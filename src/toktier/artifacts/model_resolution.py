"""Resolve a model repository to a content-bound executable artifact.

This module owns the boundary between a mutable repository name and TokTier's
immutable artifact identities.  It downloads one recorded source file, hashes
the bytes, consults the verified-sibling table, and returns either a packaged
canonical artifact or a verified local reference-only artifact.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from ..backends.protocol import TOKENIZER_FILE
from ..config import Config
from ..errors import ArtifactNotFound, RegistryInvalid
from .conversion import recipe_for
from .manifest import ArtifactEntry, ArtifactManifest
from .sibling_aliases import SiblingAliasRecord, SiblingAliasRegistry
from .sources import LocalDirectorySource
from .store import ArtifactStore, sha256_file

__all__ = ["ModelResolution", "ResolvedModelArtifact", "resolve_model_repository"]


@dataclass(frozen=True)
class ModelResolution:
    """How a model-repository request became one executable artifact."""

    requested_repo: str
    requested_revision: str | None
    resolved_revision: str
    source_file: str
    source_sha256: str
    source_size: int
    registry_root_digest: str
    basis: str | None
    evidence_repo: str | None
    canonical_family: str | None
    canonical_anchor_sha256: str | None
    admitted: bool
    refusal_reason: str | None

    def report(self, *, execution_artifact_sha256: str) -> dict[str, object]:
        """Machine-readable provenance block embedded by ``explain()``."""
        return {
            "requested_repo": self.requested_repo,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "registry_root_digest": self.registry_root_digest,
            "basis": self.basis,
            "evidence_repo": self.evidence_repo,
            "canonical_family": self.canonical_family,
            "canonical_anchor_sha256": self.canonical_anchor_sha256,
            "admitted": self.admitted,
            "refusal_reason": self.refusal_reason,
            "execution_artifact_sha256": execution_artifact_sha256,
        }


@dataclass(frozen=True)
class ResolvedModelArtifact:
    """Family/manifest pair plus the repository-to-artifact explanation."""

    family: str
    manifest: ArtifactManifest
    resolution: ModelResolution


def _download_model_file(
    repo_id: str,
    filename: str,
    revision: str,
    *,
    offline: bool,
) -> Path:
    """Resolve one Hub file lazily; kept separate for network-free tests."""
    try:
        hub = importlib.import_module("huggingface_hub")
        downloaded = hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_files_only=offline,
        )
    except Exception as error:
        raise ArtifactNotFound(
            f"cannot resolve {filename!r} from {repo_id!r} at {revision!r}",
            details={
                "repo_id": repo_id,
                "revision": revision,
                "file": filename,
                "offline": offline,
                "failure": str(error),
            },
        ) from error
    path = Path(str(downloaded))
    if not path.is_file():
        raise ArtifactNotFound(
            f"the Hub returned no file for {repo_id!r}/{filename}",
            details={
                "repo_id": repo_id,
                "revision": revision,
                "file": filename,
                "searched": [str(path)],
                "offline": offline,
            },
        )
    return path


def _manifest_repo_entry(
    manifest: ArtifactManifest, repo_id: str
) -> ArtifactEntry | None:
    """Unique canonical manifest entry published from ``repo_id``."""
    matches = [entry for entry in manifest.entries.values() if entry.repo_id == repo_id]
    if len(matches) > 1:
        raise RegistryInvalid(
            f"artifact manifest maps repository {repo_id!r} more than once",
            details={
                "repo_id": repo_id,
                "families": sorted(entry.family for entry in matches),
            },
        )
    return matches[0] if matches else None


def _manifest_anchor_match(
    manifest: ArtifactManifest, source_file: str, source_sha256: str
) -> ArtifactEntry | None:
    """Unique packaged canonical entry with this exact tokenizer identity."""
    if source_file != TOKENIZER_FILE:
        return None
    matches = [
        entry
        for entry in manifest.entries.values()
        if entry.file(TOKENIZER_FILE).sha256 == source_sha256
    ]
    if len(matches) > 1:
        raise RegistryInvalid(
            "artifact manifest has an ambiguous tokenizer content identity",
            details={
                "artifact_sha256": source_sha256,
                "families": sorted(entry.family for entry in matches),
            },
        )
    return matches[0] if matches else None


def _canonical_alias_entry(
    manifest: ArtifactManifest, record: SiblingAliasRecord
) -> ArtifactEntry | None:
    """The record's packaged anchor, with disagreement treated as corruption."""
    if not record.canonical_packaged:
        return None
    entry = manifest.entries.get(record.canonical_family)
    if entry is None:
        return None
    observed = entry.file(TOKENIZER_FILE).sha256
    if observed != record.canonical_anchor_sha256:
        raise RegistryInvalid(
            "sibling alias and artifact manifest disagree on the canonical anchor",
            details={
                "repo_id": record.repo_id,
                "family": record.canonical_family,
                "alias_anchor_sha256": record.canonical_anchor_sha256,
                "manifest_anchor_sha256": observed,
            },
        )
    return entry


def _install_external_reference(
    *,
    repo_id: str,
    source_path: Path,
    source_sha256: str,
    source_size: int,
    config: Config,
) -> tuple[str, ArtifactManifest]:
    """Import downloaded bytes into the verified cache under a content id."""
    family = f"external_{source_sha256[:24]}"
    manifest = ArtifactManifest.from_mapping(
        {
            family: {
                "repo_id": repo_id,
                "revision": source_sha256[:40],
                "local_dir": str(source_path.parent),
                "files": {
                    TOKENIZER_FILE: {
                        "sha256": source_sha256,
                        "size": source_size,
                    }
                },
            }
        },
        source=f"from_pretrained:{repo_id}",
    )
    # This is a verified local import from the path the Hub resolver already
    # returned. Clearing the store's generic offline gate cannot permit a
    # network access: LocalDirectorySource has no network implementation. The
    # original config, including offline=True, is used by the Tokenizer itself.
    ArtifactStore(
        manifest,
        config=config.replace(offline=False),
        source=LocalDirectorySource(),
    ).ensure(family)
    return family, manifest


def resolve_model_repository(
    repo_id: str,
    *,
    revision: str | None,
    config: Config,
    manifest: ArtifactManifest,
    aliases: SiblingAliasRegistry,
) -> ResolvedModelArtifact:
    """Resolve, hash, and safely classify one model repository tokenizer."""
    alias_hint = aliases.for_repo(repo_id)
    manifest_hint = _manifest_repo_entry(manifest, repo_id)
    if alias_hint is not None:
        source_file = alias_hint.source_file
        default_revision = alias_hint.revision
    elif manifest_hint is not None:
        # A family whose artifact is derived locally has no tokenizer.json
        # upstream, so its own repository is read through the same file the
        # conversion reads. Which file that is comes from the conversion
        # routing data, not from a second list here.
        recipe = recipe_for(manifest_hint.family)
        source_file = TOKENIZER_FILE if recipe is None else recipe.inputs[0].name
        default_revision = manifest_hint.revision
    else:
        source_file = TOKENIZER_FILE
        default_revision = "main"
    resolved_revision = revision or default_revision
    source_path = _download_model_file(
        repo_id,
        source_file,
        resolved_revision,
        offline=config.offline,
    )
    source_sha256, source_size = sha256_file(source_path)

    match = aliases.match(source_file, source_sha256, repo_id=repo_id)
    if match is not None and source_size != match.source_size:
        raise RegistryInvalid(
            "sibling alias size disagrees with matching content identity",
            details={
                "repo_id": repo_id,
                "source_sha256": source_sha256,
                "recorded_size": match.source_size,
                "observed_size": source_size,
            },
        )
    exact_anchor = _manifest_anchor_match(manifest, source_file, source_sha256)
    canonical_entry = (
        _canonical_alias_entry(manifest, match) if match is not None else None
    )

    if canonical_entry is not None:
        assert match is not None  # canonical_entry is derived only from this record
        admitted_entry = canonical_entry
        basis: str | None = match.basis
        evidence_repo: str | None = match.repo_id
    elif exact_anchor is not None:
        admitted_entry = exact_anchor
        basis = "exact_anchor"
        evidence_repo = exact_anchor.repo_id
    else:
        admitted_entry = None
        basis = match.basis if match is not None else None
        evidence_repo = match.repo_id if match is not None else None

    if admitted_entry is not None:
        return ResolvedModelArtifact(
            family=admitted_entry.family,
            manifest=manifest,
            resolution=ModelResolution(
                requested_repo=repo_id,
                requested_revision=revision,
                resolved_revision=resolved_revision,
                source_file=source_file,
                source_sha256=source_sha256,
                source_size=source_size,
                registry_root_digest=aliases.root_digest,
                basis=basis,
                evidence_repo=evidence_repo,
                canonical_family=admitted_entry.family,
                canonical_anchor_sha256=admitted_entry.file(
                    TOKENIZER_FILE
                ).sha256,
                admitted=True,
                refusal_reason=None,
            ),
        )

    if source_file != TOKENIZER_FILE:
        raise ArtifactNotFound(
            f"{repo_id!r} resolves to {source_file!r}, which this release cannot "
            "materialize as a Hugging Face tokenizer.json",
            details={
                "repo_id": repo_id,
                "revision": resolved_revision,
                "source_file": source_file,
                "source_sha256": source_sha256,
                "canonical_family": (
                    match.canonical_family if match is not None else None
                ),
                "remedy": (
                    "use a repository that ships tokenizer.json, or one whose "
                    "source file matches a packaged canonical artifact"
                ),
            },
        )

    if match is None:
        refusal_reason = "content_not_registered"
        canonical_family = None
        canonical_anchor = None
    elif not match.canonical_packaged:
        refusal_reason = "canonical_artifact_not_packaged"
        canonical_family = match.canonical_family
        canonical_anchor = match.canonical_anchor_sha256
    else:
        refusal_reason = "canonical_artifact_not_in_active_manifest"
        canonical_family = match.canonical_family
        canonical_anchor = match.canonical_anchor_sha256
    dynamic_family, dynamic_manifest = _install_external_reference(
        repo_id=repo_id,
        source_path=source_path,
        source_sha256=source_sha256,
        source_size=source_size,
        config=config,
    )
    return ResolvedModelArtifact(
        family=dynamic_family,
        manifest=dynamic_manifest,
        resolution=ModelResolution(
            requested_repo=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            source_file=source_file,
            source_sha256=source_sha256,
            source_size=source_size,
            registry_root_digest=aliases.root_digest,
            basis=basis,
            evidence_repo=evidence_repo,
            canonical_family=canonical_family,
            canonical_anchor_sha256=canonical_anchor,
            admitted=False,
            refusal_reason=refusal_reason,
        ),
    )
