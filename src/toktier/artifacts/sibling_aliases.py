"""Strict, content-bound view of the packaged sibling-tokenizer registry.

Repository names select a recorded revision and file name; they never grant an
accelerated route.  Admission is by the sha256 of the bytes actually resolved.
The selected canonical artifact is then loaded through the ordinary manifest
and routing registry, so this table cannot bypass any CPU or GPU certificate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .._jcs import CanonicalizationError, canonical_json
from ..backends.protocol import TOKENIZER_FILE
from ..errors import RegistryInvalid
from .manifest import ArtifactManifest
from .tables import ARTIFACT_MANIFEST, SIBLING_ALIASES

__all__ = [
    "ALIAS_BASES",
    "SiblingAliasRecord",
    "SiblingAliasRegistry",
    "load_sibling_aliases",
    "shipped_sibling_aliases",
]

ALIAS_BASES = frozenset(
    {
        "identical",
        "identical_source",
        "equivalent_canonicalisation",
        "equivalent_serialisation",
    }
)

_DOMAIN_TAG = b"toktier.sibling_aliases.v1\x00"
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_REVISION = re.compile(r"\A[0-9a-f]{40}\Z")
_FAMILY = re.compile(r"\A[a-z0-9][a-z0-9._-]{0,127}\Z")
_REPO = re.compile(r"\A[^\s/]+/[^\s/]+\Z")
_FILES = frozenset({"tokenizer.json", "tiktoken.model"})
_EXPECTED_COUNTS = {
    "identical": 150,
    "identical_source": 13,
    "equivalent_canonicalisation": 10,
    "equivalent_serialisation": 38,
    "total": 211,
    "packaged": 204,
    "reference_only": 7,
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "generated_by",
    "root_digest",
    "provenance",
    "counts",
    "aliases",
}
_GENERATED_KEYS = {"tool", "tool_version", "source_commit", "generated_at"}
_PROVENANCE_KEYS = {
    "audit_date",
    "audit_rows",
    "audit_sha256",
    "support_matrix_sha256",
    "selection",
}
_ALIAS_KEYS = {
    "repo_id",
    "revision",
    "source_file",
    "source_sha256",
    "source_size",
    "canonical_family",
    "canonical_anchor_sha256",
    "basis",
    "canonical_packaged",
}


class _DuplicateKey(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKey(f"duplicate JSON member {key!r}")
        value[key] = item
    return value


def _invalid(path: Path | str, failure: str) -> RegistryInvalid:
    return RegistryInvalid(
        f"invalid sibling alias registry: {failure}",
        details={"path": str(path), "failure": failure},
    )


def _exact_keys(
    value: Mapping[str, object], expected: set[str], *, path: Path | str, what: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise _invalid(
            path,
            f"{what} members differ: missing {sorted(expected - actual)}, "
            f"unknown {sorted(actual - expected)}",
        )


def _required_text(
    value: object,
    *,
    path: Path | str,
    what: str,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(path, f"{what} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise _invalid(path, f"{what} has invalid value {value!r}")
    return value


@dataclass(frozen=True)
class SiblingAliasRecord:
    """One audited model-repository tokenizer identity."""

    repo_id: str
    revision: str
    source_file: str
    source_sha256: str
    source_size: int
    canonical_family: str
    canonical_anchor_sha256: str
    basis: str
    canonical_packaged: bool


@dataclass(frozen=True)
class SiblingAliasRegistry:
    """Indexes verified aliases by repository hint and by content identity."""

    records: tuple[SiblingAliasRecord, ...]
    root_digest: str
    _by_repo: Mapping[str, SiblingAliasRecord] = field(
        init=False, repr=False, compare=False
    )
    _by_content: Mapping[tuple[str, str], tuple[SiblingAliasRecord, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        by_repo: dict[str, SiblingAliasRecord] = {}
        by_content: dict[tuple[str, str], list[SiblingAliasRecord]] = {}
        for record in self.records:
            by_repo[record.repo_id] = record
            by_content.setdefault(
                (record.source_file, record.source_sha256), []
            ).append(record)
        object.__setattr__(self, "_by_repo", MappingProxyType(by_repo))
        object.__setattr__(
            self,
            "_by_content",
            MappingProxyType(
                {key: tuple(value) for key, value in by_content.items()}
            ),
        )

    def for_repo(self, repo_id: str) -> SiblingAliasRecord | None:
        """Recorded lookup hint for ``repo_id``, if one exists."""
        return self._by_repo.get(repo_id)

    def match(
        self, source_file: str, source_sha256: str, *, repo_id: str | None = None
    ) -> SiblingAliasRecord | None:
        """Return an audited content match, preferring the requested repo row."""
        matches = self._by_content.get((source_file, source_sha256), ())
        if repo_id is not None:
            for record in matches:
                if record.repo_id == repo_id:
                    return record
        return matches[0] if matches else None

    def validate_manifest(
        self, manifest: ArtifactManifest, *, path: Path | str
    ) -> None:
        """Require every packaged flag and anchor to agree with ``manifest``."""
        for record in self.records:
            entry = manifest.entries.get(record.canonical_family)
            packaged = entry is not None
            if packaged != record.canonical_packaged:
                raise _invalid(
                    path,
                    f"{record.repo_id}: canonical_packaged={record.canonical_packaged} "
                    f"but manifest presence is {packaged}",
                )
            if entry is not None:
                observed = entry.file(TOKENIZER_FILE).sha256
                if observed != record.canonical_anchor_sha256:
                    raise _invalid(
                        path,
                        f"{record.repo_id}: canonical anchor "
                        f"{record.canonical_anchor_sha256} disagrees with manifest "
                        f"digest {observed}",
                    )

    @classmethod
    def from_document(
        cls, document: Mapping[str, object], *, path: Path | str = "<memory>"
    ) -> SiblingAliasRegistry:
        """Validate a root-checked document's runtime-relevant structure."""
        _exact_keys(document, _TOP_LEVEL_KEYS, path=path, what="top-level")
        if document.get("schema_version") != 1:
            raise _invalid(path, "schema_version must be 1")
        generated = document.get("generated_by")
        if not isinstance(generated, Mapping):
            raise _invalid(path, "generated_by must be an object")
        _exact_keys(generated, _GENERATED_KEYS, path=path, what="generated_by")
        if generated.get("tool") != "tools/generate_sibling_aliases.py":
            raise _invalid(path, "generated_by.tool is not the owning generator")
        _required_text(
            generated.get("tool_version"), path=path, what="generated_by.tool_version"
        )
        source_commit = _required_text(
            generated.get("source_commit"),
            path=path,
            what="generated_by.source_commit",
        )
        if re.fullmatch(r"[0-9a-f]{7,40}", source_commit) is None:
            raise _invalid(path, "generated_by.source_commit is not a git object id")
        _required_text(
            generated.get("generated_at"), path=path, what="generated_by.generated_at"
        )
        provenance = document.get("provenance")
        if not isinstance(provenance, Mapping):
            raise _invalid(path, "provenance must be an object")
        _exact_keys(provenance, _PROVENANCE_KEYS, path=path, what="provenance")
        audit_date = _required_text(
            provenance.get("audit_date"), path=path, what="provenance.audit_date"
        )
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", audit_date) is None:
            raise _invalid(path, "provenance.audit_date is not an ISO date")
        if provenance.get("audit_rows") != 470:
            raise _invalid(path, "provenance.audit_rows must be 470")
        for key in ("audit_sha256", "support_matrix_sha256"):
            _required_text(
                provenance.get(key),
                path=path,
                what=f"provenance.{key}",
                pattern=_SHA256,
            )
        _required_text(
            provenance.get("selection"), path=path, what="provenance.selection"
        )
        counts = document.get("counts")
        if not isinstance(counts, Mapping) or dict(counts) != _EXPECTED_COUNTS:
            raise _invalid(
                path,
                f"counts drifted: expected {_EXPECTED_COUNTS}, got {counts!r}",
            )
        raw_records = document.get("aliases")
        if not isinstance(raw_records, list):
            raise _invalid(path, "aliases must be an array")
        records: list[SiblingAliasRecord] = []
        repo_ids: set[str] = set()
        content_targets: dict[tuple[str, str], tuple[str, str, int]] = {}
        for index, raw in enumerate(raw_records):
            if not isinstance(raw, Mapping):
                raise _invalid(path, f"aliases[{index}] must be an object")
            _exact_keys(raw, _ALIAS_KEYS, path=path, what=f"aliases[{index}]")
            repo_id = _required_text(
                raw.get("repo_id"), path=path, what=f"aliases[{index}].repo_id"
            )
            if len(repo_id) > 192 or _REPO.fullmatch(repo_id) is None:
                raise _invalid(path, f"aliases[{index}].repo_id is invalid")
            if any(part in {".", ".."} for part in repo_id.split("/")):
                raise _invalid(path, f"aliases[{index}].repo_id contains traversal")
            if repo_id in repo_ids:
                raise _invalid(path, f"repo_id {repo_id!r} is listed twice")
            repo_ids.add(repo_id)
            revision = _required_text(
                raw.get("revision"),
                path=path,
                what=f"aliases[{index}].revision",
                pattern=_REVISION,
            )
            source_file = _required_text(
                raw.get("source_file"),
                path=path,
                what=f"aliases[{index}].source_file",
            )
            if source_file not in _FILES:
                raise _invalid(path, f"unsupported source file {source_file!r}")
            source_sha = _required_text(
                raw.get("source_sha256"),
                path=path,
                what=f"aliases[{index}].source_sha256",
                pattern=_SHA256,
            )
            source_size = raw.get("source_size")
            if (
                isinstance(source_size, bool)
                or not isinstance(source_size, int)
                or source_size <= 0
            ):
                raise _invalid(path, f"aliases[{index}].source_size must be positive")
            family = _required_text(
                raw.get("canonical_family"),
                path=path,
                what=f"aliases[{index}].canonical_family",
                pattern=_FAMILY,
            )
            anchor = _required_text(
                raw.get("canonical_anchor_sha256"),
                path=path,
                what=f"aliases[{index}].canonical_anchor_sha256",
                pattern=_SHA256,
            )
            basis = _required_text(
                raw.get("basis"), path=path, what=f"aliases[{index}].basis"
            )
            if basis not in ALIAS_BASES:
                raise _invalid(path, f"unknown alias basis {basis!r}")
            if (basis == "identical_source") != (source_file == "tiktoken.model"):
                raise _invalid(
                    path,
                    f"{repo_id}: identical_source and tiktoken.model must coincide",
                )
            packaged = raw.get("canonical_packaged")
            if not isinstance(packaged, bool):
                raise _invalid(path, f"{repo_id}: canonical_packaged must be boolean")
            content_key = (source_file, source_sha)
            content_target = (family, anchor, source_size)
            previous = content_targets.setdefault(content_key, content_target)
            if previous != content_target:
                raise _invalid(
                    path,
                    f"content {source_file}@{source_sha} has conflicting targets",
                )
            records.append(
                SiblingAliasRecord(
                    repo_id=repo_id,
                    revision=revision,
                    source_file=source_file,
                    source_sha256=source_sha,
                    source_size=source_size,
                    canonical_family=family,
                    canonical_anchor_sha256=anchor,
                    basis=basis,
                    canonical_packaged=packaged,
                )
            )
        if [item.repo_id for item in records] != sorted(
            (item.repo_id for item in records), key=str.casefold
        ):
            raise _invalid(path, "aliases are not sorted by repository id")
        basis_counts = Counter(item.basis for item in records)
        recomputed = {
            "identical": basis_counts["identical"],
            "identical_source": basis_counts["identical_source"],
            "equivalent_canonicalisation": basis_counts[
                "equivalent_canonicalisation"
            ],
            "equivalent_serialisation": basis_counts[
                "equivalent_serialisation"
            ],
            "total": len(records),
            "packaged": sum(item.canonical_packaged for item in records),
            "reference_only": sum(not item.canonical_packaged for item in records),
        }
        if recomputed != _EXPECTED_COUNTS:
            raise _invalid(path, f"record counts do not close: {recomputed}")
        root = _required_text(
            document.get("root_digest"), path=path, what="root_digest"
        )
        return cls(records=tuple(records), root_digest=root)


def load_sibling_aliases(path: Path) -> SiblingAliasRegistry:
    """Read, duplicate-check, root-check, and structurally validate a table."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise _invalid(path, f"cannot read file: {error}") from error
    try:
        raw = json.loads(text, object_pairs_hook=_object_without_duplicates)
    except (ValueError, _DuplicateKey) as error:
        raise _invalid(path, f"cannot parse JSON: {error}") from error
    if not isinstance(raw, dict):
        raise _invalid(path, "top-level value must be an object")
    recorded = raw.get("root_digest")
    if not isinstance(recorded, str):
        raise _invalid(path, "root_digest is missing")
    body = {key: value for key, value in raw.items() if key != "root_digest"}
    try:
        canonical = canonical_json(body)
    except CanonicalizationError as error:
        raise _invalid(path, f"cannot canonicalize document: {error}") from error
    expected = "sha256:" + hashlib.sha256(_DOMAIN_TAG + canonical).hexdigest()
    if recorded != expected:
        raise _invalid(
            path, f"root digest mismatch (recorded {recorded}, computed {expected})"
        )
    return SiblingAliasRegistry.from_document(raw, path=path)


@lru_cache(maxsize=1)
def shipped_sibling_aliases() -> SiblingAliasRegistry:
    """The packaged alias registry, bound to the packaged artifact manifest."""
    registry = load_sibling_aliases(SIBLING_ALIASES)
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    registry.validate_manifest(manifest, path=SIBLING_ALIASES)
    return registry
