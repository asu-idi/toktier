"""Artifact manifests: per-file content identity for tokenizer artifacts.

Contract reference: ``docs/contracts/registry.md`` section 4. An
artifact manifest is the fetch-side companion of the support registry
and pins **per-file sha256** for every file of an artifact, not merely a
repository revision: verification is per file against these digests, and
a revision pin without content digests is not sufficient. A manifest
that carries no digests is therefore rejected at load time rather than
accepted with weaker checking.

Overlay semantics are add-only. An overlay may introduce families the
base manifest does not have; it can never rewrite an entry the base
already pins, so an in-service family cannot be redirected by dropping a
file next to it.

This module is dependency-free (standard library only).
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..errors import ArtifactNotFound, RegistryInvalid

__all__ = [
    "ArtifactEntry",
    "ArtifactFile",
    "ArtifactManifest",
]

_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")

#: Family ids: lower case, digits, underscores, dots and dashes, with an
#: alphanumeric first character. Family names become path components of
#: the cache layout, so the grammar admits no separators, no leading dot
#: and nothing a platform treats as traversal.
_FAMILY_PATTERN = re.compile(r"\A[a-z0-9][a-z0-9._-]*\Z")

#: Revisions: git object names and tag-like names. Same path-safety
#: rules as families, with upper case admitted.
_REVISION_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")

#: Upper bound on identifier length, so a derived directory name stays a
#: valid file name on every supported filesystem.
_IDENTIFIER_MAX = 128

#: Characters that would make a manifest file name escape the artifact
#: directory or collide with the store's own bookkeeping.
_REJECTED_NAME_PARTS = ("..", "")


def _check_identifier(value: str, *, what: str, pattern: re.Pattern[str]) -> str:
    """Validate a path-bound identifier against its documented grammar."""
    if not pattern.match(value) or len(value) > _IDENTIFIER_MAX:
        raise ValueError(
            f"{what} {value!r} is not a valid identifier: it becomes a cache "
            "path component, so it must match "
            f"{pattern.pattern!r} and stay under {_IDENTIFIER_MAX} characters"
        )
    return value


def _fail(source: str, failure: str) -> RegistryInvalid:
    return RegistryInvalid(
        f"invalid artifact manifest: {failure}",
        details={"path": source, "failure": failure},
    )


def _require_str(value: object, *, source: str, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(source, f"{what} must be a non-empty string")
    return value


def _check_relative_name(name: str, *, source: str) -> str:
    """Validate a manifest file name as a safe artifact-relative path."""
    if name != name.strip() or not name:
        raise _fail(source, f"file name {name!r} is empty or padded")
    if "\\" in name or name.startswith("/"):
        raise _fail(source, f"file name {name!r} must be a relative POSIX path")
    parts = name.split("/")
    for part in parts:
        if part in _REJECTED_NAME_PARTS or part == ".":
            raise _fail(source, f"file name {name!r} must not traverse directories")
    if name.startswith("."):
        raise _fail(source, f"file name {name!r} must not start with a dot")
    return name


@dataclass(frozen=True)
class ArtifactFile:
    """One file of an artifact, identified by its content digest."""

    #: Artifact-relative POSIX path (for example ``tokenizer.json``).
    name: str
    #: Lower-case hexadecimal sha256 of the file content.
    sha256: str
    #: Expected byte length when the manifest records it. Checked before
    #: hashing, so a truncated transfer is caught by the cheap test.
    size: int | None = None


@dataclass(frozen=True)
class ArtifactEntry:
    """Everything needed to fetch and verify one tokenizer artifact.

    ``family`` and ``revision`` become path components of the cache
    layout through :attr:`directory_name`, so construction enforces the
    slug grammar on them; an entry that exists is safe to join to the
    cache root.
    """

    #: Canonical family id (lower case with underscores by convention).
    family: str
    #: Upstream repository the artifact was frozen from.
    repo_id: str
    #: Frozen upstream revision. Recorded for provenance; verification
    #: is done against the per-file digests, not against this value.
    revision: str
    #: Per-file digests. Never empty: an entry without digests is not a
    #: valid artifact identity under the registry contract.
    files: tuple[ArtifactFile, ...]
    #: Directory holding the files for a local source, if the manifest
    #: names one. Relative paths are resolved by the source.
    local_dir: str | None = None
    #: Alternative spellings accepted for this family.
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_identifier(self.family, what="family", pattern=_FAMILY_PATTERN)
        _check_identifier(
            self.revision, what="revision", pattern=_REVISION_PATTERN
        )

    @property
    def directory_name(self) -> str:
        """Cache directory name: family plus a revision prefix.

        The revision prefix makes a re-freeze a new directory rather
        than an in-place rewrite of a verified one.
        """
        return f"{self.family}-{self.revision[:12]}"

    def file(self, name: str) -> ArtifactFile:
        """Return the entry for ``name`` or raise :class:`ArtifactNotFound`."""
        for artifact_file in self.files:
            if artifact_file.name == name:
                return artifact_file
        raise ArtifactNotFound(
            f"artifact {self.family!r} has no file named {name!r}",
            details={
                "family": self.family,
                "searched": [item.name for item in self.files],
            },
        )


@dataclass(frozen=True)
class ArtifactManifest:
    """An immutable set of artifact entries keyed by family id."""

    entries: Mapping[str, ArtifactEntry] = field(default_factory=dict)
    #: Files this manifest was read from, in overlay order. Reported in
    #: ``ArtifactNotFound`` details so a missing family says where we
    #: looked.
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))
        object.__setattr__(self, "sources", tuple(self.sources))

    # -- construction --------------------------------------------------

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, source: str | Path = "<memory>"
    ) -> ArtifactManifest:
        """Parse and validate a manifest mapping (fail closed)."""
        origin = str(source)
        entries: dict[str, ArtifactEntry] = {}
        for family, raw in data.items():
            if not isinstance(family, str) or not family:
                raise _fail(origin, "family ids must be non-empty strings")
            if not isinstance(raw, Mapping):
                raise _fail(origin, f"entry {family!r} must be a table")
            entries[family] = _parse_entry(family, raw, source=origin)
        return cls(entries=entries, sources=(origin,))

    @classmethod
    def load(cls, path: str | Path) -> ArtifactManifest:
        """Read a manifest from a JSON file."""
        manifest_path = Path(path)
        try:
            raw_text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise _fail(str(manifest_path), f"cannot read manifest: {exc}") from exc
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise _fail(str(manifest_path), f"cannot parse manifest: {exc}") from exc
        if not isinstance(data, dict):
            raise _fail(str(manifest_path), "manifest must be a JSON object")
        return cls.from_mapping(data, source=manifest_path)

    @classmethod
    def load_layered(
        cls, base: str | Path, *extras: str | Path
    ) -> ArtifactManifest:
        """Read a base manifest and overlay additional ones, add-only.

        Later files may add families; they never rewrite an entry an
        earlier file already pins.
        """
        manifest = cls.load(base)
        for extra in extras:
            manifest = manifest.overlay(cls.load(extra))
        return manifest

    # -- queries -------------------------------------------------------

    def overlay(self, other: ArtifactManifest) -> ArtifactManifest:
        """Return a manifest with ``other``'s new families added.

        Entries already present here win; the overlay only fills gaps.
        """
        merged = dict(self.entries)
        for family, entry in other.entries.items():
            merged.setdefault(family, entry)
        return ArtifactManifest(
            entries=merged, sources=self.sources + other.sources
        )

    def get(self, family: str) -> ArtifactEntry:
        """Resolve a family id or alias to its entry."""
        entry = self.entries.get(family)
        if entry is not None:
            return entry
        for candidate in self.entries.values():
            if family in candidate.aliases:
                return candidate
        suggestions = difflib.get_close_matches(family, self.families(), n=3)
        message = f"unknown tokenizer family {family!r}"
        if suggestions:
            matches = ", ".join(repr(suggestion) for suggestion in suggestions)
            message = f"{message}; closest valid family IDs: {matches}"
        raise ArtifactNotFound(
            message,
            details={
                "family": family,
                "searched": list(self.sources),
                "suggestions": suggestions,
            },
        )

    def families(self) -> tuple[str, ...]:
        """Canonical family ids, sorted."""
        return tuple(sorted(self.entries))

    def __contains__(self, family: object) -> bool:
        if not isinstance(family, str):
            return False
        try:
            self.get(family)
        except ArtifactNotFound:
            return False
        return True

    def __len__(self) -> int:
        return len(self.entries)


def _parse_entry(
    family: str, raw: Mapping[str, Any], *, source: str
) -> ArtifactEntry:
    """Build one entry, rejecting anything that weakens verification."""
    repo_id = _require_str(raw.get("repo_id"), source=source, what=f"{family}.repo_id")
    revision = _require_str(
        raw.get("revision"), source=source, what=f"{family}.revision"
    )
    files = _parse_files(family, raw.get("files"), source=source)
    local_dir_raw = raw.get("local_dir")
    if local_dir_raw is not None and not isinstance(local_dir_raw, str):
        raise _fail(source, f"{family}.local_dir must be a string")
    aliases_raw = raw.get("aliases", ())
    if isinstance(aliases_raw, str) or not isinstance(aliases_raw, Iterable):
        raise _fail(source, f"{family}.aliases must be a list of strings")
    aliases = tuple(
        _require_str(alias, source=source, what=f"{family}.aliases[]")
        for alias in aliases_raw
    )
    try:
        return ArtifactEntry(
            family=family,
            repo_id=repo_id,
            revision=revision,
            files=files,
            local_dir=local_dir_raw,
            aliases=aliases,
        )
    except ValueError as exc:
        raise _fail(source, str(exc)) from exc


def _parse_files(
    family: str, raw: object, *, source: str
) -> tuple[ArtifactFile, ...]:
    """Parse the per-file digests, in either accepted spelling.

    Accepted shapes are a mapping ``{name: {sha256, size}}`` and a list
    of ``{name, sha256, size}`` tables. Both must be non-empty: a
    revision pin without content digests is not a valid identity.
    """
    items: list[tuple[str, Any]] = []
    if isinstance(raw, Mapping):
        items = list(raw.items())
    elif isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        for element in raw:
            if not isinstance(element, Mapping):
                raise _fail(source, f"{family}.files entries must be tables")
            name = _require_str(
                element.get("name"), source=source, what=f"{family}.files[].name"
            )
            items.append((name, element))
    else:
        raise _fail(
            source,
            f"{family} has no per-file sha256 digests; a revision pin alone "
            "is not sufficient",
        )
    if not items:
        raise _fail(
            source,
            f"{family} has no per-file sha256 digests; a revision pin alone "
            "is not sufficient",
        )
    files: list[ArtifactFile] = []
    seen: set[str] = set()
    for name, spec in items:
        checked_name = _check_relative_name(
            _require_str(name, source=source, what=f"{family}.files key"),
            source=source,
        )
        if checked_name in seen:
            raise _fail(source, f"{family}.files lists {checked_name!r} twice")
        seen.add(checked_name)
        files.append(_parse_file(family, checked_name, spec, source=source))
    return tuple(files)


def _parse_file(
    family: str, name: str, spec: object, *, source: str
) -> ArtifactFile:
    if isinstance(spec, str):
        digest = spec
        size: int | None = None
    elif isinstance(spec, Mapping):
        digest = _require_str(
            spec.get("sha256"), source=source, what=f"{family}.files[{name}].sha256"
        )
        raw_size = spec.get("size")
        if raw_size is None:
            size = None
        elif isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise _fail(source, f"{family}.files[{name}].size must be an integer")
        elif raw_size < 0:
            raise _fail(source, f"{family}.files[{name}].size must not be negative")
        else:
            size = raw_size
    else:
        raise _fail(source, f"{family}.files[{name}] must be a digest or a table")
    if not _SHA256_PATTERN.match(digest):
        raise _fail(
            source,
            f"{family}.files[{name}].sha256 must be 64 lower-case hex digits",
        )
    return ArtifactFile(name=name, sha256=digest, size=size)
