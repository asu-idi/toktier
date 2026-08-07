"""Verified artifact handles for the GPU engine.

Contract reference: ``docs/contracts/registry.md`` Section 4 (artifact
manifests pin per-file sha256; verification is per file) and
``toktier.backends.protocol.ArtifactHandle`` (the handle shape every
backend consumes).

The GPU engine takes verified handles only. It never reads a manifest of
its own and never accepts a bare directory: resolution and per-file
verification are the artifact layer's job (:class:`ArtifactStore`), and
this module is the one adapter from that layer's output to the handle
shape. A handle carries the per-file digest map, so a consumer can
re-check the single file it opens without a second manifest reader.

This module is torch-free: building and verifying the inputs of a GPU
run is host work.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ...artifacts.store import ArtifactStore
from ...backends.protocol import TOKENIZER_FILE

__all__ = ["VerifiedHandle", "verified_handle", "verified_handles"]


@dataclass(frozen=True)
class VerifiedHandle:
    """A resolved artifact whose files have been hash-verified.

    Structural implementation of ``toktier.backends.protocol.
    ArtifactHandle``. Instances are produced by :func:`verified_handle`
    only after the artifact store checked every manifest-listed file
    against its recorded sha256.
    """

    family: str
    root: Path
    artifact_sha256: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    def path(self, relative_name: str) -> Path:
        """Absolute path of one verified file inside the artifact."""
        return self.root / relative_name


def verified_handle(store: ArtifactStore, family: str) -> VerifiedHandle:
    """Resolve one family through the store's fetch-and-verify path.

    Raises whatever the artifact layer raises: ``ArtifactNotFound`` for
    an unknown family or a missing file, ``ArtifactHashMismatch`` when
    the bytes do not match the manifest. Nothing that failed
    verification ever becomes a handle.
    """
    verified = store.ensure(family)
    entry = store.manifest.get(family)
    # Raises ArtifactNotFound when the manifest does not pin the
    # tokenizer file; an artifact without it has no behavior to verify.
    tokenizer = entry.file(TOKENIZER_FILE)
    return VerifiedHandle(
        family=entry.family,
        root=verified.directory,
        artifact_sha256=tokenizer.sha256,
        files={item.name: item.sha256 for item in entry.files},
    )


def verified_handles(
    store: ArtifactStore, families: Iterable[str]
) -> dict[str, VerifiedHandle]:
    """Verified handles for several families, keyed by family id."""
    return {family: verified_handle(store, family) for family in families}
