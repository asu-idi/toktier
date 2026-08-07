"""Backend protocol and the artifact call surface backends depend on.

Contract reference: ``docs/contracts/routing.md`` Section 4 (backend id
namespace), ``docs/contracts/api.md`` Sections 3-4 (encode surface), and
``docs/contracts/registry.md`` Section 4 (artifact manifests pin
per-file sha256).

Two families of protocol live here:

- :class:`Backend` -- what every backend implements. The reference
  backend (``hf``) is in this package; the CUDA backend (``gpu``) is
  ``toktier.engine.gpu.backend.GpuBackend`` and satisfies this
  protocol. No placeholder implementation is shipped here, so there is
  exactly one definition of each backend.
- :class:`ArtifactHandle` / :class:`ArtifactResolver` -- the shapes this
  lane calls against. The concrete implementation belongs to the
  artifacts subsystem (``toktier.artifacts``); these structural
  protocols exist so backends can be written and tested before that
  implementation lands, and so the integration point is written down
  rather than assumed.

This module is dependency-free (standard library only) and imports no
accelerator runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "ADDED_TOKENS_FILE",
    "TOKENIZER_FILE",
    "ArtifactHandle",
    "ArtifactResolver",
    "Backend",
    "BackendFactory",
]

#: The artifact file that defines tokenization behavior. Verification is
#: per file against the manifest digests (registry.md Section 4).
TOKENIZER_FILE = "tokenizer.json"

#: Optional sidecar consulted by the added-token frontend when present.
ADDED_TOKENS_FILE = "added_tokens.json"


@runtime_checkable
class Backend(Protocol):
    """One tokenization backend.

    Implementations return token ids that are equal to the reference
    oracle for every configuration the registry certifies. A backend
    that cannot guarantee that for an input raises instead of guessing:
    the routing layer turns the exception into a counted fallback along
    the plan's chain, and the reference result is returned.
    """

    @property
    def backend_id(self) -> str:
        """Backend identifier from the frozen namespace (``hf``, ``gpu``)."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one document to token ids."""

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode a batch; row ``i`` equals ``encode(texts[i])``."""

    def close(self) -> None:
        """Release backend resources. Idempotent."""


class BackendFactory(Protocol):
    """Builds a backend for a resolved artifact.

    Kept separate from :class:`Backend` so the routing layer can plan
    without constructing anything: planning is a pure function, and a
    factory is only invoked once a plan has selected the backend.
    """

    def __call__(self, artifact: ArtifactHandle) -> Backend:
        """Construct a backend over the given verified artifact."""


@runtime_checkable
class ArtifactHandle(Protocol):
    """A resolved artifact whose files have already been verified.

    Interface alignment note (artifacts lane owns the implementation):
    a handle is produced only after every file listed in the manifest
    matched its recorded sha256. Backends never fetch, never resolve
    repository ids, and never accept an unverified path -- they consume
    handles. ``files`` is the per-file digest map from the manifest, so
    a backend can re-check the one file it opens without a second
    manifest reader.
    """

    @property
    def family(self) -> str:
        """Canonical family id this artifact was resolved for."""

    @property
    def root(self) -> Path:
        """Directory holding the verified artifact files."""

    @property
    def artifact_sha256(self) -> str:
        """Lowercase hex sha256 of the artifact bytes (``tokenizer.json``)."""

    @property
    def files(self) -> Mapping[str, str]:
        """Relative file name -> lowercase hex sha256, from the manifest."""

    def path(self, relative_name: str) -> Path:
        """Absolute path of one verified file inside the artifact."""


class ArtifactResolver(Protocol):
    """Resolves a family id to a verified artifact.

    Interface alignment note: ``family`` accepts registry ids only
    (decision 0002 item 4); local paths, if ever supported, arrive as a
    separate parameter rather than as an overload of ``family``.
    Resolution failures raise ``ArtifactNotFound``; digest failures
    raise ``ArtifactHashMismatch`` -- never a silent acceptance.
    """

    def resolve(self, family: str) -> ArtifactHandle:
        """Return a verified handle for ``family``."""
