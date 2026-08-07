"""Reference backend: the pinned Hugging Face ``tokenizers`` path.

Contract reference: ``docs/contracts/api.md`` Sections 3-4,
``docs/contracts/routing.md`` Section 4 (``hf`` is always present and
always last in every fallback chain), ``docs/contracts/registry.md``
Section 2 (oracle version policy).

Design rules this module keeps:

- **The artifact is executed as written.** The backend opens the
  verified ``tokenizer.json`` through the ``tokenizers`` crate and
  passes no loader flag of any kind. A loader flag that changes the
  pipeline changes token ids, so it would silently move the artifact
  out of the configuration every certification reading was taken
  against. Flags are rejected at construction, not ignored -- see
  :data:`REJECTED_LOADER_FLAGS`.
- **Local files only.** The crate is handed an absolute path to an
  already-verified file. Nothing here resolves a repository id, and
  nothing here reaches the network; missing files raise
  ``ArtifactNotFound``.
- **Verify what we open.** The manifest digest for the file we read is
  re-checked here as well. The artifacts layer verifies on fetch; this
  check costs one hash of a file we already read and turns
  time-of-check/time-of-use drift into ``ArtifactHashMismatch``.
- **Output-rewriting modes are refused.** An artifact that declares
  truncation or padding cannot produce the core token stream the rest
  of the system stores, so it is rejected with ``UNSUPPORTED_CONFIG``
  rather than quietly re-encoded.

The module imports no accelerator runtime, and the ``tokenizers``
import is deferred to construction so that importing ``toktier`` stays
cheap and side-effect free.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from .._oracle import ORACLE_PACKAGE, import_oracle, oracle_version
from ..errors import (
    ArtifactHashMismatch,
    ArtifactNotFound,
    UnsupportedConfig,
)
from ..policy import BACKEND_REFERENCE
from .protocol import TOKENIZER_FILE, ArtifactHandle

__all__ = [
    "ORACLE_PACKAGE",
    "REJECTED_LOADER_FLAGS",
    "HfBackend",
    "oracle_version",
]

#: Loader flags that must never be set, with the reason recorded next to
#: each one. The mapping is not an allowlist by omission: *every* loader
#: flag is rejected (see :meth:`HfBackend.open`); named entries only get
#: a more specific message because they have bitten a judged
#: configuration before.
REJECTED_LOADER_FLAGS: Mapping[str, str] = {
    "fix_mistral_regex": (
        "this flag rewrites the artifact's pre-tokenizer pattern. Every "
        "certification reading for the affected family was taken with the "
        "pattern as written, and the loader warning that suggests the flag "
        "is itself the evidence that it was not set. Setting it would move "
        "the running configuration outside every certified record."
    ),
}

#: Artifact sections that would rewrite the output stream.
_REWRITING_SECTIONS = ("truncation", "padding")


class _CrateEncoding(Protocol):
    @property
    def ids(self) -> Sequence[int]:
        """Token ids of one encoded sequence."""


class _CrateTokenizer(Protocol):
    def encode(
        self, sequence: str, add_special_tokens: bool = True
    ) -> _CrateEncoding:
        """Encode one sequence."""

    def encode_batch(
        self, input: Sequence[str], add_special_tokens: bool = True
    ) -> Sequence[_CrateEncoding]:
        """Encode a batch of sequences."""


def _load_crate_tokenizer(path: Path) -> _CrateTokenizer:
    """Open a verified ``tokenizer.json`` through the oracle package."""
    return cast(_CrateTokenizer, import_oracle().Tokenizer.from_file(str(path)))


class HfBackend:
    """The reference backend (backend id ``hf``).

    Instances are constructed from a verified :class:`ArtifactHandle`;
    there is no constructor that takes an unverified path, because the
    reference backend defines correct output and must not be pointed at
    bytes nobody checked.
    """

    def __init__(
        self,
        *,
        family: str,
        artifact_sha256: str,
        tokenizer_path: Path,
        tokenizer: _CrateTokenizer,
    ) -> None:
        self._family = family
        self._artifact_sha256 = artifact_sha256
        self._tokenizer_path = tokenizer_path
        self._tokenizer: _CrateTokenizer | None = tokenizer

    # -- construction --------------------------------------------------

    @classmethod
    def open(
        cls,
        artifact: ArtifactHandle,
        *,
        loader_flags: Mapping[str, object] | None = None,
    ) -> HfBackend:
        """Open the reference backend over a verified artifact.

        ``loader_flags`` exists only so that a caller who tries to pass
        one receives a specific, documented refusal instead of having
        the argument silently dropped.
        """
        if loader_flags:
            name = sorted(loader_flags)[0]
            note = REJECTED_LOADER_FLAGS.get(
                name,
                "the reference backend runs the artifact as written; loader "
                "flags change token ids and would leave every certified "
                "record behind.",
            )
            raise UnsupportedConfig(
                f"loader flag {name!r} is not available: {note}",
                details={
                    "option": name,
                    "value": loader_flags[name],
                    "reason": "reference backend runs the artifact as written",
                },
            )

        path = artifact.path(TOKENIZER_FILE)
        if not path.is_file():
            raise ArtifactNotFound(
                f"{TOKENIZER_FILE} is missing from the resolved artifact",
                details={
                    "family": artifact.family,
                    "searched": [str(path)],
                    "offline": None,
                },
            )
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        expected = artifact.files.get(TOKENIZER_FILE, artifact.artifact_sha256)
        if expected and observed != expected:
            raise ArtifactHashMismatch(
                f"{TOKENIZER_FILE} does not match its recorded digest",
                details={
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "path": str(path),
                    "remedy": (
                        "re-fetch the artifact; a cached file that no longer "
                        "matches the manifest is never accepted"
                    ),
                },
            )
        cls._reject_rewriting_sections(path, raw)
        return cls(
            family=artifact.family,
            artifact_sha256=observed,
            tokenizer_path=path,
            tokenizer=_load_crate_tokenizer(path),
        )

    @staticmethod
    def _reject_rewriting_sections(path: Path, raw: bytes) -> None:
        """Refuse artifacts that declare truncation or padding."""
        document = json.loads(raw.decode("utf-8"))
        if not isinstance(document, dict):
            raise UnsupportedConfig(
                f"{TOKENIZER_FILE} is not a JSON object",
                details={
                    "option": TOKENIZER_FILE,
                    "value": str(path),
                    "reason": "unexpected artifact shape",
                },
            )
        for section in _REWRITING_SECTIONS:
            if document.get(section) is not None:
                raise UnsupportedConfig(
                    f"artifact declares {section!r}; output-rewriting modes "
                    "are outside the supported envelope",
                    details={
                        "option": section,
                        "value": document[section],
                        "reason": (
                            "the stored stream is the pre-postprocessor core "
                            "stream; truncation and padding cannot be "
                            "represented losslessly"
                        ),
                    },
                )

    # -- identity ------------------------------------------------------

    @property
    def backend_id(self) -> str:
        """Frozen backend identifier of the reference path."""
        return BACKEND_REFERENCE

    @property
    def family(self) -> str:
        """Family id this backend was opened for."""
        return self._family

    @property
    def artifact_sha256(self) -> str:
        """Digest of the artifact bytes this backend executes."""
        return self._artifact_sha256

    @property
    def tokenizer_path(self) -> Path:
        """Path of the verified artifact file in use."""
        return self._tokenizer_path

    # -- encoding ------------------------------------------------------

    def _live(self) -> _CrateTokenizer:
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("backend is closed")
        return tokenizer

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one document to token ids."""
        encoding = self._live().encode(text, add_special_tokens=add_special_tokens)
        return [int(token_id) for token_id in encoding.ids]

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode a batch; row ``i`` equals ``encode(texts[i])``."""
        if not texts:
            return []
        encodings = self._live().encode_batch(
            list(texts), add_special_tokens=add_special_tokens
        )
        return [[int(token_id) for token_id in item.ids] for item in encodings]

    def close(self) -> None:
        """Release the loaded tokenizer. Idempotent."""
        self._tokenizer = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HfBackend(family={self._family!r}, "
            f"artifact_sha256={self._artifact_sha256[:12]!r})"
        )
