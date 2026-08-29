"""Reference backend: the pinned Hugging Face ``tokenizers`` path.

Contract reference: ``docs/contracts/api.md`` Sections 3-4,
``docs/contracts/routing.md`` Section 4 (``hf`` is always present and
always last in every fallback chain), ``docs/contracts/registry.md``
Section 2 (oracle version policy).

Design rules this module keeps:

- **The reference is the loader face.** The artifact identity covers the
  verified ``tokenizer.json`` plus the added tokens its
  ``tokenizer_config.json`` declares beyond that file, and the reference
  executes exactly that subject. When the configuration declares no such
  token, the artifact document already is the loader face and is opened
  through the ``tokenizers`` crate directly; when it does, the backend
  materializes the live loader object once (the same construction the
  fast CPU backend certifies against) and executes its exact
  serialization. Either way no caller-supplied loader flag is accepted:
  a flag that changes the pipeline changes token ids, so it would
  silently move the running configuration out of the one every
  certification reading was taken against. Flags are rejected at
  construction, not ignored -- see :data:`REJECTED_LOADER_FLAGS`.
- **Local files only.** The crate is handed an absolute path to an
  already-verified file (or the serialization of a locally materialized
  loader object over such files). Nothing here resolves a repository
  id, and nothing here reaches the network; missing files raise
  ``ArtifactNotFound``.
- **Verify what we open.** The manifest digest for the file we read is
  re-checked here as well. The artifacts layer verifies on fetch; this
  check costs one hash of a file we already read and turns
  time-of-check/time-of-use drift into ``ArtifactHashMismatch``.
- **Output-rewriting modes are refused.** An artifact that declares
  truncation or padding cannot produce the core token stream the rest
  of the system stores, so it is rejected with ``UNSUPPORTED_CONFIG``
  rather than quietly re-encoded.

The module imports no accelerator runtime, and the ``tokenizers`` and
``transformers`` imports are deferred to construction so that importing
``toktier`` stays cheap and side-effect free.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from .._oracle import ORACLE_PACKAGE, oracle_version
from ..errors import (
    ArtifactHashMismatch,
    ArtifactNotFound,
    UnsupportedConfig,
)
from ..policy import BACKEND_REFERENCE
from .loader_face import (
    config_added_token_rows,
    live_tokenizer_json,
    load_live_tokenizer,
    verify_declared_config_added_tokens,
)
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


class _CrateTokenizer(Protocol):
    def encode(
        self, sequence: str, add_special_tokens: bool = True
    ) -> Sequence[int]:
        """Encode one sequence."""

    def encode_batch(
        self, input: Sequence[str], add_special_tokens: bool = True
    ) -> Sequence[Sequence[int]]:
        """Encode a batch of sequences."""


def _load_crate_tokenizer(path: Path) -> _CrateTokenizer:
    """Open a verified ``tokenizer.json`` through the native oracle.

    The Rust crate is exactly ``tokenizers==0.22.2``, matching the Python
    package used for certification. Request execution releases the GIL and no
    longer calls through the Python wrapper.
    """
    from .. import _native

    return cast(_CrateTokenizer, _native.ReferenceEngine(str(path)))


def _load_crate_tokenizer_from_json(document: str) -> _CrateTokenizer:
    """Open a materialized loader-face document through the native oracle.

    Same engine, same crate version; the bytes are the exact
    serialization of the live loader object rather than the artifact
    file, which is how configuration-side added tokens reach the
    reference.
    """
    from .. import _native

    return cast(
        _CrateTokenizer,
        _native.ReferenceEngine.from_bytes(document.encode("utf-8")),
    )


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
        loader_face_json: str | None = None,
    ) -> None:
        self._family = family
        self._artifact_sha256 = artifact_sha256
        self._tokenizer_path = tokenizer_path
        self._tokenizer: _CrateTokenizer | None = tokenizer
        self._loader_face_json = loader_face_json

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
                "the reference backend runs the certified subject -- the "
                "artifact plus its declared configuration-side added tokens "
                "-- exactly as recorded; loader flags change token ids and "
                "would leave every certified record behind.",
            )
            raise UnsupportedConfig(
                f"loader flag {name!r} is not available: {note}",
                details={
                    "option": name,
                    "value": loader_flags[name],
                    "reason": "reference backend runs the certified subject "
                    "as recorded",
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
        document = cls._reject_rewriting_sections(path, raw)
        config_rows = config_added_token_rows(path.parent, document)
        verify_declared_config_added_tokens(
            family=artifact.family,
            observed_rows=config_rows,
            declared=getattr(artifact, "config_added_tokens_claim", None),
        )
        if not config_rows:
            # The artifact document already is the loader face's
            # added-token vocabulary; execute the verified bytes directly.
            return cls(
                family=artifact.family,
                artifact_sha256=observed,
                tokenizer_path=path,
                tokenizer=_load_crate_tokenizer(path),
            )
        # The configuration declares added tokens the artifact file does
        # not carry: materialize the live loader object once and execute
        # its exact serialization, the same construction the fast CPU
        # backend certifies against.
        live = load_live_tokenizer(path.parent)
        loader_face = live_tokenizer_json(live)
        return cls(
            family=artifact.family,
            artifact_sha256=observed,
            tokenizer_path=path,
            tokenizer=_load_crate_tokenizer_from_json(loader_face),
            loader_face_json=loader_face,
        )

    @staticmethod
    def _reject_rewriting_sections(path: Path, raw: bytes) -> dict[str, object]:
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
        return document

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

    def materialized_tokenizer_json(self) -> str:
        """The exact tokenizer JSON document this backend executes.

        When the configuration sidecar contributed added tokens, this is
        the serialization of the live loader object captured at
        construction; otherwise it is the verified artifact document
        itself. Consumers that build a second engine (for example the
        facade's decode oracle) construct from this text so that every
        reference face in one process executes the same document.
        """
        if self._loader_face_json is not None:
            return self._loader_face_json
        return self._tokenizer_path.read_text(encoding="utf-8")

    # -- encoding ------------------------------------------------------

    def _live(self) -> _CrateTokenizer:
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("backend is closed")
        return tokenizer

    def native_engine(self) -> object:
        """Return the shared native reference handle for the Rust runtime."""
        return self._live()

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one document to token ids."""
        encoded = self._live().encode(text, add_special_tokens=add_special_tokens)
        return [int(token_id) for token_id in encoded]

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode a batch; row ``i`` equals ``encode(texts[i])``."""
        if not texts:
            return []
        rows = self._live().encode_batch(
            list(texts), add_special_tokens=add_special_tokens
        )
        return [[int(token_id) for token_id in item] for item in rows]

    def close(self) -> None:
        """Release the loaded tokenizer. Idempotent."""
        self._tokenizer = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HfBackend(family={self._family!r}, "
            f"artifact_sha256={self._artifact_sha256[:12]!r})"
        )
