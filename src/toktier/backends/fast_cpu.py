"""Fast CPU backend: a vendored, data-version-pinned native engine.

Contract reference: ``docs/contracts/routing.md`` Section 4 (backend id
``fast_cpu``), ``docs/contracts/registry.md`` Sections 2-3 (status
vocabulary; the entry binds the engine binary digest), and the remediation
record of the pinned build: the engine is derived from
`gigatoken <https://github.com/marcelroed/gigatoken>`_, MIT licensed,
rebuilt as ``0.10.0+toktier.pinned.1`` so that its Unicode data versions
match the reference stack this project certifies against, and shipped as the
private module ``toktier._vendor.gigatoken_rs``.  No separately installed
package named ``gigatoken`` participates in routing.

The certificate binding set has four axes, spelled out in the registry
entry and verified before the backend is planned or opened:

1. engine version (``engine_version``, exact string match against the
   shipped provenance manifest);
2. the engine's Unicode data versions (``engine_unicode_data``,
   declarative: they are properties of the pinned build and are bound
   transitively by the binary digest);
3. oracle version (the record's oracle id; the shared oracle check);
4. patch-set digest (``patch_sha256``, declarative, same transitivity),
   with the engine's native module bound directly by ``binary_digest``.

Any mismatch closes the accelerated entry and the plan degrades to the
reference backend with reason ``R_ENGINE_BINDING_MISMATCH``.

Three loading-surface rules are this backend's own (they are the
package-side form of the judged front-end constraints):

- **Live-object construction only.** The engine's own path/repository
  loading does not see added tokens that exist only in the tokenizer
  configuration file, so this backend never hands the engine a path: the
  tokenizer is materialized as a live Hugging Face object first (via
  ``transformers``, from the verified artifact directory, local files
  only) and the live object is passed to the engine. A path-like value
  offered as a live object is refused with a specific error.
- **No silently ignored options.** The one call surface is the
  :class:`~toktier.backends.protocol.Backend` protocol; ``open`` refuses
  every engine option instead of dropping it, and nothing here forwards
  unknown keyword arguments to the engine.
- **Input validation ahead of the engine.** Text holding a lone
  surrogate cannot be encoded to UTF-8 and is routed to the reference
  backend (a counted ``R_EXEC_FAULT``) rather than handed to the native
  engine; the pinned build additionally validates UTF-8 at its own
  input boundaries, so this pre-check is a second, independent guard.

The engine loads lazily on the first encode. A load failure raises
:class:`~toktier.errors.BackendExecutionFault`, so the routing executor
re-runs the affected input on the reference backend and counts the
degradation; nothing is silently different. This module imports neither
the engine nor ``transformers`` at import time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..errors import (
    ArtifactHashMismatch,
    ArtifactNotFound,
    BackendExecutionFault,
    UnsupportedConfig,
)
from ..policy import BACKEND_FAST_CPU
from .protocol import TOKENIZER_FILE, ArtifactHandle

__all__ = [
    "ENGINE_MODULE",
    "ENGINE_PACKAGE",
    "PINNED_ENGINE_VERSION",
    "FastCpuBackend",
    "FastCpuEngineFacts",
    "fast_cpu_engine_facts",
]

#: Distribution containing the private engine.  Kept as a public constant for
#: diagnostics that historically called this value the engine package.
ENGINE_PACKAGE = "toktier"

#: Private import path of the vendored native module.
ENGINE_MODULE = "toktier._vendor.gigatoken_rs"

#: Version string of the pinned build this project ships certificates
#: for. Informational here: the certified value lives in the registry
#: entry (``engine_version``) and the planner verifies against that, so
#: there is exactly one authoritative copy per record.
PINNED_ENGINE_VERSION = "0.10.0+toktier.pinned.1"

#: File suffixes that identify the engine's native extension module.
_NATIVE_SUFFIXES = (".so", ".pyd", ".dylib")

_VENDOR_DIR = Path(__file__).resolve().parents[1] / "_vendor"
_VENDOR_MANIFEST = _VENDOR_DIR / "gigatoken_build.json"
_VENDOR_SCHEMA = "toktier.vendored_gigatoken.v1"

#: Exception types the engine is expected to raise for input- or
#: state-dependent failures. These become recoverable faults; any other
#: type is an interface regression and propagates.
_ENGINE_FAULTS = (ValueError, RuntimeError)


# ---------------------------------------------------------------------
# probe-side facts
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FastCpuEngineFacts:
    """What the probe can say about the installed engine.

    A field left ``None`` means "not observed", which the planner treats
    as a failed verification -- never as a pass.
    """

    #: Installed distribution version, from package metadata.
    version: str | None = None
    #: SHA-256 of the engine's installed native extension module. The
    #: registry entry binds this as ``binary_digest``.
    binary_digest: str | None = None
    #: SHA-256 of the packaged repair-family table.  This binds the exact
    #: margins, normalizer guards, retry limits and pclass table identity.
    config_digest: str | None = None


def fast_cpu_engine_facts() -> FastCpuEngineFacts:
    """Observe the vendored engine without importing it.

    The version and expected path come from the shipped provenance manifest;
    the executable bytes are always hashed independently.  A missing,
    malformed or self-inconsistent manifest yields empty facts and therefore
    fails the certified route closed.
    """

    repair_table = (
        Path(__file__).resolve().parents[1]
        / "repair"
        / "tables"
        / "fast_repair_families.v1.json"
    )
    try:
        config_digest = hashlib.sha256(repair_table.read_bytes()).hexdigest()
    except OSError:
        config_digest = None
    try:
        manifest = json.loads(_VENDOR_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return FastCpuEngineFacts(config_digest=config_digest)
    if not isinstance(manifest, dict) or manifest.get("schema") != _VENDOR_SCHEMA:
        return FastCpuEngineFacts(config_digest=config_digest)
    if (
        manifest.get("engine") != "gigatoken"
        or manifest.get("engine_version") != PINNED_ENGINE_VERSION
        or manifest.get("delivery") != "vendored"
        or manifest.get("module") != ENGINE_MODULE
    ):
        return FastCpuEngineFacts(config_digest=config_digest)
    native_name = manifest.get("native_file")
    expected_digest = manifest.get("native_sha256")
    if (
        not isinstance(native_name, str)
        or Path(native_name).name != native_name
        or not native_name.endswith(_NATIVE_SUFFIXES)
        or not isinstance(expected_digest, str)
    ):
        return FastCpuEngineFacts(config_digest=config_digest)
    try:
        raw = (_VENDOR_DIR / native_name).read_bytes()
    except OSError:
        return FastCpuEngineFacts(config_digest=config_digest)
    observed_digest = hashlib.sha256(raw).hexdigest()
    if observed_digest != expected_digest:
        return FastCpuEngineFacts(config_digest=config_digest)
    return FastCpuEngineFacts(
        version=PINNED_ENGINE_VERSION,
        binary_digest=observed_digest,
        config_digest=config_digest,
    )


# ---------------------------------------------------------------------
# engine call surface
# ---------------------------------------------------------------------


class _EngineTokenizer(Protocol):
    """The slice of the engine tokenizer object this backend calls."""

    def encode(self, text: str) -> Sequence[int]:
        """Core-stream token ids for one document."""

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        """Per-document id rows for one batch."""

    @property
    def vocab(self) -> Mapping[int, bytes]:
        """Raw byte vocabulary used to reconstruct window spans."""

    @property
    def vocab_size(self) -> int:
        """Number of vocabulary ids addressable by the engine."""


class _NativeEngine(Protocol):
    def encode_batch_list(
        self, texts: list[str], *, parallel: bool
    ) -> list[list[int]]: ...

    @property
    def vocab(self) -> Mapping[int, bytes]: ...

    @property
    def vocab_size(self) -> int: ...


class _EngineFactory(Protocol):
    """Builds the engine tokenizer from a live HF tokenizer object."""

    def __call__(self, hf_tokenizer: object) -> _EngineTokenizer:
        """Construct the engine over the live object."""


class _VendoredEngine:
    """Small TokTier-owned adapter over the judged native call surface.

    ``encode_batch_list`` returns ordinary Python lists and therefore avoids
    Gigatoken's optional NumPy/Awkward result adapters.  The core wheel ships
    only the native module; its unrelated CLI, loaders and compatibility
    wrappers are deliberately absent.
    """

    def __init__(self, native: _NativeEngine) -> None:
        self._native = native

    def encode(self, text: str) -> Sequence[int]:
        rows = self._native.encode_batch_list([text], parallel=False)
        return rows[0]

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        return self._native.encode_batch_list(list(texts), parallel=True)

    @property
    def vocab(self) -> Mapping[int, bytes]:
        return self._native.vocab

    @property
    def vocab_size(self) -> int:
        return int(self._native.vocab_size)


def _live_tokenizer_json(hf_tokenizer: object) -> str:
    """Serialize the already materialized fast HF tokenizer."""
    backend = getattr(hf_tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        to_str = getattr(hf_tokenizer, "to_str", None)
    if not callable(to_str):
        raise UnsupportedConfig(
            "the live fast tokenizer cannot serialize its backend",
            details={
                "option": "hf_tokenizer",
                "value": type(hf_tokenizer).__name__,
                "reason": "serializable live backend required",
            },
        )
    data = to_str()
    if not isinstance(data, str):
        raise UnsupportedConfig(
            "the live fast tokenizer returned a non-text serialization",
            details={
                "option": "hf_tokenizer",
                "value": type(data).__name__,
                "reason": "tokenizer JSON text required",
            },
        )
    return data


def _default_engine_factory(hf_tokenizer: object) -> _EngineTokenizer:
    """Lazily import the private native module and load the live tokenizer."""
    from importlib import import_module

    engine = import_module(ENGINE_MODULE)
    native = engine.load_hf_json(_live_tokenizer_json(hf_tokenizer))
    return _VendoredEngine(native)


#: Name of the loader-side configuration sidecar; the file that can
#: declare added tokens the artifact itself does not carry.
_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


def _config_only_added_tokens(root: Path) -> list[str]:
    """Added-token literals declared only in the configuration sidecar.

    These are the tokens a ``tokenizer.json``-only construction cannot
    see. The list gates the loading fallback below: it may only be
    taken when this list is empty, because a fallback that silently
    dropped an added token would encode differently from the reference.
    """
    config_path = root / _TOKENIZER_CONFIG_FILE
    if not config_path.is_file():
        return []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder = config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        return []
    declared = [
        str(item["content"])
        for item in decoder.values()
        if isinstance(item, dict) and "content" in item
    ]
    if not declared:
        return []
    artifact = json.loads((root / TOKENIZER_FILE).read_text(encoding="utf-8"))
    carried = {
        token.get("content")
        for token in artifact.get("added_tokens") or ()
        if isinstance(token, dict)
    }
    vocabulary = artifact.get("model", {}).get("vocab", {})
    return [
        content
        for content in declared
        if content not in carried and content not in vocabulary
    ]


def _load_live_tokenizer(root: Path) -> object:
    """Materialize the live HF tokenizer from a verified directory.

    Uses the base installation's pinned ``transformers`` with local files
    only: the artifact was verified by the artifacts layer, and nothing
    here reaches the network. Loading through ``transformers`` is what
    makes configuration-only added tokens visible; the engine is then
    handed the live object, never a path.

    Some artifact configurations name loader classes the installed
    ``transformers`` does not know (for example a ``tokenizer_class``
    from a newer release). For those, the documented fallback is a
    ``PreTrainedTokenizerFast`` over the artifact file alone -- taken
    only after verifying that the configuration declares no added token
    the artifact does not carry, since that is exactly what a
    file-only construction cannot see. When the verification fails, the
    original loading error propagates (and surfaces as a recoverable
    fault, so the input runs on the reference backend).
    """
    from importlib import import_module

    transformers = import_module("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(root), use_fast=True, local_files_only=True
        )
    except Exception:
        if _config_only_added_tokens(root):
            raise
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_file=str(root / TOKENIZER_FILE)
        )
    if not getattr(tokenizer, "is_fast", False):
        raise UnsupportedConfig(
            "the loaded tokenizer object is not a fast tokenizer; the "
            "engine consumes the fast backend object",
            details={
                "option": "tokenizer",
                "value": type(tokenizer).__name__,
                "reason": "fast tokenizer object required",
            },
        )
    return tokenizer


def _require_live_object(candidate: object) -> object:
    """The live-object construction rule, applied to injected objects."""
    if isinstance(candidate, (str, bytes)) or hasattr(candidate, "__fspath__"):
        raise UnsupportedConfig(
            "path or repository-id construction is not offered by the fast "
            "CPU backend: the engine's own path loading does not see added "
            "tokens that exist only in the tokenizer configuration file. "
            "Load the tokenizer first and pass the live object.",
            details={
                "option": "hf_tokenizer",
                "value": str(candidate),
                "reason": "live-object construction only",
            },
        )
    if not hasattr(candidate, "get_vocab"):
        raise UnsupportedConfig(
            "expected a live HF tokenizer object; added tokens are "
            "materialized by the loader and supplied to the engine",
            details={
                "option": "hf_tokenizer",
                "value": type(candidate).__name__,
                "reason": "live-object construction only",
            },
        )
    return candidate


# ---------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------


class FastCpuBackend:
    """The fast CPU backend (backend id ``fast_cpu``).

    Constructed from a verified :class:`ArtifactHandle`, like the
    reference backend: there is no constructor taking an unverified
    path. The engine itself loads lazily on the first encode; a load
    failure is a recoverable fault the executor routes around, so a
    machine where the engine cannot serve this family still returns
    reference results, counted.
    """

    def __init__(
        self,
        *,
        family: str,
        artifact_sha256: str,
        root: Path,
        adds_special_tokens: bool,
        hf_tokenizer: object | None,
        engine_factory: _EngineFactory,
    ) -> None:
        self._family = family
        self._artifact_sha256 = artifact_sha256
        self._root = root
        self._adds_special_tokens = adds_special_tokens
        self._hf_tokenizer = hf_tokenizer
        self._engine_factory = engine_factory
        self._engine: _EngineTokenizer | None = None
        self._load_error: BackendExecutionFault | None = None
        self._closed = False

    # -- construction --------------------------------------------------

    @classmethod
    def open(
        cls,
        artifact: ArtifactHandle,
        *,
        hf_tokenizer: object | None = None,
        engine_factory: _EngineFactory | None = None,
        engine_options: Mapping[str, object] | None = None,
    ) -> FastCpuBackend:
        """Open the fast CPU backend over a verified artifact.

        ``hf_tokenizer`` optionally injects an already-loaded live
        tokenizer object (it is checked against the live-object rule);
        by default the object is materialized from the verified artifact
        directory. ``engine_options`` exists only so that a caller who
        tries to pass one receives a documented refusal instead of
        having the option silently dropped -- the engine runs in exactly
        the configuration the certificates were judged in.
        """
        if engine_options:
            name = sorted(engine_options)[0]
            raise UnsupportedConfig(
                f"engine option {name!r} is not available: the fast CPU "
                "backend runs the engine exactly as judged, and an option "
                "that changes behavior would leave every certified record "
                "behind",
                details={
                    "option": name,
                    "value": engine_options[name],
                    "reason": "fast CPU backend runs the engine as judged",
                },
            )
        if hf_tokenizer is not None:
            _require_live_object(hf_tokenizer)

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
        document = json.loads(raw.decode("utf-8"))
        inserts = (
            isinstance(document, dict)
            and document.get("post_processor") is not None
        )
        return cls(
            family=artifact.family,
            artifact_sha256=observed,
            root=path.parent,
            adds_special_tokens=bool(inserts),
            hf_tokenizer=hf_tokenizer,
            engine_factory=engine_factory or _default_engine_factory,
        )

    # -- identity ------------------------------------------------------

    @property
    def backend_id(self) -> str:
        """Frozen backend identifier of the fast CPU path."""
        return BACKEND_FAST_CPU

    @property
    def family(self) -> str:
        """Family id this backend was opened for."""
        return self._family

    @property
    def artifact_sha256(self) -> str:
        """Digest of the artifact bytes this backend executes."""
        return self._artifact_sha256

    # -- engine lifecycle ----------------------------------------------

    def _live(self) -> _EngineTokenizer:
        if self._closed:
            raise RuntimeError("backend is closed")
        if self._engine is not None:
            return self._engine
        if self._load_error is not None:
            # The first load failed; the outcome will not change within
            # this process, so the recorded fault is re-raised cheaply
            # and every affected input is still counted by the executor.
            raise self._load_error
        try:
            live = (
                self._hf_tokenizer
                if self._hf_tokenizer is not None
                else _load_live_tokenizer(self._root)
            )
            live = _require_live_object(live)
            self._hf_tokenizer = live
            self._engine = self._engine_factory(live)
        except Exception as exc:
            fault = BackendExecutionFault(
                f"fast CPU engine failed to load family "
                f"{self._family!r}: {exc}",
                details={
                    "backend": BACKEND_FAST_CPU,
                    "stage": "engine_load",
                    "family": self._family,
                    "error": type(exc).__name__,
                },
            )
            self._load_error = fault
            raise fault from exc
        return self._engine

    def repair_components(self) -> tuple[_EngineTokenizer, object]:
        """Return the long-lived engine and the exact live HF object.

        Session repair uses the engine for window encodes and the HF object
        only for construction-time vocabulary/normalizer guards.  Keeping the
        two objects paired prevents a caller from accidentally certifying a
        Gigatoken instance against a different tokenizer configuration.
        """
        engine = self._live()
        live = self._hf_tokenizer
        if live is None:  # pragma: no cover - guarded by _live
            raise RuntimeError("fast CPU backend lost its live tokenizer")
        return engine, live

    def _require_core_stream(self, add_special_tokens: bool) -> None:
        if add_special_tokens and self._adds_special_tokens:
            raise BackendExecutionFault(
                "the artifact's post-processor inserts special tokens; the "
                "fast CPU engine produces the core stream, so this input "
                "runs on the reference backend",
                details={
                    "backend": BACKEND_FAST_CPU,
                    "stage": "add_special_tokens",
                },
            )

    @staticmethod
    def _require_utf8(text: str) -> None:
        """Route text the engine must not see to the reference backend.

        A lone surrogate has no UTF-8 encoding; the reference backend
        defines the correct output for such input, so the input is
        raised as a recoverable fault rather than either guessed at or
        handed to the native engine.
        """
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise BackendExecutionFault(
                f"input holds a lone surrogate at index {exc.start}; the "
                "input runs on the reference backend",
                details={
                    "backend": BACKEND_FAST_CPU,
                    "stage": "input_validation",
                    "index": exc.start,
                },
            ) from None

    # -- encoding ------------------------------------------------------

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one document to token ids."""
        self._require_core_stream(add_special_tokens)
        self._require_utf8(text)
        engine = self._live()
        try:
            return [int(token_id) for token_id in engine.encode(text)]
        except _ENGINE_FAULTS as exc:
            raise BackendExecutionFault(
                f"fast CPU encode failed: {exc}",
                details={"backend": BACKEND_FAST_CPU, "stage": "encode"},
            ) from exc

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode a batch; row ``i`` equals ``encode(texts[i])``."""
        self._require_core_stream(add_special_tokens)
        if not texts:
            return []
        for text in texts:
            self._require_utf8(text)
        engine = self._live()
        try:
            rows = engine.encode_batch(list(texts))
            return [[int(token_id) for token_id in row] for row in rows]
        except _ENGINE_FAULTS as exc:
            raise BackendExecutionFault(
                f"fast CPU batch encode failed: {exc}",
                details={"backend": BACKEND_FAST_CPU, "stage": "encode_batch"},
            ) from exc

    def close(self) -> None:
        """Release the engine references. Idempotent."""
        self._closed = True
        self._engine = None
        self._hf_tokenizer = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FastCpuBackend(family={self._family!r}, "
            f"artifact_sha256={self._artifact_sha256[:12]!r})"
        )
