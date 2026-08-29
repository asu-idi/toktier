"""Fast CPU backend: an integrated, data-version-pinned native engine.

Contract reference: ``docs/contracts/routing.md`` Section 4 (backend id
``fast_cpu``), ``docs/contracts/registry.md`` Sections 2-3 (status vocabulary;
the entry binds the integrated engine source/build identity), and the
remediation record of the pinned build: the engine is derived from
`gigatoken <https://github.com/marcelroed/gigatoken>`_, MIT licensed,
rebuilt as ``0.10.0+toktier.pinned.1`` so that its Unicode data versions
match the reference stack this project certifies against, and linked directly
into ``toktier._native``. No separately installed package named ``gigatoken``
participates in routing.

The certificate binding set has four axes, spelled out in the registry
entry and verified before the backend is planned or opened:

1. engine version (``engine_version``, exact string match against the
   shipped provenance manifest);
2. the engine's Unicode data versions (``engine_unicode_data``), bound by the
   integrated source identity and pinned provenance record;
3. oracle version (the record's oracle id; the shared oracle check);
4. patch-set digest (``patch_sha256``), with the compiled implementation
   bound by a domain-separated ``source_digest``, exact Rust toolchain, and
   release build flags reported by the extension itself.

Any mismatch closes the accelerated entry and the plan degrades to the
reference backend with reason ``R_ENGINE_BINDING_MISMATCH``.

Three loading-surface rules are this backend's own (they are the
package-side form of the judged front-end constraints):

- **Exact materialization.** When ``tokenizer.json`` already carries every
  added token, the integrated engine consumes those verified bytes directly
  and shares the Rust HF reference parsed from the same digest. If
  ``tokenizer_config.json`` contributes an otherwise missing token, the
  backend instead materializes a live Hugging Face object locally and
  serializes that exact live state. A caller-injected tokenizer must likewise
  be a live object; path-like substitutes are refused.
- **No silently ignored options.** The one call surface is the
  :class:`~toktier.backends.protocol.Backend` protocol; ``open`` refuses
  every engine option instead of dropping it, and nothing here forwards
  unknown keyword arguments to the engine.
- **Input validation ahead of the engine.** Text holding a lone
  surrogate cannot be encoded to UTF-8 and is routed to the reference
  backend (a counted ``R_EXEC_FAULT``) rather than handed to the native
  engine; the pinned build additionally validates UTF-8 at its own
  input boundaries, so this pre-check is a second, independent guard.

The single corrected-Gigatoken core is validated when the native runtime is
constructed, while payload-sized batch worker caches remain lazy. A load failure raises
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
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from .._native import ReferenceEngine

from ..errors import (
    ArtifactHashMismatch,
    ArtifactNotFound,
    BackendExecutionFault,
    UnsupportedConfig,
)
from ..policy import BACKEND_FAST_CPU
from .loader_face import (
    config_added_token_rows as _config_added_token_rows,
)
from .loader_face import (
    live_tokenizer_json as _live_tokenizer_json,
)
from .loader_face import (
    load_live_tokenizer as _load_live_tokenizer,
)
from .loader_face import (
    verify_declared_config_added_tokens as _verify_declared_config_added_tokens,
)
from .protocol import TOKENIZER_FILE, ArtifactHandle

__all__ = [
    "ENGINE_DELIVERY",
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

#: Private import path of the extension that owns the integrated engine.
ENGINE_MODULE = "toktier._native"

#: The corrected implementation is linked into the core extension rather
#: than imported from a second private extension module.
ENGINE_DELIVERY = "integrated"

#: Version string of the pinned build this project ships certificates
#: for. Informational here: the certified value lives in the registry
#: entry (``engine_version``) and the planner verifies against that, so
#: there is exactly one authoritative copy per record.
PINNED_ENGINE_VERSION = "0.10.0+toktier.pinned.1"

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
    #: Legacy binary identity slot. Integrated source-certified builds leave
    #: it unset; retained so injected older probe fixtures fail closed rather
    #: than changing shape abruptly.
    binary_digest: str | None = None
    #: Domain-separated digest of every source/build input that can affect
    #: corrected full encode or append repair.
    source_digest: str | None = None
    #: Exact release build description emitted by Cargo's build script.
    build_flags: tuple[str, ...] = ()
    #: Exact Rust compiler identity used for this extension.
    toolchain: str | None = None
    #: SHA-256 of the packaged repair-family table.  This binds the exact
    #: margins, normalizer guards, retry limits and pclass table identity.
    config_digest: str | None = None


def fast_cpu_engine_facts() -> FastCpuEngineFacts:
    """Read the integrated engine identity emitted by the native build.

    Importing the core extension executes no tokenizer work. A missing or
    malformed fact closes the certified route; package metadata is never
    allowed to speak on behalf of different executing code.
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
        from .. import _native

        observed: object = _native.fast_cpu_build_facts()
    except (ImportError, RuntimeError, TypeError, ValueError):
        return FastCpuEngineFacts(config_digest=config_digest)
    if not isinstance(observed, Mapping):
        return FastCpuEngineFacts(config_digest=config_digest)
    source_digest = observed.get("source_digest")
    build_flags = observed.get("build_flags")
    toolchain = observed.get("toolchain")
    if (
        observed.get("engine") != "gigatoken"
        or observed.get("engine_version") != PINNED_ENGINE_VERSION
        or observed.get("engine_delivery") != ENGINE_DELIVERY
        or observed.get("engine_module") != ENGINE_MODULE
        or not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
        or not isinstance(build_flags, list)
        or not build_flags
        or not all(isinstance(value, str) for value in build_flags)
        or not isinstance(toolchain, str)
        or not toolchain
    ):
        return FastCpuEngineFacts(config_digest=config_digest)
    return FastCpuEngineFacts(
        version=PINNED_ENGINE_VERSION,
        source_digest=source_digest,
        build_flags=tuple(build_flags),
        toolchain=toolchain,
        config_digest=config_digest,
    )


# ---------------------------------------------------------------------
# engine call surface
# ---------------------------------------------------------------------


class _EngineTokenizer(Protocol):
    """The slice of the engine tokenizer object this backend calls."""

    def encode(self, text: str) -> Sequence[int]:
        """Core-stream token ids for one document."""

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Per-document id rows for one batch."""

    @property
    def vocab(self) -> Mapping[int, bytes]:
        """Raw byte vocabulary used to reconstruct window spans."""

    @property
    def vocab_size(self) -> int:
        """Number of vocabulary ids addressable by the engine."""


class _EngineFactory(Protocol):
    """Builds the engine tokenizer from a live HF tokenizer object."""

    def __call__(self, hf_tokenizer: object) -> _EngineTokenizer:
        """Construct the engine over the live object."""


def _native_engine_factory(family: str, artifact_sha256: str) -> _EngineFactory:
    """Build the corrected engine inside TokTier's one-call Rust runtime."""

    def build(hf_tokenizer: object) -> _EngineTokenizer:
        from .. import _native
        from ..repair.registry import family_spec, pclass_table

        spec = family_spec(family, artifact_sha256)
        if spec is None:
            raise UnsupportedConfig(
                "the artifact is not in the certified repair roster",
                details={
                    "backend": BACKEND_FAST_CPU,
                    "family": family,
                    "artifact_sha256": artifact_sha256,
                },
            )
        return _native.CallbackEncoder.native_fast_cpu(
            _live_tokenizer_json(hf_tokenizer).encode("utf-8"),
            spec.family,
            spec.artifact_sha256,
            spec.margin,
            spec.effective_l_max,
            spec.has_normalizer,
            pclass_table(),
        )

    return build


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
        artifact_json: bytes,
        config_only_added_tokens: tuple[str, ...],
        adds_special_tokens: bool,
        hf_tokenizer: object | None,
        engine_factory: _EngineFactory,
        integrated_factory: bool,
    ) -> None:
        self._family = family
        self._artifact_sha256 = artifact_sha256
        self._root = root
        self._artifact_json = artifact_json
        self._config_only_added_tokens = config_only_added_tokens
        self._adds_special_tokens = adds_special_tokens
        self._hf_tokenizer = hf_tokenizer
        self._engine_factory = engine_factory
        self._integrated_factory = integrated_factory
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

        ``hf_tokenizer`` optionally injects an already-loaded live tokenizer
        object (it is checked against the live-object rule). By default the
        integrated engine uses the verified artifact bytes, escalating to a
        local live-object load only for configuration-only added tokens.
        ``engine_options`` exists only so that a caller who
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
        if not isinstance(document, Mapping):
            raise UnsupportedConfig(
                f"{TOKENIZER_FILE} is not a JSON object",
                details={
                    "option": TOKENIZER_FILE,
                    "value": str(path),
                    "reason": "unexpected artifact shape",
                },
            )
        config_rows = _config_added_token_rows(path.parent, document)
        _verify_declared_config_added_tokens(
            family=artifact.family,
            observed_rows=config_rows,
            declared=getattr(artifact, "config_added_tokens_claim", None),
        )
        inserts = document.get("post_processor") is not None
        integrated_factory = engine_factory is None
        return cls(
            family=artifact.family,
            artifact_sha256=observed,
            root=path.parent,
            artifact_json=raw,
            config_only_added_tokens=tuple(
                str(row["content"]) for row in config_rows
            ),
            adds_special_tokens=bool(inserts),
            hf_tokenizer=hf_tokenizer,
            engine_factory=(
                engine_factory
                if engine_factory is not None
                else _native_engine_factory(artifact.family, observed)
            ),
            integrated_factory=integrated_factory,
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
                f"fast CPU engine failed to load family {self._family!r}: {exc}",
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

    def materialized_tokenizer_json(self) -> str:
        """Return the exact tokenizer JSON without loading Gigatoken.

        The verified artifact is already the exact live document when no
        configuration-only added token exists, so the common certified path
        reads it directly and avoids importing ``transformers``. An injected
        live object, or a sidecar that contributes an otherwise missing added
        token, is serialized through the Hugging Face loader instead. Both
        branches preserve the same added-token contract.
        """
        if self._closed:
            raise RuntimeError("backend is closed")
        try:
            live = self._hf_tokenizer
            if live is None and not self._config_only_added_tokens:
                return self._artifact_json.decode("utf-8")
            if live is None:
                live = _load_live_tokenizer(self._root)
                live = _require_live_object(live)
                self._hf_tokenizer = live
            return _live_tokenizer_json(live)
        except Exception as exc:
            fault = BackendExecutionFault(
                f"fast CPU live tokenizer failed to load family "
                f"{self._family!r}: {exc}",
                details={
                    "backend": BACKEND_FAST_CPU,
                    "stage": "native_engine_materialization",
                    "family": self._family,
                    "error": type(exc).__name__,
                },
            )
            self._load_error = fault
            raise fault from exc

    def native_session_engine(self, reference: object | None = None) -> object | None:
        """Return the shared Rust engine when the default factory owns it."""
        if self._integrated_factory and self._engine is None:
            from .. import _native
            from ..repair.registry import family_spec, pclass_table

            spec = family_spec(self._family, self._artifact_sha256)
            if spec is None:
                return None
            shared_reference = cast(
                "ReferenceEngine | None",
                reference if not self._config_only_added_tokens else None,
            )
            self._engine = _native.CallbackEncoder.native_fast_cpu(
                self.materialized_tokenizer_json().encode("utf-8"),
                spec.family,
                spec.artifact_sha256,
                spec.margin,
                spec.effective_l_max,
                spec.has_normalizer,
                pclass_table(),
                shared_reference,
            )
            return self._engine
        engine = self._live()
        return engine if getattr(engine, "native_request_path", False) else None

    @property
    def postprocessor_adds_tokens(self) -> bool:
        """Whether the verified artifact inserts tokens when requested."""
        return self._adds_special_tokens

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
