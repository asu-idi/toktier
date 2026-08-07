"""The facade: load a family once, then use it like a tokenizer.

Contract reference: ``docs/contracts/facade.md`` (0.x surface). The
facade is a thin composition of shipped pieces: artifacts resolve
through the manifest and the verified cache, routing follows the
standing policy semantics (GPU and corrected CPU backends are selected
only for exact certified bindings), and the session/lookup paths run on
the entry store. Every path returns ids equal to a
from-scratch reference encode; store layers can only decline to serve,
never answer differently.

Heavy work is deferred: importing this module loads no oracle and no
native store. The oracle loads when the tokenizer is constructed (the
reference backend executes it); the native store loads on the first
session or content-lookup call, and the GPU engine loads only when an
eligible input crosses the configured byte threshold.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol, cast

from .._oracle import ORACLE_PACKAGE, import_oracle, oracle_version
from ..artifacts import ArtifactManifest, ArtifactStore, HuggingFaceSource
from ..artifacts.tables import ARTIFACT_MANIFEST
from ..backends.fast_cpu import FastCpuBackend
from ..backends.hf import HfBackend
from ..backends.protocol import TOKENIZER_FILE
from ..config import Config
from ..engine.gpu.backend import GpuBackend, LazyGpuBackend
from ..engine.gpu.host_probe import CudaHostProbe
from ..errors import BackendExecutionFault, BackendUnavailable, UnsupportedConfig
from ..frontend.added import AddedTokenFrontend
from ..policy import (
    BACKEND_FAST_CPU,
    BACKEND_GPU,
    BACKEND_REFERENCE,
    PlanReason,
    ReasonCode,
    RoutePlan,
    RoutingPolicy,
)
from ..repair.fastokens import FastokensFullRepair
from ..repair.gigatoken import GigatokenRepair, WindowUnsupported
from ..repair.registry import family_spec
from ..routing.added_route import AddedTokenRouter
from ..routing.execute import RoutedExecutor
from ..routing.explain import build_explanation
from ..routing.plan import assessments_for
from ..routing.plan import plan as build_plan
from ..routing.probe import probe
from ..routing.registry_load import shipped_registry
from ..routing.registry_view import ArtifactRecord
from .store import DEFAULT_CACHE_BUDGET_BYTES, EntryStore

__all__ = ["Encoding", "Tokenizer", "load"]

#: Engine identity bound into the semantic fingerprint. Changes when the
#: session-path engine semantics change; stored sessions then miss and
#: re-encode instead of being reused across meanings.
_ENGINE_ID = "facade-session/v3"

#: Session API semantic version (versioning.md Section 2).
_SESSION_SEMVER = "1"

_LOOKUP_VALUES = (None, "auto", "off")
_REPAIR_BACKENDS = ("auto", "reference", "fastokens")
_DEVICES = ("auto", "cpu", "cuda")
_GPU_DELIVERIES = ("auto", "prebuilt", "jit")

#: Measured crossover used by the automatic facade. It is public through
#: ``gpu_min_bytes`` and ``explain()`` rather than hidden in an engine.
DEFAULT_GPU_MIN_BYTES = 64 * 1024

_SessionRepair = GigatokenRepair | FastokensFullRepair


class _RouteEveryInput:
    """Fail-closed scanner for an added-token shape we cannot model."""

    def scan(self, text: str) -> list[tuple[str, int | None]]:
        return [(text, None)]


def _module_present(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _resolve_gpu_delivery(requested: str) -> str:
    """Resolve an installation profile to one concrete lazy delivery."""
    if requested not in _GPU_DELIVERIES:
        raise ValueError(
            f"gpu_delivery must be one of {_GPU_DELIVERIES!r}, not {requested!r}"
        )
    if requested != "auto":
        return requested
    # Extras do not leave a runtime flag in Python package metadata. Ninja
    # is intentionally unique to ``gpu-jit`` and therefore serves as its
    # profile marker; ``gpu`` has torch but not ninja and selects prebuilt.
    return "jit" if _module_present("ninja") else "prebuilt"


def _jit_toolchain_rejection(route_plan: RoutePlan) -> PlanReason | None:
    """The specific fail-closed JIT toolchain reason, when present."""
    return next(
        (
            reason
            for reason in route_plan.reasons
            if reason.backend == BACKEND_GPU
            and reason.code is ReasonCode.R_UNCERTIFIED_ARTIFACT
            and reason.detail.get("cause") == "toolchain_unverified"
        ),
        None,
    )


def _uncertified_jit_remedy(family: str) -> str:
    """Explicit CLI opt-in for an unjudged JIT toolchain."""
    return f"toktier gpu compile {family} --accept-uncertified-jit"


class _OracleTokenizer(Protocol):
    """The slice of the oracle tokenizer object the facade calls."""

    def encode(self, sequence: str, add_special_tokens: bool = True) -> Any:
        """Encode one sequence; the result carries ``ids`` and ``offsets``."""

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode ids back to text."""


@dataclass(frozen=True)
class _ResolvedArtifact:
    """Verified artifact handle over the cache directory."""

    family: str
    root: Path
    artifact_sha256: str
    files: Mapping[str, str]

    def path(self, relative_name: str) -> Path:
        return self.root / relative_name


@dataclass(frozen=True)
class Encoding:
    """Result of one encode. ``ids`` is the token id sequence."""

    ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ids", tuple(self.ids))

    def __len__(self) -> int:
        return len(self.ids)


class Tokenizer:
    """One loaded family behind the facade surface.

    Construction resolves and verifies the artifact, probes the machine
    and fixes an immutable route plan. The optional ``store`` names a
    directory for persistent sessions; without it, session and lookup
    state lives in this process only.
    """

    def __init__(
        self,
        family: str,
        config: Config | None = None,
        *,
        policy: RoutingPolicy | str | None = None,
        store: str | os.PathLike[str] | None = None,
        device: str = "auto",
        manifest: ArtifactManifest | None = None,
        cache_budget_bytes: int | None = None,
        repair_backend: str = "auto",
        gpu_delivery: str = "auto",
        gpu_min_bytes: int = DEFAULT_GPU_MIN_BYTES,
    ) -> None:
        if device not in _DEVICES:
            raise ValueError(f"device must be one of {_DEVICES!r}, not {device!r}")
        if type(gpu_min_bytes) is not int or gpu_min_bytes < 0:
            raise ValueError("gpu_min_bytes must be a non-negative integer")
        self._config = config if config is not None else Config.resolve()
        self._device_request = device
        self._gpu_delivery_request = gpu_delivery
        self._gpu_delivery = _resolve_gpu_delivery(gpu_delivery)
        self._gpu_min_bytes = gpu_min_bytes
        resolved_policy = (
            RoutingPolicy.coerce(policy)
            if policy is not None
            else self._config.routing_policy
        )
        if repair_backend not in _REPAIR_BACKENDS:
            raise ValueError(
                f"repair_backend must be one of {_REPAIR_BACKENDS!r}, "
                f"not {repair_backend!r}"
            )
        if (
            repair_backend == "fastokens"
            and resolved_policy is not RoutingPolicy.EXPERIMENTAL
        ):
            raise UnsupportedConfig(
                "Fastokens is an explicit experimental repair backend and "
                "cannot be selected by a certified policy",
                details={
                    "option": "repair_backend",
                    "value": repair_backend,
                    "required_policy": RoutingPolicy.EXPERIMENTAL.value,
                    "exact_id_guarantee": False,
                },
            )
        self._repair_backend_request = repair_backend
        active_manifest = (
            manifest
            if manifest is not None
            else ArtifactManifest.load(ARTIFACT_MANIFEST)
        )
        entry = active_manifest.get(family)
        verified = ArtifactStore(
            active_manifest, config=self._config, source=HuggingFaceSource()
        ).ensure(family)
        self._artifact_sha256 = entry.file(TOKENIZER_FILE).sha256
        handle = _ResolvedArtifact(
            family=entry.family,
            root=verified.directory,
            artifact_sha256=self._artifact_sha256,
            files={item.name: item.sha256 for item in entry.files},
        )
        self._artifact_handle = handle
        self._backend = HfBackend.open(handle)
        document = json.loads(self._backend.tokenizer_path.read_text(encoding="utf-8"))
        added_token_rows = document.get("added_tokens") or []
        self._seal_end_guard_chars = max(
            (
                len(str(row.get("content", "")))
                for row in added_token_rows
                if isinstance(row, Mapping)
            ),
            default=0,
        )
        try:
            frontend = AddedTokenFrontend(
                {
                    "family": entry.family,
                    "normalizer": document.get("normalizer"),
                    "added_tokens": document.get("added_tokens") or [],
                }
            )
        except UnsupportedConfig:
            # A scanner that cannot prove a miss routes every document to
            # the reference. This costs speed only and cannot change ids.
            self._added_router = AddedTokenRouter(_RouteEveryInput())
        else:
            self._added_router = AddedTokenRouter(frontend)
        self._registry = shipped_registry()
        device_probe = (
            CudaHostProbe(
                config=self._config,
                delivery=self._gpu_delivery,
            )
            if device in ("auto", "cuda")
            and resolved_policy is not RoutingPolicy.REFERENCE
            else None
        )
        self._snapshot = probe(
            family=entry.family,
            registry=self._registry,
            artifact_sha256=self._artifact_sha256,
            device_probe=device_probe,
        )
        self._plan = build_plan(
            self._snapshot, resolved_policy, self._registry, self._config
        )
        jit_rejection = (
            _jit_toolchain_rejection(self._plan)
            if self._gpu_delivery == "jit"
            else None
        )
        if jit_rejection is not None and device == "auto":
            observed = jit_rejection.detail.get("observed", "unknown")
            constraint = jit_rejection.detail.get("constraint", "unknown")
            warnings.warn(
                "TokTier refused uncertified JIT acceleration: observed "
                f"{observed}; certified constraint: {constraint}. Requests "
                f"will use {self._plan.backend!r}. To compile and evaluate "
                "this combination anyway, run `"
                f"{_uncertified_jit_remedy(entry.family)}`. That explicit "
                "opt-in runs the JIT combination outside TokTier's certified "
                "exact-ID guarantee.",
                RuntimeWarning,
                stacklevel=2,
            )
        if device == "cuda" and self._plan.backend != BACKEND_GPU:
            reason = next(
                (item for item in self._plan.reasons if item.backend == BACKEND_GPU),
                None,
            )
            details: dict[str, object] = {
                "backend": BACKEND_GPU,
                "reason_code": reason.code.value if reason else None,
                "reason": dict(reason.detail) if reason else {},
            }
            message = (
                "device='cuda' requires an eligible GPU route, but the "
                "certified planner closed it"
            )
            if jit_rejection is not None:
                remedy = _uncertified_jit_remedy(entry.family)
                details["remedy"] = remedy
                message += (
                    "; to compile this unjudged combination for explicit "
                    f"evaluation, run `{remedy}`"
                )
            self._backend.close()
            raise BackendUnavailable(message, details=details)
        routed_backends: dict[str, Any] = {BACKEND_REFERENCE: self._backend}
        self._fast_backend: FastCpuBackend | None = None
        if BACKEND_FAST_CPU in self._plan.fallback_chain:
            self._fast_backend = FastCpuBackend.open(handle)
            routed_backends[BACKEND_FAST_CPU] = self._fast_backend
        self._gpu_engine: Any | None = None
        self._gpu_backend: LazyGpuBackend | None = None
        self._gpu_device = self._select_gpu_device()
        if BACKEND_GPU in self._plan.fallback_chain:
            self._gpu_backend = LazyGpuBackend(self._open_gpu_backend)
            routed_backends[BACKEND_GPU] = self._gpu_backend
        thresholds = (
            {BACKEND_GPU: self._gpu_min_bytes}
            if BACKEND_GPU in self._plan.fallback_chain
            else None
        )
        self._executor = RoutedExecutor(
            self._plan,
            routed_backends,
            added_router=self._added_router,
            diagnostics=self._config.diagnostics,
            minimum_input_bytes=thresholds,
        )
        self._store_directory = (
            Path(os.fspath(store)).expanduser() if store is not None else None
        )
        self._cache_budget = (
            DEFAULT_CACHE_BUDGET_BYTES
            if cache_budget_bytes is None
            else cache_budget_bytes
        )
        self._oracle_handle: _OracleTokenizer | None = None
        self._entry_store: EntryStore | None = None
        self._session_repair: _SessionRepair | None = None
        self._session_repair_initialised = False
        self._session_repair_guard: dict[str, object] | None = None
        self._state_encode_counts: dict[str, int] = {}
        self._last_state_encode: dict[str, object] | None = None

    # -- identity ------------------------------------------------------

    @property
    def family(self) -> str:
        """Canonical family id this object was loaded for."""
        return self._backend.family

    @property
    def plan(self) -> RoutePlan:
        """The immutable route plan fixed at construction."""
        return self._plan

    # -- encoding ------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        session: str | None = None,
        lookup: str | None = None,
        add_special_tokens: bool = False,
    ) -> Encoding:
        """Encode one document.

        ``session`` names a store entry: when its stored text is a
        prefix of ``text``, only the remainder is appended; otherwise
        the entry is re-encoded whole and overwritten. Without a
        session, content lookup (``lookup="auto"``, the default) asks
        the store whether a stored text is a prefix of this input;
        ``lookup="off"`` skips the store. Every variant returns the ids
        a from-scratch reference encode would return.
        """
        if lookup not in _LOOKUP_VALUES:
            raise ValueError(
                f"lookup must be one of {_LOOKUP_VALUES!r}, not {lookup!r}"
            )
        if session is not None and lookup is not None:
            raise ValueError(
                "a session names the entry directly; lookup does not combine"
            )
        if add_special_tokens and (session is not None or lookup == "auto"):
            raise UnsupportedConfig(
                "store-backed paths hold the core token stream; "
                "add_special_tokens=True cannot be served from them",
                details={
                    "option": "add_special_tokens",
                    "value": True,
                    "reason": "sessions and content lookup store the "
                    "pre-postprocessor core stream",
                },
            )
        ids: list[int] | None = None
        if session is not None:
            ids = self._store().encode_session(session, text)
        elif lookup != "off" and not add_special_tokens:
            ids = self._store().encode_auto(text)
        if ids is None:
            ids = self._executor.encode(text, add_special_tokens=add_special_tokens)
        return Encoding(ids=tuple(ids))

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = False,
    ) -> list[Encoding]:
        """Encode a batch; row ``i`` equals ``encode(texts[i])``.

        Batches run the plain routed path; per-document content lookup
        belongs to :meth:`encode`.
        """
        rows = self._executor.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [Encoding(ids=tuple(row)) for row in rows]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text through the reference oracle."""
        return self._oracle().decode(list(ids), skip_special_tokens=skip_special_tokens)

    # -- diagnostics ---------------------------------------------------

    def explain(self) -> dict[str, object]:
        """The active plan, its reasons, and accumulated counters.

        The report is the routing layer's own explanation
        (:func:`toktier.routing.explain.build_explanation`) plus the
        facade keys (``family``, ``store_directory``, ``store``). The
        requested routing policy is reported as ``routing_policy``; the
        ``certification`` block is a separate answer to a separate
        question. The facade plans against the digest-verified shipped
        registry. Outside ``REFERENCE``, its default ``device="auto"`` path
        performs an honest device probe when the GPU runtime is installed;
        ``device="cpu"``
        deliberately skips enumeration and reports
        ``R_ACCELERATOR_NOT_ADOPTED`` without making a hardware claim.

        "Not adopted" and "not available" are separate statements, and
        the report keeps them separate: the ``kernel_deliveries`` block
        carries the read-only shipped facts (whether a prebuilt fatbin
        and the JIT sources are installed -- the same answer ``toktier
        doctor`` gives) together with the per-delivery,
        per-architecture certification statuses of this artifact's
        record in the shipped support registry. The ``session_repair``
        block separately reports whether store appends use the certified
        corrected-Gigatoken callback or exact HF full re-encoding.
        ``runtime_policy``, ``gpu_backend``, and ``state_encode`` report
        the automatic crossover and the path that actually ran.
        """
        report = build_explanation(
            route_plan=self._plan,
            snapshot=self._snapshot,
            assessments=assessments_for(
                self._snapshot, self._plan.policy, self._registry, self._config
            ),
            fallback_counts=self._executor.fallback_counts,
            delivery_record=self._shipped_record(),
        )
        from ..engine.gpu.loader import KernelLoader

        actual_delivery = KernelLoader.delivery()
        report["kernel_delivery"] = actual_delivery
        delivery_report = report.get("kernel_deliveries")
        if isinstance(delivery_report, dict):
            for name in ("prebuilt", "jit"):
                item = delivery_report.get(name)
                if isinstance(item, dict):
                    item["loaded"] = actual_delivery == name
        report["family"] = self.family
        report["store_directory"] = (
            str(self._store_directory) if self._store_directory else None
        )
        if self._entry_store is not None:
            report["store"] = self._entry_store.stats()
        report["session_repair"] = self._session_repair_report()
        report["runtime_policy"] = {
            "device": self._device_request,
            "gpu_delivery_request": self._gpu_delivery_request,
            "gpu_delivery_selected": self._gpu_delivery,
            "gpu_min_bytes": self._gpu_min_bytes,
            "execution_counts": dict(self._executor.execution_counts),
            "last_execution": self._executor.last_execution,
        }
        report["gpu_backend"] = {
            "planned": BACKEND_GPU in self._plan.fallback_chain,
            "device": self._gpu_device if self._gpu_backend is not None else None,
            "loaded": self._gpu_backend.loaded if self._gpu_backend else False,
            "load_error": (
                str(self._gpu_backend.load_error)
                if self._gpu_backend and self._gpu_backend.load_error
                else None
            ),
        }
        report["state_encode"] = {
            "counts": dict(sorted(self._state_encode_counts.items())),
            "last": (
                dict(self._last_state_encode)
                if self._last_state_encode is not None
                else None
            ),
        }
        return report

    def close(self) -> None:
        """Release the loaded backend. Idempotent."""
        self._backend.close()
        if self._fast_backend is not None:
            self._fast_backend.close()
        if self._gpu_backend is not None:
            self._gpu_backend.close()

    # -- internals -----------------------------------------------------

    def _select_gpu_device(self) -> str:
        """First device covered by the selected delivery, or device zero."""
        devices = self._snapshot.devices
        match = self._snapshot.certification
        allowed: set[str] = set()
        if match is not None:
            entry = match.record.backends.get(BACKEND_GPU)
            if entry is not None:
                view = entry.deliveries.get(self._gpu_delivery, entry)
                allowed = set(view.devices)
        for device in devices:
            if not allowed or device.architecture in allowed:
                return f"cuda:{device.index}"
        return f"cuda:{devices[0].index}" if devices else "cuda:0"

    def _open_gpu_backend(self) -> GpuBackend:
        """Materialize the selected GPU delivery on first routed input."""
        from ..engine.gpu.engine import GpuEngine
        from ..engine.gpu.options import GpuOptions

        engine = GpuEngine.create(
            {self.family: self._artifact_handle},
            config=self._config,
            options=GpuOptions(device=self._gpu_device),
            delivery=self._gpu_delivery,
        )
        backend = engine.backend(self.family)
        self._gpu_engine = engine
        return backend

    def _shipped_record(self) -> ArtifactRecord | None:
        """This artifact's record in the shipped support registry.

        The same digest-verified record drives planning and reporting.
        Returns ``None`` when the artifact has no registry identity.
        """
        match = self._snapshot.certification
        return match.record if match is not None else None

    def _oracle(self) -> _OracleTokenizer:
        if self._oracle_handle is None:
            crate = import_oracle().Tokenizer.from_file(
                str(self._backend.tokenizer_path)
            )
            self._oracle_handle = cast("_OracleTokenizer", crate)
        return self._oracle_handle

    def _reference_encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Core-stream reference encode with per-token spans."""
        encoded = self._oracle().encode(text, add_special_tokens=False)
        return (
            [int(token_id) for token_id in encoded.ids],
            [(int(a), int(b)) for a, b in encoded.offsets],
        )

    def _count_state_encode(self, path: str, **detail: object) -> None:
        self._state_encode_counts[path] = self._state_encode_counts.get(path, 0) + 1
        self._last_state_encode = {"path": path, **detail}

    def _state_encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Seed store state through the active full-encode route.

        A certified Gigatoken repair object supplies the byte-closure and
        normalizer guards needed to reconstruct spans around ids returned
        by either the CPU engine or GPU. If those premises do not hold,
        or this family has no certified repair path, state is seeded by HF.
        """
        repair = self._repair_callback()
        if not isinstance(repair, GigatokenRepair):
            self._count_state_encode("hf_no_certified_span_bridge")
            return self._reference_encode(text)
        if self._added_router.holds_literal(text):
            self._count_state_encode("hf_added_token")
            return self._reference_encode(text)
        ids = self._executor.encode(text, add_special_tokens=False)
        execution = self._executor.last_execution or {}
        try:
            spans = repair.spans_for_ids(text, ids)
        except (UnicodeError, ValueError, WindowUnsupported) as error:
            self._count_state_encode(
                "hf_span_guard",
                error=type(error).__name__,
                message=str(error),
            )
            return self._reference_encode(text)
        backend = str(execution.get("executed_backend", "unknown"))
        self._count_state_encode(
            "accelerated_with_reconstructed_spans",
            backend=backend,
            input_bytes=len(text.encode("utf-8")),
        )
        return ids, spans

    def _store(self) -> EntryStore:
        if self._entry_store is None:
            repair = self._repair_callback()
            self._entry_store = EntryStore(
                fingerprint=self._semantic_fingerprint(repair),
                encode=self._state_encode,
                append=repair,
                append_stats=repair.stats if repair is not None else None,
                certified_bpe_witness=isinstance(repair, GigatokenRepair),
                bpe_sync_pclass=(
                    repair.bpe_sync_pclass
                    if isinstance(repair, GigatokenRepair)
                    else None
                ),
                seal_end_guard_chars=(
                    max(
                        self._seal_end_guard_chars,
                        repair.minimum_seal_tail_chars,
                    )
                    if isinstance(repair, GigatokenRepair)
                    else 0
                ),
                directory=self._store_directory,
                cache_budget_bytes=self._cache_budget,
            )
        return self._entry_store

    def _repair_callback(self) -> _SessionRepair | None:
        """Build the certified append callback once; otherwise use HF."""
        if self._session_repair_initialised:
            return self._session_repair
        self._session_repair_initialised = True
        if self._repair_backend_request == "reference":
            return None
        spec = family_spec(self.family, self._artifact_sha256)
        if self._repair_backend_request == "fastokens":
            if spec is None:
                raise UnsupportedConfig(
                    "the experimental Fastokens adapter has no repair-table "
                    f"entry for {self.family!r}",
                    details={
                        "option": "repair_backend",
                        "value": "fastokens",
                        "family": self.family,
                    },
                )
            self._session_repair = FastokensFullRepair.open(
                spec=spec,
                tokenizer_path=self._backend.tokenizer_path,
                hf_tokenizer=cast(Any, self._oracle()),
                reference_encode=self._reference_encode,
            )
            return self._session_repair
        backend = self._fast_backend
        if backend is None or BACKEND_FAST_CPU not in self._plan.fallback_chain:
            return None
        if spec is None:
            self._session_repair_guard = {
                "reason": "artifact_not_in_certified_repair_roster"
            }
            return None
        try:
            engine, hf_tokenizer = backend.repair_components()
            self._session_repair = GigatokenRepair(
                spec=spec,
                engine=engine,
                hf_tokenizer=cast(Any, hf_tokenizer),
                reference_encode=self._reference_encode,
            )
        except (BackendExecutionFault, WindowUnsupported) as error:
            self._session_repair_guard = {
                "reason": "repair_initialisation_guard",
                "error": type(error).__name__,
                "message": str(error),
            }
        return self._session_repair

    def _session_repair_report(self) -> dict[str, object]:
        if self._session_repair is not None:
            return {"status": "active", **self._session_repair.stats()}
        if self._session_repair_guard is not None:
            return {
                "status": "reference_fallback",
                "backend": BACKEND_REFERENCE,
                **self._session_repair_guard,
            }
        if not self._session_repair_initialised:
            requested = self._repair_backend_request
            eligible = (
                "fastokens"
                if requested == "fastokens"
                else (
                    BACKEND_FAST_CPU
                    if BACKEND_FAST_CPU in self._plan.fallback_chain
                    else None
                )
            )
            return {
                "status": "not_initialised" if eligible else "reference_only",
                "eligible_backend": eligible,
            }
        return {"status": "reference_only", "backend": BACKEND_REFERENCE}

    def _semantic_fingerprint(self, repair: _SessionRepair | None = None) -> bytes:
        """32-byte identity binding artifact, oracle and engine semantics.

        Any component changing produces a different fingerprint, so
        stored sessions from another meaning miss and re-encode instead
        of being replayed. The preimage is internal to the facade.
        """
        digest = hashlib.sha256(b"toktier.facade.fingerprint.v1\0")
        repair_stats = repair.stats() if repair is not None else {}
        for component in (
            self._artifact_sha256,
            ORACLE_PACKAGE,
            oracle_version() or "",
            _ENGINE_ID,
            _SESSION_SEMVER,
            (
                str(repair_stats.get("backend"))
                if repair is not None
                else BACKEND_REFERENCE
            ),
            repair.config_id if repair is not None else "",
            str(repair_stats.get("engine_version") or ""),
            str(repair_stats.get("engine_digest") or ""),
            self._snapshot.fast_cpu_engine.version or "",
            self._snapshot.fast_cpu_engine.binary_digest or "",
            self._snapshot.fast_cpu_engine.config_digest or "",
        ):
            raw = component.encode("utf-8")
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
        return digest.digest()


def load(
    family: str,
    *,
    store: str | os.PathLike[str] | None = None,
    device: str = "auto",
    config: Config | None = None,
    policy: RoutingPolicy | str | None = None,
    manifest: ArtifactManifest | None = None,
    cache_budget_bytes: int | None = None,
    repair_backend: str = "auto",
    gpu_delivery: str = "auto",
    gpu_min_bytes: int = DEFAULT_GPU_MIN_BYTES,
) -> Tokenizer:
    """Load a family and return a ready :class:`Tokenizer`.

    ``store`` names a directory for persistent session state (state, not
    cache: deleting it loses sessions). ``device="auto"`` (the default)
    selects a certified GPU for inputs at least ``gpu_min_bytes`` long
    and the certified CPU path below that boundary; ``"cpu"`` disables
    GPU adoption and ``"cuda"`` requires it. ``gpu_delivery="auto"``
    maps the ``gpu`` installation profile to prebuilt delivery and the
    ``gpu-jit`` profile to local JIT. The
    ``repair_backend="fastokens"`` is an explicit experimental full-session
    re-encode path and requires ``policy="experimental"``; it carries no
    exact-ID guarantee. The remaining keywords inject a configuration,
    routing policy, artifact manifest, or in-process cache budget without
    touching process environment.
    """
    return Tokenizer(
        family,
        config,
        policy=policy,
        store=store,
        device=device,
        manifest=manifest,
        cache_budget_bytes=cache_budget_bytes,
        repair_backend=repair_backend,
        gpu_delivery=gpu_delivery,
        gpu_min_bytes=gpu_min_bytes,
    )
