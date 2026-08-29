"""The facade: load a family once, then use it like a tokenizer.

Contract reference: ``docs/contracts/facade.md`` (0.x surface). The
facade is a thin composition of shipped pieces: artifacts resolve
through the manifest and the verified cache, routing follows the
standing policy semantics (GPU and corrected CPU backends are selected
only for exact certified bindings), and the session/lookup paths run on
the entry store. Certified and reference paths return ids equal to a
from-scratch reference encode; store layers can only decline to serve,
never answer differently. The explicitly selected experimental Fastokens
repair adapter is outside that exact-ID guarantee and reports it.

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
import threading
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

from .._oracle import ORACLE_PACKAGE, import_oracle, oracle_version
from ..artifacts import (
    ArtifactManifest,
    ArtifactStore,
    HuggingFaceSource,
    shipped_sibling_aliases,
)
from ..artifacts.model_resolution import ModelResolution, resolve_model_repository
from ..artifacts.tables import ARTIFACT_MANIFEST
from ..backends.fast_cpu import FastCpuBackend
from ..backends.hf import HfBackend
from ..backends.protocol import TOKENIZER_FILE
from ..config import Config
from ..engine.gpu.backend import GpuBackend, LazyGpuBackend
from ..engine.gpu.host_probe import CudaHostProbe
from ..errors import (
    BackendExecutionFault,
    BackendUnavailable,
    KernelIncompatible,
    UncertifiedTokenizer,
    UnsupportedConfig,
)
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
from ..repair.registry import family_spec, pclass_table
from ..routing.added_route import AddedTokenRouter
from ..routing.execute import RoutedExecutor
from ..routing.explain import build_explanation
from ..routing.plan import assessments_for
from ..routing.plan import plan as build_plan
from ..routing.probe import probe
from ..routing.registry_load import shipped_registry
from ..routing.registry_view import ArtifactRecord
from ..session import SessionUpdate
from .store import DEFAULT_CACHE_BUDGET_BYTES, EntryStore

if TYPE_CHECKING:
    from ..verify_local import VerificationKey

__all__ = ["Encoding", "Session", "Tokenizer", "from_pretrained", "load"]

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
_ACCELERATED_STATE_ENCODE_PATHS = {
    "materialized": "accelerated_with_reconstructed_spans",
    "lazy_seed": "accelerated_with_lazy_span_checkpoints",
}

#: Measured crossover used by the automatic facade. It is public through
#: ``gpu_min_bytes`` and ``explain()`` rather than hidden in an engine.
DEFAULT_GPU_MIN_BYTES = 64 * 1024

_SessionRepair = GigatokenRepair | FastokensFullRepair


class _RouteEveryInput:
    """Fail-closed scanner for an added-token shape we cannot model."""

    def scan(self, text: str) -> list[tuple[str, int | None]]:
        return [(text, None)]


def _config_added_tokens_claim(
    registry: Any, artifact_sha256: str
) -> Mapping[str, object] | None:
    """The certification record's configuration-side added-token claim.

    ``None`` when no exact record exists (an uncertified artifact makes no
    claim to verify); the record's declared ``config_added_tokens`` mapping
    when it carries one; and an explicit empty claim (count 0) for a record
    without the section, so a certified artifact whose local sidecar has
    since grown extra added tokens fails closed instead of executing a
    loader face nobody judged.
    """
    match = registry.certification(artifact_sha256=artifact_sha256)
    if match is None or match.identity != "exact":
        return None
    declared = getattr(match.record, "config_added_tokens", None)
    if declared is not None:
        return dict(declared)
    return {"sha256": None, "count": 0}


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


def _unjudged_toolchain_ways_forward(family: str) -> str:
    """The two ways past a refusal that is only about coverage.

    Since 0.2.6 the default ``SUPPORTED`` policy admits an unjudged
    compiler pair and labels the route ``supported_untested``, so a
    reader who reached this text under ``CERTIFIED`` has a choice to
    make rather than a single escape hatch. Both are named, and neither
    is described as safe: the first says who has not measured this
    combination, the second says the same and asks the caller to say it
    out loud.
    """
    return (
        "select the default policy (policy='supported'), which runs this "
        "combination and labels it supported_untested, or keep this policy "
        f"and opt in once with `{_uncertified_jit_remedy(family)}`; either "
        "way nobody has measured this pair, and `toktier verify-local "
        f"--family {family}` compares it with the reference engine on your "
        "own text"
    )


def _last_execution_view(report: Mapping[str, object]) -> Mapping[str, object]:
    """The routing ledger's last-request record, or an empty mapping."""
    runtime = report.get("runtime_policy")
    if not isinstance(runtime, Mapping):
        return {}
    last = runtime.get("last_execution")
    return last if isinstance(last, Mapping) else {}


def _explanation_summary(report: Mapping[str, object]) -> dict[str, object]:
    """Select the flat headline view from one full facade explanation.

    The keys deliberately name their time scope, because the facts here
    belong to different ones: what the last request did, what the
    immutable plan says, what this process has loaded, and what has
    happened over the process lifetime. Every key of the 0.2.0 summary
    keeps its meaning; the ``last_execution_*`` group and the two
    delivery keys are additions that let a reader answer "what just
    happened?" without opening the full report.
    """
    certification = cast(Mapping[str, object], report["certification"])
    fallback_counts = cast(Mapping[str, int], report["fallback_counts"])
    runtime = report.get("runtime_policy")
    last = _last_execution_view(report)
    executed = last.get("executed_backend")
    selected_start = last.get("selected_start")
    return {
        "family": report["family"],
        "backend": report["backend"],
        "backend_basis": report["backend_basis"],
        "planned_backend": report["planned_backend"],
        "kernel_delivery": report["kernel_delivery"],
        "certification_state": certification["state"],
        "effective_verdict": certification["effective_verdict"],
        "fallback_occurred": bool(fallback_counts),
        # -- the last request, explicitly scoped ----------------------
        "last_execution_backend": executed if isinstance(executed, str) else None,
        "last_execution_path": (
            last["path"] if isinstance(last.get("path"), str) else None
        ),
        "last_execution_source": (
            last["source"] if isinstance(last.get("source"), str) else None
        ),
        # True when that request finished on a backend other than the
        # one the router selected for it -- a mid-request fault or
        # guard route. The crossover decision happens before selection
        # and a bounded session repair starts where it runs, so neither
        # sets this.
        "last_execution_fallback": (
            isinstance(executed, str)
            and isinstance(selected_start, str)
            and executed != selected_start
        ),
        # -- process lifetime and process/plan facts ------------------
        "fallback_ever_occurred": bool(fallback_counts),
        "selected_kernel_delivery": (
            runtime.get("gpu_delivery_selected")
            if isinstance(runtime, Mapping)
            else None
        ),
        "loaded_kernel_delivery": report["kernel_delivery"],
    }


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
    #: The certification record's configuration-side added-token claim for
    #: this exact artifact: the declared ``config_added_tokens`` mapping, an
    #: empty claim (count 0) for a record that declares none, or ``None``
    #: when no record makes a claim. Backends verify the subset they
    #: observe against it and fail closed on a mismatch.
    config_added_tokens_claim: Mapping[str, object] | None = None

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


#: Elements compared per block while looking for the first difference
#: between two token streams. Tuple slice equality compares in C, so
#: locating the differing block costs a fraction of an element-by-element
#: Python loop. 4096 was the fastest of 256 through 262,144 on sessions
#: from 500 to 500,000 tokens, and is within noise of its neighbours.
_PREFIX_BLOCK = 4096


def _first_difference(
    previous: tuple[int, ...], current: tuple[int, ...]
) -> int:
    """The first index the two streams disagree on, or their shared length.

    This is exactly ``next(i for i in range(shared) if previous[i] !=
    current[i])`` with ``shared`` as the default -- the same index, from
    the same definition. It is written as a block search because an
    append otherwise pays an element-by-element Python loop over the
    whole accumulated stream, which grows without bound as a session
    does while the append itself stays small.
    """
    shared = min(len(previous), len(current))
    start = 0
    while start < shared:
        stop = min(start + _PREFIX_BLOCK, shared)
        if previous[start:stop] != current[start:stop]:
            return next(
                index
                for index in range(start, stop)
                if previous[index] != current[index]
            )
        start = stop
    return shared


class Session:
    """A live session over one tokenizer (``api.md`` Section 5).

    Every field speaks about the pre-postprocessor core token stream, the
    stream the store holds; special tokens are applied at read time by
    :meth:`final_ids`. The object is a view over the tokenizer's own
    session path -- it holds no encoder of its own -- so the ids it
    reports are the ids that path returns, equal to a from-scratch
    reference encode of the accumulated text.

    A session has one owner: it is not safe to share across threads.
    """

    def __init__(self, tokenizer: Tokenizer, session_id: str) -> None:
        self._tokenizer = tokenizer
        self._session_id = session_id
        self._text = ""
        self._ids: tuple[int, ...] = ()
        self._writes = 0

    @property
    def session_id(self) -> str:
        """The name this session's state is stored under."""
        return self._session_id

    @property
    def ids(self) -> Sequence[int]:
        """The current core token stream."""
        return self._ids

    @property
    def text(self) -> str:
        """The text accumulated so far."""
        return self._text

    @property
    def revision(self) -> int:
        """Monotone write counter.

        When the durable store holds this session, this is the store's
        own revision -- the value an optimistic write is checked
        against. Before the first write, or while the session lives
        somewhere the store does not track it, it counts the writes made
        through this object, which is monotone but not a store fact.
        """
        stored = self._tokenizer.store_session_revision(self._session_id)
        return self._writes if stored is None else stored

    def _adopt(self, text: str) -> None:
        """Take ``text`` as the transcript this session already holds.

        Opening is not a write: when the stored record recognizes this
        text the store serves it, and the revision stays the store's.
        """
        self._ids = tuple(
            self._tokenizer.encode(text, session=self._session_id).ids
        )
        self._text = text

    def append(self, text: str) -> SessionUpdate:
        """Extend the session and report the effect on the core stream.

        The returned update satisfies the frozen invariant
        ``all_ids == old_ids[:replace_from] + replacement_ids``, and
        ``replace_from`` is the longest prefix that survived -- at least
        as tight as any cut the engine made internally, and ``0`` when
        the whole stream was re-encoded.
        """
        previous = self._ids
        combined = self._text + text
        current = tuple(
            self._tokenizer.encode(combined, session=self._session_id).ids
        )
        cut = _first_difference(previous, current)
        self._text = combined
        self._ids = current
        self._writes += 1
        return SessionUpdate(
            replace_from=cut,
            replacement_ids=current[cut:],
            all_ids=current,
        )

    def final_ids(self, add_special_tokens: bool = True) -> list[int]:
        """The read-time view: postprocessor applied over the core stream."""
        if not add_special_tokens:
            return list(self._ids)
        return list(
            self._tokenizer.encode(self._text, add_special_tokens=True).ids
        )


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
        self._registry = shipped_registry()
        handle = _ResolvedArtifact(
            family=entry.family,
            root=verified.directory,
            artifact_sha256=self._artifact_sha256,
            files={item.name: item.sha256 for item in entry.files},
            config_added_tokens_claim=_config_added_tokens_claim(
                self._registry, self._artifact_sha256
            ),
        )
        self._artifact_handle = handle
        self._backend = HfBackend.open(handle)
        # Everything below reads the loader face the reference executes:
        # for an artifact whose configuration sidecar contributes added
        # tokens this is the serialized live loader object, so the
        # added-token router, the seal guard and the reference itself
        # answer from one document.
        document = json.loads(self._backend.materialized_tokenizer_json())
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
        self._postprocessor_adds_tokens = document.get("post_processor") is not None
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
                f"will use {self._plan.backend!r}. To run this combination, "
                f"{_unjudged_toolchain_ways_forward(entry.family)}. Neither "
                "route is covered by TokTier's certified exact-ID guarantee.",
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
                # The facts the refusal rests on belong in the message,
                # not only in ``details``: a reader who sees only the
                # text should still be able to tell which of the three
                # binding axes did not match.
                observed = jit_rejection.detail.get("observed", "unknown")
                constraint = jit_rejection.detail.get("constraint", "unknown")
                remedy = _uncertified_jit_remedy(entry.family)
                details["remedy"] = remedy
                details["ways_forward"] = _unjudged_toolchain_ways_forward(
                    entry.family
                )
                message += (
                    f"; observed {observed}; certified constraint: "
                    f"{constraint}; to run this unjudged combination, "
                    f"{_unjudged_toolchain_ways_forward(entry.family)}"
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
        self._native_session_encoder: Any | None = None
        self._native_session_encoder_initialised = False
        self._native_request: Any | None = None
        self._native_request_initialised = False
        self._native_request_lock = threading.Lock()
        self._native_request_guard: dict[str, object] | None = None
        self._native_gpu_prepared: Any | None = None
        self._native_gpu_publication_guard: dict[str, object] | None = None
        self._session_repair_guard: dict[str, object] | None = None
        self._state_encode_counts: dict[str, int] = {}
        self._last_state_encode: dict[str, object] | None = None
        self._model_resolution: ModelResolution | None = None
        self._closed = False

    # -- identity ------------------------------------------------------

    @property
    def family(self) -> str:
        """Canonical family id this object was loaded for."""
        return self._backend.family

    @property
    def plan(self) -> RoutePlan:
        """The immutable route plan fixed at construction."""
        return self._plan

    # -- local verification --------------------------------------------

    def verification_key(self, engine: str) -> VerificationKey | None:
        """What a local check of one engine on this machine is about.

        ``engine`` is ``"gpu"`` or ``"cpu"``. The key gathers the facts a
        measurement would depend on -- the device, the delivery, the
        image, the compiler, the driver, the two source identities and
        the exact artifact -- so a record taken under it is read back
        only while all of them still hold. ``None`` for any other engine
        name, which is the fail-closed answer: a combination this
        tokenizer cannot name carries no record.

        The key is not a certificate and does not become one. It is the
        filing address of something a person measured here.
        """
        from ..verify_local import VerificationKey

        cache = self._snapshot.kernel_cache
        if engine == "gpu":
            delivery = (
                cache.delivery or cache.preferred_delivery or self._gpu_delivery
            )
            return VerificationKey(
                engine="gpu",
                family=self.family,
                artifact_sha256=self._artifact_sha256,
                architecture=next(
                    (
                        device.architecture
                        for device in self._snapshot.devices
                    ),
                    None,
                ),
                delivery=delivery,
                image_digest=(
                    cache.binary_digest
                    if delivery == "prebuilt"
                    else cache.source_digest
                ),
                class_table_digest=cache.class_table_digest,
                build_flags=tuple(cache.build_flags),
                toolchain=(
                    cache.toolchain
                    if delivery == "jit"
                    else cache.host_toolchain
                ),
                driver_version=self._snapshot.driver_version,
                host_source_digest=cache.host_source_digest,
                engine_source_digest=cache.source_digest,
            )
        if engine == "cpu":
            facts = self._snapshot.fast_cpu_engine
            return VerificationKey(
                engine="cpu",
                family=self.family,
                artifact_sha256=self._artifact_sha256,
                build_flags=tuple(facts.build_flags),
                toolchain=facts.toolchain,
                engine_source_digest=facts.source_digest,
                config_digest=facts.config_digest,
            )
        return None

    def _locally_verified(self) -> bool:
        """Whether the planned accelerated route carries a passing check.

        Read-only and quiet: a missing record, an unreadable one and one
        taken on another combination all answer ``False``, which is the
        label a route has when nobody has measured it here.
        """
        from ..verify_local import is_locally_verified

        engine = {BACKEND_GPU: "gpu", BACKEND_FAST_CPU: "cpu"}.get(
            self._plan.backend
        )
        if engine is None:
            return False
        key = self.verification_key(engine)
        return key is not None and is_locally_verified(self._config, key)

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
        session, an omitted ``lookup`` behaves as ``"auto"`` for
        core-stream calls: content lookup asks the store whether a
        stored text is a prefix of this input; ``lookup="off"`` skips
        the store. With ``add_special_tokens=True``, omitting
        ``lookup`` selects the plain routed path, while an explicit
        ``session`` or ``lookup="auto"`` raises ``UNSUPPORTED_CONFIG``.
        Under certified and reference policies every variant returns
        the ids a from-scratch reference encode would return; an
        explicitly selected experimental repair backend labels itself
        outside that guarantee.
        """
        self._require_open()
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
        native = self._native_request_runtime()
        if native is not None:
            native_ids = native.encode(
                text,
                session=session,
                lookup_auto=lookup != "off",
                add_special_tokens=add_special_tokens,
            )
            if self._native_gpu_prepared is not None:
                self._observe_native_gpu_open(native)
            return Encoding(ids=tuple(native_ids))
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
        self._require_open()
        native = self._native_request_runtime()
        if native is not None:
            rows = native.encode_batch(
                list(texts), add_special_tokens=add_special_tokens
            )
            if self._native_gpu_prepared is not None:
                self._observe_native_gpu_open(native)
            return [Encoding(ids=tuple(row)) for row in rows]
        rows = self._executor.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [Encoding(ids=tuple(row)) for row in rows]

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text through the reference oracle."""
        self._require_open()
        return self._oracle().decode(list(ids), skip_special_tokens=skip_special_tokens)

    # -- diagnostics ---------------------------------------------------

    def explain(self, *, summary: bool = False) -> dict[str, object]:
        """The active plan, its reasons, and accumulated counters.

        With ``summary=True``, the fully assembled report is reduced to
        its scalar headline fields. Those fields name their time scope:
        the ``last_execution_*`` group describes the request that most
        recently returned, ``planned_backend`` the immutable plan,
        ``selected_kernel_delivery`` / ``loaded_kernel_delivery`` the
        process's delivery, and ``fallback_ever_occurred`` the process
        lifetime. The no-argument call returns the complete
        machine-readable report described below.

        The report is the routing layer's own explanation
        (:func:`toktier.routing.explain.build_explanation`) plus the
        facade keys (``family``, ``store_directory``, ``store``). The
        requested routing policy is reported as ``routing_policy``; the
        ``certification`` block is a separate answer to a separate
        question.

        The headline ``backend`` is the backend that actually returned
        the last result once this tokenizer has run anything, and the
        planned backend before that; ``backend_basis``
        (``"last_execution"`` / ``"plan"``) says which, and
        ``planned_backend`` reports the plan either way. A per-input
        safety fallback therefore shows up in the headline rather than
        only in ``runtime_policy.last_execution``.

        The facade plans against the digest-verified shipped
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
        block separately reports the active repair engine and request
        path: Rust-native corrected repair, the compatibility callback,
        an explicitly selected experimental Fastokens adapter, or exact
        HF full re-encoding.
        ``runtime_policy``, ``gpu_backend``, and ``state_encode`` report
        the automatic crossover and the path that actually ran. Successful
        accelerated state paths use
        ``accelerated_with_lazy_span_checkpoints`` for a native lazy seed and
        ``accelerated_with_reconstructed_spans`` for a materialized payload.
        """
        native_runtime = (
            self._native_request if self._native_request_initialised else None
        )
        if native_runtime is not None and self._native_gpu_prepared is not None:
            self._observe_native_gpu_open(native_runtime)
        native_stats = (
            native_runtime.runtime_stats() if native_runtime is not None else None
        )
        last_execution = (
            native_stats["last_execution"]
            if native_stats is not None
            else self._executor.last_execution
        )
        from ..engine.gpu.loader import KernelLoader

        actual_delivery = KernelLoader.delivery()
        report = build_explanation(
            route_plan=self._plan,
            snapshot=self._snapshot,
            assessments=assessments_for(
                self._snapshot, self._plan.policy, self._registry, self._config
            ),
            fallback_counts=(
                native_stats["fallback_counts"]
                if native_stats is not None
                else self._executor.fallback_counts
            ),
            delivery_record=self._shipped_record(),
            last_execution=(
                last_execution if isinstance(last_execution, Mapping) else None
            ),
            gpu_delivery=actual_delivery or self._gpu_delivery,
            locally_verified=self._locally_verified(),
        )
        report["kernel_delivery"] = actual_delivery
        delivery_report = report.get("kernel_deliveries")
        if isinstance(delivery_report, dict):
            for name in ("prebuilt", "jit"):
                item = delivery_report.get(name)
                if isinstance(item, dict):
                    item["loaded"] = actual_delivery == name
        # The snapshot was taken at construction, so its delivery field
        # can predate the first kernel load. The loader's answer is a
        # read-only process fact; reporting the stale ``None`` beside a
        # top-level ``prebuilt`` would make the two disagree.
        probe_report = report.get("probe")
        if isinstance(probe_report, dict) and actual_delivery is not None:
            probe_report["kernel_delivery"] = actual_delivery
        report["family"] = self.family
        report["model_resolution"] = (
            self._model_resolution.report(
                execution_artifact_sha256=self._artifact_sha256
            )
            if self._model_resolution is not None
            else None
        )
        report["store_directory"] = (
            str(self._store_directory) if self._store_directory else None
        )
        if native_runtime is not None:
            report["store"] = native_runtime.store_stats()
        elif self._entry_store is not None:
            report["store"] = self._entry_store.stats()
        report["session_repair"] = self._session_repair_report()
        report["runtime_policy"] = {
            "device": self._device_request,
            "gpu_delivery_request": self._gpu_delivery_request,
            "gpu_delivery_selected": self._gpu_delivery,
            "gpu_min_bytes": self._gpu_min_bytes,
            "execution_counts": (
                dict(native_stats["execution_counts"])
                if native_stats is not None
                else dict(self._executor.execution_counts)
            ),
            "last_execution": last_execution,
            "request_path": (
                "rust_native" if native_runtime is not None else "python_adapter"
            ),
            "python_to_native_calls": (
                native_stats["python_to_native_calls"]
                if native_stats is not None
                else None
            ),
        }
        # On the native request path the engine opens below PyO3, on the
        # first request routed to the GPU; the runtime's once-cell is the
        # authoritative record of whether (and how) that open happened.
        native_gpu_loaded = (
            bool(native_runtime.gpu_engine_loaded)
            if native_runtime is not None
            else False
        )
        native_gpu_open_error = (
            native_runtime.gpu_engine_open_error if native_runtime is not None else None
        )
        legacy_gpu_loaded = (
            self._gpu_backend.loaded if self._gpu_backend is not None else False
        )
        gpu_report: dict[str, object] = {
            "planned": BACKEND_GPU in self._plan.fallback_chain,
            "device": (
                self._gpu_device
                if self._gpu_backend is not None or native_gpu_loaded
                else None
            ),
            "loaded": legacy_gpu_loaded or native_gpu_loaded,
            "load_error": (
                str(self._gpu_backend.load_error)
                if self._gpu_backend and self._gpu_backend.load_error
                else native_gpu_open_error
            ),
        }
        if self._native_gpu_publication_guard is not None:
            gpu_report["publication_guard"] = dict(self._native_gpu_publication_guard)
        report["gpu_backend"] = gpu_report
        report["state_encode"] = (
            {
                "counts": dict(native_stats["state_encode_counts"]),
                "last": native_stats["last_state_encode"],
            }
            if native_stats is not None
            else {
                "counts": dict(sorted(self._state_encode_counts.items())),
                "last": (
                    dict(self._last_state_encode)
                    if self._last_state_encode is not None
                    else None
                ),
            }
        )
        if self._native_request_guard is not None:
            report["native_request_guard"] = dict(self._native_request_guard)
        return _explanation_summary(report) if summary else report

    @contextmanager
    def session(
        self,
        store: str | os.PathLike[str] | None = None,
        *,
        session_id: str | None = None,
        text: str = "",
    ) -> Iterator[Session]:
        """Open a session over this tokenizer (``api.md`` Section 5).

        The yielded :class:`Session` accumulates text and reports each
        append as a :class:`~toktier.SessionUpdate`. Where the state
        lives was decided when the tokenizer was loaded: ``load(store=)``
        makes it persistent, and without it the session is in-memory for
        this process. ``store`` here may therefore be omitted, or repeat
        that same directory; naming a different one is refused rather
        than silently ignored.

        ``session_id`` names the state. An unnamed session gets a fresh
        name, readable as :attr:`Session.session_id` -- necessary for a
        persistent one, which is otherwise written where nothing can
        find it again. Leaving the block does not delete state: a
        persistent session is meant to outlive the process.

        ``text`` is the transcript this session already holds, and is how
        a stored conversation is resumed. It is not optional bookkeeping:
        a session object starts empty, and appending one new turn to an
        empty object would replace the stored conversation with that
        turn rather than continue it. The store recognizes a transcript
        it already holds and serves it instead of re-encoding, so
        resuming costs a lookup. (``api.md`` Section 5 does not carry
        this argument; the deviation is recorded in ``facade.md``
        Section 7, which is also where the store binding lives.)
        """
        self._require_open()
        if store is not None:
            requested = Path(os.fspath(store)).expanduser()
            if requested != self._store_directory:
                raise UnsupportedConfig(
                    "the session store is bound when the tokenizer is loaded",
                    details={
                        "option": "store",
                        "value": str(requested),
                        "reason": (
                            "pass the directory to load(store=...); this "
                            "tokenizer holds "
                            + (
                                f"{self._store_directory}"
                                if self._store_directory is not None
                                else "no store directory"
                            )
                        ),
                    },
                )
        name = session_id or f"session-{uuid4().hex}"
        session = Session(self, name)
        if text:
            session._adopt(text)
        yield session

    def store_session_revision(self, session_id: str) -> int | None:
        """The store's revision for one session, when it holds it.

        A tokenizer bound to a store directory asks whichever store it
        routes sessions to. Which one that is depends on the plan: the
        certified configurations serve sessions from the native request
        runtime and never build the Python entry store at all, so
        consulting only the latter answered ``None`` for every session
        the product actually stores. Both now read the revision out of
        the record when the entry is not resident, so the answer
        survives a restart -- the record has carried it since format v1.
        The store is opened to answer, through the same latched decision
        an encode makes, because a store that has not been opened yet
        cannot truthfully say ``None`` about a session it holds.

        A tokenizer with no store directory is unchanged: it holds
        nothing durable, opens nothing, and ``None`` means what it always
        meant.
        """
        if self._store_directory is not None and not self._closed:
            native = self._native_request if self._native_request_initialised else None
            if native is None:
                native = self._native_request_runtime()
            if native is not None:
                revision = cast("int | None", native.session_revision(session_id))
                if revision is not None:
                    return revision
            store = self._entry_store
            if store is None:
                store = self._store()
            return store.session_revision(session_id)
        if self._entry_store is None:
            return None
        return self._entry_store.session_revision(session_id)

    def close(self) -> None:
        """Release the loaded backend. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._native_request = None
        self._native_gpu_prepared = None
        self._native_session_encoder = None
        self._oracle_handle = None
        self._backend.close()
        if self._fast_backend is not None:
            self._fast_backend.close()
        if self._gpu_backend is not None:
            self._gpu_backend.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("backend is closed")

    def _observe_native_gpu_open(self, native: Any) -> None:
        """Publish the loaded delivery once the deferred open happened.

        The engine opens below PyO3, so the process-wide loader fact is
        recorded at the first Python-visible moment afterwards -- the end
        of the request that opened it, or an ``explain()`` call.  A
        publication the loader refuses has already voided the process
        certificate there; the refusal is kept for ``explain()`` instead
        of failing the request that carried it.
        """
        prepared = self._native_gpu_prepared
        if prepared is None or prepared.published:
            return
        if not native.gpu_engine_loaded:
            return
        try:
            prepared.publish_loaded()
        except KernelIncompatible as error:
            self._native_gpu_publication_guard = {
                "reason": "native_gpu_loaded_publication_guard",
                "error": type(error).__name__,
                "message": str(error),
            }
            prepared.published = True

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
            # The oracle executes the same document as the reference
            # backend: the loader-face serialization when the
            # configuration sidecar contributed added tokens, and the
            # verified artifact file itself otherwise.
            loader_face = getattr(
                self._backend, "materialized_tokenizer_json", None
            )
            if callable(loader_face):
                crate = import_oracle().Tokenizer.from_str(loader_face())
            else:
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

        This compatibility callback materializes spans around ids returned
        by either the CPU engine or GPU. The native request runtime instead
        records its closure-verified lazy seed under the other accelerated
        payload name. If the reconstruction premises do not hold, or this
        family has no certified repair path, state is seeded by HF.
        """
        repair = self._repair_callback()
        if not isinstance(repair, GigatokenRepair):
            path = "hf_no_certified_span_bridge"
            result = self._reference_encode(text)
            self._count_state_encode(path)
            self._executor.record_reference_result(
                text,
                reason=ReasonCode.R_INPUT_GUARD_ROUTED,
                path=path,
            )
            return result
        if self._added_router.holds_literal(text):
            path = "hf_added_token"
            result = self._reference_encode(text)
            self._count_state_encode(path)
            self._executor.record_reference_result(
                text,
                reason=ReasonCode.R_INPUT_ADDED_TOKEN,
                path=path,
            )
            return result
        ids = self._executor.encode(text, add_special_tokens=False)
        execution = self._executor.last_execution or {}
        try:
            spans = repair.spans_for_ids(text, ids)
        except (UnicodeError, ValueError, WindowUnsupported) as error:
            path = "hf_span_guard"
            result = self._reference_encode(text)
            self._count_state_encode(
                path,
                error=type(error).__name__,
                message=str(error),
            )
            self._executor.record_reference_result(
                text,
                reason=ReasonCode.R_INPUT_GUARD_ROUTED,
                path=path,
                replaces_last=True,
                error=type(error).__name__,
                message=str(error),
            )
            return result
        backend = str(execution.get("executed_backend", "unknown"))
        self._count_state_encode(
            _ACCELERATED_STATE_ENCODE_PATHS["materialized"],
            backend=backend,
            input_bytes=len(text.encode("utf-8")),
        )
        return ids, spans

    def _gigatoken_append(
        self,
        tail_text: str,
        tail_ids: list[int],
        tail_spans: Sequence[tuple[int, int]],
        delta: str,
    ) -> tuple[list[int], list[tuple[int, int]], int, str]:
        """Run one adapter repair and publish its bounded execution facts."""
        repair = self._session_repair
        assert isinstance(repair, GigatokenRepair)
        result = repair(tail_text, tail_ids, tail_spans, delta)
        path = result[3]
        if path == "gigatoken_repair":
            last = repair.stats()["last"]
            assert isinstance(last, dict)
            window_chars = last["window_chars"]
            assert isinstance(window_chars, int)
            window_text = tail_text[-window_chars:] + delta
            self._executor.record_repair_result(
                input_bytes=len(window_text.encode("utf-8")),
                path=path,
            )
        elif path.startswith("hf_full_"):
            # The bounded window did not hold, so the whole grown text was
            # re-encoded on the reference engine. That ran; the ledger says
            # so. (``gigatoken_repair_noop`` is the third case: an empty
            # delta executes nothing, so there is nothing to record.)
            self._executor.record_repair_reference_result(
                input_bytes=len((tail_text + delta).encode("utf-8")),
                path=path,
            )
        return result

    def _store(self) -> EntryStore:
        if self._entry_store is None:
            # The pure-native session encoder performs full/seed encodes on
            # the corrected CPU engine below PyO3, without consulting the
            # routed executor. That is exactly right for CPU-only plans, but
            # a plan that admits the GPU backend routes large inputs by size
            # (``gpu_min_bytes``), and this store only serves the Python
            # adapter (the native request runtime owns the store paths under
            # prebuilt delivery). So when GPU is admitted, full and seed
            # encodes go through the callback lane instead: its
            # ``encode`` callback (:meth:`_state_encode`) runs the routed
            # executor -- dispatching the admitted GPU backend and recording
            # ``last_execution`` -- while append repair stays on the
            # certified Gigatoken BPE-sync witness machinery.
            native_encoder = (
                None
                if BACKEND_GPU in self._plan.fallback_chain
                else self._native_repair_encoder()
            )
            repair = None if native_encoder is not None else self._repair_callback()
            native_repair = native_encoder is not None
            self._entry_store = EntryStore(
                fingerprint=self._semantic_fingerprint(
                    repair, native_fast_cpu=native_repair
                ),
                encode=self._state_encode,
                append=(
                    self._gigatoken_append
                    if isinstance(repair, GigatokenRepair)
                    else repair
                ),
                append_stats=(
                    native_encoder.stats
                    if native_encoder is not None
                    else (repair.stats if repair is not None else None)
                ),
                certified_bpe_witness=(
                    native_repair or isinstance(repair, GigatokenRepair)
                ),
                bpe_sync_pclass=(
                    None
                    if native_repair
                    else (
                        repair.bpe_sync_pclass
                        if isinstance(repair, GigatokenRepair)
                        else None
                    )
                ),
                seal_end_guard_chars=(
                    max(
                        self._seal_end_guard_chars,
                        native_encoder.minimum_seal_tail_chars,
                    )
                    if native_encoder is not None
                    else (
                        max(
                            self._seal_end_guard_chars,
                            repair.minimum_seal_tail_chars,
                        )
                        if isinstance(repair, GigatokenRepair)
                        else 0
                    )
                ),
                native_encoder=native_encoder,
                directory=self._store_directory,
                cache_budget_bytes=self._cache_budget,
            )
        return self._entry_store

    def _native_request_runtime(self) -> Any | None:
        """Return the one latched native request-runtime decision."""
        if self._native_request_initialised:
            return self._native_request
        with self._native_request_lock:
            # Another thread may have completed construction while this
            # thread waited for the lock. Mypy does not model that interleaving.
            if self._native_request_initialised:
                return self._native_request  # type: ignore[unreachable]
            try:
                return self._construct_native_request_runtime()
            finally:
                self._native_request_initialised = True

    def _construct_native_request_runtime(self) -> Any | None:
        """Construct the one-call native request path when every engine fits.

        JIT delivery and the explicitly experimental Fastokens repair adapter
        retain their existing Python hosts.  Certified CPU/reference plans use
        this runtime directly; prebuilt CUDA joins it through the native driver
        host rather than through a callback.
        """
        if self._repair_backend_request == "fastokens":
            return None
        if (
            BACKEND_GPU in self._plan.fallback_chain
            and self._gpu_delivery != "prebuilt"
        ):
            return None
        native_reference = getattr(self._backend, "native_engine", None)
        if not callable(native_reference):
            return None
        try:
            from .. import _native

            reference = native_reference()
            fast_encoder = None
            if BACKEND_FAST_CPU in self._plan.fallback_chain:
                fast_encoder = self._native_repair_encoder(reference)
                if fast_encoder is None:
                    return None
            repair_fast_cpu = (
                self._repair_backend_request == "auto" and fast_encoder is not None
            )
            seal_guard = 0
            if repair_fast_cpu:
                assert fast_encoder is not None
                seal_guard = max(
                    self._seal_end_guard_chars,
                    int(fast_encoder.minimum_seal_tail_chars),
                )
            gpu_encoder = None
            if BACKEND_GPU in self._plan.fallback_chain:
                from ..engine.gpu.native import prepare_native_prebuilt_gpu

                device_ordinal = int(self._gpu_device.partition(":")[2] or "0")
                device = next(
                    (
                        item
                        for item in self._snapshot.devices
                        if item.index == device_ordinal
                    ),
                    None,
                )
                if device is None:
                    raise RuntimeError(
                        "the native GPU device disappeared after planning"
                    )
                # Projection and the delivery reservation happen here; the
                # engine itself opens on the first request the native
                # runtime routes to the GPU, at or above ``gpu_min_bytes``.
                prepared = prepare_native_prebuilt_gpu(
                    artifact=self._artifact_handle,
                    family=self.family,
                    cache_dir=Path(self._config.cache_dir),
                    device_ordinal=device_ordinal,
                    architecture=device.architecture,
                    reference=reference,
                )
                gpu_encoder = prepared.engine
                self._native_gpu_prepared = prepared
            self._native_request = _native.NativeRuntime(
                list(self._plan.fallback_chain),
                [
                    self._gpu_min_bytes if backend == BACKEND_GPU else 0
                    for backend in self._plan.fallback_chain
                ],
                reference,
                fast_encoder,
                gpu_encoder,
                repair_fast_cpu,
                self._semantic_fingerprint(None, native_fast_cpu=repair_fast_cpu),
                seal_guard,
                self._postprocessor_adds_tokens,
                self._config.diagnostics,
                (
                    str(self._store_directory)
                    if self._store_directory is not None
                    else None
                ),
                self._cache_budget,
            )
        except (
            BackendExecutionFault,
            KernelIncompatible,
            UncertifiedTokenizer,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            self._native_request_guard = {
                "reason": "native_request_initialisation_guard",
                "error": type(error).__name__,
                "message": str(error),
            }
            self._native_request = None
        return self._native_request

    def _native_repair_encoder(self, reference: object | None = None) -> Any | None:
        """Build the one-call Rust session engine once when certified.

        Explicit reference and experimental Fastokens requests retain their
        existing adapters. The default corrected-Gigatoken route instead
        materializes the exact live tokenizer once, then moves full encode,
        append repair, boundary certification and HF fallback below PyO3.
        """
        if self._native_session_encoder_initialised:
            return self._native_session_encoder
        self._native_session_encoder_initialised = True
        if self._repair_backend_request != "auto":
            return None
        backend = self._fast_backend
        if backend is None or BACKEND_FAST_CPU not in self._plan.fallback_chain:
            return None
        shared_engine = getattr(backend, "native_session_engine", None)
        materialize = getattr(backend, "materialized_tokenizer_json", None)
        if not callable(shared_engine) and not callable(materialize):
            # Test/injected backends written for the callback protocol retain
            # that protocol. Production FastCpuBackend always supplies the
            # native materialization surface.
            return None
        spec = family_spec(self.family, self._artifact_sha256)
        if spec is None:
            self._session_repair_guard = {
                "reason": "artifact_not_in_certified_repair_roster"
            }
            return None
        try:
            from .. import _native

            if callable(shared_engine):
                self._native_session_encoder = (
                    shared_engine(reference)
                    if isinstance(backend, FastCpuBackend)
                    else shared_engine()
                )
            if self._native_session_encoder is None:
                assert callable(materialize)
                tokenizer_json = materialize().encode("utf-8")
                self._native_session_encoder = _native.CallbackEncoder.native_fast_cpu(
                    tokenizer_json,
                    spec.family,
                    spec.artifact_sha256,
                    spec.margin,
                    spec.effective_l_max,
                    spec.has_normalizer,
                    pclass_table(),
                )
        except (BackendExecutionFault, RuntimeError, ValueError) as error:
            self._session_repair_guard = {
                "reason": "native_repair_initialisation_guard",
                "error": type(error).__name__,
                "message": str(error),
            }
            self._native_session_encoder = None
        return self._native_session_encoder

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
        if self._native_session_encoder is not None:
            return {
                "status": "active",
                "request_path": "rust_native",
                **self._native_session_encoder.stats(),
            }
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
            if eligible is None:
                return {"status": "reference_only", "backend": BACKEND_REFERENCE}
            return {"status": "not_initialised", "eligible_backend": eligible}
        return {"status": "reference_only", "backend": BACKEND_REFERENCE}

    def _semantic_fingerprint(
        self,
        repair: _SessionRepair | None = None,
        *,
        native_fast_cpu: bool = False,
    ) -> bytes:
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
                BACKEND_FAST_CPU
                if native_fast_cpu
                else (
                    str(repair_stats.get("backend"))
                    if repair is not None
                    else BACKEND_REFERENCE
                )
            ),
            (
                "toktier-fast-repair-v1"
                if native_fast_cpu
                else (repair.config_id if repair is not None else "")
            ),
            str(repair_stats.get("engine_version") or ""),
            str(repair_stats.get("engine_digest") or ""),
            self._snapshot.fast_cpu_engine.version or "",
            self._snapshot.fast_cpu_engine.source_digest or "",
            "\x1f".join(self._snapshot.fast_cpu_engine.build_flags),
            self._snapshot.fast_cpu_engine.toolchain or "",
            self._snapshot.fast_cpu_engine.config_digest or "",
        ):
            raw = component.encode("utf-8")
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
        return digest.digest()


def from_pretrained(
    repo_id: str,
    *,
    revision: str | None = None,
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
    """Load a model repository, admitting acceleration by tokenizer content.

    The repository name chooses where to look, not whether acceleration is
    safe.  TokTier hashes the resolved tokenizer file and admits a canonical
    family only when those bytes equal a packaged anchor or an exact identity
    in the digest-protected verified-sibling registry.  Registered
    canonicalisation and serialisation variants execute the audited canonical
    anchor through the ordinary CPU/GPU router.  Unknown or changed content is
    still usable under policies that permit the reference route, and then only
    through Hugging Face; ``REQUIRE_ACCELERATED`` raises when no certified
    accelerated path is eligible.

    When ``revision`` is omitted, audited sibling and canonical repositories
    use their recorded immutable revision; an otherwise unknown repository
    resolves ``main``.  ``load(family)`` remains the family-id API and is not
    affected by this entry point.
    """
    if not isinstance(repo_id, str) or not repo_id or repo_id != repo_id.strip():
        raise ValueError("repo_id must be a non-empty, unpadded string")
    if revision is not None and (
        not isinstance(revision, str) or not revision or revision != revision.strip()
    ):
        raise ValueError("revision must be a non-empty, unpadded string or None")
    resolved_config = config if config is not None else Config.resolve()
    active_manifest = (
        manifest if manifest is not None else ArtifactManifest.load(ARTIFACT_MANIFEST)
    )
    resolved = resolve_model_repository(
        repo_id=repo_id,
        revision=revision,
        config=resolved_config,
        manifest=active_manifest,
        aliases=shipped_sibling_aliases(),
    )
    tokenizer = load(
        resolved.family,
        store=store,
        device=device,
        config=resolved_config,
        policy=policy,
        manifest=resolved.manifest,
        cache_budget_bytes=cache_budget_bytes,
        repair_backend=repair_backend,
        gpu_delivery=gpu_delivery,
        gpu_min_bytes=gpu_min_bytes,
    )
    tokenizer._model_resolution = resolved.resolution
    return tokenizer


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
