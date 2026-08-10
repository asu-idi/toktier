"""Routing stage 2: a pure function from facts to an immutable plan.

Contract reference: ``docs/contracts/routing.md`` Sections 2, 3 and 5.
``plan(snapshot, policy, registry, config) -> RoutePlan`` has no I/O, no
clock and no randomness: the same facts always produce the same plan, so
a plan can be produced in a test without hardware.

The plan records the selected backend, the ordered fallback chain
(always ending at the reference backend), and one reason entry for every
accelerated option that was considered and not selected.

Check order (stable, and the order the reason codes are assigned in):

1. policy is REFERENCE                      -> ``R_POLICY_REFERENCE``
2. GPU disabled by configuration            -> ``R_GPU_DISABLED``
3. backend module not importable            -> ``R_BACKEND_UNAVAILABLE``
4. device facts absent -- two honest cases:
   enumeration never performed              -> ``R_ACCELERATOR_NOT_ADOPTED``
   probe ran, no usable device              -> ``R_NO_GPU_DETECTED``
5. no eligible registry identity/status     -> ``R_UNCERTIFIED_ARTIFACT``
6. certificate premises lost in-process     -> ``R_UNCERTIFIED_ARTIFACT``
7. oracle version outside certified set     -> ``R_ORACLE_MISMATCH``
8. CPU-engine binding does not verify       -> ``R_ENGINE_BINDING_MISMATCH``
9. device architecture not listed           -> ``R_SM_UNCERTIFIED``
10. driver below the certified minimum      -> ``R_DRIVER_TOO_OLD``
11. kernel build attempted and failed       -> ``R_KERNEL_BUILD_FAILED``
12. a bound digest does not verify          -> ``R_KERNEL_DIGEST_MISMATCH``
13. bound flags/toolchain do not verify     -> ``R_UNCERTIFIED_ARTIFACT``

Checks 1-4 are facts about the machine or the requesting path (the
unprobed case of check 4 is a fact about the caller, not the hardware)
and hold under every policy.
Checks 5-9, 11 and 12 are certification statements, so ``EXPERIMENTAL``
may proceed past them; each one it passes is recorded as a waiver and
reported by ``explain()``, never silently. Check 10 blocks under every
policy, because a kernel that failed to build cannot run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..backends.fast_cpu import ENGINE_DELIVERY, ENGINE_MODULE
from ..config import Config
from ..errors import (
    BackendUnavailable,
    CudaDriverTooOld,
    KernelIncompatible,
    OracleVersionUnsupported,
    ToktierError,
    UncertifiedTokenizer,
    UnsupportedConfig,
)
from ..policy import (
    BACKEND_FAST_CPU,
    BACKEND_GPU,
    BACKEND_REFERENCE,
    PlanReason,
    ReasonCode,
    RoutePlan,
    RoutingPolicy,
)
from .probe import ProbeSnapshot
from .registry_view import (
    ELIGIBLE_STATUSES,
    STATUS_CERTIFIED,
    STATUS_CERTIFIED_SOURCE,
    STATUS_EXPERIMENTAL,
    STATUS_UNSUPPORTED,
    BackendEntry,
    RegistryView,
)

__all__ = [
    "ACCELERATED_BACKENDS",
    "BackendAssessment",
    "assess_backend",
    "assessments_for",
    "plan",
    "reference_plan",
]

#: Accelerated backend ids, in the order they are considered. Mirrors
#: the frozen backend namespace of routing.md Section 4; it is not a
#: routing mapping, so it does not belong in the registry.
ACCELERATED_BACKENDS: tuple[str, ...] = (BACKEND_GPU, BACKEND_FAST_CPU)


@dataclass(frozen=True)
class BackendAssessment:
    """Result of evaluating one accelerated backend.

    ``blocking`` is the reason the backend was not selected, or ``None``
    when it is eligible. ``waived`` lists certification gaps that only
    ``EXPERIMENTAL`` policy permitted; they are diagnostics, not plan
    reasons, because the backend they describe was selected.
    """

    backend: str
    eligible: bool
    blocking: PlanReason | None = None
    waived: tuple[PlanReason, ...] = ()


def _reason(
    code: ReasonCode, backend: str, **detail: object
) -> PlanReason:
    return PlanReason(code=code, backend=backend, detail=dict(detail))


def _version_tuple(value: str) -> tuple[int, ...] | None:
    """Leading dotted-numeric part of a version string."""
    parts: list[int] = []
    for chunk in value.split("."):
        digits = ""
        for character in chunk:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def _driver_satisfies(installed: str | None, minimum: str) -> bool:
    """Whether the installed driver meets the certified minimum.

    An absent or unparseable version is not treated as satisfying: the
    certified floor is a claim we either verify or do not.
    """
    if installed is None:
        return False
    left = _version_tuple(installed)
    right = _version_tuple(minimum)
    if left is None or right is None:
        return False
    return left >= right


class _Blocked(Exception):
    """Internal control flow: a check refused the backend."""

    def __init__(self, reason: PlanReason) -> None:
        super().__init__(reason.code.value)
        self.reason = reason


class _RefuseFn(Protocol):
    """Callback a check uses to refuse (or waive) an option."""

    def __call__(self, reason: PlanReason, *, waivable: bool) -> None:
        """Refuse the backend, unless the policy waives this check."""


def assess_backend(
    backend: str,
    snapshot: ProbeSnapshot,
    policy: RoutingPolicy,
    registry: RegistryView,
    config: Config,
) -> BackendAssessment:
    """Evaluate one accelerated backend. Pure; no I/O."""
    waived: list[PlanReason] = []

    def refuse(reason: PlanReason, *, waivable: bool) -> None:
        if waivable and policy is RoutingPolicy.EXPERIMENTAL:
            waived.append(reason)
            return
        raise _Blocked(reason)

    try:
        _assess(backend, snapshot, policy, registry, config, refuse)
    except _Blocked as blocked:
        return BackendAssessment(
            backend=backend,
            eligible=False,
            blocking=blocked.reason,
            waived=tuple(waived),
        )
    return BackendAssessment(
        backend=backend, eligible=True, waived=tuple(waived)
    )


def _assess(
    backend: str,
    snapshot: ProbeSnapshot,
    policy: RoutingPolicy,
    registry: RegistryView,
    config: Config,
    refuse: _RefuseFn,
) -> None:
    # 1. REFERENCE considers no accelerated option at all.
    if policy is RoutingPolicy.REFERENCE:
        refuse(
            _reason(ReasonCode.R_POLICY_REFERENCE, backend),
            waivable=False,
        )

    # 2. Configuration switched the GPU off. This is a fact about the
    #    requested configuration, so no policy overrides it.
    if backend == BACKEND_GPU and config.disable_gpu:
        refuse(
            _reason(ReasonCode.R_GPU_DISABLED, backend, source="config"),
            waivable=False,
        )

    # 3. The backend has to exist before anything else matters.
    if backend not in snapshot.importable_backends:
        refuse(
            _reason(
                ReasonCode.R_BACKEND_UNAVAILABLE,
                backend,
                missing=backend,
            ),
            waivable=False,
        )

    # 4. And it has to have something to run on. Two honest cases: a
    #    probe that ran and found nothing is a hardware fact; a probe
    #    that was never supplied means the caller adopts no accelerator
    #    runtime on this path, and the reason must not claim otherwise.
    if backend == BACKEND_GPU and not snapshot.devices:
        if snapshot.devices_probed:
            refuse(
                _reason(ReasonCode.R_NO_GPU_DETECTED, backend),
                waivable=False,
            )
        else:
            refuse(
                _reason(
                    ReasonCode.R_ACCELERATOR_NOT_ADOPTED,
                    backend,
                    cause="no_device_probe_supplied",
                ),
                waivable=False,
            )

    # 5. Certification identity and per-backend status.
    match = snapshot.certification
    entry: BackendEntry | None = None
    if match is not None:
        entry = match.record.backends.get(backend)
    if match is None or entry is None:
        refuse(
            _reason(
                ReasonCode.R_UNCERTIFIED_ARTIFACT,
                backend,
                cause="no_record" if match is None else "no_backend_entry",
                artifact_sha256=snapshot.artifact_sha256,
                family=snapshot.family,
            ),
            waivable=True,
        )
    elif entry.status == STATUS_UNSUPPORTED:
        # Known not to work: never planned, not even under EXPERIMENTAL.
        refuse(
            _reason(
                ReasonCode.R_UNCERTIFIED_ARTIFACT,
                backend,
                cause="status_unsupported",
                status=entry.status,
            ),
            waivable=False,
        )
    elif entry.status not in ELIGIBLE_STATUSES:
        refuse(
            _reason(
                ReasonCode.R_UNCERTIFIED_ARTIFACT,
                backend,
                cause="status_not_eligible",
                status=entry.status,
            ),
            waivable=entry.status == STATUS_EXPERIMENTAL,
        )

    if match is None or entry is None:
        # Only reachable under EXPERIMENTAL, which waived check 5; the
        # remaining checks all read the record, so there is nothing
        # further to verify.
        return

    # 5b. Binding view of the kernel delivery in effect (registry
    #     records may refine the entry per delivery; JIT-era records
    #     verify as themselves). A refined view that is itself not an
    #     eligible status refuses like check 5.
    view = _delivery_view(entry, snapshot)
    if view is not entry and view.status not in ELIGIBLE_STATUSES:
        refuse(
            _reason(
                ReasonCode.R_UNCERTIFIED_ARTIFACT,
                backend,
                cause="delivery_status_not_eligible",
                delivery=(
                    snapshot.kernel_cache.delivery
                    or snapshot.kernel_cache.preferred_delivery
                ),
                status=view.status,
            ),
            waivable=view.status == STATUS_EXPERIMENTAL,
        )
    if (
        backend == BACKEND_GPU
        and view is not entry
        and view.status == STATUS_CERTIFIED
        and (
            view.host_source_digest is None
            or not view.host_build_flags
            or view.host_toolchain is None
        )
    ):
        refuse(
            _reason(
                ReasonCode.R_UNCERTIFIED_ARTIFACT,
                backend,
                cause="native_host_binding_missing",
                delivery="prebuilt",
            ),
            waivable=True,
        )

    # 6. One loader, one flag set per process (registry.md 3.2).
    if backend == BACKEND_GPU and snapshot.kernel_cache.loaded_flag_sets > 1:
        refuse(
            _reason(
                ReasonCode.R_UNCERTIFIED_ARTIFACT,
                backend,
                cause="multiple_kernel_flag_sets",
                loaded_flag_sets=snapshot.kernel_cache.loaded_flag_sets,
            ),
            waivable=True,
        )

    # 7. Oracle version inside the certified set for this record.
    oracle = registry.oracle(match.record.oracle_id)
    certified_versions = oracle.certified_versions if oracle else ()
    if snapshot.oracle_version not in certified_versions:
        refuse(
            _reason(
                ReasonCode.R_ORACLE_MISMATCH,
                backend,
                package=snapshot.oracle_package,
                installed=snapshot.oracle_version,
                certified=list(certified_versions),
            ),
            waivable=True,
        )

    # 8. The corrected engine is linked into the core native extension. Its
    #    source set, release build flags and exact Rust compiler are reported
    #    by the executing extension and bound by the registry. Its row carries
    #    no device/kernel facts, so a successful check terminates here.
    if backend == BACKEND_FAST_CPU:
        facts = snapshot.fast_cpu_engine
        if entry.engine != "gigatoken":
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="engine",
                    installed="gigatoken",
                    certified=entry.engine,
                ),
                waivable=True,
            )
        if entry.engine_delivery != ENGINE_DELIVERY:
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="engine_delivery",
                    installed=ENGINE_DELIVERY,
                    certified=entry.engine_delivery,
                ),
                waivable=True,
            )
        if entry.engine_module != ENGINE_MODULE:
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="engine_module",
                    installed=ENGINE_MODULE,
                    certified=entry.engine_module,
                ),
                waivable=True,
            )
        if facts.version != entry.engine_version:
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="engine_version",
                    installed=facts.version,
                    certified=entry.engine_version,
                ),
                waivable=True,
            )
        if entry.status != STATUS_CERTIFIED_SOURCE:
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="status",
                    expected=STATUS_CERTIFIED_SOURCE,
                    observed=entry.status,
                ),
                waivable=True,
            )
        if facts.source_digest != entry.source_digest:
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="source_digest",
                    expected_digest=entry.source_digest,
                    observed_digest=facts.source_digest,
                ),
                waivable=True,
            )
        if facts.build_flags != entry.build_flags:
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="build_flags",
                    expected=list(entry.build_flags),
                    observed=list(facts.build_flags),
                ),
                waivable=True,
            )
        if facts.toolchain != entry.toolchain:
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="toolchain",
                    expected=entry.toolchain,
                    observed=facts.toolchain,
                ),
                waivable=True,
            )
        if entry.config_id != "toktier-fast-repair-v1":
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="config_id",
                    expected="toktier-fast-repair-v1",
                    observed=entry.config_id,
                ),
                waivable=True,
            )
        if facts.config_digest != entry.config_digest:
            refuse(
                _reason(
                    ReasonCode.R_ENGINE_BINDING_MISMATCH,
                    backend,
                    axis="config_digest",
                    expected_digest=entry.config_digest,
                    observed_digest=facts.config_digest,
                ),
                waivable=True,
            )
        return

    # 9. Device architecture explicitly judged.
    if view.devices:
        architectures = {device.architecture for device in snapshot.devices}
        if not architectures & set(view.devices):
            refuse(
                _reason(
                    ReasonCode.R_SM_UNCERTIFIED,
                    backend,
                    observed=sorted(architectures),
                    certified=list(view.devices),
                ),
                waivable=True,
            )

    # 10. Driver at or above the certified floor.
    if view.driver_min is not None and not _driver_satisfies(
        snapshot.driver_version, view.driver_min
    ):
        refuse(
            _reason(
                ReasonCode.R_DRIVER_TOO_OLD,
                backend,
                installed=snapshot.driver_version,
                required=view.driver_min,
            ),
            waivable=True,
        )

    # 11. A build that failed leaves nothing to run.
    if snapshot.kernel_cache.build_failed:
        refuse(
            _reason(
                ReasonCode.R_KERNEL_BUILD_FAILED,
                backend,
                error=snapshot.kernel_cache.build_error,
            ),
            waivable=False,
        )

    # 12. Bound digests. A value that was never computed counts as a
    #     failed verification, never as a pass.
    for label, expected, observed in _bound_digests(view, snapshot):
        if expected is not None and observed != expected:
            refuse(
                _reason(
                    ReasonCode.R_KERNEL_DIGEST_MISMATCH,
                    backend,
                    digest=label,
                    expected_digest=expected,
                    observed_digest=observed,
                ),
                waivable=True,
            )

    # A prebuilt certificate is the conjunction of the exact CUDA image and
    # the source-certified Rust host that selects, launches, postprocesses,
    # and falls back from it.  These are deliberately distinct bindings: a
    # stable fatbin cannot certify drifted host behavior.
    if view.status == STATUS_CERTIFIED and view.host_source_digest is not None:
        if tuple(view.host_build_flags) != tuple(
            snapshot.kernel_cache.host_build_flags
        ):
            refuse(
                _reason(
                    ReasonCode.R_UNCERTIFIED_ARTIFACT,
                    backend,
                    cause="host_build_flags_mismatch",
                    certified=list(view.host_build_flags),
                    observed=list(snapshot.kernel_cache.host_build_flags),
                ),
                waivable=True,
            )
        if view.host_toolchain != snapshot.kernel_cache.host_toolchain:
            refuse(
                _reason(
                    ReasonCode.R_UNCERTIFIED_ARTIFACT,
                    backend,
                    cause="host_toolchain_mismatch",
                    certified=view.host_toolchain,
                    observed=snapshot.kernel_cache.host_toolchain,
                ),
                waivable=True,
            )

    # 13. Bound build flags and toolchain constraint.
    if view.status == STATUS_CERTIFIED_SOURCE:
        if tuple(view.build_flags) != tuple(snapshot.kernel_cache.build_flags):
            refuse(
                _reason(
                    ReasonCode.R_UNCERTIFIED_ARTIFACT,
                    backend,
                    cause="build_flags_mismatch",
                    certified=list(view.build_flags),
                    observed=list(snapshot.kernel_cache.build_flags),
                ),
                waivable=True,
            )
        if view.toolchain is not None and (
            snapshot.kernel_cache.toolchain_satisfied is not True
        ):
            refuse(
                _reason(
                    ReasonCode.R_UNCERTIFIED_ARTIFACT,
                    backend,
                    cause="toolchain_unverified",
                    constraint=view.toolchain,
                    observed=snapshot.kernel_cache.toolchain,
                ),
                waivable=True,
            )


def _delivery_view(entry: BackendEntry, snapshot: ProbeSnapshot) -> BackendEntry:
    """The binding view for the kernel delivery in effect.

    A loaded process verifies the sub-entry of the delivery it loaded.
    Before a load, the view is the delivery the loader would prefer:
    prebuilt when a fatbin is shipped and the record refines it, JIT
    otherwise. Records without delivery refinements verify as
    themselves, which is exactly the JIT-era behavior.
    """
    if not entry.deliveries:
        return entry
    active = (
        snapshot.kernel_cache.delivery
        or snapshot.kernel_cache.preferred_delivery
    )
    if active is None:
        active = (
            "prebuilt"
            if snapshot.kernel_cache.prebuilt_available
            and "prebuilt" in entry.deliveries
            else "jit"
        )
    return entry.deliveries.get(active, entry)


def _bound_digests(
    entry: BackendEntry, snapshot: ProbeSnapshot
) -> tuple[tuple[str, str | None, str | None], ...]:
    """(label, certified digest, observed digest) triples to verify."""
    cache = snapshot.kernel_cache
    if entry.status == STATUS_CERTIFIED:
        return (
            ("binary", entry.binary_digest, cache.binary_digest),
            (
                "host_source",
                entry.host_source_digest,
                cache.host_source_digest,
            ),
        )
    if entry.status == STATUS_CERTIFIED_SOURCE:
        return (
            ("source", entry.source_digest, cache.source_digest),
            ("class_table", entry.class_table_digest, cache.class_table_digest),
        )
    return ()


def reference_plan(policy: RoutingPolicy, reasons: Sequence[PlanReason]) -> RoutePlan:
    """A plan that runs the reference backend and nothing else."""
    return RoutePlan(
        policy=policy,
        backend=BACKEND_REFERENCE,
        fallback_chain=(BACKEND_REFERENCE,),
        reasons=tuple(reasons),
    )


def plan(
    snapshot: ProbeSnapshot,
    policy: RoutingPolicy,
    registry: RegistryView,
    config: Config,
) -> RoutePlan:
    """Build the immutable route plan. Pure function.

    Raises ``OracleVersionUnsupported`` when the reference backend
    itself cannot run, and -- under ``REQUIRE_ACCELERATED`` only -- the
    specific error for the reason no accelerated backend was eligible.
    """
    if BACKEND_REFERENCE not in snapshot.importable_backends:
        raise OracleVersionUnsupported(
            f"the {snapshot.oracle_package} package is not installed, so the "
            "reference backend cannot run",
            details={
                "package": snapshot.oracle_package,
                "installed": snapshot.oracle_version,
                "certified": None,
            },
        )

    assessments = assessments_for(snapshot, policy, registry, config)
    eligible = tuple(item.backend for item in assessments if item.eligible)
    reasons = tuple(
        item.blocking
        for item in assessments
        if item.blocking is not None
    )

    if not eligible:
        if policy is RoutingPolicy.REQUIRE_ACCELERATED:
            raise _require_accelerated_error(assessments, snapshot)
        return reference_plan(policy, reasons)

    return RoutePlan(
        policy=policy,
        backend=eligible[0],
        fallback_chain=(*eligible, BACKEND_REFERENCE),
        reasons=reasons,
    )


def assessments_for(
    snapshot: ProbeSnapshot,
    policy: RoutingPolicy,
    registry: RegistryView,
    config: Config,
) -> tuple[BackendAssessment, ...]:
    """Per-backend assessments behind a plan, for diagnostics."""
    backends: list[str] = [BACKEND_GPU]
    match = snapshot.certification
    if (
        match is not None
        and BACKEND_FAST_CPU in match.record.backends
    ):
        backends.append(BACKEND_FAST_CPU)
    return tuple(
        assess_backend(backend, snapshot, policy, registry, config)
        for backend in backends
    )


def _require_accelerated_error(
    assessments: Sequence[BackendAssessment],
    snapshot: ProbeSnapshot,
) -> ToktierError:
    """Map the first blocking reason to its specific error.

    ``REQUIRE_ACCELERATED`` constrains plan time only: it asks for the
    cause, not for a generic refusal, so the reason code travels into
    ``details`` in every case.
    """
    blocking = [item.blocking for item in assessments if item.blocking is not None]
    reason = blocking[0] if blocking else None
    if reason is None:  # pragma: no cover - defensive
        return BackendUnavailable(
            "no accelerated backend is configured",
            details={"backend": None, "missing": None},
        )
    code = reason.code
    detail = dict(reason.detail)
    backend = reason.backend
    if code is ReasonCode.R_GPU_DISABLED:
        return UnsupportedConfig(
            "policy requires an accelerated backend while the configuration "
            "disables the GPU",
            details={
                "option": "disable_gpu",
                "value": True,
                "reason": code.value,
            },
        )
    if code is ReasonCode.R_BACKEND_UNAVAILABLE:
        return BackendUnavailable(
            f"backend {backend!r} is not installed",
            details={"backend": backend, "missing": detail.get("missing")},
        )
    if code is ReasonCode.R_NO_GPU_DETECTED:
        return BackendUnavailable(
            f"backend {backend!r} found no usable device",
            details={"backend": backend, "missing": "cuda_device"},
        )
    if code is ReasonCode.R_ACCELERATOR_NOT_ADOPTED:
        return BackendUnavailable(
            f"backend {backend!r} is not adopted on this path: device "
            "enumeration was not performed (the integrating layer supplied "
            "no device probe), so no accelerated backend could be planned; "
            "this says nothing about the machine's hardware",
            details={"backend": backend, "missing": "device_probe"},
        )
    if code is ReasonCode.R_DRIVER_TOO_OLD:
        return CudaDriverTooOld(
            "the installed driver is below the certified minimum",
            details={
                "installed": detail.get("installed"),
                "required": detail.get("required"),
            },
        )
    if code in (
        ReasonCode.R_SM_UNCERTIFIED,
        ReasonCode.R_KERNEL_DIGEST_MISMATCH,
        ReasonCode.R_KERNEL_BUILD_FAILED,
    ):
        return KernelIncompatible(
            "the kernel constraints did not verify for this configuration",
            details={
                "backend": backend,
                "reason_code": code.value,
                "sm": detail.get("observed"),
                "expected_digest": detail.get("expected_digest"),
                "observed_digest": detail.get("observed_digest"),
                "class_table_digest": detail.get("digest"),
            },
        )
    # R_UNCERTIFIED_ARTIFACT, R_ORACLE_MISMATCH, and
    # R_ENGINE_BINDING_MISMATCH mean: this
    # artifact has no eligible certification identity under the current
    # conditions. The reason code in details keeps them distinguishable.
    return UncertifiedTokenizer(
        "no certified accelerated path is eligible for this artifact",
        details={
            "artifact_sha256": snapshot.artifact_sha256,
            "family": snapshot.family,
            "reason_code": code.value,
            **{
                key: value
                for key, value in detail.items()
                if key
                in (
                    "cause",
                    "installed",
                    "certified",
                    "package",
                    "status",
                    "axis",
                    "expected_digest",
                    "observed_digest",
                )
            },
        },
    )
