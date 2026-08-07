"""Planning matrix: every policy against every probe state.

Contract reference: ``docs/contracts/routing.md`` Sections 2, 3 and 5.

The table below is the readable form of the routing contract: one row
per machine state, one column per policy, and the expected plan plus its
reason codes in each cell. A change in routing behavior has to change
this table, which is the point.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import _support as support
import pytest

from toktier.config import Config
from toktier.errors import (
    BackendUnavailable,
    CudaDriverTooOld,
    KernelIncompatible,
    OracleVersionUnsupported,
    ToktierError,
    UncertifiedTokenizer,
    UnsupportedConfig,
)
from toktier.policy import (
    BACKEND_GPU,
    BACKEND_REFERENCE,
    ReasonCode,
    RoutingPolicy,
)
from toktier.routing.plan import assessments_for, plan
from toktier.routing.probe import DeviceInfo, ProbeSnapshot
from toktier.routing.registry_view import RegistryView

CERTIFIED = RoutingPolicy.CERTIFIED
REFERENCE = RoutingPolicy.REFERENCE
REQUIRE = RoutingPolicy.REQUIRE_ACCELERATED
EXPERIMENTAL = RoutingPolicy.EXPERIMENTAL

#: Every plan-time reason code of routing.md Section 5.1. The matrix is
#: expected to exercise all of them; see ``test_matrix_covers_...``.
PLAN_TIME_CODES = frozenset(
    {
        ReasonCode.R_POLICY_REFERENCE,
        ReasonCode.R_UNCERTIFIED_ARTIFACT,
        ReasonCode.R_ORACLE_MISMATCH,
        ReasonCode.R_BACKEND_UNAVAILABLE,
        ReasonCode.R_GPU_DISABLED,
        ReasonCode.R_NO_GPU_DETECTED,
        ReasonCode.R_ACCELERATOR_NOT_ADOPTED,
        ReasonCode.R_DRIVER_TOO_OLD,
        ReasonCode.R_SM_UNCERTIFIED,
        ReasonCode.R_KERNEL_DIGEST_MISMATCH,
        ReasonCode.R_KERNEL_BUILD_FAILED,
    }
)


@dataclass(frozen=True)
class Expect:
    """Expected outcome of one (state, policy) cell."""

    backend: str | None = None
    reasons: tuple[ReasonCode, ...] = ()
    waived: tuple[ReasonCode, ...] = ()
    error: type[ToktierError] | None = None


@dataclass(frozen=True)
class State:
    """One machine state plus its expected plan under each policy."""

    build: Callable[[], tuple[ProbeSnapshot, RegistryView, Config]]
    expect: dict[RoutingPolicy, Expect] = field(default_factory=dict)


def _blocked(code: ReasonCode) -> Expect:
    return Expect(backend=BACKEND_REFERENCE, reasons=(code,))


def _reference_only() -> Expect:
    return _blocked(ReasonCode.R_POLICY_REFERENCE)


def _eligible(waived: tuple[ReasonCode, ...] = ()) -> Expect:
    return Expect(backend=BACKEND_GPU, waived=waived)


def _state(
    *,
    registry_kwargs: dict[str, Any] | None = None,
    snapshot_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
) -> Callable[[], tuple[ProbeSnapshot, RegistryView, Config]]:
    def build() -> tuple[ProbeSnapshot, RegistryView, Config]:
        view = support.registry(**(registry_kwargs or {}))
        snapshot = support.snapshot(
            registry_view=view, **(snapshot_kwargs or {})
        )
        return snapshot, view, support.config(**(config_kwargs or {}))

    return build


def _uncertified_case(
    *,
    registry_kwargs: dict[str, Any] | None = None,
    snapshot_kwargs: dict[str, Any] | None = None,
    config_kwargs: dict[str, Any] | None = None,
    code: ReasonCode = ReasonCode.R_UNCERTIFIED_ARTIFACT,
    error: type[ToktierError] = UncertifiedTokenizer,
    waivable: bool = True,
) -> State:
    """A state where the accelerated path is closed for one reason."""
    return State(
        build=_state(
            registry_kwargs=registry_kwargs,
            snapshot_kwargs=snapshot_kwargs,
            config_kwargs=config_kwargs,
        ),
        expect={
            CERTIFIED: _blocked(code),
            REFERENCE: _reference_only(),
            REQUIRE: Expect(error=error),
            EXPERIMENTAL: _eligible((code,)) if waivable else _blocked(code),
        },
    )


STATES: dict[str, State] = {
    # -- the accelerated path is open ---------------------------------
    "certified_source_all_green": State(
        build=_state(),
        expect={
            CERTIFIED: _eligible(),
            REFERENCE: _reference_only(),
            REQUIRE: _eligible(),
            EXPERIMENTAL: _eligible(),
        },
    ),
    "certified_binary_green": State(
        build=_state(
            registry_kwargs={"backends": {BACKEND_GPU: support.certified_entry()}}
        ),
        expect={
            CERTIFIED: _eligible(),
            REFERENCE: _reference_only(),
            REQUIRE: _eligible(),
            EXPERIMENTAL: _eligible(),
        },
    ),
    "composition_grant": State(
        build=_state(
            registry_kwargs={
                "artifact_sha256": "9" * 64,
                "compositions": ((support.PIPELINE_ID, support.ADDED_FRONTEND_ID),),
            }
        ),
        expect={
            CERTIFIED: _eligible(),
            REFERENCE: _reference_only(),
            REQUIRE: _eligible(),
            EXPERIMENTAL: _eligible(),
        },
    ),
    # -- machine facts: never waived ----------------------------------
    "gpu_disabled_by_config": State(
        build=_state(config_kwargs={"disable_gpu": True}),
        expect={
            CERTIFIED: _blocked(ReasonCode.R_GPU_DISABLED),
            REFERENCE: _reference_only(),
            REQUIRE: Expect(error=UnsupportedConfig),
            EXPERIMENTAL: _blocked(ReasonCode.R_GPU_DISABLED),
        },
    ),
    "backend_not_importable": State(
        build=_state(snapshot_kwargs={"gpu_importable": False}),
        expect={
            CERTIFIED: _blocked(ReasonCode.R_BACKEND_UNAVAILABLE),
            REFERENCE: _reference_only(),
            REQUIRE: Expect(error=BackendUnavailable),
            EXPERIMENTAL: _blocked(ReasonCode.R_BACKEND_UNAVAILABLE),
        },
    ),
    "no_device_present": State(
        build=_state(snapshot_kwargs={"devices": ()}),
        expect={
            CERTIFIED: _blocked(ReasonCode.R_NO_GPU_DETECTED),
            REFERENCE: _reference_only(),
            REQUIRE: Expect(error=BackendUnavailable),
            EXPERIMENTAL: _blocked(ReasonCode.R_NO_GPU_DETECTED),
        },
    ),
    "device_probe_never_supplied": State(
        # No device probe ran (the facade path): the honest reason is
        # that no accelerator runtime was adopted, not a hardware claim.
        build=_state(
            snapshot_kwargs={"devices": (), "devices_probed": False}
        ),
        expect={
            CERTIFIED: _blocked(ReasonCode.R_ACCELERATOR_NOT_ADOPTED),
            REFERENCE: _reference_only(),
            REQUIRE: Expect(error=BackendUnavailable),
            EXPERIMENTAL: _blocked(ReasonCode.R_ACCELERATOR_NOT_ADOPTED),
        },
    ),
    "kernel_build_failed": State(
        build=_state(
            snapshot_kwargs={
                "kernel_cache": support.gpu_ready_kernel_cache(
                    built=False, build_failed=True, build_error="nvcc missing"
                )
            }
        ),
        expect={
            CERTIFIED: _blocked(ReasonCode.R_KERNEL_BUILD_FAILED),
            REFERENCE: _reference_only(),
            REQUIRE: Expect(error=KernelIncompatible),
            EXPERIMENTAL: _blocked(ReasonCode.R_KERNEL_BUILD_FAILED),
        },
    ),
    # -- certification statements: EXPERIMENTAL may waive -------------
    "artifact_not_in_registry": _uncertified_case(
        snapshot_kwargs={"artifact_sha256": "1" * 64, "pipeline_fingerprint": None}
    ),
    "composition_absent": _uncertified_case(
        registry_kwargs={"artifact_sha256": "9" * 64}
    ),
    "backend_entry_absent": _uncertified_case(registry_kwargs={"backends": {}}),
    "status_experimental": _uncertified_case(
        registry_kwargs={
            "backends": {BACKEND_GPU: support.certified_source_entry(
                status="experimental"
            )}
        }
    ),
    "status_unsupported": _uncertified_case(
        registry_kwargs={
            "backends": {BACKEND_GPU: {"status": "unsupported"}}
        },
        waivable=False,
    ),
    "oracle_outside_certified_set": _uncertified_case(
        snapshot_kwargs={"oracle_version": "0.23.0"},
        code=ReasonCode.R_ORACLE_MISMATCH,
    ),
    "device_architecture_unlisted": _uncertified_case(
        snapshot_kwargs={
            "devices": (DeviceInfo(index=0, name="older", architecture="sm_75"),)
        },
        code=ReasonCode.R_SM_UNCERTIFIED,
        error=KernelIncompatible,
    ),
    "driver_below_certified_minimum": _uncertified_case(
        snapshot_kwargs={"driver_version": "550.0"},
        code=ReasonCode.R_DRIVER_TOO_OLD,
        error=CudaDriverTooOld,
    ),
    "class_table_digest_mismatch": _uncertified_case(
        snapshot_kwargs={
            "kernel_cache": support.gpu_ready_kernel_cache(
                class_table_digest="0" * 64
            )
        },
        code=ReasonCode.R_KERNEL_DIGEST_MISMATCH,
        error=KernelIncompatible,
    ),
    "source_digest_never_computed": _uncertified_case(
        snapshot_kwargs={
            "kernel_cache": support.gpu_ready_kernel_cache(source_digest=None)
        },
        code=ReasonCode.R_KERNEL_DIGEST_MISMATCH,
        error=KernelIncompatible,
    ),
    "binary_digest_mismatch": _uncertified_case(
        registry_kwargs={"backends": {BACKEND_GPU: support.certified_entry()}},
        snapshot_kwargs={
            "kernel_cache": support.gpu_ready_kernel_cache(binary_digest="0" * 64)
        },
        code=ReasonCode.R_KERNEL_DIGEST_MISMATCH,
        error=KernelIncompatible,
    ),
    "two_kernel_flag_sets_in_process": _uncertified_case(
        snapshot_kwargs={
            "kernel_cache": support.gpu_ready_kernel_cache(loaded_flag_sets=2)
        }
    ),
    "build_flags_differ_from_certificate": _uncertified_case(
        snapshot_kwargs={
            "kernel_cache": support.gpu_ready_kernel_cache(build_flags=("-O2",))
        }
    ),
    "toolchain_constraint_unverified": _uncertified_case(
        snapshot_kwargs={
            "kernel_cache": support.gpu_ready_kernel_cache(toolchain_satisfied=None)
        }
    ),
}

MATRIX = [
    (name, policy)
    for name, state in STATES.items()
    for policy in (CERTIFIED, REFERENCE, REQUIRE, EXPERIMENTAL)
    if policy in state.expect
]


@pytest.mark.parametrize(("name", "policy"), MATRIX, ids=lambda value: str(value))
def test_plan_matrix(name: str, policy: RoutingPolicy) -> None:
    """Each machine state plans as the contract describes, per policy."""
    state = STATES[name]
    snapshot, registry, config = state.build()
    expected = state.expect[policy]

    if expected.error is not None:
        with pytest.raises(expected.error) as caught:
            plan(snapshot, policy, registry, config)
        assert caught.value.code
        assert "reason_code" in caught.value.details or caught.value.details
        return

    route_plan = plan(snapshot, policy, registry, config)
    assert route_plan.backend == expected.backend
    assert route_plan.fallback_chain[-1] == BACKEND_REFERENCE
    assert tuple(reason.code for reason in route_plan.reasons) == expected.reasons

    assessments = assessments_for(snapshot, policy, registry, config)
    waived = tuple(
        reason.code
        for assessment in assessments
        if assessment.eligible
        for reason in assessment.waived
    )
    assert waived == expected.waived


def test_matrix_covers_every_plan_time_reason_code() -> None:
    """The matrix exercises all of routing.md Section 5.1."""
    seen: set[ReasonCode] = set()
    for name, state in STATES.items():
        for policy in state.expect:
            snapshot, registry, config = state.build()
            try:
                route_plan = plan(snapshot, policy, registry, config)
            except ToktierError:
                continue
            seen.update(reason.code for reason in route_plan.reasons)
            assessments = assessments_for(snapshot, policy, registry, config)
            seen.update(
                reason.code
                for assessment in assessments
                for reason in assessment.waived
            )
        assert name  # keeps the loop variable meaningful
    assert seen >= PLAN_TIME_CODES


def test_plan_is_a_pure_function() -> None:
    """Same facts, same plan -- twice, and independently constructed."""
    snapshot, registry, config = STATES["certified_source_all_green"].build()
    first = plan(snapshot, CERTIFIED, registry, config)
    second = plan(snapshot, CERTIFIED, registry, config)
    assert first == second

    other_snapshot, other_registry, other_config = STATES[
        "certified_source_all_green"
    ].build()
    assert plan(other_snapshot, CERTIFIED, other_registry, other_config) == first


def test_route_plan_is_immutable() -> None:
    """A plan cannot be edited after construction."""
    snapshot, registry, config = STATES["certified_source_all_green"].build()
    route_plan = plan(snapshot, CERTIFIED, registry, config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        route_plan.backend = BACKEND_REFERENCE  # type: ignore[misc]


def test_chain_always_ends_at_the_reference_backend() -> None:
    """Every plan can fall back to the definition of correct output."""
    for name, state in STATES.items():
        for policy, expected in state.expect.items():
            if expected.error is not None:
                continue
            snapshot, registry, config = state.build()
            route_plan = plan(snapshot, policy, registry, config)
            assert route_plan.fallback_chain[-1] == BACKEND_REFERENCE, name
            assert route_plan.backend == route_plan.fallback_chain[0], name


def test_missing_oracle_package_is_an_error_not_a_fallback() -> None:
    """Without the oracle package there is no correct output to produce."""
    snapshot, registry, config = STATES["certified_source_all_green"].build()
    without_reference = dataclasses.replace(
        snapshot, importable_backends=frozenset({BACKEND_GPU})
    )
    with pytest.raises(OracleVersionUnsupported) as caught:
        plan(without_reference, CERTIFIED, registry, config)
    assert caught.value.code == "ORACLE_VERSION_UNSUPPORTED"
    assert caught.value.details["package"] == "tokenizers"


def test_require_accelerated_reports_the_specific_cause() -> None:
    """The raised error names the reason, not a generic refusal."""
    snapshot, registry, config = STATES[
        "driver_below_certified_minimum"
    ].build()
    with pytest.raises(CudaDriverTooOld) as caught:
        plan(snapshot, REQUIRE, registry, config)
    assert caught.value.details["installed"] == "550.0"
    assert caught.value.details["required"] == "560.0"


def test_require_accelerated_does_not_blame_hardware_nobody_probed() -> None:
    """An unprobed path raises ``missing: device_probe``, not a device claim.

    The distinction matters on a machine that does have a GPU: a path
    that never enumerated devices must not report that none exist.
    """
    snapshot, registry, config = STATES["device_probe_never_supplied"].build()
    with pytest.raises(BackendUnavailable) as caught:
        plan(snapshot, REQUIRE, registry, config)
    assert caught.value.details["missing"] == "device_probe"
    assert caught.value.details["backend"] == BACKEND_GPU


def test_experimental_waivers_are_reported_not_hidden() -> None:
    """An uncertified path that ran says which checks it skipped."""
    snapshot, registry, config = STATES["artifact_not_in_registry"].build()
    assessments = assessments_for(snapshot, EXPERIMENTAL, registry, config)
    waived = [reason.code for item in assessments for reason in item.waived]
    assert ReasonCode.R_UNCERTIFIED_ARTIFACT in waived
    route_plan = plan(snapshot, EXPERIMENTAL, registry, config)
    assert route_plan.backend == BACKEND_GPU
    # The plan itself records no reason: nothing was excluded.
    assert route_plan.reasons == ()
