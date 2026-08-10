"""Delivery-aware routing: the planner verifies the delivery in effect.

Contract reference: ``docs/contracts/registry.md`` Section 3 (the
``certified`` status binds a binary digest; ``certified_source`` binds
the source set) and PLAN/148 (registry records may refine the GPU entry
per kernel delivery). The planner must verify the sub-entry of the
delivery the process actually runs -- prebuilt facts must be judged
against the fatbin's binary digest, JIT facts against the source
binding set -- and records without delivery refinements must plan
exactly as before.
"""

from __future__ import annotations

from typing import Any

import _support as support
import pytest

from toktier.policy import BACKEND_GPU, ReasonCode, RoutingPolicy
from toktier.routing.plan import assess_backend

PREBUILT_DIGEST = "d" * 64

#: The prebuilt driver floor named in the registry entry under test.
DRIVER_MIN = "580.65.06"

#: A driver above the floor (the support default, 570.1, is below it on
#: purpose: the floor test needs both sides).
DRIVER_OK = "580.82.07"


def _entry_with_deliveries(**prebuilt_overrides: Any) -> dict[str, Any]:
    prebuilt = support.certified_entry(
        binary_digest=PREBUILT_DIGEST,
        devices=["sm_89", "sm_120"],
        devices_experimental=["sm_75", "sm_80", "sm_86", "sm_90", "sm_100"],
        driver_min=DRIVER_MIN,
    )
    prebuilt.update(prebuilt_overrides)
    entry: dict[str, Any] = support.certified_source_entry(
        deliveries={
            "jit": support.certified_source_entry(),
            "prebuilt": prebuilt,
        }
    )
    return entry


def _assessment(
    entry: dict[str, Any],
    *,
    driver_version: str = DRIVER_OK,
    **cache_overrides: Any,
) -> Any:
    view = support.registry(backends={BACKEND_GPU: entry})
    snapshot = support.snapshot(
        registry_view=view,
        driver_version=driver_version,
        kernel_cache=support.gpu_ready_kernel_cache(**cache_overrides),
    )
    return assess_backend(
        BACKEND_GPU,
        snapshot,
        RoutingPolicy.CERTIFIED,
        view,
        support.config(),
    )


def test_prebuilt_delivery_verifies_binary_digest() -> None:
    """A prebuilt process is judged against the fatbin binary digest."""
    assessment = _assessment(
        _entry_with_deliveries(),
        delivery="prebuilt",
        prebuilt_available=True,
        binary_digest=PREBUILT_DIGEST,
    )
    assert assessment.eligible, assessment.blocking


def test_prebuilt_delivery_with_wrong_binary_refuses() -> None:
    """A drifted fatbin closes the accelerated path, stated not silent."""
    assessment = _assessment(
        _entry_with_deliveries(),
        delivery="prebuilt",
        prebuilt_available=True,
        binary_digest="e" * 64,
    )
    assert not assessment.eligible
    assert assessment.blocking is not None
    assert assessment.blocking.code is ReasonCode.R_KERNEL_DIGEST_MISMATCH


def test_prebuilt_delivery_with_wrong_native_host_source_refuses() -> None:
    """A stable fatbin cannot certify a drifted Rust request host."""
    assessment = _assessment(
        _entry_with_deliveries(),
        delivery="prebuilt",
        prebuilt_available=True,
        binary_digest=PREBUILT_DIGEST,
        host_source_digest="e" * 64,
    )
    assert not assessment.eligible
    assert assessment.blocking is not None
    assert assessment.blocking.code is ReasonCode.R_KERNEL_DIGEST_MISMATCH
    assert assessment.blocking.detail["digest"] == "host_source"


def test_prebuilt_delivery_without_native_host_binding_refuses() -> None:
    """A refined prebuilt row must name every native-host binding axis."""
    entry = _entry_with_deliveries()
    prebuilt = entry["deliveries"]["prebuilt"]
    del prebuilt["host_source_digest"]
    del prebuilt["host_build_flags"]
    del prebuilt["host_toolchain"]
    assessment = _assessment(
        entry,
        delivery="prebuilt",
        prebuilt_available=True,
        binary_digest=PREBUILT_DIGEST,
    )
    assert not assessment.eligible
    assert assessment.blocking is not None
    assert assessment.blocking.detail["cause"] == "native_host_binding_missing"


def test_prebuilt_delivery_with_wrong_native_host_build_refuses() -> None:
    """Host flags and toolchain are part of the prebuilt certificate."""
    assessment = _assessment(
        _entry_with_deliveries(),
        delivery="prebuilt",
        prebuilt_available=True,
        binary_digest=PREBUILT_DIGEST,
        host_build_flags=("profile=debug",),
    )
    assert not assessment.eligible
    assert assessment.blocking is not None
    assert assessment.blocking.code is ReasonCode.R_UNCERTIFIED_ARTIFACT
    assert assessment.blocking.detail["cause"] == "host_build_flags_mismatch"

    assessment = _assessment(
        _entry_with_deliveries(),
        delivery="prebuilt",
        prebuilt_available=True,
        binary_digest=PREBUILT_DIGEST,
        host_toolchain="rustc drifted",
    )
    assert not assessment.eligible
    assert assessment.blocking is not None
    assert assessment.blocking.detail["cause"] == "host_toolchain_mismatch"


def test_prebuilt_delivery_on_unlisted_architecture_refuses() -> None:
    """sm not in the prebuilt device list -> R_SM_UNCERTIFIED."""
    assessment = _assessment(
        _entry_with_deliveries(devices=["sm_89"]),  # snapshot is sm_120
        delivery="prebuilt",
        prebuilt_available=True,
        binary_digest=PREBUILT_DIGEST,
    )
    assert not assessment.eligible
    assert assessment.blocking is not None
    assert assessment.blocking.code is ReasonCode.R_SM_UNCERTIFIED


def test_jit_delivery_still_verifies_source_bindings() -> None:
    """A JIT process keeps the certified_source verification path."""
    assessment = _assessment(
        _entry_with_deliveries(),
        delivery="jit",
        prebuilt_available=True,
    )
    assert assessment.eligible, assessment.blocking


def test_preload_prefers_prebuilt_when_shipped() -> None:
    """Before any load, a shipped fatbin selects the prebuilt view."""
    assessment = _assessment(
        _entry_with_deliveries(),
        delivery=None,
        prebuilt_available=True,
        binary_digest=PREBUILT_DIGEST,
    )
    assert assessment.eligible, assessment.blocking


def test_preload_without_fatbin_verifies_jit_view() -> None:
    """No fatbin shipped -> the JIT sub-entry is what gets verified."""
    assessment = _assessment(
        _entry_with_deliveries(),
        delivery=None,
        prebuilt_available=False,
        binary_digest=None,
    )
    assert assessment.eligible, assessment.blocking


def test_record_without_deliveries_plans_as_before() -> None:
    """JIT-era records verify as themselves under either cache state."""
    cases: tuple[dict[str, Any], ...] = (
        {},
        {"delivery": "jit", "prebuilt_available": True},
    )
    for overrides in cases:
        assessment = _assessment(
            support.certified_source_entry(), **overrides
        )
        assert assessment.eligible, assessment.blocking


def test_prebuilt_delivery_experimental_status_refuses() -> None:
    """A non-eligible delivery status refuses under CERTIFIED."""
    assessment = _assessment(
        _entry_with_deliveries(status="experimental"),
        delivery="prebuilt",
        prebuilt_available=True,
        binary_digest=PREBUILT_DIGEST,
    )
    assert not assessment.eligible
    assert assessment.blocking is not None
    assert assessment.blocking.code is ReasonCode.R_UNCERTIFIED_ARTIFACT
    assert (
        assessment.blocking.detail.get("cause")
        == "delivery_status_not_eligible"
    )


def test_prebuilt_driver_floor_enforced() -> None:
    """A driver below the prebuilt floor refuses with the floor named."""
    assessment = _assessment(
        _entry_with_deliveries(),
        driver_version="570.1",
        delivery="prebuilt",
        prebuilt_available=True,
        binary_digest=PREBUILT_DIGEST,
    )
    assert not assessment.eligible
    assert assessment.blocking is not None
    assert assessment.blocking.code is ReasonCode.R_DRIVER_TOO_OLD
    assert assessment.blocking.detail["required"] == DRIVER_MIN


def test_explain_reports_per_architecture_delivery_status() -> None:
    """explain() labels judged and unjudged architectures distinctly."""
    from toktier.routing.explain import build_explanation
    from toktier.routing.plan import plan

    view = support.registry(
        backends={BACKEND_GPU: _entry_with_deliveries()}
    )
    snapshot = support.snapshot(
        registry_view=view,
        driver_version=DRIVER_OK,
        kernel_cache=support.gpu_ready_kernel_cache(
            delivery="prebuilt",
            prebuilt_available=True,
            binary_digest=PREBUILT_DIGEST,
        ),
    )
    route = plan(
        snapshot, RoutingPolicy.CERTIFIED, view, support.config()
    )
    explanation = build_explanation(route_plan=route, snapshot=snapshot)
    assert explanation["kernel_delivery"] == "prebuilt"
    assert explanation["prebuilt_available"] is True
    certification = explanation["certification"]
    assert isinstance(certification, dict)
    deliveries = certification["deliveries"][BACKEND_GPU]
    architectures = deliveries["prebuilt"]["architectures"]
    assert architectures["sm_120"] == "certified"
    assert architectures["sm_75"] == "experimental"
    assert deliveries["jit"]["status"] == "certified_source"


def test_certification_headline_follows_the_loaded_delivery() -> None:
    """The headline labels the delivery that runs, not the one beside it.

    The record's backend-level GPU row is ``certified_source`` (the JIT
    view). With the judged prebuilt image loaded, a headline repeating
    that row would label a judged binary with a source certificate.
    """
    from toktier.routing.explain import build_explanation
    from toktier.routing.plan import plan

    view = support.registry(backends={BACKEND_GPU: _entry_with_deliveries()})
    snapshot = support.snapshot(
        registry_view=view,
        driver_version=DRIVER_OK,
        kernel_cache=support.gpu_ready_kernel_cache(
            delivery="prebuilt",
            prebuilt_available=True,
            binary_digest=PREBUILT_DIGEST,
        ),
    )
    route = plan(snapshot, RoutingPolicy.CERTIFIED, view, support.config())
    assert route.backend == BACKEND_GPU

    prebuilt = build_explanation(
        route_plan=route, snapshot=snapshot, gpu_delivery="prebuilt"
    )
    certification = prebuilt["certification"]
    assert isinstance(certification, dict)
    assert certification["backend_status"][BACKEND_GPU] == "certified"
    assert certification["state"] == "certified"
    assert certification["gpu_delivery"] == "prebuilt"
    # The nested per-delivery detail is unchanged by the headline.
    assert certification["deliveries"][BACKEND_GPU]["jit"]["status"] == (
        "certified_source"
    )

    jit = build_explanation(
        route_plan=route, snapshot=snapshot, gpu_delivery="jit"
    )
    jit_certification = jit["certification"]
    assert isinstance(jit_certification, dict)
    assert jit_certification["backend_status"][BACKEND_GPU] == "certified_source"
    assert jit_certification["state"] == "certified_source"
    assert jit_certification["gpu_delivery"] == "jit"

    # With no delivery named, the backend-level row is reported as before
    # and ``gpu_delivery`` says the label belongs to no delivery.
    unknown = build_explanation(route_plan=route, snapshot=snapshot)
    unknown_certification = unknown["certification"]
    assert isinstance(unknown_certification, dict)
    assert unknown_certification["backend_status"][BACKEND_GPU] == (
        "certified_source"
    )
    assert unknown_certification["gpu_delivery"] is None


@pytest.mark.parametrize("delivery", ["prebuilt", "jit"])
def test_delivery_view_falls_back_to_entry_for_unknown(
    delivery: str,
) -> None:
    """A delivery the record does not refine verifies the record itself."""
    entry = support.certified_source_entry(
        deliveries={"jit": support.certified_source_entry()}
    )
    assessment = _assessment(
        entry, delivery=delivery, prebuilt_available=True
    )
    assert assessment.eligible, assessment.blocking
