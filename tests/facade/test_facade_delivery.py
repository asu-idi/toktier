"""``gpu_delivery="auto"`` prefers prebuilt whenever it can serve.

The planner is replaced by a stand-in that opens the GPU route for a
chosen set of deliveries and architectures, so each case states exactly
which plan the facade was offered and checks which one it kept. No case
here loads a kernel or needs a device.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from toktier.facade import api as facade_api
from toktier.facade import delivery
from toktier.policy import (
    BACKEND_GPU,
    BACKEND_REFERENCE,
    PlanReason,
    ReasonCode,
    RoutePlan,
    RoutingPolicy,
)
from toktier.routing.probe import DeviceInfo, KernelCacheState, ProbeSnapshot
from toktier.routing.registry_load import load_registry_document
from toktier.routing.tables import SUPPORT_REGISTRY

from .conftest import Rig

_DEVICES = (
    DeviceInfo(index=0, name="card zero", architecture="sm_75"),
    DeviceInfo(index=1, name="card one", architecture="sm_120"),
)


class _FakeHostProbe:
    """Stands in for ``CudaHostProbe``: two cards, no kernel loaded."""

    instances: ClassVar[list[_FakeHostProbe]] = []

    def __init__(self, *, config: Any = None, delivery: str = "auto") -> None:
        self.delivery = delivery
        _FakeHostProbe.instances.append(self)

    def devices(self) -> tuple[DeviceInfo, ...]:
        return _DEVICES

    def driver_version(self) -> str | None:
        return "999.0"

    def kernel_cache(self) -> KernelCacheState:
        return KernelCacheState(
            source_digest=None,
            class_table_digest=None,
            preferred_delivery=self.delivery,
        )


def _plan(holds_gpu: bool) -> RoutePlan:
    if holds_gpu:
        return RoutePlan(
            policy=RoutingPolicy.SUPPORTED,
            backend=BACKEND_GPU,
            fallback_chain=(BACKEND_GPU, BACKEND_REFERENCE),
            reasons=(),
        )
    return RoutePlan(
        policy=RoutingPolicy.SUPPORTED,
        backend=BACKEND_REFERENCE,
        fallback_chain=(BACKEND_REFERENCE,),
        reasons=(
            PlanReason(
                ReasonCode.R_UNCERTIFIED_ARTIFACT, BACKEND_GPU, {"cause": "stand-in"}
            ),
        ),
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    serving: set[str],
    architectures: set[str] | None = None,
    jit_installed: bool,
) -> list[str]:
    """Open the GPU route for ``serving`` deliveries (and architectures)."""
    _FakeHostProbe.instances = []
    offered: list[str] = []
    monkeypatch.setattr(facade_api, "CudaHostProbe", _FakeHostProbe)
    monkeypatch.setattr(facade_api, "profile_jit_installed", lambda: jit_installed)

    def build_plan(snapshot: ProbeSnapshot, *_args: object) -> RoutePlan:
        chosen = snapshot.kernel_cache.preferred_delivery or "auto"
        offered.append(chosen)
        archs = {device.architecture for device in snapshot.devices}
        admitted = architectures is None or bool(archs & architectures)
        return _plan(chosen in serving and admitted and bool(snapshot.devices))

    monkeypatch.setattr(facade_api, "build_plan", build_plan)
    return offered


@pytest.mark.parametrize(
    ("serving", "jit_installed", "expected", "offered"),
    [
        ({"prebuilt", "jit"}, True, "prebuilt", ["prebuilt"]),
        ({"prebuilt"}, True, "prebuilt", ["prebuilt"]),
        ({"prebuilt"}, False, "prebuilt", ["prebuilt"]),
        ({"jit"}, True, "jit", ["prebuilt", "jit"]),
        ({"jit"}, False, "prebuilt", ["prebuilt"]),
        (set(), True, "prebuilt", ["prebuilt", "jit"]),
        (set(), False, "prebuilt", ["prebuilt"]),
    ],
)
def test_auto_takes_prebuilt_whenever_it_can_serve(
    rig: Rig,
    monkeypatch: pytest.MonkeyPatch,
    serving: set[str],
    jit_installed: bool,
    expected: str,
    offered: list[str],
) -> None:
    seen = _install(monkeypatch, serving=serving, jit_installed=jit_installed)
    tokenizer = rig.tokenizer()
    try:
        assert seen == offered
        assert tokenizer._gpu_delivery == expected
        assert (BACKEND_GPU in tokenizer.plan.fallback_chain) == bool(
            serving and (expected in serving)
        )
        runtime = tokenizer.explain()["runtime_policy"]
        assert isinstance(runtime, dict)
        assert runtime["gpu_delivery_request"] == "auto"
        assert runtime["gpu_delivery_selected"] == expected
    finally:
        tokenizer.close()


@pytest.mark.parametrize("requested", ["prebuilt", "jit"])
def test_an_explicit_delivery_is_planned_alone(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    seen = _install(monkeypatch, serving={"prebuilt", "jit"}, jit_installed=True)
    tokenizer = rig.tokenizer(gpu_delivery=requested)
    try:
        assert seen == [requested]
        assert tokenizer._gpu_delivery == requested
    finally:
        tokenizer.close()


def test_cpu_device_probes_nothing_and_reads_prebuilt(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, serving={"jit"}, jit_installed=True)
    tokenizer = rig.tokenizer(device="cpu")
    try:
        assert _FakeHostProbe.instances == []
        assert tokenizer._gpu_delivery == "prebuilt"
    finally:
        tokenizer.close()


def _host_default(
    *,
    prebuilt_available: bool = True,
    native_host_ready: bool = True,
    status: str = "certified",
    jit_installed: bool,
) -> str:
    return delivery.host_default_delivery(
        prebuilt_available=prebuilt_available,
        native_host_ready=native_host_ready,
        observed_architectures=["sm_120"],
        prebuilt_statuses={"sm_120": status},
        admitted_statuses={"certified", "certified_source"},
        jit_installed=jit_installed,
    )


def test_host_default_follows_the_same_preference() -> None:
    assert _host_default(jit_installed=True) == "prebuilt"
    assert _host_default(prebuilt_available=False, jit_installed=True) == "jit"
    assert _host_default(native_host_ready=False, jit_installed=True) == "jit"
    assert _host_default(status="uncertified", jit_installed=True) == "jit"
    assert _host_default(prebuilt_available=False, jit_installed=False) == (
        "prebuilt"
    )


def test_shipped_prebuilt_judged_devices_cover_the_jit_judged_devices() -> None:
    """The property that keeps ``auto`` from trading a judged JIT pair for a
    weaker prebuilt label on the device axis."""
    document = load_registry_document(SUPPORT_REGISTRY)
    assert delivery.prebuilt_covers_jit_devices(document) == []
    synthetic = {
        "artifacts": [
            {
                "family": "f",
                "backends": {
                    "gpu": {
                        "deliveries": {
                            "jit": {"devices": ["sm_89", "sm_61"]},
                            "prebuilt": {"devices": ["sm_89"]},
                        }
                    }
                },
            }
        ]
    }
    assert delivery.prebuilt_covers_jit_devices(synthetic) == ["f: ['sm_61']"]
