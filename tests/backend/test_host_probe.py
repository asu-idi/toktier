"""The concrete host probe feeds the real probe and planner.

Contract reference: ``docs/contracts/routing.md`` Section 2. These tests
run the production ``CudaHostProbe`` through ``probe()`` and ``plan()``
rather than through synthetic snapshots, so a drift between what the
probe reports and what the planner expects fails here.
"""

from __future__ import annotations

from types import SimpleNamespace

import _support as support
import pytest

from toktier.engine.gpu import host_probe
from toktier.engine.gpu.host_probe import CudaHostProbe
from toktier.policy import BACKEND_REFERENCE, ReasonCode, RoutingPolicy
from toktier.routing.plan import plan
from toktier.routing.probe import probe


def test_probe_reports_shipped_facts_without_a_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without torch, the probe still reports the host-computable facts."""
    monkeypatch.setattr(host_probe, "_torch_runtime", lambda: None)
    device_probe = CudaHostProbe(config=support.config())

    assert device_probe.devices() == ()
    assert device_probe.driver_version() is None
    cache = device_probe.kernel_cache()
    assert cache.built is False
    assert cache.loaded_flag_sets == 0
    assert cache.source_digest == support.ENGINE_BINDINGS.source_digest
    assert cache.class_table_digest == support.ENGINE_BINDINGS.class_table_digest


def test_explicit_delivery_is_visible_before_the_lazy_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planner judges the delivery the facade will eventually load."""
    fake_torch = SimpleNamespace(
        __version__="2.11.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(host_probe, "_torch_runtime", lambda: fake_torch)
    monkeypatch.setattr(host_probe, "find_spec", lambda name: object())

    cache = CudaHostProbe(config=support.config(), delivery="jit").kernel_cache()
    assert cache.delivery is None
    assert cache.preferred_delivery == "jit"
    assert cache.toolchain == "CUDA 12.8 / torch 2.11.0+cu128"
    assert cache.toolchain_satisfied is True

    prebuilt = CudaHostProbe(
        config=support.config(), delivery="prebuilt"
    ).kernel_cache()
    assert prebuilt.delivery is None
    assert prebuilt.preferred_delivery == "prebuilt"
    assert prebuilt.toolchain_satisfied is None


def test_unjudged_jit_pair_fails_the_toolchain_fact_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.12.0+cu129",
        version=SimpleNamespace(cuda="12.9"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(host_probe, "_torch_runtime", lambda: fake_torch)
    monkeypatch.setattr(host_probe, "find_spec", lambda name: object())
    cache = CudaHostProbe(config=support.config(), delivery="jit").kernel_cache()
    assert cache.toolchain_satisfied is False


def test_driver_uses_the_system_version_when_torch_has_no_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, driver_version=None)
    )
    monkeypatch.setattr(host_probe, "_torch_runtime", lambda: fake_torch)
    monkeypatch.setattr(host_probe, "_system_driver_version", lambda: "580.65.06")
    assert CudaHostProbe().driver_version() == "580.65.06"


def test_host_probe_flows_through_the_real_planner() -> None:
    """A snapshot from the production probe yields a coherent plan."""
    view = support.registry()
    snapshot = probe(
        family="test_family",
        registry=view,
        artifact_sha256=support.ARTIFACT_SHA,
        pipeline_fingerprint=support.PIPELINE_FINGERPRINT,
        added_frontend_fingerprint=support.ADDED_FINGERPRINT,
        device_probe=CudaHostProbe(config=support.config()),
        installed_oracle_version=support.ORACLE_VERSION,
    )
    assert snapshot.kernel_cache.source_digest == (
        support.ENGINE_BINDINGS.source_digest
    )

    route_plan = plan(snapshot, RoutingPolicy.CERTIFIED, view, support.config())
    assert route_plan.fallback_chain[0] == route_plan.backend
    assert route_plan.fallback_chain[-1] == BACKEND_REFERENCE
    if not snapshot.devices:
        # This environment has no usable accelerator (or no torch), so
        # the real planner must land on the reference backend and say
        # why.
        assert route_plan.backend == BACKEND_REFERENCE
        assert set(route_plan.reason_codes()) & {
            ReasonCode.R_BACKEND_UNAVAILABLE,
            ReasonCode.R_NO_GPU_DETECTED,
        }
