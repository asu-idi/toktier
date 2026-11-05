"""Examined artifacts: runnable on the fast CPU engine, never certified.

Contract reference: ``docs/support-matrix.md`` (Status vocabulary, the
examined DeepSeek-V4.1-Flash section) and ``routing/plan.py`` check 5a.
The tests read the shipped tables, so they also pin what this release
says about the one examined artifact it carries.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from toktier.backends.fast_cpu import FastCpuEngineFacts
from toktier.config import Config
from toktier.policy import (
    BACKEND_FAST_CPU,
    BACKEND_REFERENCE,
    ReasonCode,
    RoutingPolicy,
)
from toktier.repair.registry import examined_families, families, family_spec
from toktier.routing.explain import build_explanation
from toktier.routing.plan import assessments_for, plan
from toktier.routing.probe import ProbeSnapshot
from toktier.routing.registry_load import shipped_registry

FAMILY = "deepseek_v4_1_flash"
ARTIFACT = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"


def _matching_engine_facts() -> FastCpuEngineFacts:
    """Engine facts that verify against the peer's fast CPU binding."""
    match = shipped_registry().certification(artifact_sha256=ARTIFACT)
    assert match is not None
    entry = match.record.backends[BACKEND_FAST_CPU]
    return FastCpuEngineFacts(
        version=entry.engine_version,
        source_digest=entry.source_digest,
        build_flags=tuple(entry.build_flags),
        toolchain=entry.toolchain,
        config_digest=entry.config_digest,
    )


def _snapshot(**overrides: object) -> ProbeSnapshot:
    registry = shipped_registry()
    values: dict[str, object] = {
        "family": FAMILY,
        "artifact_sha256": ARTIFACT,
        "oracle_version": "0.22.2",
        "loader_face_sha256": ARTIFACT,
        "importable_backends": frozenset({BACKEND_REFERENCE, BACKEND_FAST_CPU}),
        "fast_cpu_engine": _matching_engine_facts(),
        "certification": registry.certification(artifact_sha256=ARTIFACT),
    }
    values.update(overrides)
    return ProbeSnapshot(**values)  # type: ignore[arg-type]


def test_the_examined_row_is_not_a_certification_record() -> None:
    registry = shipped_registry()
    examined = registry.examined(ARTIFACT)
    assert examined is not None
    assert examined.family == FAMILY
    assert examined.missing_evidence == (
        "corpus_absence_certificate",
        "cross_artifact_carryover_record",
    )
    match = registry.certification(artifact_sha256=ARTIFACT)
    assert match is not None
    assert match.identity == "examined"
    assert set(match.record.backends) == {BACKEND_FAST_CPU}
    assert {"deepseek_v3", "deepseek_v4_flash"} <= set(match.pipeline_peers)
    certified = [
        registry.certification(artifact_sha256=item.artifact_sha256)
        for item in families()
    ]
    assert all(item is not None and item.identity == "exact" for item in certified)
    assert FAMILY not in {item.record.family for item in certified if item}


def test_the_certified_repair_roster_is_unchanged() -> None:
    assert len(families()) == 11
    assert all(item.family != FAMILY for item in families())
    spec = family_spec(FAMILY, ARTIFACT)
    assert spec is not None
    assert spec in examined_families()
    assert (spec.margin, spec.effective_l_max, spec.has_normalizer) == (
        36,
        128,
        True,
    )
    assert family_spec(FAMILY, "0" * 64) is None


@pytest.mark.parametrize(
    "policy", [RoutingPolicy.SUPPORTED, RoutingPolicy.REQUIRE_ACCELERATED]
)
def test_supported_runs_the_fast_cpu_engine_and_labels_it(
    policy: RoutingPolicy,
) -> None:
    snapshot = _snapshot()
    route = plan(snapshot, policy, shipped_registry(), Config())
    assert route.backend == BACKEND_FAST_CPU
    assessments = assessments_for(snapshot, policy, shipped_registry(), Config())
    fast = next(item for item in assessments if item.backend == BACKEND_FAST_CPU)
    assert fast.eligible
    causes = [reason.detail.get("cause") for reason in fast.supported]
    assert "examined_not_certified" in causes
    report = build_explanation(
        route_plan=route, snapshot=snapshot, assessments=assessments
    )
    certification = report["certification"]
    assert isinstance(certification, dict)
    assert certification["state"] == "supported_untested"
    assert certification["effective_verdict"] == "supported"
    assert certification["identity"] == "examined"
    assert certification["certified_family"] is None
    assert certification["evidence_id"] is None
    assert certification["backend_status"] == {BACKEND_FAST_CPU: "examined"}
    examined = certification["examined"]
    assert isinstance(examined, dict)
    assert examined["missing_evidence"] == [
        "corpus_absence_certificate",
        "cross_artifact_carryover_record",
    ]


def test_certified_refuses_with_the_uncertified_artifact_reason() -> None:
    route = plan(_snapshot(), RoutingPolicy.CERTIFIED, shipped_registry(), Config())
    assert route.backend == BACKEND_REFERENCE
    refusal = next(
        reason for reason in route.reasons if reason.backend == BACKEND_FAST_CPU
    )
    assert refusal.code is ReasonCode.R_UNCERTIFIED_ARTIFACT
    assert refusal.detail["cause"] == "examined_not_certified"


def test_reference_policy_considers_no_accelerated_option() -> None:
    route = plan(_snapshot(), RoutingPolicy.REFERENCE, shipped_registry(), Config())
    assert route.backend == BACKEND_REFERENCE


def test_every_binding_still_verifies() -> None:
    """Coverage admission waives nothing that has to verify."""
    registry = shipped_registry()
    other_face = plan(
        _snapshot(loader_face_sha256="e" * 64),
        RoutingPolicy.SUPPORTED,
        registry,
        Config(),
    )
    assert other_face.backend == BACKEND_REFERENCE
    assert ReasonCode.R_LOADER_FACE_MISMATCH in {
        reason.code for reason in other_face.reasons
    }
    other_engine = plan(
        _snapshot(
            fast_cpu_engine=replace(_matching_engine_facts(), source_digest="1" * 64)
        ),
        RoutingPolicy.SUPPORTED,
        registry,
        Config(),
    )
    assert other_engine.backend == BACKEND_REFERENCE
    assert ReasonCode.R_ENGINE_BINDING_MISMATCH in {
        reason.code for reason in other_engine.reasons
    }
    other_oracle = plan(
        _snapshot(oracle_version="0.21.0"),
        RoutingPolicy.SUPPORTED,
        registry,
        Config(),
    )
    assert other_oracle.backend == BACKEND_REFERENCE
