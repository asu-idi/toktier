"""Examined artifacts: runnable on the fast CPU engine, never certified.

Contract reference: ``docs/contracts/registry.md`` Section 3.0 and
``routing/plan.py`` check 5a. The shipped examined-artifact table carries
no row in this release; the mechanism is exercised through a fixture table
read by the same digest-verified loader the shipped table goes through.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from toktier.backends.fast_cpu import FastCpuEngineFacts
from toktier.config import Config
from toktier.policy import (
    BACKEND_FAST_CPU,
    BACKEND_REFERENCE,
    ReasonCode,
    RoutingPolicy,
)
from toktier.repair import registry as repair_registry
from toktier.repair.registry import examined_families, families, family_spec
from toktier.routing.explain import build_explanation
from toktier.routing.plan import assessments_for, plan
from toktier.routing.probe import ProbeSnapshot
from toktier.routing.registry_load import (
    load_examined_artifacts,
    load_registry_document,
    shipped_registry,
)
from toktier.routing.registry_view import RegistryView
from toktier.routing.tables import EXAMINED_ARTIFACTS, SUPPORT_REGISTRY

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "examined_artifacts.fixture.json"
)
FAMILY = "examined_fixture"
ARTIFACT = "e" * 64


def _fixture_registry() -> RegistryView:
    """The shipped registry with the fixture examined table beside it."""
    return RegistryView.from_document(
        load_registry_document(SUPPORT_REGISTRY),
        path=str(SUPPORT_REGISTRY),
        examined=load_examined_artifacts(FIXTURE),
    )


@pytest.fixture
def fixture_table(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the repair-parameter lookup at the fixture table."""
    import toktier.routing.tables as tables

    monkeypatch.setattr(tables, "EXAMINED_ARTIFACTS", FIXTURE)
    examined_families.cache_clear()
    repair_registry._sources.cache_clear()
    yield
    examined_families.cache_clear()
    repair_registry._sources.cache_clear()


def test_the_shipped_examined_table_is_empty() -> None:
    assert load_examined_artifacts(EXAMINED_ARTIFACTS) == ()
    assert examined_families() == ()
    registry = shipped_registry()
    assert registry.examined(ARTIFACT) is None
    # The artifact that was runnable through the table is certified now.
    match = registry.certification(
        artifact_sha256=(
            "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
        )
    )
    assert match is not None and match.identity == "exact"
    assert match.record.family == "deepseek_v4_1_flash"


def _matching_engine_facts() -> FastCpuEngineFacts:
    """Engine facts that verify against the peer's fast CPU binding."""
    match = _fixture_registry().certification(artifact_sha256=ARTIFACT)
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
    registry = _fixture_registry()
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
    registry = _fixture_registry()
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
    assert {"deepseek_v3", "deepseek_v4_1_flash", "deepseek_v4_flash"} <= set(
        match.pipeline_peers
    )
    certified = [
        registry.certification(artifact_sha256=item.artifact_sha256)
        for item in families()
    ]
    assert all(item is not None and item.identity == "exact" for item in certified)
    assert FAMILY not in {item.record.family for item in certified if item}


@pytest.mark.usefixtures("fixture_table")
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
    route = plan(snapshot, policy, _fixture_registry(), Config())
    assert route.backend == BACKEND_FAST_CPU
    assessments = assessments_for(snapshot, policy, _fixture_registry(), Config())
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
    route = plan(_snapshot(), RoutingPolicy.CERTIFIED, _fixture_registry(), Config())
    assert route.backend == BACKEND_REFERENCE
    refusal = next(
        reason for reason in route.reasons if reason.backend == BACKEND_FAST_CPU
    )
    assert refusal.code is ReasonCode.R_UNCERTIFIED_ARTIFACT
    assert refusal.detail["cause"] == "examined_not_certified"


def test_reference_policy_considers_no_accelerated_option() -> None:
    route = plan(_snapshot(), RoutingPolicy.REFERENCE, _fixture_registry(), Config())
    assert route.backend == BACKEND_REFERENCE


def test_every_binding_still_verifies() -> None:
    """Coverage admission waives nothing that has to verify."""
    registry = _fixture_registry()
    other_face = plan(
        _snapshot(loader_face_sha256="d" * 64),
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
