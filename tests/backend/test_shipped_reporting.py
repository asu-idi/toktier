"""Shipped facts are reported the same everywhere, truthfully.

Contract references: ``docs/contracts/registry.md`` Section 2 (the
explicit-engine oracle rule) and ``docs/contracts/facade.md`` Section 5
("not adopted" and "not available" are separate statements). Three
surfaces answer the prebuilt question -- ``toktier doctor``, the
routing probe behind ``explain()``, and the explicit engine's reports
-- and these tests pin them to one shared answer so they cannot state
different prebuilt facts for one installation.
"""

from __future__ import annotations

import json
from pathlib import Path

import _support as support
import pytest

from toktier.engine.gpu.certify import family_certification, oracle_binding
from toktier.errors import RegistryInvalid
from toktier.kernels.bindings import bare_sha256
from toktier.kernels.prebuilt import shipped_prebuilt_facts
from toktier.policy import BACKEND_GPU, RoutingPolicy
from toktier.routing.explain import build_explanation
from toktier.routing.plan import plan
from toktier.routing.probe import NoDevices
from toktier.routing.registry_load import (
    load_registry,
    load_registry_document,
    shipped_registry,
)
from toktier.routing.registry_view import ArtifactRecord, RegistryView
from toktier.routing.tables import SUPPORT_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_REGISTRY = ROOT / "tables" / "support_registry.json"


# ---------------------------------------------------------------------
# shipped registry: packaged copy and runtime loader
# ---------------------------------------------------------------------


def test_packaged_registry_copy_is_byte_identical() -> None:
    """The installed copy must not drift from the repository copy."""
    assert SUPPORT_REGISTRY.read_bytes() == REPOSITORY_REGISTRY.read_bytes()


def test_shipped_registry_loads_and_resolves_records() -> None:
    """Root digest verifies, records resolve, oracle sets are stated."""
    view = shipped_registry()
    document = json.loads(SUPPORT_REGISTRY.read_text(encoding="utf-8"))
    first = document["artifacts"][0]
    match = view.certification(artifact_sha256=first["artifact_sha256"])
    assert match is not None
    assert match.record.family == first["family"]
    oracle = view.oracle(match.record.oracle_id)
    assert oracle is not None
    assert oracle.certified_versions


def test_tampered_registry_is_refused(tmp_path: Path) -> None:
    """A changed byte fails the root digest; nothing loads unverified."""
    document = json.loads(SUPPORT_REGISTRY.read_text(encoding="utf-8"))
    document["artifacts"][0]["family"] = "renamed_family"
    tampered = tmp_path / "support_registry.v1.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryInvalid):
        load_registry(tampered)


def test_registry_without_root_digest_is_refused(tmp_path: Path) -> None:
    document = json.loads(SUPPORT_REGISTRY.read_text(encoding="utf-8"))
    del document["root_digest"]
    stripped = tmp_path / "support_registry.v1.json"
    stripped.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryInvalid):
        load_registry_document(stripped)


# ---------------------------------------------------------------------
# probe: shipped facts on the no-probe path
# ---------------------------------------------------------------------


def test_no_probe_path_reports_shipped_facts_truthfully() -> None:
    """The no-probe kernel cache answers with doctor's own answer.

    Device facts stay absent (nothing was enumerated), but the shipped
    prebuilt fact is a read-only property of the installation and must
    match the one shared helper, digest included.
    """
    available, digest = shipped_prebuilt_facts()
    cache = NoDevices().kernel_cache()
    assert cache.prebuilt_available is available
    assert cache.binary_digest == (bare_sha256(digest) if digest else None)
    # This checkout ships the fatbin and the JIT sources.
    assert cache.prebuilt_available is True
    assert cache.source_digest is not None
    assert cache.delivery is None
    assert cache.built is False


def test_not_adopted_and_not_available_stay_separate() -> None:
    """A shipped prebuilt with no probe supplied keeps both statements.

    The plan reason says the path adopted no accelerator; the delivery
    block says the prebuilt is shipped. Neither may swallow the other.
    """
    view = support.registry()
    snapshot = support.snapshot(
        registry_view=view,
        devices=(),
        devices_probed=False,
        kernel_cache=support.gpu_ready_kernel_cache(
            built=False,
            loaded_flag_sets=0,
            delivery=None,
            prebuilt_available=True,
        ),
    )
    route = plan(snapshot, RoutingPolicy.CERTIFIED, view, support.config())
    explanation = build_explanation(route_plan=route, snapshot=snapshot)
    reasons = explanation["plan_reasons"]
    assert isinstance(reasons, list)
    codes = {reason["code"] for reason in reasons}
    assert "R_ACCELERATOR_NOT_ADOPTED" in codes
    assert explanation["prebuilt_available"] is True
    deliveries = explanation["kernel_deliveries"]
    assert isinstance(deliveries, dict)
    assert deliveries["prebuilt"]["shipped"] is True
    assert deliveries["prebuilt"]["loaded"] is False
    assert deliveries["jit"]["loaded"] is False


# ---------------------------------------------------------------------
# explain: the delivery-dimension status map
# ---------------------------------------------------------------------


def _delivery_entry(**prebuilt_overrides: object) -> dict[str, object]:
    prebuilt = support.certified_entry(
        binary_digest="d" * 64,
        devices=["sm_89", "sm_120"],
        devices_experimental=["sm_75"],
        driver_min="580.65.06",
    )
    prebuilt.update(prebuilt_overrides)
    entry: dict[str, object] = support.certified_source_entry(
        deliveries={
            "jit": support.certified_source_entry(),
            "prebuilt": prebuilt,
        }
    )
    return entry


def test_kernel_deliveries_reports_statuses_from_the_record() -> None:
    view = support.registry(backends={BACKEND_GPU: _delivery_entry()})
    snapshot = support.snapshot(
        registry_view=view,
        driver_version="580.82.07",
        kernel_cache=support.gpu_ready_kernel_cache(
            delivery="prebuilt",
            prebuilt_available=True,
            binary_digest="d" * 64,
        ),
    )
    route = plan(snapshot, RoutingPolicy.CERTIFIED, view, support.config())
    explanation = build_explanation(route_plan=route, snapshot=snapshot)
    deliveries = explanation["kernel_deliveries"]
    assert isinstance(deliveries, dict)
    prebuilt = deliveries["prebuilt"]
    assert prebuilt["shipped"] is True
    assert prebuilt["loaded"] is True
    assert prebuilt["status"] == "certified"
    assert prebuilt["architectures"]["sm_120"] == "certified"
    assert prebuilt["architectures"]["sm_75"] == "experimental"
    assert prebuilt["driver_min"] == "580.65.06"
    jit = deliveries["jit"]
    assert jit["loaded"] is False
    assert jit["status"] == "certified_source"


def test_kernel_deliveries_without_a_record_claims_no_status() -> None:
    """No record consulted -> shipped facts only, no status claims."""
    unrelated = support.registry(
        backends={BACKEND_GPU: _delivery_entry()},
        artifact_sha256="e" * 64,
    )
    snapshot = support.snapshot(
        registry_view=unrelated,
        artifact_sha256="f" * 64,
        pipeline_fingerprint=None,
        added_fingerprint=None,
        kernel_cache=support.gpu_ready_kernel_cache(
            delivery=None,
            prebuilt_available=True,
        ),
    )
    route = plan(
        snapshot,
        RoutingPolicy.CERTIFIED,
        unrelated,
        support.config(),
    )
    explanation = build_explanation(route_plan=route, snapshot=snapshot)
    deliveries = explanation["kernel_deliveries"]
    assert isinstance(deliveries, dict)
    assert deliveries["prebuilt"]["shipped"] is True
    assert deliveries["prebuilt"]["status"] is None
    assert deliveries["prebuilt"]["architectures"] == {}
    assert deliveries["jit"]["status"] is None


def test_kernel_deliveries_delivery_record_override() -> None:
    """A caller may supply the record read-only (the 0.x facade path)."""
    view = support.registry(backends={BACKEND_GPU: _delivery_entry()})
    record_match = view.certification(artifact_sha256=support.ARTIFACT_SHA)
    assert record_match is not None
    from toktier.routing.registry_view import empty_registry

    empty = empty_registry()
    snapshot = support.snapshot(
        registry_view=empty,
        pipeline_fingerprint=None,
        added_fingerprint=None,
        kernel_cache=support.gpu_ready_kernel_cache(
            delivery=None,
            prebuilt_available=True,
        ),
    )
    route = plan(snapshot, RoutingPolicy.CERTIFIED, empty, support.config())
    explanation = build_explanation(
        route_plan=route,
        snapshot=snapshot,
        delivery_record=record_match.record,
    )
    deliveries = explanation["kernel_deliveries"]
    assert isinstance(deliveries, dict)
    prebuilt = deliveries["prebuilt"]
    assert prebuilt["status"] == "certified"
    assert prebuilt["architectures"]["sm_89"] == "certified"
    # The certification block still answers for the active request only.
    certification = explanation["certification"]
    assert isinstance(certification, dict)
    assert certification["state"] == "uncertified"


# ---------------------------------------------------------------------
# explicit engine: oracle binding and per-family verdicts
# ---------------------------------------------------------------------


def _record_and_registry() -> tuple[RegistryView, ArtifactRecord]:
    view = support.registry(backends={BACKEND_GPU: _delivery_entry()})
    match = view.certification(artifact_sha256=support.ARTIFACT_SHA)
    assert match is not None
    return view, match.record


def test_oracle_binding_inside_certified_set() -> None:
    view, record = _record_and_registry()
    report = oracle_binding(
        view, {"test_family": record}, installed=support.ORACLE_VERSION
    )
    assert report["in_certified_set"] is True
    assert support.ORACLE_VERSION in report["certified_versions"]


def test_oracle_binding_outside_certified_set() -> None:
    view, record = _record_and_registry()
    report = oracle_binding(
        view, {"test_family": record}, installed="0.23.1"
    )
    assert report["installed"] == "0.23.1"
    assert report["in_certified_set"] is False


def test_oracle_binding_without_records_fails_closed() -> None:
    view, _record = _record_and_registry()
    report = oracle_binding(
        view, {"test_family": None}, installed=support.ORACLE_VERSION
    )
    assert report["certified_versions"] == []
    assert report["in_certified_set"] is False


def test_family_certification_attaches_only_when_everything_matches() -> None:
    view, record = _record_and_registry()
    verdict = family_certification(
        registry=view,
        record=record,
        delivery="prebuilt",
        architecture="sm_120",
        certificate_void=False,
        installed_oracle=support.ORACLE_VERSION,
    )
    assert verdict["state"] == "certified"
    assert verdict["reasons"] == []


def test_family_certification_labels_out_of_set_oracle() -> None:
    view, record = _record_and_registry()
    verdict = family_certification(
        registry=view,
        record=record,
        delivery="prebuilt",
        architecture="sm_120",
        certificate_void=False,
        installed_oracle="0.23.1",
    )
    assert verdict["state"] == "uncertified"
    assert "oracle_outside_certified_set" in verdict["reasons"]
    # The registry status is still reported, distinctly from the verdict.
    assert verdict["status"] == "certified"


def test_family_certification_experimental_architecture() -> None:
    view, record = _record_and_registry()
    verdict = family_certification(
        registry=view,
        record=record,
        delivery="prebuilt",
        architecture="sm_75",
        certificate_void=False,
        installed_oracle=support.ORACLE_VERSION,
    )
    assert verdict["state"] == "experimental"
    assert verdict["reasons"] == ["architecture_experimental"]


def test_family_certification_without_record() -> None:
    view, _record = _record_and_registry()
    verdict = family_certification(
        registry=view,
        record=None,
        delivery="prebuilt",
        architecture="sm_120",
        certificate_void=False,
        installed_oracle=support.ORACLE_VERSION,
    )
    assert verdict["state"] == "uncertified"
    assert verdict["reasons"] == ["no_certification_record"]
