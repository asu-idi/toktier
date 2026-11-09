"""Cross-artifact carry-over: carried records and their release check.

Contract reference: ``docs/contracts/evidence-carryover.md`` Section 3.1
(conditions C1-C7) and ``docs/contracts/registry.md`` Section 3.0 (the
registry-held repair member). The positive tests pin the shipped state:
two records carried from an anchor, each holding the anchor's readings and
engine binding, its own capability and repair member, a certificate and an
engine-parity reading. The negative tests tamper with a copy of the
shipped files and require the release check (``carryover_problems``) to
refuse each change.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import generate_registry
import pytest
from generate_registry import carryover_problems

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "tables" / "support_registry.json"
ROSTER = ROOT / "src" / "toktier" / "repair" / "tables" / "fast_repair_families.v1.json"
CARRIED = {"deepseek_v4_1_flash": "deepseek_v4_flash", "mimo_v2_6": "qwen3_8b"}


def _document() -> dict[str, Any]:
    value = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _row(document: dict[str, Any], family: str) -> dict[str, Any]:
    return next(item for item in document["artifacts"] if item["family"] == family)


def _roster() -> dict[str, dict[str, Any]]:
    rows = json.loads(ROSTER.read_text(encoding="utf-8"))["families"]
    return {row["family"]: row for row in rows}


# -- positive: the shipped state -------------------------------------------


def test_carryover_check_passes_on_the_shipped_registry() -> None:
    assert carryover_problems(SHIPPED) == []


@pytest.mark.parametrize(("family", "anchor"), sorted(CARRIED.items()))
def test_cross_artifact_record_mirrors_its_anchor(family: str, anchor: str) -> None:
    document = _document()
    row, base = _row(document, family), _row(document, anchor)
    for key in ("readings", "evidence_id", "suite_version", "oracle_id", "pipeline_id"):
        assert row[key] == base[key], key
    assert row["backends"]["gpu"] == base["backends"]["gpu"]
    assert row["added_frontend_id"] != base["added_frontend_id"]
    assert row["loader_face_sha256"] != base["loader_face_sha256"]
    certificate = json.loads(
        (ROOT / row["carryover"]["certificate"]["path"]).read_text(encoding="utf-8")
    )
    assert certificate["subject"]["artifact_sha256"] == row["artifact_sha256"]
    assert certificate["subject"]["loader_face_sha256"] == row["loader_face_sha256"]
    assert certificate["subject"]["from_artifact_sha256"] == base["artifact_sha256"]
    assert "aliases" not in row and "config_added_tokens" not in row
    node = row["carryover"]
    assert (node["from_family"], node["from_artifact_sha256"]) == (
        anchor,
        base["artifact_sha256"],
    )
    assert "carryover" not in base
    assert {
        (item["pipeline_id"], item["added_frontend_id"], item["evidence_id"])
        for item in document["compositions"]
    } >= {(row["pipeline_id"], row["added_frontend_id"], row["evidence_id"])}


@pytest.mark.parametrize(("family", "anchor"), sorted(CARRIED.items()))
def test_carried_fast_cpu_entry_equals_anchor_plus_repair(
    family: str, anchor: str
) -> None:
    document = _document()
    entry = copy.deepcopy(_row(document, family)["backends"]["fast_cpu"])
    repair = entry.pop("repair")
    assert entry == _row(document, anchor)["backends"]["fast_cpu"]
    roster = _roster()[anchor]
    for key in ("margin", "effective_l_max", "has_normalizer"):
        assert repair[key] == roster[key], key
    digest = repair["source_table_sha256"]
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    assert family not in _roster()


def test_family_spec_resolves_carried_pairs_after_the_roster() -> None:
    from toktier.repair.registry import carried_families, families, family_spec

    assert len(families()) == 11
    document = _document()
    carried = {item.family for item in carried_families()}
    assert carried == set(CARRIED)
    for family in CARRIED:
        sha = _row(document, family)["artifact_sha256"]
        spec = family_spec(family, sha)
        assert spec is not None and spec.family == family
        assert family_spec(family, "0" * 64) is None


@pytest.mark.parametrize("family", sorted(CARRIED))
def test_carried_family_plans_fast_cpu_under_certified(family: str) -> None:
    from toktier.backends.fast_cpu import FastCpuEngineFacts
    from toktier.config import Config
    from toktier.policy import BACKEND_FAST_CPU, BACKEND_REFERENCE, RoutingPolicy
    from toktier.routing.explain import build_explanation
    from toktier.routing.plan import assessments_for, plan
    from toktier.routing.probe import ProbeSnapshot
    from toktier.routing.registry_load import shipped_registry

    registry = shipped_registry()
    row = _row(_document(), family)
    match = registry.certification(artifact_sha256=row["artifact_sha256"])
    assert match is not None and match.identity == "exact"
    entry = match.record.backends[BACKEND_FAST_CPU]
    snapshot = ProbeSnapshot(
        family=family,
        artifact_sha256=row["artifact_sha256"],
        oracle_version="0.22.2",
        loader_face_sha256=row["loader_face_sha256"],
        importable_backends=frozenset({BACKEND_REFERENCE, BACKEND_FAST_CPU}),
        fast_cpu_engine=FastCpuEngineFacts(
            version=entry.engine_version,
            source_digest=entry.source_digest,
            build_flags=tuple(entry.build_flags),
            toolchain=entry.toolchain,
            config_digest=entry.config_digest,
        ),
        certification=match,
    )
    route = plan(snapshot, RoutingPolicy.CERTIFIED, registry, Config())
    assert route.backend == BACKEND_FAST_CPU
    assessments = assessments_for(snapshot, RoutingPolicy.CERTIFIED, registry, Config())
    report = build_explanation(
        route_plan=route, snapshot=snapshot, assessments=assessments
    )
    certification = report["certification"]
    assert isinstance(certification, dict)
    assert certification["certified_family"] == family
    assert certification["identity"] == "exact"
    assert "R_UNCERTIFIED_ARTIFACT" not in json.dumps(report, default=str)


def test_a_name_in_two_repair_sources_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from toktier.errors import RegistryInvalid
    from toktier.repair import registry as repair_registry

    roster = repair_registry.families()
    duplicate = next(item for item in roster if item.family == "qwen3_8b")
    monkeypatch.setattr(repair_registry, "carried_families", lambda: (duplicate,))
    repair_registry._sources.cache_clear()
    try:
        with pytest.raises(RegistryInvalid):
            repair_registry.family_spec("qwen3_8b", duplicate.artifact_sha256)
    finally:
        monkeypatch.undo()
        repair_registry._sources.cache_clear()


# -- negative: a tampered copy of the shipped files -------------------------


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A copy of the files the release check reads, rooted at ``tmp_path``."""
    for relative in ("readings", "evidence"):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    (tmp_path / ROSTER.relative_to(ROOT)).parent.mkdir(parents=True)
    shutil.copy(ROSTER, tmp_path / ROSTER.relative_to(ROOT))
    (tmp_path / "tables").mkdir()
    shutil.copy(SHIPPED, tmp_path / "tables" / "support_registry.json")
    monkeypatch.setattr(generate_registry, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        generate_registry, "REPAIR_ROSTER_PATH", tmp_path / ROSTER.relative_to(ROOT)
    )
    yield tmp_path


def _edit_registry(tree: Path, change: Callable[[dict[str, Any]], None]) -> Path:
    path = tree / "tables" / "support_registry.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _edit_reading(
    tree: Path,
    family: str,
    member: str,
    change: Callable[[dict[str, Any]], None],
) -> Path:
    """Change one reading named by a carried node and re-pin its digest."""
    registry = tree / "tables" / "support_registry.json"
    document = json.loads(registry.read_text(encoding="utf-8"))
    node = _row(document, family)["carryover"][member]
    path = tree / node["path"]
    reading = json.loads(path.read_text(encoding="utf-8"))
    change(reading)
    path.write_text(json.dumps(reading), encoding="utf-8")
    node["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    registry.write_text(json.dumps(document), encoding="utf-8")
    return registry


def _problems(path: Path) -> str:
    return "\n".join(carryover_problems(path))


def test_the_copy_passes(tree: Path) -> None:
    assert carryover_problems(tree / "tables" / "support_registry.json") == []


def test_a_certificate_with_one_wrong_literal_is_refused(tree: Path) -> None:
    def change(reading: dict[str, Any]) -> None:
        reading["subject"]["divergence_set"][0]["content"] += "x"

    path = _edit_reading(tree, "deepseek_v4_1_flash", "certificate", change)
    assert "certificate divergence set differs from the registry" in _problems(path)


def test_a_node_naming_a_wrong_anchor_digest_is_refused(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        _row(document, "mimo_v2_6")["carryover"]["from_artifact_sha256"] = "a" * 64

    problems = _problems(_edit_registry(tree, change))
    assert "carry-over names anchor artifact" in problems
    assert "certificate subject from_artifact_sha256" in problems


def test_a_certificate_naming_a_wrong_anchor_digest_is_refused(tree: Path) -> None:
    def change(reading: dict[str, Any]) -> None:
        reading["subject"]["from_artifact_sha256"] = "a" * 64

    path = _edit_reading(tree, "mimo_v2_6", "certificate", change)
    assert "certificate subject from_artifact_sha256" in _problems(path)


def test_an_anchor_that_is_itself_carried_is_refused(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        row = _row(document, "mimo_v2_6")
        base = _row(document, "qwen3_5_08b")
        row["carryover"]["from_family"] = "qwen3_5_08b"
        row["carryover"]["from_artifact_sha256"] = base["artifact_sha256"]

    assert "carry-over does not chain" in _problems(_edit_registry(tree, change))


def test_a_repair_parameter_off_by_one_is_refused(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        _row(document, "deepseek_v4_1_flash")["backends"]["fast_cpu"]["repair"][
            "margin"
        ] += 1

    assert "repair margin=37 differs" in _problems(_edit_registry(tree, change))


def test_repair_parameters_without_a_cross_artifact_node_are_refused(
    tree: Path,
) -> None:
    def change(document: dict[str, Any]) -> None:
        _row(document, "qwen3_8b")["backends"]["fast_cpu"]["repair"] = {
            "margin": 20,
            "effective_l_max": 128,
            "has_normalizer": True,
            "source_table_sha256": "0" * 64,
        }

    assert "without a cross-artifact carry-over" in _problems(
        _edit_registry(tree, change)
    )


def test_readings_one_document_off_are_refused(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        _row(document, "mimo_v2_6")["readings"]["docs"] -= 1

    assert "readings differs from the anchor" in _problems(_edit_registry(tree, change))


def test_the_anchors_added_frontend_is_refused_as_a_sibling(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        row = _row(document, "mimo_v2_6")
        row["added_frontend_id"] = _row(document, "qwen3_8b")["added_frontend_id"]

    assert "a sibling, not a carry-over" in _problems(_edit_registry(tree, change))


def test_a_certificate_corpus_at_another_revision_is_refused(tree: Path) -> None:
    def change(reading: dict[str, Any]) -> None:
        reading["corpus"]["revision"] = "0" * 40

    path = _edit_reading(tree, "deepseek_v4_1_flash", "certificate", change)
    assert "is not the corpus of the evidence manifest" in _problems(path)


def test_a_zero_positive_control_is_refused(tree: Path) -> None:
    def change(reading: dict[str, Any]) -> None:
        reading["scan"]["controls"]["the"] = 0

    path = _edit_reading(tree, "mimo_v2_6", "certificate", change)
    assert "positive control 'the' is not nonzero" in _problems(path)


def test_a_parity_reading_with_a_mismatch_is_refused(tree: Path) -> None:
    def change(reading: dict[str, Any]) -> None:
        reading["mismatches"] = 1

    path = _edit_reading(tree, "mimo_v2_6", "engine_parity", change)
    assert "engine-parity reading reports mismatches" in _problems(path)


def test_a_parity_reading_of_another_artifact_is_refused(tree: Path) -> None:
    def change(reading: dict[str, Any]) -> None:
        reading["subject"]["artifact_sha256"] = "b" * 64

    path = _edit_reading(tree, "deepseek_v4_1_flash", "engine_parity", change)
    assert "engine-parity reading is of another artifact" in _problems(path)


def test_a_carried_record_without_a_parity_reading_is_refused(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        del _row(document, "mimo_v2_6")["carryover"]["engine_parity"]

    assert "has no engine_parity reading" in _problems(_edit_registry(tree, change))


# -- C7 binds the version-normalized engine identity -------------------------


def _bump_engine_v1(document: dict[str, Any], digest: str) -> None:
    """What a version bump alone does to the registry: every certified fast
    CPU entry gets a new per-build ``source_digest`` and keeps its
    ``source_digest_v2``."""
    for item in document["artifacts"]:
        fast = (item.get("backends") or {}).get("fast_cpu") or {}
        if fast.get("status") == "certified_source":
            fast["source_digest"] = digest


def test_a_parity_reading_bound_by_v2_survives_a_version_bump(tree: Path) -> None:
    v2 = _row(_document(), "mimo_v2_6")["backends"]["fast_cpu"]["source_digest_v2"]

    def record_v2(reading: dict[str, Any]) -> None:
        reading["engine"]["source_digest_v2"] = v2

    path = _edit_reading(tree, "mimo_v2_6", "engine_parity", record_v2)
    assert "mimo_v2_6" not in _problems(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    _bump_engine_v1(document, "c" * 64)
    path.write_text(json.dumps(document), encoding="utf-8")
    problems = _problems(path)
    assert "mimo_v2_6: engine-parity" not in problems, problems
    # The reading without the version-normalized identity is held to the
    # per-build identity, so the same bump asks for a retake.
    assert (
        "deepseek_v4_1_flash: engine-parity reading was taken on another fast CPU"
        in problems
    )
    assert "retake it recording engine.source_digest_v2" in problems


def test_a_parity_reading_of_another_v2_is_refused(tree: Path) -> None:
    def change(reading: dict[str, Any]) -> None:
        reading["engine"]["source_digest_v2"] = "d" * 64

    path = _edit_reading(tree, "deepseek_v4_1_flash", "engine_parity", change)
    assert (
        "deepseek_v4_1_flash: engine-parity reading was taken on another "
        "version-normalized fast CPU engine source identity"
    ) in _problems(path)


def test_a_parity_reading_without_v2_is_held_to_the_per_build_identity(
    tree: Path,
) -> None:
    def change(reading: dict[str, Any]) -> None:
        reading["engine"]["source_digest"] = "e" * 64

    path = _edit_reading(tree, "mimo_v2_6", "engine_parity", change)
    assert "mimo_v2_6: engine-parity reading was taken on another fast CPU" in (
        _problems(path)
    )


def test_a_parity_reading_bound_by_v2_still_names_its_per_build_identity(
    tree: Path,
) -> None:
    """``source_digest_v2`` is computed from the tree; ``source_digest`` is
    what the executing extension reported, and has to be there too."""
    v2 = _row(_document(), "mimo_v2_6")["backends"]["fast_cpu"]["source_digest_v2"]

    def v2_only(reading: dict[str, Any]) -> None:
        reading["engine"]["source_digest_v2"] = v2
        del reading["engine"]["source_digest"]

    path = _edit_reading(tree, "mimo_v2_6", "engine_parity", v2_only)
    assert "records source_digest_v2 but no per-build engine identity" in (
        _problems(path)
    )

    def malformed(reading: dict[str, Any]) -> None:
        reading["engine"]["source_digest_v2"] = v2
        reading["engine"]["source_digest"] = "not a digest"

    path = _edit_reading(tree, "mimo_v2_6", "engine_parity", malformed)
    assert "malformed per-build engine identity" in _problems(path)
