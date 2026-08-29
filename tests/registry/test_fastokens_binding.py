"""The pinned Fastokens registry node and the binding that generates it.

Acceptance surface: the shipped registry carries the node the binding
describes, the binding's numbers are the readings' numbers, a placeholder
wheel digest is refused by the release check, the evidence id resolves to a
manifest, and the guard set the node carries is the full-domain reading.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import generate_registry
import pytest
import update_fastokens_registry as tool
from registry_common import (
    PLACEHOLDER_SHA256,
    GenerationError,
    load_json,
    schema_violations,
)

ROOT = Path(__file__).resolve().parents[2]


def _json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _registry() -> dict[str, Any]:
    return cast(dict[str, Any], _json("tables/support_registry.json"))


def _binding() -> dict[str, Any]:
    return cast(dict[str, Any], _json("tools/fastokens_binding.json"))


def test_shipped_registry_carries_the_bound_node() -> None:
    registry = _registry()
    node = registry["engine_distributions"]["fastokens"]
    assert node == tool.node_from_binding(_binding())
    packaged = _json("src/toktier/routing/tables/support_registry.v1.json")
    assert packaged == registry
    assert tool.main(["--check"]) == 0
    assert generate_registry.fastokens_binding_problems(tool.DEFAULT_REGISTRY) == []


def test_the_node_says_what_was_published_and_measured() -> None:
    node = _registry()["engine_distributions"]["fastokens"]
    assert node["backend"] == "fastokens"
    assert node["admission"] == "experimental"
    assert node["distribution"]["name"] == "toktier-fastokens"
    assert node["distribution"]["version"] == "0.3.1.1"
    assert node["distribution"]["import_name"] == "fastokens"
    assert node["recognised_distributions"] == ["toktier-fastokens", "fastokens"]
    (wheel,) = node["known_wheels"]
    assert wheel["filename"] == (
        "toktier_fastokens-0.3.1.1-cp39-abi3-manylinux_2_28_x86_64.whl"
    )
    assert wheel["sha256"] == (
        "b99f2765fa1b900afe181844a85ed8eb784ba87972ac92e22cc924322d9c5468"
    )
    assert wheel["engine_digest"] == (
        "0bcf3ada9268e5aef1c9da515555f5e2ea6fc8d7a8accfbc444789853edfdfec"
    )
    assert wheel["published"]["index"] == "PyPI"
    assert node["sdist"]["sha256"] == (
        "7c275f907d26107d2f4605821372e9104c6679e240f88b41f22a521445b86969"
    )
    assert node["oracle"] == {"package": "tokenizers", "version": "0.22.2"}
    assert len(node["families"]) == 15
    evidence = node["evidence"]
    assert evidence["evidence_id"] == "ev-fastokens-pinned-v1"
    assert evidence["docs_per_family"] == 998857881
    assert evidence["comparisons"] == 14982868215
    assert evidence["mismatch_raw"] == 28
    assert evidence["mismatch_guarded"] == 0
    assert evidence["engine_error"] == 0
    assert evidence["routed_reference_per_family"] == 505
    assert evidence["visible_cpus"] == 8
    assert evidence["gate3"]["topologies"] == 6
    assert evidence["gate4"] == {
        "families": 13,
        "splices": 157872,
        "edits": 23400,
        "failing": 0,
    }


def test_every_family_the_repair_table_reaches_is_inside_the_evidence() -> None:
    """The premise behind an assurance state the shipped tables cannot reach.

    ``engine_assurance: family_outside_evidence`` asks whether the pinned
    readings cover a family. The adapter is only ever opened for a family
    in the certified repair table, so the state is reachable only if that
    table reaches outside the evidence. It does not, and the difference
    runs the other way: four covered families have no repair entry, which
    is the separate ``fastokens_family_admitted: false`` answer. Pinning
    the containment keeps ``docs/contracts/facade.md`` true about which
    of the four combinations the shipped package can produce.
    """
    node = _registry()["engine_distributions"]["fastokens"]
    covered = {(row["family"], row["artifact_sha256"]) for row in node["families"]}
    repair = json.loads(
        (
            ROOT
            / "src"
            / "toktier"
            / "repair"
            / "tables"
            / "fast_repair_families.v1.json"
        ).read_text(encoding="utf-8")
    )
    rows = repair["families"] if isinstance(repair, dict) else repair
    reachable = {(row["family"], row["artifact_sha256"]) for row in rows}

    assert len(reachable) == 11
    assert reachable <= covered
    assert {family for family, _ in covered - reachable} == {
        "hy3",
        "kimi_k3",
        "laguna_s_2_1",
        "ling_3_0_flash",
    }


def test_the_evidence_id_resolves_to_a_manifest() -> None:
    node = _registry()["engine_distributions"]["fastokens"]
    manifest = _json("evidence/evidence_manifest_fastokens_pinned.json")
    assert manifest["evidence_id"] == node["evidence"]["evidence_id"]
    assert manifest["run"]["suite_version"] == node["evidence"]["suite_version"]
    assert manifest["totals"]["docs"] == node["evidence"]["docs_per_family"]
    assert manifest["totals"]["mismatches"] == node["evidence"]["mismatch_guarded"]
    assert (
        manifest["totals"]["routed"] == node["evidence"]["routed_reference_per_family"]
    )
    assert {row["family"] for row in manifest["artifacts"]} == {
        row["family"] for row in node["families"]
    }


def test_the_guard_set_is_the_full_domain_reading() -> None:
    node = _registry()["engine_distributions"]["fastokens"]
    guard = node["guard"]
    reading = _json("readings/fastokens_pinned_guard_full_domain.json")
    codepoints = tool.guard_codepoints(guard["ranges"])
    assert len(codepoints) == guard["codepoints"] == 154
    assert [f"U+{cp:04X}" for cp in codepoints] == reading["codepoints"]
    assert (
        tool.guard_set_digest(codepoints)
        == guard["set_sha256"]
        == reading["set_sha256"]
    )
    assert reading["comparison"]["verdict"] == "SUPERSET"
    assert reading["comparison"]["archived_size"] == 108
    assert len(reading["comparison"]["only_in_full_domain"]) == 46
    assert reading["engine_digest"] == node["known_wheels"][0]["engine_digest"]


def test_every_reading_was_taken_on_the_published_wheel() -> None:
    node = _registry()["engine_distributions"]["fastokens"]
    engine = node["known_wheels"][0]["engine_digest"]
    readings = node["evidence"]["readings"]
    assert _json(readings["gate1"])["subject"]["engine_digest"] == engine
    for gate in ("gate2", "gate3", "gate4", "guard"):
        assert _json(readings[gate])["engine_digest"] == engine
    gate1 = _json(readings["gate1"])
    assert gate1["verdict_guarded"] == "PASS"
    assert gate1["verdict_raw"] == "FAIL"
    assert gate1["raw_mismatch_classification"]["unclassified"] == 0


def test_the_schema_accepts_the_node_and_rejects_an_unknown_member() -> None:
    schema = _json("schemas/support_registry.schema.json")
    registry = _registry()
    assert schema_violations(registry, schema) == []
    edited = copy.deepcopy(registry)
    edited["engine_distributions"]["fastokens"]["extra"] = 1
    assert schema_violations(edited, schema) != []
    edited = copy.deepcopy(registry)
    del edited["engine_distributions"]["fastokens"]["guard"]
    assert schema_violations(edited, schema) != []


def test_the_release_check_refuses_a_placeholder_wheel_digest(tmp_path: Path) -> None:
    registry = _registry()
    edited = copy.deepcopy(registry)
    wheel = edited["engine_distributions"]["fastokens"]["known_wheels"][0]
    wheel["engine_digest"] = PLACEHOLDER_SHA256
    path = tmp_path / "support_registry.json"
    path.write_text(json.dumps(edited), encoding="utf-8")
    problems = generate_registry.release_problems(path)
    assert any("known_wheels[0]: engine_digest is a placeholder" in p for p in problems)
    assert generate_registry.release_problems(tool.DEFAULT_REGISTRY) == []


def test_a_binding_that_drifts_from_its_readings_is_refused() -> None:
    registry = _registry()
    binding = _binding()
    drifted = copy.deepcopy(binding)
    drifted["evidence"]["mismatch_guarded"] = 1
    with pytest.raises(GenerationError, match="differs from the gate1 reading"):
        tool.augmented_document(registry, drifted)
    drifted = copy.deepcopy(binding)
    drifted["guard"]["ranges"] = drifted["guard"]["ranges"][:-1]
    with pytest.raises(GenerationError, match=r"guard\.codepoints"):
        tool.augmented_document(registry, drifted)
    drifted = copy.deepcopy(binding)
    drifted["known_wheels"][0]["engine_digest"] = "0" * 64
    with pytest.raises(GenerationError, match="not derived on the first known wheel"):
        tool.augmented_document(registry, drifted)


def test_the_adapter_source_is_bound() -> None:
    binding = _binding()
    adapter = binding["adapter"]
    observed = hashlib.sha256((ROOT / adapter["path"]).read_bytes()).hexdigest()
    assert observed == adapter["source_sha256"]
    assert adapter["guard_set_sha256"] == binding["guard"]["set_sha256"]


def test_the_legal_material_is_bound() -> None:
    legal = _binding()["legal"]
    for path_key, digest_key in (
        ("license_path", "license_sha256"),
        ("notice_path", "notice_sha256"),
        ("sbom_path", "sbom_sha256"),
        ("license_bundle_path", "license_bundle_sha256"),
    ):
        observed = hashlib.sha256((ROOT / legal[path_key]).read_bytes()).hexdigest()
        assert observed == legal[digest_key], path_key
    series = _binding()["source"]["patch_series"]
    assert [entry["changes_code"] for entry in series] == [True] * 5 + [False]
    for entry in series:
        observed = hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest()
        assert observed == entry["sha256"], entry["file"]


def test_the_document_check_helper_sees_the_node(tmp_path: Path) -> None:
    """``load_json`` of the packaged copy is what the adapter reads at runtime."""
    packaged = load_json(
        ROOT / "src" / "toktier" / "routing" / "tables" / "support_registry.v1.json"
    )
    assert "engine_distributions" in packaged
    from toktier.routing.registry_load import load_registry_document
    from toktier.routing.tables import SUPPORT_REGISTRY

    # The typed document names the frozen v1 keys; the optional node is
    # read the way the adapter reads it, as an untyped member.
    document = cast(dict[str, Any], load_registry_document(SUPPORT_REGISTRY))
    assert document["engine_distributions"]["fastokens"]["backend"] == "fastokens"
