"""Directly judged records: record-held repair without a carry-over.

A fast CPU entry may hold a ``repair`` member in a second form beside the
carried one: a face judged directly, with no carry-over node, certified by
a full-corpus engine differential of its own
(``schemas/engine_differential_reading.schema.json``) and named in
``tools/fast_cpu_binding.json`` under ``direct_families``. The shipped
registry holds no such record yet; these tests build one on a copy of the
shipped files (family ``hy4`` with its derived parameters) and require
the binding tool to produce the entry and the release check
(``carryover_problems``, conditions D1-D6) to accept it, then refuse each
change that breaks a condition.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import check_reading_forms as forms
import generate_registry
import pytest
import update_fast_cpu_registry
from compute_identity_v2 import source_digest as source_digest_v2
from engine_core_identity import engine_core_digests
from registry_common import carried_repair_records

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "tables" / "support_registry.json"
ROSTER = ROOT / "src" / "toktier" / "repair" / "tables" / "fast_repair_families.v1.json"
BINDING = ROOT / "tools" / "fast_cpu_binding.json"
FAMILY = "hy4"
REPAIR = {
    "margin": 29,
    "effective_l_max": 33,
    "has_normalizer": True,
    "source_table_sha256": (
        "aaa934b6fd13a56a2921f3d4e761984669f3b9ad860a95589ec58c70d67d4cd1"
    ),
}
READING = f"readings/fast_cpu_engine_differential_{FAMILY}.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _row(document: dict[str, Any], family: str) -> dict[str, Any]:
    return next(item for item in document["artifacts"] if item["family"] == family)


def _build(v2: str) -> dict[str, Any]:
    binding = _load(BINDING)
    return {
        "source_digest": binding["source_digest"],
        "source_digest_v2": v2,
        "build_flags": binding["build_flags"],
        "toolchain": binding["toolchain"],
    }


def _reading(artifact: dict[str, Any]) -> dict[str, Any]:
    documents = int(_load(BINDING)["validation"]["documents"])
    characters = int(_load(BINDING)["validation"]["characters"])
    v2 = source_digest_v2("fast_cpu")
    residue = 6_000_000
    bridging = documents // 100
    return {
        "schema": "toktier.fast_cpu.engine_differential.v1",
        "subject": {
            "family": artifact["family"],
            "artifact_sha256": artifact["artifact_sha256"],
            "loader_face_sha256": artifact["loader_face_sha256"],
            "repair": dict(REPAIR),
            "longest_vocabulary_token_chars": 33,
        },
        "engine_core": engine_core_digests(),
        "build": _build("1" * 64),
        "corpus": {
            "manifest_sha256": "2" * 64,
            "documents": documents,
            "characters": characters,
            "oracle_version": "0.22.2",
            "harness_sha256": "3" * 64,
        },
        "main_pass": {
            "documents": documents,
            "characters": characters,
            "mismatches": 0,
            "path_classes": {
                "gigatoken_full": documents - residue,
                "hf_full_added_token": residue,
                "hf_full_engine_guard": 0,
            },
            "residue": {"count": residue, "list_sha256": "4" * 64},
        },
        "append_replay": {
            "documents": documents // 40,
            "mismatches": 0,
            "sampling": "1 in 40, seeded cut",
            "threshold_decisive": 0,
        },
        "constructed_arm": {
            "strings": 1_000_000,
            "mismatches": 0,
            "threshold_decisive": 100_000,
            "comparison_l_max": 128,
            "build": _build(v2),
        },
        "residue_supplement": {
            "documents": residue,
            "mismatches_full": 0,
            "mismatches_append": 0,
            "path_classes": {
                "gigatoken_full": 0,
                "gigatoken_segmented": residue - 2,
                "hf_added_token": 2,
                "hf_engine_guard": 0,
            },
            "reference_answered": {"count": 2, "list_sha256": "5" * 64},
            "build": _build(v2),
        },
        "bridging_check": {
            "documents": bridging,
            "sample_share": 0.01,
            "mismatches_full": 0,
            "mismatches_append": 0,
            "path_class_changes_literal_free": 0,
            "build": _build(v2),
        },
        "sensitivity_limit": {
            "per_document": 1 / bridging,
            "statement": (
                "the bridging sample cannot resolve a frozen-build defect rate "
                "below about one in its document count on literal-free input"
            ),
        },
    }


def _parity_row(artifact: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    documents = 130_679
    return {
        "artifact_sha256": artifact["artifact_sha256"],
        "characters": 403_376_460,
        "documents": documents,
        "family": artifact["family"],
        "full_encode_equal_hf": True,
        "mismatches": 0,
        "repair_equal_hf": True,
        "special_tokens": {
            "postprocessor_class": "identity",
            "prefix_ids": [],
            "documents": documents,
            "mismatches": 0,
            "served_by": {"facade": documents, "reference": 0},
            "evidence_role": "regression_guard",
        },
        "literal_segments": {
            "literal_class": "served",
            "added_frontend_fingerprint": fingerprint,
            "battery": {
                "generator_sha256": "6" * 64,
                "strings": 120_000,
                "mismatches": 0,
            },
            "entry_points": [
                {
                    "entry_point": "id_route",
                    "documents": 1000,
                    "mismatches": 0,
                    "paths": {
                        "gigatoken_segmented": 1000,
                        "gpu_segmented": 0,
                        "hf_added_token": 0,
                        "hf_engine_guard": 0,
                    },
                    "segments": {
                        "literals": 4000,
                        "text": 5000,
                        "largest_text_bytes": 9000,
                        "bytes": {"gpu": 0, "fast_cpu": 900_000, "hf": 0},
                    },
                    "guard_documents": {"count": 0, "bytes_hf": 0, "bound": 1},
                }
            ],
        },
    }


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A copy of the shipped files with ``hy4`` judged directly."""
    for relative in ("readings", "evidence"):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    roster = tmp_path / ROSTER.relative_to(ROOT)
    roster.parent.mkdir(parents=True)
    shutil.copy(ROSTER, roster)
    examined = Path("src/toktier/routing/tables/examined_artifacts.v1.json")
    (tmp_path / examined).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / examined, tmp_path / examined)
    (tmp_path / "tables").mkdir()
    (tmp_path / "tools").mkdir()

    document = _load(SHIPPED)
    artifact = _row(document, FAMILY)
    uniform = copy.deepcopy(_row(document, "deepseek_v3")["backends"]["fast_cpu"])
    reading = _reading(artifact)
    _dump(tmp_path / READING, reading)
    digest = hashlib.sha256((tmp_path / READING).read_bytes()).hexdigest()
    artifact["backends"]["fast_cpu"] = {
        **uniform,
        "repair": dict(REPAIR),
        "engine_differential": {
            "reading": READING,
            "sha256": digest,
            "documents": reading["main_pass"]["documents"],
            "mismatches": 0,
        },
    }
    _dump(tmp_path / "tables" / "support_registry.json", document)

    binding = _load(BINDING)
    del binding["rejected_families"][FAMILY]
    binding["direct_families"] = {FAMILY: {"reading": READING, "sha256": digest}}
    _dump(tmp_path / "tools" / "fast_cpu_binding.json", binding)

    parity_path = tmp_path / "readings" / "fast_cpu_native_frontend_parity.json"
    parity = _load(parity_path)
    fingerprint = forms.registry_fingerprints(document)[FAMILY]
    parity["rows"].append(_parity_row(artifact, fingerprint))
    parity["unique_artifacts"] += 1
    parity["model_families"] += 1
    _dump(parity_path, parity)

    monkeypatch.setattr(generate_registry, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(generate_registry, "REPAIR_ROSTER_PATH", roster)
    monkeypatch.setattr(
        generate_registry,
        "FAST_CPU_BINDING_PATH",
        tmp_path / "tools" / "fast_cpu_binding.json",
    )
    monkeypatch.setattr(update_fast_cpu_registry, "READINGS_ROOT", tmp_path)
    yield tmp_path


def _registry(tree: Path) -> Path:
    return tree / "tables" / "support_registry.json"


def _problems(tree: Path) -> list[str]:
    return [
        problem
        for problem in generate_registry.carryover_problems(_registry(tree))
        if problem.startswith(FAMILY)
    ]


def _edit_registry(tree: Path, change: Callable[[dict[str, Any]], None]) -> None:
    document = _load(_registry(tree))
    change(document)
    _dump(_registry(tree), document)


def _edit_reading(tree: Path, change: Callable[[dict[str, Any]], None]) -> None:
    reading = _load(tree / READING)
    change(reading)
    _dump(tree / READING, reading)
    digest = hashlib.sha256((tree / READING).read_bytes()).hexdigest()

    def repin(document: dict[str, Any]) -> None:
        _row(document, FAMILY)["backends"]["fast_cpu"]["engine_differential"][
            "sha256"
        ] = digest

    _edit_registry(tree, repin)
    binding_path = tree / "tools" / "fast_cpu_binding.json"
    binding = _load(binding_path)
    binding["direct_families"][FAMILY]["sha256"] = digest
    _dump(binding_path, binding)


# -- positive ---------------------------------------------------------------


def test_the_shipped_registry_holds_no_direct_record() -> None:
    binding = _load(BINDING)
    assert binding.get("direct_families", {}) == {}
    for artifact in _load(SHIPPED)["artifacts"]:
        assert "engine_differential" not in (
            artifact["backends"].get("fast_cpu") or {}
        )


def test_a_directly_judged_record_passes(tree: Path) -> None:
    assert _problems(tree) == []
    # ... and the release check of the whole copy is otherwise unchanged.
    assert generate_registry.carryover_problems(_registry(tree)) == []


def test_the_binding_tool_produces_the_direct_entry(tree: Path) -> None:
    document = _load(_registry(tree))
    binding = _load(tree / "tools" / "fast_cpu_binding.json")
    generated = update_fast_cpu_registry.augmented_document(document, binding)
    entry = _row(generated, FAMILY)["backends"]["fast_cpu"]
    uniform = _row(generated, "deepseek_v3")["backends"]["fast_cpu"]
    assert entry["repair"] == REPAIR
    assert entry["engine_differential"]["reading"] == READING
    assert entry["engine_differential"]["mismatches"] == 0
    assert {
        key: value
        for key, value in entry.items()
        if key not in ("repair", "engine_differential")
    } == uniform
    # The schema admits the entry.
    schema = _load(ROOT / "schemas" / "support_registry.schema.json")
    from registry_common import schema_violations

    assert schema_violations(generated, schema) == []
    # The chunked admission roster does not select it (no carry-over node).
    assert FAMILY not in carried_repair_records(generated)


def test_the_binding_tool_refuses_a_direct_family_with_a_carry_over(
    tree: Path,
) -> None:
    document = _load(_registry(tree))
    _row(document, FAMILY)["carryover"] = {"divergence_set": []}
    binding = _load(tree / "tools" / "fast_cpu_binding.json")
    with pytest.raises(Exception, match="holds no carry-over node"):
        update_fast_cpu_registry.augmented_document(document, binding)


def test_the_binding_tool_requires_both_columns_on_the_direct_row(
    tree: Path,
) -> None:
    parity_path = tree / "readings" / "fast_cpu_native_frontend_parity.json"
    parity = _load(parity_path)
    del parity["rows"][-1]["literal_segments"]
    _dump(parity_path, parity)
    document = _load(_registry(tree))
    binding = _load(tree / "tools" / "fast_cpu_binding.json")
    with pytest.raises(Exception, match="carries no literal_segments column"):
        update_fast_cpu_registry.augmented_document(document, binding)


# -- negative: one condition at a time ---------------------------------------


def test_d1_a_record_not_listed_in_the_binding_is_refused(tree: Path) -> None:
    binding_path = tree / "tools" / "fast_cpu_binding.json"
    binding = _load(binding_path)
    binding["direct_families"] = {}
    _dump(binding_path, binding)
    assert any(
        "does not list the family as directly judged" in p for p in _problems(tree)
    )


def test_d1_a_record_in_the_examined_table_is_refused(tree: Path) -> None:
    """D1: a directly judged record is not an examined artifact either (the
    shipped examined table is empty, so the check is exercised here)."""
    examined_path = tree / "src" / "toktier" / "routing" / "tables"
    examined_path /= "examined_artifacts.v1.json"
    examined = _load(examined_path)
    artifact = _row(_load(_registry(tree)), FAMILY)
    examined.setdefault("artifacts", []).append(
        {"family": FAMILY, "artifact_sha256": artifact["artifact_sha256"]}
    )
    _dump(examined_path, examined)
    assert any("is in the examined table" in p for p in _problems(tree))


def test_d1_a_carry_over_node_is_refused(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        _row(document, FAMILY)["carryover"] = {"divergence_set": []}

    _edit_registry(tree, change)
    assert any("holds no carry-over node" in p for p in _problems(tree))


def test_d2_parameters_other_than_the_reading_are_refused(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        _row(document, FAMILY)["backends"]["fast_cpu"]["repair"]["effective_l_max"] = 34

    _edit_registry(tree, change)
    assert any("but the record holds" in p for p in _problems(tree))


def test_d2_a_window_below_the_longest_token_is_refused(tree: Path) -> None:
    def change(reading: dict[str, Any]) -> None:
        reading["subject"]["longest_vocabulary_token_chars"] = 40

    _edit_reading(tree, change)
    assert any("below the longest vocabulary token" in p for p in _problems(tree))


def test_d3_an_entry_other_than_the_uniform_one_is_refused(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        _row(document, FAMILY)["backends"]["fast_cpu"]["toolchain"] = "rustc 0.0.0"

    _edit_registry(tree, change)
    assert any("not the uniform certified entry" in p for p in _problems(tree))


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda r: r["main_pass"].update(mismatches=1),
            "main pass 1 mismatch",
        ),
        (
            lambda r: r["engine_core"].update(core_content_sha256="7" * 64),
            "engine core core_content_sha256",
        ),
        (
            lambda r: r["corpus"].update(documents=r["corpus"]["documents"] - 1),
            "the campaign corpus holds",
        ),
        (
            lambda r: r["main_pass"]["residue"].update(count=1),
            "the residue lists 1 documents",
        ),
        (
            lambda r: r["residue_supplement"]["reference_answered"].update(count=0),
            "listed as reference-answered",
        ),
        (
            lambda r: r["residue_supplement"].update(mismatches_append=1),
            "residue supplement records mismatches",
        ),
        (
            lambda r: r["bridging_check"].update(path_class_changes_literal_free=3),
            "3 literal-free document(s) changed path class",
        ),
        (
            lambda r: r["bridging_check"].update(documents=10),
            "its sample share plans",
        ),
        (
            lambda r: r["sensitivity_limit"].update(per_document=1e-12),
            "finer than the bridging sample can resolve",
        ),
        (
            lambda r: r["constructed_arm"].update(strings=999_999),
            "fewer than 1000000",
        ),
        (
            lambda r: r["constructed_arm"].update(threshold_decisive=0),
            "the window bound decided 0 of 1000000 outcomes",
        ),
        (
            lambda r: r["constructed_arm"].update(threshold_decisive=9_999),
            "decided 9999 of 1000000 outcomes, below 1%",
        ),
        (
            lambda r: r["constructed_arm"].update(comparison_l_max=64),
            "compared with window bound 64, not 128",
        ),
        (
            lambda r: r["corpus"].update(characters=r["corpus"]["characters"] - 1)
            or r["main_pass"].update(characters=r["corpus"]["characters"]),
            "characters, the campaign corpus holds 12328592579973",
        ),
        (
            lambda r: r["bridging_check"].update(
                sample_share=0.004, documents=r["bridging_check"]["documents"] * 2 // 5
            ),
            "bridging sample share 0.004 is below 1%",
        ),
        (
            lambda r: r["append_replay"].update(documents=1000),
            "the append replay covered 1000 documents, below 2%",
        ),
        (
            lambda r: r["corpus"].update(oracle_version="0.0.0"),
            "judged against tokenizers 0.0.0, the campaign's reference is 0.22.2",
        ),
        (
            lambda r: r["constructed_arm"].update(threshold_decisive=1_000_001),
            "counts 1000001 decided outcomes of 1000000 strings",
        ),
        (
            lambda r: r["append_replay"].update(
                documents=r["corpus"]["documents"] + 1
            ),
            "more than the corpus holds",
        ),
        (
            lambda r: r["bridging_check"].update(sample_share=float("nan")),
            "sample_share: not a finite number",
        ),
        (
            lambda r: r["residue_supplement"]["path_classes"].update(
                gigatoken_segmented=0, hf_added_token=6_000_000
            )
            or r["residue_supplement"]["reference_answered"].update(count=6_000_000),
            "above 1% for a served face",
        ),
        (
            lambda r: r["bridging_check"]["build"].update(source_digest_v2="8" * 64),
            "bridging_check was taken on fast CPU v2",
        ),
        (
            lambda r: r["subject"].update(loader_face_sha256="9" * 64),
            "subject loader_face_sha256",
        ),
    ],
)
def test_d4_d6_the_reading_is_held(
    tree: Path, change: Callable[[dict[str, Any]], None], expected: str
) -> None:
    _edit_reading(tree, change)
    problems = _problems(tree)
    assert any(expected in problem for problem in problems), problems


def test_d4_a_reading_with_another_digest_is_refused(tree: Path) -> None:
    reading = _load(tree / READING)
    reading["main_pass"]["characters"] += 0
    (tree / READING).write_text(json.dumps(reading), encoding="utf-8")
    assert any("does not match the recorded" in p for p in _problems(tree))


def test_d5_the_parity_row_and_its_columns_are_required(tree: Path) -> None:
    parity_path = tree / "readings" / "fast_cpu_native_frontend_parity.json"
    parity = _load(parity_path)
    row = parity["rows"][-1]
    del row["special_tokens"]
    row["literal_segments"]["entry_points"][0]["paths"]["hf_added_token"] = 1
    row["literal_segments"]["entry_points"][0]["paths"]["gigatoken_segmented"] = 999
    _dump(parity_path, parity)
    problems = _problems(tree)
    assert any("carries no special_tokens column" in p for p in problems)
    assert any("went whole to the reference for a literal" in p for p in problems)
    parity["rows"] = parity["rows"][:-1]
    _dump(parity_path, parity)
    assert any("does not list the face" in p for p in _problems(tree))


def test_a_face_at_the_comparison_bound_counts_no_decided_outcome(
    tree: Path,
) -> None:
    """``effective_l_max == 128``: the arm cannot tell the two engines apart,
    so its count has to be zero (and is accepted at zero)."""

    def at_bound(reading: dict[str, Any]) -> None:
        reading["subject"]["repair"]["effective_l_max"] = 128
        reading["constructed_arm"]["threshold_decisive"] = 0

    _edit_reading(tree, at_bound)
    problems = _problems(tree)
    assert not any("constructed arm" in p for p in problems), problems
    _edit_reading(tree, lambda r: r["constructed_arm"].update(threshold_decisive=5))
    assert any("equals the comparison bound 128" in p for p in _problems(tree))


def test_engine_differential_beside_a_carry_over_is_refused(tree: Path) -> None:
    def change(document: dict[str, Any]) -> None:
        carried = _row(document, "mimo_v2_6")["backends"]["fast_cpu"]
        carried["engine_differential"] = {
            "reading": READING,
            "sha256": "0" * 64,
            "documents": 1,
            "mismatches": 0,
        }

    _edit_registry(tree, change)
    problems = generate_registry.carryover_problems(_registry(tree))
    assert any(
        problem.startswith("mimo_v2_6: engine_differential belongs to the directly")
        for problem in problems
    )
