"""The 0.3.2 parity-reading columns and their checks.

``tools/check_reading_forms.py`` defines the per-family columns a
native-front-end parity row may carry. These tests pin the rules beyond
the schemas: the ``add_special_tokens=True`` column (``special_tokens``)
is evidence only when the facade answered on an identity or prefix face,
says on an identity face that it is a regression guard, and treats
``loader_true_equal`` as informative.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import aggregate_native_frontend_parity as aggregate  # noqa: E402
import check_reading_forms as forms  # noqa: E402
from registry_common import GenerationError  # noqa: E402


def _true_column(kind: str = "prefix", documents: int = 100) -> dict[str, Any]:
    return {
        "postprocessor_class": kind,
        "prefix_ids": [128000] if kind == "prefix" else [],
        "documents": documents,
        "mismatches": 0,
        "served_by": (
            {"facade": 0, "reference": documents}
            if kind == "reference"
            else {"facade": documents, "reference": 0}
        ),
        "evidence_role": forms.EVIDENCE_ROLE[kind],
        "loader_true_equal": {
            "loader": {"package": "transformers", "version": "4.57.6"},
            "strings": 3000,
            "equal": 3000,
            "difference": "none",
        },
    }


@pytest.mark.parametrize("kind", ["identity", "prefix", "reference"])
def test_a_well_formed_column_passes(kind: str) -> None:
    assert forms.special_tokens_problems("f", _true_column(kind), documents=100) == []


def test_a_silent_fallback_to_the_reference_is_refused() -> None:
    column = _true_column("prefix")
    column["served_by"] = {"facade": 99, "reference": 1}
    problems = forms.special_tokens_problems("f", column)
    assert any("answered by the reference" in problem for problem in problems)
    column = _true_column("identity")
    column["served_by"] = {"facade": 0, "reference": 100}
    assert any(
        "compare the reference with itself" in problem
        for problem in forms.special_tokens_problems("f", column)
    )


@pytest.mark.parametrize("kind", ["identity", "prefix"])
def test_the_vacuity_guard_is_judged_on_the_facade_surface_only(kind: str) -> None:
    # The same reference-answered column: refused on the facade surface
    # (named, or by default when absent), reported on the native route.
    column = _true_column(kind)
    column["served_by"] = {"facade": 0, "reference": 100}
    for surface in (None, "facade"):
        if surface is not None:
            column["surface"] = surface
        assert any(
            "answered by the reference" in problem
            for problem in forms.special_tokens_problems("f", column)
        )
    column["surface"] = "native_route"
    assert forms.special_tokens_problems("f", column) == []
    column["evidence_role"] = "reference_route"
    assert forms.special_tokens_problems("f", column) == []
    notes = forms.special_tokens_notes("f", column)
    assert len(notes) == 1 and "native_route surface 100 document(s)" in notes[0]
    assert "judged on the facade surface only" in notes[0]
    # Off the facade surface, a column the facade answered keeps its role.
    served = _true_column(kind)
    served["surface"] = "native_route"
    assert forms.special_tokens_problems("f", served) == []
    assert forms.special_tokens_notes("f", served) == []
    served["evidence_role"] = "reference_route"
    assert any(
        "not 'reference_route'" in problem
        for problem in forms.special_tokens_problems("f", served)
    )
    # Mismatches count on every surface, and the surface names are closed.
    column["mismatches"] = 1
    assert any(
        "1 mismatch" in problem
        for problem in forms.special_tokens_problems("f", column)
    )
    column["mismatches"] = 0
    column["surface"] = "elsewhere"
    assert forms.special_tokens_problems("f", column)


def test_the_identity_column_is_named_a_regression_guard() -> None:
    column = _true_column("identity")
    column["evidence_role"] = "prefix_rule"
    problems = forms.special_tokens_problems("f", column)
    assert any(
        "'regression_guard'" in problem and "core stream with a core stream" in problem
        for problem in problems
    )


def test_counts_prefix_ids_and_mismatches_are_held() -> None:
    column = _true_column("prefix")
    column["prefix_ids"] = []
    assert any(
        "exactly one prefix id" in problem
        for problem in forms.special_tokens_problems("f", column)
    )
    column = _true_column("identity")
    column["prefix_ids"] = [1]
    assert any(
        "only a prefix face" in problem
        for problem in forms.special_tokens_problems("f", column)
    )
    column = _true_column("prefix")
    column["mismatches"] = 1
    assert any(
        "1 mismatch" in problem
        for problem in forms.special_tokens_problems("f", column)
    )
    column = _true_column("prefix")
    assert any(
        "the parity row judged 101" in problem
        for problem in forms.special_tokens_problems("f", column, documents=101)
    )
    column["served_by"] = {"facade": 98, "reference": 0}
    assert any(
        "does not add up" in problem
        for problem in forms.special_tokens_problems("f", column)
    )
    column = _true_column("reference")
    column["served_by"] = {"facade": 100, "reference": 0}
    assert any(
        "not served by the facade" in problem
        for problem in forms.special_tokens_problems("f", column)
    )


def test_the_schema_is_closed() -> None:
    column = _true_column("prefix")
    column["extra"] = 1
    assert any(
        "Additional properties" in problem
        for problem in forms.special_tokens_problems("f", column)
    )


def test_loader_true_equal_is_informative_not_a_gate() -> None:
    column = _true_column("identity")
    column["loader_true_equal"] = {
        "loader": {"package": "transformers", "version": "4.57.6"},
        "strings": 3265,
        "equal": 0,
        "difference": "fixed_affix",
        "loader_prefix_ids": [0],
        "loader_suffix_ids": [],
    }
    assert forms.special_tokens_problems("deepseek_v3", column) == []
    notes = forms.special_tokens_notes("deepseek_v3", column)
    assert len(notes) == 1 and "informative only" in notes[0]
    assert "prefix [0]" in notes[0]
    # Internal consistency is still checked.
    column["loader_true_equal"]["difference"] = "none"
    assert any(
        "says none" in problem
        for problem in forms.special_tokens_problems("deepseek_v3", column)
    )
    column["loader_true_equal"].update(difference="fixed_affix", loader_prefix_ids=[])
    assert any(
        "names no affix" in problem
        for problem in forms.special_tokens_problems("deepseek_v3", column)
    )
    del column["loader_true_equal"]
    assert forms.special_tokens_problems("deepseek_v3", column) == []
    assert forms.special_tokens_notes("deepseek_v3", column) == []


def _reading(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "toktier.fast_cpu.native_frontend_parity.v1", "rows": rows}


def test_parity_problems_require_and_check_columns() -> None:
    shipped = json.loads(
        (ROOT / "readings" / "fast_cpu_native_frontend_parity.json").read_text(
            encoding="utf-8"
        )
    )
    # The shipped readings carry no 0.3.2 column: they pass as they are and
    # fail only when the column is required.
    assert forms.parity_problems(shipped) == ([], [])
    problems, _notes = forms.parity_problems(shipped, ["special_tokens"])
    assert len(problems) == len(shipped["rows"])
    rows = copy.deepcopy(shipped["rows"])
    for row in rows:
        row["special_tokens"] = _true_column("identity", int(row["documents"]))
    assert forms.parity_problems(_reading(rows), ["special_tokens"]) == ([], [])
    rows[0]["special_tokens"]["served_by"] = {
        "facade": int(rows[0]["documents"]) - 1,
        "reference": 1,
    }
    problems, _notes = forms.parity_problems(_reading(rows))
    assert problems and rows[0]["family"] in problems[0]
    with pytest.raises(GenerationError, match="unknown column"):
        forms.parity_problems(shipped, ["no_such_column"])


def test_the_aggregator_carries_checked_columns() -> None:
    column = _true_column("prefix", 10)
    row: dict[str, Any] = {"family": "f", "documents": 10, "special_tokens": column}
    assert aggregate._columns(row, ()) == {"special_tokens": row["special_tokens"]}
    assert aggregate._columns({"family": "f", "documents": 10}, ()) == {}
    with pytest.raises(GenerationError, match="carries no special_tokens"):
        aggregate._columns({"family": "f", "documents": 10}, ("special_tokens",))
    column["served_by"] = {"facade": 9, "reference": 1}
    with pytest.raises(GenerationError, match="column did not pass"):
        aggregate._columns(row, ())


def test_the_command_line_reports(tmp_path: Path) -> None:
    column = _true_column("prefix", 5)
    rows = [{"family": "f", "documents": 5, "special_tokens": column}]
    path = tmp_path / "reading.json"
    path.write_text(json.dumps(_reading(rows)), encoding="utf-8")
    command = [sys.executable, str(TOOLS / "check_reading_forms.py"), "parity"]
    result = subprocess.run(
        [*command, str(path), "--require", "special_tokens"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    column["mismatches"] = 2
    path.write_text(json.dumps(_reading(rows)), encoding="utf-8")
    result = subprocess.run(
        [*command, str(path)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 1 and "2 mismatch(es)" in result.stderr


# -- literal_segments ---------------------------------------------------------


def _registry() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        (ROOT / "tables" / "support_registry.json").read_text(encoding="utf-8")
    )
    return document


def _segments_column(
    arm: str = "cpu", served: bool = True, fingerprint: str = "a" * 64
) -> dict[str, Any]:
    gpu_bytes = 600 if arm == "gpu" else 0
    entry: dict[str, Any] = {
        "entry_point": "id_route",
        "documents": 1000,
        "mismatches": 0,
        "paths": {
            "gigatoken_segmented": (990 if arm == "cpu" else 0) if served else 0,
            "gpu_segmented": (990 if arm == "gpu" else 0) if served else 0,
            "hf_added_token": 0 if served else 990,
            "hf_engine_guard": 10,
        },
        "segments": {
            "literals": 5000,
            "text": 6000,
            "largest_text_bytes": 70000,
            "bytes": {"gpu": gpu_bytes, "fast_cpu": 300, "hf": 100},
        },
        "guard_documents": {"count": 10, "bytes_hf": 100, "bound": 10},
    }
    if arm == "gpu":
        entry["gpu_share"] = 600 / 1000
        entry["gpu_min_bytes"] = 65536
    return {
        "literal_class": "served" if served else "unserved",
        "added_frontend_fingerprint": fingerprint,
        "battery": {"generator_sha256": "b" * 64, "strings": 120000, "mismatches": 0},
        "entry_points": [entry],
    }


@pytest.mark.parametrize("arm", ["cpu", "gpu"])
@pytest.mark.parametrize("served", [True, False])
def test_a_well_formed_segments_column_passes(arm: str, served: bool) -> None:
    column = _segments_column(arm, served)
    if not served:
        column["entry_points"][0]["guard_documents"]["bytes_hf"] = 0
        column["entry_points"][0]["paths"]["hf_engine_guard"] = 10
    assert forms.literal_segments_problems("f", column, arm=arm) == []


def test_reference_bytes_outside_the_guard_documents_are_refused() -> None:
    column = _segments_column("cpu")
    column["entry_points"][0]["segments"]["bytes"]["hf"] = 150
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("50 byte(s) outside the listed guard documents" in p for p in problems)


def test_a_whole_document_fallback_on_a_served_face_is_refused() -> None:
    column = _segments_column("cpu")
    paths = column["entry_points"][0]["paths"]
    paths["gigatoken_segmented"] -= 5
    paths["hf_added_token"] += 5
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("went whole to the reference for a literal" in p for p in problems)


def test_the_gpu_arm_needs_bytes_on_the_gpu_and_a_share() -> None:
    column = _segments_column("gpu")
    entry = column["entry_points"][0]
    entry["segments"]["bytes"]["gpu"] = 0
    entry["paths"]["gpu_segmented"] = 0
    entry["paths"]["gigatoken_segmented"] = 990
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("no segment reached the GPU" in p for p in problems)
    column = _segments_column("gpu")
    del column["entry_points"][0]["gpu_share"]
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("reports no gpu_share" in p for p in problems)
    column = _segments_column("gpu")
    column["entry_points"][0]["gpu_share"] = 0.5
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("gpu_share does not match" in p for p in problems)


def test_guard_documents_are_counted_and_bounded() -> None:
    column = _segments_column("cpu")
    column["entry_points"][0]["guard_documents"]["count"] = 9
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("9 guard document(s) listed, 10 counted" in p for p in problems)
    column = _segments_column("cpu")
    guard = column["entry_points"][0]["guard_documents"]
    guard["bound"] = 50
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("is above 1% of the documents" in p for p in problems)
    guard["bound"] = 9
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("exceed the stated bound 9" in p for p in problems)


def test_an_unserved_face_shows_no_segmented_document() -> None:
    column = _segments_column("cpu", served=False)
    paths = column["entry_points"][0]["paths"]
    paths["hf_added_token"] -= 1
    paths["gigatoken_segmented"] += 1
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("unserved face shows segmented documents" in p for p in problems)


def test_entry_points_counts_and_battery_are_held() -> None:
    column = _segments_column("cpu")
    column["entry_points"][0]["entry_point"] = "state_route"
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("the ID route is not measured" in p for p in problems)
    column = _segments_column("cpu")
    column["entry_points"].append(copy.deepcopy(column["entry_points"][0]))
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("listed twice" in p for p in problems)
    column = _segments_column("cpu")
    column["entry_points"][0]["paths"]["hf_engine_guard"] += 1
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("do not add up" in p for p in problems)
    column = _segments_column("cpu")
    column["battery"]["mismatches"] = 3
    column["entry_points"][0]["mismatches"] = 1
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("battery 3 mismatch" in p for p in problems)
    assert any("[id_route]: 1 mismatch" in p for p in problems)
    column = _segments_column("cpu")
    column["entry_points"][0]["segments"]["bytes"]["gpu"] = 5
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("CPU arm records GPU segments" in p for p in problems)


def test_the_column_is_held_to_the_records_added_token_table() -> None:
    registry = _registry()
    fingerprints = forms.registry_fingerprints(registry)
    family = "deepseek_v3"
    assert len(fingerprints[family]) == 64
    column = _segments_column("cpu", fingerprint=fingerprints[family])
    rows = [{"family": family, "documents": 1000, "literal_segments": column}]
    reading = {"schema": "toktier.fast_cpu.native_frontend_parity.v1", "rows": rows}
    assert forms.parity_problems(reading, ["literal_segments"], registry) == ([], [])
    column["added_frontend_fingerprint"] = "c" * 64
    problems, _notes = forms.parity_problems(reading, (), registry)
    assert problems and "the record names" in problems[0]
    # The arm comes from the reading's schema: the same column read as a
    # GPU reading has no GPU bytes.
    column["added_frontend_fingerprint"] = fingerprints[family]
    reading["schema"] = "toktier.gpu.native_frontend_parity.v1"
    problems, _notes = forms.parity_problems(reading, (), registry)
    assert any("no segment reached the GPU" in p for p in problems)


# -- release readings (S8) ----------------------------------------------------


def _complete_row(
    row: dict[str, Any], arm: str, fingerprints: dict[str, str]
) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out["special_tokens"] = _true_column("identity", int(row["documents"]))
    out["literal_segments"] = _segments_column(
        arm, fingerprint=fingerprints[str(row["family"])]
    )
    return out


def test_a_release_row_needs_both_columns_on_the_facade_surface() -> None:
    fingerprints = forms.registry_fingerprints(_registry())
    row = {"family": "deepseek_v3", "documents": 100}
    problems = forms.release_column_problems(row, arm="cpu")
    assert sum("a release reading requires it" in p for p in problems) == 2
    complete = _complete_row(row, "cpu", fingerprints)
    assert (
        forms.release_column_problems(
            complete,
            arm="cpu",
            added_frontend_fingerprint=fingerprints["deepseek_v3"],
        )
        == []
    )
    # A native-route column cannot speak for the prefix rule.
    complete["special_tokens"]["surface"] = "native_route"
    problems = forms.release_column_problems(complete, arm="cpu")
    assert any("taken on the native_route surface" in p for p in problems)


def test_require_special_tokens_is_not_met_by_a_native_route_column() -> None:
    column = _true_column("prefix", 5)
    column["surface"] = "native_route"
    rows = [{"family": "f", "documents": 5, "special_tokens": column}]
    problems, _notes = forms.parity_problems(_reading(rows), ["special_tokens"])
    assert any("taken on the native_route surface" in p for p in problems)
    # Without the requirement the column is checked as itself and passes.
    assert forms.parity_problems(_reading(rows)) == ([], [])


def test_the_release_check_requires_the_columns_on_certified_rows(
    tmp_path: Path,
) -> None:
    import generate_registry

    registry_path = ROOT / "tables" / "support_registry.json"
    # The readings in the tree predate the 0.3.2 columns: refused.
    shipped = generate_registry.release_reading_problems(registry_path)
    assert shipped and any(p.startswith("cpu/") for p in shipped)
    assert any(p.startswith("sm_120/") for p in shipped)
    fingerprints = forms.registry_fingerprints(_registry())
    cpu = json.loads(
        (ROOT / "readings" / "fast_cpu_native_frontend_parity.json").read_text(
            encoding="utf-8"
        )
    )
    cpu["rows"] = [_complete_row(row, "cpu", fingerprints) for row in cpu["rows"]]
    cpu_path = tmp_path / "cpu.json"
    cpu_path.write_text(json.dumps(cpu), encoding="utf-8")
    gpu_paths: dict[str, Path] = {}
    for architecture, source in generate_registry.PREBUILT_HARDWARE_READINGS.items():
        gpu = json.loads(source.read_text(encoding="utf-8"))
        gpu["rows"] = [_complete_row(row, "gpu", fingerprints) for row in gpu["rows"]]
        gpu_paths[architecture] = tmp_path / f"{architecture}.json"
        gpu_paths[architecture].write_text(json.dumps(gpu), encoding="utf-8")
    assert (
        generate_registry.release_reading_problems(
            registry_path, cpu_reading=cpu_path, gpu_readings=gpu_paths
        )
        == []
    )
    # One GPU row taken on the native route is refused, and named.
    gpu = json.loads(gpu_paths["sm_89"].read_text(encoding="utf-8"))
    gpu["rows"][0]["special_tokens"]["surface"] = "native_route"
    gpu_paths["sm_89"].write_text(json.dumps(gpu), encoding="utf-8")
    problems = generate_registry.release_reading_problems(
        registry_path, cpu_reading=cpu_path, gpu_readings=gpu_paths
    )
    assert len(problems) == 1 and problems[0].startswith("sm_89/")


# -- literal_class is held (S9) ------------------------------------------------


def test_a_self_declared_unserved_class_does_not_pass_where_served_is_expected(
    tmp_path: Path,
) -> None:
    column = _segments_column("cpu", served=False)
    column["entry_points"][0]["guard_documents"]["bytes_hf"] = 0
    # Taken as itself, an unserved column is consistent ...
    assert forms.literal_segments_problems("f", column, arm="cpu") == []
    # ... but a certified face is served in 0.3.2.
    problems = forms.literal_segments_problems(
        "f", column, arm="cpu", expected_literal_class="served"
    )
    assert any("the face is 'served'" in p for p in problems)
    fingerprints = forms.registry_fingerprints(_registry())
    row = _complete_row({"family": "qwen3_8b", "documents": 100}, "cpu", fingerprints)
    row["literal_segments"] = column
    column["added_frontend_fingerprint"] = fingerprints["qwen3_8b"]
    problems = forms.release_column_problems(row, arm="cpu")
    assert any("the face is 'served'" in p for p in problems)
    # The command line expects served by default; "any" takes the column.
    path = tmp_path / "reading.json"
    path.write_text(json.dumps(_reading([row])), encoding="utf-8")
    command = [sys.executable, str(TOOLS / "check_reading_forms.py"), "parity"]
    result = subprocess.run(
        [*command, str(path)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 1 and "the face is 'served'" in result.stderr
    result = subprocess.run(
        [*command, str(path), "--literal-class", "any"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
