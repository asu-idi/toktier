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
        "executed_by": (
            {"fast_cpu": 0, "gpu": 0, "hf": documents}
            if kind == "reference"
            else {"fast_cpu": documents, "gpu": 0, "hf": 0}
        ),
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
    gpu_bytes = 60000 if arm == "gpu" else 0
    entry: dict[str, Any] = {
        "entry_point": "id_route",
        "documents": 1000,
        "mismatches": 0,
        "paths": {
            "gigatoken_segmented": (990 if arm == "cpu" else 0) if served else 0,
            "gpu_segmented": (990 if arm == "gpu" else 0) if served else 0,
            "hf_added_token": 0 if served else 990,
            "hf_engine_guard": 10,
            "hf_segments_below_crossover": 0,
            "hf_span_guard": 0,
        },
        "segments": {
            "literals": 5000,
            "text": 6000,
            "largest_text_bytes": 70000,
            "bytes": {"gpu": gpu_bytes, "fast_cpu": 30000, "hf": 100},
        },
        "guard_documents": {"count": 10, "bytes_hf": 100, "bound": 10},
    }
    if arm == "gpu":
        entry["gpu_share"] = 60000 / 90100
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
    assert any(
        "50 byte(s) outside the listed guard, below-crossover and span-guard" in p
        for p in problems
    )


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


def _below_crossover(column: dict[str, Any], documents: int, size: int) -> None:
    """Move ``documents`` GPU-segmented documents of ``size`` bytes each to
    the below-crossover class: the reference encodes them whole."""
    entry = column["entry_points"][0]
    entry["paths"]["gpu_segmented"] -= documents
    entry["paths"]["hf_segments_below_crossover"] += documents
    entry["segments"]["bytes"]["gpu"] -= documents * size
    entry["segments"]["bytes"]["hf"] += documents * size
    entry["below_crossover_documents"] = {
        "count": documents,
        "bytes_hf": documents * size,
    }
    total = sum(entry["segments"]["bytes"].values())
    entry["gpu_share"] = entry["segments"]["bytes"]["gpu"] / total


def test_below_crossover_documents_are_their_own_path_class() -> None:
    column = _segments_column("gpu")
    _below_crossover(column, 40, 100)
    assert forms.literal_segments_problems("f", column, arm="gpu") == []
    # The served-face guard on hf_added_token keeps its meaning: the same
    # documents counted there are refused.
    paths = column["entry_points"][0]["paths"]
    paths["hf_added_token"] += 40
    paths["hf_segments_below_crossover"] -= 40
    column["entry_points"][0]["below_crossover_documents"]["count"] = 0
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("went whole to the reference for a literal" in p for p in problems)


def test_below_crossover_documents_are_listed_with_their_bytes() -> None:
    column = _segments_column("gpu")
    _below_crossover(column, 40, 100)
    del column["entry_points"][0]["below_crossover_documents"]
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("without below_crossover_documents" in p for p in problems)
    column = _segments_column("gpu")
    _below_crossover(column, 40, 100)
    column["entry_points"][0]["below_crossover_documents"]["count"] = 39
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("39 below-crossover document(s) listed, 40" in p for p in problems)
    column = _segments_column("gpu")
    _below_crossover(column, 40, 100)
    column["entry_points"][0]["below_crossover_documents"]["bytes_hf"] = 3000
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("1000 byte(s) outside the listed guard" in p for p in problems)


def test_below_crossover_documents_appear_only_on_a_served_gpu_arm() -> None:
    column = _segments_column("cpu")
    entry = column["entry_points"][0]
    entry["paths"]["gigatoken_segmented"] -= 5
    entry["paths"]["hf_segments_below_crossover"] = 5
    entry["segments"]["bytes"]["hf"] += 500
    entry["below_crossover_documents"] = {"count": 5, "bytes_hf": 500}
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("the CPU arm records GPU segments" in p for p in problems)
    column = _segments_column("gpu", served=False)
    entry = column["entry_points"][0]
    entry["guard_documents"]["bytes_hf"] = 0
    entry["paths"]["hf_added_token"] -= 5
    entry["paths"]["hf_segments_below_crossover"] = 5
    entry["below_crossover_documents"] = {"count": 5, "bytes_hf": 500}
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("unserved face shows segmented documents" in p for p in problems)


def test_the_below_crossover_class_is_a_required_key() -> None:
    column = _segments_column("gpu")
    del column["entry_points"][0]["paths"]["hf_segments_below_crossover"]
    assert forms.literal_segments_problems("f", column, arm="gpu")


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
    if arm == "gpu":
        out["special_tokens"]["executed_by"] = {
            "fast_cpu": 0,
            "gpu": int(row["documents"]),
            "hf": 0,
        }
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


# -- floors (S10) ---------------------------------------------------------------


def test_a_column_of_a_handful_of_documents_does_not_pass_the_ratios() -> None:
    column = _segments_column("cpu")
    entry = column["entry_points"][0]
    entry["documents"] = 2
    entry["paths"] = {
        "gigatoken_segmented": 2,
        "gpu_segmented": 0,
        "hf_added_token": 0,
        "hf_engine_guard": 0,
        "hf_segments_below_crossover": 0,
        "hf_span_guard": 0,
    }
    entry["guard_documents"] = {"count": 0, "bytes_hf": 0, "bound": 0}
    entry["segments"]["bytes"]["hf"] = 0
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("2 documents, fewer than 1000" in p for p in problems)
    column = _segments_column("cpu")
    column["battery"]["strings"] = 4
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("battery judged 4 strings" in p for p in problems)


def test_guard_documents_are_bounded_in_bytes_too() -> None:
    column = _segments_column("cpu")
    entry = column["entry_points"][0]
    entry["segments"]["bytes"]["hf"] = 20000
    entry["guard_documents"]["bytes_hf"] = 20000
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("guard documents hold 20000 of 50000 bytes" in p for p in problems)


def test_the_gpu_arm_holds_a_gpu_share_floor_on_the_id_route() -> None:
    column = _segments_column("gpu")
    entry = column["entry_points"][0]
    entry["segments"]["bytes"].update(gpu=1000, fast_cpu=89000)
    entry["gpu_share"] = 1000 / 90100
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("is below 0.5" in p for p in problems)
    # The state route is held to bytes on the GPU, not to the share floor.
    entry["entry_point"] = "state_route"
    column["entry_points"].insert(0, _segments_column("gpu")["entry_points"][0])
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert not any("is below 0.5" in p for p in problems)


def _riders_document() -> dict[str, Any]:
    return {
        "schema": "toktier.chunked_cuts_riders.v1",
        "tool_sha256": "c" * 64,
        "chunked_identity": "sha256:" + "d" * 64,
        "fast_cpu_source_digest_v2": "e" * 64,
        "rider_share": 0.05,
        "units": 3,
        "policy": "experimental",
        "riders": ["literal_segments", "special_tokens"],
        "families": {
            "qwen3_8b": {
                "special_tokens": {
                    "documents": 40,
                    "mismatches": 0,
                    "served_by": {"facade": 40},
                    "classes": {"identity": 40},
                    "vacuous": 0,
                    "executed_by": {"fast_cpu": 40},
                },
                "literal_segments": {
                    "documents": 40,
                    "mismatches": 0,
                    "literals": 120,
                    "paths": {"gigatoken_segmented": 40},
                    "bytes": {"gpu": 0, "fast_cpu": 9000, "hf": 0},
                    "segments_reported": 40,
                },
            }
        },
    }


def test_a_complete_rider_document_passes() -> None:
    assert forms.riders_problems(_riders_document()) == []
    assert forms.riders_problems(_riders_document(), ["special_tokens"]) == []


@pytest.mark.parametrize(
    ("change", "named"),
    [
        ("vacuous", "answered by the reference"),
        ("not_reported", "served_by not reported"),
        ("class_not_reported", "classes not reported"),
        ("path_not_reported", "path was not reported"),
        ("no_documents", "0 documents, fewer than 1"),
        ("arm_missing", "rider literal_segments: not run"),
        ("mismatch", "1 mismatch"),
    ],
)
def test_the_rider_document_refuses_what_would_read_as_zero_mismatches(
    change: str, named: str
) -> None:
    document = _riders_document()
    row = document["families"]["qwen3_8b"]
    special, segments = row["special_tokens"], row["literal_segments"]
    if change == "vacuous":
        special.update(vacuous=40, served_by={"reference": 40})
    elif change == "not_reported":
        special["served_by"] = {"facade": 39, "not_reported": 1}
    elif change == "class_not_reported":
        special["classes"] = {"identity": 39, "not_reported": 1}
    elif change == "path_not_reported":
        segments["paths"] = {"gigatoken_segmented": 39, "not_reported": 1}
    elif change == "no_documents":
        segments.update(documents=0, paths={}, literals=0, segments_reported=0)
    elif change == "arm_missing":
        del row["literal_segments"]
    else:
        special["mismatches"] = 1
    problems = forms.riders_problems(document)
    assert any(named in p for p in problems), problems


def test_the_rider_document_names_a_required_arm_the_campaign_did_not_run(
    tmp_path: Path,
) -> None:
    document = _riders_document()
    document["riders"] = ["special_tokens"]
    del document["families"]["qwen3_8b"]["literal_segments"]
    assert forms.riders_problems(document) == []
    problems = forms.riders_problems(document, ["literal_segments"])
    assert any("did not run ['literal_segments']" in p for p in problems)
    document["extra"] = 1
    assert any("Additional properties" in p for p in forms.riders_problems(document))
    path = tmp_path / "riders.json"
    path.write_text(json.dumps(_riders_document()), encoding="utf-8")
    command = [sys.executable, str(TOOLS / "check_reading_forms.py"), "riders"]
    result = subprocess.run(
        [*command, str(path), "--require", "special_tokens"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# -- codex B-01, B-04 -----------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_is_refused_before_any_rule(value: float) -> None:
    column = _segments_column("gpu")
    column["entry_points"][0]["gpu_share"] = value
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("gpu_share: not a finite number" in p for p in problems), problems
    # The token json.loads accepts reaches the same rule.
    parsed = json.loads(json.dumps(column))
    assert forms.literal_segments_problems("f", parsed, arm="gpu")
    document = _riders_document()
    document["rider_share"] = value
    assert any("not a finite number" in p for p in forms.riders_problems(document))


@pytest.mark.parametrize("documents", [2, 3, 50, 99])
def test_one_guard_document_is_above_one_percent_of_few_documents(
    documents: int,
) -> None:
    column = _segments_column("cpu")
    entry = column["entry_points"][0]
    entry["documents"] = documents
    entry["paths"] = {
        "gigatoken_segmented": documents - 1,
        "gpu_segmented": 0,
        "hf_added_token": 0,
        "hf_engine_guard": 1,
        "hf_segments_below_crossover": 0,
        "hf_span_guard": 0,
    }
    entry["guard_documents"] = {"count": 1, "bytes_hf": 100, "bound": 1}
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("is above 1% of the documents" in p for p in problems), problems


def test_the_guard_bound_is_one_percent_of_the_documents() -> None:
    column = _segments_column("cpu")
    guard = column["entry_points"][0]["guard_documents"]
    guard["bound"] = 10
    assert forms.literal_segments_problems("f", column, arm="cpu") == []
    guard["bound"] = 11
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("is above 1% of the documents" in p for p in problems)


def _span_guard(column: dict[str, Any], documents: int, size: int) -> None:
    """Make the column's entry the state route and move ``documents``
    GPU-segmented documents of ``size`` bytes each to the span guard: the
    device row was refused and the reference re-encoded them whole."""
    entry = column["entry_points"][0]
    entry["entry_point"] = "state_route"
    entry["paths"]["gpu_segmented"] -= documents
    entry["paths"]["hf_span_guard"] += documents
    entry["segments"]["bytes"]["gpu"] -= documents * size
    entry["segments"]["bytes"]["hf"] += documents * size
    entry["span_guard_documents"] = {"count": documents, "bytes_hf": documents * size}
    total = sum(entry["segments"]["bytes"].values())
    entry["gpu_share"] = entry["segments"]["bytes"]["gpu"] / total
    id_route = copy.deepcopy(_segments_column("gpu")["entry_points"][0])
    column["entry_points"].insert(0, id_route)


def test_span_guard_documents_are_their_own_path_class() -> None:
    column = _segments_column("gpu")
    _span_guard(column, 40, 100)
    assert forms.literal_segments_problems("f", column, arm="gpu") == []
    # Counted under the engine guard instead, the same documents break the
    # guard listing (and the guard bound): the two are kept apart.
    entry = column["entry_points"][1]
    entry["paths"]["hf_engine_guard"] += 40
    entry["paths"]["hf_span_guard"] -= 40
    del entry["span_guard_documents"]
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("counted as hf_engine_guard" in p for p in problems), problems


def test_span_guard_documents_are_listed_with_their_bytes() -> None:
    column = _segments_column("gpu")
    _span_guard(column, 40, 100)
    del column["entry_points"][1]["span_guard_documents"]
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("hf_span_guard without span_guard_documents" in p for p in problems)
    column = _segments_column("gpu")
    _span_guard(column, 40, 100)
    column["entry_points"][1]["span_guard_documents"]["count"] = 39
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("39 span-guard document(s) listed, 40" in p for p in problems)
    column = _segments_column("gpu")
    _span_guard(column, 40, 100)
    column["entry_points"][1]["span_guard_documents"]["bytes_hf"] = 3000
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("1000 byte(s) outside the listed guard" in p for p in problems)


def test_span_guard_documents_appear_only_on_a_served_gpu_state_route() -> None:
    # On the ID route
    column = _segments_column("gpu")
    _span_guard(column, 40, 100)
    column["entry_points"] = column["entry_points"][1:]
    column["entry_points"][0]["entry_point"] = "id_route"
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("hf_span_guard outside the state route" in p for p in problems)
    # In the CPU arm
    column = _segments_column("cpu")
    entry = copy.deepcopy(column["entry_points"][0])
    entry["entry_point"] = "state_route"
    entry["paths"]["gigatoken_segmented"] -= 5
    entry["paths"]["hf_span_guard"] = 5
    entry["segments"]["bytes"]["hf"] += 500
    entry["span_guard_documents"] = {"count": 5, "bytes_hf": 500}
    column["entry_points"].append(entry)
    problems = forms.literal_segments_problems("f", column, arm="cpu")
    assert any("the CPU arm records GPU segments" in p for p in problems)
    # On an unserved face
    column = _segments_column("gpu", served=False)
    entry = column["entry_points"][0]
    entry["entry_point"] = "state_route"
    entry["guard_documents"]["bytes_hf"] = 0
    entry["paths"]["hf_added_token"] -= 5
    entry["paths"]["hf_span_guard"] = 5
    entry["span_guard_documents"] = {"count": 5, "bytes_hf": 500}
    id_route = copy.deepcopy(_segments_column("gpu", served=False)["entry_points"][0])
    column["entry_points"].insert(0, id_route)
    column["entry_points"][0]["guard_documents"]["bytes_hf"] = 0
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("unserved face shows segmented documents" in p for p in problems)


def test_the_span_guard_class_is_a_required_key() -> None:
    column = _segments_column("gpu")
    del column["entry_points"][0]["paths"]["hf_span_guard"]
    assert forms.literal_segments_problems("f", column, arm="gpu")


# -- device-declined documents (the reference answers the whole document) ----


def _device_declined(
    column: dict[str, Any],
    documents: int,
    size: int,
    *,
    entry_index: int = 0,
    chain: list[str] | None = None,
) -> dict[str, Any]:
    """Move ``documents`` GPU-segmented documents of ``size`` bytes each of
    one entry point to the device-declined class: the device declined a
    text segment (R_EXEC_FAULT, scope segment) and the reference, next in
    the chain, encoded the whole document. The entry's engine-guard
    documents are moved back to the GPU first, so the shared bound has
    room."""
    entry: dict[str, Any] = column["entry_points"][entry_index]
    entry["paths"]["gpu_segmented"] += entry["paths"]["hf_engine_guard"]
    entry["segments"]["bytes"]["hf"] -= entry["guard_documents"]["bytes_hf"]
    entry["segments"]["bytes"]["gpu"] += entry["guard_documents"]["bytes_hf"]
    entry["paths"]["hf_engine_guard"] = 0
    entry["guard_documents"] = {"count": 0, "bytes_hf": 0, "bound": 10}
    entry["paths"]["gpu_segmented"] -= documents
    entry["paths"]["hf_device_declined"] = documents
    entry["segments"]["bytes"]["gpu"] -= documents * size
    entry["segments"]["bytes"]["hf"] += documents * size
    entry["device_declined_documents"] = {
        "count": documents,
        "whole_documents": documents,
        "bytes_hf": documents * size,
        "chain": chain if chain is not None else ["gpu", "hf"],
        "event": {"code": "R_EXEC_FAULT", "backend": "gpu", "scope": "segment"},
    }
    total = sum(entry["segments"]["bytes"].values())
    entry["gpu_share"] = entry["segments"]["bytes"]["gpu"] / total
    return entry


def _both_routes_gpu() -> dict[str, Any]:
    column = _segments_column("gpu")
    state = copy.deepcopy(column["entry_points"][0])
    state["entry_point"] = "state_route"
    column["entry_points"].append(state)
    return column


def test_device_declined_documents_are_their_own_path_class() -> None:
    column = _both_routes_gpu()
    _device_declined(column, 3, 100, entry_index=0)
    _device_declined(column, 3, 100, entry_index=1)
    assert forms.literal_segments_problems("f", column, arm="gpu") == []
    # Counted under hf_added_token instead, the same documents are refused on
    # a served face: the class keeps that guard's meaning.
    entry = column["entry_points"][0]
    entry["paths"]["hf_added_token"] += 3
    entry["paths"]["hf_device_declined"] -= 3
    del entry["device_declined_documents"]
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("went whole to the reference for a literal" in p for p in problems)


def test_the_device_declined_class_is_optional_and_reads_zero_when_absent() -> None:
    # Readings taken before the class existed carry no key: they read 0.
    column = _segments_column("gpu")
    assert "hf_device_declined" not in column["entry_points"][0]["paths"]
    assert forms.literal_segments_problems("f", column, arm="gpu") == []
    column["entry_points"][0]["paths"]["hf_device_declined"] = 0
    assert forms.literal_segments_problems("f", column, arm="gpu") == []


def test_device_declined_documents_are_listed_with_their_bytes() -> None:
    column = _segments_column("gpu")
    _device_declined(column, 3, 100)
    del column["entry_points"][0]["device_declined_documents"]
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any(
        "hf_device_declined without device_declined_documents" in p for p in problems
    ), problems
    column = _segments_column("gpu")
    _device_declined(column, 3, 100)["device_declined_documents"]["count"] = 2
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("2 device-declined document(s) listed, 3" in p for p in problems)
    column = _segments_column("gpu")
    _device_declined(column, 3, 100)["device_declined_documents"]["bytes_hf"] = 200
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("100 byte(s) outside the listed guard" in p for p in problems)


def test_device_declined_documents_share_the_guard_bounds() -> None:
    # 11 documents of 1,000: above floor(1 %) = 10.
    column = _segments_column("gpu")
    _device_declined(column, 11, 10)
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("exceed the bound 10" in p for p in problems), problems
    # Together with engine-guard documents: 5 + 6 > 10.
    column = _segments_column("gpu")
    entry = _device_declined(column, 5, 10)
    entry["paths"]["gpu_segmented"] -= 6
    entry["paths"]["hf_engine_guard"] = 6
    entry["guard_documents"] = {"count": 6, "bytes_hf": 60, "bound": 10}
    entry["segments"]["bytes"]["gpu"] -= 60
    entry["segments"]["bytes"]["hf"] += 60
    entry["gpu_share"] = entry["segments"]["bytes"]["gpu"] / sum(
        entry["segments"]["bytes"].values()
    )
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("exceed the bound 10" in p for p in problems), problems
    # Bytes: 3 documents of 400 bytes hold 1,200 of 90,100 bytes, above 1 %.
    column = _segments_column("gpu")
    _device_declined(column, 3, 400)
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any(
        "guard and device-declined documents hold 1200 of 90100 bytes" in p
        for p in problems
    ), problems


def test_device_declined_documents_appear_only_where_the_device_hands_to_the_reference(
) -> None:
    # A chain with a CPU engine between the device and the reference: a
    # declined segment would go to the CPU engine, not whole to the reference.
    column = _segments_column("gpu")
    _device_declined(column, 3, 100, chain=["gpu", "fast_cpu", "hf"])
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("on the chain ['gpu', 'fast_cpu', 'hf']" in p for p in problems), (
        problems
    )
    # The record has to show the device's own decline.
    column = _segments_column("gpu")
    _device_declined(column, 3, 100)["device_declined_documents"]["event"] = {
        "code": "R_INPUT_ADDED_TOKEN",
        "backend": "gpu",
        "scope": "segment",
    }
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("the device's own decline" in p for p in problems), problems
    # Not in the CPU arm, nor where the arm is not named.
    column = _segments_column("cpu")
    entry = column["entry_points"][0]
    entry["paths"]["gigatoken_segmented"] -= 3
    entry["paths"]["hf_device_declined"] = 3
    entry["segments"]["bytes"]["hf"] += 300
    entry["device_declined_documents"] = {
        "count": 3,
        "whole_documents": 3,
        "bytes_hf": 300,
        "chain": ["gpu", "hf"],
        "event": {"code": "R_EXEC_FAULT", "backend": "gpu", "scope": "segment"},
    }
    for arm in ("cpu", None):
        problems = forms.literal_segments_problems("f", column, arm=arm)
        assert any("outside the GPU arm" in p for p in problems), (arm, problems)
    # Not on an unserved face.
    column = _segments_column("gpu", served=False)
    entry = column["entry_points"][0]
    entry["guard_documents"]["bytes_hf"] = 0
    entry["paths"]["hf_added_token"] -= 3
    entry["paths"]["hf_device_declined"] = 3
    entry["device_declined_documents"] = {
        "count": 3,
        "whole_documents": 3,
        "bytes_hf": 300,
        "chain": ["gpu", "hf"],
        "event": {"code": "R_EXEC_FAULT", "backend": "gpu", "scope": "segment"},
    }
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("unserved face shows segmented documents" in p for p in problems)


def test_device_declined_documents_may_keep_their_other_segments_on_the_gpu() -> None:
    """A document of several text segments of which the device declined
    some: the others ran on the GPU, the reference encoded the declined
    ones. It counts as device-declined (not gpu_segmented), with only the
    declined segments' bytes on the reference."""
    column = _segments_column("gpu")
    entry = _device_declined(column, 3, 100)
    # Two of the three ran their other segments on the GPU.
    entry["device_declined_documents"]["whole_documents"] = 1
    assert forms.literal_segments_problems("f", column, arm="gpu") == []
    entry["device_declined_documents"]["whole_documents"] = 4
    problems = forms.literal_segments_problems("f", column, arm="gpu")
    assert any("answered whole, more than the 3 listed" in p for p in problems)
    del entry["device_declined_documents"]["whole_documents"]
    assert forms.literal_segments_problems("f", column, arm="gpu")


def test_the_aggregator_checks_the_column_in_its_arm() -> None:
    column = _segments_column("gpu")
    _device_declined(column, 3, 100)
    row: dict[str, Any] = {"family": "f", "documents": 1000, "literal_segments": column}
    assert aggregate._columns(row, ("literal_segments",), arm="gpu") == {
        "literal_segments": column
    }
    with pytest.raises(GenerationError, match="outside the GPU arm"):
        aggregate._columns(row, ("literal_segments",), arm="cpu")


def test_a_gpu_reading_with_device_declined_documents_passes_the_command_line(
    tmp_path: Path,
) -> None:
    column = _both_routes_gpu()
    _device_declined(column, 3, 100, entry_index=0)
    _device_declined(column, 3, 100, entry_index=1)
    reading = {
        "schema": "toktier.gpu.native_frontend_parity.v1",
        "rows": [{"family": "f", "documents": 1000, "literal_segments": column}],
    }
    path = tmp_path / "reading.json"
    path.write_text(json.dumps(reading), encoding="utf-8")
    command = [sys.executable, str(TOOLS / "check_reading_forms.py"), "parity"]
    result = subprocess.run(
        [*command, str(path), "--require", "literal_segments"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    column["entry_points"][1]["device_declined_documents"]["chain"] = [
        "gpu",
        "fast_cpu",
        "hf",
    ]
    path.write_text(json.dumps(reading), encoding="utf-8")
    result = subprocess.run(
        [*command, str(path), "--require", "literal_segments"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "fast_cpu" in result.stderr


# -- special_tokens: which engine produced the core streams ------------------


@pytest.mark.parametrize("kind", ["identity", "prefix"])
def test_a_column_answered_by_the_reference_engine_is_refused(kind: str) -> None:
    """Lane RCP's case: the facade served every document (served_by), but
    every core stream came from the reference (a build the registry does not
    bind yet, under the default policy)."""
    column = _true_column(kind)
    column["executed_by"] = {"fast_cpu": 0, "gpu": 0, "hf": 100}
    for arm in ("cpu", "gpu", None):
        problems = forms.special_tokens_problems("f", column, documents=100, arm=arm)
        assert any("below 50%" in p for p in problems), problems
    # ... and through the row form a parity reading uses.
    row = {"family": "f", "documents": 100, "special_tokens": column}
    assert any("below 50%" in p for p in forms.column_problems(row, arm="cpu"))


def test_the_arms_engine_carries_the_column() -> None:
    column = _true_column("prefix")
    column["executed_by"] = {"fast_cpu": 51, "gpu": 0, "hf": 49}
    assert forms.special_tokens_problems("f", column, arm="cpu") == []
    problems = forms.special_tokens_problems("f", column, arm="gpu")
    assert any("came from gpu" in p for p in problems)
    column["executed_by"] = {"fast_cpu": 49, "gpu": 0, "hf": 51}
    assert any(
        "49 of 100 core stream(s) came from fast_cpu" in p
        for p in forms.special_tokens_problems("f", column, arm="cpu")
    )
    # The GPU arm: short documents below the crossover may run on the CPU
    # engine; the GPU has to carry at least half.
    column["executed_by"] = {"fast_cpu": 40, "gpu": 60, "hf": 0}
    assert forms.special_tokens_problems("f", column, arm="gpu") == []
    assert forms.special_tokens_problems("f", column) == []


def test_the_engine_record_is_required_and_not_empty() -> None:
    column = _true_column("identity")
    del column["executed_by"]
    assert any("executed_by" in p for p in forms.special_tokens_problems("f", column))
    column["executed_by"] = {"fast_cpu": 0, "gpu": 0, "hf": 0}
    assert any(
        "records no execution" in p for p in forms.special_tokens_problems("f", column)
    )


def test_the_engine_floor_does_not_apply_to_reference_routes() -> None:
    # A reference face is served by the reference engine by design ...
    assert (
        forms.special_tokens_problems("f", _true_column("reference"), arm="cpu") == []
    )
    # ... and a native-route column is a reference-route reading.
    column = _true_column("identity")
    column["surface"] = "native_route"
    column["served_by"] = {"facade": 0, "reference": 100}
    column["evidence_role"] = "reference_route"
    column["executed_by"] = {"fast_cpu": 0, "gpu": 0, "hf": 100}
    assert forms.special_tokens_problems("f", column, arm="cpu") == []


# -- rider document: engine record, reference paths, policy -------------------


def test_the_rider_document_names_its_policy() -> None:
    document = _riders_document()
    del document["policy"]
    assert any("policy" in p for p in forms.riders_problems(document))
    document["policy"] = None
    assert forms.riders_problems(document) == []


def test_a_rider_true_arm_made_by_the_reference_engine_is_refused() -> None:
    """served_by reads facade on every document while every core stream came
    from the reference (lane RCP's default-policy case on an unbound build)."""
    document = _riders_document()
    arm = document["families"]["qwen3_8b"]["special_tokens"]
    arm["executed_by"] = {"hf": 40}
    problems = forms.riders_problems(document)
    assert any("0 of 40 identity or prefix document(s)" in p for p in problems)
    arm["executed_by"] = {"fast_cpu": 20, "hf": 20}
    assert forms.riders_problems(document) == []
    arm["executed_by"] = {"fast_cpu": 19, "hf": 20, "not_reported": 1}
    problems = forms.riders_problems(document)
    assert any("below 50%" in p for p in problems)
    assert any("executed_by not reported" in p for p in problems)
    # A reference face is answered by the reference by design.
    arm["classes"] = {"reference": 40}
    arm["served_by"] = {"reference": 40}
    arm["executed_by"] = {"hf": 40}
    assert forms.riders_problems(document) == []


def test_the_rider_literal_arm_bounds_its_reference_paths() -> None:
    document = _riders_document()
    arm = document["families"]["qwen3_8b"]["literal_segments"]
    arm["documents"] = 1000
    arm["paths"] = {
        "gigatoken_segmented": 990,
        "hf_engine_guard": 5,
        "hf_span_guard": 5,
    }
    assert forms.riders_problems(document) == []
    # The span guard is counted under its own name, and with the engine
    # guard bounded to 1 % of the documents.
    arm["paths"] = {
        "gigatoken_segmented": 988,
        "hf_engine_guard": 6,
        "hf_span_guard": 6,
    }
    problems = forms.riders_problems(document)
    assert any("12 document(s) on reference paths" in p for p in problems)
    assert any("'hf_span_guard': 6" in p for p in problems)
    # No document goes whole to the reference for a literal.
    arm["paths"] = {"gigatoken_segmented": 999, "hf_added_token": 1}
    problems = forms.riders_problems(document)
    assert any("went whole to the reference for a literal" in p for p in problems)
    # Reference bytes, and a literal arm that segmented nothing.
    arm["paths"] = {"gigatoken_segmented": 1000}
    arm["bytes"] = {"gpu": 0, "fast_cpu": 9000, "hf": 100}
    assert any("above 1%" in p for p in forms.riders_problems(document))
    arm["bytes"] = {"gpu": 0, "fast_cpu": 0, "hf": 0}
    arm["paths"] = {"gigatoken_full": 1000}
    problems = forms.riders_problems(document)
    assert any("no document was segmented" in p for p in problems)
