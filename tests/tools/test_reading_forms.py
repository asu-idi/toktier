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
