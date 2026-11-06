"""The family roster of the Rust API matrix that records a runtime build.

Contract reference: ``docs/contracts/registry.md`` Section 1.1 (the
``runtime_builds`` block and the matrix behind it) and Section 3.0 (examined
artifacts). The matrix is certification evidence, so its roster is the
registry's certified artifact records: the shipped readings answer for
exactly those families, a reading that adds an examined artifact or drops a
certified one is refused, and no examined artifact is ever a certified
record.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for entry in (str(ROOT), str(TOOLS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import update_rust_api_registry as tool  # noqa: E402
from registry_common import GenerationError  # noqa: E402

REGISTRY = ROOT / "tables/support_registry.json"
EXAMINED = ROOT / "src/toktier/routing/tables/examined_artifacts.v1.json"


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _roster() -> set[str]:
    return tool.certified_roster(_read(REGISTRY))


def _readings() -> dict[str, dict[str, Any]]:
    return {target: _read(path) for target, path in tool.MATRIX_READINGS.items()}


def _binding(readings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The identity tuple the readings themselves carry, so that only the
    roster decides the outcome of the checks below."""
    cpu = readings["cpu"]
    return {
        "source_digest": cpu["runtime_source_digest"],
        "fast_cpu_source_digest": cpu["fast_cpu_source_digest"],
        "native_host_source_digest": cpu["native_host_source_digest"],
        "toolchain": cpu["toolchain"],
        "build_flags": cpu["build_flags"],
    }


def _install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    readings: dict[str, dict[str, Any]],
) -> None:
    paths = {}
    for target, reading in readings.items():
        path = tmp_path / f"rust_api_matrix_{target}.json"
        path.write_text(json.dumps(reading), encoding="utf-8")
        paths[target] = path
    monkeypatch.setattr(tool, "MATRIX_READINGS", paths)


def test_roster_is_the_certified_records_and_excludes_examined() -> None:
    roster = _roster()
    examined = {str(row["family"]) for row in _read(EXAMINED)["artifacts"]}
    assert len(roster) == tool.MATRIX_FAMILIES
    assert not roster & examined


def test_shipped_readings_answer_for_the_certified_roster() -> None:
    roster = _roster()
    for target, reading in _readings().items():
        families = [str(row["family"]) for row in reading["rows"]]
        assert sorted(families) == sorted(roster), target


def test_shipped_readings_pass_the_roster_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    readings = _readings()
    _install(monkeypatch, tmp_path, readings)
    tool.verify_public_matrix(_binding(readings), bootstrap=False, roster=_roster())


def test_an_examined_row_in_place_of_a_certified_one_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    readings = copy.deepcopy(_readings())
    # The shipped examined table may be empty; any family outside the
    # certified roster stands in for an examined one.
    rows = _read(EXAMINED)["artifacts"]
    examined = str(rows[0]["family"]) if rows else "examined_fixture"
    readings["cpu"]["rows"][0]["family"] = examined
    _install(monkeypatch, tmp_path, readings)
    with pytest.raises(GenerationError, match="certified artifact roster"):
        tool.verify_public_matrix(
            _binding(readings), bootstrap=False, roster=_roster()
        )


def test_a_repeated_row_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    readings = copy.deepcopy(_readings())
    rows = readings["sm_89"]["rows"]
    rows[1]["family"] = rows[0]["family"]
    _install(monkeypatch, tmp_path, readings)
    with pytest.raises(GenerationError, match="certified artifact roster"):
        tool.verify_public_matrix(
            _binding(readings), bootstrap=False, roster=_roster()
        )


def test_a_roster_of_another_size_is_refused() -> None:
    roster = _roster()
    with pytest.raises(
        GenerationError, match=f"certifies {tool.MATRIX_FAMILIES - 1} artifacts"
    ):
        tool.verify_public_matrix(
            _binding(_readings()), bootstrap=False, roster=set(sorted(roster)[1:])
        )
