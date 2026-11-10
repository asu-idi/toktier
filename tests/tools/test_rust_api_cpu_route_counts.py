"""The Rust API tool derives its CPU counts from the fast CPU binding.

``tools/update_rust_api_registry.py`` checks two things that depend on how
many families run on the integrated CPU engine: the CPU native-front-end
parity reading's ``unique_artifacts`` and ``model_families``, and the
backend each family answers through in the CPU venue's Rust API matrix.
Neither is written beside the tool. Both follow from
``tools/fast_cpu_binding.json``: the loadable families, the families
carried from one of them, and the directly judged families whose
engine-differential reading is recorded and present. A directly judged
family whose reading is still pending answers through the reference
engine. A reading whose counts or backends disagree with that derivation
is refused.
"""

from __future__ import annotations

import copy
import hashlib
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
from compute_identity_v2 import source_digest as source_digest_v2  # noqa: E402
from registry_common import GenerationError  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _fast_binding() -> dict[str, Any]:
    return _read(ROOT / "tools/fast_cpu_binding.json")


def _roster() -> set[str]:
    return tool.certified_roster(_read(ROOT / "tables/support_registry.json"))


def _matrix_readings() -> dict[str, dict[str, Any]]:
    return {
        target: _read(ROOT / "readings" / f"rust_api_matrix_{suffix}.json")
        for target, suffix in (("cpu", "cpu"), ("sm_89", "sm89"), ("sm_120", "sm120"))
    }


def _matrix_binding(readings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The identity tuple the readings carry, so that only the backend
    roster decides the outcome."""
    cpu = readings["cpu"]
    return {
        "source_digest": cpu["runtime_source_digest"],
        "fast_cpu_source_digest": cpu["fast_cpu_source_digest"],
        "native_host_source_digest": cpu["native_host_source_digest"],
        "toolchain": cpu["toolchain"],
        "build_flags": cpu["build_flags"],
    }


def _four_face_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Any], list[str]]:
    """A scratch tree where every directly judged family has its reading.

    Each directly judged family gets a reading file under ``readings/``
    and its digest in a copy of the binding; the tool's root and binding
    path point at the scratch copy.
    """
    binding = copy.deepcopy(_fast_binding())
    direct = sorted(binding["direct_families"])
    assert direct, "the binding lists no directly judged family"
    for family in direct:
        node = binding["direct_families"][family]
        path = tmp_path / node["reading"]
        _write(path, {"family": family, "fixture": True})
        node["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    binding_path = _write(tmp_path / "tools/fast_cpu_binding.json", binding)
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "FAST_CPU_BINDING", binding_path, raising=False)
    return binding, direct


def _pending_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Any], list[str]]:
    """A scratch tree where every directly judged family is still pending.

    A copy of the binding records no digest for any directly judged family
    (``"sha256": null``), the shape a face has while its reading is being
    taken; the tool's root and binding path point at the scratch copy.
    """
    binding = copy.deepcopy(_fast_binding())
    direct = sorted(binding["direct_families"])
    assert direct, "the binding lists no directly judged family"
    for family in direct:
        binding["direct_families"][family]["sha256"] = None
    binding_path = _write(tmp_path / "tools/fast_cpu_binding.json", binding)
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "FAST_CPU_BINDING", binding_path, raising=False)
    return binding, direct


def _loadable_reading(binding: dict[str, Any]) -> dict[str, Any]:
    """The shipped CPU parity reading cut to the loadable families' rows.

    This is the reading as it reads while every directly judged face is
    pending: one row per loadable family and the coverage counts of the
    judged campaign.
    """
    reading = copy.deepcopy(
        _read(ROOT / "readings/fast_cpu_native_frontend_parity.json")
    )
    loadable = set(binding["loadable_families"])
    reading["rows"] = [row for row in reading["rows"] if row["family"] in loadable]
    assert {row["family"] for row in reading["rows"]} == loadable
    reading["unique_artifacts"] = len(reading["rows"])
    reading["model_families"] = binding["coverage"]["model_families"]
    return reading


def _cpu_reading(binding: dict[str, Any], direct: list[str]) -> dict[str, Any]:
    """The loadable-only CPU parity reading extended by one row per direct
    face."""
    reading = _loadable_reading(binding)
    template = reading["rows"][0]
    for family in direct:
        row = copy.deepcopy(template)
        row["family"] = family
        reading["rows"].append(row)
    reading["unique_artifacts"] = len(reading["rows"])
    reading["model_families"] = binding["coverage"]["model_families"] + len(direct)
    return reading


def _pending_matrix(direct: list[str]) -> dict[str, dict[str, Any]]:
    """The shipped matrix readings with the directly judged faces answering
    through the reference engine at the CPU venue."""
    readings = copy.deepcopy(_matrix_readings())
    for row in readings["cpu"]["rows"]:
        if row["family"] in direct:
            row["backend"] = "HuggingFace"
    return readings


def _install_cpu_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reading: dict[str, Any]
) -> dict[str, Any]:
    # The identity checks are not what this test is about either: the
    # reading names the tree's own v2 so that only the counts decide.
    reading = copy.deepcopy(reading)
    reading["engine"]["source_digest_v2"] = source_digest_v2("fast_cpu")
    path = _write(tmp_path / "readings/fast_cpu_native_frontend_parity.json", reading)
    monkeypatch.setattr(tool, "CPU_READING", path)
    # The GPU readings are not what this test is about.
    monkeypatch.setattr(tool, "GPU_READINGS", {})
    return {"fast_cpu_source_digest": reading["engine"]["source_digest"]}


def _install_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    readings: dict[str, dict[str, Any]],
) -> None:
    paths = {
        target: _write(tmp_path / f"matrix/rust_api_matrix_{target}.json", reading)
        for target, reading in readings.items()
    }
    monkeypatch.setattr(tool, "MATRIX_READINGS", paths)


def _counts(backends: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for backend in backends.values():
        counts[backend] = counts.get(backend, 0) + 1
    return counts


# The recorded binding: every directly judged face has its reading.


def test_the_recorded_binding_puts_every_direct_face_on_the_route() -> None:
    binding = _fast_binding()
    route = tool.fast_cpu_route(binding)
    assert route["pending"] == set()
    assert route["direct"] == set(binding["direct_families"])
    expected = route["loadable"] | route["carried"] | route["direct"]
    backends = tool.cpu_matrix_backends(binding, _roster())
    assert {f for f, b in backends.items() if b == "FastCpu"} == expected
    assert _counts(backends) == {
        "FastCpu": len(expected),
        "HuggingFace": len(_roster()) - len(expected),
    }
    shipped_matrix = {
        row["family"]: row["backend"] for row in _matrix_readings()["cpu"]["rows"]
    }
    assert backends == shipped_matrix
    counts = tool.cpu_parity_counts(binding)
    assert counts["rows"] == route["loadable"] | route["direct"]
    assert counts["unique_artifacts"] == len(route["loadable"]) + len(route["direct"])
    shipped = _read(ROOT / "readings/fast_cpu_native_frontend_parity.json")
    assert {row["family"] for row in shipped["rows"]} == counts["rows"]
    assert shipped["unique_artifacts"] == counts["unique_artifacts"]
    assert shipped["model_families"] == counts["model_families"]


# A pending direct face stays on the reference engine.


def test_a_pending_direct_face_stays_on_the_reference_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding, direct = _pending_tree(monkeypatch, tmp_path)
    route = tool.fast_cpu_route(binding)
    assert route["pending"] == set(direct)
    assert route["direct"] == set()
    expected = route["loadable"] | route["carried"]
    backends = tool.cpu_matrix_backends(binding, _roster())
    assert {f for f, b in backends.items() if b == "FastCpu"} == expected
    assert _counts(backends) == {
        "FastCpu": len(expected),
        "HuggingFace": len(_roster()) - len(expected),
    }
    counts = tool.cpu_parity_counts(binding)
    assert counts["rows"] == route["loadable"] | route["direct"]
    assert counts["unique_artifacts"] == len(route["loadable"]) + len(route["direct"])


def test_readings_for_pending_faces_answer_through_the_reference_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding, direct = _pending_tree(monkeypatch, tmp_path)
    readings = _pending_matrix(direct)
    _install_matrix(monkeypatch, tmp_path, readings)
    tool.verify_public_matrix(
        _matrix_binding(readings), bootstrap=False, roster=_roster()
    )
    tool.verify_shared_evidence(
        _install_cpu_reading(monkeypatch, tmp_path, _loadable_reading(binding))
    )


def test_the_shipped_readings_are_refused_while_the_faces_are_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _binding, direct = _pending_tree(monkeypatch, tmp_path)
    readings = _matrix_readings()
    _install_matrix(monkeypatch, tmp_path, readings)
    with pytest.raises(GenerationError, match="families on another backend") as raised:
        tool.verify_public_matrix(
            _matrix_binding(readings), bootstrap=False, roster=_roster()
        )
    for family in direct:
        assert family in str(raised.value)
    reading = _read(ROOT / "readings/fast_cpu_native_frontend_parity.json")
    with pytest.raises(GenerationError, match="rows are not the fast CPU route"):
        tool.verify_shared_evidence(
            _install_cpu_reading(monkeypatch, tmp_path, reading)
        )


def test_the_shipped_readings_still_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    readings = _matrix_readings()
    _install_matrix(monkeypatch, tmp_path, readings)
    tool.verify_public_matrix(
        _matrix_binding(readings), bootstrap=False, roster=_roster()
    )
    reading = _read(ROOT / "readings/fast_cpu_native_frontend_parity.json")
    tool.verify_shared_evidence(_install_cpu_reading(monkeypatch, tmp_path, reading))


# Every directly judged face with its reading: the four-face case.


def test_direct_faces_with_readings_are_admitted_on_the_cpu_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding, direct = _four_face_tree(monkeypatch, tmp_path)
    reading = _cpu_reading(binding, direct)
    loadable = len(binding["loadable_families"])
    assert reading["unique_artifacts"] == loadable + len(direct)
    tool.verify_shared_evidence(_install_cpu_reading(monkeypatch, tmp_path, reading))


def test_direct_faces_with_readings_run_fast_cpu_in_the_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding, direct = _four_face_tree(monkeypatch, tmp_path)
    readings = copy.deepcopy(_matrix_readings())
    for row in readings["cpu"]["rows"]:
        if row["family"] in direct:
            row["backend"] = "FastCpu"
    on_route = (
        set(binding["loadable_families"])
        | set(binding["carried_families"])
        | set(direct)
    )
    assert sum(r["backend"] == "FastCpu" for r in readings["cpu"]["rows"]) == len(
        on_route
    )
    _install_matrix(monkeypatch, tmp_path, readings)
    tool.verify_public_matrix(
        _matrix_binding(readings), bootstrap=False, roster=_roster()
    )


# A count or backend that disagrees with the binding is refused.


@pytest.mark.parametrize("key", ["unique_artifacts", "model_families"])
def test_a_wrong_cpu_count_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str
) -> None:
    binding, direct = _four_face_tree(monkeypatch, tmp_path)
    reading = _cpu_reading(binding, direct)
    reading[key] -= 1
    with pytest.raises(GenerationError, match=key):
        tool.verify_shared_evidence(
            _install_cpu_reading(monkeypatch, tmp_path, reading)
        )


def test_the_old_eleven_and_twelve_are_refused_once_the_faces_have_readings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding, _direct = _four_face_tree(monkeypatch, tmp_path)
    reading = _loadable_reading(binding)
    with pytest.raises(GenerationError, match="rows are not the fast CPU route"):
        tool.verify_shared_evidence(
            _install_cpu_reading(monkeypatch, tmp_path, reading)
        )


def test_a_direct_face_left_on_the_reference_engine_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _binding, direct = _four_face_tree(monkeypatch, tmp_path)
    readings = copy.deepcopy(_matrix_readings())
    for row in readings["cpu"]["rows"]:
        if row["family"] in direct:
            row["backend"] = "HuggingFace" if row["family"] == direct[0] else "FastCpu"
    _install_matrix(monkeypatch, tmp_path, readings)
    with pytest.raises(GenerationError, match=direct[0]):
        tool.verify_public_matrix(
            _matrix_binding(readings), bootstrap=False, roster=_roster()
        )


def test_a_swapped_backend_with_equal_counts_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    readings = copy.deepcopy(_matrix_readings())
    rows = readings["cpu"]["rows"]
    fast = next(row for row in rows if row["backend"] == "FastCpu")
    reference = next(row for row in rows if row["backend"] == "HuggingFace")
    fast["backend"], reference["backend"] = "HuggingFace", "FastCpu"
    _install_matrix(monkeypatch, tmp_path, readings)
    with pytest.raises(GenerationError, match="families on another backend"):
        tool.verify_public_matrix(
            _matrix_binding(readings), bootstrap=False, roster=_roster()
        )


# The binding itself is read fail-closed.


def test_a_recorded_reading_that_is_missing_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding, direct = _four_face_tree(monkeypatch, tmp_path)
    (tmp_path / binding["direct_families"][direct[0]]["reading"]).unlink()
    with pytest.raises(GenerationError, match="file is missing"):
        tool.fast_cpu_route(binding)


def test_a_recorded_reading_with_another_digest_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding, direct = _four_face_tree(monkeypatch, tmp_path)
    binding["direct_families"][direct[0]]["sha256"] = "0" * 64
    with pytest.raises(GenerationError, match="does not match the recorded"):
        tool.fast_cpu_route(binding)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda b: b.update(loadable_families={}), "loadable_families"),
        (lambda b: b.update(loadable_families=[]), "loadable_families"),
        (lambda b: b.update(carried_families=["x"]), "carried_families"),
        (
            lambda b: b["carried_families"].update(extra_face="not_loadable"),
            "not a loadable family",
        ),
        (lambda b: b.update(direct_families=["x"]), "direct_families"),
        (
            lambda b: b["direct_families"].update(extra_face={"reading": "x.json"}),
            "under readings/",
        ),
        (
            lambda b: b["direct_families"].update(
                extra_face={"reading": "readings/x.json", "sha256": "abc"}
            ),
            "not a SHA-256",
        ),
        (
            lambda b: b["carried_families"].update(
                {b["loadable_families"][0]: b["loadable_families"][1]}
            ),
            "more than one",
        ),
    ],
)
def test_an_unexpected_binding_shape_is_refused(change: Any, match: str) -> None:
    binding = copy.deepcopy(_fast_binding())
    change(binding)
    with pytest.raises(GenerationError, match=match):
        tool.fast_cpu_route(binding)


def test_a_coverage_that_does_not_count_the_loadable_families_is_refused() -> None:
    binding = copy.deepcopy(_fast_binding())
    binding["coverage"]["unique_tokenizer_artifacts"] += 1
    with pytest.raises(GenerationError, match="coverage does not count"):
        tool.cpu_parity_counts(binding)


def test_a_route_family_outside_the_roster_is_refused() -> None:
    binding = _fast_binding()
    roster = _roster() - {sorted(binding["loadable_families"])[0]}
    with pytest.raises(GenerationError, match="outside the certified roster"):
        tool.cpu_matrix_backends(binding, roster)
