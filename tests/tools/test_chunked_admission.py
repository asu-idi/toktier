"""The chunked admission table is what the readings derive, and nothing more.

Offline tier: the shipped table, manifest and readings pass their own
``--check``; the table is a subset of the certified repair roster; the two
identities it records are this tree's; a regenerated pair is byte-identical
to the shipped pair (``generated_by`` excepted); the generator refuses a
family with a recorded mismatch, an artifact outside the roster, a
digest that is not the roster's, and a certified row without a recorded
loader face; ``--release-check`` names every roster
family without a row rather than inventing one; and the tools-only
``chunked`` identity stays out of the three build-time identities.
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

import compute_identity_v2  # noqa: E402
import update_chunked_admission as tool  # noqa: E402
import validate_registry  # noqa: E402
from registry_common import (  # noqa: E402
    CHUNKED_ADMISSION_DOMAIN_TAG,
    CHUNKED_READINGS_DOMAIN_TAG,
    GenerationError,
    load_json,
    serialise_document,
    with_root_digest,
)
from source_identity_common import (  # noqa: E402
    BUILD_SCRIPT_LISTS,
    IDENTITIES,
    TOOL_SIDE_IDENTITIES,
    list_synchronization_problems,
)

from toktier.repair.registry import families as roster_families  # noqa: E402

GENERATED_BY = {
    "tool": tool.TOOL_NAME,
    "tool_version": tool.TOOL_VERSION,
    "source_commit": "0" * 40,
    "generated_at": "1970-01-01T00:00:00Z",
}


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOLS / "update_chunked_admission.py"), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _readings() -> dict[str, Any]:
    readings: dict[str, Any] = load_json(tool.READINGS_PATH)
    return readings


def _derive(readings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    table, manifest = tool.derive(
        readings,
        identity=tool.chunked_identity(),
        engine_v2=compute_identity_v2.source_digest("fast_cpu"),
        generated_by=GENERATED_BY,
    )
    return table, manifest


# -- the shipped documents --------------------------------------------------


def test_shipped_documents_pass_check() -> None:
    result = _run("--check")
    assert result.returncode == 0, result.stderr
    assert "check passed" in result.stdout


def test_table_is_a_roster_subset_bound_to_this_tree() -> None:
    table: dict[str, Any] = load_json(tool.TABLE_PATH)
    roster = {item.family: item.artifact_sha256 for item in roster_families()}
    rows = table["rows"]
    assert rows, "the shipped table has no rows"
    assert [row["family"] for row in rows] == sorted(row["family"] for row in rows)
    for row in rows:
        assert row["artifact_sha256"] == roster[row["family"]], row["family"]
        face = row["loader_face_sha256"]
        assert face is None or (len(face) == 64 and int(face, 16) >= 0)
        assert row["evidence_id"] == tool.EVIDENCE_ID
        assert row["corpus_docs"] or row["adversarial_strings"]
        levels = set(row["evidence"]["python"])
        if row["status"] == "certified":
            assert levels == set(tool.EVIDENCE_LEVELS) and "python" in row["paths"]
            assert row["corpus_docs"] and row["adversarial_strings"]
        else:
            assert row["paths"] == [] and len(levels) == 1
            assert row["status"] == next(iter(levels)) + "_only"
        if "native" in row["paths"]:
            assert set(row["evidence"]["native"]) == set(tool.EVIDENCE_LEVELS)
    assert any(row["status"] == "certified" for row in rows)
    assert table["chunked_identity"] == tool.chunked_identity()
    assert table["fast_cpu_source_digest_v2"] == compute_identity_v2.source_digest(
        "fast_cpu"
    )
    manifest: dict[str, Any] = load_json(tool.MANIFEST_PATH)
    assert manifest["evidence_id"] == tool.EVIDENCE_ID
    assert {row["family"] for row in manifest["artifacts"]} == {
        row["family"] for row in rows
    }
    assert manifest["totals"]["mismatches"] == 0


def test_regeneration_is_byte_identical_except_generated_by(tmp_path: Path) -> None:
    result = _run(
        "--table",
        str(tmp_path / "table.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--source-commit",
        "0" * 40,
    )
    assert result.returncode == 0, result.stderr
    for fresh, shipped in (
        (tmp_path / "table.json", tool.TABLE_PATH),
        (tmp_path / "manifest.json", tool.MANIFEST_PATH),
    ):
        regenerated = json.loads(fresh.read_text(encoding="utf-8"))
        existing = json.loads(shipped.read_text(encoding="utf-8"))
        regenerated["generated_by"] = existing["generated_by"]
        regenerated.pop("root_digest")
        existing.pop("root_digest")
        assert regenerated == existing, shipped.name


def test_shipped_documents_validate_standalone() -> None:
    for path, kind in (
        (tool.TABLE_PATH, "chunked_admission"),
        (tool.READINGS_PATH, "chunked_readings"),
        (tool.MANIFEST_PATH, "evidence"),
    ):
        document = validate_registry.load_document(path)
        assert validate_registry.detect_schema(document) == kind, path.name
        assert validate_registry.check_schema(document, kind) == (True, ""), path.name
        digest_ok = validate_registry.check_root_digest(document, kind)
        assert digest_ok == (True, ""), path.name


# -- the derivation refuses rather than rounds --------------------------------


def test_a_recorded_mismatch_refuses_the_derivation() -> None:
    readings = _readings()
    assert isinstance(readings["families"], list) and readings["families"]
    tampered = copy.deepcopy(readings)
    tampered["families"][0]["mismatches"]["python_vs_oracle"] = 1
    with pytest.raises(GenerationError, match="recorded mismatch"):
        _derive(tampered)
    tampered = copy.deepcopy(readings)
    tampered["adversarial"]["families"][0]["mismatches"]["serial_vs_oracle"] = 2
    with pytest.raises(GenerationError, match="recorded mismatch"):
        _derive(tampered)


def test_a_family_outside_the_roster_or_with_another_digest_is_refused() -> None:
    readings = _readings()
    tampered = copy.deepcopy(readings)
    tampered["families"][0]["family"] = "kimi_k3"
    with pytest.raises(GenerationError, match="outside the certified repair roster"):
        _derive(tampered)
    tampered = copy.deepcopy(readings)
    tampered["families"][0]["artifact_sha256"] = "0" * 64
    with pytest.raises(GenerationError, match="is not the roster artifact"):
        _derive(tampered)


def _with_native(row: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out["paths"] = ["python", "native"]
    out["mismatches"]["native_vs_oracle"] = 0
    out["mismatches"]["python_vs_native"] = 0
    if "docs" in out:
        out["native"] = {
            "chunks": 1,
            "chunked_docs": 1,
            "serial_fallbacks": 0,
            "serial_reasons": {},
            "plan_disagreements": 0,
        }
    return out


def _python_only(row: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out["paths"] = ["python"]
    out["mismatches"]["native_vs_oracle"] = None
    out["mismatches"]["python_vs_native"] = None
    if "docs" in out:
        out["native"] = None
    return out


def _row_index(rows: list[dict[str, Any]], family: str) -> int:
    return next(i for i, row in enumerate(rows) if row["family"] == family)


def test_native_path_needs_both_levels_and_complete_columns() -> None:
    readings = _readings()
    certified = next(
        row["family"]
        for row in load_json(tool.TABLE_PATH)["rows"]
        if row["status"] == "certified"
    )
    corpus_i = _row_index(readings["families"], certified)
    battery_i = _row_index(readings["adversarial"]["families"], certified)
    # Start from a Python-only reading of this family at both levels,
    # whatever the shipped readings record for the native path.
    readings["families"][corpus_i] = _python_only(readings["families"][corpus_i])
    readings["adversarial"]["families"][battery_i] = _python_only(
        readings["adversarial"]["families"][battery_i]
    )

    def granted(document: dict[str, Any]) -> list[str]:
        table, _manifest = _derive(document)
        row = next(r for r in table["rows"] if r["family"] == certified)
        assert row["status"] == "certified"
        return list(row["paths"])

    # Native judged by the corpus campaign alone: recorded, not granted.
    corpus_only = copy.deepcopy(readings)
    corpus_only["families"][corpus_i] = _with_native(readings["families"][corpus_i])
    assert granted(corpus_only) == ["python"]
    # Native judged by both levels, but a column missing at one: not granted.
    missing = copy.deepcopy(corpus_only)
    battery_row = _with_native(readings["adversarial"]["families"][battery_i])
    battery_row["mismatches"]["python_vs_native"] = None
    missing["adversarial"]["families"][battery_i] = battery_row
    assert granted(missing) == ["python"]
    # Both levels, both columns at zero: granted.
    complete = copy.deepcopy(corpus_only)
    complete["adversarial"]["families"][battery_i] = _with_native(
        readings["adversarial"]["families"][battery_i]
    )
    assert granted(complete) == ["python", "native"]
    _table, manifest = _derive(complete)
    assert manifest["totals"]["mismatches"] == 0


def test_one_evidence_level_gives_a_named_partial_status() -> None:
    readings = _readings()
    certified = next(
        row["family"]
        for row in load_json(tool.TABLE_PATH)["rows"]
        if row["status"] == "certified"
    )
    # A Python-only battery row, whatever the shipped readings record for
    # the native path.
    battery_i = _row_index(readings["adversarial"]["families"], certified)
    readings["adversarial"]["families"][battery_i] = _python_only(
        readings["adversarial"]["families"][battery_i]
    )
    battery_only = copy.deepcopy(readings)
    battery_only["families"] = [
        row for row in readings["families"] if row["family"] != certified
    ]
    table, manifest = _derive(battery_only)
    row = next(r for r in table["rows"] if r["family"] == certified)
    assert row["status"] == "adversarial_only" and row["paths"] == []
    assert row["evidence"] == {"python": ["adversarial"], "native": []}
    assert row["corpus_docs"] == 0 and row["adversarial_strings"] > 0
    assert {a["family"] for a in manifest["artifacts"]} >= {certified}
    corpus_only = copy.deepcopy(readings)
    corpus_only["adversarial"]["families"] = [
        row for row in readings["adversarial"]["families"] if row["family"] != certified
    ]
    table, _manifest = _derive(corpus_only)
    row = next(r for r in table["rows"] if r["family"] == certified)
    assert row["status"] == "corpus_only" and row["paths"] == []
    assert tool.partial_rows(table) and tool.absent_families(readings, table) == (
        tool.absent_families(readings, load_json(tool.TABLE_PATH))
    )


def test_a_certified_row_needs_a_recorded_loader_face(tmp_path: Path) -> None:
    """The run-time admission compares the live loader face with the face a
    certified row records, so the derivation refuses a certified row whose
    readings recorded no face at either level; one level recording it is
    enough, and a row that is not certified may go without (it is refused
    at run time anyway). ``--check`` runs the same derivation, and also
    names a certified row without a face in the table it is given."""
    readings = _readings()
    certified = next(
        row["family"]
        for row in load_json(tool.TABLE_PATH)["rows"]
        if row["status"] == "certified"
    )
    corpus_i = _row_index(readings["families"], certified)
    battery_i = _row_index(readings["adversarial"]["families"], certified)
    face = (
        readings["families"][corpus_i]["loader_face_sha256"]
        or readings["adversarial"]["families"][battery_i]["loader_face_sha256"]
    )
    assert isinstance(face, str) and len(face) == 64

    one_level = copy.deepcopy(readings)
    one_level["families"][corpus_i]["loader_face_sha256"] = None
    one_level["adversarial"]["families"][battery_i]["loader_face_sha256"] = face
    table, _manifest = _derive(one_level)
    row = next(r for r in table["rows"] if r["family"] == certified)
    assert row["status"] == "certified" and row["loader_face_sha256"] == face

    faceless = copy.deepcopy(one_level)
    faceless["adversarial"]["families"][battery_i]["loader_face_sha256"] = None
    with pytest.raises(GenerationError, match="certified row needs the loader face"):
        _derive(faceless)

    partial = copy.deepcopy(faceless)
    partial["families"] = [r for r in partial["families"] if r["family"] != certified]
    table, _manifest = _derive(partial)
    row = next(r for r in table["rows"] if r["family"] == certified)
    assert row["status"] == "adversarial_only" and row["loader_face_sha256"] is None

    path = tmp_path / "readings.json"
    path.write_bytes(
        serialise_document(with_root_digest(faceless, CHUNKED_READINGS_DOMAIN_TAG))
    )
    result = _run("--check", "--readings", str(path))
    assert result.returncode == 2
    assert "certified row needs the loader face" in result.stderr

    shipped: dict[str, Any] = load_json(tool.TABLE_PATH)
    for entry in shipped["rows"]:
        if entry["family"] == certified:
            entry["loader_face_sha256"] = None
    edited = tmp_path / "table.json"
    edited.write_bytes(
        serialise_document(with_root_digest(shipped, CHUNKED_ADMISSION_DOMAIN_TAG))
    )
    result = _run("--check", "--table", str(edited))
    assert result.returncode == 1
    assert f"{certified}: a certified row records no loader_face_sha256" in (
        result.stderr
    )


def test_release_check_names_the_absent_roster_families(tmp_path: Path) -> None:
    readings = _readings()
    table: dict[str, Any] = load_json(tool.TABLE_PATH)
    roster = {item.family for item in roster_families()}
    present = {row["family"] for row in table["rows"]}
    expected = sorted(roster - present)
    assert tool.absent_families(readings, table) == expected
    result = _run("--release-check")
    for family in expected:
        assert f"release: {family} is in the certified repair roster" in result.stderr
    for row in tool.partial_rows(table):
        assert f"release: {row['family']} has chunked-path evidence" in result.stderr
    if (
        expected
        or tool.partial_rows(table)
        or readings["chunked_identity"] != table["chunked_identity"]
    ):
        assert result.returncode == 1
    # A family recorded as not reasoned about is not reported as absent.
    if expected:
        excluded = copy.deepcopy(readings)
        roster_digest = {
            item.family: item.artifact_sha256 for item in roster_families()
        }
        excluded["not_reasoned"] = [
            {
                "family": expected[0],
                "artifact_sha256": roster_digest[expected[0]],
                "refusal": "normalizer_not_supported:NFKC",
            }
        ]
        assert tool.absent_families(excluded, table) == expected[1:]
        # ... but it may not also carry a judged row.
        excluded["families"].append(
            dict(
                readings["families"][0],
                family=expected[0],
                artifact_sha256=roster_digest[expected[0]],
            )
        )
        with pytest.raises(GenerationError, match="not reasoned about"):
            _derive(excluded)


def test_check_rejects_a_hand_edited_table(tmp_path: Path) -> None:
    table: dict[str, Any] = load_json(tool.TABLE_PATH)
    table["rows"][0]["paths"] = ["python", "native"]
    edited = tmp_path / "table.json"
    edited.write_bytes(
        serialise_document(with_root_digest(table, CHUNKED_ADMISSION_DOMAIN_TAG))
    )
    result = _run("--check", "--table", str(edited))
    assert result.returncode == 1
    assert "regenerated content differs" in result.stderr


# -- the tools-only identity --------------------------------------------------


def test_chunked_identity_is_tools_only_and_covers_the_three_files() -> None:
    chunked = TOOL_SIDE_IDENTITIES["chunked"]
    assert chunked.domain == b"toktier.chunked.source.v1\0"
    assert chunked.files == (
        "src/toktier/chunked.py",
        "src/toktier/repair/tables/fast_repair_families.v1.json",
        "src/toktier/repair/tables/repair_pclass.v1.zlib",
    )
    assert chunked.trees == ()
    assert "chunked" not in IDENTITIES
    assert "chunked" not in {name for name, _module, _function in BUILD_SCRIPT_LISTS}
    assert "chunked" not in compute_identity_v2.DOMAINS
    assert list_synchronization_problems() == []
    assert tool.chunked_identity() == "sha256:" + chunked.source_digest()
