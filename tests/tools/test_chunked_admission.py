"""The chunked admission table is what the readings derive, and nothing more.

Offline tier: the shipped table, manifest and readings pass their own
``--check``; the table is a subset of the certified repair roster; the two
identities it records are this tree's; a regenerated pair is byte-identical
to the shipped pair (``generated_by`` excepted); the generator refuses a
family with a recorded mismatch, an artifact outside the roster, and a
digest that is not the roster's; ``--release-check`` names every roster
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
        assert "python" in row["paths"]
        assert row["evidence_id"] == tool.EVIDENCE_ID
        assert row["corpus_docs"] or row["adversarial_strings"]
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


def test_native_path_needs_both_native_columns_and_the_python_path() -> None:
    readings = _readings()
    row = copy.deepcopy(readings["families"][0])
    row["paths"] = ["python", "native"]
    row["native"] = {
        "chunks": 1,
        "chunked_docs": 1,
        "serial_fallbacks": 0,
        "serial_reasons": {},
        "plan_disagreements": 0,
    }
    row["mismatches"]["native_vs_oracle"] = 0
    # One native column missing: the native path is not granted.
    readings_a = copy.deepcopy(readings)
    readings_a["families"][0] = copy.deepcopy(row)
    table, _ = _derive(readings_a)
    granted = next(r for r in table["rows"] if r["family"] == row["family"])
    assert granted["paths"] == ["python"]
    # Both native columns at zero: granted, and the manifest still binds.
    row["mismatches"]["python_vs_native"] = 0
    readings_b = copy.deepcopy(readings)
    readings_b["families"][0] = row
    table, manifest = _derive(readings_b)
    granted = next(r for r in table["rows"] if r["family"] == row["family"])
    assert granted["paths"] == ["python", "native"]
    assert manifest["totals"]["mismatches"] == 0


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
    if expected or readings["chunked_identity"] != table["chunked_identity"]:
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
