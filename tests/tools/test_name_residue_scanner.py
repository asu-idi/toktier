"""Refusal tests for the internal-name residue scanner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import scan_name_residue  # noqa: E402

# Built by concatenation so that this file does not match the scanner.
PROTOTYPE = "Inc" + "Tok"
LEGACY = "Lo" + "PT"


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_clean_tree_reports_nothing(tmp_path: Path) -> None:
    _write(tmp_path, "docs/serving.md", "Chunk-parallel encode.\n")

    assert scan_name_residue.scan(tmp_path) == []


@pytest.mark.parametrize(
    "spelling",
    [PROTOTYPE, PROTOTYPE.lower(), PROTOTYPE.upper(), LEGACY, LEGACY.lower()],
)
def test_each_forbidden_name_is_reported_in_any_case(
    tmp_path: Path, spelling: str
) -> None:
    _write(tmp_path, "src/pkg/module.py", f"ok = 1\n# see {spelling} notes\n")

    assert scan_name_residue.scan(tmp_path) == [
        f"src/pkg/module.py:2:# see {spelling} notes"
    ]


@pytest.mark.parametrize(
    "relative",
    ["evidence/readings.json", "CITATION.cff", ".github/workflows/ci.yml"],
)
def test_data_and_metadata_files_are_covered(tmp_path: Path, relative: str) -> None:
    _write(tmp_path, relative, f'"name": "{PROTOTYPE.lower()}"\n')

    assert scan_name_residue.scan(tmp_path) == [
        f'{relative}:1:"name": "{PROTOTYPE.lower()}"'
    ]


def test_skipped_build_directories_are_not_read(tmp_path: Path) -> None:
    _write(tmp_path, "target/debug/notes.txt", PROTOTYPE)
    _write(tmp_path, ".git/COMMIT_EDITMSG", PROTOTYPE)

    assert scan_name_residue.scan(tmp_path) == []


def test_main_fails_closed_on_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("# none\n", encoding="utf-8")
    _write(tmp_path, "README.md", f"{PROTOTYPE}\n")
    monkeypatch.setattr(scan_name_residue, "ROOT", tmp_path)
    monkeypatch.setattr(scan_name_residue, "ALLOWLIST", allowlist)

    assert scan_name_residue.main() == 1
