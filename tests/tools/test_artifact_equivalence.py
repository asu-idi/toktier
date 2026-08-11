"""Fast applicability tests for the sentinel artifact-equivalence judge."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import verify_artifact_equivalence as equivalence  # noqa: E402


@pytest.fixture
def tree_pair(tmp_path: Path) -> tuple[Path, Path, equivalence.Coverage]:
    old = tmp_path / "old"
    files = {
        "Cargo.lock": "lock\n",
        "rust-toolchain.toml": "toolchain\n",
        "pyproject.toml": "project\n",
        "Cargo.toml": "[workspace]\n",
        "crates/core/Cargo.toml": "[package]\nname = \"core\"\n",
        "crates/core/src/lib.rs": "pub fn value() -> u8 { 1 }\n",
        "crates/core/NOTICE.md": "notice\n",
        "src/toktier/backends/fast_cpu.py": "VALUE = 1\n",
        "src/toktier/repair/tables/table.json": "{}\n",
        "README.md": "outside coverage\n",
    }
    for relative, content in files.items():
        path = old / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    new = tmp_path / "new"
    shutil.copytree(old, new)
    coverage = equivalence.Coverage(
        files=frozenset(
            {
                Path("Cargo.lock"),
                Path("rust-toolchain.toml"),
                Path("pyproject.toml"),
                Path("Cargo.toml"),
                Path("crates/core/Cargo.toml"),
                Path("crates/core/src/lib.rs"),
                Path("crates/core/NOTICE.md"),
                Path("src/toktier/backends/fast_cpu.py"),
                Path("src/toktier/repair/tables/table.json"),
            }
        ),
        trees=(Path("crates/core"),),
        byte_shipped=frozenset(
            {
                Path("src/toktier/backends/fast_cpu.py"),
                Path("src/toktier/repair/tables/table.json"),
            }
        ),
    )
    return old, new, coverage


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("crates/core/src/lib.rs", "pub fn value() -> u8 { 1 }\n// comment\n"),
        (
            "crates/core/Cargo.toml",
            "[package]\nname = \"core\"\n[dev-dependencies]\nfoo = \"=1\"\n",
        ),
        ("crates/core/NOTICE.md", "corrected notice\n"),
    ],
)
def test_covered_non_byte_shipped_changes_are_applicable(
    tree_pair: tuple[Path, Path, equivalence.Coverage],
    relative: str,
    content: str,
) -> None:
    old, new, coverage = tree_pair
    (new / relative).write_text(content, encoding="utf-8")

    result = equivalence.check_applicability(old, new, coverage)

    assert result.applicable
    assert result.witness["cargo_lock_unchanged"] is True
    assert result.witness["protected_files_unchanged"] is True


@pytest.mark.parametrize(
    ("relative", "failure", "failed_precondition"),
    [
        (
            "README.md",
            "tree_diff_outside_covered_non_byte_shipped_files",
            "covered_non_byte_shipped_diff_only",
        ),
        ("Cargo.lock", "cargo_lock_changed", "cargo_lock_byte_equal"),
        (
            "pyproject.toml",
            "protected_file_changed",
            "toolchain_packaging_and_byte_shipped_files_equal",
        ),
        (
            "src/toktier/backends/fast_cpu.py",
            "protected_file_changed",
            "toolchain_packaging_and_byte_shipped_files_equal",
        ),
    ],
)
def test_first_failed_precondition_returns_not_applicable(
    tree_pair: tuple[Path, Path, equivalence.Coverage],
    relative: str,
    failure: str,
    failed_precondition: str,
) -> None:
    old, new, coverage = tree_pair
    (new / relative).write_text("changed\n", encoding="utf-8")

    result = equivalence.check_applicability(old, new, coverage)

    assert not result.applicable
    assert result.witness["failure"] == failure
    assert result.witness["checks"][-1] == {
        "precondition": failed_precondition,
        "passed": False,
        **(
            {"outside_files": ["README.md"]}
            if failed_precondition == "covered_non_byte_shipped_diff_only"
            else (
                {"changed_files": [relative]}
                if failed_precondition
                == "toolchain_packaging_and_byte_shipped_files_equal"
                else {}
            )
        ),
    }
