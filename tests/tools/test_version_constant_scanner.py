"""Coverage and refusal tests for the identity-v2 version-read gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import scan_version_constants  # noqa: E402


def _scan(tmp_path: Path, relative: str, source: str) -> list[str]:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return scan_version_constants.scan_paths(
        tmp_path,
        (Path(relative),),
        audit_allowlist=False,
    )


def test_current_coverage_has_only_the_enumerated_build_fact_sites() -> None:
    assert not scan_version_constants.scan_paths(
        ROOT,
        scan_version_constants.covered_code_paths(),
    )


@pytest.mark.parametrize(
    ("relative", "source", "description"),
    [
        (
            "crates/toktier-routing-core/src/violation.rs",
            'const VERSION: &str = option_env!("CARGO_PKG_VERSION_MAJOR").unwrap();\n',
            "Cargo package-version constant",
        ),
        (
            "src/toktier/backends/fast_cpu.py",
            "import importlib.metadata\n",
            "Python distribution-metadata read",
        ),
        (
            "src/toktier/engine/gpu/native.py",
            "selected = dependency.__version__\n",
            "Python __version__ read",
        ),
    ],
)
def test_synthetic_version_reads_are_rejected(
    tmp_path: Path,
    relative: str,
    source: str,
    description: str,
) -> None:
    violations = _scan(tmp_path, relative, source)

    assert len(violations) == 1
    assert violations[0].startswith(f"{relative}:1: {description}:")


def test_allowlisted_pattern_is_capped_at_todays_count(tmp_path: Path) -> None:
    line = 'crate_version: env!("CARGO_PKG_VERSION").to_owned(),\n'
    violations = _scan(
        tmp_path,
        "crates/toktier/src/runtime.rs",
        line * 3,
    )

    assert violations == [
        "crates/toktier/src/runtime.rs:3: Cargo package-version constant: "
        'crate_version: env!("CARGO_PKG_VERSION").to_owned(),'
    ]
