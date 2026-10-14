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


# ---------------------------------------------- Rust dependency examples --

_MANIFEST = '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "1.4.2"\n'


def _dependency_tree(tmp_path: Path, files: dict[str, str]) -> list[str]:
    (tmp_path / "Cargo.toml").write_text(_MANIFEST, encoding="utf-8")
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return scan_version_constants.dependency_version_violations(tmp_path)


def test_this_tree_names_the_workspace_version_in_its_dependency_examples() -> None:
    assert scan_version_constants.dependency_version_violations(ROOT) == []
    # The check reads real lines here, so it cannot pass by matching none.
    readme = (ROOT / "crates/toktier/README.md").read_text(encoding="utf-8")
    version = scan_version_constants.workspace_version(ROOT)
    assert f'toktier = {{ version = "{version}"' in readme


@pytest.mark.parametrize(
    "line",
    [
        'toktier = { version = "1.4.2", features = ["network"] }',
        'toktier = "1.4.2"',
        'toktier = "=1.4.2"',
        '//! toktier = { version = "1.4.2" }',
        "cargo add toktier@1.4.2",
        # Not a dependency on a version: no finding whatever the manifest says.
        'toktier = { path = "../../crates/toktier" }',
        "toktier = 'toktier.cli:main'",
        'toktier = "toktier.cli:main"',
        "cargo add toktier --features network",
    ],
)
def test_a_dependency_example_on_the_workspace_version_passes(
    tmp_path: Path, line: str
) -> None:
    assert _dependency_tree(tmp_path, {"docs/example.md": f"```\n{line}\n```\n"}) == []


@pytest.mark.parametrize(
    ("relative", "line", "named"),
    [
        (
            "crates/toktier/README.md",
            'toktier = { version = "1.4.1", features = ["network"] }',
            "1.4.1",
        ),
        ("docs/rust-lifecycle.md", 'toktier = "1.4"', "1.4"),
        ("crates/toktier/src/lib.rs", '//! toktier = { version = "1.3.0" }', "1.3.0"),
        (
            "examples/Cargo.toml",
            'toktier = { path = "..", version = "=1.4.1" }',
            "=1.4.1",
        ),
        ("README.md", "    cargo add toktier@1.4.1", "1.4.1"),
    ],
)
def test_a_dependency_example_on_another_version_is_refused(
    tmp_path: Path, relative: str, line: str, named: str
) -> None:
    violations = _dependency_tree(tmp_path, {relative: f"intro\n{line}\n"})

    assert violations == [
        f"{relative}:2: Rust dependency example names {named!r}, not '1.4.2': "
        f"{line.strip()}"
    ]


def test_a_release_note_is_held_to_its_own_version(tmp_path: Path) -> None:
    line = 'toktier = { version = "1.3.0" }'
    violations = _dependency_tree(
        tmp_path,
        {
            "docs/releases/v1.3.0.md": f"{line}\n",
            "docs/releases/v1.4.2.md": f"{line}\n",
        },
    )

    assert violations == [
        "docs/releases/v1.4.2.md:1: Rust dependency example names '1.3.0', "
        f"not '1.4.2': {line}"
    ]


def test_build_output_and_unrelated_files_are_not_read(tmp_path: Path) -> None:
    line = 'toktier = { version = "0.0.1" }\n'
    assert (
        _dependency_tree(
            tmp_path,
            {
                "target/doc/README.md": line,
                ".git/README.md": line,
                "notes.json": line,
            },
        )
        == []
    )


def test_main_reports_a_stale_dependency_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _dependency_tree(tmp_path, {"docs/guide.md": 'toktier = "1.4.1"\n'})
    monkeypatch.setattr(scan_version_constants, "ROOT", tmp_path)
    # The covered-code half is exercised above; this case is about the
    # dependency half reaching the report and the exit status.
    monkeypatch.setattr(scan_version_constants, "covered_code_paths", lambda: ())
    monkeypatch.setattr(scan_version_constants, "scan_paths", lambda root, paths: [])

    assert scan_version_constants.main() == 1
    assert capsys.readouterr().out == (
        "docs/guide.md:1: Rust dependency example names '1.4.1', not '1.4.2': "
        'toktier = "1.4.1"\n'
    )
