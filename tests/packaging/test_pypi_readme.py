"""PyPI long-description generation and link portability."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "README.pypi.md"

_RELATIVE_MARKDOWN = re.compile(
    r"]\((?!https?://|mailto:|#)[^)]+\)", re.MULTILINE
)
_RELATIVE_HTML = re.compile(
    r'(?:src|srcset|href)="(?!https?://|#)[^"]+"', re.MULTILINE
)


def test_pypi_readme_is_generated_from_canonical_readme() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_pypi_readme.py"), "--check"],
        check=True,
    )


def test_pypi_readme_has_no_repository_relative_targets() -> None:
    payload = OUTPUT.read_text(encoding="utf-8")
    assert _RELATIVE_MARKDOWN.search(payload) is None
    assert _RELATIVE_HTML.search(payload) is None
    assert "<picture>" not in payload
    assert (
        "https://raw.githubusercontent.com/asu-idi/toktier/v0.2.9/"
        "docs/figures/hero_session_vs_reencode.svg"
    ) in payload
    assert (
        "https://github.com/asu-idi/toktier/blob/v0.2.9/README.zh-CN.md"
    ) in payload


def test_project_metadata_uses_generated_long_description() -> None:
    pyproject = ROOT / "pyproject.toml"
    assert 'readme = "README.pypi.md"' in pyproject.read_text(encoding="utf-8")
