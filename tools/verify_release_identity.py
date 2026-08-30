#!/usr/bin/env python3
"""Refuse a release event whose tag and checked-in identities disagree."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(
    r"(?ms)^\[workspace\.package]\s*.*?^version\s*=\s*\"([^\"]+)\""
)

#: The release date this tree is cut for. It is checked against
#: CITATION.cff rather than derived from it, so a release cannot go out
#: carrying the previous release's date by omission. Moving the release
#: day means moving both, together.
RELEASE_DATE = "2026-08-30"


def project_version() -> str:
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(cargo)
    if match is None:
        raise ValueError("Cargo.toml has no workspace package version")
    return match.group(1)


def verify(tag: str) -> None:
    version = project_version()
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ValueError(f"release tag {tag!r} must be {expected_tag!r}")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if f'version: "{version}"' not in citation:
        raise ValueError("CITATION.cff version differs from Cargo.toml")
    if f'date-released: "{RELEASE_DATE}"' not in citation:
        raise ValueError(
            f"CITATION.cff does not carry the frozen release date {RELEASE_DATE}"
        )

    generator = (ROOT / "tools" / "generate_pypi_readme.py").read_text(
        encoding="utf-8"
    )
    if f'RELEASE_REF = "{expected_tag}"' not in generator:
        raise ValueError("PyPI README links do not target the release tag")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'readme = "README.pypi.md"' not in pyproject:
        raise ValueError("package metadata does not use the PyPI README")
    if 'requires = ["maturin==1.14.1"]' not in pyproject:
        raise ValueError("build backend is not exactly pinned")

    toolchain = (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    if 'channel = "1.93.1"' not in toolchain:
        raise ValueError("Rust release toolchain is not exactly pinned")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default=os.environ.get("GITHUB_REF_NAME"),
        help="release tag (defaults to GITHUB_REF_NAME)",
    )
    arguments = parser.parse_args()
    if not arguments.tag:
        parser.error("--tag is required outside a GitHub tag event")
    try:
        verify(str(arguments.tag))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"release identity verified: {arguments.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
