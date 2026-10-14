#!/usr/bin/env python3
"""Refuse version reads inside the three source-identity coverage sets.

Identity-v2 deliberately tolerates changes to enumerated version fields, so
covered code must be structurally unable to branch on package versions.  The
only exceptions are enumerated build-fact embedding sites that report the
version without selecting behavior; each is capped by file, pattern, and
today's occurrence count.

The same run checks the other direction of the version boundary: the
places that tell a reader which version to depend on. A Rust dependency
line naming this workspace's crate -- ``toktier = "X"``,
``toktier = { version = "X", ... }``, either of them inside a Rust or TOML
comment, or ``cargo add toktier@X`` -- must name the workspace version, so
a release cannot ship documentation that points at the one before it. A
release note under ``docs/releases/`` names its own version instead,
which is what keeps the historical notes valid after later bumps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from scan_common import iter_files, read_text
from source_identity_common import IDENTITIES

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Rule:
    name: str
    suffixes: frozenset[str]
    pattern: re.Pattern[str]
    description: str


@dataclass(frozen=True)
class AllowedSite:
    path: str
    rule: str
    line_pattern: re.Pattern[str]
    expected_hits: int


RULES = (
    Rule(
        "cargo_package_version",
        frozenset({".rs"}),
        re.compile(r"\bCARGO_PKG_VERSION(?:_(?:MAJOR|MINOR|PATCH|PRE))?\b"),
        "Cargo package-version constant",
    ),
    Rule(
        "cargo_metadata_api",
        frozenset({".rs"}),
        re.compile(r"\b(?:cargo_metadata|MetadataCommand)\b"),
        "cargo-metadata API",
    ),
    Rule(
        "cargo_metadata_command",
        frozenset({".py", ".rs"}),
        re.compile(
            r"(?:Command::new\s*\(\s*['\"]cargo['\"]\s*\)|"
            r"['\"]cargo['\"]\s*,)[\s\S]{0,512}?['\"]metadata['\"]"
        ),
        "cargo metadata command",
    ),
    Rule(
        "python_distribution_metadata",
        frozenset({".py"}),
        re.compile(
            r"\bimportlib(?:\.|_)metadata\b|"
            r"\bfrom\s+importlib\s+import\s+metadata\b"
        ),
        "Python distribution-metadata read",
    ),
    Rule(
        "python_dunder_version",
        frozenset({".py"}),
        re.compile(r"\b__version__\b"),
        "Python __version__ read",
    ),
)

# These values are emitted as build/report facts only. They do not control a
# branch, feature, route, tokenization operation, or serialized format.
ALLOWLIST = (
    AllowedSite(
        "crates/toktier/src/artifact.rs",
        "cargo_package_version",
        re.compile(r'\s*concat!\("toktier-rust/", env!\("CARGO_PKG_VERSION"\)\),\s*'),
        1,
    ),
    AllowedSite(
        "crates/toktier/src/runtime.rs",
        "cargo_package_version",
        re.compile(r'\s*crate_version: env!\("CARGO_PKG_VERSION"\)\.to_owned\(\),\s*'),
        2,
    ),
)


def covered_code_paths(root: Path = ROOT) -> tuple[Path, ...]:
    """Return Rust and Python files in the union of all three v1 coverages."""
    paths: set[Path] = set()
    for identity in IDENTITIES.values():
        paths.update(
            Path(value)
            for value in identity.files
            if Path(value).suffix in {".py", ".rs"}
        )
        for tree in identity.trees:
            paths.update(
                path.relative_to(root)
                for path in (root / tree).rglob("*")
                if path.is_file() and path.suffix in {".py", ".rs"}
            )
    return tuple(sorted(paths, key=lambda path: path.parts))


def scan_paths(
    root: Path,
    paths: tuple[Path, ...],
    *,
    audit_allowlist: bool = True,
) -> list[str]:
    """Scan explicit covered paths and return stable, line-oriented refusals."""
    used = {site: 0 for site in ALLOWLIST}
    violations: list[str] = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            violations.append(f"{relative.as_posix()}: covered code file is missing")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for rule in RULES:
            if relative.suffix not in rule.suffixes:
                continue
            for match in rule.pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                line = lines[line_number - 1] if lines else ""
                allowed = next(
                    (
                        site
                        for site in ALLOWLIST
                        if site.path == relative.as_posix()
                        and site.rule == rule.name
                        and site.line_pattern.fullmatch(line)
                        and used[site] < site.expected_hits
                    ),
                    None,
                )
                if allowed is not None:
                    used[allowed] += 1
                    continue
                violations.append(
                    f"{relative.as_posix()}:{line_number}: "
                    f"{rule.description}: {line.strip()}"
                )
    if audit_allowlist:
        for site, count in used.items():
            if count != site.expected_hits:
                violations.append(
                    f"{site.path}: stale {site.rule} allowlist entry: "
                    f"expected {site.expected_hits} hit(s), found {count}"
                )
    return violations


#: The workspace version, as ``Cargo.toml`` declares it for every member.
_WORKSPACE_VERSION = re.compile(
    r"(?ms)^\[workspace\.package]\s*.*?^version\s*=\s*\"([^\"]+)\""
)

#: A dependency line that names the ``toktier`` crate and a version. The
#: line may sit in a ``//``, ``///``, ``//!`` or ``#`` comment, so doc
#: comments and commented manifests are read as well. A table without a
#: ``version`` key (a path dependency) does not match, and neither does a
#: quoted value that is not a version requirement (the console-script
#: entry of ``pyproject.toml`` has the same left-hand side).
_DEPENDENCY_LINE = re.compile(
    r'^\s*(?:(?://[/!]?|#)\s*)?toktier\s*=\s*'
    r'(?:"(?P<bare>[=^~]?\s*[0-9][^"\n]*)"'
    r'|\{[^}\n]*?\bversion\s*=\s*"(?P<table>[^"\n]*)")'
)
_CARGO_ADD = re.compile(r"\bcargo\s+add\s+toktier@(?P<version>[^\s`'\"]+)")

#: Text files that can carry a dependency example.
DEPENDENCY_EXAMPLE_SUFFIXES = frozenset({".md", ".rs", ".rst", ".toml", ".txt"})

#: A release note, which names the version it describes.
_RELEASE_NOTE = re.compile(r"docs/releases/v(?P<version>[0-9][0-9A-Za-z.+-]*)\.md")


def workspace_version(root: Path = ROOT) -> str:
    """The ``[workspace.package]`` version of the manifest under ``root``."""
    match = _WORKSPACE_VERSION.search(
        (root / "Cargo.toml").read_text(encoding="utf-8")
    )
    if match is None:
        raise ValueError("Cargo.toml has no [workspace.package] version")
    return match.group(1)


def _requirement_version(requirement: str) -> str:
    """The version a requirement names, without its comparison operator."""
    return requirement.strip().lstrip("=^~").strip()


def dependency_version_violations(
    root: Path, expected: str | None = None
) -> list[str]:
    """Dependency examples under ``root`` that name another version.

    ``expected`` defaults to the workspace version read from ``root``; a
    release note is held to the version in its own file name instead.
    """
    workspace = workspace_version(root) if expected is None else expected
    violations: list[str] = []
    for path, relative in iter_files(root):
        if Path(relative).suffix not in DEPENDENCY_EXAMPLE_SUFFIXES:
            continue
        text = read_text(path)
        if text is None:
            continue
        note = _RELEASE_NOTE.fullmatch(relative)
        wanted = note.group("version") if note is not None else workspace
        for line_number, line in enumerate(text.splitlines(), start=1):
            named: list[str] = []
            dependency = _DEPENDENCY_LINE.match(line)
            if dependency is not None:
                bare, table = dependency.group("bare", "table")
                named.append(bare if bare is not None else table)
            named.extend(match.group("version") for match in _CARGO_ADD.finditer(line))
            for requirement in named:
                if _requirement_version(requirement) != wanted:
                    violations.append(
                        f"{relative}:{line_number}: Rust dependency example names "
                        f"{requirement!r}, not {wanted!r}: {line.strip()}"
                    )
    return violations


def main() -> int:
    violations = scan_paths(ROOT, covered_code_paths())
    violations.extend(dependency_version_violations(ROOT))
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
