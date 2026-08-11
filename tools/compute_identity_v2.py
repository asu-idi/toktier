#!/usr/bin/env python3
"""Compute version-normalized v2 identities over the exact v1 source sets.

Only the adopted version axis is normalized: the root workspace-package
version, the 11 enumerated internal production dependency constraints, and
the seven enumerated workspace-package versions in Cargo.lock.  Every
transform is line-local and anchored by its file, section, and key.  Use
``--show-diff`` to review the bytes changed before hashing.
"""

from __future__ import annotations

import argparse
import difflib
import re
from hashlib import sha256
from pathlib import Path

from source_identity_common import IDENTITIES

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {
    "fast_cpu": b"toktier.fast_cpu.integrated_source.v2\0",
    "native_host": b"toktier.prebuilt.native_host_source.v2\0",
    "rust_api": b"toktier.rust_api.integrated_source.v2\0",
}
WORKSPACE_PACKAGES = frozenset(
    {
        "toktier",
        "toktier-cuda-driver",
        "toktier-gigatoken-core",
        "toktier-py",
        "toktier-routing-core",
        "toktier-store-core",
        "toktier-store-sqlite",
    }
)
INTERNAL_DEPENDENCIES = {
    Path("crates/toktier/Cargo.toml"): frozenset(
        {
            "toktier-routing-core",
            "toktier-cuda-driver",
            "toktier-store-core",
            "toktier-store-sqlite",
        }
    ),
    Path("crates/toktier-py/Cargo.toml"): frozenset(
        {"toktier-routing-core", "toktier-store-core", "toktier-store-sqlite"}
    ),
    Path("crates/toktier-routing-core/Cargo.toml"): frozenset(
        {"toktier-gigatoken-core", "toktier-cuda-driver", "toktier-store-core"}
    ),
    Path("crates/toktier-store-sqlite/Cargo.toml"): frozenset({"toktier-store-core"}),
}
_VERSION_FIELD = re.compile(r'(?P<head>\bversion\s*=\s*")=[^"]+(?P<tail>")')
_PATH_FIELD = re.compile(r"(?:^|[,{])\s*path\s*=")


def _quoted_value(line: str) -> str | None:
    rest = line.partition("=")[2].lstrip()
    if not rest.startswith('"') or (end := rest.find('"', 1)) < 0:
        return None
    return rest[1:end]


def _replace_assignment(line: str, value: str) -> str:
    left, separator, rest = line.partition("=")
    leading = rest[: len(rest) - len(rest.lstrip())]
    body = rest.lstrip()
    if not separator or not body.startswith('"') or (end := body.find('"', 1)) < 0:
        return line
    return f'{left}{separator}{leading}"{value}"{body[end + 1 :]}'


def _normalize_manifest(relative: Path, text: str) -> str:
    allowed = INTERNAL_DEPENDENCIES.get(relative, ())
    section = ""
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        key, separator, value = line.partition("=")
        key = key.strip()
        if (
            relative == Path("Cargo.toml")
            and section == "[workspace.package]" and key == "version"
        ):
            line = _replace_assignment(line, "0.0.0")
        elif (
            section == "[dependencies]" and key in allowed
            and separator and _PATH_FIELD.search(value)
        ):
            line = _VERSION_FIELD.sub(r"\g<head>=0.0.0\g<tail>", line, count=1)
        output.append(line)
    return "".join(output)


def _normalize_lock(text: str) -> str:
    lines = text.splitlines(keepends=True)
    starts = [
        index for index, line in enumerate(lines) if line.strip() == "[[package]]"
    ]
    for start, end in zip(starts, [*starts[1:], len(lines)], strict=True):
        fields = {
            line.partition("=")[0].strip(): index
            for index in range(start + 1, end)
            if "=" in (line := lines[index])
            and line.partition("=")[0].strip() in {"name", "version"}
        }
        name_index = fields.get("name")
        version_index = fields.get("version")
        if (
            name_index is not None and version_index is not None
            and _quoted_value(lines[name_index]) in WORKSPACE_PACKAGES
        ):
            lines[version_index] = _replace_assignment(lines[version_index], "0.0.0")
    return "".join(lines)


def normalize(relative: Path, content: bytes) -> bytes:
    """Return bytes with exactly the three enumerated field classes normalized."""
    if relative == Path("Cargo.lock"):
        return _normalize_lock(content.decode()).encode()
    if relative.name == "Cargo.toml":
        return _normalize_manifest(relative, content.decode()).encode()
    return content


def source_paths(root: Path, identity: str) -> tuple[Path, ...]:
    """Return the v1 coverage set rooted at ``root``."""
    definition = IDENTITIES[identity]
    paths = {Path(value) for value in definition.files}
    for tree in definition.trees:
        paths.update(
            path.relative_to(root)
            for path in (root / tree).rglob("*")
            if path.is_file()
        )
    return tuple(sorted(paths, key=lambda path: path.parts))


def source_digest(identity: str, root: Path = ROOT) -> str:
    """Hash one v1 source set after the exact version-axis normalization."""
    digest = sha256(DOMAINS[identity])
    for relative in source_paths(root, identity):
        rendered = relative.as_posix().encode()
        content = normalize(relative, (root / relative).read_bytes())
        digest.update(len(rendered).to_bytes(8, "little"))
        digest.update(rendered)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def normalization_diff(root: Path, identities: tuple[str, ...]) -> str:
    """Render one unified review diff for all normalized covered files."""
    chunks: list[str] = []
    covered = {path for name in identities for path in source_paths(root, name)}
    for relative in sorted(covered, key=lambda path: path.parts):
        raw = (root / relative).read_bytes()
        normalized = normalize(relative, raw)
        if raw != normalized:
            chunks.extend(difflib.unified_diff(
                raw.decode().splitlines(True), normalized.decode().splitlines(True),
                f"a/{relative.as_posix()}", f"b/{relative.as_posix()}",
            ))
    return "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identities", nargs="*", choices=tuple(DOMAINS))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--show-diff", action="store_true")
    arguments = parser.parse_args()
    identities = tuple(arguments.identities) or tuple(DOMAINS)
    if arguments.show_diff:
        print(normalization_diff(arguments.root, identities), end="")
    for identity in identities:
        print(f"{identity} {source_digest(identity, arguments.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
