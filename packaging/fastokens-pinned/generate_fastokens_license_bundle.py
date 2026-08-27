#!/usr/bin/env python3
"""Generate the license bundle for the pinned Fastokens distribution.

The input is Cargo's locked metadata for the patched Fastokens checkout,
rooted at the ``fastokens-python`` extension crate.  Every normal dependency
reachable on any target is included.  This is a deliberately conservative
superset of the Linux binary's linked closure; it prevents a target predicate
or procedural-macro edge from silently dropping attribution material.

Usage::

    cargo metadata --offline --locked --format-version 1 \
        --manifest-path python/Cargo.toml > metadata.json
    python3 packaging/fastokens-pinned/generate_fastokens_license_bundle.py \
        metadata.json packaging/fastokens-pinned/THIRD_PARTY_LICENSES-fastokens.txt
"""


from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HEADER = """THIRD-PARTY LICENSE BUNDLE FOR THE PINNED FASTOKENS DISTRIBUTION
=================================================================

Generated from the locked Cargo metadata of toktier-fastokens 0.3.1.1
(upstream fastokens tag v0.3.1, commit
fe854299553524f2156a22036a2cb4d1f2ef4d97, with the toktier patch series
applied), rooted at the fastokens-python extension crate. The dependency
walk includes every normal edge on every target, so it is a conservative
superset of the Linux x86-64 wheel's linked dependency closure.

The package/version roster and SPDX expressions below are the accounting
index. License and notice files found in the corresponding published crate
sources follow, deduplicated byte-for-byte. Packages whose crate archive has
no standalone license file are called out separately and are covered by the
matching standard text elsewhere in this bundle or in the distribution's
NOTICE and THIRD_PARTY_NOTICES files.
"""

_LICENSE_NAME_MARKERS = (
    "license",
    "licence",
    "copying",
    "notice",
    "copyright",
    "authors",
    "apache",
    "mit",
    "bsd",
    "mpl",
    "lgpl",
    "gpl",
    "isc",
    "unlicense",
    "cc0",
    "zlib",
    "cdla",
    "unicode",
    "0bsd",
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _normal_closure(metadata: dict[str, Any]) -> set[str]:
    resolve = _mapping(metadata.get("resolve"), "resolve")
    root = resolve.get("root")
    if not isinstance(root, str):
        raise ValueError("metadata has no resolve root")
    nodes_raw = resolve.get("nodes")
    if not isinstance(nodes_raw, list):
        raise ValueError("metadata has no resolve nodes")
    nodes = {
        str(node["id"]): node for node in nodes_raw if isinstance(node, dict)
    }
    seen: set[str] = set()
    pending = [root]
    while pending:
        package_id = pending.pop()
        if package_id in seen:
            continue
        seen.add(package_id)
        node = _mapping(nodes.get(package_id), f"resolve node {package_id}")
        dependencies = node.get("deps")
        if not isinstance(dependencies, list):
            raise ValueError(f"resolve node {package_id} has no dependencies")
        for dependency in dependencies:
            dependency = _mapping(dependency, "dependency")
            kinds = dependency.get("dep_kinds")
            if not isinstance(kinds, list):
                continue
            if any(
                isinstance(kind, dict) and kind.get("kind") in (None, "normal")
                for kind in kinds
            ):
                pending.append(str(dependency["pkg"]))
    return seen


def _license_files(package: dict[str, Any]) -> list[Path]:
    root = Path(str(package["manifest_path"])).resolve().parent
    paths = [
        path
        for path in root.iterdir()
        if path.is_file()
        and any(marker in path.name.lower() for marker in _LICENSE_NAME_MARKERS)
    ]
    license_file = package.get("license_file")
    if isinstance(license_file, str):
        explicit = (root / license_file).resolve()
        if explicit.is_file():
            paths.append(explicit)
    return sorted(set(paths), key=lambda path: path.name.lower())


def generate(metadata: dict[str, Any]) -> bytes:
    packages_raw = metadata.get("packages")
    if not isinstance(packages_raw, list):
        raise ValueError("metadata has no packages")
    packages = {
        str(package["id"]): package
        for package in packages_raw
        if isinstance(package, dict)
    }
    closure = _normal_closure(metadata)
    if not closure <= packages.keys():
        raise ValueError("resolve graph references packages absent from metadata")
    ordered = sorted(
        (packages[package_id] for package_id in closure),
        key=lambda package: (str(package["name"]), str(package["version"])),
    )

    grouped: dict[str, list[tuple[str, str, str, bytes]]] = defaultdict(list)
    missing: list[str] = []
    roster: list[str] = []
    for package in ordered:
        name = str(package["name"])
        version = str(package["version"])
        license_expression = str(package.get("license") or "NOT DECLARED")
        roster.append(f"  {name} {version}: {license_expression}")
        files = _license_files(package)
        if not files:
            missing.append(f"  {name} {version}: {license_expression}")
        for path in files:
            payload = path.read_bytes().replace(b"\r\n", b"\n").rstrip() + b"\n"
            digest = hashlib.sha256(payload).hexdigest()
            grouped[digest].append((name, version, path.name, payload))

    lines = [HEADER.rstrip(), "", "PACKAGE AND SPDX INDEX", "----------------------"]
    lines.extend(roster)
    lines.extend(
        [
            "",
            "PACKAGES WITHOUT A STANDALONE LICENSE FILE IN THE CRATE ARCHIVE",
            "---------------------------------------------------------------",
        ]
    )
    lines.extend(missing or ["  (none)"])
    lines.extend(
        [
            "",
            "DEDUPLICATED LICENSE AND NOTICE TEXTS",
            "-------------------------------------",
        ]
    )
    for digest in sorted(grouped):
        entries = grouped[digest]
        labels = sorted(
            {
                f"{name} {version}/{filename}"
                for name, version, filename, _ in entries
            }
        )
        license_text = entries[0][3].decode("utf-8", errors="replace").rstrip()
        lines.extend(
            [
                "",
                f"sha256: {digest}",
                "applies to:",
                *(f"  {label}" for label in labels),
                "",
                license_text,
            ]
        )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    metadata = json.loads(arguments.metadata.read_text(encoding="utf-8"))
    payload = generate(_mapping(metadata, "metadata"))
    if arguments.check:
        if not arguments.output.is_file() or arguments.output.read_bytes() != payload:
            raise SystemExit(f"{arguments.output} is not the generated license bundle")
        return 0
    arguments.output.write_bytes(payload)
    print(f"wrote {arguments.output} ({len(payload)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
