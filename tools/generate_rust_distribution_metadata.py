#!/usr/bin/env python3
"""Generate the public Rust crate's deterministic CycloneDX and license bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SBOM = ROOT / "crates/toktier/data/sbom/toktier.cyclonedx.json"
LICENSES = ROOT / "crates/toktier/data/licenses/RUST_DEPENDENCY_LICENSES.txt"


def metadata() -> dict[str, Any]:
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--locked",
            "--format-version",
            "1",
            "--all-features",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, Any], json.loads(result.stdout))


def lock_checksums() -> dict[tuple[str, str], str]:
    module_name = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    parser = importlib.import_module(module_name)
    document = cast(
        dict[str, Any], parser.loads((ROOT / "Cargo.lock").read_text())
    )
    return {
        (row["name"], row["version"]): row["checksum"]
        for row in document["package"]
        if "checksum" in row
    }


def public_closure(
    document: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    packages = {row["id"]: row for row in document["packages"]}
    nodes = {row["id"]: row for row in document["resolve"]["nodes"]}
    roots = [
        row["id"]
        for row in document["packages"]
        if row["name"] == "toktier"
        and Path(row["manifest_path"]).resolve() == ROOT / "crates/toktier/Cargo.toml"
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one public toktier package, found {roots}")
    root = roots[0]
    seen: set[str] = set()
    queue = deque([root])
    graph: dict[str, list[str]] = {}
    while queue:
        package_id = queue.popleft()
        if package_id in seen:
            continue
        seen.add(package_id)
        dependencies: list[str] = []
        for dependency in nodes[package_id].get("deps", []):
            kinds = dependency.get("dep_kinds") or []
            if kinds and all(kind.get("kind") == "dev" for kind in kinds):
                continue
            target = dependency["pkg"]
            dependencies.append(target)
            queue.append(target)
        graph[package_id] = sorted(set(dependencies))
    return [packages[value] for value in sorted(seen)], graph


def bom_ref(package: dict[str, Any]) -> str:
    return f"pkg:cargo/{package['name']}@{package['version']}"


def generate_sbom(packages: list[dict[str, Any]], graph: dict[str, list[str]]) -> bytes:
    checksums = lock_checksums()
    by_id = {package["id"]: package for package in packages}
    components = []
    for package in sorted(packages, key=lambda row: (row["name"], row["version"])):
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": bom_ref(package),
            "name": package["name"],
            "version": package["version"],
            "purl": bom_ref(package),
        }
        if package.get("license"):
            component["licenses"] = [{"expression": package["license"]}]
        checksum = checksums.get((package["name"], package["version"]))
        if checksum:
            component["hashes"] = [{"alg": "SHA-256", "content": checksum}]
        source = package.get("source")
        if source:
            component["properties"] = [{"name": "cargo:source", "value": source}]
        components.append(component)
    root = next(package for package in packages if package["name"] == "toktier")
    root_component = next(
        component for component in components if component["bom-ref"] == bom_ref(root)
    )
    dependencies = [
        {
            "ref": bom_ref(package),
            "dependsOn": [
                bom_ref(by_id[target])
                for target in graph.get(package["id"], [])
                if target in by_id
            ],
        }
        for package in sorted(packages, key=lambda row: (row["name"], row["version"]))
    ]
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": root_component,
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "TokTier Rust distribution metadata generator",
                        "version": "1",
                    }
                ]
            },
        },
        "components": [
            component
            for component in components
            if component["bom-ref"] != bom_ref(root)
        ],
        "dependencies": dependencies,
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def license_candidates(package: dict[str, Any]) -> list[Path]:
    root = Path(package["manifest_path"]).parent
    patterns = ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*")
    paths = {
        path
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file() and path.stat().st_size <= 2 * 1024 * 1024
    }
    if package.get("license_file"):
        candidate = root / package["license_file"]
        if candidate.is_file():
            paths.add(candidate)
    return sorted(paths, key=lambda path: path.name)


def canonical_license_text(raw: bytes) -> bytes:
    """Normalize display-only whitespace while preserving license wording."""

    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    return ("\n".join(lines).rstrip("\n") + "\n").encode()


def generate_licenses(packages: list[dict[str, Any]]) -> bytes:
    texts: dict[str, bytes] = {}
    package_rows: list[tuple[str, str, list[str]]] = []
    for package in sorted(packages, key=lambda row: (row["name"], row["version"])):
        digests = []
        for path in license_candidates(package):
            raw = canonical_license_text(path.read_bytes())
            digest = hashlib.sha256(raw).hexdigest()
            texts.setdefault(digest, raw)
            digests.append(f"{path.name}=sha256:{digest}")
        package_rows.append(
            (
                f"{package['name']} {package['version']}",
                package.get("license") or "NOT DECLARED",
                digests,
            )
        )
    parts = [
        "TokTier Rust dependency license bundle\n",
        "Generated from the exact Cargo all-features non-dev dependency closure.\n",
        "Package declarations and included upstream license/notice texts follow.\n\n",
        "PACKAGE INDEX\n",
        "=============\n",
    ]
    for name, expression, digests in package_rows:
        rendered = ", ".join(digests) if digests else "no license file shipped by crate"
        parts.append(f"{name} | {expression} | {rendered}\n")
    for digest, raw in sorted(texts.items()):
        parts.extend(
            [
                "\n\nLICENSE TEXT sha256:",
                digest,
                "\n===============================================\n",
                raw.decode("utf-8", errors="replace"),
            ]
        )
        if not parts[-1].endswith("\n"):
            parts.append("\n")
    return "".join(parts).encode()


def write_or_check(path: Path, content: bytes, check: bool) -> bool:
    if check:
        if not path.is_file() or path.read_bytes() != content:
            print(f"error: {path} is missing or stale")
            return False
        print(f"{path}: check passed")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o644)
    print(f"{path}: updated")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    document = metadata()
    packages, graph = public_closure(document)
    okay = write_or_check(SBOM, generate_sbom(packages, graph), arguments.check)
    okay &= write_or_check(LICENSES, generate_licenses(packages), arguments.check)
    print(f"Rust distribution closure: {len(packages)} packages")
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
