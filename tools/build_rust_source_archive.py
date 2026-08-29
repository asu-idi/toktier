#!/usr/bin/env python3
"""Build a deterministic, fully vendored, offline Rust source archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_WORKSPACE_VERSION = re.compile(
    r"(?ms)^\[workspace\.package]\s*.*?^version\s*=\s*\"([^\"]+)\""
)


def _workspace_version() -> str:
    """The version this archive is named after.

    The crate inherits the workspace version, so the name is read from
    the manifest rather than restated here: a constant would have to be
    remembered at every release, and forgetting it names the archive
    after the previous one.
    """
    manifest = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = _WORKSPACE_VERSION.search(manifest)
    if match is None:
        raise SystemExit("Cargo.toml has no [workspace.package] version")
    return match.group(1)


VERSION = _workspace_version()
DEFAULT_OUTPUT = ROOT / f"dist/rust/toktier-rust-source-{VERSION}.tar.gz"
TOP = f"toktier-rust-source-{VERSION}"

ROOT_FILES = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "pyproject.toml",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES",
    "README.md",
    "README.zh-CN.md",
    # The generated PyPI long description. `pyproject.toml` travels in
    # this archive and names it, and it is the text a reader would
    # compare against the project's PyPI front page, so leaving it out
    # made the archive point at a file it did not carry.
    "README.pypi.md",
    "ARCHITECTURE.md",
    "ROADMAP.md",
    "CITATION.cff",
)
TREES = (
    "crates",
    "src/toktier",
    "data",
    "evidence",
    "schemas",
    "tables",
    "readings",
    "packaging",
    "docs",
    "tools",
)
INTERNAL_PACKAGES = (
    "toktier-cuda-driver",
    "toktier-gigatoken-core",
    "toktier-routing-core",
    "toktier-store-core",
    "toktier-store-sqlite",
)


def copy_inputs(destination: Path) -> None:
    for relative in ROOT_FILES:
        source = ROOT / relative
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for relative in TREES:
        source = ROOT / relative
        shutil.copytree(
            source,
            destination / relative,
            # Preserve links so the explicit archive audit below can reject
            # them. Dereferencing here could silently copy data from outside
            # the declared source tree.
            symlinks=True,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                "*.so",
                "*.dylib",
                "*.dll",
                "*.whl",
                "target",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
            ),
        )


def toml_string(value: str) -> str:
    """Return a TOML basic string using JSON's compatible escaping rules."""

    return json.dumps(value, ensure_ascii=False)


def dependency_line(dependency: dict[str, Any]) -> str:
    name = str(dependency.get("rename") or dependency["name"])
    fields = [f"version = {toml_string(str(dependency['req']))}"]
    if dependency.get("rename"):
        fields.append(f"package = {toml_string(str(dependency['name']))}")
    if dependency.get("optional"):
        fields.append("optional = true")
    if not dependency.get("uses_default_features", True):
        fields.append("default-features = false")
    features = dependency.get("features") or []
    if features:
        encoded = ", ".join(toml_string(str(feature)) for feature in features)
        fields.append(f"features = [{encoded}]")
    return f"{name} = {{ {', '.join(fields)} }}"


def normalized_manifest(package: dict[str, Any]) -> str:
    """Build the registry-style manifest Cargo expects in a directory source.

    Cargo normalizes path dependencies to registry dependencies when packaging.
    The source archive therefore needs workspace crates in the vendored source as
    packages, rather than only as workspace paths.  Development dependencies are
    deliberately omitted: these copies are dependency inputs for the packaged
    public crate, while the complete workspace (including its tests) remains next
    to them in ``crates/`` and is tested separately.
    """

    lines = [
        "# Automatically generated for TokTier's offline directory source.",
        "# The original workspace manifest is preserved as Cargo.toml.orig.",
        "",
        "[package]",
        f"name = {toml_string(str(package['name']))}",
        f"version = {toml_string(str(package['version']))}",
        f"edition = {toml_string(str(package['edition']))}",
    ]
    for field in ("rust_version", "license", "repository", "description"):
        value = package.get(field)
        if value:
            cargo_field = "rust-version" if field == "rust_version" else field
            lines.append(f"{cargo_field} = {toml_string(str(value))}")
    readme = package.get("readme")
    if readme:
        lines.append(f"readme = {toml_string(Path(str(readme)).name)}")

    features = package.get("features") or {}
    if features:
        lines.extend(("", "[features]"))
        for name, members in sorted(features.items()):
            encoded = ", ".join(toml_string(str(member)) for member in members)
            lines.append(f"{name} = [{encoded}]")

    groups: dict[tuple[str | None, str], list[dict[str, Any]]] = {}
    for raw_dependency in package.get("dependencies") or []:
        dependency = dict(raw_dependency)
        kind = str(dependency.get("kind") or "normal")
        if kind == "dev":
            continue
        target = dependency.get("target")
        key = (str(target) if target else None, kind)
        groups.setdefault(key, []).append(dependency)
    for (target, kind), dependencies in sorted(
        groups.items(), key=lambda item: ((item[0][0] or ""), item[0][1])
    ):
        section = "build-dependencies" if kind == "build" else "dependencies"
        if target:
            lines.extend(("", f"[target.{toml_string(target)}.{section}]"))
        else:
            lines.extend(("", f"[{section}]"))
        for dependency in sorted(
            dependencies, key=lambda item: str(item.get("rename") or item["name"])
        ):
            lines.append(dependency_line(dependency))
    return "\n".join(lines) + "\n"


def add_internal_packages(destination: Path, vendor_root: Path) -> None:
    completed = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cargo metadata failed:\n{completed.stderr}")
    metadata = json.loads(completed.stdout)
    packages = {package["name"]: package for package in metadata["packages"]}
    for name in INTERNAL_PACKAGES:
        package = packages.get(name)
        if package is None:
            raise RuntimeError(f"workspace metadata is missing internal package {name}")
        version = str(package["version"])
        source = destination / "crates" / name
        target = vendor_root / f"{name}-{version}"
        shutil.copytree(source, target, symlinks=True)
        original = target / "Cargo.toml"
        shutil.copyfile(original, target / "Cargo.toml.orig")
        original.write_text(normalized_manifest(package))

        checksums = {}
        for relative in files(target):
            if relative.name == ".cargo-checksum.json":
                continue
            checksums[relative.as_posix()] = hashlib.sha256(
                (target / relative).read_bytes()
            ).hexdigest()
        (target / ".cargo-checksum.json").write_text(
            json.dumps(
                {"files": checksums, "package": None},
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def vendor(destination: Path) -> None:
    vendor_root = destination / "vendor"
    completed = subprocess.run(
        [
            "cargo",
            "vendor",
            "--locked",
            "--versioned-dirs",
            str(vendor_root),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cargo vendor failed:\n{completed.stderr}")
    add_internal_packages(destination, vendor_root)
    config = destination / ".cargo/config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "[source.crates-io]\n"
        'replace-with = "vendored-sources"\n\n'
        "[source.vendored-sources]\n"
        'directory = "vendor"\n\n'
        "[net]\n"
        "offline = true\n"
    )


def files(root: Path) -> list[Path]:
    output = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"source archive refuses symlink {path}")
        if path.is_file():
            output.append(path.relative_to(root))
    return sorted(output, key=lambda path: path.as_posix())


def add_manifest(root: Path) -> None:
    rows = []
    digest = hashlib.sha256(b"toktier.rust_source_archive.v1\0")
    for relative in files(root):
        raw = (root / relative).read_bytes()
        name = relative.as_posix().encode()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        )
    document = {
        "schema": "toktier.rust_source_archive.v1",
        "files": rows,
        "root_digest": "sha256:" + digest.hexdigest(),
    }
    (root / "SOURCE-MANIFEST.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )


def normalized_mode(path: Path) -> int:
    executable = path.suffix in {".sh", ".py"} or path.name in {
        "configure",
        "config.guess",
        "config.sub",
    }
    return 0o755 if executable else 0o644


def build_archive(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with (
        open(temporary, "wb") as raw_output,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, mtime=0
        ) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
        ) as archive,
    ):
        for relative in files(staging):
            source = staging / relative
            info = tarfile.TarInfo(f"{TOP}/{relative.as_posix()}")
            info.size = source.stat().st_size
            info.mode = normalized_mode(source)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with open(source, "rb") as handle:
                archive.addfile(info, handle)
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="toktier-rust-source-") as temporary:
        staging = Path(temporary) / TOP
        staging.mkdir()
        copy_inputs(staging)
        vendor(staging)
        add_manifest(staging)
        build_archive(staging, arguments.output)
    digest = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
    size = arguments.output.stat().st_size
    print(f"{arguments.output}: sha256:{digest} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
