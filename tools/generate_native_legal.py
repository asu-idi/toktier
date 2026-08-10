#!/usr/bin/env python3
"""Generate the native extension's locked SBOM and license bundle.

The wheel contains one Rust extension, ``toktier._native``.  Its legal
artifacts must therefore describe the dependency closure rooted at
``toktier-py`` rather than a historical standalone Gigatoken wheel.  The walk
follows every activated normal edge on every target in Cargo's locked resolve
graph.  That is deliberately a conservative superset of the Linux x86-64
binary: target-only crates and procedural macros may remain in the accounting
record, but a linked dependency can never disappear from it silently.

The historical filenames under ``packaging/fast_cpu`` are stable distribution
paths.  Their generated contents describe the complete integrated native
extension, including the corrected Gigatoken core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ROOT_PACKAGE = "toktier-py"
SBOM_PATH = ROOT / "packaging/fast_cpu/gigatoken.cyclonedx.json"
LICENSE_BUNDLE_PATH = (
    ROOT / "packaging/fast_cpu/THIRD_PARTY_LICENSES-gigatoken.txt"
)
TOOL_NAME = "tools/generate_native_legal.py"
TOOL_VERSION = "1.1.0"

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
    "boost",
)

# Recursive sweep: crates keep mandatory notices for embedded third-party
# sources in subdirectories (oniguruma/COPYING, licenses/LICENSE-yyjson,
# src/unicode_tables/LICENSE-UNICODE, ...). Filenames beginning with these
# prefixes are collected anywhere below the crate root, skipping build
# ``target`` directories. Collecting too much is acceptable here; missing a
# mandatory notice is not.
_LICENSE_FILE_PREFIXES = ("license", "licence", "copying", "notice")

# Bundle-entry annotations, keyed by (package name, crate-relative label).
_LABEL_NOTES: dict[tuple[str, str], str] = {
    ("libsqlite3-sys", "sqlcipher/LICENSE"): (
        "collected conservatively; the sqlcipher feature is not enabled "
        "in this build, and the statically linked SQLite is public domain"
    ),
}

_BLAKE2B_SIMD_CURATED = """\
Curated statement for blake2b_simd 1.0.3: the published crate archive ships
no standalone license file, and the Cargo metadata declares MIT. The MIT
license text below is reproduced from the upstream repository
https://github.com/oconnor663/blake2_simd (file LICENSE at commit
48306863ceb221f75f9b82d66f412222601f5f58, sha256
27e0387973d6b8507cb15f825f6f26e0278d4d5857082c1f56b49a7b39a90183).

MIT License

Copyright (c) 2018 Jack O'Connor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

_ESAXX_SAIS_CURATED = """\
Curated statement for esaxx-rs 0.1.10: the crate archive carries an
Apache-2.0 license file at its root, and its embedded C++ sources carry
their own MIT declarations in the source headers. The declaration block
below is reproduced in full from the crate's src/sais.hxx.

/*
 * sais.hxx for sais-lite
 * Copyright (c) 2008-2009 Yuta Mori All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
"""

_ESAXX_ESA_CURATED = """\
Curated statement for esaxx-rs 0.1.10: the declaration block below is
reproduced in full from the crate's src/esa.hxx.

/*
 * esa.hxx
 * Copyright (c) 2010 Daisuke Okanohara All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
"""

# Curated bundle entries, keyed by (package name, version). They cover the
# cases a file walk cannot reach: crate archives that ship no standalone
# license file, and embedded sources whose declarations live in code
# headers rather than in license-named files.
_CURATED_TEXTS: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("blake2b_simd", "1.0.3"): (
        ("upstream LICENSE [curated]", _BLAKE2B_SIMD_CURATED),
    ),
    ("esaxx-rs", "0.1.10"): (
        ("src/sais.hxx notice [curated]", _ESAXX_SAIS_CURATED),
        ("src/esa.hxx notice [curated]", _ESAXX_ESA_CURATED),
    ),
}

# Non-Cargo components statically registered for the SBOM: the prebuilt
# fatbin (src/toktier/kernels/prebuilt/pretok_kernel.fatbin) contains device
# code instantiated from NVIDIA CCCL 3.2.0 headers by the CUDA 13.2
# toolchain. These entries record that embedded material explicitly, since
# a Cargo resolve walk cannot see it. The corresponding license material is
# reproduced in the repository-level THIRD_PARTY_NOTICES, section 6.
_EMBEDDED_FATBIN_COMPONENTS: tuple[dict[str, Any], ...] = (
    {
        "type": "library",
        "bom-ref": "pkg:github/nvidia/cccl@v3.2.0#cub",
        "name": "CUB (NVIDIA CCCL)",
        "version": "3.2.0",
        "scope": "required",
        "purl": "pkg:github/nvidia/cccl@v3.2.0#cub",
        "description": (
            "CUB device code embedded in the prebuilt CUDA fatbin via "
            "template instantiation (DeviceSelectSweepKernel and "
            "DeviceScanKernel symbols in the _V_300200 namespace)."
        ),
        "licenses": [{"expression": "BSD-3-Clause"}],
        "properties": [
            {
                "name": "toktier:embedded:channel",
                "value": "src/toktier/kernels/prebuilt/pretok_kernel.fatbin",
            },
            {
                "name": "toktier:embedded:basis",
                "value": (
                    "curated static registration of fatbin-embedded device "
                    "code; CUDA 13.2 toolchain headers, CUB_VERSION 300200"
                ),
            },
            {
                "name": "toktier:embedded:license-source",
                "value": "https://github.com/NVIDIA/cccl/blob/v3.2.0/LICENSE",
            },
        ],
    },
    {
        "type": "library",
        "bom-ref": "pkg:github/nvidia/cccl@v3.2.0#thrust",
        "name": "Thrust (NVIDIA CCCL)",
        "version": "3.2.0",
        "scope": "required",
        "purl": "pkg:github/nvidia/cccl@v3.2.0#thrust",
        "description": (
            "Thrust device code embedded in the prebuilt CUDA fatbin via "
            "template instantiation (counting_iterator template arguments "
            "in the _V_300200 namespace)."
        ),
        "licenses": [
            {
                "license": {
                    "name": (
                        "Apache-2.0 with the specific exceptions listed in "
                        "the combined CCCL license text"
                    )
                }
            }
        ],
        "properties": [
            {
                "name": "toktier:embedded:channel",
                "value": "src/toktier/kernels/prebuilt/pretok_kernel.fatbin",
            },
            {
                "name": "toktier:embedded:basis",
                "value": (
                    "curated static registration of fatbin-embedded device "
                    "code; CUDA 13.2 toolchain headers, CCCL 3.2.0"
                ),
            },
            {
                "name": "toktier:embedded:license-source",
                "value": "https://github.com/NVIDIA/cccl/blob/v3.2.0/LICENSE",
            },
        ],
    },
    {
        "type": "library",
        "bom-ref": "pkg:github/nvidia/cccl@v3.2.0#libcudacxx",
        "name": "libcu++ (NVIDIA CCCL)",
        "version": "3.2.0",
        "scope": "required",
        "purl": "pkg:github/nvidia/cccl@v3.2.0#libcudacxx",
        "description": (
            "libcu++ support headers used by the kernel translation unit "
            "that produced the prebuilt CUDA fatbin."
        ),
        "licenses": [{"expression": "Apache-2.0 WITH LLVM-exception"}],
        "properties": [
            {
                "name": "toktier:embedded:channel",
                "value": "src/toktier/kernels/prebuilt/pretok_kernel.fatbin",
            },
            {
                "name": "toktier:embedded:basis",
                "value": (
                    "curated static registration of fatbin-embedded device "
                    "code; CUDA 13.2 toolchain headers, CCCL 3.2.0"
                ),
            },
            {
                "name": "toktier:embedded:license-source",
                "value": "https://github.com/NVIDIA/cccl/blob/v3.2.0/LICENSE",
            },
        ],
    },
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


class ToolUnavailable(Exception):
    """A prerequisite of this tool is not installed."""


def _locked_metadata() -> dict[str, Any]:
    """Cargo's locked resolve graph for the workspace.

    The legal artifacts describe a Rust dependency closure, so ``cargo``
    is a hard prerequisite of both generating and checking them. Saying
    so plainly is more useful than the bare ``FileNotFoundError`` a
    missing executable would otherwise raise: it names what is missing
    and what to do about it.
    """
    command = ["cargo", "metadata", "--offline", "--locked", "--format-version", "1"]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise ToolUnavailable(
            "this tool requires cargo: it reads the workspace's locked "
            "resolve graph with `cargo metadata --offline --locked`. "
            "Install the Rust toolchain pinned by rust-toolchain.toml "
            "(https://rustup.rs) and re-run. There is no cargo-free mode: "
            "the shipped SBOM and license bundle describe a Rust "
            "dependency closure and cannot be verified without it."
        ) from error
    except subprocess.CalledProcessError as error:
        raise ToolUnavailable(
            "`cargo metadata --offline --locked` failed; the workspace "
            "resolve graph could not be read. From a fresh checkout this "
            "usually means the local Cargo cache is not populated yet: "
            "run `cargo fetch --locked` once (network required; a plain "
            "`cargo build` is not enough, since the legal closure covers "
            "every target), then retry. cargo reported:\n"
            f"{error.stderr.strip()}"
        ) from error
    return _mapping(json.loads(completed.stdout), "cargo metadata")


def _root_id(metadata: dict[str, Any]) -> str:
    matches = [
        str(package["id"])
        for package in metadata.get("packages", [])
        if isinstance(package, dict)
        and package.get("name") == ROOT_PACKAGE
        and package.get("source") is None
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one workspace package {ROOT_PACKAGE!r}, found {len(matches)}"
        )
    return matches[0]


def _normal_closure(metadata: dict[str, Any]) -> set[str]:
    resolve = _mapping(metadata.get("resolve"), "resolve")
    nodes_raw = resolve.get("nodes")
    if not isinstance(nodes_raw, list):
        raise ValueError("metadata has no resolve nodes")
    nodes = {
        str(node["id"]): node for node in nodes_raw if isinstance(node, dict)
    }
    seen: set[str] = set()
    pending = [_root_id(metadata)]
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


def _packages(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = metadata.get("packages")
    if not isinstance(raw, list):
        raise ValueError("metadata has no packages")
    return {
        str(package["id"]): package
        for package in raw
        if isinstance(package, dict)
    }


def _nodes(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    resolve = _mapping(metadata.get("resolve"), "resolve")
    raw = resolve.get("nodes")
    if not isinstance(raw, list):
        raise ValueError("metadata has no resolve nodes")
    return {str(node["id"]): node for node in raw if isinstance(node, dict)}


def _lock_checksums() -> dict[tuple[str, str, str], str]:
    """Read package checksums from Cargo's generated, locked TOML shape.

    Avoiding a TOML runtime dependency keeps this repository check runnable in
    the Rust-only CI job and on Python 3.10. Cargo itself remains the authority:
    ``cargo metadata --locked`` has already parsed and accepted this file.
    """
    document = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    result: dict[tuple[str, str, str], str] = {}
    for block in document.split("[[package]]")[1:]:
        fields = dict(
            re.findall(r'^([a-z_]+)\s*=\s*"([^"\r\n]*)"\s*$', block, re.MULTILINE)
        )
        if all(key in fields for key in ("name", "version", "source", "checksum")):
            result[(fields["name"], fields["version"], fields["source"])] = fields[
                "checksum"
            ]
    return result


def _bom_ref(package: dict[str, Any]) -> str:
    """Checkout-path-independent reference: the canonical Cargo purl.

    Cargo's raw package IDs embed ``path+file:///...`` for workspace
    members, so hashing them binds the document to one absolute checkout
    path and the shipped SBOM can never re-verify from another one. The
    purl (``pkg:cargo/name@version``) is derived from name and version
    only; :func:`generate_sbom` refuses to emit a document in which two
    closure packages would share one reference.
    """
    return _purl(package)


def _purl(package: dict[str, Any]) -> str:
    name = urllib.parse.quote(str(package["name"]), safe="")
    version = urllib.parse.quote(str(package["version"]), safe=".+-")
    return f"pkg:cargo/{name}@{version}"


def _spdx_expression(value: object) -> str | None:
    """Normalize legacy Cargo slash spelling into an SPDX ``OR`` expression."""
    if not isinstance(value, str) or not value:
        return None
    return re.sub(r"\s*/\s*", " OR ", value)


def _component(
    package: dict[str, Any], checksums: dict[tuple[str, str, str], str]
) -> dict[str, Any]:
    component: dict[str, Any] = {
        "type": "library",
        "bom-ref": _bom_ref(package),
        "name": str(package["name"]),
        "version": str(package["version"]),
        "scope": "required",
        "purl": _purl(package),
    }
    description = package.get("description")
    if isinstance(description, str) and description:
        component["description"] = description
    license_expression = _spdx_expression(package.get("license"))
    if license_expression is not None:
        component["licenses"] = [{"expression": license_expression}]
    source = package.get("source")
    if isinstance(source, str):
        checksum = checksums.get(
            (str(package["name"]), str(package["version"]), source)
        )
        if checksum is not None:
            component["hashes"] = [{"alg": "SHA-256", "content": checksum}]
        component["properties"] = [{"name": "toktier:cargo:source", "value": source}]
    else:
        component["properties"] = [
            {"name": "toktier:cargo:source", "value": "workspace"}
        ]
    return component


def generate_sbom(metadata: dict[str, Any]) -> bytes:
    packages = _packages(metadata)
    nodes = _nodes(metadata)
    closure = _normal_closure(metadata)
    if not closure <= packages.keys() or not closure <= nodes.keys():
        raise ValueError("resolve graph references packages absent from metadata")
    root_id = _root_id(metadata)
    checksums = _lock_checksums()
    ordered_ids = sorted(
        closure,
        key=lambda package_id: (
            str(packages[package_id]["name"]),
            str(packages[package_id]["version"]),
            package_id,
        ),
    )
    references: dict[str, str] = {}
    for package_id in ordered_ids:
        reference = _bom_ref(packages[package_id])
        previous = references.get(reference)
        if previous is not None:
            raise ValueError(
                f"bom-ref collision: packages {previous!r} and "
                f"{package_id!r} both render as {reference!r}; the SBOM "
                "reference form needs a disambiguating component before "
                "this document can be generated"
            )
        references[reference] = package_id
    for embedded in _EMBEDDED_FATBIN_COMPONENTS:
        reference = str(embedded["bom-ref"])
        if reference in references:
            raise ValueError(
                f"bom-ref collision: curated embedded component {reference!r} "
                "clashes with a Cargo closure package"
            )
        references[reference] = f"embedded:{embedded['name']}"
    dependencies = []
    for package_id in ordered_ids:
        direct = []
        for dependency in nodes[package_id].get("deps", []):
            if not isinstance(dependency, dict):
                continue
            kinds = dependency.get("dep_kinds")
            if not isinstance(kinds, list) or not any(
                isinstance(kind, dict) and kind.get("kind") in (None, "normal")
                for kind in kinds
            ):
                continue
            child = str(dependency["pkg"])
            if child in closure:
                direct.append(_bom_ref(packages[child]))
        row: dict[str, Any] = {"ref": _bom_ref(packages[package_id])}
        if direct:
            row["dependsOn"] = sorted(set(direct))
        dependencies.append(row)
    root = _component(packages[root_id], checksums)
    root["type"] = "application"
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "tools": [
                {
                    "vendor": "TokTier",
                    "name": TOOL_NAME,
                    "version": TOOL_VERSION,
                }
            ],
            "component": root,
            "properties": [
                {"name": "cdx:rustc:sbom:target:all_targets", "value": "true"},
                {"name": "toktier:cargo:lock", "value": "Cargo.lock"},
                {
                    "name": "toktier:cargo:root-package",
                    "value": ROOT_PACKAGE,
                },
            ],
        },
        "components": [
            _component(packages[package_id], checksums)
            for package_id in ordered_ids
            if package_id != root_id
        ]
        + [dict(embedded) for embedded in _EMBEDDED_FATBIN_COMPONENTS],
        "dependencies": dependencies,
    }
    return (json.dumps(document, indent=2, ensure_ascii=True) + "\n").encode()


def _license_files(package: dict[str, Any]) -> list[Path]:
    root = Path(str(package["manifest_path"])).resolve().parent
    paths = [
        path
        for path in root.iterdir()
        if path.is_file()
        and any(marker in path.name.lower() for marker in _LICENSE_NAME_MARKERS)
    ]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "target" in path.relative_to(root).parts[:-1]:
            continue
        if path.name.lower().startswith(_LICENSE_FILE_PREFIXES):
            paths.append(path)
    license_file = package.get("license_file")
    if isinstance(license_file, str):
        explicit = (root / license_file).resolve()
        if explicit.is_file():
            paths.append(explicit)
    name = str(package["name"])
    if package.get("source") is None and name.startswith("toktier-"):
        paths.append(ROOT / "LICENSE")
    if name == "toktier-gigatoken-core":
        paths.extend(
            [
                ROOT / "packaging/fast_cpu/LICENSE-gigatoken",
                ROOT / "packaging/fast_cpu/NOTICE-gigatoken-pinned",
            ]
        )
    return sorted(
        {path for path in paths if path.is_file()},
        key=lambda path: path.as_posix().lower(),
    )


def generate_license_bundle(metadata: dict[str, Any]) -> bytes:
    packages = _packages(metadata)
    closure = _normal_closure(metadata)
    ordered = sorted(
        (packages[package_id] for package_id in closure),
        key=lambda package: (
            str(package["name"]),
            str(package["version"]),
            str(package["id"]),
        ),
    )
    header = """THIRD-PARTY LICENSE BUNDLE FOR TOKTIER'S INTEGRATED NATIVE EXTENSION
============================================================================

Generated from the repository's locked Cargo metadata with toktier-py as the
root package. The dependency walk includes every activated normal edge on
every target, so it is a conservative superset of the Linux x86-64 wheel's
linked dependency closure. It covers the corrected Gigatoken core, native HF
reference, Rust router/store, and CUDA Driver host compiled into the single
toktier._native extension.

The package/version roster and SPDX expressions below are the accounting
index. License and notice files found anywhere in the corresponding
published crate sources (recursively, skipping build target directories)
follow, normalized to LF with trailing whitespace removed and then
deduplicated by content. Packages whose crate archive has no standalone
license file are called out separately and are covered by the matching
standard text elsewhere in this bundle or THIRD_PARTY_NOTICES. Curated
statements, marked [curated], carry material a file walk cannot reach:
upstream license texts that are absent from the crate archive, and
declarations embedded in source headers.
"""
    grouped: dict[str, list[tuple[str, str, str, bytes]]] = defaultdict(list)
    missing: list[str] = []
    roster: list[str] = []
    for package in ordered:
        name = str(package["name"])
        version = str(package["version"])
        license_expression = _spdx_expression(package.get("license")) or "NOT DECLARED"
        source = "crates.io" if package.get("source") is not None else "workspace"
        roster.append(f"  {name} {version} [{source}]: {license_expression}")
        crate_root = Path(str(package["manifest_path"])).resolve().parent
        files = _license_files(package)
        curated = _CURATED_TEXTS.get((name, version), ())
        if not files:
            row = f"  {name} {version}: {license_expression}"
            if curated:
                row += " (a curated upstream statement is included below)"
            missing.append(row)
        for path in files:
            try:
                label = path.relative_to(crate_root).as_posix()
            except ValueError:
                label = path.name
            note = _LABEL_NOTES.get((name, label))
            if note is not None:
                label = f"{label} ({note})"
            normalized = path.read_bytes().replace(b"\r\n", b"\n")
            payload = (
                b"\n".join(line.rstrip() for line in normalized.split(b"\n")).rstrip()
                + b"\n"
            )
            digest = hashlib.sha256(payload).hexdigest()
            grouped[digest].append((name, version, label, payload))
        for label, text in curated:
            payload = (
                b"\n".join(
                    line.rstrip() for line in text.encode().split(b"\n")
                ).rstrip()
                + b"\n"
            )
            digest = hashlib.sha256(payload).hexdigest()
            grouped[digest].append((name, version, label, payload))

    lines = [header.rstrip(), "", "PACKAGE AND SPDX INDEX", "----------------------"]
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
                for name, version, filename, _payload in entries
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
    return ("\n".join(lines).rstrip() + "\n").encode()


def _write_or_check(path: Path, payload: bytes, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_bytes() != payload:
            raise SystemExit(f"{path}: generated native legal artifact drifted")
        print(f"{path}: check passed")
        return
    path.write_bytes(payload)
    print(f"wrote {path} ({len(payload)} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        metadata = _locked_metadata()
    except ToolUnavailable as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    _write_or_check(SBOM_PATH, generate_sbom(metadata), arguments.check)
    _write_or_check(
        LICENSE_BUNDLE_PATH,
        generate_license_bundle(metadata),
        arguments.check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
