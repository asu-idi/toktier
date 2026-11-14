#!/usr/bin/env python3
"""Derive the property-data guard table of the Rust API crate.

The fast CPU pre-tokenizer reads its character classes (General_Category,
White_Space, Script=Han) from whichever ``icu_properties`` and
``icu_properties_data`` copies the build resolves, and the workspace
requirement on ``icu_properties`` is a caret one. The judged build resolves
one pair; a consumer whose lockfile resolves another pair gets classes that
can differ from the judged ones on some code points, and with them ids that
can differ from the reference engine's. The guard in
``crates/toktier/src/property_guard.rs`` sends every text that holds such a
code point to the reference engine. This tool produces the two documents it
rests on:

* ``crates/toktier/src/property_guard_table.rs`` -- the admitted copies of
  both crates as ``(name, version, registry checksum)``, the judged pair
  marked, and one set of code points as sorted inclusive ranges: the union,
  over every admitted pair, of the code points whose engine classes differ
  from the judged pair's.
* ``tools/property_guard_manifest.json`` -- the evidence behind the table:
  per pair the versions and checksums, the sha256 of its dump, its own
  differing set as ranges with their counts, and the packages its dump
  build resolved; the generation date and the index state read.

How a pair is judged. For every published ``icu_properties`` version the
workspace requirement admits, and every ``icu_properties_data`` version
that version's own requirement admits, the tool builds
``tools/property_dump`` with exactly that pair pinned in a lockfile of its
own. That program compiles the engine's
``crates/toktier-gigatoken-core/src/pretokenize/unicode.rs`` unchanged and
writes, for every code point below 0x110000, the value of each class
accessor the engine uses. A code point is in a pair's differing set when
any of its six values differs from the judged pair's dump. The comparison
is exhaustive, so outside the union every admitted pair and the judged pair
give the engine the same classes. A pair whose dump cannot be built is left
off the list, which makes a build resolving it decline the fast route.

Usage::

    python tools/generate_property_guard.py --work-dir DIR   # regenerate both
    python tools/generate_property_guard.py --work-dir DIR --check
    python tools/generate_property_guard.py --check-offline

``--check`` regenerates in memory and compares with the files on disk; the
table must be byte-for-byte equal and the manifest equal apart from the
generation date, the raw index digests and the packages each dump build
resolved beside its pair (reported when they move; the pair list derived
from the index and every dump digest are compared). It needs ``cargo`` and
the registry (network, or a populated ``CARGO_HOME`` with ``--offline``).
``--check-offline`` needs neither: it re-renders the table from the
manifest's per-pair ranges and checks the manifest's own sums, the judged
pair against ``Cargo.lock``, the requirement against ``Cargo.toml``, and
the engine's ``unicode.rs`` and the dump program against the digests the
manifest records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "crates" / "toktier" / "src" / "property_guard_table.rs"
MANIFEST = ROOT / "tools" / "property_guard_manifest.json"
DUMP_CRATE = ROOT / "tools" / "property_dump"
DUMP_MAIN = DUMP_CRATE / "src" / "main.rs"
ENGINE_UNICODE = (
    ROOT / "crates" / "toktier-gigatoken-core" / "src" / "pretokenize" / "unicode.rs"
)

SCHEMA = "toktier.property_guard_manifest.v1"
GENERATED_BY = "tools/generate_property_guard.py"
PROPERTIES = "icu_properties"
DATA = "icu_properties_data"
CRATES = (PROPERTIES, DATA)
REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
INDEX_URL = "https://index.crates.io/{prefix}/{name}"

CODE_POINTS = 0x110000
COLUMNS = (
    "class_of",
    "ds_class_of",
    "o200k_class_of",
    "kimi_class_of",
    "general_category",
    "white_space",
)
GENERAL_CATEGORY = COLUMNS.index("general_category")
#: ICU4X's numbering of General_Category: 0 is Unassigned (Cn).
UNASSIGNED = 0

#: Manifest keys that differ between two runs over the same evidence.
VOLATILE_KEYS = ("generated_at", "index_state")


class GuardError(RuntimeError):
    """A condition under which no table may be written."""


# ---------------------------------------------------------------------
# Versions and requirements (the subset of Cargo's rules these crates use)
# ---------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int
    patch: int
    pre: str = ""

    @classmethod
    def parse(cls, text: str) -> Version:
        match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?", text)
        if match is None:
            raise GuardError(f"unreadable version {text!r}")
        return cls(int(match[1]), int(match[2]), int(match[3]), match[4] or "")

    def __str__(self) -> str:
        core = f"{self.major}.{self.minor}.{self.patch}"
        return f"{core}-{self.pre}" if self.pre else core


def admits(requirement: str, version: Version) -> bool:
    """Whether a Cargo requirement admits a version.

    Only the forms these two crates use are read (caret, tilde, exact, one
    comparator); anything else is refused rather than guessed. A
    pre-release is never admitted, which is Cargo's rule for requirements
    without a pre-release part.
    """
    text = requirement.strip()
    if "," in text or " " in text:
        raise GuardError(f"requirement {requirement!r} is not one comparator")
    if version.pre:
        return False
    if text[:1] in ("=", "^", "~"):
        operator, rest = text[0], text[1:].strip()
    else:
        operator, rest = "^", text
    parts = rest.split(".")
    if not 1 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        raise GuardError(f"requirement {requirement!r} is not read by this tool")
    numbers = [int(part) for part in parts]
    padded = numbers + [0] * (3 - len(numbers))
    floor = Version(padded[0], padded[1], padded[2])
    key = (version.major, version.minor, version.patch)
    low = (floor.major, floor.minor, floor.patch)
    if operator == "=":
        return key[: len(numbers)] == tuple(numbers)
    if key < low:
        return False
    if operator == "~":
        if len(numbers) == 1:
            return version.major == floor.major
        return (version.major, version.minor) == (floor.major, floor.minor)
    # caret
    if floor.major > 0 or len(numbers) == 1:
        return version.major == floor.major
    if floor.minor > 0 or len(numbers) == 2:
        return (version.major, version.minor) == (0, floor.minor)
    return key == low


# ---------------------------------------------------------------------
# Inputs from the tree
# ---------------------------------------------------------------------


def workspace_requirement(manifest_text: str) -> str:
    """The workspace's requirement on ``icu_properties``."""
    match = re.search(
        r'^icu_properties\s*=\s*\{[^}\n]*\bversion\s*=\s*"([^"]+)"',
        manifest_text,
        re.MULTILINE,
    )
    if match is None:
        raise GuardError("Cargo.toml names no workspace icu_properties version")
    return match[1]


def lock_packages(text: str) -> list[dict[str, str]]:
    """The ``[[package]]`` blocks of a lockfile, as flat string maps."""
    packages: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            if current is not None:
                packages.append(current)
            current = {} if stripped == "[[package]]" else None
            continue
        if current is None:
            continue
        match = re.fullmatch(
            r'(name|version|source|checksum)\s*=\s*"([^"]*)"', stripped
        )
        if match:
            current[match[1]] = match[2]
    if current is not None:
        packages.append(current)
    return packages


def copies_in_lock(text: str, name: str) -> list[dict[str, str]]:
    return [package for package in lock_packages(text) if package.get("name") == name]


def judged_pair(lock_text: str) -> dict[str, dict[str, str]]:
    """The one judged copy of each crate, from the workspace lockfile."""
    pair: dict[str, dict[str, str]] = {}
    for name in CRATES:
        found = copies_in_lock(lock_text, name)
        if len(found) != 1:
            raise GuardError(f"Cargo.lock holds {len(found)} copies of {name}, not one")
        copy = found[0]
        if copy.get("source") != REGISTRY_SOURCE or not copy.get("checksum"):
            raise GuardError(
                f"Cargo.lock's {name} is not a registry copy with a checksum"
            )
        pair[name] = {"version": copy["version"], "checksum": copy["checksum"]}
    return pair


def toolchain_channel() -> str:
    text = (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    match = re.search(r'^channel\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise GuardError("rust-toolchain.toml names no channel")
    return match[1]


# ---------------------------------------------------------------------
# The registry index
# ---------------------------------------------------------------------


def index_prefix(name: str) -> str:
    if len(name) <= 2:
        return str(len(name))
    if len(name) == 3:
        return f"3/{name[0]}"
    return f"{name[:2]}/{name[2:4]}"


def read_index(name: str, index_dir: Path | None) -> bytes:
    if index_dir is not None:
        return (index_dir / name).read_bytes()
    url = INDEX_URL.format(prefix=index_prefix(name), name=name)
    request = urllib.request.Request(url, headers={"User-Agent": "property-guard-tool"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body: bytes = response.read()
    return body


@dataclass(frozen=True)
class Release:
    name: str
    version: Version
    checksum: str
    yanked: bool
    data_requirement: str | None


def releases(name: str, raw: bytes) -> list[Release]:
    found: list[Release] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("name") != name:
            raise GuardError(f"the {name} index holds an entry for {entry.get('name')}")
        requirement = None
        for dependency in entry.get("deps", []):
            if dependency.get("name") == DATA and dependency.get("kind") in (
                None,
                "normal",
            ):
                requirement = str(dependency["req"])
        found.append(
            Release(
                name=name,
                version=Version.parse(entry["vers"]),
                checksum=str(entry["cksum"]),
                yanked=bool(entry.get("yanked", False)),
                data_requirement=requirement,
            )
        )
    return found


def admitted_pairs(
    requirement: str, properties: Sequence[Release], data: Sequence[Release]
) -> list[tuple[Release, Release]]:
    """Every pair Cargo could resolve under the workspace requirement.

    Yanked releases are kept: a lockfile written before a release was
    yanked keeps resolving it.
    """
    pairs: list[tuple[Release, Release]] = []
    for release in sorted(properties, key=lambda item: item.version):
        if not admits(requirement, release.version):
            continue
        if release.data_requirement is None:
            raise GuardError(
                f"{PROPERTIES} {release.version} names no {DATA} requirement"
            )
        for candidate in sorted(data, key=lambda item: item.version):
            if admits(release.data_requirement, candidate.version):
                pairs.append((release, candidate))
    return pairs


# ---------------------------------------------------------------------
# Building and reading dumps
# ---------------------------------------------------------------------


def pair_label(pair: tuple[Release, Release]) -> str:
    return f"{pair[0].version}+{pair[1].version}"


def dump_manifest(properties_version: Version) -> str:
    return (
        "[package]\n"
        'name = "toktier-property-dump"\n'
        'version = "0.0.0"\n'
        'edition = "2024"\n'
        "publish = false\n\n"
        "[workspace]\n\n"
        "[features]\n"
        "upstream-tests = []\n\n"
        "[[bin]]\n"
        'name = "toktier-property-dump"\n'
        f"path = {json.dumps(str(DUMP_MAIN))}\n\n"
        "[dependencies]\n"
        f'icu_properties = {{ version = "={properties_version}", '
        'features = ["compiled_data"] }\n'
    )


def run(command: Sequence[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"+ {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise GuardError(
            f"{' '.join(command)} failed (rc={completed.returncode}); see {log}"
        )


def build_dump(
    pair: tuple[Release, Release],
    work: Path,
    jobs: int,
    offline: bool,
    pin: dict[str, str] | None = None,
) -> tuple[bytes, dict[str, str]]:
    """Build the dump program with ``pair`` pinned and return its output.

    ``pin`` moves every other package the dump build shares with it to the
    version it names (the judged build's own versions, for the second build
    of the judged pair). Also returns every package the dump build resolved
    (name to version), so the manifest says what else was linked beside the
    pair.
    """
    properties, data = pair
    suffix = "+workspace-pins" if pin else ""
    directory = work / "pairs" / (pair_label(pair) + suffix)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        dump_manifest(properties.version), encoding="utf-8"
    )
    log = directory / "build.log"
    env = dict(os.environ)
    env.setdefault("RUSTUP_TOOLCHAIN", toolchain_channel())
    env["CARGO_TARGET_DIR"] = str(work / "target")
    env["CARGO_TERM_COLOR"] = "never"
    network = ["--offline"] if offline else []
    run(["cargo", "generate-lockfile", *network], directory, env, log)
    run(
        [
            "cargo",
            "update",
            *network,
            "--package",
            DATA,
            "--precise",
            str(data.version),
        ],
        directory,
        env,
        log,
    )
    # One package at a time can be refused while a dependent still needs
    # the newer version, so the pins are retried until no pass moves one.
    moved = True
    while pin and moved:
        moved = False
        for name, version in sorted(pin.items()):
            current = copies_in_lock(
                (directory / "Cargo.lock").read_text(encoding="utf-8"), name
            )
            if len(current) != 1 or current[0].get("version") == version:
                continue
            spec = f"{name}@{current[0]['version']}"
            command = ["cargo", "update", *network, "--package", spec]
            try:
                run([*command, "--precise", version], directory, env, log)
            except GuardError:
                continue
            moved = True
    lock_text = (directory / "Cargo.lock").read_text(encoding="utf-8")
    for name, version in sorted((pin or {}).items()):
        found = copies_in_lock(lock_text, name)
        if len(found) == 1 and found[0].get("version") != version:
            raise GuardError(f"the pinned dump build could not take {name} {version}")
    for release in pair:
        found = copies_in_lock(lock_text, release.name)
        if len(found) != 1 or found[0].get("version") != str(release.version):
            raise GuardError(
                f"the dump build of {pair_label(pair)} resolved "
                f"{release.name} as {found}"
            )
        if found[0].get("source") != REGISTRY_SOURCE:
            raise GuardError(
                f"the dump build's {release.name} is not the registry copy"
            )
        if found[0].get("checksum") != release.checksum:
            raise GuardError(
                f"the dump build's {release.name} {release.version} checksum "
                f"{found[0].get('checksum')} is not the index's {release.checksum}"
            )
    run(
        ["cargo", "build", *network, "--release", "--locked", "--jobs", str(jobs)],
        directory,
        env,
        log,
    )
    output = directory / "classes.bin"
    binary = work / "target" / "release" / "toktier-property-dump"
    run([str(binary), str(output)], directory, env, log)
    dump = output.read_bytes()
    if len(dump) != CODE_POINTS * len(COLUMNS):
        raise GuardError(f"the dump of {pair_label(pair)} has {len(dump)} bytes")
    resolved = {
        package["name"]: package["version"]
        for package in lock_packages(lock_text)
        if package.get("name") != "toktier-property-dump"
    }
    return dump, resolved


def ranges_of(code_points: Iterable[int]) -> list[list[int]]:
    """Sorted inclusive ranges of a sorted run of code points."""
    found: list[list[int]] = []
    for code_point in code_points:
        if found and found[-1][1] == code_point - 1:
            found[-1][1] = code_point
        else:
            found.append([code_point, code_point])
    return found


def range_count(ranges: Iterable[Sequence[int]]) -> int:
    return sum(high - low + 1 for low, high in ranges)


def compare(judged: bytes, other: bytes) -> dict[str, Any]:
    """Every code point whose six values differ, and how."""
    width = len(COLUMNS)
    differing: list[int] = []
    per_column = [0] * width
    unassigned_before = 0
    assigned_changed: list[dict[str, Any]] = []
    # Compare 4096-code-point blocks first; most are equal.
    block = 4096 * width
    for start in range(0, len(judged), block):
        if judged[start : start + block] == other[start : start + block]:
            continue
        for offset in range(start, min(start + block, len(judged)), width):
            left = judged[offset : offset + width]
            right = other[offset : offset + width]
            if left == right:
                continue
            code_point = offset // width
            differing.append(code_point)
            for column in range(width):
                if left[column] != right[column]:
                    per_column[column] += 1
            if left[GENERAL_CATEGORY] == UNASSIGNED:
                unassigned_before += 1
            else:
                assigned_changed.append(
                    {
                        "code_point": f"U+{code_point:04X}",
                        "judged": list(left),
                        "resolved": list(right),
                    }
                )
    ranges = ranges_of(differing)
    return {
        "differing_code_points": len(differing),
        "differing_ranges": len(ranges),
        "unassigned_at_judged": unassigned_before,
        "assigned_at_judged": assigned_changed,
        "per_column": dict(zip(COLUMNS, per_column, strict=True)),
        "ranges": ranges,
    }


def union(range_lists: Iterable[Sequence[Sequence[int]]]) -> list[list[int]]:
    code_points: set[int] = set()
    for ranges in range_lists:
        for low, high in ranges:
            code_points.update(range(low, high + 1))
    return ranges_of(sorted(code_points))


def ranges_sha256(ranges: Sequence[Sequence[int]]) -> str:
    canonical = json.dumps([list(item) for item in ranges], separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------


def admitted_copies(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The distinct crate copies of the listed pairs, judged ones marked."""
    copies: dict[tuple[str, str, str], bool] = {}
    for pair in pairs:
        for name in CRATES:
            copy = pair[name]
            key = (name, copy["version"], copy["checksum"])
            copies[key] = copies.get(key, False) or bool(pair["judged"])
    return [
        {"name": name, "version": version, "checksum": checksum, "judged": judged}
        for (name, version, checksum), judged in sorted(
            copies.items(), key=lambda item: (item[0][0], Version.parse(item[0][1]))
        )
    ]


def render_table(manifest: dict[str, Any]) -> str:
    pairs = manifest["pairs"]
    moved = manifest["union"]["ranges"]
    lines = [
        "// @generated by tools/generate_property_guard.py from",
        "// tools/property_guard_manifest.json; do not edit by hand. Regenerate",
        "// with `python3 tools/generate_property_guard.py --work-dir <dir>` and",
        "// verify with `--check-offline`.",
        "//",
        "// Workspace requirement: icu_properties "
        f"{manifest['workspace_requirement']}.",
        f"// Pairs judged against the judged pair: {len(pairs)}.",
        f"// Code points whose engine class some admitted pair moves: "
        f"{manifest['union']['code_points']} in {len(moved)} ranges.",
        "",
        "/// The registry source every admitted copy carries in a lockfile.",
        f"pub(crate) const REGISTRY_SOURCE: &str = {json.dumps(REGISTRY_SOURCE)};",
        "",
        "/// One crate copy the table speaks for.",
        "#[derive(Debug, Clone, Copy, PartialEq, Eq)]",
        "pub(crate) struct AdmittedCopy {",
        "    pub(crate) name: &'static str,",
        "    pub(crate) version: &'static str,",
        "    pub(crate) checksum: &'static str,",
        "    /// A copy of the judged pair.",
        "    pub(crate) judged: bool,",
        "}",
        "",
        "/// Every copy of `icu_properties` and `icu_properties_data` that some",
        "/// admitted pair holds, sorted by name and version.",
        "pub(crate) const ADMITTED_COPIES: &[AdmittedCopy] = &[",
    ]
    for copy in admitted_copies(pairs):
        lines += [
            "    AdmittedCopy {",
            f"        name: {json.dumps(copy['name'])},",
            f"        version: {json.dumps(copy['version'])},",
            f"        checksum: {json.dumps(copy['checksum'])},",
            f"        judged: {'true' if copy['judged'] else 'false'},",
            "    },",
        ]
    lines += [
        "];",
        "",
        "/// The union of the differing sets of every admitted pair, as sorted,",
        "/// disjoint, non-adjacent inclusive ranges.",
        "pub(crate) const MOVED_RANGES: &[(u32, u32)] = &[",
    ]
    for low, high in moved:
        lines.append(f"    (0x{low:04X}, 0x{high:04X}),")
    lines += [
        "];",
        "",
        "/// The number of code points in [`MOVED_RANGES`].",
        "pub(crate) const MOVED_CODE_POINTS: u32 = "
        f"{manifest['union']['code_points']};",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Generation and the two checks
# ---------------------------------------------------------------------


def generate(
    work: Path,
    jobs: int,
    offline: bool,
    index_dir: Path | None,
    keep_dumps: Path | None,
) -> dict[str, Any]:
    requirement = workspace_requirement(
        (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    )
    judged = judged_pair((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    raw = {name: read_index(name, index_dir) for name in CRATES}
    listed = {name: releases(name, raw[name]) for name in CRATES}
    pairs = admitted_pairs(requirement, listed[PROPERTIES], listed[DATA])
    judged_key = (judged[PROPERTIES]["version"], judged[DATA]["version"])
    if not any((str(p.version), str(d.version)) == judged_key for p, d in pairs):
        raise GuardError(
            f"the judged pair {judged_key} is not among the admitted pairs"
        )
    for name in CRATES:
        match = [r for r in listed[name] if str(r.version) == judged[name]["version"]]
        if len(match) != 1 or match[0].checksum != judged[name]["checksum"]:
            raise GuardError(
                f"the index does not list the judged {name} with its checksum"
            )
    work.mkdir(parents=True, exist_ok=True)
    # Keep the index state read, so that a later run can replay it with
    # --index-dir.
    (work / "index").mkdir(exist_ok=True)
    for name in CRATES:
        (work / "index" / name).write_bytes(raw[name])
    dumps: dict[str, bytes] = {}
    resolved: dict[str, dict[str, str]] = {}
    failures: list[dict[str, str]] = []
    for pair in pairs:
        label = pair_label(pair)
        print(f"building the dump of {label}", flush=True)
        try:
            dumps[label], resolved[label] = build_dump(pair, work, jobs, offline)
        except GuardError as error:
            if (str(pair[0].version), str(pair[1].version)) == judged_key:
                raise
            print(f"  left off the list: {error}", flush=True)
            failures.append({"pair": label, "reason": str(error)})
            continue
        if keep_dumps is not None:
            keep_dumps.mkdir(parents=True, exist_ok=True)
            (keep_dumps / f"classes_{label}.bin").write_bytes(dumps[label])
    judged_dump = dumps[f"{judged_key[0]}+{judged_key[1]}"]
    # The judged pair once more, with every package the dump build shares
    # with the workspace lockfile at the workspace's version: the classes
    # the judged build itself computes. They must equal the first dump.
    workspace_versions: dict[str, list[str]] = {}
    for package in lock_packages((ROOT / "Cargo.lock").read_text(encoding="utf-8")):
        workspace_versions.setdefault(package["name"], []).append(package["version"])
    pins = {
        name: versions[0]
        for name, versions in workspace_versions.items()
        if len(versions) == 1 and name not in CRATES
    }
    judged_release = next(
        pair
        for pair in pairs
        if (str(pair[0].version), str(pair[1].version)) == judged_key
    )
    print(
        "building the dump of the judged pair with the workspace's versions", flush=True
    )
    pinned_dump, pinned_resolved = build_dump(judged_release, work, jobs, offline, pins)
    if pinned_dump != judged_dump:
        raise GuardError(
            "the judged pair's dump differs between the newest support crates and the "
            "workspace lockfile's; the table cannot speak for the judged build"
        )
    records: list[dict[str, Any]] = []
    for pair in pairs:
        label = pair_label(pair)
        if label not in dumps:
            continue
        is_judged = (str(pair[0].version), str(pair[1].version)) == judged_key
        record: dict[str, Any] = {
            "pair": label,
            "judged": is_judged,
        }
        for release in pair:
            record[release.name] = {
                "version": str(release.version),
                "checksum": release.checksum,
                "yanked": release.yanked,
            }
        record["dump_sha256"] = hashlib.sha256(dumps[label]).hexdigest()
        record.update(compare(judged_dump, dumps[label]))
        record["resolved_packages"] = resolved[label]
        records.append(record)
    moved = union(record["ranges"] for record in records)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index_state": {
            name: {
                "sha256": hashlib.sha256(raw[name]).hexdigest(),
                "source": "index-dir"
                if index_dir is not None
                else INDEX_URL.format(prefix=index_prefix(name), name=name),
            }
            for name in CRATES
        },
        "workspace_requirement": requirement,
        "toolchain": toolchain_channel(),
        "engine_unicode_sha256": hashlib.sha256(
            ENGINE_UNICODE.read_bytes()
        ).hexdigest(),
        "dump_program_sha256": hashlib.sha256(DUMP_MAIN.read_bytes()).hexdigest(),
        "columns": list(COLUMNS),
        "judged_pair": f"{judged_key[0]}+{judged_key[1]}",
        "judged_pair_with_workspace_versions": {
            "dump_sha256": hashlib.sha256(pinned_dump).hexdigest(),
            "resolved_packages": pinned_resolved,
        },
        "pairs": records,
        "pairs_left_off": failures,
        "union": {
            "code_points": range_count(moved),
            "range_count": len(moved),
            "ranges_sha256": ranges_sha256(moved),
            "ranges": moved,
        },
    }
    table = render_table(manifest)
    manifest["table_sha256"] = hashlib.sha256(table.encode("ascii")).hexdigest()
    return manifest


def manifest_text(manifest: dict[str, Any]) -> str:
    """The manifest as written: indented, with every list of integers on one
    line (a range, a dump row), so that a reader sees one range per line."""
    text = json.dumps(manifest, indent=1, sort_keys=True)
    text = re.sub(
        r"\[\s*(-?\d+(?:,\s*-?\d+)*)\s*\]",
        lambda match: "[" + ", ".join(re.split(r",\s*", match[1])) + "]",
        text,
    )
    return text + "\n"


def offline_problems(manifest: dict[str, Any], table_text: str) -> list[str]:
    """Everything ``--check-offline`` refuses, as sentences."""
    problems: list[str] = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"the manifest's schema is not {SCHEMA}")
        return problems
    requirement = workspace_requirement(
        (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    )
    if manifest["workspace_requirement"] != requirement:
        problems.append(
            f"Cargo.toml requires icu_properties {requirement}, the manifest was "
            f"generated for {manifest['workspace_requirement']}"
        )
    judged = judged_pair((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    judged_records = [record for record in manifest["pairs"] if record["judged"]]
    if len(judged_records) != 1:
        problems.append("the manifest does not mark exactly one judged pair")
    else:
        record = judged_records[0]
        for name in CRATES:
            if {k: record[name][k] for k in ("version", "checksum")} != judged[name]:
                problems.append(f"the manifest's judged {name} is not Cargo.lock's")
        if record["differing_code_points"] != 0:
            problems.append("the judged pair differs from itself")
        pinned = manifest.get("judged_pair_with_workspace_versions", {})
        if pinned.get("dump_sha256") != record["dump_sha256"]:
            problems.append(
                "the judged pair's dump with the workspace lockfile's versions is "
                "not recorded as equal to its dump"
            )
    if (
        manifest["engine_unicode_sha256"]
        != hashlib.sha256(ENGINE_UNICODE.read_bytes()).hexdigest()
    ):
        problems.append(
            "the engine's unicode.rs changed since the manifest was generated; "
            "regenerate the table"
        )
    if (
        manifest["dump_program_sha256"]
        != hashlib.sha256(DUMP_MAIN.read_bytes()).hexdigest()
    ):
        problems.append("the dump program changed since the manifest was generated")
    for record in manifest["pairs"]:
        ranges = record["ranges"]
        flat = [value for item in ranges for value in item]
        if flat != sorted(flat) or any(
            ranges[i][1] + 1 >= ranges[i + 1][0] for i in range(len(ranges) - 1)
        ):
            problems.append(
                f"the ranges of {record['pair']} are not sorted and disjoint"
            )
        if range_count(ranges) != record["differing_code_points"]:
            problems.append(f"the code point count of {record['pair']} does not add up")
        if len(ranges) != record["differing_ranges"]:
            problems.append(f"the range count of {record['pair']} does not add up")
    moved = union(record["ranges"] for record in manifest["pairs"])
    if moved != manifest["union"]["ranges"]:
        problems.append("the union is not the union of the pairs' ranges")
    if range_count(moved) != manifest["union"]["code_points"]:
        problems.append("the union's code point count does not add up")
    if ranges_sha256(moved) != manifest["union"]["ranges_sha256"]:
        problems.append("the union's digest does not match its ranges")
    expected = render_table(manifest)
    if table_text != expected:
        problems.append(
            f"{TABLE.relative_to(ROOT)} is not the rendering of the manifest"
        )
    if (
        hashlib.sha256(table_text.encode("utf-8")).hexdigest()
        != manifest["table_sha256"]
    ):
        problems.append("the table's digest is not the one the manifest records")
    return problems


def stable(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest without what may differ between two runs on the same evidence.

    The packages a dump build resolved beside the pair (``resolved_packages``)
    follow the registry's newest compatible releases, so they are reported by
    ``--check`` when they move but do not fail it; the dump's digest, which
    they would move if they mattered, is compared.
    """
    kept = {key: value for key, value in manifest.items() if key not in VOLATILE_KEYS}
    kept["pairs"] = [
        {key: value for key, value in record.items() if key != "resolved_packages"}
        for record in manifest.get("pairs", [])
    ]
    return kept


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="regenerate and compare")
    mode.add_argument(
        "--check-offline",
        action="store_true",
        help="verify the table against the manifest",
    )
    parser.add_argument(
        "--work-dir", type=Path, help="scratch directory for the dump builds"
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="cargo build jobs (default 4)"
    )
    parser.add_argument(
        "--offline", action="store_true", help="pass --offline to cargo"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="read the two index files from here, not the network",
    )
    parser.add_argument("--keep-dumps", type=Path, help="also write every dump here")
    args = parser.parse_args(argv)
    try:
        if args.check_offline:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            problems = offline_problems(manifest, TABLE.read_text(encoding="utf-8"))
            for problem in problems:
                print(f"error: {problem}", file=sys.stderr)
            if problems:
                return 1
            print(
                "property guard table: offline check passed "
                f"({len(manifest['pairs'])} pairs, "
                f"{manifest['union']['code_points']} code points)"
            )
            return 0
        if args.work_dir is None:
            parser.error("--work-dir is required unless --check-offline is given")
        manifest = generate(
            args.work_dir.resolve(),
            args.jobs,
            args.offline,
            args.index_dir,
            args.keep_dumps,
        )
        table = render_table(manifest)
        if args.check:
            on_disk = json.loads(MANIFEST.read_text(encoding="utf-8"))
            problems = []
            if TABLE.read_text(encoding="utf-8") != table:
                problems.append(
                    f"{TABLE.relative_to(ROOT)} differs from a regeneration"
                )
            if stable(on_disk) != stable(manifest):
                keys = sorted(
                    key
                    for key in set(stable(on_disk)) | set(stable(manifest))
                    if stable(on_disk).get(key) != stable(manifest).get(key)
                )
                problems.append(f"the manifest differs from a regeneration in {keys}")
            before = {r["pair"]: r.get("resolved_packages") for r in on_disk["pairs"]}
            for record in manifest["pairs"]:
                if before.get(record["pair"]) != record["resolved_packages"]:
                    print(
                        f"note: the dump build of {record['pair']} resolved other "
                        "packages than the recorded one; its dump is unchanged"
                        if not problems
                        else f"note: the dump build of {record['pair']} resolved "
                        "other packages than the recorded one"
                    )
            for problem in problems:
                print(f"error: {problem}", file=sys.stderr)
            if problems:
                return 1
            print("property guard table: check passed")
            return 0
        TABLE.write_text(table, encoding="ascii")
        MANIFEST.write_text(manifest_text(manifest), encoding="ascii")
        print(
            f"wrote {TABLE.relative_to(ROOT)} and {MANIFEST.relative_to(ROOT)}: "
            f"{len(manifest['pairs'])} pairs, {manifest['union']['code_points']} code "
            f"points in {manifest['union']['range_count']} ranges"
        )
        return 0
    except GuardError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
