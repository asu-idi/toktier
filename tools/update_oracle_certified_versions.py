#!/usr/bin/env python3
"""Widen or check the certified reference versions of the oracle, on evidence.

The oracle record of ``tables/support_registry.json`` lists the exact
``tokenizers`` releases that are *certified reference versions*: the
embedded reference every certification reading was judged against (the
``tokenizers`` crate compiled into the native extension, 0.22.2), and any
later Python release judged equivalent to it on the certified surface.
Package metadata admits a range of releases; this set says which of them
the certificates speak for. The routing plan refuses the accelerated
routes for an installed release outside it (``R_ORACLE_MISMATCH``), the
chunked path refuses with ``oracle_differs_from_chunked_evidence``, the
explicit GPU engine reports ``uncertified_oracle``, and ``toktier doctor``
reports ``uncertified_reference`` (``docs/contracts/registry.md`` Section
2). Widening the behavior class behind the record's ``semantic_id``
requires certification evidence that the releases agree on the certified
surface (``docs/contracts/fingerprint.md`` Section 7); the semantic id
itself does not move, so stored sessions stay valid.

The evidence for a release ``v`` other than the embedded reference is a
set of compatibility readings under ``readings/oracle_compat/``, taken
with ``v`` installed by the same harness and aggregator that produce the
certification readings they mirror::

    readings/oracle_compat/fast_cpu_native_frontend_parity.tokenizers-<v>.json
    readings/oracle_compat/gpu_native_frontend_<arch>_parity.tokenizers-<v>.json
        for every architecture in GPU_ARCHITECTURES

Each compatibility reading has to record ``v`` as its oracle, zero
mismatches with every per-batch flag true, the same families, artifacts,
document and character counts as the certification reading it mirrors,
and the engine identity of this tree: the version-normalized fast CPU
identity for the CPU reading, the version-normalized native-host identity
plus the shipped fatbin and architecture digests for a GPU reading. The
package metadata range has to admit every certified version, so that a
release the certificates speak for is one that installs.

Usage::

    python tools/update_oracle_certified_versions.py --check
    python tools/update_oracle_certified_versions.py --add 0.23.3 [--add ...]

``--add`` verifies the evidence for each release, writes the widened set
on the oracle record of the repository table and its packaged copy, and
recomputes the root digest; ``tools/sync_rust_package_data.py`` then
carries the packaged copy into the Rust crate's data payload. ``--check``
passes when the two copies are the same serialised document, the embedded
reference is in the set, every other member has passing evidence, no
compatibility reading attests a release outside the set, and the metadata
range admits every member.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

from compute_identity_v2 import source_digest as source_digest_v2
from registry_common import (
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    load_json,
    schema_violations,
    serialise_document,
    verify_file,
    with_root_digest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: The oracle package the set describes.
ORACLE_PACKAGE = "tokenizers"
#: The reference compiled into the native extension (``crates/toktier``'s
#: ``ORACLE`` and ``jit.rs``'s ``FROZEN_ORACLE_VERSION`` name the same
#: release). It is always a certified reference version: the readings were
#: judged against it.
EMBEDDED_REFERENCE_VERSION = "0.22.2"
#: Architectures whose GPU parity has to be re-read under a release before
#: it is certified: the venues the certification readings themselves come
#: from that this project can re-run.
GPU_ARCHITECTURES = ("sm_120", "sm_89")

COMPAT_DIRECTORY = Path("readings") / "oracle_compat"
CPU_READING = Path("readings") / "fast_cpu_native_frontend_parity.json"
_METADATA_RANGE = re.compile(
    r'"tokenizers>=(?P<floor>[0-9][^,"]*),<(?P<ceiling>[0-9][^"]*)"'
)
_COMPAT_FILE = re.compile(
    r"(?P<stem>fast_cpu_native_frontend_parity|gpu_native_frontend_sm[0-9]+_parity)"
    r"\.tokenizers-(?P<version>[^/]+)\.json"
)


def registry_path(root: Path) -> Path:
    return root / "tables" / "support_registry.json"


def packaged_path(root: Path) -> Path:
    return root / "src" / "toktier" / "routing" / "tables" / "support_registry.v1.json"


def schema_path(root: Path) -> Path:
    return root / "schemas" / "support_registry.schema.json"


def gpu_reading(architecture: str) -> Path:
    # The shipped readings spell the architecture without the underscore.
    short = architecture.replace("_", "")
    return Path("readings") / f"gpu_native_frontend_{short}_parity.json"


def compat_path(root: Path, mirrored: Path, version: str) -> Path:
    return root / COMPAT_DIRECTORY / f"{mirrored.stem}.tokenizers-{version}.json"


def version_key(version: str) -> tuple[tuple[int, int | str], ...]:
    """Order dotted release numbers numerically, other components as text."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in version.split(".")
    )


def tree_identities(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    """The version-normalized identities the compatibility readings bind."""
    return {
        "fast_cpu_v2": source_digest_v2("fast_cpu", root),
        "native_host_v2": source_digest_v2("native_host", root),
    }


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be a JSON object")
    return value


def oracle_records(registry: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        _mapping(raw, "registry oracle record") for raw in registry.get("oracles", [])
    ]
    named = [record for record in records if record.get("package") == ORACLE_PACKAGE]
    if not named:
        raise GenerationError(f"the registry has no oracle record for {ORACLE_PACKAGE}")
    return named


def certified_versions(registry: dict[str, Any]) -> list[str]:
    versions: set[str] = set()
    for record in oracle_records(registry):
        versions.update(
            str(version) for version in record.get("certified_versions", [])
        )
    return sorted(versions, key=version_key)


def metadata_range(root: Path) -> tuple[str, str]:
    """The ``tokenizers`` floor and ceiling the package metadata declares."""
    match = _METADATA_RANGE.search(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    if match is None:
        raise GenerationError("pyproject.toml does not declare a tokenizers range")
    return match["floor"], match["ceiling"]


def _rows_by_family(reading: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = reading.get("rows")
    if not isinstance(rows, list) or not rows:
        raise GenerationError(f"{label} has no rows")
    return {str(row.get("family")): _mapping(row, f"{label} row") for row in rows}


def reading_problems(
    compat: dict[str, Any],
    mirrored: dict[str, Any],
    *,
    version: str,
    label: str,
    identities: dict[str, str],
) -> list[str]:
    """Why ``compat`` does not attest ``version`` for what ``mirrored`` covers."""
    problems: list[str] = []
    oracle = compat.get("oracle")
    if not isinstance(oracle, dict) or oracle.get("package") != ORACLE_PACKAGE:
        problems.append(f"{label}: no {ORACLE_PACKAGE} oracle is recorded")
    elif oracle.get("version") != version:
        problems.append(
            f"{label}: records oracle version {oracle.get('version')!r}, "
            f"the file name says {version}"
        )
    if compat.get("schema") != mirrored.get("schema"):
        problems.append(f"{label}: schema differs from the certification reading")
    for key in ("documents", "characters"):
        if compat.get(key) != mirrored.get(key):
            problems.append(
                f"{label}: {key} {compat.get(key)!r} is not the certification "
                f"reading's {mirrored.get(key)!r}"
            )
    if int(compat.get("mismatches", -1)) != 0:
        problems.append(f"{label}: records {compat.get('mismatches')!r} mismatches")
    for flag in (
        "all_ids_equal_hf",
        "one_python_to_rust_call_per_batch",
        "gil_released",
    ):
        if compat.get(flag) is not True:
            problems.append(f"{label}: {flag} is not true")
    try:
        compat_rows = _rows_by_family(compat, label)
        mirrored_rows = _rows_by_family(mirrored, "certification reading")
    except GenerationError as error:
        return [*problems, str(error)]
    if set(compat_rows) != set(mirrored_rows):
        problems.append(f"{label}: families differ from the certification reading")
    for family in sorted(set(compat_rows) & set(mirrored_rows)):
        row, expected = compat_rows[family], mirrored_rows[family]
        if row.get("artifact_sha256") != expected.get("artifact_sha256"):
            problems.append(f"{label}/{family}: artifact differs")
        if int(row.get("mismatches", -1)) != 0 or int(row.get("documents", 0)) < 1:
            problems.append(f"{label}/{family}: row did not pass")
        for key in ("documents", "characters"):
            if row.get(key) != expected.get(key):
                problems.append(f"{label}/{family}: {key} differs")
    if compat.get("schema") == "toktier.fast_cpu.native_frontend_parity.v1":
        engine = compat.get("engine")
        if not isinstance(engine, dict):
            problems.append(f"{label}: no engine identity is recorded")
        elif engine.get("source_digest_v2") != identities["fast_cpu_v2"]:
            problems.append(
                f"{label}: engine identity is not this tree's fast CPU engine"
            )
        for key in ("unique_artifacts", "model_families"):
            if compat.get(key) != mirrored.get(key):
                problems.append(
                    f"{label}: {key} differs from the certification reading"
                )
    elif compat.get("schema") == "toktier.gpu.native_frontend_parity.v1":
        host = compat.get("native_host_build_facts")
        if not isinstance(host, dict):
            problems.append(f"{label}: no native-host identity is recorded")
        elif host.get("host_source_digest_v2") != identities["native_host_v2"]:
            problems.append(f"{label}: host identity is not this tree's native host")
        for key in ("architecture", "fatbin_digest", "architecture_digest", "families"):
            if compat.get(key) != mirrored.get(key):
                problems.append(
                    f"{label}: {key} differs from the certification reading"
                )
    return problems


def evidence_problems(
    root: Path, version: str, identities: dict[str, str]
) -> list[str]:
    """Every reason the compatibility readings do not attest ``version``."""
    problems: list[str] = []
    mirrored_paths = [CPU_READING] + [gpu_reading(arch) for arch in GPU_ARCHITECTURES]
    for mirrored in mirrored_paths:
        path = compat_path(root, mirrored, version)
        label = path.relative_to(root).as_posix()
        if not path.is_file():
            problems.append(f"{label}: compatibility reading is missing")
            continue
        if not (root / mirrored).is_file():
            problems.append(f"{mirrored.as_posix()}: certification reading is missing")
            continue
        try:
            compat = _mapping(load_json(path), label)
            certification = _mapping(load_json(root / mirrored), mirrored.as_posix())
        except (OSError, ValueError, GenerationError) as error:
            problems.append(f"{label}: unreadable: {error}")
            continue
        problems.extend(
            reading_problems(
                compat,
                certification,
                version=version,
                label=label,
                identities=identities,
            )
        )
    return problems


def attested_versions(root: Path) -> set[str]:
    """Releases some compatibility reading under the tree names."""
    directory = root / COMPAT_DIRECTORY
    if not directory.is_dir():
        return set()
    versions: set[str] = set()
    for path in directory.iterdir():
        match = _COMPAT_FILE.fullmatch(path.name)
        if match is not None:
            versions.add(match["version"])
    return versions


def _within(version: str, floor: str, ceiling: str) -> bool:
    return version_key(floor) <= version_key(version) < version_key(ceiling)


def check_problems(
    root: Path = REPOSITORY_ROOT, identities: dict[str, str] | None = None
) -> list[str]:
    """Everything ``--check`` would report."""
    identities = tree_identities(root) if identities is None else identities
    problems = verify_file(
        registry_path(root), load_json(schema_path(root)), REGISTRY_DOMAIN_TAG
    )
    try:
        registry = _mapping(load_json(registry_path(root)), "support registry")
    except (OSError, ValueError, GenerationError) as error:
        return [*problems, str(error)]
    payload = serialise_document(registry)
    packaged = packaged_path(root)
    if not packaged.is_file() or packaged.read_bytes() != payload:
        problems.append(f"{packaged} is not the serialised registry")
    try:
        versions = certified_versions(registry)
        floor, ceiling = metadata_range(root)
    except GenerationError as error:
        return [*problems, str(error)]
    if EMBEDDED_REFERENCE_VERSION not in versions:
        problems.append(
            f"the embedded reference {EMBEDDED_REFERENCE_VERSION} is not a "
            "certified reference version"
        )
    for version in versions:
        if not _within(version, floor, ceiling):
            problems.append(
                f"certified reference version {version} is outside the package "
                f"metadata range >={floor},<{ceiling}"
            )
        if version != EMBEDDED_REFERENCE_VERSION:
            problems.extend(evidence_problems(root, version, identities))
    for version in sorted(attested_versions(root) - set(versions), key=version_key):
        problems.append(
            f"readings/oracle_compat attests {version}, which is not a certified "
            "reference version"
        )
    return problems


def widened_document(registry: dict[str, Any], versions: list[str]) -> dict[str, Any]:
    """``registry`` with ``versions`` added to every oracle record, digest redone."""
    completed = copy.deepcopy(registry)
    for record in oracle_records(completed):
        current = {str(version) for version in record.get("certified_versions", [])}
        record["certified_versions"] = sorted(current | set(versions), key=version_key)
    return with_root_digest(completed, REGISTRY_DOMAIN_TAG)


def add_versions(
    root: Path, versions: list[str], identities: dict[str, str] | None = None
) -> dict[str, Any]:
    """Verify the evidence for ``versions``, write the widened registry copies."""
    identities = tree_identities(root) if identities is None else identities
    registry = _mapping(load_json(registry_path(root)), "support registry")
    floor, ceiling = metadata_range(root)
    problems: list[str] = []
    for version in versions:
        if version == EMBEDDED_REFERENCE_VERSION:
            continue
        if not _within(version, floor, ceiling):
            problems.append(
                f"{version} is outside the package metadata range >={floor},<{ceiling}"
            )
        problems.extend(evidence_problems(root, version, identities))
    if problems:
        raise GenerationError(
            "the evidence does not certify the requested release(s):\n  "
            + "\n  ".join(problems)
        )
    generated = widened_document(registry, versions)
    violations = schema_violations(generated, load_json(schema_path(root)))
    if violations:
        raise GenerationError(
            "widened registry violates its schema:\n  " + "\n  ".join(violations)
        )
    payload = serialise_document(generated)
    for path in (registry_path(root), packaged_path(root)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Widen or check the certified reference versions of the oracle."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify set and evidence")
    mode.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="VERSION",
        help="certify this release on the compatibility readings under the tree",
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.check:
            problems = check_problems()
            for problem in problems:
                print(f"error: {problem}", file=sys.stderr)
            if problems:
                return 1
            registry = _mapping(load_json(registry_path(REPOSITORY_ROOT)), "registry")
            print(
                f"{registry_path(REPOSITORY_ROOT)}: check passed (certified reference "
                f"versions: {', '.join(certified_versions(registry))})"
            )
            return 0
        generated = add_versions(REPOSITORY_ROOT, list(arguments.add))
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"updated {registry_path(REPOSITORY_ROOT)} and "
        f"{packaged_path(REPOSITORY_ROOT)} (certified reference versions: "
        f"{', '.join(certified_versions(generated))}); run "
        "tools/sync_rust_package_data.py to carry the packaged copy into the crate"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
