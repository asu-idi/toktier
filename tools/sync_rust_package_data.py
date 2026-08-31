#!/usr/bin/env python3
"""Synchronize the self-contained data payload of the public Rust crate.

Besides copying, this tool checks the digests the crate pins by hand over
the payload it embeds.  ``crates/toktier/src/manifest.rs`` refuses to
load its embedded data when a payload does not hash to the constant
written beside it, and no generator writes those constants: a data file
synchronized here without its constant moving with it is a refusal that
only a Rust test run reports.  The constants are read back out of the
source, so one added later is covered the day it is added.
"""

from __future__ import annotations

import argparse
import re
import shutil
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "crates" / "toktier"
DESTINATION = CRATE / "data"
MANIFEST_SOURCE = CRATE / "src" / "manifest.rs"

FILES = (
    "THIRD_PARTY_NOTICES",
    "packaging/fast_cpu/LICENSE-gigatoken",
    "packaging/fast_cpu/NOTICE-gigatoken-pinned",
    "packaging/fast_cpu/THIRD_PARTY_LICENSES-gigatoken.txt",
    "packaging/fast_cpu/gigatoken-toktier-pinned-1.patch",
    "schemas/evidence_manifest.schema.json",
    "schemas/sibling_alias_source.schema.json",
    "schemas/sibling_aliases.schema.json",
    "schemas/support_registry.schema.json",
    "src/toktier/artifacts/tables/artifact_conversions.v1.json",
    "src/toktier/artifacts/tables/artifact_manifest.v1.json",
    "src/toktier/artifacts/tables/sibling_aliases.v1.json",
    "src/toktier/kernels/prebuilt/build_manifest.json",
    "src/toktier/kernels/prebuilt/pretok_kernel.fatbin",
    "src/toktier/kernels/prebuilt_unit.cu",
    "src/toktier/kernels/pretok_kernel.cu",
    "src/toktier/kernels/tables/kernel_families.v1.json",
    "src/toktier/kernels/tables/nfc_quick_check.v1.meta.json",
    "src/toktier/kernels/tables/nfc_quick_check.v1.npy",
    "src/toktier/kernels/tables/pretok_classes_cl100k.v3.npy",
    "src/toktier/kernels/tables/pretok_classes_cl100k_marks_as_letters.v3.npy",
    "src/toktier/kernels/tables/pretok_classes_deepseek.v1.meta.json",
    "src/toktier/kernels/tables/pretok_classes_deepseek.v1.npy",
    "src/toktier/kernels/tables/pretok_classes_kimi.v1.meta.json",
    "src/toktier/kernels/tables/pretok_classes_kimi.v1.npy",
    "src/toktier/kernels/tables/pretok_classes_o200k.v4.npy",
    "src/toktier/repair/tables/fast_repair_families.v1.json",
    "src/toktier/repair/tables/repair_pclass.v1.zlib",
    "src/toktier/routing/tables/support_registry.v1.json",
)
#: Sources copied under a different packaged name. The lockfile is the
#: judged dependency graph: an unpacked registry copy cannot see the
#: workspace, so it carries this byte-identical copy and the build
#: script compares the graph that governs the consumer's build against
#: it. Keeping the raw lockfile (rather than a parsed summary) leaves
#: exactly one closure implementation in the tree, in
#: crates/toktier/build_support/source_identity.rs.
RENAMED = (("Cargo.lock", "build/judged_dependencies.lock"),)
#: Written by other maintainer tools into the same payload; this one
#: checks that they are present, and each of those tools has its own
#: `--check` for what they contain.
GENERATED = (
    "build/judged_compiled_closure.json",
    "build/source_identity.json",
    "licenses/RUST_DEPENDENCY_LICENSES.txt",
    "sbom/toktier.cyclonedx.json",
)


#: An embedded payload: the constant naming it, and the path it is
#: included from, relative to the crate directory.
_EMBEDDED_PAYLOAD = re.compile(
    r"const\s+(?P<name>[A-Z0-9_]+)_BYTES\s*:\s*&\[u8\]\s*="
    r"\s*include_bytes!\(\s*concat!\(\s*"
    r'env!\("CARGO_MANIFEST_DIR"\)\s*,\s*"(?P<path>[^"]+)"\s*,?\s*\)\s*\)\s*;'
)
#: A digest constant beside those payloads.  The value is matched loosely
#: on purpose: a constant that is neither a build-script value nor a
#: well-formed literal is reported rather than passed over.
_PINNED_DIGEST = re.compile(
    r"const\s+(?P<name>[A-Z0-9_]+)_SHA256\s*:\s*&str\s*=\s*(?P<value>[^;]+);"
)
_HEX_LITERAL = re.compile(r'^"([0-9a-f]{64})"$')


def target(relative: str) -> Path:
    return DESTINATION / relative


def hand_pinned_digest_names(source: str) -> list[str]:
    """Digest constants the crate source pins by hand, in file order.

    A constant whose value comes from ``env!`` is left out: the build
    script hashes the file it reads, so nothing there is written by hand.
    """
    return [
        match["name"]
        for match in _PINNED_DIGEST.finditer(source)
        if not match["value"].strip().startswith("env!")
    ]


def embedded_digest_problems(
    source: str | None = None, crate: Path | None = None
) -> list[str]:
    """Recompute every hand-pinned digest over the payload it pins.

    Each ``<NAME>_SHA256`` constant is paired with the ``<NAME>_BYTES``
    payload declared in the same file, and the payload is hashed from
    disk.  A constant with no payload to check it against, a value that
    is neither a build-script value nor a lowercase hex literal, and a
    source that has stopped declaring any hand-pinned constant at all are
    each reported: this check is worth nothing if it can pass by finding
    nothing to do.
    """
    crate = CRATE if crate is None else crate
    if source is None:
        source = MANIFEST_SOURCE.read_text(encoding="utf-8")
    payloads = {
        match["name"]: match["path"] for match in _EMBEDDED_PAYLOAD.finditer(source)
    }
    values = {
        match["name"]: match["value"].strip()
        for match in _PINNED_DIGEST.finditer(source)
    }
    issues: list[str] = []
    names = hand_pinned_digest_names(source)
    if not names:
        issues.append(
            f"no hand-pinned digest constant found in {MANIFEST_SOURCE.name}; "
            f"this check has nothing to compare"
        )
    for name in names:
        literal = _HEX_LITERAL.match(values[name])
        if literal is None:
            issues.append(
                f"pinned digest {name}_SHA256 is neither a build-script value "
                f"nor a lowercase 64-character hex literal: {values[name]}"
            )
            continue
        relative = payloads.get(name)
        if relative is None:
            issues.append(
                f"pinned digest {name}_SHA256 has no {name}_BYTES payload "
                f"declared beside it to check it against"
            )
            continue
        embedded = crate / relative.lstrip("/")
        if not embedded.is_file():
            issues.append(f"embedded payload is missing: {relative.lstrip('/')}")
            continue
        observed = sha256(embedded.read_bytes()).hexdigest()
        if observed != literal.group(1):
            issues.append(
                f"pinned digest {name}_SHA256 is stale: "
                f"{relative.lstrip('/')} hashes to {observed}"
            )
    return issues


def problems() -> list[str]:
    source_backed = {Path(value): Path(value) for value in FILES}
    source_backed.update(
        {Path(packaged): Path(source) for source, packaged in RENAMED}
    )
    expected = set(source_backed) | {Path(value) for value in GENERATED}
    issues: list[str] = []
    for relative in sorted(source_backed):
        source = ROOT / source_backed[relative]
        packaged = target(relative.as_posix())
        if not source.is_file():
            issues.append(f"source is missing: {source_backed[relative]}")
        elif not packaged.is_file():
            issues.append(f"packaged copy is missing: {relative}")
        elif packaged.read_bytes() != source.read_bytes():
            issues.append(f"packaged copy drifted: {relative}")
    for relative in map(Path, GENERATED):
        if not target(relative.as_posix()).is_file():
            issues.append(f"generated packaged file is missing: {relative}")
    if DESTINATION.is_dir():
        observed = {
            path.relative_to(DESTINATION)
            for path in DESTINATION.rglob("*")
            if path.is_file()
        }
        for relative in sorted(observed - expected):
            issues.append(f"unexpected packaged file: {relative}")
    issues.extend(embedded_digest_problems())
    return issues


def sync() -> None:
    pairs = [(value, value) for value in FILES] + list(RENAMED)
    for source_relative, packaged_relative in pairs:
        source = ROOT / source_relative
        packaged = target(packaged_relative)
        packaged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, packaged)
        packaged.chmod(0o644)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if not arguments.check:
        sync()
    issues = problems()
    for issue in issues:
        print(f"error: {issue}")
    if issues:
        return 1
    pinned = len(
        hand_pinned_digest_names(MANIFEST_SOURCE.read_text(encoding="utf-8"))
    )
    print(
        f"{DESTINATION}: {'check passed' if arguments.check else 'updated'} "
        f"({pinned} hand-pinned embedded digests verified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
