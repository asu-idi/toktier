#!/usr/bin/env python3
"""Synchronize the self-contained data payload of the public Rust crate."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "crates" / "toktier" / "data"

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
GENERATED = (
    "build/source_identity.json",
    "licenses/RUST_DEPENDENCY_LICENSES.txt",
    "sbom/toktier.cyclonedx.json",
)


def target(relative: str) -> Path:
    return DESTINATION / relative


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
    print(f"{DESTINATION}: {'check passed' if arguments.check else 'updated'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
