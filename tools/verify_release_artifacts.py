#!/usr/bin/env python3
"""Verify the exact, single-wheel artifact set allowed for release 0.2.0."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WHEEL = "toktier-0.2.0-cp310-abi3-manylinux_2_34_x86_64.whl"


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _one(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        _fail(f"expected one *{suffix}, found {matches}")
    return matches[0]


def _verify_record(archive: zipfile.ZipFile, names: list[str]) -> None:
    record_name = _one(names, ".dist-info/RECORD")
    rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
    recorded_names = {row[0] for row in rows}
    file_names = {name for name in names if not name.endswith("/")}
    if recorded_names != file_names:
        _fail("wheel RECORD does not account for every file exactly once")
    for name, digest_field, size_field in rows:
        if name == record_name:
            if digest_field or size_field:
                _fail("RECORD's own row must have empty hash and size")
            continue
        payload = archive.read(name)
        if size_field != str(len(payload)):
            _fail(f"RECORD size mismatch for {name}")
        algorithm, separator, encoded = digest_field.partition("=")
        if separator != "=" or algorithm != "sha256":
            _fail(f"RECORD uses a non-sha256 digest for {name}")
        observed = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(
            b"="
        )
        if observed.decode("ascii") != encoded:
            _fail(f"RECORD digest mismatch for {name}")


def _verify_legal_material(archive: zipfile.ZipFile, names: list[str]) -> None:
    sources = (
        ROOT / "LICENSE",
        ROOT / "NOTICE",
        ROOT / "THIRD_PARTY_NOTICES",
        ROOT / "packaging" / "fast_cpu" / "LICENSE-gigatoken",
        ROOT / "packaging" / "fast_cpu" / "NOTICE-gigatoken-pinned",
        ROOT
        / "packaging"
        / "fast_cpu"
        / "THIRD_PARTY_LICENSES-gigatoken.txt",
    )
    license_entries = [name for name in names if ".dist-info/licenses/" in name]
    payloads = {archive.read(name) for name in license_entries}
    for source in sources:
        if source.read_bytes() not in payloads:
            _fail(f"wheel does not carry exact legal material from {source}")


def verify(wheel: Path) -> None:
    if wheel.name != EXPECTED_WHEEL:
        _fail(f"unexpected wheel name {wheel.name!r}; expected {EXPECTED_WHEEL!r}")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if any(
            name.startswith("gigatoken/")
            or re.match(r"^gigatoken-[^/]+\.dist-info/", name)
            for name in names
        ):
            _fail("core wheel exposes a top-level Gigatoken package")
        if any(
            name.endswith("toktier/_vendor/gigatoken_rs.abi3.so")
            or name.endswith("toktier/_vendor/gigatoken_build.json")
            for name in names
        ):
            _fail("wheel still carries the obsolete second CPU extension")

        native_name = _one(names, "toktier/_native.abi3.so")
        if not archive.read(native_name).startswith(b"\x7fELF"):
            _fail("TokTier core extension is not an ELF binary")

        binding = json.loads((ROOT / "tools/fast_cpu_binding.json").read_bytes())
        sbom_name = _one(names, ".dist-info/sboms/gigatoken.cyclonedx.json")
        legal = binding.get("legal") or {}
        if hashlib.sha256(archive.read(sbom_name)).hexdigest() != legal.get(
            "sbom_sha256"
        ):
            _fail("integrated Gigatoken SBOM digest differs from its binding")

        registry_name = _one(
            names, "toktier/routing/tables/support_registry.v1.json"
        )
        registry = json.loads(archive.read(registry_name))
        certified_cpu = [
            row["backends"]["fast_cpu"]
            for row in registry.get("artifacts", [])
            if row.get("backends", {}).get("fast_cpu", {}).get("status")
            == "certified_source"
        ]
        if len(certified_cpu) != 11:
            _fail("wheel registry does not certify eleven integrated CPU artifacts")
        for entry in certified_cpu:
            if (
                entry.get("engine_delivery") != "integrated"
                or entry.get("engine_module") != "toktier._native"
                or entry.get("source_digest") != binding.get("source_digest")
                or entry.get("build_flags") != binding.get("build_flags")
                or entry.get("toolchain") != binding.get("toolchain")
            ):
                _fail("wheel registry carries another integrated CPU identity")

        metadata_name = _one(names, ".dist-info/METADATA")
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_name))
        if metadata["Name"] != "toktier" or metadata["Version"] != "0.2.0":
            _fail("wheel metadata has the wrong distribution identity")
        requirements = metadata.get_all("Requires-Dist", failobj=[])
        if any(re.match(r"(?i)^gigatoken(?:\s|\[|;|$)", item) for item in requirements):
            _fail("wheel metadata requires a second Gigatoken distribution")
        extras = set(metadata.get_all("Provides-Extra", failobj=[]))
        if "fast" in extras:
            _fail("wheel metadata still exposes the obsolete fast extra")
        if not {"gpu", "gpu-jit"}.issubset(extras):
            _fail("wheel metadata does not expose both GPU delivery profiles")
        normalized = [item.replace(" ", "").lower() for item in requirements]
        if "tokenizers==0.22.2" not in normalized:
            _fail("base metadata does not pin tokenizers==0.22.2")
        if "transformers==4.57.6" not in normalized:
            _fail("base metadata does not pin transformers==4.57.6")

        wheel_metadata_name = _one(names, ".dist-info/WHEEL")
        wheel_metadata = archive.read(wheel_metadata_name).decode("utf-8")
        if "Tag: cp310-abi3-manylinux_2_34_x86_64" not in wheel_metadata:
            _fail("wheel metadata does not carry the certified platform tag")

        entry_name = _one(names, ".dist-info/entry_points.txt")
        entry_points = archive.read(entry_name).decode("utf-8")
        if re.search(r"(?m)^toktier\s*=\s*toktier\.cli:main\s*$", entry_points) is None:
            _fail("toktier CLI entry point is missing")
        if "gigatoken" in entry_points.lower():
            _fail("core wheel exposes the upstream Gigatoken CLI")

        alias_name = _one(
            names, "toktier/artifacts/tables/sibling_aliases.v1.json"
        )
        alias_source = (
            ROOT
            / "src"
            / "toktier"
            / "artifacts"
            / "tables"
            / "sibling_aliases.v1.json"
        )
        if archive.read(alias_name) != alias_source.read_bytes():
            _fail("wheel sibling registry differs from the generated source")

        _verify_legal_material(archive, names)
        _verify_record(archive, names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path, nargs="?", default=ROOT / "dist")
    arguments = parser.parse_args()
    dist = arguments.dist.resolve()
    wheels = sorted(dist.glob("*.whl")) if dist.is_dir() else [dist]
    forbidden = (
        sorted(dist.glob("*.tar.gz")) + sorted(dist.glob("*.zip"))
        if dist.is_dir()
        else []
    )
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, found {len(wheels)}")
    if forbidden:
        raise SystemExit(f"source distributions are not admitted: {forbidden}")
    try:
        verify(wheels[0])
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"release artifact verified: {wheels[0]}")
    print(f"sha256: {hashlib.sha256(wheels[0].read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
