#!/usr/bin/env python3
"""Bind an exact Rust direct-NVCC tuple after its hardware matrix passes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import native_host_source_identity
import rust_api_source_identity
from compute_identity_v2 import source_digest as source_digest_v2
from registry_common import (
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    load_json,
    schema_violations,
    serialise_document,
    with_root_digest,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tables/support_registry.json"
PYTHON_COPY = ROOT / "src/toktier/routing/tables/support_registry.v1.json"
SCHEMA = ROOT / "schemas/support_registry.schema.json"
READING = ROOT / "readings/rust_direct_jit_sm120.json"
EVIDENCE_ID = "ev-rust-direct-jit-sm120-plan160-v1"
FLAGS = [
    "-fatbin",
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-DTOKTIER_DEVICE_ONLY",
]


def direct_source_digest() -> str:
    digest = hashlib.sha256(b"toktier.rust_jit_source.v1\0")
    for name in ("prebuilt_unit.cu", "pretok_kernel.cu"):
        raw = (ROOT / "src/toktier/kernels" / name).read_bytes()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be an object")
    return value


def validate(reading: dict[str, Any]) -> None:
    expected = {
        "schema": "toktier.rust_direct_jit.matrix.v1",
        "architecture": "sm_120",
        "runtime_source_digest": rust_api_source_identity.source_digest(),
        "native_host_source_digest": native_host_source_identity.source_digest(),
        "source_digest": direct_source_digest(),
        "direct_build_flags": FLAGS,
        "families": 14,
        "documents": 98,
        "mismatches": 0,
    }
    for key, value in expected.items():
        if reading.get(key) != value:
            raise GenerationError(
                f"direct-JIT reading {key}={reading.get(key)!r}; expected {value!r}"
            )
    expected_v2 = {
        "runtime_source_digest_v2": source_digest_v2("rust_api"),
        "native_host_source_digest_v2": source_digest_v2("native_host"),
    }
    for key, value in expected_v2.items():
        if reading.get(key) not in {None, value}:
            raise GenerationError(f"direct-JIT reading {key} does not match the tree")
    if reading.get("compiler_world_writable_component") is not None:
        raise GenerationError("direct-JIT compiler has a world-writable path component")
    for key in ("compiler_release", "compiler_build", "compiler_sha256"):
        if not isinstance(reading.get(key), str) or not reading[key]:
            raise GenerationError(f"direct-JIT reading has no {key}")
    if len(reading["compiler_sha256"]) != 64:
        raise GenerationError("direct-JIT compiler digest is not a bare SHA-256")
    rows = reading.get("rows")
    if not isinstance(rows, list) or len(rows) != 14:
        raise GenerationError("direct-JIT reading has an incomplete family roster")
    registry = mapping(load_json(REGISTRY), "support registry")
    expected_artifacts = {
        str(artifact["family"]): artifact
        for raw in registry.get("artifacts", [])
        if (artifact := mapping(raw, "registry artifact"))
    }
    families: set[str] = set()
    for raw in rows:
        row = mapping(raw, "direct-JIT family row")
        family = str(row.get("family"))
        families.add(family)
        expected_artifact = expected_artifacts.get(family)
        if expected_artifact is None or row.get(
            "artifact_sha256"
        ) != expected_artifact.get("artifact_sha256"):
            raise GenerationError(
                f"{family}: direct-JIT artifact identity does not match the registry"
            )
        if (
            row.get("oracle_id") != expected_artifact.get("oracle_id")
            or row.get("oracle_version") != "0.22.2"
            or row.get("artifact_evidence_id") != expected_artifact.get("evidence_id")
        ):
            raise GenerationError(
                f"{family}: direct-JIT oracle/evidence lineage does not match "
                "the registry"
            )
        if row.get("certified") is True:
            if row.get("jit_evidence_id") != EVIDENCE_ID:
                raise GenerationError(
                    f"{family}: certified direct-JIT row has no matching evidence id"
                )
        elif (
            row.get("jit_evidence_id") is not None
            or reading.get("experimental_opt_in") is not True
        ):
            raise GenerationError(
                f"{family}: uncertified direct-JIT row lacks an explicit "
                "experimental run"
            )
        if (
            row.get("mismatches") != 0
            or row.get("backend") != "Gpu"
            or row.get("delivery") != "jit"
            or row.get("first_compile_cache_hit") is not False
            or row.get("second_compile_cache_hit") is not True
        ):
            raise GenerationError(f"{family}: direct-JIT row did not pass")
    if families != set(expected_artifacts):
        raise GenerationError("direct-JIT family roster does not equal the registry")


def _records_v2(registry: dict[str, Any]) -> bool:
    return any(
        "direct_host_source_digest_v2" in jit
        for raw in registry.get("artifacts", [])
        if isinstance(raw, dict)
        for gpu in [(raw.get("backends") or {}).get("gpu")]
        if isinstance(gpu, dict)
        for jit in [(gpu.get("deliveries") or {}).get("jit")]
        if isinstance(jit, dict)
    )


def augmented(
    registry: dict[str, Any],
    reading: dict[str, Any],
    *,
    include_v2: bool | None = None,
) -> dict[str, Any]:
    if include_v2 is None:
        include_v2 = _records_v2(registry)
    validate(reading)
    completed = dict(registry)
    artifacts = []
    row_families = {str(row["family"]) for row in reading["rows"]}
    for source in registry.get("artifacts", []):
        artifact = dict(mapping(source, "registry artifact"))
        family = str(artifact.get("family"))
        gpu = mapping(
            mapping(artifact.get("backends"), "artifact backends").get("gpu"), "GPU row"
        )
        deliveries = mapping(gpu.get("deliveries"), "GPU deliveries")
        jit = dict(mapping(deliveries.get("jit"), "JIT delivery"))
        if family not in row_families:
            raise GenerationError(
                f"registry family {family} is absent from direct-JIT evidence"
            )
        jit.update(
            {
                "direct_source_digest": reading["source_digest"],
                "direct_host_source_digest": reading["native_host_source_digest"],
                "direct_build_flags": FLAGS,
                "direct_devices": [reading["architecture"]],
                "direct_toolchains": [
                    {
                        "release": reading["compiler_release"],
                        "build": reading["compiler_build"],
                        "compiler_sha256": reading["compiler_sha256"],
                        "architecture": reading["architecture"],
                        "evidence_id": EVIDENCE_ID,
                    }
                ],
            }
        )
        if include_v2:
            jit["direct_host_source_digest_v2"] = source_digest_v2("native_host")
        new_deliveries = dict(deliveries)
        new_deliveries["jit"] = jit
        new_gpu = dict(gpu)
        new_gpu["deliveries"] = new_deliveries
        new_backends = dict(artifact["backends"])
        new_backends["gpu"] = new_gpu
        artifact["backends"] = new_backends
        artifacts.append(artifact)
    completed["artifacts"] = artifacts
    return with_root_digest(completed, REGISTRY_DOMAIN_TAG)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reading", type=Path, default=READING)
    arguments = parser.parse_args()
    reading = mapping(load_json(arguments.reading), "direct-JIT reading")
    registry = mapping(load_json(REGISTRY), "support registry")
    document = augmented(
        registry,
        reading,
        include_v2=not arguments.check or _records_v2(registry),
    )
    violations = schema_violations(document, load_json(SCHEMA))
    if violations:
        raise GenerationError(
            "direct-JIT registry violates schema:\n  " + "\n  ".join(violations)
        )
    payload = serialise_document(document)
    if arguments.check:
        stale = [
            path for path in (REGISTRY, PYTHON_COPY) if path.read_bytes() != payload
        ]
        if stale:
            print(
                "error: direct-JIT registry copies are stale: "
                + ", ".join(map(str, stale))
            )
            return 1
        print(f"{REGISTRY}: direct-JIT check passed")
        return 0
    REGISTRY.write_bytes(payload)
    PYTHON_COPY.write_bytes(payload)
    print(f"{REGISTRY}: bound {EVIDENCE_ID}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"error: {error}")
        raise SystemExit(2) from error
