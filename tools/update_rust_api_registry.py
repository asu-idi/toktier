#!/usr/bin/env python3
"""Apply or verify the evidence-bound public Rust runtime build row."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

from compute_identity_v2 import source_digest as source_digest_v2
from fast_cpu_source_identity import source_digest as fast_cpu_source_digest
from native_host_source_identity import source_digest as native_host_source_digest
from registry_common import (
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    load_json,
    schema_violations,
    serialise_document,
    with_root_digest,
)
from rust_api_source_identity import source_digest as rust_api_source_digest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tables" / "support_registry.json"
PACKAGED_REGISTRY = ROOT / "src/toktier/routing/tables/support_registry.v1.json"
SCHEMA = ROOT / "schemas/support_registry.schema.json"
BINDING = ROOT / "tools/rust_api_binding.json"
CPU_READING = ROOT / "readings/fast_cpu_native_frontend_parity.json"
GPU_READINGS = {
    "sm_89": ROOT / "readings/gpu_native_frontend_sm89_parity.json",
    "sm_120": ROOT / "readings/gpu_native_frontend_sm120_parity.json",
}
MATRIX_READINGS = {
    "cpu": ROOT / "readings/rust_api_matrix_cpu.json",
    "sm_89": ROOT / "readings/rust_api_matrix_sm89.json",
    "sm_120": ROOT / "readings/rust_api_matrix_sm120.json",
}


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be an object")
    return value


def _verify_build(build: dict[str, Any], *, label: str) -> None:
    if not isinstance(build.get("build_flags"), list) or not build["build_flags"]:
        raise GenerationError(f"{label} has no exact build flags")
    if not isinstance(build.get("toolchain"), str) or not build["toolchain"]:
        raise GenerationError(f"{label} has no exact toolchain")
    if not isinstance(build.get("evidence_id"), str) or not build["evidence_id"]:
        raise GenerationError(f"{label} has no evidence id")


def verify_binding(binding: dict[str, Any]) -> None:
    expected = {
        "runtime": "rust_api",
        "source_digest": rust_api_source_digest(),
        "fast_cpu_source_digest": fast_cpu_source_digest(),
        "native_host_source_digest": native_host_source_digest(),
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise GenerationError(
                f"Rust API binding {key} does not match the current tree "
                f"(recorded={binding.get(key)!r}, current={value!r})"
            )
    expected_v2 = {
        "source_digest_v2": source_digest_v2("rust_api"),
        "fast_cpu_source_digest_v2": source_digest_v2("fast_cpu"),
        "native_host_source_digest_v2": source_digest_v2("native_host"),
    }
    present_v2 = set(expected_v2) & set(binding)
    if present_v2 and present_v2 != set(expected_v2):
        raise GenerationError("Rust API binding has an incomplete v2 identity tuple")
    for key in present_v2:
        if binding[key] != expected_v2[key]:
            raise GenerationError(f"Rust API binding {key} does not match the tree")
    _verify_build(binding, label="Rust API default build")
    additional = binding.get("additional_builds", [])
    if not isinstance(additional, list):
        raise GenerationError("Rust API additional_builds must be an array")
    names: set[str] = set()
    identities = {(tuple(binding["build_flags"]), binding["toolchain"])}
    for index, raw in enumerate(additional):
        build = _mapping(raw, label=f"Rust API additional build {index}")
        name = build.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise GenerationError("Rust API additional build names must be unique")
        names.add(name)
        _verify_build(build, label=f"Rust API additional build {name}")
        identity = (tuple(build["build_flags"]), build["toolchain"])
        if identity in identities:
            raise GenerationError(f"Rust API additional build {name} is a duplicate")
        identities.add(identity)
        if build.get("evidence_reading") != "rust_direct_jit_sm120":
            raise GenerationError(
                f"Rust API additional build {name} has an unsupported evidence reading"
            )


def verify_shared_evidence(binding: dict[str, Any]) -> None:
    cpu = _mapping(load_json(CPU_READING), label="CPU native-front-end reading")
    engine = _mapping(cpu.get("engine"), label="CPU engine facts")
    if (
        cpu.get("schema") != "toktier.fast_cpu.native_frontend_parity.v1"
        or cpu.get("unique_artifacts") != 11
        or cpu.get("model_families") != 12
        or cpu.get("mismatches") != 0
        or cpu.get("all_ids_equal_hf") is not True
        or engine.get("source_digest") != binding["fast_cpu_source_digest"]
    ):
        raise GenerationError(
            "CPU native-front-end evidence does not admit this Rust build"
        )
    if engine.get("source_digest_v2") not in {
        None,
        source_digest_v2("fast_cpu"),
    }:
        raise GenerationError("CPU native-front-end v2 identity drifted")
    for architecture, path in GPU_READINGS.items():
        reading = _mapping(load_json(path), label=f"{architecture} GPU reading")
        host = _mapping(reading.get("native_host_build_facts"), label="GPU host facts")
        if (
            reading.get("schema") != "toktier.gpu.native_frontend_parity.v1"
            or reading.get("architecture") != architecture
            or reading.get("families") != 14
            or reading.get("mismatches") != 0
            or reading.get("all_ids_equal_hf") is not True
            or host.get("host_source_digest") != binding["native_host_source_digest"]
        ):
            raise GenerationError(
                f"{architecture} native-front-end evidence does not admit "
                "this Rust build"
            )
        if host.get("host_source_digest_v2") not in {
            None,
            source_digest_v2("native_host"),
        }:
            raise GenerationError(f"{architecture} native-host v2 identity drifted")


def verify_public_matrix(binding: dict[str, Any], *, bootstrap: bool) -> None:
    for target, path in MATRIX_READINGS.items():
        reading = _mapping(load_json(path), label=f"Rust API {target} matrix")
        if (
            reading.get("schema") != "toktier.rust_api.matrix.v1"
            or reading.get("runtime_source_digest") != binding["source_digest"]
            or reading.get("fast_cpu_source_digest")
            != binding["fast_cpu_source_digest"]
            or reading.get("native_host_source_digest")
            != binding["native_host_source_digest"]
            or reading.get("toolchain") != binding["toolchain"]
            or reading.get("build_flags") != binding["build_flags"]
            or reading.get("families") != 14
            or reading.get("documents") != 98
            or reading.get("mismatches") != 0
            or (not bootstrap and reading.get("runtime_build_certified") is not True)
        ):
            raise GenerationError(
                f"Rust API {target} matrix does not match the binding"
            )
        rows = reading.get("rows")
        if not isinstance(rows, list) or len(rows) != 14:
            raise GenerationError(
                f"Rust API {target} matrix has an incomplete family roster"
            )
        backend_counts: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("mismatches") != 0:
                raise GenerationError(f"Rust API {target} matrix has a failing row")
            backend = str(row.get("backend"))
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        expected = (
            {"FastCpu": 11, "HuggingFace": 3}
            if target == "cpu"
            else {"Gpu": 14}
        )
        if backend_counts != expected:
            raise GenerationError(
                f"Rust API {target} matrix backend roster is {backend_counts}, "
                f"expected {expected}"
            )


def verify_additional_builds(binding: dict[str, Any], *, bootstrap: bool) -> None:
    for raw in binding.get("additional_builds", []):
        build = _mapping(raw, label="Rust API additional build")
        name = str(build["name"])
        if build["evidence_reading"] != "rust_direct_jit_sm120":
            raise GenerationError(f"unsupported evidence for additional build {name}")
        reading = _mapping(
            load_json(ROOT / "readings/rust_direct_jit_sm120.json"),
            label=f"Rust API additional build {name} evidence",
        )
        if (
            reading.get("schema") != "toktier.rust_direct_jit.matrix.v1"
            or reading.get("runtime_source_digest") != binding["source_digest"]
            or reading.get("native_host_source_digest")
            != binding["native_host_source_digest"]
            or reading.get("rust_toolchain") != build["toolchain"]
            or reading.get("rust_build_flags") != build["build_flags"]
            or reading.get("families") != 14
            or reading.get("documents") != 98
            or reading.get("mismatches") != 0
            or (
                not bootstrap
                and reading.get("runtime_build_certified") is not True
            )
            or (not bootstrap and reading.get("experimental_opt_in") is not False)
        ):
            raise GenerationError(
                f"Rust API additional build {name} does not match its evidence"
            )
        rows = reading.get("rows")
        if not isinstance(rows, list) or len(rows) != 14 or any(
            not isinstance(row, dict)
            or row.get("mismatches") != 0
            or row.get("certified") is not True
            for row in rows
        ):
            raise GenerationError(
                f"Rust API additional build {name} has incomplete certified rows"
            )


def _records_v2(registry: dict[str, Any]) -> bool:
    return any(
        isinstance(row, dict) and "source_digest_v2" in row
        for row in registry.get("runtime_builds", [])
    )


def augmented_document(
    registry: dict[str, Any],
    binding: dict[str, Any],
    *,
    bootstrap: bool = False,
    include_v2: bool | None = None,
) -> dict[str, Any]:
    if include_v2 is None:
        include_v2 = _records_v2(registry)
    verify_binding(binding)
    verify_shared_evidence(binding)
    verify_public_matrix(binding, bootstrap=bootstrap)
    verify_additional_builds(binding, bootstrap=bootstrap)
    completed = copy.deepcopy(registry)
    common = {
        key: binding[key]
        for key in (
            "runtime",
            "source_digest",
            "fast_cpu_source_digest",
            "native_host_source_digest",
        )
    }
    if include_v2:
        common.update(
            {
                "source_digest_v2": source_digest_v2("rust_api"),
                "fast_cpu_source_digest_v2": source_digest_v2("fast_cpu"),
                "native_host_source_digest_v2": source_digest_v2("native_host"),
            }
        )
    builds = [binding, *binding.get("additional_builds", [])]
    completed["runtime_builds"] = [
        {
            **common,
            **{
                key: build[key]
                for key in ("build_flags", "toolchain", "evidence_id")
            },
        }
        for build in builds
    ]
    return with_root_digest(completed, REGISTRY_DOMAIN_TAG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="admit exact candidate builds before the final certified replay",
    )
    arguments = parser.parse_args(argv)
    if arguments.check and arguments.bootstrap:
        parser.error("--check and --bootstrap are mutually exclusive")
    registry = _mapping(load_json(REGISTRY), label="support registry")
    binding = _mapping(load_json(BINDING), label="Rust API binding")
    generated = augmented_document(
        registry,
        binding,
        bootstrap=arguments.bootstrap,
        include_v2=(
            not arguments.check or arguments.bootstrap or _records_v2(registry)
        ),
    )
    violations = schema_violations(generated, load_json(SCHEMA))
    if violations:
        raise GenerationError(
            "Rust API registry violates its schema:\n  " + "\n  ".join(violations)
        )
    payload = serialise_document(generated)
    if arguments.check:
        problems = [
            str(path)
            for path in (REGISTRY, PACKAGED_REGISTRY)
            if not path.is_file() or path.read_bytes() != payload
        ]
        for path in problems:
            print(
                f"error: {path} is not the generated Rust API registry",
                file=sys.stderr,
            )
        return 1 if problems else 0
    REGISTRY.write_bytes(payload)
    PACKAGED_REGISTRY.write_bytes(payload)
    print(f"updated {REGISTRY} and {PACKAGED_REGISTRY}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
