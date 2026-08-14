#!/usr/bin/env python3
"""Aggregate independently judged PLAN/153 native-front-end family rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compute_identity_v2 import source_digest as source_digest_v2
from registry_common import GenerationError, load_json

ROOT = Path(__file__).resolve().parents[1]

#: Families the GPU parity readings must cover. The registry check refuses a
#: reading whose family set differs from the certified set, so this is the
#: same number stated where the reading is built rather than only where it is
#: consumed: an aggregation over a short roster would otherwise produce a
#: well-formed document that only fails much later.
GPU_FAMILIES = 15


def _rows(directory: Path, backend: str) -> list[dict[str, Any]]:
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise GenerationError(f"{directory} has no family readings")
    rows: list[dict[str, Any]] = []
    for path in paths:
        row = load_json(path)
        if (
            row.get("schema") != "plan153.native_frontend.family_parity.v1"
            or row.get("backend") != backend
            or int(row.get("mismatches", -1)) != 0
            or row.get("all_ids_equal_hf") is not True
            or row.get("one_python_to_rust_call_per_batch") is not True
            or row.get("gil_released") is not True
        ):
            raise GenerationError(f"{path}: native-front-end row did not pass")
        rows.append(row)
    families = [str(row.get("family")) for row in rows]
    if len(families) != len(set(families)):
        raise GenerationError("native-front-end inputs repeat a family")
    return rows


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def aggregate_cpu(directory: Path) -> dict[str, Any]:
    rows = _rows(directory, "cpu")
    if len(rows) != 11:
        raise GenerationError(f"CPU parity requires 11 rows, found {len(rows)}")
    fact_rows = [row.get("fast_cpu_build_facts") for row in rows]
    if not all(
        isinstance(value, dict) and value == fact_rows[0] for value in fact_rows
    ):
        raise GenerationError("CPU parity rows do not share one build identity")
    facts = fact_rows[0]
    assert isinstance(facts, dict)
    oracle_versions = {
        str((row.get("environment") or {}).get("tokenizers")) for row in rows
    }
    if len(oracle_versions) != 1:
        raise GenerationError("CPU parity rows do not share one oracle version")
    output_rows = []
    for row in rows:
        repair = row.get("repair")
        if not isinstance(repair, dict) or repair.get("all_ids_equal_hf") is not True:
            raise GenerationError(f"{row.get('family')}: repair probe did not pass")
        append_paths = repair.get("append_paths") or {}
        if int(append_paths.get("gigatoken_repair", 0)) < 1:
            raise GenerationError(
                f"{row.get('family')}: repair probe did not execute Gigatoken"
            )
        output_rows.append(
            {
                "family": row["family"],
                "artifact_sha256": row["artifact_sha256"],
                "documents": row["documents"],
                "characters": row["characters"],
                "mismatches": 0,
                "full_encode_equal_hf": True,
                "repair_equal_hf": True,
            }
        )
    return {
        "schema": "toktier.fast_cpu.native_frontend_parity.v1",
        "engine": {
            "upstream_project": facts["engine"],
            "version": facts["engine_version"],
            "delivery": facts["engine_delivery"],
            "module": facts["engine_module"],
            "source_digest": facts["source_digest"],
            "source_digest_v2": source_digest_v2("fast_cpu"),
            "build_flags": facts["build_flags"],
            "toolchain": facts["toolchain"],
        },
        "oracle": {
            "package": "tokenizers",
            "version": oracle_versions.pop(),
        },
        "unique_artifacts": 11,
        "model_families": 12,
        "documents": sum(int(row["documents"]) for row in rows),
        "characters": sum(int(row["characters"]) for row in rows),
        "mismatches": 0,
        "all_ids_equal_hf": True,
        "one_python_to_rust_call_per_batch": True,
        "gil_released": True,
        "rows": output_rows,
    }


#: What a GPU parity reading covers. ``full`` is the whole per-family
#: campaign of the certification protocol; ``spot`` is the bounded
#: re-take that rests on the cross-architecture record already on file.
#: Both are certified readings with zero mismatches; they differ in
#: scale, and a reader cannot tell which from a document count alone.
GPU_SCALES = ("full", "spot")


def aggregate_gpu(
    directory: Path, architecture: str, scale: str
) -> dict[str, Any]:
    from toktier.kernels.bindings import bare_sha256
    from toktier.kernels.prebuilt import (
        fatbin_digest,
        fatbin_path,
        load_manifest,
    )

    rows = _rows(directory, "gpu")
    if len(rows) != GPU_FAMILIES:
        raise GenerationError(
            f"GPU parity requires {GPU_FAMILIES} rows, found {len(rows)}"
        )
    manifest = load_manifest()
    host_rows = [row.get("native_host_build_facts") for row in rows]
    if not all(
        isinstance(value, dict) and value == host_rows[0] for value in host_rows
    ):
        raise GenerationError("GPU parity rows do not share one native-host identity")
    host_facts = host_rows[0]
    assert isinstance(host_facts, dict)
    bound_host_facts = {
        "host_source_digest": host_facts.get("source_digest"),
        "host_source_digest_v2": source_digest_v2("native_host"),
        "host_build_flags": host_facts.get("build_flags"),
        "host_toolchain": host_facts.get("toolchain"),
    }
    architecture_facts = manifest["architectures"].get(architecture)
    if not isinstance(architecture_facts, dict):
        raise GenerationError(f"the fatbin has no {architecture} image")
    devices = sorted({str((row.get("gpu") or {}).get("nvidia_smi")) for row in rows})
    if any(
        architecture.removeprefix("sm_") not in value.replace(".", "")
        for value in devices
    ):
        raise GenerationError(f"GPU readings do not report {architecture} hardware")
    output_rows = [
        {
            "family": row["family"],
            "artifact_sha256": row["artifact_sha256"],
            "documents": row["documents"],
            "characters": row["characters"],
            "mismatches": 0,
            "all_ids_equal_hf": True,
            "one_python_to_rust_call_per_batch": True,
            "gil_released": True,
        }
        for row in rows
    ]
    return {
        "schema": "toktier.gpu.native_frontend_parity.v1",
        "architecture": architecture,
        "scale": scale,
        "fatbin_digest": bare_sha256(fatbin_digest(fatbin_path().read_bytes())),
        "architecture_digest": bare_sha256(str(architecture_facts["digest"])),
        "toolchain": manifest["toolchain"],
        "native_host_build_facts": bound_host_facts,
        "devices": devices,
        "families": GPU_FAMILIES,
        "documents": sum(int(row["documents"]) for row in rows),
        "characters": sum(int(row["characters"]) for row in rows),
        "mismatches": 0,
        "all_ids_equal_hf": True,
        "one_python_to_rust_call_per_batch": True,
        "gil_released": True,
        "rows": output_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--architecture")
    parser.add_argument(
        "--scale",
        choices=GPU_SCALES,
        help="what this GPU reading covers; required for GPU aggregation",
    )
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.backend == "cpu":
        if arguments.architecture is not None:
            raise GenerationError("--architecture is only valid for GPU readings")
        if arguments.scale is not None:
            raise GenerationError("--scale is only valid for GPU readings")
        document = aggregate_cpu(arguments.input_dir)
    else:
        if arguments.architecture not in {"sm_89", "sm_120"}:
            raise GenerationError(
                "GPU aggregation requires --architecture sm_89 or sm_120"
            )
        if arguments.scale is None:
            raise GenerationError(
                "GPU aggregation requires --scale full or --scale spot: the "
                "reading has to say what it covers, and no document count "
                "says it on its own"
            )
        document = aggregate_gpu(
            arguments.input_dir, arguments.architecture, arguments.scale
        )
    _write(arguments.out, document)
    print(f"wrote {arguments.out}: {len(document['rows'])} rows, zero divergence")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"error: {error}")
        raise SystemExit(2) from error
