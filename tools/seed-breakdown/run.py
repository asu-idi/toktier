#!/usr/bin/env python3
"""Run independent-process seed-breakdown samples and aggregate them."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import tempfile
from typing import Any

DEFAULT_PHASES = [
    "public_encode",
    "public_seed_memory",
    "reference_core",
    "fast_cpu_encode",
    "router_encode",
    "router_append",
    "encoding_clone",
    "tail_fill",
    "precomputed_append",
    "boundary_search",
    "store_put_real_memory",
    "store_put_real_tracked",
    "store_put_precomputed_full_memory",
    "store_put_precomputed_full_tracked",
    "store_put_precomputed_no_post",
    "store_put_precomputed_blocks_no_boundary",
    "store_put_precomputed_boundary_only",
    "store_put_precomputed_no_tracking",
    "content_digest",
    "recovery_sha256",
    "all_ids",
    "sqlite_save",
    "sqlite_load",
    "public_seed_sqlite",
]

# PLAN/163 W3 direct cells: span-bridge, lazy-span, and payload-hash
# measurements. Every cell is a host-side CPU measurement; prototype cells
# assert element/bit equality against the product implementation inside the
# measured process before the sample is accepted.
W3_PHASES = [
    "store_seed_soa_shape",
    "store_seed_lazy_shape",
    "store_seed_lazy_shape_overlap",
    "store_seed_soa_shape_unicode",
    "store_seed_lazy_shape_unicode",
    "store_seed_lazy_shape_overlap_unicode",
    "store_seed_concurrent4",
    "store_seed_concurrent4_overlap",
    "store_seed_overlap_longrun",
    "spans_direct",
    "spans_soa_direct",
    "spans_soa_proto",
    "spans_lazy_closure",
    "spans_lazy_tail_window",
    "spans_checkpoint_build",
    "spans_checkpoint_window",
    "spans_direct_unicode",
    "spans_soa_direct_unicode",
    "spans_soa_proto_unicode",
    "spans_lazy_closure_unicode",
    "spans_lazy_tail_window_unicode",
    "spans_checkpoint_build_unicode",
    "spans_checkpoint_window_unicode",
    "payload_digest_seed_shape",
    "payload_digest_append_shape",
    "payload_digest_incremental_proto",
    "payload_digest_incremental_direct",
    "payload_digest_chunked",
    "added_gate_scan",
]


def percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for phase in sorted({row["phase"] for row in rows}):
        selected = [row for row in rows if row["phase"] == phase]
        values = [int(row["elapsed_ns"]) for row in selected]
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        representative = min(selected, key=lambda row: abs(row["elapsed_ns"] - median))
        output[phase] = {
            "n": len(values),
            "min_ns": min(values),
            "p50_ns": median,
            "p95_ns": percentile(values, 0.95),
            "p99_ns": percentile(values, 0.99),
            "max_ns": max(values),
            "mad_ns": statistics.median(deviations),
            "token_count": representative["token_count"],
            "actual_backend": representative.get("actual_backend"),
            "actual_path": representative.get("actual_path"),
            "representative_details": representative.get("details", {}),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--binary", type=pathlib.Path)
    parser.add_argument("--phase", action="append", dest="phases")
    parser.add_argument("--numactl", action="store_true")
    parser.add_argument("--suite", choices=("plan161", "w3"), default="plan161")
    parser.add_argument("--sample-timeout", type=float, default=900.0)
    parser.add_argument(
        "--rayon-threads",
        type=int,
        default=1,
        help=(
            "Bounded worker pool size for the measured process "
            "(RAYON_NUM_THREADS). The overlap cells run their digest scan "
            "on this pool; the value is recorded in the aggregate "
            "environment so overlap readings stay bound to their pool shape."
        ),
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.rayon_threads < 1:
        parser.error("--rayon-threads must be positive")

    root = pathlib.Path(__file__).resolve().parents[2]
    binary = (
        args.binary
        or root / "tools/seed-breakdown/target/release/toktier-seed-breakdown"
    )
    if not binary.is_file():
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(root / "tools/seed-breakdown/Cargo.toml"),
            ],
            cwd=root,
            check=True,
        )
    commit = os.environ.get("TOKTIER_PROFILE_PRODUCT_COMMIT")
    if not commit:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    thread_environment = {
        "TOKENIZERS_PARALLELISM": "false",
        "RAYON_NUM_THREADS": str(args.rayon_threads),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    environment = os.environ.copy()
    environment.update(thread_environment)
    environment["TOKTIER_PROFILE_PRODUCT_COMMIT"] = commit
    if args.phases is not None:
        phases = args.phases
    elif args.suite == "w3":
        phases = W3_PHASES
    elif args.device == "gpu":
        phases = [
            "public_encode",
            "public_encode_offsets",
            "public_seed_memory",
            "public_seed_sqlite",
        ]
    else:
        phases = DEFAULT_PHASES
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_path = args.output.with_suffix(".jsonl")
    with raw_path.open("w", encoding="utf-8") as raw:
        for phase in phases:
            for sample in range(args.samples):
                with tempfile.TemporaryDirectory(
                    prefix=f"toktier-seed-{phase}-"
                ) as home:
                    command = [
                        str(binary),
                        "--phase",
                        phase,
                        "--artifact",
                        str(args.artifact),
                        "--device",
                        args.device,
                        "--home",
                        home,
                    ]
                    if args.numactl:
                        command = ["numactl", "--membind=0,1", *command]
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=root,
                            env=environment,
                            check=True,
                            text=True,
                            stdout=subprocess.PIPE,
                            timeout=args.sample_timeout,
                        )
                    except subprocess.TimeoutExpired as error:
                        raise SystemExit(
                            f"{phase} sample {sample} exceeded "
                            f"{args.sample_timeout}s; no partial row was recorded"
                        ) from error
                    except subprocess.CalledProcessError as error:
                        raise SystemExit(
                            f"{phase} sample {sample} exited with "
                            f"{error.returncode}; no partial row was recorded"
                        ) from error
                    row = json.loads(completed.stdout)
                    row["sample_index"] = sample
                    rows.append(row)
                    raw.write(json.dumps(row, sort_keys=True) + "\n")
                    raw.flush()
    identities = {
        (
            row["product_commit"],
            row["rust_api_source_sha256"],
            row["fast_cpu_source_sha256"],
            row["native_host_source_sha256"],
        )
        for row in rows
    }
    if len(identities) != 1:
        raise SystemExit(f"observations used multiple product identities: {identities}")
    payload = {
        "schema": "toktier.seed_breakdown.aggregate.v1",
        "device": args.device,
        "samples_per_phase": args.samples,
        "artifact": str(args.artifact),
        "input_bytes": 4 * 1024 * 1024,
        "identity": next(iter(identities)),
        "raw_jsonl": raw_path.name,
        # CHANGE-162 C5: the bounded-pool shape is part of the reading's
        # environment; overlap on/off cells stay separate distributions.
        "thread_environment": thread_environment,
        "phases": aggregate(rows),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
