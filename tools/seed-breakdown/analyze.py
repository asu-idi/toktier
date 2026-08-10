#!/usr/bin/env python3
"""Derive labelled median contrasts from seed-breakdown aggregate files."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, cast


def load(path: pathlib.Path) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema") != "toktier.seed_breakdown.aggregate.v1":
        raise SystemExit(f"unexpected aggregate schema in {path}")
    for name, row in payload["phases"].items():
        if row["n"] < 30:
            raise SystemExit(f"{path}: {name} has only {row['n']} observations")
    return payload


def p50(payload: dict[str, Any], phase: str) -> float:
    return float(payload["phases"][phase]["p50_ns"])


def direct(payload: dict[str, Any], phase: str) -> dict[str, Any]:
    row = payload["phases"][phase]
    return {
        "kind": "direct_p50",
        "phase": phase,
        "ns": row["p50_ns"],
        "n": row["n"],
        "mad_ns": row["mad_ns"],
    }


def contrast(payload: dict[str, Any], minuend: str, subtrahend: str) -> dict[str, Any]:
    return {
        "kind": "difference_of_independent_p50s",
        "minuend": minuend,
        "subtrahend": subtrahend,
        "ns": p50(payload, minuend) - p50(payload, subtrahend),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True, type=pathlib.Path)
    parser.add_argument("--cpu-supplement", type=pathlib.Path)
    parser.add_argument("--gpu", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    cpu = load(args.cpu)
    if args.cpu_supplement is not None:
        supplement = load(args.cpu_supplement)
        if supplement["identity"] != cpu["identity"]:
            raise SystemExit("CPU supplement does not bind the primary CPU identity")
        overlap = set(cpu["phases"]) & set(supplement["phases"])
        if overlap:
            phases = sorted(overlap)
            raise SystemExit(f"CPU supplement repeats primary phases: {phases}")
        cpu["phases"].update(supplement["phases"])
    gpu = load(args.gpu)
    if cpu["identity"] != gpu["identity"]:
        raise SystemExit("CPU and GPU aggregates do not bind the same product identity")

    cpu_seed = p50(cpu, "public_seed_memory")
    gpu_seed = p50(gpu, "public_seed_memory")
    modeled_store_after_append = p50(cpu, "store_put_precomputed_full_memory") - p50(
        cpu, "precomputed_append"
    )
    modeled_public_remainder = (
        cpu_seed - p50(cpu, "router_append") - modeled_store_after_append
    )
    result = {
        "schema": "toktier.seed_breakdown.analysis.v1",
        "identity": cpu["identity"],
        "interpretation": {
            "direct_p50": "the median of 31 directly timed independent processes",
            "difference_of_independent_p50s": (
                "an explanatory contrast, not an internal exclusive timer; negative "
                "values and factorial interactions must remain visible"
            ),
            "timed_boundary": (
                "runtime/tokenizer construction and correctness comparison are "
                "excluded; the public seed timer begins immediately before "
                "Session::seed and ends when its complete Rust Encoding has returned"
            ),
        },
        "cpu": {
            "public_seed": direct(cpu, "public_seed_memory"),
            "engine_fast_cpu": direct(cpu, "fast_cpu_encode"),
            "router_after_engine": contrast(cpu, "router_encode", "fast_cpu_encode"),
            "empty_tail_population": contrast(cpu, "router_append", "router_encode"),
            "real_store_after_router_append_validation": contrast(
                cpu, "store_put_real_memory", "router_append"
            ),
            "public_after_real_store_validation": contrast(
                cpu, "public_seed_memory", "store_put_real_memory"
            ),
            "store_after_precomputed_append": {
                "kind": "counterfactual_difference_of_independent_p50s",
                "ns": modeled_store_after_append,
                "minuend": "store_put_precomputed_full_memory",
                "subtrahend": "precomputed_append",
            },
            "public_remainder_after_modeled_store": {
                "kind": "algebraic_closure_residual",
                "ns": modeled_public_remainder,
            },
            "all_ids_copy": direct(cpu, "all_ids"),
            "public_remainder_after_all_ids": {
                "kind": "modeled_public_remainder_minus_direct_p50",
                "ns": modeled_public_remainder - p50(cpu, "all_ids"),
            },
            "store_counterfactuals": {
                "encoding_clone": direct(cpu, "encoding_clone"),
                "tail_fill_with_precloned_encoding": direct(cpu, "tail_fill"),
                "precomputed_append": direct(cpu, "precomputed_append"),
                "precomputed_append_closure_residual_ns": (
                    p50(cpu, "precomputed_append")
                    - p50(cpu, "encoding_clone")
                    - p50(cpu, "tail_fill")
                ),
                "store_after_precomputed_append": contrast(
                    cpu,
                    "store_put_precomputed_full_memory",
                    "precomputed_append",
                ),
                "content_tracking_contrast": contrast(
                    cpu,
                    "store_put_precomputed_full_memory",
                    "store_put_precomputed_no_tracking",
                ),
                "recovery_tracking_contrast": contrast(
                    cpu,
                    "store_put_precomputed_full_tracked",
                    "store_put_precomputed_full_memory",
                ),
                "post_text_full_contrast": contrast(
                    cpu,
                    "store_put_precomputed_full_memory",
                    "store_put_precomputed_no_post",
                ),
                "block_chain_only_contrast": contrast(
                    cpu,
                    "store_put_precomputed_blocks_no_boundary",
                    "store_put_precomputed_no_post",
                ),
                "boundary_seal_only_contrast": contrast(
                    cpu,
                    "store_put_precomputed_boundary_only",
                    "store_put_precomputed_no_post",
                ),
                "post_text_factorial_interaction_ns": (
                    p50(cpu, "store_put_precomputed_full_memory")
                    - p50(cpu, "store_put_precomputed_blocks_no_boundary")
                    - p50(cpu, "store_put_precomputed_boundary_only")
                    + p50(cpu, "store_put_precomputed_no_post")
                ),
                "boundary_search_direct": direct(cpu, "boundary_search"),
                "content_digest_direct": direct(cpu, "content_digest"),
                "recovery_sha256_direct": direct(cpu, "recovery_sha256"),
            },
            "durability": {
                "public_sqlite_increment": contrast(
                    cpu, "public_seed_sqlite", "public_seed_memory"
                ),
                "sqlite_save_direct": direct(cpu, "sqlite_save"),
                "sqlite_load_and_verify_direct": direct(cpu, "sqlite_load"),
            },
            "share_of_memory_seed": {
                "engine_fast_cpu_percent": p50(cpu, "fast_cpu_encode") / cpu_seed * 100,
                "router_after_engine_percent": (
                    p50(cpu, "router_encode") - p50(cpu, "fast_cpu_encode")
                )
                / cpu_seed
                * 100,
                "empty_tail_population_percent": (
                    p50(cpu, "router_append") - p50(cpu, "router_encode")
                )
                / cpu_seed
                * 100,
                "store_after_append_percent": (
                    modeled_store_after_append / cpu_seed * 100
                ),
                "public_after_store_percent": modeled_public_remainder / cpu_seed * 100,
            },
        },
        "gpu": {
            "public_seed": direct(gpu, "public_seed_memory"),
            "id_only_route": direct(gpu, "public_encode"),
            "known_id_span_bridge": contrast(
                gpu, "public_encode_offsets", "public_encode"
            ),
            "state_store_and_return_after_spans": contrast(
                gpu, "public_seed_memory", "public_encode_offsets"
            ),
            "sqlite_increment": contrast(
                gpu, "public_seed_sqlite", "public_seed_memory"
            ),
            "share_of_memory_seed": {
                "id_only_percent": p50(gpu, "public_encode") / gpu_seed * 100,
                "span_bridge_percent": (
                    p50(gpu, "public_encode_offsets") - p50(gpu, "public_encode")
                )
                / gpu_seed
                * 100,
                "state_store_return_percent": (
                    gpu_seed - p50(gpu, "public_encode_offsets")
                )
                / gpu_seed
                * 100,
            },
        },
        "cross_host_context_only": {
            "cpu_seed_p50_ns": cpu_seed,
            "gpu_seed_p50_ns": gpu_seed,
            "cpu_over_gpu_seed_ratio": cpu_seed / gpu_seed,
            "warning": (
                "CPU and GPU observations were collected on different host CPUs; this "
                "ratio is context, not a controlled device-only speedup"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
