#!/usr/bin/env python3
"""Run the focused 11-artifact parity gate through an installed core wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import tokenizers

import toktier

NATIVE_SHA256 = (
    "9a701047dafa1cdebc168851d0548a0ca"
    "af08d0523d70911cc7a24112ccf92a3"
)
ENGINE_MODULE = "toktier._vendor.gigatoken_rs"

_BASE = (
    "user: Explain why incremental tokenization must repair a boundary.\n"
    "assistant: The next byte can change the final merge, so retain "
    "evidence. 世界 123\n"
) * 96
_APPENDS = (
    "user: Give the exact invariant in one sentence.\n",
    "assistant: Stored-prefix repair must equal a fresh HF encode, token by token.\n",
    "user: Include UTF-8: café, Ελληνικά, العربية, हिन्दी, 🌵.\n",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _artifact_path(home: Path, family: str) -> Path:
    candidates = []
    for marker in (home / "cache" / "artifacts").glob("*/.toktier-verified.json"):
        record = _json(marker)
        if record.get("family") == family:
            candidates.append(marker.parent / "tokenizer.json")
    if len(candidates) != 1 or not candidates[0].is_file():
        raise ValueError(f"{family}: expected one verified cached tokenizer")
    return candidates[0]


def _external_gigatoken_present() -> bool:
    try:
        importlib.metadata.distribution("gigatoken")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def run(home: Path, binding_path: Path, wheel: Path) -> dict[str, Any]:
    binding = _json(binding_path)
    families = binding.get("loadable_families")
    if not isinstance(families, list) or len(families) != 11:
        raise ValueError("binding must name eleven loadable artifacts")
    if toktier.__version__ != "0.1.0":
        raise ValueError("focused gate must run from installed toktier 0.1.0")
    if _external_gigatoken_present() or importlib.util.find_spec("gigatoken"):
        raise ValueError("a top-level Gigatoken distribution contaminates the gate")

    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="toktier-focused-parity-") as temporary:
        stores = Path(temporary)
        for family_value in families:
            family = str(family_value)
            tokenizer_path = _artifact_path(home, family)
            artifact_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
            reference = tokenizers.Tokenizer.from_file(str(tokenizer_path))
            tok = toktier.load(family, store=stores / family)
            transcript = _BASE
            counts: list[int] = []
            all_equal = True
            for turn in range(4):
                if turn:
                    transcript += _APPENDS[turn - 1]
                actual = tok.encode(transcript, session="focused-wheel-gate")
                expected = reference.encode(
                    transcript, add_special_tokens=False
                ).ids
                equal = list(actual.ids) == list(expected)
                all_equal = all_equal and equal
                counts.append(len(actual.ids))
                if not equal:
                    raise ValueError(f"{family}: turn {turn} diverged from HF")
            report = tok.explain()
            repair = report.get("session_repair")
            if not isinstance(repair, dict):
                raise ValueError(f"{family}: no session-repair report")
            path_counts = repair.get("path_counts")
            if not isinstance(path_counts, dict):
                raise ValueError(f"{family}: no repair path counts")
            normalized_paths = Counter(
                {str(name): int(count) for name, count in path_counts.items()}
            )
            if normalized_paths["gigatoken_repair"] < 1:
                raise ValueError(f"{family}: append did not execute Gigatoken repair")
            if report.get("backend") != "fast_cpu":
                raise ValueError(f"{family}: planner did not select fast_cpu")
            rows.append(
                {
                    "family": family,
                    "artifact_sha256": artifact_sha256,
                    "plan_backend": "fast_cpu",
                    "path_counts": dict(sorted(normalized_paths.items())),
                    "turn_token_counts": counts,
                    "all_turns_equal_hf": all_equal,
                }
            )
            tok.close()

    return {
        "schema": "toktier.fast_cpu.focused_parity.v2",
        "release": {
            "distribution": "toktier",
            "version": "0.1.0",
            "wheel": wheel.name,
            "wheel_sha256": wheel_sha256,
            "external_gigatoken_distribution_present": False,
        },
        "oracle": {"package": "tokenizers", "version": tokenizers.__version__},
        "engine": {
            "upstream_project": "gigatoken",
            "version": str(binding["engine_version"]),
            "delivery": "vendored",
            "module": ENGINE_MODULE,
            "native_sha256": NATIVE_SHA256,
        },
        "unique_artifacts": 11,
        "model_families": 12,
        "turns_per_artifact": 4,
        "all_ids_equal_hf": True,
        "all_executed_gigatoken_repair": True,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toktier-home", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    document = run(
        arguments.toktier_home.resolve(),
        arguments.binding.resolve(),
        arguments.wheel.resolve(),
    )
    arguments.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {arguments.output}: 11/11 artifacts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
