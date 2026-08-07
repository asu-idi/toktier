#!/usr/bin/env python3
"""Import the frozen, judged repair predicates into the product tree.

This is a one-way, digest-checked importer.  It does not derive Unicode
properties from the running Python or Rust libraries: doing so would make a
certificate depend on whichever Unicode release happened to be installed.
The source tables were generated from the judged HF tokenizers stack and are
copied into a compact, deterministic runtime representation here.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY_ROOT / "src" / "toktier" / "repair" / "tables"

FAMILIES = (
    "deepseek_v3",
    "deepseek_v4_flash",
    "glm_5_2",
    "gpt_oss_120b",
    "llama_3_1_8b",
    "minimax_m3",
    "ministral_3_8b",
    "nemotron_3_nano_4b",
    "olmo_3_7b",
    "qwen3_5_08b",
    "qwen3_8b",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def import_tables(source: Path) -> None:
    shared = _load(source / "_shared.json")
    raw = base64.b64decode(shared["pclass_b64"], validate=True)
    observed = hashlib.sha256(raw).hexdigest()
    expected = str(shared["pclass_sha256"])
    if observed != expected:
        raise ValueError(
            f"repair pclass digest mismatch: {observed} != {expected}"
        )
    if len(raw) != 0x110000:
        raise ValueError(f"repair pclass has {len(raw)} rows, expected 0x110000")

    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        item = _load(source / f"{family}.json")
        if item.get("family") != family:
            raise ValueError(f"{family}.json names {item.get('family')!r}")
        if item.get("boundary_kind") != "letter_space":
            raise ValueError(f"{family}: expected the byte-BPE family gate")
        if item.get("pclass_sha256") != expected:
            raise ValueError(f"{family}: shared pclass binding drifted")
        rows.append(
            {
                "family": family,
                "artifact_sha256": item["tokenizer_json_sha256"],
                "margin": item["margin"],
                "effective_l_max": item["effective_l_max"],
                "has_normalizer": item["has_normalizer"],
                "source_table_sha256": item["content_sha256"],
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compressed = zlib.compress(raw, level=9)
    (OUTPUT_DIR / "repair_pclass.v1.zlib").write_bytes(compressed)
    payload = {
        "schema": "toktier.fast_repair_families.v1",
        "config_id": "toktier-fast-repair-v1",
        "source": "archived frozen repair tables",
        "pclass": {
            "file": "repair_pclass.v1.zlib",
            "encoding": "zlib(raw uint8[0x110000])",
            "labels": "OSLNM",
            "raw_sha256": expected,
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        },
        "families": rows,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    (OUTPUT_DIR / "fast_repair_families.v1.json").write_text(
        rendered, encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        help="directory containing the archived _shared.json and family tables",
    )
    args = parser.parse_args()
    import_tables(args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
