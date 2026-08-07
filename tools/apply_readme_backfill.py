#!/usr/bin/env python3
"""Apply resolved readings from readings/readme_backfill.json to the docs.

Replaces ``<!-- TODO:<kind>:<key> -->`` markers with formatted values for
every key whose value is resolved (non-null). Unresolved keys keep their
markers so the remaining work stays visible. Traceability lives in the
backfill file itself (value + source per key); the documents carry the
plain values.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [
    ROOT / "README.md",
    ROOT / "docs" / "support-matrix.md",
    ROOT / "docs" / "integration" / "dynamo.md",
    ROOT / "ARCHITECTURE.md",
    ROOT / "ROADMAP.md",
    ROOT / "CITATION.cff",
    ROOT / "NOTICE",
]


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}" if abs(value) >= 10_000 else str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def main() -> int:
    backfill = json.loads((ROOT / "readings" / "readme_backfill.json").read_text())
    keys: dict[str, dict[str, object]] = backfill["keys"]
    replaced: dict[str, int] = {}
    unresolved: list[str] = []

    for key, entry in keys.items():
        if entry.get("value") is None:
            unresolved.append(key)
            continue
        marker = re.compile(
            r"<!--\s*TODO:(?:number|pending):" + re.escape(key) + r"\s*-->"
        )
        text_value = format_value(entry["value"])
        for path in DOC_FILES:
            if not path.exists():
                continue
            original = path.read_text(encoding="utf-8")
            updated, count = marker.subn(text_value, original)
            if count:
                path.write_text(updated, encoding="utf-8")
                replaced[key] = replaced.get(key, 0) + count

    print(f"replaced {sum(replaced.values())} markers for {len(replaced)} keys")
    if unresolved:
        print("unresolved (markers kept):", ", ".join(sorted(unresolved)))
    leftovers = 0
    for path in DOC_FILES:
        if path.exists():
            leftovers += len(re.findall(r"<!--\s*TODO:", path.read_text()))
    print(f"markers remaining across documents: {leftovers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
