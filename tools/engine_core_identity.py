#!/usr/bin/env python3
"""Digests of the corrected CPU engine's core, for readings taken on it.

A full-corpus engine differential of a directly judged face is a reading
about the engine core, not about one build: it stays valid while the
engine core is unchanged. The objects that make up that core are:

* the vendored engine crate ``crates/toktier-gigatoken-core`` (every
  file, by content);
* the two repair tables, ``src/toktier/repair/tables/fast_repair_families.v1.json``
  and ``src/toktier/repair/tables/repair_pclass.v1.zlib``;
* the engine patch ``packaging/fast_cpu/gigatoken-toktier-pinned-1.patch``.

``engine_core_digests()`` returns their sha256 values. The crate digest is
domain-separated and covers the sorted list of ``<sha256>  <path>`` lines
of every regular file under the crate (paths relative to the crate,
``/``-separated), so it is the same for a checkout, an archive and an
unpacked source tree. A reading records these values
(``engine_core`` in ``schemas/engine_differential_reading.schema.json``)
and ``tools/generate_registry.py`` requires them to equal the tree's.

Usage::

    python tools/engine_core_identity.py          # prints the digests as JSON
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORE_CRATE = Path("crates") / "toktier-gigatoken-core"
REPAIR_TABLES = Path("src") / "toktier" / "repair" / "tables"
REPAIR_FAMILIES = REPAIR_TABLES / "fast_repair_families.v1.json"
REPAIR_PCLASS = REPAIR_TABLES / "repair_pclass.v1.zlib"
ENGINE_PATCH = Path("packaging") / "fast_cpu" / "gigatoken-toktier-pinned-1.patch"
CONTENT_DOMAIN = b"toktier.engine_core.content.v1\0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core_content_sha256(root: Path = REPOSITORY_ROOT) -> str:
    """Domain-separated digest of every file of the engine crate."""
    crate = root / CORE_CRATE
    if not crate.is_dir():
        raise FileNotFoundError(f"the engine crate is missing ({crate})")
    lines = sorted(
        f"{_sha256(path)}  {path.relative_to(crate).as_posix()}\n"
        for path in crate.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256(CONTENT_DOMAIN)
    for line in lines:
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def engine_core_digests(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    """The four engine-core digests of the tree at ``root``."""
    return {
        "core_content_sha256": core_content_sha256(root),
        "repair_families_sha256": _sha256(root / REPAIR_FAMILIES),
        "repair_pclass_sha256": _sha256(root / REPAIR_PCLASS),
        "engine_patch_sha256": _sha256(root / ENGINE_PATCH),
    }


def main() -> int:
    print(json.dumps(engine_core_digests(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
