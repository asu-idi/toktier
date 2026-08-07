"""Data shipped with the kernels: routing data and generated tables.

``kernel_families.v1.json`` is the family-to-band routing data, generated
by the registry tooling. The generated lookup tables (character classes,
NFC quick check) are written here or into the cache directory by
``tools/generate_class_tables.py``; each carries a sha256 that the kernel
certificate binds.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["FAMILY_TABLE", "TABLE_DIR"]

#: Directory of the shipped data files.
TABLE_DIR = Path(__file__).resolve().parent

#: The family routing data shipped with the package.
FAMILY_TABLE = TABLE_DIR / "kernel_families.v1.json"
