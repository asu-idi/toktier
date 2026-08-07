"""Data shipped with the routing package: the support registry.

``support_registry.v1.json`` is a byte-identical installed copy of the
machine-generated certification registry (``tables/support_registry.json``
in the repository); ``tools/generate_registry.py --check`` verifies the
two copies stay identical, so hand edits and drift are rejected the same
way they are for the repository copy.

The file lives inside the package, not only next to the repository's
generated tables, because an installed wheel has to be able to report
certification statuses -- the per-delivery, per-architecture status maps
``explain()`` and the explicit GPU engine's reports carry -- without a
source checkout.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["SUPPORT_REGISTRY", "TABLE_DIR"]

#: Directory of the shipped data files.
TABLE_DIR = Path(__file__).resolve().parent

#: The support registry shipped with the package.
SUPPORT_REGISTRY = TABLE_DIR / "support_registry.v1.json"
