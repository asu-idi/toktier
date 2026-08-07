"""Loading generated lookup tables, with digest verification.

Contract reference: ``docs/contracts/registry.md`` Section 3.1 --
generated lookup tables are first-class artifacts. Their digest
(``class_table_digest``) is part of the kernel certificate's binding set,
and a table that does not match the bound digest closes the accelerated
path exactly as a kernel source mismatch would.

Why this module only *loads*
----------------------------
Kernel split behaviour depends on character-class tables derived from the
reference tokenizer package, and therefore from its Unicode version. The
prototype this was ported from built the tables lazily on first use by probing the
installed reference engine codepoint by codepoint. That is convenient and
unsafe as a release shape: a reference-package upgrade would silently
change kernel split behaviour while the certificate stayed green.

So the probe lives in ``tools/generate_class_tables.py`` and the result is
an artifact with a sha256. This module resolves a table, verifies its
bytes against the digest the registry bound, and refuses to invent one at
load time. When a table is missing, the error names the command that
generates it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...errors import KernelIncompatible
from .families import ClassTableSpec, KernelFamilyTable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = [
    "PACKAGED_TABLE_DIR",
    "ClassTableStore",
    "LoadedClassTable",
    "class_table_digest",
    "file_sha256",
]

#: Tables shipped inside the wheel, next to the routing data.
PACKAGED_TABLE_DIR = Path(__file__).resolve().parents[2] / "kernels" / "tables"

_BINDING_DOMAIN = b"toktier.class_tables.v1\x00"


def file_sha256(path: Path) -> str:
    """``sha256:<hex>`` over a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def class_table_digest(specs: list[ClassTableSpec]) -> str:
    """One digest over a set of tables, for the certificate binding set.

    Order-independent by construction (the entries are sorted by id), so
    the same set of tables always yields the same value. Metadata
    sidecar digests are part of the preimage: sidecar fields change
    tokenization behavior, so a sidecar edit must change the bound
    identity exactly as a table edit would.
    """
    digest = hashlib.sha256()
    digest.update(_BINDING_DOMAIN)
    for spec in sorted(specs, key=lambda item: item.table_id):
        digest.update(spec.table_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update((spec.sha256 or "").encode("utf-8"))
        digest.update(b"\x00")
        digest.update((spec.meta_sha256 or "").encode("utf-8"))
        digest.update(b"\x00")
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class LoadedClassTable:
    """A verified table plus where it came from."""

    spec: ClassTableSpec
    path: Path
    array: np.ndarray[Any, Any]
    observed_sha256: str
    #: Sidecar metadata, when the generator emitted one (the
    #: three-splitter table carries the constants the kernel
    #: cross-checks against).
    meta: dict[str, Any] | None = None


class ClassTableStore:
    """Resolves, verifies and caches the generated lookup tables.

    Search order for a table file:

    1. an explicit directory handed to the constructor;
    2. the tables shipped inside the package;
    3. the generated-table directory under the resolved cache directory.

    Whichever file is found is verified against the digest the routing
    data carries. When the routing data has no digest yet (before the
    first generator run), the observed digest is recorded and reported
    but not enforced, and callers can see that in the binding set.
    """

    def __init__(
        self,
        family_table: KernelFamilyTable,
        *,
        table_dir: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._families = family_table
        self._explicit_dir = Path(table_dir) if table_dir is not None else None
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._loaded: dict[str, LoadedClassTable] = {}

    # -- resolution ----------------------------------------------------

    def search_dirs(self) -> tuple[Path, ...]:
        """Directories searched for table files, in order."""
        dirs: list[Path] = []
        if self._explicit_dir is not None:
            dirs.append(self._explicit_dir)
        dirs.append(PACKAGED_TABLE_DIR)
        if self._cache_dir is not None:
            dirs.append(self._cache_dir / "class_tables")
        return tuple(dirs)

    def locate(self, table_id: str) -> Path | None:
        """First existing file for this table id, or ``None``."""
        spec = self._families.class_table(table_id)
        for directory in self.search_dirs():
            candidate = directory / spec.file
            if candidate.is_file():
                return candidate
        return None

    # -- loading -------------------------------------------------------

    def load(self, table_id: str) -> LoadedClassTable:
        """Load and verify one table, caching the result."""
        cached = self._loaded.get(table_id)
        if cached is not None:
            return cached

        import numpy

        spec = self._families.class_table(table_id)
        path = self.locate(table_id)
        if path is None:
            raise KernelIncompatible(
                f"generated class table {table_id!r} is not present",
                details={
                    "backend": "gpu",
                    "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                    "class_table_digest": spec.sha256,
                    "searched": [str(d / spec.file) for d in self.search_dirs()],
                    "remedy": (
                        "python tools/generate_class_tables.py "
                        f"--table {table_id}"
                    ),
                },
            )
        observed = file_sha256(path)
        if spec.sha256 is not None and observed != spec.sha256:
            raise KernelIncompatible(
                f"class table {table_id!r} does not match the bound digest",
                details={
                    "backend": "gpu",
                    "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                    "expected_digest": spec.sha256,
                    "observed_digest": observed,
                    "class_table_digest": spec.sha256,
                    "path": str(path),
                },
            )
        array = numpy.load(path)
        if spec.shape and tuple(array.shape) != spec.shape:
            raise KernelIncompatible(
                f"class table {table_id!r} has shape {tuple(array.shape)}, "
                f"expected {spec.shape}",
                details={
                    "backend": "gpu",
                    "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                    "path": str(path),
                },
            )
        if spec.dtype and array.dtype.name != spec.dtype:
            raise KernelIncompatible(
                f"class table {table_id!r} has dtype {array.dtype.name}, "
                f"expected {spec.dtype}",
                details={
                    "backend": "gpu",
                    "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                    "path": str(path),
                },
            )
        meta = self._load_meta(spec, path)
        loaded = LoadedClassTable(
            spec=spec,
            path=path,
            array=array,
            observed_sha256=observed,
            meta=meta,
        )
        self._loaded[table_id] = loaded
        return loaded

    def _load_meta(
        self, spec: ClassTableSpec, path: Path
    ) -> dict[str, Any] | None:
        """Load and verify the metadata sidecar next to a table.

        When the routing data binds a sidecar digest, a missing or
        drifted sidecar closes the accelerated path: its fields change
        tokenization behavior, so it is held to the table's standard.
        Without a bound digest the sidecar is read as found, exactly
        like a table whose own digest is not yet recorded.
        """
        meta_path = path.with_suffix(".meta.json")
        if spec.meta_sha256 is not None and not meta_path.is_file():
            raise KernelIncompatible(
                f"class table {spec.table_id!r} binds a metadata sidecar "
                "that is not present",
                details={
                    "backend": "gpu",
                    "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                    "expected_digest": spec.meta_sha256,
                    "path": str(meta_path),
                },
            )
        if not meta_path.is_file():
            return None
        if spec.meta_sha256 is not None:
            observed = file_sha256(meta_path)
            if observed != spec.meta_sha256:
                raise KernelIncompatible(
                    f"metadata sidecar of {spec.table_id!r} does not match "
                    "the bound digest",
                    details={
                        "backend": "gpu",
                        "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                        "expected_digest": spec.meta_sha256,
                        "observed_digest": observed,
                        "path": str(meta_path),
                    },
                )
        loaded: dict[str, Any] = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
        return loaded

    def load_role(self, role: str) -> LoadedClassTable:
        """Load the table declared for a named role."""
        return self.load(self._families.table_for_role(role))

    def for_family(self, family: str) -> LoadedClassTable:
        """Load the table the named family needs."""
        return self.load(self._families.get(family).class_table)

    # -- certificate support -------------------------------------------

    def binding_digest(self) -> str:
        """``class_table_digest`` over every table in the routing data."""
        return class_table_digest(list(self._families.class_tables()))

    def observed_digests(self) -> dict[str, str | None]:
        """Observed digest of every table that could be located."""
        result: dict[str, str | None] = {}
        for spec in self._families.class_tables():
            path = self.locate(spec.table_id)
            result[spec.table_id] = file_sha256(path) if path is not None else None
        return result
