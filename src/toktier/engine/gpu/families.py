"""Family to kernel-band resolution, read from registry data.

Contract reference: ``docs/contracts/registry.md`` Section 3.3 -- the
registry is the only data source for family-to-kernel mappings, and
runtime code must not carry a second copy of any mapping the registry
expresses.

The prototype this was ported from kept two independently maintained
tables: a five-branch dispatch function next to the encoders, and a set
of band constants in the judgement harness. Adding a family meant
editing both, and one release cycle did exactly that in one place only.
Here there is a single data file, and the dispatch is a lookup.

The data file is the normative routing data, maintained in this
repository and consumed both by this module and by the table generator
(``tools/generate_class_tables.py``). This module only reads it, and its
content digest travels in the certificate binding set (see
``GpuEngine.binding_set``), so a drifted copy cannot select a kernel the
certificate never covered. Nothing in this package hardcodes a family
name, a band, a ruleset id, a digits-max value or a class-table id.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import RegistryInvalid, UncertifiedTokenizer

__all__ = [
    "DEFAULT_FAMILY_TABLE_PATH",
    "ClassTableSpec",
    "KernelBand",
    "KernelFamily",
    "KernelFamilyTable",
]

#: Location of the routing data shipped inside the package.
DEFAULT_FAMILY_TABLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "kernels"
    / "tables"
    / "kernel_families.v1.json"
)

_SCHEMA = "toktier.kernel_families.v1"


@dataclass(frozen=True)
class ClassTableSpec:
    """One generated lookup table, as the registry describes it.

    ``sha256`` is part of the kernel certificate binding set: a table
    whose bytes do not match closes the accelerated path with
    ``R_KERNEL_DIGEST_MISMATCH``. It is ``None`` only before the
    generator has been run and the registry regenerated. ``meta_sha256``
    binds the metadata sidecar the same way where one exists: sidecar
    fields (for example ``digits_max``) change tokenization behavior,
    so they are part of the same identity as the table bytes.
    """

    table_id: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str | None
    generator: str
    note: str = ""
    meta_file: str | None = None
    meta_sha256: str | None = None


@dataclass(frozen=True)
class KernelBand:
    """One kernel band: the implementation entry points that serve it.

    ``pretok`` and ``encoder`` name entry points from the engine's
    dispatch registry (``toktier.engine.gpu.entry_points``). The code
    declares which implementations exist; *this data* declares which
    band each one serves, so the runtime carries no band-to-class
    mapping of its own (registry contract Section 3.3).
    """

    name: str
    #: Whether this band has an end-to-end encoder. When false, encode
    #: requests fall back to the reference backend.
    e2e: bool
    #: Entry point of the piece-start (split) layer. Every band has one.
    pretok: str
    #: Entry point of the end-to-end encoder pair; ``None`` exactly when
    #: the band is split-only.
    encoder: str | None = None
    #: Whether the batched channel needs windowed piece starts (the
    #: sparse-window splitter group).
    windowed_starts: bool = False
    note: str = ""


@dataclass(frozen=True)
class KernelFamily:
    """Everything the GPU backend needs to route one family."""

    name: str
    #: Kernel band: which set of entry points serves this family.
    band: str
    #: Ruleset selector handed to the pre-tokenization kernel.
    ruleset: str
    #: Maximum digits per piece; ``None`` means the value is carried by
    #: the class-table metadata (mechanically extracted from the
    #: artifact's own pattern) rather than duplicated here.
    digits_max: int | None
    #: Identifier of the generated class table this family needs.
    class_table: str
    #: Whether the splitter has the contraction alternative.
    contractions: bool
    #: Whether this band has an end-to-end encoder. When false, encode
    #: requests fall back to the reference backend.
    e2e: bool
    #: Another family whose model section (vocabulary and merges) is
    #: identical, so one exported table serves both. The claim is
    #: verified by hashing both model sections before every use.
    shares_model_with: str | None = None
    note: str = ""


class KernelFamilyTable:
    """Read-only view over the family routing data."""

    def __init__(
        self,
        document: Mapping[str, Any],
        *,
        source: Path | None = None,
        content_sha256: str | None = None,
    ) -> None:
        self._source = source
        self._content_sha256 = content_sha256
        schema = document.get("schema")
        if schema != _SCHEMA:
            raise RegistryInvalid(
                f"unexpected family table schema {schema!r}",
                details={"path": str(source), "failure": "schema"},
            )
        bands = document.get("bands")
        families = document.get("families")
        tables = document.get("class_tables")
        if not isinstance(bands, Mapping) or not isinstance(families, Mapping):
            raise RegistryInvalid(
                "family table must carry 'bands' and 'families' objects",
                details={"path": str(source), "failure": "structure"},
            )
        if not isinstance(tables, Mapping):
            raise RegistryInvalid(
                "family table must carry a 'class_tables' object",
                details={"path": str(source), "failure": "structure"},
            )
        self._bands: dict[str, KernelBand] = {
            str(k): self._parse_band(str(k), v, source)
            for k, v in bands.items()
            if isinstance(v, Mapping)
        }
        self._class_tables = {
            str(table_id): ClassTableSpec(
                table_id=str(table_id),
                file=str(spec["file"]),
                dtype=str(spec.get("dtype", "uint8")),
                shape=tuple(int(x) for x in spec.get("shape", ())),
                sha256=(str(spec["sha256"]) if spec.get("sha256") else None),
                generator=str(spec.get("generator", "")),
                note=str(spec.get("note", "")),
                meta_file=(
                    str(spec["meta_file"]) if spec.get("meta_file") else None
                ),
                meta_sha256=(
                    str(spec["meta_sha256"]) if spec.get("meta_sha256") else None
                ),
            )
            for table_id, spec in tables.items()
            if isinstance(spec, Mapping)
        }
        self._table_roles: dict[str, str] = {
            str(role): str(table_id)
            for role, table_id in (document.get("table_roles") or {}).items()
        }
        for role, table_id in self._table_roles.items():
            if table_id not in self._class_tables:
                raise RegistryInvalid(
                    f"table role {role!r} names unknown table {table_id!r}",
                    details={
                        "path": str(source),
                        "failure": "unknown_class_table",
                    },
                )
        self._families: dict[str, KernelFamily] = {}
        for name, entry in families.items():
            if not isinstance(entry, Mapping):
                continue
            band = str(entry["band"])
            band_spec = self._bands.get(band)
            if band_spec is None:
                raise RegistryInvalid(
                    f"family {name!r} names unknown band {band!r}",
                    details={"path": str(source), "failure": "unknown_band"},
                )
            class_table = str(entry["class_table"])
            if class_table not in self._class_tables:
                raise RegistryInvalid(
                    f"family {name!r} names unknown class table {class_table!r}",
                    details={"path": str(source), "failure": "unknown_class_table"},
                )
            digits_max = entry.get("digits_max")
            self._families[str(name)] = KernelFamily(
                name=str(name),
                band=band,
                ruleset=str(entry["ruleset"]),
                digits_max=None if digits_max is None else int(digits_max),
                class_table=class_table,
                contractions=bool(entry.get("contractions", False)),
                e2e=band_spec.e2e,
                shares_model_with=(
                    str(entry["shares_model_with"])
                    if entry.get("shares_model_with")
                    else None
                ),
                note=str(entry.get("note", "")),
            )

    @staticmethod
    def _parse_band(
        name: str, spec: Mapping[str, Any], source: Path | None
    ) -> KernelBand:
        """Build one typed band entry, refusing incoherent shapes."""
        e2e = bool(spec.get("e2e", False))
        pretok = spec.get("pretok")
        encoder = spec.get("encoder")
        if not isinstance(pretok, str) or not pretok:
            raise RegistryInvalid(
                f"band {name!r} declares no pretok entry point",
                details={"path": str(source), "failure": "band_pretok"},
            )
        if e2e and (not isinstance(encoder, str) or not encoder):
            raise RegistryInvalid(
                f"band {name!r} is end-to-end but declares no encoder "
                "entry point",
                details={"path": str(source), "failure": "band_encoder"},
            )
        if not e2e and encoder is not None:
            raise RegistryInvalid(
                f"band {name!r} declares an encoder entry point but is "
                "not end-to-end",
                details={"path": str(source), "failure": "band_encoder"},
            )
        return KernelBand(
            name=name,
            e2e=e2e,
            pretok=pretok,
            encoder=encoder if e2e else None,
            windowed_starts=bool(spec.get("windowed_starts", False)),
            note=str(spec.get("note", "")),
        )

    # -- construction --------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> KernelFamilyTable:
        """Load the routing data, defaulting to the packaged copy.

        The exact bytes on disk are hashed before parsing, so the
        content digest reported in the binding set is the digest of what
        was actually read, not of a re-serialization.
        """
        resolved = Path(path) if path is not None else DEFAULT_FAMILY_TABLE_PATH
        try:
            raw = resolved.read_bytes()
        except FileNotFoundError as exc:
            raise RegistryInvalid(
                "kernel family routing data is missing",
                details={"path": str(resolved), "failure": "missing"},
            ) from exc
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryInvalid(
                f"kernel family routing data is not valid JSON: {exc}",
                details={"path": str(resolved), "failure": "json"},
            ) from exc
        return cls(
            document,
            source=resolved,
            content_sha256=hashlib.sha256(raw).hexdigest(),
        )

    # -- lookups -------------------------------------------------------

    @property
    def source(self) -> Path | None:
        """Where this table was read from, when it came from a file."""
        return self._source

    @property
    def content_sha256(self) -> str | None:
        """Bare-hex digest of the exact routing-data bytes.

        ``None`` for a table built from an in-memory document; the
        binding set then reports no family-table digest, which every
        verifier treats as a failed verification, never a pass.
        """
        return self._content_sha256

    def names(self) -> tuple[str, ...]:
        """Every family the routing data knows, sorted."""
        return tuple(sorted(self._families))

    def bands(self) -> tuple[str, ...]:
        """Every band the routing data knows, sorted."""
        return tuple(sorted(self._bands))

    def band_spec(self, band: str) -> KernelBand:
        """The typed dispatch entry for one band."""
        spec = self._bands.get(band)
        if spec is None:
            raise RegistryInvalid(
                f"unknown kernel band {band!r}",
                details={"path": str(self._source), "failure": "unknown_band"},
            )
        return spec

    def find(self, family: str) -> KernelFamily | None:
        """Look a family up, returning ``None`` when it is not listed.

        The planner uses this: an unlisted family is not an error under
        the default policy, it is a reference configuration reported with
        ``R_UNCERTIFIED_ARTIFACT``.
        """
        return self._families.get(family)

    def get(self, family: str) -> KernelFamily:
        """Look a family up, raising when it is not listed."""
        found = self.find(family)
        if found is None:
            raise UncertifiedTokenizer(
                f"no GPU kernel entry for family {family!r}",
                details={"family": family, "artifact_sha256": None},
            )
        return found

    def class_table(self, table_id: str) -> ClassTableSpec:
        """Look up a generated class table by id."""
        spec = self._class_tables.get(table_id)
        if spec is None:
            raise RegistryInvalid(
                f"unknown class table {table_id!r}",
                details={"path": str(self._source), "failure": "unknown_class_table"},
            )
        return spec

    def table_for_role(self, role: str) -> str:
        """Table id for a named role, for example the NFC quick check.

        Roles are declared in the routing data so that no module has to
        spell a table id: which table plays which part is data, like the
        family mappings are.
        """
        table_id = self._table_roles.get(role)
        if table_id is None:
            raise RegistryInvalid(
                f"no table is declared for role {role!r}",
                details={"path": str(self._source), "failure": "unknown_role"},
            )
        return table_id

    def class_tables(self) -> tuple[ClassTableSpec, ...]:
        """Every generated class table the routing data describes."""
        return tuple(self._class_tables[key] for key in sorted(self._class_tables))

    def shared_model_map(self) -> dict[str, str]:
        """Families that share another family's exported tables.

        The table exporter consumes this instead of carrying its own
        copy: which families share a model section is routing data, and
        routing data has one source.
        """
        return {
            name: entry.shares_model_with
            for name, entry in self._families.items()
            if entry.shares_model_with is not None
        }

    def supports_e2e(self, family: str) -> bool:
        """Whether this family has an end-to-end GPU encoder."""
        found = self.find(family)
        return bool(found and found.e2e)
