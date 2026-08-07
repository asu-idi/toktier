"""Generate the GPU backend's lookup tables as explicit artifacts.

The split kernels classify text through lookup tables covering the whole
codepoint space. Those tables are derived from the reference tokenizer
package, and therefore from the Unicode version that package carries.
The prototype this was ported from built them lazily on first use,
inside the kernel loading path, by probing the installed reference engine
codepoint by codepoint, so a reference-package upgrade could change
kernel split behaviour silently.

The release rule is the opposite: each table is a first-class artifact
with a sha256, the digest is part of the kernel certificate's binding
set, and generation is an explicit, separate command. This tool is that
command; ``toktier.engine.gpu.class_tables`` only loads and verifies.

Usage::

    python tools/generate_class_tables.py --out-dir DIR [--table ID ...]
        [--check] [--manifest PATH] [--extra-manifest PATH ...]

Table ids, file names, dtypes and shapes come from the routing data
(``src/toktier/kernels/tables/kernel_families.v1.json``); nothing here
carries a second copy of them. With no ``--table``, every table listed
there is built. The default output directory is the generated-table
directory under the resolved cache directory, which is the third entry
in the loader's search order.

Artifact inputs: most tables are probed from the reference engine
alone, but the ``deepseek_v1`` table reads the split patterns of its
band's artifacts (``deepseek_v3``, ``deepseek_v4_flash``, ``hy3``) and
the ``kimi_v1`` table checks its family's frozen splitter fingerprint
(``kimi_k3``). Families named by an explicit ``--manifest`` /
``--extra-manifest`` are taken from there; any family those do not
define is looked up in the local toktier artifact cache (populated by
``toktier artifacts fetch <family>``, honoring ``TOKTIER_HOME``).
``kimi_k3`` has no artifact identity in the packaged manifest in this
release, so it always needs an explicit manifest entry; a run without
one can still check every other table via ``--table``.

Streams: the JSON summary is the only thing written to stdout, so the
registry generator can consume it directly. Progress lines and one
``sha256:<hex>`` line per file go to stderr.

Exit codes: 0 for success, 1 when ``--check`` finds a difference, 2 when
a structural check refuses the build (in which case nothing was
written).

Two properties must survive any future edit:

* Category masks are probed from the reference engine's own regex
  splitter (``behavior="removed"``, whole codepoint space in chunks,
  surrogates skipped). They are deliberately *not* taken from
  ``unicodedata`` and not from the PyPI ``regex`` module. Both were
  tried and both were wrong, in opposite directions: the standard
  library table was older than the reference engine, so codepoints the
  engine already classified as letters were treated as other and the
  kernel over-split; the ``regex`` module was newer than the engine, so
  codepoints the engine still treats as unassigned were classified as
  letters and the skew ran the other way. Only the engine whose output
  the kernel must reproduce can define these classes. ``unicodedata``
  appears in this file solely as a cross-check source inside the NFC
  quick-check builder, exactly as the original did, and never as the
  source of a category mask.
* Nothing is written until every structural check has passed. Table
  bytes are produced once and then either written or compared, so the
  digest reported is the digest of the bytes on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from typing import TypeAlias

    import numpy as np

    NDArray: TypeAlias = np.ndarray[Any, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Routing data: the single source of truth for table ids and files.
FAMILY_TABLE_PATH = (
    REPO_ROOT / "src" / "toktier" / "kernels" / "tables" / "kernel_families.v1.json"
)

#: Generator ids in the routing data are ``class_tables:<key>``.
GENERATOR_PREFIX = "class_tables:"

CODEPOINT_COUNT = 0x110000
SURROGATE_START = 0xD800
SURROGATE_STOP = 0xE000

#: Codepoints per probe string. Large enough that the per-call overhead
#: of the reference splitter does not dominate, small enough to keep the
#: intermediate lists modest.
PROBE_CHUNK = 0x8000

#: Properties probed from the reference engine. The five coarse ones
#: (L, M, N, P, S) partition everything the splitters name; the five
#: fine letter subcategories are needed by the case-aware rulesets.
MASK_PROPS: tuple[str, ...] = ("Lu", "Lt", "Ll", "Lm", "Lo", "L", "M", "N", "P", "S")

#: The reference engine's ``\s`` is ``\p{White_Space}``: these 25
#: codepoints. U+001C..U+001F are ``str.isspace()`` in Python but are not
#: Unicode White_Space, and the engine does not treat them as whitespace,
#: so they are deliberately absent.
WHITE_SPACE: tuple[int, ...] = (
    0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x0020, 0x0085, 0x00A0,
    0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
    0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F,
    0x3000,
)

# Four-class enumeration of the GPT-style splitter group.
CL_P, CL_L, CL_N, CL_S = 0, 1, 2, 3

# Seven-class enumeration shared by the o200k table and the kimi table:
# punctuation-or-other, uppercase/titlecase, lowercase, other letter,
# number, whitespace, mark.
CLS_P, CLS_U, CLS_L, CLS_C, CLS_N, CLS_S, CLS_M = range(7)

# The kimi table adds three Han subclasses on top of the shared seven.
CLS_HL, CLS_HN, CLS_HP = 7, 8, 9

# Seven-class enumeration of the three-splitter ruleset: homeless
# (controls, format characters, unassigned, private use), letter, mark,
# number, punctuation-or-symbol, non-CRLF whitespace, CR/LF.
DS_O, DS_L, DS_M, DS_N, DS_PS, DS_WS, DS_CRLF = range(7)
DS_CLASS_NAMES: dict[int, str] = {
    DS_O: "O",
    DS_L: "L",
    DS_M: "M",
    DS_N: "N",
    DS_PS: "PS",
    DS_WS: "WS",
    DS_CRLF: "CRLF",
}

#: Frozen fingerprint of the kimi splitter pattern the ten-class table
#: was solved for. The prototype builder pinned this against a stored
#: certificate draft; here it is checked against the artifact's own
#: pattern, which is the same value with one less indirection.
KIMI_PATTERN_SHA256 = (
    "de5781783b193d5ccf5b1b28edfa70fa816ce78d54603fdc422cfd8d4ea4411f"
)

#: Han subclass sizes observed when the certification numbers were
#: measured. A reference-package Unicode change moves these, and moving
#: them silently is exactly what this pin prevents.
KIMI_EXPECTED_COUNTS: dict[str, int] = {
    "HL": 98_685,
    "HN": 13,
    "HP": 332,
    "Han": 99_030,
}

# NFC quick-check constants.
NFC_UNSAFE = 255
HANGUL_START, HANGUL_END = 0xAC00, 0xD7A3


class TableBuildError(RuntimeError):
    """A structural check failed, so nothing may be written."""


def _require(condition: bool, message: str) -> None:
    """Structural check that fails the build before anything is written.

    The prototype used bare ``assert`` for these. They are the whole
    safety argument of the generator, so they are raised explicitly here
    and survive ``python -O``.
    """
    if not condition:
        raise TableBuildError(message)


def log(message: str) -> None:
    """Progress and digests go to stderr; stdout carries only JSON."""
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Routing data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSpec:
    """One generated table as the routing data describes it."""

    table_id: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    generator: str
    #: Digest recorded in the routing data, ``None`` before the first
    #: backfill. ``--check`` compares it against the regenerated bytes.
    sha256: str | None = None
    meta_sha256: str | None = None

    @property
    def meta_file(self) -> str:
        """Sidecar name the loader looks for next to the table."""
        return Path(self.file).with_suffix(".meta.json").name

    @property
    def builder_key(self) -> str:
        """Builder selected by the routing data's ``generator`` field."""
        _require(
            self.generator.startswith(GENERATOR_PREFIX),
            f"table {self.table_id!r} has generator {self.generator!r}, "
            f"which does not start with {GENERATOR_PREFIX!r}",
        )
        return self.generator[len(GENERATOR_PREFIX) :]


def load_routing_data(
    path: Path,
) -> tuple[dict[str, TableSpec], dict[str, tuple[str, ...]]]:
    """Read table specs and the family lists that reference them."""
    document = json.loads(path.read_text(encoding="utf-8"))
    specs: dict[str, TableSpec] = {}
    for table_id, entry in document["class_tables"].items():
        specs[table_id] = TableSpec(
            table_id=table_id,
            file=str(entry["file"]),
            dtype=str(entry["dtype"]),
            shape=tuple(int(value) for value in entry["shape"]),
            generator=str(entry["generator"]),
            sha256=(str(entry["sha256"]) if entry.get("sha256") else None),
            meta_sha256=(
                str(entry["meta_sha256"]) if entry.get("meta_sha256") else None
            ),
        )
    families: dict[str, list[str]] = {}
    for name, entry in document["families"].items():
        families.setdefault(str(entry["class_table"]), []).append(name)
    return specs, {key: tuple(value) for key, value in families.items()}


# ---------------------------------------------------------------------------
# Tokenizer manifests
# ---------------------------------------------------------------------------


def resolve_local_dir(manifest_path: Path, local_dir: Path) -> Path:
    """Turn a manifest's ``local_dir`` into a directory on disk.

    An absolute path is used as it stands. A relative one is resolved
    against the manifest's own directory, so a manifest can travel with
    the artifact tree it describes; if that does not exist and the
    manifest's parent directory does resolve it, that is used instead,
    which covers manifests stored one level inside the tree they
    describe. Both candidates are named in the error when neither
    resolves, so a wrong assumption is visible rather than silent.
    """
    if local_dir.is_absolute():
        return local_dir
    base = manifest_path.resolve().parent
    candidates = [base / local_dir, base.parent / local_dir]
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file():
            return candidate
    return candidates[0]


def read_manifest(path: Path) -> dict[str, Path]:
    """Family to artifact directory, from one manifest file."""
    document = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(document, dict),
        f"manifest {path} must map family names to objects",
    )
    entries: dict[str, Path] = {}
    for family, entry in document.items():
        _require(
            isinstance(entry, dict) and "local_dir" in entry,
            f"manifest {path} entry {family!r} has no 'local_dir'",
        )
        entries[family] = resolve_local_dir(path, Path(str(entry["local_dir"])))
    return entries


def load_manifests(primary: Path | None, extras: Sequence[Path]) -> dict[str, Path]:
    """Merge manifests, additively: an overlay never overrides.

    Extra manifests only fill in families the earlier ones do not
    define, which is how new candidate families are staged without
    touching the entries a published run was measured against.
    """
    merged: dict[str, Path] = {}
    if primary is not None:
        merged.update(read_manifest(primary))
    for extra in extras:
        for family, local_dir in read_manifest(extra).items():
            merged.setdefault(family, local_dir)
    return merged


def packaged_families() -> frozenset[str]:
    """Families with an artifact identity in the packaged manifest.

    These are the families ``toktier artifacts fetch`` can resolve, and
    therefore the ones the default cache lookup below can find without
    an explicit ``--manifest``.
    """
    manifest_path = (
        REPO_ROOT
        / "src"
        / "toktier"
        / "artifacts"
        / "tables"
        / "artifact_manifest.v1.json"
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    return frozenset(str(family) for family in document)


def default_cache_manifest() -> dict[str, Path]:
    """Family directories found in the local toktier artifact cache.

    The default when no ``--manifest`` names a family: artifacts fetched
    with ``toktier artifacts fetch <family>`` are consulted where the
    runtime verified them, honoring ``TOKTIER_HOME``. Explicit manifests
    always win (``load_manifests`` merges them first); this only fills
    families they do not define. Families without a packaged artifact
    identity (no fetch path) still need an explicit manifest entry.

    The lookup imports the package from ``src`` for the manifest and
    cache-layout definitions, so this tool cannot drift from the layout
    the runtime writes. When the import cannot be satisfied (a minimal
    environment without the runtime dependencies), the default is empty
    and the per-family error message states the remedies.
    """
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from toktier.artifacts import ArtifactManifest
        from toktier.artifacts.tables import ARTIFACT_MANIFEST
        from toktier.config import Config
        from toktier.paths import artifact_cache_dir
    except ImportError as error:
        log(f"note: default artifact-cache lookup unavailable ({error})")
        return {}
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    cache_root = artifact_cache_dir(Config.resolve())
    entries: dict[str, Path] = {}
    for family in manifest.families():
        directory = cache_root / manifest.get(family).directory_name
        if (directory / "tokenizer.json").is_file():
            entries[family] = directory
    return entries


# ---------------------------------------------------------------------------
# Reference engine probes
# ---------------------------------------------------------------------------


class ReferenceProbe:
    """Unicode category masks read out of the reference engine itself.

    The method is fixed: run the engine's own regex splitter over the
    whole codepoint space with ``behavior="removed"`` and keep whatever
    it did *not* remove as the complement of the property. Isolated
    surrogates are skipped because the engine's binding rejects them;
    every consumer treats that range by its own explicit convention.
    """

    def __init__(self) -> None:
        self._chunks: list[tuple[list[int], str]] | None = None
        self._masks: dict[str, NDArray] | None = None
        self._han: NDArray | None = None

    def chunks(self) -> list[tuple[list[int], str]]:
        """Codepoint blocks and their string form, built once."""
        if self._chunks is None:
            chunks: list[tuple[list[int], str]] = []
            for base in range(0, CODEPOINT_COUNT, PROBE_CHUNK):
                stop = min(base + PROBE_CHUNK, CODEPOINT_COUNT)
                codes = [
                    code
                    for code in range(base, stop)
                    if not SURROGATE_START <= code < SURROGATE_STOP
                ]
                if codes:
                    chunks.append((codes, "".join(map(chr, codes))))
            self._chunks = chunks
        return self._chunks

    def property_mask(self, prop: str) -> NDArray:
        """Boolean mask of ``\\p{<prop>}`` over the codepoint space."""
        import numpy as np

        # Imported here, not at module scope, so this file stays
        # importable (and ``--help`` stays usable) without the reference
        # tokenizer package installed.
        from tokenizers import Regex, pre_tokenizers

        splitter = pre_tokenizers.Split(
            Regex(r"\p{" + prop + "}"), behavior="removed"
        )
        mask = np.zeros(CODEPOINT_COUNT, dtype=bool)
        for codes, text in self.chunks():
            keep = np.ones(len(codes), dtype=bool)
            for _piece, (start, stop) in splitter.pre_tokenize_str(text):
                keep[start:stop] = False
            # Vectorized form of the original element-wise scatter: the
            # same indices are set, in the same order-independent way.
            mask[np.asarray(codes, dtype=np.int64)[keep]] = True
        return mask

    def masks(self) -> dict[str, NDArray]:
        """All probed category masks, computed once per run."""
        if self._masks is None:
            masks: dict[str, NDArray] = {}
            for prop in MASK_PROPS:
                log(f"probe: category mask \\p{{{prop}}}")
                masks[prop] = self.property_mask(prop)
            self._masks = masks
        return self._masks

    def han_mask(self) -> NDArray:
        """``\\p{Han}`` mask, probed the same way as the categories."""
        if self._han is None:
            log("probe: script mask \\p{Han}")
            self._han = self.property_mask("Han")
        return self._han


def white_space_mask() -> NDArray:
    """Boolean mask of the reference engine's whitespace set."""
    import numpy as np

    mask = np.zeros(CODEPOINT_COUNT, dtype=bool)
    for code in WHITE_SPACE:
        mask[code] = True
    return mask


def all_codepoints() -> list[int]:
    """Every codepoint except the surrogate range."""
    return [
        code
        for code in range(CODEPOINT_COUNT)
        if not SURROGATE_START <= code < SURROGATE_STOP
    ]


# ---------------------------------------------------------------------------
# Build context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltTable:
    """A table and the sidecar metadata it carries, if any."""

    array: NDArray
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class BuildContext:
    """Everything a builder may consult."""

    probe: ReferenceProbe
    manifest: Mapping[str, Path]
    families_by_table: Mapping[str, tuple[str, ...]]
    procs: int
    #: Families ``toktier artifacts fetch`` can resolve; used only to
    #: state the right remedy when a directory is missing.
    fetchable: frozenset[str] = frozenset()

    def local_dir(self, family: str) -> Path:
        """Artifact directory of a family, from the manifest."""
        local_dir = self.manifest.get(family)
        if local_dir is None:
            if family in self.fetchable:
                remedy = (
                    f"run 'toktier artifacts fetch {family}' (the local "
                    "artifact cache is consulted automatically, honoring "
                    "TOKTIER_HOME), or pass --manifest"
                )
            else:
                remedy = (
                    "this family has no artifact identity in the packaged "
                    "manifest, so it cannot be fetched; pass --manifest "
                    "(or --extra-manifest) with a 'local_dir' for it, or "
                    "restrict the run with --table to the tables that do "
                    "not read this artifact"
                )
            raise TableBuildError(
                f"no artifact directory for family {family!r}; {remedy}"
            )
        _require(
            (local_dir / "tokenizer.json").is_file(),
            f"family {family!r} has no tokenizer.json under {local_dir}",
        )
        return local_dir

    def families_for(self, table_id: str) -> tuple[str, ...]:
        """Families the routing data points at this table."""
        families = self.families_by_table.get(table_id, ())
        _require(
            bool(families),
            f"the routing data lists no family using table {table_id!r}",
        )
        return families


Builder = Callable[[BuildContext, TableSpec], BuiltTable]


def split_patterns(local_dir: Path) -> list[str]:
    """Every ``Split`` pattern in an artifact's pre-tokenizer, in order."""
    document = json.loads(
        (local_dir / "tokenizer.json").read_text(encoding="utf-8")
    )
    patterns: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "Split":
                patterns.append(node["pattern"]["Regex"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document["pre_tokenizer"])
    return patterns


# ---------------------------------------------------------------------------
# Four-class tables (GPT-style splitter group)
# ---------------------------------------------------------------------------


def build_cl100k(ctx: BuildContext, spec: TableSpec) -> BuiltTable:
    """Codepoint to {punctuation, letter, number, whitespace}.

    Structurally identical to the sequential reference classifier: the
    default is the punctuation-or-other class, letters and numbers come
    from the probed masks, and whitespace is written last.
    """
    import numpy as np

    masks = ctx.probe.masks()
    table = np.zeros(CODEPOINT_COUNT, dtype=np.uint8)  # default: P
    table[masks["L"]] = CL_L
    table[masks["N"]] = CL_N
    for code in WHITE_SPACE:
        table[code] = CL_S
    return BuiltTable(array=table)


def build_cl100k_marks_as_letters(
    ctx: BuildContext, spec: TableSpec
) -> BuiltTable:
    """The four-class table with Mark codepoints folded into letters.

    One family's splitter differs from the others in exactly two places:
    its letter run is ``[\\p{L}\\p{M}]+`` and its punctuation class
    excludes ``\\p{M}``. Both differences are absorbed by moving marks
    into the letter class here, which is the only change from the base
    table.
    """
    import numpy as np

    masks = ctx.probe.masks()
    table = np.zeros(CODEPOINT_COUNT, dtype=np.uint8)  # default: P
    table[masks["L"]] = CL_L
    table[masks["M"]] = CL_L  # the single difference from the base table
    table[masks["N"]] = CL_N
    for code in WHITE_SPACE:
        table[code] = CL_S
    return BuiltTable(array=table)


# ---------------------------------------------------------------------------
# Seven-class table (o200k splitter group)
# ---------------------------------------------------------------------------


def build_o200k(ctx: BuildContext, spec: TableSpec) -> BuiltTable:
    """Codepoint to the seven-class enumeration of the o200k group.

    Marks get their own class rather than joining the letters: they have
    three roles at once (letter-run member, punctuation-run member and
    prefix character). The vectorized rules treat them by their letter
    role and the rare punctuation-adjacent cases are resolved by a local
    fallback in the kernel, so the class must stay distinguishable.
    """
    import numpy as np

    masks = ctx.probe.masks()
    table = np.zeros(CODEPOINT_COUNT, dtype=np.uint8)  # default: P
    table[masks["Lu"] | masks["Lt"]] = CLS_U
    table[masks["Ll"]] = CLS_L
    table[masks["Lm"] | masks["Lo"]] = CLS_C
    table[masks["M"]] = CLS_M
    table[masks["N"]] = CLS_N
    for code in WHITE_SPACE:
        table[code] = CLS_S
    return BuiltTable(array=table)


# ---------------------------------------------------------------------------
# Ten-class table (kimi splitter)
# ---------------------------------------------------------------------------


def build_kimi(ctx: BuildContext, spec: TableSpec) -> BuiltTable:
    """Seven-class table plus three Han subclasses.

    The splitter's letter branches exclude Han through a character-class
    difference, while its first branch matches Han runs on their own.
    That difference is resolved here, at table build time, by giving Han
    three subclasses so the kernel never needs a set difference:

    * HL = Han and (Lm or Lo), the plain-letter Han that only the Han
      branch can take;
    * HN = Han and N, absorbed by the digit branch;
    * HP = Han and neither L nor N, absorbed by the punctuation branch
      (this is where Han marks land).

    The prototype builder read these three ranges from a stored solution
    and required an independent ``\\p{Han}`` probe to agree with them
    bit for bit. Here the subclasses are derived from that probe
    directly, which is the same result with one less indirection, and
    every cross-check the original performed against the base masks is
    kept.
    """
    import numpy as np

    families = ctx.families_for(spec.table_id)
    for family in families:
        patterns = split_patterns(ctx.local_dir(family))
        _require(
            len(patterns) == 1,
            f"family {family!r} has {len(patterns)} split patterns, "
            "expected exactly one",
        )
        digest = hashlib.sha256(patterns[0].encode()).hexdigest()
        _require(
            digest == KIMI_PATTERN_SHA256,
            f"family {family!r} splitter fingerprint is {digest}, expected "
            f"{KIMI_PATTERN_SHA256}; the table was solved for the frozen "
            "pattern and must not be reused for a different one",
        )

    masks = ctx.probe.masks()
    han = ctx.probe.han_mask()
    han_letters = han & (masks["Lm"] | masks["Lo"])
    han_numbers = han & masks["N"]
    han_punct = han & ~masks["L"] & ~masks["N"]

    _require(
        not (han_letters & han_numbers).any()
        and not (han_letters & han_punct).any()
        and not (han_numbers & han_punct).any(),
        "the Han subclasses overlap",
    )
    _require(
        ((han_letters | han_numbers | han_punct) == han).all(),
        "the Han subclasses do not cover \\p{Han}: some Han codepoint is "
        "cased, which the splitter's letter branches would then take",
    )
    whitespace = white_space_mask()
    _require(
        not (han & whitespace).any(),
        "\\p{Han} intersects the whitespace set, so the punctuation "
        "subclass is no longer equivalent to the pattern's negated class",
    )
    counts = {
        "HL": int(han_letters.sum()),
        "HN": int(han_numbers.sum()),
        "HP": int(han_punct.sum()),
        "Han": int(han.sum()),
    }
    _require(
        counts == KIMI_EXPECTED_COUNTS,
        f"Han subclass sizes are {counts}, expected {KIMI_EXPECTED_COUNTS}; "
        "the reference package's Unicode tables moved under this table",
    )

    # Base classes first (Han members included), Han subclasses last.
    table = np.zeros(CODEPOINT_COUNT, dtype=np.uint8)  # default: P
    table[masks["Lu"] | masks["Lt"]] = CLS_U
    table[masks["Ll"]] = CLS_L
    table[masks["Lm"] | masks["Lo"]] = CLS_C
    table[masks["M"]] = CLS_M
    table[masks["N"]] = CLS_N
    for code in WHITE_SPACE:
        table[code] = CLS_S
    table[han_letters] = CLS_HL
    table[han_numbers] = CLS_HN
    table[han_punct] = CLS_HP

    # Final checks: Han is exactly the subclass range, no cased Han, and
    # the non-Han letter classes still mean what the seven-class table
    # means.
    _require(((table >= CLS_HL) == han).all(), "Han is not the subclass range")
    _require(
        not ((table == CLS_U) & han).any(),
        "a Han codepoint is uppercase or titlecase",
    )
    _require(
        ((table == CLS_C) == ((masks["Lm"] | masks["Lo"]) & ~han)).all(),
        "the other-letter class is not the non-Han part of Lm or Lo",
    )
    _require(
        (
            (table == CLS_M)
            == (masks["M"] & ~han & ~(masks["Lm"] | masks["Lo"]))
        ).all(),
        "the mark class is not the non-Han part of M",
    )

    class_enum = {
        "P": CLS_P,
        "U": CLS_U,
        "L": CLS_L,
        "C": CLS_C,
        "N": CLS_N,
        "S": CLS_S,
        "M": CLS_M,
        "HL": CLS_HL,
        "HN": CLS_HN,
        "HP": CLS_HP,
    }
    meta = {
        "table": spec.file,
        "class_enum": class_enum,
        "pattern_sha256": KIMI_PATTERN_SHA256,
        "families": list(families),
        "counts": counts,
        "class_sizes": {
            name: int((table == value).sum())
            for name, value in class_enum.items()
        },
        "table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "source": (
            "generate_class_tables.py (category and script masks probed "
            "from the reference engine; Han subclasses resolved at build "
            "time)"
        ),
    }
    return BuiltTable(array=table, meta=meta)


# ---------------------------------------------------------------------------
# Seven-class table (three-splitter ruleset)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassItem:
    """One member of a bracketed character class."""

    #: ``"r"`` for a range, ``"c"`` for a single codepoint.
    kind: str
    low: int
    high: int


@dataclass(frozen=True)
class SplitterConstants:
    """Constants extracted mechanically from the artifact's patterns."""

    digits_max: int
    cjk_ranges: list[tuple[int, int]]
    apunct: list[int]
    alpha_ranges: list[tuple[int, int]]
    a3_space: int
    crlf_cps: list[int]


def parse_char_class(source: str, index: int) -> tuple[list[ClassItem], int]:
    """Parse a bracketed character class mechanically.

    Handles literals, ``\\<c>`` escapes and ``a-b`` ranges only. Nested
    classes, negation and ``\\p`` inside the class are refused rather
    than guessed at, because a silent misparse here would produce a
    plausible-looking but wrong constant.
    """
    _require(source[index] == "[", f"expected '[' at {index} in {source!r}")
    index += 1
    _require(source[index] != "^", "negated classes are out of scope")
    codes: list[int] = []
    literal: list[bool] = []  # whether a '-' came from an escape
    while source[index] != "]":
        char = source[index]
        _require(char != "[", "nested '[' is out of scope")
        if char == "\\":
            following = source[index + 1]
            _require(
                following in "\\-[]^`'\""
                or following in "!#$%&()*+,./:;<=>?@_{|}~",
                f"unknown escape \\{following}",
            )
            codes.append(ord(following))
            literal.append(True)
            index += 2
        else:
            codes.append(ord(char))
            literal.append(False)
            index += 1
    index += 1  # consume ']'
    items: list[ClassItem] = []
    position = 0
    while position < len(codes):
        if (
            position + 2 < len(codes)
            and codes[position + 1] == ord("-")
            and not literal[position + 1]
        ):
            low, high = codes[position], codes[position + 2]
            _require(low <= high, f"inverted range {hex(low)}-{hex(high)}")
            items.append(ClassItem("r", low, high))
            position += 3
        else:
            items.append(ClassItem("c", codes[position], codes[position]))
            position += 1
    return items, index


def split_top_alternatives(pattern: str) -> list[str]:
    """Split on top-level ``|`` only (bracket and escape aware)."""
    alternatives: list[str] = []
    buffer: list[str] = []
    index = 0
    in_class = False
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            buffer.append(pattern[index : index + 2])
            index += 2
            continue
        if char == "[" and not in_class:
            in_class = True
        elif char == "]" and in_class:
            in_class = False
        elif char == "|" and not in_class:
            alternatives.append("".join(buffer))
            buffer = []
            index += 1
            continue
        buffer.append(char)
        index += 1
    alternatives.append("".join(buffer))
    return alternatives


def extract_constants(patterns: Sequence[str]) -> SplitterConstants:
    """Three patterns to their constants, with the shapes pinned.

    Every interval, punctuation character and alphabet range the kernel
    carries inline is read out of the artifact's own patterns here. Hand
    copying an interval bound is exactly the kind of error that stays
    invisible until a rare codepoint arrives, so it is never done; the
    assertions below pin the frozen pattern shapes so that a pattern
    change fails the build instead of silently shifting a constant.
    """
    _require(len(patterns) == 3, f"expected three patterns, got {len(patterns)}")
    first, second, third = patterns

    # Splitter 0: a bounded digit run.
    match = re.fullmatch(r"\\p\{N\}\{1,(\d+)\}", first)
    if match is None:
        raise TableBuildError(f"unexpected digit splitter shape: {first!r}")
    digits_max = int(match.group(1))

    # Splitter 1: one bracketed class of ranges, one or more.
    items, index = parse_char_class(second, 0)
    _require(second[index:] == "+", f"unexpected tail {second[index:]!r}")
    _require(
        all(item.kind == "r" for item in items),
        "the ideograph class is expected to be ranges only",
    )
    cjk_ranges = [(item.low, item.high) for item in items]

    # Splitter 2: six alternatives.
    alternatives = split_top_alternatives(third)
    _require(
        len(alternatives) == 6,
        f"expected six alternatives, got {len(alternatives)}",
    )

    # Alternative 1: [<punctuation>][<alphabet>]+
    punct_items, index = parse_char_class(alternatives[0], 0)
    alpha_items, tail = parse_char_class(alternatives[0], index)
    _require(
        alternatives[0][tail:] == "+",
        f"unexpected tail {alternatives[0][tail:]!r}",
    )
    apunct: set[int] = set()
    for item in punct_items:
        if item.kind == "c":
            apunct.add(item.low)
        else:
            apunct.update(range(item.low, item.high + 1))
    _require(
        all(item.kind == "r" for item in alpha_items),
        "the alphabet class is expected to be ranges only",
    )
    alpha_ranges = [(item.low, item.high) for item in alpha_items]

    # Alternatives 2..6 carry no inline constants (their semantics come
    # from the probed masks), but their literal shape is pinned so that a
    # pattern change cannot pass unnoticed.
    _require(
        alternatives[1] == r"[^" + "\r\n" + r"\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+",
        f"unexpected alternative 2: {alternatives[1]!r}",
    )
    _require(
        alternatives[2] == r" ?[\p{P}\p{S}]+[" + "\r\n" + r"]*",
        f"unexpected alternative 3: {alternatives[2]!r}",
    )
    _require(
        alternatives[3] == r"\s*[" + "\r\n" + r"]+",
        f"unexpected alternative 4: {alternatives[3]!r}",
    )
    _require(
        alternatives[4] == r"\s+(?!\S)",
        f"unexpected alternative 5: {alternatives[4]!r}",
    )
    _require(
        alternatives[5] == r"\s+",
        f"unexpected alternative 6: {alternatives[5]!r}",
    )

    # The optional leading space of alternative 3 and the CR/LF tail are
    # extracted too, rather than assumed. In the pattern these are the
    # decoded control characters, not escapes.
    a3_space = ord(alternatives[2][0])
    crlf_items, tail = parse_char_class(
        alternatives[2], alternatives[2].rindex("[")
    )
    _require(
        alternatives[2][tail:] == "*",
        f"unexpected tail {alternatives[2][tail:]!r}",
    )
    _require(
        all(item.kind == "c" for item in crlf_items),
        "the line-break class is expected to be single codepoints",
    )
    crlf_cps = sorted(item.low for item in crlf_items)
    _require(crlf_cps == [10, 13], f"unexpected line-break class {crlf_cps}")

    return SplitterConstants(
        digits_max=digits_max,
        cjk_ranges=cjk_ranges,
        apunct=sorted(apunct),
        alpha_ranges=alpha_ranges,
        a3_space=a3_space,
        crlf_cps=crlf_cps,
    )


def build_deepseek(ctx: BuildContext, spec: TableSpec) -> BuiltTable:
    """Seven-class table plus the sidecar the kernel cross-checks.

    The class assignment is only a partition because the probed masks
    are provably disjoint, so the disjointness is checked over the whole
    codepoint space before anything is assigned, and the histogram is
    required to account for every codepoint afterwards.
    """
    import numpy as np

    families = ctx.families_for(spec.table_id)
    patterns_by_family = {
        family: split_patterns(ctx.local_dir(family)) for family in families
    }
    reference_family = families[0]
    patterns = patterns_by_family[reference_family]
    _require(
        len(patterns) == 3,
        f"family {reference_family!r} has {len(patterns)} split patterns, "
        "expected three",
    )
    for family in families[1:]:
        _require(
            patterns_by_family[family] == patterns,
            f"family {family!r} does not share the group's splitters",
        )
    constants = extract_constants(patterns)

    masks = ctx.probe.masks()

    # Pairwise disjointness of the five coarse masks, and of each of
    # them against the whitespace set.
    coarse = {name: masks[name] for name in ("L", "M", "N", "P", "S")}
    whitespace = white_space_mask()
    crlf = np.zeros(CODEPOINT_COUNT, dtype=bool)
    for code in constants.crlf_cps:
        crlf[code] = True
    _require(bool(crlf[10]) and bool(crlf[13]), "CR and LF are not both set")
    _require(
        bool(whitespace[10]) and bool(whitespace[13]),
        "CR and LF must be inside the frozen whitespace set",
    )
    non_crlf_whitespace = whitespace & ~crlf
    overlaps: dict[str, int] = {}
    names = list(coarse)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            key = f"{names[left]}&{names[right]}"
            overlaps[key] = int((coarse[names[left]] & coarse[names[right]]).sum())
    for name, mask in coarse.items():
        overlaps[f"{name}&WS"] = int((mask & whitespace).sum())
    _require(
        all(value == 0 for value in overlaps.values()),
        f"the class masks are not disjoint: {overlaps}",
    )

    # The digit splitter and the ideograph splitter must not compete for
    # any codepoint, or their relative order would matter.
    ideographs = np.zeros(CODEPOINT_COUNT, dtype=bool)
    for low, high in constants.cjk_ranges:
        ideographs[low : high + 1] = True
    digits_in_ideographs = int((masks["N"] & ideographs).sum())
    _require(
        digits_in_ideographs == 0,
        f"the digit and ideograph splitters overlap on "
        f"{digits_in_ideographs} codepoints",
    )

    # The inline constants must sit inside the classes they claim.
    punct_or_symbol = masks["P"] | masks["S"]
    _require(
        bool(punct_or_symbol[np.array(constants.apunct)].all()),
        "the punctuation constant is not a subset of P or S",
    )
    alphabet = np.zeros(CODEPOINT_COUNT, dtype=bool)
    for low, high in constants.alpha_ranges:
        alphabet[low : high + 1] = True
    _require(
        bool(masks["L"][alphabet].all()),
        "the alphabet constant is not a subset of L",
    )
    _require(
        bool(whitespace[constants.a3_space])
        and not bool(punct_or_symbol[constants.a3_space]),
        "the leading-space constant is not whitespace, or is punctuation",
    )

    # Assignment is a partition, given the disjointness proved above.
    table = np.zeros(CODEPOINT_COUNT, dtype=np.uint8)  # default: O
    table[masks["L"]] = DS_L
    table[masks["M"]] = DS_M
    table[masks["N"]] = DS_N
    table[punct_or_symbol] = DS_PS
    _require(
        bool((table[non_crlf_whitespace] == DS_O).all()),
        "whitespace intersects a category class",
    )
    table[non_crlf_whitespace] = DS_WS
    _require(bool((table[crlf] == DS_O).all()), "CR or LF is already classed")
    table[crlf] = DS_CRLF

    # Isolated surrogates cannot be probed, so they take the convention
    # every splitter applies to them: punctuation-or-symbol.
    surrogates = np.zeros(CODEPOINT_COUNT, dtype=bool)
    surrogates[SURROGATE_START:SURROGATE_STOP] = True
    _require(
        bool((table[surrogates] == DS_O).all()),
        "a surrogate codepoint was classified by a probe",
    )
    table[surrogates] = DS_PS

    histogram = {
        name: int((table == value).sum())
        for value, name in DS_CLASS_NAMES.items()
    }
    _require(
        sum(histogram.values()) == CODEPOINT_COUNT,
        f"the classes do not cover the codepoint space: {histogram}",
    )

    meta = {
        "table": spec.file,
        "enum": {name: value for value, name in DS_CLASS_NAMES.items()},
        "digits_max": constants.digits_max,
        "cjk_ranges": [[low, high] for low, high in constants.cjk_ranges],
        "apunct": constants.apunct,
        "alpha_ranges": [[low, high] for low, high in constants.alpha_ranges],
        "a3_space": constants.a3_space,
        "crlf_cps": constants.crlf_cps,
        "families": list(families),
        "patterns": patterns,
        "pattern_sha256": [
            hashlib.sha256(pattern.encode()).hexdigest() for pattern in patterns
        ],
        "class_histogram": histogram,
        "source": (
            "generate_class_tables.py (constants extracted mechanically "
            "from the artifact's own patterns; category masks probed from "
            "the reference engine)"
        ),
    }
    return BuiltTable(array=table, meta=meta)


# ---------------------------------------------------------------------------
# NFC quick-check table
# ---------------------------------------------------------------------------
#
# Values:
#   0        safe starter (combining class 0 and unable to take part in
#            any NFC transformation)
#   1..K     rank of a safe non-starter's combining class, order
#            isomorphic to the reference engine's own ordering, K <= 254
#   255      unsafe: single-character NFC is not the identity, or the
#            codepoint can be the second element of a composition, or it
#            is a surrogate. These always take the slow path.
#
# Predicate (the kernel implements the same one):
#   pass(s) <=> no 255 and no adjacent pair with v[i-1] > v[i] != 0
# Theorem (conservative quick check): pass(s) implies that the reference
# normalizer leaves s unchanged.
#
# The table is probed from the reference engine's own NFC/NFD
# normalizers. Note that the frozen package's normalizer carries an
# older Unicode version than its regex splitter, which is why the
# normalizer is probed separately instead of being assumed consistent
# with the category masks; ``unicodedata`` is used below only to
# cross-check, never as a data source, and any disagreement resolves
# conservatively towards 255.

_NORMALIZERS: dict[str, Any] = {}

#: Worker state for the second-element scan (populated after fork).
_WORKER: dict[str, Any] = {}
_WORKER_UNSAFE: frozenset[int] = frozenset()


def _reference_normalizers() -> tuple[Any, Any]:
    """The reference engine's NFC and NFD normalizers, built once."""
    if not _NORMALIZERS:
        # Lazy, so this module imports without the reference package.
        from tokenizers import normalizers

        _NORMALIZERS["nfc"] = normalizers.NFC()
        _NORMALIZERS["nfd"] = normalizers.NFD()
    return _NORMALIZERS["nfc"], _NORMALIZERS["nfd"]


def singleton_scan(kind: str) -> dict[int, str]:
    """Codepoints whose single-character NFC/NFD is not the identity.

    Probed in space-separated batches: U+0020 has no canonical
    composition or decomposition role and never appears in canonical
    output, so it is a safe batch separator. A batch whose output does
    not split back into the same number of pieces falls back to one
    probe per character, and U+0020 itself is probed on its own.
    """
    nfc, nfd = _reference_normalizers()
    normalize = (nfc if kind == "NFC" else nfd).normalize_str
    changed: dict[int, str] = {}
    codes = [code for code in all_codepoints() if code != 0x20]
    batch = 20000
    for start in range(0, len(codes), batch):
        block = codes[start : start + batch]
        text = " ".join(map(chr, block))
        pieces = normalize(text).split(" ")
        if len(pieces) == len(block):
            for code, piece in zip(block, pieces, strict=True):
                if piece != chr(code):
                    changed[code] = piece
        else:  # output contained a space: fall back to one probe each
            for code in block:
                piece = normalize(chr(code))
                if piece != chr(code):
                    changed[code] = piece
    piece = normalize(" ")
    if piece != " ":
        changed[0x20] = piece
    return changed


def _worker_init() -> None:
    """Per-worker state for the second-element scan."""
    from tokenizers import normalizers

    _WORKER["nfc"] = normalizers.NFC().normalize_str
    # Codepoints already known to be unsafe are removed from the pool:
    # they are 255 regardless, and keeping them would only add noise to
    # the bisection.
    _WORKER["chr_all"] = [
        chr(code) for code in all_codepoints() if code not in _WORKER_UNSAFE
    ]


def _scan_recursive(first: str, seconds: list[str], hits: list[str]) -> None:
    """Bisect a batch of candidate second elements.

    The probe string interleaves the first element with each candidate.
    If the whole batch normalizes to itself, no candidate in it composes
    with the first element; otherwise the batch is halved, and every
    single hit is confirmed pairwise before it is recorded.
    """
    probe = first + first.join(seconds)
    if _WORKER["nfc"](probe) == probe:
        return
    if len(seconds) == 1:
        pair = first + seconds[0]
        if _WORKER["nfc"](pair) != pair:
            hits.append(seconds[0])
        return
    middle = len(seconds) // 2
    _scan_recursive(first, seconds[:middle], hits)
    _scan_recursive(first, seconds[middle:], hits)


def _scan_first(first_code: int) -> tuple[int, list[int]]:
    """Every second element that composes with one first element."""
    first = chr(first_code)
    hits: list[str] = []
    candidates: list[str] = _WORKER["chr_all"]
    batch = 4096
    for start in range(0, len(candidates), batch):
        _scan_recursive(first, candidates[start : start + batch], hits)
    return first_code, [ord(hit) for hit in hits]


def second_element_scan(
    firsts: Sequence[int], unsafe: frozenset[int], procs: int
) -> set[int]:
    """Scan every first element against the whole codepoint space."""
    import multiprocessing

    global _WORKER_UNSAFE
    _WORKER_UNSAFE = unsafe  # inherited by the forked workers
    seconds: set[int] = set()
    pairs = 0
    started = time.time()
    context = multiprocessing.get_context("fork")
    with context.Pool(procs, initializer=_worker_init) as pool:
        results = pool.imap_unordered(_scan_first, list(firsts), chunksize=8)
        for index, (_first, hits) in enumerate(results):
            seconds.update(hits)
            pairs += len(hits)
            if (index + 1) % 500 == 0:
                log(
                    f"nfc: {index + 1}/{len(firsts)} first elements done, "
                    f"{len(seconds)} distinct second elements so far"
                )
    log(
        f"nfc: second-element scan done, {pairs} composing pairs, "
        f"{len(seconds)} distinct second elements "
        f"({time.time() - started:.1f}s)"
    )
    return seconds


def detect_nonstarters(candidates: Sequence[int]) -> set[int]:
    """Codepoints with a non-zero combining class, by reordering probes.

    Two NFD probes: ``"a" + c + U+0334`` changes exactly when the
    combining class of ``c`` is greater than 1, and ``"a" + U+0345 + c``
    changes exactly when it is between 1 and 239. Their union is every
    non-zero combining class. Candidates are decomposition-free, so the
    output has the same length as the input and each triple can be read
    back by slicing.
    """
    _, nfd = _reference_normalizers()
    normalize = nfd.normalize_str
    nonstarters: set[int] = set()
    batch = 8000
    for probe_kind in ("A", "B"):
        for start in range(0, len(candidates), batch):
            block = candidates[start : start + batch]
            if probe_kind == "A":
                text = "".join("a" + chr(code) + "\u0334" for code in block)
            else:
                text = "".join("a\u0345" + chr(code) for code in block)
            result = normalize(text)
            _require(
                len(result) == len(text),
                "a decomposable codepoint reached the reordering probe",
            )
            for offset, code in enumerate(block):
                start_index = 3 * offset
                stop_index = start_index + 3
                if result[start_index:stop_index] != text[start_index:stop_index]:
                    nonstarters.add(code)
    return nonstarters


def rank_nonstarters(
    nonstarters: Sequence[int],
) -> tuple[dict[int, int], list[list[int]]]:
    """Order non-starters by combining class, via reordering comparisons.

    ``NFD("a" + u + v) != "a" + u + v`` exactly when the combining class
    of ``u`` is greater than that of ``v``, which is a total preorder;
    insertion sort over equivalence classes turns it into ranks 1..K.
    """
    _, nfd = _reference_normalizers()
    normalize = nfd.normalize_str
    cache: dict[tuple[int, int], bool] = {}

    def greater(left: int, right: int) -> bool:
        key = (left, right)
        if key not in cache:
            probe = "a" + chr(left) + chr(right)
            cache[key] = normalize(probe) != probe
        return cache[key]

    classes: list[list[int]] = []  # ascending, represented by their first
    for code in nonstarters:
        low, high = 0, len(classes)
        placed = False
        while low < high:
            middle = (low + high) // 2
            representative = classes[middle][0]
            if greater(code, representative):
                low = middle + 1
            elif greater(representative, code):
                high = middle
            else:
                classes[middle].append(code)
                placed = True
                break
        if not placed:
            classes.insert(low, [code])
    ranks: dict[int, int] = {}
    for index, members in enumerate(classes):
        for code in members:
            ranks[code] = index + 1
    return ranks, classes


def quick_check_passes(text: str, table: NDArray) -> bool:
    """Reference form of the predicate the kernel implements."""
    import numpy as np

    if not text:
        return True
    values = table[np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)]
    if (values == NFC_UNSAFE).any():
        return False
    if values.size < 2:
        return True
    reordered = (values[:-1] > values[1:]) & (values[1:] != 0)
    return not reordered.any()


def _nfc_pin_checks(table: NDArray) -> None:
    """Codepoints whose value is known independently of the probe."""
    must_be_unsafe = {
        "U+0301 combining acute (composition candidate)": 0x0301,
        "U+0300 combining grave (composition candidate)": 0x0300,
        "U+0308 combining diaeresis (composition candidate)": 0x0308,
        "U+2126 ohm sign (singleton decomposition)": 0x2126,
        "U+0344 (decomposes to two marks)": 0x0344,
        "U+0F73 (composition exclusion)": 0x0F73,
        "U+0958 (composition exclusion)": 0x0958,
        "U+1161 Hangul vowel": 0x1161,
        "U+11A8 Hangul trailing consonant": 0x11A8,
        "U+0BBE Tamil AA (second element of a matrix)": 0x0BBE,
        "U+09BE Bengali AA": 0x09BE,
    }
    for name, code in must_be_unsafe.items():
        _require(
            int(table[code]) == NFC_UNSAFE,
            f"pin failed: {name} should be {NFC_UNSAFE}, got {int(table[code])}",
        )
    must_be_safe_starters = {
        "space": 0x20,
        "a": 0x61,
        "e with acute (recomposable)": 0xE9,
        "a CJK ideograph": 0x4E2D,
        "U+1100 Hangul leading consonant": 0x1100,
        "U+AC00 Hangul syllable": 0xAC00,
    }
    for name, code in must_be_safe_starters.items():
        _require(
            int(table[code]) == 0,
            f"pin failed: {name} should be 0, got {int(table[code])}",
        )
    # Known combining classes 1 < 10 < 220 < 222 must keep their order.
    # U+0345 would be the natural 240 probe but it is a composition
    # candidate and therefore 255, so U+059A stands in for it.
    ranks = tuple(int(table[code]) for code in (0x334, 0x5B0, 0x316, 0x59A))
    _require(
        0 < ranks[0] < ranks[1] < ranks[2] < ranks[3] < NFC_UNSAFE,
        f"the combining-class ranks are out of order: {ranks}",
    )


def _nfc_cross_check_pairs(table: NDArray, nfc: Any) -> list[int]:
    """Cross-check against the standard library's composition pairs.

    The standard library's Unicode version is newer than the one the
    reference normalizer carries, so a pair it knows may simply be inert
    for the reference engine. Every such gap is therefore probed
    directly: the reference engine must really leave the pair alone. If
    it does not, the scan was incomplete and the build fails.
    """
    pairs: dict[int, list[int]] = {}
    for code in range(CODEPOINT_COUNT):
        if SURROGATE_START <= code < SURROGATE_STOP:
            continue
        decomposition = unicodedata.decomposition(chr(code))
        if decomposition and not decomposition.startswith("<"):
            parts = decomposition.split()
            if len(parts) == 2:
                first, second = int(parts[0], 16), int(parts[1], 16)
                composed = unicodedata.normalize(
                    "NFC", chr(first) + chr(second)
                )
                if composed == chr(code):
                    pairs.setdefault(second, []).append(first)
    gap = sorted(code for code in pairs if int(table[code]) != NFC_UNSAFE)
    for code in gap:
        for first in pairs[code]:
            probe = chr(first) + chr(code)
            _require(
                nfc.normalize_str(probe) == probe,
                f"incomplete scan: the reference engine composes "
                f"({hex(first)}, {hex(code)}) but the table does not flag it",
            )
    return gap


def _nfc_version_ladder(nfd: Any) -> dict[str, bool]:
    """Pin which Unicode version the reference normalizer carries.

    Each probe asks whether the normalizer knows a combining class that
    a given Unicode version introduced. The pinned outcome is that it
    knows the older one and not the newer one; if that ever changes, the
    table's provenance changed with it.
    """
    ladder: dict[str, bool] = {}
    for name, code in (
        ("knows_U+08E3", 0x08E3),
        ("knows_U+1E944", 0x1E944),
        ("knows_U+11A34", 0x11A34),
        ("knows_U+08D3", 0x08D3),
        ("knows_U+1ABF", 0x1ABF),
        ("knows_U+10D69", 0x10D69),
    ):
        probe = "a\u0345" + chr(code)
        ladder[name] = nfd.normalize_str(probe) != probe
    _require(
        ladder["knows_U+1E944"] and not ladder["knows_U+11A34"],
        f"the reference normalizer's Unicode version moved: {ladder}",
    )
    return ladder


def _nfc_cross_check_ranks(table: NDArray, ranks: Mapping[int, int]) -> None:
    """Rank order must agree with the standard library where both know.

    Only codepoints both sides have data for are compared: a codepoint
    the standard library has not assigned carries no information here.
    """
    by_rank: dict[int, set[int]] = {}
    for code, rank in ranks.items():
        if int(table[code]) == NFC_UNSAFE:
            continue
        combining = unicodedata.combining(chr(code))
        if combining == 0 and unicodedata.category(chr(code)) == "Cn":
            continue  # unassigned for the standard library
        by_rank.setdefault(rank, set()).add(combining)
    violations: list[tuple[int, list[int]]] = []
    seen: list[int] = []
    for rank in sorted(by_rank):
        values = by_rank[rank] - {0}
        if len(by_rank[rank]) > 1 or (seen and values and min(values) <= max(seen)):
            violations.append((rank, sorted(by_rank[rank])))
        seen.extend(values)
    _require(
        not violations,
        f"the rank order disagrees with the standard library: {violations[:5]}",
    )


def _nfc_planted_checks(table: NDArray, nfc: Any) -> None:
    """Strings the predicate must reject, as a live check of its wiring."""
    planted = {
        "e + combining acute (a composing pair)": "e\u0301",
        "U+2126 (singleton decomposition)": "\u2126",
        "a + two safe marks in the wrong order": "a\u0316\u0334",
        "Hangul leading consonant + vowel": "\u1100\u1161",
        "Tamil E + AA (matrix composition)": "\u0BC6\u0BBE",
        "a defective sequence starting with a mark": "\u0301x",
    }
    for name, text in planted.items():
        _require(
            not quick_check_passes(text, table),
            f"the predicate did not reject: {name} "
            f"(reference identity: {nfc.normalize_str(text) == text})",
        )


def _nfc_soundness_battery(table: NDArray, nfc: Any) -> tuple[int, int]:
    """Randomized soundness gate: pass implies reference identity.

    Three kinds of string are drawn: safe starters only, safe starters
    with dense ascending mark runs, and uniform draws over the whole
    codepoint space (mostly rejected). Any string the predicate accepts
    that the reference normalizer would change is a counterexample and
    fails the build.
    """
    import numpy as np

    rng = np.random.default_rng(2026)
    safe_starters = np.flatnonzero(table == 0)
    safe_marks = np.flatnonzero((table > 0) & (table < NFC_UNSAFE))
    marks_by_rank = sorted(safe_marks, key=lambda code: table[code])
    passed = checked = 0
    for trial in range(30000):
        kind = trial % 3
        if kind == 0:  # safe starters only
            codes = rng.choice(safe_starters, size=rng.integers(1, 60))
        elif kind == 1:  # starters with dense ascending mark runs
            values: list[int] = []
            for _ in range(rng.integers(1, 12)):
                values.append(int(rng.choice(safe_starters)))
                count = int(rng.integers(0, 6))
                picks = sorted(rng.choice(len(marks_by_rank), size=count))
                values.extend(int(marks_by_rank[pick]) for pick in picks)
            codes = np.array(values or [0x61])
        else:  # uniform over the codepoint space
            codes = rng.integers(0, CODEPOINT_COUNT, size=rng.integers(1, 60))
            codes = codes[(codes < SURROGATE_START) | (codes >= SURROGATE_STOP)]
            if codes.size == 0:
                continue
        text = "".join(map(chr, codes.tolist()))
        checked += 1
        if quick_check_passes(text, table):
            passed += 1
            _require(
                nfc.normalize_str(text) == text,
                "soundness counterexample: "
                f"{[hex(ord(char)) for char in text]}",
            )
    return checked, passed


def build_nfc_qc(ctx: BuildContext, spec: TableSpec) -> BuiltTable:
    """Build the NFC quick-check table from the reference normalizer."""
    import numpy as np
    import tokenizers

    nfc, nfd = _reference_normalizers()
    started = time.time()
    log(
        "nfc: standard library Unicode version "
        f"{unicodedata.unidata_version} (cross-check source only)"
    )

    # Step 1: single-character scan. Anything whose NFC is not the
    # identity is unsafe outright.
    log("nfc: single-character NFC/NFD scan")
    nfc_changed = singleton_scan("NFC")
    nfd_changed = singleton_scan("NFD")
    unsafe: set[int] = set(nfc_changed)

    # Step 2 (before the first-element set is built): non-starters. The
    # first element of a composition pair is always a starter, because
    # recomposition only attaches marks to starters. Decomposable
    # codepoints whose first element is a mark exist, and letting them
    # into the first-element set would flag safe lower-class marks
    # through the reordering surface, which belongs to the rank check
    # instead. Candidates are the decomposition-free codepoints; a
    # decomposable codepoint that is still safe must be a starter, and
    # the standard library is used to assert that, resolving any
    # violation conservatively towards unsafe.
    candidates = [code for code in all_codepoints() if code not in nfd_changed]
    decomposable_safe = [code for code in nfd_changed if code not in unsafe]
    combining_violations = [
        code for code in decomposable_safe if unicodedata.combining(chr(code))
    ]
    for code in combining_violations:
        unsafe.add(code)
    log("nfc: non-starter detection")
    nonstarters = sorted(detect_nonstarters(candidates))
    nonstarter_set = set(nonstarters)
    log(f"nfc: {len(nonstarters)} non-starters")

    # Step 3: the first-element set. Every first element of a
    # composition pair is either the first element of some canonical
    # decomposition, or a recomposable composite (including Hangul LV,
    # excluding Hangul LVT, which is never a first element). Unsafe
    # singles are dropped: they cannot appear on the fast path.
    firsts = {
        ord(value[0]) for value in nfd_changed.values() if len(value) >= 2
    }
    composites = {
        code
        for code, value in nfd_changed.items()
        if code not in nfc_changed
        and len(value) >= 2
        and not (
            HANGUL_START <= code <= HANGUL_END
            and (code - HANGUL_START) % 28 != 0
        )
    }
    first_elements = sorted(
        code
        for code in firsts | composites
        if code not in nfc_changed
        and code not in nonstarter_set
        and not SURROGATE_START <= code < SURROGATE_STOP
    )
    log(
        f"nfc: {len(first_elements)} first elements "
        f"({len(firsts)} decomposition heads, {len(composites)} composites)"
    )

    # Step 4: second-element scan over the whole codepoint space.
    log(f"nfc: second-element scan with {ctx.procs} workers")
    seconds = second_element_scan(
        first_elements, frozenset(unsafe), ctx.procs
    )
    unsafe |= seconds

    # Step 5: rank the non-starters by combining class.
    ranks, classes = rank_nonstarters(nonstarters)
    _require(
        len(classes) <= 254,
        f"{len(classes)} combining-class ranks exceed the uint8 budget",
    )
    log(f"nfc: {len(classes)} combining-class ranks")

    # Step 6: assemble and audit.
    table = np.zeros(CODEPOINT_COUNT, dtype=np.uint8)
    for code, rank in ranks.items():
        table[code] = rank
    for code in unsafe:
        table[code] = NFC_UNSAFE
    table[SURROGATE_START:SURROGATE_STOP] = NFC_UNSAFE  # cannot be probed

    _nfc_pin_checks(table)
    _nfc_cross_check_pairs(table, nfc)
    log(f"nfc: normalizer version probes {_nfc_version_ladder(nfd)}")
    _nfc_cross_check_ranks(table, ranks)
    # A codepoint the standard library gives a non-zero combining class
    # while this table calls it a safe starter should not exist under
    # normalization stability; if one does, it is forced unsafe.
    forced = [
        code
        for code in all_codepoints()
        if int(table[code]) == 0 and unicodedata.combining(chr(code))
    ]
    for code in forced:
        table[code] = NFC_UNSAFE
    if forced:
        log(f"nfc: forced unsafe on stability grounds: {[hex(c) for c in forced]}")
    _nfc_planted_checks(table, nfc)
    log("nfc: soundness battery")
    checked, passed = _nfc_soundness_battery(table, nfc)
    log(f"nfc: {checked} strings, {passed} accepted, 0 counterexamples")

    histogram = {
        "safe_starter(0)": int((table == 0).sum()),
        "safe_nonstarter(1..K)": int(
            ((table > 0) & (table < NFC_UNSAFE)).sum()
        ),
        "unsafe(255)": int((table == NFC_UNSAFE).sum()),
    }
    meta = {
        "table": spec.file,
        "semantics": {
            "0": "safe starter",
            "1..K": "combining class rank",
            "255": "unsafe, take the slow path",
        },
        "predicate": (
            "pass <=> no 255 and no adjacent v[i-1] > v[i] != 0; pass "
            "implies the reference normalizer leaves the string unchanged"
        ),
        "K": len(classes),
        "judge": {
            "lib": "tokenizers",
            "version": tokenizers.__version__,
            "api": "normalizers.NFC/NFD .normalize_str",
        },
        "table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "histogram": histogram,
        "separator_note": (
            "batch probes are separated by U+0020, which has no canonical "
            "composition or decomposition role; the second-element scan "
            "covers (x, U+0020) for every first element x"
        ),
        "source": (
            "generate_class_tables.py (reference normalizer probed over "
            "the whole codepoint space: single-character identity, first "
            "elements against the whole space by bisection, combining "
            "class order by NFD reordering probes)"
        ),
    }
    log(f"nfc: build took {time.time() - started:.1f}s")
    return BuiltTable(array=table, meta=meta)


BUILDERS: dict[str, Builder] = {
    "cl100k": build_cl100k,
    "cl100k_m2l": build_cl100k_marks_as_letters,
    "deepseek": build_deepseek,
    "o200k": build_o200k,
    "kimi": build_kimi,
    "nfc_qc": build_nfc_qc,
}


# ---------------------------------------------------------------------------
# Payloads, writing and checking
# ---------------------------------------------------------------------------


def encode_npy(array: NDArray) -> bytes:
    """Serialize an array in ``.npy`` form, byte for byte as saved."""
    from numpy.lib import format as npy_format

    buffer = io.BytesIO()
    npy_format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def encode_meta(meta: Mapping[str, Any]) -> bytes:
    """Serialize sidecar metadata deterministically.

    ASCII-escaped so a sidecar stays byte-safe wherever it is stored,
    including next to the packaged tables; the frozen patterns it
    records contain non-ASCII characters.
    """
    return (json.dumps(meta, indent=1, ensure_ascii=True) + "\n").encode("utf-8")


def digest(payload: bytes) -> str:
    """``sha256:<hex>``, the form the registry binds and compares."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def write_atomic(path: Path, payload: bytes) -> None:
    """Write through a temporary file in the destination, then rename."""
    handle, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def update_family_table(path: Path, tables: Mapping[str, Any]) -> bool:
    """Backfill built digests into the routing data, atomically.

    Only the entries of the tables just built are touched; every other
    field of the document is preserved. The routing data is the single
    consumer-facing record of table identity, so the generator that
    produced the bytes is the one writer of their digests.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for table_id, entry in tables.items():
        spec = document["class_tables"][table_id]
        for key in ("sha256", "meta_file", "meta_sha256"):
            if key in entry and spec.get(key) != entry[key]:
                spec[key] = entry[key]
                changed = True
    if changed:
        payload = (
            json.dumps(document, indent=2, ensure_ascii=True) + "\n"
        ).encode("ascii")
        write_atomic(path, payload)
    return changed


def validate_shape(built: BuiltTable, spec: TableSpec) -> None:
    """Refuse to write a table the routing data does not describe."""
    _require(
        tuple(built.array.shape) == spec.shape,
        f"table {spec.table_id!r} has shape {tuple(built.array.shape)}, "
        f"routing data says {spec.shape}",
    )
    _require(
        built.array.dtype.name == spec.dtype,
        f"table {spec.table_id!r} has dtype {built.array.dtype.name}, "
        f"routing data says {spec.dtype}",
    )


def default_out_dir() -> Path:
    """Generated-table directory under the resolved cache directory.

    This is the third entry of the loader's search order. The cache
    directory is resolved by the library's own configuration chain
    rather than re-derived here, so there is one definition of it.
    """
    source_dir = REPO_ROOT / "src"
    if source_dir.is_dir() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    try:
        config_module = importlib.import_module("toktier.config")
    except ImportError as error:  # pragma: no cover - depends on layout
        raise TableBuildError(
            "cannot import toktier to resolve the cache directory; "
            "pass --out-dir explicitly"
        ) from error
    cache_dir = Path(config_module.Config.resolve().cache_dir)
    return cache_dir / "class_tables"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_class_tables.py",
        description=(
            "Generate the GPU backend's lookup tables. Each table is "
            "written as a .npy artifact, with a .meta.json sidecar where "
            "the kernel needs the constants. The JSON summary goes to "
            "stdout; progress and digests go to stderr."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "where to write the tables; defaults to the generated-table "
            "directory under the resolved cache directory"
        ),
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        metavar="ID",
        help=(
            "table id from the routing data; repeatable, defaults to all "
            "of them"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "regenerate in memory and compare against the existing files "
            "without writing anything; any difference exits non-zero"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "JSON manifest mapping family name to an object with a "
            "'local_dir'; needed by the tables whose constants are read "
            "from the artifacts themselves. Families it does not define "
            "are looked up in the local toktier artifact cache "
            "(toktier artifacts fetch <family>, honoring TOKTIER_HOME)"
        ),
    )
    parser.add_argument(
        "--extra-manifest",
        action="append",
        dest="extra_manifests",
        default=[],
        type=Path,
        metavar="PATH",
        help=(
            "additional manifest that only fills in families the primary "
            "one does not define, never overriding it; repeatable"
        ),
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=min(24, os.cpu_count() or 8),
        metavar="N",
        help=(
            "worker processes for the NFC second-element scan; affects "
            "runtime only, never the result"
        ),
    )
    return parser


def resolve_selection(
    requested: Sequence[str] | None, specs: Mapping[str, TableSpec]
) -> list[str]:
    """Table ids to build, validated against the routing data."""
    if not requested:
        return list(specs)
    unknown = [table_id for table_id in requested if table_id not in specs]
    _require(
        not unknown,
        f"unknown table id(s) {unknown}; the routing data defines "
        f"{sorted(specs)}",
    )
    seen: list[str] = []
    for table_id in requested:
        if table_id not in seen:
            seen.append(table_id)
    return seen


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs, families_by_table = load_routing_data(FAMILY_TABLE_PATH)
    selection = resolve_selection(args.tables, specs)
    out_dir = args.out_dir if args.out_dir is not None else default_out_dir()
    manifest = load_manifests(args.manifest, args.extra_manifests)
    for family, local_dir in default_cache_manifest().items():
        manifest.setdefault(family, local_dir)
    context = BuildContext(
        probe=ReferenceProbe(),
        manifest=manifest,
        families_by_table=families_by_table,
        procs=max(1, int(args.procs)),
        fetchable=packaged_families(),
    )

    if not args.check:
        out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, Any] = {}
    differences: list[dict[str, str]] = []

    for table_id in selection:
        spec = specs[table_id]
        builder = BUILDERS.get(spec.builder_key)
        _require(
            builder is not None,
            f"no builder for generator {spec.generator!r} (table "
            f"{table_id!r})",
        )
        assert builder is not None  # narrowed by the check above
        log(f"{table_id}: building")
        built = builder(context, spec)
        validate_shape(built, spec)

        payloads: list[tuple[str, bytes]] = [
            (spec.file, encode_npy(built.array))
        ]
        if built.meta is not None:
            payloads.append((spec.meta_file, encode_meta(built.meta)))

        entry: dict[str, Any] = {
            "file": spec.file,
            "sha256": digest(payloads[0][1]),
            "shape": list(spec.shape),
            "dtype": spec.dtype,
        }
        if built.meta is not None:
            entry["meta_file"] = spec.meta_file
            entry["meta_sha256"] = digest(payloads[1][1])

        for name, payload in payloads:
            path = out_dir / name
            if args.check:
                if not path.is_file():
                    differences.append({"file": name, "reason": "missing"})
                    log(f"{table_id}: {name} MISSING")
                    continue
                existing = path.read_bytes()
                if existing != payload:
                    differences.append(
                        {"file": name, "reason": "content differs"}
                    )
                    log(
                        f"{table_id}: {name} DIFFERS "
                        f"(on disk {digest(existing)}, "
                        f"regenerated {digest(payload)})"
                    )
                else:
                    log(f"{table_id}: {name} matches {digest(payload)}")
            else:
                write_atomic(path, payload)
                log(f"{digest(payload)}  {path}")

        if args.check:
            recorded = {"sha256": spec.sha256, "meta_sha256": spec.meta_sha256}
            for key in ("sha256", "meta_sha256"):
                relevant = key in entry or recorded[key] is not None
                if relevant and recorded[key] != entry.get(key):
                    differences.append(
                        {
                            "file": spec.file,
                            "reason": f"routing data {key} differs",
                        }
                    )
                    log(
                        f"{table_id}: routing data records "
                        f"{key}={recorded[key]}, regenerated "
                        f"{entry.get(key)}"
                    )

        tables[table_id] = entry

    if not args.check and update_family_table(FAMILY_TABLE_PATH, tables):
        log(f"routing data updated: {FAMILY_TABLE_PATH}")

    summary: dict[str, Any] = {
        "tables": tables,
        "out_dir": str(out_dir),
        "family_table": str(FAMILY_TABLE_PATH),
        "mode": "check" if args.check else "write",
    }
    if args.check:
        summary["differences"] = differences
    print(json.dumps(summary, indent=2))
    return 1 if differences else 0


def run(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: a refused build is not a crash."""
    try:
        return main(argv)
    except TableBuildError as error:
        log(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
