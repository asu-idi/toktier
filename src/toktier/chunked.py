"""Certified chunk-parallel encode for long requests on the CPU route.

A long document is cut at *certified stable boundaries*, the chunks are
encoded as one batch on the loaded tokenizer (the native batch path runs
the rows on its worker pool), and the id rows are concatenated. There is
no overlap, no seam matching and no retry, because a cut is placed only
where the pre-tokenizer of the certified families cannot carry a
pre-token across it: the synchronizing-transition predicate the
session-repair witness already uses (``BpeSyncBoundary`` in the routing
core -- eight class pairs over the frozen O/S/L/N/M table, minus two
carve-outs), extended here by a guard that keeps every cut outside any
added-token literal and, for a family whose artifact normalizes its
input, by the normalization-stability condition described below. The
concatenation is therefore the serial encoding by construction, and a
cut that cannot be certified is simply not made: fewer chunks, never a
different id.

The synchronizing predicate speaks about the text the pre-tokenizer
sees, which for an NFC family is the normalized text rather than the
input. A cut certified on input characters alone can be wrong there:
U+0958 is a letter, but NFC rewrites it as U+0915 followed by the
combining mark U+093C, so a cut between U+0958 and a following ``/`` is
a letter-to-punctuation transition in the input and a mark-to-
punctuation one in the normalized text, where the mark and the slash
share a pre-token. For such a family a candidate cut is accepted only
when it is also a certified cut of the normalized text. The input on
each side of the cut is anchored at the nearest *stable split point*: a
character before which NFC can be applied piecewise, which is one that
is not a mark, is unchanged by NFC on its own, does not reorder against
a combining mark placed on either side of it, and does not compose after
a Hangul consonant jamo or a Hangul syllable -- all decided with the
artifact's own normalizer (the reference engine's normalizer object, so
there is no Unicode-version skew), never with the standard library, and
memoized per code point; every ASCII character qualifies without a
test. The test is sound because a character can be changed by what
precedes it only by reordering (it then has a nonzero combining class,
and one of the two probe marks reorders against it) or by composition
(it is then the second element of a canonical composition, and in the
reference crate's tables every such element is a mark or a Hangul
vowel or trailing jamo; the shipped test re-checks this against the
installed crate). Then: no added-token literal may touch the anchored
window, because the reference normalizes each literal-free segment on
its own; the normalizer must map the window to the concatenation of
its two halves; and the predicate must hold on the two normalized
characters that meet at the cut. Together these make the normalized
document the concatenation of the normalized chunks with a certified
boundary at every seam. Three stable characters around the cut satisfy
all of this without any window, which is the common case in prose,
code and CJK text alike. The normalizer is used for equality only;
every character class is read from the frozen property table.

Two things this module deliberately is not. It is not a new route: the
chunks run through the tokenizer's own batch path, so the plan, the
per-input guards and the fallback ledger apply to each chunk exactly as
they apply to a whole document (a chunk that trips a guard is routed to
the reference on its own, which localizes that cost to one chunk). And
by default it is not a native change: the cut search and the
concatenation are Python, so the parallelism available is the batch
worker pool of the native engine -- the Rayon thread count at first use,
and at most one worker per 64 KiB of payload.

A native counterpart exists (``native=True``, and
``Tokenizer.encode(..., parallel=...)``): the same plan, position for
position, computed inside the native runtime, with the chunks encoded on
a persistent pool of corrected-Gigatoken workers and concatenated there
(``crates/toktier-routing-core/src/chunked.rs``). Its facts come back in
``ChunkedEncoding.native_facts``; the counters of this module count both
paths.

Admission is the certified repair roster, narrowed to the normalizers
this module reasons about, narrowed again to the artifacts whose chunked
path has recorded evidence. A tokenizer whose family and artifact digest
have a repair-table entry -- the eleven byte-level BPE artifacts the
shipped sync profile was certified on -- may be chunked when its loader
face carries no normalizer, the empty normalizer sequence, or NFC with
every added token matched on the input text (``normalized: false``),
with no added token that sets ``lstrip``, ``rstrip`` or ``single_word``
(the reference then matches a literal together with its neighbourhood,
which the cut rule does not reason about; refusal
``added_token_strip_or_single_word``), and when the shipped admission table
(``routing/tables/chunked_admission.v1.json``, generated from the
campaign readings by ``tools/update_chunked_admission.py``) has a
certified row for that family and artifact -- one with both levels of
chunked-path evidence, the adversarial normalization battery and the
corpus cut campaign: the set this module may chunk is the set the
chunked path was judged on. Any other tokenizer is encoded serially,
the refusal is counted under ``R_CHUNK_FAMILY_UNSUPPORTED`` and named in
``explain()["chunked"]["refusal"]`` (``artifact_without_chunked_evidence``
for a roster artifact without a row,
``chunked_evidence_incomplete:<status>`` for a row with one level
recorded, ``oracle_differs_from_chunked_evidence:<version>`` when the
facade's probe found the installed oracle outside the set the artifact's
record was judged with, ``loader_face_differs_from_chunked_evidence:<digest>``
when the live loader face is not the one the row's evidence was taken
on). A document in which no certified
cut can be found is encoded serially under ``R_CHUNK_NO_CUT``. Both
counters, and the per-call facts, are read back from
``Tokenizer.explain()["chunked"]`` once this module has served the
tokenizer, or from :func:`stats`.

The table also says which paths were judged for a row. With
``native=None`` (the default) the native encoder is used only when it can
run for the tokenizer -- the loaded runtime carries it and the route
carries the corrected CPU engine it is built on -- and the row records
the ``native`` path; otherwise the Python path runs. ``native=True``
demands the native encoder and raises when the runtime does not carry
it, when the tokenizer is not admitted, or when the row lacks the native
path; ``native=False`` is the Python path.
Collecting the evidence a row is made of needs the gate lifted for the
tokenizer under judgement; the campaign tool does that through the
private planner constructor, and ``explain()["chunked"]["admission_gate"]``
says so when it happens.

Source text in this repository is ASCII, so the characters named in the
carve-outs and in the example above are written as escapes.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from typing import Any

from ._jcs import canonical_json
from .backends.loader_face import loader_face_sha256
from .errors import RegistryInvalid
from .facade.api import Encoding, Tokenizer
from .policy import ReasonCode
from .repair.registry import family_spec, pclass_table
from .routing.tables import CHUNKED_ADMISSION

__all__ = [
    "CHUNK_TABLE",
    "DEFAULT_SCAN_BUDGET",
    "MIN_BYTES_PER_WORKER",
    "MIN_CHARS",
    "SYNC_PAIRS",
    "ChunkPlan",
    "ChunkedEncoding",
    "admission",
    "default_chunks",
    "encode",
    "is_admitted",
    "native_available",
    "plan",
    "stats",
    "warm",
]

# Class indices of the frozen O/S/L/N/M property table.
_O, _S, _L, _N, _M = 0, 1, 2, 3, 4

#: The eight synchronizing class transitions, as ``(left, right)`` class
#: pairs: a pre-token of the certified byte-level BPE families never
#: spans one of them. This is the set the routing core's
#: ``BpeSyncBoundary::accepts`` matches on; keep the two in step.
SYNC_PAIRS: frozenset[tuple[int, int]] = frozenset(
    {
        (_L, _N),
        (_L, _O),
        (_L, _S),
        (_N, _L),
        (_N, _O),
        (_N, _S),
        (_O, _N),
        (_O, _S),
    }
)
_SYNC: tuple[tuple[bool, ...], ...] = tuple(
    tuple((left, right) in SYNC_PAIRS for right in range(5)) for left in range(5)
)

#: The native batch path opens at most one worker per this many payload
#: bytes (``crates/toktier-routing-core/src/fast_cpu.rs``); more chunks
#: than that only add concatenation work.
MIN_BYTES_PER_WORKER = 64 * 1024

#: Chunk-count staircase ``(characters below, chunks)``, read top down;
#: the last row has no bound. The values are the argmin of the measured
#: batch-path latency on the certified CPU engine over three families
#: (qwen3_8b, llama_3_1_8b, gpt_oss_120b) and four corpora (prose, code,
#: chat transcripts, a multilingual mix) at 64 K, 256 K, 1 M and 4 M
#: characters on a 16-core NUMA node, rounded to one conservative
#: staircase (the slower engine of the three preferred 32 already at 1 M;
#: the two faster ones preferred 16, and the 64 KiB worker floor makes the
#: difference immaterial there). Below 256 K characters the batch path is
#: not faster than a single encode, so the table says one chunk.
CHUNK_TABLE: tuple[tuple[int | None, int], ...] = (
    (256_000, 1),
    (1_000_000, 4),
    (4_000_000, 16),
    (None, 32),
)

#: Characters below which :func:`default_chunks` answers one chunk.
MIN_CHARS: int = 256_000

#: How far below a target position the cut search looks before it gives
#: that cut up (characters).
DEFAULT_SCAN_BUDGET = 4096

#: The first, cheap part of every cut search; the guard window is built
#: for this span first and widened only on a miss. A smaller
#: ``scan_budget`` bounds it.
_QUICK_SCAN = 48

#: Documents shorter than this are never cut, whatever ``k`` says.
_MIN_TEXT_CHARS = 256

#: How far (characters) the normalization-stability check looks for a
#: stable split point on each side of a candidate cut before giving that
#: candidate up. Far above the longest combining sequence any real text
#: carries (the stream-safe text format bounds those at 30), so the limit
#: only matters inside long runs of marks or jamo, where the candidate is
#: skipped and the search moves on.
_ANCHOR_SCAN = 256

#: Probes of the stable-split-point test (module docstring): a combining
#: mark of canonical class 230 placed before the character and one of
#: class 1 placed after it -- any character of nonzero class reorders
#: against one of the two, and neither mark composes with anything -- and
#: a leading Hangul consonant jamo and syllable, after which a vowel or
#: trailing jamo composes.
_PROBE_BEFORE = "\u0301"
_PROBE_AFTER = "\u0335"
_PROBE_JAMO_L = "\u1100"
_PROBE_SYLLABLE = "\uac00"
_STABLE_CACHE_LIMIT = 1 << 16

#: Loader-face normalizer shapes this module reasons about.
_NORMALIZER_NONE = "none"
_NORMALIZER_NFC = "nfc"

#: Added-token flags under which the reference matches a literal by more
#: than its own text: ``lstrip`` and ``rstrip`` take the neighbouring
#: whitespace into the match, and ``single_word`` matches the literal only
#: where no word character touches it. With ``rstrip`` or ``single_word``
#: a cut the sync rule accepts beside such a literal can change the ids;
#: ``lstrip`` takes whitespace on the literal's left only, where the rule
#: never cuts, and is refused with the other two so that admission does
#: not rest on that property of the profile. A loader face with an added
#: token that sets any of them is not admitted (refusal
#: ``added_token_strip_or_single_word``).
_LITERAL_FLAGS = ("lstrip", "rstrip", "single_word")
_R_LITERAL_FLAGS = "added_token_strip_or_single_word"

_R_NO_CUT = ReasonCode.R_CHUNK_NO_CUT.value
_R_FAMILY = ReasonCode.R_CHUNK_FAMILY_UNSUPPORTED.value

#: The refusal of a roster artifact the admission table has no row for,
#: and the prefix of the refusal of a row whose evidence has one level
#: only (``chunked_evidence_incomplete:adversarial_only`` and the like).
_R_EVIDENCE = "artifact_without_chunked_evidence"
_R_INCOMPLETE = "chunked_evidence_incomplete"
_R_ORACLE = "oracle_differs_from_chunked_evidence"
_R_FACE = "loader_face_differs_from_chunked_evidence"
_STATUS_CERTIFIED = "certified"

#: The shipped admission table and the domain its root digest is taken in
#: (``tools/registry_common.py`` writes the same tag).
_ADMISSION_TABLE = CHUNKED_ADMISSION
_ADMISSION_SCHEMA = "toktier.chunked_admission.v1"
_ADMISSION_DOMAIN = b"toktier.chunked_admission.v1\0"
_PATH_NATIVE = "native"


def _table_failure(message: str, **details: object) -> RegistryInvalid:
    return RegistryInvalid(
        message, details={"path": str(_ADMISSION_TABLE), **details}
    )


@lru_cache(maxsize=1)
def _admission_rows() -> dict[tuple[str, str], Mapping[str, object]]:
    """The admission table, keyed by ``(family, artifact_sha256)``.

    The root digest is recomputed before a row is trusted, as the repair
    registry does for its tables: a table that does not hash to its own
    digest is refused rather than read around.
    """
    try:
        raw = json.loads(_ADMISSION_TABLE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _table_failure("the chunked admission table cannot be read") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != _ADMISSION_SCHEMA
        or raw.get("schema_version") != 1
    ):
        raise _table_failure("the chunked admission table has an unknown schema")
    body = {key: value for key, value in raw.items() if key != "root_digest"}
    expected = "sha256:" + hashlib.sha256(
        _ADMISSION_DOMAIN + canonical_json(body)
    ).hexdigest()
    if raw.get("root_digest") != expected:
        raise _table_failure(
            "the chunked admission table root digest does not match",
            observed=raw.get("root_digest"),
            expected=expected,
        )
    rows = raw.get("rows")
    if not isinstance(rows, list):
        raise _table_failure("the chunked admission table has no rows")
    out: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("paths"), list):
            raise _table_failure("a chunked admission row is malformed")
        key = (str(row.get("family")), str(row.get("artifact_sha256")))
        if key in out:
            raise _table_failure(
                "the chunked admission table repeats an artifact", family=key[0]
            )
        out[key] = row
    return out


def default_chunks(chars: int) -> int:
    """The chunk count the table selects for a document of ``chars``.

    The table value is capped by the number of batch workers the payload
    can use: one per started :data:`MIN_BYTES_PER_WORKER`, the native
    batch path's own rule, counted here in characters (which never
    overstates the byte length).
    """
    if chars < 0:
        raise ValueError("chars must be non-negative")
    selected = 1
    for bound, chunks in CHUNK_TABLE:
        if bound is None or chars < bound:
            selected = chunks
            break
    useful = -(-chars // MIN_BYTES_PER_WORKER)
    return max(1, min(selected, useful))


def _pair_ok(table: bytes, previous: str, current: str) -> bool:
    """The sync-witness predicate on two adjacent characters."""
    left = table[ord(previous)]
    right = table[ord(current)]
    if not _SYNC[left][right]:
        return False
    # Punctuation absorbs a following line break into its pre-token.
    if left == _O and right == _S and current in "\r\n":
        return False
    # An apostrophe after letters may open a contraction suffix.
    return not (left == _L and right == _O and current == "'")


def _boundary_ok(table: bytes, text: str, position: int) -> bool:
    """Whether a cut between ``text[position - 1]`` and ``text[position]``
    is a certified stable boundary of ``text`` as written (the predicate
    of the sync witness on input characters)."""
    return _pair_ok(table, text[position - 1], text[position])


class _LiteralGuard:
    """Cut positions strictly inside an added-token literal are forbidden.

    Conservative on purpose: every occurrence of every literal counts,
    overlapping ones included, and at each start the longest literal that
    matches there is taken. That is a superset of what any leftmost-longest
    extraction would actually consume, so a cut the guard allows never
    splits a literal the reference would have matched, and a window the
    guard reports untouched holds no literal the reference would extract.
    """

    __slots__ = ("_pattern", "count", "lmax")

    def __init__(self, literals: Sequence[str]) -> None:
        unique = sorted({item for item in literals if item}, key=lambda s: (-len(s), s))
        self.count = len(unique)
        self.lmax = max((len(item) for item in unique), default=0)
        self._pattern: re.Pattern[str] | None = (
            re.compile("(?=(" + "|".join(re.escape(item) for item in unique) + "))")
            if unique
            else None
        )

    def forbidden(self, text: str, low: int, high: int) -> set[int]:
        """Positions in ``[low, high]`` that fall inside some occurrence."""
        if self._pattern is None or self.lmax < 2:
            return set()
        start = max(0, low - self.lmax + 1)
        end = min(len(text), high + self.lmax - 1)
        out: set[int] = set()
        for match in self._pattern.finditer(text, start, end):
            first = match.start()
            last = first + len(match.group(1))
            out.update(range(max(first + 1, low), min(last, high + 1)))
        return out

    def touches(self, text: str, low: int, high: int) -> bool:
        """Whether some occurrence intersects the span ``[low, high)``."""
        if self._pattern is None:
            return False
        start = max(0, low - self.lmax + 1)
        end = min(len(text), high + self.lmax - 1)
        for match in self._pattern.finditer(text, start, end):
            first = match.start()
            if first >= high:
                break
            if first + len(match.group(1)) > low:
                return True
        return False


class _NfcRule:
    """The normalization-stability condition of an NFC family.

    ``normalize`` is the artifact's own normalizer. With ``left`` the
    nearest stable split point before the cut and ``right`` the nearest
    one after it (the start and the end of the text count as well), the
    normalized document is ``N(text[:left]) + N(text[left:right]) +
    N(text[right:])`` provided no added-token literal touches
    ``[left, right)`` (the reference normalizes literal-free segments,
    and the guard's occurrence set is a superset of the extracted ones);
    when ``N(text[left:right])`` equals ``N(text[left:position]) +
    N(text[position:right])`` it is also ``N(text[:position]) +
    N(text[position:])``, and the two characters meeting at the cut are
    the last of the first half and the first of the second, on which the
    sync predicate is then decided.
    """

    __slots__ = ("_cache", "_guard", "_normalize", "_table")

    def __init__(
        self, table: bytes, guard: _LiteralGuard, normalize: Callable[[str], str]
    ) -> None:
        self._table = table
        self._guard = guard
        self._normalize = normalize
        self._cache: dict[str, bool] = {}

    def stable(self, char: str) -> bool:
        """Whether ``char`` is a stable split point (module docstring)."""
        if char.isascii():
            return True
        cached = self._cache.get(char)
        if cached is not None:
            return cached
        normalize = self._normalize
        verdict = (
            self._table[ord(char)] != _M
            and normalize(char) == char
            and normalize(_PROBE_BEFORE + char) == _PROBE_BEFORE + char
            and normalize(char + _PROBE_AFTER) == char + _PROBE_AFTER
            and normalize(_PROBE_JAMO_L + char) == _PROBE_JAMO_L + char
            and normalize(_PROBE_SYLLABLE + char) == _PROBE_SYLLABLE + char
        )
        if len(self._cache) < _STABLE_CACHE_LIMIT:
            self._cache[char] = verdict
        return verdict

    def accepts(self, text: str, position: int) -> bool:
        """Whether the cut at ``position`` is also a certified cut of the
        normalized text."""
        around = text[position - 1 : position + 2]
        if around.isascii() or all(self.stable(char) for char in around):
            # Stable on both sides and beyond: the cut is its own
            # normalized cut, and the classes are the input classes.
            return True
        stable = self.stable
        length = len(text)
        left = position - 1
        floor_index = max(0, position - _ANCHOR_SCAN)
        while left >= floor_index and not stable(text[left]):
            left -= 1
        if left < floor_index:
            if floor_index > 0:
                return False
            left = 0  # the start of the text anchors as well
        right = position + 1
        ceiling = min(length, position + 1 + _ANCHOR_SCAN)
        while right < ceiling and not stable(text[right]):
            right += 1
        if right == ceiling and ceiling < length:
            return False
        if self._guard.touches(text, left, right):
            return False
        normalize = self._normalize
        before = normalize(text[left:position])
        after = normalize(text[position:right])
        if not before or not after:
            return False
        if normalize(text[left:right]) != before + after:
            return False
        return _pair_ok(self._table, before[-1], after[0])


def _flagged_literal(document: Mapping[str, object]) -> bool:
    """Whether an added token of the face sets a flag in :data:`_LITERAL_FLAGS`."""
    rows = document.get("added_tokens") or []
    return isinstance(rows, Sequence) and any(
        isinstance(row, Mapping) and any(row.get(flag) for flag in _LITERAL_FLAGS)
        for row in rows
    )


def _normalizer_mode(document: Mapping[str, object]) -> tuple[str | None, str | None]:
    """``(mode, refusal)`` for the loader face: its normalizer section
    first, then the added-token flags the cut rule does not reason about
    (:data:`_LITERAL_FLAGS`)."""
    section = document.get("normalizer")
    if section is None:
        mode = _NORMALIZER_NONE
    elif not isinstance(section, Mapping):
        return None, "normalizer_not_supported"
    else:
        kind = section.get("type")
        if kind == "Sequence" and not section.get("normalizers"):
            mode = _NORMALIZER_NONE
        elif kind == "NFC":
            rows = document.get("added_tokens") or []
            if isinstance(rows, Sequence) and any(
                isinstance(row, Mapping) and row.get("normalized") for row in rows
            ):
                # Such a literal is matched on the normalized text, which
                # the input-text guard does not see; not reasoned about here.
                return None, "nfc_with_normalized_added_tokens"
            mode = _NORMALIZER_NFC
        else:
            return None, f"normalizer_not_supported:{kind}"
    if _flagged_literal(document):
        return None, _R_LITERAL_FLAGS
    return mode, None


def _oracle_outside_certified_set(tokenizer: object) -> str | None:
    """The installed oracle version when the facade's probe found it
    outside the set the artifact's certification record was judged with,
    ``None`` when it is inside or there is no record to compare with.

    This is the comparison behind the planner's ``R_ORACLE_MISMATCH``
    (``toktier.routing.plan``, the oracle-version check), read from the
    tokenizer's own probe snapshot and registry view, so it answers the
    same under every routing policy, including the ones that never reach
    that check.
    """
    snapshot = getattr(tokenizer, "_snapshot", None)
    registry = getattr(tokenizer, "_registry", None)
    match = getattr(snapshot, "certification", None)
    if match is None or registry is None:
        return None
    oracle = registry.oracle(match.record.oracle_id)
    certified = oracle.certified_versions if oracle is not None else ()
    installed = getattr(snapshot, "oracle_version", None)
    if installed in certified:
        return None
    return str(installed)


@dataclass(frozen=True)
class ChunkPlan:
    """Where a document was cut, and what the search had to give up."""

    #: Half-open character spans, in order, covering the document.
    spans: tuple[tuple[int, int], ...]
    #: Requested chunk count.
    k: int
    #: Target positions for which no certified cut was found.
    cut_misses: int
    #: Candidate positions rejected because they fell inside a literal.
    literal_rejects: int
    #: Candidate positions rejected by the normalization-stability
    #: condition (NFC families only).
    norm_rejects: int = 0

    @property
    def cuts(self) -> tuple[int, ...]:
        return tuple(span[0] for span in self.spans[1:])


@dataclass(frozen=True)
class ChunkedEncoding(Encoding):
    """An :class:`Encoding` that also says how it was produced.

    ``chunks`` is the number of rows that ran; ``serial_reason`` names the
    reason code when the document was encoded serially instead, and is
    ``None`` for a chunked result and for a document below the threshold.
    """

    chunks: int = 1
    k_requested: int = 1
    cuts: tuple[int, ...] = ()
    cut_misses: int = 0
    literal_rejects: int = 0
    norm_rejects: int = 0
    serial_reason: str | None = None
    #: The native encoder's own facts (row sources, phase timings) when
    #: the native path produced this result; ``None`` otherwise.
    native_facts: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "cuts", tuple(self.cuts))


class _Planner:
    """Per-tokenizer admission, guard, normalizer and counters.

    Admission is decided once, in this order: the certified repair roster
    (family and artifact digest), a loader face to read, a normalizer
    shape the rule reasons about, no added token with a strip or
    single-word flag, and a row in the shipped admission table (certified,
    the installed oracle inside the set the record was judged with, and
    the live face the row's). ``evidence_gate=False`` leaves the last
    step out; it exists for the campaign that produces the table's rows
    and for the equivalence tests of the two planners, and the report
    says when it was used.
    """

    def __init__(self, tokenizer: Tokenizer, *, evidence_gate: bool = True) -> None:
        self._table = pclass_table()
        self.family = tokenizer.family
        self.refusal: str | None = None
        self.normalizer: str | None = None
        self.evidence_gate = evidence_gate
        self.evidence_id: str | None = None
        self.evidence_status: str | None = None
        self.evidence_paths: tuple[str, ...] = ()
        self.loader_face_sha256: str | None = None
        self._rule: _NfcRule | None = None
        digest = getattr(tokenizer, "_artifact_sha256", None)
        if not isinstance(digest, str) or family_spec(self.family, digest) is None:
            self.refusal = "artifact_not_in_certified_repair_roster"
        literals: list[str] = []
        loader_face = getattr(
            getattr(tokenizer, "_backend", None), "materialized_tokenizer_json", None
        )
        if self.refusal is None:
            if not callable(loader_face):
                self.refusal = "no_loader_face_document"
            else:
                face_text = loader_face()
                self.loader_face_sha256 = loader_face_sha256(face_text)
                document = json.loads(face_text)
                literals = [
                    str(row.get("content", ""))
                    for row in document.get("added_tokens") or []
                    if isinstance(row, Mapping)
                ]
                self.normalizer, self.refusal = _normalizer_mode(document)
        row = None
        if isinstance(digest, str):
            row = _admission_rows().get((self.family, digest))
        if row is not None:
            paths = row.get("paths")
            self.evidence_id = str(row.get("evidence_id"))
            self.evidence_status = str(row.get("status"))
            if isinstance(paths, list):
                self.evidence_paths = tuple(str(path) for path in paths)
        if self.refusal is None and evidence_gate:
            recorded_face = None if row is None else row.get("loader_face_sha256")
            if row is None:
                self.refusal = _R_EVIDENCE
            elif self.evidence_status != _STATUS_CERTIFIED:
                # One evidence level recorded (the battery or the corpus
                # campaign, not both): named, and refused like an absent row.
                self.refusal = f"{_R_INCOMPLETE}:{self.evidence_status}"
            elif (oracle := _oracle_outside_certified_set(tokenizer)) is not None:
                # The installed reference is not one the artifact's record
                # was judged with (the facade's probe found it outside the
                # certified set): the chunked evidence was taken against
                # the certified oracle and does not speak for this one.
                self.refusal = f"{_R_ORACLE}:{oracle}"
            elif recorded_face is not None and recorded_face != self.loader_face_sha256:
                # The live loader face is not the one the evidence was taken
                # on (another loader version materializing another
                # pre-tokenizer, for one): the evidence does not speak for it.
                self.refusal = f"{_R_FACE}:{str(self.loader_face_sha256)[:12]}"
        self.guard = _LiteralGuard(literals)
        if self.normalizer == _NORMALIZER_NFC:
            # The reference engine's own NFC, not the standard library's:
            # same crate, same Unicode tables.
            from tokenizers.normalizers import NFC

            self._rule = _NfcRule(self._table, self.guard, NFC().normalize_str)
        self._lock = threading.Lock()
        self._calls = 0
        self._chunked = 0
        self._serial = 0
        self._below_threshold = 0
        self._chunks_total = 0
        self._cut_misses = 0
        self._literal_rejects = 0
        self._norm_rejects = 0
        self._native_calls = 0
        self._fallback_counts: dict[str, int] = {}
        self._last: dict[str, object] | None = None

    @property
    def admitted(self) -> bool:
        return self.refusal is None

    @property
    def native_evidence(self) -> bool:
        """Whether the native path may run: its evidence is recorded, or
        the gate is lifted for collecting it."""
        return _PATH_NATIVE in self.evidence_paths or not self.evidence_gate

    def _scan(self, text: str, low: int, high: int, rejects: list[int]) -> int:
        """The highest certified cut in ``[low, high]``, or ``-1``."""
        if high < low:
            return -1
        table = self._table
        rule = self._rule
        forbidden = self.guard.forbidden(text, low, high)
        position = high
        while position >= low:
            if _boundary_ok(table, text, position):
                if position in forbidden:
                    rejects[0] += 1
                elif rule is not None and not rule.accepts(text, position):
                    rejects[1] += 1
                else:
                    return position
            position -= 1
        return -1

    def plan(
        self, text: str, k: int, min_chunk: int | None, scan_budget: int
    ) -> ChunkPlan:
        length = len(text)
        if k <= 1 or length < _MIN_TEXT_CHARS:
            return ChunkPlan(((0, length),), k, 0, 0)
        floor = (
            max(16, min(1024, length // (4 * k))) if min_chunk is None else min_chunk
        )
        cuts: list[int] = []
        misses = 0
        rejects = [0, 0]
        previous = 0
        quick = min(_QUICK_SCAN, scan_budget)
        for index in range(1, k):
            target = (length * index) // k
            if target <= previous:
                misses += 1
                continue
            low = max(previous + 1, target - quick)
            found = self._scan(text, low, target, rejects)
            if found < 0 and scan_budget > quick and low > previous + 1:
                high = low - 1
                low = max(previous + 1, target - scan_budget)
                found = self._scan(text, low, high, rejects)
            # A cut that would leave either neighbouring chunk below the
            # floor -- the one before it, or the tail after it -- is not
            # made; the text stays with the preceding chunk.
            if found < 0 or found - previous < floor or length - found < floor:
                misses += 1
                continue
            cuts.append(found)
            previous = found
        bounds = [0, *cuts, length]
        spans = tuple((bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1))
        return ChunkPlan(spans, k, misses, rejects[0], rejects[1])

    def record(
        self,
        *,
        chunks: int,
        k: int,
        chars: int,
        tokens: int,
        plan: ChunkPlan | None,
        serial_reason: str | None,
        below_threshold: bool,
        native: bool = False,
    ) -> None:
        with self._lock:
            self._calls += 1
            if serial_reason is not None:
                self._serial += 1
                self._fallback_counts[serial_reason] = (
                    self._fallback_counts.get(serial_reason, 0) + 1
                )
            elif below_threshold:
                self._below_threshold += 1
            else:
                self._chunked += 1
            self._chunks_total += chunks
            if plan is not None:
                self._cut_misses += plan.cut_misses
                self._literal_rejects += plan.literal_rejects
                self._norm_rejects += plan.norm_rejects
            if native:
                self._native_calls += 1
            self._last = {
                "chars": chars,
                "tokens": tokens,
                "k_requested": k,
                "chunks": chunks,
                "serial_reason": serial_reason,
                "cut_misses": plan.cut_misses if plan is not None else 0,
                "literal_rejects": plan.literal_rejects if plan is not None else 0,
                "norm_rejects": plan.norm_rejects if plan is not None else 0,
                "native": native,
            }

    def admission_facts(self) -> dict[str, object]:
        """The admission decision and what it rests on (no counters)."""
        return {
            "admitted": self.admitted,
            "refusal": self.refusal,
            "family": self.family,
            "normalizer": self.normalizer,
            "literals": self.guard.count,
            "loader_face_sha256": self.loader_face_sha256,
            "evidence_id": self.evidence_id,
            "evidence_status": self.evidence_status,
            "evidence_paths": list(self.evidence_paths),
            "admission_gate": (
                "evidence_table"
                if self.evidence_gate
                else "lifted_for_evidence_collection"
            ),
        }

    def report(self) -> dict[str, object]:
        with self._lock:
            return {
                **self.admission_facts(),
                "calls": self._calls,
                "applied": self._chunked,
                "serial": self._serial,
                "below_threshold": self._below_threshold,
                "chunks_total": self._chunks_total,
                "cut_misses": self._cut_misses,
                "literal_rejects": self._literal_rejects,
                "norm_rejects": self._norm_rejects,
                "native_calls": self._native_calls,
                "fallback_counts": dict(sorted(self._fallback_counts.items())),
                "last": dict(self._last) if self._last is not None else None,
            }


_PLANNERS: weakref.WeakKeyDictionary[Tokenizer, _Planner] = weakref.WeakKeyDictionary()
_PLANNERS_LOCK = threading.Lock()


def _planner_for(
    tokenizer: Tokenizer, *, evidence_gate: bool | None = None
) -> _Planner:
    """The planner of ``tokenizer``, created on first use.

    ``evidence_gate`` is ``None`` for the ordinary call (an existing
    planner is returned, a new one is gated on the admission table). A
    boolean replaces the planner with one built that way, counters
    reset; ``False`` is the lift the campaign tool and the equivalence
    tests use before any other call has served the tokenizer.
    """
    with _PLANNERS_LOCK:
        planner = _PLANNERS.get(tokenizer)
        if planner is None or (
            evidence_gate is not None and planner.evidence_gate != evidence_gate
        ):
            gated = True if evidence_gate is None else evidence_gate
            planner = _Planner(tokenizer, evidence_gate=gated)
            _PLANNERS[tokenizer] = planner
            register = getattr(tokenizer, "_register_report", None)
            if callable(register):
                register("chunked", planner.report)
        return planner


def is_admitted(tokenizer: Tokenizer) -> bool:
    """Whether this tokenizer may be chunked: certified repair roster, a
    loader face this module reasons about (normalizer and added-token
    flags), and a certified row in the admission table that holds for the
    installed oracle and the live face (:func:`admission` names the
    refusal)."""
    return _planner_for(tokenizer).admitted


def stats(tokenizer: Tokenizer) -> dict[str, object]:
    """The counters this module keeps for ``tokenizer``."""
    return _planner_for(tokenizer).report()


def plan(
    tokenizer: Tokenizer,
    text: str,
    *,
    k: int | None = None,
    min_chunk: int | None = None,
    scan_budget: int = DEFAULT_SCAN_BUDGET,
    native: bool | None = None,
) -> ChunkPlan:
    """The cut plan :func:`encode` would use, without encoding.

    Raises ``ValueError`` for a tokenizer outside the admitted set; the
    encode entry point prefers to fall back, this one prefers to say so.
    ``native=True`` asks the native runtime for its plan instead (and
    raises ``RuntimeError`` when that path may not run, as :func:`encode`
    does); the two are meant to agree position for position.
    """
    planner = _planner_for(tokenizer)
    if not planner.admitted:
        raise ValueError(
            f"{tokenizer.family!r} has no certified chunk-cut rule: {planner.refusal}"
        )
    chunks = _resolve_k(k, len(text))
    budget = _check_budget(scan_budget)
    if native:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        runtime = _require_native(tokenizer, planner)
        facts = runtime.chunked_plan(text, chunks, min_chunk, budget)
        return ChunkPlan(
            spans=tuple((int(low), int(high)) for low, high in facts["spans"]),
            k=chunks,
            cut_misses=int(facts["cut_misses"]),
            literal_rejects=int(facts["literal_rejects"]),
            norm_rejects=int(facts["norm_rejects"]),
        )
    return planner.plan(text, chunks, min_chunk, budget)


_NO_NATIVE_SURFACE = "the loaded runtime has no native chunked encoder"


def _native_runtime(tokenizer: Tokenizer) -> Any:
    """The native request runtime with the chunked surface, or raise."""
    runtime = tokenizer._native_request_runtime()
    if runtime is None or not callable(getattr(runtime, "chunked_encode", None)):
        raise RuntimeError(_NO_NATIVE_SURFACE)
    return runtime


def _native_state(tokenizer: Tokenizer) -> tuple[dict[str, object] | None, str | None]:
    """The native planner's admission facts, or why there are none.

    ``(facts, None)`` when the native chunk encoder can run for
    ``tokenizer``: the loaded runtime carries the chunked surface, and its
    route carries the corrected CPU engine the native planner is built on,
    which the runtime confirms by describing that planner. ``(None,
    reason)`` otherwise: no surface, or a route without the corrected CPU
    engine (the reference-only route, for one), where the surface is
    present but the native encoder would only encode serially.
    """
    runtime = tokenizer._native_request_runtime()
    if runtime is None or not callable(getattr(runtime, "chunked_encode", None)):
        return None, _NO_NATIVE_SURFACE
    try:
        return dict(runtime.chunked_admission()), None
    except RuntimeError as exc:
        return None, str(exc)


def native_available(tokenizer: Tokenizer) -> bool:
    """Whether the native chunk encoder can run for ``tokenizer``.

    True when the loaded runtime carries the native chunked surface *and*
    its route carries the corrected CPU engine the native planner runs on.
    A runtime that carries the surface on a route without that engine
    answers False: the native encoder would encode every input serially
    there (``R_CHUNK_ROUTE_NOT_FAST_CPU``), so ``native=None`` takes the
    Python path instead. Admission is a separate question, answered by
    :func:`admission`, which also names the reason when this is False.
    """
    return _native_state(tokenizer)[1] is None


def _require_native(tokenizer: Tokenizer, planner: _Planner) -> Any:
    """The native runtime when the native path may run for this tokenizer.

    Raises ``RuntimeError`` naming what is missing: the runtime surface,
    the admission itself, or the recorded evidence for the native path.
    """
    runtime = _native_runtime(tokenizer)
    if not planner.admitted:
        raise RuntimeError(
            f"{planner.family!r} is not admitted to the chunked path "
            f"({planner.refusal}), and the native chunk encoder was asked for "
            "explicitly"
        )
    if not planner.native_evidence:
        raise RuntimeError(
            f"{planner.family!r}: the admission table records no evidence for the "
            f"native chunked path of this artifact (recorded paths: "
            f"{list(planner.evidence_paths)}); pass native=None to let the module "
            "choose, or native=False for the Python path"
        )
    return runtime


def _select_native(
    tokenizer: Tokenizer, planner: _Planner, native: bool | None
) -> bool:
    """Whether :func:`encode` runs the native path for this call."""
    if native is None:
        return (
            planner.admitted and planner.native_evidence and native_available(tokenizer)
        )
    if native:
        _require_native(tokenizer, planner)
        return True
    return False


def admission(tokenizer: Tokenizer) -> dict[str, object]:
    """The admission facts, merged.

    This module's decision and what it rests on (roster, loader face,
    admission-table row and the paths it records), whether the native
    encoder can run (``native_available``, as :func:`native_available`
    decides it) and whether ``native=None`` would select it, and under
    ``"native"`` the native planner's own admission facts when it can run
    (``None`` otherwise, with ``"native_unavailable_reason"`` saying why:
    no native surface, or a route without the corrected CPU engine the
    native planner is built on). ``native_by_default`` is true only when
    ``native_available`` is, so it is never reported beside an
    unavailable reason.
    """
    planner = _planner_for(tokenizer)
    native_facts, unavailable = _native_state(tokenizer)
    available = unavailable is None
    facts: dict[str, object] = planner.admission_facts()
    facts["native_available"] = available
    facts["native_by_default"] = (
        planner.admitted and planner.native_evidence and available
    )
    facts["native"] = native_facts
    facts["native_unavailable_reason"] = unavailable
    return facts


def warm(
    tokenizer: Tokenizer, *, workers: int | None = None, text: str | None = None
) -> int:
    """Fork (and, with ``text``, pre-warm) the native chunk worker pool.

    Returns the pool size. Without the call the pool is forked on first use.
    """
    size: int = _native_runtime(tokenizer).chunked_warm(workers, text)
    return size


def _resolve_k(k: int | None, chars: int) -> int:
    if k is None:
        return default_chunks(chars)
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer or None")
    return k


def _check_budget(scan_budget: int) -> int:
    if type(scan_budget) is not int or scan_budget < 1:
        raise ValueError("scan_budget must be a positive integer")
    return scan_budget


def _encode_rows(tokenizer: Tokenizer, chunks: Sequence[str]) -> list[Sequence[int]]:
    core = getattr(tokenizer, "_encode_batch_core", None)
    if callable(core):
        rows: list[Sequence[int]] = core(chunks)
        return rows
    return [row.ids for row in tokenizer.encode_batch(chunks)]


def encode(
    tokenizer: Tokenizer,
    text: str,
    *,
    k: int | None = None,
    min_chunk: int | None = None,
    scan_budget: int = DEFAULT_SCAN_BUDGET,
    native: bool | None = None,
) -> ChunkedEncoding:
    """Encode ``text`` in certified chunks; ids equal ``tokenizer.encode``.

    ``k`` is the requested chunk count (``None`` reads :data:`CHUNK_TABLE`
    for the document length); ``min_chunk`` the smallest chunk the plan
    may produce, first and last included (``None`` selects
    ``min(1024, len // (4 k))``); ``scan_budget`` how far below each
    target the cut search looks. The result is the core stream, exactly
    as ``encode(text, lookup="off")`` returns it: special tokens are a
    read-time template, as for sessions.

    A tokenizer outside the admitted set (the certified roster, with a
    normalizer this module reasons about, and a row in the admission
    table), and a document in which no certified cut exists, are encoded
    serially on the tokenizer's plain route; the result says so in
    ``serial_reason`` and the tokenizer's ``explain()["chunked"]`` block
    counts it.

    ``native`` selects the native chunk encoder, which runs the plan and
    the chunk encodes inside the native runtime. ``None`` (the default)
    uses it when it can run for this tokenizer (:func:`native_available`:
    the loaded runtime carries it and the route carries the corrected CPU
    engine) and the admission table records the native path for this
    artifact, and the Python path otherwise; ``True`` demands it and
    raises ``RuntimeError`` when the runtime does not carry it, the
    tokenizer is not admitted, or the native path has no recorded
    evidence, and on an input whose route does not start at the corrected
    CPU engine it encodes serially under ``R_CHUNK_ROUTE_NOT_FAST_CPU``;
    ``False`` is the Python path.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    planner = _planner_for(tokenizer)
    chars = len(text)
    requested = _resolve_k(k, chars)
    budget = _check_budget(scan_budget)
    if _select_native(tokenizer, planner, native):
        return _native_chunked_encode(
            tokenizer, planner, text, requested, min_chunk, budget
        )
    if not planner.admitted:
        ids = tokenizer.encode(text, lookup="off").ids
        planner.record(
            chunks=1,
            k=requested,
            chars=chars,
            tokens=len(ids),
            plan=None,
            serial_reason=_R_FAMILY,
            below_threshold=False,
        )
        return ChunkedEncoding(
            ids=ids, chunks=1, k_requested=requested, serial_reason=_R_FAMILY
        )
    if requested <= 1 or chars < _MIN_TEXT_CHARS:
        ids = tokenizer.encode(text, lookup="off").ids
        planner.record(
            chunks=1,
            k=requested,
            chars=chars,
            tokens=len(ids),
            plan=None,
            serial_reason=None,
            below_threshold=True,
        )
        return ChunkedEncoding(ids=ids, chunks=1, k_requested=requested)
    plan = planner.plan(text, requested, min_chunk, budget)
    if len(plan.spans) == 1:
        ids = tokenizer.encode(text, lookup="off").ids
        planner.record(
            chunks=1,
            k=requested,
            chars=chars,
            tokens=len(ids),
            plan=plan,
            serial_reason=_R_NO_CUT,
            below_threshold=False,
        )
        return ChunkedEncoding(
            ids=ids,
            chunks=1,
            k_requested=requested,
            cut_misses=plan.cut_misses,
            literal_rejects=plan.literal_rejects,
            norm_rejects=plan.norm_rejects,
            serial_reason=_R_NO_CUT,
        )
    rows = _encode_rows(tokenizer, [text[low:high] for low, high in plan.spans])
    ids = tuple(chain.from_iterable(rows))
    planner.record(
        chunks=len(rows),
        k=requested,
        chars=chars,
        tokens=len(ids),
        plan=plan,
        serial_reason=None,
        below_threshold=False,
    )
    return ChunkedEncoding(
        ids=ids,
        chunks=len(rows),
        k_requested=requested,
        cuts=plan.cuts,
        cut_misses=plan.cut_misses,
        literal_rejects=plan.literal_rejects,
        norm_rejects=plan.norm_rejects,
    )


def _native_chunked_encode(
    tokenizer: Tokenizer,
    planner: _Planner,
    text: str,
    requested: int,
    min_chunk: int | None,
    budget: int,
) -> ChunkedEncoding:
    """The native path of :func:`encode`."""
    runtime = _native_runtime(tokenizer)
    ids, facts = runtime.chunked_encode(text, requested, min_chunk, budget)
    serial_reason = facts["serial_reason"]
    below = bool(facts["below_threshold"])
    plan = ChunkPlan(
        spans=(),
        k=requested,
        cut_misses=int(facts["cut_misses"]),
        literal_rejects=int(facts["literal_rejects"]),
        norm_rejects=int(facts["norm_rejects"]),
    )
    planner.record(
        chunks=int(facts["chunks"]),
        k=requested,
        chars=len(text),
        tokens=len(ids),
        plan=None if below or serial_reason == _R_FAMILY else plan,
        serial_reason=None if serial_reason is None else str(serial_reason),
        below_threshold=below,
        native=True,
    )
    return ChunkedEncoding(
        ids=tuple(ids),
        chunks=int(facts["chunks"]),
        k_requested=requested,
        cuts=tuple(int(cut) for cut in facts["cuts"]),
        cut_misses=plan.cut_misses,
        literal_rejects=plan.literal_rejects,
        norm_rejects=plan.norm_rejects,
        serial_reason=None if serial_reason is None else str(serial_reason),
        native_facts=dict(facts),
    )
