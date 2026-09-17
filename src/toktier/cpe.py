"""Certified chunk-parallel encode (CPE) for long requests on the CPU route.

A long document is cut at *certified stable boundaries*, the chunks are
encoded as one batch on the loaded tokenizer (the native batch path runs
the rows on its worker pool), and the id rows are concatenated. There is
no overlap, no seam matching and no retry, because a cut is placed only
where the pre-tokenizer of the certified families cannot carry a
pre-token across it: the synchronizing-transition predicate the
session-repair witness already uses (``BpeSyncBoundary`` in the routing
core -- eight class pairs over the frozen O/S/L/N/M table, minus two
carve-outs), extended here by a guard that keeps every cut outside any
added-token literal. The concatenation is therefore the serial encoding
by construction, and a cut that cannot be certified is simply not made:
fewer chunks, never a different id.

Two things this module deliberately is not. It is not a new route: the
chunks run through the tokenizer's own batch path, so the plan, the
per-input guards and the fallback ledger apply to each chunk exactly as
they apply to a whole document (a chunk that trips a guard is routed to
the reference on its own, which localizes that cost to one chunk). And it
is not a native change: the cut search and the concatenation are Python,
so the parallelism available is the batch worker pool of the native
engine -- the Rayon thread count at first use, and at most one worker per
64 KiB of payload.

Admission is the certified repair roster. A tokenizer whose family and
artifact digest have a repair-table entry -- the eleven byte-level BPE
artifacts the shipped sync profile was certified on -- may be chunked;
any other tokenizer is encoded serially and the refusal is counted under
``R_CPE_FAMILY_UNSUPPORTED``. A document in which no certified cut can be
found is encoded serially under ``R_CPE_NO_CUT``. Both counters, and the
per-call facts, are read back from ``Tokenizer.explain()["cpe"]`` once
this module has served the tokenizer, or from :func:`stats`.

Source text in this repository is ASCII, so the characters named in the
carve-outs are written as escapes.
"""

from __future__ import annotations

import json
import re
import threading
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain

from .facade.api import Encoding, Tokenizer
from .policy import ReasonCode
from .repair.registry import family_spec, pclass_table

__all__ = [
    "CHUNK_TABLE",
    "DEFAULT_SCAN_BUDGET",
    "MIN_BYTES_PER_WORKER",
    "MIN_CHARS",
    "SYNC_PAIRS",
    "ChunkPlan",
    "ChunkedEncoding",
    "chunked_encode",
    "default_chunks",
    "is_admitted",
    "plan_chunks",
    "stats",
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
#: for this span first and widened only on a miss.
_QUICK_SCAN = 48

#: Documents shorter than this are never cut, whatever ``k`` says.
_MIN_TEXT_CHARS = 256

_R_NO_CUT = ReasonCode.R_CPE_NO_CUT.value
_R_FAMILY = ReasonCode.R_CPE_FAMILY_UNSUPPORTED.value


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


def _boundary_ok(table: bytes, text: str, position: int) -> bool:
    """Whether a cut between ``text[position - 1]`` and ``text[position]``
    is a certified stable boundary (the predicate of the sync witness)."""
    left = table[ord(text[position - 1])]
    right = table[ord(text[position])]
    if not _SYNC[left][right]:
        return False
    current = text[position]
    # Punctuation absorbs a following line break into its pre-token.
    if left == _O and right == _S and current in "\r\n":
        return False
    # An apostrophe after letters may open a contraction suffix.
    return not (left == _L and right == _O and current == "'")


class _LiteralGuard:
    """Cut positions strictly inside an added-token literal are forbidden.

    Conservative on purpose: every occurrence of every literal counts,
    overlapping ones included, and at each start the longest literal that
    matches there is taken. That is a superset of what any leftmost-longest
    extraction would actually consume, so a cut the guard allows never
    splits a literal the reference would have matched.
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
    serial_reason: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "cuts", tuple(self.cuts))


class _Planner:
    """Per-tokenizer admission, guard and counters."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._table = pclass_table()
        self.family = tokenizer.family
        self.refusal: str | None = None
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
                document = json.loads(loader_face())
                literals = [
                    str(row.get("content", ""))
                    for row in document.get("added_tokens") or []
                    if isinstance(row, Mapping)
                ]
        self.guard = _LiteralGuard(literals)
        self._lock = threading.Lock()
        self._calls = 0
        self._chunked = 0
        self._serial = 0
        self._below_threshold = 0
        self._chunks_total = 0
        self._cut_misses = 0
        self._literal_rejects = 0
        self._fallback_counts: dict[str, int] = {}
        self._last: dict[str, object] | None = None

    @property
    def admitted(self) -> bool:
        return self.refusal is None

    def plan(
        self, text: str, k: int, min_chunk: int | None, scan_budget: int
    ) -> ChunkPlan:
        length = len(text)
        if k <= 1 or length < _MIN_TEXT_CHARS:
            return ChunkPlan(((0, length),), k, 0, 0)
        floor = (
            max(16, min(1024, length // (4 * k))) if min_chunk is None else min_chunk
        )
        table = self._table
        guard = self.guard
        cuts: list[int] = []
        misses = 0
        rejects = 0
        previous = 0
        for index in range(1, k):
            target = (length * index) // k
            if target <= previous:
                misses += 1
                continue
            found = -1
            low = max(previous + 1, target - _QUICK_SCAN)
            forbidden = guard.forbidden(text, low, target)
            position = target
            while position >= low:
                if _boundary_ok(table, text, position):
                    if position in forbidden:
                        rejects += 1
                    else:
                        found = position
                        break
                position -= 1
            if found < 0 and low > previous + 1:
                high = low - 1
                low = max(previous + 1, target - scan_budget)
                if high >= low:
                    forbidden = guard.forbidden(text, low, high)
                    position = high
                    while position >= low:
                        if _boundary_ok(table, text, position):
                            if position in forbidden:
                                rejects += 1
                            else:
                                found = position
                                break
                        position -= 1
            if found < 0 or found - previous < floor:
                misses += 1
                continue
            cuts.append(found)
            previous = found
        bounds = [0, *cuts, length]
        spans = tuple((bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1))
        return ChunkPlan(spans, k, misses, rejects)

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
            self._last = {
                "chars": chars,
                "tokens": tokens,
                "k_requested": k,
                "chunks": chunks,
                "serial_reason": serial_reason,
                "cut_misses": plan.cut_misses if plan is not None else 0,
                "literal_rejects": plan.literal_rejects if plan is not None else 0,
            }

    def report(self) -> dict[str, object]:
        with self._lock:
            return {
                "admitted": self.admitted,
                "refusal": self.refusal,
                "family": self.family,
                "literals": self.guard.count,
                "calls": self._calls,
                "chunked": self._chunked,
                "serial": self._serial,
                "below_threshold": self._below_threshold,
                "chunks_total": self._chunks_total,
                "cut_misses": self._cut_misses,
                "literal_rejects": self._literal_rejects,
                "fallback_counts": dict(sorted(self._fallback_counts.items())),
                "last": dict(self._last) if self._last is not None else None,
            }


_PLANNERS: weakref.WeakKeyDictionary[Tokenizer, _Planner] = weakref.WeakKeyDictionary()
_PLANNERS_LOCK = threading.Lock()


def _planner_for(tokenizer: Tokenizer) -> _Planner:
    with _PLANNERS_LOCK:
        planner = _PLANNERS.get(tokenizer)
        if planner is None:
            planner = _Planner(tokenizer)
            _PLANNERS[tokenizer] = planner
            register = getattr(tokenizer, "_register_report", None)
            if callable(register):
                register("cpe", planner.report)
        return planner


def is_admitted(tokenizer: Tokenizer) -> bool:
    """Whether this tokenizer may be chunked (certified repair roster)."""
    return _planner_for(tokenizer).admitted


def stats(tokenizer: Tokenizer) -> dict[str, object]:
    """The counters this module keeps for ``tokenizer``."""
    return _planner_for(tokenizer).report()


def plan_chunks(
    tokenizer: Tokenizer,
    text: str,
    *,
    k: int | None = None,
    min_chunk: int | None = None,
    scan_budget: int = DEFAULT_SCAN_BUDGET,
) -> ChunkPlan:
    """The cut plan :func:`chunked_encode` would use, without encoding.

    Raises ``ValueError`` for a tokenizer outside the certified roster; the
    encode entry point prefers to fall back, this one prefers to say so.
    """
    planner = _planner_for(tokenizer)
    if not planner.admitted:
        raise ValueError(
            f"{tokenizer.family!r} has no certified chunk-cut rule: {planner.refusal}"
        )
    chunks = _resolve_k(k, len(text))
    return planner.plan(text, chunks, min_chunk, _check_budget(scan_budget))


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


def chunked_encode(
    tokenizer: Tokenizer,
    text: str,
    *,
    k: int | None = None,
    min_chunk: int | None = None,
    scan_budget: int = DEFAULT_SCAN_BUDGET,
) -> ChunkedEncoding:
    """Encode ``text`` in certified chunks; ids equal ``tokenizer.encode``.

    ``k`` is the requested chunk count (``None`` reads :data:`CHUNK_TABLE`
    for the document length); ``min_chunk`` the smallest chunk the plan
    may produce (``None`` selects ``min(1024, len // (4 k))``);
    ``scan_budget`` how far below each target the cut search looks. The
    result is the core stream, exactly as ``encode(text, lookup="off")``
    returns it: special tokens are a read-time template, as for sessions.

    A tokenizer outside the certified roster, and a document in which no
    certified cut exists, are encoded serially on the tokenizer's plain
    route; the result says so in ``serial_reason`` and the tokenizer's
    ``explain()["cpe"]`` block counts it.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    planner = _planner_for(tokenizer)
    chars = len(text)
    requested = _resolve_k(k, chars)
    budget = _check_budget(scan_budget)
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
    )
