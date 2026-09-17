"""Chunk-parallel encode returns the ids of the serial encode.

Two tiers. The offline tier runs on the facade tier's tiny real
byte-level artifact and pins the geometry itself: the boundary
predicate against the shipped routing-core profile, the literal guard,
the chunk table, the admission refusal and the input edge cases that do
not need a certified family. The artifact tier runs when
``TOKTIER_TEST_ARTIFACTS`` names a directory of local artifacts (the
backend tier's convention: one subdirectory per artifact, each holding
the manifest's files) and judges ``chunked_encode`` against a
from-scratch encode by the oracle package on the certified families it
finds there, over three corpus types and three document lengths.
``TOKTIER_TEST_CPE_CORPUS`` may name a directory whose subdirectories
each hold UTF-8 text files (one corpus type per subdirectory); when it
is unset, the three corpora are generated and labelled as such. Both
variables are development inputs, read once at import, and every
artifact is copied into an isolated home before it is loaded.

Source text in this repository is ASCII, so the non-ASCII probe
characters below are written as escapes.
"""

from __future__ import annotations

import functools
import json
import os
import random
import re
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import toktier
from toktier import Config, ReasonCode, Tokenizer, cpe
from toktier.artifacts.manifest import ArtifactManifest
from toktier.artifacts.tables import ARTIFACT_MANIFEST
from toktier.paths import artifact_cache_dir
from toktier.repair.registry import families as roster_families
from toktier.repair.registry import pclass_table

from .facade.conftest import build_rig

REPO_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACTS = os.environ.get("TOKTIER_TEST_ARTIFACTS")
_CORPUS = os.environ.get("TOKTIER_TEST_CPE_CORPUS")

#: Families judged in the artifact tier, in this order, when their
#: artifacts are present; the first three are the ones the shipped chunk
#: table was measured on.
CANDIDATE_FAMILIES = ("qwen3_8b", "llama_3_1_8b", "gpt_oss_120b", "deepseek_v3")
SIZES = (64_000, 256_000, 1_000_000)

_O, _S, _L, _N, _M = 0, 1, 2, 3, 4


# -- corpora ---------------------------------------------------------------

_WORDS = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "a",
    "lazy",
    "dog",
    "while",
    "seven",
    "engineers",
    "debate",
    "whether",
    "tokenization",
    "is",
    "a",
    "solved",
    "problem",
    "and",
    "nobody",
    "agrees",
    "on",
    "anything",
    "except",
    "lunch",
    "caf\u00e9",
    "na\u00efve",
    "r\u00e9sum\u00e9",
    "stra\u00dfe",
    "M\u00fcnchen",
    "\u4e2d\u6587",
    "\u6d4b\u8bd5",
    "\u3042\u3044\u3046",
    "\ud55c\uae00",
    "don't",
    "it's",
    "we're",
    "they'll",
    "o'clock",
    "1990s",
    "42",
    "3.14",
    "1,000,000",
    "e\u0301",
    "\U0001f642",
    "\U0001f469\u200d\U0001f4bb",
]
_PUNCT = (".", ",", ";", ":", "!", "?", " --", "...", "\n", "\n\n", '"', "'", "(", ")")
_CODE_LINES = (
    "def f{n}(x, y=None):",
    "    return x + {n} if y is None else x * y  # note",
    "for i in range({n}):",
    "    items.append({{'k': i, 'v': str(i)}})",
    "class C{n}(Base):",
    '    """Docstring with \u00fcnicode and a URL http://example.com/a?b={n}"""',
    "    value: int = 0x{n:x}",
    "if __name__ == '__main__':",
    "    print(f'{{value!r}} {{x:>{n}}}')",
    "    # TODO(person): don't forget the edge-case at i == {n}",
    "",
    "\t\treturn [a, b, c][{n} % 3]",
)
_LITERALS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|begin_of_text|>",
    "<|eot_id|>",
    "<|start|>",
    "<|end|>",
    "<|message|>",
)


def _prose(rng: random.Random, chars: int) -> str:
    parts: list[str] = []
    length = 0
    while length < chars:
        word = rng.choice(_WORDS)
        piece = (" " if parts and rng.random() < 0.92 else "") + word
        if rng.random() < 0.12:
            piece += rng.choice(_PUNCT)
        parts.append(piece)
        length += len(piece)
    return "".join(parts)[:chars]


def _code(rng: random.Random, chars: int) -> str:
    lines: list[str] = []
    length = 0
    while length < chars:
        line = rng.choice(_CODE_LINES).format(n=rng.randint(0, 99999))
        newline = "\r\n" if rng.random() < 0.1 else "\n"
        lines.append(line + newline)
        length += len(line) + len(newline)
    return "".join(lines)[:chars]


def _transcript(rng: random.Random, chars: int) -> str:
    parts: list[str] = []
    length = 0
    while length < chars:
        role = rng.choice(("user", "assistant", "system", "tool"))
        body = _prose(rng, rng.randint(40, 600))
        literal = rng.choice(_LITERALS) if rng.random() < 0.3 else ""
        turn = f"{literal}{role}\n{body}\n{rng.choice(_LITERALS)}\n"
        if rng.random() < 0.2:
            turn = turn.replace("\n", "\r\n")
        parts.append(turn)
        length += len(turn)
    return "".join(parts)[:chars]


def _generated_corpora(chars: int) -> dict[str, str]:
    rng = random.Random(0xC9E0917)
    return {
        "prose(generated)": _prose(rng, chars),
        "code(generated)": _code(rng, chars),
        "transcript(generated)": _transcript(rng, chars),
    }


def _external_corpora(chars: int) -> dict[str, str]:
    root = Path(_CORPUS or "")
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for sub in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(path for path in sub.iterdir() if path.is_file())
        if not files:
            continue
        text = files[0].read_text(encoding="utf-8", errors="strict")
        if len(text) >= chars:
            out[f"{sub.name}({files[0].name})"] = text[:chars]
    return out


@functools.cache
def corpora(chars: int) -> dict[str, str]:
    """Three or more corpora of ``chars`` characters each (cached)."""
    out = _external_corpora(chars)
    if len(out) < 3:
        out.update(_generated_corpora(chars))
    return out


# -- artifact tier fixtures -------------------------------------------------


def _artifact_source(family: str) -> Path | None:
    if not _ARTIFACTS:
        return None
    root = Path(_ARTIFACTS)
    entry = ArtifactManifest.load(ARTIFACT_MANIFEST).get(family)
    for name in (entry.directory_name, family):
        candidate = root / name
        if (candidate / "tokenizer.json").is_file():
            return candidate
    return None


def available_families() -> list[str]:
    return [name for name in CANDIDATE_FAMILIES if _artifact_source(name)]


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Tokenizer]]:
    """Every available candidate family, loaded once from an isolated home."""
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    handles: dict[str, Tokenizer] = {}
    for family in available_families():
        source = _artifact_source(family)
        assert source is not None
        home = tmp_path_factory.mktemp(f"cpe-{family}")
        config = Config(home=home, offline=True)
        cache = artifact_cache_dir(config)
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, cache / manifest.get(family).directory_name)
        handles[family] = toktier.load(family, config=config, device="cpu")
    yield handles
    for handle in handles.values():
        handle.close()


@functools.cache
def _reference(tokenizer: Tokenizer) -> Callable[[str], list[int]]:
    """From-scratch core-stream encode by the oracle package itself."""
    import tokenizers

    face = tokenizer._backend.materialized_tokenizer_json()
    handle = tokenizers.Tokenizer.from_str(face)

    def encode(text: str) -> list[int]:
        return [int(i) for i in handle.encode(text, add_special_tokens=False).ids]

    return encode


def _family(loaded: dict[str, Tokenizer], family: str) -> Tokenizer:
    if family not in loaded:
        pytest.skip(f"{family}: artifact not available under TOKTIER_TEST_ARTIFACTS")
    return loaded[family]


# -- offline tier: the geometry --------------------------------------------


def test_sync_pairs_match_the_routing_core_profile() -> None:
    """The Python pair set is the one ``BpeSyncBoundary::accepts`` uses."""
    source = REPO_ROOT / "crates" / "toktier-routing-core" / "src" / "lib.rs"
    if not source.is_file():
        pytest.skip("routing core source not present in this checkout")
    text = source.read_text(encoding="utf-8")
    body = text[text.index("fn accepts(&self, previous: char, current: char)") :]
    body = body[: body.index("}\n")]
    pairs = {(int(a), int(b)) for a, b in re.findall(r"\((\d), (\d)\)", body)}
    assert pairs == set(cpe.SYNC_PAIRS)
    assert "'\\r' | '\\n'" in body and "== '\\''" in body


def test_boundary_predicate_examples() -> None:
    table = pclass_table()
    assert table[ord("a")] == _L and table[ord("1")] == _N
    assert table[ord(" ")] == _S and table[ord(".")] == _O
    assert table[ord("\u0301")] == _M

    def ok(text: str, position: int) -> bool:
        return cpe._boundary_ok(table, text, position)

    assert ok("ab 12", 2)  # L -> S
    assert ok("ab 12", 3) is False  # S -> N: a leading space belongs right
    assert ok("ab12", 2)  # L -> N
    assert ok("12ab", 2)  # N -> L
    assert ok("ab.", 2)  # L -> O
    assert ok(".ab", 1) is False  # O -> L is not a sync pair
    assert ok("x.\n", 2) is False  # punctuation absorbs the line break
    assert ok("x. y", 2)  # punctuation before a space is fine
    assert ok("don't", 3) is False  # apostrophe after letters
    assert ok("don'x", 3) is False
    assert ok("12'x", 2)  # N -> O with an apostrophe is not the carve-out
    assert ok("e\u0301 x", 1) is False  # a combining mark never starts a chunk
    assert ok("\u4e2d\u6587", 1) is False  # two letters


def test_literal_guard_is_conservative_over_overlaps() -> None:
    guard = cpe._LiteralGuard(["ab", "bcd", "<|x|>"])
    text = "xabcdy <|x|> ab"
    assert guard.forbidden(text, 0, len(text)) == {2, 3, 4, 8, 9, 10, 11, 14}
    assert guard.forbidden(text, 3, 3) == {3}
    assert guard.forbidden(text, 5, 7) == set()
    assert guard.forbidden(text, 12, 15) == {14}
    assert cpe._LiteralGuard([]).forbidden(text, 0, 15) == set()
    assert cpe._LiteralGuard(["a"]).forbidden(text, 0, 15) == set()


def test_default_chunks_follows_the_table_and_the_worker_floor() -> None:
    assert cpe.default_chunks(0) == 1
    assert cpe.default_chunks(255_999) == 1
    assert cpe.default_chunks(256_000) == 4
    assert cpe.default_chunks(400_000) == 4
    assert cpe.default_chunks(1_000_000) == 16  # 16 started 64 KiB units
    assert cpe.default_chunks(1_000_001) == 16
    assert cpe.default_chunks(2_000_000) == 16
    assert cpe.default_chunks(4_000_000) == 32
    assert cpe.default_chunks(40_000_000) == 32
    with pytest.raises(ValueError):
        cpe.default_chunks(-1)


def test_tiny_family_is_refused_and_encoded_serially(tmp_path: Path) -> None:
    rig = build_rig(tmp_path)
    tokenizer = rig.tokenizer()
    text = "hello world " * 30_000
    assert cpe.is_admitted(tokenizer) is False
    encoded = cpe.chunked_encode(tokenizer, text, k=8)
    assert encoded.serial_reason == ReasonCode.R_CPE_FAMILY_UNSUPPORTED.value
    assert encoded.chunks == 1
    assert list(encoded.ids) == list(tokenizer.encode(text, lookup="off").ids)
    with pytest.raises(ValueError):
        cpe.plan_chunks(tokenizer, text)
    report = tokenizer.explain()["cpe"]
    assert isinstance(report, dict)
    assert report["admitted"] is False
    assert report["fallback_counts"] == {"R_CPE_FAMILY_UNSUPPORTED": 1}
    assert report["calls"] == 1 and report["serial"] == 1
    assert cpe.stats(tokenizer) == report
    assert "cpe" not in tokenizer.explain(summary=True)


def test_argument_validation(tmp_path: Path) -> None:
    rig = build_rig(tmp_path)
    tokenizer = rig.tokenizer()
    with pytest.raises(TypeError):
        cpe.chunked_encode(tokenizer, b"bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cpe.chunked_encode(tokenizer, "x", k=0)
    with pytest.raises(ValueError):
        cpe.chunked_encode(tokenizer, "x", scan_budget=0)
    with pytest.raises(ValueError):
        cpe.chunked_encode(tokenizer, "x", k=True)  # bool is not accepted


# -- artifact tier: identity against the oracle -----------------------------


def _cases() -> list[object]:
    return [
        pytest.param(
            family,
            corpus,
            size,
            marks=[pytest.mark.slow] if size >= 1_000_000 else [],
        )
        for family in available_families()
        for size in SIZES
        for corpus in corpora(size)
    ]


@pytest.mark.parametrize(("family", "corpus", "size"), _cases())
def test_chunked_ids_equal_reference(
    loaded: dict[str, Tokenizer], family: str, corpus: str, size: int
) -> None:
    tokenizer = _family(loaded, family)
    text = corpora(size)[corpus]
    expected = _reference(tokenizer)(text)
    assert list(tokenizer.encode(text, lookup="off").ids) == expected
    for k in (None, 8, 32):
        encoded = cpe.chunked_encode(tokenizer, text, k=k)
        assert list(encoded.ids) == expected, (family, corpus, size, k)
        assert encoded.serial_reason is None
        if k is None:
            assert encoded.k_requested == cpe.default_chunks(size)
        else:
            assert encoded.k_requested == k
            assert encoded.chunks >= 2, (family, corpus, size, k)
    plan = cpe.plan_chunks(tokenizer, text, k=32)
    assert plan.cut_misses == 0, (family, corpus, size)


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_plan_invariants(loaded: dict[str, Tokenizer], family: str) -> None:
    tokenizer = _family(loaded, family)
    table = pclass_table()
    planner = cpe._planner_for(tokenizer)
    text = _generated_corpora(300_000)["transcript(generated)"]
    for k in (2, 5, 16, 64):
        plan = cpe.plan_chunks(tokenizer, text, k=k)
        assert plan.spans[0][0] == 0 and plan.spans[-1][1] == len(text)
        assert all(a < b for a, b in plan.spans)
        assert all(
            plan.spans[i][1] == plan.spans[i + 1][0] for i in range(len(plan.spans) - 1)
        )
        assert len(plan.spans) <= k
        floor = max(16, min(1024, len(text) // (4 * k)))
        assert all(b - a >= floor for a, b in plan.spans[:-1])
        for cut in plan.cuts:
            assert cpe._boundary_ok(table, text, cut)
            assert cut not in planner.guard.forbidden(text, cut, cut)
    tight = cpe.plan_chunks(tokenizer, text, k=8, min_chunk=100_000)
    assert len(tight.spans) <= 3 and tight.cut_misses >= 5


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_cut_never_lands_inside_an_added_token_literal(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    """Negative control: every target position is placed inside a literal."""
    tokenizer = _family(loaded, family)
    planner = cpe._planner_for(tokenizer)
    face = json.loads(tokenizer._backend.materialized_tokenizer_json())
    literals = sorted({str(row["content"]) for row in face["added_tokens"]}, key=len)
    literal = literals[-1]
    rng = random.Random(7)
    k = 16
    base = _prose(rng, 400_000)
    # Rebuild the text so that literal occurrences straddle every target
    # position, then verify no cut falls inside any occurrence.
    pieces: list[str] = []
    previous = 0
    occurrences: list[tuple[int, int]] = []
    for index in range(1, k):
        target = (len(base) * index) // k
        start = target - len(literal) // 2
        pieces.append(base[previous:start])
        pieces.append(literal)
        previous = start
    pieces.append(base[previous:])
    text = "".join(pieces)
    position = 0
    while True:
        position = text.find(literal, position)
        if position < 0:
            break
        occurrences.append((position, position + len(literal)))
        position += 1
    assert len(occurrences) >= k - 1
    plan = cpe.plan_chunks(tokenizer, text, k=k)
    for cut in plan.cuts:
        assert not any(a < cut < b for a, b in occurrences), (family, cut)
    assert plan.literal_rejects >= 1
    encoded = cpe.chunked_encode(tokenizer, text, k=k)
    assert encoded.chunks >= 2
    assert list(encoded.ids) == _reference(tokenizer)(text)
    # A literal exactly at a target position is forbidden strictly inside
    # only; the two ends remain legal cut candidates.
    tight = literal.join(["ab 12 ", " cd 34 "] * 4)
    forbidden = planner.guard.forbidden(tight, 0, len(tight))
    for start, end in (
        (m.start(), m.end()) for m in re.finditer(re.escape(literal), tight)
    ):
        assert start not in forbidden and end not in forbidden
        assert all(p in forbidden for p in range(start + 1, end))


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_no_certified_cut_falls_back_to_serial(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    tokenizer = _family(loaded, family)
    counts = cpe.stats(tokenizer)["fallback_counts"]
    assert isinstance(counts, dict)
    before = counts.get("R_CPE_NO_CUT", 0)
    # Letters only: no synchronizing transition exists anywhere.
    text = "\u4e2d\u6587\u6d4b\u8bd5" * 10_000
    encoded = cpe.chunked_encode(tokenizer, text, k=4, scan_budget=512)
    assert encoded.serial_reason == ReasonCode.R_CPE_NO_CUT.value
    assert encoded.chunks == 1 and encoded.cut_misses == 3
    assert list(encoded.ids) == list(tokenizer.encode(text, lookup="off").ids)
    report = tokenizer.explain()["cpe"]
    assert isinstance(report, dict)
    assert report["fallback_counts"]["R_CPE_NO_CUT"] == before + 1
    assert report["last"] == {
        "chars": len(text),
        "tokens": len(encoded.ids),
        "k_requested": 4,
        "chunks": 1,
        "serial_reason": "R_CPE_NO_CUT",
        "cut_misses": 3,
        "literal_rejects": 0,
    }


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_edge_inputs(loaded: dict[str, Tokenizer], family: str) -> None:
    tokenizer = _family(loaded, family)
    reference = _reference(tokenizer)
    for text in ("", " ", "a", "\n", "ab 12", " " * 300_000, "a" * 300):
        encoded = cpe.chunked_encode(tokenizer, text, k=8)
        assert list(encoded.ids) == reference(text), repr(text[:8])
        assert list(encoded.ids) == list(tokenizer.encode(text, lookup="off").ids)
        assert encoded.chunks == 1
    below = cpe.chunked_encode(tokenizer, "ab 12 " * 20, k=8)
    assert below.chunks == 1 and below.serial_reason is None
    below_count = cpe.stats(tokenizer)["below_threshold"]
    assert isinstance(below_count, int) and below_count >= 1
    # A lone surrogate is not UTF-8; both routes refuse it the same way.
    bad = ("ab 12 " * 60_000) + "\ud800" + ("cd 34 " * 60_000)
    with pytest.raises(UnicodeEncodeError):
        tokenizer.encode(bad, lookup="off")
    with pytest.raises(UnicodeEncodeError):
        cpe.chunked_encode(tokenizer, bad, k=8)


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_explain_carries_the_cpe_block(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    tokenizer = _family(loaded, family)
    text = _generated_corpora(300_000)["prose(generated)"]
    cpe.chunked_encode(tokenizer, text, k=4)
    report = tokenizer.explain()
    block = report["cpe"]
    assert isinstance(block, dict)
    assert block["admitted"] is True and block["refusal"] is None
    assert block["family"] == family
    assert block["literals"] >= 1
    assert block["chunked"] >= 1 and block["chunks_total"] >= 2
    assert set(block) >= {
        "calls",
        "chunked",
        "serial",
        "below_threshold",
        "chunks_total",
        "cut_misses",
        "literal_rejects",
        "fallback_counts",
        "last",
    }
    ledger = report["fallback_counts"]
    assert isinstance(ledger, dict)
    assert "R_CPE_NO_CUT" not in ledger


def test_roster_families_are_the_admitted_set() -> None:
    names = {item.family for item in roster_families()}
    assert set(CANDIDATE_FAMILIES) <= names
    assert "kimi_k3" not in names and "bert_uncased" not in names
