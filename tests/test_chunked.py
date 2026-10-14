"""Chunk-parallel encode returns the ids of the serial encode.

Two tiers. The offline tier runs on the facade tier's tiny real
byte-level artifact and pins the geometry itself: the boundary
predicate against the shipped routing-core profile, the literal guard,
the chunk table, the admission refusal and the input edge cases that do
not need a certified family. The artifact tier runs when
``TOKTIER_TEST_ARTIFACTS`` names a directory of local artifacts (the
backend tier's convention: one subdirectory per artifact, each holding
the manifest's files) and judges ``chunked.encode`` against a
from-scratch encode by the oracle package on the certified families it
finds there, over three corpus types and three document lengths.
``TOKTIER_TEST_CHUNKED_CORPUS`` may name a directory whose subdirectories
each hold UTF-8 text files (one corpus type per subdirectory); when it
is unset, the three corpora are generated and labelled as such.
``TOKTIER_TEST_CHUNKED_ADVERSARIAL`` scales the generated normalization
corpus (strings per family; default 400). All three variables are
development inputs, read once at import, and every artifact is copied
into an isolated home before it is loaded.

Source text in this repository is ASCII, so the non-ASCII probe
characters below are written as escapes.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import toktier
from toktier import Config, ReasonCode, Tokenizer, chunked
from toktier._jcs import canonical_json
from toktier.artifacts.manifest import ArtifactManifest
from toktier.artifacts.tables import ARTIFACT_MANIFEST
from toktier.errors import RegistryInvalid
from toktier.paths import artifact_cache_dir
from toktier.repair.registry import families as roster_families
from toktier.repair.registry import pclass_table
from toktier.routing.tables import CHUNKED_ADMISSION

from .facade.conftest import build_rig

REPO_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACTS = os.environ.get("TOKTIER_TEST_ARTIFACTS")
_CORPUS = os.environ.get("TOKTIER_TEST_CHUNKED_CORPUS")
_ADVERSARIAL = int(os.environ.get("TOKTIER_TEST_CHUNKED_ADVERSARIAL", "400"))

def _admission_table() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(CHUNKED_ADMISSION.read_text(encoding="utf-8"))
    return document


#: Every row of the shipped admission table (family -> row), and the
#: certified subset: the families the artifact tier judges as admitted.
#: A row with one evidence level recorded is refused at run time, and the
#: tier checks that refusal instead.
TABLE_ROWS: dict[str, dict[str, Any]] = {
    str(row["family"]): row for row in _admission_table()["rows"]
}
TABLE_FAMILIES = tuple(TABLE_ROWS)
CANDIDATE_FAMILIES = tuple(
    family for family, row in TABLE_ROWS.items() if row["status"] == "certified"
)
PARTIAL_FAMILIES = tuple(
    family for family, row in TABLE_ROWS.items() if row["status"] != "certified"
)
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


# -- adversarial normalization corpus ----------------------------------------

# The generator lives with the tools so that the campaign tool and these
# tests judge one and the same corpus (tools/chunked_adversarial_corpus.py).
sys.path.insert(0, str(REPO_ROOT / "tools"))
from chunked_adversarial_corpus import adversarial  # noqa: E402

_adversarial = adversarial

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


def available_families(candidates: Sequence[str] = CANDIDATE_FAMILIES) -> list[str]:
    return [name for name in candidates if _artifact_source(name)]


def load_isolated(
    family: str,
    tmp_path_factory: pytest.TempPathFactory,
    *,
    loader: Callable[..., Tokenizer] | None = None,
    **options: Any,
) -> Tokenizer:
    """Load ``family`` from a copy of its local artifact in an isolated home.

    ``loader`` is ``toktier.load`` unless a test that patches it hands the
    original in.
    """
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    source = _artifact_source(family)
    assert source is not None
    home = tmp_path_factory.mktemp(f"chunked-{family}")
    config = Config(home=home, offline=True)
    cache = artifact_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, cache / manifest.get(family).directory_name)
    load = toktier.load if loader is None else loader
    return load(family, config=config, device="cpu", **options)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Tokenizer]]:
    """Every available candidate family, loaded once from an isolated home."""
    handles: dict[str, Tokenizer] = {}
    for family in available_families():
        handles[family] = load_isolated(family, tmp_path_factory)
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
    assert pairs == set(chunked.SYNC_PAIRS)
    assert "'\\r' | '\\n'" in body and "== '\\''" in body


def test_boundary_predicate_examples() -> None:
    table = pclass_table()
    assert table[ord("a")] == _L and table[ord("1")] == _N
    assert table[ord(" ")] == _S and table[ord(".")] == _O
    assert table[ord("\u0301")] == _M

    def ok(text: str, position: int) -> bool:
        return chunked._boundary_ok(table, text, position)

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
    guard = chunked._LiteralGuard(["ab", "bcd", "<|x|>"])
    text = "xabcdy <|x|> ab"
    assert guard.forbidden(text, 0, len(text)) == {2, 3, 4, 8, 9, 10, 11, 14}
    assert guard.forbidden(text, 3, 3) == {3}
    assert guard.forbidden(text, 5, 7) == set()
    assert guard.forbidden(text, 12, 15) == {14}
    assert chunked._LiteralGuard([]).forbidden(text, 0, 15) == set()
    assert chunked._LiteralGuard(["a"]).forbidden(text, 0, 15) == set()


def test_literal_guard_touch_reports_any_intersecting_occurrence() -> None:
    guard = chunked._LiteralGuard(["ab", "bcd", "<|x|>", "z"])
    text = "xabcdy <|x|> ab z"
    assert guard.touches(text, 0, 1) is False  # before "ab"
    assert guard.touches(text, 1, 2)  # "ab" starts at 1
    assert guard.touches(text, 5, 7) is False  # "y " between occurrences
    assert guard.touches(text, 5, 8)  # reaches into "<|x|>"
    assert guard.touches(text, 12, 13) is False  # the space after "<|x|>"
    assert guard.touches(text, 12, 14)  # reaches the "ab" at 13
    assert guard.touches(text, 16, 17)  # the single-character literal
    assert chunked._LiteralGuard([]).touches(text, 0, len(text)) is False


def test_normalization_stability_predicate() -> None:
    """The condition NFC families add to the input predicate."""
    from tokenizers.normalizers import NFC

    table = pclass_table()
    rule = chunked._NfcRule(
        table, chunked._LiteralGuard(["<|x|>"]), NFC().normalize_str
    )
    stable = rule.accepts

    # release-candidate review, finding 1: letter / punctuation as written,
    # mark / punctuation once normalized (U+0958 -> U+0915 U+093C).
    text = "a" * 127 + "\u0958" + "/k" + " " * 126
    assert chunked._boundary_ok(table, text, 128)
    assert stable(text, 128) is False
    # Three stable characters around the cut need no window: ASCII, and
    # CJK ideographs with full-width punctuation alike.
    assert stable("ab 12", 2) and stable("x\u0958b /k", 3)
    assert stable("\u5b57\u5b57\uff0c\u5b57", 2)
    # Jamo compose across the cut: not a cut of the normalized text.
    assert stable("12\u1100\u1161" + "3", 3) is False
    # A digit before a vowel jamo: nothing composes, (N, L) holds.
    assert stable("12\u1161" + "3", 2)
    # The character left of the cut is judged on its normalized self: a
    # composing sequence becomes the letter (L, S); a mark that composes
    # with nothing stays a mark (M, S).
    assert stable("caf\u00e9 x", 4)
    assert stable("cafe\u0301 x", 5)
    assert stable("cafb\u0301 x", 5) is False
    # A literal touching the anchored window: the reference normalizes the
    # segment after the literal on its own, so the window is not judged
    # (U+0338 composes with the literal's closing bracket into U+226F,
    # which would pass as punctuation / space).
    assert stable("x>\u00e9 1", 3)
    assert stable("ab>\u0338 1", 4)
    assert stable("<|x|>\u0338 1", 6) is False
    # No stable split point within reach (vowel jamo compose after a
    # consonant jamo, so they never anchor): the candidate is given up;
    # the start and the end of the text anchor.
    assert stable("\u1161" * 300 + " a", 300) is False
    assert stable("\u1161" * 30 + " a", 30)
    assert stable("ab " + "\u1161" * 300, 2) is False
    assert stable("ab " + "\u1161" * 100, 2)
    # A mark right of the cut is refused by the class predicate.
    assert stable("ab\u0301 c", 2) is False
    # The stable-split-point test itself.
    for char in "a \u5b57\u00e9\u0915\u3042\u0627\u05d0\u1100\uac00":
        assert rule.stable(char), hex(ord(char))
    for char in "\u0301\u093c\u0958\u1161\u11a8\u0b3e\u3099\u0344\u2126":
        assert not rule.stable(char), hex(ord(char))


def test_stable_split_point_probes_agree_with_the_reference_crate() -> None:
    """The facts the stable-split-point test rests on, re-checked against
    the installed reference crate and the frozen property table: every
    second element of a canonical composition is a mark or a Hangul
    vowel/trailing jamo; the two reorder probes flag no NFC-invariant
    non-mark; the Hangul probes flag exactly the vowel and trailing
    jamo."""
    from tokenizers.normalizers import NFC, NFD

    nfc = NFC().normalize_str
    nfd = NFD().normalize_str
    table = pclass_table()
    before, after = chunked._PROBE_BEFORE, chunked._PROBE_AFTER
    jamo_l, syllable = chunked._PROBE_JAMO_L, chunked._PROBE_SYLLABLE
    vowels = {chr(c) for c in range(0x1161, 0x1176)}
    trailing = {chr(c) for c in range(0x11A8, 0x11C3)}
    seconds: set[str] = set()
    reacting_inert: list[int] = []
    after_l: set[str] = set()
    after_lv: set[str] = set()
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        char = chr(cp)
        decomposed = nfd(char)
        if len(decomposed) >= 2 and nfc(decomposed) == char:
            seconds.update(decomposed[1:])
        normalized = nfc(char)
        if (
            (
                nfc(before + char) != before + normalized
                or nfc(char + after) != normalized + after
            )
            and normalized == char
            and table[cp] != _M
        ):
            reacting_inert.append(cp)
        if nfc(jamo_l + char) != jamo_l + normalized:
            after_l.add(char)
        if nfc(syllable + char) != syllable + normalized:
            after_lv.add(char)
    assert all(table[ord(c)] == _M or c in vowels | trailing for c in seconds)
    assert reacting_inert == []
    assert after_l == vowels and after_lv == trailing


def test_normalizer_mode_admits_none_identity_and_nfc_only() -> None:
    assert chunked._normalizer_mode({}) == ("none", None)
    assert chunked._normalizer_mode({"normalizer": None}) == ("none", None)
    assert chunked._normalizer_mode(
        {"normalizer": {"type": "Sequence", "normalizers": []}}
    ) == ("none", None)
    assert chunked._normalizer_mode({"normalizer": {"type": "NFC"}}) == ("nfc", None)
    assert chunked._normalizer_mode(
        {
            "normalizer": {"type": "NFC"},
            "added_tokens": [{"content": "x", "normalized": True}],
        }
    ) == (None, "nfc_with_normalized_added_tokens")
    assert chunked._normalizer_mode({"normalizer": {"type": "NFKC"}}) == (
        None,
        "normalizer_not_supported:NFKC",
    )
    assert chunked._normalizer_mode(
        {"normalizer": {"type": "Sequence", "normalizers": [{"type": "NFC"}]}}
    ) == (None, "normalizer_not_supported:Sequence")
    assert chunked._normalizer_mode({"normalizer": "NFC"}) == (
        None,
        "normalizer_not_supported",
    )


_LITERAL_FLAG_REFUSAL = "added_token_strip_or_single_word"


def test_strip_and_single_word_literals_are_not_admitted() -> None:
    """An added token with ``lstrip``, ``rstrip`` or ``single_word`` set is
    matched by the reference together with its neighbourhood, which the
    cut rule does not reason about: the face is refused whatever its
    normalizer. The normalizer step keeps its own refusals first, and a
    face whose tokens set all three flags false is admitted as before."""
    normalizers: list[object] = [
        None,
        {"type": "Sequence", "normalizers": []},
        {"type": "NFC"},
    ]
    for flag in ("lstrip", "rstrip", "single_word"):
        for truthy in (True, 1, "yes"):
            row = {"content": "<|x|>", "normalized": False, flag: truthy}
            for normalizer in normalizers:
                face = {
                    "normalizer": normalizer,
                    "added_tokens": [{"content": "<|y|>", "normalized": False}, row],
                }
                assert chunked._normalizer_mode(face) == (None, _LITERAL_FLAG_REFUSAL)
    plain = {
        "content": "<|x|>",
        "normalized": False,
        "lstrip": False,
        "rstrip": False,
        "single_word": False,
    }
    assert chunked._normalizer_mode({"added_tokens": [plain]}) == ("none", None)
    assert chunked._normalizer_mode(
        {"normalizer": {"type": "NFC"}, "added_tokens": [plain]}
    ) == ("nfc", None)
    flagged = {"content": "<|x|>", "rstrip": True}
    assert chunked._normalizer_mode(
        {"normalizer": {"type": "NFKC"}, "added_tokens": [flagged]}
    ) == (None, "normalizer_not_supported:NFKC")
    normalized = {**flagged, "normalized": True}
    assert chunked._normalizer_mode(
        {"normalizer": {"type": "NFC"}, "added_tokens": [normalized]}
    ) == (None, "nfc_with_normalized_added_tokens")


def test_strip_and_single_word_flags_move_ids_at_a_certified_cut() -> None:
    """Why the flags are refused: on a byte-level BPE (bytes as the
    vocabulary, no merges) a cut that the sync predicate accepts and the
    literal guard allows splits the encoding once ``rstrip`` or
    ``single_word`` is set, and does not without them. ``rstrip`` takes
    the space after the literal into the match; ``single_word`` stops
    matching a literal that a word character touches."""
    from tokenizers import AddedToken, models, pre_tokenizers
    from tokenizers import Tokenizer as RawTokenizer

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())

    def build(**flags: bool) -> RawTokenizer:
        raw = RawTokenizer(
            models.BPE(vocab={char: i for i, char in enumerate(alphabet)}, merges=[])
        )
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.add_special_tokens(
            [AddedToken("<|x|>", normalized=False, special=True, **flags)]
        )
        return raw

    def ids(raw: RawTokenizer, text: str) -> list[int]:
        return list(raw.encode(text, add_special_tokens=False).ids)

    table = pclass_table()
    guard = chunked._LiteralGuard(["<|x|>"])
    for flag, (left, right) in {
        "rstrip": ("<|x|>", " hello"),
        "single_word": ("foo", "<|x|> bar"),
    }.items():
        text, position = left + right, len(left)
        assert chunked._boundary_ok(table, text, position)
        assert position not in guard.forbidden(text, position, position)
        plain, flagged = build(), build(**{flag: True})
        assert ids(plain, text) == ids(plain, left) + ids(plain, right)
        assert ids(flagged, text) != ids(flagged, left) + ids(flagged, right), flag


def test_default_chunks_follows_the_table_and_the_worker_floor() -> None:
    assert chunked.default_chunks(0) == 1
    assert chunked.default_chunks(255_999) == 1
    assert chunked.default_chunks(256_000) == 4
    assert chunked.default_chunks(400_000) == 4
    assert chunked.default_chunks(1_000_000) == 16  # 16 started 64 KiB units
    assert chunked.default_chunks(1_000_001) == 16
    assert chunked.default_chunks(2_000_000) == 16
    assert chunked.default_chunks(4_000_000) == 32
    assert chunked.default_chunks(40_000_000) == 32
    with pytest.raises(ValueError):
        chunked.default_chunks(-1)


def test_tiny_family_is_refused_and_encoded_serially(tmp_path: Path) -> None:
    rig = build_rig(tmp_path)
    tokenizer = rig.tokenizer()
    text = "hello world " * 30_000
    assert chunked.is_admitted(tokenizer) is False
    encoded = chunked.encode(tokenizer, text, k=8)
    assert encoded.serial_reason == ReasonCode.R_CHUNK_FAMILY_UNSUPPORTED.value
    assert encoded.chunks == 1
    assert list(encoded.ids) == list(tokenizer.encode(text, lookup="off").ids)
    with pytest.raises(ValueError):
        chunked.plan(tokenizer, text)
    report = tokenizer.explain()["chunked"]
    assert isinstance(report, dict)
    assert report["admitted"] is False
    assert report["fallback_counts"] == {"R_CHUNK_FAMILY_UNSUPPORTED": 1}
    assert report["calls"] == 1 and report["serial"] == 1
    assert chunked.stats(tokenizer) == report
    assert "chunked" not in tokenizer.explain(summary=True)


def test_argument_validation(tmp_path: Path) -> None:
    rig = build_rig(tmp_path)
    tokenizer = rig.tokenizer()
    with pytest.raises(TypeError):
        chunked.encode(tokenizer, b"bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        chunked.encode(tokenizer, "x", k=0)
    with pytest.raises(ValueError):
        chunked.encode(tokenizer, "x", scan_budget=0)
    with pytest.raises(ValueError):
        chunked.encode(tokenizer, "x", k=True)  # bool is not accepted


def test_native_demand_on_a_refused_tokenizer_raises(tmp_path: Path) -> None:
    """``native=True`` is a demand: a tokenizer outside the admitted set
    raises instead of falling back, while ``native=None`` keeps the
    serial fallback and the merged admission facts say which is which."""
    rig = build_rig(tmp_path)
    tokenizer = rig.tokenizer()
    text = "ab 12 " * 100
    if not chunked.native_available(tokenizer):
        pytest.skip("the tiny family's route has no native runtime here")
    with pytest.raises(RuntimeError, match="not admitted"):
        chunked.encode(tokenizer, text, k=2, native=True)
    with pytest.raises(ValueError, match="no certified chunk-cut rule"):
        chunked.plan(tokenizer, text, k=2, native=True)  # plan prefers to say so
    encoded = chunked.encode(tokenizer, text, k=2)
    assert encoded.serial_reason == ReasonCode.R_CHUNK_FAMILY_UNSUPPORTED.value
    assert encoded.native_facts is None
    facts = chunked.admission(tokenizer)
    assert facts["admitted"] is False
    assert facts["refusal"] == "artifact_not_in_certified_repair_roster"
    assert facts["native_available"] is True and facts["native_by_default"] is False
    assert facts["evidence_id"] is None and facts["evidence_paths"] == []
    assert facts["admission_gate"] == "evidence_table"
    # The tiny family's route has no corrected CPU engine, so the native
    # planner cannot be described; the merged facts say so instead of raising.
    assert facts["native"] is None
    assert "no corrected CPU engine" in str(facts["native_unavailable_reason"])


# -- offline tier: the admission table --------------------------------------


def _roster_digest(family: str) -> str:
    return next(
        item.artifact_sha256 for item in roster_families() if item.family == family
    )


def _stand_in(family: str, digest: str, face: dict[str, Any]) -> Any:
    """What ``_Planner`` reads from a tokenizer: family, artifact digest and
    the loader face, without loading an artifact."""
    backend = SimpleNamespace(materialized_tokenizer_json=lambda: json.dumps(face))
    return SimpleNamespace(family=family, _artifact_sha256=digest, _backend=backend)


_NFC_FACE: dict[str, Any] = {
    "normalizer": {"type": "NFC"},
    "added_tokens": [{"content": "<|x|>", "normalized": False}],
}
_NFC_FACE_SHA256 = hashlib.sha256(json.dumps(_NFC_FACE).encode("utf-8")).hexdigest()


def _bind_face(document: dict[str, Any], family: str = "qwen3_8b") -> None:
    """Make the table copy's row record the stand-in face as the judged one."""
    for row in document["rows"]:
        if row["family"] == family:
            row["loader_face_sha256"] = _NFC_FACE_SHA256


@pytest.fixture
def admission_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Callable[..., dict[str, Any]]]:
    """Point the module at an edited copy of the shipped admission table.

    The returned installer takes a mutation of the table document and
    rewrites the root digest unless ``redigest=False`` asks for a stale
    one; the module's cached rows are dropped around each installation.
    """
    shipped: dict[str, Any] = json.loads(CHUNKED_ADMISSION.read_text(encoding="utf-8"))

    def install(
        mutate: Callable[[dict[str, Any]], None] | None = None, *, redigest: bool = True
    ) -> dict[str, Any]:
        document = copy.deepcopy(shipped)
        if mutate is not None:
            mutate(document)
        if redigest:
            body = {k: v for k, v in document.items() if k != "root_digest"}
            digest = hashlib.sha256(chunked._ADMISSION_DOMAIN + canonical_json(body))
            document = {**body, "root_digest": "sha256:" + digest.hexdigest()}
        path = tmp_path / "chunked_admission.v1.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        monkeypatch.setattr(chunked, "_ADMISSION_TABLE", path)
        chunked._admission_rows.cache_clear()
        return document

    yield install
    chunked._admission_rows.cache_clear()


@pytest.fixture
def stand_in_table(
    admission_table: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """The shipped table with qwen3_8b's row bound to the stand-in face, so
    that a stand-in planner is judged on everything but the face."""
    admission_table(_bind_face)
    return admission_table


def test_admission_table_rows_are_roster_artifacts_with_the_python_path() -> None:
    rows = chunked._admission_rows()
    roster = {(item.family, item.artifact_sha256) for item in roster_families()}
    assert rows and set(rows) <= roster
    assert CANDIDATE_FAMILIES, "the shipped table certifies no family"
    for (family, _digest), row in rows.items():
        assert row["evidence_id"] == "ev-chunked-cuts-v1"
        paths = row["paths"]
        assert isinstance(paths, list)
        if row["status"] == "certified":
            assert "python" in paths and family in CANDIDATE_FAMILIES
        else:
            assert paths == [] and family in PARTIAL_FAMILIES


def test_a_row_with_one_evidence_level_is_refused_by_name(
    admission_table: Callable[..., dict[str, Any]],
) -> None:
    """A row that is not certified (one level of evidence recorded) is
    refused like an absent row, with the status in the refusal."""
    digest = _roster_digest("qwen3_8b")

    def one_level(document: dict[str, Any]) -> None:
        for row in document["rows"]:
            if row["family"] == "qwen3_8b":
                row["status"] = "adversarial_only"
                row["paths"] = []
                row["evidence"] = {"python": ["adversarial"], "native": []}

    admission_table(one_level)
    planner = chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE))
    assert planner.admitted is False
    assert planner.refusal == "chunked_evidence_incomplete:adversarial_only"
    assert planner.evidence_status == "adversarial_only"
    assert planner.evidence_id == "ev-chunked-cuts-v1" and planner.evidence_paths == ()
    stand_in = _stand_in("qwen3_8b", digest, _NFC_FACE)
    lifted = chunked._Planner(stand_in, evidence_gate=False)
    assert lifted.admitted is True and lifted.evidence_status == "adversarial_only"


def test_live_loader_face_must_be_the_one_the_evidence_saw(
    admission_table: Callable[..., dict[str, Any]],
) -> None:
    """A certified row names the loader face its evidence was taken on; a
    tokenizer whose live face hashes differently (another loader version
    materializing another pre-tokenizer) is refused by name, a row without
    a recorded face is not held to one, and the lift records the live face."""
    digest = _roster_digest("qwen3_8b")
    admission_table()  # the shipped row: the campaign's face, not the stand-in's
    shipped_face = TABLE_ROWS["qwen3_8b"]["loader_face_sha256"]
    planner = chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE))
    if shipped_face is None:
        assert planner.admitted is True
    else:
        assert planner.admitted is False
        assert planner.refusal == (
            f"loader_face_differs_from_chunked_evidence:{_NFC_FACE_SHA256[:12]}"
        )
    assert planner.loader_face_sha256 == _NFC_FACE_SHA256
    assert planner.admission_facts()["loader_face_sha256"] == _NFC_FACE_SHA256
    stand_in = _stand_in("qwen3_8b", digest, _NFC_FACE)
    assert chunked._Planner(stand_in, evidence_gate=False).admitted is True

    def unrecorded(document: dict[str, Any]) -> None:
        for row in document["rows"]:
            if row["family"] == "qwen3_8b":
                row["loader_face_sha256"] = None

    admission_table(unrecorded)
    assert chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE)).admitted is True
    admission_table(_bind_face)
    assert chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE)).admitted is True


def test_normalized_added_token_refuses_an_nfc_roster_artifact(
    stand_in_table: Callable[..., dict[str, Any]],
) -> None:
    """A roster artifact whose NFC face carries a ``normalized: true`` added
    token is refused on the normalizer step, before and regardless of the
    evidence step; the same face with the token matched on the input text
    is admitted with its table row."""
    digest = _roster_digest("qwen3_8b")
    face = copy.deepcopy(_NFC_FACE)
    face["added_tokens"][0]["normalized"] = True
    planner = chunked._Planner(_stand_in("qwen3_8b", digest, face))
    assert planner.admitted is False
    assert planner.refusal == "nfc_with_normalized_added_tokens"
    lifted = chunked._Planner(_stand_in("qwen3_8b", digest, face), evidence_gate=False)
    assert lifted.refusal == "nfc_with_normalized_added_tokens"
    planner = chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE))
    assert planner.admitted is True and planner.normalizer == "nfc"
    assert planner.evidence_id == "ev-chunked-cuts-v1"
    assert "python" in planner.evidence_paths
    assert planner.admission_facts()["admission_gate"] == "evidence_table"


@pytest.mark.parametrize("flag", ["lstrip", "rstrip", "single_word"])
def test_flagged_added_token_refuses_a_roster_artifact(
    stand_in_table: Callable[..., dict[str, Any]], flag: str
) -> None:
    """A roster artifact whose face carries an added token with a strip or
    single-word flag is refused on the shape step, before the table step
    and also with the evidence gate lifted (the lift covers the table
    step only)."""
    digest = _roster_digest("qwen3_8b")
    face = copy.deepcopy(_NFC_FACE)
    face["added_tokens"][0][flag] = True
    planner = chunked._Planner(_stand_in("qwen3_8b", digest, face))
    assert planner.admitted is False
    assert planner.refusal == _LITERAL_FLAG_REFUSAL
    assert planner.normalizer is None
    assert planner.admission_facts()["refusal"] == _LITERAL_FLAG_REFUSAL
    lifted = chunked._Planner(_stand_in("qwen3_8b", digest, face), evidence_gate=False)
    assert lifted.admitted is False and lifted.refusal == _LITERAL_FLAG_REFUSAL
    face["added_tokens"][0][flag] = False
    planner = chunked._Planner(_stand_in("qwen3_8b", digest, face))
    assert planner.refusal != _LITERAL_FLAG_REFUSAL


def test_roster_artifact_without_a_table_row_is_refused(
    admission_table: Callable[..., dict[str, Any]],
) -> None:
    digest = _roster_digest("qwen3_8b")

    def drop_qwen(document: dict[str, Any]) -> None:
        document["rows"] = [r for r in document["rows"] if r["family"] != "qwen3_8b"]

    admission_table(drop_qwen)
    planner = chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE))
    assert planner.admitted is False
    assert planner.refusal == "artifact_without_chunked_evidence"
    assert planner.evidence_id is None and planner.evidence_paths == ()
    assert planner.native_evidence is False
    # Lifting the evidence gate admits the artifact for collecting that
    # evidence, and the report says so; the roster step is not lifted.
    stand_in = _stand_in("qwen3_8b", digest, _NFC_FACE)
    lifted = chunked._Planner(stand_in, evidence_gate=False)
    assert lifted.admitted is True and lifted.native_evidence is True
    gate = lifted.admission_facts()["admission_gate"]
    assert gate == "lifted_for_evidence_collection"
    foreign = _stand_in("qwen3_8b", "0" * 64, _NFC_FACE)
    assert (
        chunked._Planner(foreign, evidence_gate=False).refusal
        == "artifact_not_in_certified_repair_roster"
    )
    # A row that records only the Python path leaves the native path out.
    admission_table(_bind_face)
    planner = chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE))
    assert planner.admitted is True
    assert planner.native_evidence == ("native" in planner.evidence_paths)


def test_tampered_admission_table_fails_closed(
    admission_table: Callable[..., dict[str, Any]],
) -> None:
    digest = _roster_digest("qwen3_8b")

    def grant_native(document: dict[str, Any]) -> None:
        _bind_face(document)
        for row in document["rows"]:
            row["paths"] = ["python", "native"]

    admission_table(grant_native, redigest=False)
    with pytest.raises(RegistryInvalid, match="root digest"):
        chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE))
    admission_table(lambda document: document.update(schema="toktier.other.v1"))
    with pytest.raises(RegistryInvalid, match="unknown schema"):
        chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE))
    admission_table(lambda document: document["rows"].append(document["rows"][0]))
    with pytest.raises(RegistryInvalid, match="repeats"):
        chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE))
    # The same edit with a recomputed digest is read (the digest guards
    # the file against tampering, not against regeneration).
    admission_table(grant_native)
    planner = chunked._Planner(_stand_in("qwen3_8b", digest, _NFC_FACE))
    assert planner.native_evidence is True


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
        encoded = chunked.encode(tokenizer, text, k=k)
        assert list(encoded.ids) == expected, (family, corpus, size, k)
        assert encoded.serial_reason is None
        if k is None:
            assert encoded.k_requested == chunked.default_chunks(size)
        else:
            assert encoded.k_requested == k
            assert encoded.chunks >= 2, (family, corpus, size, k)
    plan = chunked.plan(tokenizer, text, k=32)
    assert plan.cut_misses == 0, (family, corpus, size)


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_parallel_keyword_follows_the_admission_table(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    """``Tokenizer.encode(parallel=...)`` lets the module choose the path:
    on a row that records only the Python path it runs the Python path
    (and does not raise, whatever the runtime offers); on a row with the
    native path it runs the native encoder when the runtime has it."""
    tokenizer = _family(loaded, family)
    row = TABLE_ROWS[family]
    assert row["status"] == "certified"
    text = _generated_corpora(300_000)["prose(generated)"]
    expected = _reference(tokenizer)(text)
    auto = tokenizer.encode(text, parallel="auto")
    fixed = tokenizer.encode(text, parallel=8, lookup="off")
    assert isinstance(auto, chunked.ChunkedEncoding)
    assert isinstance(fixed, chunked.ChunkedEncoding)
    for encoded in (auto, fixed):
        assert list(encoded.ids) == expected
        assert encoded.serial_reason is None and encoded.chunks >= 2
        if "native" in row["paths"] and chunked.native_available(tokenizer):
            assert encoded.native_facts is not None
        else:
            assert encoded.native_facts is None
    assert auto.k_requested == chunked.default_chunks(len(text))
    assert fixed.k_requested == 8
    facts = chunked.admission(tokenizer)
    assert facts["admitted"] is True and facts["evidence_status"] == "certified"
    assert facts["native_by_default"] == (
        "native" in row["paths"] and facts["native_available"]
    )
    if "native" not in row["paths"] and facts["native_available"]:
        with pytest.raises(RuntimeError, match="no evidence for the native"):
            chunked.encode(tokenizer, text, k=4, native=True)


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_plan_invariants(loaded: dict[str, Tokenizer], family: str) -> None:
    tokenizer = _family(loaded, family)
    table = pclass_table()
    planner = chunked._planner_for(tokenizer)
    text = _generated_corpora(300_000)["transcript(generated)"]
    for k in (2, 5, 16, 64):
        plan = chunked.plan(tokenizer, text, k=k)
        assert plan.spans[0][0] == 0 and plan.spans[-1][1] == len(text)
        assert all(a < b for a, b in plan.spans)
        assert all(
            plan.spans[i][1] == plan.spans[i + 1][0] for i in range(len(plan.spans) - 1)
        )
        assert len(plan.spans) <= k
        floor = max(16, min(1024, len(text) // (4 * k)))
        assert all(b - a >= floor for a, b in plan.spans)
        for cut in plan.cuts:
            assert chunked._boundary_ok(table, text, cut)
            assert cut not in planner.guard.forbidden(text, cut, cut)
    tight = chunked.plan(tokenizer, text, k=8, min_chunk=100_000)
    assert len(tight.spans) <= 3 and tight.cut_misses >= 5


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_cut_never_lands_inside_an_added_token_literal(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    """Negative control: every target position is placed inside a literal."""
    tokenizer = _family(loaded, family)
    planner = chunked._planner_for(tokenizer)
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
    plan = chunked.plan(tokenizer, text, k=k)
    for cut in plan.cuts:
        assert not any(a < cut < b for a, b in occurrences), (family, cut)
    assert plan.literal_rejects >= 1
    encoded = chunked.encode(tokenizer, text, k=k)
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
    counts = chunked.stats(tokenizer)["fallback_counts"]
    assert isinstance(counts, dict)
    before = counts.get("R_CHUNK_NO_CUT", 0)
    # Letters only: no synchronizing transition exists anywhere.
    text = "\u4e2d\u6587\u6d4b\u8bd5" * 10_000
    encoded = chunked.encode(tokenizer, text, k=4, scan_budget=512)
    assert encoded.serial_reason == ReasonCode.R_CHUNK_NO_CUT.value
    assert encoded.chunks == 1 and encoded.cut_misses == 3
    assert list(encoded.ids) == list(tokenizer.encode(text, lookup="off").ids)
    report = tokenizer.explain()["chunked"]
    assert isinstance(report, dict)
    assert report["fallback_counts"]["R_CHUNK_NO_CUT"] == before + 1
    last = report["last"]
    assert isinstance(last, dict)
    # Judged key by key: the block may grow keys (the native path added
    # ``native``), and each fact of this call is checked on its own.
    expected = {
        "chars": len(text),
        "tokens": len(encoded.ids),
        "k_requested": 4,
        "chunks": 1,
        "serial_reason": "R_CHUNK_NO_CUT",
        "cut_misses": 3,
        "literal_rejects": 0,
        "norm_rejects": 0,
        "native": False,
    }
    for key, value in expected.items():
        assert key in last, key
        assert last[key] == value, (key, last[key], value)


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_edge_inputs(loaded: dict[str, Tokenizer], family: str) -> None:
    tokenizer = _family(loaded, family)
    reference = _reference(tokenizer)
    for text in ("", " ", "a", "\n", "ab 12", " " * 300_000, "a" * 300):
        encoded = chunked.encode(tokenizer, text, k=8)
        assert list(encoded.ids) == reference(text), repr(text[:8])
        assert list(encoded.ids) == list(tokenizer.encode(text, lookup="off").ids)
        assert encoded.chunks == 1
    below = chunked.encode(tokenizer, "ab 12 " * 20, k=8)
    assert below.chunks == 1 and below.serial_reason is None
    below_count = chunked.stats(tokenizer)["below_threshold"]
    assert isinstance(below_count, int) and below_count >= 1
    # A lone surrogate is not UTF-8; both routes refuse it the same way.
    bad = ("ab 12 " * 60_000) + "\ud800" + ("cd 34 " * 60_000)
    with pytest.raises(UnicodeEncodeError):
        tokenizer.encode(bad, lookup="off")
    with pytest.raises(UnicodeEncodeError):
        chunked.encode(tokenizer, bad, k=8)


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_explain_carries_the_chunked_block(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    tokenizer = _family(loaded, family)
    text = _generated_corpora(300_000)["prose(generated)"]
    chunked.encode(tokenizer, text, k=4)
    report = tokenizer.explain()
    block = report["chunked"]
    assert isinstance(block, dict)
    assert block["admitted"] is True and block["refusal"] is None
    assert block["family"] == family
    assert block["normalizer"] in ("none", "nfc")
    assert block["literals"] >= 1
    assert block["evidence_id"] == "ev-chunked-cuts-v1"
    assert block["evidence_status"] == "certified"
    assert "python" in block["evidence_paths"]
    assert block["admission_gate"] == "evidence_table"
    assert block["applied"] >= 1 and block["chunks_total"] >= 2
    assert set(block) >= {
        "calls",
        "applied",
        "serial",
        "below_threshold",
        "chunks_total",
        "cut_misses",
        "literal_rejects",
        "norm_rejects",
        "fallback_counts",
        "last",
    }
    ledger = report["fallback_counts"]
    assert isinstance(ledger, dict)
    assert "R_CHUNK_NO_CUT" not in ledger


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_min_chunk_and_scan_budget_bound_the_plan(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    """release-candidate review, notes 5 and 6: the documented bounds hold."""
    tokenizer = _family(loaded, family)
    text = "a" * 110 + " " + "b" * 189
    plan = chunked.plan(tokenizer, text, k=2, scan_budget=1)
    assert plan.spans == ((0, 300),) and plan.cut_misses == 1
    assert chunked.plan(tokenizer, text, k=2, scan_budget=40).cuts == (110,)
    text = "ab 12 " * 50
    plan = chunked.plan(tokenizer, text, k=3, min_chunk=150)
    assert plan.spans == ((0, 300),) and plan.cut_misses == 2
    plan = chunked.plan(tokenizer, text, k=3, min_chunk=100)
    assert len(plan.spans) == 2 and all(b - a >= 100 for a, b in plan.spans)


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_cut_the_normalizer_moves_is_not_made(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    """release-candidate review, finding 1.

    Between U+0958 and ``/`` the input shows a letter-to-punctuation
    transition; NFC writes U+0958 as U+0915 U+093C, and the mark then
    shares a pre-token with the slash. An NFC family must not cut there;
    a family without a normalizer sees the input as the pre-tokenizer
    does and may.
    """
    tokenizer = _family(loaded, family)
    reference = _reference(tokenizer)
    text = "a" * 127 + "\u0958" + "/k" + " " * 126
    assert chunked._boundary_ok(pclass_table(), text, 128)
    encoded = chunked.encode(tokenizer, text, k=2)
    assert list(encoded.ids) == reference(text), family
    assert list(encoded.ids) == list(tokenizer.encode(text, lookup="off").ids)
    if chunked.stats(tokenizer)["normalizer"] == "nfc":
        assert encoded.chunks == 1
        assert encoded.serial_reason == ReasonCode.R_CHUNK_NO_CUT.value
        assert encoded.norm_rejects >= 1
    else:
        assert encoded.chunks == 2 and encoded.cuts == (128,)
    # The same shape with plain letters at the cut is chunked, exactly.
    healthy = "a" * 120 + "\u0958" + "b" * 7 + "/k" + " " * 126
    encoded = chunked.encode(tokenizer, healthy, k=2)
    assert encoded.chunks == 2 and encoded.cuts == (128,)
    assert list(encoded.ids) == reference(healthy)


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_adversarial_normalization_corpus_is_exact(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    tokenizer = _family(loaded, family)
    reference = _reference(tokenizer)
    face = json.loads(tokenizer._backend.materialized_tokenizer_json())
    literals = [str(row["content"]) for row in face["added_tokens"]]
    rng = random.Random(sum(ord(c) for c in family) * 7919 + 0x5EED)
    n_chunked = 0
    for index in range(_ADVERSARIAL):
        text = _adversarial(rng, rng.randint(300, 1500), literals)
        k = rng.choice((2, 3, 4, 8))
        encoded = chunked.encode(tokenizer, text, k=k)
        assert list(encoded.ids) == reference(text), (family, index, k)
        n_chunked += encoded.chunks >= 2
    # The corpus must actually exercise cuts, not only refusals.
    assert n_chunked >= _ADVERSARIAL // 4, (family, n_chunked, _ADVERSARIAL)


def test_roster_families_are_the_admitted_set(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The table is a subset of the roster, and the run-time admitted set is
    exactly the table's certified rows: every roster family whose artifact
    is present locally is loaded and asked; a certified row is admitted, a
    partial row is refused with its status, an absent family is refused as
    without evidence. The artifact part runs when the artifacts are there."""
    names = {item.family for item in roster_families()}
    assert set(TABLE_FAMILIES) <= names
    assert set(CANDIDATE_FAMILIES) <= set(TABLE_FAMILIES)
    assert "kimi_k3" not in names and "bert_uncased" not in names
    present = available_families(sorted(names))
    if not present:
        pytest.skip("roster artifacts not available under TOKTIER_TEST_ARTIFACTS")
    admitted: set[str] = set()
    for family in present:
        tokenizer = load_isolated(family, tmp_path_factory)
        try:
            facts = chunked.admission(tokenizer)
            if chunked.is_admitted(tokenizer):
                admitted.add(family)
            row = TABLE_ROWS.get(family)
            if row is None:
                assert facts["refusal"] == "artifact_without_chunked_evidence", family
                assert facts["evidence_status"] is None
            elif row["status"] != "certified":
                assert facts["refusal"] == (
                    f"chunked_evidence_incomplete:{row['status']}"
                ), family
                assert facts["evidence_status"] == row["status"]
            else:
                assert facts["refusal"] is None and facts["admitted"] is True, family
                assert facts["evidence_paths"] == row["paths"]
                if row["loader_face_sha256"] is not None:
                    assert facts["loader_face_sha256"] == row["loader_face_sha256"]
        finally:
            tokenizer.close()
    assert admitted == set(CANDIDATE_FAMILIES) & set(present)


def test_packaged_artifacts_set_no_strip_or_single_word_flag() -> None:
    """The refusal of flagged literals leaves every packaged artifact as it
    was: none of the loader faces the reference executes (the artifact
    file, or the pinned loader's serialization when the configuration
    sidecar contributes added tokens) carries an added token with
    ``lstrip``, ``rstrip`` or ``single_word`` set. Runs over every
    packaged family whose artifact is present locally."""
    from toktier.backends.loader_face import (
        config_added_token_rows,
        live_tokenizer_json,
        load_live_tokenizer,
    )

    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    checked: list[str] = []
    for family in manifest.families():
        root = _artifact_source(family)
        if root is None:
            continue
        text = (root / "tokenizer.json").read_text(encoding="utf-8")
        document = json.loads(text)
        if config_added_token_rows(root, document):
            document = json.loads(live_tokenizer_json(load_live_tokenizer(root)))
        assert chunked._flagged_literal(document) is False, family
        assert chunked._normalizer_mode(document)[1] != _LITERAL_FLAG_REFUSAL, family
        checked.append(family)
    if not checked:
        pytest.skip("packaged artifacts not available under TOKTIER_TEST_ARTIFACTS")
    assert set(available_families(sorted(TABLE_FAMILIES))) <= set(checked)


@pytest.mark.parametrize("family", PARTIAL_FAMILIES)
def test_partial_row_is_encoded_serially_with_the_refusal_named(
    tmp_path_factory: pytest.TempPathFactory, family: str
) -> None:
    """A roster artifact whose row has one evidence level is a serial
    tokenizer for this module: ids equal the plain encode, the fallback is
    counted, and the report names the status."""
    if not _artifact_source(family):
        pytest.skip(f"{family}: artifact not available under TOKTIER_TEST_ARTIFACTS")
    tokenizer = load_isolated(family, tmp_path_factory)
    try:
        text = _generated_corpora(300_000)["prose(generated)"]
        encoded = chunked.encode(tokenizer, text, k=8)
        assert encoded.serial_reason == ReasonCode.R_CHUNK_FAMILY_UNSUPPORTED.value
        assert encoded.chunks == 1 and encoded.native_facts is None
        assert list(encoded.ids) == list(tokenizer.encode(text, lookup="off").ids)
        assert list(encoded.ids) == _reference(tokenizer)(text)
        wrapped = tokenizer.encode(text, parallel="auto")
        assert isinstance(wrapped, chunked.ChunkedEncoding)
        assert wrapped.serial_reason == ReasonCode.R_CHUNK_FAMILY_UNSUPPORTED.value
        report = tokenizer.explain()["chunked"]
        assert isinstance(report, dict)
        assert report["admitted"] is False
        assert report["refusal"] == (
            f"chunked_evidence_incomplete:{TABLE_ROWS[family]['status']}"
        )
        assert report["fallback_counts"] == {"R_CHUNK_FAMILY_UNSUPPORTED": 2}
        with pytest.raises(ValueError, match="no certified chunk-cut rule"):
            chunked.plan(tokenizer, text, k=8)
    finally:
        tokenizer.close()


def test_removing_a_row_turns_a_certified_family_serial(
    tmp_path_factory: pytest.TempPathFactory,
    admission_table: Callable[..., dict[str, Any]],
) -> None:
    """The negative control on a real artifact: with its row removed from
    the table a certified family is encoded serially under
    R_CHUNK_FAMILY_UNSUPPORTED, and explain() names the missing evidence;
    with the shipped table restored the same artifact is chunked again."""
    present = available_families()
    if not present:
        pytest.skip("no certified artifact available under TOKTIER_TEST_ARTIFACTS")
    family = present[0]

    def drop(document: dict[str, Any]) -> None:
        document["rows"] = [r for r in document["rows"] if r["family"] != family]

    text = _generated_corpora(300_000)["code(generated)"]
    admission_table(drop)
    tokenizer = load_isolated(family, tmp_path_factory)
    try:
        assert chunked.is_admitted(tokenizer) is False
        encoded = chunked.encode(tokenizer, text, k=8)
        assert encoded.serial_reason == ReasonCode.R_CHUNK_FAMILY_UNSUPPORTED.value
        assert encoded.chunks == 1
        assert list(encoded.ids) == _reference(tokenizer)(text)
        report = tokenizer.explain()["chunked"]
        assert isinstance(report, dict)
        assert report["refusal"] == "artifact_without_chunked_evidence"
        assert report["evidence_id"] is None and report["evidence_paths"] == []
        assert report["fallback_counts"] == {"R_CHUNK_FAMILY_UNSUPPORTED": 1}
    finally:
        tokenizer.close()
    admission_table()
    tokenizer = load_isolated(family, tmp_path_factory)
    try:
        assert chunked.is_admitted(tokenizer) is True
        encoded = chunked.encode(tokenizer, text, k=8)
        assert encoded.serial_reason is None and encoded.chunks >= 2
        assert list(encoded.ids) == _reference(tokenizer)(text)
    finally:
        tokenizer.close()
