#!/usr/bin/env python3
"""The adversarial normalization corpus of the chunk-parallel path.

One generator, shared by the chunked tests (``tests/test_chunked.py``) and
the campaign tool (``tools/run_chunked_cut_campaign.py adversarial``), so
the battery the admission table's rows rest on is the corpus the tests
judge. :func:`adversarial` builds one string of about ``chars`` characters
from pieces that stress NFC and the cut predicate: decomposing characters
against ASCII punctuation, combining-mark runs, jamo that compose across a
cut, kana voicing marks, ``<=>`` with U+0338, added-token literals beside
marks, contraction suffixes after marks, digits of several scripts, format
and control characters, emoji sequences and plain prose, joined by assorted
separators. ``literals`` are the added-token strings of the family under
judgement.

Source text in this repository is ASCII, so every non-ASCII probe character
is written as an escape.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

__all__ = ["GENERATOR", "adversarial", "battery_k", "battery_rng"]

#: How the tests and the campaign tool describe this corpus in a reading.
GENERATOR = (
    "adversarial normalization corpus (tools/chunked_adversarial_corpus.py): "
    "strings of 300-1,500 characters, k drawn from {2, 3, 4, 8}, one random "
    "stream per family seeded from the family name"
)


def battery_rng(family: str, offset: int = 0) -> random.Random:
    """The per-family random stream the tests and the tool share."""
    return random.Random(sum(ord(c) for c in family) * 7919 + 0x5EED + offset)


def battery_k(rng: random.Random) -> int:
    """The requested chunk count drawn after each string."""
    return int(rng.choice((2, 3, 4, 8)))


# -- adversarial normalization corpus ----------------------------------------

#: Characters NFC rewrites: letters that decompose into a letter and a
#: combining mark (composition exclusions), singletons whose NFC form is
#: another character, marks that decompose into two marks, compatibility
#: ideographs, and musical symbols with combining parts.
_DECOMPOSING = (
    "\u0958\u0959\u095a\u095b\u095c\u095d\u095e\u095f\u09dc\u09dd\u09df"
    "\u0a33\u0a36\u0a59\u0a5a\u0a5b\u0a5e\u0b5c\u0b5d"
    "\ufb1d\ufb1f\ufb2a\ufb2b\ufb2c\ufb2d\ufb2e\ufb2f\ufb30\ufb31\ufb32\ufb33"
    "\ufb34\ufb35\ufb36\ufb38\ufb39\ufb3a\ufb3b\ufb3c\ufb3e\ufb40\ufb41\ufb43"
    "\ufb44\ufb46\ufb47\ufb48\ufb49\ufb4a\ufb4b\ufb4c\ufb4d\ufb4e"
    "\u0f43\u0f4d\u0f52\u0f57\u0f5c\u0f69\u0f76\u0f78\u0f93\u0f9d\u0fa2\u0fa7"
    "\u0fac\u0fb9"
    "\u2126\u212a\u212b\u037e\u0387\u1fef\u1ffd\u1fee\u2000\u2001\u2329\u232a"
    "\u0340\u0341\u0343\u0344\u0374\u1f71\u1f73\u1f75\u1f77\u1f79\u1f7b\u1f7d"
    "\u1fbb\u0f73\u0f75\u0f81\u2adc"
    "\uf900\uf901\uf902\uf90a\uf95f\ufa10\ufa12\ufa15\ufa30\ufa6a"
    "\U0001d15e\U0001d15f\U0001d160\U0001d1bb\U0001d1bc"
)
#: Combining marks (and a few spacing marks that compose with a preceding
#: vowel sign) across scripts, plus variation selectors.
_MARKS = (
    "\u0300\u0301\u0302\u0303\u0304\u0306\u0307\u0308\u030a\u030b\u030c\u0323"
    "\u0324\u0325\u0327\u0328\u0331\u0338\u0345\u034f"
    "\u0483\u0484\u0591\u0592\u05b0\u05b4\u05b7\u05bc"
    "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670"
    "\u093c\u093e\u093f\u0940\u0941\u0942\u0947\u0948\u094b\u094c\u094d\u0951"
    "\u0952\u09be\u09bf\u09c7\u09c8\u09cb\u09cc\u09cd\u09d7\u0b3e\u0b47\u0b4b"
    "\u0b56\u0b57\u0bbe\u0bc6\u0bc7\u0bca\u0bd7\u0d3e\u0d46\u0d4a\u0d57\u0dca"
    "\u0dcf\u0dd9\u0ddc\u0ddf"
    "\u0e31\u0e34\u0e35\u0e38\u0e39\u0e3a\u0e47\u0e48\u0e49\u0e4a\u0e4b"
    "\u0f71\u0f72\u0f74\u0f7a\u0f7b\u0f7c\u0f7d\u0f80"
    "\u1ab0\u1dc0\u1dc1\u20d0\u20d1\u20dd\u20e3\u3099\u309a"
    "\ufe00\ufe0e\ufe0f\U000e0100\U0001d165\U0001d16e"
)
_SCRIPT_BASES = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "".join(chr(c) for c in range(0x0391, 0x03AA))
    + "".join(chr(c) for c in range(0x03B1, 0x03CA)),
    "".join(chr(c) for c in range(0x0410, 0x0450)),
    "".join(chr(c) for c in range(0x0915, 0x093A)),  # Devanagari consonants
    "".join(chr(c) for c in range(0x0995, 0x09A9)),  # Bengali consonants
    "".join(chr(c) for c in range(0x0B15, 0x0B29)),  # Oriya consonants
    "".join(chr(c) for c in range(0x0B95, 0x0B9A))
    + "\u0b9e\u0b9f\u0ba3\u0ba4",  # Tamil
    "".join(chr(c) for c in range(0x0D15, 0x0D29)),  # Malayalam consonants
    "".join(chr(c) for c in range(0x0D9A, 0x0DB2)),  # Sinhala consonants
    "".join(chr(c) for c in range(0x0E01, 0x0E2F)),  # Thai
    "".join(chr(c) for c in range(0x0621, 0x064B)),  # Arabic
    "".join(chr(c) for c in range(0x05D0, 0x05EB)),  # Hebrew
    "".join(chr(c) for c in range(0x0F40, 0x0F6B)),  # Tibetan
    "".join(chr(c) for c in range(0x3041, 0x3097)),  # Hiragana
    "".join(chr(c) for c in range(0x30A1, 0x30FB)),  # Katakana
    "".join(chr(c) for c in range(0x4E00, 0x4E00 + 200)),  # CJK
    "".join(chr(c) for c in range(0xAC00, 0xAC00 + 300, 7)),  # Hangul syllables
)
_JAMO_L = "".join(chr(c) for c in range(0x1100, 0x1113))
_JAMO_V = "".join(chr(c) for c in range(0x1161, 0x1176))
_JAMO_T = "".join(chr(c) for c in range(0x11A8, 0x11C3))
_DIGIT_POOLS = (
    "0123456789",
    "".join(chr(c) for c in range(0x0660, 0x066A)),  # Arabic-Indic
    "".join(chr(c) for c in range(0x0966, 0x0970)),  # Devanagari
    "".join(chr(c) for c in range(0xFF10, 0xFF1A)),  # fullwidth
    "\u00bd\u00bc\u217b\u2167",
)
_FORMATS = "\u200b\u200c\u200d\ufeff\u200e\u200f\u202a\u202e\u202c\u2066\u2069"
_CONTROLS = (
    "".join(chr(c) for c in range(0x00, 0x09))
    + "\x0b\x0c"
    + "".join(chr(c) for c in range(0x0E, 0x20))
    + "\x7f"
)
_SPACES = (
    " ",
    " ",
    " ",
    " ",
    "\u00a0",
    "\u2003",
    "\u3000",
    "\t",
    "\n",
    "\r\n",
    "\u2028",
    "\x0b",
)
_ASCII_PUNCT = ".,;:!?()[]{}<>/\\'\"`~@#$%^&*-_=+|"
_EMOJI = (
    "\U0001f642",
    "\U0001f680",
    "\u2764\ufe0f",
    "\u2728",
    "\U0001f469\u200d\U0001f4bb",
    "\U0001f468\u200d\U0001f469\u200d\U0001f467",
    "\U0001f3f3\ufe0f\u200d\U0001f308",
    "\U0001f1fa\U0001f1f8",
    "\U0001f44d\U0001f3fd",
)


def adversarial(rng: random.Random, chars: int, literals: Sequence[str]) -> str:
    """A string of about ``chars`` characters built from pieces that stress
    NFC: decomposing characters against ASCII punctuation (the reviewed shape),
    Zalgo runs, jamo that compose across a cut, kana voicing marks,
    ``<=>`` with U+0338, added-token literals beside marks, contraction
    suffixes after marks, digits of several scripts, formats, controls,
    emoji sequences and plain prose, joined by assorted separators."""
    parts: list[str] = []
    length = 0
    lits = [item for item in literals if item] or ["<|endoftext|>"]
    while length < chars:
        kind = rng.random()
        if kind < 0.18:
            piece = "".join(
                rng.choice("abcdefghijklmnopqrstuvwxyz")
                for _ in range(rng.randint(1, 9))
            )
            if rng.random() < 0.15:
                piece += rng.choice(("'s", "'t", "n't", "'re", "'ll"))
        elif kind < 0.26:
            pool = rng.choice(_DIGIT_POOLS)
            piece = "".join(rng.choice(pool) for _ in range(rng.randint(1, 6)))
            if rng.random() < 0.2:
                piece += rng.choice(("'x", "'", "."))
        elif kind < 0.44:
            bases = rng.choice(_SCRIPT_BASES)
            out: list[str] = []
            for _ in range(rng.randint(1, 8)):
                out.append(
                    rng.choice(_DECOMPOSING)
                    if rng.random() < 0.2
                    else rng.choice(bases)
                )
                if rng.random() < 0.35:
                    out.extend(rng.choice(_MARKS) for _ in range(rng.randint(1, 3)))
            piece = "".join(out)
        elif kind < 0.50:
            piece = (
                rng.choice(_DECOMPOSING) + rng.choice(_ASCII_PUNCT) + rng.choice("kx1 ")
            )
        elif kind < 0.56:
            piece = rng.choice(_SCRIPT_BASES[0] + _SCRIPT_BASES[1]) + "".join(
                rng.choice(_MARKS) for _ in range(rng.randint(1, 40))
            )
        elif kind < 0.62:
            out = []
            for _ in range(rng.randint(1, 4)):
                out.append(rng.choice(_JAMO_L))
                out.append(rng.choice(_JAMO_V))
                if rng.random() < 0.5:
                    out.append(rng.choice(_JAMO_T))
                if rng.random() < 0.3:
                    out.append(rng.choice(_SCRIPT_BASES[-1]))
            piece = "".join(out)
        elif kind < 0.66:
            piece = "".join(
                rng.choice(_SCRIPT_BASES[13] + _SCRIPT_BASES[14])
                + rng.choice("\u3099\u309a")
                for _ in range(rng.randint(1, 4))
            )
        elif kind < 0.70:
            piece = rng.choice("<=>") + "\u0338" + rng.choice("ab1 ")
        elif kind < 0.76:
            literal = rng.choice(lits)
            piece = rng.choice(
                (
                    literal + rng.choice(_MARKS),
                    rng.choice(_MARKS) + literal,
                    rng.choice(_DECOMPOSING) + literal + rng.choice(_DECOMPOSING),
                    literal,
                    literal[: len(literal) // 2]
                    + rng.choice(_MARKS)
                    + literal[len(literal) // 2 :],
                )
            )
        elif kind < 0.80:
            piece = (
                rng.choice(("don", "isn", "we", "12", "x"))
                + rng.choice(_MARKS)
                + rng.choice(("'t", "'re", "'s", "'"))
            )
        elif kind < 0.84:
            piece = (
                rng.choice(_FORMATS)
                + rng.choice(_SCRIPT_BASES[0])
                + rng.choice(_FORMATS)
            )
        elif kind < 0.87:
            piece = rng.choice(_CONTROLS)
        elif kind < 0.92:
            piece = rng.choice(_EMOJI)
        elif kind < 0.96:
            piece = "".join(rng.choice(_ASCII_PUNCT) for _ in range(rng.randint(1, 4)))
        else:
            piece = rng.choice(_MARKS) * rng.randint(1, 5)
        separator = rng.random()
        if separator < 0.30:
            joiner = ""
        elif separator < 0.75:
            joiner = " "
        elif separator < 0.90:
            joiner = rng.choice(_SPACES)
        else:
            joiner = rng.choice(_ASCII_PUNCT)
        parts.append(piece + joiner)
        length += len(piece) + len(joiner)
    return "".join(parts)
