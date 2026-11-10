"""The table facts the session seal's normalized-cut rule rests on.

The rule that decides a seal on the characters the reference reads
(``crates/toktier-routing-core/src/seal_cut.rs``, with the chunk planner's
``NfcRule``) is sufficient because of five facts about the reference
normalizer's own tables against the frozen O/S/L/N/M class table:

* F1 every non-starter of the normalizer is class M;
* F2 for every character outside class M, the NFD image begins with a
  starter of the same class and the NFC image begins with a character of
  the same class (the NFC image ends in class M for 73 characters: 59 L and
  14 O -- the family of the seal defect of this release);
* F3 every second member of a composition pair is class M; the only
  characters outside class M that are stable alone and yet combine with a
  preceding character are the conjoining vowels and finals (21 + 27, L);
* F4 no composition changes the class of its first member;
* F5 no stable split point (the planner's test) is a non-starter or a second
  member of a composition pair.

The facts are recomputed here from the normalizer ``tokenizers`` executes
(not from Python's ``unicodedata``) and the shipped class table, so a
change of the ``tokenizers`` pin or of the class table turns this test red
instead of silently removing a premise of the rule. The counts are those
of ``tokenizers`` 0.22.2 and the frozen table.
"""

from __future__ import annotations

import collections
from typing import Any

import pytest

tokenizers = pytest.importorskip("tokenizers")
normalizers = pytest.importorskip("tokenizers.normalizers")

M = 4
NAMES = "OSLNM"
#: Marks of several combining classes; a character that reorders against one
#: of them has a nonzero combining class for the normalizer.
PROBES = [
    "́", "̵", "̣", "ְ", "゙", "्", "ุ", "ཱ",
    "ͅ",
]  # fmt: skip
#: The stable-split-point probes of the chunk planner (``chunked.rs``).
PROBE_BEFORE, PROBE_AFTER, PROBE_JAMO_L, PROBE_SYLLABLE = (
    "́",
    "̵",
    "ᄀ",
    "가",
)


@pytest.fixture(scope="module")
def facts() -> dict[str, Any]:
    from toktier.repair.registry import pclass_table

    table = pclass_table()
    assert len(table) == 0x110000
    nfc = normalizers.NFC().normalize_str
    nfd = normalizers.NFD().normalize_str
    scalars = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    nfc_of: dict[int, str] = {}
    nfd_of: dict[int, str] = {}
    for c in scalars:
        ch = chr(c)
        if nfc(ch) != ch:
            nfc_of[c] = nfc(ch)
        if nfd(ch) != ch:
            nfd_of[c] = nfd(ch)
    nonstarter = set()
    for c in scalars:
        if c in nfd_of:
            continue
        ch = chr(c)
        for probe in PROBES:
            if nfd(probe + ch) != probe + ch or nfd(ch + probe) != ch + probe:
                nonstarter.add(c)
                break
    out: dict[str, Any] = {"nonstarters": len(nonstarter)}
    out["F1"] = [c for c in sorted(nonstarter) if table[c] != M]
    nfd_first_nonstarter, nfd_first_class, nfc_first_class = [], [], []
    nfc_last = collections.Counter()
    for c in scalars:
        if table[c] == M:
            continue
        d = nfd_of.get(c)
        if d is not None:
            if ord(d[0]) in nonstarter:
                nfd_first_nonstarter.append(c)
            if table[ord(d[0])] != table[c]:
                nfd_first_class.append(c)
        a = nfc_of.get(c)
        if a is not None:
            if table[ord(a[0])] != table[c]:
                nfc_first_class.append(c)
            if table[ord(a[-1])] != table[c]:
                nfc_last[f"{NAMES[table[c]]}->{NAMES[table[ord(a[-1])]]}"] += 1
    out["F2"] = (nfd_first_nonstarter, nfd_first_class, nfc_first_class)
    out["nfc_last_class_differs"] = dict(nfc_last)
    pairs: dict[tuple[str, str], str] = {}
    unresolved = []
    for c, d in nfd_of.items():
        ch = chr(c)
        if nfc(ch) != ch or 0xAC00 <= c <= 0xD7A3:
            continue
        # The last character of the image first; canonical order may place
        # the composing mark elsewhere, so then every other position.
        head = nfc(d[:-1])
        if len(head) == 1 and nfc(head + d[-1]) == ch:
            pairs[(head, d[-1])] = ch
            continue
        found = False
        for i in range(1, len(d)):
            head = nfc(d[:i] + d[i + 1 :])
            if len(head) == 1 and nfc(head + d[i]) == ch:
                pairs[(head, d[i])] = ch
                found = True
        if not found:
            unresolved.append(c)
    out["pairs"] = len(pairs)
    out["unresolved"] = unresolved
    seconds = {last for (_head, last) in pairs}
    out["seconds"] = len(seconds)
    out["F3_second_not_M"] = sorted(x for x in seconds if table[ord(x)] != M)
    out["F4"] = [
        (head, last)
        for (head, last), comp in pairs.items()
        if table[ord(head)] != table[ord(comp)]
    ]

    def stable_split_point(ch: str) -> bool:
        if ord(ch) < 0x80:
            return True
        return (
            table[ord(ch)] != M
            and nfc(ch) == ch
            and nfc(PROBE_BEFORE + ch) == PROBE_BEFORE + ch
            and nfc(ch + PROBE_AFTER) == ch + PROBE_AFTER
            and nfc(PROBE_JAMO_L + ch) == PROBE_JAMO_L + ch
            and nfc(PROBE_SYLLABLE + ch) == PROBE_SYLLABLE + ch
        )

    split = {c for c in scalars if stable_split_point(chr(c))}
    out["split_points"] = len(split)
    out["F5"] = (
        sorted(split & nonstarter),
        sorted(ord(x) for x in seconds if ord(x) in split),
    )
    combining = collections.Counter()
    for c in scalars:
        if table[c] == M or c in split or c in nfc_of:
            continue
        ch = chr(c)
        kind = (
            "vowel"
            if nfc(PROBE_JAMO_L + ch) != PROBE_JAMO_L + ch
            else "final"
            if nfc(PROBE_SYLLABLE + ch) != PROBE_SYLLABLE + ch
            else "other"
        )
        combining[f"{kind}:{NAMES[table[c]]}"] += 1
    out["F3_combining_starters"] = dict(combining)
    return out


def test_f1_every_non_starter_is_class_m(facts: dict[str, Any]) -> None:
    assert facts["F1"] == []
    assert facts["nonstarters"] == 810


def test_f2_a_character_outside_m_keeps_its_class_at_the_front(
    facts: dict[str, Any],
) -> None:
    assert facts["F2"] == ([], [], [])
    # The family of the defect: NFC images that end in a mark.
    assert facts["nfc_last_class_differs"] == {"L->M": 59, "O->M": 14}


def test_f3_second_members_are_marks_except_the_conjoining_jamo(
    facts: dict[str, Any],
) -> None:
    assert facts["unresolved"] == []
    assert (facts["pairs"], facts["seconds"]) == (940, 62)
    assert facts["F3_second_not_M"] == []
    assert facts["F3_combining_starters"] == {"vowel:L": 21, "final:L": 27}


def test_f4_no_composition_changes_the_class_of_its_first_member(
    facts: dict[str, Any],
) -> None:
    assert facts["F4"] == []


def test_f5_no_stable_split_point_is_a_non_starter_or_a_second_member(
    facts: dict[str, Any],
) -> None:
    assert facts["F5"] == ([], [])
    assert facts["split_points"] == 1108410
