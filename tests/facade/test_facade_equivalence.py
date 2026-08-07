"""Every facade path returns the ids of a from-scratch reference encode.

Property-style: seeded random documents (multi-byte code points,
combining marks, a ZWJ sequence, long runs) are pushed through the
plain, session and content-lookup paths, in memory and against a
persistent store, and each result is judged against the oracle. The
unicode boundary cases place multi-byte and combining sequences exactly
across the checkpoint byte positions the auto index digests at.

Source text in this repository is ASCII, so every non-ASCII probe
character is written as an escape.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from toktier.facade.index import MARK_FLOOR_BYTES
from toktier.facade.store import AUTO_MIN_BYTES

from .conftest import Rig

#: Alphabet with 1-4 byte code points, a combining mark and a ZWJ.
_ALPHABET = (
    "abcdefghij XYZ0123456789.,;:!?()[]"
    "\u00e9\u00df\u4e2d\u6587\u3042\ud55c"
    "\u0301\u200d\U0001f642\U0001f469"
)


def _document(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(length))


def test_all_paths_match_reference_over_random_documents(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    rng = random.Random(0xFACADE)
    tokenizer = rig.tokenizer(store=rig.store_path())
    for round_index in range(24):
        text = _document(rng, rng.randint(0, 3000))
        expected = reference(text)
        assert list(tokenizer.encode(text, lookup="off").ids) == expected
        assert list(tokenizer.encode(text).ids) == expected
        assert (
            list(tokenizer.encode(text, session=f"doc-{round_index}").ids)
            == expected
        )


def test_session_growth_and_rewrites_match_reference(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    rng = random.Random(0x5E5510)
    tokenizer = rig.tokenizer(store=rig.store_path())
    for round_index in range(6):
        session = f"chat-{round_index}"
        text = _document(rng, rng.randint(0, 400))
        for _ in range(8):
            # Grow (append path) most of the time; occasionally rewrite
            # to a non-extension (overwrite path).
            if rng.random() < 0.75:
                text = text + _document(rng, rng.randint(1, 400))
            else:
                text = _document(rng, rng.randint(0, 400))
            observed = tokenizer.encode(text, session=session)
            assert list(observed.ids) == reference(text)


def test_auto_extension_chain_matches_reference(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    rng = random.Random(0xA0701)
    tokenizer = rig.tokenizer(store=rig.store_path())
    text = _document(rng, 3 * AUTO_MIN_BYTES)
    for _ in range(6):
        assert list(tokenizer.encode(text).ids) == reference(text)
        text = text + _document(rng, rng.randint(1, 800))
    report = tokenizer.explain()
    store_stats = report["store"]
    assert isinstance(store_stats, dict)
    assert store_stats["auto_appends"] >= 1


def test_unicode_sequences_across_checkpoint_boundaries(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    """Multi-byte and combining sequences straddling digest positions."""
    tokenizer = rig.tokenizer(store=rig.store_path())
    straddlers = (
        "\u00e9",  # 2 bytes
        "\u4e2d",  # 3 bytes
        "\U0001f642",  # 4 bytes
        "e\u0301\u0301",  # combining marks
        "\U0001f469\u200d\U0001f4bb",  # ZWJ sequence
    )
    for boundary in (MARK_FLOOR_BYTES, 2 * MARK_FLOOR_BYTES):
        for probe in straddlers:
            for shift in (-3, -2, -1, 0, 1):
                head = "a" * (boundary + shift)
                base = head + probe
                extended = base + " tail" + probe
                for text in (base, extended):
                    assert list(tokenizer.encode(text).ids) == reference(text)
                session = f"u-{boundary}-{shift}-{ord(probe[0]):x}"
                assert (
                    list(tokenizer.encode(base, session=session).ids)
                    == reference(base)
                )
                assert (
                    list(tokenizer.encode(extended, session=session).ids)
                    == reference(extended)
                )


def test_in_memory_store_matches_reference_without_persistence(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    rng = random.Random(0x1B4D)
    tokenizer = rig.tokenizer()  # no store directory
    text = _document(rng, 2 * AUTO_MIN_BYTES)
    for _ in range(4):
        assert list(tokenizer.encode(text).ids) == reference(text)
        assert list(tokenizer.encode(text, session="mem").ids) == reference(text)
        text = text + _document(rng, 200)
    assert not rig.store_path().exists()
