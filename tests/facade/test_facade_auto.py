"""Content lookup: checkpoint index behavior and its hard gates.

The digest index only proposes candidates; every judgment below checks
both halves of that split: endpoint hits are exact (one byte either
side of the endpoint must miss), the longest stored prefix wins, a
forged digest is stopped by the byte re-check, and a lost or corrupt
index degrades to rebuild-or-full-encode with correct ids throughout.
"""

from __future__ import annotations

import json
import random
import string
from collections.abc import Callable
from pathlib import Path

import pytest

from toktier.facade import index as index_module
from toktier.facade import store as store_module
from toktier.facade.index import (
    CheckpointIndex,
    IndexEntry,
    entry_for,
    mark_positions,
    prefix_digests,
)
from toktier.facade.store import AUTO_MIN_BYTES, EntryStore

from .conftest import TEST_FINGERPRINT, Rig, SpanEncode

_rng = random.Random(0xC0FFEE)


def _text(length: int) -> str:
    return "".join(_rng.choice(string.ascii_letters + " .,") for _ in range(length))


def _store(
    span_reference: SpanEncode,
    directory: Path | None = None,
    **keywords: object,
) -> EntryStore:
    return EntryStore(
        fingerprint=TEST_FINGERPRINT,
        encode=span_reference,
        directory=directory,
        **keywords,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------- index --


def test_mark_positions_double_from_the_floor() -> None:
    assert mark_positions(0) == ()
    assert mark_positions(4096) == ()
    assert mark_positions(4097) == (4096,)
    assert mark_positions(100_000) == (4096, 8192, 16384, 32768, 65536)


def test_prefix_digests_match_direct_recomputation() -> None:
    data = _text(20_000).encode("utf-8")
    lengths = [1, 4096, 8192, 19_999, len(data)]
    streamed = prefix_digests(data, lengths)
    for length in lengths:
        assert streamed[length] == entry_for(data[:length]).end_digest


def test_index_payload_round_trip_and_malformation() -> None:
    index = CheckpointIndex()
    index.put("a-x", entry_for(_text(9000).encode("utf-8")))
    rebuilt = CheckpointIndex.from_payload(index.to_payload())
    assert rebuilt is not None
    assert rebuilt.to_payload() == index.to_payload()

    assert CheckpointIndex.from_payload(None) is None
    assert CheckpointIndex.from_payload({"format": 99, "entries": {}}) is None
    assert (
        CheckpointIndex.from_payload({"format": 1, "entries": {"x": {"bytes": "3"}}})
        is None
    )


# ------------------------------------------------------------ endpoints --


def test_endpoint_hit_is_exact_and_one_byte_off_misses(
    span_reference: SpanEncode, reference: Callable[[str], list[int]]
) -> None:
    store = _store(span_reference)
    base = _text(2 * AUTO_MIN_BYTES)

    assert store.encode_auto(base) == reference(base)  # miss -> entry
    assert store.stats()["auto_misses"] == 1

    assert store.encode_auto(base) == reference(base)  # exact endpoint hit
    assert store.stats()["auto_hits"] == 1

    extended = base + "!"
    assert store.encode_auto(extended) == reference(extended)
    assert store.stats()["auto_appends"] == 1

    # One byte short of a stored endpoint: no endpoint can match.
    before = store.stats()
    shorter = base[:-1]
    assert store.encode_auto(shorter) == reference(shorter)
    after = store.stats()
    assert after["auto_hits"] == before["auto_hits"]

    # Divergence in the final byte: digests cannot collide into a hit.
    diverged = base[:-1] + ("A" if base[-1] != "A" else "B") + "tail"
    assert store.encode_auto(diverged) == reference(diverged)
    assert store.stats()["collision_rejects"] == 0


def test_longest_stored_prefix_wins(
    span_reference: SpanEncode, reference: Callable[[str], list[int]]
) -> None:
    store = _store(span_reference)
    short = _text(AUTO_MIN_BYTES + 100)
    longer = short + _text(AUTO_MIN_BYTES)
    # Longer first: storing the shorter one afterwards misses (no stored
    # text is a prefix of it), leaving two independent entries.
    assert store.encode_auto(longer) == reference(longer)
    assert store.encode_auto(short) == reference(short)
    assert store.stats()["entries"] == 2

    query = longer + " and more"
    assert store.encode_auto(query) == reference(query)
    texts = {
        entry.text
        for entry in store._entries.values()
        if entry.text is not None
    }
    # The longer entry was extended in place; the shorter one is intact.
    assert query in texts
    assert short in texts
    assert longer not in texts


def test_forged_digest_is_stopped_by_the_byte_gate(
    span_reference: SpanEncode, reference: Callable[[str], list[int]]
) -> None:
    store = _store(span_reference)
    stored = _text(AUTO_MIN_BYTES + 50)
    assert store.encode_auto(stored) == reference(stored)

    # Forge the stored entry's index row so it claims to be the prefix
    # of an unrelated query: same length, same digest as the query's
    # prefix, but the entry text underneath differs.
    query = _text(AUTO_MIN_BYTES + 300)
    assert not query.startswith(stored)
    forged_length = len(stored.encode("utf-8"))
    forged_digest = entry_for(
        query.encode("utf-8")[:forged_length]
    ).end_digest
    (name,) = store._index.names()
    store._index.put(
        name,
        IndexEntry(byte_length=forged_length, end_digest=forged_digest, marks=()),
    )

    assert store.encode_auto(query) == reference(query)
    collision_rejects = store.stats()["collision_rejects"]
    assert isinstance(collision_rejects, int)
    assert collision_rejects >= 1


def test_below_threshold_texts_bypass_the_store(
    span_reference: SpanEncode,
) -> None:
    store = _store(span_reference)
    assert store.encode_auto("short text") is None
    assert store.stats()["entries"] == 0


def test_auto_entry_cap_evicts_the_oldest(
    span_reference: SpanEncode,
    reference: Callable[[str], list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_module, "MAX_AUTO_ENTRIES", 2)
    store = _store(span_reference)
    texts = [_text(AUTO_MIN_BYTES + i) for i in range(3)]
    for text in texts:
        assert store.encode_auto(text) == reference(text)
    assert store.stats()["entries"] == 2
    # The displaced first entry misses now, and is still answered right.
    assert store.encode_auto(texts[0]) == reference(texts[0])


# ---------------------------------------------------- derived index file --


def test_corrupt_index_is_rebuilt_from_records(
    rig: Rig,
    span_reference: SpanEncode,
    reference: Callable[[str], list[int]],
) -> None:
    directory = rig.store_path()
    first = _store(span_reference, directory)
    stored = _text(AUTO_MIN_BYTES + 10)
    assert first.encode_auto(stored) == reference(stored)

    (directory / "index.json").write_bytes(b"\x00 not json")
    second = _store(span_reference, directory)
    assert second.stats()["index_rebuilds"] == 1
    assert second.encode_auto(stored) == reference(stored)
    assert second.stats()["auto_hits"] == 1


def test_missing_index_is_rebuilt_from_records(
    rig: Rig,
    span_reference: SpanEncode,
    reference: Callable[[str], list[int]],
) -> None:
    directory = rig.store_path()
    first = _store(span_reference, directory)
    stored = _text(AUTO_MIN_BYTES + 10)
    assert first.encode_auto(stored) == reference(stored)

    (directory / "index.json").unlink()
    second = _store(span_reference, directory)
    assert second.stats()["index_rebuilds"] == 1
    assert second.encode_auto(stored + "!!") == reference(stored + "!!")


def test_corrupt_record_degrades_to_a_correct_miss(
    rig: Rig,
    span_reference: SpanEncode,
    reference: Callable[[str], list[int]],
) -> None:
    directory = rig.store_path()
    first = _store(span_reference, directory)
    stored = _text(AUTO_MIN_BYTES + 10)
    assert first.encode_auto(stored) == reference(stored)

    (record_path,) = (directory / "entries").glob("*.rec")
    raw = bytearray(record_path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    record_path.write_bytes(bytes(raw))
    (directory / "index.json").unlink()

    second = _store(span_reference, directory)
    assert second.encode_auto(stored) == reference(stored)
    assert second.stats()["auto_misses"] == 1


def test_index_file_shape_is_stable_json(
    rig: Rig,
    span_reference: SpanEncode,
    reference: Callable[[str], list[int]],
) -> None:
    directory = rig.store_path()
    store = _store(span_reference, directory)
    stored = _text(AUTO_MIN_BYTES + 10)
    assert store.encode_auto(stored) == reference(stored)
    payload = json.loads((directory / "index.json").read_text())
    assert payload["format"] == index_module.INDEX_FORMAT
    (row,) = payload["entries"].values()
    assert row["bytes"] == len(stored.encode("utf-8"))
