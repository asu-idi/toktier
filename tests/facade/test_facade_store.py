"""Session entries, persistence, identity binding, and the cache budget."""

from __future__ import annotations

import json
import random
import string
from collections.abc import Callable

import pytest

from toktier import records
from toktier.errors import (
    SessionStateMismatch,
    StoreCorrupt,
    StoreFormatUnsupported,
)
from toktier.facade import store as store_module
from toktier.facade.store import EntryStore

from .conftest import TEST_FINGERPRINT, Rig, SpanEncode, build_rig, byte_level_document

_rng = random.Random(0xD1CE)


def _text(length: int) -> str:
    return "".join(_rng.choice(string.ascii_letters + " .,") for _ in range(length))


def _store_stats(tokenizer_report: dict[str, object]) -> dict[str, int]:
    stats = tokenizer_report["store"]
    assert isinstance(stats, dict)
    return stats


# ---------------------------------------------------------------- session --


def test_session_hit_append_and_overwrite_counters(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    tokenizer = rig.tokenizer(store=rig.store_path())
    base = _text(300)

    assert list(tokenizer.encode(base, session="s").ids) == reference(base)
    assert list(tokenizer.encode(base, session="s").ids) == reference(base)
    grown = base + " more"
    assert list(tokenizer.encode(grown, session="s").ids) == reference(grown)
    rewritten = "something else entirely"
    assert (
        list(tokenizer.encode(rewritten, session="s").ids)
        == reference(rewritten)
    )

    stats = _store_stats(tokenizer.explain())
    assert stats["session_misses"] == 1
    assert stats["session_hits"] == 1
    assert stats["session_appends"] == 1
    assert stats["session_overwrites"] == 1


def test_session_ids_survive_a_process_boundary(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    text = _text(500)
    first = rig.tokenizer(store=rig.store_path())
    assert list(first.encode(text, session="chat-1").ids) == reference(text)

    second = rig.tokenizer(store=rig.store_path())
    assert list(second.encode(text, session="chat-1").ids) == reference(text)
    grown = text + " continued"
    assert (
        list(second.encode(grown, session="chat-1").ids) == reference(grown)
    )
    stats = _store_stats(second.explain())
    assert stats["session_hits"] == 1
    assert stats["session_appends"] == 1
    assert stats["session_misses"] == 0


def test_arbitrary_session_id_spellings(rig: Rig) -> None:
    tokenizer = rig.tokenizer(store=rig.store_path())
    for session_id in ("", "a/b\\c", "sp ace", "\u00e9\u4e2d", "x" * 200):
        encoding = tokenizer.encode("payload " + session_id, session=session_id)
        assert len(encoding.ids) > 0


# ------------------------------------------------------------- identity --


def test_store_written_under_another_fingerprint_is_refused(
    rig: Rig,
) -> None:
    store_dir = rig.store_path()
    first = rig.tokenizer(store=store_dir)
    first.encode("hello", session="s")

    document = byte_level_document()
    document["decoder"]["trim_offsets"] = False  # different artifact bytes
    other = build_rig(rig.base / "other", document=document)
    tokenizer = other.tokenizer(store=store_dir)
    with pytest.raises(SessionStateMismatch) as caught:
        tokenizer.encode("hello", session="s")
    assert caught.value.code == "SESSION_STATE_MISMATCH"


def test_unknown_store_meta_format_is_refused(
    rig: Rig, span_reference: SpanEncode
) -> None:
    directory = rig.store_path()
    directory.mkdir()
    (directory / "meta.json").write_text(
        json.dumps({"format": 99, "fingerprint": TEST_FINGERPRINT.hex()})
    )
    with pytest.raises(StoreFormatUnsupported):
        EntryStore(
            fingerprint=TEST_FINGERPRINT,
            encode=span_reference,
            directory=directory,
        )


# ------------------------------------------------------------ records --


def test_persisted_records_decode_and_carry_the_stream(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    tokenizer = rig.tokenizer(store=rig.store_path())
    text = _text(700)
    tokenizer.encode(text, session="chat")
    tokenizer.encode(text + " extended", session="chat")

    (record_path,) = (rig.store_path() / "entries").glob("*.rec")
    view = records.decode_record(record_path.read_bytes())
    assert view.witness_category == 0
    assert view.stable_prefix_byte_length == 0
    assert view.text_tail == text + " extended"
    assert view.ids == reference(text + " extended")
    assert view.session_revision == 1

    tampered = bytearray(record_path.read_bytes())
    tampered[-1] ^= 0x01
    with pytest.raises(StoreCorrupt):
        records.decode_record(bytes(tampered))

    newer = bytearray(record_path.read_bytes())
    newer[8:10] = (2).to_bytes(2, "little")
    with pytest.raises(StoreFormatUnsupported):
        records.decode_record(bytes(newer))


def test_corrupt_session_record_degrades_to_overwrite(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    text = _text(600)
    first = rig.tokenizer(store=rig.store_path())
    first.encode(text, session="chat")

    (record_path,) = (rig.store_path() / "entries").glob("*.rec")
    record_path.write_bytes(b"garbage")

    second = rig.tokenizer(store=rig.store_path())
    grown = text + " more"
    assert list(second.encode(grown, session="chat").ids) == reference(grown)
    # The refreshed record is valid again.
    view = records.decode_record(record_path.read_bytes())
    assert view.text_tail == grown


# --------------------------------------------------------- cache budget --


def test_tiny_budget_evicts_but_persistent_store_still_serves(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    tokenizer = rig.tokenizer(store=rig.store_path(), cache_budget_bytes=0)
    first = _text(400)
    second = _text(400)
    assert list(tokenizer.encode(first, session="a").ids) == reference(first)
    assert list(tokenizer.encode(second, session="b").ids) == reference(second)
    stats = _store_stats(tokenizer.explain())
    assert stats["entries_evicted"] >= 1

    # Entry "a" was evicted from memory; the record reloads and appends.
    grown = first + " again"
    assert list(tokenizer.encode(grown, session="a").ids) == reference(grown)
    stats = _store_stats(tokenizer.explain())
    assert stats["session_appends"] == 1


def test_tiny_budget_in_memory_store_stays_correct(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    tokenizer = rig.tokenizer(cache_budget_bytes=0)
    first = _text(400)
    second = _text(400)
    assert list(tokenizer.encode(first, session="a").ids) == reference(first)
    assert list(tokenizer.encode(second, session="b").ids) == reference(second)
    # "a" is gone (no persistence); the re-encode is a counted miss.
    assert list(tokenizer.encode(first, session="a").ids) == reference(first)
    stats = _store_stats(tokenizer.explain())
    assert stats["session_misses"] >= 2
    assert stats["entries_evicted"] >= 1


def test_native_capacity_churn_never_changes_ids(
    rig: Rig,
    reference: Callable[[str], list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store_module, "_NATIVE_MAX_SESSIONS", 1)
    tokenizer = rig.tokenizer(store=rig.store_path())
    first = _text(300)
    second = _text(300)
    assert list(tokenizer.encode(first, session="a").ids) == reference(first)
    assert list(tokenizer.encode(second, session="b").ids) == reference(second)
    grown = first + " back again"
    assert list(tokenizer.encode(grown, session="a").ids) == reference(grown)
