"""Session entries, persistence, identity binding, and the cache budget."""

from __future__ import annotations

import json
import os
import random
import stat
import string
from collections.abc import Callable
from pathlib import Path

import pytest

from toktier import records
from toktier.errors import (
    SessionStateMismatch,
    StoreCorrupt,
    StoreFormatUnsupported,
    UnsupportedConfig,
)
from toktier.facade import store as store_module
from toktier.facade.recovery import RecoveryBinding
from toktier.facade.store import EntryStore
from toktier.paths import FILE_MODE
from toktier.policy import BACKEND_REFERENCE

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


def test_native_reference_append_replaces_the_seed_ledger(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    tokenizer = rig.tokenizer(store=rig.store_path(), device="cpu")
    assert tokenizer.plan.fallback_chain == (BACKEND_REFERENCE,)
    base = "native café reference seed"
    grown = base + " appended 中"

    assert list(tokenizer.encode(base, session="ledger").ids) == reference(base)
    seeded = tokenizer.explain()["runtime_policy"]
    assert isinstance(seeded, dict)
    assert seeded["request_path"] == "rust_native"
    assert seeded["last_execution"] == {
        "input_bytes": len(base.encode("utf-8")),
        "selected_start": BACKEND_REFERENCE,
        "executed_backend": BACKEND_REFERENCE,
        "source": "state_encode",
        "path": "hf_no_certified_span_bridge",
    }

    assert list(tokenizer.encode(grown, session="ledger").ids) == reference(grown)
    runtime = tokenizer.explain()["runtime_policy"]
    assert isinstance(runtime, dict)
    assert runtime["execution_counts"] == {BACKEND_REFERENCE: 2}
    assert runtime["last_execution"] == {
        "input_bytes": len(grown.encode("utf-8")),
        "selected_start": BACKEND_REFERENCE,
        "executed_backend": BACKEND_REFERENCE,
        "path": "native_hf_full_reencode",
    }

    assert list(tokenizer.encode(grown, session="ledger").ids) == reference(grown)
    after_hit = tokenizer.explain()["runtime_policy"]
    assert isinstance(after_hit, dict)
    assert after_hit["execution_counts"] == runtime["execution_counts"]
    assert after_hit["last_execution"] == runtime["last_execution"]


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


def test_persistent_write_uses_incremental_recovery_material(
    rig: Rig,
    reference: Callable[[str], list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binding persistence must not invoke the one-shot history scanner."""

    def forbidden_create(*_args: object, **_kwargs: object) -> RecoveryBinding:
        raise AssertionError("RecoveryBinding.create rescanned full history")

    monkeypatch.setattr(RecoveryBinding, "create", classmethod(forbidden_create))
    tokenizer = rig.tokenizer(store=rig.store_path())
    base = _text(12_000) + " \u4f60\u597d"
    grown = base + " incremental \U0001f680"

    assert list(tokenizer.encode(base, session="hot-path").ids) == reference(base)
    assert list(tokenizer.encode(grown, session="hot-path").ids) == reference(grown)
    (binding_path,) = (rig.store_path() / "entries").glob("*.binding")
    binding = RecoveryBinding.from_bytes(binding_path.read_bytes())
    assert binding.index_entry.byte_length == len(grown.encode())

    # The one-call Rust runtime intentionally no longer exposes its session
    # handles through the legacy Python EntryStore.  Verify the persisted
    # material itself: it must bind the current native record and exact text,
    # while the monkeypatch above proves that the Python one-shot constructor
    # did not create it.
    (record_path,) = (rig.store_path() / "entries").glob("*.rec")
    view = records.decode_record(record_path.read_bytes())
    assert binding.matches(grown.encode(), view.curr_block_hash)


def test_native_recovery_binding_is_byte_identical_to_python_golden(
    rig: Rig,
) -> None:
    tokenizer = rig.tokenizer(store=rig.store_path())
    text = _text(14_000) + " boundary 你好 🚀"
    tokenizer.encode(text, session="golden")

    entries = rig.store_path() / "entries"
    (record_path,) = entries.glob("*.rec")
    (binding_path,) = entries.glob("*.binding")
    view = records.decode_record(record_path.read_bytes())
    expected = RecoveryBinding.create(
        text.encode("utf-8"), view.curr_block_hash
    ).to_bytes()
    assert binding_path.read_bytes() == expected


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


def test_atomic_write_syncs_payload_before_rename_and_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entry files follow the write-fsync-rename-fsync discipline.

    The payload has to be durable before the rename publishes it, and
    the directory entry is flushed afterwards -- the same discipline as
    the artifact store's installer, so a crash never leaves a torn but
    visible record. The recording wrappers call through, so the write
    is also checked for real.
    """
    events: list[tuple[str, str]] = []
    descriptor_paths: dict[int, str] = {}
    real_open = os.open
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_open(path: str | os.PathLike[str], flags: int) -> int:
        descriptor = real_open(path, flags)
        descriptor_paths[descriptor] = str(path)
        return descriptor

    def recording_fsync(descriptor: int) -> None:
        events.append(("fsync", descriptor_paths.get(descriptor, "<unknown>")))
        real_fsync(descriptor)

    def recording_replace(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        events.append(("replace", str(destination)))
        real_replace(source, destination)

    target = tmp_path / "durable.rec"
    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    store_module._atomic_write(target, b"payload")
    monkeypatch.undo()

    assert target.read_bytes() == b"payload"
    assert stat.S_IMODE(target.stat().st_mode) == FILE_MODE
    assert [kind for kind, _ in events] == ["fsync", "replace", "fsync"]
    payload_sync, publish, directory_sync = events
    assert payload_sync[1].endswith(".tmp")
    assert publish[1] == str(target)
    assert directory_sync[1] == str(tmp_path)


# ------------------------------------------------- session context manager --


def test_session_context_manager_reports_exact_updates(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    """The frozen shape of api.md Section 5, over the shipped facade."""
    tokenizer = rig.tokenizer(store=rig.store_path())
    turns = [_text(300), " and then " + _text(120), " " + _text(40)]

    with tokenizer.session(session_id="conversation") as session:
        assert session.session_id == "conversation"
        assert list(session.ids) == []
        accumulated = ""
        previous: list[int] = []
        for turn in turns:
            update = session.append(turn)
            accumulated += turn
            # The splice invariant, and the correctness invariant under it.
            assert list(update.all_ids) == (
                previous[: update.replace_from] + list(update.replacement_ids)
            )
            assert list(update.all_ids) == reference(accumulated)
            assert list(session.ids) == reference(accumulated)
            assert session.text == accumulated
            previous = list(update.all_ids)

        # An append is a write, and the durable store owns the count.
        assert session.revision >= len(turns)
        assert session.final_ids(add_special_tokens=False) == list(session.ids)


def test_session_context_manager_serves_a_reopened_conversation(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    """State outlives the block: leaving it is not a delete."""
    store = rig.store_path()
    first = rig.tokenizer(store=store)
    opening = _text(300)
    with first.session(session_id="kept") as session:
        session.append(opening)
    first.close()

    second = rig.tokenizer(store=store)
    with second.session(session_id="kept", text=opening) as session:
        assert list(session.ids) == reference(opening)
        update = session.append(" continued")
    assert list(update.all_ids) == reference(opening + " continued")
    # Resuming kept the prefix: the append spliced rather than re-encoded.
    assert update.replace_from > 0
    second.close()


def test_session_context_manager_refuses_a_second_store(rig: Rig) -> None:
    """The store is bound at load; a different one is refused, not ignored."""
    tokenizer = rig.tokenizer(store=rig.store_path())

    with (
        pytest.raises(UnsupportedConfig) as raised,
        tokenizer.session(store=rig.store_path("elsewhere")),
    ):
        pass  # pragma: no cover - the manager refuses before yielding

    assert raised.value.code == "UNSUPPORTED_CONFIG"
    assert raised.value.details["option"] == "store"

    # Repeating the bound directory is accepted.
    with tokenizer.session(store=rig.store_path()) as session:
        assert session.append("hello").replace_from == 0


def test_an_unnamed_session_names_itself(rig: Rig) -> None:
    """Without a store the session is in-memory, and still identifiable."""
    tokenizer = rig.tokenizer()

    with tokenizer.session() as first, tokenizer.session() as second:
        assert first.session_id != second.session_id
        first.append("one")
        # Nothing durable holds it, so the revision is this object's count.
        assert first.revision == 1
