"""Lifetime of a tokenizer whose Python-owned entry store uses callbacks.

When the entry store is not given the native session encoder -- a plan
that admits the GPU backend (JIT delivery keeps this Python adapter), or
an explicit ``repair_backend`` other than ``"auto"`` -- it hands Python
callbacks to a native encoder. The native object is not visible to the
cycle collector, so the callbacks must not hold the tokenizer strongly: a
tokenizer its owner dropped is released, with or without
:meth:`~toktier.Tokenizer.close`. These tests use the fake backends of
``test_fast_repair`` and need no device.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from typing import Any

import pytest

from toktier.backends.fast_cpu import FastCpuBackend
from toktier.facade import api as facade_api
from toktier.facade.store import EntryStore
from toktier.policy import RoutingPolicy
from toktier.repair.fastokens import FastokensFullRepair

from .conftest import Rig
from .test_fast_repair import (
    _CountingEngine,
    _FakeFastBackend,
    _FakeGpuBackend,
    _gpu_registry,
    _install_certified_probe,
    _install_gpu_probe,
    _registry,
    _spec,
)

_BASE = ("alpha 123 beta 456 " * 60) + "tail"
_DELTAS = (" appended text 789", " and one more 0", " last piece.")
_LANES = ("gpu-jit", "reference-repair", "fastokens-repair")


def _install(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, lane: str
) -> tuple[Any, dict[str, object]]:
    """Fakes for one callback lane; the live reference and the load options."""
    import tokenizers

    live = tokenizers.Tokenizer.from_file(str(rig.artifact_path))
    engine = _CountingEngine(live)
    fast_backend = _FakeFastBackend(rig.family, engine, live)
    monkeypatch.setattr(facade_api, "family_spec", lambda *_args: _spec(rig))
    monkeypatch.setattr(
        FastCpuBackend,
        "open",
        classmethod(lambda _cls, _artifact: fast_backend),
    )
    if lane == "gpu-jit":
        gpu_backend = _FakeGpuBackend(live)
        _install_gpu_probe(monkeypatch, rig, _gpu_registry(rig))
        monkeypatch.setattr(
            facade_api.Tokenizer,
            "_open_gpu_backend",
            lambda _self: gpu_backend,
        )
        return live, {"gpu_min_bytes": 64, "gpu_delivery": "jit"}
    _install_certified_probe(monkeypatch, rig, _registry(rig))
    if lane == "reference-repair":
        return live, {"repair_backend": "reference"}

    def reference_encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = live.encode(text, add_special_tokens=False)
        return (
            [int(value) for value in encoded.ids],
            [(int(start), int(end)) for start, end in encoded.offsets],
        )

    repair = FastokensFullRepair(
        spec=_spec(rig),
        engine=live,
        engine_version="0.3.1",
        engine_digest="e" * 64,
        hf_tokenizer=live,
        reference_encode=reference_encode,
    )
    monkeypatch.setattr(
        FastokensFullRepair,
        "open",
        classmethod(lambda _cls, **_keywords: repair),
    )
    return live, {
        "policy": RoutingPolicy.EXPERIMENTAL,
        "repair_backend": "fastokens",
    }


def _reference(live: Any, text: str) -> list[int]:
    return [int(value) for value in live.encode(text, add_special_tokens=False).ids]


def _callback_store(tokenizer: facade_api.Tokenizer) -> EntryStore:
    """The tokenizer's entry store, checked to be on the callback lane."""
    from toktier import _native

    store = tokenizer._entry_store
    assert store is not None
    assert isinstance(store._encoder, _native.CallbackEncoder)
    assert store._native_encoder is None
    return store


def _session_repair_of(tokenizer: facade_api.Tokenizer) -> object:
    return tokenizer._session_repair


def _directory_files(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("lane", _LANES)
@pytest.mark.parametrize("close_first", [True, False], ids=["closed", "not-closed"])
@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "directory"])
def test_dropped_tokenizer_and_its_store_are_collected(
    rig: Rig,
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
    close_first: bool,
    persistent: bool,
) -> None:
    live, options = _install(rig, monkeypatch, lane)
    if persistent:
        options["store"] = rig.store_path()
    tokenizer = rig.tokenizer(**options)
    text = _BASE
    with tokenizer.session(session_id="chat", text=text) as session:
        for delta in _DELTAS:
            text += delta
            update = session.append(delta)
            assert list(update.all_ids) == _reference(live, text)
    del session
    # A default call that reaches the content store takes the same lane.
    long_text = "content lookup 42 " * 300
    assert list(tokenizer.encode(long_text).ids) == _reference(live, long_text)
    store = _callback_store(tokenizer)
    tokenizer_ref = weakref.ref(tokenizer)
    store_ref = weakref.ref(store)
    del store
    if close_first:
        tokenizer.close()
    del tokenizer
    gc.collect()
    assert tokenizer_ref() is None
    assert store_ref() is None


@pytest.mark.parametrize("lane", _LANES)
def test_live_session_keeps_its_tokenizer_until_the_session_goes(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    live, options = _install(rig, monkeypatch, lane)
    tokenizer = rig.tokenizer(**options)
    opened = tokenizer.session(session_id="chat", text=_BASE)
    session = opened.__enter__()
    _callback_store(tokenizer)
    tokenizer_ref = weakref.ref(tokenizer)
    del tokenizer
    gc.collect()
    # The caller dropped its own reference; the session still answers.
    assert tokenizer_ref() is not None
    text = _BASE
    for delta in _DELTAS:
        text += delta
        update = session.append(delta)
        assert list(update.all_ids) == _reference(live, text)
    assert session.revision == len(_DELTAS)
    opened.__exit__(None, None, None)
    del opened, session
    gc.collect()
    assert tokenizer_ref() is None


def test_close_releases_the_store_and_the_repair_adapter(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closed tokenizer still held by its owner keeps no native store."""
    _, options = _install(rig, monkeypatch, "gpu-jit")
    tokenizer = rig.tokenizer(store=rig.store_path(), **options)
    with tokenizer.session(session_id="chat", text=_BASE) as session:
        session.append(_DELTAS[0])
    store = _callback_store(tokenizer)
    assert _session_repair_of(tokenizer) is not None
    tokenizer.close()
    assert store._store is None
    assert store._encoder is None
    assert _session_repair_of(tokenizer) is None
    assert store.encode_session("chat", _BASE) is None
    assert store.encode_auto(_BASE * 4) is None


def test_close_keeps_what_later_calls_see(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a call after ``close()`` sees, pinned as it was before the repair.

    Later work is refused, the store directory is not touched,
    ``explain()`` still reports the store counters and the session-repair
    facts as they were, a held session keeps its ids, text and revision,
    and a new tokenizer resumes the stored session from the directory.
    """
    live, options = _install(rig, monkeypatch, "gpu-jit")
    directory = rig.store_path()
    tokenizer = rig.tokenizer(store=directory, **options)
    opened = tokenizer.session(session_id="chat", text=_BASE)
    session = opened.__enter__()
    text = _BASE
    for delta in _DELTAS:
        text += delta
        session.append(delta)
    _callback_store(tokenizer)
    report_before = tokenizer.explain()
    files_before = _directory_files(directory)
    ids_before = tuple(session.ids)
    revision_before = session.revision
    assert revision_before == len(_DELTAS)

    tokenizer.close()
    tokenizer.close()

    assert _directory_files(directory) == files_before
    report_after = tokenizer.explain()
    for key in ("store", "session_repair", "state_encode"):
        assert report_after[key] == report_before[key], key
    assert tuple(session.ids) == ids_before
    assert session.text == text
    assert session.revision == revision_before
    assert tokenizer.store_session_revision("chat") == revision_before
    assert tokenizer.store_session_revision("other") is None
    with pytest.raises(RuntimeError, match="backend is closed"):
        session.append(" after close")
    with pytest.raises(RuntimeError, match="backend is closed"):
        tokenizer.encode(text, session="chat")
    with pytest.raises(RuntimeError, match="backend is closed"), tokenizer.session(
        session_id="chat"
    ):
        pass
    assert tuple(session.ids) == ids_before
    assert _directory_files(directory) == files_before

    reopened = rig.tokenizer(store=directory, **options)
    with reopened.session(session_id="chat", text=text) as resumed:
        assert list(resumed.ids) == _reference(live, text)
        assert resumed.revision == revision_before
        update = resumed.append(" resumed")
        assert list(update.all_ids) == _reference(live, text + " resumed")
    stats = reopened.explain()["store"]
    assert isinstance(stats, dict)
    assert stats["session_hits"] == 1
    reopened.close()
