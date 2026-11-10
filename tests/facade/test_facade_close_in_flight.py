"""``close()`` while a store call is in flight, and a callback whose tokenizer is gone.

The Python-owned entry store hands its callbacks to a native encoder, and
a store call holds the native store for its whole length, callbacks
included. ``close()`` may land inside such a call: from another thread,
or from inside one of the callbacks. It must not raise because the store
is busy, and it must not stop half-way: the store and the session-repair
adapter are released together once the call has ended, the adapter only
after the store. The call itself ends with ids equal to the reference or
with the tokenizer's ``RuntimeError("backend is closed")``.

A callback that outlives its tokenizer (reachable only through private
objects) is a miss for the store, never ids.

``close()`` may also land at a Python step of a store call, when the
native store is not held at that moment. Nothing native defers the
release then: the store's own record of the calls in flight does. The
call still ends on the store it started with, and the adapter stays
until that store has released.

These tests use the fake backends of ``test_fast_repair`` and need no
device.
"""

from __future__ import annotations

import gc
import random
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import pytest

from toktier.facade import api as facade_api
from toktier.facade.store import EntryStore

from .conftest import Rig
from .test_facade_lifetime import (
    _BASE,
    _DELTAS,
    _callback_store,
    _directory_files,
    _install,
    _reference,
)

_CLOSED = "backend is closed"


def _released(tokenizer: facade_api.Tokenizer, store: EntryStore) -> bool:
    return (
        store._store is None
        and store._encoder is None
        and tokenizer._session_repair is None
    )


def _outcome(live: Any, call: Any, text: str) -> str:
    """``equal``, ``closed`` or a description of anything else."""
    try:
        ids = call()
    except RuntimeError as error:
        return "closed" if str(error) == _CLOSED else f"RuntimeError: {error}"
    except Exception as error:  # every other outcome is a failure
        return f"{type(error).__name__}: {error}"
    return "equal" if list(ids) == _reference(live, text) else "UNEQUAL"


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "directory"])
@pytest.mark.parametrize(
    ("method", "when", "at_call"),
    [
        ("_state_encode", "before", 1),
        ("_state_encode", "after", 1),
        ("_gigatoken_append", "before", 1),
        ("_gigatoken_append", "after", 1),
        ("_gigatoken_append", "before", 2),
    ],
)
def test_close_from_inside_a_store_callback_returns_and_completes(
    rig: Rig,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    when: str,
    at_call: int,
    persistent: bool,
) -> None:
    """The deterministic form: ``close()`` while the native store is held."""
    live, options = _install(rig, monkeypatch, "gpu-jit")
    directory = rig.store_path() if persistent else None
    tokenizer = rig.tokenizer(store=directory, **options)
    original = getattr(tokenizer, method)
    seen: dict[str, object] = {"calls": 0, "closed": False, "close_error": None}

    def close_inside() -> None:
        seen["closed"] = True
        try:
            tokenizer.close()
        except Exception as error:  # recorded and asserted below
            seen["close_error"] = f"{type(error).__name__}: {error}"

    def wrapper(*args: Any) -> Any:
        calls = int(seen["calls"]) + 1  # type: ignore[call-overload]
        seen["calls"] = calls
        if calls == at_call and when == "before":
            close_inside()
        result = original(*args)
        if calls == at_call and when == "after":
            close_inside()
        return result

    # On the instance and before the store is built: the store's callback
    # looks the method up on the tokenizer, so the wrapper is what it calls.
    setattr(tokenizer, method, wrapper)
    outcomes: list[str] = []
    text = _BASE
    session: Any = None
    try:
        session = tokenizer.session(session_id="chat", text=text).__enter__()
    except RuntimeError as error:
        outcomes.append("closed" if str(error) == _CLOSED else f"RuntimeError: {error}")
    else:
        equal = list(session.ids) == _reference(live, text)
        outcomes.append("equal" if equal else "UNEQUAL")
    if session is not None:
        for delta in _DELTAS:
            grown = text + delta
            outcome = _outcome(live, lambda d=delta: session.append(d).all_ids, grown)
            outcomes.append(outcome)
            if outcome == "equal":
                text = grown
    assert seen["closed"], seen
    assert seen["close_error"] is None, seen
    assert set(outcomes) <= {"equal", "closed"}, outcomes

    # The call that held the store has ended: the release is complete
    # without a second ``close()``.
    store = _callback_store_or_released(tokenizer)
    assert _released(tokenizer, store)
    tokenizer.close()
    assert _released(tokenizer, store)
    report = tokenizer.explain()
    assert isinstance(report["store"], dict)
    assert isinstance(report["session_repair"], dict)

    if directory is not None:
        files = _directory_files(directory)
        reopened = rig.tokenizer(store=directory, **options)
        for candidate in (text, text + "".join(_DELTAS)):
            with reopened.session(session_id="chat", text=candidate) as resumed:
                assert list(resumed.ids) == _reference(live, candidate)
        reopened.close()
        assert set(files) <= set(_directory_files(directory))


def _callback_store_or_released(tokenizer: facade_api.Tokenizer) -> EntryStore:
    store = tokenizer._entry_store
    assert isinstance(store, EntryStore)
    return store


def test_close_from_another_thread_during_a_store_call(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated trials: ``close()`` lands while a worker is inside the store."""
    live, options = _install(rig, monkeypatch, "gpu-jit")
    rng = random.Random(20261110)
    failures: list[dict[str, object]] = []
    for trial in range(30):
        tokenizer = rig.tokenizer(**options)
        inside = threading.Event()
        original = tokenizer._state_encode

        def slow_encode(
            text: str, _original: Any = original, _inside: Any = inside
        ) -> Any:
            _inside.set()
            time.sleep(0.004)  # lets the closing thread run inside the call
            return _original(text)

        tokenizer._state_encode = slow_encode  # type: ignore[method-assign]
        answers: list[str] = []

        def work(
            _tokenizer: facade_api.Tokenizer = tokenizer, _answers: list[str] = answers
        ) -> None:
            for step in range(4):
                text = f"[{step}] " + _BASE
                _answers.append(
                    _outcome(
                        live,
                        lambda t=text, s=f"s{step}": (
                            _tokenizer.encode(t, session=s).ids
                        ),
                        text,
                    )
                )

        worker = threading.Thread(target=work)
        worker.start()
        assert inside.wait(10.0)
        time.sleep(rng.uniform(0.0, 0.006))
        close_error = None
        try:
            tokenizer.close()
        except Exception as error:  # recorded and asserted below
            close_error = f"{type(error).__name__}: {error}"
        worker.join(30.0)
        assert not worker.is_alive()
        store = tokenizer._entry_store
        assert isinstance(store, EntryStore)
        released = _released(tokenizer, store)
        tokenizer.close()
        released_after_second = _released(tokenizer, store)
        if (
            close_error is not None
            or not released
            or not released_after_second
            or not set(answers) <= {"equal", "closed"}
        ):
            failures.append(
                {
                    "trial": trial,
                    "close_error": close_error,
                    "released": released,
                    "released_after_second_close": released_after_second,
                    "answers": answers,
                }
            )
    assert failures == [], failures


_LONG = "content lookup 42 " * 300


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "directory"])
@pytest.mark.parametrize(
    "shape", ["session-hit", "session-append", "auto-hit", "auto-extension"]
)
def test_close_between_native_steps_defers_to_the_call_in_flight(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, shape: str, persistent: bool
) -> None:
    """``close()`` from a Python step of a store call (no native borrow held).

    The release waits for the call: right after ``close()`` returns, the
    native store the call started with and the session-repair adapter are
    both still present; the call ends with ids equal to the reference or
    with ``RuntimeError("backend is closed")``; when it has ended both
    are released without a second ``close()``.
    """
    live, options = _install(rig, monkeypatch, "gpu-jit")
    directory = rig.store_path() if persistent else None
    tokenizer = rig.tokenizer(store=directory, **options)
    session: Any = None
    if shape.startswith("session"):
        session = tokenizer.session(session_id="chat", text=_BASE).__enter__()
        text = _BASE if shape == "session-hit" else _BASE + _DELTAS[0]
    else:
        assert list(tokenizer.encode(_LONG).ids) == _reference(live, _LONG)
        text = _LONG if shape == "auto-hit" else _LONG + _DELTAS[0]
    store = _callback_store(tokenizer)
    native_store = store._store
    assert native_store is not None
    assert tokenizer._session_repair is not None
    original_touch = store._touch
    seen: dict[str, object] = {}

    def touch_then_close(entry: Any) -> None:
        original_touch(entry)
        if "close_error" in seen:
            return
        try:
            tokenizer.close()
        except Exception as error:  # recorded and asserted below
            seen["close_error"] = f"{type(error).__name__}: {error}"
        else:
            seen["close_error"] = None
        seen["store_kept"] = store._store is native_store
        seen["encoder_kept"] = store._encoder is not None
        seen["adapter_kept"] = tokenizer._session_repair is not None

    store._touch = touch_then_close  # type: ignore[method-assign]
    if session is not None and shape == "session-append":
        outcome = _outcome(live, lambda: session.append(_DELTAS[0]).all_ids, text)
    elif session is not None:
        outcome = _outcome(
            live, lambda: tokenizer.encode(text, session="chat").ids, text
        )
    else:
        outcome = _outcome(live, lambda: tokenizer.encode(text).ids, text)

    assert "close_error" in seen, "the store call did not reach a Python step"
    assert seen["close_error"] is None, seen
    # Deferred, not partial: the call in flight kept its store, and the
    # adapter that serves the store's callbacks stayed with it.
    assert seen["store_kept"] and seen["encoder_kept"], seen
    assert seen["adapter_kept"], seen
    assert outcome in {"equal", "closed"}, outcome

    # The call has ended: the release is complete without a second close().
    assert _released(tokenizer, store)
    assert not store._calls
    with pytest.raises(RuntimeError, match=_CLOSED):
        tokenizer.encode(text, session="chat")
    tokenizer.close()
    assert _released(tokenizer, store)
    report = tokenizer.explain()
    assert isinstance(report["store"], dict)
    assert isinstance(report["session_repair"], dict)


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "directory"])
def test_a_callback_whose_tokenizer_is_gone_is_a_miss_never_ids(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    """The ``repair_backend="reference"`` lane: the store alone can outlive
    its tokenizer there."""
    live, options = _install(rig, monkeypatch, "reference-repair")
    directory: Path | None = rig.store_path() if persistent else None
    tokenizer = rig.tokenizer(store=directory, **options)
    with tokenizer.session(session_id="chat", text=_BASE) as session:
        session.append(_DELTAS[0])
    long_text = "content lookup 42 " * 300
    assert list(tokenizer.encode(long_text).ids) == _reference(live, long_text)
    store = _callback_store(tokenizer)
    native_store, encoder = store._store, store._encoder
    assert native_store is not None
    key_id = store._key_id
    files_before = _directory_files(directory) if directory is not None else None
    degraded_before = store.stats()["degraded"]
    tokenizer_ref = weakref.ref(tokenizer)
    del tokenizer, session
    gc.collect()
    assert tokenizer_ref() is None

    assert store.encode_session("new", _BASE + " new") is None
    assert store.encode_session("chat", _BASE + _DELTAS[0] + _DELTAS[1]) is None
    assert store.encode_auto("another long text 7 " * 300) is None
    assert store.stats()["degraded"] == int(degraded_before) + 3  # type: ignore[call-overload]
    if directory is not None:
        assert _directory_files(directory) == files_before

    # The bare native objects: the failed callback is an error, not ids.
    with pytest.raises(RuntimeError, match="ReferenceError"):
        native_store.put(key_id, _BASE, encoder)
