"""The Python-owned entry store writes no block chain.

This owner finds stored prefixes through its content index and never
reads the native store's sealed-block map (no ``lookup``, no node export),
so its native store is built with ``node_tail_cap_bytes=0``: no node is
created, and ids, seals, store counters and append paths are what the
shipped cap gives. The tests drive the store through the facade on a
plan that admits a fake GPU backend under JIT delivery (the lane that
builds this owner) and need no device.
"""

from __future__ import annotations

from typing import Any

import pytest

from toktier import _native
from toktier.facade import api as facade_api
from toktier.facade.store import EntryStore

from .conftest import Rig
from .test_facade_lifetime import _install, _reference

#: The value a native store takes when the owner passes nothing.
_SHIPPED_NODE_TAIL_CAP = 65536

_PIECE = "alpha 123 beta 456 gamma 789 delta. " * 25


def _traffic(tokenizer: facade_api.Tokenizer, live: Any) -> list[bool]:
    """Sessions past one block, default calls first, repeated and extended."""
    equal: list[bool] = []
    for index in range(3):
        text = f"[{index}] " + _PIECE
        with tokenizer.session(session_id=f"chat{index}", text=text) as session:
            equal.append(list(session.ids) == _reference(live, text))
            for step in range(12):
                delta = f" turn {step} of {index}. " + _PIECE[: 300 + 97 * step]
                text += delta
                update = session.append(delta)
                equal.append(list(update.all_ids) == _reference(live, text))
        assert len(text) > 2 * 4096, len(text)
    for index in range(3):
        text = f"<{index}> " + _PIECE * 6
        assert len(text.encode("utf-8")) >= 4096
        for candidate in (text, text, text + " and an extension 42."):
            ids = list(tokenizer.encode(candidate).ids)
            equal.append(ids == _reference(live, candidate))
    return equal


def _native_store(tokenizer: facade_api.Tokenizer) -> Any:
    store = tokenizer._entry_store
    assert isinstance(store, EntryStore)
    assert isinstance(store._encoder, _native.CallbackEncoder)
    assert store._store is not None
    return store._store


def _facts(tokenizer: facade_api.Tokenizer) -> dict[str, object]:
    counters = tokenizer.explain()["store"]
    assert isinstance(counters, dict)
    native_stats = _native_store(tokenizer).stats()
    return {
        "counters": {k: v for k, v in counters.items() if k != "repair"},
        "seals": native_stats["seals"],
        "sealed_tokens": native_stats["sealed_tokens"],
    }


def test_python_owned_store_creates_no_block_nodes(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    live, options = _install(rig, monkeypatch, "gpu-jit")
    tokenizer = rig.tokenizer(**options)
    equal = _traffic(tokenizer, live)
    assert all(equal), equal
    assert _native_store(tokenizer).stats()["node_count"] == 0
    tokenizer.close()


def test_no_block_nodes_leaves_ids_and_counters_as_the_shipped_cap(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    live, options = _install(rig, monkeypatch, "gpu-jit")
    product = rig.tokenizer(**options)
    product_equal = _traffic(product, live)
    product_facts = _facts(product)
    assert _native_store(product).stats()["node_count"] == 0
    product.close()

    original = _native.SessionStore

    def shipped_cap(*args: Any, **keywords: Any) -> Any:
        keywords["node_tail_cap_bytes"] = _SHIPPED_NODE_TAIL_CAP
        return original(*args, **keywords)

    monkeypatch.setattr(_native, "SessionStore", shipped_cap)
    control = rig.tokenizer(**options)
    control_equal = _traffic(control, live)
    control_facts = _facts(control)
    # The traffic does build a chain under the shipped cap.
    assert _native_store(control).stats()["node_count"] > 0
    control.close()

    assert product_equal == control_equal
    assert all(product_equal), product_equal
    assert product_facts == control_facts
    counters = product_facts["counters"]
    assert isinstance(counters, dict)
    assert counters["session_appends"] > 0
    assert counters["auto_hits"] > 0
    assert counters["auto_appends"] > 0
