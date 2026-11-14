"""The SGLang adapter's lifecycle: what the wrapper says about itself, when
it gives way to the supplied tokenizer, and how its pool and its encoder
state are kept and closed.

None of these tests asks for different ids: each one checks that a call
returns what the supplied tokenizer (or the stub it stands for) returns,
whatever the wrapper did around it.
"""

from __future__ import annotations

from typing import Any

import pytest

from toktier.integrations.sglang import (
    AdapterConfig,
    SGLangToktierTokenizer,
    wrap_tokenizer,
)

from .test_sglang_adapter import _FakeToktier, _StubHf

# -- the wrapper says whether it accelerates -------------------------------


class _ExternalToktier(_FakeToktier):
    """A family toktier knows only as external: the reference engine alone."""

    family = "external_0123456789abcdef"

    class plan:
        backend = "hf"


def test_accelerates_is_true_on_an_accelerated_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="off"))
    assert w.accelerates is True
    assert w.stats()["accelerates"] is True
    assert "accelerates=True" in repr(w)
    # is_fast keeps the Hugging Face meaning
    assert w.is_fast is True
    assert w.encode("ab", add_special_tokens=False) == [97, 98]


def test_accelerates_is_false_in_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="passthrough"))
    assert w.accelerates is False
    assert w.stats()["accelerates"] is False
    assert "accelerates=False" in repr(w)
    assert w.is_fast is True
    # a supplied object toktier declines: passthrough for good, not accelerated
    drifted = _StubHf()
    drifted.backend_tokenizer.add_tokens(["toktierprobeunique"])
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: _FakeToktier()
    )
    w2 = wrap_tokenizer(drifted, AdapterConfig(session_key="off"))
    assert w2.identity["status"] == "mismatch" and w2.accelerates is False


def test_accelerates_is_false_for_an_external_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = _ExternalToktier()
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: external
    )
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="auto"))
    # admitted and answering through toktier, but on the reference engine alone
    assert w.identity["status"] == "equivalent"
    assert w.planned_backend == "hf" and w.toktier is external
    assert w.accelerates is False
    assert w.stats()["accelerates"] is False
    assert "accelerates=False" in repr(w)
    assert w.encode("ab", add_special_tokens=False) == [97, 98]


def test_accelerates_follows_a_later_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    hf = _StubHf()
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off"))
    assert w.accelerates is True
    w.add_tokens(["toktierprobeunique"])
    assert w.encode("ab", add_special_tokens=False) == hf.encode(
        "ab", add_special_tokens=False
    )
    assert w.stats()["accelerates"] is False
    assert not w.accelerates


def test_the_start_up_record_carries_accelerates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import json

    log = tmp_path / "timing.jsonl"
    external = _ExternalToktier()
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: external
    )
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="off", timing_log=str(log)))
    w.flush()
    first = json.loads(log.read_text().splitlines()[0])
    assert first["kind"] == "init" and first["accelerates"] is False
    w.close()
