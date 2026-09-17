"""Unit tests for the SGLang adapter that need neither sglang nor an artifact.

The toktier path itself is exercised end to end by the PLAN/184 smoke
(shadow check against the Hugging Face stream on a live server); these
tests pin the pure-Python contract: delegation, the session-key plumbing,
the special-token template derivation, and the manager attachment.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from toktier.integrations.sglang import (
    SESSION_KEY,
    AdapterConfig,
    SGLangToktierTokenizer,
    _derive_special_template,
    attach_to_tokenizer_manager,
    bind_session_key,
    session_key_from_request,
    wrap_tokenizer,
)


class _StubHf:
    """A tiny tokenizer: one id per character; optional BOS/EOS template."""

    name_or_path = "stub/model"
    eos_token_id = 999
    is_fast = True

    def __init__(self, bos: int | None = None, eos: int | None = None) -> None:
        self._bos, self._eos = bos, eos

    def _core(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def __call__(
        self, text: Any, add_special_tokens: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        if isinstance(text, str):
            return {"input_ids": self._one(text, add_special_tokens), "kwargs": kwargs}
        return {
            "input_ids": [self._one(t, add_special_tokens) for t in text],
            "kwargs": kwargs,
        }

    def _one(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = self._core(text)
        if add_special_tokens:
            ids = (
                ([self._bos] if self._bos is not None else [])
                + ids
                + ([self._eos] if self._eos is not None else [])
            )
        return ids

    def encode(
        self, text: str, add_special_tokens: bool = True, **kwargs: Any
    ) -> list[int]:
        return self._one(text, add_special_tokens)

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "".join(chr(i) for i in ids)


def test_config_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError):
        AdapterConfig(session_key="nope")
    with pytest.raises(ValueError):
        AdapterConfig(shadow="sometimes")
    with pytest.raises(ValueError):
        AdapterConfig(shadow_rate=1.5)


def test_config_from_env_maps_names(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "TOKTIER_SGLANG_SESSION_KEY": "contextvar",
        "TOKTIER_SGLANG_SHADOW": "1",
        "TOKTIER_SGLANG_SHADOW_RATE": "0.05",
        "TOKTIER_SGLANG_FAMILY": "qwen3_8b",
        "TOKTIER_SGLANG_EXPLAIN_EVERY": "10",
    }
    cfg = AdapterConfig.from_env(env)
    assert cfg.session_key == "contextvar"
    assert cfg.shadow == "sync"
    assert cfg.shadow_rate == 0.05
    assert cfg.family == "qwen3_8b"
    assert cfg.explain_every == 10
    assert AdapterConfig.from_env({}).session_key == "auto"


def test_special_template_derivation() -> None:
    assert _derive_special_template(_StubHf()) == ([], [], True)
    assert _derive_special_template(_StubHf(bos=1)) == ([1], [], True)
    assert _derive_special_template(_StubHf(bos=1, eos=2)) == ([1], [2], True)


def test_passthrough_wrapper_delegates_everything() -> None:
    hf = _StubHf(bos=1)
    w = wrap_tokenizer(hf, AdapterConfig(session_key="passthrough"))
    assert isinstance(w, SGLangToktierTokenizer)
    assert w.is_fast is True
    assert w.eos_token_id == 999
    assert w("ab")["input_ids"] == [1, 97, 98]
    assert w(["ab", "c"])["input_ids"] == [[1, 97, 98], [1, 99]]
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    assert w.decode([97]) == "a"
    assert w.stats()["delegated"] == 3
    assert wrap_tokenizer(w) is w


class _Req:
    def __init__(self, **kw: Any) -> None:
        self.session_id = None
        self.session_params = None
        self.cache_salt = None
        self.rid = None
        self.__dict__.update(kw)


def test_session_key_precedence() -> None:
    assert session_key_from_request(_Req(session_id="s1", cache_salt="c1")) == "s1"
    assert session_key_from_request(_Req(cache_salt="c1")) == "c1"
    assert (
        session_key_from_request(_Req(rid="conv7-turn3"), rid_separator="-turn")
        == "conv7"
    )
    assert session_key_from_request(_Req(rid="conv7-turn3")) is None
    # server-side continuation sends a delta, not a transcript: no key
    assert (
        session_key_from_request(_Req(session_id="s1", session_params={"id": "s1"}))
        is None
    )


def test_bind_session_key_restores() -> None:
    assert SESSION_KEY.get() is None
    with bind_session_key("k"):
        assert SESSION_KEY.get() == "k"
    assert SESSION_KEY.get() is None


def test_attach_to_manager_wraps_and_binds() -> None:
    hf = _StubHf()
    seen: dict[str, str | None] = {}

    class Manager:
        def __init__(self) -> None:
            self.tokenizer: Any = hf
            self.async_dynamic_batch_tokenizer = None
            self.rid_to_state: dict[str, Any] = {}

        async def _tokenize_one_request(self, obj: Any) -> Any:
            seen["key"] = SESSION_KEY.get()
            return self.tokenizer(obj.text)["input_ids"]

    assert attach_to_tokenizer_manager(
        Manager, AdapterConfig(session_key="passthrough")
    )
    assert (
        attach_to_tokenizer_manager(Manager, AdapterConfig(session_key="passthrough"))
        is False
    )
    m = Manager()
    assert isinstance(m.tokenizer, SGLangToktierTokenizer)
    ids = asyncio.run(
        m._tokenize_one_request(_Req(text="hi", session_id="conv", rid="r"))
    )
    assert ids == [104, 105]
    assert seen["key"] == "conv"
    assert SESSION_KEY.get() is None


# -- chunk-parallel encode wiring (0.2.10) ----------------------------------


class _Ids:
    def __init__(self, ids: tuple[int, ...]) -> None:
        self.ids = ids


class _FakeToktier:
    """Stands in for ``toktier.Tokenizer`` inside the wrapper."""

    family = "fake_family"

    class plan:  # mirrors the attribute the wrapper reads
        backend = "fast_cpu"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, text: str, **kwargs: object) -> _Ids:
        self.calls.append(dict(kwargs))
        return _Ids(tuple(ord(c) for c in text))

    def explain(self, summary: bool = False) -> dict[str, object]:
        return {"store": None, "fallback_counts": {}, "cpe": {"calls": 1}}

    def close(self) -> None:
        pass


class _Chunked:
    def __init__(self, text: str, k: int | None) -> None:
        self.ids = tuple(ord(c) for c in text)
        self.chunks = 2
        self.k_requested = k or 4
        self.serial_reason = None


def test_config_cpe_fields_from_env_and_validation() -> None:
    cfg = AdapterConfig.from_env(
        {"TOKTIER_SGLANG_CPE_MIN_CHARS": "4000", "TOKTIER_SGLANG_CPE_K": "4"}
    )
    assert cfg.cpe_min_chars == 4000 and cfg.cpe_k == 4
    assert AdapterConfig.from_env({}).cpe_min_chars == 0
    assert AdapterConfig.from_env({}).cpe_k == 0
    with pytest.raises(ValueError):
        AdapterConfig(cpe_min_chars=-1)
    with pytest.raises(ValueError):
        AdapterConfig(cpe_k=-1)


def test_cpe_serves_long_stateless_requests_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    seen: list[tuple[str, int | None]] = []

    def chunked(tok: object, text: str, *, k: int | None = None) -> _Chunked:
        assert tok is fake
        seen.append((text, k))
        return _Chunked(text, k)

    monkeypatch.setattr("toktier.cpe.chunked_encode", chunked)
    long_text = "x" * 20
    # "off": short stays on the stateless route, long takes CPE.
    w = wrap_tokenizer(
        _StubHf(), AdapterConfig(session_key="off", cpe_min_chars=10, cpe_k=4)
    )
    assert w.encode("short", add_special_tokens=False) == [ord(c) for c in "short"]
    assert fake.calls == [{"lookup": "off"}]
    assert w.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert seen == [(long_text, 4)]
    assert w.stats()["how"] == {"off": 1, "cpe": 1}
    assert w.stats()["cpe"] == {"calls": 1}
    # "contextvar": a keyed request stays on the session path whatever its
    # length; a keyless long one takes CPE (the table, since cpe_k is 0).
    fake.calls.clear()
    seen.clear()
    w2 = wrap_tokenizer(
        _StubHf(), AdapterConfig(session_key="contextvar", cpe_min_chars=10)
    )
    with bind_session_key("conv"):
        assert w2.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert fake.calls == [{"session": "conv"}]
    assert w2.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert seen == [(long_text, None)]
    assert w2.encode("short", add_special_tokens=False) == [ord(c) for c in "short"]
    assert fake.calls[-1] == {}
    assert w2.stats()["how"] == {"session": 1, "cpe": 1, "auto": 1}
    # "auto": content lookup by design; CPE never fires.
    fake.calls.clear()
    seen.clear()
    w3 = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="auto", cpe_min_chars=10))
    assert w3.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert seen == [] and fake.calls == [{}]
    # Off by default: nothing changes for a wrapper without the option.
    seen.clear()
    w4 = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="off"))
    assert w4.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert seen == []


def test_explicit_long_request_encoder_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    monkeypatch.setattr(
        "toktier.cpe.chunked_encode",
        lambda *a, **k: pytest.fail("cpe must not run behind an explicit hook"),
    )
    w = wrap_tokenizer(
        _StubHf(),
        AdapterConfig(
            session_key="off",
            long_request_encoder=lambda text: [1, 2, 3],
            long_request_min_chars=5,
            cpe_min_chars=5,
        ),
    )
    assert w.encode("x" * 8, add_special_tokens=False) == [1, 2, 3]
    assert w.stats()["how"] == {"long_request_encoder": 1}
