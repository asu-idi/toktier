"""The per-request check of the SGLang wrapper, one kind of change at a time.

After admission the supplied object can change. Before every accelerated
encode the wrapper compares the object's encode-affecting surface with the
admitted one (``_surface_unchanged``); on a difference the full comparison
runs and either latches passthrough or re-admits. Each test below makes one
change directly on the retained object (not through the wrapper, so the
stale mark does not help) and then makes a request: the supplied tokenizer
must answer, and the wrapper must be in passthrough with the differing
section named. Each part of the check is needed by at least one of these
tests: with any one part removed, some test here fails.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from toktier.integrations import sglang as sglang_module
from toktier.integrations.sglang import (
    AdapterConfig,
    SGLangToktierTokenizer,
    wrap_tokenizer,
)

from .test_sglang_adapter import _FakeToktier, _StubHf

tokenizers = pytest.importorskip("tokenizers")
AddedToken = tokenizers.AddedToken

#: The flags an added token carries besides its content, with their values
#: on the token the base tokenizer is admitted with.
_FLAGS = {
    "special": False,
    "normalized": False,
    "lstrip": False,
    "rstrip": False,
    "single_word": False,
}


def _base() -> _StubHf:
    """A stub whose backend has added tokens and a pre-tokenizer with a
    setting that can be changed in place."""
    from tokenizers import pre_tokenizers

    hf = _StubHf()
    backend = hf.backend_tokenizer
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.add_tokens([AddedToken("<a1>", **_FLAGS)])
    backend.add_special_tokens([AddedToken("<s1>", special=True, normalized=False)])
    return hf


class _FaceOf(_FakeToktier):
    """A fake toktier whose loader face is the given object's backend as it
    is now (the subject the wrapper admits it against)."""

    def __init__(self, hf: Any) -> None:
        super().__init__()
        face: str = hf.backend_tokenizer.to_str()

        class _Backend:
            @staticmethod
            def materialized_tokenizer_json() -> str:
                return face

        self._backend = cast("type[_FakeToktier._backend]", _Backend)


def _wrap(
    monkeypatch: pytest.MonkeyPatch, hf: Any, mode: str = "off", **cfg: Any
) -> SGLangToktierTokenizer:
    fake = _FaceOf(hf)
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    w: SGLangToktierTokenizer = wrap_tokenizer(
        hf, AdapterConfig(session_key=mode, **cfg)
    )
    assert w.identity["status"] == "equivalent", w.identity
    assert w.encode("ab <a1> c", add_special_tokens=False) == hf.encode(
        "ab <a1> c", add_special_tokens=False
    )
    first = "off" if mode == "off" else "auto"  # a keyless request
    assert w.stats()["how"] == {first: 1} and w.stats()["delegated"] == 0
    return w


def _other_model_same_shape(hf: Any) -> Any:
    """A tokenizer equal to ``hf``'s backend in everything but the model's
    vocabulary content (same size, same components, same added tokens)."""
    import json

    from tokenizers import Tokenizer

    document = json.loads(hf.backend_tokenizer.to_str())
    vocab = document["model"]["vocab"]
    vocab["d"] = vocab.pop("c")
    return Tokenizer.from_str(json.dumps(document))


def _bigger_model() -> Any:
    from tokenizers import models

    return models.BPE(
        vocab={"<unk>": 0, "a": 1, "b": 2, "ab": 3, "c": 4, "d": 5},
        merges=[("a", "b")],
        unk_token="<unk>",
    )


def _reflag(field: str) -> Callable[[Any], Any]:
    def change(hf: Any) -> Any:
        flags = dict(_FLAGS)
        flags[field] = not flags[field]
        return hf.backend_tokenizer.add_tokens([AddedToken("<a1>", **flags)])

    return change


def _set(obj: Callable[[Any], Any], name: str, value: Any) -> Callable[[Any], Any]:
    return lambda hf: setattr(obj(hf), name, value)


def _backend(hf: Any) -> Any:
    return hf.backend_tokenizer


def _replace_backend(make: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def change(hf: Any) -> None:
        hf.backend_tokenizer = make(hf)

    return change


def _processors() -> Any:
    from tokenizers import processors

    return processors


def _normalizers() -> Any:
    from tokenizers import normalizers

    return normalizers


def _pre_tokenizers() -> Any:
    from tokenizers import pre_tokenizers

    return pre_tokenizers


#: kind of change -> (the change, the section the full comparison names)
_CHANGES: dict[str, tuple[Callable[[Any], Any], str]] = {
    "a new added token": (
        lambda hf: hf.backend_tokenizer.add_tokens(["<zz_new>"]),
        "added_tokens",
    ),
    "a token the model already has": (
        lambda hf: hf.backend_tokenizer.add_tokens(["ab"]),
        "added_tokens",
    ),
    "a special token": (
        lambda hf: hf.backend_tokenizer.add_special_tokens(["<zz_sp>"]),
        "added_tokens",
    ),
    **{
        f"an existing added token with {field} changed": (
            _reflag(field),
            "added_tokens",
        )
        for field in _FLAGS
    },
    "add_special_tokens of an existing added token": (
        lambda hf: hf.backend_tokenizer.add_special_tokens(
            [AddedToken("<a1>", special=True, normalized=False)]
        ),
        "added_tokens",
    ),
    "the split flag on the object": (
        lambda hf: setattr(hf, "split_special_tokens", True),
        "split_special_tokens",
    ),
    "the split flag on the backend": (
        _set(_backend, "encode_special_tokens", True),
        "split_special_tokens",
    ),
    "the normalizer replaced": (
        lambda hf: setattr(
            hf.backend_tokenizer, "normalizer", _normalizers().Lowercase()
        ),
        "normalizer",
    ),
    "the pre-tokenizer replaced": (
        lambda hf: setattr(
            hf.backend_tokenizer, "pre_tokenizer", _pre_tokenizers().Whitespace()
        ),
        "pre_tokenizer",
    ),
    "the pre-tokenizer changed in place": (
        lambda hf: setattr(
            hf.backend_tokenizer.pre_tokenizer, "add_prefix_space", True
        ),
        "pre_tokenizer",
    ),
    "the post-processor replaced": (
        lambda hf: setattr(
            hf.backend_tokenizer,
            "post_processor",
            _processors().TemplateProcessing(single="$A $A"),
        ),
        "post_processor",
    ),
    "truncation": (lambda hf: hf.backend_tokenizer.enable_truncation(4), "truncation"),
    "padding": (lambda hf: hf.backend_tokenizer.enable_padding(length=8), "padding"),
    "the backend object replaced by another tokenizer": (
        _replace_backend(_other_model_same_shape),
        "model",
    ),
    "the model replaced by one of another size": (
        lambda hf: setattr(hf.backend_tokenizer, "model", _bigger_model()),
        "model",
    ),
}


@pytest.mark.parametrize("kind", sorted(_CHANGES))
def test_a_change_on_the_retained_object_is_noticed(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf = _base()
    w = _wrap(monkeypatch, hf)
    change, section = _CHANGES[kind]
    change(hf)
    text = "ab <a1> c <s1> ab"
    expected = hf.encode(text, add_special_tokens=False)
    assert w.encode(text, add_special_tokens=False) == expected
    assert w.stats()["mode"] == "passthrough", (kind, w.identity)
    assert w.toktier is None and w.stats()["delegated"] == 1
    assert section in w.identity["sections"], (kind, w.identity)


def test_an_equal_copy_of_the_backend_is_readmitted_after_one_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenizers import Tokenizer

    hf = _base()
    w = _wrap(monkeypatch, hf)
    calls: list[str] = []
    full = sglang_module.check_core_identity

    def counted(*args: Any) -> Any:
        verdict = full(*args)
        calls.append(verdict["status"])
        return verdict

    monkeypatch.setattr(sglang_module, "check_core_identity", counted)
    hf.backend_tokenizer = Tokenizer.from_str(hf.backend_tokenizer.to_str())
    for _ in range(3):
        assert w.encode("ab c", add_special_tokens=False) == [97, 98, 32, 99]
    assert calls == ["equivalent"]
    assert w.stats()["mode"] == "off" and w.stats()["how"] == {"off": 4}
    assert w.identity["status"] == "equivalent"


def test_an_unchanged_object_takes_no_full_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf = _base()
    w = _wrap(monkeypatch, hf)
    calls: list[Any] = []
    monkeypatch.setattr(
        sglang_module, "check_core_identity", lambda *a: calls.append(a)
    )
    for _ in range(5):
        w.encode("ab c", add_special_tokens=False)
    assert calls == [] and w.stats()["how"] == {"off": 6}


def test_added_token_equality_covers_the_content_and_each_flag() -> None:
    """The dictionary comparison of the added-token table rests on this: a
    tokenizers release that changes it turns this test red."""
    base = {"content": "<q>", **_FLAGS}
    assert AddedToken(**base) == AddedToken(**base)
    for field, other in (("content", "<r>"), *((f, True) for f in _FLAGS)):
        changed = dict(base)
        changed[field] = other
        assert AddedToken(**base) != AddedToken(**changed), field
        assert {7: AddedToken(**base)} != {7: AddedToken(**changed)}, field


# -- short requests: answered by the supplied tokenizer, without the check -----


def _count_checks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = sglang_module._surface_unchanged

    def counted(*args: Any) -> bool:
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(sglang_module, "_surface_unchanged", counted)
    return calls


def test_the_default_threshold_is_one_character_per_added_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf = _base()  # two added tokens
    w = _wrap(monkeypatch, hf)
    assert w.stats()["short_chars"] == 2 == len(
        hf.backend_tokenizer.get_added_tokens_decoder()
    )
    checks = _count_checks(monkeypatch)
    # below the threshold: the supplied tokenizer answers, and no check runs
    assert w.encode("a", add_special_tokens=False) == [97]
    assert w("b")["input_ids"] == hf("b")["input_ids"]
    assert checks == [] and w.stats()["how"] == {"off": 1, "short": 2}
    assert w.toktier is not None and w.toktier.calls == [{"lookup": "off"}]
    # at the threshold: the accelerated route, after the check
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    assert checks == [1] and w.stats()["how"] == {"off": 2, "short": 2}
    assert w.accelerates and w.stats()["delegated"] == 0


def test_the_setting_overrides_the_threshold_and_zero_turns_it_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert AdapterConfig.from_env({}).short_chars is None
    unset = AdapterConfig.from_env({"TOKTIER_SGLANG_SHORT_CHARS": ""})
    assert unset.short_chars is None
    assert AdapterConfig.from_env({"TOKTIER_SGLANG_SHORT_CHARS": "7"}).short_chars == 7
    for bad in ("-1", "many", "1.5"):
        with pytest.raises(ValueError, match="TOKTIER_SGLANG_SHORT_CHARS"):
            AdapterConfig.from_env({"TOKTIER_SGLANG_SHORT_CHARS": bad})
    with pytest.raises(ValueError, match="SHORT_CHARS"):
        AdapterConfig(short_chars=-1)
    w = _wrap(monkeypatch, _base(), short_chars=5)
    assert w.stats()["short_chars"] == 5
    w.encode("abcd", add_special_tokens=False)
    w.encode("abcde", add_special_tokens=False)
    assert w.stats()["how"] == {"off": 2, "short": 1}
    off = _wrap(monkeypatch, _base(), short_chars=0)
    off.encode("a", add_special_tokens=False)
    assert off.stats()["short_chars"] == 0 and off.stats()["how"] == {"off": 2}


def test_a_change_is_noticed_by_the_first_long_request_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf = _base()
    w = _wrap(monkeypatch, hf, short_chars=4)
    hf.backend_tokenizer.add_tokens(["<zz_new>"])
    for text in ("a", "ab", "abc"):  # short: the supplied tokenizer, by definition
        assert w.encode(text, add_special_tokens=False) == hf.encode(
            text, add_special_tokens=False
        )
    assert w.stats()["mode"] == "off" and w.stats()["how"]["short"] == 3
    assert w.encode("abcd", add_special_tokens=False) == hf.encode(
        "abcd", add_special_tokens=False
    )
    assert w.stats()["mode"] == "passthrough" and w.toktier is None
    assert w.identity["sections"] == ["added_tokens"]


def test_batches_are_decided_text_by_text_and_pairs_keep_their_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf = _base()
    w = _wrap(monkeypatch, hf, short_chars=3)
    rows = w(["a", "abc", "ab", "abcd"], add_special_tokens=False)["input_ids"]
    assert rows == [[97], [97, 98, 99], [97, 98], [97, 98, 99, 100]]
    assert w.stats()["how"] == {"off": 3, "short": 2}
    w("a", text_pair="b")  # a pair: delegated as before, not counted as short
    assert w.stats()["delegated"] == 1 and w.stats()["how"]["short"] == 2


def test_a_short_text_does_not_enter_the_session_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from toktier.integrations.sglang import bind_session_key

    hf = _base()
    w = _wrap(monkeypatch, hf, mode="contextvar", short_chars=3)
    assert w.toktier is not None
    with bind_session_key("conversation"):
        assert w.encode("ab", add_special_tokens=False) == [97, 98]
        assert w.toktier.calls == [{}]  # the session saw nothing
        assert w.encode("ab c", add_special_tokens=False) == [97, 98, 32, 99]
    assert w.toktier.calls == [{}, {"session": "conversation"}]
    assert w.stats()["how"] == {"auto": 1, "short": 1, "session": 1}


def test_an_async_short_request_is_answered_on_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    w = _wrap(monkeypatch, _base(), short_chars=3)
    assert asyncio.run(w.encode_async("ab", add_special_tokens=False)) == [97, 98]
    assert asyncio.run(w.encode_async("abc", add_special_tokens=False)) == [97, 98, 99]
    assert w.stats()["how"] == {"off": 2, "short": 1}


def test_a_short_text_does_not_go_to_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from pathlib import Path

    from toktier.integrations.sglang import encode_texts_with_pool
    from toktier.integrations.sglang_sched import SchedulerConfig

    here = str(Path(__file__).resolve().parent)
    hf = _base()
    fake = _FaceOf(hf)
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    pool = SchedulerConfig(
        workers=1, loader="_sched_stub:load", loader_path=here, start_timeout_s=120.0
    )
    cfg = AdapterConfig(session_key="off", scheduler=pool, short_chars=3)
    w = wrap_tokenizer(hf, cfg)
    try:
        assert w.scheduler is not None
        # the rule starts after one encode completed on the accelerated route
        assert w.encode("abcd", add_special_tokens=False) == [97, 98, 99, 100]
        assert asyncio.run(encode_texts_with_pool(w, ["ab", "abcd"])) == [
            [97, 98],
            [97, 98, 99, 100],
        ]
        assert w.encode("a", add_special_tokens=False) == [97]
        assert w.stats()["how"] == {"short": 2, "sched_free": 2}
        assert w.stats()["scheduler"]["submitted"] == 2
    finally:
        w.close()


def test_the_rule_starts_after_the_first_accelerated_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Until one encode has completed on the accelerated route, a short text
    takes that route, after the check, as it did before the rule existed
    (so a caller's first encode, whatever its length, opens the engine and
    the session store and reports what is wrong with them); from then on
    it is answered by the supplied tokenizer."""
    hf = _base()
    fake = _FaceOf(hf)
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off", short_chars=5))
    checks = _count_checks(monkeypatch)
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    assert checks == [1] and w.stats()["how"] == {"off": 1}
    assert fake.calls == [{"lookup": "off"}]
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    assert checks == [1] and w.stats()["how"] == {"off": 1, "short": 1}
    # An accelerated encode that raises does not start the rule: every
    # short call meets the same error, as before.
    broken = _FaceOf(hf)

    def refuse(text: str, **kwargs: object) -> Any:
        raise RuntimeError("the store cannot be read")

    broken.encode = refuse  # type: ignore[method-assign]
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: broken)
    w2 = wrap_tokenizer(hf, AdapterConfig(session_key="off", short_chars=5))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cannot be read"):
            w2.encode("ab", add_special_tokens=False)
    assert "short" not in w2.stats()["how"]
