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
from typing import Any

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
        face = hf.backend_tokenizer.to_str()

        class _Backend:
            @staticmethod
            def materialized_tokenizer_json() -> str:
                return face

        self._backend = _Backend


def _wrap(
    monkeypatch: pytest.MonkeyPatch, hf: Any, mode: str = "off", **cfg: Any
) -> SGLangToktierTokenizer:
    fake = _FaceOf(hf)
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    w = wrap_tokenizer(hf, AdapterConfig(session_key=mode, **cfg))
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

