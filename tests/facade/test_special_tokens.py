"""``add_special_tokens=True`` on the facade: classifier, self-check, serving.

``True`` means the post-processor of the reference document. Faces whose
post-processor adds nothing (``identity``) or one special-token id in
front (``prefix``) are answered by the facade over the core stream of the
plain routed path; every other shape keeps the reference route. Every
id-level judgment below compares with the oracle package encoding the
same document with ``add_special_tokens=True``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from toktier.errors import UnsupportedConfig
from toktier.facade.special_tokens import (
    KIND_IDENTITY,
    KIND_PREFIX,
    KIND_REFERENCE,
    classify_postprocessor,
    plan_special_tokens,
    self_check_probes,
)

from .conftest import Rig, build_rig, byte_level_document

_A = {"Sequence": {"id": "A", "type_id": 0}}
_B = {"Sequence": {"id": "B", "type_id": 1}}


def _special(name: str, type_id: int = 0) -> dict[str, Any]:
    return {"SpecialToken": {"id": name, "type_id": type_id}}


def _template(
    single: list[dict[str, Any]],
    tokens: dict[str, list[int]] | None = None,
    pair: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "type": "TemplateProcessing",
        "single": single,
        "pair": pair if pair is not None else [_A, _B],
        "special_tokens": {
            name: {"id": name, "ids": ids, "tokens": [name] * len(ids)}
            for name, ids in (tokens or {}).items()
        },
    }


_BYTE_LEVEL = {
    "type": "ByteLevel",
    "add_prefix_space": False,
    "trim_offsets": True,
    "use_regex": True,
}
_PREFIX = _template([_special("<s>"), _A], {"<s>": [256]})


def _document(post_processor: object = None, **extra: object) -> dict[str, Any]:
    document = byte_level_document()
    document["post_processor"] = copy.deepcopy(post_processor)
    document["added_tokens"] = [
        {
            "id": 256,
            "content": "<s>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    ]
    document.update(extra)
    return document


# -- classifier ------------------------------------------------------------


@pytest.mark.parametrize(
    ("post_processor", "expected"),
    [
        (None, (KIND_IDENTITY, [])),
        (_BYTE_LEVEL, (KIND_IDENTITY, [])),
        (_template([_A]), (KIND_IDENTITY, [])),
        (_PREFIX, (KIND_PREFIX, [256])),
        (
            {"type": "Sequence", "processors": [_BYTE_LEVEL, _PREFIX]},
            (KIND_PREFIX, [256]),
        ),
        (
            {"type": "Sequence", "processors": [_PREFIX, _BYTE_LEVEL]},
            (KIND_PREFIX, [256]),
        ),
        ({"type": "Sequence", "processors": [_BYTE_LEVEL]}, (KIND_IDENTITY, [])),
        ({"type": "Sequence", "processors": []}, (KIND_IDENTITY, [])),
    ],
)
def test_classifier_admits_identity_and_single_prefix_shapes(
    post_processor: object, expected: tuple[str, list[int]]
) -> None:
    assert classify_postprocessor(_document(post_processor)) == expected


_REFERENCE_SHAPES: dict[str, object] = {
    "suffix": _template([_A, _special("</s>")], {"</s>": [2]}),
    "both_sides": _template(
        [_special("<s>"), _A, _special("</s>")], {"<s>": [1], "</s>": [2]}
    ),
    "two_prefix_tokens": _template(
        [_special("<s>"), _special("<t>"), _A], {"<s>": [1], "<t>": [3]}
    ),
    "repeated_sequence": _template([_A, _A]),
    "empty_single": _template([]),
    "prefix_with_two_ids": _template([_special("<s>"), _A], {"<s>": [1, 2]}),
    "prefix_without_ids": _template([_special("<s>"), _A], {"<s>": []}),
    "prefix_id_above_u32": _template([_special("<s>"), _A], {"<s>": [2**32]}),
    "prefix_id_negative": _template([_special("<s>"), _A], {"<s>": [-1]}),
    "prefix_type_id_one": _template([_special("<s>", 1), _A], {"<s>": [1]}),
    "missing_special_entry": _template([_special("<s>"), _A], {}),
    "bert": {
        "type": "BertProcessing",
        "sep": ["[SEP]", 102],
        "cls": ["[CLS]", 101],
    },
    "roberta": {
        "type": "RobertaProcessing",
        "sep": ["</s>", 2],
        "cls": ["<s>", 0],
        "trim_offsets": True,
        "add_prefix_space": False,
    },
    "unknown_type": {"type": "SomethingNew"},
    "not_a_mapping": ["ByteLevel"],
    # A template after the prefix member would receive two encodings and
    # run its pair form (the second review lane's specimen T1-a).
    "two_templates": {
        "type": "Sequence",
        "processors": [_PREFIX, _template([_A], pair=[_special("<s>"), _A, _B])],
    },
    "nested_sequence": {
        "type": "Sequence",
        "processors": [{"type": "Sequence", "processors": [_PREFIX]}],
    },
    "sequence_with_bert": {
        "type": "Sequence",
        "processors": [
            _BYTE_LEVEL,
            {"type": "BertProcessing", "sep": ["[SEP]", 2], "cls": ["[CLS]", 1]},
        ],
    },
}


@pytest.mark.parametrize("name", sorted(_REFERENCE_SHAPES))
def test_classifier_sends_every_other_shape_to_the_reference(name: str) -> None:
    assert classify_postprocessor(_document(_REFERENCE_SHAPES[name])) == (
        KIND_REFERENCE,
        [],
    )


def test_classifier_looks_the_special_token_up_by_the_piece_id() -> None:
    processor = _template([_special("<s>"), _A], {"<s>": [256]})
    processor["special_tokens"] = {
        "<x>": {"id": "<s>", "ids": [256], "tokens": ["<s>"]}
    }
    assert classify_postprocessor(_document(processor))[0] == KIND_REFERENCE
    processor["special_tokens"] = {
        "<s>": {"id": "<x>", "ids": [256], "tokens": ["<s>"]}
    }
    assert classify_postprocessor(_document(processor))[0] == KIND_REFERENCE


def test_classifier_rejects_boolean_ids() -> None:
    processor = _template([_special("<s>"), _A], {"<s>": [1]})
    processor["special_tokens"]["<s>"]["ids"] = [True]
    assert classify_postprocessor(_document(processor))[0] == KIND_REFERENCE


@pytest.mark.parametrize("section", ["truncation", "padding"])
def test_classifier_refuses_rewriting_sections(section: str) -> None:
    document = _document(None, **{section: {"max_length": 8}})
    assert classify_postprocessor(document) == (KIND_REFERENCE, [])


# -- self-check ------------------------------------------------------------


def test_self_check_probes_hold_an_added_token_and_non_ascii_text() -> None:
    probes = self_check_probes(_document(_PREFIX))
    assert "<s>" in probes
    assert any("<s>" in probe and probe != "<s>" for probe in probes)
    assert any(not probe.isascii() for probe in probes)
    assert "" in probes


def _oracle_encode(document: dict[str, Any]) -> Callable[[str, bool], list[int]]:
    import json

    import tokenizers

    handle = tokenizers.Tokenizer.from_str(json.dumps(document))
    return lambda text, special: list(
        handle.encode(text, add_special_tokens=special).ids
    )


def test_self_check_demotes_a_kind_the_engine_contradicts() -> None:
    document = _document(_PREFIX)
    honest = plan_special_tokens(document, _oracle_encode(document))
    assert honest.kind == KIND_PREFIX
    assert honest.prefix_ids == (256,)
    assert honest.self_check["passed"] is True

    def wrong(text: str, special: bool) -> Sequence[int]:
        core = [ord(ch) % 256 for ch in text]
        return [256, *core, 9] if special else core

    demoted = plan_special_tokens(document, wrong)
    assert demoted.kind == KIND_REFERENCE
    assert demoted.document_kind == KIND_PREFIX
    assert demoted.prefix_ids == ()
    assert demoted.self_check["passed"] is False

    def failing(text: str, special: bool) -> Sequence[int]:
        raise RuntimeError("synthetic engine failure")

    errored = plan_special_tokens(document, failing)
    assert errored.kind == KIND_REFERENCE
    assert errored.self_check["error"] == "RuntimeError"


def test_reference_kind_is_not_self_checked() -> None:
    def never(text: str, special: bool) -> Sequence[int]:
        raise AssertionError("a reference face is not probed")

    plan = plan_special_tokens(_document(_REFERENCE_SHAPES["suffix"]), never)
    assert plan.kind == KIND_REFERENCE
    assert plan.self_check["passed"] is None


# -- the facade ------------------------------------------------------------

_TEXTS = [
    "",
    "a",
    "hello world",
    "<s>",
    "<s><s>",
    "hi <s> there",
    "h\u00e9llo \u4f60\u597d \U0001f642",
    "x" * 5000,
    ("chat <s> turn " * 400),
]


def _rig_for(tmp_path: Path, post_processor: object) -> tuple[Rig, Any]:
    import tokenizers

    rig = build_rig(tmp_path, _document(post_processor))
    oracle = tokenizers.Tokenizer.from_file(str(rig.artifact_path))
    return rig, oracle


@pytest.mark.parametrize(
    ("post_processor", "kind", "served_by"),
    [
        (_PREFIX, KIND_PREFIX, "facade_prefix"),
        (
            {"type": "Sequence", "processors": [_BYTE_LEVEL, _PREFIX]},
            KIND_PREFIX,
            "facade_prefix",
        ),
        (_BYTE_LEVEL, KIND_IDENTITY, "core_stream"),
        (None, KIND_IDENTITY, "core_stream"),
        (_REFERENCE_SHAPES["suffix"], KIND_REFERENCE, "reference"),
        (_REFERENCE_SHAPES["both_sides"], KIND_REFERENCE, "reference"),
    ],
)
def test_true_equals_the_reference_document_post_processor(
    tmp_path: Path, post_processor: object, kind: str, served_by: str
) -> None:
    rig, oracle = _rig_for(tmp_path, post_processor)
    tokenizer = rig.tokenizer()
    for text in _TEXTS:
        expected = list(oracle.encode(text, add_special_tokens=True).ids)
        assert list(tokenizer.encode(text, add_special_tokens=True).ids) == expected
        assert (
            list(tokenizer.encode(text, lookup="off", add_special_tokens=True).ids)
            == expected
        )
    rows = tokenizer.encode_batch(_TEXTS, add_special_tokens=True)
    assert [list(row.ids) for row in rows] == [
        list(oracle.encode(text, add_special_tokens=True).ids) for text in _TEXTS
    ]
    assert [list(row.ids) for row in tokenizer.encode_batch(_TEXTS)] == [
        list(oracle.encode(text, add_special_tokens=False).ids) for text in _TEXTS
    ]
    block = tokenizer.explain()["special_tokens"]
    assert isinstance(block, dict)
    assert block["postprocessor_class"] == kind
    assert block["served_by"] == served_by
    calls = 3 * len(_TEXTS)
    requests = block["requests"]
    if kind == KIND_REFERENCE:
        assert requests == {"facade": 0, "reference": calls}
    else:
        assert requests == {"facade": calls, "reference": 0}
    # The summary keys do not change.
    assert "special_tokens" not in tokenizer.explain(summary=True)


@pytest.mark.parametrize(
    ("post_processor", "adds"),
    [
        (None, False),
        (_BYTE_LEVEL, False),
        (_PREFIX, True),
        (_REFERENCE_SHAPES["suffix"], True),
    ],
)
def test_router_constructor_value_is_false_only_for_identity_faces(
    tmp_path: Path, post_processor: object, adds: bool
) -> None:
    rig, _oracle = _rig_for(tmp_path, post_processor)
    tokenizer = rig.tokenizer()
    assert tokenizer._postprocessor_adds_tokens is adds


def test_true_runs_the_plain_route_and_never_consults_the_store(
    tmp_path: Path,
) -> None:
    rig, oracle = _rig_for(tmp_path, _PREFIX)
    tokenizer = rig.tokenizer(store=rig.store_path())
    text = "stored text " * 600
    tokenizer.encode(text)  # a core-stream call with content lookup
    before = cast("dict[str, int]", tokenizer.explain()["store"])
    for _ in range(3):
        assert list(tokenizer.encode(text, add_special_tokens=True).ids) == list(
            oracle.encode(text, add_special_tokens=True).ids
        )
    after = cast("dict[str, int]", tokenizer.explain()["store"])
    for key in ("auto_hits", "auto_appends", "auto_misses", "entries"):
        assert after[key] == before[key], key
    with pytest.raises(UnsupportedConfig):
        tokenizer.encode(text, session="s", add_special_tokens=True)
    with pytest.raises(UnsupportedConfig):
        tokenizer.encode(text, lookup="auto", add_special_tokens=True)
    with pytest.raises(ValueError):
        tokenizer.encode(text, parallel=2, add_special_tokens=True)


@pytest.mark.parametrize(
    "post_processor", [_PREFIX, _BYTE_LEVEL, _REFERENCE_SHAPES["suffix"]]
)
def test_session_final_ids_equals_the_reference(
    tmp_path: Path, post_processor: object
) -> None:
    rig, oracle = _rig_for(tmp_path, post_processor)
    tokenizer = rig.tokenizer(store=rig.store_path())
    accumulated = ""
    with tokenizer.session(session_id="chat") as session:
        for turn in ("hello", " <s> world", " " + "y" * 3000, " end<s>"):
            session.append(turn)
            accumulated += turn
            assert session.final_ids(True) == list(
                oracle.encode(accumulated, add_special_tokens=True).ids
            )
            assert session.final_ids(False) == list(
                oracle.encode(accumulated, add_special_tokens=False).ids
            )


def test_final_ids_prefix_shortcut_needs_no_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig, oracle = _rig_for(tmp_path, _PREFIX)
    tokenizer = rig.tokenizer()
    with tokenizer.session() as session:
        session.append("some text <s> here")

        def refuse(*_args: object, **_keywords: object) -> object:
            raise AssertionError("final_ids(True) encoded again")

        monkeypatch.setattr(tokenizer, "encode", refuse)
        monkeypatch.setattr(tokenizer, "_encode_ids", refuse)
        assert session.final_ids(True) == list(
            oracle.encode("some text <s> here", add_special_tokens=True).ids
        )


def test_final_ids_keeps_the_re_encode_outside_the_exact_id_guarantee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig, oracle = _rig_for(tmp_path, _PREFIX)
    tokenizer = rig.tokenizer(policy="experimental")
    calls: list[str] = []
    original = tokenizer.encode

    def tracking(text: str, **keywords: Any) -> Any:
        calls.append(text)
        return original(text, **keywords)

    with tokenizer.session() as session:
        session.append("experimental text")
        monkeypatch.setattr(tokenizer, "encode", tracking)
        assert session.final_ids(True) == list(
            oracle.encode("experimental text", add_special_tokens=True).ids
        )
    assert calls == ["experimental text"]
