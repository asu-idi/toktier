"""``Session.append_patch`` reports the same splice as ``append``.

Judged on the tiny byte-level artifact against a from-scratch encode by
the oracle package: two sessions fed the same turns, one through
``append`` and one through ``append_patch``, must report the same
``replace_from`` and the same replacement, and a caller applying the
patches to a stream of its own must end with the reference stream.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

import toktier
from toktier import SessionPatch, SessionUpdate

from .conftest import Rig

_ALPHABET = "abcdefghij XYZ0123456789.,;:!?()[]\u00e9\u00df\u4e2d\u6587\u0301\U0001f642"


def _turn(rng: random.Random) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(1, 300)))


def test_session_patch_value_object() -> None:
    patch = SessionPatch(replace_from=2, replacement_ids=[5, 6])
    assert patch.replacement_ids == (5, 6)
    assert isinstance(patch.replacement_ids, tuple)
    with pytest.raises(ValueError):
        SessionPatch(replace_from=-1, replacement_ids=())
    assert "SessionPatch" in toktier.__all__


def test_append_patch_matches_append_and_reference(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    rng = random.Random(0x9A7C4)
    tokenizer = rig.tokenizer(store=rig.store_path())
    for round_index in range(6):
        opening = _turn(rng) if rng.random() < 0.5 else ""
        with (
            tokenizer.session(session_id=f"u-{round_index}", text=opening) as full,
            tokenizer.session(session_id=f"p-{round_index}", text=opening) as thin,
        ):
            stream = list(thin.ids)
            assert stream == reference(opening)
            for _ in range(8):
                text = _turn(rng)
                update = full.append(text)
                patch = thin.append_patch(text)
                assert isinstance(update, SessionUpdate)
                assert isinstance(patch, SessionPatch)
                assert patch.replace_from == update.replace_from
                assert patch.replacement_ids == update.replacement_ids
                stream[patch.replace_from :] = patch.replacement_ids
                assert stream == list(update.all_ids)
                assert stream == reference(full.text)
                assert list(thin.ids) == stream
                assert thin.text == full.text
            assert thin.revision == full.revision


def test_ids_is_an_immutable_snapshot_after_either_append(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    tokenizer = rig.tokenizer()
    with tokenizer.session() as session:
        session.append_patch("ab 12")
        first = session.ids
        assert isinstance(first, tuple)
        assert session.ids is first  # cached until the next write
        session.append(" cd")
        second = session.ids
        assert isinstance(second, tuple)
        assert list(second) == reference("ab 12 cd")
        patch = session.append_patch(" ef")
        assert list(session.ids) == reference("ab 12 cd ef")
        assert (
            list(session.ids[: patch.replace_from])
            == list(second)[: patch.replace_from]
        )
        assert session.final_ids(add_special_tokens=False) == list(session.ids)


def test_empty_append_patch_reports_no_change(rig: Rig) -> None:
    tokenizer = rig.tokenizer()
    with tokenizer.session(text="hello world") as session:
        before = tuple(session.ids)
        patch = session.append_patch("")
        assert patch.replace_from == len(before)
        assert patch.replacement_ids == ()
        assert tuple(session.ids) == before
