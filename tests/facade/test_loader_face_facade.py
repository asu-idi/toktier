"""The facade over an artifact whose sidecar declares extra added tokens.

Contract reference: ``docs/contracts/facade.md`` Section 5 (reference =
the loader face), ``docs/contracts/routing.md`` Section 5.2
(``R_INPUT_ADDED_TOKEN``).

Since 0.2.8 every face the facade serves reads the same loader-face
document: the reference backend, the decode oracle, and the added-token
router. These tests pin that agreement on a tiny artifact whose
``tokenizer_config.json`` declares one added token beyond the artifact
file -- the shape on which 0.2.7's two certified routes could answer
differently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import Rig, build_rig

EXTRA = "<x_extra>"
EXTRA_ID = 256


def _sidecar_rig(tmp_path: Path) -> Rig:
    rig = build_rig(tmp_path)
    sidecar = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "added_tokens_decoder": {
            str(EXTRA_ID): {
                "content": EXTRA,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
        },
    }
    (rig.artifact_path.parent / "tokenizer_config.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return rig


@pytest.fixture
def sidecar_rig(tmp_path: Path) -> Rig:
    pytest.importorskip("transformers")
    return _sidecar_rig(tmp_path)


def test_default_and_reference_policies_agree_on_the_literal(
    sidecar_rig: Rig,
) -> None:
    """One loader face, every policy: the literal maps to its id."""
    text = f"a{EXTRA}b"
    with_default = sidecar_rig.tokenizer()
    with_reference = sidecar_rig.tokenizer(policy="reference")
    try:
        default_ids = list(with_default.encode(text).ids)
        reference_ids = list(with_reference.encode(text).ids)
        assert default_ids == reference_ids == [97, EXTRA_ID, 98]
        assert list(with_default.encode("ab").ids) == [97, 98]
    finally:
        with_default.close()
        with_reference.close()


def test_router_and_execution_answer_alike_about_the_literal(
    sidecar_rig: Rig,
) -> None:
    """The two scanning surfaces agree (FINDING follow-up, 0.2.8).

    On 0.2.7 the facade's Python router was built from the artifact
    file's added tokens while execution saw the loader face, so the
    router could deny a literal that execution then extracted. Both now
    read the loader face.
    """
    tokenizer = sidecar_rig.tokenizer()
    try:
        assert tokenizer._added_router.holds_literal(f"a{EXTRA}b") is True
        assert tokenizer._added_router.holds_literal("plain") is False
        plan = tokenizer._added_router.plan_for(f"a{EXTRA}b")
        assert plan is not None
        assert (EXTRA, EXTRA_ID) in list(plan)
    finally:
        tokenizer.close()


def test_decode_oracle_reads_the_same_document(sidecar_rig: Rig) -> None:
    tokenizer = sidecar_rig.tokenizer()
    try:
        assert tokenizer.decode([97, EXTRA_ID, 98], skip_special_tokens=False) == (
            f"a{EXTRA}b"
        )
    finally:
        tokenizer.close()


def test_loader_face_matches_the_pinned_loader(sidecar_rig: Rig) -> None:
    """The facade's ids equal AutoTokenizer's own on the same directory."""
    import transformers

    loader = transformers.AutoTokenizer.from_pretrained(
        str(sidecar_rig.artifact_path.parent),
        use_fast=True,
        local_files_only=True,
    )
    tokenizer = sidecar_rig.tokenizer()
    try:
        for text in (f"a{EXTRA}b", "plain text", f"{EXTRA}", f"x{EXTRA}"):
            assert list(tokenizer.encode(text).ids) == loader.encode(
                text, add_special_tokens=False
            )
    finally:
        tokenizer.close()


def test_seal_guard_covers_configuration_side_literals(sidecar_rig: Rig) -> None:
    """The session seal guard reads the loader face's longest literal."""
    tokenizer = sidecar_rig.tokenizer()
    try:
        assert tokenizer._seal_end_guard_chars >= len(EXTRA)
    finally:
        tokenizer.close()


def test_certification_block_reports_the_two_annotations(
    sidecar_rig: Rig,
) -> None:
    """``explain()`` carries the claim and carry-over keys, honest nulls here.

    The rig artifact has no certification record, so both read ``None``:
    the absence of a claim, not a claim of absence. The shipped-table
    values are pinned by ``tests/registry/test_carryover_sections.py``.
    """
    tokenizer = sidecar_rig.tokenizer()
    try:
        certification = tokenizer.explain()["certification"]
        assert isinstance(certification, dict)
        assert certification["config_added_tokens"] is None
        assert certification["carryover"] is None
    finally:
        tokenizer.close()
