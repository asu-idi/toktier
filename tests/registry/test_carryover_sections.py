"""The configuration-side subject extension and its carry-over evidence.

Contract reference: ``docs/contracts/registry.md`` Section 1 (exact
artifact identity, extended by the declared configuration-side added
tokens) and ``docs/contracts/evidence-carryover.md`` Section 3
(corpus-equivalence carry-over).

Two kinds of facts are pinned here. The shipped-table sweep asserts the
0.2.8 state of the world: exactly one packaged artifact
(``qwen3_5_08b``) has a loader face that carries added tokens beyond its
artifact file, its readings are annotated with the corpus-equivalence
carry-over, and its certificate verifies byte for byte -- every other
record carries neither section, which is the machine-checkable form of
"the two faces are the same function there". The schema tests keep the
new sections strict, so a malformed annotation cannot ride along.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "tables" / "support_registry.json"
SCHEMA = json.loads(
    (ROOT / "schemas" / "support_registry.schema.json").read_text(encoding="utf-8")
)

pytest.importorskip("jsonschema", reason="schema validation needs jsonschema")


def _shipped() -> dict[str, Any]:
    document = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


# ---------------------------------------------------------------------
# shipped-table sweep
# ---------------------------------------------------------------------


def test_exactly_one_artifact_declares_configuration_side_tokens() -> None:
    document = _shipped()
    declaring = {
        row["family"]: row["config_added_tokens"]
        for row in document["artifacts"]
        if "config_added_tokens" in row
    }
    assert set(declaring) == {"qwen3_5_08b"}
    claim = declaring["qwen3_5_08b"]
    assert claim["count"] == 7
    assert claim["source"] == "tokenizer_config.json"


def test_every_other_record_carries_neither_section() -> None:
    """For every other packaged artifact the two faces are one function."""
    document = _shipped()
    for row in document["artifacts"]:
        if row["family"] == "qwen3_5_08b":
            continue
        assert "config_added_tokens" not in row, row["family"]
        assert "carryover" not in row, row["family"]


def test_the_carried_record_names_its_certificate() -> None:
    document = _shipped()
    row = next(
        item for item in document["artifacts"] if item["family"] == "qwen3_5_08b"
    )
    carryover = row["carryover"]
    assert carryover["mechanism"] == "corpus_equivalence"
    assert len(carryover["divergence_set"]) == 7
    assert row["config_added_tokens"]["count"] == len(
        carryover["divergence_set"]
    )
    certificate = carryover["certificate"]
    path = ROOT / certificate["path"]
    assert path.is_file()
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == certificate["sha256"]
    )
    reading = json.loads(path.read_text(encoding="utf-8"))
    assert reading["subject"]["artifact_sha256"] == row["artifact_sha256"]
    assert reading["scan"]["verdict"] == "ABSENT"
    assert reading["scan"]["docs_with_any_target"] == 0
    literals = {
        item["content"] for item in reading["subject"]["divergence_set"]
    }
    assert literals == set(carryover["divergence_set"])
    for literal in carryover["divergence_set"]:
        assert reading["scan"]["occurrences"][literal] == 0
    # The subset relation the carry-over rests on: the certificate corpus
    # covers at least every document the readings were taken on.
    assert row["readings"]["docs"] <= certificate["docs"]
    assert certificate["occurrences"] == 0


def test_verification_half_accepts_the_shipped_table() -> None:
    from generate_registry import carryover_problems

    assert carryover_problems(SHIPPED) == []


# ---------------------------------------------------------------------
# schema strictness
# ---------------------------------------------------------------------


def _validate(document: dict[str, Any]) -> list[str]:
    import jsonschema

    validator = jsonschema.Draft202012Validator(SCHEMA)
    return [error.message for error in validator.iter_errors(document)]


def _with_sections(**overrides: Any) -> dict[str, Any]:
    document = _shipped()
    row = next(
        item for item in document["artifacts"] if item["family"] == "qwen3_5_08b"
    )
    for key, value in overrides.items():
        section, _, member = key.partition("__")
        if member:
            row[section][member] = value
        else:
            row[section] = value
    return document


def test_shipped_table_passes_the_schema() -> None:
    assert _validate(_shipped()) == []


def test_nonzero_occurrences_are_rejected() -> None:
    document = _shipped()
    row = next(
        item for item in document["artifacts"] if item["family"] == "qwen3_5_08b"
    )
    row["carryover"]["certificate"]["occurrences"] = 1
    assert _validate(document)


def test_unknown_carryover_members_are_rejected() -> None:
    assert _validate(_with_sections(carryover__reason="freeform"))


def test_the_removed_cross_artifact_members_are_no_longer_accepted() -> None:
    """0.2.9 removed two members 0.2.8 had reserved and nothing wrote.

    Loader-face equality makes the cross-artifact form unnecessary (two
    artifacts with equal faces have an empty divergence set and hold the
    same capability ids), so the reservation went rather than staying as
    dead configuration. They are rejected like any other unknown member,
    which is what keeps them from creeping back in unvalidated.
    """
    assert _validate(_with_sections(carryover__from_artifact_sha256="a" * 64))
    assert _validate(
        _with_sections(
            carryover__supporting_readings=[
                {"path": "readings/example.json", "sha256": "b" * 64}
            ]
        )
    )


def test_config_added_tokens_needs_a_positive_count() -> None:
    assert _validate(_with_sections(config_added_tokens__count=0))


def test_config_added_tokens_source_is_fixed() -> None:
    assert _validate(_with_sections(config_added_tokens__source="other.json"))
