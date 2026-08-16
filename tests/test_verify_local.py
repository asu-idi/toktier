"""The local verification record, its key, and the rule-only generator.

Acceptance surface: a record is read back only while every fact it was
taken under still holds; a check that disagreed leaves the route exactly
as unlabelled as no check at all; and the generated documents come from
rules in this package rather than from anyone's corpus.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from toktier.config import Config
from toktier.verify_local import (
    FRAGMENTS,
    RECORD_SCHEMA,
    Comparison,
    VerificationKey,
    compare,
    forget_record,
    generate,
    input_digest,
    is_locally_verified,
    read_record,
    record_for,
    record_path,
    split_documents,
    verify_cache_dir,
    write_record,
)


def _key(**overrides: str) -> VerificationKey:
    base: dict[str, Any] = {
        "engine": "gpu",
        "family": "qwen3_8b",
        "artifact_sha256": "a" * 64,
        "architecture": "sm_100",
        "delivery": "prebuilt",
        "image_digest": "b" * 64,
        "driver_version": "610.43.02",
        "host_source_digest": "c" * 64,
        "engine_source_digest": "d" * 64,
    }
    base.update(overrides)
    return VerificationKey(**base)


def _record(key: VerificationKey, *, status: str = "passed") -> object:
    comparison = Comparison(
        documents=3,
        bytes=120,
        mismatches=0 if status == "passed" else 1,
        first_mismatch=None if status == "passed" else (2, 7),
        served=3,
    )
    return record_for(
        key, comparison, documents=["a", "b", "c"], source="generated"
    )


@pytest.mark.parametrize(
    "field",
    [
        "engine",
        "family",
        "artifact_sha256",
        "architecture",
        "delivery",
        "image_digest",
        "driver_version",
        "host_source_digest",
        "engine_source_digest",
        "tool_version",
    ],
)
def test_every_key_field_files_the_record_somewhere_else(field: str) -> None:
    """Each fact a measurement depended on is part of its address."""
    moved = _key(**{field: "moved"})
    assert moved.digest() != _key().digest()


def test_a_record_is_read_back_only_under_its_own_key(tmp_path: Path) -> None:
    config = Config.resolve(home=tmp_path)
    key = _key()
    write_record(config, _record(key))  # type: ignore[arg-type]

    assert is_locally_verified(config, key) is True
    # A driver that moved is a different combination, so the record does
    # not describe it and nothing has to be swept to say so.
    assert is_locally_verified(config, _key(driver_version="611.0")) is False


def test_a_check_that_disagreed_leaves_the_route_unlabelled(
    tmp_path: Path,
) -> None:
    config = Config.resolve(home=tmp_path)
    key = _key()
    write_record(config, _record(key, status="failed"))  # type: ignore[arg-type]

    stored = read_record(config, key)
    assert stored is not None
    assert stored.status == "failed"
    assert stored.mismatches == 1
    # Running the tool never makes a combination more restricted than
    # not running it: the label is the one a route without any record
    # would have.
    assert is_locally_verified(config, key) is False


def test_a_record_of_another_schema_is_not_read(tmp_path: Path) -> None:
    config = Config.resolve(home=tmp_path)
    key = _key()
    path = write_record(config, _record(key))  # type: ignore[arg-type]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "toktier.local_verification.v2"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_record(config, key) is None


def test_reading_never_creates_the_directory(tmp_path: Path) -> None:
    config = Config.resolve(home=tmp_path)

    assert read_record(config, _key()) is None
    assert not verify_cache_dir(config).exists()


def test_forget_removes_one_combination(tmp_path: Path) -> None:
    config = Config.resolve(home=tmp_path)
    key = _key()
    write_record(config, _record(key))  # type: ignore[arg-type]

    assert forget_record(config, key) is True
    assert forget_record(config, key) is False
    assert not record_path(config, key).exists()


def test_the_record_is_owner_only(tmp_path: Path) -> None:
    config = Config.resolve(home=tmp_path)
    path = write_record(config, _record(_key()))  # type: ignore[arg-type]

    assert path.stat().st_mode & 0o077 == 0


def test_generated_documents_are_the_same_everywhere() -> None:
    """Same seed, same documents: a run can be repeated exactly."""
    assert generate(8, 512, 7) == generate(8, 512, 7)
    assert generate(8, 512, 7) != generate(8, 512, 8)


def test_generated_documents_stay_within_the_bound() -> None:
    documents = generate(40, 256, 3)
    assert len(documents) == 40
    assert all(len(text.encode("utf-8")) <= 256 for text in documents)


def test_a_small_run_still_reaches_every_shape() -> None:
    """Each document opens on a different fragment, in order."""
    documents = generate(len(FRAGMENTS), 64, 11)
    assert [text[:12] for text in documents] == [
        fragment[:12] for fragment in FRAGMENTS
    ]


def test_the_measured_sentinels_are_among_the_shapes() -> None:
    """The three code points FINDING 044 measured are generated."""
    joined = "".join(FRAGMENTS)
    for sentinel in ("\U00010940", "\U000323b0", "\u0295"):
        assert sentinel in joined


def test_documents_split_one_per_non_empty_line() -> None:
    assert split_documents("one\n\ntwo\n") == ["one", "two"]
    # Text without a line break is one document, not none.
    assert split_documents("just this") == ["just this"]
    assert split_documents("") == []


def test_the_input_digest_separates_two_runs() -> None:
    assert input_digest(["ab", "c"]) != input_digest(["a", "bc"])


class _Tokenizer:
    """A tokenizer stand-in: fixed ids, and a backend it claims to run."""

    def __init__(self, ids: dict[str, tuple[int, ...]], backend: str) -> None:
        self._ids = ids
        self._backend = backend

    def encode(self, text: str, *, lookup: str | None = None) -> object:
        return dataclasses.make_dataclass("E", ["ids"])(self._ids[text])

    def explain(self, *, summary: bool = False) -> dict[str, object]:
        return {"last_execution_backend": self._backend}


def test_compare_names_the_first_document_that_disagreed() -> None:
    documents = ["a", "b", "c"]
    subject = _Tokenizer({"a": (1, 2), "b": (3, 9), "c": (5,)}, "gpu")
    judge = _Tokenizer({"a": (1, 2), "b": (3, 4), "c": (5,)}, "hf")

    result = compare(subject, judge, documents, expected_backend="gpu")

    assert result.mismatches == 1
    assert result.first_mismatch == (1, 1)
    assert result.served == 3
    assert result.passed is False


def test_a_route_that_fell_back_did_not_measure_itself() -> None:
    """Agreement over documents the engine never served is not a pass."""
    documents = ["a", "b"]
    subject = _Tokenizer({"a": (1,), "b": (2,)}, "hf")
    judge = _Tokenizer({"a": (1,), "b": (2,)}, "hf")

    result = compare(subject, judge, documents, expected_backend="gpu")

    assert result.mismatches == 0
    assert result.served == 0
    assert result.passed is False


def test_a_record_carries_the_schema_and_what_was_compared() -> None:
    key = _key()
    record = record_for(
        key,
        Comparison(
            documents=2, bytes=9, mismatches=0, first_mismatch=None, served=2
        ),
        documents=["ab", "cdefghi"],
        source="your text",
    )

    assert record.schema == RECORD_SCHEMA
    assert record.status == "passed"
    assert record.input == "your text"
    assert record.input_digest == input_digest(["ab", "cdefghi"])
