"""The rider arms of the chunk-cut campaign tool.

The riders (``run --riders special_tokens,literal_segments``) make two
extra comparisons on a content-chosen subsample of the documents a unit
reads anyway. These offline tests pin what they may and may not do: the
selection depends on the content only; a chat-shaped variant is a
deterministic function of its inputs and places the literals it reports;
the riders count, report and save their own findings; and a unit that
runs them judges the same documents, with the same stream digest and the
same four chunked columns, as a unit that does not.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import run_chunked_cut_campaign as tool  # noqa: E402

from ..facade.conftest import build_rig  # noqa: E402


def test_selection_is_by_content_and_follows_the_share() -> None:
    texts = [f"document {index} " * 20 for index in range(4000)]
    chosen = [text for text in texts if tool.rider_selected(text, 0.1)]
    assert 300 < len(chosen) < 500
    assert chosen == [text for text in texts if tool.rider_selected(text, 0.1)]
    assert not any(tool.rider_selected(text, 0.0) for text in texts)
    assert all(tool.rider_selected(text, 1.0) for text in texts)
    # A larger share selects a superset.
    assert set(chosen) <= {text for text in texts if tool.rider_selected(text, 0.2)}


def test_a_chat_variant_is_deterministic_and_places_the_literals() -> None:
    literals = ["<|im_start|>", "<|im_end|>", "<think>"]
    text = "The quick brown fox jumps over the lazy dog. " * 10
    variant, placed = tool.chat_variant(text, literals, "f\0seed")
    assert (variant, placed) == tool.chat_variant(text, literals, "f\0seed")
    assert placed >= 0
    stripped = variant
    for literal in literals:
        stripped = stripped.replace(literal, "")
    assert stripped == text
    assert sum(variant.count(literal) for literal in literals) == placed
    seeds = {tool.chat_variant(text, literals, f"f\0{i}")[0] for i in range(50)}
    assert len(seeds) > 10
    assert any(
        tool.chat_variant(text, literals, f"f\0{i}")[1] > 0 for i in range(50)
    )
    assert tool.chat_variant(text, [], "f") == (text, 0)
    # One character or none: no cut, literals placed only around it.
    for short in ("", "x"):
        variant, placed = tool.chat_variant(short, literals, "f\0short")
        stripped = variant
        for literal in literals:
            stripped = stripped.replace(literal, "")
        assert stripped == short and placed >= 0


def _family(tmp_path: Path, riders: tuple[str, ...]) -> Any:
    rig = build_rig(tmp_path)
    return tool.Family("tiny_bytes", rig.tokenizer(), native="off", riders=riders)


def test_the_riders_count_and_name_their_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = _family(tmp_path, tool.RIDERS)
    try:
        text = "ab 12 cd " * 80
        assert tool.ride(family, text, tool.RIDERS) == []
        special = family.riders["special_tokens"]
        segments = family.riders["literal_segments"]
        assert special["documents"] == 1 and special["mismatches"] == 0
        assert sum(special["served_by"].values()) == 1
        # The facade reports the class and moves one request counter per
        # call (explain()["special_tokens"]): served_by names that counter.
        assert set(special["served_by"]) <= {"facade", "reference"}
        assert sum(special["classes"].values()) == 1
        assert set(special["classes"]) <= {"identity", "prefix", "reference"}
        assert special["vacuous"] == 0
        assert segments["documents"] == 1 and segments["mismatches"] == 0
        assert sum(segments["paths"].values()) == 1
        # An injected difference is counted and named, column by column.
        monkeypatch.setattr(family, "reference", lambda _text: [1, 2, 3])
        findings = tool.ride(family, text, ("literal_segments",))
        assert [finding["unequal"] for finding in findings] == [["chat_vs_oracle"]]
        assert segments["mismatches"] == 1 and special["documents"] == 1
        rows = tool.family_rows({"tiny_bytes": family})
        assert rows["tiny_bytes"]["riders"]["literal_segments"]["mismatches"] == 1
        assert tool._rider_mismatches({"tiny_bytes": family}) == 1
    finally:
        family.close()


def test_riders_leave_the_stream_and_the_chunked_columns_alone(
    tmp_path: Path,
) -> None:
    docs = ["ab 12 " * 100, "cd 34 " * 120, "ef 56 " * 150, "gh 78 " * 90]

    def unit(riders: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
        family = _family(tmp_path / ("r" if riders else "n"), riders)
        try:
            stream, _stopped = tool.run_unit(
                {"tiny_bytes": family},
                (("memory", d) for d in docs),
                min_chars=512,
                max_chars=4000,
                chunk_chars=300,
                min_chunk=64,
                max_docs=10,
                skip_docs=0,
                time_budget_s=0.0,
                report_every=0,
                mismatch_dir=None,
                flush=lambda *_args: None,
                started=0.0,
                riders=riders,
                rider_share=1.0,
            )
            row = tool.family_rows({"tiny_bytes": family})["tiny_bytes"]
            return stream.facts(), row
        finally:
            family.close()

    plain_stream, plain_row = unit(())
    ridden_stream, ridden_row = unit(tool.RIDERS)
    assert plain_stream == ridden_stream
    assert "riders" not in plain_row
    riders = ridden_row.pop("riders")
    assert ridden_row["mismatches"] == plain_row["mismatches"]
    assert ridden_row["docs"] == plain_row["docs"] == len(docs)
    assert riders["special_tokens"]["documents"] == len(docs)
    assert riders["literal_segments"]["documents"] == len(docs)


class _ReportingTokenizer:
    """A facade stand-in whose special_tokens block is set by the test."""

    def __init__(self, kind: str, answer: str) -> None:
        self.kind = kind
        self.answer = answer
        self.requests = {"facade": 0, "reference": 0}

    def encode(self, text: str, **_kwargs: Any) -> Any:
        if self.answer in self.requests:
            self.requests[self.answer] += 1
        return argparse.Namespace(ids=[1, 2])

    def explain(self) -> dict[str, Any]:
        if self.answer == "silent":
            return {}
        return {
            "special_tokens": {
                "postprocessor_class": self.kind,
                "served_by": "facade_prefix",
                "requests": dict(self.requests),
            }
        }


class _Oracle:
    def encode(self, _text: str, **_kwargs: Any) -> Any:
        return argparse.Namespace(ids=[1, 2])


@pytest.mark.parametrize(
    ("kind", "answer", "served", "vacuous"),
    [
        ("prefix", "facade", "facade", 0),
        ("identity", "facade", "facade", 0),
        ("prefix", "reference", "reference", 1),
        ("identity", "reference", "reference", 1),
        ("reference", "reference", "reference", 0),
        ("prefix", "silent", "not_reported", 0),
    ],
)
def test_the_special_tokens_rider_reads_the_facade_request_counters(
    kind: str, answer: str, served: str, vacuous: int
) -> None:
    family = argparse.Namespace(
        name="f",
        tokenizer=_ReportingTokenizer(kind, answer),
        oracle=_Oracle(),
        riders={"special_tokens": tool._zero_rider("special_tokens")},
    )
    assert tool.ride(family, "text", ("special_tokens",)) == []
    row = family.riders["special_tokens"]
    assert row["served_by"] == {served: 1}
    assert row["classes"] == {(kind if answer != "silent" else "not_reported"): 1}
    assert row["vacuous"] == vacuous


def test_rider_counters_merge_over_units() -> None:
    first = tool._zero_rider("literal_segments")
    first.update(documents=3, literals=7, paths={"gigatoken_segmented": 3})
    first["bytes"]["fast_cpu"] = 100
    second = tool._zero_rider("literal_segments")
    second.update(documents=2, mismatches=1, paths={"hf_added_token": 2})
    units = [
        {"families": {"f": {"riders": {"literal_segments": first}}}},
        {"families": {"f": {"riders": {"literal_segments": second}}, "g": {}}},
    ]
    merged = tool.merge_riders(units)
    assert set(merged) == {"f"}
    row = merged["f"]["literal_segments"]
    assert row["documents"] == 5 and row["mismatches"] == 1 and row["literals"] == 7
    assert row["paths"] == {"gigatoken_segmented": 3, "hf_added_token": 2}
    assert row["bytes"] == {"gpu": 0, "fast_cpu": 100, "hf": 0}


def test_rider_arguments_are_checked() -> None:
    def parsed(riders: str, share: float) -> tuple[str, ...]:
        return tool._riders(argparse.Namespace(riders=riders, rider_share=share))

    assert parsed("", 0.1) == ()
    assert parsed("special_tokens,literal_segments", 0.1) == tool.RIDERS
    with pytest.raises(tool.SetupError, match="unknown rider"):
        parsed("other", 0.1)
    with pytest.raises(tool.SetupError, match="rider-share"):
        parsed("special_tokens", 0.0)
