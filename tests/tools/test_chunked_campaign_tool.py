"""The campaign tool judges four ways, records what ran, and stops on a finding.

Offline tier (the facade tier's tiny artifact): the judge compares the
chunked, serial and oracle ids and reports equality; an injected inequality
is a finding with the columns named and the unit ends with status 3; the
document stream digest is the corpus plan's definition and the part window
passes earlier documents over without judging them; the tool's copy of the
chunked identity path list is the tools-side identity; and a summary of a
family outside the roster is refused by the table generator.

Artifact tier (``TOKTIER_TEST_ARTIFACTS``): ``run``, ``adversarial`` and
``summarize`` end to end on one certified family over a two-document text
corpus, the readings validate, and the table generator derives a certified
row from them; the native arm is refused with status 4 on a plan that does
not start on the corrected CPU engine, and runs under a policy that does.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import compute_identity_v2  # noqa: E402
import run_chunked_cut_campaign as tool  # noqa: E402
import update_chunked_admission  # noqa: E402
import validate_registry  # noqa: E402
from registry_common import GenerationError, load_json  # noqa: E402
from source_identity_common import TOOL_SIDE_IDENTITIES  # noqa: E402

import toktier  # noqa: E402
from toktier import ReasonCode  # noqa: E402

from ..facade.conftest import build_rig  # noqa: E402
from ..test_chunked import (  # noqa: E402
    CANDIDATE_FAMILIES,
    _artifact_source,
    _generated_corpora,
    load_isolated,
)

GENERATED_BY = {
    "tool": update_chunked_admission.TOOL_NAME,
    "tool_version": update_chunked_admission.TOOL_VERSION,
    "source_commit": "0" * 40,
    "generated_at": "1970-01-01T00:00:00Z",
}


def _stream_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        payload = text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


# -- offline tier -------------------------------------------------------------


def test_identity_path_list_is_the_tools_side_identity() -> None:
    chunked_identity = TOOL_SIDE_IDENTITIES["chunked"]
    assert chunked_identity.domain == tool.CHUNKED_IDENTITY_DOMAIN
    assert tuple(name for name, _installed in tool.CHUNKED_IDENTITY_FILES) == (
        chunked_identity.files
    )
    for name, installed in tool.CHUNKED_IDENTITY_FILES:
        assert name == f"src/toktier/{installed}"
    # The tests run on the source tree as the installed package, so the
    # recomputation from the package equals the tree's digest.
    assert tool.installed_chunked_identity() == (
        "sha256:" + chunked_identity.source_digest()
    )


def test_judge_reports_equality_and_names_an_injected_inequality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = build_rig(tmp_path)
    tokenizer = rig.tokenizer()
    family = tool.Family("tiny_bytes", tokenizer, native="off")
    try:
        # Not in the roster: the lift does not admit it, and the judge
        # still compares the serial fallback against the oracle.
        assert family.admission["admitted"] is False
        assert family.admission["admission_gate"] == "lifted_for_evidence_collection"
        text = "ab 12 " * 200
        assert family.judge(text, k=4, min_chunk=64) is None
        counters = family.counters
        assert counters["docs"] == 1 and counters["chunked_docs"] == 0
        assert counters["serial_reasons"] == {"R_CHUNK_FAMILY_UNSUPPORTED": 1}
        assert counters["mismatches"] == {
            "python_vs_oracle": 0,
            "native_vs_oracle": None,
            "python_vs_native": None,
            "serial_vs_oracle": 0,
        }
        monkeypatch.setattr(family, "reference", lambda _text: [1, 2, 3])
        finding = family.judge(text, k=4, min_chunk=64)
        assert finding is not None
        assert finding["unequal"] == ["python_vs_oracle", "serial_vs_oracle"]
        assert finding["family"] == "tiny_bytes" and finding["text"] == text
        assert counters["mismatches"]["python_vs_oracle"] == 1
        assert counters["mismatches"]["serial_vs_oracle"] == 1
        tool.save_finding(tmp_path / "findings", finding)
        saved = list((tmp_path / "findings").glob("tiny_bytes_*.json"))
        assert len(saved) == 1 and "python_vs_oracle" in saved[0].name
    finally:
        family.close()


def test_run_unit_digest_and_part_window(tmp_path: Path) -> None:
    rig = build_rig(tmp_path)
    tokenizer = rig.tokenizer()
    family = tool.Family("tiny_bytes", tokenizer, native="off")
    docs = ["short", "ab 12 " * 100, "tiny", "cd 34 " * 120, "ef 56 " * 150]
    accepted = [d for d in docs if 512 <= len(d) <= 4000]
    flushed: list[tuple[int, bool, str]] = []

    def flush(stream: tool.Stream, final: bool, stopped: str) -> None:
        flushed.append((stream.docs_judged, final, stopped))

    try:
        stream, stopped = tool.run_unit(
            {"tiny_bytes": family},
            (("memory", d) for d in docs),
            min_chars=512,
            max_chars=4000,
            chunk_chars=300,
            min_chunk=64,
            max_docs=10,
            skip_docs=0,
            time_budget_s=0.0,
            report_every=1,
            mismatch_dir=None,
            flush=flush,
            started=0.0,
        )
        assert stopped == "source_exhausted"
        facts = stream.facts()
        assert facts["docs_seen"] == 5 and facts["docs_skipped"] == 2
        assert facts["docs_judged"] == 3 and facts["docs_passed_over"] == 0
        assert facts["chars"] == sum(len(d) for d in accepted)
        assert facts["doc_stream_sha256"] == _stream_digest(accepted)
        assert len(flushed) == 3 and all(not final for _, final, _ in flushed)
        # A part window: pass over the first accepted document, judge one.
        family.counters = tool._zero_counters()
        family.counters["mismatches"]["python_vs_oracle"] = 0
        stream, stopped = tool.run_unit(
            {"tiny_bytes": family},
            (("memory", d) for d in docs),
            min_chars=512,
            max_chars=4000,
            chunk_chars=300,
            min_chunk=64,
            max_docs=1,
            skip_docs=1,
            time_budget_s=0.0,
            report_every=0,
            mismatch_dir=None,
            flush=flush,
            started=0.0,
        )
        assert stopped == "max_docs"
        facts = stream.facts()
        assert facts["docs_passed_over"] == 1 and facts["docs_judged"] == 1
        # seen/skipped count the window only: "tiny" was skipped inside it.
        assert facts["docs_seen"] == 2 and facts["docs_skipped"] == 1
        assert facts["doc_stream_sha256"] == _stream_digest(accepted[1:2])
    finally:
        family.close()


def test_summary_of_a_family_outside_the_roster_is_refused(tmp_path: Path) -> None:
    rig = build_rig(tmp_path)
    tokenizer = rig.tokenizer()
    family = tool.Family("tiny_bytes", tokenizer, native="off")
    try:
        family.judge("ab 12 " * 200, k=4, min_chunk=64)
        rows = tool.family_rows({"tiny_bytes": family})
    finally:
        family.close()
    readings = _readings_from_rows(rows)
    with pytest.raises(GenerationError, match="outside the certified repair roster"):
        update_chunked_admission.derive(
            readings,
            identity=update_chunked_admission.chunked_identity(),
            engine_v2=compute_identity_v2.source_digest("fast_cpu"),
            generated_by=GENERATED_BY,
        )


def _readings_from_rows(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    families = []
    for name, row in rows.items():
        families.append(
            {
                "family": name,
                "artifact_sha256": row["artifact_sha256"],
                "loader_face_sha256": row["loader_face_sha256"],
                "normalizer": row["normalizer"] or "none",
                "paths": ["python"],
                "docs": row["docs"],
                "chars": row["chars"],
                "tokens": row["tokens"],
                "chunks": row["chunks"],
                "chunked_docs": row["chunked_docs"],
                "cut_misses": row["cut_misses"],
                "literal_rejects": row["literal_rejects"],
                "norm_rejects": row["norm_rejects"],
                "serial_fallbacks": row["serial_fallbacks"],
                "serial_reasons": row["serial_reasons"],
                "native": None,
                "mismatches": row["mismatches"],
            }
        )
    base: dict[str, Any] = load_json(update_chunked_admission.READINGS_PATH)
    base["families"] = families
    base["adversarial"] = {"generator": "not run", "families": []}
    base["corpus"] = [{"corpus_id": "memory", "docs": 1, "chars": 1200}]
    return base


# -- artifact tier --------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path: Path) -> str:
    """A two-document text corpus and its listing, as a ``--source`` value."""
    directory = tmp_path / "texts"
    directory.mkdir()
    generated = _generated_corpora(3_000)
    # Two documents: the blank line separates them, so neither may contain one.

    def one_document(text: str) -> str:
        return "\n".join(line for line in text.split("\n") if line.strip())

    prose = one_document(generated["prose(generated)"])
    code = one_document(generated["code(generated)"])
    (directory / "a.txt").write_text(prose + "\n\n" + code, encoding="utf-8")
    (directory / "unlisted.txt").write_text("not read " * 200, encoding="utf-8")
    listing = tmp_path / "texts.txt"
    listing.write_text("# listing\n" + "0" * 64 + "  a.txt\n", encoding="utf-8")
    return f"texts:{directory}:{listing}"


def test_sources_come_from_the_listing_not_the_directory(
    tmp_path: Path, corpus: str
) -> None:
    sources = tool.select_sources([corpus], 0, 1)
    assert len(sources) == 1 and sources[0].kind == "texts"
    assert [p.name for p in sources[0].files] == ["a.txt"]  # unlisted.txt is not read
    record = sources[0].record()
    assert record["files"] == ["a.txt"] and len(record["listing_sha256"]) == 64
    docs = list(tool.documents(sources, 0, 1, "text"))
    assert len(docs) == 2 and all(name == "a.txt" for name, _text in docs)
    # Round robin by listing position: shard 1 of 2 gets nothing here.
    assert tool.select_sources([corpus], 1, 2)[0].files == ()
    # A directory alone is refused, and so is a listed file that is missing.
    with pytest.raises(tool.SetupError, match="never scanned"):
        tool.select_sources([corpus.rsplit(":", 1)[0]], 0, 1)
    listing = tmp_path / "missing.txt"
    listing.write_text("a.txt\nb.txt\n", encoding="utf-8")
    root = corpus.split(":")[1]
    with pytest.raises(tool.SetupError, match="are not under"):
        tool.select_sources([f"texts:{root}:{listing}"], 0, 1)
    # ... but a missing file the shard does not select is no concern.
    assert [
        p.name for p in tool.select_sources([f"texts:{root}:{listing}"], 0, 2)[0].files
    ] == ["a.txt"]


@pytest.fixture
def isolated_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """Route the tool's ``toktier.load`` to the local artifact copies."""
    original = toktier.load

    def load(family: str, **options: Any) -> Any:
        options.pop("device", None)
        return load_isolated(family, tmp_path_factory, loader=original, **options)

    monkeypatch.setattr(toktier, "load", load)
    yield


def _certified_family() -> str:
    for family in CANDIDATE_FAMILIES:
        if _artifact_source(family):
            return family
    pytest.skip("no certified artifact available under TOKTIER_TEST_ARTIFACTS")
    raise AssertionError("unreachable")


def test_run_adversarial_and_summarize_end_to_end(
    tmp_path: Path, corpus: str, isolated_loader: None
) -> None:
    family = _certified_family()
    out = tmp_path / "out"
    assert (
        tool.main(
            [
                "run",
                "--source",
                corpus,
                "--families",
                family,
                "--min-chars",
                "512",
                "--chunk-chars",
                "500",
                "--mismatch-dir",
                str(out / "mismatch"),
                "--out",
                str(out / "shard_0000.json"),
                "--tag",
                "test",
            ]
        )
        == tool.STATUS_OK
    )
    shard = json.loads((out / "shard_0000.json").read_text(encoding="utf-8"))
    assert shard["schema"] == tool.SHARD_SCHEMA
    assert shard["manifest"]["final"] is True
    assert shard["manifest"]["admission_gate"] == "lifted_for_evidence_collection"
    assert shard["manifest"]["chunked_identity"] == tool.installed_chunked_identity()
    assert shard["stream"]["docs_judged"] == 2 and shard["totals"]["checks"] == 2
    assert shard["unit"]["sources"][0]["files"] == ["a.txt"]
    row = shard["families"][family]
    assert len(row["loader_face_sha256"]) == 64
    assert row["docs"] == 2 and row["chunked_docs"] == 2 and row["chunks"] >= 4
    assert row["mismatches"]["python_vs_oracle"] == 0
    assert row["mismatches"]["native_vs_oracle"] is None
    assert row["admission"]["admitted"] is True
    assert row["report"]["admission_gate"] == "lifted_for_evidence_collection"
    assert not (out / "mismatch").exists()

    assert (
        tool.main(
            [
                "adversarial",
                "--families",
                family,
                "--strings",
                "25",
                "--out",
                str(out / "battery.json"),
            ]
        )
        == tool.STATUS_OK
    )
    battery = json.loads((out / "battery.json").read_text(encoding="utf-8"))
    assert battery["schema"] == tool.BATTERY_SCHEMA
    assert battery["families"][family]["strings"] == 25
    assert battery["families"][family]["chunked"] >= 6

    readings_path = out / "readings.json"
    assert (
        tool.main(
            [
                "summarize",
                "--shards",
                str(out / "shard_*.json"),
                "--adversarial",
                str(out / "battery.json"),
                "--out",
                str(readings_path),
                "--corpus-id",
                "generated two-document text corpus",
                "--run-id",
                "chunked-cuts-test",
                "--date",
                "2026-10-14",
                "--source-commit",
                "0" * 40,
                "--expected-units",
                "1",
            ]
        )
        == tool.STATUS_OK
    )
    document = validate_registry.load_document(readings_path)
    assert validate_registry.detect_schema(document) == "chunked_readings"
    assert validate_registry.check_schema(document, "chunked_readings") == (True, "")
    digest_ok = validate_registry.check_root_digest(document, "chunked_readings")
    assert digest_ok == (True, "")
    assert isinstance(document, dict)
    assert document["chunked_identity"] == tool.installed_chunked_identity()
    assert document["shards"] == {"planned": 1, "final": 1}
    assert document["corpus"][0]["docs"] == 2
    table, manifest = update_chunked_admission.derive(
        document,
        identity=update_chunked_admission.chunked_identity(),
        engine_v2=compute_identity_v2.source_digest("fast_cpu"),
        generated_by=GENERATED_BY,
    )
    assert [r["family"] for r in table["rows"]] == [family]
    assert table["rows"][0]["loader_face_sha256"] == row["loader_face_sha256"]
    assert table["rows"][0]["status"] == "certified"
    assert table["rows"][0]["paths"] == ["python"]
    assert table["rows"][0]["corpus_docs"] == 2
    assert table["rows"][0]["adversarial_strings"] == 25
    assert manifest["totals"] == {
        "docs": 2,
        "bytes": document["corpus"][0]["chars"],
        "mismatches": 0,
    }


def test_family_groups_of_one_unit_merge_once(
    tmp_path: Path, corpus: str, isolated_loader: None
) -> None:
    """Two files of the same unit, one family each, summarize to one unit
    with the documents counted once; the same family twice, or a different
    stream under the same unit key, is refused."""
    families = [f for f in CANDIDATE_FAMILIES if _artifact_source(f)]
    if len(families) < 2:
        pytest.skip("two certified artifacts are needed under TOKTIER_TEST_ARTIFACTS")
    out = tmp_path / "out"
    for group, family in enumerate(families[:2]):
        assert (
            tool.main(
                [
                    "run",
                    "--source",
                    corpus,
                    "--families",
                    family,
                    "--chunk-chars",
                    "500",
                    "--out",
                    str(out / f"unit_g{group}.json"),
                ]
            )
            == tool.STATUS_OK
        )
    summarize = [
        "summarize",
        "--shards",
        str(out / "unit_g*.json"),
        "--out",
        str(out / "readings.json"),
        "--corpus-id",
        "generated",
        "--run-id",
        "chunked-cuts-test",
        "--date",
        "2026-10-14",
        "--source-commit",
        "0" * 40,
    ]
    assert tool.main(summarize) == tool.STATUS_OK
    document = json.loads((out / "readings.json").read_text(encoding="utf-8"))
    assert document["shards"] == {"planned": 1, "final": 1}
    assert document["corpus"][0]["docs"] == 2
    assert [row["family"] for row in document["families"]] == sorted(families[:2])
    assert all(row["docs"] == 2 for row in document["families"])
    # The same family in two files of one unit is refused.
    duplicate = json.loads((out / "unit_g0.json").read_text(encoding="utf-8"))
    (out / "unit_g2.json").write_text(json.dumps(duplicate), encoding="utf-8")
    assert tool.main(summarize) == tool.STATUS_SETUP
    # A file of the unit that judged other documents is refused.
    duplicate["families"] = {}
    duplicate["stream"]["doc_stream_sha256"] = "0" * 64
    (out / "unit_g2.json").write_text(json.dumps(duplicate), encoding="utf-8")
    assert tool.main(summarize) == tool.STATUS_SETUP


def test_native_arm_needs_the_corrected_cpu_route(
    tmp_path: Path, corpus: str, isolated_loader: None
) -> None:
    family = _certified_family()
    out = tmp_path / "out"
    arguments = [
        "run",
        "--source",
        corpus,
        "--families",
        family,
        "--chunk-chars",
        "500",
        "--native",
        "both",
        "--native-workers",
        "1",
        "--out",
        str(out / "shard.json"),
    ]
    status = tool.main(arguments)
    if status == tool.STATUS_NATIVE_ROUTE:
        # The default policy did not plan the corrected CPU route (the
        # identities are not yet rebound); the experimental policy does.
        status = tool.main([*arguments, "--policy", "experimental"])
    if status == tool.STATUS_NATIVE_ROUTE:
        pytest.skip(f"{family}: no corrected CPU route on this host")
    assert status == tool.STATUS_OK
    shard = json.loads((out / "shard.json").read_text(encoding="utf-8"))
    row = shard["families"][family]
    assert row["native"] is not None and row["native"]["plan_disagreements"] == 0
    assert row["native"]["chunked_docs"] == 2
    assert row["mismatches"]["native_vs_oracle"] == 0
    assert row["mismatches"]["python_vs_native"] == 0
    assert (
        ReasonCode.R_CHUNK_ROUTE_NOT_FAST_CPU.value
        not in (row["native"]["serial_reasons"])
    )
