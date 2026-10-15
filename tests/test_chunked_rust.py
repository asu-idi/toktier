"""The Rust crate's chunked surface answers as ``toktier.chunked`` does.

Artifact tier only (``TOKTIER_TEST_ARTIFACTS``, see ``test_chunked.py``),
and only on a host with ``cargo`` on its PATH: the crate's ``chunked``
example is built once (release profile; ``CARGO_TARGET_DIR`` is respected)
and run per family in its ``--json`` mode, which prints the admission the
Rust surface decided, the plan it made and a digest of the ids it
produced. The Python side loads the same artifact from the same directory
and answers with ``toktier.chunked``; the two are compared key by key.

Admission is compared under the default policy, where the Rust face reads
the shipped admission table for the native path and the Python face reads
it for its own. The refusal strings are one vocabulary: a roster artifact
whose native-path evidence is incomplete reads
``chunked_evidence_incomplete:<status>`` on the Rust face, which is the
string the Python face gives when its own path is in that state, and the
string ``chunked.encode(..., native=True)`` refuses with for the same
artifact. Plans and ids are compared under ``Policy::Experimental``, which
lifts the table step the way the Python planner's gate is lifted here,
and admits the engine of an unregistered build.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "4")

import toktier
from toktier import chunked
from toktier.artifacts.tables import ARTIFACT_MANIFEST
from toktier.repair.registry import families as roster_families

from .test_chunked import (
    _ARTIFACTS,
    REPO_ROOT,
    TABLE_ROWS,
    _artifact_source,
    _generated_corpora,
    _reference,
    load_isolated,
)

SHIPPED: tuple[str, ...] = tuple(
    sorted(json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8")))
)
ROSTER: frozenset[str] = frozenset(item.family for item in roster_families())

#: The structural refusals the native encoder itself answers with; the
#: Rust face reports them verbatim, as the Python face does.
_STRUCTURAL = ("normalizer_not_supported", "nfc_with_normalized_added_tokens")
_LITERAL_FLAGS = "added_token_strip_or_single_word"
_ROSTER_REFUSAL = "artifact_not_in_certified_repair_roster"


def _ids_sha256(ids: Any) -> str:
    values = [int(value) for value in ids]
    return hashlib.sha256(struct.pack(f"<{len(values)}I", *values)).hexdigest()


@pytest.fixture(scope="module")
def rust_example() -> Iterator[Path]:
    """The crate's ``chunked`` example, built once in the release profile."""
    if not _ARTIFACTS:
        pytest.skip("TOKTIER_TEST_ARTIFACTS is not set")
    cargo = shutil.which("cargo")
    if not cargo:
        pytest.skip("cargo is not on PATH")
    command = [str(cargo), "build", "--release", "-p", "toktier", "--example"]
    command += ["chunked", "--locked"]
    if os.environ.get("CARGO_NET_OFFLINE") == "true":
        command.append("--offline")
    built = subprocess.run(
        command, cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    assert built.returncode == 0, built.stderr[-4000:]
    target = Path(os.environ.get("CARGO_TARGET_DIR") or REPO_ROOT / "target")
    binary = target / "release" / "examples" / "chunked"
    assert binary.is_file(), binary
    yield binary


def _run(binary: Path, family: str, *arguments: str) -> dict[str, Any]:
    assert _ARTIFACTS is not None
    command = [str(binary), "--json", "--family", family, "--artifacts", _ARTIFACTS]
    completed = subprocess.run(
        [*command, *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (family, completed.stderr[-4000:])
    report: dict[str, Any] = json.loads(completed.stdout)
    return report


def _native_status(levels: list[str]) -> str | None:
    adversarial = "adversarial" in levels
    corpus = "corpus" in levels
    if adversarial and corpus:
        return "certified"
    if adversarial:
        return "adversarial_only"
    if corpus:
        return "corpus_only"
    return None


def _expected_rust_refusal(
    python_refusal: str | None, row: dict[str, Any] | None, artifact_sha256: str
) -> str | None:
    """What the documented rule gives the Rust face from the Python
    face's refusal and the row: the roster and structural refusals
    verbatim, then the native path's evidence."""
    if python_refusal == _ROSTER_REFUSAL:
        return python_refusal
    if python_refusal is not None and (
        python_refusal.startswith(_STRUCTURAL) or python_refusal == _LITERAL_FLAGS
    ):
        return python_refusal
    if row is None:
        return "artifact_without_chunked_evidence"
    status = _native_status(list(row["evidence"]["native"]))
    if status is None:
        return "artifact_without_chunked_evidence"
    if status != "certified":
        return f"chunked_evidence_incomplete:{status}"
    face = row.get("loader_face_sha256")
    if face is not None and face != artifact_sha256:
        return f"loader_face_differs_from_chunked_evidence:{artifact_sha256[:12]}"
    return None


def test_rust_admission_refusals_equal_the_python_ones(
    rust_example: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    checked = 0
    for family in SHIPPED:
        if not _artifact_source(family):
            continue
        report = _run(rust_example, family, "--policy", "supported", "--chars", "300")
        rust = report["admission"]
        checked += 1
        encode = report["encode"]
        assert encode["equal_to_serial"] is True, family
        if family not in ROSTER:
            # Outside the roster the answer is the roster refusal on both
            # faces; the Python face is asked only where its loader can
            # open the artifact directory as staged here.
            assert rust["refusal"] == _ROSTER_REFUSAL, family
            assert rust["normalizer"] is None and rust["literals"] == 0
            assert encode["serial_reason"] == "R_CHUNK_FAMILY_UNSUPPORTED", family
            try:
                tokenizer = load_isolated(family, tmp_path_factory)
            except toktier.errors.ArtifactNotFound:
                continue
            try:
                assert chunked.admission(tokenizer)["refusal"] == _ROSTER_REFUSAL
            finally:
                tokenizer.close()
            continue
        tokenizer = load_isolated(family, tmp_path_factory)
        try:
            python = chunked.admission(tokenizer)
        finally:
            tokenizer.close()
        assert rust["family"] == family == python["family"]
        assert rust["gate"] == "EvidenceTable"
        assert python["admission_gate"] == "evidence_table"
        assert rust["evidence_id"] == python["evidence_id"], family
        row = TABLE_ROWS.get(family)
        native_levels = list(row["evidence"]["native"]) if row is not None else []
        assert rust["native_evidence"] == native_levels, family
        assert rust["evidence_status"] == _native_status(native_levels), family
        assert rust["admitted"] is (rust["refusal"] is None)
        python_refusal = python["refusal"]
        assert python_refusal is None or isinstance(python_refusal, str)
        python_paths = python["evidence_paths"]
        assert isinstance(python_paths, list)
        # The Rust face reads the artifact file; the Python face the
        # loader face, which carries more literals where the
        # configuration sidecar declares them.
        same_face = python["loader_face_sha256"] == rust["artifact_sha256"]
        assert rust["normalizer"] == python["normalizer"], family
        python_literals = python["literals"]
        assert isinstance(python_literals, int)
        if same_face:
            assert rust["literals"] == python_literals, family
        else:
            assert python_literals >= rust["literals"], family
        expected = _expected_rust_refusal(python_refusal, row, rust["artifact_sha256"])
        assert rust["refusal"] == expected, (family, rust, python)
        # Where the Python face refuses on evidence its own path lacks and
        # the native path lacks the same, the strings are identical; where
        # the Python face admits its own path, the native path is judged
        # apart, and native=True answers as the Rust face does.
        if (
            python_refusal is not None
            and python_refusal.startswith("chunked_evidence_incomplete")
            and row is not None
            and sorted(row["evidence"]["python"]) == sorted(native_levels)
        ):
            assert rust["refusal"] == python_refusal, family
        if python_refusal is None:
            assert ("native" in python_paths) == (
                _native_status(native_levels) == "certified"
            ), family
        if rust["refusal"] is not None:
            assert encode["serial_reason"] == "R_CHUNK_FAMILY_UNSUPPORTED", family
            assert report["plan"] is None
            assert rust["refusal"] in report["plan_error"], family
    if checked == 0:
        pytest.skip("no shipped artifact under TOKTIER_TEST_ARTIFACTS")


def test_rust_plan_and_ids_equal_the_python_chunked_path(
    rust_example: Path, tmp_path_factory: pytest.TempPathFactory, tmp_path: Path
) -> None:
    text = _generated_corpora(300_000)["transcript(generated)"]
    path = tmp_path / "text.txt"
    path.write_text(text, encoding="utf-8")
    checked = 0
    skipped: list[str] = []
    for family in sorted(ROSTER):
        if family not in TABLE_ROWS or not _artifact_source(family):
            continue
        report = _run(
            rust_example,
            family,
            "--policy",
            "experimental",
            "--k",
            "8",
            "--text",
            str(path),
        )
        rust = report["admission"]
        assert rust["gate"] == "LiftedByExperimentalPolicy"
        assert rust["admitted"] is True, (family, rust["refusal"])
        if not rust["cpu_route"]:
            skipped.append(family)
            continue
        checked += 1
        tokenizer = load_isolated(family, tmp_path_factory)
        try:
            planner = chunked._planner_for(tokenizer, evidence_gate=False)
            same_face = planner.loader_face_sha256 == rust["artifact_sha256"]
            plan = planner.plan(text, 8, None, chunked.DEFAULT_SCAN_BUDGET)
            python = chunked.encode(tokenizer, text, k=8, native=False)
            reference = _reference(tokenizer)(text)
        finally:
            tokenizer.close()
        encode = report["encode"]
        assert encode["equal_to_serial"] is True, family
        assert encode["serial_reason"] is None and encode["chunks"] >= 2, family
        assert encode["ids_len"] == len(reference), family
        assert encode["ids_sha256"] == _ids_sha256(reference), family
        assert encode["ids_sha256"] == _ids_sha256(python.ids), family
        assert encode["certification"] == "Experimental"
        if same_face:
            spans = tuple(
                (int(low), int(high)) for low, high in report["plan"]["spans"]
            )
            assert spans == plan.spans, family
            assert tuple(encode["cuts"]) == python.cuts == plan.cuts, family
            assert encode["chunks"] == python.chunks, family
            assert (
                report["plan"]["cut_misses"],
                report["plan"]["literal_rejects"],
                report["plan"]["norm_rejects"],
            ) == (plan.cut_misses, plan.literal_rejects, plan.norm_rejects), family
    if checked == 0:
        pytest.skip(
            "no roster artifact reached the corrected CPU route on the Rust "
            f"face (skipped: {skipped})"
        )
