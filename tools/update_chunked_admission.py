#!/usr/bin/env python3
"""Derive the chunk-parallel admission table and its evidence manifest.

The chunk-parallel path (``toktier.chunked``) may chunk a tokenizer only
when the chunked path itself has been judged on that artifact. This tool
turns the recorded judgement into the two shipped documents:

* ``src/toktier/routing/tables/chunked_admission.v1.json`` -- the table
  the module reads at run time: one row per certified family and
  artifact digest, naming the paths (``python``, ``native``) whose
  evidence is recorded, the evidence id, and the volumes. A family
  without a row is refused at run time
  (``explain()["chunked"]["refusal"] == "artifact_without_chunked_evidence"``).
* ``evidence/evidence_manifest_chunked_cuts.json`` -- the campaign's
  evidence manifest in the frozen shape of
  ``schemas/evidence_manifest.schema.json`` (``ev-chunked-cuts-v1``),
  verified like every other manifest by ``tools/generate_evidence.py``.

Both are derived from one input, ``readings/chunked_cuts_campaign.json``
(``schemas/chunked_readings.schema.json``), which the campaign tool
``tools/run_chunked_cut_campaign.py summarize`` writes. The derivation is
total and refuses rather than rounds: a family with any recorded mismatch,
an artifact digest that differs from the certified repair roster, or a
family outside the roster stops generation.

Two identities bind the table to the sources it speaks about. The
``chunked_identity`` (``tools/source_identity_common.py``,
``TOOL_SIDE_IDENTITIES["chunked"]``) covers the module and the two frozen
repair tables it reads; the ``fast_cpu_source_digest_v2`` is the
version-normalized identity of the native engine. Both are recomputed by
``--check``, so a change to any covered file makes the gate ask for the
table to be regenerated, which is the moment to re-examine the evidence.

Usage::

    python tools/update_chunked_admission.py            # regenerate both
    python tools/update_chunked_admission.py --check    # the sweep gate
    python tools/update_chunked_admission.py --release-check

``--check`` passes when the two documents are byte-for-byte what the
readings derive (``generated_by`` excepted), their digests and schemas
hold, and the two identities equal the tree's. ``--release-check`` adds
what a release must be able to say: that the readings were recorded on
this tree's chunked source (and, when a native path is admitted, on this
tree's engine identity), and which roster families still have no row. A
family that is absent is reported, not invented; the exit status is
nonzero so the release notes have to acknowledge it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compute_identity_v2 import source_digest as source_digest_v2
from registry_common import (
    CHUNKED_ADMISSION_DOMAIN_TAG,
    CHUNKED_READINGS_DOMAIN_TAG,
    EVIDENCE_DOMAIN_TAG,
    GenerationError,
    check_regenerated,
    git_commit,
    load_json,
    verify_file,
    write_document,
)
from source_identity_common import TOOL_SIDE_IDENTITIES

from toktier.repair.registry import families as roster_families

TOOL_NAME = "tools/update_chunked_admission.py"
TOOL_VERSION = "1.0.0"
EVIDENCE_ID = "ev-chunked-cuts-v1"
SUITE_VERSION = "chunked-cuts-four-way-v1"
#: The two levels of chunked-path evidence a certified row needs for the
#: Python path (and the native path, for it to be listed in ``paths``).
EVIDENCE_LEVELS = ("adversarial", "corpus")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
READINGS_PATH = REPOSITORY_ROOT / "readings" / "chunked_cuts_campaign.json"
TABLE_PATH = (
    REPOSITORY_ROOT
    / "src"
    / "toktier"
    / "routing"
    / "tables"
    / "chunked_admission.v1.json"
)
MANIFEST_PATH = REPOSITORY_ROOT / "evidence" / "evidence_manifest_chunked_cuts.json"
TABLE_SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "chunked_admission.schema.json"
READINGS_SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "chunked_readings.schema.json"
EVIDENCE_SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "evidence_manifest.schema.json"

MISMATCH_COLUMNS = (
    "python_vs_oracle",
    "native_vs_oracle",
    "python_vs_native",
    "serial_vs_oracle",
)


def chunked_identity() -> str:
    """The ``sha256:``-prefixed chunked source identity of this tree."""
    return "sha256:" + TOOL_SIDE_IDENTITIES["chunked"].source_digest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be a JSON object")
    return value


def _rows(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise GenerationError(f"{label} must be a list of objects")
    return value


def _count(value: object, label: str) -> int | None:
    """A recorded count: a non-negative integer, or ``None`` for 'not run'."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GenerationError(f"{label} must be a non-negative integer or null")
    return value


def _mismatches(row: dict[str, Any], label: str) -> dict[str, int | None]:
    recorded = _mapping(row.get("mismatches"), f"{label}.mismatches")
    out = {column: _count(recorded.get(column), f"{label}.mismatches.{column}")
           for column in MISMATCH_COLUMNS}
    for column, value in out.items():
        if value:
            raise GenerationError(
                f"{label}: {value} recorded mismatch(es) in {column}; a family "
                "with a recorded mismatch cannot be admitted to the chunked "
                "path, and the readings are kept as read"
            )
    return out


def _paths(row: dict[str, Any], label: str) -> tuple[str, ...]:
    paths = row.get("paths")
    if (
        not isinstance(paths, list)
        or not paths
        or len(set(paths)) != len(paths)
        or any(path not in ("python", "native") for path in paths)
    ):
        raise GenerationError(f"{label}.paths must name python and/or native once")
    return tuple(str(path) for path in paths)


def _roster() -> dict[str, str]:
    return {item.family: item.artifact_sha256 for item in roster_families()}


def _family_row(
    row: dict[str, Any], label: str, roster: dict[str, str]
) -> tuple[str, str, str]:
    family = str(row.get("family"))
    if family not in roster:
        raise GenerationError(
            f"{label}: {family!r} is outside the certified repair roster, "
            "which is the roster the chunk-cut rule was certified on"
        )
    digest = row.get("artifact_sha256")
    if digest != roster[family]:
        raise GenerationError(
            f"{label}: {family}: artifact digest {digest!r} is not the roster "
            f"artifact {roster[family]!r}"
        )
    normalizer = row.get("normalizer")
    if normalizer not in ("none", "nfc"):
        raise GenerationError(f"{label}: {family}: normalizer must be none or nfc")
    return family, str(digest), str(normalizer)


def derive(
    readings: dict[str, Any],
    *,
    identity: str,
    engine_v2: str,
    generated_by: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(admission table, evidence manifest)`` derived from the readings.

    A family gets a row when either level of evidence judged its Python
    chunked ids against the oracle and the serial route; the row is
    ``certified`` only when both levels did (the adversarial battery and
    the corpus campaign), and only then does ``paths`` list ``python``,
    plus ``native`` when both levels also judged the native chunked ids
    against the oracle and the Python ids. A row with one level carries
    that as its status and no paths. Any recorded mismatch, anywhere,
    refuses the whole derivation.
    """
    roster = _roster()
    corpus_rows = _rows(readings.get("families"), "families")
    adversarial = _mapping(readings.get("adversarial"), "adversarial")
    adversarial_rows = _rows(adversarial.get("families"), "adversarial.families")
    excluded = _rows(readings.get("not_reasoned"), "not_reasoned")

    # Per family: which evidence levels (adversarial battery, corpus
    # campaign) judged which path with complete, zero-mismatch columns.
    levels: dict[str, dict[str, set[str]]] = {}
    facts: dict[str, dict[str, Any]] = {}
    total_mismatches = 0

    def enroll(
        family: str, digest: str, normalizer: str, label: str
    ) -> dict[str, Any]:
        known = facts.get(family)
        if known is None:
            known = {
                "family": family,
                "artifact_sha256": digest,
                "normalizer": normalizer,
                "corpus_docs": 0,
                "adversarial_strings": 0,
            }
            facts[family] = known
            levels[family] = {"python": set(), "native": set()}
        elif known["normalizer"] != normalizer:
            raise GenerationError(
                f"{label}: {family}: normalizer differs between the corpus "
                "campaign and the adversarial battery"
            )
        return known

    def judged(
        row: dict[str, Any], label: str, family: str, level: str
    ) -> dict[str, int | None]:
        paths = _paths(row, label)
        mismatches = _mismatches(row, label)
        python_complete = (
            "python" in paths
            and mismatches["python_vs_oracle"] is not None
            and mismatches["serial_vs_oracle"] is not None
        )
        native_complete = (
            "native" in paths
            and mismatches["native_vs_oracle"] is not None
            and mismatches["python_vs_native"] is not None
        )
        if not python_complete:
            raise GenerationError(
                f"{label}: {family}: the {level} readings must judge the Python "
                "path against the oracle and the serial route"
            )
        levels[family]["python"].add(level)
        if native_complete:
            levels[family]["native"].add(level)
        return mismatches

    for index, row in enumerate(corpus_rows):
        label = f"families[{index}]"
        family, digest, normalizer = _family_row(row, label, roster)
        known = enroll(family, digest, normalizer, label)
        if known["corpus_docs"]:
            raise GenerationError(f"{label}: {family} appears twice")
        docs = _count(row.get("docs"), f"{label}.docs")
        if not docs:
            raise GenerationError(f"{label}: {family}: no documents were judged")
        if "native" in _paths(row, label) and row.get("native") is None:
            raise GenerationError(
                f"{label}: {family}: a native-path row needs the native counters"
            )
        mismatches = judged(row, label, family, "corpus")
        total_mismatches += sum(value or 0 for value in mismatches.values())
        known["corpus_docs"] = docs

    for index, row in enumerate(adversarial_rows):
        label = f"adversarial.families[{index}]"
        family, digest, normalizer = _family_row(row, label, roster)
        known = enroll(family, digest, normalizer, label)
        if known["adversarial_strings"]:
            raise GenerationError(f"{label}: {family} appears twice")
        strings = _count(row.get("strings"), f"{label}.strings")
        if not strings:
            raise GenerationError(f"{label}: {family}: no strings were judged")
        mismatches = judged(row, label, family, "adversarial")
        total_mismatches += sum(value or 0 for value in mismatches.values())
        known["adversarial_strings"] = strings

    rows: dict[str, dict[str, Any]] = {}
    for family, known in facts.items():
        python_levels = levels[family]["python"]
        native_levels = levels[family]["native"]
        if python_levels == set(EVIDENCE_LEVELS):
            status = "certified"
        elif python_levels == {"adversarial"}:
            status = "adversarial_only"
        else:
            status = "corpus_only"
        paths: list[str] = []
        if status == "certified":
            paths.append("python")
            if native_levels == set(EVIDENCE_LEVELS):
                paths.append("native")
        rows[family] = {
            "family": family,
            "artifact_sha256": known["artifact_sha256"],
            "normalizer": known["normalizer"],
            "status": status,
            "paths": paths,
            "evidence": {
                "python": sorted(python_levels),
                "native": sorted(native_levels),
            },
            "evidence_id": EVIDENCE_ID,
            "corpus_docs": known["corpus_docs"],
            "adversarial_strings": known["adversarial_strings"],
        }

    for index, row in enumerate(excluded):
        label = f"not_reasoned[{index}]"
        family = str(row.get("family"))
        if family not in roster or row.get("artifact_sha256") != roster[family]:
            raise GenerationError(f"{label}: {family!r} is not a roster artifact")
        if family in rows:
            raise GenerationError(
                f"{label}: {family} is recorded as not reasoned about and also "
                "has a judged row"
            )

    table = {
        "schema": "toktier.chunked_admission.v1",
        "schema_version": 1,
        "generated_by": dict(generated_by),
        "chunked_identity": identity,
        "fast_cpu_source_digest_v2": engine_v2,
        "rows": [rows[family] for family in sorted(rows)],
    }

    campaign = _mapping(readings.get("campaign"), "campaign")
    corpus = _rows(readings.get("corpus"), "corpus")
    corpora: list[dict[str, Any]] = []
    for index, entry in enumerate(corpus):
        docs = _count(entry.get("docs"), f"corpus[{index}].docs")
        chars = _count(entry.get("chars"), f"corpus[{index}].chars")
        if docs is None or chars is None:
            raise GenerationError(f"corpus[{index}]: docs and chars are required")
        corpora.append(
            {"corpus_id": str(entry.get("corpus_id")), "docs": docs, "bytes": chars}
        )
    if not corpora:
        raise GenerationError("corpus: at least one judged corpus is required")
    environment: dict[str, Any] = {
        "os": str(readings.get("os")),
        "python": str(readings.get("python")),
    }
    hosts = readings.get("hosts")
    if isinstance(hosts, list) and hosts:
        environment["hosts"] = [str(host) for host in hosts]
    manifest = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "generated_by": dict(generated_by),
        "run": {
            "run_id": str(campaign.get("run_id")),
            "date": str(campaign.get("date")),
            "suite_version": SUITE_VERSION,
        },
        "oracle": {"package": "tokenizers", "version": str(readings.get("tokenizers"))},
        "artifacts": [
            {"artifact_sha256": rows[family]["artifact_sha256"], "family": family}
            for family in sorted(rows)
        ],
        "corpora": corpora,
        "totals": {
            "docs": sum(int(entry["docs"]) for entry in corpora),
            "bytes": sum(int(entry["bytes"]) for entry in corpora),
            "mismatches": total_mismatches,
        },
        "environment": environment,
    }
    return table, manifest


def _generated_by(source_commit: str | None) -> dict[str, str]:
    commit = source_commit or git_commit(REPOSITORY_ROOT)
    if commit is None:
        raise GenerationError(
            "cannot determine the source commit; pass --source-commit"
        )
    return {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "source_commit": commit,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def partial_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Table rows that are not certified (one evidence level recorded)."""
    return [row for row in table.get("rows", []) if row.get("status") != "certified"]


def absent_families(readings: dict[str, Any], table: dict[str, Any]) -> list[str]:
    """Roster families with neither a table row nor a not-reasoned record."""
    present = {str(row["family"]) for row in table.get("rows", [])}
    excluded = {
        str(row.get("family")) for row in readings.get("not_reasoned", [])
        if isinstance(row, dict)
    }
    return sorted(set(_roster()) - present - excluded)


def check(
    *,
    readings_path: Path,
    table_path: Path,
    manifest_path: Path,
    release: bool,
) -> list[str]:
    """Return the problems ``--check`` (and ``--release-check``) report."""
    problems = verify_file(
        readings_path, load_json(READINGS_SCHEMA_PATH), CHUNKED_READINGS_DOMAIN_TAG
    )
    if problems:
        return problems
    readings = _mapping(load_json(readings_path), "readings")
    identity = chunked_identity()
    engine_v2 = source_digest_v2("fast_cpu")
    placeholder = {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "source_commit": "0" * 40,
        "generated_at": "1970-01-01T00:00:00Z",
    }
    table, manifest = derive(
        readings, identity=identity, engine_v2=engine_v2, generated_by=placeholder
    )
    table_schema = load_json(TABLE_SCHEMA_PATH)
    evidence_schema = load_json(EVIDENCE_SCHEMA_PATH)
    problems.extend(
        check_regenerated(table_path, table, table_schema, CHUNKED_ADMISSION_DOMAIN_TAG)
    )
    problems.extend(
        check_regenerated(manifest_path, manifest, evidence_schema, EVIDENCE_DOMAIN_TAG)
    )
    if table_path.exists():
        shipped = _mapping(load_json(table_path), "admission table")
        if shipped.get("chunked_identity") != identity:
            problems.append(
                f"{table_path}: chunked_identity {shipped.get('chunked_identity')!r} "
                f"is not this tree's {identity!r}; the chunked sources moved, so "
                "the table has to be regenerated and the evidence re-examined"
            )
        if shipped.get("fast_cpu_source_digest_v2") != engine_v2:
            problems.append(
                f"{table_path}: fast_cpu_source_digest_v2 is not this tree's "
                f"{engine_v2!r}; the native engine sources moved"
            )
    if not release:
        return problems
    recorded_identity = readings.get("chunked_identity")
    if recorded_identity != identity:
        problems.append(
            f"release: the readings were recorded under chunked_identity "
            f"{recorded_identity!r}, not this tree's {identity!r}; the Python-path "
            "evidence does not bind to this tree"
        )
    if any("native" in row["paths"] for row in table["rows"]):
        recorded_v2 = readings.get("fast_cpu_source_digest_v2")
        if recorded_v2 != engine_v2:
            problems.append(
                f"release: the readings were recorded under "
                f"fast_cpu_source_digest_v2 {recorded_v2!r}, not this tree's "
                f"{engine_v2!r}; the native-path evidence does not bind to this tree"
            )
    for row in partial_rows(table):
        problems.append(
            f"release: {row['family']} has chunked-path evidence at one level only "
            f"({row['status']}); it is not certified and is refused at run time"
        )
    for family in absent_families(readings, table):
        problems.append(
            f"release: {family} is in the certified repair roster but has no "
            "chunked-path evidence; it is refused at run time"
        )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive or check the chunked admission table and manifest."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--release-check", action="store_true")
    parser.add_argument("--readings", type=Path, default=READINGS_PATH)
    parser.add_argument("--table", type=Path, default=TABLE_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--source-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.check or arguments.release_check:
        problems = check(
            readings_path=arguments.readings,
            table_path=arguments.table,
            manifest_path=arguments.manifest,
            release=arguments.release_check,
        )
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"{arguments.table}: check passed")
        print(f"{arguments.manifest}: check passed")
        return 0
    problems = verify_file(
        arguments.readings,
        load_json(READINGS_SCHEMA_PATH),
        CHUNKED_READINGS_DOMAIN_TAG,
    )
    if problems:
        raise GenerationError("\n".join(problems))
    readings = _mapping(load_json(arguments.readings), "readings")
    table, manifest = derive(
        readings,
        identity=chunked_identity(),
        engine_v2=source_digest_v2("fast_cpu"),
        generated_by=_generated_by(arguments.source_commit),
    )
    table_schema = load_json(TABLE_SCHEMA_PATH)
    evidence_schema = load_json(EVIDENCE_SCHEMA_PATH)
    write_document(arguments.table, table, table_schema, CHUNKED_ADMISSION_DOMAIN_TAG)
    write_document(arguments.manifest, manifest, evidence_schema, EVIDENCE_DOMAIN_TAG)
    rows = table["rows"]
    certified = sum(row["status"] == "certified" for row in rows)
    native = sum("native" in row["paths"] for row in rows)
    print(
        f"updated {arguments.table} ({len(rows)} rows, {certified} certified, "
        f"{native} with the native path) and {arguments.manifest}"
    )
    for row in partial_rows(table):
        print(
            f"note: {row['family']} has {row['status'].replace('_', ' ')} evidence "
            "and is refused at run time until both levels are recorded"
        )
    for family in absent_families(readings, table):
        print(f"note: {family} has no chunked-path evidence and is refused at run time")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
