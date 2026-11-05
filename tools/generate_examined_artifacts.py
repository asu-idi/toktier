#!/usr/bin/env python3
"""Generate and verify the packaged table of examined tokenizer artifacts.

An examined artifact is not certified. It is an artifact whose pipeline
capability is the one a certified family holds and whose added-token
capability no campaign has judged, so its only accelerated route is the
fast CPU engine, admitted under the ``supported`` policy and labelled
``supported_untested``; the ``certified`` policy refuses it
(``docs/support-matrix.md``, Status vocabulary).

The source projection in ``data/examined_artifacts.v1.json`` is written by
maintainers from the examination readings. This generator checks it against
the documents it has to agree with and writes the domain-separated,
root-digested table the runtime reads:

* the artifact is not in the support registry, under its digest or its
  family id, and its capability pair is not a registry composition -- an
  examined row never shadows a certified one;
* the support registry carries the row's pipeline capability and oracle,
  and at least one certified record with that pipeline carries an eligible
  fast CPU entry, whose engine binding the route is planned against;
* the shipped artifact manifest pins the family with the same
  ``tokenizer.json`` digest;
* the certified repair roster does not name the family or the digest.

Nothing here grants a certificate, and nothing here moves the fast CPU
engine's bound source set: the table sits outside it.

Usage::

    python tools/generate_examined_artifacts.py            # write
    python tools/generate_examined_artifacts.py --check    # verify
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

from registry_common import (
    EXAMINED_ARTIFACTS_DOMAIN_TAG,
    GenerationError,
    check_regenerated,
    git_commit,
    load_json,
    serialise_document,
    write_document,
)

TOOL_NAME = "tools/generate_examined_artifacts.py"
TOOL_VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "examined_artifacts.v1.json"
SCHEMA = ROOT / "schemas" / "examined_artifacts.schema.json"
REGISTRY = ROOT / "tables" / "support_registry.json"
ARTIFACT_MANIFEST = (
    ROOT / "src" / "toktier" / "artifacts" / "tables" / "artifact_manifest.v1.json"
)
REPAIR_TABLE = (
    ROOT / "src" / "toktier" / "repair" / "tables" / "fast_repair_families.v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "src" / "toktier" / "routing" / "tables" / "examined_artifacts.v1.json"
)

ELIGIBLE = frozenset({"certified", "certified_source"})


def _object(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise GenerationError(f"{path}: top-level value must be an object")
    return value


def _check_against_registry(rows: list[dict[str, Any]]) -> None:
    registry = _object(REGISTRY)
    records = registry.get("artifacts") or []
    families = {str(record["family"]) for record in records}
    digests = {str(record["artifact_sha256"]) for record in records}
    pipelines = {str(row["pipeline_id"]) for row in registry.get("pipelines") or []}
    oracles = {str(row["oracle_id"]) for row in registry.get("oracles") or []}
    compositions = {
        (str(row["pipeline_id"]), str(row["added_frontend_id"]))
        for row in registry.get("compositions") or []
    }
    for row in rows:
        family = str(row["family"])
        if family in families or str(row["artifact_sha256"]) in digests:
            raise GenerationError(
                f"{family}: the support registry already records this family "
                "or artifact; an examined row may not shadow a certified one"
            )
        pair = (str(row["pipeline_id"]), str(row["added_frontend_id"]))
        if pair in compositions:
            raise GenerationError(
                f"{family}: {pair} is a registry composition, so the artifact "
                "would be certified by composition, not examined"
            )
        if pair[0] not in pipelines:
            raise GenerationError(
                f"{family}: pipeline {pair[0]} is not a certified pipeline "
                "capability of the support registry"
            )
        if str(row["oracle_id"]) not in oracles:
            raise GenerationError(
                f"{family}: oracle {row['oracle_id']!r} is not in the registry"
            )
        peers = [
            str(record["family"])
            for record in records
            if str(record["pipeline_id"]) == pair[0]
            and str(
                (record.get("backends") or {}).get("fast_cpu", {}).get("status")
            )
            in ELIGIBLE
        ]
        if not peers:
            raise GenerationError(
                f"{family}: no certified record with pipeline {pair[0]} carries "
                "an eligible fast CPU entry, so there is no engine binding to "
                "plan the route against"
            )


def _check_against_manifest(rows: list[dict[str, Any]]) -> None:
    manifest = _object(ARTIFACT_MANIFEST)
    for row in rows:
        family = str(row["family"])
        entry = manifest.get(family)
        if not isinstance(entry, dict):
            raise GenerationError(
                f"{family}: the shipped artifact manifest does not pin this family"
            )
        pinned = (entry.get("files") or {}).get("tokenizer.json", {}).get("sha256")
        if pinned != row["artifact_sha256"]:
            raise GenerationError(
                f"{family}: the artifact manifest pins {pinned} for "
                f"tokenizer.json, the examined row names {row['artifact_sha256']}"
            )


def _check_against_repair_roster(rows: list[dict[str, Any]]) -> None:
    roster = _object(REPAIR_TABLE).get("families") or []
    names = {str(item["family"]) for item in roster}
    digests = {str(item["artifact_sha256"]) for item in roster}
    for row in rows:
        if row["family"] in names or row["artifact_sha256"] in digests:
            raise GenerationError(
                f"{row['family']}: the certified repair roster already names "
                "this family or artifact"
            )


def build_document(
    source_path: Path, *, source_commit: str, generated_at: str
) -> dict[str, Any]:
    source = _object(source_path)
    if source_path.read_bytes() != serialise_document(source):
        raise GenerationError(
            f"{source_path}: source is not deterministically serialized"
        )
    rows = source.get("artifacts")
    if not isinstance(rows, list):
        raise GenerationError(f"{source_path}: artifacts must be an array")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise GenerationError(f"{source_path}: artifact rows must be objects")
        for key in ("family", "artifact_sha256"):
            value = str(row.get(key))
            if value in seen:
                raise GenerationError(f"{source_path}: {value} appears twice")
            seen.add(value)
    _check_against_registry(rows)
    _check_against_manifest(rows)
    _check_against_repair_roster(rows)
    return {
        "schema_version": source.get("schema_version"),
        "generated_by": {
            "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "source_commit": source_commit,
            "generated_at": generated_at,
        },
        "root_digest": "",
        "status": source.get("status"),
        "statement": source.get("statement"),
        "artifacts": sorted(rows, key=lambda row: str(row["family"])),
    }


def _default_timestamp() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    document = build_document(
        arguments.source,
        source_commit=arguments.source_commit or git_commit(ROOT) or "0000000",
        generated_at=arguments.generated_at or _default_timestamp(),
    )
    schema = _object(SCHEMA)
    if arguments.check:
        problems = check_regenerated(
            arguments.out, document, schema, EXAMINED_ARTIFACTS_DOMAIN_TAG
        )
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"{arguments.out}: check passed")
        return 0
    completed = write_document(
        arguments.out, document, schema, EXAMINED_ARTIFACTS_DOMAIN_TAG
    )
    print(
        f"wrote {arguments.out} ({len(completed['artifacts'])} examined "
        f"artifacts, root {completed['root_digest']})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
