#!/usr/bin/env python3
"""Recompute the store-fingerprint history table from the release tags.

``src/toktier/routing/tables/store_fingerprint_history.v1.json`` lists the
releases whose facade stores this build may open (the Accept branch of
``toktier.facade.fingerprint``). Each row is a function of what the named
release shipped, so the release gate recomputes every row at its tag
(``v<release>``) and refuses a table that differs:

* the fast CPU v1 / v2 pair: ``tools/fast_cpu_binding.json`` at the tag,
  which every certified ``fast_cpu`` entry of the tag's packaged registry
  must restate;
* the native host v2: the ``host_source_digest_v2`` of the certified
  prebuilt GPU deliveries at the tag, which must agree;
* per registered artifact, the loader-face digest the record names and the
  repair parameters the tag's repair sources give it (the roster table,
  then any record-held ``repair`` member), in the facade's own canonical
  form (``repair_parameters_component``).

The table's ``note``, schema and preimage domain are fixed here; the root
digest is the facade's (``toktier.store_fingerprint_history.v1``).

Usage::

    python tools/store_history.py --check          # recompute and compare
    python tools/store_history.py --write v0.3.0 v0.3.1

``generate_registry.py --release-check`` runs the same comparison. A tree
without the table passes (nothing is listed). A tree with the table but
without the named tags cannot be checked: a shallow clone (the CI checkout)
is told so in a note and passes, since the publish workflow checks out the
full history and recomputes; any other tree is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from registry_common import GenerationError, load_json

from toktier._jcs import canonical_json
from toktier.facade.fingerprint import repair_parameters_component
from toktier.repair.registry import RepairFamily

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = (
    REPOSITORY_ROOT / "src/toktier/routing/tables/store_fingerprint_history.v1.json"
)
HISTORY_DOMAIN_TAG = b"toktier.store_fingerprint_history.v1\x00"
SCHEMA = "toktier.store_fingerprint_history.v1"
PREIMAGE_DOMAIN = "toktier.facade.fingerprint.v1"
NOTE = (
    "Releases that wrote facade store fingerprints in the v1 preimage form. "
    "Generated from the release tags; a row is consulted only when this "
    "build's bound v2 pair equals the row's."
)
_CERTIFIED = ("certified", "certified_source")


def _show(repository: Path, tag: str, path: str) -> Any:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "show", f"{tag}:{path}"],
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GenerationError(
            f"{tag}:{path}: git is not available ({error})"
        ) from error
    if result.returncode != 0:
        raise GenerationError(
            f"{tag}:{path}: cannot be read from the repository "
            f"({result.stderr.decode('utf-8', 'replace').strip()[:200]})"
        )
    return json.loads(result.stdout)


def _one(values: Iterable[tuple[str, str]], what: str) -> tuple[str, str]:
    distinct = set(values)
    if len(distinct) != 1:
        raise GenerationError(f"{what}: expected one value, found {sorted(distinct)}")
    return distinct.pop()


def _spec(family: str, sha256: str, row: Mapping[str, Any]) -> RepairFamily:
    return RepairFamily(
        family=family,
        artifact_sha256=sha256,
        margin=int(row["margin"]),
        effective_l_max=int(row["effective_l_max"]),
        has_normalizer=row["has_normalizer"] is True,
        source_table_sha256=str(row["source_table_sha256"]),
    )


def release_row(repository: Path, tag: str) -> dict[str, Any]:
    """The history row of one release, recomputed from its tag."""
    binding = _show(repository, tag, "tools/fast_cpu_binding.json")
    registry = _show(
        repository, tag, "src/toktier/routing/tables/support_registry.v1.json"
    )
    roster = _show(
        repository, tag, "src/toktier/repair/tables/fast_repair_families.v1.json"
    )
    artifacts = registry["artifacts"]
    fast_v1, fast_v2 = _one(
        (
            (entry["source_digest"], entry["source_digest_v2"])
            for record in artifacts
            for entry in [record["backends"].get("fast_cpu") or {}]
            if entry.get("status") in _CERTIFIED
        ),
        f"{tag} fast_cpu pair",
    )
    if (fast_v1, fast_v2) != (binding["source_digest"], binding["source_digest_v2"]):
        raise GenerationError(
            f"{tag}: the registry's fast_cpu pair differs from fast_cpu_binding.json"
        )
    _host_v1, host_v2 = _one(
        (
            (delivery["host_source_digest"], delivery["host_source_digest_v2"])
            for record in artifacts
            for delivery in [
                ((record["backends"].get("gpu") or {}).get("deliveries") or {}).get(
                    "prebuilt"
                )
                or {}
            ]
            if delivery.get("status") in _CERTIFIED
        ),
        f"{tag} native host pair",
    )
    parameters: dict[str, RepairFamily] = {}
    for row in roster["families"]:
        parameters[row["artifact_sha256"]] = _spec(
            row["family"], row["artifact_sha256"], row
        )
    for record in artifacts:
        repair = (record["backends"].get("fast_cpu") or {}).get("repair")
        if repair:
            parameters[record["artifact_sha256"]] = _spec(
                record["family"], record["artifact_sha256"], repair
            )
    rows = {
        record["artifact_sha256"]: {
            "family": record["family"],
            "loader_face_sha256": record.get("loader_face_sha256"),
            "repair_parameters": repair_parameters_component(
                parameters.get(record["artifact_sha256"])
            ),
        }
        for record in artifacts
    }
    return {
        "release": tag.removeprefix("v"),
        "fast_cpu_source_digest": fast_v1,
        "fast_cpu_source_digest_v2": fast_v2,
        "native_host_source_digest_v2": host_v2,
        "artifacts": dict(sorted(rows.items())),
    }


def history_document(repository: Path, tags: Iterable[str]) -> dict[str, Any]:
    """The whole table for ``tags``, root digest included."""
    body = {
        "schema": SCHEMA,
        "preimage_domain": PREIMAGE_DOMAIN,
        "note": NOTE,
        "releases": [release_row(repository, tag) for tag in tags],
    }
    digest = (
        "sha256:"
        + hashlib.sha256(HISTORY_DOMAIN_TAG + canonical_json(body)).hexdigest()
    )
    return {**body, "root_digest": digest}


def _git_answer(repository: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def history_problems(
    path: Path = HISTORY_PATH, repository: Path = REPOSITORY_ROOT
) -> tuple[list[str], list[str]]:
    """``(problems, notes)`` of the shipped table against its recomputation."""
    if not path.exists():
        return [], []
    shipped = load_json(path)
    label = path.name
    releases = shipped.get("releases") if isinstance(shipped, dict) else None
    if not isinstance(releases, list) or not all(
        isinstance(row, dict) and isinstance(row.get("release"), str)
        for row in releases
    ):
        return [f"{label}: the table names no releases"], []
    tags = [f"v{row['release']}" for row in releases]
    absent = [
        tag
        for tag in tags
        if _git_answer(
            repository, "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"
        )
        is None
    ]
    if (
        absent
        and _git_answer(repository, "rev-parse", "--is-shallow-repository") == "true"
    ):
        return [], [
            f"note: {label}: tags {absent} are not in this shallow clone; the "
            "table is recomputed where the full history is checked out (the "
            "publish workflow)"
        ]
    try:
        expected = history_document(repository, tags)
    except GenerationError as error:
        return [f"{label}: cannot be recomputed at {tags}: {error}"], []
    problems: list[str] = []
    for key in ("schema", "preimage_domain", "note"):
        if shipped.get(key) != expected[key]:
            problems.append(f"{label}: {key} differs from the generated form")
    for row, want in zip(releases, expected["releases"], strict=True):
        if row != want:
            fields = sorted(
                key for key in set(row) | set(want) if row.get(key) != want.get(key)
            )
            problems.append(
                f"{label}: the {row['release']} row differs from its "
                f"recomputation at v{row['release']} ({', '.join(fields)})"
            )
    if shipped.get("root_digest") != expected["root_digest"]:
        problems.append(
            f"{label}: root digest {shipped.get('root_digest')} differs from "
            f"the recomputed {expected['root_digest']}"
        )
    return problems, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="recompute and compare")
    mode.add_argument("--write", nargs="+", metavar="TAG", help="write the table")
    parser.add_argument("--out", type=Path, default=HISTORY_PATH)
    arguments = parser.parse_args(argv)
    if arguments.write:
        document = history_document(REPOSITORY_ROOT, arguments.write)
        arguments.out.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{arguments.out}: {document['root_digest']}")
        return 0
    problems, notes = history_problems(arguments.out)
    for note in notes:
        print(note)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    if notes:
        print(f"{arguments.out}: not recomputed in this checkout")
    else:
        print(f"{arguments.out}: recomputed at the named tags and equal")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:  # pragma: no cover - command line surface
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
