#!/usr/bin/env python3
"""Check the reading forms the 0.3.2 evidence adds to the parity readings.

The native-front-end parity readings (``readings/fast_cpu_native_frontend_parity.json``
and ``readings/gpu_native_frontend_sm*_parity.json``, built by
``tools/aggregate_native_frontend_parity.py``) carry one row per family.
This module defines the per-family columns those rows may carry and the
checks a column has to pass beyond its schema:

``special_tokens`` (``schemas/special_tokens_column.schema.json``)
    ``add_special_tokens=True`` on the same documents as the parity pass.
    The column is only evidence when the facade answered: on an
    ``identity`` or ``prefix`` face ``served_by.reference`` has to be
    zero, or a silent fallback to the reference would read as zero
    mismatches. On an ``identity`` face the column compares a core stream
    with a core stream, so it is a regression guard and says so
    (``evidence_role``). ``loader_true_equal`` is informative: it is
    reported, never a reason to refuse.

Usage::

    python tools/check_reading_forms.py parity <reading.json> [--require special_tokens]

The command exits 0 when every present column passes (and every required
column is present), 1 otherwise, and prints the informative notes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from registry_common import GenerationError, load_json, schema_violations

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = REPOSITORY_ROOT / "schemas"
SPECIAL_TOKENS_SCHEMA = SCHEMAS / "special_tokens_column.schema.json"

#: The per-family columns a parity row may carry, by name.
COLUMNS = ("special_tokens",)

#: What the ``True`` column shows, by post-processor class.
EVIDENCE_ROLE = {
    "identity": "regression_guard",
    "prefix": "prefix_rule",
    "reference": "reference_route",
}


def _schema_problems(label: str, value: object, schema_path: Path) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{label}: is not an object"]
    return [
        f"{label}: {message}"
        for message in schema_violations(value, load_json(schema_path))
    ]


def special_tokens_problems(
    family: str, column: object, *, documents: int | None = None
) -> list[str]:
    """Problems of one family's ``special_tokens`` column (empty: it passes).

    ``documents`` is the parity row's document count; the column is taken
    on the same documents, so the two must be equal when it is given.
    """
    label = f"{family}: special_tokens"
    problems = _schema_problems(label, column, SPECIAL_TOKENS_SCHEMA)
    if problems:
        return problems
    assert isinstance(column, Mapping)
    kind = str(column["postprocessor_class"])
    prefix = list(column["prefix_ids"])
    served = column["served_by"]
    count = int(column["documents"])
    if kind == "prefix" and len(prefix) != 1:
        problems.append(f"{label}: a prefix face names exactly one prefix id")
    if kind != "prefix" and prefix:
        problems.append(f"{label}: only a prefix face names prefix ids")
    if column["evidence_role"] != EVIDENCE_ROLE[kind]:
        problems.append(
            f"{label}: on a {kind} face the column is "
            f"{EVIDENCE_ROLE[kind]!r}, not {column['evidence_role']!r}"
            + (
                " (it compares a core stream with a core stream)"
                if kind == "identity"
                else ""
            )
        )
    if kind != "reference" and int(served["reference"]) != 0:
        problems.append(
            f"{label}: {served['reference']} document(s) on a {kind} face were "
            "answered by the reference; the column would compare the reference "
            "with itself"
        )
    if kind == "reference" and int(served["facade"]) != 0:
        problems.append(f"{label}: a reference face is not served by the facade")
    if int(served["facade"]) + int(served["reference"]) != count:
        problems.append(f"{label}: served_by does not add up to the documents")
    if documents is not None and count != documents:
        problems.append(
            f"{label}: {count} documents, the parity row judged {documents}; the "
            "column is taken on the same documents"
        )
    if int(column["mismatches"]) != 0:
        problems.append(f"{label}: {column['mismatches']} mismatch(es)")
    loader = column.get("loader_true_equal")
    if isinstance(loader, Mapping):
        strings, equal = int(loader["strings"]), int(loader["equal"])
        difference = loader["difference"]
        if equal > strings:
            problems.append(
                f"{label}: loader_true_equal counts more equal than strings"
            )
        if difference == "none" and equal != strings:
            problems.append(
                f"{label}: loader_true_equal says none but counts a difference"
            )
        if difference == "not_built" and strings != 0:
            problems.append(f"{label}: loader_true_equal not_built counts strings")
        if difference == "fixed_affix" and not (
            loader.get("loader_prefix_ids") or loader.get("loader_suffix_ids")
        ):
            problems.append(f"{label}: loader_true_equal fixed_affix names no affix")
    return problems


def special_tokens_notes(family: str, column: object) -> list[str]:
    """Informative lines: where the pinned loader's object answers otherwise."""
    if not isinstance(column, Mapping):
        return []
    loader = column.get("loader_true_equal")
    if not isinstance(loader, Mapping) or loader.get("difference") == "none":
        return []
    affix = ""
    if loader.get("difference") == "fixed_affix":
        affix = (
            f" (prefix {loader.get('loader_prefix_ids') or []}, "
            f"suffix {loader.get('loader_suffix_ids') or []})"
        )
    return [
        f"note: {family}: the pinned loader's object answers add_special_tokens=True "
        f"differently ({loader.get('difference')}{affix}; "
        f"{loader.get('equal')}/{loader.get('strings')} equal); informative only"
    ]


def column_problems(row: Mapping[str, Any]) -> list[str]:
    """Problems of every 0.3.2 column a parity row carries."""
    family = str(row.get("family", "<unknown>"))
    documents = row.get("documents")
    count = int(documents) if isinstance(documents, int) else None
    problems: list[str] = []
    if "special_tokens" in row:
        problems += special_tokens_problems(
            family, row["special_tokens"], documents=count
        )
    return problems


def parity_problems(
    reading: Mapping[str, Any], require: Iterable[str] = ()
) -> tuple[list[str], list[str]]:
    """``(problems, notes)`` of a parity reading's rows and their columns."""
    required = set(require)
    unknown = required - set(COLUMNS)
    if unknown:
        raise GenerationError(f"unknown column(s): {sorted(unknown)}")
    rows = reading.get("rows")
    if not isinstance(rows, list) or not rows:
        return ["the reading has no rows"], []
    problems: list[str] = []
    notes: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            problems.append("a row is not an object")
            continue
        family = str(row.get("family", "<unknown>"))
        for name in sorted(required):
            if name not in row:
                problems.append(f"{family}: the row carries no {name} column")
        problems += column_problems(row)
        notes += special_tokens_notes(family, row.get("special_tokens"))
    return problems, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    parity = commands.add_parser("parity", help="check a native-front-end reading")
    parity.add_argument("reading", type=Path)
    parity.add_argument(
        "--require",
        action="append",
        default=[],
        choices=COLUMNS,
        help="a column every row must carry (repeatable)",
    )
    arguments = parser.parse_args(argv)
    reading = load_json(arguments.reading)
    if not isinstance(reading, dict):
        raise GenerationError(f"{arguments.reading}: not a JSON object")
    problems, notes = parity_problems(reading, arguments.require)
    for note in notes:
        print(note)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"{arguments.reading}: reading forms passed ({len(reading['rows'])} rows)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
