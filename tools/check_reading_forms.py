#!/usr/bin/env python3
"""Check the reading forms the 0.3.2 evidence adds to the parity readings.

The native-front-end parity readings (``readings/fast_cpu_native_frontend_parity.json``
and ``readings/gpu_native_frontend_sm*_parity.json``, built by
``tools/aggregate_native_frontend_parity.py``) carry one row per family.
This module defines the per-family columns those rows may carry and the
checks a column has to pass beyond its schema:

``special_tokens`` (``schemas/special_tokens_column.schema.json``)
    ``add_special_tokens=True`` on the same documents as the parity pass.
    The column is only evidence when the facade answered. The vacuity
    guard is judged on the facade surface (``surface`` = ``facade``, the
    default when absent): there, on an ``identity`` or ``prefix`` face
    ``served_by.reference`` has to be zero, or a silent fallback to the
    reference would read as zero mismatches. A column taken on the native
    router directly (``surface`` = ``native_route``), which does not apply
    the facade's prefix rule, may be answered by the reference; it is then
    a reference-route reading and the count is printed as a note. On an
    ``identity`` face the column compares a core stream
    with a core stream, so it is a regression guard and says so
    (``evidence_role``). ``loader_true_equal`` is informative: it is
    reported, never a reason to refuse.

``literal_segments`` (``schemas/literal_segments_column.schema.json``)
    Documents holding added-token literals, per measured entry point of
    the native router (``id_route``, ``state_route``). The vacuity guard
    is on bytes: on a served face the reference encodes bytes only inside
    the listed guard documents, their count is reported and bounded, and
    in the GPU arm some segment bytes reach the GPU (``gpu_share``
    reported). An unserved face keeps the whole-document path, so it
    shows no segmented document. The column names the added-token table
    it was taken on (``added_frontend_fingerprint``), which a registry
    check holds to the record's.

The module also checks the engine-differential reading of a directly
judged face (``schemas/engine_differential_reading.schema.json``), the
evidence a fast CPU entry with record-held repair parameters and no
carry-over rests on: :func:`engine_differential_problems`.

Usage::

    python tools/check_reading_forms.py parity <reading.json> \
        [--require special_tokens] [--require literal_segments] [--registry FILE]

The arm (CPU or GPU) is read from the reading's ``schema``.

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
LITERAL_SEGMENTS_SCHEMA = SCHEMAS / "literal_segments_column.schema.json"
ENGINE_DIFFERENTIAL_SCHEMA = SCHEMAS / "engine_differential_reading.schema.json"

#: The constructed arm of an engine differential has to judge at least this
#: many strings per face on the frozen build (condition D6b).
CONSTRUCTED_ARM_MIN_STRINGS = 1_000_000

#: The per-family columns a parity row may carry, by name.
COLUMNS = ("special_tokens", "literal_segments")

#: The two arms of the parity readings, by reading schema.
ARMS = {
    "toktier.fast_cpu.native_frontend_parity.v1": "cpu",
    "toktier.gpu.native_frontend_parity.v1": "gpu",
}

#: Upper bound on guard documents (sent whole to the reference by the
#: engine guard) as a share of an entry point's documents. Above it the
#: column speaks mostly about the reference, not about segmentation.
GUARD_DOCUMENT_SHARE = 0.01

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
    surface = str(column.get("surface", "facade"))
    prefix = list(column["prefix_ids"])
    served = column["served_by"]
    count = int(column["documents"])
    if kind == "prefix" and len(prefix) != 1:
        problems.append(f"{label}: a prefix face names exactly one prefix id")
    if kind != "prefix" and prefix:
        problems.append(f"{label}: only a prefix face names prefix ids")
    allowed_roles = {EVIDENCE_ROLE[kind]}
    if surface != "facade" and int(served["reference"]) != 0:
        allowed_roles.add("reference_route")
    if column["evidence_role"] not in allowed_roles:
        problems.append(
            f"{label}: on a {kind} face the column is "
            f"{EVIDENCE_ROLE[kind]!r}, not {column['evidence_role']!r}"
            + (
                " (it compares a core stream with a core stream)"
                if kind == "identity"
                else ""
            )
        )
    if surface == "facade" and kind != "reference" and int(served["reference"]) != 0:
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
    """Informative lines: where the pinned loader's object answers otherwise,
    and where a column off the facade surface was answered by the reference."""
    if not isinstance(column, Mapping):
        return []
    notes: list[str] = []
    served = column.get("served_by")
    surface = column.get("surface", "facade")
    if (
        surface != "facade"
        and column.get("postprocessor_class") in ("identity", "prefix")
        and isinstance(served, Mapping)
        and int(served.get("reference", 0)) != 0
    ):
        notes.append(
            f"note: {family}: on the {surface} surface {served.get('reference')} "
            "document(s) of the add_special_tokens=True column were answered by "
            "the reference; the vacuity guard is judged on the facade surface "
            "only, informative here"
        )
    loader = column.get("loader_true_equal")
    if not isinstance(loader, Mapping) or loader.get("difference") == "none":
        return notes
    affix = ""
    if loader.get("difference") == "fixed_affix":
        affix = (
            f" (prefix {loader.get('loader_prefix_ids') or []}, "
            f"suffix {loader.get('loader_suffix_ids') or []})"
        )
    notes.append(
        f"note: {family}: the pinned loader's object answers add_special_tokens=True "
        f"differently ({loader.get('difference')}{affix}; "
        f"{loader.get('equal')}/{loader.get('strings')} equal); informative only"
    )
    return notes


def literal_segments_problems(
    family: str,
    column: object,
    *,
    arm: str | None = None,
    added_frontend_fingerprint: str | None = None,
) -> list[str]:
    """Problems of one family's ``literal_segments`` column (empty: it passes).

    ``arm`` (``cpu`` or ``gpu``) applies the arm's byte rule when given;
    ``added_frontend_fingerprint`` is the registry record's, when known.
    """
    label = f"{family}: literal_segments"
    problems = _schema_problems(label, column, LITERAL_SEGMENTS_SCHEMA)
    if problems:
        return problems
    assert isinstance(column, Mapping)
    served = column["literal_class"] == "served"
    if (
        added_frontend_fingerprint is not None
        and column["added_frontend_fingerprint"] != added_frontend_fingerprint
    ):
        problems.append(
            f"{label}: taken on added-token table "
            f"{column['added_frontend_fingerprint']}, the record names "
            f"{added_frontend_fingerprint}"
        )
    battery = column["battery"]
    if int(battery["mismatches"]) != 0:
        problems.append(f"{label}: battery {battery['mismatches']} mismatch(es)")
    seen: set[str] = set()
    for entry in column["entry_points"]:
        name = str(entry["entry_point"])
        where = f"{label}[{name}]"
        if name in seen:
            problems.append(f"{where}: the entry point is listed twice")
        seen.add(name)
        documents = int(entry["documents"])
        paths = entry["paths"]
        segment_bytes = entry["segments"]["bytes"]
        guard = entry["guard_documents"]
        if int(entry["mismatches"]) != 0:
            problems.append(f"{where}: {entry['mismatches']} mismatch(es)")
        if sum(int(value) for value in paths.values()) != documents:
            problems.append(f"{where}: the path classes do not add up to the documents")
        if int(guard["count"]) != int(paths["hf_engine_guard"]):
            problems.append(
                f"{where}: {guard['count']} guard document(s) listed, "
                f"{paths['hf_engine_guard']} counted as hf_engine_guard"
            )
        if int(guard["count"]) > int(guard["bound"]):
            problems.append(
                f"{where}: {guard['count']} guard documents exceed the stated "
                f"bound {guard['bound']}"
            )
        if int(guard["bound"]) > max(1, int(documents * GUARD_DOCUMENT_SHARE)):
            problems.append(
                f"{where}: the guard-document bound {guard['bound']} is above "
                f"{GUARD_DOCUMENT_SHARE:.0%} of the documents"
            )
        if int(guard["bytes_hf"]) > int(segment_bytes["hf"]):
            problems.append(
                f"{where}: guard documents hold more reference bytes than were "
                "encoded by the reference"
            )
        if not served:
            if int(paths["gigatoken_segmented"]) or int(paths["gpu_segmented"]):
                problems.append(
                    f"{where}: an unserved face shows segmented documents; it "
                    "keeps the whole-document path"
                )
            continue
        if int(paths["hf_added_token"]) != 0:
            problems.append(
                f"{where}: {paths['hf_added_token']} document(s) on a served face "
                "went whole to the reference for a literal"
            )
        if int(segment_bytes["hf"]) != int(guard["bytes_hf"]):
            problems.append(
                f"{where}: the reference encoded "
                f"{int(segment_bytes['hf']) - int(guard['bytes_hf'])} byte(s) "
                "outside the listed guard documents"
            )
        if arm == "cpu":
            if int(segment_bytes["gpu"]) or int(paths["gpu_segmented"]):
                problems.append(f"{where}: the CPU arm records GPU segments")
            if int(paths["gigatoken_segmented"]) < 1 or not int(
                segment_bytes["fast_cpu"]
            ):
                problems.append(
                    f"{where}: no document was segmented on the CPU engine"
                )
        elif arm == "gpu":
            total = sum(int(value) for value in segment_bytes.values())
            if int(segment_bytes["gpu"]) < 1 or int(paths["gpu_segmented"]) < 1:
                problems.append(
                    f"{where}: no segment reached the GPU; the GPU arm would "
                    "read another engine"
                )
            share = entry.get("gpu_share")
            if share is None:
                problems.append(f"{where}: the GPU arm reports no gpu_share")
            elif total and abs(float(share) - int(segment_bytes["gpu"]) / total) > 1e-6:
                problems.append(f"{where}: gpu_share does not match the bytes")
    if "id_route" not in seen:
        problems.append(f"{label}: the ID route is not measured")
    return problems


def column_problems(
    row: Mapping[str, Any],
    *,
    arm: str | None = None,
    added_frontend_fingerprint: str | None = None,
) -> list[str]:
    """Problems of every 0.3.2 column a parity row carries."""
    family = str(row.get("family", "<unknown>"))
    documents = row.get("documents")
    count = int(documents) if isinstance(documents, int) else None
    problems: list[str] = []
    if "special_tokens" in row:
        problems += special_tokens_problems(
            family, row["special_tokens"], documents=count
        )
    if "literal_segments" in row:
        problems += literal_segments_problems(
            family,
            row["literal_segments"],
            arm=arm,
            added_frontend_fingerprint=added_frontend_fingerprint,
        )
    return problems


def engine_differential_problems(
    reading: object,
    *,
    artifact: Mapping[str, Any],
    expected_documents: int,
    engine_core: Mapping[str, str],
    frozen_source_digest_v2: str,
) -> list[str]:
    """Problems of a directly judged face's engine-differential reading.

    ``artifact`` is the registry record (its ``backends.fast_cpu.repair``
    member is the parameter set the reading has to have run with);
    ``expected_documents`` is the campaign corpus size the roster's own
    differential covered; ``engine_core`` the tree's
    ``tools/engine_core_identity.py`` digests; ``frozen_source_digest_v2``
    the version-normalized fast CPU identity of the tree, which the arms
    taken on the frozen build have to name.
    """
    family = str(artifact.get("family", "<unknown>"))
    label = f"{family}: engine differential"
    problems = _schema_problems(label, reading, ENGINE_DIFFERENTIAL_SCHEMA)
    if problems:
        return problems
    assert isinstance(reading, Mapping)
    subject = reading["subject"]
    for key in ("family", "artifact_sha256", "loader_face_sha256"):
        if subject[key] != artifact.get(key):
            problems.append(
                f"{label}: subject {key} {subject[key]!r} is not the record's "
                f"{artifact.get(key)!r}"
            )
    repair = ((artifact.get("backends") or {}).get("fast_cpu") or {}).get("repair")
    if subject["repair"] != repair:
        problems.append(
            f"{label}: ran with repair parameters {subject['repair']} but the "
            f"record holds {repair}"
        )
    if int(subject["repair"]["effective_l_max"]) < int(
        subject["longest_vocabulary_token_chars"]
    ):
        problems.append(
            f"{label}: effective_l_max {subject['repair']['effective_l_max']} is "
            "below the longest vocabulary token "
            f"({subject['longest_vocabulary_token_chars']} characters)"
        )
    for key, value in engine_core.items():
        if reading["engine_core"].get(key) != value:
            problems.append(
                f"{label}: engine core {key} {reading['engine_core'].get(key)} is "
                f"not the tree's {value}"
            )
    corpus, main = reading["corpus"], reading["main_pass"]
    if int(corpus["documents"]) != expected_documents:
        problems.append(
            f"{label}: corpus of {corpus['documents']} documents, the campaign "
            f"corpus holds {expected_documents}"
        )
    if (int(main["documents"]), int(main["characters"])) != (
        int(corpus["documents"]),
        int(corpus["characters"]),
    ):
        problems.append(f"{label}: the main pass does not cover the corpus")
    if int(main["mismatches"]) != 0:
        problems.append(f"{label}: main pass {main['mismatches']} mismatch(es)")
    classes = main["path_classes"]
    if sum(int(value) for value in classes.values()) != int(main["documents"]):
        problems.append(f"{label}: main-pass path classes do not add up")
    residue = int(classes["hf_full_added_token"]) + int(
        classes["hf_full_engine_guard"]
    )
    if int(main["residue"]["count"]) != residue:
        problems.append(
            f"{label}: the residue lists {main['residue']['count']} documents, the "
            f"path classes leave {residue} not served by the engine"
        )
    append = reading["append_replay"]
    if int(append["mismatches"]) != 0:
        problems.append(f"{label}: append replay {append['mismatches']} mismatch(es)")
    constructed = reading["constructed_arm"]
    if int(constructed["strings"]) < CONSTRUCTED_ARM_MIN_STRINGS:
        problems.append(
            f"{label}: the constructed arm judged {constructed['strings']} strings, "
            f"fewer than {CONSTRUCTED_ARM_MIN_STRINGS}"
        )
    if int(constructed["mismatches"]) != 0:
        problems.append(
            f"{label}: constructed arm {constructed['mismatches']} mismatch(es)"
        )
    if int(constructed["threshold_decisive"]) < 1:
        problems.append(
            f"{label}: in the constructed arm the window bound decided no outcome, "
            "so the arm says nothing about effective_l_max"
        )
    supplement = reading["residue_supplement"]
    if int(supplement["documents"]) != int(main["residue"]["count"]):
        problems.append(
            f"{label}: the residue supplement re-ran {supplement['documents']} of "
            f"{main['residue']['count']} residue documents"
        )
    if int(supplement["mismatches_full"]) or int(supplement["mismatches_append"]):
        problems.append(f"{label}: the residue supplement records mismatches")
    supplement_classes = supplement["path_classes"]
    if sum(int(value) for value in supplement_classes.values()) != int(
        supplement["documents"]
    ):
        problems.append(f"{label}: residue-supplement path classes do not add up")
    if int(supplement_classes["hf_added_token"]) != int(
        supplement["reference_answered"]["count"]
    ):
        problems.append(
            f"{label}: {supplement_classes['hf_added_token']} residue document(s) "
            "went to the reference for a literal on the frozen build, "
            f"{supplement['reference_answered']['count']} are listed as "
            "reference-answered"
        )
    bridging = reading["bridging_check"]
    if int(bridging["mismatches_full"]) or int(bridging["mismatches_append"]):
        problems.append(f"{label}: the bridging check records mismatches")
    if int(bridging["path_class_changes_literal_free"]) != 0:
        problems.append(
            f"{label}: {bridging['path_class_changes_literal_free']} literal-free "
            "document(s) changed path class on the frozen build"
        )
    planned = float(bridging["sample_share"]) * int(corpus["documents"])
    if abs(int(bridging["documents"]) - planned) > max(1.0, 0.01 * planned):
        problems.append(
            f"{label}: the bridging check ran {bridging['documents']} documents, "
            f"its sample share plans {planned:.0f}"
        )
    limit = reading["sensitivity_limit"]
    if float(limit["per_document"]) * int(bridging["documents"]) < 1 - 1e-9:
        problems.append(
            f"{label}: the stated sensitivity limit {limit['per_document']} is finer "
            "than the bridging sample can resolve"
        )
    for arm in ("constructed_arm", "residue_supplement", "bridging_check"):
        recorded = reading[arm]["build"]["source_digest_v2"]
        if recorded != frozen_source_digest_v2:
            problems.append(
                f"{label}: {arm} was taken on fast CPU v2 {recorded}, not the "
                f"frozen tree's {frozen_source_digest_v2}"
            )
    return problems


def registry_fingerprints(registry: Mapping[str, Any]) -> dict[str, str]:
    """``{family: added_frontend_fingerprint}`` of a support registry."""
    frontends = {
        str(row.get("added_frontend_id")): str(row.get("added_frontend_fingerprint"))
        for row in registry.get("added_frontends") or ()
        if isinstance(row, Mapping)
    }
    return {
        str(artifact.get("family")): frontends[str(artifact.get("added_frontend_id"))]
        for artifact in registry.get("artifacts") or ()
        if isinstance(artifact, Mapping)
        and str(artifact.get("added_frontend_id")) in frontends
    }


def parity_problems(
    reading: Mapping[str, Any],
    require: Iterable[str] = (),
    registry: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """``(problems, notes)`` of a parity reading's rows and their columns."""
    required = set(require)
    unknown = required - set(COLUMNS)
    if unknown:
        raise GenerationError(f"unknown column(s): {sorted(unknown)}")
    rows = reading.get("rows")
    if not isinstance(rows, list) or not rows:
        return ["the reading has no rows"], []
    arm = ARMS.get(str(reading.get("schema")))
    fingerprints = registry_fingerprints(registry) if registry is not None else {}
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
        problems += column_problems(
            row, arm=arm, added_frontend_fingerprint=fingerprints.get(family)
        )
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
    parity.add_argument(
        "--registry",
        type=Path,
        default=REPOSITORY_ROOT / "tables" / "support_registry.json",
        help="support registry naming each family's added-token table",
    )
    arguments = parser.parse_args(argv)
    reading = load_json(arguments.reading)
    if not isinstance(reading, dict):
        raise GenerationError(f"{arguments.reading}: not a JSON object")
    registry = load_json(arguments.registry)
    problems, notes = parity_problems(reading, arguments.require, registry)
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
