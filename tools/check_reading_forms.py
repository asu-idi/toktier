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
    (``evidence_role``). ``executed_by`` records which engine produced
    the core streams (the native request runtime's execution counts read
    around the column): ``served_by`` alone cannot show it, and under a
    policy that plans the reference the facade still applies the prefix
    rule to reference-made core streams. On the facade surface, on an
    ``identity`` or ``prefix`` face, the arm's engine accounts for at
    least :data:`SPECIAL_TOKENS_ENGINE_SHARE` of the executions.
    ``loader_true_equal`` is informative: it is reported, never a reason
    to refuse.

``literal_segments`` (``schemas/literal_segments_column.schema.json``)
    Documents holding added-token literals, per measured entry point of
    the native router (``id_route``, ``state_route``). The vacuity guard
    is on bytes: on a served face the reference encodes bytes only inside
    the listed guard documents, their count is reported and bounded, and
    in the GPU arm some segment bytes reach the GPU (``gpu_share``
    reported). A served face sends no document whole to the reference for
    a literal (``hf_added_token`` = 0); the one exception is a class of
    its own, ``hf_segments_below_crossover`` (GPU arm only: a route whose
    only backend after the GPU is the reference, every text segment below
    the crossover), listed with its bytes in ``below_crossover_documents``.
    A second class of its own is ``hf_span_guard`` (state route, GPU arm
    only): the state-route acceptance refused a device row because the
    text is not stable under the normalizer (or the span bridge failed),
    and the reference re-encoded the document; it is listed with its bytes
    in ``span_guard_documents`` and counted apart from the engine guard.
    A third is ``hf_device_declined`` (GPU arm only, either entry point;
    an optional key that reads 0 when absent): the device declined a text
    segment itself (the execution record shows ``R_EXEC_FAULT`` from the
    ``gpu`` backend, scope ``segment``) and the reference, the only entry
    of the chain after the device, encoded the whole document. Those
    documents are listed with their bytes, chain and event in
    ``device_declined_documents`` and share the guard documents' bounds
    (:data:`GUARD_DOCUMENT_SHARE` of the documents,
    :data:`GUARD_BYTE_SHARE` of the bytes). The class stands only where
    both hold: the chain has no CPU engine between the device and the
    reference (:data:`DEVICE_DECLINED_CHAIN`), and the record shows the
    device's own decline (:data:`DEVICE_DECLINED_EVENT`).
    An unserved face keeps the whole-document path, so it shows no
    segmented, below-crossover, span-guard or device-declined document.
    The column names
    the added-token table it was taken on (``added_frontend_fingerprint``),
    which a registry check holds to the record's.

The module also checks the engine-differential reading of a directly
judged face (``schemas/engine_differential_reading.schema.json``), the
evidence a fast CPU entry with record-held repair parameters and no
carry-over rests on: :func:`engine_differential_problems`.

A release reading (``generate_registry.py --release-check``) requires both
columns on every certified fast CPU row and every GPU parity row, with the
``special_tokens`` column taken on the facade surface:
:func:`release_column_problems`. ``--require special_tokens`` asks the
same of the column: a ``native_route`` column does not satisfy it, since
the native router does not apply the facade's prefix rule.

The rider document of the chunk-cut campaign
(``schemas/chunked_cuts_riders.schema.json``, written by
``run_chunked_cut_campaign.py summarize --riders-out``) is checked by
:func:`riders_problems`; ``summarize`` runs that check on the document it
writes.

Usage::

    python tools/check_reading_forms.py parity <reading.json> \
        [--require special_tokens] [--require literal_segments] [--registry FILE]
    python tools/check_reading_forms.py riders <riders.json> \
        [--require special_tokens] [--require literal_segments]

The arm (CPU or GPU) is read from the reading's ``schema``.

The command exits 0 when every present column passes (and every required
column is present), 1 otherwise, and prints the informative notes.
"""

from __future__ import annotations

import argparse
import math
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
RIDERS_SCHEMA = SCHEMAS / "chunked_cuts_riders.schema.json"

#: The constructed arm of an engine differential has to judge at least this
#: many strings per face on the frozen build (condition D6b).
CONSTRUCTED_ARM_MIN_STRINGS = 1_000_000

#: The window bound the constructed arm compares the face's own with: the
#: roster's standing value. Below it the arm has to show its power: at
#: least this share of its strings decided by the bound (lane E's dry run
#: read 10-12 % on the three faces below 128). At it the arm cannot
#: distinguish the two engines, so the count has to be zero and the face
#: rests on the roster's standing value.
CONSTRUCTED_ARM_COMPARISON_L_MAX = 128
CONSTRUCTED_ARM_DECISIVE_SHARE = 0.01

#: The bridging check re-runs at least this share of the corpus on the
#: frozen build (the campaign plans exactly 1 %).
BRIDGING_MIN_SHARE = 0.01

#: The append replay covers at least this share of the corpus documents
#: (the campaign replays one document in 40, 2.5 %; the floor leaves room
#: for documents too short to cut).
APPEND_REPLAY_MIN_SHARE = 0.02

#: On a served face at most this share of the residue documents may go to
#: the reference for a literal on the frozen build (the literal segmenter
#: answers them otherwise).
RESIDUE_REFERENCE_SHARE = 0.01

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

#: Upper bound on the reference bytes of the guard documents as a share of
#: all bytes an entry point encoded: the document bound alone would admit
#: a few long guard documents carrying most of the bytes.
GUARD_BYTE_SHARE = 0.01

#: The chain and the execution-record event a ``hf_device_declined``
#: document has to show (the narrower of the two readings, both required):
#: the reference follows the device directly, so a declined segment sends
#: the whole document to it, and the device itself declined (a fallback
#: fault of the gpu backend on a segment), not a crossover, a size bound or
#: an unserved literal.
DEVICE_DECLINED_CHAIN = ("gpu", "hf")
DEVICE_DECLINED_EVENT = {"code": "R_EXEC_FAULT", "backend": "gpu", "scope": "segment"}

#: Absolute floors beside the ratio checks: a column of a handful of
#: documents passes every ratio. An entry point of the literal_segments
#: column judges at least this many documents ...
LITERAL_ENTRY_POINT_MIN_DOCUMENTS = 1_000
#: ... the literal-dense battery at least this many strings per family
#: (the battery generator reaches 15,000 to 87,000 strings per family; the
#: families with the fewest literals set the floor) ...
LITERAL_BATTERY_MIN_STRINGS = 10_000
#: ... and in the GPU arm the ID route sends at least this share of its
#: segment bytes to the GPU (a share near zero would read the CPU engine
#: under the GPU arm's name).
GPU_SHARE_FLOOR = 0.5

#: Each rider arm a campaign requested runs on at least this many
#: documents of every family in the rider document.
RIDER_MIN_DOCUMENTS = 1

#: The literal class every certified family has in 0.3.2: every face the
#: registry certifies is served by the native literal segmenter. The
#: column's own ``literal_class`` is held to it in a release reading (and
#: by default on the command line), so a face demoted at run time, or a
#: harness that writes ``unserved``, does not pass as an unserved face.
EXPECTED_LITERAL_CLASS = "served"

#: On the facade surface, on an identity or prefix face, the share of the
#: ``special_tokens`` column's executions the arm's engine has to account
#: for (lane RCP measured a column answered entirely by the reference, on a
#: build the registry does not bind yet, under the default policy).
SPECIAL_TOKENS_ENGINE_SHARE = 0.5

#: The engines that count as the arm's own in the ``special_tokens`` and
#: rider engine records (any accelerated engine when the arm is unknown).
ARM_ENGINES = {"cpu": ("fast_cpu",), "gpu": ("gpu",), None: ("fast_cpu", "gpu")}

#: What the ``True`` column shows, by post-processor class.
EVIDENCE_ROLE = {
    "identity": "regression_guard",
    "prefix": "prefix_rule",
    "reference": "reference_route",
}


def _non_finite(value: object, path: str = "") -> list[str]:
    """Paths of non-finite numbers (``NaN``, ``Infinity``) inside ``value``.

    ``json.loads`` accepts those tokens and every comparison with ``NaN`` is
    false, so a bound such as ``minimum`` or ``share < floor`` would let one
    through; they are refused before any rule reads the document.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return [path or "<root>"]
    if isinstance(value, Mapping):
        return [
            found
            for key, item in value.items()
            for found in _non_finite(item, f"{path}/{key}" if path else str(key))
        ]
    if isinstance(value, list):
        return [
            found
            for index, item in enumerate(value)
            for found in _non_finite(item, f"{path}/{index}")
        ]
    return []


def _schema_problems(label: str, value: object, schema_path: Path) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{label}: is not an object"]
    non_finite = _non_finite(value)
    if non_finite:
        return [f"{label}: {path}: not a finite number" for path in non_finite]
    return [
        f"{label}: {message}"
        for message in schema_violations(value, load_json(schema_path))
    ]


def special_tokens_problems(
    family: str,
    column: object,
    *,
    documents: int | None = None,
    arm: str | None = None,
) -> list[str]:
    """Problems of one family's ``special_tokens`` column (empty: it passes).

    ``documents`` is the parity row's document count; the column is taken
    on the same documents, so the two must be equal when it is given.
    ``arm`` (``cpu`` or ``gpu``) names the engine the executions have to
    come from; when it is not given, either accelerated engine counts.
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
    problems += _engine_share_problems(
        label,
        column["executed_by"],
        arm=arm,
        judged=surface == "facade" and kind != "reference",
    )
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


def _engine_share_problems(
    label: str,
    executed: Mapping[str, Any],
    *,
    arm: str | None,
    judged: bool,
    documents: int | None = None,
) -> list[str]:
    """The engine-level vacuity guard of a ``True`` column or rider arm.

    ``executed`` counts executions by the engine that produced the core
    stream. A column with no execution recorded speaks about nothing; when
    ``judged`` (an identity or prefix face on the facade surface), the
    arm's engines (:data:`ARM_ENGINES`) account for at least
    :data:`SPECIAL_TOKENS_ENGINE_SHARE` of them (or of ``documents``, when
    each document records one execution).
    """
    total = sum(int(value) for value in executed.values())
    if total < 1:
        return [f"{label}: executed_by records no execution"]
    if not judged:
        return []
    engines = ARM_ENGINES.get(arm, ARM_ENGINES[None])
    own = sum(int(executed.get(engine, 0)) for engine in engines)
    base = total if documents is None else max(total, documents)
    if own < SPECIAL_TOKENS_ENGINE_SHARE * base:
        return [
            f"{label}: {own} of {base} core stream(s) came from "
            f"{'/'.join(engines)}, below {SPECIAL_TOKENS_ENGINE_SHARE:.0%}; the "
            f"column reads another engine ({dict(executed)}), e.g. the reference "
            "under a policy that does not bind the accelerated engine"
        ]
    return []


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
    expected_literal_class: str | None = None,
) -> list[str]:
    """Problems of one family's ``literal_segments`` column (empty: it passes).

    ``arm`` (``cpu`` or ``gpu``) applies the arm's byte rule when given;
    ``added_frontend_fingerprint`` is the registry record's, when known;
    ``expected_literal_class`` is the class the face has to have, when
    known (the column's own value is otherwise self-declared).
    """
    label = f"{family}: literal_segments"
    problems = _schema_problems(label, column, LITERAL_SEGMENTS_SCHEMA)
    if problems:
        return problems
    assert isinstance(column, Mapping)
    served = column["literal_class"] == "served"
    if (
        expected_literal_class is not None
        and column["literal_class"] != expected_literal_class
    ):
        problems.append(
            f"{label}: the column reads literal_class "
            f"{column['literal_class']!r}; the face is "
            f"{expected_literal_class!r}"
        )
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
    if int(battery["strings"]) < LITERAL_BATTERY_MIN_STRINGS:
        problems.append(
            f"{label}: the battery judged {battery['strings']} strings, fewer "
            f"than {LITERAL_BATTERY_MIN_STRINGS}"
        )
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
        below_count = int(paths["hf_segments_below_crossover"])
        below = entry.get("below_crossover_documents")
        below_bytes = 0
        span_count = int(paths["hf_span_guard"])
        span = entry.get("span_guard_documents")
        span_bytes = 0
        declined_count = int(paths.get("hf_device_declined", 0))
        declined = entry.get("device_declined_documents")
        declined_bytes = 0
        if declined is not None:
            declined_bytes = int(declined["bytes_hf"])
            if int(declined["count"]) != declined_count:
                problems.append(
                    f"{where}: {declined['count']} device-declined document(s) "
                    f"listed, {declined_count} counted as hf_device_declined"
                )
        elif declined_count:
            problems.append(
                f"{where}: {declined_count} document(s) counted as "
                "hf_device_declined without device_declined_documents"
            )
        if declined_count and arm != "gpu":
            # Only a device declines a segment; a reading whose arm is not
            # named cannot show that the documents came from one.
            problems.append(
                f"{where}: {declined_count} device-declined document(s) outside "
                "the GPU arm"
            )
        if declined_count and declined is not None:
            chain = [str(item) for item in declined["chain"]]
            if tuple(chain) != DEVICE_DECLINED_CHAIN:
                problems.append(
                    f"{where}: device-declined documents on the chain {chain}; "
                    "the class stands only where the reference follows the "
                    f"device directly ({list(DEVICE_DECLINED_CHAIN)}, no CPU "
                    "engine between them)"
                )
            if dict(declined["event"]) != DEVICE_DECLINED_EVENT:
                problems.append(
                    f"{where}: device-declined documents recorded with event "
                    f"{dict(declined['event'])}; the class needs the device's "
                    f"own decline ({DEVICE_DECLINED_EVENT})"
                )
        if span is not None:
            span_bytes = int(span["bytes_hf"])
            if int(span["count"]) != span_count:
                problems.append(
                    f"{where}: {span['count']} span-guard document(s) listed, "
                    f"{span_count} counted as hf_span_guard"
                )
        elif span_count:
            problems.append(
                f"{where}: {span_count} document(s) counted as hf_span_guard "
                "without span_guard_documents"
            )
        if span_count and name != "state_route":
            # The span guard decides on a device row entering stored state;
            # the ID route adopts no state.
            problems.append(
                f"{where}: {span_count} document(s) counted as hf_span_guard "
                "outside the state route"
            )
        if below is not None:
            below_bytes = int(below["bytes_hf"])
            if int(below["count"]) != below_count:
                problems.append(
                    f"{where}: {below['count']} below-crossover document(s) "
                    f"listed, {below_count} counted as "
                    "hf_segments_below_crossover"
                )
        elif below_count:
            problems.append(
                f"{where}: {below_count} document(s) counted as "
                "hf_segments_below_crossover without below_crossover_documents"
            )
        if int(entry["mismatches"]) != 0:
            problems.append(f"{where}: {entry['mismatches']} mismatch(es)")
        if documents < LITERAL_ENTRY_POINT_MIN_DOCUMENTS:
            problems.append(
                f"{where}: {documents} documents, fewer than "
                f"{LITERAL_ENTRY_POINT_MIN_DOCUMENTS}"
            )
        total_bytes = sum(int(value) for value in segment_bytes.values())
        if int(guard["bytes_hf"]) > GUARD_BYTE_SHARE * total_bytes:
            problems.append(
                f"{where}: guard documents hold {guard['bytes_hf']} of "
                f"{total_bytes} bytes, above {GUARD_BYTE_SHARE:.0%}"
            )
        if declined_count:
            # The device-declined documents count toward the guard
            # documents' bounds: together at most the stated bound (itself at
            # most 1 % of the documents) and 1 % of the bytes.
            held = int(guard["count"]) + declined_count
            if held > int(guard["bound"]):
                problems.append(
                    f"{where}: {guard['count']} guard and {declined_count} "
                    f"device-declined document(s) exceed the bound {guard['bound']}"
                )
            held_bytes = int(guard["bytes_hf"]) + declined_bytes
            if held_bytes > GUARD_BYTE_SHARE * total_bytes:
                problems.append(
                    f"{where}: guard and device-declined documents hold "
                    f"{held_bytes} of {total_bytes} bytes, above "
                    f"{GUARD_BYTE_SHARE:.0%}"
                )
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
        if int(guard["bound"]) > int(documents * GUARD_DOCUMENT_SHARE):
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
            if (
                int(paths["gigatoken_segmented"])
                or int(paths["gpu_segmented"])
                or below_count
                or span_count
                or declined_count
            ):
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
        listed_hf = int(guard["bytes_hf"]) + below_bytes + span_bytes + declined_bytes
        if int(segment_bytes["hf"]) != listed_hf:
            problems.append(
                f"{where}: the reference encoded "
                f"{int(segment_bytes['hf']) - listed_hf} byte(s) "
                "outside the listed guard, below-crossover and span-guard "
                "documents and the device-declined documents"
            )
        if arm == "cpu":
            if (
                int(segment_bytes["gpu"])
                or int(paths["gpu_segmented"])
                or below_count
                or span_count
            ):
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
            elif name == "id_route" and float(share) < GPU_SHARE_FLOOR:
                problems.append(
                    f"{where}: gpu_share {share} is below {GPU_SHARE_FLOOR}"
                )
    if "id_route" not in seen:
        problems.append(f"{label}: the ID route is not measured")
    return problems


def column_problems(
    row: Mapping[str, Any],
    *,
    arm: str | None = None,
    added_frontend_fingerprint: str | None = None,
    expected_literal_class: str | None = None,
) -> list[str]:
    """Problems of every 0.3.2 column a parity row carries."""
    family = str(row.get("family", "<unknown>"))
    documents = row.get("documents")
    count = int(documents) if isinstance(documents, int) else None
    problems: list[str] = []
    if "special_tokens" in row:
        problems += special_tokens_problems(
            family, row["special_tokens"], documents=count, arm=arm
        )
    if "literal_segments" in row:
        problems += literal_segments_problems(
            family,
            row["literal_segments"],
            arm=arm,
            added_frontend_fingerprint=added_frontend_fingerprint,
            expected_literal_class=expected_literal_class,
        )
    return problems


def _surface_problems(family: str, row: Mapping[str, Any]) -> list[str]:
    """The ``special_tokens`` column of ``row`` has to be the facade's."""
    column = row.get("special_tokens")
    if not isinstance(column, Mapping):
        return []
    surface = str(column.get("surface", "facade"))
    if surface == "facade":
        return []
    return [
        f"{family}: special_tokens was taken on the {surface} surface; the "
        "column has to be the facade's, the only surface that applies the "
        "prefix rule"
    ]


def release_column_problems(
    row: Mapping[str, Any],
    *,
    arm: str | None = None,
    added_frontend_fingerprint: str | None = None,
) -> list[str]:
    """Problems of a certified row of a release reading (empty: it passes).

    Both 0.3.2 columns are present, the ``special_tokens`` column is the
    facade's, the ``literal_segments`` column reads the class every
    certified family has (:data:`EXPECTED_LITERAL_CLASS`), and every
    column passes :func:`column_problems`.
    """
    family = str(row.get("family", "<unknown>"))
    problems = [
        f"{family}: the row carries no {name} column; a release reading "
        "requires it"
        for name in COLUMNS
        if name not in row
    ]
    problems += _surface_problems(family, row)
    problems += column_problems(
        row,
        arm=arm,
        added_frontend_fingerprint=added_frontend_fingerprint,
        expected_literal_class=EXPECTED_LITERAL_CLASS,
    )
    return problems


def engine_differential_problems(
    reading: object,
    *,
    artifact: Mapping[str, Any],
    expected_documents: int,
    engine_core: Mapping[str, str],
    frozen_source_digest_v2: str,
    expected_characters: int | None = None,
    expected_oracle_version: str | None = None,
) -> list[str]:
    """Problems of a directly judged face's engine-differential reading.

    ``artifact`` is the registry record (its ``backends.fast_cpu.repair``
    member is the parameter set the reading has to have run with);
    ``expected_documents`` and ``expected_characters`` are the campaign
    corpus the roster's own differential covered
    (``tools/fast_cpu_binding.json`` ``validation``), and
    ``expected_oracle_version`` the reference version it was judged against
    (the binding's ``oracle.tokenizers``); ``engine_core`` the tree's
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
    if (
        expected_oracle_version is not None
        and corpus["oracle_version"] != expected_oracle_version
    ):
        problems.append(
            f"{label}: judged against tokenizers {corpus['oracle_version']}, the "
            f"campaign's reference is {expected_oracle_version}"
        )
    if (
        expected_characters is not None
        and int(corpus["characters"]) != expected_characters
    ):
        problems.append(
            f"{label}: corpus of {corpus['characters']} characters, the campaign "
            f"corpus holds {expected_characters}"
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
    if int(append["documents"]) > int(corpus["documents"]):
        problems.append(
            f"{label}: the append replay counts {append['documents']} documents, "
            f"more than the corpus holds ({corpus['documents']})"
        )
    if int(append["threshold_decisive"]) > int(append["documents"]):
        problems.append(
            f"{label}: the append replay counts more decided outcomes than "
            "documents"
        )
    if int(append["documents"]) < APPEND_REPLAY_MIN_SHARE * int(corpus["documents"]):
        problems.append(
            f"{label}: the append replay covered {append['documents']} documents, "
            f"below {APPEND_REPLAY_MIN_SHARE:.0%} of the corpus"
        )
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
    if int(constructed["threshold_decisive"]) > int(constructed["strings"]):
        problems.append(
            f"{label}: the constructed arm counts {constructed['threshold_decisive']} "
            f"decided outcomes of {constructed['strings']} strings"
        )
    if int(constructed["mismatches"]) > int(constructed["strings"]):
        problems.append(
            f"{label}: the constructed arm counts more mismatches than strings"
        )
    own_l_max = int(subject["repair"]["effective_l_max"])
    comparison = int(constructed["comparison_l_max"])
    decisive = int(constructed["threshold_decisive"])
    if comparison != CONSTRUCTED_ARM_COMPARISON_L_MAX:
        problems.append(
            f"{label}: the constructed arm compared with window bound {comparison}, "
            f"not {CONSTRUCTED_ARM_COMPARISON_L_MAX}"
        )
    elif own_l_max < comparison:
        needed = CONSTRUCTED_ARM_DECISIVE_SHARE * int(constructed["strings"])
        if decisive < 1 or decisive < needed:
            problems.append(
                f"{label}: in the constructed arm the window bound decided "
                f"{decisive} of {constructed['strings']} outcomes, below "
                f"{CONSTRUCTED_ARM_DECISIVE_SHARE:.0%}, so the arm says too little "
                f"about effective_l_max {own_l_max} against {comparison}"
            )
    elif own_l_max == comparison and decisive != 0:
        problems.append(
            f"{label}: effective_l_max equals the comparison bound {comparison}, "
            f"yet {decisive} outcome(s) are counted as decided by it"
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
    if int(supplement["reference_answered"]["count"]) > int(
        RESIDUE_REFERENCE_SHARE * int(supplement["documents"])
    ):
        problems.append(
            f"{label}: {supplement['reference_answered']['count']} of "
            f"{supplement['documents']} residue documents went to the reference "
            f"for a literal on the frozen build, above {RESIDUE_REFERENCE_SHARE:.0%} "
            "for a served face"
        )
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
    if int(bridging["documents"]) > int(corpus["documents"]):
        problems.append(
            f"{label}: the bridging check counts {bridging['documents']} documents, "
            f"more than the corpus holds ({corpus['documents']})"
        )
    if float(bridging["sample_share"]) < BRIDGING_MIN_SHARE:
        problems.append(
            f"{label}: the bridging sample share {bridging['sample_share']} is "
            f"below {BRIDGING_MIN_SHARE:.0%}"
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


def riders_problems(document: object, require: Iterable[str] = ()) -> list[str]:
    """Problems of a chunk-cut campaign's rider document (empty: it passes).

    The document is ``summarize --riders-out``'s
    (``schemas/chunked_cuts_riders.schema.json``). Every family ran every
    arm the document names (and every arm in ``require``) on at least
    :data:`RIDER_MIN_DOCUMENTS` documents; no arm records a mismatch; the
    ``special_tokens`` arm has no vacuous document (an identity or prefix
    face answered by the reference) and none whose answering counter or
    executing engine the facade did not report, and on its identity and
    prefix documents an accelerated engine produced at least
    :data:`SPECIAL_TOKENS_ENGINE_SHARE` of the core streams
    (``executed_by``); the ``literal_segments`` arm has no document whose
    path was not reported, segments at least one document on an
    accelerated engine, sends no document whole to the reference for a
    literal (every family of the campaign is a served face), and keeps
    the reference paths -- the engine guard and the span guard, each
    counted under its own name -- to :data:`GUARD_DOCUMENT_SHARE` of its
    documents and the reference bytes to :data:`GUARD_BYTE_SHARE` of the
    reported bytes. Otherwise a corpus-scale reading could say "0
    mismatches" with every document answered by the reference. The
    document names the policy the units ran under (``policy``).
    """
    problems = _schema_problems("riders", document, RIDERS_SCHEMA)
    if problems:
        return problems
    assert isinstance(document, Mapping)
    requested = set(document["riders"])
    missing = set(require) - requested
    if missing:
        problems.append(f"riders: the campaign did not run {sorted(missing)}")
    for family, arms in sorted(document["families"].items()):
        for name in sorted(requested | set(require)):
            row = arms.get(name)
            where = f"{family}: rider {name}"
            if not isinstance(row, Mapping):
                problems.append(f"{where}: not run")
                continue
            documents = int(row["documents"])
            if documents < RIDER_MIN_DOCUMENTS:
                problems.append(
                    f"{where}: {documents} documents, fewer than "
                    f"{RIDER_MIN_DOCUMENTS}"
                )
            if int(row["mismatches"]) != 0:
                problems.append(f"{where}: {row['mismatches']} mismatch(es)")
            if name == "special_tokens":
                if int(row["vacuous"]) != 0:
                    problems.append(
                        f"{where}: {row['vacuous']} document(s) of an identity or "
                        "prefix face were answered by the reference"
                    )
                judged = int(row["classes"].get("identity", 0)) + int(
                    row["classes"].get("prefix", 0)
                )
                executed = row["executed_by"]
                accelerated = int(executed.get("fast_cpu", 0)) + int(
                    executed.get("gpu", 0)
                )
                if judged and accelerated < SPECIAL_TOKENS_ENGINE_SHARE * judged:
                    problems.append(
                        f"{where}: {accelerated} of {judged} identity or prefix "
                        "document(s) had their core stream made by an accelerated "
                        f"engine, below {SPECIAL_TOKENS_ENGINE_SHARE:.0%} "
                        f"({dict(executed)})"
                    )
                for key in ("served_by", "classes", "executed_by"):
                    counts = row[key]
                    if int(counts.get("not_reported", 0)) != 0:
                        problems.append(
                            f"{where}: {counts['not_reported']} document(s) with "
                            f"{key} not reported"
                        )
                    if sum(int(value) for value in counts.values()) != documents:
                        problems.append(f"{where}: {key} does not add up")
            else:
                paths = row["paths"]
                if int(paths.get("not_reported", 0)) != 0:
                    problems.append(
                        f"{where}: {paths['not_reported']} document(s) whose path "
                        "was not reported"
                    )
                if sum(int(value) for value in paths.values()) != documents:
                    problems.append(f"{where}: paths do not add up")
                problems += _rider_literal_problems(where, row, documents)
    return problems


def _rider_literal_problems(
    where: str, row: Mapping[str, Any], documents: int
) -> list[str]:
    """The reference-path bounds of a rider document's literal arm."""
    problems: list[str] = []
    paths = row["paths"]
    segment_bytes = row["bytes"]
    if int(paths.get("hf_added_token", 0)) != 0:
        problems.append(
            f"{where}: {paths['hf_added_token']} document(s) went whole to the "
            "reference for a literal"
        )
    reference = {
        key: int(value)
        for key, value in paths.items()
        if key.startswith("hf_") and key != "hf_added_token" and int(value)
    }
    if sum(reference.values()) > int(documents * GUARD_DOCUMENT_SHARE):
        problems.append(
            f"{where}: {sum(reference.values())} document(s) on reference paths "
            f"{reference}, above {GUARD_DOCUMENT_SHARE:.0%} of the documents"
        )
    total = sum(int(value) for value in segment_bytes.values())
    if int(segment_bytes["hf"]) > GUARD_BYTE_SHARE * total:
        problems.append(
            f"{where}: the reference encoded {segment_bytes['hf']} of {total} "
            f"reported bytes, above {GUARD_BYTE_SHARE:.0%}"
        )
    segmented = int(paths.get("gigatoken_segmented", 0)) + int(
        paths.get("gpu_segmented", 0)
    )
    if segmented < 1 or int(segment_bytes["fast_cpu"]) + int(segment_bytes["gpu"]) < 1:
        problems.append(
            f"{where}: no document was segmented on an accelerated engine"
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
    *,
    expected_literal_class: str | None = None,
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
        if "special_tokens" in required:
            problems += _surface_problems(family, row)
        problems += column_problems(
            row,
            arm=arm,
            added_frontend_fingerprint=fingerprints.get(family),
            expected_literal_class=expected_literal_class,
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
    parity.add_argument(
        "--literal-class",
        choices=("served", "unserved", "any"),
        default=EXPECTED_LITERAL_CLASS,
        help=(
            "the literal class every literal_segments column must read "
            f"(default {EXPECTED_LITERAL_CLASS}, the class of every certified "
            "family; any: take the column's own value)"
        ),
    )
    riders = commands.add_parser(
        "riders", help="check a chunk-cut campaign's rider document"
    )
    riders.add_argument("document", type=Path)
    riders.add_argument(
        "--require",
        action="append",
        default=[],
        choices=COLUMNS,
        help="a rider arm the campaign must have run (repeatable)",
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "riders":
        rider_problems = riders_problems(
            load_json(arguments.document), arguments.require
        )
        for problem in rider_problems:
            print(f"error: {problem}", file=sys.stderr)
        if rider_problems:
            return 1
        print(f"{arguments.document}: rider document passed")
        return 0
    reading = load_json(arguments.reading)
    if not isinstance(reading, dict):
        raise GenerationError(f"{arguments.reading}: not a JSON object")
    registry = load_json(arguments.registry)
    problems, notes = parity_problems(
        reading,
        arguments.require,
        registry,
        expected_literal_class=(
            None if arguments.literal_class == "any" else arguments.literal_class
        ),
    )
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
