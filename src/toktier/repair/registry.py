"""Digest-checked data for the certified Gigatoken repair route."""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..errors import RegistryInvalid

CONFIG_ID = "toktier-fast-repair-v1"
_TABLE_DIR = Path(__file__).with_name("tables")
_REGISTRY_PATH = _TABLE_DIR / "fast_repair_families.v1.json"


@dataclass(frozen=True)
class RepairFamily:
    """One artifact-specific set of proven repair parameters."""

    family: str
    artifact_sha256: str
    margin: int
    effective_l_max: int
    has_normalizer: bool
    source_table_sha256: str
    window_chars: int = 512
    max_retries: int = 5
    min_match_tokens: int = 2


def _failure(message: str, **details: object) -> RegistryInvalid:
    return RegistryInvalid(
        message,
        details={"path": str(_REGISTRY_PATH), **details},
    )


@lru_cache(maxsize=1)
def _document() -> dict[str, Any]:
    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _failure("the fast-repair registry cannot be read") from exc
    if not isinstance(raw, dict) or raw.get("schema") != (
        "toktier.fast_repair_families.v1"
    ):
        raise _failure("the fast-repair registry has an unknown schema")
    if raw.get("config_id") != CONFIG_ID:
        raise _failure(
            "the fast-repair configuration id drifted",
            observed=raw.get("config_id"),
            expected=CONFIG_ID,
        )
    return raw


@lru_cache(maxsize=1)
def pclass_table() -> bytes:
    """Return the frozen O/S/L/N/M Unicode-property table."""
    document = _document()
    metadata = document.get("pclass")
    if not isinstance(metadata, dict):
        raise _failure("the fast-repair registry has no pclass metadata")
    path = _TABLE_DIR / str(metadata.get("file", ""))
    try:
        compressed = path.read_bytes()
    except OSError as exc:
        raise _failure("the fast-repair pclass table is missing") from exc
    compressed_sha = hashlib.sha256(compressed).hexdigest()
    if compressed_sha != metadata.get("compressed_sha256"):
        raise _failure(
            "the fast-repair pclass archive digest does not match",
            observed=compressed_sha,
            expected=metadata.get("compressed_sha256"),
        )
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        raise _failure("the fast-repair pclass archive is corrupt") from exc
    raw_sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != 0x110000 or raw_sha != metadata.get("raw_sha256"):
        raise _failure(
            "the fast-repair pclass payload does not match",
            rows=len(raw),
            observed=raw_sha,
            expected=metadata.get("raw_sha256"),
        )
    return raw


@lru_cache(maxsize=1)
def families() -> tuple[RepairFamily, ...]:
    rows = _document().get("families")
    if not isinstance(rows, list):
        raise _failure("the fast-repair family list is missing")
    try:
        parsed = tuple(
            RepairFamily(
                family=str(row["family"]),
                artifact_sha256=str(row["artifact_sha256"]),
                margin=int(row["margin"]),
                effective_l_max=int(row["effective_l_max"]),
                has_normalizer=bool(row["has_normalizer"]),
                source_table_sha256=str(row["source_table_sha256"]),
            )
            for row in rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _failure("a fast-repair family row is malformed") from exc
    if len(parsed) != 11:
        raise _failure(
            "the certified repair roster must contain 11 unique artifacts",
            observed=len(parsed),
        )
    if len({item.artifact_sha256 for item in parsed}) != len(parsed):
        raise _failure("the fast-repair roster repeats an artifact digest")
    return parsed


@lru_cache(maxsize=1)
def examined_families() -> tuple[RepairFamily, ...]:
    """Repair parameters of the examined (not certified) artifacts.

    Read from the shipped examined-artifact table, digest-verified, and
    kept apart from :func:`families`: the certified roster stays exactly
    the eleven artifacts its readings cover, and its file stays inside the
    fast CPU engine's bound source set unchanged. Whether an examined
    artifact's route may run, and under which label, is the planner's
    decision (``routing/plan.py`` check 5), never this table's.
    """
    from ..routing.registry_load import load_examined_artifacts
    from ..routing.tables import EXAMINED_ARTIFACTS

    try:
        return tuple(
            RepairFamily(
                family=record.family,
                artifact_sha256=record.artifact_sha256,
                margin=int(str(record.repair["margin"])),
                effective_l_max=int(str(record.repair["effective_l_max"])),
                has_normalizer=record.repair["has_normalizer"] is True,
                source_table_sha256=str(record.repair["source_table_sha256"]),
            )
            for record in load_examined_artifacts(EXAMINED_ARTIFACTS)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _failure("an examined repair row is malformed") from exc


@lru_cache(maxsize=1)
def carried_families() -> tuple[RepairFamily, ...]:
    """Repair parameters held by certified registry records themselves.

    A record certified through a cross-artifact carry-over
    (``docs/contracts/evidence-carryover.md`` Section 3.1) is not in the
    repair roster; its fast CPU entry carries the parameters in a ``repair``
    member, equal to its anchor's roster row (condition C6, checked by the
    release check). Read from the digest-verified packaged registry, and
    kept apart from :func:`families`, which stays exactly the eleven
    artifacts the campaign judged.
    """
    from ..routing.registry_load import load_registry
    from ..routing.tables import SUPPORT_REGISTRY

    rows: list[RepairFamily] = []
    try:
        for record in load_registry(SUPPORT_REGISTRY).artifact_records():
            entry = record.backends.get("fast_cpu")
            if entry is None or entry.repair is None:
                continue
            rows.append(
                RepairFamily(
                    family=record.family,
                    artifact_sha256=record.artifact_sha256,
                    margin=int(str(entry.repair["margin"])),
                    effective_l_max=int(str(entry.repair["effective_l_max"])),
                    has_normalizer=entry.repair["has_normalizer"] is True,
                    source_table_sha256=str(entry.repair["source_table_sha256"]),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise _failure("a carried repair row is malformed") from exc
    return tuple(rows)


@lru_cache(maxsize=1)
def _sources() -> tuple[
    dict[tuple[str, str], RepairFamily], frozenset[str]
]:
    """Every (family, artifact) row, and the families named twice.

    A family or an artifact digest that two sources name (the roster, the
    carried records, the examined table) is an inconsistent installation:
    the parameters a route runs with must have one source. Such families
    are refused by :func:`family_spec` rather than resolved by order.
    """
    index: dict[tuple[str, str], RepairFamily] = {}
    family_sources: dict[str, set[str]] = {}
    digest_sources: dict[str, set[str]] = {}
    digest_families: dict[str, set[str]] = {}
    for source, rows in (
        ("roster", families()),
        ("carried", carried_families()),
        ("examined", examined_families()),
    ):
        for item in rows:
            family_sources.setdefault(item.family, set()).add(source)
            digest_sources.setdefault(item.artifact_sha256, set()).add(source)
            digest_families.setdefault(item.artifact_sha256, set()).add(item.family)
            index.setdefault((item.family, item.artifact_sha256), item)
    conflicts = {
        family for family, sources in family_sources.items() if len(sources) > 1
    }
    for digest, sources in digest_sources.items():
        if len(sources) > 1:
            conflicts |= digest_families[digest]
    return index, frozenset(conflicts)


def family_spec(family: str, artifact_sha256: str) -> RepairFamily | None:
    """Exact family-and-artifact repair parameters, or ``None``.

    The certified roster is consulted first, then the certified records
    carried from an anchor (:func:`carried_families`), then the examined
    artifacts. A row from the last answers "with which parameters would
    the engine run this artifact", not "is it certified": the plan refuses
    an examined artifact under ``CERTIFIED`` and labels it
    ``supported_untested`` under ``SUPPORTED``. A family or digest that two
    of the sources name raises ``RegistryInvalid``.
    """
    index, conflicts = _sources()
    if family in conflicts:
        raise _failure(
            "a repair family is named by more than one parameter source",
            family=family,
        )
    return index.get((family, artifact_sha256))
