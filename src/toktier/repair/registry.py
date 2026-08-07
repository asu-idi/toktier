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


def family_spec(family: str, artifact_sha256: str) -> RepairFamily | None:
    """Exact family-and-artifact match, or ``None`` when not certified."""
    return next(
        (
            item
            for item in families()
            if item.family == family and item.artifact_sha256 == artifact_sha256
        ),
        None,
    )
