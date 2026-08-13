"""Integrity and coverage checks for the packaged sibling registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from toktier.artifacts import ArtifactManifest
from toktier.artifacts.sibling_aliases import (
    load_sibling_aliases,
    shipped_sibling_aliases,
)
from toktier.artifacts.tables import SIBLING_ALIASES
from toktier.errors import RegistryInvalid


def test_shipped_registry_closes_to_the_documented_coverage() -> None:
    registry = shipped_sibling_aliases()
    assert len(registry.records) == 211
    equivalent = [
        record
        for record in registry.records
        if record.basis.startswith("equivalent_")
    ]
    assert len(equivalent) == 48
    assert sum(record.canonical_packaged for record in equivalent) == 46
    assert sum(record.canonical_packaged for record in registry.records) == 204
    assert registry.root_digest.startswith("sha256:")


def test_packaged_registry_passes_its_generator_check() -> None:
    tools = str(Path(__file__).resolve().parents[1] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import generate_sibling_aliases

    assert generate_sibling_aliases.main(["--check"]) == 0


def test_shipped_registry_carries_full_content_identities() -> None:
    registry = shipped_sibling_aliases()
    record = registry.for_repo("Qwen/Qwen3-235B-A22B-Thinking-2507")
    assert record is not None
    assert record.revision == "6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5"
    assert record.source_sha256 == (
        "19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d"
    )
    assert record.canonical_family == "qwen3_8b"
    assert record.basis == "equivalent_serialisation"


def test_tampering_is_rejected_by_the_root_digest(tmp_path: Path) -> None:
    payload = SIBLING_ALIASES.read_text(encoding="utf-8").replace(
        '"source_size": 7032399', '"source_size": 7032400', 1
    )
    path = tmp_path / "aliases.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RegistryInvalid, match="root digest mismatch"):
        load_sibling_aliases(path)


def test_duplicate_json_members_are_rejected(tmp_path: Path) -> None:
    payload = SIBLING_ALIASES.read_text(encoding="utf-8").replace(
        '"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,', 1
    )
    path = tmp_path / "aliases.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RegistryInvalid, match="duplicate JSON member"):
        load_sibling_aliases(path)


def test_packaged_flags_are_bound_to_the_artifact_manifest() -> None:
    registry = shipped_sibling_aliases()
    with pytest.raises(RegistryInvalid, match="manifest presence"):
        registry.validate_manifest(ArtifactManifest(), path="empty-manifest")
