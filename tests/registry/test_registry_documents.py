# Exercise the canonical form, the root digest rule and the check mode.
"""Tests for the generated registry and evidence documents.

These tests use small hand-built documents, so they run anywhere: the rules
under test are the canonical form, the digest construction and the check
mode, not the contents of the shipped tables.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import registry_common
from registry_common import (
    EVIDENCE_DOMAIN_TAG,
    PLACEHOLDER_SHA256,
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    canonical_json,
    check_regenerated,
    root_digest,
    serialise_document,
    verify_file,
    with_root_digest,
    write_document,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_SCHEMA = json.loads(
    (ROOT / "schemas" / "support_registry.schema.json").read_text(encoding="utf-8")
)
EVIDENCE_SCHEMA = json.loads(
    (ROOT / "schemas" / "evidence_manifest.schema.json").read_text(encoding="utf-8")
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64

pytest.importorskip("jsonschema", reason="schema validation needs jsonschema")


def minimal_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_by": {
            "tool": "tools/generate_registry.py",
            "tool_version": "1.0.0",
            "source_commit": "0123abc",
            "generated_at": "2026-08-05T22:00:00Z",
        },
        "root_digest": "",
        "oracles": [
            {
                "oracle_id": "tokenizers",
                "package": "tokenizers",
                "certified_versions": ["0.22.2"],
                "semantic_id": "tokenizers.0.22.2",
            }
        ],
        "pipelines": [
            {"pipeline_id": "pipeline.aaaa", "pipeline_fingerprint": DIGEST_B}
        ],
        "added_frontends": [
            {
                "added_frontend_id": "added-frontend.aaaa",
                "added_frontend_fingerprint": DIGEST_C,
            }
        ],
        "artifacts": [
            {
                "artifact_sha256": DIGEST_A,
                "family": "example_family",
                "pipeline_id": "pipeline.aaaa",
                "added_frontend_id": "added-frontend.aaaa",
                "oracle_id": "tokenizers",
                "suite_version": "suite-v1",
                "evidence_id": "ev-example-v1",
                "readings": {"docs": 10, "bytes": 100, "mismatches": 0},
                "backends": {
                    "gpu": {
                        "status": "certified_source",
                        "source_digest": PLACEHOLDER_SHA256,
                        "build_flags": ["-O3"],
                        "toolchain": "cuda 13",
                        "class_table_digest": PLACEHOLDER_SHA256,
                        "devices": ["sm_89"],
                    }
                },
            }
        ],
        "compositions": [
            {
                "pipeline_id": "pipeline.aaaa",
                "added_frontend_id": "added-frontend.aaaa",
                "evidence_id": "ev-example-v1",
            }
        ],
    }


def minimal_evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "ev-example-v1",
        "generated_by": {
            "tool": "tools/generate_evidence.py",
            "tool_version": "1.0.0",
            "source_commit": "0123abc",
            "generated_at": "2026-08-05T22:00:00Z",
        },
        "root_digest": "",
        "run": {
            "run_id": "example-run",
            "date": "2026-08-05",
        },
        "oracle": {"package": "tokenizers", "version": "0.22.2"},
        "artifacts": [{"artifact_sha256": DIGEST_A, "family": "example_family"}],
        "corpora": [{"corpus_id": "example/corpus@1", "docs": 10, "bytes": 100}],
        "totals": {"docs": 10, "bytes": 100, "mismatches": 0, "routed": 0},
        "environment": {"os": "Linux", "python": "3.12.3"},
    }


# --- canonical form -------------------------------------------------------


def test_canonical_json_sorts_keys_and_strips_whitespace() -> None:
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert canonical_json([1, {"z": None, "y": True}]) == b'[1,{"y":true,"z":null}]'


def test_canonical_json_escapes_the_required_characters() -> None:
    assert canonical_json('a"b\\c\nd\x01') == b'"a\\"b\\\\c\\nd\\u0001"'


def test_canonical_json_keeps_non_ascii_literal() -> None:
    assert canonical_json("\u00e9") == '"\u00e9"'.encode()


def test_canonical_json_canonicalizes_fractional_numbers() -> None:
    assert canonical_json(0.0) == b"0"
    assert canonical_json(0.5) == b"0.5"
    assert canonical_json(1e-7) == b"1e-7"
    with pytest.raises(GenerationError):
        canonical_json(float("inf"))


def test_canonical_json_refuses_unknown_kinds() -> None:
    with pytest.raises(GenerationError):
        canonical_json({1, 2})


# --- root digest ----------------------------------------------------------


def test_root_digest_ignores_the_recorded_digest_and_key_order() -> None:
    document = minimal_registry()
    first = root_digest(document, REGISTRY_DOMAIN_TAG)
    document["root_digest"] = first
    assert root_digest(document, REGISTRY_DOMAIN_TAG) == first
    reordered = dict(reversed(list(document.items())))
    assert root_digest(reordered, REGISTRY_DOMAIN_TAG) == first


def test_root_digest_is_domain_separated() -> None:
    document = minimal_registry()
    assert root_digest(document, REGISTRY_DOMAIN_TAG) != root_digest(
        document, EVIDENCE_DOMAIN_TAG
    )


def test_root_digest_follows_any_change() -> None:
    document = minimal_registry()
    before = root_digest(document, REGISTRY_DOMAIN_TAG)
    document["artifacts"][0]["readings"]["mismatches"] = 1
    assert root_digest(document, REGISTRY_DOMAIN_TAG) != before


def test_root_digest_matches_the_documented_construction() -> None:
    import hashlib

    document = minimal_evidence()
    stripped = {key: value for key, value in document.items() if key != "root_digest"}
    expected = hashlib.sha256(
        EVIDENCE_DOMAIN_TAG + canonical_json(stripped)
    ).hexdigest()
    assert root_digest(document, EVIDENCE_DOMAIN_TAG) == f"sha256:{expected}"


# --- schema, positive and negative ---------------------------------------


def test_valid_documents_pass_their_schema() -> None:
    registry = with_root_digest(minimal_registry(), REGISTRY_DOMAIN_TAG)
    evidence = with_root_digest(minimal_evidence(), EVIDENCE_DOMAIN_TAG)
    assert registry_common.schema_violations(registry, REGISTRY_SCHEMA) == []
    assert registry_common.schema_violations(evidence, EVIDENCE_SCHEMA) == []


def test_v2_identity_columns_are_additive_under_schema_version_one() -> None:
    document = minimal_registry()
    backend = document["artifacts"][0]["backends"]["gpu"]
    backend.update(
        {
            "source_digest_v2": DIGEST_A,
            "host_source_digest_v2": DIGEST_B,
            "direct_host_source_digest_v2": DIGEST_C,
        }
    )
    document["runtime_builds"] = [
        {
            "runtime": "rust_api",
            "source_digest": DIGEST_A,
            "source_digest_v2": DIGEST_B,
            "fast_cpu_source_digest": DIGEST_A,
            "fast_cpu_source_digest_v2": DIGEST_B,
            "native_host_source_digest": DIGEST_A,
            "native_host_source_digest_v2": DIGEST_B,
            "build_flags": ["profile=release"],
            "toolchain": "rustc test",
            "evidence_id": "ev-example-v1",
        }
    ]
    registry = with_root_digest(document, REGISTRY_DOMAIN_TAG)

    assert registry["schema_version"] == 1
    assert registry_common.schema_violations(registry, REGISTRY_SCHEMA) == []


def test_certified_source_without_its_bindings_is_rejected() -> None:
    document = with_root_digest(minimal_registry(), REGISTRY_DOMAIN_TAG)
    del document["artifacts"][0]["backends"]["gpu"]["source_digest"]
    assert registry_common.schema_violations(document, REGISTRY_SCHEMA)


def test_certified_status_needs_a_binary_digest() -> None:
    document = with_root_digest(minimal_registry(), REGISTRY_DOMAIN_TAG)
    document["artifacts"][0]["backends"]["gpu"] = {"status": "certified"}
    assert registry_common.schema_violations(document, REGISTRY_SCHEMA)


def test_malformed_digest_is_rejected() -> None:
    document = with_root_digest(minimal_registry(), REGISTRY_DOMAIN_TAG)
    document["artifacts"][0]["artifact_sha256"] = "not-a-digest"
    assert registry_common.schema_violations(document, REGISTRY_SCHEMA)


def test_unknown_member_is_rejected() -> None:
    document = with_root_digest(minimal_registry(), REGISTRY_DOMAIN_TAG)
    document["extra_member"] = True
    assert registry_common.schema_violations(document, REGISTRY_SCHEMA)


def test_negative_mismatch_count_is_rejected() -> None:
    document = with_root_digest(minimal_evidence(), EVIDENCE_DOMAIN_TAG)
    document["totals"]["mismatches"] = -1
    assert registry_common.schema_violations(document, EVIDENCE_SCHEMA)


def test_evidence_without_a_corpus_is_rejected() -> None:
    document = with_root_digest(minimal_evidence(), EVIDENCE_DOMAIN_TAG)
    document["corpora"] = []
    assert registry_common.schema_violations(document, EVIDENCE_SCHEMA)


# --- check mode -----------------------------------------------------------


def test_written_document_passes_verification(tmp_path: Path) -> None:
    path = tmp_path / "support_registry.json"
    write_document(path, minimal_registry(), REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG)
    assert verify_file(path, REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG) == []


def test_write_refuses_a_document_that_fails_its_schema(tmp_path: Path) -> None:
    document = minimal_registry()
    document["artifacts"][0]["readings"]["docs"] = -1
    with pytest.raises(GenerationError):
        write_document(
            tmp_path / "out.json", document, REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG
        )
    assert not (tmp_path / "out.json").exists()


def test_verification_catches_an_edited_value(tmp_path: Path) -> None:
    path = tmp_path / "support_registry.json"
    write_document(path, minimal_registry(), REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["artifacts"][0]["readings"]["mismatches"] = 0xC0FFEE
    path.write_bytes(serialise_document(document))
    problems = verify_file(path, REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG)
    assert any("root digest mismatch" in problem for problem in problems)


def test_verification_catches_a_reflowed_file(tmp_path: Path) -> None:
    path = tmp_path / "support_registry.json"
    write_document(path, minimal_registry(), REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG)
    document = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(document, indent=4), encoding="utf-8")
    problems = verify_file(path, REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG)
    assert any("deterministic serialised form" in problem for problem in problems)


def test_verification_reports_a_missing_file(tmp_path: Path) -> None:
    problems = verify_file(
        tmp_path / "absent.json", REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG
    )
    assert problems and "missing" in problems[0]


def test_regeneration_check_ignores_the_generation_stamp(tmp_path: Path) -> None:
    path = tmp_path / "support_registry.json"
    write_document(path, minimal_registry(), REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG)
    regenerated = minimal_registry()
    regenerated["generated_by"]["generated_at"] = "2027-01-01T00:00:00Z"
    regenerated["generated_by"]["source_commit"] = "beefbee"
    assert (
        check_regenerated(path, regenerated, REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG) == []
    )


def test_regeneration_check_reports_changed_readings(tmp_path: Path) -> None:
    path = tmp_path / "support_registry.json"
    write_document(path, minimal_registry(), REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG)
    regenerated = minimal_registry()
    regenerated["artifacts"][0]["readings"]["docs"] = 11
    problems = check_regenerated(
        path, regenerated, REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG
    )
    assert any("differs from the file on disk" in problem for problem in problems)


def test_serialisation_is_ascii_only_and_newline_terminated() -> None:
    document = minimal_registry()
    document["artifacts"][0]["family"] = "example_family"
    document["pipelines"][0]["description"] = "caf\u00e9"
    raw = serialise_document(with_root_digest(document, REGISTRY_DOMAIN_TAG))
    assert raw.endswith(b"\n")
    assert all(byte < 128 for byte in raw)


# --- the shipped files ----------------------------------------------------


@pytest.mark.parametrize(
    ("relative", "schema", "tag"),
    [
        ("tables/support_registry.json", REGISTRY_SCHEMA, REGISTRY_DOMAIN_TAG),
        ("evidence/evidence_manifest.json", EVIDENCE_SCHEMA, EVIDENCE_DOMAIN_TAG),
        (
            "evidence/evidence_manifest_added_families.json",
            EVIDENCE_SCHEMA,
            EVIDENCE_DOMAIN_TAG,
        ),
        (
            "evidence/evidence_manifest_kimi_band.json",
            EVIDENCE_SCHEMA,
            EVIDENCE_DOMAIN_TAG,
        ),
    ],
)
def test_shipped_documents_verify(
    relative: str, schema: dict[str, Any], tag: bytes
) -> None:
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} has not been generated in this tree")
    assert verify_file(path, schema, tag) == []


def test_every_gpu_parity_reading_says_what_it_covers() -> None:
    """Scale is in the reading, not only in the prose around it.

    Both GPU architectures report zero mismatches and both are certified
    rows. They differ in how much each wave re-takes: `sm_120` runs the
    full per-family campaign and `sm_89` a bounded spot check that rests
    on the cross-architecture record already on file. Before the field
    existed the only thing separating them was an unlabelled document
    count, which says nothing on its own about the protocol a campaign
    followed.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    expected = {"sm_120": "full", "sm_89": "spot"}
    per_family: dict[str, int] = {}
    for architecture, scale in expected.items():
        suffix = architecture.replace("sm_", "sm")
        path = root / f"readings/gpu_native_frontend_{suffix}_parity.json"
        reading = json.loads(path.read_text(encoding="utf-8"))
        assert reading["architecture"] == architecture
        assert reading["scale"] == scale, path
        rows = reading["rows"]
        counts = {int(row["documents"]) for row in rows}
        assert len(counts) == 1, f"{path}: rows disagree on documents per family"
        per_family[scale] = counts.pop()

    # The label and the numbers have to agree: a spot check is smaller.
    assert per_family["spot"] < per_family["full"]
