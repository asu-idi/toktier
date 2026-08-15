"""Schema and chain tests for add-only evidence carry-over records."""

from __future__ import annotations

import copy
import json
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import verify_carryover  # noqa: E402

# The certified identity triple recorded by the tree under test.  A carry-over
# record anchors to the live campaign only when its source identities appear in
# the evidence it points at, so these literals track the registry and are
# refreshed by each recertification.
# The identities the shipped runtime-build row records. A carryover record
# anchors to a campaign only when its source identities are exactly the ones
# the registry carries, so these move with every recertification wave.
CERTIFIED = {
    "fast_cpu": "a10687cea7e9b187a4a5b7146e350ee9eca7ff479015d61c5a4c7c35fb2205e2",
    "native_host": "1158dbf5586056cca76d6f74ba0acbbcd188a448d094ae711027700cd3fb7dc9",
    "rust_api": "8d4f3fcc2ca04602ac14e107f7ebdb623b9aa19aaddcd68588d09ab611340016",
}


def _identity(character: str) -> dict[str, str]:
    return {key: character * 64 for key in verify_carryover.IDENTITY_KEYS}


def _record(
    source: dict[str, str] | None = None,
    target: dict[str, str] | None = None,
) -> dict[str, Any]:
    tree = "/tmp/toktier_equiv/tree"
    cargo_home = "/home/builder/.cargo"
    target_dir = "/tmp/toktier_equiv/target"
    rustflags = (
        f"--remap-path-prefix={tree}=/toktier "
        f"--remap-path-prefix={cargo_home}=/cargo"
    )
    return {
        "record": "evidence_carryover.v1",
        "mechanism": "artifact_equivalence",
        "from_source_identity": source or CERTIFIED,
        "to_source_identity": target or _identity("a"),
        "witness": {
            "sentinel_artifacts": [
                {
                    "artifact": "_native.abi3.so",
                    "sha256_both": "b" * 64,
                    "bytes": 100,
                    "byte_equal": True,
                },
                {
                    "artifact": "libtoktier.rlib",
                    "sha256_both": "c" * 64,
                    "bytes": 200,
                    "byte_equal": True,
                },
            ],
            "recipe": {
                "tree_path": tree,
                "cargo_target_dir": target_dir,
                "rustflags": rustflags,
                "cargo_home_roots": [cargo_home],
                "locked": True,
                "fresh_target_for_each_tree": True,
                "sequential_same_path_builds": True,
                "commands": [
                    {
                        "purpose": "sentinel_native_extension",
                        "cwd": tree,
                        "argv": [
                            "nice",
                            "-n",
                            "5",
                            "maturin",
                            "build",
                            "--locked",
                        ],
                    },
                    {
                        "purpose": "sentinel_whole_rust_api_rlib",
                        "cwd": tree,
                        "argv": [
                            "nice",
                            "-n",
                            "5",
                            "cargo",
                            "build",
                            "--locked",
                            "-p",
                            "toktier",
                        ],
                    },
                ],
                "toolchain": {
                    "rust_toolchain_sha256": "d" * 64,
                    "rustc": "rustc 1.93.1",
                    "cargo": "cargo 1.93.1",
                    "maturin": "maturin 1.14.1",
                },
                "ambient_environment": {"RUSTFLAGS": None},
                "effective_environment": {
                    "CARGO_TARGET_DIR": target_dir,
                    "RUSTFLAGS": rustflags,
                    "TOKTIER_IDENTITY_SENTINEL": "1",
                },
                "cargo_configs": [],
                "host_fingerprint": {
                    "description": "build environment fingerprint without a host name",
                    "sha256": "e" * 64,
                    "facts": {"system": "Linux"},
                },
                "same_host_only": True,
            },
            "applicability": {
                "diff_files": ["crates/core/src/lib.rs"],
                "cargo_lock_unchanged": True,
                "protected_files_unchanged": True,
            },
        },
        "carried_evidence": [
            "tables/support_registry.json#/runtime_builds/0"
        ],
    }


def _write(directory: Path, name: str, document: dict[str, Any]) -> None:
    path = directory / "v0.2" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_artifact_record_schema_and_internal_consistency() -> None:
    schema = json.loads(verify_carryover.DEFAULT_SCHEMA.read_text(encoding="utf-8"))

    problems, campaign_anchor = verify_carryover.record_problems(
        _record(), Path("record.json"), schema, ROOT
    )

    assert problems == []
    assert campaign_anchor


def test_schema_rejects_a_missing_record_discriminator() -> None:
    schema = json.loads(verify_carryover.DEFAULT_SCHEMA.read_text(encoding="utf-8"))
    document = _record()
    del document["record"]

    problems = verify_carryover.schema_problems(document, schema)

    assert any("'record' is a required property" in problem for problem in problems)


def test_code_identity_v2_allows_the_enumerated_lock_change() -> None:
    schema = json.loads(verify_carryover.DEFAULT_SCHEMA.read_text(encoding="utf-8"))
    document = _record()
    document["mechanism"] = "code_identity_v2"
    witness = document["witness"]
    del witness["sentinel_artifacts"]
    witness["applicability"]["cargo_lock_unchanged"] = False
    witness["code_identity_v2"] = {
        "value_both": _identity("f"),
        "normalization_diff": "--- a/Cargo.toml\n+++ b/Cargo.toml\n",
    }
    witness["recipe"]["commands"] = [
        {
            "purpose": "version_normalized_source_identity",
            "cwd": witness["recipe"]["tree_path"],
            "argv": [
                "python",
                "tools/compute_identity_v2.py",
                "--show-diff",
            ],
        }
    ]

    problems, campaign_anchor = verify_carryover.record_problems(
        document, Path("record.json"), schema, ROOT
    )

    assert problems == []
    assert campaign_anchor


def test_registry_pointer_must_select_an_original_record() -> None:
    schema = json.loads(verify_carryover.DEFAULT_SCHEMA.read_text(encoding="utf-8"))
    document = _record()
    document["carried_evidence"] = ["tables/support_registry.json"]

    problems, campaign_anchor = verify_carryover.record_problems(
        document, Path("record.json"), schema, ROOT
    )

    assert any(
        "registry pointer must select one original record" in item
        for item in problems
    )
    assert not campaign_anchor


def test_four_consecutive_carryovers_are_rejected(tmp_path: Path) -> None:
    records = tmp_path / "carryover"
    nodes = [CERTIFIED, *(_identity(character) for character in "1234")]
    for index, (source, target) in enumerate(pairwise(nodes), 1):
        document = copy.deepcopy(_record(source, target))
        _write(records, f"{index:02d}.json", document)

    problems, count = verify_carryover.check_records(records)

    assert count == 4
    assert any("consecutive carry-over depth 4 exceeds 3" in item for item in problems)


def test_each_represented_minor_needs_a_campaign_anchor(tmp_path: Path) -> None:
    records = tmp_path / "carryover"
    first = _record(CERTIFIED, _identity("1"))
    _write(records, "01.json", first)
    second = _record(_identity("1"), _identity("2"))
    path = records / "v0.3/01.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(second), encoding="utf-8")

    problems, count = verify_carryover.check_records(records)

    assert count == 2
    assert any("v0.3: no real-campaign anchor" in item for item in problems)
