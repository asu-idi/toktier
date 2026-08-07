"""Accounting and provenance gates for the optional CPU repair engines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _json(relative: str) -> dict[str, object]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _sha256_without_terminal_newline(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes().rstrip(b"\n")).hexdigest()


def test_fast_cpu_registry_counts_unique_artifacts_without_inflation() -> None:
    binding = _json("tools/fast_cpu_binding.json")
    registry = _json("tables/support_registry.json")
    coverage = binding["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["unique_tokenizer_artifacts"] == 11
    assert coverage["model_families"] == 12

    artifacts = registry["artifacts"]
    assert isinstance(artifacts, list)
    states = {
        row["family"]: row["backends"]["fast_cpu"]["status"]
        for row in artifacts
        if isinstance(row, dict) and "fast_cpu" in row["backends"]
    }
    assert sum(status == "certified" for status in states.values()) == 11
    assert {
        family for family, status in states.items() if status == "unsupported"
    } == {"hy3", "laguna_s_2_1", "ling_3_0_flash"}

    inherited = coverage["exact_artifact_inheritance"]
    assert isinstance(inherited, dict)
    assert inherited["inherits_from"] == "qwen3_8b"
    assert inherited["artifact_sha256"] == next(
        row["artifact_sha256"]
        for row in artifacts
        if isinstance(row, dict) and row["family"] == "qwen3_8b"
    )
    repositories = inherited["repositories"]
    assert isinstance(repositories, list)
    assert len(repositories) == 3


def test_corrected_gigatoken_patch_and_notices_are_digest_bound() -> None:
    binding = _json("tools/fast_cpu_binding.json")
    manifest = _json("src/toktier/_vendor/gigatoken_build.json")
    assert binding["engine_distribution"] == "toktier"
    assert binding["engine_delivery"] == "vendored"
    assert binding["engine_module"] == "toktier._vendor.gigatoken_rs"
    assert manifest["module"] == binding["engine_module"]
    assert manifest["native_sha256"] == binding["binary_digest"]
    assert _sha256(str(binding["vendored_native_path"])) == binding["binary_digest"]
    assert _sha256(str(binding["vendored_sbom_path"])) == binding[
        "vendored_sbom_sha256"
    ]
    assert _sha256(str(binding["vendored_license_bundle_path"])) == binding[
        "vendored_license_bundle_sha256"
    ]
    assert binding["patch_sha256"] == _sha256(
        "packaging/fast_cpu/gigatoken-toktier-pinned-1.patch"
    )
    assert (ROOT / "packaging/fast_cpu/LICENSE-gigatoken").is_file()
    notice = (ROOT / "packaging/fast_cpu/NOTICE-gigatoken-pinned").read_text(
        encoding="utf-8"
    )
    assert str(binding["patch_sha256"]) in notice
    recipe = (ROOT / "packaging/fast_cpu/build_pinned.sh").read_text(
        encoding="utf-8"
    )
    patch = (
        ROOT / "packaging/fast_cpu/gigatoken-toktier-pinned-1.patch"
    ).read_text(encoding="utf-8")
    assert str(binding["patch_sha256"]) in recipe
    assert "NOTICE-TOKTIER.md" in recipe
    assert "diff --git a/NOTICE-TOKTIER.md b/NOTICE-TOKTIER.md" in patch

    assert binding["engine_wheel_sha256"] == (
        "9fbfe0fda617763ec65dab98de15c28c94223f515ffd71a4a296716c60f220e7"
    )
    equivalence = binding["native_equivalence"]
    assert isinstance(equivalence, dict)
    assert equivalence["campaign_binary_digest"] == binding["binary_digest"]
    assert equivalence["release_binary_digest"] == binding["binary_digest"]


def test_focused_public_api_parity_covers_every_certified_artifact() -> None:
    binding = _json("tools/fast_cpu_binding.json")
    reading = _json("readings/fast_cpu_focused_parity.json")
    assert reading["schema"] == "toktier.fast_cpu.focused_parity.v2"
    release = reading["release"]
    assert isinstance(release, dict)
    assert release["distribution"] == "toktier"
    assert release["external_gigatoken_distribution_present"] is False
    engine = reading["engine"]
    assert isinstance(engine, dict)
    assert engine["delivery"] == "vendored"
    assert engine["module"] == "toktier._vendor.gigatoken_rs"
    assert engine["native_sha256"] == binding["binary_digest"]
    assert reading["unique_artifacts"] == 11
    assert reading["model_families"] == 12
    assert reading["all_ids_equal_hf"] is True
    assert reading["all_executed_gigatoken_repair"] is True
    rows = reading["rows"]
    assert isinstance(rows, list)
    by_family = {
        str(row["family"]): row for row in rows if isinstance(row, dict)
    }
    loadable = binding["loadable_families"]
    assert isinstance(loadable, list)
    assert set(by_family) == set(loadable)
    assert all(row["all_turns_equal_hf"] is True for row in by_family.values())
    assert all(
        int(row["path_counts"].get("gigatoken_repair", 0)) >= 1
        for row in by_family.values()
    )


def test_fastokens_v031_license_materials_are_exact_upstream_files() -> None:
    assert _sha256_without_terminal_newline("packaging/fastokens/LICENSE") == (
        "0cec06e0e55fbc3dc5cee4fca9b607f66cb8f4e4dbcf3b3c013594dd156732e9"
    )
    assert _sha256_without_terminal_newline("packaging/fastokens/NOTICES.txt") == (
        "22ad47758c67a1ed81791611b6095d59b93d1bfa8cad6476cdbf2e7524191cf0"
    )
