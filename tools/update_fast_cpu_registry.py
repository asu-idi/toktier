#!/usr/bin/env python3
"""Apply the integrated corrected-Gigatoken binding to the support registry.

The active CPU certificate binds the source/build identity emitted by the
single ``toktier._native`` extension.  Historical wheel and binary digests are
retained in ``fast_cpu_binding.json`` as campaign lineage, but they never
authorize the executing backend.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import zlib
from pathlib import Path
from typing import Any

from check_reading_forms import column_problems as reading_column_problems
from check_reading_forms import registry_fingerprints
from compute_identity_v2 import source_digest as source_digest_v2
from fast_cpu_source_identity import source_digest
from registry_common import (
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    load_json,
    schema_violations,
    serialise_document,
    sha256_of_file,
    with_root_digest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "tables" / "support_registry.json"
PACKAGED_REGISTRY = (
    REPOSITORY_ROOT
    / "src"
    / "toktier"
    / "routing"
    / "tables"
    / "support_registry.v1.json"
)
BINDING_PATH = REPOSITORY_ROOT / "tools" / "fast_cpu_binding.json"
#: Root the bound readings (``readings/...``) are read under.
READINGS_ROOT = REPOSITORY_ROOT
REPAIR_TABLE_PATH = (
    REPOSITORY_ROOT
    / "src"
    / "toktier"
    / "repair"
    / "tables"
    / "fast_repair_families.v1.json"
)
REPAIR_PCLASS_PATH = REPAIR_TABLE_PATH.with_name("repair_pclass.v1.zlib")
PATCH_PATH = (
    REPOSITORY_ROOT
    / "packaging"
    / "fast_cpu"
    / "gigatoken-toktier-pinned-1.patch"
)
LICENSE_PATH = REPOSITORY_ROOT / "packaging" / "fast_cpu" / "LICENSE-gigatoken"
NOTICE_PATH = (
    REPOSITORY_ROOT / "packaging" / "fast_cpu" / "NOTICE-gigatoken-pinned"
)
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "support_registry.schema.json"


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be a JSON object")
    return value


def _require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GenerationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_identity(binding: dict[str, Any]) -> None:
    if binding.get("version") != "fast-cpu-binding-v2":
        raise GenerationError("unknown fast CPU binding version")
    if binding.get("backend") != "fast_cpu":
        raise GenerationError("fast CPU binding names the wrong backend")
    if binding.get("repair_config_id") != "toktier-fast-repair-v1":
        raise GenerationError("fast CPU binding names the wrong repair config")
    if (
        binding.get("engine_distribution") != "toktier"
        or binding.get("engine_module") != "toktier._native"
        or binding.get("engine_delivery") != "integrated"
    ):
        raise GenerationError("fast CPU binding names the wrong delivery identity")

    recorded_source = _require_digest(
        binding.get("source_digest"), label="source_digest"
    )
    observed_source = source_digest()
    if recorded_source != observed_source:
        raise GenerationError(
            "integrated fast CPU source digest drifted: "
            f"recorded={recorded_source}, observed={observed_source}"
        )
    if binding.get("source_digest_v2") is not None:
        recorded_v2 = _require_digest(
            binding.get("source_digest_v2"), label="source_digest_v2"
        )
        if recorded_v2 != source_digest_v2("fast_cpu"):
            raise GenerationError("integrated fast CPU v2 source digest drifted")
    _require_digest(binding.get("patch_sha256"), label="patch_sha256")
    if sha256_of_file(PATCH_PATH) != binding.get("patch_sha256"):
        raise GenerationError("the shipped Gigatoken patch digest does not match")
    if not LICENSE_PATH.is_file() or not NOTICE_PATH.is_file():
        raise GenerationError("the Gigatoken license or modification notice is missing")

    build_flags = binding.get("build_flags")
    if not isinstance(build_flags, list) or not build_flags or not all(
        isinstance(value, str) and value for value in build_flags
    ):
        raise GenerationError("build_flags must be a non-empty string array")
    toolchain = binding.get("toolchain")
    if not isinstance(toolchain, str) or not toolchain.startswith("rustc "):
        raise GenerationError("toolchain must carry the exact rustc identity")

    legal = _require_mapping(binding.get("legal"), label="legal")
    for path_field, digest_field in (
        ("sbom_path", "sbom_sha256"),
        ("license_bundle_path", "license_bundle_sha256"),
    ):
        path = REPOSITORY_ROOT / str(legal.get(path_field, ""))
        expected = _require_digest(legal.get(digest_field), label=digest_field)
        if not path.is_file() or sha256_of_file(path) != expected:
            raise GenerationError(f"{path_field} does not match {digest_field}")

    # These values are evidence lineage only.  Checking them here prevents a
    # later rewrite from making the 12.4-TB campaign appear to have judged a
    # different historical binary.
    lineage = _require_mapping(binding.get("lineage"), label="lineage")
    historical_binary = _require_digest(
        lineage.get("campaign_binary_digest"),
        label="lineage.campaign_binary_digest",
    )
    if lineage.get("release_binary_digest") != historical_binary:
        raise GenerationError("historical campaign/release binary lineage drifted")
    _require_digest(
        lineage.get("engine_wheel_sha256"),
        label="lineage.engine_wheel_sha256",
    )
    _require_digest(
        lineage.get("campaign_wheel_sha256"),
        label="lineage.campaign_wheel_sha256",
    )


def _carried_families(binding: dict[str, Any]) -> dict[str, str]:
    """Carried family to anchor family (third class beside the two above).

    A carried family is certified through a cross-artifact carry-over
    (``docs/contracts/evidence-carryover.md`` Section 3.1). Its fast CPU
    entry is the anchor's certified entry plus the ``repair`` member the
    registry record already holds; it adds no row to the repair roster and
    is not one of the eleven artifacts the campaign judged, so the counts
    below stay those of the judged artifacts.
    """
    raw = binding.get("carried_families")
    if raw is None:
        return {}
    carried = _require_mapping(raw, label="carried_families")
    return {str(family): str(anchor) for family, anchor in carried.items()}


def _direct_families(binding: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Directly judged family to its engine-differential reading (fourth class).

    A directly judged family holds its repair parameters in its registry
    record with no carry-over node, and is certified by a full-corpus
    engine differential of its own
    (``schemas/engine_differential_reading.schema.json``), named here by
    path and SHA-256. Like a carried family it adds no row to
    the repair roster and is not one of the eleven artifacts the campaign
    judged, so the counts below stay those of the judged artifacts.
    """
    raw = binding.get("direct_families")
    if raw is None:
        return {}
    direct = _require_mapping(raw, label="direct_families")
    out: dict[str, dict[str, str]] = {}
    for family, value in direct.items():
        node = _require_mapping(value, label=f"direct_families.{family}")
        reading = node.get("reading")
        if not isinstance(reading, str) or not reading.startswith("readings/"):
            raise GenerationError(
                f"{family}: a directly judged family names its reading under readings/"
            )
        _require_digest(node.get("sha256"), label=f"direct_families.{family}.sha256")
        out[str(family)] = {"reading": reading, "sha256": str(node["sha256"])}
    return out


def _verify_coverage(
    registry: dict[str, Any], binding: dict[str, Any]
) -> tuple[
    set[str], set[str], dict[str, str], dict[str, dict[str, Any]], dict[str, Any]
]:
    loadable = set(binding.get("loadable_families") or ())
    rejected = set(
        _require_mapping(
            binding.get("rejected_families"), label="rejected_families"
        )
    )
    carried = _carried_families(binding)
    direct = _direct_families(binding)
    if len(loadable) != 11:
        raise GenerationError(
            f"fast CPU binding must certify 11 unique artifacts, found {len(loadable)}"
        )
    if loadable & rejected:
        raise GenerationError("a family is both loadable and rejected")
    if set(carried) & (loadable | rejected):
        raise GenerationError(
            "a carried family is also classified loadable or rejected"
        )
    if set(direct) & (loadable | rejected | set(carried)):
        raise GenerationError(
            "a directly judged family is also classified loadable, rejected or carried"
        )
    for family, anchor in sorted(carried.items()):
        if anchor not in loadable:
            raise GenerationError(
                f"{family}: the carry-over anchor {anchor} is not a loadable family"
            )

    coverage = _require_mapping(binding.get("coverage"), label="coverage")
    if coverage.get("unique_tokenizer_artifacts") != len(loadable):
        raise GenerationError("fast CPU coverage must count 11 unique artifacts")
    if coverage.get("model_families") != 12:
        raise GenerationError("fast CPU coverage must count 12 model families")
    inheritance = _require_mapping(
        coverage.get("exact_artifact_inheritance"),
        label="coverage.exact_artifact_inheritance",
    )
    if (
        inheritance.get("inherits_from") != "qwen3_8b"
        or inheritance.get("artifact_sha256")
        != "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
        or inheritance.get("basis") != "tokenizer.json bytes are identical"
    ):
        raise GenerationError("the Nemotron-Terminal inheritance record drifted")
    repositories = inheritance.get("repositories")
    if not isinstance(repositories, list) or len(repositories) != 3:
        raise GenerationError("the exact-artifact inheritance must name three repos")

    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list):
        raise GenerationError("support registry carries no artifacts list")
    by_family = {
        str(row.get("family")): row
        for row in artifacts
        if isinstance(row, dict)
    }
    missing = loadable - set(by_family)
    if missing:
        raise GenerationError(
            "certified fast CPU families are absent from the registry: "
            + ", ".join(sorted(missing))
        )
    known_rejected = rejected & set(by_family)
    known_carried = {
        family: anchor for family, anchor in carried.items() if family in by_family
    }
    known_direct = set(direct) & set(by_family)
    if set(by_family) != loadable | known_rejected | set(known_carried) | known_direct:
        unexpected = (
            set(by_family)
            - loadable
            - known_rejected
            - set(known_carried)
            - known_direct
        )
        raise GenerationError(
            "every shipped artifact must be classified; unclassified: "
            + ", ".join(sorted(unexpected))
        )
    return loadable, known_rejected, known_carried, by_family, coverage


def _verify_repair_tables(
    binding: dict[str, Any],
    loadable: set[str],
    registry_by_family: dict[str, dict[str, Any]],
) -> str:
    repair = _require_mapping(load_json(REPAIR_TABLE_PATH), label="fast repair table")
    if repair.get("config_id") != binding.get("repair_config_id"):
        raise GenerationError("the packaged repair table uses another config id")
    rows = repair.get("families")
    if not isinstance(rows, list):
        raise GenerationError("the packaged repair table has no family rows")
    by_family = {
        str(row.get("family")): row for row in rows if isinstance(row, dict)
    }
    if set(by_family) != loadable:
        raise GenerationError("repair roster differs from certified CPU roster")
    for family, row in by_family.items():
        if row.get("artifact_sha256") != registry_by_family[family].get(
            "artifact_sha256"
        ):
            raise GenerationError(
                f"{family}: repair table and registry artifact digests differ"
            )

    pclass = _require_mapping(repair.get("pclass"), label="pclass metadata")
    compressed = REPAIR_PCLASS_PATH.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != pclass.get("compressed_sha256"):
        raise GenerationError("the packaged pclass archive digest does not match")
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as error:
        raise GenerationError("the packaged pclass archive is corrupt") from error
    if (
        len(raw) != 0x110000
        or hashlib.sha256(raw).hexdigest() != pclass.get("raw_sha256")
    ):
        raise GenerationError("the packaged pclass payload does not match")
    return sha256_of_file(REPAIR_TABLE_PATH)


def _verify_native_reading(
    binding: dict[str, Any],
    loadable: set[str],
    registry_by_family: dict[str, dict[str, Any]],
    fingerprints: dict[str, str] | None = None,
    direct: set[str] | None = None,
) -> None:
    """The build-bound native-front-end parity reading.

    Its rows are the judged roster plus the directly judged families
    present in the registry; a directly judged row carries both 0.3.2
    columns (``special_tokens``, ``literal_segments``).
    """
    judged = loadable | (direct or set())
    validation = _require_mapping(binding.get("validation"), label="validation")
    relative = validation.get("native_frontend_parity")
    if relative != "readings/fast_cpu_native_frontend_parity.json":
        raise GenerationError("the integrated native-frontend reading is not bound")
    reading = _require_mapping(
        load_json(READINGS_ROOT / str(relative)),
        label="integrated native-frontend parity reading",
    )
    engine = _require_mapping(reading.get("engine"), label="reading.engine")
    oracle = _require_mapping(reading.get("oracle"), label="reading.oracle")
    oracle_binding = _require_mapping(binding.get("oracle"), label="oracle")
    if (
        reading.get("schema") != "toktier.fast_cpu.native_frontend_parity.v1"
        or reading.get("unique_artifacts") != len(judged)
        or reading.get("model_families") != 12 + len(judged - loadable)
        or reading.get("all_ids_equal_hf") is not True
        or reading.get("one_python_to_rust_call_per_batch") is not True
        or reading.get("gil_released") is not True
        or int(reading.get("mismatches", -1)) != 0
        or oracle.get("package") != "tokenizers"
        or oracle.get("version") != oracle_binding.get("tokenizers")
        or engine.get("upstream_project") != binding.get("engine")
        or engine.get("version") != binding.get("engine_version")
        or engine.get("delivery") != binding.get("engine_delivery")
        or engine.get("module") != binding.get("engine_module")
        or engine.get("source_digest") != binding.get("source_digest")
        or engine.get("build_flags") != binding.get("build_flags")
        or engine.get("toolchain") != binding.get("toolchain")
    ):
        raise GenerationError("integrated native-frontend parity binding drifted")
    if engine.get("source_digest_v2") not in {
        None,
        source_digest_v2("fast_cpu"),
    }:
        raise GenerationError("integrated native-frontend v2 binding drifted")
    rows = reading.get("rows")
    if not isinstance(rows, list):
        raise GenerationError("native-frontend reading has no rows")
    by_family = {
        str(row.get("family")): row for row in rows if isinstance(row, dict)
    }
    if set(by_family) != judged:
        raise GenerationError("native-frontend parity roster differs from binding")
    for family, row in by_family.items():
        if (
            row.get("artifact_sha256")
            != registry_by_family[family].get("artifact_sha256")
            or row.get("full_encode_equal_hf") is not True
            or row.get("repair_equal_hf") is not True
            or int(row.get("documents", 0)) < 1
            or int(row.get("mismatches", -1)) != 0
        ):
            raise GenerationError(f"{family}: native-frontend parity did not pass")
        # The 0.3.2 columns, when the row carries them, pass their own checks;
        # a directly judged row has to carry both.
        if family in judged - loadable:
            for column in ("special_tokens", "literal_segments"):
                if column not in row:
                    raise GenerationError(
                        f"{family}: a directly judged row carries no {column} column"
                    )
        problems = reading_column_problems(
            row,
            arm="cpu",
            added_frontend_fingerprint=(fingerprints or {}).get(family),
        )
        if problems:
            raise GenerationError(
                "native-frontend parity column did not pass:\n  "
                + "\n  ".join(problems)
            )


def _carried_repair(row: dict[str, Any], anchor: str) -> dict[str, Any]:
    """The repair member of one carried record, checked against its anchor.

    The member is written by the generation channel beside the record's
    cross-artifact carry-over node; this tool keeps it and refuses when the
    node names another anchor or the parameters differ from the anchor's
    roster row (contract C6).
    """
    family = str(row.get("family"))
    carryover = row.get("carryover")
    if not isinstance(carryover, dict) or carryover.get("from_family") != anchor:
        raise GenerationError(
            f"{family}: the registry record carries no cross-artifact "
            f"carry-over from {anchor}"
        )
    existing = ((row.get("backends") or {}).get("fast_cpu") or {}).get("repair")
    repair = _require_mapping(existing, label=f"{family}: backends.fast_cpu.repair")
    roster = {
        str(item.get("family")): item
        for item in load_json(REPAIR_TABLE_PATH).get("families") or ()
        if isinstance(item, dict)
    }
    anchor_row = roster.get(anchor)
    if anchor_row is None:
        raise GenerationError(f"{family}: the anchor {anchor} has no roster row")
    for key in ("margin", "effective_l_max", "has_normalizer"):
        if repair.get(key) != anchor_row.get(key):
            raise GenerationError(
                f"{family}: repair {key} differs from the {anchor} roster row"
            )
    _require_digest(
        repair.get("source_table_sha256"),
        label=f"{family}: repair.source_table_sha256",
    )
    return {
        "margin": int(repair["margin"]),
        "effective_l_max": int(repair["effective_l_max"]),
        "has_normalizer": bool(repair["has_normalizer"]),
        "source_table_sha256": str(repair["source_table_sha256"]),
    }


def _direct_repair(row: dict[str, Any]) -> dict[str, Any]:
    """The repair member of one directly judged record, kept as recorded.

    The member is written with the record (lane of the generation channel:
    the exporter's recomputation over the frozen directory); this tool
    keeps it, refuses a record that also holds a carry-over node, and
    leaves the comparison with the engine-differential reading to
    ``tools/generate_registry.py`` (conditions D1-D6).
    """
    family = str(row.get("family"))
    if "carryover" in row:
        raise GenerationError(
            f"{family}: a directly judged record holds no carry-over node"
        )
    existing = ((row.get("backends") or {}).get("fast_cpu") or {}).get("repair")
    repair = _require_mapping(existing, label=f"{family}: backends.fast_cpu.repair")
    _require_digest(
        repair.get("source_table_sha256"),
        label=f"{family}: repair.source_table_sha256",
    )
    return {
        "margin": int(repair["margin"]),
        "effective_l_max": int(repair["effective_l_max"]),
        "has_normalizer": bool(repair["has_normalizer"]),
        "source_table_sha256": str(repair["source_table_sha256"]),
    }


def _engine_differential(family: str, node: dict[str, str]) -> dict[str, Any]:
    """The registry's ``engine_differential`` member, read from the reading."""
    path = READINGS_ROOT / node["reading"]
    if not path.is_file():
        raise GenerationError(f"{family}: engine-differential reading is missing")
    if sha256_of_file(path) != node["sha256"]:
        raise GenerationError(
            f"{family}: engine-differential reading digest differs from the binding"
        )
    reading = _require_mapping(load_json(path), label=f"{family}: reading")
    main = _require_mapping(reading.get("main_pass"), label=f"{family}: main_pass")
    if int(main.get("mismatches", -1)) != 0:
        raise GenerationError(f"{family}: engine-differential reading has mismatches")
    return {
        "reading": node["reading"],
        "sha256": node["sha256"],
        "documents": int(main["documents"]),
        "mismatches": 0,
    }


def _records_v2(registry: dict[str, Any]) -> bool:
    return any(
        isinstance(entry, dict) and "source_digest_v2" in entry
        for row in registry.get("artifacts", [])
        if isinstance(row, dict)
        for entry in [(row.get("backends") or {}).get("fast_cpu")]
    )


def augmented_document(
    registry: dict[str, Any],
    binding: dict[str, Any],
    *,
    include_v2: bool | None = None,
) -> dict[str, Any]:
    """Return ``registry`` with exactly the binding-owned entries applied."""
    if include_v2 is None:
        include_v2 = _records_v2(registry)
    _verify_identity(binding)
    completed = copy.deepcopy(registry)
    loadable, rejected, carried, by_family, _coverage = _verify_coverage(
        completed, binding
    )
    direct = {
        family: node
        for family, node in _direct_families(binding).items()
        if family in by_family
    }
    config_digest = _verify_repair_tables(binding, loadable, by_family)
    _verify_native_reading(
        binding, loadable, by_family, registry_fingerprints(completed), set(direct)
    )

    certified_entry = {
        "status": "certified_source",
        "source_digest": str(binding["source_digest"]),
        "build_flags": list(binding["build_flags"]),
        "toolchain": str(binding["toolchain"]),
        "engine": str(binding["engine"]),
        "engine_version": str(binding["engine_version"]),
        "engine_delivery": str(binding["engine_delivery"]),
        "engine_module": str(binding["engine_module"]),
        "engine_unicode_data": str(binding["engine_unicode_data"]),
        "patch_sha256": str(binding["patch_sha256"]),
        "config_id": str(binding["repair_config_id"]),
        "config_digest": config_digest,
    }
    if include_v2:
        certified_entry["source_digest_v2"] = source_digest_v2("fast_cpu")
    artifacts = completed["artifacts"]
    assert isinstance(artifacts, list)
    for row in artifacts:
        if not isinstance(row, dict):
            raise GenerationError("support registry has a non-object artifact row")
        family = str(row.get("family"))
        backends = row.get("backends")
        if not isinstance(backends, dict):
            raise GenerationError(f"{family}: backends must be an object")
        if family in loadable:
            backends["fast_cpu"] = dict(certified_entry)
        elif family in carried:
            backends["fast_cpu"] = {
                **certified_entry,
                "repair": _carried_repair(row, carried[family]),
            }
        elif family in direct:
            backends["fast_cpu"] = {
                **certified_entry,
                "repair": _direct_repair(row),
                "engine_differential": _engine_differential(family, direct[family]),
            }
        elif family in rejected:
            backends["fast_cpu"] = {
                "status": "unsupported",
                "engine": str(binding["engine"]),
                "engine_version": str(binding["engine_version"]),
                "engine_delivery": str(binding["engine_delivery"]),
                "engine_module": str(binding["engine_module"]),
                "config_id": str(binding["repair_config_id"]),
                "config_digest": config_digest,
            }

    return with_root_digest(completed, REGISTRY_DOMAIN_TAG)


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialise_document(document))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply or check the integrated fast CPU registry binding."
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)

    registry = _require_mapping(load_json(DEFAULT_REGISTRY), label="registry")
    binding = _require_mapping(load_json(BINDING_PATH), label="binding")
    generated = augmented_document(
        registry,
        binding,
        include_v2=not arguments.check or _records_v2(registry),
    )
    violations = schema_violations(generated, load_json(SCHEMA_PATH))
    if violations:
        raise GenerationError(
            "augmented registry violates its schema:\n  " + "\n  ".join(violations)
        )
    payload = serialise_document(generated)
    if arguments.check:
        problems = []
        for path in (DEFAULT_REGISTRY, PACKAGED_REGISTRY):
            if not path.is_file() or path.read_bytes() != payload:
                problems.append(f"{path} is not the generated fast CPU registry")
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1 if problems else 0
    _write(DEFAULT_REGISTRY, generated)
    _write(PACKAGED_REGISTRY, generated)
    print(f"updated {DEFAULT_REGISTRY} and {PACKAGED_REGISTRY}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
