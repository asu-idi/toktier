#!/usr/bin/env python3
"""Apply or verify the evidence-bound public Rust runtime build row."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

from compute_identity_v2 import source_digest as source_digest_v2
from fast_cpu_source_identity import source_digest as fast_cpu_source_digest
from native_host_source_identity import source_digest as native_host_source_digest
from registry_common import (
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    load_json,
    schema_violations,
    serialise_document,
    sha256_of_file,
    with_root_digest,
)
from rust_api_source_identity import source_digest as rust_api_source_digest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tables" / "support_registry.json"
PACKAGED_REGISTRY = ROOT / "src/toktier/routing/tables/support_registry.v1.json"
SCHEMA = ROOT / "schemas/support_registry.schema.json"
BINDING = ROOT / "tools/rust_api_binding.json"
FAST_CPU_BINDING = ROOT / "tools/fast_cpu_binding.json"
CPU_READING = ROOT / "readings/fast_cpu_native_frontend_parity.json"
GPU_READINGS = {
    "sm_80": ROOT / "readings/gpu_native_frontend_sm80_parity.json",
    "sm_89": ROOT / "readings/gpu_native_frontend_sm89_parity.json",
    "sm_90": ROOT / "readings/gpu_native_frontend_sm90_parity.json",
    "sm_120": ROOT / "readings/gpu_native_frontend_sm120_parity.json",
}
#: The venues a Rust API matrix is taken at, and where each reading lands.
#: The key is also the architecture the reading has to name: every GPU
#: venue asks for `cuda:0`, so the requested device does not tell two of
#: them apart and the reading says which device answered.
MATRIX_READINGS = {
    "cpu": ROOT / "readings/rust_api_matrix_cpu.json",
    "sm_89": ROOT / "readings/rust_api_matrix_sm89.json",
    "sm_120": ROOT / "readings/rust_api_matrix_sm120.json",
}

#: Families every venue matrix has to cover, and the documents that follows
#: from the seven fixed cases each family is asked. A matrix that answers for
#: fewer families is a hole, not a smaller campaign.
#:
#: The roster is the certified artifact records of the support registry,
#: one row per family (see :func:`certified_roster`). The matrix is the
#: evidence that records a Rust build in ``runtime_builds``, so it speaks
#: only for artifacts some campaign has judged: an examined artifact
#: (``src/toktier/routing/tables/examined_artifacts.v1.json``) is never in
#: the registry and is therefore not a row of the matrix, and a reading that
#: carries one does not match the roster.
MATRIX_FAMILIES = 21
MATRIX_DOCUMENTS = MATRIX_FAMILIES * 7

#: Families the GPU parity readings cover; the same roster.
GPU_FAMILIES = MATRIX_FAMILIES

#: The CPU venue's backend roster and the CPU parity reading's counts are
#: not written here: both follow from ``tools/fast_cpu_binding.json`` (see
#: :func:`fast_cpu_route` and :func:`cpu_matrix_backends`).


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def fast_cpu_route(fast_binding: dict[str, Any]) -> dict[str, set[str]]:
    """The families the fast CPU binding places on the integrated CPU engine.

    Three classes run there: the loadable families the campaign judged, the
    families carried from one of them, and the directly judged families
    whose own engine-differential reading is recorded (path and SHA-256)
    and present with that digest. A directly judged family registered
    before its reading exists (``"sha256": null``) is not on the route yet
    and answers through the reference engine. Any other shape is refused.
    """
    loadable_raw = fast_binding.get("loadable_families")
    if (
        not isinstance(loadable_raw, list)
        or not loadable_raw
        or not all(isinstance(item, str) and item for item in loadable_raw)
        or len(set(loadable_raw)) != len(loadable_raw)
    ):
        raise GenerationError(
            "fast CPU binding loadable_families must be a non-empty list of "
            "distinct family names"
        )
    loadable = set(loadable_raw)
    carried_raw = fast_binding.get("carried_families") or {}
    if not isinstance(carried_raw, dict) or not all(
        isinstance(family, str) and family and isinstance(anchor, str)
        for family, anchor in carried_raw.items()
    ):
        raise GenerationError(
            "fast CPU binding carried_families must map a family to its anchor"
        )
    for family, anchor in sorted(carried_raw.items()):
        if anchor not in loadable:
            raise GenerationError(
                f"fast CPU binding carries {family} from {anchor}, which is not "
                "a loadable family"
            )
    carried = set(carried_raw)
    direct_raw = fast_binding.get("direct_families") or {}
    if not isinstance(direct_raw, dict):
        raise GenerationError(
            "fast CPU binding direct_families must map a family to its reading"
        )
    direct: set[str] = set()
    pending: set[str] = set()
    for family, node in sorted(direct_raw.items()):
        reading = node.get("reading") if isinstance(node, dict) else None
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(node, dict)
            or not isinstance(reading, str)
            or not reading.startswith("readings/")
        ):
            raise GenerationError(
                f"fast CPU binding direct_families.{family} must name its "
                "reading under readings/"
            )
        recorded = node.get("sha256")
        if recorded is None:
            pending.add(family)
            continue
        if not _digest(recorded):
            raise GenerationError(
                f"fast CPU binding direct_families.{family}.sha256 is not a "
                "SHA-256 digest"
            )
        path = ROOT / reading
        if not path.is_file():
            raise GenerationError(
                f"{family}: the fast CPU binding records the engine-differential "
                f"reading {reading}, and the file is missing"
            )
        observed = sha256_of_file(path)
        if observed != recorded:
            raise GenerationError(
                f"{family}: engine-differential reading digest {observed} does "
                f"not match the recorded {recorded}"
            )
        direct.add(family)
    classes = (loadable, carried, direct | pending)
    if any(
        first & second
        for index, first in enumerate(classes)
        for second in classes[index + 1 :]
    ):
        raise GenerationError(
            "fast CPU binding lists a family in more than one of the loadable, "
            "carried and directly judged classes"
        )
    return {
        "loadable": loadable,
        "carried": carried,
        "direct": direct,
        "pending": pending,
    }


def cpu_parity_counts(fast_binding: dict[str, Any]) -> dict[str, Any]:
    """What the CPU native-front-end parity reading has to read.

    Its rows are the loadable families and the directly judged families on
    the route (carried families inherit their anchor's row), so
    ``unique_artifacts`` is their number. ``model_families`` is the
    binding's coverage count of the judged campaign (which includes the
    exact-artifact inheritance) plus one for each directly judged family.
    """
    route = fast_cpu_route(fast_binding)
    coverage = fast_binding.get("coverage")
    if not isinstance(coverage, dict):
        raise GenerationError("fast CPU binding has no coverage object")
    unique = coverage.get("unique_tokenizer_artifacts")
    families = coverage.get("model_families")
    if (
        not isinstance(unique, int)
        or isinstance(unique, bool)
        or not isinstance(families, int)
        or isinstance(families, bool)
        or unique != len(route["loadable"])
        or families < unique
    ):
        raise GenerationError(
            "fast CPU binding coverage does not count its loadable families "
            f"(unique_tokenizer_artifacts={unique!r}, model_families={families!r}, "
            f"loadable={len(route['loadable'])})"
        )
    rows = route["loadable"] | route["direct"]
    return {
        "rows": rows,
        "unique_artifacts": len(rows),
        "model_families": families + len(route["direct"]),
    }


def cpu_matrix_backends(
    fast_binding: dict[str, Any], roster: set[str]
) -> dict[str, str]:
    """The backend each roster family answers through at the CPU venue.

    A family on the fast CPU route (:func:`fast_cpu_route`) runs on the
    integrated engine (``FastCpu``); every other certified family answers
    through the reference engine (``HuggingFace``).
    """
    route = fast_cpu_route(fast_binding)
    on_route = route["loadable"] | route["carried"] | route["direct"]
    outside = on_route - roster
    if outside:
        raise GenerationError(
            "fast CPU binding places families outside the certified roster on "
            f"the CPU route: {sorted(outside)}"
        )
    return {
        family: ("FastCpu" if family in on_route else "HuggingFace")
        for family in roster
    }


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be an object")
    return value


def _verify_build(build: dict[str, Any], *, label: str) -> None:
    if not isinstance(build.get("build_flags"), list) or not build["build_flags"]:
        raise GenerationError(f"{label} has no exact build flags")
    if not isinstance(build.get("toolchain"), str) or not build["toolchain"]:
        raise GenerationError(f"{label} has no exact toolchain")
    if not isinstance(build.get("evidence_id"), str) or not build["evidence_id"]:
        raise GenerationError(f"{label} has no evidence id")


def verify_binding(binding: dict[str, Any]) -> None:
    expected = {
        "runtime": "rust_api",
        "source_digest": rust_api_source_digest(),
        "fast_cpu_source_digest": fast_cpu_source_digest(),
        "native_host_source_digest": native_host_source_digest(),
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise GenerationError(
                f"Rust API binding {key} does not match the current tree "
                f"(recorded={binding.get(key)!r}, current={value!r})"
            )
    expected_v2 = {
        "source_digest_v2": source_digest_v2("rust_api"),
        "fast_cpu_source_digest_v2": source_digest_v2("fast_cpu"),
        "native_host_source_digest_v2": source_digest_v2("native_host"),
    }
    present_v2 = set(expected_v2) & set(binding)
    if present_v2 and present_v2 != set(expected_v2):
        raise GenerationError("Rust API binding has an incomplete v2 identity tuple")
    for key in present_v2:
        if binding[key] != expected_v2[key]:
            raise GenerationError(f"Rust API binding {key} does not match the tree")
    _verify_build(binding, label="Rust API default build")
    additional = binding.get("additional_builds", [])
    if not isinstance(additional, list):
        raise GenerationError("Rust API additional_builds must be an array")
    names: set[str] = set()
    identities = {(tuple(binding["build_flags"]), binding["toolchain"])}
    for index, raw in enumerate(additional):
        build = _mapping(raw, label=f"Rust API additional build {index}")
        name = build.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise GenerationError("Rust API additional build names must be unique")
        names.add(name)
        _verify_build(build, label=f"Rust API additional build {name}")
        identity = (tuple(build["build_flags"]), build["toolchain"])
        if identity in identities:
            raise GenerationError(f"Rust API additional build {name} is a duplicate")
        identities.add(identity)
        if build.get("evidence_reading") != "rust_direct_jit_sm120":
            raise GenerationError(
                f"Rust API additional build {name} has an unsupported evidence reading"
            )


def verify_shared_evidence(binding: dict[str, Any]) -> None:
    cpu = _mapping(load_json(CPU_READING), label="CPU native-front-end reading")
    engine = _mapping(cpu.get("engine"), label="CPU engine facts")
    expected = cpu_parity_counts(
        _mapping(load_json(FAST_CPU_BINDING), label="fast CPU binding")
    )
    rows = cpu.get("rows")
    listed = (
        [str(row.get("family")) for row in rows if isinstance(row, dict)]
        if isinstance(rows, list)
        else []
    )
    if (
        not isinstance(rows, list)
        or len(listed) != len(rows)
        or len(set(listed)) != len(listed)
        or set(listed) != expected["rows"]
    ):
        raise GenerationError(
            "CPU native-front-end reading rows are not the fast CPU route "
            f"(outside it: {sorted(set(listed) - expected['rows'])}, "
            f"missing: {sorted(expected['rows'] - set(listed))})"
        )
    for key in ("unique_artifacts", "model_families"):
        if cpu.get(key) != expected[key]:
            raise GenerationError(
                f"CPU native-front-end reading {key} is {cpu.get(key)!r}; the "
                f"fast CPU binding gives {expected[key]}"
            )
    if (
        cpu.get("schema") != "toktier.fast_cpu.native_frontend_parity.v1"
        or cpu.get("mismatches") != 0
        or cpu.get("all_ids_equal_hf") is not True
        or engine.get("source_digest") != binding["fast_cpu_source_digest"]
    ):
        raise GenerationError(
            "CPU native-front-end evidence does not admit this Rust build"
        )
    if engine.get("source_digest_v2") not in {
        None,
        source_digest_v2("fast_cpu"),
    }:
        raise GenerationError("CPU native-front-end v2 identity drifted")
    for architecture, path in GPU_READINGS.items():
        reading = _mapping(load_json(path), label=f"{architecture} GPU reading")
        host = _mapping(reading.get("native_host_build_facts"), label="GPU host facts")
        if (
            reading.get("schema") != "toktier.gpu.native_frontend_parity.v1"
            or reading.get("architecture") != architecture
            or reading.get("families") != GPU_FAMILIES
            or reading.get("mismatches") != 0
            or reading.get("all_ids_equal_hf") is not True
            or host.get("host_source_digest") != binding["native_host_source_digest"]
        ):
            raise GenerationError(
                f"{architecture} native-front-end evidence does not admit "
                "this Rust build"
            )
        if host.get("host_source_digest_v2") not in {
            None,
            source_digest_v2("native_host"),
        }:
            raise GenerationError(f"{architecture} native-host v2 identity drifted")


def certified_roster(registry: dict[str, Any]) -> set[str]:
    """The families of the registry's certified artifact records.

    Examined artifacts live in their own table and never in the registry
    (``docs/contracts/registry.md`` Section 3.0), so this set leaves them
    out by construction.
    """
    return {
        str(raw["family"])
        for raw in registry.get("artifacts", [])
        if isinstance(raw, dict) and "family" in raw
    }


def verify_public_matrix(
    binding: dict[str, Any], *, bootstrap: bool, roster: set[str]
) -> None:
    if len(roster) != MATRIX_FAMILIES:
        raise GenerationError(
            f"support registry certifies {len(roster)} artifacts; the Rust API "
            f"matrix roster is {MATRIX_FAMILIES}"
        )
    cpu_route_binding = _mapping(load_json(FAST_CPU_BINDING), label="fast CPU binding")
    for target, path in MATRIX_READINGS.items():
        reading = _mapping(load_json(path), label=f"Rust API {target} matrix")
        if (
            reading.get("schema") != "toktier.rust_api.matrix.v1"
            or reading.get("runtime_source_digest") != binding["source_digest"]
            or reading.get("fast_cpu_source_digest")
            != binding["fast_cpu_source_digest"]
            or reading.get("native_host_source_digest")
            != binding["native_host_source_digest"]
            or reading.get("toolchain") != binding["toolchain"]
            or reading.get("build_flags") != binding["build_flags"]
            or reading.get("architecture") != (None if target == "cpu" else target)
            or reading.get("families") != MATRIX_FAMILIES
            or reading.get("documents") != MATRIX_DOCUMENTS
            or reading.get("mismatches") != 0
            or (not bootstrap and reading.get("runtime_build_certified") is not True)
        ):
            raise GenerationError(
                f"Rust API {target} matrix does not match the binding"
            )
        rows = reading.get("rows")
        if not isinstance(rows, list) or len(rows) != MATRIX_FAMILIES:
            raise GenerationError(
                f"Rust API {target} matrix has an incomplete family roster"
            )
        families = [str(row.get("family")) for row in rows if isinstance(row, dict)]
        if len(families) != len(rows) or set(families) != roster or len(
            set(families)
        ) != len(families):
            raise GenerationError(
                f"Rust API {target} matrix roster is not the certified artifact "
                f"roster (outside it: {sorted(set(families) - roster)}, "
                f"missing: {sorted(roster - set(families))})"
            )
        expected_backends = (
            cpu_matrix_backends(cpu_route_binding, roster)
            if target == "cpu"
            else dict.fromkeys(roster, "Gpu")
        )
        backend_counts: dict[str, int] = {}
        expected_counts: dict[str, int] = {}
        for backend in expected_backends.values():
            expected_counts[backend] = expected_counts.get(backend, 0) + 1
        differing: list[str] = []
        for row in rows:
            if not isinstance(row, dict) or row.get("mismatches") != 0:
                raise GenerationError(f"Rust API {target} matrix has a failing row")
            backend = str(row.get("backend"))
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            if backend != expected_backends[str(row.get("family"))]:
                differing.append(str(row.get("family")))
        if backend_counts != expected_counts or differing:
            raise GenerationError(
                f"Rust API {target} matrix backend roster is {backend_counts}, "
                f"expected {expected_counts} (families on another backend: "
                f"{sorted(differing)})"
            )


def verify_additional_builds(binding: dict[str, Any], *, bootstrap: bool) -> None:
    for raw in binding.get("additional_builds", []):
        build = _mapping(raw, label="Rust API additional build")
        name = str(build["name"])
        if build["evidence_reading"] != "rust_direct_jit_sm120":
            raise GenerationError(f"unsupported evidence for additional build {name}")
        reading = _mapping(
            load_json(ROOT / "readings/rust_direct_jit_sm120.json"),
            label=f"Rust API additional build {name} evidence",
        )
        if (
            reading.get("schema") != "toktier.rust_direct_jit.matrix.v1"
            or reading.get("runtime_source_digest") != binding["source_digest"]
            or reading.get("native_host_source_digest")
            != binding["native_host_source_digest"]
            or reading.get("rust_toolchain") != build["toolchain"]
            or reading.get("rust_build_flags") != build["build_flags"]
            or reading.get("families") != MATRIX_FAMILIES
            or reading.get("documents") != MATRIX_DOCUMENTS
            or reading.get("mismatches") != 0
            or (
                not bootstrap
                and reading.get("runtime_build_certified") is not True
            )
            or (not bootstrap and reading.get("experimental_opt_in") is not False)
        ):
            raise GenerationError(
                f"Rust API additional build {name} does not match its evidence"
            )
        rows = reading.get("rows")
        if not isinstance(rows, list) or len(rows) != MATRIX_FAMILIES or any(
            not isinstance(row, dict)
            or row.get("mismatches") != 0
            or row.get("certified") is not True
            for row in rows
        ):
            raise GenerationError(
                f"Rust API additional build {name} has incomplete certified rows"
            )


def _records_v2(registry: dict[str, Any]) -> bool:
    return any(
        isinstance(row, dict) and "source_digest_v2" in row
        for row in registry.get("runtime_builds", [])
    )


def augmented_document(
    registry: dict[str, Any],
    binding: dict[str, Any],
    *,
    bootstrap: bool = False,
    include_v2: bool | None = None,
) -> dict[str, Any]:
    if include_v2 is None:
        include_v2 = _records_v2(registry)
    verify_binding(binding)
    verify_shared_evidence(binding)
    verify_public_matrix(
        binding, bootstrap=bootstrap, roster=certified_roster(registry)
    )
    verify_additional_builds(binding, bootstrap=bootstrap)
    completed = copy.deepcopy(registry)
    common = {
        key: binding[key]
        for key in (
            "runtime",
            "source_digest",
            "fast_cpu_source_digest",
            "native_host_source_digest",
        )
    }
    if include_v2:
        common.update(
            {
                "source_digest_v2": source_digest_v2("rust_api"),
                "fast_cpu_source_digest_v2": source_digest_v2("fast_cpu"),
                "native_host_source_digest_v2": source_digest_v2("native_host"),
            }
        )
    builds = [binding, *binding.get("additional_builds", [])]
    completed["runtime_builds"] = [
        {
            **common,
            **{
                key: build[key]
                for key in ("build_flags", "toolchain", "evidence_id")
            },
        }
        for build in builds
    ]
    return with_root_digest(completed, REGISTRY_DOMAIN_TAG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="admit exact candidate builds before the final certified replay",
    )
    arguments = parser.parse_args(argv)
    if arguments.check and arguments.bootstrap:
        parser.error("--check and --bootstrap are mutually exclusive")
    registry = _mapping(load_json(REGISTRY), label="support registry")
    binding = _mapping(load_json(BINDING), label="Rust API binding")
    generated = augmented_document(
        registry,
        binding,
        bootstrap=arguments.bootstrap,
        include_v2=(
            not arguments.check or arguments.bootstrap or _records_v2(registry)
        ),
    )
    violations = schema_violations(generated, load_json(SCHEMA))
    if violations:
        raise GenerationError(
            "Rust API registry violates its schema:\n  " + "\n  ".join(violations)
        )
    payload = serialise_document(generated)
    if arguments.check:
        problems = [
            str(path)
            for path in (REGISTRY, PACKAGED_REGISTRY)
            if not path.is_file() or path.read_bytes() != payload
        ]
        for path in problems:
            print(
                f"error: {path} is not the generated Rust API registry",
                file=sys.stderr,
            )
        return 1 if problems else 0
    REGISTRY.write_bytes(payload)
    PACKAGED_REGISTRY.write_bytes(payload)
    print(f"updated {REGISTRY} and {PACKAGED_REGISTRY}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
