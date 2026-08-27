#!/usr/bin/env python3
"""Apply the pinned Fastokens distribution binding to the support registry.

The explicit experimental Fastokens adapter recognises the wheels the toktier
project publishes by the content digest of the installed ``fastokens/``
package.  This tool carries ``tools/fastokens_binding.json`` -- the one
hand-written source -- into the registry's ``engine_distributions.fastokens``
node after checking it against the files it names: the readings the numbers
come from, the evidence manifest the evidence id resolves to, the patch
series and legal material in ``packaging/fastokens-pinned``, the registry's
own artifact roster and oracle, and the adapter source that compiles the
guard.  The node is not a backend status and admits no route; it records
which published bytes the adapter's evidence describes.

The adapter file is not part of the three source identities.  Its binding
here is the one that ties the code reporting ``engine_assurance`` to the node
it reads: a change to either without the other fails ``--check``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

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
BINDING_PATH = REPOSITORY_ROOT / "tools" / "fastokens_binding.json"
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "support_registry.schema.json"
EVIDENCE_PATH = REPOSITORY_ROOT / "evidence" / "evidence_manifest_fastokens_pinned.json"

#: Domain tag of the guard-set digest; the registry node states the same
#: construction in words so a reader can recompute it.
GUARD_SET_DOMAIN = b"toktier.fastokens.unicode_guard.v1\0"

#: The keys of the binding that become the registry node, in node order.
NODE_KEYS = (
    "backend",
    "admission",
    "distribution",
    "recognised_distributions",
    "upstream",
    "source",
    "known_wheels",
    "sdist",
    "oracle",
    "families",
    "guard",
    "evidence",
)


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be a JSON object")
    return value


def _require_list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise GenerationError(f"{label} must be a non-empty JSON array")
    return value


def _require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GenerationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_file_digest(relative: object, expected: object, *, label: str) -> None:
    path = REPOSITORY_ROOT / str(relative)
    digest = _require_digest(expected, label=f"{label} digest")
    if not path.is_file():
        raise GenerationError(f"{label}: {relative} is missing")
    if sha256_of_file(path) != digest:
        raise GenerationError(f"{label}: {relative} does not match its recorded digest")


def parse_codepoint(text: object) -> int:
    """``U+XXXX`` to an integer; refuses anything else."""
    if not isinstance(text, str) or not text.startswith("U+"):
        raise GenerationError(f"code point {text!r} is not written as U+XXXX")
    try:
        value = int(text[2:], 16)
    except ValueError as error:
        raise GenerationError(f"code point {text!r} is not hexadecimal") from error
    if not 0 <= value <= 0x10FFFF or len(text) < 6:
        raise GenerationError(f"code point {text!r} is outside Unicode")
    return value


def guard_codepoints(ranges: object) -> list[int]:
    """Expand ``[["U+..", "U+.."], ...]`` into sorted, distinct code points."""
    out: list[int] = []
    previous_end = -1
    for pair in _require_list(ranges, label="guard.ranges"):
        if not isinstance(pair, list) or len(pair) != 2:
            raise GenerationError("guard.ranges entries must be [lo, hi] pairs")
        low, high = (parse_codepoint(pair[0]), parse_codepoint(pair[1]))
        if low > high or low <= previous_end:
            raise GenerationError("guard.ranges must be ascending and disjoint")
        out.extend(range(low, high + 1))
        previous_end = high
    return out


def guard_set_digest(codepoints: list[int]) -> str:
    joined = "\n".join(f"U+{value:04X}" for value in sorted(codepoints))
    return hashlib.sha256(GUARD_SET_DOMAIN + joined.encode("ascii")).hexdigest()


def _verify_wheels(binding: dict[str, Any]) -> str:
    wheels = _require_list(binding.get("known_wheels"), label="known_wheels")
    version = str(
        _require_mapping(binding.get("distribution"), label="distribution")["version"]
    )
    engine_digests: set[str] = set()
    for index, wheel in enumerate(wheels):
        wheel = _require_mapping(wheel, label=f"known_wheels[{index}]")
        name = str(wheel.get("filename", ""))
        if not name.startswith(f"toktier_fastokens-{version}-") or not name.endswith(
            ".whl"
        ):
            raise GenerationError(
                f"known_wheels[{index}] is not a {version} wheel of the distribution"
            )
        _require_digest(wheel.get("sha256"), label=f"known_wheels[{index}].sha256")
        engine = _require_digest(
            wheel.get("engine_digest"), label=f"known_wheels[{index}].engine_digest"
        )
        if engine in engine_digests:
            raise GenerationError("two known wheels carry the same engine digest")
        engine_digests.add(engine)
        files = _require_list(
            wheel.get("code_files"), label=f"known_wheels[{index}].code_files"
        )
        names = [
            str(_require_mapping(entry, label="code_files entry").get("name"))
            for entry in files
        ]
        if names != sorted(names) or any(not n.startswith("fastokens/") for n in names):
            raise GenerationError(
                f"known_wheels[{index}].code_files must be sorted fastokens/ paths"
            )
        for entry in files:
            _require_digest(entry.get("sha256"), label="code_files entry digest")
    return str(wheels[0]["engine_digest"])


def _verify_families(registry: dict[str, Any], binding: dict[str, Any]) -> None:
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list):
        raise GenerationError("support registry carries no artifacts list")
    by_family = {
        str(row.get("family")): str(row.get("artifact_sha256"))
        for row in artifacts
        if isinstance(row, dict)
    }
    listed: dict[str, str] = {}
    for entry in _require_list(binding.get("families"), label="families"):
        entry = _require_mapping(entry, label="families entry")
        family = str(entry.get("family"))
        if family in listed:
            raise GenerationError(f"{family} is listed twice in the evidence families")
        listed[family] = _require_digest(
            entry.get("artifact_sha256"), label=f"{family} artifact"
        )
    missing = set(listed) - set(by_family)
    if missing:
        raise GenerationError(
            "evidence families absent from the registry: " + ", ".join(sorted(missing))
        )
    for family, digest in listed.items():
        if by_family[family] != digest:
            raise GenerationError(
                f"{family}: evidence artifact digest differs from the registry"
            )
    oracle = _require_mapping(binding.get("oracle"), label="oracle")
    certified = {
        str(version)
        for row in registry.get("oracles", [])
        if isinstance(row, dict) and row.get("package") == oracle.get("package")
        for version in row.get("certified_versions", [])
    }
    if str(oracle.get("version")) not in certified:
        raise GenerationError(
            "the evidence oracle is not a certified oracle version of the registry"
        )


def _verify_guard(binding: dict[str, Any], engine_digest: str) -> None:
    guard = _require_mapping(binding.get("guard"), label="guard")
    codepoints = guard_codepoints(guard.get("ranges"))
    if guard.get("codepoints") != len(codepoints):
        raise GenerationError(
            "guard.codepoints does not equal the size of guard.ranges"
        )
    if guard.get("set_sha256") != guard_set_digest(codepoints):
        raise GenerationError("guard.set_sha256 does not match guard.ranges")
    derived = _require_mapping(
        guard.get("derived_against"), label="guard.derived_against"
    )
    if derived.get("engine_digest") != engine_digest:
        raise GenerationError("the guard was not derived on the first known wheel")
    reading = _require_mapping(
        load_json(REPOSITORY_ROOT / str(guard.get("selftest_reading"))),
        label="guard selftest reading",
    )
    if reading.get("engine_digest") != engine_digest:
        raise GenerationError(
            "the guard reading was not taken on the first known wheel"
        )
    if [parse_codepoint(cp) for cp in reading.get("codepoints", [])] != codepoints:
        raise GenerationError(
            "guard.ranges does not enumerate the guard reading's code points"
        )
    if reading.get("set_sha256") != guard.get("set_sha256"):
        raise GenerationError("guard reading and binding disagree on the set digest")
    comparison = _require_mapping(
        reading.get("comparison"), label="guard reading comparison"
    )
    if comparison.get("verdict") != "SUPERSET" or comparison.get("only_in_archived"):
        raise GenerationError(
            "the guard reading must be a superset of the archived set"
        )


def _verify_evidence(binding: dict[str, Any], engine_digest: str) -> None:
    evidence = _require_mapping(binding.get("evidence"), label="evidence")
    readings = _require_mapping(evidence.get("readings"), label="evidence.readings")
    loaded: dict[str, dict[str, Any]] = {}
    for gate in ("gate1", "gate2", "gate3", "gate4", "guard"):
        path = REPOSITORY_ROOT / str(readings.get(gate, ""))
        if not path.is_file():
            raise GenerationError(f"evidence.readings.{gate} is missing")
        loaded[gate] = _require_mapping(load_json(path), label=f"{gate} reading")
    gate1 = loaded["gate1"]
    subject = _require_mapping(gate1.get("subject"), label="gate1 subject")
    if subject.get("engine_digest") != engine_digest:
        raise GenerationError(
            "the gate1 reading was not taken on the first known wheel"
        )
    for gate in ("gate2", "gate3", "gate4"):
        if loaded[gate].get("engine_digest") != engine_digest:
            raise GenerationError(
                f"the {gate} reading was not taken on the first known wheel"
            )
    totals = _require_mapping(gate1.get("totals"), label="gate1 totals")
    expectations = {
        "docs_per_family": gate1.get("docs_per_family"),
        "families": gate1.get("families"),
        "comparisons": gate1.get("comparisons"),
        "mismatch_raw": totals.get("mismatch_raw"),
        "mismatch_guarded": totals.get("mismatch_guarded"),
        "engine_error": totals.get("engine_error"),
        "routed_reference_per_family": totals.get("routed_reference_per_family"),
        "visible_cpus": gate1.get("visible_cpus"),
        "guard_codepoints_in_run": _require_mapping(
            gate1.get("guard"), label="gate1 guard"
        ).get("codepoints_in_harness"),
    }
    for key, expected in expectations.items():
        if evidence.get(key) != expected:
            raise GenerationError(f"evidence.{key} differs from the gate1 reading")
    if int(evidence.get("comparisons") or 0) != int(
        evidence.get("docs_per_family") or 0
    ) * int(evidence.get("families") or 0):
        raise GenerationError("evidence.comparisons is not docs_per_family x families")
    if gate1.get("verdict_guarded") != "PASS":
        raise GenerationError("the gate1 guarded verdict is not PASS")
    gate2 = _require_mapping(evidence.get("gate2"), label="evidence.gate2")
    arms = _require_mapping(loaded["gate2"].get("arms"), label="gate2 arms")
    if (
        gate2.get("drift_events") != loaded["gate2"].get("total_drift_events")
        or gate2.get("targeted_cases") != arms["targeted"].get("cases")
        or gate2.get("matrix_steps") != arms["matrix"].get("steps")
        or loaded["gate2"].get("verdict") != "PASS"
    ):
        raise GenerationError("evidence.gate2 differs from the gate2 reading")
    gate3 = _require_mapping(evidence.get("gate3"), label="evidence.gate3")
    matrix = _require_mapping(loaded["gate3"].get("matrix"), label="gate3 matrix")
    counts = _require_mapping(loaded["gate3"].get("counts"), label="gate3 counts")
    if (
        gate3.get("cells") != matrix.get("cells")
        or gate3.get("visible_cpus") != matrix.get("visible_cpus")
        or gate3.get("mechanisms") != matrix.get("mechanisms")
        or gate3.get("topologies")
        != len(matrix.get("visible_cpus", [])) * len(matrix.get("mechanisms", []))
        or gate3.get("id_mismatch") != counts.get("id_mismatch_vs_full_core")
        or counts.get("ok") != matrix.get("cells")
        or loaded["gate3"].get("verdict") != "PASS"
    ):
        raise GenerationError("evidence.gate3 differs from the gate3 reading")
    gate4 = _require_mapping(evidence.get("gate4"), label="evidence.gate4")
    volume = _require_mapping(loaded["gate4"].get("volume"), label="gate4 volume")
    if (
        gate4.get("families") != len(loaded["gate4"].get("families", []))
        or gate4.get("splices") != volume.get("splice_steps")
        or volume.get("certified_splice_accepts") != volume.get("splice_steps")
        or gate4.get("edits") != volume.get("edits")
        or gate4.get("failing") != len(loaded["gate4"].get("failing", []))
        or loaded["gate4"].get("verdict") != "PASS"
    ):
        raise GenerationError("evidence.gate4 differs from the gate4 reading")
    manifest = _require_mapping(
        load_json(EVIDENCE_PATH), label="fastokens evidence manifest"
    )
    if manifest.get("evidence_id") != evidence.get("evidence_id"):
        raise GenerationError("the evidence manifest carries another evidence id")
    manifest_totals = _require_mapping(manifest.get("totals"), label="manifest totals")
    if (
        manifest_totals.get("docs") != evidence.get("docs_per_family")
        or manifest_totals.get("mismatches") != evidence.get("mismatch_guarded")
        or manifest_totals.get("routed") != evidence.get("routed_reference_per_family")
        or _require_mapping(manifest.get("run"), label="manifest run").get(
            "suite_version"
        )
        != evidence.get("suite_version")
    ):
        raise GenerationError("the evidence manifest disagrees with the binding")
    manifest_families = {
        (str(row.get("family")), str(row.get("artifact_sha256")))
        for row in manifest.get("artifacts", [])
        if isinstance(row, dict)
    }
    binding_families = {
        (str(row.get("family")), str(row.get("artifact_sha256")))
        for row in binding.get("families", [])
        if isinstance(row, dict)
    }
    if manifest_families != binding_families:
        raise GenerationError(
            "the evidence manifest lists other families than the binding"
        )


def _verify_packaging(binding: dict[str, Any]) -> None:
    source = _require_mapping(binding.get("source"), label="source")
    series = _require_list(source.get("patch_series"), label="source.patch_series")
    recipe = (REPOSITORY_ROOT / str(source.get("recipe"))).read_text(encoding="utf-8")
    for entry in series:
        entry = _require_mapping(entry, label="patch_series entry")
        _require_file_digest(entry.get("file"), entry.get("sha256"), label="patch")
        if str(entry.get("sha256")) not in recipe:
            raise GenerationError(f"the recipe does not check {entry.get('file')}")
    if (
        str(source.get("code_tree")) not in recipe
        or str(source.get("noticed_tree")) not in recipe
    ):
        raise GenerationError("the recipe does not check the recorded tree hashes")
    legal = _require_mapping(binding.get("legal"), label="legal")
    for path_key, digest_key in (
        ("license_path", "license_sha256"),
        ("notice_path", "notice_sha256"),
        ("sbom_path", "sbom_sha256"),
        ("license_bundle_path", "license_bundle_sha256"),
    ):
        _require_file_digest(legal.get(path_key), legal.get(digest_key), label=path_key)


def _verify_adapter(binding: dict[str, Any]) -> None:
    adapter = _require_mapping(binding.get("adapter"), label="adapter")
    _require_file_digest(
        adapter.get("path"), adapter.get("source_sha256"), label="adapter source"
    )
    guard = _require_mapping(binding.get("guard"), label="guard")
    if adapter.get("guard_set_sha256") != guard.get("set_sha256"):
        raise GenerationError("adapter.guard_set_sha256 differs from guard.set_sha256")
    source = (REPOSITORY_ROOT / str(adapter.get("path"))).read_text(encoding="utf-8")
    if f'"{adapter.get("config_id")}"' not in source:
        raise GenerationError("the adapter source does not carry the bound config id")
    if f'"{guard.get("id")}"' not in source:
        raise GenerationError("the adapter source does not name the bound guard id")


def verify_binding(registry: dict[str, Any], binding: dict[str, Any]) -> None:
    """Every check the node rests on; raises ``GenerationError`` on the first."""
    if binding.get("version") != "fastokens-binding-v1":
        raise GenerationError("unknown fastokens binding version")
    if (
        binding.get("backend") != "fastokens"
        or binding.get("admission") != "experimental"
    ):
        raise GenerationError(
            "the fastokens binding names the wrong backend or admission"
        )
    distribution = _require_mapping(binding.get("distribution"), label="distribution")
    recognised = _require_list(
        binding.get("recognised_distributions"), label="recognised_distributions"
    )
    upstream = _require_mapping(binding.get("upstream"), label="upstream")
    if (
        distribution.get("name") not in recognised
        or upstream.get("distribution") not in recognised
    ):
        raise GenerationError(
            "recognised_distributions must name the pinned and the upstream "
            "distribution"
        )
    engine_digest = _verify_wheels(binding)
    _verify_families(registry, binding)
    _verify_guard(binding, engine_digest)
    _verify_evidence(binding, engine_digest)
    _verify_packaging(binding)
    _verify_adapter(binding)


def node_from_binding(binding: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(binding[key]) for key in NODE_KEYS}


def augmented_document(
    registry: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    """Return ``registry`` with exactly the binding-owned node applied."""
    verify_binding(registry, binding)
    completed = copy.deepcopy(registry)
    distributions = completed.get("engine_distributions")
    if not isinstance(distributions, dict):
        distributions = {}
    distributions["fastokens"] = node_from_binding(binding)
    completed["engine_distributions"] = distributions
    return with_root_digest(completed, REGISTRY_DOMAIN_TAG)


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialise_document(document))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply or check the pinned Fastokens registry binding."
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)

    registry = _require_mapping(load_json(DEFAULT_REGISTRY), label="registry")
    binding = _require_mapping(load_json(BINDING_PATH), label="binding")
    generated = augmented_document(registry, binding)
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
                problems.append(f"{path} is not the generated fastokens registry")
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
