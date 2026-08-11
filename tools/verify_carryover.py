#!/usr/bin/env python3
"""Validate every add-only evidence carry-over record and its chain rules.

Records live below ``evidence/carryover/vMAJOR.MINOR/``.  The directory names
provide the minor-version scope without adding a field to the frozen
``evidence_carryover.v1`` record.  A carried-evidence pointer that binds all
three ``from_source_identity`` values is a real-campaign anchor and resets the
consecutive carry-over count.  Every represented minor needs such an anchor,
and no chain may exceed three consecutive carry-overs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover - exercised by the CLI environment
    jsonschema = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "evidence/carryover"
DEFAULT_SCHEMA = ROOT / "schemas/evidence_carryover.schema.json"
IDENTITY_KEYS = ("fast_cpu", "native_host", "rust_api")
ARTIFACT_NAMES = frozenset({"_native.abi3.so", "libtoktier.rlib"})
MINOR_DIRECTORY = re.compile(r"^v[0-9]+\.[0-9]+$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

Identity = tuple[str, str, str]


@dataclass(frozen=True)
class RecordInfo:
    """Validated fields needed to derive one graph edge."""

    path: Path
    minor: str
    source: Identity
    target: Identity
    campaign_anchor: bool
    document: dict[str, Any]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _location(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    return "/".join(parts) if parts else "<record>"


def schema_problems(document: Any, schema: Mapping[str, Any]) -> list[str]:
    """Return stable, path-qualified JSON Schema failures."""
    if jsonschema is None:
        return [
            "the jsonschema package is required; install the pinned test "
            "dependency group"
        ]
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    return [f"{_location(error)}: {error.message}" for error in errors]


def _identity(value: Any) -> Identity | None:
    if not isinstance(value, Mapping):
        return None
    values = tuple(value.get(key) for key in IDENTITY_KEYS)
    if not all(isinstance(item, str) and SHA256_HEX.fullmatch(item) for item in values):
        return None
    return values  # type: ignore[return-value]


def _decode_pointer(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")


def _pointer_fragment(document: Any, fragment: str) -> Any:
    if not fragment:
        return document
    if not fragment.startswith("/"):
        raise ValueError("fragment must be an RFC 6901 JSON pointer")
    current = document
    for raw in fragment[1:].split("/"):
        component = _decode_pointer(raw)
        if isinstance(current, Mapping):
            if component not in current:
                raise ValueError(f"JSON pointer component {component!r} is absent")
            current = current[component]
        elif isinstance(current, list):
            try:
                current = current[int(component)]
            except (ValueError, IndexError) as error:
                raise ValueError(
                    f"JSON pointer list index {component!r} is invalid"
                ) from error
        else:
            raise ValueError(f"JSON pointer cannot descend through {component!r}")
    return current


def _carried_fragment(pointer: str, repository_root: Path) -> Any:
    relative_value, separator, fragment = pointer.partition("#")
    relative = Path(relative_value)
    allowed = relative == Path("tables/support_registry.json") or (
        len(relative.parts) == 2
        and relative.parts[0] == "readings"
        and relative.suffix == ".json"
    )
    if not allowed or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("pointer must target the registry or a readings JSON file")
    if relative == Path("tables/support_registry.json") and not fragment:
        raise ValueError("a registry pointer must select one original record")
    path = repository_root / relative
    if not path.is_file():
        raise ValueError(f"pointed evidence file does not exist: {relative}")
    return _pointer_fragment(_json(path), fragment if separator else "")


def _strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        found: set[str] = set()
        for item in value.values():
            found.update(_strings(item))
        return found
    if isinstance(value, list):
        found = set()
        for item in value:
            found.update(_strings(item))
        return found
    return set()


def _validate_artifact_recipe(
    witness: Mapping[str, Any], path: Path
) -> list[str]:
    problems: list[str] = []
    artifacts_value = witness.get("sentinel_artifacts")
    if not isinstance(artifacts_value, list):
        return [f"{path}: artifact_equivalence lacks sentinel_artifacts"]
    artifacts = {
        item.get("artifact"): item
        for item in artifacts_value
        if isinstance(item, Mapping) and isinstance(item.get("artifact"), str)
    }
    if set(artifacts) != ARTIFACT_NAMES:
        problems.append(
            f"{path}: sentinel artifact set must be exactly "
            "_native.abi3.so and libtoktier.rlib"
        )
    for name in sorted(ARTIFACT_NAMES):
        item = artifacts.get(name)
        if item is None:
            continue
        digest = item.get("sha256_both")
        size = item.get("bytes")
        if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
            problems.append(f"{path}: {name} lacks a valid sha256_both")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            problems.append(f"{path}: {name} lacks a positive byte size")
        if item.get("byte_equal") is False:
            problems.append(f"{path}: {name} is recorded as byte-unequal")

    recipe = witness.get("recipe")
    if not isinstance(recipe, Mapping):
        return [*problems, f"{path}: artifact_equivalence lacks a recipe"]
    required = {
        "tree_path",
        "cargo_target_dir",
        "rustflags",
        "cargo_home_roots",
        "locked",
        "fresh_target_for_each_tree",
        "sequential_same_path_builds",
        "commands",
        "toolchain",
        "ambient_environment",
        "effective_environment",
        "host_fingerprint",
        "same_host_only",
    }
    missing = sorted(required - set(recipe))
    if missing:
        problems.append(f"{path}: recipe lacks {', '.join(missing)}")
        return problems
    tree_path = recipe.get("tree_path")
    target = recipe.get("cargo_target_dir")
    rustflags = recipe.get("rustflags")
    roots = recipe.get("cargo_home_roots")
    if not isinstance(tree_path, str) or not Path(tree_path).is_absolute():
        problems.append(f"{path}: recipe tree_path must be absolute")
    if not isinstance(target, str) or not Path(target).is_absolute():
        problems.append(f"{path}: recipe cargo_target_dir must be absolute")
    if not isinstance(rustflags, str) or not isinstance(tree_path, str):
        problems.append(f"{path}: recipe rustflags must be one string")
    else:
        source_remap = f"--remap-path-prefix={tree_path}=/toktier"
        if source_remap not in rustflags.split():
            problems.append(f"{path}: rustflags lacks the canonical tree remap")
        if "\n" in rustflags:
            problems.append(f"{path}: rustflags must be one line")
    if not isinstance(roots, list) or not roots:
        problems.append(f"{path}: recipe cargo_home_roots must be nonempty")
    elif isinstance(rustflags, str):
        flags = rustflags.split()
        for root in roots:
            expected = f"--remap-path-prefix={root}=/cargo"
            if not isinstance(root, str) or expected not in flags:
                problems.append(f"{path}: rustflags lacks Cargo root remap {root}")

    commands = recipe.get("commands")
    purposes: set[str] = set()
    if isinstance(commands, list):
        for command in commands:
            if not isinstance(command, Mapping):
                continue
            purpose = command.get("purpose")
            argv = command.get("argv")
            cwd = command.get("cwd")
            if isinstance(purpose, str):
                purposes.add(purpose)
            if not isinstance(argv, list) or "--locked" not in argv:
                problems.append(f"{path}: every artifact build command needs --locked")
            if cwd != tree_path:
                problems.append(f"{path}: build command cwd must equal tree_path")
    expected_purposes = {
        "sentinel_native_extension",
        "sentinel_whole_rust_api_rlib",
    }
    if purposes != expected_purposes:
        problems.append(
            f"{path}: recipe must carry native-extension and whole-rlib commands"
        )
    effective = recipe.get("effective_environment")
    if not isinstance(effective, Mapping):
        problems.append(f"{path}: recipe effective_environment is invalid")
    else:
        expected_environment: dict[str, Any] = {
            "CARGO_TARGET_DIR": target,
            "RUSTFLAGS": rustflags,
            "TOKTIER_IDENTITY_SENTINEL": "1",
        }
        for key, expected in expected_environment.items():
            if effective.get(key) != expected:
                problems.append(f"{path}: effective {key} does not match recipe")
    for boolean in (
        "locked",
        "fresh_target_for_each_tree",
        "sequential_same_path_builds",
        "same_host_only",
    ):
        if recipe.get(boolean) is not True:
            problems.append(f"{path}: recipe {boolean} must be true")
    if recipe.get("cargo_configs") not in (None, []):
        problems.append(f"{path}: v1 recipe does not accept ambient Cargo config")
    return problems


def _validate_v2_recipe(witness: Mapping[str, Any], path: Path) -> list[str]:
    problems: list[str] = []
    code_identity = witness.get("code_identity_v2")
    if not isinstance(code_identity, Mapping):
        return [f"{path}: code_identity_v2 mechanism lacks its v2 witness"]
    value = code_identity.get("value_both")
    if isinstance(value, Mapping):
        if _identity(value) is None:
            problems.append(f"{path}: code_identity_v2 value_both is invalid")
    elif not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
        problems.append(f"{path}: code_identity_v2 value_both is invalid")
    if not isinstance(code_identity.get("normalization_diff"), str) or not (
        code_identity.get("normalization_diff")
    ):
        problems.append(f"{path}: code_identity_v2 normalization_diff is empty")
    recipe = witness.get("recipe")
    commands = recipe.get("commands") if isinstance(recipe, Mapping) else None
    argv_values = [
        " ".join(command.get("argv", []))
        for command in commands or []
        if isinstance(command, Mapping) and isinstance(command.get("argv"), list)
    ]
    if not any("tools/compute_identity_v2.py" in argv for argv in argv_values):
        problems.append(f"{path}: v2 recipe lacks the identity-v2 command line")
    return problems


def record_problems(
    document: Any,
    path: Path,
    schema: Mapping[str, Any],
    repository_root: Path,
) -> tuple[list[str], bool]:
    """Validate one record and report whether its source is campaign-bound."""
    problems = [f"{path}: {problem}" for problem in schema_problems(document, schema)]
    if problems or not isinstance(document, dict):
        return problems, False
    source = _identity(document.get("from_source_identity"))
    target = _identity(document.get("to_source_identity"))
    if source is None or target is None:
        return [*problems, f"{path}: source identities are invalid"], False
    if source == target:
        problems.append(f"{path}: from/to source identity sets must differ")
    witness = document.get("witness")
    applicability = (
        witness.get("applicability")
        if isinstance(witness, Mapping)
        else None
    )
    if not isinstance(applicability, Mapping):
        problems.append(f"{path}: witness applicability is absent")
    else:
        diff_files = applicability.get("diff_files")
        if not isinstance(diff_files, list) or not diff_files:
            problems.append(f"{path}: witness diff_files must be nonempty")
    mechanism = document.get("mechanism")
    if isinstance(witness, Mapping) and mechanism == "artifact_equivalence":
        if not isinstance(applicability, Mapping) or (
            applicability.get("cargo_lock_unchanged") is not True
        ):
            problems.append(
                f"{path}: artifact witness does not bind an unchanged Cargo.lock"
            )
        if not isinstance(applicability, Mapping) or (
            applicability.get("protected_files_unchanged") is not True
        ):
            problems.append(f"{path}: artifact witness lacks protected-file equality")
        problems.extend(_validate_artifact_recipe(witness, path))
    elif isinstance(witness, Mapping) and mechanism == "code_identity_v2":
        problems.extend(_validate_v2_recipe(witness, path))

    fragments: list[Any] = []
    carried = document.get("carried_evidence")
    if isinstance(carried, list):
        for pointer in carried:
            if not isinstance(pointer, str):
                continue
            try:
                fragments.append(_carried_fragment(pointer, repository_root))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                problems.append(
                    f"{path}: invalid carried-evidence pointer {pointer!r}: {error}"
                )
    source_values = set(source)
    campaign_anchor = any(source_values <= _strings(fragment) for fragment in fragments)
    return problems, campaign_anchor


def _load_records(
    records_directory: Path,
    schema: Mapping[str, Any],
    repository_root: Path,
) -> tuple[list[RecordInfo], list[str]]:
    records: list[RecordInfo] = []
    problems: list[str] = []
    if not records_directory.is_dir():
        return records, [f"{records_directory}: records directory does not exist"]
    for path in sorted(records_directory.rglob("*.json")):
        relative = path.relative_to(records_directory)
        minor = relative.parts[0] if len(relative.parts) > 1 else ""
        if not MINOR_DIRECTORY.fullmatch(minor):
            problems.append(
                f"{path}: record must be below a vMAJOR.MINOR directory"
            )
            continue
        try:
            document = _json(path)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{path}: invalid JSON: {error}")
            continue
        record_errors, campaign_anchor = record_problems(
            document, path, schema, repository_root
        )
        problems.extend(record_errors)
        if record_errors or not isinstance(document, dict):
            continue
        source = _identity(document["from_source_identity"])
        target = _identity(document["to_source_identity"])
        if source is None or target is None:
            continue
        records.append(
            RecordInfo(path, minor, source, target, campaign_anchor, document)
        )
    return records, problems


def _chain_problems(records: list[RecordInfo]) -> list[str]:
    problems: list[str] = []
    by_target: dict[Identity, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_target[record.target].append(index)
    represented_minors = sorted({record.minor for record in records})
    for minor in represented_minors:
        if not any(
            record.minor == minor and record.campaign_anchor for record in records
        ):
            problems.append(
                f"{minor}: no real-campaign anchor; every represented minor "
                "version needs at least one real campaign"
            )

    memo: dict[int, int] = {}
    visiting: set[int] = set()
    cycle_reported: set[int] = set()

    def depth(index: int) -> int:
        if index in memo:
            return memo[index]
        record = records[index]
        if record.campaign_anchor:
            memo[index] = 1
            return 1
        if index in visiting:
            if index not in cycle_reported:
                problems.append(f"{record.path}: carry-over chain contains a cycle")
                cycle_reported.add(index)
            return 4
        predecessors = by_target.get(record.source, [])
        if not predecessors:
            problems.append(
                f"{record.path}: chain has no preceding carry-over or "
                "real-campaign anchor"
            )
            memo[index] = 1
            return 1
        visiting.add(index)
        value = 1 + max(depth(predecessor) for predecessor in predecessors)
        visiting.remove(index)
        memo[index] = value
        return value

    for index, record in enumerate(records):
        value = depth(index)
        if value > 3:
            problems.append(
                f"{record.path}: consecutive carry-over depth {value} exceeds 3"
            )
    return problems


def check_records(
    records_directory: Path = DEFAULT_RECORDS,
    schema_path: Path = DEFAULT_SCHEMA,
    repository_root: Path = ROOT,
) -> tuple[list[str], int]:
    """Return all schema, consistency, pointer, and chain-rule failures."""
    try:
        schema = _json(schema_path)
    except (OSError, json.JSONDecodeError) as error:
        return [f"{schema_path}: invalid schema: {error}"], 0
    if not isinstance(schema, Mapping):
        return [f"{schema_path}: schema root must be an object"], 0
    records, problems = _load_records(records_directory, schema, repository_root)
    if not problems:
        problems.extend(_chain_problems(records))
    return problems, len(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.check:
        print("error: this verifier requires --check", file=sys.stderr)
        return 2
    problems, count = check_records(
        arguments.records_dir,
        arguments.schema,
        arguments.repository_root,
    )
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"{arguments.records_dir}: check passed ({count} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
