#!/usr/bin/env python3
"""Validate a toktier registry or evidence manifest and its root digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from toktier._jcs import CanonicalizationError, canonical_json  # noqa: E402

SCHEMAS: dict[str, tuple[Path, bytes]] = {
    "registry": (
        REPOSITORY_ROOT / "schemas" / "support_registry.schema.json",
        b"toktier.registry.v1\0",
    ),
    "evidence": (
        REPOSITORY_ROOT / "schemas" / "evidence_manifest.schema.json",
        b"toktier.evidence.v1\0",
    ),
}
REGISTRY_MARKERS = frozenset(
    {"oracles", "pipelines", "added_frontends", "compositions"}
)
EVIDENCE_MARKERS = frozenset({"evidence_id", "run", "corpora", "totals", "environment"})


class InputError(ValueError):
    """Raised when the input cannot be interpreted as a supported document."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise InputError(f"non-JSON numeric constant {value}")


def load_document(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def detect_schema(document: object) -> str:
    if not isinstance(document, dict):
        raise InputError("top-level JSON value must be an object")
    keys = set(document)
    registry_match = bool(keys & REGISTRY_MARKERS)
    evidence_match = bool(keys & EVIDENCE_MARKERS)
    if registry_match == evidence_match:
        raise InputError(
            "cannot unambiguously auto-detect schema from top-level fields"
        )
    return "registry" if registry_match else "evidence"


def load_schema(kind: str) -> dict[str, object]:
    loaded: object = json.loads(SCHEMAS[kind][0].read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError("schema root is not an object")
    return cast(dict[str, object], loaded)


def check_schema(document: object, kind: str) -> tuple[bool, str]:
    validator = Draft202012Validator(
        load_schema(kind), format_checker=FormatChecker()
    )
    error = next(validator.iter_errors(document), None)
    if error is None:
        return True, ""
    location = "$"
    for component in error.absolute_path:
        location += f"[{component}]" if isinstance(component, int) else f".{component}"
    return False, f"{location}: {error.message}"


def check_root_digest(document: object, kind: str) -> tuple[bool, str]:
    if not isinstance(document, dict):
        return False, "top-level JSON value must be an object"
    if "root_digest" not in document:
        return False, "root_digest member is missing"
    embedded = document["root_digest"]
    if not isinstance(embedded, str):
        return False, "root_digest member must be a string"
    digest_input = dict(document)
    del digest_input["root_digest"]
    try:
        canonical = canonical_json(digest_input)
    except CanonicalizationError as error:
        return False, str(error)
    expected = "sha256:" + hashlib.sha256(SCHEMAS[kind][1] + canonical).hexdigest()
    if embedded == expected:
        return True, ""
    return False, f"expected {expected}, found {embedded}"


def print_result(check: str, passed: bool, detail: str = "") -> None:
    suffix = f": {' '.join(detail.splitlines())}" if detail else ""
    print(f"{check}: {'PASS' if passed else 'FAIL'}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a toktier registry or evidence manifest."
    )
    parser.add_argument("json_file", type=Path, metavar="FILE")
    parser.add_argument("--schema", choices=("registry", "evidence"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        document = load_document(arguments.json_file)
        kind = arguments.schema or detect_schema(document)
    except (InputError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print_result("schema", False, str(error))
        print_result("root_digest", False, str(error))
        return 1

    schema_passed, schema_detail = check_schema(document, kind)
    digest_passed, digest_detail = check_root_digest(document, kind)
    print_result("schema", schema_passed, schema_detail)
    print_result("root_digest", digest_passed, digest_detail)
    return 0 if schema_passed and digest_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
