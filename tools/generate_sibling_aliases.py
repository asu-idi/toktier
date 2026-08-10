#!/usr/bin/env python3
"""Generate and verify the packaged verified-sibling registry.

The public source projection in ``data/sibling_aliases.v1.json`` is produced
from the archived 470-repository audit by maintainer tooling.  This product-side
generator validates that projection, binds each canonical anchor to the shipped
artifact manifest, adds package-availability facts, and writes the
domain-separated, root-digested table consumed at runtime.

Nothing here grants a new certificate: an alias only points at an already
certified canonical artifact, and the runtime still applies every ordinary
artifact, oracle, CPU-engine, GPU-delivery, device, and input guard.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from registry_common import (
    SIBLING_ALIAS_DOMAIN_TAG,
    GenerationError,
    check_regenerated,
    git_commit,
    load_json,
    schema_violations,
    serialise_document,
    write_document,
)

TOOL_NAME = "tools/generate_sibling_aliases.py"
TOOL_VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "sibling_aliases.v1.json"
SOURCE_SCHEMA = ROOT / "schemas" / "sibling_alias_source.schema.json"
OUTPUT_SCHEMA = ROOT / "schemas" / "sibling_aliases.schema.json"
ARTIFACT_MANIFEST = (
    ROOT / "src" / "toktier" / "artifacts" / "tables" / "artifact_manifest.v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "src" / "toktier" / "artifacts" / "tables" / "sibling_aliases.v1.json"
)


def _load_schema(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise GenerationError(f"{path}: schema must be an object")
    return value


def _load_source(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise GenerationError(f"{path}: top-level value must be an object")
    violations = schema_violations(value, _load_schema(SOURCE_SCHEMA))
    if violations:
        raise GenerationError(
            f"{path} does not satisfy {SOURCE_SCHEMA.name}:\n  "
            + "\n  ".join(violations)
        )
    if path.read_bytes() != serialise_document(value):
        raise GenerationError(
            f"{path}: source is not deterministically serialized; regenerate it"
        )
    return value


def _manifest_anchors() -> dict[str, str]:
    """Return the tokenizer digest of each family the wheel can load."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from toktier.artifacts import ArtifactManifest
    from toktier.backends.protocol import TOKENIZER_FILE

    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    return {
        family: entry.file(TOKENIZER_FILE).sha256
        for family, entry in manifest.entries.items()
    }


def build_document(
    source_path: Path,
    *,
    source_commit: str,
    generated_at: str,
) -> dict[str, Any]:
    source = _load_source(source_path)
    anchors = _manifest_anchors()
    raw_aliases = source["aliases"]
    if not isinstance(raw_aliases, list):  # already schema-checked; narrows type
        raise GenerationError(f"{source_path}: aliases must be an array")

    aliases: list[dict[str, object]] = []
    repo_ids: set[str] = set()
    digest_targets: dict[str, tuple[str, str]] = {}
    for raw in raw_aliases:
        if not isinstance(raw, dict):
            raise GenerationError(f"{source_path}: alias rows must be objects")
        row = dict(raw)
        repo_id = str(row["repo_id"])
        if repo_id in repo_ids:
            raise GenerationError(f"{source_path}: duplicate repo_id {repo_id!r}")
        repo_ids.add(repo_id)
        family = str(row["canonical_family"])
        anchor = str(row["canonical_anchor_sha256"])
        source_digest = str(row["source_sha256"])
        target = (family, anchor)
        previous = digest_targets.setdefault(source_digest, target)
        if previous != target:
            raise GenerationError(
                f"{source_path}: source digest {source_digest} maps to both "
                f"{previous} and {target}"
            )
        packaged_digest = anchors.get(family)
        packaged = packaged_digest is not None
        if packaged and packaged_digest != anchor:
            raise GenerationError(
                f"{repo_id}: alias anchor {anchor} disagrees with the shipped "
                f"{family} artifact {packaged_digest}"
            )
        row["canonical_packaged"] = packaged
        aliases.append(row)

    aliases.sort(key=lambda row: str(row["repo_id"]).casefold())
    basis_counts = Counter(str(row["basis"]) for row in aliases)
    packaged_count = sum(bool(row["canonical_packaged"]) for row in aliases)
    equivalent = [
        row for row in aliases if str(row["basis"]).startswith("equivalent_")
    ]
    equivalent_packaged = sum(
        bool(row["canonical_packaged"]) for row in equivalent
    )
    if len(equivalent) != 48 or equivalent_packaged != 46:
        raise GenerationError(
            "equivalent sibling coverage drifted: expected 48 total / 46 "
            f"packaged, got {len(equivalent)} / {equivalent_packaged}"
        )
    counts = {
        "identical": basis_counts["identical"],
        "identical_source": basis_counts["identical_source"],
        "equivalent_canonicalisation": basis_counts[
            "equivalent_canonicalisation"
        ],
        "equivalent_serialisation": basis_counts["equivalent_serialisation"],
        "total": len(aliases),
        "packaged": packaged_count,
        "reference_only": len(aliases) - packaged_count,
    }
    return {
        "schema_version": 1,
        "generated_by": {
            "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "source_commit": source_commit,
            "generated_at": generated_at,
        },
        "root_digest": "",
        "provenance": source["provenance"],
        "counts": counts,
        "aliases": aliases,
    }


def _default_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    source_commit = arguments.source_commit or git_commit(ROOT) or "0000000"
    generated_at = arguments.generated_at or _default_timestamp()
    document = build_document(
        arguments.source,
        source_commit=source_commit,
        generated_at=generated_at,
    )
    schema = _load_schema(OUTPUT_SCHEMA)
    if arguments.check:
        problems = check_regenerated(
            arguments.out, document, schema, SIBLING_ALIAS_DOMAIN_TAG
        )
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"{arguments.out}: check passed")
        return 0
    completed = write_document(
        arguments.out, document, schema, SIBLING_ALIAS_DOMAIN_TAG
    )
    print(
        f"wrote {arguments.out} ({completed['counts']['total']} aliases, "
        f"root {completed['root_digest']})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
