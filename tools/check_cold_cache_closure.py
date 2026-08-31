#!/usr/bin/env python3
# Standing cold-cache gate: the manifest closure rebuilds the loader face.
"""Simulate a fresh machine's first fetch from the shipped manifest.

Contract reference: ``docs/contracts/registry.md`` Section 4 (the
manifest pins the loader-face input closure). The certified loader face
is materialized on the frozen artifact directory's file group; on a
machine that has never fetched, the cache holds exactly what the
manifest pins. This gate builds such a cache -- with zero network: the
files come from a local directory holding the frozen artifact
directories, through the ordinary fetch-and-verify path -- and requires
the certified flagship family to load and serve the loader-face ids.

Four arms:

1. **closure fetch** -- a temporary home is populated through
   ``ArtifactStore.ensure`` over the shipped manifest with a local
   source; the resulting cache must hold exactly the pinned file group.
2. **cold load** -- ``load(family)`` succeeds offline on that cache;
   every configuration-side literal of the family's registry claim
   encodes to the same ids on the default policy, the reference policy,
   and the pinned loader over the same directory.
3. **repository spellings** -- ``from_pretrained`` on each named
   repository resolves offline (from the local hub cache; reported and
   skipped when the snapshot is absent), is admitted onto the anchor,
   and serves the anchor's ids.
4. **fail-closed remainder** -- a cache built from a one-file manifest
   (the 0.2.8 fetch shape) still fails closed with
   ``ARTIFACT_HASH_MISMATCH`` / ``config_added_tokens_mismatch``, and a
   cache whose pinned sidecar is removed fails closed on the missing
   file. The certification never silently degrades to the file-only
   face when the record declares configuration-side tokens.

The pytest suite deliberately never reads real artifacts, so this gate
lives here, beside ``check_loader_face_alignment.py``: it is meant for
the release gates and any machine holding the frozen artifact
directories. It refuses to conclude anything when those are absent
(exit code 3, distinct from a violation).

Usage::

    python tools/check_cold_cache_closure.py --artifact-root <dir>
    python tools/check_cold_cache_closure.py --artifact-root <dir> \
        --repo Qwen/Qwen3.8-27B --repo Qwen/Qwen3.8-Flash-Next
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

#: Absent-input exit code: nothing was checked, nothing failed.
ABSENT = 3

FAMILY = "qwen3_5_08b"

DEFAULT_REPOS = ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-Flash-Next")

CONTROL_PROBE = "plain text with no literals"


def _claim(family: str) -> dict[str, Any] | None:
    from toktier.routing.tables import SUPPORT_REGISTRY

    registry = json.loads(SUPPORT_REGISTRY.read_text(encoding="utf-8"))
    for record in registry["artifacts"]:
        if record.get("family") == family:
            claim = record.get("config_added_tokens")
            return claim if isinstance(claim, dict) else None
    return None


def _build_cache(
    home: Path, manifest: Any, family: str, artifact_root: Path
) -> Any:
    """Populate a cold home through the ordinary fetch-and-verify path."""
    from toktier import Config
    from toktier.artifacts.sources import LocalDirectorySource
    from toktier.artifacts.store import ArtifactStore

    # ``offline=False`` opens the store's generic gate only; the local
    # source has no network implementation, so nothing can reach out.
    store = ArtifactStore(
        manifest,
        config=Config(home=home, offline=False),
        source=LocalDirectorySource(root=artifact_root),
    )
    return store.ensure(family)


def check_closure_fetch(
    home: Path, family: str, artifact_root: Path
) -> tuple[int, dict[str, object]]:
    """Arm 1: the cold cache holds exactly the pinned file group."""
    from toktier.artifacts import ArtifactManifest
    from toktier.artifacts.bundle import VERIFIED_MARKER_NAME
    from toktier.artifacts.tables import ARTIFACT_MANIFEST

    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    entry = manifest.get(family)
    reading: dict[str, object] = {
        "arm": "closure_fetch",
        "family": family,
        "pinned": sorted(item.name for item in entry.files),
    }
    source_dir = artifact_root / entry.directory_name
    if not source_dir.is_dir():
        reading["skipped"] = f"no frozen directory at {source_dir}"
        return ABSENT, reading
    verified = _build_cache(home, manifest, family, artifact_root)
    cached = sorted(
        path.name
        for path in verified.directory.iterdir()
        if path.name != VERIFIED_MARKER_NAME
    )
    reading["cached"] = cached
    if cached != reading["pinned"]:
        reading["error"] = "cache contents differ from the pinned group"
        return 1, reading
    return 0, reading


def check_cold_load(home: Path, family: str) -> tuple[int, dict[str, object]]:
    """Arm 2: the cold cache serves the loader-face ids."""
    import transformers

    from toktier import Config, load
    from toktier.artifacts import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.backends.loader_face import config_added_token_rows
    from toktier.paths import artifact_cache_dir

    config = Config(home=home, offline=True)
    entry = ArtifactManifest.load(ARTIFACT_MANIFEST).get(family)
    root = artifact_cache_dir(config) / entry.directory_name
    rows = config_added_token_rows(root)
    claim = _claim(family)
    reading: dict[str, object] = {
        "arm": "cold_load",
        "family": family,
        "declared_count": None if claim is None else claim.get("count"),
        "observed_count": len(rows),
    }
    loader = transformers.AutoTokenizer.from_pretrained(
        str(root), use_fast=True, local_files_only=True
    )
    with_default = load(family, config=config)
    with_reference = load(family, config=config, policy="reference")
    literal_ids: dict[str, list[int]] = {}
    mismatches: list[dict[str, object]] = []
    try:
        probes = [str(row["content"]) for row in rows] + [CONTROL_PROBE]
        for text in probes:
            expected = loader.encode(text, add_special_tokens=False)
            observed_default = list(with_default.encode(text).ids)
            observed_reference = list(with_reference.encode(text).ids)
            if observed_default != expected or observed_reference != expected:
                mismatches.append(
                    {
                        "text": text,
                        "loader": expected,
                        "default": observed_default,
                        "reference": observed_reference,
                    }
                )
            elif text != CONTROL_PROBE:
                literal_ids[text] = observed_default
    finally:
        with_default.close()
        with_reference.close()
    reading["literal_ids"] = literal_ids
    reading["mismatches"] = mismatches
    if mismatches:
        return 1, reading
    if claim is not None and len(rows) != int(claim.get("count") or 0):
        reading["error"] = "observed subset differs from the registry claim"
        return 1, reading
    return 0, reading


def check_repository_spellings(
    home: Path, family: str, repos: list[str]
) -> tuple[int, dict[str, object]]:
    """Arm 3: the admitted repository spellings execute the anchor."""
    from toktier import Config, load
    from toktier import from_pretrained as toktier_from_pretrained
    from toktier.errors import ArtifactNotFound

    config = Config(home=home, offline=True)
    probe = "hello <|audio_start|> world"
    anchor = load(family, config=config)
    try:
        anchor_ids = list(anchor.encode(probe).ids)
    finally:
        anchor.close()
    reading: dict[str, object] = {
        "arm": "repository_spellings",
        "probe": probe,
        "anchor_ids": anchor_ids,
    }
    results: list[dict[str, object]] = []
    codes = [0]
    for repo in repos:
        row: dict[str, object] = {"repo": repo}
        try:
            tokenizer = toktier_from_pretrained(repo, config=config)
        except ArtifactNotFound as error:
            row["skipped"] = f"snapshot not in the local hub cache: {error}"
            codes.append(ABSENT)
            results.append(row)
            continue
        try:
            raw = tokenizer.explain().get("model_resolution")
            resolution: dict[str, Any] = raw if isinstance(raw, dict) else {}
            row["admitted"] = resolution.get("admitted")
            row["basis"] = resolution.get("basis")
            row["canonical_family"] = resolution.get("canonical_family")
            row["ids"] = list(tokenizer.encode(probe).ids)
        finally:
            tokenizer.close()
        if (
            row["admitted"] is not True
            or row["canonical_family"] != family
            or row["ids"] != anchor_ids
        ):
            row["error"] = "not admitted onto the anchor, or ids differ"
            codes.append(1)
        results.append(row)
    reading["repositories"] = results
    return max(codes), reading


def check_fail_closed(
    family: str, artifact_root: Path
) -> tuple[int, dict[str, object]]:
    """Arm 4: a cache missing the sidecar never certifies silently."""
    from toktier import Config, load
    from toktier.artifacts import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.backends.protocol import TOKENIZER_FILE
    from toktier.errors import ArtifactHashMismatch, ArtifactNotFound
    from toktier.paths import artifact_cache_dir

    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    entry = manifest.get(family)
    reading: dict[str, object] = {"arm": "fail_closed", "family": family}

    # One-file manifest: the fetch shape shipped through 0.2.8.
    trimmed_entry = dataclasses.replace(
        entry, files=(entry.file(TOKENIZER_FILE),)
    )
    trimmed = ArtifactManifest(
        entries={family: trimmed_entry}, sources=("cold-cache-gate",)
    )
    with tempfile.TemporaryDirectory(prefix="toktier-cold-a-") as scratch:
        home = Path(scratch)
        _build_cache(home, trimmed, family, artifact_root)
        try:
            load(family, config=Config(home=home, offline=True), manifest=trimmed)
        except ArtifactHashMismatch as error:
            details = error.details
            reading["one_file_manifest"] = {
                "outcome": "ARTIFACT_HASH_MISMATCH",
                "reason": details.get("reason"),
                "expected_count": details.get("expected_count"),
                "observed_count": details.get("observed_count"),
                "remedy": details.get("remedy"),
            }
            if details.get("reason") != "config_added_tokens_mismatch":
                reading["error"] = "unexpected mismatch reason"
                return 1, reading
        else:
            reading["error"] = (
                "a cache without the declared sidecar was accepted"
            )
            return 1, reading

    # Full manifest, sidecar removed after the fetch.
    claim = _claim(family)
    sidecar = None if claim is None else claim.get("source")
    if isinstance(sidecar, str):
        with tempfile.TemporaryDirectory(prefix="toktier-cold-b-") as scratch:
            home = Path(scratch)
            verified = _build_cache(home, manifest, family, artifact_root)
            (verified.directory / sidecar).unlink()
            config = Config(home=home, offline=True)
            root = artifact_cache_dir(config) / entry.directory_name
            try:
                load(family, config=config)
            except (ArtifactNotFound, ArtifactHashMismatch) as error:
                reading["removed_sidecar"] = {
                    "outcome": type(error).__name__,
                    "message": str(error)[:200],
                    "searched": str(root),
                }
            else:
                reading["error"] = "a cache missing a pinned file was accepted"
                return 1, reading
    return 0, reading


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cold-cache closure gate over real artifacts."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Directory holding the frozen artifact directories.",
    )
    parser.add_argument("--family", default=FAMILY)
    parser.add_argument(
        "--repo",
        action="append",
        default=None,
        help="Repository spelling to resolve (repeatable).",
    )
    parser.add_argument(
        "--hub-cache",
        type=Path,
        default=None,
        help=(
            "Hub cache to resolve repository snapshots from (sets HF_HOME "
            "for this run; the layout is the hub client's own, and the "
            "resolution stays local-files-only)."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.hub_cache is not None:
        # Before any hub client import, so the client resolves its cache
        # paths against this run's setting.
        os.environ["HF_HOME"] = str(arguments.hub_cache)
    repos = list(DEFAULT_REPOS) if arguments.repo is None else arguments.repo

    codes = []
    with tempfile.TemporaryDirectory(prefix="toktier-cold-") as scratch:
        home = Path(scratch)
        code, reading = check_closure_fetch(
            home, arguments.family, arguments.artifact_root
        )
        codes.append(code)
        print(json.dumps(reading, ensure_ascii=False))
        if code == 0:
            code, reading = check_cold_load(home, arguments.family)
            codes.append(code)
            print(json.dumps(reading, ensure_ascii=False))
            code, reading = check_repository_spellings(
                home, arguments.family, repos
            )
            codes.append(code)
            print(json.dumps(reading, ensure_ascii=False))
    if codes[0] == 0:
        code, reading = check_fail_closed(
            arguments.family, arguments.artifact_root
        )
        codes.append(code)
        print(json.dumps(reading, ensure_ascii=False))
    if 1 in codes:
        return 1
    if ABSENT in codes:
        return ABSENT
    return 0


if __name__ == "__main__":
    sys.exit(main())
