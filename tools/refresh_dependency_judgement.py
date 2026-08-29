#!/usr/bin/env python3
"""Refresh lock-derived judgement data early in a release cycle.

Run this before the release certification battery so that the battery judges
the dependency resolution intended for the release. The write mode updates
the lockfile, populates Cargo's all-target cache, regenerates every record that
consumes the lockfile, and finishes by running the corresponding checks.

**The write mode's first step edits `Cargo.lock`, and by default it edits it
offline.** `cargo update --workspace --offline` re-resolves this workspace's
own members against the packages already in the local cache. It is the
narrow operation this tool exists for. An unrestricted `cargo update` is a
different operation: it reaches the network and moves every transitive
third-party version that has published since the lockfile was written. A
0.2.7 release wave met that difference the hard way -- twelve unrelated
packages were lifted, `Cargo.lock` went from a seven-line diff to a
thirty-seven-line one, and the version-normalised source identities moved
with it. Ask for it explicitly with `--allow-network-update` when a release
really is meant to take new upstream versions.

Usage::

    python3 tools/refresh_dependency_judgement.py
    python3 tools/refresh_dependency_judgement.py --allow-network-update
    python3 tools/refresh_dependency_judgement.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The fast CPU binding records the digest of the two legal artifacts it
#: names, and `tools/generate_native_legal.py` rewrites those artifacts from
#: the lockfile. Rebinding the digests here keeps a refresh self-consistent:
#: they were hand-held before, and a refresh that moved the files without
#: them left a mismatch for a later gate to find.
FAST_CPU_BINDING = ROOT / "tools/fast_cpu_binding.json"
LEGAL_KEYS = (
    ("sbom_path", "sbom_sha256"),
    ("license_bundle_path", "license_bundle_sha256"),
)

GENERATORS = (
    "tools/generate_judged_closure.py",
    "tools/generate_native_legal.py",
    "tools/generate_rust_distribution_metadata.py",
    "tools/update_rust_package_identity.py",
    "tools/sync_rust_package_data.py",
)


def run(command: list[str]) -> int:
    """Run one visible step from the repository root."""

    print(f"+ {shlex.join(command)}", flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def verification_commands() -> list[list[str]]:
    return [[sys.executable, generator, "--check"] for generator in GENERATORS]


def legal_digest_problems(*, rewrite: bool) -> list[str]:
    """Answer for, or restate, the two legal digests the binding records."""

    document = json.loads(FAST_CPU_BINDING.read_text(encoding="utf-8"))
    legal = document["legal"]
    problems: list[str] = []
    changed = False
    for path_key, digest_key in LEGAL_KEYS:
        observed = hashlib.sha256((ROOT / legal[path_key]).read_bytes()).hexdigest()
        if legal[digest_key] == observed:
            continue
        if rewrite:
            legal[digest_key] = observed
            changed = True
        else:
            problems.append(
                f"{FAST_CPU_BINDING}: {digest_key} records "
                f"{legal[digest_key]}, but {legal[path_key]} hashes to {observed}"
            )
    if changed:
        FAST_CPU_BINDING.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{FAST_CPU_BINDING}: updated", flush=True)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh lock-derived judgement data early in a release cycle."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the lock-derived records without updating or rewriting them",
    )
    parser.add_argument(
        "--allow-network-update",
        action="store_true",
        help=(
            "let the lock update reach the network and move transitive "
            "third-party versions, instead of re-resolving this workspace "
            "against the packages already cached. Say this only when the "
            "release is meant to take new upstream versions: it moves the "
            "source identities with them."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.check and arguments.allow_network_update:
        parser.error("--check writes nothing, so --allow-network-update means nothing")

    commands: list[list[str]] = []
    if not arguments.check:
        # Offline and workspace-scoped unless asked otherwise; see the
        # module docstring for what the unrestricted form does.
        update = (
            ["cargo", "update"]
            if arguments.allow_network_update
            else ["cargo", "update", "--workspace", "--offline"]
        )
        commands.extend((update, ["cargo", "fetch", "--locked"]))
        commands.extend([[sys.executable, generator] for generator in GENERATORS])
    commands.extend(verification_commands())

    for command in commands:
        returncode = run(command)
        if returncode != 0:
            print(
                f"error: {shlex.join(command)} exited with status {returncode}",
                file=sys.stderr,
            )
            return returncode
        if not arguments.check and command[-1].endswith("generate_native_legal.py"):
            legal_digest_problems(rewrite=True)
    problems = legal_digest_problems(rewrite=False)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"{FAST_CPU_BINDING}: legal digests check passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
