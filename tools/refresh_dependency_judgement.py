#!/usr/bin/env python3
"""Refresh lock-derived judgement data early in a release cycle.

Run this before the release certification battery so that the battery judges
the dependency resolution intended for the release. The default mode updates
the lockfile, populates Cargo's all-target cache, regenerates every record that
consumes the lockfile, and finishes by running the corresponding checks.

Usage::

    python3 tools/refresh_dependency_judgement.py
    python3 tools/refresh_dependency_judgement.py --check
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the lock-derived records without updating or rewriting them",
    )
    arguments = parser.parse_args()

    commands: list[list[str]] = []
    if not arguments.check:
        commands.extend((["cargo", "update"], ["cargo", "fetch", "--locked"]))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
