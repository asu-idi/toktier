#!/usr/bin/env python3
"""Record the set of packages the judged Rust API build actually compiles.

``crates/toktier/build_support/source_identity.rs`` compares this build's
resolved dependency graph against the one the certification campaign was
taken on. Until 0.2.4 the judged side of that comparison was the whole
lockfile closure, and a lockfile's dependency lists are the union over
every feature and every target: a Linux consumer was refused over a
WebAssembly binding that never entered the artifact. The comparison now
stands on the packages Cargo compiles for the judged build, which is what
this tool writes down.

Where the answer comes from, and why here rather than in the build
script: only this workspace can ask Cargo the question with the whole
manifest graph in hand, offline and reproducibly. A consumer's build
script cannot -- it would have to invoke Cargo inside a Cargo build,
against a workspace it cannot see from an unpacked registry copy. So the
set is taken once, at release time, and travels with the crate as data,
the way the judged lockfile already does.

``cargo tree`` rather than ``cargo metadata``: ``metadata``'s resolve
graph carries edges for optional dependencies that no feature enabled
(this workspace's ``faststr -> rkyv`` among them), which would enrol
fourteen packages nothing compiles. ``cargo tree`` prunes by the features
actually on.

The file carries names and versions only. Content hashes and origins stay
in the judged lockfile, so the two records cannot disagree about the same
package; the build script requires every name here to appear there.

Usage::

    python tools/generate_judged_closure.py
    python tools/generate_judged_closure.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "tables" / "support_registry.json"
OUTPUT = ROOT / "crates" / "toktier" / "data" / "build" / "judged_compiled_closure.json"
SCHEMA = "toktier.rust_compiled_closure.v1"
TOOL_NAME = "tools/generate_judged_closure.py"
ROOT_PACKAGE = "toktier"

#: `cargo tree` prints `name vVERSION` optionally followed by an origin and
#: the `(*)` marker it uses for a subtree it already printed.
PACKAGE_LINE = re.compile(r"^(?P<name>[^\s]+) v(?P<version>[^\s]+)")


class GenerationError(RuntimeError):
    """A condition that must stop generation rather than be written out."""


def judged_selections() -> list[dict[str, str]]:
    """The (target, features) pairs the shipped registry certifies.

    The compiled set is a function of exactly these two, and admission
    already requires both to match -- they are keys of ``build_flags`` --
    so a build that could be certified is a build one of these describes.
    """
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    selections: list[dict[str, str]] = []
    for row in document.get("runtime_builds", []):
        if row.get("runtime") != "rust_api":
            continue
        flags = {
            key: value
            for key, _, value in (
                flag.partition("=") for flag in row.get("build_flags", [])
            )
        }
        try:
            selection = {"target": flags["target"], "features": flags["features"]}
        except KeyError as error:
            raise GenerationError(
                f"{REGISTRY_PATH}: a rust_api row has no {error.args[0]} flag"
            ) from error
        if selection not in selections:
            selections.append(selection)
    if not selections:
        raise GenerationError(f"{REGISTRY_PATH} carries no rust_api runtime build")
    return selections


def compiled_packages(selection: dict[str, str]) -> set[tuple[str, str]]:
    """Every package Cargo compiles for one judged selection.

    ``-e normal,build`` is the whole of what enters the artifact: the
    linked crates, the proc macros whose expansion becomes source, and the
    build dependencies whose output is linked (``cc`` and what it pulls).
    Development dependencies are excluded because a consumer never builds
    them. The line between judged and not is "does Cargo compile it",
    which one command answers and anyone can re-run; "is the compiled code
    ever called" would need a cross-language call graph and could not be
    re-checked in a gate.
    """
    features = [
        feature
        for feature in selection["features"].split(",")
        # `default` is on unless it is turned off, and naming it here as
        # well would be the same set said twice.
        if feature and feature != "default"
    ]
    command = [
        "cargo",
        "tree",
        "--locked",
        "--offline",
        "--package",
        ROOT_PACKAGE,
        "--edges",
        "normal,build",
        "--target",
        selection["target"],
        "--prefix",
        "none",
        "--format",
        "{p}",
    ]
    if features:
        command += ["--features", ",".join(features)]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    if completed.returncode != 0:
        raise GenerationError(
            f"cargo tree failed for {selection}: {completed.stderr.strip()}"
        )
    packages: set[tuple[str, str]] = set()
    for line in completed.stdout.splitlines():
        match = PACKAGE_LINE.match(line.strip())
        if match is None:
            if line.strip():
                raise GenerationError(
                    f"cargo tree printed an unreadable line: {line!r}"
                )
            continue
        packages.add((match["name"], match["version"]))
    if not any(name == ROOT_PACKAGE for name, _ in packages):
        raise GenerationError(f"cargo tree did not report {ROOT_PACKAGE} itself")
    return packages


def build_document() -> dict[str, object]:
    selections = judged_selections()
    packages: set[tuple[str, str]] = set()
    for selection in selections:
        # The union over the judged selections. Where they differ, judging
        # the union is the conservative direction: an extra package can
        # only add a refusal, never remove one.
        packages |= compiled_packages(selection)
    return {
        "schema": SCHEMA,
        "generated_by": TOOL_NAME,
        "root": ROOT_PACKAGE,
        "selections": selections,
        "packages": [
            {"name": name, "version": version} for name, version in sorted(packages)
        ],
    }


def serialise(document: dict[str, object]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        rendered = serialise(build_document())
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if arguments.check:
        if not OUTPUT.is_file():
            print(f"error: {OUTPUT} is missing", file=sys.stderr)
            return 1
        if OUTPUT.read_text(encoding="utf-8") != rendered:
            print(
                f"error: {OUTPUT} is not what {TOOL_NAME} generates from this tree",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT}: check passed")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(json.loads(rendered)['packages'])} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
