#!/usr/bin/env python3
"""Record the set of packages the judged Rust API build actually compiles,
and which of them the certificate speaks for.

``crates/toktier/build_support/source_identity.rs`` compares this build's
resolved dependency graph against the one the certification campaign was
taken on. Until 0.2.4 the judged side of that comparison was the whole
lockfile closure, and a lockfile's dependency lists are the union over
every feature and every target: a Linux consumer was refused over a
WebAssembly binding that never entered the artifact. The comparison now
stands on the packages Cargo compiles for the judged build, which is what
this tool writes down.

Since 0.2.6 each package also carries the tier that decides what it can
do to the certificate. The certified core -- TokTier's own crates, the
packages they call directly, and the text-semantics libraries beneath
them -- decides `certified`; everything else the build compiles is
compared as before and reported as an advisory. The three rules that
draw that line are mechanical, so anyone can re-derive the split from
this repository:

- **R0**: a workspace member of this repository.
- **R1**: a non-development direct dependency of one of those members
  that is also named from an encode-path source file. Every one of them
  is pinned exactly by at least one of our own edges, which is what
  ``--check`` requires.
- **R2**: a text-semantics library (regex engine; Unicode property,
  normalization or segmentation data and algorithms; SentencePiece
  normalization) reachable through normal edges from an engine crate.
  These are the packages whose correct behaviour is defined by an
  evolving external standard, so a version of one of them really can
  change ids by design rather than by fault.

The classification of R2 is data -- ``TEXT_SEMANTICS_TABLE`` -- and its
completeness is a gate: any package in the closure whose name matches the
text-semantics name net has to be classified explicitly, with a reason,
or generation stops.

Where the answer comes from, and why here rather than in the build
script: only this workspace can ask Cargo the question with the whole
manifest graph in hand, offline and reproducibly. A consumer's build
script cannot -- it would have to invoke Cargo inside a Cargo build,
against a workspace it cannot see from an unpacked registry copy. So the
set is taken once, at release time, and travels with the crate as data,
the way the judged lockfile already does.

``cargo tree`` rather than ``cargo metadata`` for the compiled set:
``metadata``'s resolve graph carries edges for optional dependencies that
no feature enabled (this workspace's ``faststr -> rkyv`` among them),
which would enrol fourteen packages nothing compiles. ``cargo tree``
prunes by the features actually on. The edges and their kinds, which
``cargo tree`` does not report, come from ``cargo metadata``'s resolve
graph, read by package id rather than by name: connecting by name would
attach gigatoken-core's disabled optional ``rand`` to the ``rand`` that
``tokenizers`` brings in.

The file carries names, versions, tiers and behaviour versions only.
Content hashes and origins stay in the judged lockfile, so the two
records cannot disagree about the same package; the build script
requires every name here to appear there.

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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

#: Cargo's own JSON, read loosely: this tool asks a handful of keys of it
#: and states what it expects of each one where it reads it.
Node = Any
Metadata = Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "tables" / "support_registry.json"
OUTPUT = ROOT / "crates" / "toktier" / "data" / "build" / "judged_compiled_closure.json"
SCHEMA = "toktier.rust_compiled_closure.v2"
TOOL_NAME = "tools/generate_judged_closure.py"
ROOT_PACKAGE = "toktier"

#: `cargo tree` prints `name vVERSION` optionally followed by an origin and
#: the `(*)` marker it uses for a subtree it already printed.
PACKAGE_LINE = re.compile(r"^(?P<name>[^\s]+) v(?P<version>[^\s]+)")

#: The crates whose transitive normal edges reach the code that turns text
#: into ids. Reachability from these, rather than from the facade, is what
#: keeps a package that only the artifact lifecycle touches out of R2.
ENGINE_CRATES = (
    "toktier-gigatoken-core",
    "toktier-routing-core",
    "toktier-cuda-driver",
)

#: Files of `crates/toktier/src` that are not on the encode path: artifact
#: acquisition and packaging, the filesystem helpers they use, the serving
#: pool, the suggestion helper, the error and diagnostic records, the local
#: verification command, and the CLI. A direct dependency named only from
#: these is reported rather than certified, which is why `fs2` and `tar`
#: are periphery. The verification command drives the encode path rather
#: than being on it: it is a diagnostic a person runs, and a package only
#: it names is not part of what answers a request.
FACADE_LIFECYCLE_SOURCES = (
    "artifact.rs",
    "bundle.rs",
    "diagnostics.rs",
    "error.rs",
    "fsutil.rs",
    "package_data.rs",
    "serving.rs",
    "suggest.rs",
    "verify_local.rs",
)

#: Files of `crates/toktier/src` that are on the encode path. Listing them
#: rather than deriving them by subtraction is the point: a new file in
#: this directory has to be classified by a person, and `--check` says so
#: until it is.
FACADE_ENCODE_SOURCES = (
    "behavior_version.rs",
    "buffer.rs",
    "gpu_data.rs",
    "jit.rs",
    "lib.rs",
    "manifest.rs",
    "runtime.rs",
    "session.rs",
)

#: Any package in the closure whose name matches this is a candidate for
#: carrying Unicode or regex semantics, so it has to be classified by hand
#: rather than fall into the periphery by default.
NAME_NET = re.compile(
    r"^(regex|fancy-regex|onig|pcre|unicode[-_]|icu_|ucd|spm_"
    r"|.*normaliz|.*segment|.*graphem|tokenizers)"
)

#: Which packages carry text semantics, and why. `core` here means R2 when
#: the package is also reachable from an engine crate, which `--check`
#: requires; `periphery` records a name-net match that was looked at and
#: found to carry no Unicode or regex knowledge of its own.
TEXT_SEMANTICS_TABLE: dict[str, dict[str, str]] = {
    "regex": {
        "tier": "core",
        "reason": (
            "the regex crate the reference engine uses for added-token "
            "boundaries and whitespace stripping; its Unicode class "
            "semantics decide where those splits fall"
        ),
    },
    "regex-automata": {
        "tier": "core",
        "reason": "the engine behind regex",
    },
    "regex-syntax": {
        "tier": "core",
        "reason": "the parser and Unicode class tables behind regex",
    },
    "onig": {
        "tier": "core",
        "reason": (
            "the Oniguruma binding the reference engine uses for every "
            "byte-level BPE family's split pattern"
        ),
    },
    "onig_sys": {
        "tier": "core",
        "reason": (
            "the Oniguruma C library behind onig; its built-in Unicode "
            "tables define the classes those patterns name"
        ),
    },
    "unicode-segmentation": {
        "tier": "core",
        "reason": (
            "grapheme and word segmentation in the reference engine's "
            "BERT pre-tokenizer, strip normalizer and normalizer core; "
            "the algorithm moves with the Unicode version"
        ),
    },
    "unicode_categories": {
        "tier": "core",
        "reason": (
            "general-category tests in the reference engine's punctuation "
            "pre-tokenizer and BERT normalizer"
        ),
    },
    "icu_properties_data": {
        "tier": "core",
        "reason": (
            "the Unicode property tables behind icu_properties, which are "
            "the fast CPU pre-tokenizer's character classes"
        ),
    },
    "icu_collections": {
        "tier": "periphery",
        "reason": (
            "code point set and trie containers; they carry no Unicode "
            "knowledge of their own"
        ),
    },
    "icu_locale_core": {
        "tier": "periphery",
        "reason": "locale identifier plumbing for the compiled-data provider",
    },
    "icu_provider": {
        "tier": "periphery",
        "reason": "data provider plumbing for the compiled-data provider",
    },
    "unicode-ident": {
        "tier": "periphery",
        "reason": (
            "the identifier tables proc-macro2 reads while macros expand; "
            "not linked into the running artifact"
        ),
    },
    "unicode-width": {
        "tier": "periphery",
        "reason": (
            "display width for the progress bars of indicatif and console; "
            "not on the encode path"
        ),
    },
}

#: The behaviour unit each R2 package belongs to, and where that unit's
#: behaviour version is read from. A unit is the thing whose version
#: decides ids: the Unicode tables a regex engine carries, the Oniguruma
#: library a binding links, the segmentation tables a crate ships. Package
#: versions of these are compared too, but in the advisory rather than in
#: the certificate (`docs/rust-api.md`).
BEHAVIOUR_UNITS: dict[str, str] = {
    "regex": "regex",
    "regex-automata": "regex",
    "regex-syntax": "regex",
    "onig": "onig",
    "onig_sys": "onig",
    "unicode-segmentation": "unicode-segmentation",
    "unicode_categories": "unicode_categories",
    "icu_properties_data": "icu_properties_data",
}

#: Where the version of each behaviour unit is read at generation time,
#: and what the runtime probe of the same unit reads. A unit with no
#: readable behaviour version falls back to its crate version, which is
#: compared exactly, so the fallback is the strict direction.
BEHAVIOUR_SOURCES: dict[str, str] = {
    "regex": "probe:regex-age",
    "onig": "runtime:onig::version",
    "unicode-segmentation": "const:UNICODE_VERSION",
    "unicode_categories": "crate-version",
    "icu_properties_data": "crate-version",
}

#: The package whose vendored sources answer for each unit at generation
#: time. The runtime probe asks the linked library the same question.
BEHAVIOUR_WITNESS: dict[str, str] = {
    "regex": "regex-syntax",
    "onig": "onig_sys",
    "unicode-segmentation": "unicode-segmentation",
}

TIER_STATEMENT = (
    "The certified core is R0 the workspace members, R1 their non-development "
    "direct dependencies that encode-path sources name, and R2 the "
    "text-semantics libraries reachable from an engine crate through normal "
    "edges. Everything else this build compiles is periphery: it is compared "
    "and reported, and it does not decide certification. R0 and R1 are pinned "
    "exactly by our own edges, so they cannot move by version; R2 is compared "
    "by behaviour version, and its package versions are reported in the "
    "advisory."
)


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


def selected_features(selection: dict[str, str]) -> list[str]:
    return [
        feature
        for feature in selection["features"].split(",")
        # `default` is on unless it is turned off, and naming it here as
        # well would be the same set said twice.
        if feature and feature != "default"
    ]


def run_cargo(command: list[str]) -> str:
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
            f"{' '.join(command)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


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
    features = selected_features(selection)
    if features:
        command += ["--features", ",".join(features)]
    packages: set[tuple[str, str]] = set()
    for line in run_cargo(command).splitlines():
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


def resolve_graph(selection: dict[str, str]) -> Metadata:
    """Cargo's resolve graph for one judged selection.

    ``--filter-platform`` keeps the edges of other targets out, and the
    graph is read by package id: a manifest-name join would connect
    gigatoken-core's optional ``rand``, which no feature enables, to the
    ``rand`` that ``tokenizers`` resolves.
    """
    command = [
        "cargo",
        "metadata",
        "--format-version",
        "1",
        "--locked",
        "--offline",
        "--filter-platform",
        selection["target"],
    ]
    features = selected_features(selection)
    if features:
        command += ["--features", ",".join(features)]
    parsed: Metadata = json.loads(run_cargo(command))
    return parsed


class ResolveGraph(Protocol):
    """What the tier rules ask of a resolve graph.

    Named as an interface so the rules can be exercised against a graph
    small enough to state in a test, rather than only against whatever
    this workspace happens to resolve today.
    """

    workspace_members: set[str]
    nodes: dict[str, Node]

    def name(self, identifier: str) -> str: ...

    def version(self, identifier: str) -> str: ...

    def is_proc_macro(self, identifier: str) -> bool: ...

    def id_of(self, name: str) -> str | None: ...

    def edges(self, identifier: str, kinds: set[str | None]) -> list[str]: ...

    def reachable(self, roots: list[str], kinds: set[str | None]) -> set[str]: ...

    def own_requirements(self, name: str) -> list[str]: ...


class Graph:
    """The parts of one resolve graph this tool asks about."""

    def __init__(self, metadata: Metadata) -> None:
        self.packages: dict[str, Node] = {
            str(package["id"]): package for package in metadata["packages"]
        }
        self.workspace_members = {
            str(member) for member in metadata["workspace_members"]
        }
        self.nodes: dict[str, Node] = {
            str(node["id"]): node for node in metadata["resolve"]["nodes"]
        }

    def name(self, identifier: str) -> str:
        return str(self.packages[identifier]["name"])

    def version(self, identifier: str) -> str:
        return str(self.packages[identifier]["version"])

    def is_proc_macro(self, identifier: str) -> bool:
        return any(
            "proc-macro" in target.get("kind", [])
            for target in self.packages[identifier].get("targets", [])
        )

    def manifest_dir(self, identifier: str) -> Path:
        return Path(str(self.packages[identifier]["manifest_path"])).parent

    def id_of(self, name: str) -> str | None:
        for identifier in self.nodes:
            if self.name(identifier) == name:
                return identifier
        return None

    def edges(self, identifier: str, kinds: set[str | None]) -> list[str]:
        node = self.nodes.get(identifier)
        if node is None:
            return []
        reached = []
        for dependency in node.get("deps", []):
            dependency_kinds = {
                entry.get("kind") for entry in dependency.get("dep_kinds", [])
            } or {None}
            if dependency_kinds & kinds:
                reached.append(dependency["pkg"])
        return reached

    def reachable(self, roots: list[str], kinds: set[str | None]) -> set[str]:
        seen: set[str] = set()
        queue = list(roots)
        while queue:
            identifier = queue.pop()
            if identifier in seen:
                continue
            seen.add(identifier)
            if self.is_proc_macro(identifier):
                # A proc macro is compiled for the host and expands into
                # source; nothing of it is linked into the artifact, so it
                # cannot carry a run-time text semantic.
                continue
            queue.extend(self.edges(identifier, kinds))
        return seen

    def own_requirements(self, name: str) -> list[str]:
        """Every version requirement our own crates put on a package."""

        requirements = []
        for identifier in self.workspace_members:
            for dependency in self.packages[identifier].get("dependencies", []):
                if dependency["name"] != name:
                    continue
                if dependency.get("kind") == "dev":
                    continue
                requirements.append(dependency.get("req", ""))
        return requirements


def encode_path_sources() -> list[Path]:
    """The source files the R1 reference test reads.

    Every source file of the six crates, less the facade files that serve
    the artifact lifecycle, the serving pool, the diagnostics records and
    the CLI.
    """
    sources: list[Path] = []
    for crate in sorted(path.name for path in (ROOT / "crates").iterdir()):
        if crate == "toktier-py":
            # The wheel's extension module is a separate artifact with its
            # own identity; the Rust API certificate does not speak for it.
            continue
        source_root = ROOT / "crates" / crate / "src"
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.rs")):
            if crate != "toktier":
                sources.append(path)
                continue
            relative = path.relative_to(source_root)
            if relative.parts[0] == "bin":
                continue
            if str(relative) in FACADE_LIFECYCLE_SOURCES:
                continue
            sources.append(path)
    return sources


def check_facade_source_classification() -> None:
    """Every file of the facade's own sources is classified."""

    source_root = ROOT / "crates" / "toktier" / "src"
    known = set(FACADE_LIFECYCLE_SOURCES) | set(FACADE_ENCODE_SOURCES)
    unclassified = []
    for path in sorted(source_root.rglob("*.rs")):
        relative = path.relative_to(source_root)
        if relative.parts[0] == "bin":
            continue
        if str(relative) not in known:
            unclassified.append(str(relative))
    if unclassified:
        raise GenerationError(
            "these files of crates/toktier/src are in neither "
            f"FACADE_ENCODE_SOURCES nor FACADE_LIFECYCLE_SOURCES of {TOOL_NAME}: "
            + ", ".join(unclassified)
            + " -- classify each one, because whether a dependency it names "
            "is part of the certified core depends on the answer"
        )
    missing = sorted(
        name
        for name in known
        if not (source_root / name).is_file()
    )
    if missing:
        raise GenerationError(
            f"{TOOL_NAME} classifies files crates/toktier/src no longer holds: "
            + ", ".join(missing)
        )


def referenced_from_encode_path(name: str, sources: list[Path]) -> int:
    """How many encode-path files name a package's library."""

    identifier = name.replace("-", "_")
    pattern = re.compile(
        rf"(?:^|[^A-Za-z0-9_]){re.escape(identifier)}\s*::|\buse\s+{re.escape(identifier)}\b"
    )
    return sum(
        1
        for path in sources
        if pattern.search(path.read_text(encoding="utf-8"))
    )


def behaviour_version(unit: str, graph: Graph, versions: dict[str, str]) -> str:
    """The version of one behaviour unit, read where that unit records it.

    A crate whose text semantics have no version of their own answers with
    its package version, which the build script then compares exactly.
    """
    source = BEHAVIOUR_SOURCES[unit]
    if source == "crate-version":
        return f"crate:{versions[unit]}"
    witness = BEHAVIOUR_WITNESS[unit]
    identifier = graph.id_of(witness)
    if identifier is None:
        raise GenerationError(
            f"the behaviour version of {unit} is read from {witness}, "
            "which this graph does not resolve"
        )
    directory = graph.manifest_dir(identifier)
    if unit == "regex":
        # The highest Unicode age the tables carry is the Unicode version
        # of the tables, and it is what the runtime probe finds by asking
        # the engine which `\p{Age=...}` values it accepts.
        text = (directory / "src/unicode_tables/age.rs").read_text(encoding="utf-8")
        ages = re.findall(r'\("V(\d+)_(\d+)"', text)
        if not ages:
            raise GenerationError(f"{witness} carries no age table to read")
        major, minor = max((int(a), int(b)) for a, b in ages)
        return f"{major}.{minor}"
    if unit == "onig":
        text = (directory / "oniguruma/src/oniguruma.h").read_text(encoding="utf-8")
        fields = []
        for field in ("MAJOR", "MINOR", "TEENY"):
            match = re.search(rf"ONIGURUMA_VERSION_{field}\s+(\d+)", text)
            if match is None:
                raise GenerationError(
                    f"{witness} does not declare ONIGURUMA_VERSION_{field}"
                )
            fields.append(match.group(1))
        return ".".join(fields)
    if unit == "unicode-segmentation":
        text = (directory / "src/tables.rs").read_text(encoding="utf-8")
        match = re.search(
            r"pub const UNICODE_VERSION:\s*\([^)]*\)\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)",
            text,
        )
        if match is None:
            raise GenerationError(f"{witness} declares no UNICODE_VERSION")
        return ".".join(match.groups())
    raise GenerationError(f"no behaviour-version reader for {unit}")


Package = tuple[str, str]


def check_table_covers(compiled: set[Package]) -> None:
    """Every classified package is one this build still compiles.

    The table is how a text-semantics package earns or loses a place in
    the certified core, so an entry the closure no longer holds is a
    classification nobody is reading: it is corrected here rather than
    left to look like coverage.
    """
    names = {name for name, _ in compiled}
    stale = sorted(set(TEXT_SEMANTICS_TABLE) - names)
    if stale:
        raise GenerationError(
            f"{TOOL_NAME}'s TEXT_SEMANTICS_TABLE classifies packages this build "
            "no longer compiles: "
            + ", ".join(stale)
            + " -- remove each one, or say why the closure should still hold it"
        )




def classify(
    compiled: set[Package], graphs: Sequence[ResolveGraph]
) -> tuple[dict[Package, str], dict[Package, str]]:
    """The tier and criterion of every compiled package.

    Keyed by name and version, because a graph can hold two versions of
    one package and only one of them can be the copy an own edge names:
    `base64 0.22.1` arrives on our own pinned edge, and `base64 0.13.1`
    arrives under `spm_precompiled`.
    """
    sources = encode_path_sources()

    workspace: set[Package] = set()
    direct: set[Package] = set()
    reachable: set[Package] = set()
    proc_macro: set[Package] = set()
    for graph in graphs:
        for identifier in graph.workspace_members:
            package = (graph.name(identifier), graph.version(identifier))
            if package in compiled:
                workspace.add(package)
            for edge in graph.edges(identifier, {None, "build"}):
                candidate = (graph.name(edge), graph.version(edge))
                if candidate in compiled:
                    direct.add(candidate)
        roots = [
            root
            for engine in ENGINE_CRATES
            if (root := graph.id_of(engine)) is not None
        ]
        for identifier in graph.reachable(roots, {None}):
            candidate = (graph.name(identifier), graph.version(identifier))
            if candidate in compiled:
                reachable.add(candidate)
        for identifier in graph.nodes:
            if graph.is_proc_macro(identifier):
                proc_macro.add((graph.name(identifier), graph.version(identifier)))

    table_core = {
        name
        for name, entry in TEXT_SEMANTICS_TABLE.items()
        if entry["tier"] == "core"
    }
    own = {name for name, _ in workspace}
    references = {
        name: referenced_from_encode_path(name, sources)
        for name in sorted({name for name, _ in direct} - own)
    }

    tiers: dict[Package, str] = {}
    criteria: dict[Package, str] = {}
    for package in sorted(compiled):
        name, _ = package
        if package in workspace:
            tiers[package], criteria[package] = "core", "R0"
        elif name in table_core:
            # The table wins over R1 so that a package we take a direct
            # edge on in order to probe it -- regex, onig,
            # unicode-segmentation -- is still judged by behaviour version
            # rather than pinned exactly. It applies to every version of
            # that name in the graph, which is the conservative direction.
            tiers[package], criteria[package] = "core", "R2"
        elif package in direct and references.get(name, 0) > 0:
            tiers[package], criteria[package] = "core", "R1"
        elif name in TEXT_SEMANTICS_TABLE:
            tiers[package], criteria[package] = "periphery", "periphery:classified"
        elif package in direct:
            tiers[package] = "periphery"
            criteria[package] = "periphery:lifecycle-only-direct-dependency"
        elif package in proc_macro:
            tiers[package], criteria[package] = "periphery", "periphery:proc-macro"
        elif package not in reachable:
            tiers[package] = "periphery"
            criteria[package] = "periphery:not-reachable-from-an-engine-crate"
        else:
            tiers[package] = "periphery"
            criteria[package] = "periphery:reachable-not-text-semantics"

    unclassified = sorted(
        f"{name} {version}"
        for (name, version) in compiled
        if NAME_NET.match(name)
        and name not in TEXT_SEMANTICS_TABLE
        and criteria[(name, version)] not in ("R0", "R1")
    )
    if unclassified:
        raise GenerationError(
            "these packages match the text-semantics name net and are not "
            f"classified by {TOOL_NAME}'s TEXT_SEMANTICS_TABLE: "
            + ", ".join(unclassified)
            + " -- add each one with a reason, on either side"
        )

    unpinned = sorted(
        f"{name} {version}"
        for (name, version), criterion in criteria.items()
        if criterion == "R1"
        and not any(
            requirement.startswith("=")
            for graph in graphs
            for requirement in graph.own_requirements(name)
        )
    )
    if unpinned:
        raise GenerationError(
            "these packages are in the certified core by R1 and no edge of "
            "ours pins them exactly: "
            + ", ".join(unpinned)
            + " -- pin one edge with `=`, or classify the package instead"
        )

    unreachable_core = sorted(
        f"{name} {version}"
        for (name, version) in compiled
        if name in table_core and (name, version) not in reachable
    )
    if unreachable_core:
        raise GenerationError(
            "these packages are classified as text-semantics core and are not "
            "reachable through normal edges from an engine crate: "
            + ", ".join(unreachable_core)
            + " -- nothing of them is linked into the running artifact"
        )
    return tiers, criteria


def build_document() -> dict[str, object]:
    selections = judged_selections()
    check_facade_source_classification()
    packages: set[tuple[str, str]] = set()
    graphs: list[Graph] = []
    for selection in selections:
        # The union over the judged selections. Where they differ, judging
        # the union is the conservative direction: an extra package can
        # only add a refusal, never remove one.
        packages |= compiled_packages(selection)
        graphs.append(Graph(resolve_graph(selection)))
    check_table_covers(packages)
    tiers, criteria = classify(packages, graphs)
    versions = {name: version for name, version in packages}
    behaviour_versions = {
        unit: behaviour_version(unit, graphs[0], versions)
        for unit in sorted(set(BEHAVIOUR_UNITS.values()))
    }
    entries = []
    for package in sorted(packages):
        name, version = package
        entry = {
            "name": name,
            "version": version,
            "tier": tiers[package],
            "criterion": criteria[package],
        }
        unit = BEHAVIOUR_UNITS.get(name)
        if unit is not None and criteria[package] == "R2":
            entry["behavior_unit"] = unit
            entry["behavior_version"] = behaviour_versions[unit]
            entry["behavior_source"] = BEHAVIOUR_SOURCES[unit]
        entries.append(entry)
    return {
        "schema": SCHEMA,
        "generated_by": TOOL_NAME,
        "root": ROOT_PACKAGE,
        "selections": selections,
        "tier_rule": {
            "statement": TIER_STATEMENT,
            "compiled_set_command": (
                "cargo tree --locked --offline --package toktier --edges "
                "normal,build --target <target> --prefix none --format {p} "
                "[--features <features>]"
            ),
            "graph_command": (
                "cargo metadata --format-version 1 --locked --offline "
                "--filter-platform <target> [--features <features>]"
            ),
            "engine_crates": list(ENGINE_CRATES),
            "encode_path_exclusions": [
                f"crates/toktier/src/{name}" for name in FACADE_LIFECYCLE_SOURCES
            ]
            + ["crates/toktier/src/bin/"],
            "name_net": NAME_NET.pattern,
            "text_semantics_table": TEXT_SEMANTICS_TABLE,
            "behavior_units": BEHAVIOUR_UNITS,
            "behavior_sources": BEHAVIOUR_SOURCES,
        },
        "packages": entries,
    }


def serialise(document: dict[str, object]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        document = build_document()
        rendered = serialise(document)
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
    entries = cast(list[dict[str, str]], document["packages"])
    core = sum(1 for entry in entries if entry["tier"] == "core")
    print(
        f"wrote {OUTPUT} ({len(entries)} packages: "
        f"{core} core, {len(entries) - core} periphery)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
