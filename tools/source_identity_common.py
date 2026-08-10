"""Shared implementation behind the three source-identity tools.

Each certification source identity is a bare SHA-256 over a domain tag
plus path-bound file bytes. The authoritative hashers run at compile
time (``crates/toktier-py/build.rs`` and ``crates/toktier/build.rs``);
the tools here recompute the same digests so readings, registries, and
release checks can verify them without a build.

One table (:data:`IDENTITIES`) holds the domain and path set of every
identity, so the three thin CLI wrappers cannot drift from each other.
Ordering note: Rust's ``PathBuf`` ordering compares path components,
not the rendered ``'/'`` byte against ``'-'``; sorting by ``path.parts``
matches ``build.rs`` exactly for sibling names such as ``toktier`` and
``toktier-*``. All identities sort with this one key.

This module also carries the coverage cross-checks that
``tools/generate_registry.py`` runs as part of ``--check`` and
``--release-check``:

* :func:`list_synchronization_problems` re-reads the shared Rust path
  module (``crates/toktier/build_support/source_identity.rs``, the one
  definition both build scripts include with ``#[path]``) and confirms
  that every path list there matches this table, and that both build
  scripts still consume the shared module rather than a local copy.
* :func:`routing_core_coverage_problems` watches for new ``.rs`` files
  under ``crates/toktier-routing-core/src``. The fast_cpu identity
  names that crate's files individually, so a freshly added file would
  compile into the binary while staying outside the fast_cpu digest
  unless it is enrolled (or recorded as excluded) deliberately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceIdentity:
    """A domain tag plus the path set it binds."""

    domain: bytes
    files: tuple[str, ...]
    trees: tuple[str, ...]

    def source_paths(self) -> tuple[Path, ...]:
        """The exact, sorted source set hashed by the build scripts."""
        paths = {Path(value) for value in self.files}
        for tree in self.trees:
            paths.update(
                path.relative_to(ROOT)
                for path in (ROOT / tree).rglob("*")
                if path.is_file()
            )
        return tuple(sorted(paths, key=lambda path: path.parts))

    def source_digest(self) -> str:
        """Bare SHA-256 over path-bound source bytes."""
        digest = sha256(self.domain)
        for relative in self.source_paths():
            rendered = relative.as_posix().encode()
            content = (ROOT / relative).read_bytes()
            digest.update(len(rendered).to_bytes(8, "little"))
            digest.update(rendered)
            digest.update(len(content).to_bytes(8, "little"))
            digest.update(content)
        return digest.hexdigest()


IDENTITIES: dict[str, SourceIdentity] = {
    "fast_cpu": SourceIdentity(
        domain=b"toktier.fast_cpu.integrated_source.v1\0",
        files=(
            "Cargo.lock",
            "Cargo.toml",
            "pyproject.toml",
            "rust-toolchain.toml",
            "crates/toktier/build_support/source_identity.rs",
            "crates/toktier-routing-core/Cargo.toml",
            "crates/toktier-routing-core/src/fast_cpu.rs",
            "crates/toktier-routing-core/src/lib.rs",
            "crates/toktier-routing-core/src/reference.rs",
            "crates/toktier-routing-core/src/runtime.rs",
            "crates/toktier-store-core/Cargo.toml",
            "crates/toktier-py/Cargo.toml",
            "crates/toktier-py/build.rs",
            "crates/toktier-py/src/lib.rs",
            "src/toktier/backends/fast_cpu.py",
            "src/toktier/repair/tables/fast_repair_families.v1.json",
            "src/toktier/repair/tables/repair_pclass.v1.zlib",
        ),
        trees=(
            "crates/toktier-gigatoken-core",
            "crates/toktier-store-core/src",
        ),
    ),
    "native_host": SourceIdentity(
        domain=b"toktier.prebuilt.native_host_source.v1\0",
        files=(
            "Cargo.lock",
            "Cargo.toml",
            "pyproject.toml",
            "rust-toolchain.toml",
            "crates/toktier/build_support/source_identity.rs",
            "crates/toktier-py/Cargo.toml",
            "crates/toktier-py/build.rs",
            "crates/toktier-py/src/lib.rs",
            "src/toktier/engine/gpu/native.py",
            "src/toktier/kernels/bpe_tables.py",
        ),
        trees=(
            "crates/toktier-cuda-driver",
            "crates/toktier-routing-core",
            "crates/toktier-store-core",
            "crates/toktier-store-sqlite",
        ),
    ),
    "rust_api": SourceIdentity(
        domain=b"toktier.rust_api.integrated_source.v1\0",
        files=(
            "Cargo.lock",
            "Cargo.toml",
            "rust-toolchain.toml",
            "crates/toktier/Cargo.toml",
            "crates/toktier/build.rs",
            "crates/toktier/build_support/source_identity.rs",
            "src/toktier/repair/tables/fast_repair_families.v1.json",
            "src/toktier/repair/tables/repair_pclass.v1.zlib",
        ),
        trees=(
            "crates/toktier/src",
            "crates/toktier-cuda-driver",
            "crates/toktier-gigatoken-core",
            "crates/toktier-routing-core",
            "crates/toktier-store-core",
            "crates/toktier-store-sqlite",
        ),
    ),
}

# ---------------------------------------------------------------------
# Coverage cross-checks (read-only; they never change a digest).
# ---------------------------------------------------------------------

#: The one Rust-side definition of the identity path lists. Both build
#: scripts include it with ``#[path]``; :data:`BUILD_SCRIPT_INCLUDES`
#: watches that they keep doing so.
SHARED_PATH_MODULE = "crates/toktier/build_support/source_identity.rs"

#: Where each identity's path list is defined on the Rust side.
BUILD_SCRIPT_LISTS: tuple[tuple[str, str, str], ...] = (
    ("fast_cpu", SHARED_PATH_MODULE, "fast_cpu_source_paths"),
    ("native_host", SHARED_PATH_MODULE, "native_host_source_paths"),
    ("rust_api", SHARED_PATH_MODULE, "rust_api_source_paths"),
)

#: The ``#[path]`` include string each build script must carry so the
#: shared module stays their only path-list source.
BUILD_SCRIPT_INCLUDES: tuple[tuple[str, str], ...] = (
    ("crates/toktier/build.rs", '#[path = "build_support/source_identity.rs"]'),
    (
        "crates/toktier-py/build.rs",
        '#[path = "../toktier/build_support/source_identity.rs"]',
    ),
)

ROUTING_CORE_SRC = "crates/toktier-routing-core/src"

#: ``.rs`` files under routing-core/src that stay outside the fast_cpu
#: named-file list on purpose: they serve the GPU and facade store
#: paths and are still covered whole-tree by the native_host and
#: rust_api identities. Enrolling a file here is a deliberate decision,
#: recorded next to the lists it amends.
FAST_CPU_ROUTING_CORE_EXCLUDED: frozenset[str] = frozenset(
    {
        "crates/toktier-routing-core/src/entry_store.rs",
        "crates/toktier-routing-core/src/gpu.rs",
    }
)


def _rust_function_body(source: str, function: str) -> str:
    """The text of a top-level ``fn`` in the shared path module."""
    pattern = re.compile(
        rf"^(?:pub(?:\(crate\))? )?fn {re.escape(function)}\("
        rf".*?(?=^(?:pub(?:\(crate\))? )?fn |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"no function {function!r} in the shared path module")
    return match.group(0)


def _rust_string_literals(body: str) -> set[str]:
    """Every plain string literal in a function body.

    The path lists in the build scripts are plain literals with no
    escapes. Any other literal that later appears inside these
    functions will surface as a mismatch, which is the safe direction:
    the cross-check asks to be updated rather than staying quiet.
    """
    return set(re.findall(r'"([^"\\]*)"', body))


def list_synchronization_problems() -> list[str]:
    """Compare this table with the shared Rust path module.

    The comparison is set-based: both sides sort and deduplicate before
    hashing, so ordering differences cannot change a digest. A missing
    or extra path on either side is reported with both locations, since
    the fix is to bring the two definitions back into agreement. The
    build scripts themselves are checked only for the ``#[path]``
    include line, which is what makes the shared module their single
    source of the lists.
    """
    problems: list[str] = []
    for identity_name, module, function in BUILD_SCRIPT_LISTS:
        expected = set(IDENTITIES[identity_name].files) | set(
            IDENTITIES[identity_name].trees
        )
        module_path = ROOT / module
        try:
            body = _rust_function_body(
                module_path.read_text(encoding="utf-8"), function
            )
        except (OSError, ValueError) as error:
            problems.append(
                f"{module}: cannot read the {identity_name} path list "
                f"({error}); the source-identity cross-check needs it"
            )
            continue
        transcribed = _rust_string_literals(body)
        for path in sorted(expected - transcribed):
            problems.append(
                f"{module} ({function}): {path!r} is in the "
                f"tools-side {identity_name} list but not in the shared "
                "path module; the two definitions need to agree"
            )
        for path in sorted(transcribed - expected):
            problems.append(
                f"{module} ({function}): {path!r} appears in the shared "
                f"path module but not in the tools-side {identity_name} "
                "list; the two definitions need to agree"
            )
    for script, include in BUILD_SCRIPT_INCLUDES:
        try:
            source = (ROOT / script).read_text(encoding="utf-8")
        except OSError as error:
            problems.append(
                f"{script}: cannot read the build script ({error}); the "
                "source-identity cross-check needs it"
            )
            continue
        if include not in source:
            problems.append(
                f"{script}: does not include the shared path module via "
                f"{include!r}; the identity path lists must come from "
                f"{SHARED_PATH_MODULE} rather than a local copy"
            )
    return problems


def routing_core_coverage_problems() -> list[str]:
    """Watch the named-file coverage of routing-core for drift.

    The fast_cpu identity covers ``crates/toktier-routing-core`` by a
    named-file list rather than a whole tree. A new ``.rs`` file there
    would compile into the shipped binary while staying outside the
    fast_cpu digest, so its appearance has to be a recorded decision:
    either enroll it in the named lists (both build scripts and
    :data:`IDENTITIES`) or record it in
    :data:`FAST_CPU_ROUTING_CORE_EXCLUDED`.
    """
    problems: list[str] = []
    named = {
        value
        for value in IDENTITIES["fast_cpu"].files
        if value.startswith(f"{ROUTING_CORE_SRC}/")
    }
    expected = named | FAST_CPU_ROUTING_CORE_EXCLUDED
    present = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / ROUTING_CORE_SRC).rglob("*.rs")
        if path.is_file()
    }
    for path in sorted(present - expected):
        problems.append(
            f"{path}: new routing-core source file is outside the "
            "fast_cpu named-file list; enroll it in both build scripts "
            "and tools/source_identity_common.py, or record it in "
            "FAST_CPU_ROUTING_CORE_EXCLUDED, so it does not silently "
            "stay out of the fast_cpu source identity"
        )
    for path in sorted(expected - present):
        problems.append(
            f"{path}: named in the routing-core coverage lists but "
            "absent from the tree; a rename or removal needs the lists "
            "updated in the same change"
        )
    return problems


def coverage_problems() -> list[str]:
    """All source-identity coverage refusals, for the release checks."""
    return list_synchronization_problems() + routing_core_coverage_problems()


if __name__ == "__main__":
    found = coverage_problems()
    for problem in found:
        print(f"error: {problem}")
    raise SystemExit(1 if found else 0)
