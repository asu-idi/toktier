#!/usr/bin/env python3
"""Judge sentinel artifact equivalence for two TokTier source trees.

The two builds run sequentially at one canonical source path and one fresh,
canonical Cargo target path.  Version 1 witnesses are deliberately same-host
only: this tool has no cross-host mode and records a neutral host fingerprint.
Cross-host rlib reproducibility has not yet been validated, so a cross-host
record must be refused until that validation changes the contract.

``not_applicable`` is a normal verdict and exits zero; it sends the change to
full recertification (or, for the enumerated package-version axis, to the
identity-v2 mechanism).  ``not_equivalent`` exits one.  An operational error
exits two without manufacturing a verdict.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, NoReturn

from source_identity_common import IDENTITIES

DEFAULT_SCRATCH_ROOT = Path("/tmp/toktier_equiv")
CANONICAL_TREE_NAME = "tree"
CANONICAL_TARGET_NAME = "target"
IDENTITY_SENTINEL_ENV = "TOKTIER_IDENTITY_SENTINEL"
IGNORED_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "target",
    }
)
BYTE_SHIPPED_FILES = frozenset(
    {
        Path("src/toktier/backends/fast_cpu.py"),
        Path("src/toktier/engine/gpu/native.py"),
        Path("src/toktier/kernels/bpe_tables.py"),
        Path("src/toktier/repair/tables/fast_repair_families.v1.json"),
        Path("src/toktier/repair/tables/repair_pclass.v1.zlib"),
    }
)
FIXED_PROTECTED_FILES = (
    Path("Cargo.lock"),
    Path("rust-toolchain.toml"),
    Path("pyproject.toml"),
)
AMBIENT_EXACT_NAMES = frozenset(
    {
        "AR",
        "CC",
        "CFLAGS",
        "CPATH",
        "CXX",
        "CXXFLAGS",
        "HOST",
        "LANG",
        "LD_LIBRARY_PATH",
        "LDFLAGS",
        "LIBRARY_PATH",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "SOURCE_DATE_EPOCH",
        "TARGET",
        "TZ",
        "VIRTUAL_ENV",
    }
)
AMBIENT_PREFIXES = (
    "CARGO_",
    "LC_",
    "PKG_CONFIG",
    "RUST",
)


class JudgeError(RuntimeError):
    """An operational condition prevented a trustworthy judgment."""


@dataclass(frozen=True)
class Coverage:
    """The covered paths and byte-shipped subset used by applicability."""

    files: frozenset[Path]
    trees: tuple[Path, ...]
    byte_shipped: frozenset[Path]


@dataclass(frozen=True)
class Applicability:
    """Ordered applicability result and its carry-over witness fragment."""

    applicable: bool
    witness: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_sha256(root: Path, relative: Path) -> str | None:
    path = root / relative
    if path.is_symlink():
        return sha256(os.readlink(path).encode()).hexdigest()
    if not path.is_file():
        return None
    return _sha256(path)


def _same_entry(left: Path, right: Path) -> bool:
    if left.is_symlink() or right.is_symlink():
        return (
            left.is_symlink()
            and right.is_symlink()
            and os.readlink(left) == os.readlink(right)
        )
    if left.is_file() and right.is_file():
        return filecmp.cmp(left, right, shallow=False)
    return not left.exists() and not right.exists()


def _tree_files(root: Path) -> set[Path]:
    files: set[Path] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = [name for name in names if name not in IGNORED_NAMES]
        directory_path = Path(directory)
        for name in filenames:
            relative = (directory_path / name).relative_to(root)
            if not any(part in IGNORED_NAMES for part in relative.parts):
                files.add(relative)
        for name in names:
            path = directory_path / name
            if path.is_symlink():
                files.add(path.relative_to(root))
    return files


def _is_within(relative: Path, tree: Path) -> bool:
    try:
        relative.relative_to(tree)
    except ValueError:
        return False
    return True


def coverage_for_roots(*roots: Path) -> Coverage:
    """Expand the shared v1 definitions against both candidate trees."""
    files = {
        Path(relative)
        for definition in IDENTITIES.values()
        for relative in definition.files
    }
    trees = tuple(
        sorted(
            {
                Path(relative)
                for definition in IDENTITIES.values()
                for relative in definition.trees
            },
            key=lambda path: path.parts,
        )
    )
    for root in roots:
        for tree in trees:
            directory = root / tree
            if directory.is_dir():
                files.update(
                    path.relative_to(root)
                    for path in directory.rglob("*")
                    if path.is_file() or path.is_symlink()
                )
    return Coverage(frozenset(files), trees, BYTE_SHIPPED_FILES)


def _allowed_non_byte_shipped(relative: Path, coverage: Coverage) -> bool:
    if relative not in coverage.files:
        return False
    if relative.suffix == ".rs" or relative.name == "Cargo.toml":
        return True
    in_coverage_tree = any(_is_within(relative, tree) for tree in coverage.trees)
    name = relative.name.upper()
    is_document = (
        relative.suffix.lower() in {".md", ".txt"}
        or name == "LICENSE"
        or name.startswith(("LICENSE-", "NOTICE"))
    )
    return in_coverage_tree and is_document


def _comparison(left: Path, right: Path, relative: Path) -> dict[str, Any]:
    left_hash = _entry_sha256(left, relative)
    right_hash = _entry_sha256(right, relative)
    return {
        "path": relative.as_posix(),
        "from_sha256": left_hash,
        "to_sha256": right_hash,
        "unchanged": left_hash is not None and left_hash == right_hash,
    }


def check_applicability(
    old_tree: Path,
    new_tree: Path,
    coverage: Coverage | None = None,
) -> Applicability:
    """Apply the three preconditions in their contractually fixed order."""
    active_coverage = coverage or coverage_for_roots(old_tree, new_tree)
    protected = frozenset(FIXED_PROTECTED_FILES) | active_coverage.byte_shipped
    relatives = _tree_files(old_tree) | _tree_files(new_tree)
    changed = tuple(
        sorted(
            (
                relative
                for relative in relatives
                if not _same_entry(old_tree / relative, new_tree / relative)
            ),
            key=lambda path: path.parts,
        )
    )
    witness: dict[str, Any] = {
        "diff_files": [path.as_posix() for path in changed],
        "checks": [],
    }

    outside = [
        path.as_posix()
        for path in changed
        if path not in protected
        and not _allowed_non_byte_shipped(path, active_coverage)
    ]
    confined = not outside
    witness["checks"].append(
        {
            "precondition": "covered_non_byte_shipped_diff_only",
            "passed": confined,
            "outside_files": outside,
        }
    )
    if not confined:
        witness["failure"] = "tree_diff_outside_covered_non_byte_shipped_files"
        return Applicability(False, witness)

    lock_comparison = _comparison(old_tree, new_tree, Path("Cargo.lock"))
    lock_unchanged = bool(lock_comparison["unchanged"])
    witness["cargo_lock_unchanged"] = lock_unchanged
    witness["cargo_lock"] = lock_comparison
    witness["checks"].append(
        {
            "precondition": "cargo_lock_byte_equal",
            "passed": lock_unchanged,
        }
    )
    if not lock_unchanged:
        witness["failure"] = "cargo_lock_changed"
        return Applicability(False, witness)

    protected_comparisons = [
        _comparison(old_tree, new_tree, relative)
        for relative in (*FIXED_PROTECTED_FILES[1:], *sorted(
            active_coverage.byte_shipped, key=lambda path: path.parts
        ))
    ]
    protected_unchanged = all(
        bool(comparison["unchanged"]) for comparison in protected_comparisons
    )
    witness["protected_files_unchanged"] = protected_unchanged
    witness["protected_files"] = protected_comparisons
    witness["checks"].append(
        {
            "precondition": "toolchain_packaging_and_byte_shipped_files_equal",
            "passed": protected_unchanged,
            "changed_files": [
                comparison["path"]
                for comparison in protected_comparisons
                if not comparison["unchanged"]
            ],
        }
    )
    if not protected_unchanged:
        witness["failure"] = "protected_file_changed"
        return Applicability(False, witness)
    return Applicability(True, witness)


def _command_output(argv: Sequence[str], cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise JudgeError(f"cannot run {' '.join(argv)!r}: {error}") from error
    return completed.stdout.strip()


def _resolve_executable(value: str, description: str) -> Path:
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.parent != Path(".")
        else Path(shutil.which(value) or "")
    )
    if not resolved.is_file():
        raise JudgeError(f"{description} executable not found: {value}")
    return resolved


def _default_maturin() -> str:
    virtual_environment = os.environ.get("VIRTUAL_ENV")
    candidates = []
    if virtual_environment:
        candidates.append(Path(virtual_environment) / "bin" / "maturin")
    candidates.append(Path(__file__).resolve().parents[1] / ".venv/bin/maturin")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "maturin"


def _metadata(
    tree: Path, cargo: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    argv = [str(cargo), "metadata", "--locked", "--format-version", "1"]
    try:
        completed = subprocess.run(
            argv,
            cwd=tree,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise JudgeError(
            f"cargo metadata failed for {tree}: {detail.strip()}"
        ) from error
    try:
        document = json.loads(completed.stdout)
        packages = document["packages"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise JudgeError(f"cargo metadata returned invalid JSON for {tree}") from error
    if not isinstance(packages, list):
        raise JudgeError(f"cargo metadata packages are invalid for {tree}")
    return packages, argv


def _cargo_home_for_manifest(manifest: Path) -> Path | None:
    parts = manifest.parts
    for marker in ("registry", "git"):
        if marker in parts:
            index = parts.index(marker)
            return Path(*parts[:index])
    return None


def enumerate_cargo_homes(
    trees: tuple[Path, Path], cargo: Path
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    """Enumerate actual dependency roots from locked Cargo metadata."""
    roots: set[Path] = set()
    external_manifests: set[str] = set()
    commands: list[dict[str, Any]] = []
    for tree in trees:
        packages, argv = _metadata(tree, cargo)
        commands.append({"argv": argv, "cwd": str(tree)})
        for package in packages:
            if not isinstance(package, dict):
                raise JudgeError("cargo metadata contains a non-object package")
            value = package.get("manifest_path")
            if not isinstance(value, str):
                raise JudgeError("cargo metadata package lacks manifest_path")
            manifest = Path(value).resolve()
            try:
                manifest.relative_to(tree)
                continue
            except ValueError:
                pass
            cargo_home = _cargo_home_for_manifest(manifest)
            if cargo_home is None:
                raise JudgeError(
                    "cannot enumerate the Cargo home for external manifest "
                    f"{manifest}"
                )
            roots.add(cargo_home)
            external_manifests.add(str(manifest))
    if not roots:
        raise JudgeError("cargo metadata enumerated no external Cargo home roots")
    rendered = "\n".join(sorted(external_manifests)).encode()
    ordered = tuple(sorted(roots, key=str))
    evidence = {
        "method": "locked_cargo_metadata_external_manifest_paths",
        "commands": commands,
        "roots": [str(path) for path in ordered],
        "external_manifest_count": len(external_manifests),
        "external_manifests_sha256": sha256(rendered).hexdigest(),
    }
    return ordered, evidence


def _ambient_environment() -> dict[str, str | None]:
    names = set(AMBIENT_EXACT_NAMES)
    names.update(
        name
        for name in os.environ
        if name.startswith(AMBIENT_PREFIXES)
    )
    names.update(
        {
            "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_INCREMENTAL",
            "CARGO_TARGET_DIR",
            "RUSTC",
            "RUSTC_WORKSPACE_WRAPPER",
            "RUSTC_WRAPPER",
            "RUSTDOC",
            "RUSTDOCFLAGS",
            "RUSTFLAGS",
        }
    )
    return {name: os.environ.get(name) for name in sorted(names)}


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware"}:
                return value.strip()
    return platform.processor() or "unknown"


def _host_fingerprint() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    facts = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "libc": f"{libc_name} {libc_version}".strip(),
    }
    rendered = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    return {
        "description": "build environment fingerprint without a host name",
        "sha256": sha256(rendered).hexdigest(),
        "facts": facts,
    }


def _optional_version(command: str, arguments: Sequence[str]) -> str:
    executable = shutil.which(command)
    if executable is None:
        return "not found"
    try:
        return _command_output([executable, *arguments]).splitlines()[0]
    except (JudgeError, IndexError):
        return "unavailable"


def _toolchain(
    tree: Path, cargo: Path, rustc: Path, maturin: Path
) -> dict[str, Any]:
    return {
        "rust_toolchain_sha256": _sha256(tree / "rust-toolchain.toml"),
        "rustc": _command_output([str(rustc), "--version", "--verbose"], tree),
        "cargo": _command_output([str(cargo), "--version"], tree),
        "maturin": _command_output([str(maturin), "--version"], tree),
        "python": platform.python_version(),
        "python_abi": sysconfig.get_config_var("SOABI") or "unknown",
        "native_tools": {
            "cc": _optional_version(os.environ.get("CC", "cc"), ["--version"]),
            "cxx": _optional_version(os.environ.get("CXX", "c++"), ["--version"]),
            "linker": _optional_version("ld", ["--version"]),
            "archiver": _optional_version(os.environ.get("AR", "ar"), ["--version"]),
        },
    }


def _cargo_config_search_paths(
    canonical_tree: Path, cargo_homes: tuple[Path, ...]
) -> tuple[list[str], list[Path]]:
    directories = {canonical_tree / ".cargo"}
    directories.update(parent / ".cargo" for parent in canonical_tree.parents)
    directories.update(cargo_homes)
    paths = [
        directory / filename
        for directory in sorted(directories, key=str)
        for filename in ("config", "config.toml")
    ]
    return [str(path) for path in paths], [path for path in paths if path.is_file()]


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(*sorted(IGNORED_NAMES)),
    )


def _run_build(argv: Sequence[str], tree: Path, environment: dict[str, str]) -> None:
    print(f"+ {' '.join(argv)}", file=sys.stderr, flush=True)
    try:
        subprocess.run(argv, cwd=tree, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise JudgeError(f"build command failed: {' '.join(argv)}") from error


def _extract_native(wheel_directory: Path, destination: Path) -> None:
    wheels = sorted(wheel_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise JudgeError(
            f"expected one wheel in {wheel_directory}, found {len(wheels)}"
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.endswith("toktier/_native.abi3.so")
        ]
        if len(members) != 1:
            raise JudgeError(
                f"expected one toktier/_native.abi3.so member, found {len(members)}"
            )
        destination.write_bytes(archive.read(members[0]))


def _materialize_build(
    label: str,
    source: Path,
    scratch_root: Path,
    recipe: dict[str, Any],
) -> dict[str, Path]:
    tree = Path(recipe["tree_path"])
    target = Path(recipe["cargo_target_dir"])
    distribution = scratch_root / "dist"
    saved = scratch_root / "witness" / label
    for path in (tree, target, distribution):
        if path.exists():
            shutil.rmtree(path)
    _copy_tree(source, tree)
    distribution.mkdir()
    saved.mkdir(parents=True)

    environment = os.environ.copy()
    environment.pop("CARGO_ENCODED_RUSTFLAGS", None)
    environment.update(recipe["effective_environment"])
    for command in recipe["commands"]:
        _run_build(command["argv"], tree, environment)

    native = saved / "_native.abi3.so"
    rlib = saved / "libtoktier.rlib"
    _extract_native(distribution, native)
    built_rlib = target / "release/libtoktier.rlib"
    if not built_rlib.is_file():
        raise JudgeError(f"whole rlib witness is missing: {built_rlib}")
    shutil.copy2(built_rlib, rlib)
    return {"_native.abi3.so": native, "libtoktier.rlib": rlib}


def _artifact_witness(
    old_artifacts: dict[str, Path], new_artifacts: dict[str, Path]
) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    equivalent = True
    for name in ("_native.abi3.so", "libtoktier.rlib"):
        old = old_artifacts[name]
        new = new_artifacts[name]
        old_hash = _sha256(old)
        new_hash = _sha256(new)
        byte_equal = filecmp.cmp(old, new, shallow=False)
        record: dict[str, Any] = {
            "artifact": name,
            "from_sha256": old_hash,
            "to_sha256": new_hash,
            "from_bytes": old.stat().st_size,
            "to_bytes": new.stat().st_size,
            "byte_equal": byte_equal,
        }
        if byte_equal:
            record["sha256_both"] = old_hash
            record["bytes"] = old.stat().st_size
        records.append(record)
        equivalent = equivalent and byte_equal
    return records, equivalent


def _recipe(
    old_tree: Path,
    new_tree: Path,
    scratch_root: Path,
    cargo_value: str,
    rustc_value: str,
    maturin_value: str,
) -> dict[str, Any]:
    cargo = _resolve_executable(cargo_value, "cargo")
    rustc = _resolve_executable(rustc_value, "rustc")
    maturin = _resolve_executable(maturin_value, "maturin")
    nice = _resolve_executable("nice", "nice")
    cargo_homes, enumeration = enumerate_cargo_homes((old_tree, new_tree), cargo)
    canonical_tree = scratch_root / CANONICAL_TREE_NAME
    canonical_target = scratch_root / CANONICAL_TARGET_NAME
    remaps = [f"--remap-path-prefix={canonical_tree}=/toktier"]
    remaps.extend(f"--remap-path-prefix={path}=/cargo" for path in cargo_homes)
    if any(any(character.isspace() for character in value) for value in remaps):
        raise JudgeError("remap paths containing whitespace are not supported")
    rustflags = " ".join(remaps)
    distribution = scratch_root / "dist"
    commands = [
        {
            "purpose": "sentinel_native_extension",
            "cwd": str(canonical_tree),
            "argv": [
                str(nice),
                "-n",
                "5",
                str(maturin),
                "build",
                "--locked",
                "--release",
                "--out",
                str(distribution),
            ],
        },
        {
            "purpose": "sentinel_whole_rust_api_rlib",
            "cwd": str(canonical_tree),
            "argv": [
                str(nice),
                "-n",
                "5",
                str(cargo),
                "build",
                "--locked",
                "--release",
                "-p",
                "toktier",
            ],
        },
    ]
    source_configs = [
        tree / ".cargo" / filename
        for tree in (old_tree, new_tree)
        for filename in ("config", "config.toml")
        if (tree / ".cargo" / filename).is_file()
    ]
    if source_configs:
        rendered = ", ".join(map(str, source_configs))
        raise JudgeError(
            "source-tree Cargo configuration is not accepted by the v1 "
            f"recipe: {rendered}"
        )
    config_search_paths, cargo_configs = _cargo_config_search_paths(
        canonical_tree, cargo_homes
    )
    if cargo_configs:
        rendered = ", ".join(map(str, cargo_configs))
        raise JudgeError(
            "ambient Cargo configuration is not accepted by the v1 recipe: "
            f"{rendered}"
        )
    return {
        "tree_path": str(canonical_tree),
        "cargo_target_dir": str(canonical_target),
        "rustflags": rustflags,
        "cargo_home_roots": [str(path) for path in cargo_homes],
        "cargo_home_enumeration": enumeration,
        "locked": True,
        "fresh_target_for_each_tree": True,
        "sequential_same_path_builds": True,
        "commands": commands,
        "toolchain": _toolchain(old_tree, cargo, rustc, maturin),
        "ambient_environment": _ambient_environment(),
        "effective_environment": {
            "CARGO_INCREMENTAL": "0",
            "CARGO_TARGET_DIR": str(canonical_target),
            "CARGO_TERM_COLOR": "never",
            "CUDA_VISIBLE_DEVICES": "",
            "RUSTFLAGS": rustflags,
            IDENTITY_SENTINEL_ENV: "1",
        },
        "removed_environment": ["CARGO_ENCODED_RUSTFLAGS"],
        "cargo_config_search_paths": config_search_paths,
        "cargo_configs": [],
        "host_fingerprint": _host_fingerprint(),
        "same_host_only": True,
    }


def judge(
    old_tree: Path,
    new_tree: Path,
    scratch_root: Path,
    cargo: str,
    rustc: str,
    maturin: str,
) -> dict[str, Any]:
    """Return one of the three verdicts and its witness JSON fragment."""
    old_tree = old_tree.resolve()
    new_tree = new_tree.resolve()
    for tree in (old_tree, new_tree):
        if not tree.is_dir() or not (tree / "Cargo.toml").is_file():
            raise JudgeError(f"not a TokTier source tree: {tree}")
    applicability = check_applicability(old_tree, new_tree)
    if not applicability.applicable:
        return {
            "verdict": "not_applicable",
            "witness": {"applicability": applicability.witness},
        }

    scratch_root = scratch_root.resolve()
    if scratch_root.exists():
        raise JudgeError(f"scratch root already exists: {scratch_root}")
    if any(
        scratch_root == tree
        or scratch_root in tree.parents
        or tree in scratch_root.parents
        for tree in (old_tree, new_tree)
    ):
        raise JudgeError("scratch root and input trees cannot contain each other")
    scratch_root.mkdir(parents=True, mode=0o700)
    created_scratch = True
    try:
        recipe = _recipe(
            old_tree,
            new_tree,
            scratch_root,
            cargo,
            rustc,
            maturin,
        )
        before_host = recipe["host_fingerprint"]
        old_artifacts = _materialize_build(
            "from", old_tree, scratch_root, recipe
        )
        new_artifacts = _materialize_build("to", new_tree, scratch_root, recipe)
        after_host = _host_fingerprint()
        if before_host["sha256"] != after_host["sha256"]:
            raise JudgeError("build host fingerprint changed between the two builds")
        artifacts, equivalent = _artifact_witness(old_artifacts, new_artifacts)
        return {
            "verdict": "equivalent" if equivalent else "not_equivalent",
            "witness": {
                "sentinel_artifacts": artifacts,
                "recipe": recipe,
                "applicability": applicability.witness,
            },
        }
    finally:
        if created_scratch and scratch_root.is_dir():
            shutil.rmtree(scratch_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_tree", type=Path)
    parser.add_argument("new_tree", type=Path)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--rustc", default="rustc")
    parser.add_argument("--maturin", default=_default_maturin())
    return parser


def _die(error: JudgeError) -> NoReturn:
    print(f"error: {error}", file=sys.stderr)
    raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = judge(
            arguments.old_tree,
            arguments.new_tree,
            arguments.scratch_root,
            arguments.cargo,
            arguments.rustc,
            arguments.maturin,
        )
    except JudgeError as error:
        _die(error)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["verdict"] == "not_equivalent" else 0


if __name__ == "__main__":
    raise SystemExit(main())
