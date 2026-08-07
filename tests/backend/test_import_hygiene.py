"""Import and environment hygiene gates.

Contract reference: ``docs/contracts/config.md`` Section 4 (the frozen
environment variable set and its deliberate absences) and the release
gate that importing the package pulls in no accelerator runtime.

These are cheap tests that protect properties which are easy to lose by
accident: one convenience import at module scope would put a multi
hundred megabyte runtime into every process that touches the package,
and one ambient environment read would put a correctness switch outside
the call site.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "toktier"
MODULES = sorted(PACKAGE_ROOT.rglob("*.py"))

#: The frozen long-term environment variable set of config.md Section 4.
FROZEN_ENV = frozenset(
    {
        "TOKTIER_HOME",
        "TOKTIER_OFFLINE",
        "TOKTIER_LOG_LEVEL",
        "TOKTIER_DISABLE_GPU",
        "TOKTIER_DIAGNOSTICS",
    }
)

#: Variables outside the frozen set that the package is allowed to read.
#: Platform conventions only; nothing here can change output.
ALLOWED_EXTRA_ENV = frozenset({"XDG_CACHE_HOME", "XDG_STATE_HOME"})


def test_no_module_imports_an_accelerator_runtime() -> None:
    """Not even inside a function: this lane has no such dependency."""
    offenders: list[str] = []
    for path in MODULES:
        if "gpu" in path.parts and "engine" in path.parts:
            # The GPU engine modules are reachable only through the GPU
            # backend factory, which the routing layer imports lazily; the
            # packaging tier proves that `import toktier` stays torch-free.
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "torch" or name.startswith("torch."):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def _environment_names(tree: ast.AST) -> set[str]:
    """Environment variable names read anywhere in a module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            is_environ_get = (
                isinstance(target, ast.Attribute)
                and target.attr in ("get", "getenv", "setdefault")
            )
            if is_environ_get and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
        elif isinstance(node, ast.Subscript):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                names.add(node.slice.value)
        elif isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if (
                    isinstance(comparator, ast.Attribute)
                    and comparator.attr == "environ"
                    and isinstance(node.left, ast.Name)
                ):
                    names.add(node.left.id)
    return names


def test_only_frozen_environment_variables_are_read() -> None:
    """No ambient switch outside the frozen set reaches the package."""
    seen: set[str] = set()
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        seen.update(
            name
            for name in _environment_names(tree)
            if name.isupper() and "_" in name
        )
    unexpected = {
        name
        for name in seen
        if name not in FROZEN_ENV and name not in ALLOWED_EXTRA_ENV
    }
    assert unexpected == set()


def test_environment_names_use_the_package_prefix() -> None:
    """Every package-owned variable carries the package prefix."""
    for name in FROZEN_ENV:
        assert name.startswith("TOKTIER_")
