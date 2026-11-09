"""The release-artifact check derives the certified CPU count from the binding.

``tools/verify_release_artifacts.py`` requires the wheel's packaged
support registry to certify as many integrated CPU artifacts as
``tools/fast_cpu_binding.json`` declares (loadable, carried and direct
families), all on the bound engine identity. The count is not written
beside the tool, and it is not read from ``tables/support_registry.json``
either: the packaged registry is a byte-identical copy of that file, so a
count read from it would compare a copy with itself.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "verify_release_artifacts.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_release_artifacts", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verify_release_artifacts", module)
    spec.loader.exec_module(module)
    return module


def _registry() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        (ROOT / "tables" / "support_registry.json").read_bytes()
    )
    return document


def _binding() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(
        (ROOT / "tools" / "fast_cpu_binding.json").read_bytes()
    )
    return document


def test_the_expected_count_is_the_binding_declaration() -> None:
    tool = _tool()
    binding = _binding()
    declared = (
        set(binding["loadable_families"])
        | set(binding.get("carried_families") or {})
        | set(binding.get("direct_families") or {})
    )
    assert len(declared) >= 1
    assert tool.expected_certified_cpu_count() == len(declared)
    assert "!= 13" not in TOOL.read_text(encoding="utf-8")
    # On the shipped tree the declaration and the repository registry agree.
    registry = _registry()
    counted = sum(
        1
        for row in registry["artifacts"]
        if (row.get("backends") or {}).get("fast_cpu", {}).get("status")
        == "certified_source"
    )
    assert counted == len(declared)


def test_a_wheel_built_from_a_registry_the_binding_does_not_declare_is_refused(
    tmp_path: Path,
) -> None:
    """The anchor is the tool-side declaration, not the registry copy."""
    tool = _tool()
    binding = copy.deepcopy(_binding())
    binding["loadable_families"] = list(binding["loadable_families"])[1:]
    path = tmp_path / "fast_cpu_binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")
    expected = tool.expected_certified_cpu_count(path)
    assert expected == tool.expected_certified_cpu_count() - 1
    with pytest.raises(ValueError, match=f"declares {expected}"):
        tool.verify_certified_cpu(_registry(), _binding(), expected)


def test_the_shipped_registry_passes_and_a_changed_count_is_named() -> None:
    tool = _tool()
    registry = _registry()
    expected = tool.expected_certified_cpu_count()
    tool.verify_certified_cpu(registry, _binding(), expected)
    named = f"fast_cpu_binding.json declares {expected + 1}"
    with pytest.raises(ValueError, match=named):
        tool.verify_certified_cpu(registry, _binding(), expected + 1)
    fewer = copy.deepcopy(registry)
    first = next(
        row
        for row in fewer["artifacts"]
        if (row.get("backends") or {}).get("fast_cpu", {}).get("status")
        == "certified_source"
    )
    first["backends"]["fast_cpu"] = {"status": "unsupported"}
    with pytest.raises(ValueError, match=f"certifies {expected - 1} integrated"):
        tool.verify_certified_cpu(fewer, _binding(), expected)


def test_another_engine_identity_is_refused() -> None:
    tool = _tool()
    registry = copy.deepcopy(_registry())
    entry = tool.certified_cpu_entries(registry)[0]
    entry["source_digest"] = "0" * 64
    with pytest.raises(ValueError, match="another integrated CPU identity"):
        tool.verify_certified_cpu(
            registry, _binding(), tool.expected_certified_cpu_count()
        )


def test_a_binding_declaring_no_certified_family_is_refused(tmp_path: Path) -> None:
    tool = _tool()
    binding = copy.deepcopy(_binding())
    binding["loadable_families"] = []
    binding["carried_families"] = {}
    binding["direct_families"] = {}
    path = tmp_path / "fast_cpu_binding.json"
    path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="declares no certified integrated CPU"):
        tool.expected_certified_cpu_count(path)
