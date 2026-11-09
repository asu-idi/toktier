"""The release-artifact check reads the certified CPU count from the registry.

``tools/verify_release_artifacts.py`` requires the wheel's packaged
support registry to certify as many integrated CPU artifacts as the
repository registry does, all on the bound engine identity. The count is
derived from ``tables/support_registry.json``, not written beside the
tool, so a record added to the registry does not leave the check behind.
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


def test_the_expected_count_is_the_repository_registry_count() -> None:
    tool = _tool()
    registry = _registry()
    counted = sum(
        1
        for row in registry["artifacts"]
        if (row.get("backends") or {}).get("fast_cpu", {}).get("status")
        == "certified_source"
    )
    assert counted >= 1
    assert tool.expected_certified_cpu_count() == counted
    assert "!= 13" not in TOOL.read_text(encoding="utf-8")


def test_the_shipped_registry_passes_and_a_changed_count_is_named() -> None:
    tool = _tool()
    registry = _registry()
    expected = tool.expected_certified_cpu_count()
    tool.verify_certified_cpu(registry, _binding(), expected)
    named = f"repository registry certifies {expected + 1}"
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


def test_a_registry_without_certified_cpu_entries_is_refused(tmp_path: Path) -> None:
    tool = _tool()
    registry = copy.deepcopy(_registry())
    for row in registry["artifacts"]:
        if (row.get("backends") or {}).get("fast_cpu", {}).get("status") == (
            "certified_source"
        ):
            row["backends"]["fast_cpu"] = {"status": "unsupported"}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="certifies no integrated CPU artifact"):
        tool.expected_certified_cpu_count(path)
