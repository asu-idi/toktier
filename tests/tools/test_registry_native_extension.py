"""How the registry check finds the extension it reads identity from.

``tools/generate_registry.py`` binds the shipped registry to the native
host that executes prebuilt GPU requests, and reads that identity out of
the compiled extension. The tools put this repository's ``src`` first on
``sys.path``, so a source tree that has never been built has no
extension there -- the ordinary shape of a snapshot checked out beside
an installed wheel. These tests cover which extension the search picks
and, above all, that the exact-identity requirement is untouched by
where it was read.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPOSITORY_ROOT / "tools"


@pytest.fixture
def registry_tool(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    if str(TOOLS) not in sys.path:
        monkeypatch.syspath_prepend(str(TOOLS))
    import generate_registry

    return generate_registry


def test_search_skips_this_source_tree(
    registry_tool: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The source package is never reported as an "installed" extension.

    Adopting it would make the fallback a no-op that looks like a
    success: the source tree is exactly the location already known to be
    empty.
    """
    source_package = REPOSITORY_ROOT / "src" / "toktier"
    monkeypatch.setattr(sys, "path", [str(REPOSITORY_ROOT / "src"), str(tmp_path)])
    assert registry_tool._installed_native_extension() is None

    installed = tmp_path / "toktier"
    installed.mkdir()
    (installed / "_native.abi3.so").write_bytes(b"not a real extension")
    found = registry_tool._installed_native_extension()
    assert found == installed / "_native.abi3.so"
    assert found != source_package / "_native.abi3.so"


def test_search_ignores_non_extension_files(
    registry_tool: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only files with a real extension suffix are candidates."""
    installed = tmp_path / "toktier"
    installed.mkdir()
    (installed / "_native.pyi").write_text("", encoding="utf-8")
    (installed / "_native.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path)])

    assert registry_tool._installed_native_extension() is None


def test_missing_extension_names_both_ways_to_supply_one(
    registry_tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is actionable rather than a bare identity mismatch."""
    from toktier.engine.gpu.native import NativeHostBuildFacts

    monkeypatch.setattr(
        "toktier.engine.gpu.native.native_host_build_facts",
        lambda: NativeHostBuildFacts(),
    )
    monkeypatch.setattr(registry_tool, "_installed_native_extension", lambda: None)

    with pytest.raises(registry_tool.GenerationError) as caught:
        registry_tool.native_host_bindings()

    message = str(caught.value)
    assert "no native host identity could be read" in message
    assert "maturin" in message
    assert "wheel is installed" in message


def test_adopted_extension_still_has_to_match_the_source_set(
    registry_tool: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reading identity elsewhere does not relax what it must equal."""
    from toktier.engine.gpu.native import NativeHostBuildFacts

    adopted = tmp_path / "site-packages" / "toktier" / "_native.abi3.so"
    adopted.parent.mkdir(parents=True)
    adopted.write_bytes(b"not a real extension")

    facts = iter(
        (
            NativeHostBuildFacts(),
            NativeHostBuildFacts(
                source_digest="c" * 64,
                build_flags=("profile=release",),
                toolchain="rustc (test fixture)",
            ),
        )
    )
    monkeypatch.setattr(
        "toktier.engine.gpu.native.native_host_build_facts", lambda: next(facts)
    )
    monkeypatch.setattr(
        registry_tool, "_adopt_installed_native_extension", lambda: adopted
    )
    monkeypatch.setattr(registry_tool, "_ADOPTED_NATIVE_EXTENSION", adopted)

    with pytest.raises(registry_tool.GenerationError) as caught:
        registry_tool.native_host_bindings()

    message = str(caught.value)
    assert "was not built from the current source set" in message
    # The message names the file it judged, so the mismatch is traceable.
    assert str(adopted) in message
