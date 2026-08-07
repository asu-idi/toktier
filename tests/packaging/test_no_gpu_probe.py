import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

#: Module objects displaced by a fresh-import experiment; restored after
#: each test so the rest of the suite keeps one enum/module identity.
_SAVED: dict[str, Any] = {}


def _forget_toktier() -> None:
    for name in tuple(sys.modules):
        if name == "toktier" or name.startswith("toktier."):
            _SAVED.setdefault(name, sys.modules.pop(name))


@pytest.fixture(autouse=True)
def _restore_toktier_modules() -> Iterator[None]:
    yield
    if _SAVED:
        for name in tuple(sys.modules):
            if name == "toktier" or name.startswith("toktier."):
                sys.modules.pop(name)
        sys.modules.update(_SAVED)
        _SAVED.clear()


def test_import_does_not_probe_cuda_or_invoke_nvcc(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.syspath_prepend(str(SRC))
    _forget_toktier()

    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    marker = tmp_path / "nvcc-invoked"
    nvcc = shim_dir / "nvcc"
    nvcc.write_text(
        '#!/bin/sh\n: > "$NVCC_SMOKE_MARKER"\nexit 99\n', encoding="utf-8"
    )
    nvcc.chmod(0o755)
    monkeypatch.setenv("NVCC_SMOKE_MARKER", str(marker))
    monkeypatch.setenv("PATH", str(shim_dir))

    importlib.import_module("toktier")

    assert not marker.exists(), "import toktier invoked nvcc"
