"""Shared test fixtures.

Every test runs against an isolated home: no test reads the developer's
real cache, state or configuration file, and no test reaches the
network.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

#: Environment variables that could leak the outer machine into a test.
_CLEARED_PREFIXES = ("TOKTIER_", "XDG_")
_CLEARED_NAMES = ("HF_HUB_OFFLINE", "HF_HOME", "HUGGINGFACE_HUB_CACHE")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the platform conventions at a temporary user home."""
    for name in list(os.environ):
        if name.startswith(_CLEARED_PREFIXES) or name in _CLEARED_NAMES:
            monkeypatch.delenv(name, raising=False)
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    yield home
