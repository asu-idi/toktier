"""Core-wheel delivery contract of the corrected CPU engine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from toktier.backends.fast_cpu import _VendoredEngine

ROOT = Path(__file__).resolve().parents[2]


class _Native:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def encode_batch_list(
        self, texts: list[str], *, parallel: bool
    ) -> list[list[int]]:
        self.calls.append((texts, parallel))
        return [[len(text), index] for index, text in enumerate(texts)]

    @property
    def vocab(self) -> dict[int, bytes]:
        return {0: b"a"}

    @property
    def vocab_size(self) -> int:
        return 1


def test_vendored_adapter_uses_dependency_free_list_surface() -> None:
    native = _Native()
    engine = _VendoredEngine(native)

    assert list(engine.encode("hello")) == [5, 0]
    assert [list(row) for row in engine.encode_batch(["a", "bb"])] == [
        [1, 0],
        [2, 1],
    ]
    assert native.calls == [(["hello"], False), (["a", "bb"], True)]
    assert engine.vocab == {0: b"a"}
    assert engine.vocab_size == 1


def test_facts_hash_the_vendored_module_without_importing_it() -> None:
    script = """
import json
import sys
from toktier.backends.fast_cpu import ENGINE_MODULE, fast_cpu_engine_facts
before = ENGINE_MODULE in sys.modules
facts = fast_cpu_engine_facts()
after = ENGINE_MODULE in sys.modules
print(json.dumps({
    "before": before,
    "after": after,
    "version": facts.version,
    "digest": facts.binary_digest,
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    report: dict[str, Any] = json.loads(completed.stdout)
    assert report == {
        "before": False,
        "after": False,
        "version": "0.10.0+toktier.pinned.1",
        "digest": (
            "9a701047dafa1cdebc168851d0548a0ca"
            "af08d0523d70911cc7a24112ccf92a3"
        ),
    }


def test_vendor_package_has_no_import_side_effects() -> None:
    script = """
import json
import sys
import toktier._vendor
print(json.dumps(sorted(name for name in sys.modules if name.startswith("gigatoken"))))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == []
