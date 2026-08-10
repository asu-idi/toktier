"""Core-wheel identity contract of the integrated corrected CPU engine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from _support import byte_level_document, write_artifact

ROOT = Path(__file__).resolve().parents[2]


def _binding() -> dict[str, Any]:
    value = json.loads((ROOT / "tools/fast_cpu_binding.json").read_text())
    assert isinstance(value, dict)
    return value


def test_integrated_build_facts_match_the_active_certificate() -> None:
    from toktier.backends.fast_cpu import (
        ENGINE_DELIVERY,
        ENGINE_MODULE,
        fast_cpu_engine_facts,
    )

    binding = _binding()
    facts = fast_cpu_engine_facts()
    assert ENGINE_MODULE == "toktier._native"
    assert ENGINE_DELIVERY == "integrated"
    assert facts.version == binding["engine_version"]
    assert facts.binary_digest is None
    assert facts.source_digest == binding["source_digest"]
    assert list(facts.build_flags) == binding["build_flags"]
    assert facts.toolchain == binding["toolchain"]
    assert facts.config_digest is not None


def test_source_identity_is_independently_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools/fast_cpu_source_identity.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == _binding()["source_digest"]


def test_probe_imports_only_the_one_core_extension() -> None:
    script = """
import json
import sys
from toktier.backends.fast_cpu import fast_cpu_engine_facts
before = 'toktier._native' in sys.modules
facts = fast_cpu_engine_facts()
print(json.dumps({
    'before': before,
    'after': 'toktier._native' in sys.modules,
    'legacy': 'toktier._vendor.gigatoken_rs' in sys.modules,
    'source_digest': facts.source_digest,
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
        "after": True,
        "legacy": False,
        "source_digest": _binding()["source_digest"],
    }


def test_vendor_namespace_has_no_legacy_engine_import_side_effects() -> None:
    script = """
import json
import sys
import toktier._vendor
print(json.dumps(sorted(name for name in sys.modules if 'gigatoken_rs' in name)))
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


def test_native_materialization_reads_the_verified_artifact_without_transformers(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from toktier.backends import fast_cpu

    document = byte_level_document()
    artifact = write_artifact(tmp_path, document)
    backend = fast_cpu.FastCpuBackend.open(artifact)

    def unexpected_loader(_root: Path) -> object:
        raise AssertionError("the artifact-only path must not import transformers")

    monkeypatch.setattr(fast_cpu, "_load_live_tokenizer", unexpected_loader)
    assert json.loads(backend.materialized_tokenizer_json()) == document


def test_native_engine_initializes_one_core_but_defers_batch_workers() -> None:
    from toktier import _native
    from toktier.repair.registry import pclass_table

    document = byte_level_document()
    raw = json.dumps(document).encode("utf-8")
    encoder = _native.CallbackEncoder.native_fast_cpu(
        raw,
        "test_family",
        "a" * 64,
        8,
        1,
        False,
        pclass_table(),
    )
    assert bool(encoder.engine_initialized)
    assert encoder.batch_worker_count == 0
    assert encoder.encode("abc") == [97, 98, 99]
    assert encoder.encode_batch(["a"] * 32) == [[97]] * 32
    assert encoder.batch_worker_count == 1


def test_shared_reference_must_match_the_certified_artifact(
    tmp_path: Path,
) -> None:
    import hashlib

    import pytest

    from toktier import _native
    from toktier.repair.registry import pclass_table

    left = json.dumps(byte_level_document()).encode("utf-8")
    right_document = byte_level_document()
    right_document["added_tokens"] = [
        {
            "id": 256,
            "content": "<different>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    ]
    right = json.dumps(right_document).encode("utf-8")
    right_path = tmp_path / "right.json"
    right_path.write_bytes(right)
    reference = _native.ReferenceEngine(str(right_path))
    with pytest.raises(ValueError, match="does not belong to tokenizer_json"):
        _native.CallbackEncoder.native_fast_cpu(
            left,
            "test_family",
            hashlib.sha256(left).hexdigest(),
            8,
            1,
            False,
            pclass_table(),
            reference,
        )
