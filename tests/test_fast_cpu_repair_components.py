"""The fast CPU backend refuses the callback-lane repair components in a
typed form when its engine belongs to the native request runtime.

Artifact tier only (``TOKTIER_TEST_ARTIFACTS``, see ``test_chunked.py``).
The native request runtime builds the backend's engine without the live
tokenizer object (``native_session_engine``). If that runtime then does not
serve the tokenizer -- the prebuilt GPU delivery in a process where a kernel
was already loaded through the JIT host (``tests/gpu/test_gpu_delivery_mix.py``)
-- the Python-owned store asks for ``repair_components``. That request is
refused with ``BackendExecutionFault``, which the store turns into the
reference path with the reason recorded, instead of an internal
``RuntimeError``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import toktier
from toktier import Config
from toktier.artifacts.manifest import ArtifactManifest
from toktier.artifacts.tables import ARTIFACT_MANIFEST
from toktier.errors import BackendExecutionFault
from toktier.paths import artifact_cache_dir

from .test_chunked import _artifact_source


def test_an_engine_of_the_native_runtime_refuses_callback_components(
    tmp_path: Path,
) -> None:
    source = _artifact_source("qwen3_8b")
    if source is None:
        pytest.skip("qwen3_8b: artifact not available under TOKTIER_TEST_ARTIFACTS")
    assert source is not None
    config = Config(home=tmp_path, offline=True)
    cache = artifact_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    entry = ArtifactManifest.load(ARTIFACT_MANIFEST).get("qwen3_8b")
    shutil.copytree(source, cache / entry.directory_name)
    tokenizer = toktier.load(
        "qwen3_8b", config=config, device="cpu", policy="experimental"
    )
    try:
        backend = tokenizer._fast_backend
        assert backend is not None
        engine = backend.native_session_engine(None)
        assert engine is not None
        with pytest.raises(BackendExecutionFault) as caught:
            backend.repair_components()
        assert caught.value.details["reason"] == (
            "engine_owned_by_native_request_runtime"
        )
        assert "native request runtime" in str(caught.value)
    finally:
        tokenizer.close()
