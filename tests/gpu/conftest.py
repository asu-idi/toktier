"""Shared fixtures and markers for the GPU backend tests.

Two kinds of test live here:

- **host tests**, which need neither a GPU nor torch. They cover the
  routing data, the digests, the loader's flag-set rules and the
  argument checking. They run in ordinary CI.
- **GPU tests**, marked ``@pytest.mark.gpu``. They compile the kernel and
  compare against the reference tokenizer. They are skipped unless a
  CUDA device and the frozen artifacts are both present.

Run the GPU suite with::

    pytest tests/gpu -m gpu --artifact-manifest PATH

``--artifact-manifest`` names a real artifact manifest
(``toktier.artifacts.manifest.ArtifactManifest``): per-file sha256 for
every file, with ``local_dir`` pointing at the frozen artifact tree. The
fixtures resolve it through ``ArtifactStore`` -- fetch from the local
directory, hash-verify, install -- so the engine under test consumes
exactly the verified handles production consumes. A digest-free
``{family: {local_dir}}`` mapping is rejected at load time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:  # editable-install-free test run
    sys.path.insert(0, str(_SRC))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: needs a CUDA device, torch, and the frozen tokenizer artifacts",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--artifact-manifest",
        action="store",
        default=None,
        help="Path to the tokenizer artifact manifest used by GPU tests.",
    )
    parser.addoption(
        "--class-table-dir",
        action="store",
        default=None,
        help="Directory holding the generated lookup tables.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip GPU-marked tests unless the machine can actually run them."""
    reason = _gpu_unavailable_reason(config)
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        # The marker, not the keyword set: these tests live in a
        # directory called "gpu", so the directory name is a keyword on
        # every item here and would skip the host tests too.
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip)


def _gpu_unavailable_reason(config: pytest.Config) -> str | None:
    try:
        import torch
    except ImportError:
        return "torch is not installed (the gpu-jit extra is absent)"
    if not torch.cuda.is_available():
        return "no CUDA device is available"
    if config.getoption("--artifact-manifest") is None:
        return "no --artifact-manifest was given"
    return None


@pytest.fixture(scope="session")
def artifact_handles(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Verified handles for every family the manifest pins.

    Built through the real chain: ``ArtifactManifest`` (rejects entries
    without per-file digests) -> ``ArtifactStore`` with a local source
    (hash-verify and install into a session cache) ->
    ``verified_handle``. What the GPU tests encode against is therefore
    verified content, not a trusted directory.
    """
    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.artifacts.sources import LocalDirectorySource
    from toktier.artifacts.store import ArtifactStore
    from toktier.config import Config
    from toktier.engine.gpu.handles import verified_handles
    from toktier.policy import RoutingPolicy

    path = request.config.getoption("--artifact-manifest")
    if path is None:
        pytest.skip("no --artifact-manifest was given")
    manifest = ArtifactManifest.load(Path(str(path)))
    cache_root = tmp_path_factory.mktemp("toktier-artifacts")
    config = Config(
        home=None,
        cache_dir=cache_root / "cache",
        state_dir=cache_root / "state",
        offline=False,
        log_level="WARNING",
        disable_gpu=False,
        diagnostics=False,
        routing_policy=RoutingPolicy.CERTIFIED,
    )
    store = ArtifactStore(manifest, config=config, source=LocalDirectorySource())
    return dict(verified_handles(store, manifest.families()))


@pytest.fixture(scope="session")
def class_table_dir(request: pytest.FixtureRequest) -> Path | None:
    """Where the generated lookup tables live, when given explicitly."""
    value = request.config.getoption("--class-table-dir")
    return Path(str(value)) if value else None


@pytest.fixture(scope="session")
def gpu_engine(
    artifact_handles: dict[str, Any],
    class_table_dir: Path | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> Any:
    """One engine for the whole session: one process, one kernel build."""
    from toktier.engine.gpu.engine import GpuEngine

    return GpuEngine.create(
        artifact_handles,
        cache_dir=tmp_path_factory.mktemp("toktier-cache"),
        class_table_dir=class_table_dir,
    )
