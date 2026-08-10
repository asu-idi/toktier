"""Facade-level JIT delivery: default lookup must execute the GPU plan.

The 0.2.0 pre-release stranger evaluation found that a ``gpu-jit``
facade with an admitted GPU plan could serve a large default-lookup
request from the store's CPU encoder while ``explain()`` still
headlined the GPU plan (``backend_basis: "plan"``, ``loaded: false``).
The test here reproduces that exact scenario -- experimental policy,
JIT delivery, explicit ``device="cuda"``, an input above the 64 KiB
crossover, and the DEFAULT lookup behavior -- and requires the honest
outcome: the JIT kernel loads, the GPU executes, the execution ledger
says so, and the IDs equal the frozen reference.

Every test is marked ``gpu`` and additionally needs ``ninja`` (the
``gpu-jit`` extra); it is skipped otherwise. The file stays ASCII, as
the repository's hygiene scan requires, so sample text uses escapes.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.gpu

FAMILY = "qwen3_8b"


@pytest.fixture(scope="module")
def jit_facade_config(
    artifact_handles: dict[str, Any],
    tmp_path_factory: pytest.TempPathFactory,
) -> Any:
    """An offline facade configuration with the frozen artifact cached.

    The verified bytes come from the GPU tier's artifact manifest; they
    are pre-placed under the shipped manifest's cache layout so
    ``toktier.load`` resolves them without any network access.
    """
    pytest.importorskip("ninja", reason="JIT delivery needs the gpu-jit extra")
    if FAMILY not in artifact_handles:
        pytest.skip(f"the artifact manifest does not pin {FAMILY}")
    handle = artifact_handles[FAMILY]
    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.backends.protocol import TOKENIZER_FILE
    from toktier.config import Config
    from toktier.paths import artifact_cache_dir

    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    entry = manifest.get(FAMILY)
    if entry.file(TOKENIZER_FILE).sha256 != handle.artifact_sha256:
        pytest.skip(
            f"the manifest's {FAMILY} artifact differs from the shipped pin"
        )
    base = tmp_path_factory.mktemp("facade-jit")
    config = Config(home=base / "home", offline=True)
    directory = artifact_cache_dir(config) / entry.directory_name
    directory.mkdir(parents=True)
    tokenizer_json = directory / TOKENIZER_FILE
    tokenizer_json.write_bytes(handle.path(TOKENIZER_FILE).read_bytes())
    return {"config": config, "tokenizer_json": tokenizer_json}


def test_jit_default_lookup_executes_gpu_and_reports_honestly(
    jit_facade_config: dict[str, Any],
) -> None:
    import tokenizers

    import toktier

    reference = tokenizers.Tokenizer.from_file(
        str(jit_facade_config["tokenizer_json"])
    )

    # The evaluation's shape: explicit CUDA, experimental policy for the
    # locally judged-or-waived JIT toolchain, input above 64 KiB.
    text = "Explicit CUDA with default lookup. \u4e2d\U0001f642 " * 4096
    assert len(text.encode("utf-8")) >= 64 * 1024

    tokenizer = toktier.load(
        FAMILY,
        device="cuda",
        policy="experimental",
        gpu_delivery="jit",
        config=jit_facade_config["config"],
    )
    try:
        encoding = tokenizer.encode(text)  # DEFAULT lookup behavior

        expected = list(reference.encode(text, add_special_tokens=False).ids)
        assert list(encoding.ids) == expected

        report = tokenizer.explain()
        assert report["backend"] == "gpu"
        assert report["backend_basis"] == "last_execution"
        assert report["kernel_delivery"] == "jit"
        gpu_report = report["gpu_backend"]
        assert isinstance(gpu_report, dict)
        assert gpu_report["loaded"] is True
        assert gpu_report["load_error"] is None
        runtime = report["runtime_policy"]
        assert isinstance(runtime, dict)
        counts = runtime["execution_counts"]
        assert isinstance(counts, dict)
        assert counts.get("gpu", 0) >= 1
        last = runtime["last_execution"]
        assert isinstance(last, dict)
        assert last["executed_backend"] == "gpu"

        # A shorter follow-up extension is still served from the store.
        grown = text + " short extension"
        followup = tokenizer.encode(grown)
        assert list(followup.ids) == list(
            reference.encode(grown, add_special_tokens=False).ids
        )
        store = tokenizer.explain()["store"]
        assert isinstance(store, dict)
        assert store["auto_appends"] >= 1
    finally:
        tokenizer.close()


def test_jit_lookup_off_still_executes_gpu(
    jit_facade_config: dict[str, Any],
) -> None:
    """The evaluation's working control keeps working after the fix."""
    import tokenizers

    import toktier

    reference = tokenizers.Tokenizer.from_file(
        str(jit_facade_config["tokenizer_json"])
    )
    text = "Store-free JIT control. \u4e2d\U0001f642 " * 4096

    tokenizer = toktier.load(
        FAMILY,
        device="cuda",
        policy="experimental",
        gpu_delivery="jit",
        config=jit_facade_config["config"],
    )
    try:
        encoding = tokenizer.encode(text, lookup="off")
        assert list(encoding.ids) == list(
            reference.encode(text, add_special_tokens=False).ids
        )
        report = tokenizer.explain()
        assert report["backend"] == "gpu"
        assert report["backend_basis"] == "last_execution"
    finally:
        tokenizer.close()
