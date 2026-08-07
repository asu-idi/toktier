"""Certified Gigatoken session routing and experimental Fastokens policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from toktier import records
from toktier.backends.fast_cpu import FastCpuBackend, FastCpuEngineFacts
from toktier.errors import UnsupportedConfig
from toktier.facade import api as facade_api
from toktier.policy import BACKEND_FAST_CPU, BACKEND_GPU, BACKEND_REFERENCE
from toktier.repair.gigatoken import GigatokenRepair
from toktier.repair.registry import RepairFamily
from toktier.routing.probe import DeviceInfo, KernelCacheState, ProbeSnapshot
from toktier.routing.registry_view import (
    ArtifactRecord,
    BackendEntry,
    OracleRecord,
    RegistryView,
)

from .conftest import Rig

_ENGINE_VERSION = "0.10.0+toktier.pinned.1"
_BINARY_DIGEST = "a" * 64
_CONFIG_DIGEST = "b" * 64
_GPU_DIGEST = "d" * 64


class _CountingEngine:
    """Gigatoken-shaped engine over the real tiny HF fixture."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.calls = 0
        self.fail = False
        self._vocab = {token_id: b"x" for token_id in range(256)}

    def encode(self, text: str) -> Sequence[int]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic window fault")
        return [
            int(value)
            for value in self.tokenizer.encode(text, add_special_tokens=False).ids
        ]

    def encode_batch(self, texts: Sequence[str]) -> Sequence[Sequence[int]]:
        return [self.encode(text) for text in texts]

    @property
    def vocab(self) -> Mapping[int, bytes]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


class _FakeFastBackend:
    backend_id = BACKEND_FAST_CPU

    def __init__(self, family: str, engine: _CountingEngine, tokenizer: Any) -> None:
        self.family = family
        self.engine = engine
        self.tokenizer = tokenizer

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [int(value) for value in self.engine.encode(text)]

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        del add_special_tokens
        return [
            [int(value) for value in row] for row in self.engine.encode_batch(texts)
        ]

    def repair_components(self) -> tuple[_CountingEngine, Any]:
        return self.engine, self.tokenizer

    def close(self) -> None:
        return None


class _FakeGpuBackend:
    backend_id = BACKEND_GPU

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.calls: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        self.calls.append(text)
        return [
            int(value)
            for value in self.tokenizer.encode(
                text, add_special_tokens=add_special_tokens
            ).ids
        ]

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        return [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]

    def close(self) -> None:
        return None


def _registry(rig: Rig) -> RegistryView:
    return RegistryView(
        artifacts=(
            ArtifactRecord(
                artifact_sha256=rig.artifact_sha256,
                family=rig.family,
                pipeline_id="pipeline",
                added_frontend_id="frontend",
                oracle_id="hf-0.22.2",
                suite_version="test",
                evidence_id="test",
                backends={
                    BACKEND_FAST_CPU: BackendEntry(
                        status="certified",
                        binary_digest=_BINARY_DIGEST,
                        engine="gigatoken",
                        engine_version=_ENGINE_VERSION,
                        engine_delivery="vendored",
                        engine_module="toktier._vendor.gigatoken_rs",
                        config_id="toktier-fast-repair-v1",
                        config_digest=_CONFIG_DIGEST,
                    )
                },
            ),
        ),
        oracles=(
            OracleRecord(
                oracle_id="hf-0.22.2",
                package="tokenizers",
                certified_versions=("0.22.2",),
                semantic_id="test",
            ),
        ),
    )


def _gpu_registry(rig: Rig) -> RegistryView:
    base = _registry(rig)
    original = base.certification(artifact_sha256=rig.artifact_sha256)
    assert original is not None
    record = original.record
    return RegistryView(
        artifacts=(
            ArtifactRecord(
                artifact_sha256=record.artifact_sha256,
                family=record.family,
                pipeline_id=record.pipeline_id,
                added_frontend_id=record.added_frontend_id,
                oracle_id=record.oracle_id,
                suite_version=record.suite_version,
                evidence_id=record.evidence_id,
                backends={
                    **record.backends,
                    BACKEND_GPU: BackendEntry(
                        status="certified",
                        binary_digest=_GPU_DIGEST,
                        devices=("sm_120",),
                    ),
                },
            ),
        ),
        oracles=(
            OracleRecord(
                oracle_id="hf-0.22.2",
                package="tokenizers",
                certified_versions=("0.22.2",),
                semantic_id="test",
            ),
        ),
    )


def _spec(rig: Rig) -> RepairFamily:
    return RepairFamily(
        family=rig.family,
        artifact_sha256=rig.artifact_sha256,
        margin=4,
        effective_l_max=2,
        has_normalizer=False,
        source_table_sha256="c" * 64,
        window_chars=128,
        max_retries=2,
        min_match_tokens=2,
    )


def _install_certified_probe(
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    registry: RegistryView,
    *,
    binary_digest: str = _BINARY_DIGEST,
) -> None:
    match = registry.certification(artifact_sha256=rig.artifact_sha256)
    assert match is not None
    snapshot = ProbeSnapshot(
        family=rig.family,
        artifact_sha256=rig.artifact_sha256,
        oracle_version="0.22.2",
        importable_backends=frozenset({BACKEND_REFERENCE, BACKEND_FAST_CPU}),
        fast_cpu_engine=FastCpuEngineFacts(
            version=_ENGINE_VERSION,
            binary_digest=binary_digest,
            config_digest=_CONFIG_DIGEST,
        ),
        certification=match,
    )
    monkeypatch.setattr(facade_api, "shipped_registry", lambda: registry)
    monkeypatch.setattr(facade_api, "probe", lambda **_keywords: snapshot)


def _install_gpu_probe(
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    registry: RegistryView,
) -> None:
    match = registry.certification(artifact_sha256=rig.artifact_sha256)
    assert match is not None
    snapshot = ProbeSnapshot(
        family=rig.family,
        artifact_sha256=rig.artifact_sha256,
        oracle_version="0.22.2",
        importable_backends=frozenset(
            {BACKEND_REFERENCE, BACKEND_FAST_CPU, BACKEND_GPU}
        ),
        devices=(DeviceInfo(0, "test GPU", "sm_120"),),
        devices_probed=True,
        kernel_cache=KernelCacheState(
            binary_digest=_GPU_DIGEST,
            prebuilt_available=True,
            preferred_delivery="prebuilt",
        ),
        fast_cpu_engine=FastCpuEngineFacts(
            version=_ENGINE_VERSION,
            binary_digest=_BINARY_DIGEST,
            config_digest=_CONFIG_DIGEST,
        ),
        certification=match,
    )
    monkeypatch.setattr(facade_api, "shipped_registry", lambda: registry)
    monkeypatch.setattr(facade_api, "probe", lambda **_keywords: snapshot)


def test_public_session_append_executes_certified_gigatoken_callback(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tokenizers

    live = tokenizers.Tokenizer.from_file(str(rig.artifact_path))
    engine = _CountingEngine(live)
    backend = _FakeFastBackend(rig.family, engine, live)
    registry = _registry(rig)
    _install_certified_probe(monkeypatch, rig, registry)
    monkeypatch.setattr(facade_api, "family_spec", lambda *_args: _spec(rig))
    monkeypatch.setattr(
        FastCpuBackend,
        "open",
        classmethod(lambda _cls, _artifact: backend),
    )

    tokenizer = rig.tokenizer(store=rig.store_path())
    base = ("alpha 123 beta 456 " * 60) + "tail"
    grown = base + " appended text 789"
    expected = live.encode(grown, add_special_tokens=False).ids

    tokenizer.encode(base, session="chat")
    actual = tokenizer.encode(grown, session="chat")

    assert list(actual.ids) == list(expected)
    assert engine.calls >= 1
    report = tokenizer.explain()
    repair = report["session_repair"]
    assert isinstance(repair, dict)
    assert repair["status"] == "active"
    assert repair["backend"] == BACKEND_FAST_CPU
    assert repair["path_counts"]["gigatoken_repair"] == 1
    assert "hf_full_window_covers_all" not in repair["path_counts"]
    state = report["state_encode"]
    assert isinstance(state, dict)
    last_state = state["last"]
    assert isinstance(last_state, dict)
    assert last_state["backend"] == BACKEND_FAST_CPU
    (record_path,) = (rig.store_path() / "entries").glob("*.rec")
    assert records.decode_record(record_path.read_bytes()).witness_category == 1


def test_automatic_facade_routes_short_cpu_and_large_gpu(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One complete install exposes the size router without extra flags."""
    import tokenizers

    live = tokenizers.Tokenizer.from_file(str(rig.artifact_path))
    engine = _CountingEngine(live)
    fast_backend = _FakeFastBackend(rig.family, engine, live)
    gpu_backend = _FakeGpuBackend(live)
    registry = _gpu_registry(rig)
    _install_gpu_probe(monkeypatch, rig, registry)
    monkeypatch.setattr(facade_api, "family_spec", lambda *_args: _spec(rig))
    monkeypatch.setattr(
        FastCpuBackend,
        "open",
        classmethod(lambda _cls, _artifact: fast_backend),
    )
    monkeypatch.setattr(
        facade_api.Tokenizer,
        "_open_gpu_backend",
        lambda _self: gpu_backend,
    )

    tokenizer = rig.tokenizer(gpu_min_bytes=64)
    small = "short request"
    large = "long request " * 20
    assert list(tokenizer.encode(small, lookup="off").ids) == list(
        live.encode(small, add_special_tokens=False).ids
    )
    assert list(tokenizer.encode(large, lookup="off").ids) == list(
        live.encode(large, add_special_tokens=False).ids
    )

    assert tokenizer.plan.fallback_chain == (
        BACKEND_GPU,
        BACKEND_FAST_CPU,
        BACKEND_REFERENCE,
    )
    assert gpu_backend.calls == [large]
    assert engine.calls == 1
    report = tokenizer.explain()
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    assert runtime["gpu_min_bytes"] == 64
    assert runtime["execution_counts"] == {
        BACKEND_FAST_CPU: 1,
        BACKEND_GPU: 1,
    }
    gpu_report = report["gpu_backend"]
    assert isinstance(gpu_report, dict)
    assert gpu_report["loaded"] is True


def test_gpu_cold_session_is_seeded_then_appended_by_gigatoken(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requested GPU-cold -> CPU-repair state transition is real."""
    import tokenizers

    live = tokenizers.Tokenizer.from_file(str(rig.artifact_path))
    engine = _CountingEngine(live)
    fast_backend = _FakeFastBackend(rig.family, engine, live)
    gpu_backend = _FakeGpuBackend(live)
    registry = _gpu_registry(rig)
    _install_gpu_probe(monkeypatch, rig, registry)
    monkeypatch.setattr(facade_api, "family_spec", lambda *_args: _spec(rig))
    monkeypatch.setattr(
        FastCpuBackend,
        "open",
        classmethod(lambda _cls, _artifact: fast_backend),
    )
    monkeypatch.setattr(
        facade_api.Tokenizer,
        "_open_gpu_backend",
        lambda _self: gpu_backend,
    )

    tokenizer = rig.tokenizer(store=rig.store_path(), gpu_min_bytes=64)
    base = ("alpha 123 beta 456 " * 60) + "tail"
    grown = base + " appended text 789"
    assert list(tokenizer.encode(base, session="chat").ids) == list(
        live.encode(base, add_special_tokens=False).ids
    )
    assert list(tokenizer.encode(grown, session="chat").ids) == list(
        live.encode(grown, add_special_tokens=False).ids
    )

    # Cold state came from GPU exactly once. The strict append bypassed
    # the size router and ran the corrected Gigatoken repair callback.
    assert gpu_backend.calls == [base]
    assert engine.calls >= 1
    report = tokenizer.explain()
    state = report["state_encode"]
    assert isinstance(state, dict)
    assert state["counts"] == {"accelerated_with_reconstructed_spans": 1}
    last_state = state["last"]
    assert isinstance(last_state, dict)
    assert last_state["backend"] == BACKEND_GPU
    repair = report["session_repair"]
    assert isinstance(repair, dict)
    assert repair["backend"] == BACKEND_FAST_CPU
    assert repair["path_counts"]["gigatoken_repair"] == 1


def test_binding_mismatch_keeps_public_session_on_reference(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(rig)
    _install_certified_probe(monkeypatch, rig, registry, binary_digest="d" * 64)

    def unexpected_open(*_args: object, **_keywords: object) -> None:
        raise AssertionError("mismatched Gigatoken binding must not be opened")

    monkeypatch.setattr(FastCpuBackend, "open", unexpected_open)
    tokenizer = rig.tokenizer()
    tokenizer.encode("hello", session="chat")
    report = tokenizer.explain()

    assert tokenizer.plan.backend == BACKEND_REFERENCE
    assert report["session_repair"] == {
        "status": "reference_only",
        "backend": BACKEND_REFERENCE,
    }
    reasons = report["plan_reasons"]
    assert isinstance(reasons, list)
    assert any(
        reason["backend"] == BACKEND_FAST_CPU
        and reason["code"] == "R_ENGINE_BINDING_MISMATCH"
        and reason["detail"]["axis"] == "binary_digest"
        for reason in reasons
    )


def test_fastokens_requires_explicit_experimental_policy(rig: Rig) -> None:
    with pytest.raises(UnsupportedConfig) as caught:
        rig.tokenizer(repair_backend="fastokens")
    assert caught.value.details == {
        "option": "repair_backend",
        "value": "fastokens",
        "required_policy": "experimental",
        "exact_id_guarantee": False,
    }


def test_gigatoken_guard_does_not_swallow_reference_errors(rig: Rig) -> None:
    import tokenizers

    live = tokenizers.Tokenizer.from_file(str(rig.artifact_path))
    engine = _CountingEngine(live)

    def reference(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        del text
        raise LookupError("reference failure must remain visible")

    repair = GigatokenRepair(
        spec=_spec(rig),
        engine=engine,
        hf_tokenizer=live,
        reference_encode=reference,
    )
    base = "alpha 123 beta 456 " * 60
    encoded = live.encode(base, add_special_tokens=False)
    engine.fail = True
    with pytest.raises(LookupError, match="must remain visible"):
        repair(
            base,
            [int(value) for value in encoded.ids],
            [(int(a), int(b)) for a, b in encoded.offsets],
            " append",
        )


def test_fastokens_version_changes_the_persistent_fingerprint(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = rig.tokenizer()

    class Repair:
        config_id = "toktier-fastokens-full-experimental-v1"

        def __init__(self, version: str) -> None:
            self.version = version

        def stats(self) -> dict[str, object]:
            return {"backend": "fastokens", "engine_version": self.version}

    first = tokenizer._semantic_fingerprint(Repair("0.3.1"))  # type: ignore[arg-type]
    second = tokenizer._semantic_fingerprint(Repair("0.3.2"))  # type: ignore[arg-type]
    assert first != second
