"""Certified Gigatoken session routing and experimental Fastokens policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from toktier import records
from toktier.backends.fast_cpu import FastCpuBackend, FastCpuEngineFacts
from toktier.errors import UnsupportedConfig
from toktier.facade import api as facade_api
from toktier.facade.recovery import RecoveryBinding
from toktier.policy import BACKEND_FAST_CPU, BACKEND_GPU, BACKEND_REFERENCE
from toktier.repair.gigatoken import GigatokenRepair, WindowUnsupported
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
_SOURCE_DIGEST = "a" * 64
_BUILD_FLAGS = ("profile=release", "opt-level=3")
_TOOLCHAIN = "rustc 1.93.1 (test fixture)"
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


class _NativeCapableFastBackend(_FakeFastBackend):
    """Callback-protocol fake that also offers the native surface.

    The production ``FastCpuBackend`` always supplies
    ``materialized_tokenizer_json``, so ``_native_repair_encoder`` can
    build the pure-native session encoder from it. Tests that need to
    observe which store lane the facade selects use this fake: with the
    plain ``_FakeFastBackend`` the native lane is never even possible.
    """

    def __init__(
        self,
        family: str,
        engine: _CountingEngine,
        tokenizer: Any,
        tokenizer_json: str,
    ) -> None:
        super().__init__(family, engine, tokenizer)
        self._tokenizer_json = tokenizer_json

    def materialized_tokenizer_json(self) -> str:
        return self._tokenizer_json


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
                        status="certified_source",
                        source_digest=_SOURCE_DIGEST,
                        build_flags=_BUILD_FLAGS,
                        toolchain=_TOOLCHAIN,
                        engine="gigatoken",
                        engine_version=_ENGINE_VERSION,
                        engine_delivery="integrated",
                        engine_module="toktier._native",
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
    source_digest: str = _SOURCE_DIGEST,
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
            source_digest=source_digest,
            build_flags=_BUILD_FLAGS,
            toolchain=_TOOLCHAIN,
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
            source_digest=_SOURCE_DIGEST,
            build_flags=_BUILD_FLAGS,
            toolchain=_TOOLCHAIN,
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


def test_added_token_store_seed_reports_the_actual_reference_result(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct HF store seeding is visible in the shared runtime ledger."""
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
    monkeypatch.setattr(tokenizer._added_router, "holds_literal", lambda _text: True)
    text = "alpha <synthetic-added-token> beta"
    actual = tokenizer.encode(text, session="chat")

    assert list(actual.ids) == list(live.encode(text, add_special_tokens=False).ids)
    report = tokenizer.explain()
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    assert runtime["execution_counts"] == {BACKEND_REFERENCE: 1}
    assert runtime["last_execution"] == {
        "input_bytes": len(text.encode("utf-8")),
        "selected_start": BACKEND_FAST_CPU,
        "executed_backend": BACKEND_REFERENCE,
        "source": "state_encode",
        "path": "hf_added_token",
    }
    assert report["fallback_counts"] == {"R_INPUT_ADDED_TOKEN": 1}
    state = report["state_encode"]
    assert isinstance(state, dict)
    assert state["last"] == {"path": "hf_added_token"}


def test_span_guard_reclassifies_store_seed_as_the_reference_result(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A discarded accelerated seed is not reported as the final backend."""
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

    def refuse_spans(
        _self: GigatokenRepair, _text: str, _ids: Sequence[int]
    ) -> list[tuple[int, int]]:
        raise WindowUnsupported("synthetic span guard")

    monkeypatch.setattr(GigatokenRepair, "spans_for_ids", refuse_spans)
    tokenizer = rig.tokenizer(store=rig.store_path())
    text = "alpha 123 beta 456 " * 60
    actual = tokenizer.encode(text, session="chat")

    assert list(actual.ids) == list(live.encode(text, add_special_tokens=False).ids)
    report = tokenizer.explain()
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    assert runtime["execution_counts"] == {BACKEND_REFERENCE: 1}
    assert runtime["last_execution"] == {
        "input_bytes": len(text.encode("utf-8")),
        "selected_start": BACKEND_FAST_CPU,
        "executed_backend": BACKEND_REFERENCE,
        "source": "state_encode",
        "path": "hf_span_guard",
    }
    assert report["fallback_counts"] == {"R_INPUT_GUARD_ROUTED": 1}
    state = report["state_encode"]
    assert isinstance(state, dict)
    assert state["last"] == {
        "path": "hf_span_guard",
        "error": "WindowUnsupported",
        "message": "synthetic span guard",
    }


def test_long_session_seals_natively_and_keeps_gigatoken_repair_active(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tail above the hard cap is sealed before its first strict append."""
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
    base = ("alpha 123 beta 456 " * 4000) + "tail"
    grown = base + " appended text 789"
    tokenizer.encode(base, session="long")
    actual = tokenizer.encode(grown, session="long")
    assert list(actual.ids) == list(live.encode(grown, add_special_tokens=False).ids)

    report = tokenizer.explain()
    store = report["store"]
    assert isinstance(store, dict)
    assert store["append_paths"]["gigatoken_repair"] == 1
    assert tokenizer._entry_store is not None
    assert tokenizer._entry_store._store is not None
    native_stats = tokenizer._entry_store._store.stats()
    seals = native_stats["seals"]
    hard_cap_degrades = native_stats["hard_cap_degrades"]
    assert isinstance(seals, int)
    assert isinstance(hard_cap_degrades, int)
    assert seals >= 1
    assert hard_cap_degrades == 0
    repair = report["session_repair"]
    assert isinstance(repair, dict)
    assert repair["path_counts"]["gigatoken_repair"] == 1

    (record_path,) = (rig.store_path() / "entries").glob("*.rec")
    view = records.decode_record(record_path.read_bytes())
    assert view.witness_category == 1
    assert view.stable_prefix_byte_length > 0


def test_sealed_session_restores_and_repairs_after_process_boundary(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record-bound caller prefix restores a sealed native session."""
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

    directory = rig.store_path("sealed-session")
    base = ("alpha \u4f60\u597d caf\u00e9 123 beta 456 " * 4000) + "tail"
    sealed = base + " first append 789"
    grown = sealed + " second append after restart"

    first = rig.tokenizer(store=directory)
    first.encode(base, session="long")
    assert list(first.encode(sealed, session="long").ids) == list(
        live.encode(sealed, add_special_tokens=False).ids
    )
    (record_path,) = (directory / "entries").glob("*.rec")
    assert records.decode_record(record_path.read_bytes()).stable_prefix_byte_length > 0
    assert len(list((directory / "entries").glob("*.binding"))) == 1

    # A new facade owns a new native store, matching a process restart.
    second = rig.tokenizer(store=directory)
    assert list(second.encode(grown, session="long").ids) == list(
        live.encode(grown, add_special_tokens=False).ids
    )
    store = second.explain()["store"]
    assert isinstance(store, dict)
    assert store["session_misses"] == 0
    assert store["session_appends"] == 1
    assert store["append_paths"]["gigatoken_repair"] == 1
    report = second.explain()
    state = report["state_encode"]
    assert isinstance(state, dict)
    last_state = state["last"]
    assert isinstance(last_state, dict)
    assert last_state["backend"] == BACKEND_FAST_CPU
    assert isinstance(last_state["input_bytes"], int)
    assert last_state["input_bytes"] < len(sealed.encode("utf-8"))


def test_sealed_content_lookup_restores_after_index_rebuild(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record-bound row rebuilds content lookup without plaintext."""
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

    directory = rig.store_path("sealed-auto")
    base = ("lookup prefix \u4f60\u597d 123 " * 4000) + "tail"
    sealed = base + " first extension"
    grown = sealed + " extension after restart"
    first = rig.tokenizer(store=directory)
    first.encode(base, lookup="auto")
    assert list(first.encode(sealed, lookup="auto").ids) == list(
        live.encode(sealed, add_special_tokens=False).ids
    )
    (directory / "index.json").unlink()

    second = rig.tokenizer(store=directory)
    assert list(second.encode(grown, lookup="auto").ids) == list(
        live.encode(grown, add_special_tokens=False).ids
    )
    store = second.explain()["store"]
    assert isinstance(store, dict)
    assert store["index_rebuilds"] == 1
    assert store["auto_misses"] == 0
    assert store["auto_appends"] == 1
    assert store["append_paths"]["gigatoken_repair"] == 1


@pytest.mark.parametrize("damage", ["missing", "corrupt", "wrong_digest"])
def test_missing_or_corrupt_sealed_binding_degrades_to_exact_cold_encode(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, damage: str
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

    directory = rig.store_path("corrupt-binding")
    base = ("binding corruption 123 " * 4000) + "tail"
    sealed = base + " first extension"
    first = rig.tokenizer(store=directory)
    first.encode(base, session="long")
    first.encode(sealed, session="long")
    (binding_path,) = (directory / "entries").glob("*.binding")
    if damage == "missing":
        binding_path.unlink()
    else:
        if damage == "corrupt":
            raw = bytearray(binding_path.read_bytes())
            raw[-1] ^= 0x01
            binding_path.write_bytes(bytes(raw))
        else:
            binding = RecoveryBinding.from_bytes(binding_path.read_bytes())
            wrong = bytearray(binding.text_digest)
            wrong[0] ^= 0x01
            binding_path.write_bytes(
                replace(binding, text_digest=bytes(wrong)).to_bytes()
            )

    grown = sealed + " after corruption"
    second = rig.tokenizer(store=directory)
    assert list(second.encode(grown, session="long").ids) == list(
        live.encode(grown, add_special_tokens=False).ids
    )
    store = second.explain()["store"]
    assert isinstance(store, dict)
    assert store["session_misses"] == 1
    assert store["session_appends"] == 0


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


def test_gpu_admitted_plan_routes_cold_auto_lookup_through_the_executor(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default-lookup cold miss on a GPU plan reaches the GPU backend.

    The store's full/seed encodes must consult the routed executor when
    the plan admits GPU: a large input is dispatched to the GPU backend
    and recorded in the execution ledger, so ``explain()`` headlines the
    execution rather than repeating the plan. The fast backend here
    carries the native materialization surface, exactly like the
    production ``FastCpuBackend``, so this test fails if the store
    silently prefers the store-internal CPU encoder instead.
    """
    import tokenizers

    live = tokenizers.Tokenizer.from_file(str(rig.artifact_path))
    engine = _CountingEngine(live)
    fast_backend = _NativeCapableFastBackend(
        rig.family, engine, live, rig.artifact_path.read_text(encoding="utf-8")
    )
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

    # JIT delivery keeps the Python adapter, as a gpu-jit install does.
    tokenizer = rig.tokenizer(gpu_min_bytes=64, gpu_delivery="jit")
    text = ("gpu content lookup 123 " * 300) + "tail"
    assert len(text.encode("utf-8")) >= 4096  # above the auto-store floor

    encoding = tokenizer.encode(text)  # default lookup

    assert list(encoding.ids) == list(live.encode(text, add_special_tokens=False).ids)
    # The routed executor ran and dispatched the admitted GPU backend.
    assert gpu_backend.calls == [text]
    report = tokenizer.explain()
    assert report["backend"] == BACKEND_GPU
    assert report["backend_basis"] == "last_execution"
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    assert runtime["execution_counts"] == {BACKEND_GPU: 1}
    last = runtime["last_execution"]
    assert isinstance(last, dict)
    assert last["executed_backend"] == BACKEND_GPU
    gpu_report = report["gpu_backend"]
    assert isinstance(gpu_report, dict)
    assert gpu_report["loaded"] is True
    state = report["state_encode"]
    assert isinstance(state, dict)
    assert state["counts"] == {"accelerated_with_reconstructed_spans": 1}
    # Appends stay on the certified Gigatoken witness lane, not the
    # pure-native store encoder.
    assert tokenizer._native_session_encoder is None
    repair = report["session_repair"]
    assert isinstance(repair, dict)
    assert repair["status"] == "active"
    assert repair["backend"] == BACKEND_FAST_CPU

    # A shorter follow-up extension is still served from the store.
    grown = text + " short extension"
    followup = tokenizer.encode(grown)
    assert list(followup.ids) == list(
        live.encode(grown, add_special_tokens=False).ids
    )
    store = tokenizer.explain()["store"]
    assert isinstance(store, dict)
    assert store["auto_appends"] == 1


def test_cpu_only_plan_keeps_the_native_store_encoder(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an admitted GPU backend the pure-native lane is kept.

    A CPU-only install must not regress: the store's full encodes stay
    on the GIL-free native session encoder, and the routed executor is
    not consulted for them.
    """
    import tokenizers

    live = tokenizers.Tokenizer.from_file(str(rig.artifact_path))
    engine = _CountingEngine(live)
    fast_backend = _NativeCapableFastBackend(
        rig.family, engine, live, rig.artifact_path.read_text(encoding="utf-8")
    )
    registry = _registry(rig)
    _install_certified_probe(monkeypatch, rig, registry)
    monkeypatch.setattr(facade_api, "family_spec", lambda *_args: _spec(rig))
    monkeypatch.setattr(
        FastCpuBackend,
        "open",
        classmethod(lambda _cls, _artifact: fast_backend),
    )
    # Pin the Python adapter, as installations without the one-call
    # native request surface use it; the store lane is then observable.
    monkeypatch.setattr(
        facade_api.Tokenizer, "_native_request_runtime", lambda _self: None
    )

    tokenizer = rig.tokenizer()
    text = ("cpu only content lookup 123 " * 300) + "tail"
    encoding = tokenizer.encode(text)  # default lookup

    assert list(encoding.ids) == list(live.encode(text, add_special_tokens=False).ids)
    # The pure-native encoder served the store; the executor stayed idle.
    assert tokenizer._native_session_encoder is not None
    runtime = tokenizer.explain()["runtime_policy"]
    assert isinstance(runtime, dict)
    assert runtime["execution_counts"] == {}
    assert engine.calls == 0
    repair = tokenizer.explain()["session_repair"]
    assert isinstance(repair, dict)
    assert repair["status"] == "active"
    assert repair["request_path"] == "rust_native"


def test_headline_backend_reports_the_below_threshold_cpu_execution(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GPU plan whose last request stayed on CPU headlines the CPU.

    The crossover is a per-input decision, so the plan alone cannot
    answer "what ran". The headline follows the execution ledger and
    ``planned_backend`` keeps the plan readable beside it.
    """
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
    planned = tokenizer.plan.backend
    assert planned == BACKEND_GPU

    # Before any request the report says so rather than guessing.
    cold = tokenizer.explain()
    assert cold["backend"] == BACKEND_GPU
    assert cold["backend_basis"] == "plan"
    assert cold["planned_backend"] == BACKEND_GPU

    tokenizer.encode("long request " * 20, lookup="off")
    tokenizer.encode("short request", lookup="off")

    report = tokenizer.explain()
    assert report["backend"] == BACKEND_FAST_CPU
    assert report["backend_basis"] == "last_execution"
    assert report["planned_backend"] == BACKEND_GPU
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    last = runtime["last_execution"]
    assert isinstance(last, dict)
    assert last["executed_backend"] == BACKEND_FAST_CPU
    assert report["backend"] == last["executed_backend"]


def test_headline_backend_reports_the_long_session_cpu_repair(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long session sealed on GPU, then repaired on CPU, headlines the CPU.

    The repair input is bounded, so it lands below the crossover and the
    corrected Gigatoken CPU engine returns the result even though the plan
    selected GPU for the transcript that seeded the session.
    """
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

    directory = rig.store_path("sealed-headline")
    base = ("alpha 123 beta 456 " * 4000) + "tail"
    sealed = base + " first append 789"
    grown = sealed + " second append after restart"

    first = rig.tokenizer(store=directory, gpu_min_bytes=4096)
    first.encode(base, session="long")
    first.encode(sealed, session="long")
    seeded = first.explain()
    assert seeded["backend"] == BACKEND_GPU
    assert seeded["backend_basis"] == "last_execution"

    # A new facade owns a new native store, matching a process restart.
    second = rig.tokenizer(store=directory, gpu_min_bytes=4096)
    assert list(second.encode(grown, session="long").ids) == list(
        live.encode(grown, add_special_tokens=False).ids
    )

    report = second.explain()
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    last = runtime["last_execution"]
    assert isinstance(last, dict)
    assert last["executed_backend"] == BACKEND_FAST_CPU
    assert last["selected_start"] == BACKEND_FAST_CPU
    # The ledger reports the bounded repair input, not the whole transcript.
    assert isinstance(last["input_bytes"], int)
    assert last["input_bytes"] < len(sealed.encode("utf-8"))
    assert report["backend"] == BACKEND_FAST_CPU
    assert report["backend_basis"] == "last_execution"
    assert report["planned_backend"] == BACKEND_GPU
    repair = report["session_repair"]
    assert isinstance(repair, dict)
    assert repair["backend"] == BACKEND_FAST_CPU
    assert repair["path_counts"]["gigatoken_repair"] == 1


def test_headline_backend_reports_the_added_token_reference_execution(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An added-token literal sends the request to HF, and the headline says so."""
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
    assert tokenizer.plan.backend == BACKEND_FAST_CPU
    monkeypatch.setattr(tokenizer._added_router, "holds_literal", lambda _text: True)
    tokenizer.encode("alpha <synthetic-added-token> beta", session="chat")

    report = tokenizer.explain()
    assert report["backend"] == BACKEND_REFERENCE
    assert report["backend_basis"] == "last_execution"
    assert report["planned_backend"] == BACKEND_FAST_CPU
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    last = runtime["last_execution"]
    assert isinstance(last, dict)
    assert last["executed_backend"] == BACKEND_REFERENCE
    assert last["path"] == "hf_added_token"


def test_binding_mismatch_keeps_public_session_on_reference(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(rig)
    _install_certified_probe(monkeypatch, rig, registry, source_digest="d" * 64)

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
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    assert runtime["execution_counts"] == {BACKEND_REFERENCE: 1}
    assert runtime["last_execution"] == {
        "input_bytes": 5,
        "selected_start": BACKEND_REFERENCE,
        "executed_backend": BACKEND_REFERENCE,
        "source": "state_encode",
        "path": "hf_no_certified_span_bridge",
    }
    assert report["fallback_counts"] == {}
    reasons = report["plan_reasons"]
    assert isinstance(reasons, list)
    assert any(
        reason["backend"] == BACKEND_FAST_CPU
        and reason["code"] == "R_ENGINE_BINDING_MISMATCH"
        and reason["detail"]["axis"] == "source_digest"
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
