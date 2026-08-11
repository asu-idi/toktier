"""Surface behavior of ``toktier.load`` and the facade ``Tokenizer``."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import toktier
from toktier.errors import (
    ArtifactHashMismatch,
    ArtifactNotFound,
    BackendUnavailable,
    UnsupportedConfig,
)
from toktier.facade import api as facade_api
from toktier.policy import (
    BACKEND_GPU,
    BACKEND_REFERENCE,
    PlanReason,
    ReasonCode,
    RoutePlan,
    RoutingPolicy,
)
from toktier.routing.probe import DeviceInfo, ProbeSnapshot

from .conftest import Rig, build_rig


def test_gpu_delivery_profile_detection_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(facade_api, "_module_present", lambda _name: False)
    assert facade_api._resolve_gpu_delivery("auto") == "prebuilt"
    monkeypatch.setattr(facade_api, "_module_present", lambda _name: True)
    assert facade_api._resolve_gpu_delivery("auto") == "jit"
    assert facade_api._resolve_gpu_delivery("prebuilt") == "prebuilt"
    assert facade_api._resolve_gpu_delivery("jit") == "jit"
    with pytest.raises(ValueError, match="gpu_delivery must be"):
        facade_api._resolve_gpu_delivery("unknown")


def _unjudged_jit_plan() -> RoutePlan:
    return RoutePlan(
        policy=RoutingPolicy.CERTIFIED,
        backend=BACKEND_REFERENCE,
        fallback_chain=(BACKEND_REFERENCE,),
        reasons=(
            PlanReason(
                ReasonCode.R_UNCERTIFIED_ARTIFACT,
                BACKEND_GPU,
                {
                    "cause": "toolchain_unverified",
                    "constraint": "CUDA 13.0 / torch 2.13.0+cu130",
                    "observed": "CUDA 13.0 / torch 2.11.0+cu130",
                },
            ),
        ),
    )


def test_auto_device_warns_with_explicit_unjudged_jit_remedy(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(facade_api, "_resolve_gpu_delivery", lambda _value: "jit")
    monkeypatch.setattr(facade_api, "build_plan", lambda *_args: _unjudged_jit_plan())

    with pytest.warns(RuntimeWarning, match="--accept-uncertified-jit") as caught:
        tokenizer = rig.tokenizer()
    try:
        message = str(caught[0].message)
        assert "CUDA 13.0 / torch 2.11.0+cu130" in message
        assert "CUDA 13.0 / torch 2.13.0+cu130" in message
        assert "outside TokTier's certified exact-ID guarantee" in message
        assert tokenizer.plan.backend == BACKEND_REFERENCE
    finally:
        tokenizer.close()


def test_cuda_device_failure_carries_unjudged_jit_remedy(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(facade_api, "_resolve_gpu_delivery", lambda _value: "jit")
    monkeypatch.setattr(facade_api, "build_plan", lambda *_args: _unjudged_jit_plan())

    with pytest.raises(BackendUnavailable) as caught:
        rig.tokenizer(device="cuda")

    remedy = "toktier gpu compile tiny_bytes --accept-uncertified-jit"
    assert caught.value.details["remedy"] == remedy
    assert caught.value.details["reason_code"] == "R_UNCERTIFIED_ARTIFACT"
    assert caught.value.details["reason"] == {
        "cause": "toolchain_unverified",
        "constraint": "CUDA 13.0 / torch 2.13.0+cu130",
        "observed": "CUDA 13.0 / torch 2.11.0+cu130",
    }
    # The message carries the same facts as the details, so a reader who
    # sees only the text can still tell what did not match.
    message = str(caught.value)
    assert remedy in message
    assert "observed CUDA 13.0 / torch 2.11.0+cu130" in message
    assert "certified constraint: CUDA 13.0 / torch 2.13.0+cu130" in message


def test_load_fixes_a_reference_plan(rig: Rig) -> None:
    tokenizer = rig.tokenizer()
    assert tokenizer.family == rig.family
    assert tokenizer.plan.backend == "hf"
    assert tokenizer.plan.fallback_chain[-1] == "hf"

    report = tokenizer.explain()
    assert report["backend"] == "hf"
    assert report["fallback_chain"] == ["hf"]
    assert isinstance(report["plan_reasons"], list)


def test_explain_is_the_routing_explanation_plus_facade_keys(rig: Rig) -> None:
    """The facade reports through the routing layer's own explanation.

    The requested routing policy travels under ``routing_policy`` -- the
    bare name ``policy`` with the value ``"certified"`` read like a
    certification state, which it is not -- and the certification block
    is present as its own answer. The facade plans against an empty
    registry view in this release, so that block says no certification
    identity was consulted.
    """
    tokenizer = rig.tokenizer()
    report = tokenizer.explain()
    assert tokenizer.explain(summary=False) == report

    assert report["family"] == rig.family
    assert report["routing_policy"] == "certified"
    assert "policy" not in report
    certification = report["certification"]
    assert isinstance(certification, dict)
    assert certification["state"] == "uncertified"
    assert certification["effective_verdict"] == "unverified"
    assert certification["identity"] is None
    probe = report["probe"]
    assert isinstance(probe, dict)
    assert probe["family"] == rig.family
    assert isinstance(probe["artifact_sha256"], str)
    assert probe["fast_cpu_engine_delivery"] == "integrated"
    assert probe["fast_cpu_engine_module"] == "toktier._native"
    assert "fast_cpu_source_digest" in probe
    assert "fast_cpu_build_flags" in probe
    assert "fast_cpu_toolchain" in probe
    assert "prebuilt_host_source_digest" in probe
    assert "prebuilt_host_build_flags" in probe
    assert "prebuilt_host_toolchain" in probe
    assert report["experimental_waivers"] == []
    assert report["store_directory"] is None
    assert "store" not in report  # the store has not been touched


def test_explain_summary_is_flat_and_agrees_with_full_report(rig: Rig) -> None:
    tokenizer = rig.tokenizer()
    full = tokenizer.explain()
    summary = tokenizer.explain(summary=True)
    certification = full["certification"]
    assert isinstance(certification, dict)
    fallback_counts = full["fallback_counts"]
    assert isinstance(fallback_counts, dict)

    runtime = full["runtime_policy"]
    assert isinstance(runtime, dict)

    assert summary == {
        "family": full["family"],
        "backend": full["backend"],
        "backend_basis": full["backend_basis"],
        "planned_backend": full["planned_backend"],
        "kernel_delivery": full["kernel_delivery"],
        "certification_state": certification["state"],
        "effective_verdict": certification["effective_verdict"],
        "fallback_occurred": bool(fallback_counts),
        "last_execution_backend": None,
        "last_execution_path": None,
        "last_execution_source": None,
        "last_execution_fallback": False,
        "fallback_ever_occurred": bool(fallback_counts),
        "selected_kernel_delivery": runtime["gpu_delivery_selected"],
        "loaded_kernel_delivery": full["kernel_delivery"],
    }
    assert not any(isinstance(value, dict) for value in summary.values())


def test_explain_summary_names_the_time_scope_of_each_fact(rig: Rig) -> None:
    """The summary answers "what just happened?" on its own.

    Before anything runs the last-execution group is empty and the flag
    is false. After a request it mirrors ``runtime_policy.last_execution``
    exactly, and the lifetime flag keeps its separate, sticky meaning.
    """
    tokenizer = rig.tokenizer()
    before = tokenizer.explain(summary=True)
    assert before["last_execution_backend"] is None
    assert before["last_execution_fallback"] is False
    assert before["backend_basis"] == "plan"

    tokenizer.encode("hello world")
    after = tokenizer.explain(summary=True)
    full = tokenizer.explain()
    runtime = full["runtime_policy"]
    assert isinstance(runtime, dict)
    last = runtime["last_execution"]
    assert isinstance(last, dict)

    assert after["last_execution_backend"] == last["executed_backend"]
    assert after["last_execution_path"] == last.get("path")
    assert after["last_execution_source"] == last.get("source")
    # This rig plans and runs on the reference backend, so the request
    # finished where it started: no failover to report.
    assert last["executed_backend"] == last["selected_start"]
    assert after["last_execution_fallback"] is False
    assert after["backend"] == last["executed_backend"]
    assert after["fallback_ever_occurred"] == after["fallback_occurred"]


def test_explain_summary_marks_a_real_per_request_failover(rig: Rig) -> None:
    """A request that finishes elsewhere than it started says so.

    The ledger is what the summary reads, so a recorded execution whose
    selected start differs from the backend that returned is exactly the
    case the flag exists for.
    """
    tokenizer = rig.tokenizer()
    tokenizer.encode("hello world")
    report = tokenizer.explain()
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    recorded = runtime["last_execution"]
    assert isinstance(recorded, dict)
    ledger = dict(recorded)
    ledger["selected_start"] = "fast_cpu"
    runtime["last_execution"] = ledger
    report["runtime_policy"] = runtime

    summary = facade_api._explanation_summary(report)

    assert summary["last_execution_backend"] == "hf"
    assert summary["last_execution_fallback"] is True


def test_auto_device_reports_the_hardware_probe_it_performed(rig: Rig) -> None:
    """The automatic facade probes and distinguishes absent runtime/device."""
    report = rig.tokenizer().explain()
    probe = report["probe"]
    assert isinstance(probe, dict)
    assert probe["devices_probed"] is True
    reasons = report["plan_reasons"]
    assert isinstance(reasons, list)
    gpu_codes = {reason["code"] for reason in reasons if reason["backend"] == "gpu"}
    assert "R_ACCELERATOR_NOT_ADOPTED" not in gpu_codes
    assert gpu_codes <= {
        "R_BACKEND_UNAVAILABLE",
        "R_NO_GPU_DETECTED",
        # On a machine with an installed CUDA runtime and a real device, the
        # synthetic fixture reaches the later (and more precise) artifact gate.
        "R_UNCERTIFIED_ARTIFACT",
    }


def test_cpu_device_deliberately_skips_the_hardware_probe(rig: Rig) -> None:
    report = rig.tokenizer(device="cpu").explain()
    probe = report["probe"]
    assert isinstance(probe, dict)
    assert probe["devices_probed"] is False
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    assert runtime["device"] == "cpu"


def test_explain_separates_shipped_facts_from_adoption(rig: Rig) -> None:
    """ "Not adopted" and "not available" are distinct statements.

    This checkout ships the prebuilt fatbin and the JIT sources, so the
    facade must report them as shipped -- the same answer ``toktier
    doctor`` gives -- while the plan reasons keep saying that this path
    adopts no accelerator. The fixture artifact has no record in the
    shipped support registry, so the delivery block claims no
    certification status for it (an absence of a claim, not a claim of
    absence).
    """
    report = rig.tokenizer().explain()
    assert report["prebuilt_available"] is True
    assert report["kernel_delivery"] is None
    deliveries = report["kernel_deliveries"]
    assert isinstance(deliveries, dict)
    prebuilt = deliveries["prebuilt"]
    assert prebuilt["shipped"] is True
    assert prebuilt["loaded"] is False
    assert isinstance(prebuilt["binary_digest"], str)
    jit = deliveries["jit"]
    assert jit["shipped"] is True
    assert jit["loaded"] is False
    assert prebuilt["status"] is None
    assert prebuilt["architectures"] == {}
    assert jit["status"] is None


def test_encode_returns_an_encoding(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    tokenizer = rig.tokenizer()
    encoding = tokenizer.encode("hello world")
    assert isinstance(encoding, toktier.Encoding)
    assert isinstance(encoding.ids, tuple)
    assert list(encoding.ids) == reference("hello world")
    assert len(encoding) == len(encoding.ids)


def test_close_releases_native_runtime_and_refuses_later_work(rig: Rig) -> None:
    tokenizer = rig.tokenizer()
    tokenizer.encode("materialize native runtime")
    tokenizer.close()
    tokenizer.close()

    with pytest.raises(RuntimeError, match="backend is closed"):
        tokenizer.encode("later")
    with pytest.raises(RuntimeError, match="backend is closed"):
        tokenizer.encode_batch(["later"])
    with pytest.raises(RuntimeError, match="backend is closed"):
        tokenizer.decode([1])


def test_encode_batch_rows_equal_single_encodes(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    tokenizer = rig.tokenizer()
    texts = ["", "a", "hello world", "\u00e9 \u00e9", "a\u4e2d\U0001f642b"]
    rows = tokenizer.encode_batch(texts)
    assert [list(row.ids) for row in rows] == [reference(text) for text in texts]


def test_public_requests_use_one_native_call_each(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    tokenizer = rig.tokenizer(device="cpu")

    assert list(tokenizer.encode("one", lookup="off").ids) == reference("one")
    report = tokenizer.explain()
    policy = report["runtime_policy"]
    assert isinstance(policy, dict)
    assert policy["request_path"] == "rust_native"
    assert policy["python_to_native_calls"] == 1

    texts = ["two", "three", "four"]
    rows = tokenizer.encode_batch(texts)
    assert [list(row.ids) for row in rows] == [reference(text) for text in texts]
    policy = tokenizer.explain()["runtime_policy"]
    assert isinstance(policy, dict)
    assert policy["python_to_native_calls"] == 2

    assert list(tokenizer.encode("session", session="native").ids) == reference(
        "session"
    )
    policy = tokenizer.explain()["runtime_policy"]
    assert isinstance(policy, dict)
    assert policy["python_to_native_calls"] == 3


class _ConcurrentNativeRuntime:
    """Native-runtime fake that records every request it serves."""

    gpu_engine_loaded = False
    gpu_engine_open_error = None

    def __init__(
        self,
        reference: Any,
        calls: list[str],
    ) -> None:
        self._reference = reference
        self._calls = calls
        self._calls_lock = threading.Lock()

    def encode(
        self,
        text: str,
        *,
        session: str | None,
        lookup_auto: bool,
        add_special_tokens: bool,
    ) -> list[int]:
        del session, lookup_auto
        with self._calls_lock:
            self._calls.append(text)
        return list(self._reference.encode(text, add_special_tokens=add_special_tokens))

    def runtime_stats(self) -> dict[str, object]:
        with self._calls_lock:
            call_count = len(self._calls)
        return {
            "fallback_counts": {},
            "execution_counts": {BACKEND_REFERENCE: call_count},
            "last_execution": None,
            "state_encode_counts": {},
            "last_state_encode": None,
            "python_to_native_calls": call_count,
        }

    def store_stats(self) -> dict[str, object]:
        return {}


def test_concurrent_first_encodes_wait_for_one_native_runtime(
    rig: Rig,
    reference: Callable[[str], list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First callers share construction instead of touching the adapter."""
    from toktier import _native
    from toktier.engine.gpu import native as gpu_native

    snapshot = ProbeSnapshot(
        family=rig.family,
        artifact_sha256=rig.artifact_sha256,
        oracle_version="0.22.2",
        importable_backends=frozenset({BACKEND_GPU, BACKEND_REFERENCE}),
        devices=(DeviceInfo(0, "synthetic GPU", "sm_test"),),
        devices_probed=True,
    )
    plan = RoutePlan(
        policy=RoutingPolicy.CERTIFIED,
        backend=BACKEND_GPU,
        fallback_chain=(BACKEND_GPU, BACKEND_REFERENCE),
    )
    monkeypatch.setattr(facade_api, "probe", lambda **_keywords: snapshot)
    monkeypatch.setattr(facade_api, "build_plan", lambda *_args: plan)

    prepared_calls = 0

    class _PreparedGpu:
        engine = object()
        published = False

    def prepare_gpu(**_keywords: Any) -> _PreparedGpu:
        nonlocal prepared_calls
        prepared_calls += 1
        return _PreparedGpu()

    monkeypatch.setattr(gpu_native, "prepare_native_prebuilt_gpu", prepare_gpu)

    legacy_loads = 0

    def refuse_legacy_load(_tokenizer: Any) -> Any:
        nonlocal legacy_loads
        legacy_loads += 1
        raise RuntimeError("the prebuilt delivery is already owned by native host")

    monkeypatch.setattr(facade_api.Tokenizer, "_open_gpu_backend", refuse_legacy_load)
    tokenizer = rig.tokenizer(device="cpu", gpu_delivery="prebuilt", gpu_min_bytes=0)

    worker_count = 8
    all_callers_entered = threading.Event()
    caller_lock = threading.Lock()
    caller_count = 0
    original_runtime = tokenizer._native_request_runtime

    def synchronized_runtime() -> Any | None:
        nonlocal caller_count
        with caller_lock:
            caller_count += 1
            if caller_count == worker_count:
                all_callers_entered.set()
        return original_runtime()

    monkeypatch.setattr(tokenizer, "_native_request_runtime", synchronized_runtime)

    construction_calls = 0
    construction_lock = threading.Lock()
    native_calls: list[str] = []

    def slow_runtime_factory(*args: Any, **_keywords: Any) -> Any:
        nonlocal construction_calls
        with construction_lock:
            construction_calls += 1
        if not all_callers_entered.wait(timeout=5):
            raise RuntimeError("concurrent callers did not reach initialization")
        return _ConcurrentNativeRuntime(args[2], native_calls)

    monkeypatch.setattr(_native, "NativeRuntime", slow_runtime_factory)

    texts = [f"concurrent native request {index}" for index in range(worker_count)]
    results: list[tuple[int, ...] | None] = [None] * worker_count
    failures: list[BaseException] = []

    def encode(index: int) -> None:
        try:
            results[index] = tokenizer.encode(texts[index], lookup="off").ids
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    workers = [
        threading.Thread(target=encode, args=(index,)) for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert not any(worker.is_alive() for worker in workers)
    assert not failures
    assert construction_calls == 1
    assert prepared_calls == 1
    assert sorted(native_calls) == sorted(texts)
    assert results == [tuple(reference(text)) for text in texts]
    assert legacy_loads == 0
    report = tokenizer.explain()
    gpu = report["gpu_backend"]
    assert isinstance(gpu, dict)
    assert gpu["load_error"] is None
    runtime = report["runtime_policy"]
    assert isinstance(runtime, dict)
    assert runtime["request_path"] == "rust_native"


def test_explain_probe_reports_the_delivery_this_process_loaded(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot taken before the first kernel load is not left stale.

    ``probe.kernel_delivery`` comes from the construction-time snapshot,
    so it can predate the load that the top-level ``kernel_delivery``
    reports. The two must not disagree about the same process fact.
    """
    from toktier.engine.gpu.loader import KernelLoader

    tokenizer = rig.tokenizer(device="cpu")
    before = tokenizer.explain()
    probe = before["probe"]
    assert isinstance(probe, dict)
    assert probe["kernel_delivery"] == before["kernel_delivery"]

    monkeypatch.setattr(KernelLoader, "delivery", classmethod(lambda _cls: "prebuilt"))
    after = tokenizer.explain()
    assert after["kernel_delivery"] == "prebuilt"
    probe = after["probe"]
    assert isinstance(probe, dict)
    assert probe["kernel_delivery"] == "prebuilt"


class _GpuStateRuntime:
    """A native runtime stand-in with a fixed deferred-GPU open state."""

    def __init__(self, inner: Any, *, loaded: bool, open_error: str | None) -> None:
        self._inner = inner
        self.gpu_engine_loaded = loaded
        self.gpu_engine_open_error = open_error

    def runtime_stats(self) -> Any:
        return self._inner.runtime_stats()

    def store_stats(self) -> Any:
        return self._inner.store_stats()


def test_explain_reports_a_loaded_native_gpu_host(rig: Rig) -> None:
    """The compatibility GPU block must include the Rust prebuilt host.

    The engine opens below PyO3 on the first request routed to the GPU,
    so ``gpu_backend.loaded`` follows the runtime's own record of that
    open rather than any construction-time fact.
    """
    tokenizer = rig.tokenizer(device="cpu")
    tokenizer.encode("prime the native runtime")
    inner = tokenizer._native_request
    assert inner is not None

    before = tokenizer.explain()["gpu_backend"]
    assert isinstance(before, dict)
    assert before["loaded"] is False

    tokenizer._native_request = _GpuStateRuntime(inner, loaded=True, open_error=None)
    tokenizer._gpu_device = "cuda:7"

    gpu = tokenizer.explain()["gpu_backend"]

    assert isinstance(gpu, dict)
    assert gpu["loaded"] is True
    assert gpu["device"] == "cuda:7"
    assert gpu["load_error"] is None


def test_explain_reports_a_latched_native_gpu_open_failure(rig: Rig) -> None:
    """A failed deferred open surfaces as the GPU block's load error."""
    tokenizer = rig.tokenizer(device="cpu")
    tokenizer.encode("prime the native runtime")
    inner = tokenizer._native_request
    assert inner is not None

    tokenizer._native_request = _GpuStateRuntime(
        inner, loaded=False, open_error="injected open failure"
    )

    gpu = tokenizer.explain()["gpu_backend"]

    assert isinstance(gpu, dict)
    assert gpu["loaded"] is False
    assert gpu["load_error"] == "injected open failure"


@pytest.mark.slow
def test_public_native_encode_releases_the_gil(rig: Rig) -> None:
    tokenizer = rig.tokenizer(device="cpu")
    entered = threading.Event()
    finished = threading.Event()
    failures: list[BaseException] = []
    text = "agent turn with unicode 世界 U0001f680\n" * 120_000

    def encode() -> None:
        try:
            entered.set()
            tokenizer.encode(text, lookup="off")
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=encode)
    worker.start()
    assert entered.wait(timeout=5)
    progress = 0
    while not finished.is_set():
        progress += 1
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert not failures
    # A native call that retained the GIL would allow, at most, the handful of
    # iterations between ``entered.set()`` and entering the extension. The
    # released-GIL encode gives this thread an extended scheduling window.
    assert progress > 1_000
    policy = tokenizer.explain()["runtime_policy"]
    assert isinstance(policy, dict)
    assert policy["request_path"] == "rust_native"
    assert policy["python_to_native_calls"] == 1


def test_decode_round_trips_the_core_stream(rig: Rig) -> None:
    tokenizer = rig.tokenizer()
    text = "hello \u00e9 \u4e2d\u6587 world"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_cuda_requires_an_eligible_gpu_and_unknown_devices_are_refused(
    rig: Rig,
) -> None:
    with pytest.raises(BackendUnavailable) as caught:
        rig.tokenizer(device="cuda")
    assert caught.value.details["backend"] == "gpu"
    with pytest.raises(ValueError, match="device must be"):
        rig.tokenizer(device="tpu")
    with pytest.raises(ValueError, match="gpu_min_bytes"):
        rig.tokenizer(gpu_min_bytes=-1)
    with pytest.raises(ValueError, match="gpu_min_bytes"):
        rig.tokenizer(gpu_min_bytes=True)


def test_lookup_argument_is_validated(rig: Rig) -> None:
    tokenizer = rig.tokenizer()
    with pytest.raises(ValueError):
        tokenizer.encode("x", lookup="always")
    with pytest.raises(ValueError):
        tokenizer.encode("x", session="s", lookup="auto")


def test_special_tokens_cannot_ride_store_paths(rig: Rig) -> None:
    tokenizer = rig.tokenizer()
    with pytest.raises(UnsupportedConfig):
        tokenizer.encode("x", session="s", add_special_tokens=True)
    with pytest.raises(UnsupportedConfig):
        tokenizer.encode("x", lookup="auto", add_special_tokens=True)


def test_special_tokens_run_the_plain_path(
    rig: Rig, reference: Callable[[str], list[int]]
) -> None:
    # The tiny artifact has no postprocessor, so the streams coincide;
    # the point here is that the call is served, not refused.
    tokenizer = rig.tokenizer()
    encoding = tokenizer.encode("hello", add_special_tokens=True)
    assert list(encoding.ids) == reference("hello")


def test_unknown_family_raises_artifact_not_found(rig: Rig) -> None:
    with pytest.raises(ArtifactNotFound):
        toktier.load("no_such_family", config=rig.config, manifest=rig.manifest)


def test_artifact_digest_mismatch_is_refused(rig: Rig) -> None:
    tampered = rig.artifact_path.read_bytes() + b" "
    rig.artifact_path.write_bytes(tampered)
    with pytest.raises(ArtifactHashMismatch):
        rig.tokenizer()


def test_reference_policy_is_accepted(rig: Rig) -> None:
    tokenizer = rig.tokenizer(policy="reference")
    assert tokenizer.plan.policy.value == "reference"
    assert tokenizer.plan.backend == "hf"
    probe = tokenizer.explain()["probe"]
    assert isinstance(probe, dict)
    assert probe["devices_probed"] is False


def test_version_reports_the_installed_distribution() -> None:
    assert isinstance(toktier.__version__, str)
    assert toktier.__version__


def test_second_rig_family_is_isolated(tmp_path: Path) -> None:
    other = build_rig(tmp_path / "other", family="tiny_bytes_b")
    tokenizer = other.tokenizer()
    assert tokenizer.family == "tiny_bytes_b"
