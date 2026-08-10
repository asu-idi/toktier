"""Surface behavior of ``toktier.load`` and the facade ``Tokenizer``."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

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

    assert report["family"] == rig.family
    assert report["routing_policy"] == "certified"
    assert "policy" not in report
    certification = report["certification"]
    assert isinstance(certification, dict)
    assert certification["state"] == "uncertified"
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


def test_explain_reports_a_loaded_native_gpu_host(rig: Rig) -> None:
    """The compatibility GPU block must include the Rust prebuilt host."""
    tokenizer = rig.tokenizer(device="cpu")
    tokenizer._native_gpu_engine = object()
    tokenizer._gpu_device = "cuda:7"

    gpu = tokenizer.explain()["gpu_backend"]

    assert isinstance(gpu, dict)
    assert gpu["loaded"] is True
    assert gpu["device"] == "cuda:7"
    assert gpu["load_error"] is None


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
