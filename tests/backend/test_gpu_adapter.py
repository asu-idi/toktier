"""The GPU backend adapter satisfies the executor-facing protocol.

Contract reference: ``docs/contracts/routing.md`` Section 4. The
adapter is exercised through :class:`RoutedExecutor` with the real plan
shape, against stub encoders (the module is torch-free by design); the
device-backed equality run lives in the GPU test tier.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import _support as support
import pytest

from toktier.backends.protocol import Backend
from toktier.engine.gpu.backend import GpuBackend, LazyGpuBackend
from toktier.errors import BackendExecutionFault, KernelIncompatible
from toktier.policy import (
    BACKEND_GPU,
    BACKEND_REFERENCE,
    ReasonCode,
    RoutePlan,
    RoutingPolicy,
)
from toktier.routing.execute import RoutedExecutor

PLAN = RoutePlan(
    policy=RoutingPolicy.CERTIFIED,
    backend=BACKEND_GPU,
    fallback_chain=(BACKEND_GPU, BACKEND_REFERENCE),
)


class StubEncoder:
    """Single-request encoder shape the adapter consumes."""

    def __init__(
        self,
        *,
        adds_special_tokens: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.adds_special_tokens = adds_special_tokens
        self.error = error
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return [len(text)]


class StubBatched:
    """Batched channel shape the adapter consumes."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_batch(self, docs: list[str]) -> list[list[int]]:
        self.calls.append(docs)
        return [[len(doc)] for doc in docs]


def _backend(**kwargs: Any) -> GpuBackend:
    return GpuBackend(StubEncoder(**kwargs))


def test_adapter_satisfies_the_backend_protocol() -> None:
    backend = _backend()
    assert isinstance(backend, Backend)
    assert backend.backend_id == BACKEND_GPU


def test_encode_and_batch_produce_core_stream_ids() -> None:
    encoder = StubEncoder()
    batched = StubBatched()
    backend = GpuBackend(encoder, batched=batched)
    assert backend.encode("abc", add_special_tokens=False) == [3]
    assert backend.encode_batch(["a", "bb"], add_special_tokens=False) == [
        [1],
        [2],
    ]
    assert batched.calls == [["a", "bb"]]
    assert backend.encode_batch([], add_special_tokens=False) == []


def test_batches_fall_back_to_the_encoder_without_a_channel() -> None:
    encoder = StubEncoder()
    backend = GpuBackend(encoder)
    assert backend.encode_batch(["a", "bb"], add_special_tokens=False) == [
        [1],
        [2],
    ]
    assert encoder.calls == ["a", "bb"]


def test_special_token_default_is_a_noop_only_when_the_artifact_adds_none() -> None:
    """True is honored natively exactly when it cannot change the ids."""
    plain = _backend(adds_special_tokens=False)
    assert plain.encode("abc") == [3]

    scaffolded = _backend(adds_special_tokens=True)
    with pytest.raises(BackendExecutionFault) as caught:
        scaffolded.encode("abc")
    assert caught.value.details["stage"] == "add_special_tokens"
    with pytest.raises(BackendExecutionFault):
        scaffolded.encode_batch(["abc"])
    # The core stream stays available.
    assert scaffolded.encode("abc", add_special_tokens=False) == [3]


def test_runtime_errors_are_wrapped_and_others_propagate() -> None:
    wrapped = _backend(error=RuntimeError("device lost"))
    with pytest.raises(BackendExecutionFault) as caught:
        wrapped.encode("abc", add_special_tokens=False)
    assert isinstance(caught.value.__cause__, RuntimeError)

    defect = _backend(error=TypeError("bad call"))
    with pytest.raises(TypeError):
        defect.encode("abc", add_special_tokens=False)


def test_close_is_idempotent_and_use_after_close_is_an_error() -> None:
    backend = _backend()
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError):
        backend.encode("abc", add_special_tokens=False)


def test_executor_falls_back_to_reference_on_a_gpu_fault() -> None:
    """Through the real executor: a fault is counted, the answer is right."""
    gpu = GpuBackend(StubEncoder(error=RuntimeError("device lost")))
    reference = support.FakeBackend(BACKEND_REFERENCE)
    executor = RoutedExecutor(
        PLAN,
        {BACKEND_GPU: gpu, BACKEND_REFERENCE: reference},
        diagnostics=True,
    )
    assert executor.encode("hello", add_special_tokens=False) == [5, 0]
    assert executor.fallback_counts == {ReasonCode.R_EXEC_FAULT.value: 1}
    assert "device lost" in str(executor.events[0].detail["message"])


def test_executor_routes_scaffolded_defaults_to_reference() -> None:
    """add_special_tokens=True on a scaffolding artifact lands on hf."""
    gpu = GpuBackend(StubEncoder(adds_special_tokens=True))
    reference = support.FakeBackend(BACKEND_REFERENCE)
    executor = RoutedExecutor(PLAN, {BACKEND_GPU: gpu, BACKEND_REFERENCE: reference})
    assert executor.encode("hello") == [5, 1]
    assert executor.fallback_counts == {ReasonCode.R_EXEC_FAULT.value: 1}


def test_lazy_adapter_does_not_materialize_until_first_nonempty_call() -> None:
    """Facade construction and empty batches do not touch CUDA."""
    build = Mock(return_value=_backend())
    backend = LazyGpuBackend(build)
    assert backend.encode_batch([], add_special_tokens=False) == []
    build.assert_not_called()
    assert backend.encode("abc", add_special_tokens=False) == [3]
    assert backend.loaded is True
    assert build.call_count == 1
    assert backend.encode("x", add_special_tokens=False) == [1]
    assert build.call_count == 1


def test_lazy_load_failure_is_cached_and_routes_to_reference() -> None:
    """A failed first load is one stable recoverable route, not a retry loop."""
    calls = [0]

    def build() -> GpuBackend:
        calls[0] += 1
        raise KernelIncompatible(
            "synthetic load failure", details={"backend": BACKEND_GPU}
        )

    gpu = LazyGpuBackend(build)
    reference = support.FakeBackend(BACKEND_REFERENCE)
    executor = RoutedExecutor(
        PLAN,
        {BACKEND_GPU: gpu, BACKEND_REFERENCE: reference},
    )
    assert executor.encode("hello", add_special_tokens=False) == [5, 0]
    assert executor.encode("again", add_special_tokens=False) == [5, 0]
    assert calls[0] == 1
    assert gpu.loaded is False
    assert gpu.load_error is not None
    assert gpu.load_error.details["stage"] == "load"
    assert executor.fallback_counts == {ReasonCode.R_EXEC_FAULT.value: 2}


def test_lazy_load_wraps_internal_non_domain_errors() -> None:
    """Import/configuration defects in lazy construction remain recoverable."""

    def build() -> GpuBackend:
        raise ValueError("synthetic loader defect")

    backend = LazyGpuBackend(build)
    with pytest.raises(BackendExecutionFault) as caught:
        backend.encode("hello", add_special_tokens=False)
    assert caught.value.details == {
        "backend": BACKEND_GPU,
        "stage": "load",
        "cause": "ValueError",
    }
