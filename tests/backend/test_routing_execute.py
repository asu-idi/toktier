"""Execution follows the plan, and every degradation is counted.

Contract reference: ``docs/contracts/routing.md`` Sections 2 and 5.2,
``docs/contracts/api.md`` Section 6.
"""

from __future__ import annotations

import _support as support
import pytest

from toktier.policy import (
    BACKEND_FAST_CPU,
    BACKEND_GPU,
    BACKEND_REFERENCE,
    ReasonCode,
    RoutePlan,
    RoutingPolicy,
)
from toktier.routing.added_route import AddedTokenRouter
from toktier.routing.execute import RoutedExecutor
from toktier.routing.explain import build_explanation
from toktier.routing.plan import assessments_for, plan

ACCELERATED_PLAN = RoutePlan(
    policy=RoutingPolicy.CERTIFIED,
    backend=BACKEND_GPU,
    fallback_chain=(BACKEND_GPU, BACKEND_REFERENCE),
)
REFERENCE_PLAN = RoutePlan(
    policy=RoutingPolicy.CERTIFIED,
    backend=BACKEND_REFERENCE,
    fallback_chain=(BACKEND_REFERENCE,),
)


def _executor(
    *,
    gpu_fails: bool = False,
    marker: str | None = None,
    route_plan: RoutePlan = ACCELERATED_PLAN,
    diagnostics: bool = True,
) -> tuple[RoutedExecutor, support.FakeBackend, support.FakeBackend]:
    gpu = support.FakeBackend(BACKEND_GPU, fail=gpu_fails, base=1000)
    reference = support.FakeBackend(BACKEND_REFERENCE, base=0)
    router = AddedTokenRouter(support.FakeScanner(marker)) if marker else None
    executor = RoutedExecutor(
        route_plan,
        {BACKEND_GPU: gpu, BACKEND_REFERENCE: reference},
        added_router=router,
        diagnostics=diagnostics,
    )
    return executor, gpu, reference


def test_selected_backend_runs_when_nothing_goes_wrong() -> None:
    """No fault, no fallback, no counters."""
    executor, gpu, reference = _executor()
    assert executor.encode("hello") == [1005, 1]
    assert gpu.calls == ["hello"]
    assert reference.calls == []
    assert executor.fallback_counts == {}


def test_execution_fault_falls_back_and_is_counted() -> None:
    """An accelerated fault returns the reference result, counted."""
    executor, gpu, reference = _executor(gpu_fails=True)
    assert executor.encode("hello") == [5, 1]
    assert gpu.calls == ["hello"]
    assert reference.calls == ["hello"]
    assert executor.fallback_counts == {ReasonCode.R_EXEC_FAULT.value: 1}
    event = executor.events[0]
    assert event.code is ReasonCode.R_EXEC_FAULT
    assert event.backend == BACKEND_GPU
    assert event.target == BACKEND_REFERENCE


def test_reference_failure_propagates() -> None:
    """There is nothing below the reference backend to fall back to."""
    reference = support.FakeBackend(
        BACKEND_REFERENCE, fail=True, error_type=RuntimeError
    )
    executor = RoutedExecutor(REFERENCE_PLAN, {BACKEND_REFERENCE: reference})
    with pytest.raises(RuntimeError):
        executor.encode("hello")
    assert executor.fallback_counts == {}


def test_unexpected_exception_types_propagate() -> None:
    """Only the recoverable fault type is a route; defects surface."""
    executor, gpu, reference = _executor()
    gpu.fail = True
    gpu.error_type = TypeError
    with pytest.raises(TypeError):
        executor.encode("hello")
    with pytest.raises(TypeError):
        executor.encode_batch(["hello"])
    assert reference.calls == []
    assert executor.fallback_counts == {}


def test_fault_events_keep_the_message_and_traceback() -> None:
    """Diagnostics start from the fault, not from the fallback."""
    executor, _gpu, _reference = _executor(gpu_fails=True)
    executor.encode("hello")
    event = executor.events[0]
    assert event.detail["message"] == "gpu refused"
    traceback_text = event.detail["traceback"]
    assert isinstance(traceback_text, str)
    assert "BackendExecutionFault" in traceback_text


def test_a_reference_plan_never_upgrades_mid_run() -> None:
    """Execution moves along the chain, never sideways or upward."""
    gpu = support.FakeBackend(BACKEND_GPU, base=1000)
    reference = support.FakeBackend(BACKEND_REFERENCE)
    executor = RoutedExecutor(
        REFERENCE_PLAN, {BACKEND_GPU: gpu, BACKEND_REFERENCE: reference}
    )
    assert executor.encode("hello") == [5, 1]
    assert gpu.calls == []


def test_added_token_literal_routes_the_input() -> None:
    """An input holding a literal takes the added-token path, counted."""
    executor, gpu, reference = _executor(marker="<sep>")
    assert executor.encode("plain text") == [1010, 1]
    assert executor.encode("a<sep>b") == [7, 1]
    assert gpu.calls == ["plain text"]
    assert reference.calls == ["a<sep>b"]
    assert executor.fallback_counts == {ReasonCode.R_INPUT_ADDED_TOKEN.value: 1}


def test_native_literal_prefilter_skips_the_exact_scanner_on_proven_misses() -> None:
    """The Rust necessary-condition gate is fast, while positives stay exact."""

    class PrefixScanner:
        def __init__(self) -> None:
            self.scans: list[str] = []

        def _native_prefilter_prefixes(self) -> tuple[tuple[int, int], ...]:
            return ((ord("<"), ord("s")),)

        def scan(self, text: str) -> list[tuple[str, int | None]] | None:
            self.scans.append(text)
            if "<sep>" not in text:
                return None
            head, _, tail = text.partition("<sep>")
            return [(head, None), ("<sep>", 99), (tail, None)]

    scanner = PrefixScanner()
    gpu = support.FakeBackend(BACKEND_GPU, base=1000)
    reference = support.FakeBackend(BACKEND_REFERENCE)
    executor = RoutedExecutor(
        ACCELERATED_PLAN,
        {BACKEND_GPU: gpu, BACKEND_REFERENCE: reference},
        added_router=AddedTokenRouter(scanner),
    )

    assert executor.encode("plain text") == [1010, 1]
    assert scanner.scans == []
    assert executor.encode("a<sep>b") == [7, 1]
    assert scanner.scans == ["a<sep>b"]
    assert reference.calls == ["a<sep>b"]


def test_batch_rows_match_single_encodes() -> None:
    """encode_batch is row-for-row equal to encode."""
    executor, _, _ = _executor()
    texts = ["a", "bb", "ccc"]
    assert executor.encode_batch(texts) == [executor.encode(t) for t in texts]


def test_batch_fault_re_runs_every_input_and_counts_each() -> None:
    """A batch fault cannot be attributed, so nothing is left uncounted."""
    executor, _gpu, reference = _executor(gpu_fails=True)
    texts = ["a", "bb", "ccc"]
    assert executor.encode_batch(texts) == [[1, 1], [2, 1], [3, 1]]
    assert reference.calls == texts
    assert executor.fallback_counts == {ReasonCode.R_EXEC_FAULT.value: 3}


def test_batch_routes_literal_inputs_one_by_one() -> None:
    """Only the inputs holding a literal leave the accelerated path."""
    executor, gpu, reference = _executor(marker="<sep>")
    texts = ["plain", "a<sep>b"]
    assert executor.encode_batch(texts) == [[1005, 1], [7, 1]]
    assert gpu.calls == ["plain"]
    assert reference.calls == ["a<sep>b"]
    assert executor.fallback_counts == {ReasonCode.R_INPUT_ADDED_TOKEN.value: 1}
    assert executor.encode_batch(texts) == [executor.encode(text) for text in texts]


def test_empty_batch_is_empty() -> None:
    """No inputs, no backend calls."""
    executor, gpu, reference = _executor()
    assert executor.encode_batch([]) == []
    assert gpu.calls == []
    assert reference.calls == []


def test_gpu_crossover_starts_small_inputs_on_fast_cpu() -> None:
    """The boundary is byte-exact and stays inside the immutable chain."""
    route_plan = RoutePlan(
        policy=RoutingPolicy.CERTIFIED,
        backend=BACKEND_GPU,
        fallback_chain=(BACKEND_GPU, BACKEND_FAST_CPU, BACKEND_REFERENCE),
    )
    gpu = support.FakeBackend(BACKEND_GPU, base=2000)
    fast_cpu = support.FakeBackend(BACKEND_FAST_CPU, base=1000)
    reference = support.FakeBackend(BACKEND_REFERENCE)
    executor = RoutedExecutor(
        route_plan,
        {
            BACKEND_GPU: gpu,
            BACKEND_FAST_CPU: fast_cpu,
            BACKEND_REFERENCE: reference,
        },
        minimum_input_bytes={BACKEND_GPU: 65_536},
        diagnostics=True,
    )

    assert executor.encode("a" * 65_535) == [66_535, 1]
    assert executor.encode("a" * 65_536) == [67_536, 1]
    assert gpu.calls == ["a" * 65_536]
    assert fast_cpu.calls == ["a" * 65_535]
    assert reference.calls == []
    assert executor.fallback_counts == {ReasonCode.R_INPUT_BELOW_GPU_THRESHOLD.value: 1}
    assert executor.execution_counts == {
        BACKEND_FAST_CPU: 1,
        BACKEND_GPU: 1,
    }
    assert executor.events[0].detail == {
        "input_bytes": 65_535,
        "threshold_bytes": 65_536,
    }
    assert executor.last_execution == {
        "input_bytes": 65_536,
        "selected_start": BACKEND_GPU,
        "executed_backend": BACKEND_GPU,
    }


def test_gpu_crossover_measures_utf8_bytes_not_code_points() -> None:
    """Multibyte text crosses on its serialized workload size."""
    route_plan = RoutePlan(
        policy=RoutingPolicy.CERTIFIED,
        backend=BACKEND_GPU,
        fallback_chain=(BACKEND_GPU, BACKEND_REFERENCE),
    )
    gpu = support.FakeBackend(BACKEND_GPU, base=1000)
    reference = support.FakeBackend(BACKEND_REFERENCE)
    executor = RoutedExecutor(
        route_plan,
        {BACKEND_GPU: gpu, BACKEND_REFERENCE: reference},
        minimum_input_bytes={BACKEND_GPU: 4},
    )

    assert executor.encode("\u4f60") == [1, 1]
    assert executor.encode("\u4f60a") == [1002, 1]
    assert reference.calls == ["\u4f60"]
    assert gpu.calls == ["\u4f60a"]


def test_unpaired_surrogate_stays_on_the_reference_path() -> None:
    """An invalid UTF-8 view is not allowed to enter an accelerated backend."""
    gpu = support.FakeBackend(BACKEND_GPU, base=1000)
    reference = support.FakeBackend(BACKEND_REFERENCE)
    executor = RoutedExecutor(
        ACCELERATED_PLAN,
        {BACKEND_GPU: gpu, BACKEND_REFERENCE: reference},
    )

    text = "\ud800"
    assert executor.encode(text) == [1, 1]
    assert gpu.calls == []
    assert reference.calls == [text]
    assert executor.last_execution == {
        "input_bytes": None,
        "selected_start": BACKEND_REFERENCE,
        "executed_backend": BACKEND_REFERENCE,
    }


def test_gpu_crossover_partitions_mixed_batches_and_preserves_order() -> None:
    """One batch may use both engines without reordering its rows."""
    route_plan = RoutePlan(
        policy=RoutingPolicy.CERTIFIED,
        backend=BACKEND_GPU,
        fallback_chain=(BACKEND_GPU, BACKEND_FAST_CPU, BACKEND_REFERENCE),
    )
    gpu = support.FakeBackend(BACKEND_GPU, base=2000)
    fast_cpu = support.FakeBackend(BACKEND_FAST_CPU, base=1000)
    reference = support.FakeBackend(BACKEND_REFERENCE)
    executor = RoutedExecutor(
        route_plan,
        {
            BACKEND_GPU: gpu,
            BACKEND_FAST_CPU: fast_cpu,
            BACKEND_REFERENCE: reference,
        },
        minimum_input_bytes={BACKEND_GPU: 4},
    )

    texts = ["a", "large", "bb", "bigger"]
    assert executor.encode_batch(texts) == [
        [1001, 1],
        [2005, 1],
        [1002, 1],
        [2006, 1],
    ]
    assert fast_cpu.calls == ["a", "bb"]
    assert gpu.calls == ["large", "bigger"]
    assert reference.calls == []
    assert executor.fallback_counts == {ReasonCode.R_INPUT_BELOW_GPU_THRESHOLD.value: 2}


@pytest.mark.parametrize(
    "thresholds",
    [
        {"unknown": 1},
        {BACKEND_GPU: -1},
        {BACKEND_GPU: 1.5},
        {BACKEND_GPU: True},
        {BACKEND_REFERENCE: 1},
    ],
)
def test_gpu_crossover_rejects_invalid_thresholds(
    thresholds: dict[str, int],
) -> None:
    """Threshold configuration cannot silently name invalid routes."""
    gpu = support.FakeBackend(BACKEND_GPU)
    reference = support.FakeBackend(BACKEND_REFERENCE)
    with pytest.raises(ValueError):
        RoutedExecutor(
            ACCELERATED_PLAN,
            {BACKEND_GPU: gpu, BACKEND_REFERENCE: reference},
            minimum_input_bytes=thresholds,
        )


def test_missing_backend_is_refused_at_construction() -> None:
    """A plan naming a backend nobody built is an error, not a surprise."""
    from toktier.errors import BackendUnavailable

    with pytest.raises(BackendUnavailable) as caught:
        RoutedExecutor(
            ACCELERATED_PLAN,
            {BACKEND_REFERENCE: support.FakeBackend(BACKEND_REFERENCE)},
        )
    assert caught.value.code == "BACKEND_UNAVAILABLE"
    assert caught.value.details["backend"] == BACKEND_GPU


def test_explain_reports_plan_reasons_waivers_and_counters() -> None:
    """explain() carries the plan, the labels, and what actually happened."""
    view = support.registry()
    snapshot = support.snapshot(registry_view=view, driver_version="550.0")
    config = support.config()
    route_plan = plan(snapshot, RoutingPolicy.CERTIFIED, view, config)
    executor, _, _ = _executor(gpu_fails=True, route_plan=ACCELERATED_PLAN)
    executor.encode("hello")

    explanation = build_explanation(
        route_plan=route_plan,
        snapshot=snapshot,
        assessments=assessments_for(snapshot, RoutingPolicy.CERTIFIED, view, config),
        fallback_counts=executor.fallback_counts,
    )
    assert explanation["routing_policy"] == "certified"
    assert "policy" not in explanation  # the bare key invited misreading
    assert explanation["backend"] == BACKEND_REFERENCE
    assert explanation["fallback_chain"] == [BACKEND_REFERENCE]
    reasons = explanation["plan_reasons"]
    assert isinstance(reasons, list)
    assert reasons[0]["code"] == ReasonCode.R_DRIVER_TOO_OLD.value
    assert explanation["fallback_counts"] == {ReasonCode.R_EXEC_FAULT.value: 1}
    certification = explanation["certification"]
    assert isinstance(certification, dict)
    assert certification["state"] == "reference"
    assert certification["identity"] == "exact"
    assert certification["backend_status"] == {BACKEND_GPU: "certified_source"}


def test_explain_distinguishes_certified_from_certified_source() -> None:
    """The JIT delivery mode is labeled as itself, everywhere."""
    view = support.registry()
    snapshot = support.snapshot(registry_view=view)
    config = support.config()
    route_plan = plan(snapshot, RoutingPolicy.CERTIFIED, view, config)
    explanation = build_explanation(route_plan=route_plan, snapshot=snapshot)
    certification = explanation["certification"]
    assert isinstance(certification, dict)
    assert certification["state"] == "certified_source"

    binary_view = support.registry(backends={BACKEND_GPU: support.certified_entry()})
    binary_snapshot = support.snapshot(registry_view=binary_view)
    binary_plan = plan(binary_snapshot, RoutingPolicy.CERTIFIED, binary_view, config)
    binary_explanation = build_explanation(
        route_plan=binary_plan, snapshot=binary_snapshot
    )
    binary_certification = binary_explanation["certification"]
    assert isinstance(binary_certification, dict)
    assert binary_certification["state"] == "certified"


def test_explain_marks_reference_only_on_oracle_mismatch() -> None:
    """An oracle outside the certified set is reported, not hidden."""
    view = support.registry()
    snapshot = support.snapshot(registry_view=view, oracle_version="0.23.0")
    config = support.config()
    route_plan = plan(snapshot, RoutingPolicy.CERTIFIED, view, config)
    explanation = build_explanation(route_plan=route_plan, snapshot=snapshot)
    certification = explanation["certification"]
    assert isinstance(certification, dict)
    assert certification["state"] == "reference_only"


def test_explain_lists_experimental_waivers() -> None:
    """Running outside the certified set says so."""
    view = support.registry()
    snapshot = support.snapshot(
        registry_view=view, artifact_sha256="1" * 64, pipeline_fingerprint=None
    )
    config = support.config()
    policy = RoutingPolicy.EXPERIMENTAL
    route_plan = plan(snapshot, policy, view, config)
    explanation = build_explanation(
        route_plan=route_plan,
        snapshot=snapshot,
        assessments=assessments_for(snapshot, policy, view, config),
    )
    waivers = explanation["experimental_waivers"]
    assert isinstance(waivers, list)
    assert waivers[0]["code"] == ReasonCode.R_UNCERTIFIED_ARTIFACT.value
    certification = explanation["certification"]
    assert isinstance(certification, dict)
    assert certification["state"] == "uncertified"
