"""Routing stage 3: follow the plan, count every degradation.

Contract reference: ``docs/contracts/routing.md`` Sections 2 and 5.2.

Execution may only move along the plan's fallback chain, never sideways
and never upward: a run that started under a reference plan does not
opportunistically upgrade mid-run, because the plan is the record of
what was decided and why. Every runtime fallback is recorded with its
reason code, so a degraded run is visible in ``explain()`` rather than
merely slower or, worse, silently different.

Three runtime routings live here:

- ``R_INPUT_BELOW_GPU_THRESHOLD`` -- a small input starts at the next
  eligible backend in the immutable chain instead of paying GPU launch cost.
- ``R_INPUT_ADDED_TOKEN`` -- the input holds an added-token literal.
  Part of the certified pipeline design, not an incident.
- ``R_EXEC_FAULT`` -- an accelerated path raised
  :class:`~toktier.errors.BackendExecutionFault`, the one exception
  type backends use for recoverable execution failures. The affected
  input is re-run on the next backend in the chain; the reference
  backend answers when the chain reaches it. (The native one-call
  runtime records this same code when a core-stream-only backend is
  bypassed for requested postprocessing.) Every other exception
  propagates: an unexpected
  error is a defect to surface, not a route, and an exception from the
  reference backend itself has nothing further to fall back to.
"""

from __future__ import annotations

import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from ..errors import BackendExecutionFault, BackendUnavailable
from ..policy import BACKEND_GPU, BACKEND_REFERENCE, ReasonCode, RoutePlan
from .added_route import AddedTokenRouter

__all__ = [
    "FallbackEvent",
    "RoutedExecutor",
]


class _Encoder(Protocol):
    """The part of a backend the executor uses."""

    @property
    def backend_id(self) -> str:
        """Backend identifier."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one document."""

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Encode a batch."""


@dataclass(frozen=True)
class FallbackEvent:
    """One recorded runtime routing decision."""

    code: ReasonCode
    #: Backend the input moved away from.
    backend: str
    #: Backend the input moved to.
    target: str
    detail: Mapping[str, object]


class RoutedExecutor:
    """Runs one immutable plan over a set of constructed backends."""

    def __init__(
        self,
        route_plan: RoutePlan,
        backends: Mapping[str, _Encoder],
        *,
        added_router: AddedTokenRouter | None = None,
        diagnostics: bool = False,
        minimum_input_bytes: Mapping[str, int] | None = None,
    ) -> None:
        missing = [
            backend_id
            for backend_id in route_plan.fallback_chain
            if backend_id not in backends
        ]
        if missing:
            raise BackendUnavailable(
                "the plan names backends that were not constructed",
                details={"backend": missing[0], "missing": missing},
            )
        self._plan = route_plan
        self._backends = dict(backends)
        self._added_router = added_router
        self._diagnostics = diagnostics
        thresholds = dict(minimum_input_bytes or {})
        unknown = set(thresholds) - set(route_plan.fallback_chain)
        if unknown:
            raise ValueError(
                "minimum_input_bytes names backends outside the route plan: "
                f"{sorted(unknown)}"
            )
        if any(type(value) is not int or value < 0 for value in thresholds.values()):
            raise ValueError(
                "minimum input byte thresholds must be non-negative integers"
            )
        if thresholds.get(BACKEND_REFERENCE, 0) != 0:
            raise ValueError(
                "the reference backend cannot have an input-size threshold"
            )
        self._minimum_input_bytes = thresholds
        from .. import _native as native

        if added_router is None:
            literal_mode = 0
            literal_prefixes: tuple[tuple[int, int], ...] = ()
        else:
            discovered = added_router._native_prefilter_prefixes()
            if discovered is None:
                literal_mode = 1
                literal_prefixes = ()
            else:
                literal_mode = 2
                literal_prefixes = discovered
        self._native_selector = native.RouteSelector(
            [thresholds.get(backend_id, 0) for backend_id in route_plan.fallback_chain],
            len(route_plan.fallback_chain) - 1,
            route_plan.fallback_chain[0] == BACKEND_GPU,
            literal_mode,
            literal_prefixes,
        )
        self._counts: dict[ReasonCode, int] = {}
        self._events: list[FallbackEvent] = []
        self._execution_counts: dict[str, int] = {}
        self._last_execution: dict[str, object] | None = None

    # -- state ---------------------------------------------------------

    @property
    def plan(self) -> RoutePlan:
        """The immutable plan being followed."""
        return self._plan

    @property
    def fallback_counts(self) -> Mapping[str, int]:
        """Reason code -> number of inputs affected."""
        return {
            code.value: count
            for code, count in sorted(
                self._counts.items(), key=lambda item: item[0].value
            )
        }

    @property
    def events(self) -> tuple[FallbackEvent, ...]:
        """Recorded events; populated only with diagnostics enabled."""
        return tuple(self._events)

    @property
    def execution_counts(self) -> Mapping[str, int]:
        """Backend id -> number of inputs whose result it returned."""
        return dict(sorted(self._execution_counts.items()))

    @property
    def minimum_input_bytes(self) -> Mapping[str, int]:
        """Per-backend runtime crossover thresholds."""
        return dict(sorted(self._minimum_input_bytes.items()))

    @property
    def last_execution(self) -> Mapping[str, object] | None:
        """The most recent per-input starting and finishing route."""
        return dict(self._last_execution) if self._last_execution else None

    def _record(
        self,
        code: ReasonCode,
        *,
        backend: str,
        target: str,
        **detail: object,
    ) -> None:
        self._counts[code] = self._counts.get(code, 0) + 1
        if self._diagnostics:
            self._events.append(
                FallbackEvent(
                    code=code, backend=backend, target=target, detail=dict(detail)
                )
            )

    # -- execution -----------------------------------------------------

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one document along the plan's chain."""
        input_bytes, start, holds_literal = self._starting_route(text)
        return self._encode_from(
            start,
            text,
            add_special_tokens=add_special_tokens,
            holds_literal=holds_literal,
            input_bytes=input_bytes,
            selected_start=start,
        )

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode a batch; row ``i`` equals ``encode(texts[i])``.

        Three cases, in order:

        - The plan already runs on the reference backend: encode the
          batch there; there is nothing to fall back to.
        - Some input holds an added-token literal: encode input by
          input, so the literal ones are routed and counted
          individually instead of pulling the whole batch off the
          accelerated path silently.
        - Otherwise: attempt the batch on the planned backend. If it
          reports a recoverable fault, the failure cannot be attributed
          to a single input, so every input is re-run from the next
          backend in the chain and each one is counted. A single fault
          must not hide an unknown number of affected inputs.
        """
        if not texts:
            return []
        routes = [self._starting_route(text) for text in texts]
        literals = [route[2] for route in routes]
        if any(literals):
            return [
                self._encode_from(
                    start,
                    text,
                    add_special_tokens=add_special_tokens,
                    holds_literal=flag,
                    input_bytes=input_bytes,
                    selected_start=start,
                )
                for text, flag, (input_bytes, start, _literal) in zip(
                    texts, literals, routes, strict=True
                )
            ]
        groups: dict[int, list[int]] = {}
        for position, (_input_bytes, start, _literal) in enumerate(routes):
            groups.setdefault(start, []).append(position)
        output: list[list[int] | None] = [None] * len(texts)
        for start, indices in groups.items():
            group_texts = [texts[index] for index in indices]
            group_bytes = [routes[index][0] for index in indices]
            rows = self._encode_batch_from(
                start,
                group_texts,
                add_special_tokens=add_special_tokens,
                input_bytes=group_bytes,
                selected_start=start,
            )
            for index, result_row in zip(indices, rows, strict=True):
                output[index] = result_row
        if any(row is None for row in output):  # defensive invariant
            raise RuntimeError("a routed batch did not produce every output row")
        return cast(list[list[int]], output)

    def _starting_route(self, text: str) -> tuple[int | None, int, bool]:
        """Return byte size, first chain index, and exact literal decision."""
        input_bytes, index, below_gpu, literal_candidate = (
            self._native_selector.route(text)
        )
        chain = self._plan.fallback_chain
        if below_gpu:
            self._record(
                ReasonCode.R_INPUT_BELOW_GPU_THRESHOLD,
                backend=BACKEND_GPU,
                target=chain[index],
                input_bytes=input_bytes,
                threshold_bytes=self._minimum_input_bytes.get(BACKEND_GPU, 0),
            )
        holds_literal = bool(
            literal_candidate
            and self._added_router is not None
            and self._added_router.holds_literal(text)
        )
        return input_bytes, index, holds_literal

    def _record_execution(
        self,
        backend: str,
        *,
        input_bytes: int | None,
        selected_start: int,
        source: str | None = None,
        path: str | None = None,
    ) -> None:
        self._execution_counts[backend] = self._execution_counts.get(backend, 0) + 1
        self._last_execution = {
            "input_bytes": input_bytes,
            "selected_start": self._plan.fallback_chain[selected_start],
            "executed_backend": backend,
        }
        if source is not None:
            self._last_execution["source"] = source
        if path is not None:
            self._last_execution["path"] = path

    def record_reference_result(
        self,
        text: str,
        *,
        reason: ReasonCode,
        path: str,
        replaces_last: bool = False,
        **detail: object,
    ) -> None:
        """Record a facade-owned reference result in the routing ledger.

        Store seeding sometimes needs token offsets that an accelerated
        backend does not expose. Those paths call the HF oracle directly,
        outside :meth:`encode`, but they are still routing outcomes and must
        not disappear from ``runtime_policy`` or ``fallback_counts``.

        ``replaces_last`` is used when an accelerated full encode completed
        but its reconstructed spans failed a guard. In that case the facade
        discards that result and returns a fresh reference result; execution
        counts therefore replace, rather than double-count, the final source.
        """
        chain = self._plan.fallback_chain
        if replaces_last and self._last_execution is not None:
            previous = self._last_execution
            input_bytes = cast(int | None, previous.get("input_bytes"))
            selected_name = str(previous.get("selected_start"))
            try:
                selected_start = chain.index(selected_name)
            except ValueError:  # defensive: the ledger must name this plan
                input_bytes, selected_start, _literal = self._starting_route(text)
            previous_backend = str(previous.get("executed_backend"))
            previous_count = self._execution_counts.get(previous_backend, 0)
            if previous_count > 1:
                self._execution_counts[previous_backend] = previous_count - 1
            elif previous_count == 1:
                del self._execution_counts[previous_backend]
            routed_from = previous_backend
        else:
            input_bytes, selected_start, _literal = self._starting_route(text)
            routed_from = chain[selected_start]

        if routed_from != BACKEND_REFERENCE:
            self._record(
                reason,
                backend=routed_from,
                target=BACKEND_REFERENCE,
                source="state_encode",
                path=path,
                **detail,
            )
        self._record_execution(
            BACKEND_REFERENCE,
            input_bytes=input_bytes,
            selected_start=selected_start,
            source="state_encode",
            path=path,
        )

    def _encode_batch_from(
        self,
        index: int,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        input_bytes: Sequence[int | None],
        selected_start: int,
    ) -> list[list[int]]:
        """Encode one same-start batch, falling down the chain as a unit."""
        chain = self._plan.fallback_chain
        backend_id = chain[index]
        try:
            rows = self._backends[backend_id].encode_batch(
                texts, add_special_tokens=add_special_tokens
            )
        except BackendExecutionFault as exc:
            if backend_id == BACKEND_REFERENCE:
                raise
            target = chain[index + 1]
            for _ in texts:
                self._record_fault(
                    exc, backend=backend_id, target=target, scope="batch"
                )
            return self._encode_batch_from(
                index + 1,
                texts,
                add_special_tokens=add_special_tokens,
                input_bytes=input_bytes,
                selected_start=selected_start,
            )
        for size in input_bytes:
            self._record_execution(
                backend_id,
                input_bytes=size,
                selected_start=selected_start,
            )
        return rows

    def _reference_index(self) -> int:
        return len(self._plan.fallback_chain) - 1

    def _record_fault(
        self,
        exc: BackendExecutionFault,
        *,
        backend: str,
        target: str,
        scope: str,
    ) -> None:
        """Count one recoverable fault, keeping its message and origin.

        The traceback is recorded only on diagnostic events: counters
        stay cheap, and the event is where debugging starts.
        """
        detail: dict[str, object] = {
            "error": type(exc).__name__,
            "message": str(exc),
            "scope": scope,
        }
        if self._diagnostics:
            detail["traceback"] = "".join(traceback.format_exception(exc))
        self._record(ReasonCode.R_EXEC_FAULT, backend=backend, target=target, **detail)

    def _encode_from(
        self,
        index: int,
        text: str,
        *,
        add_special_tokens: bool,
        holds_literal: bool | None = None,
        input_bytes: int | None = None,
        selected_start: int = 0,
    ) -> list[int]:
        # The plan guarantees the chain ends with the reference backend,
        # so the walk terminates there: the reference result is returned,
        # or its exception propagates.
        chain = self._plan.fallback_chain
        while True:
            backend_id = chain[index]
            is_reference = backend_id == BACKEND_REFERENCE
            if not is_reference and self._added_router is not None:
                if holds_literal is None:
                    holds_literal = self._added_router.holds_literal(text)
            else:
                holds_literal = False
            if not is_reference and holds_literal:
                target = chain[self._reference_index()]
                self._record(
                    ReasonCode.R_INPUT_ADDED_TOKEN,
                    backend=backend_id,
                    target=target,
                )
                index = self._reference_index()
                continue
            if is_reference:
                # The reference backend defines correct output; there is
                # nothing below it to fall back to, so nothing is caught.
                result = self._backends[backend_id].encode(
                    text, add_special_tokens=add_special_tokens
                )
                self._record_execution(
                    backend_id,
                    input_bytes=input_bytes,
                    selected_start=selected_start,
                )
                return result
            try:
                result = self._backends[backend_id].encode(
                    text, add_special_tokens=add_special_tokens
                )
                self._record_execution(
                    backend_id,
                    input_bytes=input_bytes,
                    selected_start=selected_start,
                )
                return result
            except BackendExecutionFault as exc:
                target = chain[index + 1]
                self._record_fault(
                    exc, backend=backend_id, target=target, scope="input"
                )
                index += 1
