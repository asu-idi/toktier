"""Data plane behind ``Tokenizer.explain()``.

Contract reference: ``docs/contracts/api.md`` Section 6 -- ``explain()``
is a reserved public name returning the active ``RoutePlan``, the probe
snapshot summary, and accumulated fallback reason codes. The exact key
set is informational in v1; the method name and the presence of the plan
and the reason codes are stable.

Three honesty rules are implemented here rather than described:

- ``certified`` and ``certified_source`` are reported distinctly. A
  source-certified backend binds its source digest, build flags, and exact
  toolchain instead of one judged binary; GPU JIT additionally binds its class
  table and device constraints. That difference is contract, not presentation.
- Every waiver that ``EXPERIMENTAL`` policy granted is listed with its
  reason code. A configuration that is running outside the certified
  set says so.
- The headline ``backend`` answers "what actually ran", not "what was
  planned", as soon as anything has run. A plan is a prediction; once a
  request has returned, the prediction is no longer the honest answer to
  the question a reader of a headline field asks. ``backend_basis`` says
  which of the two the value is, and ``planned_backend`` keeps the plan
  visible so nothing is lost by the correction.

Everything returned is plain data (str, int, bool, list, dict) so it can
be logged or serialized without a custom encoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..policy import BACKEND_GPU, BACKEND_REFERENCE, PlanReason, RoutePlan
from .plan import BackendAssessment
from .probe import ProbeSnapshot
from .registry_view import (
    STATUS_CERTIFIED,
    STATUS_CERTIFIED_SOURCE,
    ArtifactRecord,
    BackendEntry,
)

__all__ = ["build_explanation", "reason_to_dict"]


def reason_to_dict(reason: PlanReason) -> dict[str, object]:
    """Plain-data form of one reason entry."""
    return {
        "code": reason.code.value,
        "backend": reason.backend,
        "detail": dict(reason.detail),
    }


def _certification_state(
    snapshot: ProbeSnapshot,
    route_plan: RoutePlan,
    gpu_delivery: str | None = None,
) -> dict[str, object]:
    """How the running configuration is labeled.

    ``reference_only`` is the state where an artifact is in the registry
    but the installed oracle is outside its certified set: acceleration
    is closed, the installed reference still runs. It is distinct from
    ``uncertified`` (no record at all) because the two say different
    things about what is known.

    The GPU backend ships as two deliveries with different certification
    kinds -- a judged prebuilt binary (``certified``) and a locally
    compiled JIT build (``certified_source``) -- so a single
    backend-level label cannot describe the one that runs. When
    ``gpu_delivery`` names the delivery this process loaded or selected
    and the record carries a row for it, that row's status is the GPU
    status reported here, and ``gpu_delivery`` echoes which delivery the
    label belongs to. The per-delivery detail stays under ``deliveries``
    either way; the headline just stops labeling the loaded delivery
    with a neighbouring one's status.
    """
    match = snapshot.certification
    if match is None:
        return {
            "state": "uncertified",
            "identity": None,
            "evidence_id": None,
            "backend_status": {},
            "gpu_delivery": None,
        }
    record = match.record
    oracle_mismatch = any(
        reason.code.value == "R_ORACLE_MISMATCH" for reason in route_plan.reasons
    )
    statuses = {
        backend_id: entry.status for backend_id, entry in record.backends.items()
    }
    gpu_entry = record.backends.get(BACKEND_GPU)
    labelled_delivery: str | None = None
    if gpu_entry is not None and gpu_delivery is not None:
        delivery_entry = (getattr(gpu_entry, "deliveries", None) or {}).get(
            gpu_delivery
        )
        if delivery_entry is not None:
            statuses[BACKEND_GPU] = delivery_entry.status
            labelled_delivery = gpu_delivery
    accelerated_open = route_plan.backend != BACKEND_REFERENCE
    if accelerated_open:
        status = statuses.get(route_plan.backend)
        state = (
            "certified"
            if status == STATUS_CERTIFIED
            else "certified_source"
            if status == STATUS_CERTIFIED_SOURCE
            else "experimental"
        )
    else:
        state = "reference_only" if oracle_mismatch else "reference"
    return {
        "state": state,
        "identity": match.identity,
        "evidence_id": record.evidence_id,
        "suite_version": record.suite_version,
        "certified_family": record.family,
        "readings": {
            "docs": record.docs,
            "bytes": record.bytes_judged,
            "mismatches": record.mismatches,
        },
        "backend_status": statuses,
        # Which GPU delivery the ``gpu`` status above describes; ``None``
        # when no delivery is loaded or selected, or when the record has
        # no per-delivery rows (then the status is the backend-level one).
        "gpu_delivery": labelled_delivery,
        "deliveries": _delivery_report(record),
    }


def _delivery_report(record: object) -> dict[str, object]:
    """Per-backend, per-delivery status with per-architecture labels.

    Architectures listed under ``devices`` carry the delivery's own
    status (they are what the judgment covered); ``devices_experimental``
    architectures ship an image without judgment evidence and are labeled
    ``experimental`` -- the honesty distinction is contract, not
    presentation (registry.md Section 3).
    """
    backends = getattr(record, "backends", {}) or {}
    report: dict[str, object] = {}
    for backend_id, entry in backends.items():
        deliveries = getattr(entry, "deliveries", {}) or {}
        if not deliveries:
            continue
        per_delivery: dict[str, object] = {}
        for name, sub in deliveries.items():
            per_delivery[name] = {
                "status": sub.status,
                "architectures": _architecture_statuses(sub),
            }
        report[backend_id] = per_delivery
    return report


def _architecture_statuses(entry: BackendEntry | None) -> dict[str, str]:
    """Per-architecture status labels of one delivery entry.

    An absent entry yields an empty map: no record consulted means no
    status to claim. The labeling itself is the entry's own
    (:meth:`BackendEntry.architecture_statuses`).
    """
    if entry is None:
        return {}
    return entry.architecture_statuses()


def _kernel_deliveries(
    snapshot: ProbeSnapshot, record: ArtifactRecord | None
) -> dict[str, object]:
    """Delivery-dimension status of the GPU kernel, per delivery mode.

    Two kinds of fact, deliberately side by side and never merged:

    - **Shipped/loaded facts** come from the snapshot's kernel cache
      state -- whether a prebuilt fatbin and the JIT sources are
      installed, and which delivery this process actually loaded. These
      are read-only observations of the installation and are reported
      whether or not any registry record was consulted.
    - **Certification facts** (``status``, ``architectures``,
      ``driver_min``) come from the registry record for this artifact
      when one is available. ``None``/empty means no record was
      consulted or none exists -- an absence of a claim, not a claim of
      absence.
    """
    cache = snapshot.kernel_cache
    entry = record.backends.get(BACKEND_GPU) if record is not None else None
    prebuilt_entry: BackendEntry | None = None
    jit_entry: BackendEntry | None = None
    if entry is not None:
        if entry.deliveries:
            prebuilt_entry = entry.deliveries.get("prebuilt")
            jit_entry = entry.deliveries.get("jit")
        else:
            # JIT-era record shape: the top-level entry is the JIT view.
            jit_entry = entry
    return {
        "prebuilt": {
            "shipped": cache.prebuilt_available,
            "loaded": cache.delivery == "prebuilt",
            "binary_digest": cache.binary_digest,
            "host_source_digest": cache.host_source_digest,
            "host_build_flags": list(cache.host_build_flags),
            "host_toolchain": cache.host_toolchain,
            "status": prebuilt_entry.status if prebuilt_entry else None,
            "architectures": _architecture_statuses(prebuilt_entry),
            "driver_min": (
                prebuilt_entry.driver_min if prebuilt_entry else None
            ),
        },
        "jit": {
            "shipped": cache.source_digest is not None,
            "loaded": cache.delivery == "jit",
            "source_digest": cache.source_digest,
            "status": jit_entry.status if jit_entry else None,
            "architectures": _architecture_statuses(jit_entry),
        },
    }


def _executed_backend(last_execution: Mapping[str, object] | None) -> str | None:
    """The backend of the last returned result, when one is recorded.

    Anything other than a non-empty backend name is treated as "no
    execution to report": a headline field must not be derived from a
    value whose shape was not the one promised.
    """
    if not isinstance(last_execution, Mapping):
        return None
    value = last_execution.get("executed_backend")
    return value if isinstance(value, str) and value else None


def build_explanation(
    *,
    route_plan: RoutePlan,
    snapshot: ProbeSnapshot,
    assessments: Sequence[BackendAssessment] = (),
    fallback_counts: Mapping[str, int] | None = None,
    api_version: int = 1,
    delivery_record: ArtifactRecord | None = None,
    last_execution: Mapping[str, object] | None = None,
    gpu_delivery: str | None = None,
) -> dict[str, object]:
    """Assemble the diagnostic mapping for one tokenizer.

    ``delivery_record`` optionally names the registry record whose
    per-delivery certification statuses the ``kernel_deliveries`` block
    reports. It defaults to the record the snapshot's certification
    match carries; a caller that plans against an empty registry view
    but still wants the shipped evidence statements reported (the 0.x
    facade) passes the record it looked up read-only.

    ``last_execution`` is the routing ledger's record of the request
    that most recently returned a result. When it names a backend, that
    backend is the reported ``backend`` and ``backend_basis`` is
    ``"last_execution"``; with no execution yet the report falls back to
    the planned backend and says so with ``backend_basis="plan"``.

    ``gpu_delivery`` names the kernel delivery this process loaded, or
    selected before any load. The ``certification`` headline reports the
    status of that delivery rather than the backend-level row, so a
    loaded ``certified`` prebuilt image is not headlined with the
    ``certified_source`` label of the JIT delivery shipped beside it.
    """
    waivers = [
        reason_to_dict(reason)
        for assessment in assessments
        if assessment.eligible
        for reason in assessment.waived
    ]
    executed = _executed_backend(last_execution)
    return {
        "api_version": api_version,
        # The key says which question is answered: this is the routing
        # policy that was requested (CERTIFIED, REFERENCE, ...), not a
        # certification state. The two vocabularies share the word
        # "certified", so a bare "policy" key inviting the reading "this
        # request was certified" is exactly the ambiguity to avoid; the
        # certification state lives under the "certification" key.
        "routing_policy": route_plan.policy.value,
        # The backend that actually returned the last result, once
        # anything has returned one; the planned backend before that.
        # ``backend_basis`` names which of the two this is, and
        # ``planned_backend`` keeps the plan itself readable.
        "backend": executed if executed is not None else route_plan.backend,
        "backend_basis": "last_execution" if executed is not None else "plan",
        "planned_backend": route_plan.backend,
        "fallback_chain": list(route_plan.fallback_chain),
        "plan_reasons": [reason_to_dict(reason) for reason in route_plan.reasons],
        "experimental_waivers": waivers,
        "fallback_counts": dict(fallback_counts or {}),
        # Which kernel delivery the process runs (prebuilt / jit), or
        # None before any kernel load; prebuilt availability is the
        # read-only shipped-fatbin fact.
        "kernel_delivery": snapshot.kernel_cache.delivery,
        "prebuilt_available": snapshot.kernel_cache.prebuilt_available,
        # Delivery-dimension status map: shipped/loaded facts per kernel
        # delivery, plus the per-architecture certification statuses of
        # the record consulted for this artifact (when one is).
        "kernel_deliveries": _kernel_deliveries(
            snapshot,
            delivery_record
            if delivery_record is not None
            else (
                snapshot.certification.record
                if snapshot.certification
                else None
            ),
        ),
        "certification": _certification_state(snapshot, route_plan, gpu_delivery),
        "probe": snapshot.summary(),
    }
