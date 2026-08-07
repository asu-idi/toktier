"""Routing: probe, plan, execute.

Contract reference: ``docs/contracts/routing.md``. The split into three
stages is contract, and it exists so that the decision can be inspected,
logged and tested without running a tokenizer:

1. :mod:`~toktier.routing.probe` collects facts and changes nothing.
2. :mod:`~toktier.routing.plan` is a pure function from those facts plus
   the policy to an immutable ``RoutePlan``.
3. :mod:`~toktier.routing.execute` follows the plan and counts every
   fallback with its reason code.

:mod:`~toktier.routing.registry_view` holds the read-only registry
lookups the planner asks, and :mod:`~toktier.routing.explain` turns a
plan plus a snapshot plus the counters into the diagnostic mapping
behind ``Tokenizer.explain()``.

No module in this package imports an accelerator runtime.
"""

from __future__ import annotations

from .added_route import AddedTokenRouter, LiteralScanner
from .execute import FallbackEvent, RoutedExecutor
from .explain import build_explanation, reason_to_dict
from .plan import (
    ACCELERATED_BACKENDS,
    BackendAssessment,
    assess_backend,
    assessments_for,
    plan,
    reference_plan,
)
from .probe import (
    DeviceInfo,
    DeviceProbe,
    KernelCacheState,
    NoDevices,
    ProbeSnapshot,
    importable_backends,
    probe,
)
from .registry_view import (
    ArtifactRecord,
    BackendEntry,
    CertificationMatch,
    OracleRecord,
    RegistryView,
    empty_registry,
)

__all__ = [
    "ACCELERATED_BACKENDS",
    "AddedTokenRouter",
    "ArtifactRecord",
    "BackendAssessment",
    "BackendEntry",
    "CertificationMatch",
    "DeviceInfo",
    "DeviceProbe",
    "FallbackEvent",
    "KernelCacheState",
    "LiteralScanner",
    "NoDevices",
    "OracleRecord",
    "ProbeSnapshot",
    "RegistryView",
    "RoutedExecutor",
    "assess_backend",
    "assessments_for",
    "build_explanation",
    "empty_registry",
    "importable_backends",
    "plan",
    "probe",
    "reason_to_dict",
    "reference_plan",
]
