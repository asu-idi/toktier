"""Certification reporting for the explicit GPU engine, torch-free.

Contract reference: ``docs/contracts/registry.md`` Section 2 (the
explicit-engine rule). The engine constructs and runs below the routing
layer regardless of the installed oracle version; the honest-labeling
half of the oracle policy is implemented here: every binding set and
``explain()`` report carries the installed oracle against the certified
set, and a per-family verdict states exactly why a certificate does or
does not attach to the running process.

This module must stay importable without ``torch``: the questions it
answers (registry statuses, oracle membership, architecture coverage)
are host questions, and a machine without a GPU still has to be able to
compute and check them -- the same rule the binding-set helpers follow
(``toktier.kernels.bindings``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..._oracle import ORACLE_PACKAGE, oracle_version
from ...policy import BACKEND_GPU
from ...routing.registry_view import ArtifactRecord, RegistryView

__all__ = ["family_certification", "oracle_binding"]


def oracle_binding(
    registry: RegistryView,
    records: Mapping[str, ArtifactRecord | None],
    *,
    installed: str | None = None,
) -> dict[str, Any]:
    """The installed-oracle facts a certificate report must carry.

    ``certified_versions`` is the union of the certified sets named by
    the given records; ``in_certified_set`` is the fail-closed
    membership answer -- an empty set (no records, or an absent oracle
    record) can only produce ``False``, never a pass.
    """
    observed = installed if installed is not None else oracle_version()
    versions: set[str] = set()
    for record in records.values():
        if record is None:
            continue
        oracle = registry.oracle(record.oracle_id)
        if oracle is not None:
            versions.update(oracle.certified_versions)
    return {
        "package": ORACLE_PACKAGE,
        "installed": observed,
        "certified_versions": sorted(versions),
        "in_certified_set": observed is not None and observed in versions,
    }


def family_certification(
    *,
    registry: RegistryView,
    record: ArtifactRecord | None,
    delivery: str | None,
    architecture: str | None,
    certificate_void: bool,
    jit_toolchain_satisfied: bool | None = None,
    installed_oracle: str | None = None,
) -> dict[str, Any]:
    """One family's certification verdict for the running process.

    ``state`` answers "does the certificate attach to what this process
    is actually running?" and is fail-closed: it equals the registry
    status only when the delivery in effect, the observed device
    architecture and the installed oracle are all inside what the record
    judged, and the process certificate is intact. Every gap is listed
    in ``reasons`` (machine-readable ids), so a closed verdict names its
    cause instead of hiding it.
    """
    entry = record.backends.get(BACKEND_GPU) if record is not None else None
    if record is None or entry is None:
        return {
            "status": None,
            "architectures": {},
            "state": "uncertified",
            "reasons": ["no_certification_record"],
        }
    view = entry
    if entry.deliveries:
        view = entry.deliveries.get(delivery or "jit", entry)
    reasons: list[str] = []
    oracle = oracle_binding(
        registry, {record.family: record}, installed=installed_oracle
    )
    if not oracle["in_certified_set"]:
        reasons.append("oracle_outside_certified_set")
    if certificate_void:
        reasons.append("certificate_void")
    if delivery == "jit" and jit_toolchain_satisfied is not True:
        reasons.append("jit_toolchain_unverified")
    architecture_experimental = False
    if architecture is None:
        reasons.append("device_architecture_unobserved")
    elif architecture in view.devices:
        pass
    elif architecture in view.devices_experimental:
        architecture_experimental = True
        reasons.append("architecture_experimental")
    else:
        reasons.append("architecture_not_judged")
    if not reasons:
        state = view.status
    elif reasons == ["architecture_experimental"] and architecture_experimental:
        state = "experimental"
    else:
        state = "uncertified"
    return {
        "status": view.status,
        "architectures": view.architecture_statuses(),
        "state": state,
        "reasons": reasons,
    }
