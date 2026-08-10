"""Routing stage 1: collect facts, change nothing.

Contract reference: ``docs/contracts/routing.md`` Section 2. The probe
gathers importable backends, device inventory and driver version, kernel
cache state, registry entries for the requested family, and the
installed oracle version. It never builds kernels, never downloads
artifacts, and never mutates state.

Two consequences shape this module:

- **Availability is decided without importing.** Backend presence is
  answered with :func:`importlib.util.find_spec`, which locates a module
  without executing it. Nothing in this lane imports an accelerator
  runtime, and merely asking whether one is installed must not load it.
- **Device facts are supplied, not invented.** Enumerating CUDA devices
  needs the accelerator runtime, which lives in the GPU extra, so the
  probe takes a :class:`DeviceProbe`. When no probe is supplied the
  snapshot reports no devices *and records that enumeration was not
  performed* (``devices_probed=False``): an accelerated path is never
  planned on facts nobody produced, and the plan's reason says "not
  probed" rather than claiming a hardware observation that never
  happened.

Every value in the resulting snapshot is immutable, so the plan derived
from it cannot be invalidated by later mutation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Protocol

from ..backends.fast_cpu import (
    ENGINE_MODULE,
    FastCpuEngineFacts,
    fast_cpu_engine_facts,
)
from ..backends.hf import ORACLE_PACKAGE, oracle_version
from ..kernels.bindings import CertifiedSourceBindings
from ..policy import BACKEND_FAST_CPU, BACKEND_GPU, BACKEND_REFERENCE
from .registry_view import CertificationMatch, RegistryView

__all__ = [
    "DeviceInfo",
    "DeviceProbe",
    "KernelCacheState",
    "NoDevices",
    "ProbeSnapshot",
    "importable_backends",
    "probe",
]


@dataclass(frozen=True)
class DeviceInfo:
    """One accelerator device as reported by the runtime."""

    index: int
    name: str
    #: Device architecture in the registry's spelling, for example
    #: ``sm_120``. Comparison against the certified device list is a
    #: string match: an unlisted architecture is not eligible under
    #: CERTIFIED, whatever a build system could produce for it.
    architecture: str


@dataclass(frozen=True)
class KernelCacheState:
    """What the kernel loader can say without building anything.

    Digests here are of shipped or already-materialized inputs: hashing
    kernel sources and generated lookup tables is a read-only operation
    and is therefore allowed during probing. A field left ``None`` means
    "not verified", which the planner treats as a failed verification --
    never as a pass.
    """

    built: bool = False
    build_failed: bool = False
    build_error: str | None = None
    binary_digest: str | None = None
    source_digest: str | None = None
    class_table_digest: str | None = None
    build_flags: tuple[str, ...] = ()
    toolchain: str | None = None
    #: Domain-separated identity of the Rust host that loads the prebuilt
    #: image and owns routing/store/reference fallback below PyO3.
    host_source_digest: str | None = None
    #: Exact release-build facts embedded by that native host.
    host_build_flags: tuple[str, ...] = ()
    host_toolchain: str | None = None
    #: Kernel delivery this process loaded (``prebuilt`` / ``jit``), or
    #: ``None`` before any load. The planner verifies the registry
    #: delivery entry matching the delivery that actually runs (or, pre
    #: load, the one the loader would prefer).
    delivery: str | None = None
    #: Delivery selected for a future lazy load. Kept separate from
    #: ``delivery`` so diagnostics never call an unmaterialized profile
    #: "loaded" while the planner can still judge the exact delivery that
    #: will run.
    preferred_delivery: str | None = None
    #: Whether a prebuilt fatbin is shipped in this installation (a
    #: read-only fact; ``binary_digest`` carries its digest when so).
    prebuilt_available: bool = False
    #: Whether the installed toolchain satisfies the certificate's
    #: constraint expression. Evaluating that expression belongs to the
    #: loader that knows the toolchain; ``None`` means unevaluated.
    toolchain_satisfied: bool | None = None
    #: Number of distinct kernel build configurations loaded in this
    #: process. More than one invalidates a ``certified_source``
    #: certificate's premises (registry.md Section 3.2).
    loaded_flag_sets: int = 0

    @classmethod
    def from_bindings(
        cls, bindings: CertifiedSourceBindings, **facts: Any
    ) -> KernelCacheState:
        """Cache facts whose bound values come from a real binding set.

        The bound fields (``source_digest``, ``build_flags``,
        ``class_table_digest``) are taken from the shared
        ``CertifiedSourceBindings`` representation the kernel loader
        produces, so probe-side facts and registry-side records are
        spelled identically by construction. Remaining facts (``built``,
        toolchain observations, ...) are passed through.
        """
        return cls(
            source_digest=bindings.source_digest,
            build_flags=bindings.build_flags,
            class_table_digest=bindings.class_table_digest,
            **facts,
        )


class DeviceProbe(Protocol):
    """Reports the accelerator inventory without changing anything."""

    def devices(self) -> tuple[DeviceInfo, ...]:
        """Usable devices, empty when there are none."""

    def driver_version(self) -> str | None:
        """Installed driver version, or ``None`` when unknown."""

    def kernel_cache(self) -> KernelCacheState:
        """Kernel cache facts; no build is triggered."""


class NoDevices:
    """Device probe for a path that adopts no accelerator runtime.

    Device and driver facts stay empty -- nothing was enumerated, and
    the snapshot's ``devices_probed=False`` says so. The *shipped*
    kernel facts are different: whether a prebuilt fatbin and the JIT
    kernel sources are installed is a read-only property of the package
    on disk, needs no accelerator runtime to answer, and reporting it as
    absent would contradict ``toktier doctor`` over the same
    installation. Those facts are therefore reported truthfully here,
    through the same helpers ``doctor`` uses; whether an accelerated
    path is *adopted* remains a separate statement carried by the plan
    reasons (``R_ACCELERATOR_NOT_ADOPTED``).
    """

    def devices(self) -> tuple[DeviceInfo, ...]:
        """No devices enumerated (and the snapshot records why)."""
        return ()

    def driver_version(self) -> str | None:
        """No driver observed."""
        return None

    def kernel_cache(self) -> KernelCacheState:
        """Nothing built here; shipped and process facts still reported.

        ``delivery`` comes from the process-wide loader state: which
        kernel delivery this process loaded (through the explicit
        engine, if anything) is a read-only fact of the process, not of
        this path, and reporting ``None`` while a kernel runs in the
        same process would be false. Reading the loader state imports
        no accelerator runtime.
        """
        from ..engine.gpu.loader import KernelLoader
        from ..engine.gpu.native import native_host_build_facts
        from ..kernels import kernel_source_digest, kernel_source_paths
        from ..kernels.bindings import bare_sha256
        from ..kernels.prebuilt import shipped_prebuilt_facts

        prebuilt_available, fatbin_digest = shipped_prebuilt_facts()
        host = native_host_build_facts()
        sources_shipped = all(
            path.is_file() for path in kernel_source_paths()
        )
        return KernelCacheState(
            binary_digest=(
                bare_sha256(fatbin_digest) if fatbin_digest else None
            ),
            source_digest=(
                bare_sha256(kernel_source_digest())
                if sources_shipped
                else None
            ),
            prebuilt_available=prebuilt_available,
            delivery=KernelLoader.delivery(),
            host_source_digest=host.source_digest,
            host_build_flags=host.build_flags,
            host_toolchain=host.toolchain,
        )


@dataclass(frozen=True)
class ProbeSnapshot:
    """Immutable facts the planner reasons over."""

    family: str
    artifact_sha256: str | None = None
    pipeline_fingerprint: str | None = None
    added_frontend_fingerprint: str | None = None
    oracle_package: str = ORACLE_PACKAGE
    oracle_version: str | None = None
    importable_backends: frozenset[str] = frozenset({BACKEND_REFERENCE})
    devices: tuple[DeviceInfo, ...] = ()
    #: Whether device enumeration was actually performed by a supplied
    #: device probe. ``False`` means the empty ``devices`` tuple is the
    #: fail-closed default of a path that adopts no accelerator runtime,
    #: not an observation about the machine's hardware; the planner
    #: reports the two cases with different reason codes.
    devices_probed: bool = False
    driver_version: str | None = None
    kernel_cache: KernelCacheState = field(default_factory=KernelCacheState)
    #: Installed corrected-Gigatoken facts.  Empty facts fail the certified
    #: binding closed; package import is not required to observe them.
    fast_cpu_engine: FastCpuEngineFacts = field(
        default_factory=FastCpuEngineFacts
    )
    #: Registry match for this artifact, resolved during probing so the
    #: planner performs lookups only over values it was handed.
    certification: CertificationMatch | None = None

    def summary(self) -> dict[str, object]:
        """Diagnostic summary; informational, safe to log."""
        record = self.certification.record if self.certification else None
        return {
            "family": self.family,
            "artifact_sha256": self.artifact_sha256,
            "oracle_package": self.oracle_package,
            "oracle_version": self.oracle_version,
            "importable_backends": sorted(self.importable_backends),
            "devices_probed": self.devices_probed,
            "devices": [
                {
                    "index": device.index,
                    "name": device.name,
                    "architecture": device.architecture,
                }
                for device in self.devices
            ],
            "driver_version": self.driver_version,
            "kernel_built": self.kernel_cache.built,
            "kernel_build_failed": self.kernel_cache.build_failed,
            "kernel_delivery": self.kernel_cache.delivery,
            "kernel_preferred_delivery": self.kernel_cache.preferred_delivery,
            "prebuilt_host_source_digest": self.kernel_cache.host_source_digest,
            "prebuilt_host_build_flags": list(
                self.kernel_cache.host_build_flags
            ),
            "prebuilt_host_toolchain": self.kernel_cache.host_toolchain,
            "fast_cpu_engine_delivery": "integrated",
            "fast_cpu_engine_module": ENGINE_MODULE,
            "fast_cpu_engine_version": self.fast_cpu_engine.version,
            "fast_cpu_binary_digest": self.fast_cpu_engine.binary_digest,
            "fast_cpu_source_digest": self.fast_cpu_engine.source_digest,
            "fast_cpu_build_flags": list(self.fast_cpu_engine.build_flags),
            "fast_cpu_toolchain": self.fast_cpu_engine.toolchain,
            "fast_cpu_config_digest": self.fast_cpu_engine.config_digest,
            "certification_identity": (
                self.certification.identity if self.certification else None
            ),
            "certified_family": record.family if record else None,
            "evidence_id": record.evidence_id if record else None,
        }


def importable_backends(
    *,
    extra_requirements: Iterable[tuple[str, Sequence[str]]] = (),
) -> frozenset[str]:
    """Backend ids whose implementation modules can be imported.

    Presence is established with module lookup, not import: asking the
    question must not have the side effect of answering it differently
    next time. The reference backend counts as present only when the
    oracle package is installed, since it is the oracle package that
    executes.
    """
    present: set[str] = set()
    if find_spec(ORACLE_PACKAGE) is not None:
        present.add(BACKEND_REFERENCE)
    requirements: list[tuple[str, Sequence[str]]] = [
        (BACKEND_GPU, ("torch", "toktier.engine.gpu")),
        (BACKEND_FAST_CPU, (ENGINE_MODULE, "transformers")),
    ]
    requirements.extend(extra_requirements)
    for backend_id, modules in requirements:
        if all(_module_present(module) for module in modules):
            present.add(backend_id)
    return frozenset(present)


def _module_present(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        # A parent package that cannot be imported means the child is
        # not usable either; that is a "no", not an exception to raise
        # out of a probe.
        return False


def probe(
    *,
    family: str,
    registry: RegistryView,
    artifact_sha256: str | None = None,
    pipeline_fingerprint: str | None = None,
    added_frontend_fingerprint: str | None = None,
    device_probe: DeviceProbe | None = None,
    installed_oracle_version: str | None = None,
    engine_facts: FastCpuEngineFacts | None = None,
) -> ProbeSnapshot:
    """Collect the routing facts for one artifact. Changes nothing."""
    devices_source: DeviceProbe = device_probe or NoDevices()
    match = registry.certification(
        artifact_sha256=artifact_sha256,
        pipeline_fingerprint=pipeline_fingerprint,
        added_frontend_fingerprint=added_frontend_fingerprint,
    )
    version = (
        installed_oracle_version
        if installed_oracle_version is not None
        else oracle_version()
    )
    return ProbeSnapshot(
        family=family,
        artifact_sha256=artifact_sha256,
        pipeline_fingerprint=pipeline_fingerprint,
        added_frontend_fingerprint=added_frontend_fingerprint,
        oracle_version=version,
        importable_backends=importable_backends(),
        devices=devices_source.devices(),
        devices_probed=device_probe is not None,
        driver_version=devices_source.driver_version(),
        kernel_cache=devices_source.kernel_cache(),
        fast_cpu_engine=(
            engine_facts if engine_facts is not None else fast_cpu_engine_facts()
        ),
        certification=match,
    )
