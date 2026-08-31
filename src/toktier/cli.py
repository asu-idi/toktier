"""Command-line interface for toktier."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from . import __version__
from .artifacts import (
    ArtifactEntry,
    ArtifactManifest,
    ArtifactSource,
    ArtifactStore,
    HuggingFaceSource,
    VerifiedArtifact,
    export_bundle,
    import_bundle,
)
from .artifacts.conversion import (
    ConvertingSource,
    conversion_report,
    recipe_for,
)
from .artifacts.store import fetch_availability
from .artifacts.tables import ARTIFACT_MANIFEST
from .config import Config
from .errors import (
    ArtifactHashMismatch,
    BackendUnavailable,
    ToktierError,
    UnsupportedConfig,
)
from .paths import (
    artifact_cache_dir,
    kernel_cache_dir,
    private_dir_problem,
    store_state_dir,
)
from .policy import BACKEND_FAST_CPU, BACKEND_GPU, RoutingPolicy

if TYPE_CHECKING:
    from .backends.fast_cpu import FastCpuEngineFacts
    from .engine.gpu.toolchain import NvccFacts
    from .routing.probe import DeviceInfo, KernelCacheState

_USAGE_ERROR = 64
_TOKTIER_ERROR = 2
#: A local verification that ran and disagreed. Deliberately the same
#: code a refusal exits with: a script that treats non-zero as "do not
#: rely on this route" is reading it correctly either way.
_VERIFY_MISMATCH = _TOKTIER_ERROR


class _UsageError(Exception):
    """Signal an argparse usage failure without terminating the process."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self._print_message(f"{self.prog}: error: {message}\n", sys.stderr)
        raise _UsageError


def _toktier_version() -> str:
    try:
        return importlib.metadata.version("toktier")
    except importlib.metadata.PackageNotFoundError:
        return __version__


def _installed_version(distribution: str) -> str | None:
    """Installed distribution version without importing its runtime."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvcc_report() -> NvccFacts:
    """Inspect ``nvcc`` the way the JIT loader's build system does.

    The kernel loader delegates compilation to the torch extension
    build system, which resolves the CUDA toolkit in this order: the
    ``CUDA_HOME`` environment variable, then ``CUDA_PATH``, then the
    ``nvcc`` on ``PATH``, then the conventional ``/usr/local/cuda``
    root. This check walks the same order without importing torch and
    records every location it consulted and parses ``nvcc --version``, so
    the report names the compiler identity JIT certification actually binds
    rather than answering from a single ``PATH`` lookup.

    An explicitly set ``CUDA_HOME``/``CUDA_PATH`` is authoritative for
    the build system -- it stops the search whether or not ``nvcc`` is
    present under it -- and the same is true here.

    Environment caliber: the two variables read here are the CUDA
    toolchain's own, not toktier configuration; they are observed and
    reported, never stored and never acted on (``config.md`` Section 3
    keeps toktier's environment surface to the frozen five, read once
    by ``Config``).
    """
    from .engine.gpu.toolchain import nvcc_facts

    return nvcc_facts()


def _jit_nvcc_report() -> NvccFacts:
    """The compiler the JIT build system would actually select.

    ``_nvcc_report`` walks the toolkit roots and is what the ``nvcc_*``
    fields describe. This one asks the question the way the planner
    does, through the root PyTorch's extension builder cached when it
    was imported, so the eligibility judgement below cannot name a
    different compiler than the one a build would use. With torch
    absent or unable to expose a root it falls back to the same search.
    """
    from .engine.gpu.toolchain import selected_nvcc_facts

    return selected_nvcc_facts()


def _torch_build_facts() -> tuple[str, str] | None:
    """``(distribution version, runtime CUDA)`` of the installed torch.

    ``None`` when torch cannot be imported. The read itself lives in the
    GPU engine lane, which is the only lane allowed to touch an
    accelerator runtime; this is the seam ``doctor`` calls it through.
    """
    from .engine.gpu.toolchain import installed_torch_facts

    return installed_torch_facts()


def _jit_toolchain_judgement(
    *, delivery: str, ninja_available: bool
) -> tuple[bool | None, str | None, str | None]:
    """Judge the JIT compiler/runtime triple the way the planner does.

    Returns ``(satisfied, observed, constraint)``. The judgement itself
    is the shared one (``engine.gpu.toolchain``), so ``doctor`` and a
    refusing plan cannot disagree about the same machine. It is a pure
    check: nothing is compiled, no tokenizer is constructed, and no
    kernel is loaded.

    Under the prebuilt delivery the automatic route has no JIT toolchain
    premise, so all three values are ``None``: the keys stay present in
    every profile, and ``None`` reads as "not applicable here" exactly
    as it does for the same field name in
    ``GpuEngine.certification_report()``. Reporting ``True`` instead
    would claim a judged compiler on a machine that may have none.
    """
    if delivery != "jit":
        return None, None, None
    from .engine.gpu.toolchain import (
        JIT_TOOLCHAIN_CONSTRAINT,
        jit_toolchain_observation,
        jit_toolchain_satisfied,
    )

    torch_facts = _torch_build_facts()
    if torch_facts is None:
        return False, None, JIT_TOOLCHAIN_CONSTRAINT
    torch_version, torch_cuda = torch_facts
    compiler = _jit_nvcc_report()
    observed = jit_toolchain_observation(
        torch_cuda=torch_cuda, torch_version=torch_version, nvcc=compiler
    )
    satisfied = jit_toolchain_satisfied(
        torch_cuda=torch_cuda,
        torch_version=torch_version,
        nvcc=compiler,
        ninja_present=ninja_available,
    )
    return satisfied, observed, JIT_TOOLCHAIN_CONSTRAINT


def _prebuilt_facts() -> tuple[bool, str | None]:
    """Whether a servable prebuilt fatbin ships in this installation.

    Read-only and torch-free, like every other probe of this command:
    the fatbin file must be present and match the digest its build
    manifest records. The answer comes from the one shared helper the
    routing probe also reports through, so ``doctor`` and ``explain()``
    cannot disagree about the shipped prebuilt state. Driver and device
    facts are deliberately not consulted here -- they belong to the load
    attempt, which states its own reason when it refuses.
    """
    from toktier.kernels.prebuilt import shipped_prebuilt_facts

    return shipped_prebuilt_facts()


class _PlanProbe:
    """The devices this command observed, in the shape the planner takes.

    ``doctor`` inspects and loads no kernel, so the kernel slot carries
    the shipped, read-only facts of the installation -- the same ones
    this command prints under ``prebuilt_*`` -- rather than the loader
    state of a process that has built or loaded one. Device and driver
    facts are the ones already probed for this report, so the plan is
    read against the machine, not against an empty stand-in.
    """

    def __init__(
        self, devices: Sequence[DeviceInfo], driver_version: str | None
    ) -> None:
        self._devices = tuple(devices)
        self._driver_version = driver_version

    def devices(self) -> tuple[DeviceInfo, ...]:
        return self._devices

    def driver_version(self) -> str | None:
        return self._driver_version

    def kernel_cache(self) -> KernelCacheState:
        from .routing.probe import NoDevices

        return NoDevices().kernel_cache()


def _plan_reasons(
    family: str,
    *,
    artifact_sha256: str | None,
    config: Config,
    devices: Sequence[DeviceInfo],
    driver_version: str | None,
) -> list[dict[str, object]] | None:
    """The planner's own reasons for this family, as ``explain()`` gives them.

    ``verify-local`` sends a reader here when the plan admitted no
    accelerated route, so the report has to carry what the plan
    recorded -- the reason code and its detail, including the axis and
    the expected and observed values of an engine binding that did not
    verify. Read-only, like the rest of this command: it plans against
    the facts already gathered for this report and loads nothing.

    ``None`` when the planner cannot answer at all (the reference
    package is not installed, or the registry cannot be read); the plan
    is then not a fact this command can report, and saying nothing is
    better than printing an empty list as "no reasons".
    """
    from .routing.explain import reason_to_dict
    from .routing.plan import plan as build_plan
    from .routing.probe import probe
    from .routing.registry_load import shipped_registry

    try:
        registry = shipped_registry()
        snapshot = probe(
            family=family,
            registry=registry,
            artifact_sha256=artifact_sha256,
            device_probe=_PlanProbe(devices, driver_version),
        )
        route = build_plan(snapshot, config.routing_policy, registry, config)
    except ToktierError:
        return None
    return [reason_to_dict(reason) for reason in route.reasons]


def _policy_admits_coverage_gaps(policy: RoutingPolicy) -> bool:
    """Whether this policy proceeds past a gap nobody has measured.

    The planner has exactly two such refusals -- a device architecture and
    a compiler/runtime pair no campaign judged (``docs/contracts/routing.md``
    checks 9 and 13). Everything the record binds still has to verify; what
    is missing is coverage. ``SUPPORTED``, the default since 0.2.6, admits
    both and labels the route ``supported_untested``; ``EXPERIMENTAL``
    waives them along with everything else waivable; ``CERTIFIED`` refuses
    them as it always has.

    ``doctor`` answers "what will actually run here?", so it has to apply
    the same rule the plan applies. Reading the certified-era conjunction
    on a machine running the default policy reported an ineligible GPU
    beside a request that went straight to it.
    """
    return policy.admits_unjudged_device() or policy is RoutingPolicy.EXPERIMENTAL


def _family_report(
    family: str,
    *,
    automatic_delivery: str,
    observed_architectures: Sequence[str],
    gpu_eligible: bool,
    cpu_profile_ready: bool,
    engine_facts: FastCpuEngineFacts,
    config: Config,
    devices: Sequence[DeviceInfo] = (),
    driver_version: str | None = None,
) -> dict[str, object]:
    """What this machine would do with one named family.

    The report without a family answers for the installation: it can say
    a certified CPU fast path exists here, not that it exists for the
    family a caller is about to load. Families differ -- some route
    their CPU work to the reference engine by design -- so the effective
    backend is only exact once the family is named.
    """
    from .routing.registry_load import shipped_registry
    from .routing.registry_view import STATUS_CERTIFIED, STATUS_CERTIFIED_SOURCE

    entry = _artifact_manifest().get(family)
    artifact_sha256 = next(
        (item.sha256 for item in entry.files if item.name == "tokenizer.json"),
        None,
    )
    match = shipped_registry().certification(artifact_sha256=artifact_sha256)
    certified = {STATUS_CERTIFIED, STATUS_CERTIFIED_SOURCE}
    fast_cpu_status: str | None = None
    gpu_status: str | None = None
    fast_cpu = None
    architecture_certification: dict[str, str] = {}
    if match is not None:
        fast_cpu = match.record.backends.get("fast_cpu")
        gpu = match.record.backends.get("gpu")
        fast_cpu_status = None if fast_cpu is None else fast_cpu.status
        if gpu is not None:
            delivery = gpu.for_delivery(automatic_delivery)
            gpu_status = delivery.status
            statuses = delivery.architecture_statuses()
            architecture_certification = {
                architecture: statuses.get(architecture, "uncertified")
                for architecture in observed_architectures
            }
    # The plan refuses a record whose GPU entry is absent or carries a
    # status outside the eligible set under every policy, and treats an
    # architecture no campaign judged as a coverage gap the policy in
    # effect may admit.
    architecture_admitted = any(
        status in certified for status in architecture_certification.values()
    )
    coverage_admitted = _policy_admits_coverage_gaps(config.routing_policy)
    family_gpu_eligible = (
        gpu_eligible
        and gpu_status in certified
        and (architecture_admitted or coverage_admitted)
    )
    # Under REFERENCE the plan admits no accelerated backend at all, so
    # neither half of this family's answer survives it. ``gpu_eligible``
    # already carries that premise from the installation-level report;
    # the CPU lane needs it named here too.
    # The plan does not stop at the entry's status: check 8 verifies the
    # executing extension against the binding that entry carries, and a
    # binding that does not verify sends this family to the reference
    # engine. Reading the status alone put two answers in one object --
    # "fast_cpu will run" beside a ``plan_reasons`` entry refusing
    # fast_cpu and naming the axis that disagreed. Same list of axes, so
    # the two cannot come apart again.
    from .routing.plan import fast_cpu_binding_mismatches

    family_cpu_ready = (
        config.routing_policy is not RoutingPolicy.REFERENCE
        and cpu_profile_ready
        and fast_cpu_status in certified
        and fast_cpu is not None
        and not fast_cpu_binding_mismatches(fast_cpu, engine_facts)
    )
    return {
        "family": entry.family,
        "artifact_sha256": artifact_sha256,
        "certification_identity": None if match is None else match.identity,
        "evidence_id": None if match is None else match.record.evidence_id,
        "fast_cpu_status": fast_cpu_status,
        "gpu_status": gpu_status,
        "gpu_delivery_certification": architecture_certification,
        "automatic_gpu_eligible": family_gpu_eligible,
        # The automatic route decides by input size, so the honest answer
        # is two answers: what runs at or above the GPU threshold, and
        # what runs below it. The second is where families differ most --
        # one whose CPU lane is the reference engine reads "hf" here
        # while the installation-level report reads "fast_cpu".
        "automatic_effective_backend": (
            "gpu" if family_gpu_eligible else "fast_cpu" if family_cpu_ready else "hf"
        ),
        "automatic_effective_backend_below_gpu_threshold": (
            "fast_cpu" if family_cpu_ready else "hf"
        ),
        # Why the planner admitted what it did. The two keys above say
        # what would run; this one says what the planner recorded on the
        # way there, which is where a binding that did not verify names
        # its axis and its expected and observed values. ``verify-local``
        # points at this command for exactly that answer.
        "plan_reasons": _plan_reasons(
            entry.family,
            artifact_sha256=artifact_sha256,
            config=config,
            devices=devices,
            driver_version=driver_version,
        ),
    }


def _directory_roots_problem(config: Config) -> str | None:
    """Why this machine's resolved roots cannot hold what they are for.

    ``None`` when all three are fine. The three usually share a home, so
    one unusable location is one problem named once, with the fields it
    stands behind listed in front of it.

    Read-only, like every other probe of this command: the roots are not
    created here, because creating them to find out whether they can be
    created would make the probe an operation.
    """
    problems: dict[str, list[str]] = {}
    for name, path in (
        ("artifact_cache_dir", artifact_cache_dir(config)),
        ("kernel_cache_dir", kernel_cache_dir(config)),
        ("store_state_dir", store_state_dir(config)),
    ):
        problem = private_dir_problem(path)
        if problem is not None:
            problems.setdefault(problem, []).append(name)
    return (
        "; ".join(
            f"{', '.join(names)}: {problem}"
            for problem, names in problems.items()
        )
        or None
    )


def _doctor_report(
    config: Config, *, source: ArtifactSource | None, family: str | None = None
) -> dict[str, object]:
    # Fetch availability is reported field by field. A single "offline"
    # line would answer three different questions with one word, and the
    # interesting case is exactly the one it hides: a configuration that
    # is not offline in front of a source that is.
    availability = fetch_availability(config, source)
    # ``cuda_available`` below stays a spec lookup of the top-level cuda
    # package: it answers whether the CUDA runtime binding is installed
    # without importing an accelerator runtime for that question alone.
    # The device inventory and, under JIT delivery, the toolchain
    # judgement do consult the installed torch -- that is the machine
    # fact they report -- and neither builds or loads a kernel.
    nvcc = _nvcc_report()
    # The nvcc trail concerns the JIT delivery only; the prebuilt
    # delivery loads the shipped fatbin through the driver and needs no
    # toolkit, so the two facts are reported side by side rather than
    # letting a missing nvcc read as "no GPU path".
    prebuilt_available, prebuilt_digest = _prebuilt_facts()
    from .backends.fast_cpu import (
        ENGINE_DELIVERY,
        ENGINE_MODULE,
        fast_cpu_engine_facts,
    )
    from .engine.gpu.native import native_host_build_facts
    from .facade.api import DEFAULT_GPU_MIN_BYTES

    fast_cpu = fast_cpu_engine_facts()
    native_host = native_host_build_facts()
    from .repair import fastokens as fastokens_adapter

    # The adapter's environment-level answer: which bytes ``import fastokens``
    # would run, whose distribution they are, and what the shipped registry
    # knows about them. The family premise is applied only with --family.
    fastokens_identity = fastokens_adapter.fastokens_identity()
    torch_available = importlib.util.find_spec("torch") is not None
    transformers_available = importlib.util.find_spec("transformers") is not None
    ninja_available = importlib.util.find_spec("ninja") is not None
    tokenizers_version = _installed_version("tokenizers")
    transformers_version = _installed_version("transformers")
    automatic_delivery = "jit" if ninja_available else "prebuilt"
    from .engine.gpu.host_probe import CudaHostProbe
    from .policy import BACKEND_GPU
    from .routing.registry_load import shipped_registry

    device_probe = CudaHostProbe(config=config, delivery=automatic_delivery)
    probed_devices = device_probe.devices()
    driver_version = device_probe.driver_version()
    delivery_statuses = shipped_registry().shared_delivery_architecture_statuses(
        BACKEND_GPU, automatic_delivery
    )
    observed_architectures = dict.fromkeys(
        device.architecture for device in probed_devices
    )
    architecture_certification = {
        architecture: delivery_statuses.get(architecture, "uncertified")
        for architecture in observed_architectures
    }
    from .routing.registry_view import STATUS_CERTIFIED, STATUS_CERTIFIED_SOURCE

    # The planner admits the delivery when *some* observed architecture is
    # in its judged list, so the same "any" rule applies here.
    architecture_admitted = any(
        status in {STATUS_CERTIFIED, STATUS_CERTIFIED_SOURCE}
        for status in architecture_certification.values()
    )
    certified_cpu_profile_ready = (
        fast_cpu.version is not None
        and fast_cpu.source_digest is not None
        and bool(fast_cpu.build_flags)
        and fast_cpu.toolchain is not None
        and fast_cpu.config_digest is not None
        and tokenizers_version == "0.22.2"
        and transformers_version == "4.57.6"
    )
    gigatoken_runtime_ready = (
        fast_cpu.version is not None and transformers_available
    )
    prebuilt_native_host_ready = (
        prebuilt_available
        and native_host.source_digest is not None
        and bool(native_host.build_flags)
        and native_host.toolchain is not None
    )
    # Installation-level candidacy: the GPU runtime is installed and no
    # configuration excluded the backend. It says nothing about this
    # machine's devices or, under JIT, its compiler.
    automatic_gpu_candidate = torch_available and not config.disable_gpu
    jit_satisfied, jit_observed, jit_constraint = _jit_toolchain_judgement(
        delivery=automatic_delivery, ninja_available=ninja_available
    )
    delivery_ready = (
        ninja_available if automatic_delivery == "jit" else prebuilt_native_host_ready
    )
    # The conjunction a caller actually wants: candidacy, an observed
    # device, that delivery's own materials, and -- where the policy in
    # effect insists on them -- the two coverage premises, a judged
    # architecture and a judged compiler/runtime pair (``None`` is "not
    # applicable", not a refusal). The default policy admits both and
    # labels such a route ``supported_untested``, so requiring them here
    # would describe a stricter installation than the next request gets.
    coverage_admitted = _policy_admits_coverage_gaps(config.routing_policy)
    # REFERENCE is a third kind of premise, and the one this report used to
    # miss. It is not about coverage: check 1 of ``routing/plan.py`` refuses
    # every accelerated backend under it, unwaivably, before any of the
    # judgements above are consulted. Reading them anyway reported an
    # eligible GPU, or a fast CPU lane, on an installation whose next
    # request goes straight to the reference engine.
    accelerated_planned = config.routing_policy is not RoutingPolicy.REFERENCE
    automatic_gpu_eligible = (
        accelerated_planned
        and automatic_gpu_candidate
        and bool(probed_devices)
        and delivery_ready
        and (architecture_admitted or coverage_admitted)
        and (jit_satisfied is not False or coverage_admitted)
    )
    directory_roots_problem = _directory_roots_problem(config)
    family_artifact_sha256: str | None = None
    if family is not None:
        family_artifact_sha256 = next(
            (
                item.sha256
                for item in _artifact_manifest().get(family).files
                if item.name == "tokenizer.json"
            ),
            None,
        )
    # The same premise the family block applies, asked without a family.
    # ``certified_cpu_profile_ready`` says the engine reported its facts;
    # it never compared them with what a record binds, so this answer
    # could read ``fast_cpu`` on an installation where the plan refuses
    # that backend for every family. With no family named, the honest
    # question is whether the installed engine verifies against any
    # record eligible to take the fast path at all.
    from .policy import BACKEND_FAST_CPU
    from .routing.plan import fast_cpu_binding_mismatches

    fast_cpu_binding_verifies = any(
        not fast_cpu_binding_mismatches(entry, fast_cpu)
        for entry in shipped_registry().eligible_entries(BACKEND_FAST_CPU)
    )
    automatic_effective_backend = (
        "gpu"
        if automatic_gpu_eligible
        else "fast_cpu"
        if accelerated_planned
        and certified_cpu_profile_ready
        and gigatoken_runtime_ready
        and fast_cpu_binding_verifies
        else "hf"
    )
    return {
        "python_version": platform.python_version(),
        "toktier_version": _toktier_version(),
        # Present in every report; populated only when a family is named,
        # because the answer is only exact then.
        "family": (
            None
            if family is None
            else _family_report(
                family,
                automatic_delivery=automatic_delivery,
                observed_architectures=list(observed_architectures),
                gpu_eligible=automatic_gpu_eligible,
                cpu_profile_ready=(
                    certified_cpu_profile_ready and gigatoken_runtime_ready
                ),
                engine_facts=fast_cpu,
                config=config,
                devices=probed_devices,
                driver_version=driver_version,
            )
        ),
        "artifact_cache_dir": str(artifact_cache_dir(config)),
        "kernel_cache_dir": str(kernel_cache_dir(config)),
        "store_state_dir": str(store_state_dir(config)),
        # Whether the three paths above can actually hold what they are
        # for. Printing a layout that the next command will refuse with
        # CONFIG_INVALID would answer "what will actually run here" with
        # something that will not.
        "directory_roots_usable": directory_roots_problem is None,
        "directory_roots_problem": directory_roots_problem,
        "configured_offline": availability.configured_offline,
        "artifact_source": availability.source_name or "none",
        "source_offline": availability.source_offline,
        "artifact_fetch_available": availability.available,
        "tokenizers_version": tokenizers_version,
        "transformers_version": transformers_version,
        "certified_cpu_profile_ready": certified_cpu_profile_ready,
        "torch_available": torch_available,
        "ninja_available": ninja_available,
        "automatic_gpu_delivery": automatic_delivery,
        "automatic_gpu_min_bytes": DEFAULT_GPU_MIN_BYTES,
        # Installation-level: torch present and the GPU not disabled.
        # ``automatic_gpu_eligible`` is the full judgement.
        "automatic_gpu_candidate": automatic_gpu_candidate,
        # Which policy the three answers below were computed under. They
        # are not properties of the machine alone: the same installation
        # reads a different effective backend under REFERENCE, CERTIFIED
        # and SUPPORTED, and a report that does not say which one it
        # applied cannot be compared with another machine's. The family
        # block answers under this same policy.
        "automatic_routing_policy": config.routing_policy.value,
        "automatic_gpu_eligible": automatic_gpu_eligible,
        "automatic_effective_backend": automatic_effective_backend,
        "jit_toolchain_satisfied": jit_satisfied,
        "jit_toolchain_observed": jit_observed,
        "jit_toolchain_constraint": jit_constraint,
        "cuda_available": importlib.util.find_spec("cuda") is not None,
        "cuda_hardware_present": bool(probed_devices),
        "devices": [
            {
                "index": device.index,
                "name": device.name,
                "architecture": device.architecture,
            }
            for device in probed_devices
        ],
        "driver_version": driver_version,
        "automatic_gpu_delivery_certification": architecture_certification,
        "prebuilt_fatbin_available": prebuilt_available,
        "prebuilt_fatbin_digest": prebuilt_digest,
        "prebuilt_native_host_ready": prebuilt_native_host_ready,
        "prebuilt_host_source_digest": native_host.source_digest,
        "prebuilt_host_build_flags": list(native_host.build_flags),
        "prebuilt_host_toolchain": native_host.toolchain,
        "gigatoken_available": fast_cpu.version is not None,
        "gigatoken_delivery": ENGINE_DELIVERY,
        "gigatoken_module": ENGINE_MODULE,
        "gigatoken_runtime_ready": gigatoken_runtime_ready,
        "gigatoken_version": fast_cpu.version,
        # Retained as a compatibility key. Integrated, source-certified
        # builds intentionally have no independently certified CPU binary.
        "gigatoken_native_digest": fast_cpu.binary_digest,
        "gigatoken_source_digest": fast_cpu.source_digest,
        "gigatoken_build_flags": list(fast_cpu.build_flags),
        "gigatoken_toolchain": fast_cpu.toolchain,
        "gigatoken_repair_config_digest": fast_cpu.config_digest,
        **_fastokens_doctor_facts(
            fastokens_adapter,
            fastokens_identity,
            tokenizers_version=tokenizers_version,
            family=family,
            family_artifact=family_artifact_sha256,
        ),
        "nvcc_available": nvcc.path is not None,
        "nvcc_path": nvcc.path,
        "nvcc_resolved_path": nvcc.resolved_path,
        "nvcc_release": nvcc.release,
        "nvcc_build": nvcc.build,
        "nvcc_error": nvcc.error,
        "nvcc_checked": list(nvcc.checked),
    }


def _fastokens_doctor_facts(
    adapter: Any,
    identity: Any,
    *,
    tokenizers_version: str | None,
    family: str | None,
    family_artifact: str | None,
) -> dict[str, object]:
    """The ``fastokens_*`` doctor keys: admission word plus engine assurance.

    With ``--family`` the family premise is applied as well, and it has
    two halves that are not the same question. Whether the adapter can
    be opened for the family at all is the repair table's answer
    (``fastokens_family_admitted``); whether the pinned readings cover
    it is the evidence's (``engine_assurance``). A family the adapter
    cannot open carries no guarantee here whatever the engine is, so
    ``fastokens_exact_id_guarantee`` reads ``false`` for it while
    ``fastokens_engine_assurance`` keeps stating the engine-level fact.
    """
    entry = adapter.pinned_engine_entry()
    guard = adapter.compile_unicode_guard(entry)
    orphaned = "; ".join(owner.label for owner in identity.orphaned) or None
    admitted: bool | None = None
    admission_reason: str | None = None
    if family is not None:
        admitted = adapter.family_admitted(family, family_artifact)
        if not admitted:
            admission_reason = (
                "the adapter has no repair-table entry for this family and "
                "artifact, so a session that requests it is refused with "
                "UnsupportedConfig; the engine assurance beside this line is "
                "about the installed engine, not about this family"
            )
    if not identity.available:
        return {
            "fastokens_available": False,
            "fastokens_distribution": None,
            "fastokens_version": None,
            "fastokens_distribution_digest": None,
            "fastokens_known_wheel": None,
            "fastokens_engine_assurance": None,
            "fastokens_exact_id_guarantee": False,
            "fastokens_policy": "experimental",
            "fastokens_family_admitted": admitted,
            "fastokens_family_admission_reason": admission_reason,
            "fastokens_coinstalled": None,
            "fastokens_orphaned": orphaned,
            "fastokens_advisory": None,
        }
    report = adapter.assess(
        identity,
        entry=entry,
        guard=guard,
        oracle_version=tokenizers_version,
        family=family,
        artifact_sha256=family_artifact,
    )
    coinstalled = identity.coinstalled
    return {
        "fastokens_available": True,
        "fastokens_distribution": identity.distribution,
        "fastokens_version": identity.version,
        "fastokens_distribution_digest": identity.engine_digest,
        "fastokens_known_wheel": (
            report.known_wheel["filename"] if report.known_wheel else None
        ),
        "fastokens_engine_assurance": report.assurance,
        "fastokens_exact_id_guarantee": (
            report.exact_id_guarantee and admitted is not False
        ),
        "fastokens_policy": "experimental",
        "fastokens_family_admitted": admitted,
        "fastokens_family_admission_reason": admission_reason,
        "fastokens_coinstalled": (
            ", ".join(owner.label for owner in coinstalled)
            + (
                " (their files were overwritten; uninstalling any of them "
                "removes the shared files)"
                if len(coinstalled) > 1
                else " (its files were overwritten; uninstalling either "
                "removes the shared files)"
            )
            if coinstalled
            else None
        ),
        "fastokens_orphaned": orphaned,
        # The same sentence ``explain()`` carries, so a reader of one
        # face is not told less than a reader of the other.
        "fastokens_advisory": report.advisory,
    }


#: Fields whose printed line says what the value is for. The driver and
#: the CUDA runtime are observed and reported because a reader wants to
#: know them, not because a certificate rests on them; a registry row
#: that binds a driver floor is checked separately, as a precondition
#: for the kernel loading at all.
_DOCTOR_QUALIFIERS: dict[str, str] = {
    "driver_version": "environment fact; not a certificate premise",
    "cuda_available": "environment fact; not a certificate premise",
}

#: Qualifiers printed only when the value is ``true``: a guarantee that
#: holds says in the same line what it means.
_DOCTOR_TRUE_QUALIFIERS: dict[str, str] = {
    "fastokens_exact_id_guarantee": (
        "guarded: ids equal the pinned reference or the request is routed "
        "to it; families and evidence in explain()"
    ),
}


def _print_doctor_human(report: dict[str, object]) -> None:
    for name, value in report.items():
        if name == "family" and isinstance(value, Mapping):
            # The one nested section: printed as its own indented block
            # rather than crushed onto a single line.
            print("family:")
            for key, item in value.items():
                if isinstance(item, Mapping):
                    nested = "; ".join(
                        f"{architecture}={status}"
                        for architecture, status in item.items()
                    )
                    print(f"  {key}: {nested or 'none'}")
                elif isinstance(item, bool):
                    print(f"  {key}: {str(item).lower()}")
                else:
                    print(f"  {key}: {'none' if item is None else item}")
            continue
        if name == "devices" and isinstance(value, list):
            rendered = "; ".join(
                f"{item['index']}: {item['name']} ({item['architecture']})"
                for item in value
                if isinstance(item, Mapping)
            ) or "none"
        elif isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, list):
            rendered = "; ".join(str(item) for item in value)
        elif isinstance(value, Mapping):
            rendered = "; ".join(
                f"{key}={item}" for key, item in value.items()
            ) or "none"
        elif value is None:
            rendered = "none"
        else:
            rendered = str(value)
        qualifier = _DOCTOR_QUALIFIERS.get(name)
        if qualifier is None and value is True:
            qualifier = _DOCTOR_TRUE_QUALIFIERS.get(name)
        if qualifier is not None:
            rendered = f"{rendered} ({qualifier})"
        print(f"{name}: {rendered}")


def _artifact_manifest() -> ArtifactManifest:
    """Return the artifact identities shipped with this package.

    The file is generated by ``tools/generate_artifact_manifest.py`` and
    installed inside the package, so a wheel resolves the same families
    a source checkout does. It is read on each call: the command runs
    once per process, and a cached manifest would only add a way for a
    long-lived process to keep stale digests.
    """
    return ArtifactManifest.load(ARTIFACT_MANIFEST)


def _artifact_store(config: Config, *, source: ArtifactSource | None) -> ArtifactStore:
    return ArtifactStore(_artifact_manifest(), config=config, source=source)


def _emit(payload: Mapping[str, object], *, as_json: bool, line: str) -> None:
    """Report one command's result in the requested shape.

    The prose line is unchanged from earlier releases; ``--json`` prints
    the same facts as an object, so a script never has to parse it.
    """
    if as_json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(line)


def _artifact_payload(action: str, artifact: VerifiedArtifact) -> dict[str, object]:
    return {
        "action": action,
        "family": artifact.family,
        "directory": str(artifact.directory),
    }


def _print_artifact(
    action: str, artifact: VerifiedArtifact, *, as_json: bool = False
) -> None:
    _emit(
        _artifact_payload(action, artifact),
        as_json=as_json,
        line=f"{action} {artifact.family}: {artifact.directory}",
    )


def _fetch_source() -> ArtifactSource:
    """The source every fetching command uses, built one way.

    The hub supplies the bytes; the conversion wrapper decides, per
    family, whether those bytes are the artifact or the pinned inputs of
    a local conversion. Families without a recipe are untouched by it,
    and it imports no hub client of its own.
    """
    return ConvertingSource(HuggingFaceSource())


def _doctor(arguments: argparse.Namespace) -> int:
    # The same source ``artifacts fetch`` would use, constructed the same
    # way. It reads its environment and imports no hub client, so the
    # report describes the fetch path this machine would actually take.
    report = _doctor_report(
        Config.resolve(), source=_fetch_source(), family=arguments.family
    )
    if arguments.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        _print_doctor_human(report)
    return 0


def _artifacts_check_conversion(arguments: argparse.Namespace) -> int:
    """Re-run the conversion gate of a locally derived artifact.

    Prints the report and fails when any of its three claims does:
    the conversion is deterministic, its bytes are the digest the
    shipped manifest pins, and the added-token block is the contiguous,
    fully described table the certified artifact carries.
    """
    manifest = _artifact_manifest()
    entry = manifest.get(arguments.family)
    recipe = recipe_for(entry.family)
    if recipe is None:
        raise UnsupportedConfig(
            f"{entry.family}: this family is downloaded whole, not converted",
            details={
                "option": "family",
                "value": entry.family,
                "reason": "no conversion recipe is registered for this family",
                "remedy": f"toktier artifacts verify {entry.family}",
            },
        )
    report = conversion_report(
        entry, recipe, HuggingFaceSource(), repeats=arguments.repeats
    )
    if arguments.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(f"family: {report['family']}")
        print(f"converter: {report['converter']}")
        print(f"upstream: {report['upstream_repo']}@{report['upstream_revision']}")
        for item in report["upstream_inputs"]:
            print(f"  input {item['name']}: sha256 {item['sha256']}")
        print(f"runs: {report['runs']} (deterministic: {report['deterministic']})")
        print(f"produced sha256: {report['observed_sha256']}")
        print(f"pinned sha256:   {report['expected_sha256']}")
        print(
            f"added tokens: {report['added_tokens']} "
            f"({report['added_tokens_special']} special, contiguous from "
            f"{report['added_tokens_first_id']})"
        )
    failures = [
        name
        for name in (
            "deterministic",
            "identity_matches",
            "added_tokens_contiguous",
            "added_tokens_fully_described",
        )
        if not report[name]
    ]
    if failures:
        raise ArtifactHashMismatch(
            f"conversion gate failed: {', '.join(failures)}",
            details={
                "family": report["family"],
                "expected_sha256": report["expected_sha256"],
                "observed_sha256": report["observed_sha256"],
                "failures": failures,
                "remedy": (
                    "re-run with the pinned upstream revision; the derived "
                    "artifact is not the one the shipped manifest pins"
                ),
            },
        )
    return 0


def _artifacts_fetch(arguments: argparse.Namespace) -> int:
    config = Config.resolve()
    store = _artifact_store(config, source=_fetch_source())
    if arguments.force:
        artifact = store.verify(arguments.family)
    else:
        artifact = store.ensure(arguments.family)
    _print_artifact("fetched", artifact, as_json=arguments.json)
    return 0


def _artifacts_verify(arguments: argparse.Namespace) -> int:
    store = _artifact_store(Config.resolve(), source=None)
    artifact = store.verify(arguments.family)
    _print_artifact("verified", artifact, as_json=arguments.json)
    return 0


def _artifacts_export(arguments: argparse.Namespace) -> int:
    # Re-hash before exporting (no source, so nothing is fetched): the
    # bundle records the digests of the bytes it packs, and this check
    # ties those bytes to the shipped manifest first. Bytes that fail
    # verification are never exported.
    store = _artifact_store(Config.resolve(), source=None)
    artifact = store.verify(arguments.family)
    entry = store.manifest.get(arguments.family)
    bundle = export_bundle(
        arguments.out,
        entry.directory_name,
        {item.name: artifact.path(item.name) for item in entry.files},
    )
    _emit(
        {"action": "exported", "family": artifact.family, "bundle": str(bundle)},
        as_json=arguments.json,
        line=f"exported {artifact.family}: {bundle}",
    )
    return 0


def _artifacts_import(arguments: argparse.Namespace) -> int:
    # The bundle is validated and every file digest-checked before the
    # atomic install; `artifacts verify FAMILY` afterwards binds the
    # installed bytes to the digests the shipped manifest pins.
    target = import_bundle(arguments.bundle, artifact_cache_dir(Config.resolve()))
    _emit(
        {"action": "imported", "entry": target.name, "directory": str(target)},
        as_json=arguments.json,
        line=f"imported {target.name}: {target}",
    )
    return 0


def _inspect_entry(entry: ArtifactEntry) -> dict[str, object]:
    return {
        "family": entry.family,
        "repo_id": entry.repo_id,
        "revision": entry.revision,
        "files": {
            item.name: {"sha256": item.sha256, "size": item.size}
            for item in entry.files
        },
    }


def _inspect(arguments: argparse.Namespace) -> int:
    """Print the identity the package pins, from the shipped manifest.

    Reads no cache and no network: the answer is what the package would
    verify against, which is the value a deployment check needs. With a
    family it prints that entry; without one, every family and its
    per-file digests.
    """
    manifest = _artifact_manifest()
    if arguments.family is not None:
        entries = [manifest.get(arguments.family)]
    else:
        entries = [manifest.get(family) for family in sorted(manifest.families())]
    if arguments.json:
        report: object = (
            _inspect_entry(entries[0])
            if arguments.family is not None
            else {entry.family: _inspect_entry(entry) for entry in entries}
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    for entry in entries:
        if arguments.family is not None:
            print(f"family: {entry.family}")
            print(f"repo_id: {entry.repo_id}")
            print(f"revision: {entry.revision}")
            for item in entry.files:
                print(f"{item.name}: sha256 {item.sha256} ({item.size} bytes)")
        else:
            for item in entry.files:
                print(f"{entry.family}: {item.name} sha256 {item.sha256}")
    return 0


def _version(arguments: argparse.Namespace) -> int:
    version = _toktier_version()
    _emit({"version": version}, as_json=arguments.json, line=version)
    return 0


def _gpu_compile(arguments: argparse.Namespace) -> int:
    """Build or reuse the JIT kernel under the configured policy.

    The first attempt runs under whatever policy this machine resolves,
    which since 0.2.6 is ``SUPPORTED`` unless the configuration says
    otherwise. A compiler pair no campaign judged compiles and runs
    there, and the route is labelled ``supported_untested`` rather than
    admitted quietly. ``--accept-uncertified-jit`` is what remains for
    the stricter policies: under ``CERTIFIED`` the refusal still happens
    and the flag is the one-process opt-in past it, exactly as before.
    """
    from .facade import load

    requested_acceptance = bool(arguments.accept_uncertified_jit)
    accepted = False
    policy = Config.resolve().routing_policy.value
    warning: str | None = None
    try:
        tokenizer = load(
            arguments.family,
            device="cuda",
            policy=policy,
            gpu_delivery="jit",
            gpu_min_bytes=0,
        )
    except BackendUnavailable as error:
        reason = error.details.get("reason")
        reason_detail = reason if isinstance(reason, Mapping) else {}
        toolchain_only = reason_detail.get("cause") == "toolchain_unverified"
        if not requested_acceptance or not toolchain_only:
            raise
        accepted = True
        policy = "experimental"
        observed = reason_detail.get("observed", "unknown")
        warning = (
            f"UNCERTIFIED JIT OPT-IN: observed {observed}; this toolchain "
            "combination is being "
            "compiled under EXPERIMENTAL policy. Its GPU results are outside "
            "TokTier's certified exact-ID guarantee."
        )
        print(f"WARNING: {warning}", file=sys.stderr)
        tokenizer = load(
            arguments.family,
            device="cuda",
            policy=policy,
            gpu_delivery="jit",
            gpu_min_bytes=0,
        )
    try:
        preflight = tokenizer.explain()
        preflight_waivers = preflight.get("experimental_waivers", [])
        if accepted and (
            not isinstance(preflight_waivers, list)
            or not preflight_waivers
            or not all(_is_jit_toolchain_waiver(item) for item in preflight_waivers)
        ):
            raise BackendUnavailable(
                "--accept-uncertified-jit accepts only an unjudged JIT "
                "toolchain pair; another certification premise also failed",
                details={
                    "backend": "gpu",
                    "accepted_scope": "jit_toolchain_only",
                    "experimental_waivers": preflight_waivers,
                },
            )
        tokenizer.encode("TokTier JIT compile probe", lookup="off")
        report = tokenizer.explain()
    finally:
        tokenizer.close()
    gpu_report = report.get("gpu_backend")
    runtime = report.get("runtime_policy")
    loaded = isinstance(gpu_report, dict) and gpu_report.get("loaded") is True
    executed = runtime.get("last_execution") if isinstance(runtime, dict) else None
    executed_gpu = (
        isinstance(executed, dict) and executed.get("executed_backend") == "gpu"
    )
    if report.get("kernel_delivery") != "jit" or not loaded or not executed_gpu:
        raise BackendUnavailable(
            "the JIT compile probe did not execute the GPU backend",
            details={
                "backend": "gpu",
                "kernel_delivery": report.get("kernel_delivery"),
                "gpu_backend": gpu_report,
                "runtime_policy": runtime,
            },
        )
    payload = {
        "family": arguments.family,
        "jit_ready": True,
        "kernel_delivery": "jit",
        "policy": policy,
        "requested_uncertified_jit_opt_in": requested_acceptance,
        "accepted_uncertified_jit": accepted,
        "experimental_waivers": report.get("experimental_waivers", []),
        # What the policy admitted on coverage rather than on evidence:
        # a device or a compiler pair nobody judged. Empty on a judged
        # combination, and never folded into the waivers above, which
        # say something different.
        "supported_untested": report.get("supported_untested", []),
        "certification_state": _certification_state(report),
        "warning": warning,
    }
    if arguments.json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        for name, value in payload.items():
            if isinstance(value, list):
                rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
            elif value is None:
                rendered = "none"
            else:
                rendered = str(value).lower() if isinstance(value, bool) else str(value)
            print(f"{name}: {rendered}")
    return 0


def _verify_documents(arguments: argparse.Namespace) -> tuple[list[str], str]:
    """The documents to compare, and where they came from.

    A caller's own text is the point of the command; the generator is
    there so somebody with nothing at hand can still run it. It builds
    documents from rules written in this package, so a check costs no
    license question and no network.
    """
    from .verify_local import generate, split_documents

    source = arguments.input
    if source is not None:
        text = (
            sys.stdin.read()
            if source == "-"
            else Path(source).read_text(encoding="utf-8")
        )
        documents = split_documents(text)
        if not documents:
            raise UnsupportedConfig(
                "the input holds no documents to compare",
                details={
                    "option": "--input",
                    "value": source,
                    "reason": "one document per non-empty line",
                },
            )
        return documents, "your text"
    return (
        generate(arguments.synthetic, arguments.max_bytes, arguments.seed),
        "generated",
    )


def _verify_one_engine(
    family: str,
    engine: str,
    documents: Sequence[str],
    *,
    config: Config,
    source: str,
    forget: bool,
) -> dict[str, object]:
    """Compare one engine with the reference engine, and record it."""
    from .facade import load
    from .verify_local import (
        compare,
        forget_record,
        read_record,
        record_for,
        write_record,
    )

    backend = BACKEND_GPU if engine == "gpu" else BACKEND_FAST_CPU
    # Every document goes to the engine under test, whatever its size:
    # the automatic crossover is a performance decision and this command
    # is asking a correctness question about one route.
    subject = load(
        family,
        config=config,
        device="cuda" if engine == "gpu" else "cpu",
        gpu_min_bytes=0,
    )
    try:
        key = subject.verification_key(engine)
        if key is None:
            raise BackendUnavailable(
                f"the {engine} route on this machine cannot be named, so a "
                "local check has nothing to file its answer under",
                details={"backend": backend, "engine": engine},
            )
        if forget:
            removed = forget_record(config, key)
            return {
                "engine": engine,
                "forgot_record": removed,
                "status": "forgotten" if removed else "no_record",
            }
        judge = load(family, config=config, device="cpu", policy="reference")
        try:
            comparison = compare(
                subject, judge, documents, expected_backend=backend
            )
        finally:
            judge.close()
    finally:
        subject.close()
    if not comparison.measured:
        # Nothing about the engine was measured, so nothing is recorded.
        # Reporting a pass here would be the one dishonest thing this
        # command could do.
        return {
            "engine": engine,
            "status": "not_measured",
            "documents": comparison.documents,
            "bytes": comparison.bytes,
            "mismatches": 0,
            "first_mismatch": None,
            "served_by_engine": comparison.served,
            # Which of two states this is: a route the plan did not
            # admit, answered by ``doctor``, or one it admitted that
            # every document left for a per-input reason, answered by
            # ``explain()`` on the same input. The paths those
            # documents took are listed so the reason is in the report.
            "route_admitted": comparison.admitted,
            "unserved_paths": [
                {"path": path, "documents": count}
                for path, count in comparison.unserved_paths
            ],
            "input": source,
            "record_path": None,
            "record_readable": False,
        }
    record = record_for(key, comparison, documents=documents, source=source)
    path = write_record(config, record)
    written = read_record(config, key)
    return {
        "engine": engine,
        "status": record.status,
        "documents": record.documents,
        "bytes": record.bytes,
        "mismatches": record.mismatches,
        "first_mismatch": (
            None
            if record.first_mismatch is None
            else list(record.first_mismatch)
        ),
        # How many of those documents the engine under test actually
        # served. A route that fell back would otherwise compare the
        # judge with itself and report an agreement nobody measured.
        "served_by_engine": comparison.served,
        "input": record.input,
        "input_digest": record.input_digest,
        "record_path": str(path),
        "record_readable": written is not None,
    }


def _verify_local(arguments: argparse.Namespace) -> int:
    """Compare an accelerated route with the reference engine, here.

    This is the command the ``supported_untested`` label points at. It
    is explicit and it stays explicit: nothing runs it automatically, it
    tokenizes only what the caller hands it or what the generator
    builds from rules, and what it writes is a record of a measurement
    rather than a certificate. A disagreement is reported and nothing is
    changed on the caller's behalf -- the route keeps the label it
    already had, and choosing ``policy="certified"`` is the way to hold
    the combination on the reference route.
    """
    config = Config.resolve()
    engines = (
        ["cpu", "gpu"] if arguments.engine == "both" else [arguments.engine]
    )
    forget = bool(arguments.forget)
    documents: list[str] = []
    source = "none"
    if not forget:
        documents, source = _verify_documents(arguments)
    results: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for engine in engines:
        try:
            results.append(
                _verify_one_engine(
                    arguments.family,
                    engine,
                    documents,
                    config=config,
                    source=source,
                    forget=forget,
                )
            )
        except ToktierError as error:
            # ``both`` on a machine with no usable GPU is the ordinary
            # case, not a failure of the command: the engine that could
            # not be opened is named and the other one still runs. A
            # single explicit ``--engine`` has nothing to fall back to,
            # so its error travels to the caller unchanged.
            if arguments.engine != "both":
                raise
            skipped.append(
                {
                    "engine": engine,
                    "status": "not_available",
                    "reason": error.code,
                    "message": str(error),
                }
            )
    payload: dict[str, object] = {
        "family": arguments.family,
        "engines": results,
        "skipped": skipped,
        "documents": len(documents),
        "input": source,
    }
    failed = [item for item in results if item.get("status") == "failed"]
    if arguments.json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        _print_verify_human(payload)
    return _VERIFY_MISMATCH if failed else 0


#: Width the notes below wrap to. A person reads them at a prompt, and
#: one physical line of several hundred characters is not read.
_NOTE_WIDTH = 88


def uncovered_note(
    engine: str,
    *,
    served: int,
    documents: int,
    unserved_paths: Sequence[Mapping[str, object]],
    route_admitted: bool,
) -> str:
    """Why a run measured nothing, or not enough, about a route.

    Three states reach here and the sentence says which. The text is
    wrapped and ends in a full stop, which the Rust face has pinned with
    a test since it grew these notes and this face had not: the promise
    was written in a release note and kept on one side only.
    """
    if not route_admitted:
        body = (
            f"the plan did not admit the {engine} route, so it served none "
            f"of the {documents} documents. This run measured nothing about "
            "it and no record was written. `toktier doctor --family "
            "<family>` reports the plan's own reasons for it under "
            "`family.plan_reasons`."
        )
    elif served == 0:
        recorded = (
            ", ".join(
                f"{path['path']} x{path['documents']}" for path in unserved_paths
            )
            or "no path was recorded"
        )
        body = (
            f"the {engine} route was admitted and served none of the "
            f"{documents} documents: each one left it for a per-input reason "
            f"({recorded}). This run measured nothing about it and no record "
            "was written. `explain()` on a tokenizer for the same input names "
            "the reason, and `toktier doctor` answers about the plan rather "
            "than about one input."
        )
    else:
        body = (
            f"the {engine} route served {served} of {documents} documents; "
            "the served ones compared equal, but the run does not cover the "
            "route and no record was written. The rest went to the reference "
            "path document by document, so a record needs an input the route "
            "serves throughout."
        )
    return "\n".join(
        textwrap.wrap(f"{engine}: {body}", width=_NOTE_WIDTH, subsequent_indent="  ")
    )


def _print_verify_human(payload: Mapping[str, object]) -> None:
    """One block per engine, in the words the label uses."""
    print(f"family: {payload['family']}")
    print(f"input: {payload['input']} ({payload['documents']} documents)")
    for item in cast(Sequence[Mapping[str, object]], payload["engines"]):
        engine = item["engine"]
        status = item["status"]
        if status in {"forgotten", "no_record"}:
            removed = status == "forgotten"
            print(
                f"{engine}: "
                f"{'record removed' if removed else 'no record to remove'}"
            )
            continue
        documents = item["documents"]
        if status == "not_measured":
            served = item["served_by_engine"]
            paths = cast(
                Sequence[Mapping[str, object]], item.get("unserved_paths", [])
            )
            print(
                uncovered_note(
                    str(engine),
                    served=cast(int, served),
                    documents=cast(int, documents),
                    unserved_paths=paths,
                    route_admitted=item.get("route_admitted") is not False,
                )
            )
            continue
        if status == "passed":
            print(
                f"{engine}: locally_verified -- you compared this "
                f"machine's {engine} route with the reference engine on "
                f"{documents} documents ({item['input']}); this record is "
                "not a certificate and expires when the driver, toolchain, "
                "kernel or source identity changes"
            )
        else:
            first = item["first_mismatch"]
            where = (
                ""
                if not isinstance(first, list)
                else f" (first: doc {first[0]} at token {first[1]})"
            )
            print(
                f"{engine}: local verification failed on "
                f"{item['mismatches']} of {documents} documents{where}. The "
                f"{engine} route on this machine does not match the "
                "reference engine for those inputs; select "
                "policy='certified' to keep this combination on the "
                "reference route. Nothing was changed automatically."
            )
        served = item["served_by_engine"]
        if isinstance(served, int) and served != documents:
            print(
                f"{engine}: {served} of {documents} documents were served by "
                f"the {engine} route; the rest fell back, so they compared "
                "the reference engine with itself"
            )
        print(f"{engine}: record {item['record_path']}")
    for item in cast(Sequence[Mapping[str, object]], payload["skipped"]):
        print(f"{item['engine']}: not available here -- {item['message']}")


def _certification_state(report: Mapping[str, object]) -> str | None:
    """The label one explanation gives the route it planned."""
    certification = report.get("certification")
    if not isinstance(certification, Mapping):
        return None
    state = certification.get("state")
    return state if isinstance(state, str) else None


def _is_jit_toolchain_waiver(value: object) -> bool:
    """Whether a serialized waiver is exactly the CLI flag's narrow scope."""
    if not isinstance(value, Mapping):
        return False
    detail = value.get("detail")
    return (
        value.get("backend") == "gpu"
        and value.get("code") == "R_UNCERTIFIED_ARTIFACT"
        and isinstance(detail, Mapping)
        and detail.get("cause") == "toolchain_unverified"
    )


def _json_option() -> _ArgumentParser:
    """The shared ``--json`` flag, accepted before or after the command.

    Every subparser inherits it, and the root parser carries it too, so
    ``toktier --json doctor`` and ``toktier doctor --json`` are the same
    request. The default is suppressed rather than ``False``: argparse
    parses a subcommand into a fresh namespace and copies every name it
    holds back over the outer one, so a default here would erase a flag
    the root parser had already recorded. :func:`main` reads the absence
    as ``False`` once, for all commands.
    """
    shared = _ArgumentParser(add_help=False)
    shared.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit JSON",
    )
    return shared


def _build_parser() -> argparse.ArgumentParser:
    shared = _json_option()
    parser = _ArgumentParser(prog="toktier", parents=[shared])
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser(
        "doctor", help="show environment diagnostics", parents=[shared]
    )
    doctor.add_argument(
        "--family",
        metavar="FAMILY",
        help="also report what this machine would do with one family",
    )
    doctor.set_defaults(handler=_doctor)

    artifacts = commands.add_parser(
        "artifacts", help="manage artifacts", parents=[shared]
    )
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", required=True)

    fetch = artifact_commands.add_parser(
        "fetch", help="fetch an artifact", parents=[shared]
    )
    fetch.add_argument("family", metavar="FAMILY")
    fetch.add_argument("--force", action="store_true", help="re-hash cached files")
    fetch.set_defaults(handler=_artifacts_fetch)

    verify = artifact_commands.add_parser(
        "verify", help="verify an artifact", parents=[shared]
    )
    verify.add_argument("family", metavar="FAMILY")
    verify.set_defaults(handler=_artifacts_verify)

    check_conversion = artifact_commands.add_parser(
        "check-conversion",
        help="re-run the conversion gate of a locally derived artifact",
        parents=[shared],
    )
    check_conversion.add_argument("family", metavar="FAMILY")
    check_conversion.add_argument(
        "--repeats",
        type=int,
        default=2,
        metavar="N",
        help="independent conversions to compare (default 2)",
    )
    check_conversion.set_defaults(handler=_artifacts_check_conversion)

    export = artifact_commands.add_parser(
        "export",
        help="write a verified artifact as an air-gap bundle",
        parents=[shared],
    )
    export.add_argument("family", metavar="FAMILY")
    export.add_argument(
        "--out",
        metavar="BUNDLE",
        required=True,
        help="path of the tar bundle to write",
    )
    export.set_defaults(handler=_artifacts_export)

    import_ = artifact_commands.add_parser(
        "import",
        help="verify a bundle and install it into the cache",
        parents=[shared],
    )
    import_.add_argument("bundle", metavar="BUNDLE")
    import_.set_defaults(handler=_artifacts_import)

    inspect = commands.add_parser(
        "inspect",
        help="show the artifact identity the package pins",
        parents=[shared],
    )
    inspect.add_argument("family", metavar="FAMILY", nargs="?")
    inspect.set_defaults(handler=_inspect)

    gpu = commands.add_parser(
        "gpu", help="prepare and diagnose GPU delivery", parents=[shared]
    )
    gpu_commands = gpu.add_subparsers(dest="gpu_command", required=True)
    compile_jit = gpu_commands.add_parser(
        "compile",
        help="build or reuse the JIT kernel for one family",
        parents=[shared],
    )
    compile_jit.add_argument("family", metavar="FAMILY")
    compile_jit.add_argument(
        "--accept-uncertified-jit",
        action="store_true",
        help=(
            "compile under EXPERIMENTAL policy even when the observed JIT "
            "toolchain is outside the certified set"
        ),
    )
    compile_jit.set_defaults(handler=_gpu_compile)

    # A top-level command rather than a `gpu` subcommand: it compares the
    # CPU fast path as readily as the GPU one, and it carries the same
    # name on both faces so there is one thing to remember.
    verify = commands.add_parser(
        "verify-local",
        help=(
            "compare an accelerated route with the reference engine on "
            "this machine and record the answer"
        ),
        parents=[shared],
    )
    # Named rather than positional, so the two faces take the same
    # command line: `toktier verify-local --family qwen3_8b --engine gpu`
    # and `toktier-rust verify-local --family qwen3_8b --engine gpu`.
    verify.add_argument(
        "--family",
        metavar="FAMILY",
        required=True,
        help="the family to compare",
    )
    verify.add_argument(
        "--engine",
        choices=("cpu", "gpu", "both"),
        default="both",
        help=(
            "which accelerated route to compare (default: both; an engine "
            "this machine cannot open is named and skipped)"
        ),
    )
    verify.add_argument(
        "--input",
        metavar="PATH",
        help=(
            "your own documents, one per non-empty line; '-' reads standard "
            "input. Without this, documents are generated from rules"
        ),
    )
    verify.add_argument(
        "--synthetic",
        type=int,
        default=2000,
        metavar="N",
        help="generated documents to compare when --input is absent",
    )
    verify.add_argument(
        "--max-bytes",
        type=int,
        default=4096,
        metavar="N",
        help="largest generated document, in bytes",
    )
    verify.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="seed of the generator, so a run can be repeated exactly",
    )
    verify.add_argument(
        "--forget",
        action="store_true",
        help="remove this machine's record for the selected engines",
    )
    verify.set_defaults(handler=_verify_local)

    version = commands.add_parser(
        "version", help="show the toktier version", parents=[shared]
    )
    version.set_defaults(handler=_version)
    return parser


def _json_safe(value: object) -> object:
    """A JSON-serialisable projection of one ``details`` value.

    Error details are a machine interface, but they are also open: a
    caller may put any object there. Rendering an unknown type through
    ``repr`` keeps the envelope serialisable without dropping the fact
    it carried.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return repr(value)


def _error_payload(error: ToktierError) -> dict[str, object]:
    """The machine-readable envelope for one failed command.

    ``code`` is the stable switch key (``docs/contracts/errors.md``);
    ``message`` is the human text; ``details`` carries the same
    machine-readable facts the exception exposes in process.
    """
    return {
        "error": {
            "code": error.code,
            "message": str(error),
            "details": {
                str(key): _json_safe(value)
                for key, value in error.details.items()
            },
        }
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the toktier command-line interface."""
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
    except _UsageError:
        return _USAGE_ERROR
    # The flag is suppressed rather than defaulted where it is declared;
    # its absence means it was not asked for.
    if not hasattr(arguments, "json"):
        arguments.json = False

    handler = cast(Callable[[argparse.Namespace], int], arguments.handler)
    try:
        return handler(arguments)
    except ToktierError as error:
        # ``--json`` is a promise about the whole command, not only its
        # success path: a caller that asked for machine-readable output
        # must not have to parse prose to learn why it failed.
        if arguments.json:
            print(
                json.dumps(
                    _error_payload(error), sort_keys=True, separators=(",", ":")
                ),
                file=sys.stderr,
            )
        else:
            print(f"error {error.code}: {error}", file=sys.stderr)
        return _TOKTIER_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
