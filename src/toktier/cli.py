"""Command-line interface for toktier."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NoReturn, cast

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
from .artifacts.store import fetch_availability
from .artifacts.tables import ARTIFACT_MANIFEST
from .config import Config
from .errors import BackendUnavailable, ToktierError
from .paths import artifact_cache_dir, kernel_cache_dir, store_state_dir

if TYPE_CHECKING:
    from .engine.gpu.toolchain import NvccFacts

_USAGE_ERROR = 64
_TOKTIER_ERROR = 2


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


def _doctor_report(
    config: Config, *, source: ArtifactSource | None
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
    from .repair.fastokens import fastokens_distribution_identity

    fastokens_version, fastokens_digest = fastokens_distribution_identity()
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
    # device whose architecture the selected delivery judges, that
    # delivery's own materials, and the toolchain premise where one
    # applies (``None`` is "not applicable", not a refusal).
    automatic_gpu_eligible = (
        automatic_gpu_candidate
        and bool(probed_devices)
        and architecture_admitted
        and delivery_ready
        and jit_satisfied is not False
    )
    automatic_effective_backend = (
        "gpu"
        if automatic_gpu_eligible
        else "fast_cpu"
        if certified_cpu_profile_ready and gigatoken_runtime_ready
        else "hf"
    )
    return {
        "python_version": platform.python_version(),
        "toktier_version": _toktier_version(),
        "artifact_cache_dir": str(artifact_cache_dir(config)),
        "kernel_cache_dir": str(kernel_cache_dir(config)),
        "store_state_dir": str(store_state_dir(config)),
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
        "fastokens_available": fastokens_version is not None,
        "fastokens_version": fastokens_version,
        "fastokens_distribution_digest": fastokens_digest,
        "fastokens_policy": "experimental",
        "fastokens_exact_id_guarantee": False,
        "nvcc_available": nvcc.path is not None,
        "nvcc_path": nvcc.path,
        "nvcc_resolved_path": nvcc.resolved_path,
        "nvcc_release": nvcc.release,
        "nvcc_build": nvcc.build,
        "nvcc_error": nvcc.error,
        "nvcc_checked": list(nvcc.checked),
    }


def _print_doctor_human(report: dict[str, object]) -> None:
    for name, value in report.items():
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


def _print_artifact(action: str, artifact: VerifiedArtifact) -> None:
    print(f"{action} {artifact.family}: {artifact.directory}")


def _doctor(arguments: argparse.Namespace) -> int:
    # The same source ``artifacts fetch`` would use, constructed the same
    # way. It reads its environment and imports no hub client, so the
    # report describes the fetch path this machine would actually take.
    report = _doctor_report(Config.resolve(), source=HuggingFaceSource())
    if arguments.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        _print_doctor_human(report)
    return 0


def _artifacts_fetch(arguments: argparse.Namespace) -> int:
    config = Config.resolve()
    store = _artifact_store(config, source=HuggingFaceSource())
    if arguments.force:
        artifact = store.verify(arguments.family)
    else:
        artifact = store.ensure(arguments.family)
    _print_artifact("fetched", artifact)
    return 0


def _artifacts_verify(arguments: argparse.Namespace) -> int:
    store = _artifact_store(Config.resolve(), source=None)
    artifact = store.verify(arguments.family)
    _print_artifact("verified", artifact)
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
    print(f"exported {artifact.family}: {bundle}")
    return 0


def _artifacts_import(arguments: argparse.Namespace) -> int:
    # The bundle is validated and every file digest-checked before the
    # atomic install; `artifacts verify FAMILY` afterwards binds the
    # installed bytes to the digests the shipped manifest pins.
    target = import_bundle(arguments.bundle, artifact_cache_dir(Config.resolve()))
    print(f"imported {target.name}: {target}")
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
    del arguments
    print(_toktier_version())
    return 0


def _gpu_compile(arguments: argparse.Namespace) -> int:
    """Build or reuse the JIT kernel under a certified or explicit-risk policy."""
    from .facade import load

    requested_acceptance = bool(arguments.accept_uncertified_jit)
    accepted = False
    policy = "certified"
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


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="toktier")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="show environment diagnostics")
    doctor.add_argument("--json", action="store_true", help="emit JSON")
    doctor.set_defaults(handler=_doctor)

    artifacts = commands.add_parser("artifacts", help="manage artifacts")
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", required=True)

    fetch = artifact_commands.add_parser("fetch", help="fetch an artifact")
    fetch.add_argument("family", metavar="FAMILY")
    fetch.add_argument("--force", action="store_true", help="re-hash cached files")
    fetch.set_defaults(handler=_artifacts_fetch)

    verify = artifact_commands.add_parser("verify", help="verify an artifact")
    verify.add_argument("family", metavar="FAMILY")
    verify.set_defaults(handler=_artifacts_verify)

    export = artifact_commands.add_parser(
        "export", help="write a verified artifact as an air-gap bundle"
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
        "import", help="verify a bundle and install it into the cache"
    )
    import_.add_argument("bundle", metavar="BUNDLE")
    import_.set_defaults(handler=_artifacts_import)

    inspect = commands.add_parser(
        "inspect", help="show the artifact identity the package pins"
    )
    inspect.add_argument("family", metavar="FAMILY", nargs="?")
    inspect.add_argument("--json", action="store_true", help="emit JSON")
    inspect.set_defaults(handler=_inspect)

    gpu = commands.add_parser("gpu", help="prepare and diagnose GPU delivery")
    gpu_commands = gpu.add_subparsers(dest="gpu_command", required=True)
    compile_jit = gpu_commands.add_parser(
        "compile", help="build or reuse the JIT kernel for one family"
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
    compile_jit.add_argument("--json", action="store_true", help="emit JSON")
    compile_jit.set_defaults(handler=_gpu_compile)

    version = commands.add_parser("version", help="show the toktier version")
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

    handler = cast(Callable[[argparse.Namespace], int], arguments.handler)
    try:
        return handler(arguments)
    except ToktierError as error:
        # ``--json`` is a promise about the whole command, not only its
        # success path: a caller that asked for machine-readable output
        # must not have to parse prose to learn why it failed.
        if getattr(arguments, "json", False):
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
