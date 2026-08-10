"""The single kernel loader: prebuilt (fatbin) first, JIT as fallback.

Contract reference: ``docs/contracts/registry.md`` Section 3.2 (single
loader, single flag set) and Section 3.1 (generated tables are bound
artifacts).

Why there is exactly one loader
-------------------------------
A kernel certificate covers exactly one kernel build configuration per
process: one loader, one bound flag set, one delivery. If two builds
of the kernel with different flags were loaded into the same process, the
certificate's premises would no longer hold. The prototype this was
ported from had four independent ``cpp_extension.load`` call sites
(production, two regression gates, one A/B harness) whose flags were not
identical; that is exactly the situation this module makes impossible.

Rules implemented here:

- One process-wide loader instance. The first successful load fixes the
  bound flag set *and the delivery* for the lifetime of the process.
- Delivery preference (``delivery="auto"``): the shipped prebuilt fatbin
  when the driver can run it (a ``certified`` certificate binds its
  binary digest), otherwise the JIT build (``certified_source``), and
  the refusal reason is recorded -- an environment that cannot serve the
  prebuilt delivery is told so, never silently downgraded. An explicit
  ``delivery="prebuilt"`` or ``"jit"`` request that cannot be served
  raises instead of substituting.
- A request for a *different* flag set (or a divergent explicit
  delivery) does not silently produce a second build: it raises
  :class:`~toktier.errors.KernelIncompatible` and marks the process
  certificate as void, so any later report says uncertified rather than
  claiming a certificate whose premises no longer hold.
- The JIT build directory comes from the resolved cache directory
  (``Config.cache_dir``; see ``docs/contracts/config.md`` Section 5), not
  from a hardcoded path and not from ``TORCH_EXTENSIONS_DIR``. Built
  kernels are cache: deleting them costs build time, never data.
- Build flags and delivery are explicit arguments. The read-only toolchain
  probe observes CUDA's own ``CUDA_HOME`` / ``CUDA_PATH`` compiler-selection
  inputs because the delegated build system already honors them; they are
  reported and cache-bound, never interpreted as TokTier configuration.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from ...errors import BackendUnavailable, KernelIncompatible
from ...kernels import kernel_source_digest, kernel_source_paths
from ...kernels.bindings import CertifiedSourceBindings, bare_sha256
from .toolchain import (
    NvccFacts,
    jit_toolchain_satisfied,
    selected_nvcc_facts,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import Config

__all__ = [
    "DEFAULT_BUILD_FLAGS",
    "DELIVERIES",
    "EXTENSION_NAME",
    "BuildFlags",
    "KernelLoader",
]

#: Delivery selectors ``KernelLoader.get`` accepts. ``auto`` prefers the
#: shipped prebuilt fatbin and falls back to the JIT build with the
#: refusal reason recorded; the explicit values refuse to substitute.
DELIVERIES = ("auto", "prebuilt", "jit")

#: Name of the compiled extension module. Part of the build description.
EXTENSION_NAME = "toktier_pretok_cuda"


@dataclass(frozen=True)
class BuildFlags:
    """The immutable, certificate-bound build flag set.

    Every field here changes the produced machine code, so every field is
    part of what a ``certified_source`` record binds. Defaults reproduce
    the configuration the judged builds used.
    """

    #: Flags handed to ``nvcc`` for the CUDA sources.
    cuda_cflags: tuple[str, ...] = ("-O3",)
    #: Flags handed to the host compiler.
    cflags: tuple[str, ...] = ()
    #: Preprocessor definitions, as ``(name, value)`` pairs. The kernel
    #: exposes ``TOKTIER_TPB``, ``TOKTIER_SHORT_MAX``, ``TOKTIER_DS_*``
    #: and ``TOKTIER_PB_CONTENT_CHECK``; the certified configuration uses
    #: their built-in defaults, so this is empty.
    defines: tuple[tuple[str, str], ...] = ()

    def as_cuda_flags(self) -> list[str]:
        """Full ``nvcc`` argument list, definitions included."""
        return [*self.cuda_cflags, *(f"-D{k}={v}" for k, v in self.defines)]

    def as_host_flags(self) -> list[str]:
        """Full host-compiler argument list, definitions included."""
        return [*self.cflags, *(f"-D{k}={v}" for k, v in self.defines)]

    def as_binding_flags(self) -> tuple[str, ...]:
        """The canonical flat encoding of the flag set, for binding sets.

        The registry schema stores ``build_flags`` as one flat array of
        strings. The encoding is: the full ``nvcc`` argument list
        (definitions included), then each host-compiler flag prefixed
        ``host:``. The certified default configuration therefore encodes
        as ``("-O3",)``, which is exactly the value the judged build
        recorded (the ``extra_cuda_cflags`` of the judgement loader).
        """
        return (
            *self.as_cuda_flags(),
            *(f"host:{flag}" for flag in self.cflags),
        )

    def digest(self) -> str:
        """``sha256:<hex>`` over the canonical rendering of the flags."""
        parts = [
            "cuda:" + "\x1f".join(self.cuda_cflags),
            "host:" + "\x1f".join(self.cflags),
            "defines:" + "\x1f".join(f"{k}={v}" for k, v in self.defines),
        ]
        payload = "\x1e".join(parts).encode("utf-8")
        return "sha256:" + hashlib.sha256(
            b"toktier.kernel_build_flags.v1\x00" + payload
        ).hexdigest()


#: The flag set the certification runs used.
DEFAULT_BUILD_FLAGS = BuildFlags()


@dataclass(frozen=True)
class ToolchainFacts:
    """What the loader observed about the local build toolchain.

    These are the values a certificate's toolchain constraints are
    checked against. They are collected, never asserted, here: the
    routing layer decides what an out-of-range value means under the
    active policy.
    """

    torch_version: str
    cuda_version: str | None
    nvcc_path: str | None
    nvcc_resolved_path: str | None
    nvcc_release: str | None
    nvcc_build: str | None
    nvcc_error: str | None
    jit_toolchain_satisfied: bool
    device_name: str | None
    device_capability: str | None
    driver_version: str | None

    def cache_tag(self) -> str:
        compiler = NvccFacts(
            path=self.nvcc_path,
            resolved_path=self.nvcc_resolved_path,
            release=self.nvcc_release,
            build=self.nvcc_build,
            error=self.nvcc_error,
        )
        return compiler.cache_tag(
            torch_cuda=self.cuda_version or "unknown",
            torch_version=self.torch_version,
        )


@dataclass
class _LoadState:
    module: ModuleType | Any | None = None
    flags: BuildFlags | None = None
    certificate_void: bool = False
    void_reason: str | None = None
    toolchain: ToolchainFacts | None = None
    build_dir: Path | None = None
    #: ``"prebuilt"`` or ``"jit"`` once a load happened.
    delivery: str | None = None
    #: Identity facts of the loaded prebuilt delivery, if any.
    prebuilt: Any | None = None
    #: Why ``auto`` fell back to JIT, if it did (stated, not silent).
    prebuilt_fallback_reason: str | None = None
    #: The prebuilt image is owned by the Rust CUDA Driver host rather than
    #: by the legacy torch extension.  This is still the one process-wide
    #: certified delivery and must participate in all loader reporting and
    #: divergent-delivery guards.
    native_prebuilt: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _NativePrebuiltFacts:
    """Shape-compatible identity facts for the Rust-owned fatbin."""

    manifest: dict[str, Any]
    fatbin_digest: str
    device_architecture: str
    architecture_embedded: bool = True


class KernelLoader:
    """Process-wide singleton that owns the one kernel build.

    Use :meth:`get` to obtain it. The first call decides the build
    directory and the flag set; later calls must agree.
    """

    _lock = threading.Lock()
    _state = _LoadState()

    def __init__(self) -> None:  # pragma: no cover - not the entry point
        raise TypeError("use KernelLoader.get() instead of constructing")

    # -- introspection (safe without a GPU) ---------------------------

    @classmethod
    def is_loaded(cls) -> bool:
        """Whether a kernel build has been loaded in this process."""
        return cls._state.module is not None or cls._state.native_prebuilt

    @classmethod
    def bound_flags(cls) -> BuildFlags | None:
        """The flag set this process is bound to, if a build happened."""
        return cls._state.flags

    @classmethod
    def certificate_void(cls) -> bool:
        """Whether the process certificate has been invalidated.

        Becomes ``True`` after a second, divergent flag set is requested.
        Once void it stays void: a process cannot recover a certificate
        whose single-build premise was broken.
        """
        return cls._state.certificate_void

    @classmethod
    def void_reason(cls) -> str | None:
        """Human-readable reason the certificate was invalidated."""
        return cls._state.void_reason

    @classmethod
    def certified_source_bindings(
        cls, *, class_table_digest: str | None = None
    ) -> CertifiedSourceBindings:
        """The bound values this process would present for verification.

        Producer side of the one shared binding representation: the same
        field names and encodings the registry schema uses, so a
        registry entry and this value are comparable field by field.
        Computable without a GPU -- the source digest and the flag
        encoding need no build.
        """
        flags = cls._state.flags or DEFAULT_BUILD_FLAGS
        return CertifiedSourceBindings(
            source_digest=bare_sha256(kernel_source_digest()),
            build_flags=flags.as_binding_flags(),
            class_table_digest=(
                bare_sha256(class_table_digest) if class_table_digest else None
            ),
        )

    @classmethod
    def binding_set(
        cls,
        *,
        class_table_digest: str | None = None,
        family_table_digest: str | None = None,
    ) -> dict[str, Any]:
        """The values a ``certified_source`` record binds, as a report.

        Returned even before a build happens, because the source digest
        and the flag digest are computable without a GPU; the toolchain
        facts appear once a build has been observed. The bound fields
        are spelled exactly as the registry schema spells them
        (``source_digest``, flat ``build_flags``, bare-hex digests);
        ``family_table_digest`` additionally pins the content of the
        family routing data, so a drifted routing table is visible in
        the report rather than hidden behind an unchanged path.
        """
        flags = cls._state.flags or DEFAULT_BUILD_FLAGS
        toolchain = cls._state.toolchain
        bindings = cls.certified_source_bindings(
            class_table_digest=class_table_digest
        )
        binding: dict[str, Any] = {
            "delivery": cls._state.delivery or "jit",
            "extension_name": EXTENSION_NAME,
            **bindings.as_mapping(),
            "build_flags_digest": flags.digest(),
            "class_table_digest": bindings.class_table_digest,
            "family_table_digest": (
                bare_sha256(family_table_digest) if family_table_digest else None
            ),
            "certificate_void": cls._state.certificate_void,
        }
        from .native import native_host_build_facts

        host = native_host_build_facts()
        binding["host_source_digest"] = host.source_digest
        binding["host_build_flags"] = list(host.build_flags)
        binding["host_toolchain"] = host.toolchain
        prebuilt = cls._state.prebuilt
        if prebuilt is not None:
            manifest = prebuilt.manifest
            binding["binary_digest"] = bare_sha256(prebuilt.fatbin_digest)
            binding["prebuilt"] = {
                "fatbin_digest": prebuilt.fatbin_digest,
                "toolchain": manifest.get("toolchain"),
                "device_architecture": prebuilt.device_architecture,
                "architecture_embedded": prebuilt.architecture_embedded,
                "architectures": {
                    arch: entry.get("digest")
                    for arch, entry in manifest.get(
                        "architectures", {}
                    ).items()
                },
                "prebuilt_source_digest": manifest.get("sources", {}).get(
                    "prebuilt_source_digest"
                ),
            }
        if cls._state.prebuilt_fallback_reason is not None:
            binding["prebuilt_fallback_reason"] = (
                cls._state.prebuilt_fallback_reason
            )
        if toolchain is not None:
            binding["toolchain_facts"] = {
                "torch_version": toolchain.torch_version,
                "cuda_version": toolchain.cuda_version,
                "nvcc_path": toolchain.nvcc_path,
                "nvcc_resolved_path": toolchain.nvcc_resolved_path,
                "nvcc_release": toolchain.nvcc_release,
                "nvcc_build": toolchain.nvcc_build,
                "nvcc_error": toolchain.nvcc_error,
                "jit_toolchain_satisfied": (
                    toolchain.jit_toolchain_satisfied
                ),
                "device_name": toolchain.device_name,
                "device_capability": toolchain.device_capability,
                "driver_version": toolchain.driver_version,
            }
        return binding

    @classmethod
    def delivery(cls) -> str | None:
        """The delivery this process loaded, or ``None`` before a load."""
        return cls._state.delivery

    @classmethod
    def prebuilt_fallback_reason(cls) -> str | None:
        """Why ``auto`` fell back to JIT, if it did."""
        return cls._state.prebuilt_fallback_reason

    @classmethod
    def jit_toolchain_satisfied(cls) -> bool | None:
        """Judgement of the compiler/runtime triple, once observed."""
        facts = cls._state.toolchain
        return facts.jit_toolchain_satisfied if facts is not None else None

    # -- loading ------------------------------------------------------

    @classmethod
    def get(
        cls,
        *,
        cache_dir: Path | None = None,
        config: Config | None = None,
        flags: BuildFlags = DEFAULT_BUILD_FLAGS,
        device: str | None = None,
        delivery: str = "auto",
    ) -> ModuleType:
        """Return the kernel extension surface, loading it once.

        Args:
            cache_dir: Where JIT build products live. Defaults to the
                resolved ``Config.cache_dir``; built kernels are cache.
            config: A resolved configuration to take ``cache_dir`` from,
                when ``cache_dir`` is not given directly.
            flags: The build flag set. Must be identical for every call
                in a process. The prebuilt delivery serves exactly the
                shipped configuration (``DEFAULT_BUILD_FLAGS``); a
                custom flag set needs the JIT delivery.
            device: Optional device string; the prebuilt module loads
                for it first, and the JIT path records which
                architecture the build was observed against.
            delivery: ``"auto"`` (prebuilt when servable, JIT otherwise,
                fallback reason recorded), ``"prebuilt"`` or ``"jit"``.

        Raises:
            BackendUnavailable: ``torch`` (the ``gpu``/``gpu-jit``
                extras) is not importable.
            KernelIncompatible: a divergent flag set or delivery was
                requested, an explicit delivery cannot be served, or the
                JIT build failed.
        """
        if delivery not in DELIVERIES:
            raise KernelIncompatible(
                f"unknown kernel delivery {delivery!r}",
                details={
                    "backend": "gpu",
                    "reason_code": "R_KERNEL_BUILD_FAILED",
                    "expected_digest": None,
                    "remedy": f"use one of {DELIVERIES}",
                },
            )
        with cls._lock:
            state = cls._state
            if state.native_prebuilt:
                if flags != state.flags or delivery == "jit":
                    state.certificate_void = True
                    state.void_reason = (
                        "the Rust prebuilt host already owns the process-wide "
                        "kernel delivery and a divergent legacy load was requested"
                    )
                    raise KernelIncompatible(
                        "the native prebuilt kernel is already loaded with a "
                        "different build request",
                        details={
                            "backend": "gpu",
                            "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                            "expected_digest": (
                                state.flags or DEFAULT_BUILD_FLAGS
                            ).digest(),
                            "observed_digest": flags.digest(),
                            "remedy": "use one GPU delivery per process",
                        },
                    )
                raise KernelIncompatible(
                    "the prebuilt kernel is already owned by TokTier's native "
                    "CUDA Driver host; the legacy extension surface is unavailable",
                    details={
                        "backend": "gpu",
                        "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                        "expected_digest": None,
                        "remedy": "reuse the existing Tokenizer instance",
                    },
                )
            if state.module is not None:
                if state.flags != flags:
                    state.certificate_void = True
                    state.void_reason = (
                        "a second kernel build with a different flag set was "
                        "requested in this process; a kernel certificate "
                        "covers exactly one build configuration"
                    )
                    raise KernelIncompatible(
                        "kernel already built with a different flag set",
                        details={
                            "backend": "gpu",
                            "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                            "expected_digest": (state.flags or flags).digest(),
                            "observed_digest": flags.digest(),
                            "remedy": (
                                "use one BuildFlags value per process; "
                                "restart the process to change it"
                            ),
                        },
                    )
                if delivery != "auto" and delivery != state.delivery:
                    state.certificate_void = True
                    state.void_reason = (
                        "a second kernel load with a different delivery was "
                        "requested in this process; a kernel certificate "
                        "covers exactly one build configuration"
                    )
                    raise KernelIncompatible(
                        "kernel already loaded with a different delivery "
                        f"({state.delivery!r}; {delivery!r} was requested)",
                        details={
                            "backend": "gpu",
                            "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                            "expected_digest": None,
                            "remedy": (
                                "use one delivery per process; restart "
                                "the process to change it"
                            ),
                        },
                    )
                return state.module

            torch = _import_torch()
            if delivery in ("auto", "prebuilt"):
                loaded = cls._try_prebuilt(state, torch, flags, device, delivery)
                if loaded is not None:
                    result: ModuleType = loaded
                    return result
            toolchain = _toolchain_facts(torch, device)
            build_dir = _resolve_build_dir(
                cache_dir, config, flags, toolchain=toolchain
            )
            build_dir.mkdir(parents=True, exist_ok=True)
            module = _compile(torch, build_dir, flags)
            state.module = module
            state.flags = flags
            state.build_dir = build_dir
            state.delivery = "jit"
            state.toolchain = toolchain
            return module

    @classmethod
    def note_native_prebuilt_loaded(
        cls,
        *,
        manifest: dict[str, Any],
        fatbin_digest: str,
        architecture: str,
    ) -> None:
        """Publish a successfully loaded Rust-owned prebuilt delivery.

        The native constructor has already verified the manifest digest,
        selected an embedded image for ``architecture`` and loaded the CUDA
        module.  Recording those immutable facts here keeps ``explain()``,
        ``doctor`` and the legacy loader's single-delivery guard honest
        without importing torch or exposing the extension ABI to Rust.
        """
        facts = _NativePrebuiltFacts(
            manifest=dict(manifest),
            fatbin_digest=fatbin_digest,
            device_architecture=architecture,
        )
        with cls._lock:
            state = cls._state
            if state.module is not None:
                state.certificate_void = True
                state.void_reason = (
                    "a legacy kernel extension was already loaded before the "
                    "Rust prebuilt host requested process ownership"
                )
                raise KernelIncompatible(
                    "a kernel delivery is already loaded through the legacy host",
                    details={
                        "backend": "gpu",
                        "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                        "expected_digest": None,
                        "remedy": "use one GPU host implementation per process",
                    },
                )
            if state.native_prebuilt:
                previous = state.prebuilt
                if (
                    previous is None
                    or previous.fatbin_digest != fatbin_digest
                    or previous.device_architecture != architecture
                ):
                    state.certificate_void = True
                    state.void_reason = (
                        "a second native prebuilt identity was requested in one process"
                    )
                    raise KernelIncompatible(
                        "native prebuilt identity differs from the loaded image",
                        details={
                            "backend": "gpu",
                            "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                            "expected_digest": getattr(
                                previous, "fatbin_digest", None
                            ),
                            "observed_digest": fatbin_digest,
                            "remedy": "restart the process to change GPU images",
                        },
                    )
                return
            state.flags = DEFAULT_BUILD_FLAGS
            state.delivery = "prebuilt"
            state.prebuilt = facts
            state.native_prebuilt = True

    @classmethod
    def _try_prebuilt(
        cls,
        state: _LoadState,
        torch: Any,
        flags: BuildFlags,
        device: str | None,
        delivery: str,
    ) -> Any | None:
        """Attempt the prebuilt delivery; ``None`` = fall back to JIT.

        An explicit ``delivery="prebuilt"`` request never falls back:
        the refusal is raised with its reason. Under ``auto`` the reason
        is recorded on the state so reports can say why the process is
        on the JIT delivery.
        """
        from .prebuilt import PrebuiltUnavailable, load_prebuilt_extension

        def refuse(reason_code: str, message: str) -> None:
            if delivery == "prebuilt":
                raise KernelIncompatible(
                    f"the prebuilt kernel delivery cannot serve this "
                    f"process: {message}",
                    details={
                        "backend": "gpu",
                        "reason_code": reason_code,
                        "expected_digest": None,
                        "remedy": (
                            "use delivery='jit' (or 'auto') to build "
                            "locally, or fix the stated condition"
                        ),
                    },
                )
            state.prebuilt_fallback_reason = f"{reason_code}: {message}"

        if flags != DEFAULT_BUILD_FLAGS:
            refuse(
                "R_PREBUILT_FLAGS_UNSERVABLE",
                "the prebuilt fatbin serves exactly the shipped build "
                "configuration; a custom BuildFlags value needs the JIT "
                "delivery",
            )
            return None
        try:
            load = load_prebuilt_extension(device)
        except PrebuiltUnavailable as exc:
            refuse(exc.reason_code, str(exc))
            return None
        state.module = load.extension
        state.flags = flags
        state.delivery = "prebuilt"
        state.prebuilt = load
        state.toolchain = _toolchain_facts(
            torch, device, inspect_compiler=False
        )
        return load.extension

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Drop all loader state. Test helper; never used in production."""
        with cls._lock:
            cls._state = _LoadState()


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise BackendUnavailable(
            "the GPU backend needs the 'gpu' extra (prebuilt kernels; "
            "torch) or the 'gpu-jit' extra (local build; torch and ninja)",
            details={"backend": "gpu", "missing": "torch"},
        ) from exc
    return torch


def _resolve_build_dir(
    cache_dir: Path | None,
    config: Config | None,
    flags: BuildFlags,
    *,
    toolchain: ToolchainFacts | None = None,
) -> Path:
    """Build products go under the resolved cache directory.

    The subdirectory name carries the flag digest, so a rebuild with
    different flags cannot silently reuse another configuration's
    artefacts on disk.
    """
    if cache_dir is None:
        if config is None:
            from ...config import Config

            config = Config.resolve()
        cache_dir = Path(config.cache_dir)
    flags_tag = flags.digest().split(":", 1)[1][:16]
    toolchain_tag = toolchain.cache_tag() if toolchain is not None else "unprobed"
    return (
        Path(cache_dir)
        / "kernels"
        / f"{EXTENSION_NAME}-{flags_tag}-tc-{toolchain_tag}"
    )


def _compile(torch: Any, build_dir: Path, flags: BuildFlags) -> ModuleType:
    from torch.utils.cpp_extension import load

    sources = [str(path) for path in kernel_source_paths()]
    try:
        module = load(
            name=EXTENSION_NAME,
            sources=sources,
            extra_cflags=flags.as_host_flags(),
            extra_cuda_cflags=flags.as_cuda_flags(),
            build_directory=str(build_dir),
            verbose=False,
        )
    except Exception as exc:
        raise KernelIncompatible(
            f"kernel build failed: {exc}",
            details={
                "backend": "gpu",
                "reason_code": "R_KERNEL_BUILD_FAILED",
                "build_directory": str(build_dir),
                "expected_digest": kernel_source_digest(),
            },
        ) from exc
    loaded: ModuleType = module
    return loaded


def _toolchain_facts(
    torch: Any, device: str | None, *, inspect_compiler: bool = True
) -> ToolchainFacts:
    cuda_version = getattr(torch.version, "cuda", None)
    compiler = (
        selected_nvcc_facts()
        if inspect_compiler
        else NvccFacts(
            path=None,
            resolved_path=None,
            release=None,
            build=None,
            error="not inspected for prebuilt delivery",
        )
    )
    torch_version = str(torch.__version__)
    cuda_text = str(cuda_version) if cuda_version else None
    device_name: str | None = None
    capability: str | None = None
    driver: str | None = None
    try:
        if torch.cuda.is_available():
            index = torch.device(device).index if device else None
            device_name = torch.cuda.get_device_name(index)
            major, minor = torch.cuda.get_device_capability(index)
            capability = f"sm_{major}{minor}"
            raw_driver = getattr(torch.cuda, "driver_version", None)
            driver = str(raw_driver() if callable(raw_driver) else raw_driver)
    except RuntimeError:
        # The runtime refused to answer (driver/device initialization).
        # These are optional observations, so they stay unknown; any
        # other exception type is a defect and propagates.
        pass
    return ToolchainFacts(
        torch_version=torch_version,
        cuda_version=cuda_text,
        nvcc_path=compiler.path,
        nvcc_resolved_path=compiler.resolved_path,
        nvcc_release=compiler.release,
        nvcc_build=compiler.build,
        nvcc_error=compiler.error,
        jit_toolchain_satisfied=jit_toolchain_satisfied(
            torch_cuda=cuda_text or "unknown",
            torch_version=torch_version,
            nvcc=compiler,
            ninja_present=find_spec("ninja") is not None,
        ),
        device_name=device_name,
        device_capability=capability,
        driver_version=driver if driver and driver != "None" else None,
    )
