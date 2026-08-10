"""Concrete device probe for CUDA hosts: reports facts, builds nothing.

Contract reference: ``docs/contracts/routing.md`` Section 2. This is the
production implementation of ``toktier.routing.probe.DeviceProbe``:
device inventory and driver version come from the accelerator runtime
when it is importable, and kernel-cache facts come from the process-wide
loader state plus digests of shipped sources and routing data. Nothing
here compiles a kernel, downloads anything, or mutates state.

Importing this module does not import torch; the runtime is looked up
per call. When torch is absent or reports no devices, the probe says so,
which the planner treats as fail-closed.
"""

from __future__ import annotations

import re
import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...routing.probe import DeviceInfo, KernelCacheState
from .class_tables import ClassTableStore
from .families import KernelFamilyTable
from .loader import KernelLoader
from .toolchain import (
    NvccFacts,
    jit_toolchain_observation,
    jit_toolchain_satisfied,
    selected_nvcc_facts,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import Config

__all__ = ["CudaHostProbe"]

_DRIVER_PATTERN = re.compile(r"\b(\d{3}\.\d+(?:\.\d+)?)\b")

# A module-level alias keeps the read-only compiler probe replaceable in unit
# tests without patching subprocess globally.
_nvcc_facts = selected_nvcc_facts


def _torch_runtime() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _system_driver_version() -> str | None:
    """NVIDIA display-driver version without initializing CUDA."""
    version_file = Path("/proc/driver/nvidia/version")
    try:
        match = _DRIVER_PATTERN.search(version_file.read_text(encoding="utf-8"))
    except OSError:
        match = None
    if match is not None:
        return match.group(1)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.splitlines()[0].strip() if result.stdout else ""
    return first or None


def _jit_toolchain(torch: Any | None) -> tuple[str | None, bool]:
    """Observed compiler/runtime triple and whether it was judged."""
    if torch is None:
        return None, False
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    torch_version = str(getattr(torch, "__version__", ""))
    cuda_version = str(cuda) if cuda is not None else "unknown"
    compiler: NvccFacts = _nvcc_facts()
    observed = jit_toolchain_observation(
        torch_cuda=cuda_version,
        torch_version=torch_version,
        nvcc=compiler,
    )
    ninja_present = find_spec("ninja") is not None
    return observed, jit_toolchain_satisfied(
        torch_cuda=cuda_version,
        torch_version=torch_version,
        nvcc=compiler,
        ninja_present=ninja_present,
    )


class CudaHostProbe:
    """Device probe over the installed torch runtime and the kernel loader.

    Args:
        config: resolved configuration; its cache directory is searched
            for generated tables when reporting the class-table digest.
        class_table_dir: extra directory searched for generated tables,
            ahead of the packaged ones (mirrors ``GpuEngine.create``).
    """

    def __init__(
        self,
        *,
        config: Config | None = None,
        class_table_dir: Path | None = None,
        delivery: str = "auto",
    ) -> None:
        if delivery not in ("auto", "prebuilt", "jit"):
            raise ValueError(
                f"delivery must be 'auto', 'prebuilt', or 'jit', not {delivery!r}"
            )
        self._config = config
        self._class_table_dir = class_table_dir
        self._delivery = delivery

    def devices(self) -> tuple[DeviceInfo, ...]:
        """Usable CUDA devices, in the registry's ``sm_<cc>`` spelling."""
        torch = _torch_runtime()
        if torch is None or not torch.cuda.is_available():
            return ()
        found: list[DeviceInfo] = []
        for index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(index)
            found.append(
                DeviceInfo(
                    index=index,
                    name=torch.cuda.get_device_name(index),
                    architecture=f"sm_{major}{minor}",
                )
            )
        return tuple(found)

    def driver_version(self) -> str | None:
        """Installed driver version, or ``None`` when unknown."""
        torch = _torch_runtime()
        if torch is None or not torch.cuda.is_available():
            return None
        system = _system_driver_version()
        if system is not None:
            return system
        raw = getattr(torch.cuda, "driver_version", None)
        value = raw() if callable(raw) else raw
        text = str(value) if value is not None else None
        return text if text and text != "None" else None

    def kernel_cache(self) -> KernelCacheState:
        """Kernel facts from the loader state and shipped inputs.

        The source digest and the class-table digest are hashes of
        shipped or already-materialized files, so producing them is
        read-only. ``built`` reflects whether this process has already
        loaded the one kernel build; no build is triggered here.
        """
        families = KernelFamilyTable.load()
        cache_dir = Path(self._config.cache_dir) if self._config is not None else None
        store = ClassTableStore(
            families, table_dir=self._class_table_dir, cache_dir=cache_dir
        )
        loaded = KernelLoader.is_loaded()
        # The shipped fatbin digest is a hash of package data: read-only,
        # GPU-free, and exactly the value a ``certified`` (prebuilt)
        # record binds. Absent fatbin -> None -> a certified check can
        # only fail, never silently pass.
        from ...kernels.bindings import bare_sha256
        from .native import native_host_build_facts
        from .prebuilt import shipped_fatbin_digest

        fatbin_digest = shipped_fatbin_digest()
        host = native_host_build_facts()
        active_delivery = KernelLoader.delivery()
        preferred_delivery = self._delivery if self._delivery != "auto" else None
        judged_delivery = active_delivery or preferred_delivery
        torch = _torch_runtime()
        if judged_delivery == "jit":
            toolchain, toolchain_satisfied = _jit_toolchain(torch)
        else:
            toolchain, toolchain_satisfied = None, False
        return KernelCacheState.from_bindings(
            KernelLoader.certified_source_bindings(
                class_table_digest=store.binding_digest()
            ),
            built=loaded,
            loaded_flag_sets=1 if loaded else 0,
            binary_digest=(bare_sha256(fatbin_digest) if fatbin_digest else None),
            prebuilt_available=fatbin_digest is not None,
            delivery=active_delivery,
            preferred_delivery=preferred_delivery,
            host_source_digest=host.source_digest,
            host_build_flags=host.build_flags,
            host_toolchain=host.toolchain,
            toolchain=toolchain,
            toolchain_satisfied=(
                toolchain_satisfied if judged_delivery == "jit" else None
            ),
        )
