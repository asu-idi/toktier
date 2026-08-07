"""The CUDA backend: prebuilt or JIT kernels for split and BPE encoding.

Execution requires the ``gpu`` extra (prebuilt) or ``gpu-jit`` extra
(local compilation with torch and ninja). Importing this module
imports nothing heavy: the names below are re-exported lazily, so a
process without torch can still inspect the routing data, compute the
kernel source digest and verify table digests.

Module map:

``loader``
    The one JIT loader. One process, one build, one bound flag set.
``families``
    Family to band resolution, read from the registry's routing data.
``class_tables``
    Loading and digest-verifying the generated lookup tables.
``options``
    Explicit tuning options; nothing here reads the environment.
``pretok`` / ``pretok_o200k`` / ``pretok_kimi``
    Piece-start computation per band.
``encoder`` / ``encoder_o200k``
    End-to-end encoders, eager and fused forms.
``batched``
    The many-documents-per-pass channel behind ``encode_batch``.
``engine``
    The facade that ties them together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .families import KernelFamily, KernelFamilyTable
from .loader import DEFAULT_BUILD_FLAGS, BuildFlags, KernelLoader
from .options import DEFAULT_GPU_OPTIONS, GpuOptions

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import GpuEngine

__all__ = [
    "DEFAULT_BUILD_FLAGS",
    "DEFAULT_GPU_OPTIONS",
    "BuildFlags",
    "GpuEngine",
    "GpuOptions",
    "KernelFamily",
    "KernelFamilyTable",
    "KernelLoader",
]


def __getattr__(name: str) -> Any:
    """Import the torch-dependent facade only when it is asked for."""
    if name == "GpuEngine":
        from .engine import GpuEngine

        return GpuEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
