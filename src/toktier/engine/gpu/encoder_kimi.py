"""End-to-end GPU encoder for the kimi splitter band.

The band differs from the o200k band in the splitter, not in the BPE
layer: its pattern adds a leading Han alternative, so piece starts come
from the kimi piece-start kernel, and the rest of the chain (piece byte
boundaries, byte-level merge, id delivery) is the same code. The band's
parameters -- its ten-class table, its digit bound and its contraction
alternative -- are described by the routing data, so this module has no
family constants.

Only the eager form lives here. The other end-to-end bands also declare
a fused single-call form that a CUDA Graph can capture; this splitter
resolves its sparse cases in device windows that the host selects per
request, so its geometry depends on the input rather than on the buffer
capacity and there is no single call to capture. Declaring the eager
form alone is what keeps a request for a captured delivery a refusal
rather than a silent downgrade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .encoder import GpuTokenizer
from .options import GpuOptions
from .pretok_kimi import GpuPretokKimi

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .class_tables import LoadedClassTable
    from .families import KernelFamily

__all__ = ["GpuTokenizerKimi"]


class GpuTokenizerKimi(GpuTokenizer):
    """Eager end-to-end encoder for the kimi band."""

    def _build_pretok(
        self,
        ext: Any,
        table: LoadedClassTable,
        family: KernelFamily,
        options: GpuOptions,
    ) -> Any:
        return GpuPretokKimi(
            ext,
            table.array,
            digits_max=self.resolve_digits_max(family, table),
            device=options.device,
            options=options,
        )
