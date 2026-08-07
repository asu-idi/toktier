"""End-to-end GPU encoders for the o200k splitter band.

The band differs from the cl100k band in the splitter, not in the BPE
layer: piece starts come from the o200k piece-start kernel, and the rest
of the chain (piece byte boundaries, byte-level merge, id delivery) is
the same code.

Two families in this band share a splitter regex with the others and
differ only in their normalizer and vocabulary; two more drop the
contraction alternative and cut digits one at a time. All four are
described by the routing data, so this module has no family constants.

The fused entry for this band reports two sparse cases through flags:
a genuine contraction chain, and a span where the punctuation and mark
classes are ambiguous. Both are rare and both are resolved exactly by
redoing the request on the eager path; the fused path never tries to
correct them itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .encoder import FusedGpuTokenizer, GpuTokenizer
from .options import GpuOptions
from .pretok_o200k import GpuPretokO200k

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .class_tables import LoadedClassTable
    from .families import KernelFamily

__all__ = ["FusedGpuTokenizerO200k", "GpuTokenizerO200k"]


class GpuTokenizerO200k(GpuTokenizer):
    """Eager end-to-end encoder for the o200k band."""

    def _build_pretok(
        self,
        ext: Any,
        table: LoadedClassTable,
        family: KernelFamily,
        options: GpuOptions,
    ) -> Any:
        return GpuPretokO200k(
            ext,
            table.array,
            digits_max=self.resolve_digits_max(family, table),
            contractions=family.contractions,
            device=options.device,
            options=options,
        )


class FusedGpuTokenizerO200k(FusedGpuTokenizer, GpuTokenizerO200k):
    """Fused o200k encoder with bucketed CUDA Graph capture.

    The fused entry keeps the per-run channel on the device and reports
    ``{token count, contraction-chain flag, ambiguity flag}`` in one
    twelve-byte read back, so a request costs one host synchronisation.
    When either flag is set, the whole request is redone on the eager
    path; correcting a sparse case inside a captured graph is not
    attempted.
    """

    def _call(
        self, buffer: torch.Tensor, n_bytes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out: tuple[torch.Tensor, torch.Tensor] = self.ext.encode_fused_o200k(
            buffer,
            n_bytes,
            self.pre.table,
            self.pre.dmax,
            self.pre.contractions,
            self.pair_keys,
            self.pair_vals,
            self.byte_id,
            self.vocab_keys,
            self.vocab_vals,
            self.blob,
            self.ignore_merges,
            self.unsafe_bits,
        )
        return out

    def _graph_for(self, capacity: int) -> tuple[Any, ...]:
        if capacity not in self._graphs:
            buffer = torch.full(
                (capacity,), 0x61, dtype=torch.uint8, device=self.dev
            )  # 'a': placeholder content during capture
            n_bytes = torch.tensor(
                [capacity], dtype=torch.int32, device=self.dev
            )
            side = torch.cuda.Stream(self.dev)
            side.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(side):  # warm up before capturing
                for _ in range(3):
                    self._call(buffer, n_bytes)
            torch.cuda.current_stream(self.dev).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out, meta = self._call(buffer, n_bytes)
            self._graphs[capacity] = (graph, buffer, n_bytes, out, meta)
        return self._graphs[capacity]

    def _fused_or_none(self, text: str) -> tuple[torch.Tensor, int] | None:
        """Run the fused entry; ``None`` means a sparse flag was raised."""
        encoded = text.encode()
        n_bytes = len(encoded)
        if not self.use_graph or n_bytes > self.BUCKETS[-1]:
            buffer = torch.frombuffer(
                bytearray(encoded), dtype=torch.uint8
            ).to(self.dev)
            count_dev = torch.tensor(
                [n_bytes], dtype=torch.int32, device=self.dev
            )
            out, meta = self._call(buffer, count_dev)
        else:
            capacity = next(size for size in self.BUCKETS if size >= n_bytes)
            graph, buffer, count_dev, out, meta = self._graph_for(capacity)
            self._pinned_np[:n_bytes] = np.frombuffer(encoded, dtype=np.uint8)
            buffer[:n_bytes].copy_(self._pinned[:n_bytes], non_blocking=True)
            if n_bytes < capacity:
                buffer[n_bytes:].fill_(0x80)
            count_dev.fill_(n_bytes)
            graph.replay()
        count, chain, ambiguous = meta.cpu().tolist()  # single sync point
        if chain or ambiguous:
            return None
        return out, int(count)

    def _encode_plain(self, text: str) -> list[int]:
        # Large buffers are the eager path's territory: above the
        # crossover the copy and readback dominate, and the eager form
        # also carries the device-side normalization quick check.
        if len(text) > self.options.graph_max_bytes:
            return GpuTokenizerO200k._encode_plain(self, text)
        normalized = self.normalize(text)
        fused = self._fused_or_none(normalized)
        if fused is None:  # sparse case: redo exactly, on the eager path
            return self._encode_eager_normalized(normalized)
        out, count = fused
        result: list[int] = out[:count].cpu().tolist()
        return result

    def _encode_eager_normalized(self, text: str) -> list[int]:
        """Eager encode of text that is already normalized."""
        buffer = torch.frombuffer(
            bytearray(text.encode()), dtype=torch.uint8
        ).to(self.dev)
        result: list[int] = self.encode_dev(buffer).cpu().tolist()
        return result

    def encode_np(self, text: str) -> np.ndarray[Any, Any]:
        """Token ids as a numpy array: one device-to-host copy."""
        if not text:
            return np.empty(0, dtype=np.int32)
        plan = self._frontend_plan(text)
        if plan is not None and self._frontend is not None:
            return np.asarray(
                self._frontend.assemble(plan, self._encode_plain),
                dtype=np.int32,
            )
        if len(text) > self.options.graph_max_bytes:
            return np.asarray(
                GpuTokenizerO200k._encode_plain(self, text), dtype=np.int32
            )
        normalized = self.normalize(text)
        fused = self._fused_or_none(normalized)
        if fused is None:
            return np.asarray(
                self._encode_eager_normalized(normalized), dtype=np.int32
            )
        out, count = fused
        result: np.ndarray[Any, Any] = out[:count].cpu().numpy()
        return result

    def encode_view(self, text: str) -> torch.Tensor:
        """Device-side delivery; see the base class for the aliasing rule."""
        ids = self.encode_np(text)
        return torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int32)).to(
            self.dev
        )
