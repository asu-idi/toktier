"""Explicit tuning options for the GPU backend.

Contract reference: ``docs/contracts/config.md`` Section 4 -- the
long-term environment set has five members, and **no switch that can
change output correctness may exist in environment-variable or
configuration-file form**.

The prototype this was ported from carried these settings as prefixed environment
variables read at arbitrary points in the call graph. In the released
shape they are fields of an immutable options object, resolved once and
captured by the encoder that uses them. Nothing in this package reads
``os.environ``; a test asserts that.

What happened to each prototype flag:

===============================  ==================================
Prototype environment variable   Released form
===============================  ==================================
``*_CACHE_DIR``                  ``Config.cache_dir`` (``TOKTIER_HOME``)
``*_MANIFEST_EXTRA``             explicit manifest argument
``*_ADDED_FRONT``                :attr:`GpuOptions.added_token_frontend`
``*_MEMO``                       :attr:`GpuOptions.piece_memoization`
``*_NVCC_EXTRA``                 ``BuildFlags`` (bound by the certificate)
``*_O200K_CUDA``                 :attr:`GpuOptions.o200k_cuda_starts`
``*_O200K_HOST_WIN``             :attr:`GpuOptions.o200k_host_windows`
``*_O200K_BATCH_CUDA_MIN``       :attr:`GpuOptions.o200k_batch_cuda_min`
``*_O200K_GRAPH_MAX``            :attr:`GpuOptions.graph_max_bytes`
``*_KIMI_CUDA``                  :attr:`GpuOptions.kimi_cuda_starts`
``*_BPE_MONO_GUARD``             removed; the guard is unconditional
``*_PAR_MERGE_NONEXACT``         removed from the kernel entirely
``*_CONTENT_CHECK``              kernel build-time option, default off
``*_L2_PIN``                     removed (measured to give nothing)
``*_GPU_DIGEST``                 removed (judgement harness only)
``*_HOST_AMORTIZE``              removed (judgement harness only)
===============================  ==================================

The two removals in the middle of that table are the important ones. The
non-monotone merge-table guard is what makes batched merging exact for
merge tables that are not rank-monotone; an option to switch it off is an
option to produce wrong ids, so there is none. The non-exact parallel
plateau merge is not lossless at all and is gone from the kernel source.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

__all__ = ["DEFAULT_GPU_OPTIONS", "GpuOptions"]


@dataclass(frozen=True)
class GpuOptions:
    """Immutable GPU backend tuning options.

    Every field here is a performance or delivery choice. None of them
    changes which token ids come out: the certified path produces ids
    equal to the reference oracle under every combination of these
    values, and the ones that could have changed output were removed
    rather than exposed.
    """

    #: CUDA device the encoder runs on.
    device: str = "cuda:0"

    #: Route documents that contain added-token literals through the
    #: added-token frontend instead of the reference backend. Off by
    #: default in the first release: with it off, such documents are
    #: reported as ``R_INPUT_ADDED_TOKEN`` and run on the reference path,
    #: which is the behaviour the published judgement numbers describe.
    added_token_frontend: bool = False

    #: Capture the fused entry into a CUDA Graph per size bucket.
    use_cuda_graph: bool = True

    #: Inputs larger than this many bytes take the eager path instead of
    #: a captured graph. Measured crossover on the reference devices: the
    #: graph wins up to a quarter of a megabyte, above which the eager
    #: path wins because the copy and readback dominate.
    graph_max_bytes: int = 1 << 18

    #: Enable the per-piece memoization table on the eager path. Costs
    #: device memory; helps only on highly repetitive input.
    piece_memoization: bool = False

    #: Use the CUDA piece-start kernel for the o200k band. The pure
    #: tensor path is kept as an explicit fallback for differential
    #: comparison; it is slower and produces the same starts.
    o200k_cuda_starts: bool = True

    #: Run the o200k window phase on the host rather than on the device.
    #: Differential-comparison path only.
    o200k_host_windows: bool = False

    #: Below this many codepoints a batched o200k call stays on the host
    #: path, where the launch overhead would dominate.
    o200k_batch_cuda_min: int = 32768

    #: Use the CUDA piece-start kernel for the kimi band.
    kimi_cuda_starts: bool = True

    #: Upper bound on documents per batch in the batched encoder.
    max_batch_docs: int = 512

    #: Upper bound on characters per batch in the batched encoder. A
    #: single document longer than this becomes a batch of one.
    max_batch_chars: int = 4_000_000

    def replace(self, **changes: Any) -> GpuOptions:
        """Return a new options object with the given fields replaced."""
        return dataclasses.replace(self, **changes)

    def __post_init__(self) -> None:
        if self.graph_max_bytes <= 0:
            raise ValueError("graph_max_bytes must be positive")
        if self.max_batch_docs <= 0:
            raise ValueError("max_batch_docs must be positive")
        if self.max_batch_chars <= 0:
            raise ValueError("max_batch_chars must be positive")
        if self.o200k_batch_cuda_min < 0:
            raise ValueError("o200k_batch_cuda_min must not be negative")


#: The options the certification runs used.
DEFAULT_GPU_OPTIONS = GpuOptions()
