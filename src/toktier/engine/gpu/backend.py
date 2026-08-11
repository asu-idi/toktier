"""GPU backend adapter: the executor-facing surface of the GPU engine.

Contract reference: ``docs/contracts/routing.md`` Section 4 -- the
``gpu`` backend satisfies ``toktier.backends.protocol.Backend``. The
encoders in this package are delivery objects (single-request forms and
a batched channel) with their own call shapes; this adapter owns one of
each and presents the one protocol the routing executor runs against.

Two behaviors are the adapter's own:

- ``add_special_tokens=True`` is honored natively only when the
  artifact's post-processor inserts nothing, in which case the flag
  cannot change the ids. When the post-processor does insert tokens,
  the certified GPU chain produces the core stream only, so the call
  raises :class:`~toktier.errors.BackendExecutionFault` and the routing
  layer re-runs the input on the reference backend, counted as
  ``R_INPUT_POSTPROCESS_ROUTED``.
- Expected device and runtime failures (``RuntimeError`` from the
  accelerator stack, which is how CUDA errors surface) are wrapped in
  :class:`~toktier.errors.BackendExecutionFault`. Every other exception
  propagates unchanged: an interface regression is a defect to surface,
  not a route.

This module imports no accelerator runtime; the encoder and the batched
channel are constructed elsewhere and injected.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from ...errors import BackendExecutionFault
from ...policy import BACKEND_GPU

__all__ = ["GpuBackend", "LazyGpuBackend"]


class SingleEncoder(Protocol):
    """The single-request encoder surface the adapter consumes."""

    #: Whether the artifact's post-processor inserts special tokens.
    adds_special_tokens: bool

    def encode(self, text: str) -> list[int]:
        """Core-stream token ids for one document."""


class BatchChannel(Protocol):
    """The batched channel surface the adapter consumes."""

    def encode_batch(self, docs: list[str]) -> list[Any]:
        """Per-document id rows for one batch."""


class GpuBackend:
    """Backend-protocol adapter over one family's GPU encoders.

    Args:
        encoder: a single-request end-to-end encoder for one family.
        batched: the family's batched channel; when ``None``, batches
            are encoded input by input through ``encoder``.
    """

    def __init__(
        self,
        encoder: SingleEncoder,
        *,
        batched: BatchChannel | None = None,
    ) -> None:
        self._encoder: SingleEncoder | None = encoder
        self._batched: BatchChannel | None = batched
        self._adds_special_tokens = bool(encoder.adds_special_tokens)

    @property
    def backend_id(self) -> str:
        """Frozen backend identifier of the CUDA kernel path."""
        return BACKEND_GPU

    def _live(self) -> SingleEncoder:
        encoder = self._encoder
        if encoder is None:
            raise RuntimeError("backend is closed")
        return encoder

    def _require_core_stream(self, add_special_tokens: bool) -> None:
        if add_special_tokens and self._adds_special_tokens:
            raise BackendExecutionFault(
                "the artifact's post-processor inserts special tokens; the "
                "GPU chain produces the core stream, so this input runs on "
                "the reference backend",
                details={
                    "backend": BACKEND_GPU,
                    "stage": "add_special_tokens",
                },
            )

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one document to token ids."""
        encoder = self._live()
        self._require_core_stream(add_special_tokens)
        try:
            return encoder.encode(text)
        except RuntimeError as exc:
            raise BackendExecutionFault(
                f"GPU encode failed: {exc}",
                details={"backend": BACKEND_GPU, "stage": "encode"},
            ) from exc

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode a batch; row ``i`` equals ``encode(texts[i])``."""
        encoder = self._live()
        self._require_core_stream(add_special_tokens)
        if not texts:
            return []
        try:
            if self._batched is not None:
                rows = self._batched.encode_batch(list(texts))
                return [[int(value) for value in row] for row in rows]
            return [encoder.encode(text) for text in texts]
        except RuntimeError as exc:
            raise BackendExecutionFault(
                f"GPU batch encode failed: {exc}",
                details={"backend": BACKEND_GPU, "stage": "encode_batch"},
            ) from exc

    def close(self) -> None:
        """Release the encoder references. Idempotent.

        Device buffers, graph captures and pinned staging are owned by
        the encoder objects; dropping the references releases them once
        nothing else holds them.
        """
        self._encoder = None
        self._batched = None


class LazyGpuBackend:
    """Load one :class:`GpuBackend` only when an input reaches the GPU.

    The automatic facade can therefore construct a certified GPU plan
    without loading CUDA, allocating device buffers, or compiling a JIT
    kernel for the short requests that deliberately start on CPU. A
    domain/runtime load failure is converted to the same recoverable
    fault as an encode failure, so the executor continues down its frozen
    fallback chain and records ``R_EXEC_FAULT``.
    """

    def __init__(self, factory: Callable[[], GpuBackend]) -> None:
        self._factory = factory
        self._backend: GpuBackend | None = None
        self._load_error: BackendExecutionFault | None = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def backend_id(self) -> str:
        """Frozen backend identifier of the CUDA kernel path."""
        return BACKEND_GPU

    @property
    def loaded(self) -> bool:
        """Whether this facade instance has materialized its GPU backend."""
        return self._backend is not None

    @property
    def load_error(self) -> BackendExecutionFault | None:
        """Cached recoverable load failure, if the first load failed."""
        return self._load_error

    def _live(self) -> GpuBackend:
        if self._closed:
            raise RuntimeError("backend is closed")
        if self._backend is not None:
            return self._backend
        if self._load_error is not None:
            raise self._load_error
        with self._lock:
            # Another thread may have completed construction while this
            # thread waited for the lock. Mypy deliberately does not model
            # that interleaving.
            if self._backend is not None:
                return self._backend  # type: ignore[unreachable]
            if self._load_error is not None:
                raise self._load_error
            try:
                backend = self._factory()
            # The factory is entirely internal (artifact/table loaders,
            # torch/CUDA and our engine constructor), so every ordinary
            # exception here is an accelerated execution fault. BaseException
            # subclasses such as KeyboardInterrupt still propagate.
            except Exception as exc:
                fault = BackendExecutionFault(
                    f"GPU backend load failed: {exc}",
                    details={
                        "backend": BACKEND_GPU,
                        "stage": "load",
                        "cause": getattr(exc, "code", type(exc).__name__),
                    },
                )
                self._load_error = fault
                raise fault from exc
            self._backend = backend
            return backend

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Load on demand, then encode one document."""
        return self._live().encode(text, add_special_tokens=add_special_tokens)

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Load on demand, then encode one batch."""
        if not texts:
            return []
        return self._live().encode_batch(texts, add_special_tokens=add_special_tokens)

    def close(self) -> None:
        """Release a materialized backend; idempotent."""
        with self._lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
            self._closed = True
