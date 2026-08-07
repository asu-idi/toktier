"""Implementation entry points the routing data may name.

Contract reference: ``docs/contracts/registry.md`` Section 3.3. The
routing data (``kernel_families.v1.json``) says which entry point serves
which band; this module says which entry points exist. The two tables
answer different questions -- data identity versus code identity -- so
neither is a second copy of the other, and no module anywhere keys a
dispatch on a band or family name.

Importing this module is torch-free by design: the set of declared entry
points is checkable on a machine without a GPU. Resolving an entry point
imports its implementation module, which does need torch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ...errors import RegistryInvalid

__all__ = [
    "ENCODER_ENTRY_POINTS",
    "PRETOK_ENTRY_POINTS",
    "encoder_deliveries",
    "pretok_class",
]


def _encoder() -> Mapping[str, Any]:
    from .encoder import FusedGpuTokenizer, GpuTokenizer

    return {
        "eager": GpuTokenizer,
        "fused": FusedGpuTokenizer,
        "graph": FusedGpuTokenizer,
    }


def _encoder_o200k() -> Mapping[str, Any]:
    from .encoder_o200k import FusedGpuTokenizerO200k, GpuTokenizerO200k

    return {
        "eager": GpuTokenizerO200k,
        "fused": FusedGpuTokenizerO200k,
        "graph": FusedGpuTokenizerO200k,
    }


def _pretok() -> Any:
    from .pretok import CudaPretok

    return CudaPretok


def _pretok_o200k() -> Any:
    from .pretok_o200k import GpuPretokO200k

    return GpuPretokO200k


def _pretok_kimi() -> Any:
    from .pretok_kimi import GpuPretokKimi

    return GpuPretokKimi


#: End-to-end encoder entry points: id -> lazy loader returning the
#: delivery-kind to class mapping (``eager`` / ``fused`` / ``graph``).
ENCODER_ENTRY_POINTS: Mapping[str, Callable[[], Mapping[str, Any]]] = {
    "encoder": _encoder,
    "encoder_o200k": _encoder_o200k,
}

#: Piece-start (split layer) entry points: id -> lazy loader returning
#: the pretokenizer class. Every class exposes the uniform
#: ``from_family`` construction hook the engine calls.
PRETOK_ENTRY_POINTS: Mapping[str, Callable[[], Any]] = {
    "pretok": _pretok,
    "pretok_o200k": _pretok_o200k,
    "pretok_kimi": _pretok_kimi,
}


def encoder_deliveries(entry_point: str) -> Mapping[str, Any]:
    """Delivery-kind to encoder class mapping for one entry point."""
    loader = ENCODER_ENTRY_POINTS.get(entry_point)
    if loader is None:
        raise RegistryInvalid(
            f"routing data names unknown encoder entry point {entry_point!r}",
            details={
                "path": None,
                "failure": "unknown_entry_point",
                "declared": sorted(ENCODER_ENTRY_POINTS),
            },
        )
    return loader()


def pretok_class(entry_point: str) -> Any:
    """Pretokenizer class for one entry point."""
    loader = PRETOK_ENTRY_POINTS.get(entry_point)
    if loader is None:
        raise RegistryInvalid(
            f"routing data names unknown pretok entry point {entry_point!r}",
            details={
                "path": None,
                "failure": "unknown_entry_point",
                "declared": sorted(PRETOK_ENTRY_POINTS),
            },
        )
    return loader()
