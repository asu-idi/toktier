"""Tokenization backends and the protocol they share.

Contract reference: ``docs/contracts/routing.md`` Section 4 assigns the
frozen backend id namespace: ``hf`` (reference, always present, always
last in a fallback chain), ``fast_cpu`` (the evidence-bound corrected
Gigatoken build), and ``gpu`` (CUDA kernels).

This package ships the reference backend. The CUDA backend lives in
``toktier.engine.gpu`` (adapter: ``toktier.engine.gpu.backend``) and
satisfies :class:`~toktier.backends.protocol.Backend`; no placeholder is
shipped here, so no second definition of it can drift into existence.
"""

from __future__ import annotations

from .fast_cpu import (
    ENGINE_MODULE,
    ENGINE_PACKAGE,
    PINNED_ENGINE_VERSION,
    FastCpuBackend,
    FastCpuEngineFacts,
    fast_cpu_engine_facts,
)
from .hf import ORACLE_PACKAGE, REJECTED_LOADER_FLAGS, HfBackend, oracle_version
from .protocol import (
    ADDED_TOKENS_FILE,
    TOKENIZER_FILE,
    ArtifactHandle,
    ArtifactResolver,
    Backend,
    BackendFactory,
)

__all__ = [
    "ADDED_TOKENS_FILE",
    "ENGINE_MODULE",
    "ENGINE_PACKAGE",
    "ORACLE_PACKAGE",
    "PINNED_ENGINE_VERSION",
    "REJECTED_LOADER_FLAGS",
    "TOKENIZER_FILE",
    "ArtifactHandle",
    "ArtifactResolver",
    "Backend",
    "BackendFactory",
    "FastCpuBackend",
    "FastCpuEngineFacts",
    "HfBackend",
    "fast_cpu_engine_facts",
    "oracle_version",
]
