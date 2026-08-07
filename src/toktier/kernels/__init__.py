"""CUDA kernel sources and their content digests.

This package holds the CUDA source that the GPU backend compiles, plus
the pure-Python helpers that describe it. Importing it must never import
``torch``: the digest of the kernel source is part of the certificate
binding set (``docs/contracts/registry.md`` Section 3.1), so a machine
without a GPU still has to be able to compute and verify it.

The first release ships the kernel as source and compiles it on the
user's machine (JIT delivery). A certificate for that delivery mode has
backend status ``certified_source`` and binds the kernel source digest,
the build flags, the toolchain constraints and the generated class-table
digest -- never a binary digest, because the machine-local build product
is not bit-identical to the judged build.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = [
    "KERNEL_DIR",
    "KERNEL_SOURCES",
    "kernel_source_digest",
    "kernel_source_paths",
]

#: Directory holding the CUDA sources shipped with the package.
KERNEL_DIR = Path(__file__).resolve().parent

#: Every source file that is compiled into the extension, in the order
#: the compiler receives them. The order is part of the build
#: description and therefore part of what the certificate binds.
KERNEL_SOURCES: tuple[str, ...] = ("pretok_kernel.cu",)

#: Domain separation tag for the kernel source digest, so that the same
#: bytes hashed for another purpose cannot collide with this one.
_DIGEST_DOMAIN = b"toktier.kernel_source.v1\x00"


def kernel_source_paths() -> tuple[Path, ...]:
    """Absolute paths of the compiled sources, in compile order."""
    return tuple(KERNEL_DIR / name for name in KERNEL_SOURCES)


def kernel_source_digest() -> str:
    """Return ``sha256:<hex>`` over the kernel sources.

    The digest covers each source's name and exact bytes, in compile
    order, under a domain tag. It is stable across machines and is the
    value a ``certified_source`` record binds; a mismatch closes the
    accelerated path (``R_KERNEL_DIGEST_MISMATCH``).
    """
    digest = hashlib.sha256()
    digest.update(_DIGEST_DOMAIN)
    for name, path in zip(KERNEL_SOURCES, kernel_source_paths(), strict=True):
        data = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"
