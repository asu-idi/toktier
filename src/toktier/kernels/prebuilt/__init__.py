"""Prebuilt kernel delivery: the shipped fatbin and its identities.

This package holds the multi-architecture fatbin the prebuilt delivery
mode loads through the CUDA driver API, plus the build manifest that
pins its identity. Importing it must never import ``torch``: like the
JIT source digest (``toktier.kernels``), the binary digests here are
part of a certificate binding set and must be computable and verifiable
on a machine without a GPU.

Identity model (three digest domains, all new -- the JIT source domain
``toktier.kernel_source.v1`` is untouched and keeps covering exactly the
JIT-compiled source set):

- ``toktier.kernel_prebuilt_source.v1`` -- the compile-order source
  lineage of the prebuilt unit (the wrapper plus the included pristine
  kernel source).
- ``toktier.kernel_fatbin.v1`` -- the shipped fatbin container bytes.
  This is the binary digest a ``certified`` record binds.
- ``toktier.kernel_cubin.v1`` -- one digest per embedded architecture
  image (keyed by the architecture spelling), so per-architecture
  status (certified / experimental) can bind the exact image that
  serves that architecture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = [
    "FATBIN_NAME",
    "MANIFEST_NAME",
    "PREBUILT_DIR",
    "PREBUILT_UNIT_SOURCES",
    "cubin_digest",
    "fatbin_digest",
    "fatbin_path",
    "load_manifest",
    "manifest_path",
    "prebuilt_source_digest",
    "shipped_prebuilt_facts",
]

#: Directory holding the shipped prebuilt artifacts.
PREBUILT_DIR = Path(__file__).resolve().parent

#: File name of the shipped fatbin (package data).
FATBIN_NAME = "pretok_kernel.fatbin"

#: File name of the build manifest that pins the fatbin identity.
MANIFEST_NAME = "build_manifest.json"

#: Compile-order source lineage of the prebuilt unit, relative to the
#: ``toktier.kernels`` package directory. The wrapper includes the
#: pristine JIT source, so both files are part of the identity.
PREBUILT_UNIT_SOURCES: tuple[str, ...] = ("prebuilt_unit.cu", "pretok_kernel.cu")

_PREBUILT_SOURCE_DOMAIN = b"toktier.kernel_prebuilt_source.v1\x00"
_FATBIN_DOMAIN = b"toktier.kernel_fatbin.v1\x00"
_CUBIN_DOMAIN = b"toktier.kernel_cubin.v1\x00"


def fatbin_path() -> Path:
    """Absolute path of the shipped fatbin (may not exist in a source tree)."""
    return PREBUILT_DIR / FATBIN_NAME


def manifest_path() -> Path:
    """Absolute path of the shipped build manifest."""
    return PREBUILT_DIR / MANIFEST_NAME


def prebuilt_source_digest() -> str:
    """``sha256:<hex>`` over the prebuilt unit's source lineage.

    Same construction as ``toktier.kernels.kernel_source_digest`` (name,
    length and exact bytes of each source in compile order) under the
    prebuilt domain tag, so the two digests cannot be confused for one
    another.
    """
    kernels_dir = PREBUILT_DIR.parent
    digest = hashlib.sha256()
    digest.update(_PREBUILT_SOURCE_DOMAIN)
    for name in PREBUILT_UNIT_SOURCES:
        data = (kernels_dir / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"


def fatbin_digest(data: bytes) -> str:
    """``sha256:<hex>`` over the fatbin container bytes, domain-tagged."""
    return "sha256:" + hashlib.sha256(_FATBIN_DOMAIN + data).hexdigest()


def cubin_digest(architecture: str, data: bytes) -> str:
    """``sha256:<hex>`` over one embedded architecture image.

    The architecture spelling (``sm_89`` / ``compute_75``) is bound into
    the digest so two identical images for different architectures (or a
    relabeled image) cannot present each other's digest.
    """
    payload = _CUBIN_DOMAIN + architecture.encode("utf-8") + b"\x00" + data
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def shipped_prebuilt_facts() -> tuple[bool, str | None]:
    """Whether a servable prebuilt fatbin ships in this installation.

    Returns ``(available, digest)``: available only when the fatbin file
    is present *and* matches the digest its build manifest records, in
    which case ``digest`` is the domain-tagged fatbin digest. Read-only
    and torch-free.

    This helper is the one answer to "is a prebuilt binary shipped?".
    ``toktier doctor``, the routing probe behind ``explain()``, and the
    explicit GPU engine's reports all call it, so the three surfaces
    cannot state different prebuilt facts for one installation.
    """
    path = fatbin_path()
    if not path.is_file():
        return False, None
    try:
        manifest = load_manifest()
    except (OSError, ValueError):
        return False, None
    observed = fatbin_digest(path.read_bytes())
    if observed != str(manifest["fatbin"]["digest"]):
        return False, None
    return True, observed


def load_manifest() -> dict[str, Any]:
    """Read and structurally validate the shipped build manifest.

    Raises ``FileNotFoundError`` when the manifest is not shipped (a
    source checkout before any fatbin build) and ``ValueError`` when a
    required member is missing -- the caller treats both as "prebuilt
    delivery unavailable", never as a pass.
    """
    raw = json.loads(manifest_path().read_text(encoding="utf-8"))
    required = (
        "schema",
        "toolchain",
        "nvcc_argv",
        "architectures",
        "fatbin",
        "sources",
        "kernels",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(
            f"prebuilt build manifest is missing members: {missing}"
        )
    manifest: dict[str, Any] = raw
    return manifest
