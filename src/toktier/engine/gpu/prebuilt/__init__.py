"""Prebuilt (fatbin) delivery of the GPU kernels.

``load_prebuilt_extension`` is the single entry the kernel loader uses:
it verifies the shipped fatbin against its build manifest, checks that
the installed driver can run it, loads it through the CUDA driver API
and returns the :class:`~.launcher.PrebuiltExtension` surface plus the
facts a binding report needs. Every refusal raises
:class:`PrebuiltUnavailable` with a stated reason -- the loader then
falls back to the JIT delivery (or reference) *and says so*; a silent
downgrade is a contract violation (PLAN honesty clause).

Driver floor: the fatbin is built with a CUDA 13.x toolchain, so its
images need an r580-generation (CUDA 13) driver or newer. The check is
explicit and precedes any load attempt, so the refusal message names
the installed and required levels instead of surfacing a raw driver
error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....kernels.prebuilt import (
    fatbin_digest,
    fatbin_path,
    load_manifest,
    prebuilt_source_digest,
)
from .driver import CudaDriver, CudaDriverError, driver_available

__all__ = [
    "PrebuiltLoad",
    "PrebuiltUnavailable",
    "load_prebuilt_extension",
    "shipped_fatbin_digest",
]

#: Minimum CUDA feature level the driver must expose to run the images.
_DRIVER_FLOOR = (13, 0)
_DRIVER_FLOOR_TEXT = "a CUDA 13 (r580 generation) driver"


class PrebuiltUnavailable(RuntimeError):
    """The prebuilt delivery cannot serve this process; reason stated."""

    def __init__(self, reason_code: str, message: str):
        self.reason_code = reason_code
        super().__init__(message)


@dataclass(frozen=True)
class PrebuiltLoad:
    """A loaded prebuilt extension plus its identity facts."""

    extension: Any
    fatbin_digest: str
    manifest: dict[str, Any]
    device_architecture: str
    architecture_embedded: bool


def shipped_fatbin_digest() -> str | None:
    """Digest of the shipped fatbin, or ``None`` when not shipped.

    Read-only and GPU-free: probes may call it to report the binary
    digest a ``certified`` record would be verified against.
    """
    path = fatbin_path()
    if not path.is_file():
        return None
    return fatbin_digest(path.read_bytes())


def _manifest_or_refuse() -> dict[str, Any]:
    try:
        return load_manifest()
    except FileNotFoundError as exc:
        raise PrebuiltUnavailable(
            "R_PREBUILT_NOT_SHIPPED",
            "no prebuilt fatbin is shipped in this installation "
            f"({exc}); the JIT delivery is the fallback",
        ) from exc
    except (OSError, ValueError) as exc:
        raise PrebuiltUnavailable(
            "R_PREBUILT_MANIFEST_INVALID",
            f"the prebuilt build manifest is unreadable: {exc}",
        ) from exc


def load_prebuilt_extension(device: str | None = None) -> PrebuiltLoad:
    """Verify, load and return the prebuilt extension for one device.

    Args:
        device: torch device string the first load should target
            (``None`` = current device). Later calls through the
            returned extension may use any device; the module is loaded
            lazily per device.

    Raises:
        PrebuiltUnavailable: fatbin missing, digest mismatch, torch or
            driver missing, driver below the floor, or module load
            refused by the driver. The message states the reason.
    """
    manifest = _manifest_or_refuse()
    path = fatbin_path()
    if not path.is_file():
        raise PrebuiltUnavailable(
            "R_PREBUILT_NOT_SHIPPED",
            "the build manifest is present but the fatbin file is not "
            f"({path}); the JIT delivery is the fallback",
        )
    data = path.read_bytes()
    observed = fatbin_digest(data)
    recorded = str(manifest["fatbin"]["digest"])
    if observed != recorded:
        raise PrebuiltUnavailable(
            "R_KERNEL_DIGEST_MISMATCH",
            "the shipped fatbin does not match its build manifest "
            f"(observed {observed}, recorded {recorded})",
        )
    source_now = prebuilt_source_digest()
    source_recorded = str(manifest["sources"]["prebuilt_source_digest"])

    try:
        import torch
    except ImportError as exc:
        raise PrebuiltUnavailable(
            "R_TORCH_MISSING",
            "the prebuilt delivery still needs torch for tensors and "
            f"streams (import failed: {exc})",
        ) from exc
    if not torch.cuda.is_available():
        raise PrebuiltUnavailable(
            "R_NO_CUDA_DEVICE",
            "torch reports no usable CUDA device",
        )
    if not driver_available():
        raise PrebuiltUnavailable(
            "R_DRIVER_MISSING",
            "libcuda is not loadable (no NVIDIA driver installed?)",
        )
    try:
        driver = CudaDriver.get()
        major, minor = driver.driver_cuda_version()
    except CudaDriverError as exc:
        raise PrebuiltUnavailable(
            "R_DRIVER_MISSING", f"the CUDA driver refused early: {exc}"
        ) from exc
    if (major, minor) < _DRIVER_FLOOR:
        raise PrebuiltUnavailable(
            "R_DRIVER_TOO_OLD",
            f"the installed driver exposes CUDA {major}.{minor}, but the "
            f"prebuilt fatbin (CUDA 13.x build) needs {_DRIVER_FLOOR_TEXT}; "
            "the JIT delivery against the local toolchain is the fallback",
        )

    index = torch.device(device).index if device else None
    if index is None:
        index = torch.cuda.current_device()
    cc_major, cc_minor = torch.cuda.get_device_capability(index)
    architecture = f"sm_{cc_major}{cc_minor}"
    embedded = architecture in manifest["architectures"]

    from .launcher import PrebuiltExtension

    extension = PrebuiltExtension(data, dict(manifest["kernels"]))
    try:
        extension.load_for_device(index)
    except CudaDriverError as exc:
        raise PrebuiltUnavailable(
            "R_PREBUILT_LOAD_FAILED",
            f"the driver refused the fatbin for {architecture}: {exc}"
            + (
                ""
                if embedded
                else (
                    f" ({architecture} has no embedded image; the driver "
                    "would have to JIT the compute_75 PTX)"
                )
            ),
        ) from exc
    facts: dict[str, Any] = {
        "extension": extension,
        "fatbin_digest": observed,
        "manifest": manifest,
        "device_architecture": architecture,
        "architecture_embedded": embedded,
    }
    if source_now != source_recorded:
        # The binary digest above is the authoritative binding; a source
        # tree that drifted after the build is recorded, not fatal.
        manifest = dict(manifest)
        manifest["source_drift"] = {
            "recorded": source_recorded,
            "observed": source_now,
        }
        facts["manifest"] = manifest
    return PrebuiltLoad(**facts)
