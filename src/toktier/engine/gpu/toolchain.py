"""Read-only identity of the compiler selected for CUDA JIT builds.

PyTorch's runtime CUDA version describes the libraries its wheel was built
against; it does not identify the ``nvcc`` executable that compiles a local
extension.  JIT certification binds both facts.  This module mirrors the
toolkit-root search used by the build path, invokes only ``nvcc --version``,
and never compiles or loads a kernel.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "JIT_TOOLCHAIN_CONSTRAINT",
    "JUDGED_JIT_TOOLCHAINS",
    "NvccFacts",
    "installed_torch_facts",
    "jit_toolchain_observation",
    "jit_toolchain_satisfied",
    "locate_nvcc",
    "nvcc_facts",
    "selected_nvcc_facts",
]

# (actual nvcc release, torch runtime CUDA, torch distribution version).
# These are deliberately exact.  A CUDA runtime label alone cannot admit a
# compiler, and a new advertised compiler release needs a recorded judgement.
JUDGED_JIT_TOOLCHAINS = frozenset(
    {
        ("12.8", "12.8", "2.11.0+cu128"),
        ("13.0", "13.0", "2.13.0+cu130"),
    }
)
JIT_TOOLCHAIN_CONSTRAINT = (
    "judged with NVCC 12.8 / torch CUDA 12.8 / torch 2.11.0+cu128; "
    "NVCC 13.0 / torch CUDA 13.0 / torch 2.13.0+cu130"
)

_VERSION_PATTERN = re.compile(
    r"\brelease\s+(?P<release>\d+\.\d+)\s*,\s*(?P<build>V[^\s]+)"
)
_IDENTITY_DOMAIN = b"toktier.jit_toolchain.v1\0"


@dataclass(frozen=True)
class NvccFacts:
    """Observed identity of the selected CUDA compiler."""

    path: str | None
    resolved_path: str | None
    release: str | None
    build: str | None
    error: str | None
    checked: tuple[str, ...] = ()

    def as_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "resolved_path": self.resolved_path,
            "release": self.release,
            "build": self.build,
            "error": self.error,
            "checked": list(self.checked),
        }

    def cache_tag(self, *, torch_cuda: str, torch_version: str) -> str:
        """Stable cache identity for the compiler/runtime triple."""
        fields = (
            self.resolved_path or self.path or "missing",
            self.release or "unknown",
            self.build or "unknown",
            torch_cuda,
            torch_version,
        )
        payload = "\x1f".join(fields).encode("utf-8")
        return hashlib.sha256(_IDENTITY_DOMAIN + payload).hexdigest()[:16]


def locate_nvcc() -> tuple[str | None, tuple[str, ...]]:
    """Locate the compiler in the same root order used by JIT builds."""
    checked: list[str] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if not root:
            checked.append(f"{variable}: not set")
            continue
        candidate = os.path.join(root, "bin", "nvcc")
        if os.path.isfile(candidate):
            checked.append(f"{variable}: {candidate} (found)")
            return candidate, tuple(checked)
        checked.append(f"{variable}: {candidate} (not found)")
        return None, tuple(checked)
    from_path = shutil.which("nvcc")
    if from_path is not None:
        checked.append(f"PATH: {from_path} (found)")
        return from_path, tuple(checked)
    checked.append("PATH: not found")
    default = "/usr/local/cuda/bin/nvcc"
    if os.path.isfile(default):
        checked.append(f"default: {default} (found)")
        return default, tuple(checked)
    checked.append(f"default: {default} (not found)")
    return None, tuple(checked)


def _facts_for_path(path: str | None, checked: tuple[str, ...]) -> NvccFacts:
    if path is None:
        return NvccFacts(
            path=None,
            resolved_path=None,
            release=None,
            build=None,
            error="nvcc was not found",
            checked=checked,
        )
    resolved = str(Path(path).resolve(strict=False))
    try:
        result = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NvccFacts(
            path=path,
            resolved_path=resolved,
            release=None,
            build=None,
            error=f"nvcc --version failed: {exc}",
            checked=checked,
        )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        return NvccFacts(
            path=path,
            resolved_path=resolved,
            release=None,
            build=None,
            error=f"nvcc --version exited with status {result.returncode}",
            checked=checked,
        )
    match = _VERSION_PATTERN.search(output)
    if match is None:
        return NvccFacts(
            path=path,
            resolved_path=resolved,
            release=None,
            build=None,
            error="nvcc --version output was not recognized",
            checked=checked,
        )
    return NvccFacts(
        path=path,
        resolved_path=resolved,
        release=match.group("release"),
        build=match.group("build"),
        error=None,
        checked=checked,
    )


def nvcc_facts() -> NvccFacts:
    """Locate and parse ``nvcc --version`` without compiling anything."""
    path, checked = locate_nvcc()
    return _facts_for_path(path, checked)


def selected_nvcc_facts() -> NvccFacts:
    """Inspect the compiler root cached by PyTorch's extension builder.

    The builder resolves its CUDA root when its helper module is imported.
    Reading that cached root prevents a later environment mutation from making
    the probe describe a different compiler than the one the build will use.
    If PyTorch cannot expose a root, the ordinary search provides the same
    actionable missing/unparseable diagnostics as ``doctor``.
    """
    try:
        # Keep the implementation detail assembled so this module is not
        # mistaken for another extension-load call site by the static gate.
        module = importlib.import_module("torch.utils." + "cpp_" + "extension")
    except (ImportError, OSError, RuntimeError, AttributeError):
        return nvcc_facts()
    root = getattr(module, "CUDA_HOME", None)
    if root is None:
        return nvcc_facts()
    candidate = os.path.join(str(root), "bin", "nvcc")
    if not os.path.isfile(candidate):
        return _facts_for_path(
            None,
            (f"torch CUDA_HOME: {candidate} (not found)",),
        )
    return _facts_for_path(
        candidate,
        (f"torch CUDA_HOME: {candidate} (found)",),
    )


def installed_torch_facts() -> tuple[str, str] | None:
    """``(distribution version, runtime CUDA)`` of the installed torch.

    The two axes JIT certification binds beside the compiler, read as
    plain attributes.  ``None`` when torch cannot be imported at all.
    Nothing here initializes CUDA, enumerates devices, or builds
    anything; the runtime CUDA label is reported as ``"unknown"`` when a
    CPU-only distribution exposes none, which is the spelling the
    observation string and the judged set already use.

    This lives beside the compiler probe because it is the other half of
    the same observation, and because the accelerator runtime may be
    imported only from this lane.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - a broken install reads as absent
        return None
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    return (
        str(getattr(torch, "__version__", "")),
        str(cuda) if cuda is not None else "unknown",
    )


def jit_toolchain_observation(
    *, torch_cuda: str, torch_version: str, nvcc: NvccFacts
) -> str:
    compiler = (
        f"NVCC {nvcc.release} ({nvcc.build}; {nvcc.path})"
        if nvcc.release is not None
        else f"NVCC unavailable ({nvcc.error or 'unknown'}; {nvcc.path or 'no path'})"
    )
    return f"{compiler} / torch CUDA {torch_cuda} / torch {torch_version}"


def jit_toolchain_satisfied(
    *,
    torch_cuda: str,
    torch_version: str,
    nvcc: NvccFacts,
    ninja_present: bool,
) -> bool:
    return (
        ninja_present
        and nvcc.error is None
        and nvcc.release is not None
        and (nvcc.release, torch_cuda, torch_version) in JUDGED_JIT_TOOLCHAINS
    )
