"""Minimal CUDA driver API binding for the prebuilt fatbin delivery.

Scope: exactly the five driver calls module loading and kernel launch
need (``cuInit``, ``cuDriverGetVersion``, ``cuModuleLoadData``,
``cuModuleGetFunction``, ``cuLaunchKernel``) plus error-string helpers,
bound through :mod:`ctypes` against the user's own ``libcuda`` (the
NVIDIA driver library every CUDA process already loads).

Why ctypes and not ``cuda-python``: the pip package that exposes the
driver API (``cuda-bindings``, pulled in by both ``cuda-python`` and
``cuda-core``) is distributed under ``LicenseRef-NVIDIA-SOFTWARE-
LICENSE``, a custom non-open-source license. The driver API surface
needed here is five stable C calls, so binding them directly keeps the
``gpu`` extra free of that license question and adds no dependency at
all: ``libcuda`` comes with the driver, never with this package.

Context handling: no context is created here. The caller (the prebuilt
loader) runs a torch operation on the target device first, which makes
that device's primary context current on the calling thread; module
loads and launches then land in the same context torch uses.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
from typing import Any

__all__ = [
    "CudaDriver",
    "CudaDriverError",
    "driver_available",
]

_CUDA_SUCCESS = 0

#: Error codes worth naming in refusal messages (subset).
_ERROR_HINTS = {
    218: "the fatbin holds no image this device can run (arch not embedded)",
    222: "the embedded PTX needs a newer driver than the one installed",
    221: "the fatbin image format is not supported by this driver",
    209: "no binary for the GPU: device architecture not covered",
    35: "the installed driver is too old for this CUDA feature level",
    100: "no CUDA device is visible",
    3: "CUDA is not initialized in this process",
}


class CudaDriverError(RuntimeError):
    """A CUDA driver call returned an error status."""

    def __init__(self, call: str, status: int, name: str, detail: str):
        self.call = call
        self.status = status
        self.error_name = name
        hint = _ERROR_HINTS.get(status)
        message = f"{call} failed: {name} ({status}): {detail}"
        if hint:
            message += f" -- {hint}"
        super().__init__(message)


def _load_libcuda() -> ctypes.CDLL | None:
    for candidate in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    found = ctypes.util.find_library("cuda")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError:
            return None
    return None


def driver_available() -> bool:
    """Whether ``libcuda`` can be loaded at all (no context is touched)."""
    return _load_libcuda() is not None


class CudaDriver:
    """Process-wide libcuda handle with typed wrappers.

    Thread-safe for the calls used here; the driver itself is
    thread-safe, and the only mutable state in this class is the lazily
    created singleton.
    """

    _instance: CudaDriver | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        lib = _load_libcuda()
        if lib is None:
            raise CudaDriverError(
                "dlopen(libcuda)", -1, "CUDA_ERROR_NOT_FOUND",
                "libcuda is not present (no NVIDIA driver installed?)",
            )
        self._lib = lib
        lib.cuGetErrorName.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
        ]
        lib.cuGetErrorString.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
        ]
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        ]
        lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
        ]
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._check(lib.cuInit(0), "cuInit")

    @classmethod
    def get(cls) -> CudaDriver:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # -- helpers -------------------------------------------------------

    def _error_text(self, status: int) -> tuple[str, str]:
        name = ctypes.c_char_p()
        detail = ctypes.c_char_p()
        self._lib.cuGetErrorName(status, ctypes.byref(name))
        self._lib.cuGetErrorString(status, ctypes.byref(detail))
        return (
            (name.value or b"CUDA_ERROR_UNKNOWN").decode(),
            (detail.value or b"unknown error").decode(),
        )

    def _check(self, status: int, call: str) -> None:
        if status != _CUDA_SUCCESS:
            name, detail = self._error_text(status)
            raise CudaDriverError(call, status, name, detail)

    # -- driver calls --------------------------------------------------

    def driver_cuda_version(self) -> tuple[int, int]:
        """(major, minor) of the CUDA version the driver supports."""
        version = ctypes.c_int()
        self._check(
            self._lib.cuDriverGetVersion(ctypes.byref(version)),
            "cuDriverGetVersion",
        )
        return version.value // 1000, (version.value % 1000) // 10

    def load_module(self, image: bytes) -> int:
        """``cuModuleLoadData`` into the current context; returns handle."""
        module = ctypes.c_void_p()
        buffer = ctypes.create_string_buffer(image, len(image))
        self._check(
            self._lib.cuModuleLoadData(
                ctypes.byref(module), ctypes.cast(buffer, ctypes.c_void_p)
            ),
            "cuModuleLoadData",
        )
        return int(module.value or 0)

    def get_function(self, module: int, symbol: str) -> int:
        function = ctypes.c_void_p()
        self._check(
            self._lib.cuModuleGetFunction(
                ctypes.byref(function),
                ctypes.c_void_p(module),
                symbol.encode("ascii"),
            ),
            f"cuModuleGetFunction({symbol})",
        )
        return int(function.value or 0)

    def launch(
        self,
        function: int,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        stream: int,
        args: list[ctypes._SimpleCData[Any] | ctypes.c_void_p],
        shared_bytes: int = 0,
    ) -> None:
        """``cuLaunchKernel`` with by-value parameter packing.

        ``args`` are already-constructed ctypes scalars (``c_void_p`` for
        pointers, ``c_int`` / ``c_uint`` / ``c_bool`` for scalars); this
        builds the ``void*[]`` of their addresses. The caller keeps the
        Python references alive across the call (the launch copies the
        values synchronously, so lifetime beyond the call is a caller
        concern only for the memory the pointers name).
        """
        params = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args]
        )
        status = self._lib.cuLaunchKernel(
            ctypes.c_void_p(function),
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_bytes,
            ctypes.c_void_p(stream),
            params,
            None,
        )
        self._check(status, "cuLaunchKernel")
