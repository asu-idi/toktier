"""How ``gpu_delivery="auto"`` and ``device="cuda:N"`` are resolved.

``auto`` prefers the prebuilt delivery whenever it can serve: a
tokenizer takes ``prebuilt`` when the route plan built with that
delivery holds the GPU backend; otherwise ``jit`` when the ``gpu-jit``
installation profile is present (``ninja`` importable, the profile's
marker) and the plan built with ``jit`` holds it; otherwise
``prebuilt``, so a closed route reports the prebuilt reasons. With
``device="cpu"`` or the reference policy nothing is probed and ``auto``
reads ``prebuilt``. A process that has already loaded one delivery keeps
it, since a second cannot load beside it.

``toktier doctor`` has no artifact, so it answers with the host default
under the same preference: ``prebuilt`` when the shipped fatbin is
present and verified, the native request host reports its build facts
and some observed architecture is admitted under the prebuilt statuses;
otherwise ``jit`` when the profile is installed; otherwise ``prebuilt``.

The preference cannot pick a weaker label than the JIT pair on the
device axis while every shipped record's prebuilt judged devices contain
its JIT judged devices; :func:`prebuilt_covers_jit_devices` states that
property of the shipped data, and the test tier checks it.

``device`` accepts ``auto``, ``cpu``, ``cuda`` and ``cuda:N``, with ``N``
a decimal CUDA ordinal among the visible devices without a leading zero
(``cuda:0`` included). ``cuda:N`` is required like ``cuda``; the planner
sees device ``N`` alone (:class:`OrdinalDeviceProbe`), so another card
on a mixed host cannot open the route for it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from importlib.util import find_spec
from typing import Any, Protocol, TypeVar

__all__ = [
    "GPU_DELIVERIES",
    "OrdinalDeviceProbe",
    "auto_delivery",
    "host_default_delivery",
    "parse_device",
    "prebuilt_covers_jit_devices",
    "profile_jit_installed",
]

GPU_DELIVERIES = ("auto", "prebuilt", "jit")
_DEVICE_NAMES = ("auto", "cpu", "cuda")
_ORDINAL = re.compile(r"cuda:(0|[1-9][0-9]*)")

_Plan = TypeVar("_Plan")


def profile_jit_installed() -> bool:
    """Whether the ``gpu-jit`` profile is installed (``ninja`` importable).

    Extras leave no runtime flag in package metadata; ``ninja`` is unique
    to ``gpu-jit`` and so serves as its marker.
    """
    try:
        return find_spec("ninja") is not None
    except (ImportError, ValueError):
        return False


def parse_device(device: object) -> tuple[str, int | None]:
    """``(kind, ordinal)`` for a ``device`` argument, or ``ValueError``.

    ``kind`` is ``auto``, ``cpu`` or ``cuda``; ``ordinal`` is ``N`` for
    ``cuda:N`` and ``None`` otherwise.
    """
    if isinstance(device, str):
        if device in _DEVICE_NAMES:
            return device, None
        match = _ORDINAL.fullmatch(device)
        if match is not None:
            return "cuda", int(match.group(1))
    raise ValueError(
        "device must be one of ('auto', 'cpu', 'cuda') or 'cuda:N' with N a "
        f"decimal device ordinal, not {device!r}"
    )


def auto_delivery(
    plan_for: Callable[[str], _Plan],
    holds_gpu: Callable[[_Plan], bool],
    *,
    jit_installed: bool,
) -> tuple[str, _Plan]:
    """Resolve ``auto`` to one delivery and return the plan built with it."""
    prebuilt = plan_for("prebuilt")
    if holds_gpu(prebuilt):
        return "prebuilt", prebuilt
    if jit_installed:
        jit = plan_for("jit")
        if holds_gpu(jit):
            return "jit", jit
    return "prebuilt", prebuilt


def host_default_delivery(
    *,
    prebuilt_available: bool,
    native_host_ready: bool,
    observed_architectures: Iterable[str],
    prebuilt_statuses: Mapping[str, str],
    admitted_statuses: Iterable[str],
    jit_installed: bool,
) -> str:
    """``doctor``'s artifact-free answer for ``auto`` on this host."""
    admitted = set(admitted_statuses)
    architecture_admitted = any(
        prebuilt_statuses.get(architecture) in admitted
        for architecture in observed_architectures
    )
    if prebuilt_available and native_host_ready and architecture_admitted:
        return "prebuilt"
    return "jit" if jit_installed else "prebuilt"


def prebuilt_covers_jit_devices(document: Mapping[str, Any]) -> list[str]:
    """Records whose prebuilt judged devices miss a JIT judged device.

    Read over a support-registry document; an empty list is the property
    the ``auto`` preference rests on.
    """
    problems: list[str] = []
    for record in document.get("artifacts") or []:
        if not isinstance(record, Mapping):
            continue
        gpu = (record.get("backends") or {}).get("gpu")
        if not isinstance(gpu, Mapping):
            continue
        deliveries = gpu.get("deliveries") or {}
        jit = deliveries.get("jit", gpu)
        prebuilt = deliveries.get("prebuilt")
        if not isinstance(prebuilt, Mapping) or not isinstance(jit, Mapping):
            continue
        missing = set(jit.get("devices") or ()) - set(prebuilt.get("devices") or ())
        if missing:
            problems.append(f"{record.get('family')}: {sorted(missing)}")
    return problems


class _DeviceProbe(Protocol):
    def devices(self) -> tuple[Any, ...]: ...

    def driver_version(self) -> str | None: ...

    def kernel_cache(self) -> Any: ...


class OrdinalDeviceProbe:
    """A device probe that reports exactly one CUDA ordinal to the planner."""

    def __init__(self, inner: _DeviceProbe, ordinal: int) -> None:
        self._inner = inner
        self._ordinal = ordinal
        self._visible: tuple[Any, ...] | None = None

    @property
    def ordinal(self) -> int:
        return self._ordinal

    def visible_devices(self) -> tuple[Any, ...]:
        """Every device the inner probe reports, read once."""
        if self._visible is None:
            self._visible = tuple(self._inner.devices())
        return self._visible

    def devices(self) -> tuple[Any, ...]:
        return tuple(
            device
            for device in self.visible_devices()
            if getattr(device, "index", None) == self._ordinal
        )

    def driver_version(self) -> str | None:
        return self._inner.driver_version()

    def kernel_cache(self) -> Any:
        return self._inner.kernel_cache()
