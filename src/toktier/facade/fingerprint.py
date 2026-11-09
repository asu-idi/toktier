"""The facade's store fingerprint: which build identity a store is bound to.

A store directory (Python facade) is keyed by a 32-byte fingerprint over
the components that decide the ids a stored session holds. This module
decides which source identity enters it.

The extension reports the version-carrying (v1) source digests of the
fast CPU engine and of the native request host. Every certified fast CPU
entry of the digest-verified shipped registry records the pair (v1, v2)
for the engine, and every certified prebuilt GPU delivery the pair for
the host, where v2 is the version-normalized identity. The v2 pair is
**bound** when both v1 digests equal the ones those entries record and
all such entries agree on the v2 value -- the same equality the planner
uses to call the route certified. A store owned by the native request
runtime is then keyed on the axis ``source-v2``: it is refused when the
engine or host code, the reference library version, the build toolchain
or flags change, and not when only the version number does.

Otherwise -- a development build, a waived binding, an inconsistent
registry, or a store owned by the Python entry store and repair adapter
(whose code is outside both source identities) -- the axis is
``source-v1`` and the store stays tied to that exact build.

Besides the source identity the preimage binds the artifact, the loader
face the reference executes, the oracle package and version, the facade
engine and session semantics, the repair backend, its configuration,
version and digest, the repair parameters in force, the engine version,
build flags, toolchain and configuration digest. A store created under
this form is not opened by earlier releases.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

__all__ = [
    "AXIS_SOURCE_V1",
    "AXIS_SOURCE_V2",
    "FINGERPRINT_V2_TAG",
    "SourceBinding",
    "bound_source_v2",
    "fingerprint",
    "repair_parameters_component",
    "shipped_registry_document",
    "source_binding",
]

FINGERPRINT_V2_TAG = b"toktier.facade.fingerprint.v2\0"
AXIS_SOURCE_V1 = "source-v1"
AXIS_SOURCE_V2 = "source-v2"

_ELIGIBLE = frozenset({"certified", "certified_source"})


def fingerprint(tag: bytes, components: Iterable[str]) -> bytes:
    """SHA-256 over ``tag`` and each component as LE32 length + UTF-8."""
    digest = hashlib.sha256(tag)
    for component in components:
        raw = component.encode("utf-8")
        digest.update(len(raw).to_bytes(4, "little"))
        digest.update(raw)
    return digest.digest()


@lru_cache(maxsize=1)
def shipped_registry_document() -> Mapping[str, Any]:
    """The digest-verified shipped support registry, as a document."""
    from ..routing.registry_load import load_registry_document
    from ..routing.tables import SUPPORT_REGISTRY

    return load_registry_document(SUPPORT_REGISTRY)


def _single_v2(pairs: list[tuple[object, object]]) -> str | None:
    values = {v2 for _v1, v2 in pairs}
    if len(values) != 1:
        return None
    (value,) = values
    return value if isinstance(value, str) and value else None


def bound_source_v2(
    document: Mapping[str, Any],
    *,
    fast_cpu_v1: str | None,
    native_host_v1: str | None,
) -> tuple[str, str] | None:
    """The (fast CPU, native host) v2 pair the registry binds, or ``None``."""
    if not fast_cpu_v1 or not native_host_v1:
        return None
    engine: list[tuple[object, object]] = []
    host: list[tuple[object, object]] = []
    for record in document.get("artifacts") or []:
        if not isinstance(record, Mapping):
            continue
        backends = record.get("backends") or {}
        fast = backends.get("fast_cpu")
        if (
            isinstance(fast, Mapping)
            and fast.get("status") in _ELIGIBLE
            and fast.get("source_digest") == fast_cpu_v1
        ):
            engine.append((fast_cpu_v1, fast.get("source_digest_v2")))
        gpu = backends.get("gpu")
        prebuilt = (
            (gpu.get("deliveries") or {}).get("prebuilt")
            if isinstance(gpu, Mapping)
            else None
        )
        if (
            isinstance(prebuilt, Mapping)
            and prebuilt.get("status") in _ELIGIBLE
            and prebuilt.get("host_source_digest") == native_host_v1
        ):
            host.append((native_host_v1, prebuilt.get("host_source_digest_v2")))
    engine_v2 = _single_v2(engine)
    host_v2 = _single_v2(host)
    if engine_v2 is None or host_v2 is None:
        return None
    return engine_v2, host_v2


@dataclass(frozen=True)
class SourceBinding:
    """The source identity a store fingerprint carries."""

    axis: str
    fast_cpu: str
    native_host: str

    def report(self) -> dict[str, str]:
        return {
            "axis": self.axis,
            "fast_cpu": self.fast_cpu,
            "native_host": self.native_host,
        }


def source_binding(
    *,
    native_owned: bool,
    fast_cpu_v1: str | None,
    native_host_v1: str | None,
    document: Mapping[str, Any] | None = None,
) -> SourceBinding:
    """The axis and digests for a store owned by the native runtime or not."""
    if native_owned:
        pair = bound_source_v2(
            document if document is not None else shipped_registry_document(),
            fast_cpu_v1=fast_cpu_v1,
            native_host_v1=native_host_v1,
        )
        if pair is not None:
            return SourceBinding(AXIS_SOURCE_V2, pair[0], pair[1])
    return SourceBinding(AXIS_SOURCE_V1, fast_cpu_v1 or "", native_host_v1 or "")


def repair_parameters_component(spec: object | None) -> str:
    """Canonical text of the repair parameters in force, or ``none``."""
    if spec is None:
        return "none"
    fields = asdict(spec)  # type: ignore[call-overload]
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))
