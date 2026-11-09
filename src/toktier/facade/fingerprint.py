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

Predecessor stores. ``store_fingerprint_history.v1.json`` (digest
checked, generated from the release tags) lists the releases that wrote
the earlier (v1) preimage, with their fast CPU v1 digest, both v2
identities and, per registered artifact, the loader face and repair
parameters that release ran. When this build's bound v2 pair equals a
row's and the row names the artifact with the same loader face and
repair parameters, a native-owned store whose ``meta.json`` holds the v1
fingerprint this process computes with that row's digest substituted is
opened under that stored value. Nothing is rewritten: records keep their
key, both check levels of the store stay as they are, and the release
that wrote the store can still open it. Any other stored value is
refused as before.
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
    "FINGERPRINT_V1_TAG",
    "FINGERPRINT_V2_TAG",
    "PredecessorRelease",
    "SourceBinding",
    "bound_source_v2",
    "fingerprint",
    "predecessor_releases",
    "read_store_history",
    "repair_parameters_component",
    "shipped_registry_document",
    "source_binding",
    "store_history",
    "stored_fingerprint",
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


# -- predecessor stores (history table) ----------------------------------

#: Domain tag of the history table's root digest.
_HISTORY_DOMAIN_TAG = b"toktier.store_fingerprint_history.v1\x00"
#: Domain tag of the preimage the listed releases wrote.
FINGERPRINT_V1_TAG = b"toktier.facade.fingerprint.v1\0"
_HISTORY_NAME = "store_fingerprint_history.v1.json"


@dataclass(frozen=True)
class PredecessorRelease:
    """One release that wrote v1-form store fingerprints."""

    release: str
    fast_cpu_source_digest: str
    fast_cpu_source_digest_v2: str
    native_host_source_digest_v2: str
    #: artifact sha256 -> (loader face sha256, repair parameters component)
    artifacts: Mapping[str, tuple[str | None, str]]


def _history_path() -> Any:
    from ..routing.tables import TABLE_DIR

    return TABLE_DIR / _HISTORY_NAME


@lru_cache(maxsize=1)
def store_history() -> tuple[PredecessorRelease, ...]:
    """The digest-verified history table shipped with this installation."""
    return read_store_history(_history_path())


def read_store_history(path: Any) -> tuple[PredecessorRelease, ...]:
    """Read one history table and verify its root digest."""
    from .._jcs import CanonicalizationError, canonical_json
    from ..errors import RegistryInvalid

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in raw.items() if key != "root_digest"}
        recomputed = (
            "sha256:"
            + hashlib.sha256(_HISTORY_DOMAIN_TAG + canonical_json(body)).hexdigest()
        )
    except (OSError, ValueError, AttributeError, CanonicalizationError) as error:
        raise RegistryInvalid(
            "the store fingerprint history table cannot be read",
            details={"path": str(path), "failure": str(error)},
        ) from error
    if raw.get("root_digest") != recomputed:
        raise RegistryInvalid(
            "the store fingerprint history table digest does not match",
            details={
                "path": str(path),
                "failure": f"recorded {raw.get('root_digest')}, "
                f"recomputed {recomputed}",
            },
        )
    try:
        return tuple(
            PredecessorRelease(
                release=str(row["release"]),
                fast_cpu_source_digest=str(row["fast_cpu_source_digest"]),
                fast_cpu_source_digest_v2=str(row["fast_cpu_source_digest_v2"]),
                native_host_source_digest_v2=str(row["native_host_source_digest_v2"]),
                artifacts={
                    str(sha): (
                        entry.get("loader_face_sha256"),
                        str(entry["repair_parameters"]),
                    )
                    for sha, entry in row["artifacts"].items()
                },
            )
            for row in raw["releases"]
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise RegistryInvalid(
            "the store fingerprint history table has an unexpected shape",
            details={"path": str(path), "failure": str(error)},
        ) from error


def predecessor_releases(
    binding: SourceBinding,
    *,
    artifact_sha256: str,
    loader_face_sha256: str,
    repair_parameters: str,
    history: Iterable[PredecessorRelease] | None = None,
) -> list[PredecessorRelease]:
    """Releases whose v1-form stores this build may open under their key.

    Only a bound v2 axis consults the table, and only rows whose v2 pair
    equals this build's and that name this artifact with the same loader
    face and repair parameters qualify.
    """
    if binding.axis != AXIS_SOURCE_V2:
        return []
    rows = store_history() if history is None else tuple(history)
    return [
        row
        for row in rows
        if row.fast_cpu_source_digest_v2 == binding.fast_cpu
        and row.native_host_source_digest_v2 == binding.native_host
        and row.artifacts.get(artifact_sha256)
        == (loader_face_sha256, repair_parameters)
    ]


def stored_fingerprint(directory: Any) -> bytes | None:
    """The fingerprint a store directory's ``meta.json`` names, if readable."""
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        value = meta.get("fingerprint") if isinstance(meta, dict) else None
        raw = bytes.fromhex(value) if isinstance(value, str) else None
    except (OSError, ValueError):
        return None
    return raw if raw is not None and len(raw) == 32 else None
