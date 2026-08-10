"""Shared builders for the backend and routing tests.

Everything here is deliberately explicit: the routing tests describe
machine states as data, and the backend tests build a real (tiny)
artifact rather than mocking the oracle, so a test that passes says
something about the code that will run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from toktier.config import Config
from toktier.engine.gpu.class_tables import ClassTableStore
from toktier.engine.gpu.families import KernelFamilyTable
from toktier.engine.gpu.loader import KernelLoader
from toktier.errors import BackendExecutionFault
from toktier.policy import BACKEND_GPU, RoutingPolicy
from toktier.routing.probe import DeviceInfo, KernelCacheState, ProbeSnapshot
from toktier.routing.registry_view import RegistryView, ValidatedRegistryDocument

#: Parity readings collected during a session and reported at the end.
PARITY_READINGS: list[dict[str, object]] = []

#: The real engine binding set, produced by the same code path a
#: process presents for verification (host-computable: no GPU, no
#: build). The green routing fixtures below are derived from it, so a
#: producer/consumer schema drift fails these tests instead of hiding
#: behind hand-typed constants that happen to match themselves.
ENGINE_BINDINGS = KernelLoader.certified_source_bindings(
    class_table_digest=ClassTableStore(KernelFamilyTable.load()).binding_digest()
)

ORACLE_ID = "oracle_a"
PIPELINE_ID = "pipeline_a"
ADDED_FRONTEND_ID = "added_a"
EVIDENCE_ID = "evidence_a"
ARTIFACT_SHA = "a" * 64
PIPELINE_FINGERPRINT = "b" * 64
ADDED_FINGERPRINT = "c" * 64
SOURCE_DIGEST = ENGINE_BINDINGS.source_digest
CLASS_TABLE_DIGEST = ENGINE_BINDINGS.class_table_digest
BINARY_DIGEST = "f" * 64
ORACLE_VERSION = "0.22.2"
DEVICE = DeviceInfo(index=0, name="test device", architecture="sm_120")
BUILD_FLAGS = ENGINE_BINDINGS.build_flags
TOOLCHAIN = ">=12.0"
HOST_SOURCE_DIGEST = "1" * 64
HOST_BUILD_FLAGS = ("profile=release", "target=x86_64-unknown-linux-gnu")
HOST_TOOLCHAIN = "rustc 1.93.1 (01f9b12 2026-02-09)"


# ---------------------------------------------------------------------
# registry documents
# ---------------------------------------------------------------------


def certified_source_entry(**overrides: Any) -> dict[str, Any]:
    """A ``certified_source`` GPU entry with every bound constraint set."""
    entry = {
        "status": "certified_source",
        "source_digest": SOURCE_DIGEST,
        "class_table_digest": CLASS_TABLE_DIGEST,
        "build_flags": list(BUILD_FLAGS),
        "toolchain": TOOLCHAIN,
        "devices": [DEVICE.architecture],
        "driver_min": "560.0",
    }
    entry.update(overrides)
    return entry


def certified_entry(**overrides: Any) -> dict[str, Any]:
    """A ``certified`` GPU entry bound to a binary digest."""
    entry: dict[str, Any] = {
        "status": "certified",
        "binary_digest": BINARY_DIGEST,
        "host_source_digest": HOST_SOURCE_DIGEST,
        "host_build_flags": list(HOST_BUILD_FLAGS),
        "host_toolchain": HOST_TOOLCHAIN,
        "devices": [DEVICE.architecture],
        "driver_min": "560.0",
    }
    entry.update(overrides)
    return entry


def registry_document(
    *,
    backends: Mapping[str, Any] | None = None,
    artifact_sha256: str = ARTIFACT_SHA,
    certified_versions: Sequence[str] = (ORACLE_VERSION,),
    compositions: Sequence[tuple[str, str]] = (),
) -> ValidatedRegistryDocument:
    """A minimal registry document in the schema-v1 shape."""
    default = {BACKEND_GPU: certified_source_entry()}
    entries = dict(backends) if backends is not None else default
    document: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": {
            "tool": "test",
            "tool_version": "0",
            "source_commit": "0000000",
            "generated_at": "2026-01-01T00:00:00Z",
        },
        "root_digest": "sha256:" + "0" * 64,
        "oracles": [
            {
                "oracle_id": ORACLE_ID,
                "package": "tokenizers",
                "certified_versions": list(certified_versions),
                "semantic_id": "oracle_a_v1",
            }
        ],
        "pipelines": [
            {"pipeline_id": PIPELINE_ID, "pipeline_fingerprint": PIPELINE_FINGERPRINT}
        ],
        "added_frontends": [
            {
                "added_frontend_id": ADDED_FRONTEND_ID,
                "added_frontend_fingerprint": ADDED_FINGERPRINT,
            }
        ],
        "artifacts": [
            {
                "artifact_sha256": artifact_sha256,
                "family": "test_family",
                "pipeline_id": PIPELINE_ID,
                "added_frontend_id": ADDED_FRONTEND_ID,
                "oracle_id": ORACLE_ID,
                "suite_version": "suite-1",
                "evidence_id": EVIDENCE_ID,
                "readings": {"docs": 1000, "bytes": 4096, "mismatches": 0},
                "backends": entries,
            }
        ],
        "compositions": [
            {
                "pipeline_id": pipeline_id,
                "added_frontend_id": added_id,
                "evidence_id": EVIDENCE_ID,
            }
            for pipeline_id, added_id in compositions
        ],
    }
    # The helper supplies every required schema field; backend overrides stay
    # dynamic so individual tests can vary the records they exercise.
    return cast(ValidatedRegistryDocument, document)


def registry(**kwargs: Any) -> RegistryView:
    """A registry view over :func:`registry_document`."""
    return RegistryView.from_document(registry_document(**kwargs))


# ---------------------------------------------------------------------
# probe snapshots
# ---------------------------------------------------------------------


def gpu_ready_kernel_cache(**overrides: Any) -> KernelCacheState:
    """Kernel facts that satisfy a ``certified_source`` certificate.

    Derived from the real engine binding set through the shared
    ``from_bindings`` path, so these facts are spelled exactly the way a
    live probe would spell them.
    """
    state = KernelCacheState.from_bindings(
        ENGINE_BINDINGS,
        built=True,
        binary_digest=BINARY_DIGEST,
        toolchain="12.4",
        toolchain_satisfied=True,
        loaded_flag_sets=1,
        host_source_digest=HOST_SOURCE_DIGEST,
        host_build_flags=HOST_BUILD_FLAGS,
        host_toolchain=HOST_TOOLCHAIN,
    )
    return dataclasses.replace(state, **overrides) if overrides else state


def snapshot(
    *,
    registry_view: RegistryView | None = None,
    gpu_importable: bool = True,
    devices: Sequence[DeviceInfo] = (DEVICE,),
    devices_probed: bool = True,
    driver_version: str | None = "570.1",
    oracle_version: str | None = ORACLE_VERSION,
    artifact_sha256: str | None = ARTIFACT_SHA,
    pipeline_fingerprint: str | None = PIPELINE_FINGERPRINT,
    added_fingerprint: str | None = ADDED_FINGERPRINT,
    kernel_cache: KernelCacheState | None = None,
    reference_importable: bool = True,
) -> ProbeSnapshot:
    """Build a probe snapshot describing one machine state."""
    view = registry_view if registry_view is not None else registry()
    backends = set()
    if reference_importable:
        backends.add("hf")
    if gpu_importable:
        backends.add(BACKEND_GPU)
    return ProbeSnapshot(
        family="test_family",
        artifact_sha256=artifact_sha256,
        pipeline_fingerprint=pipeline_fingerprint,
        added_frontend_fingerprint=added_fingerprint,
        oracle_version=oracle_version,
        importable_backends=frozenset(backends),
        devices=tuple(devices),
        devices_probed=devices_probed,
        driver_version=driver_version,
        kernel_cache=kernel_cache or gpu_ready_kernel_cache(),
        certification=view.certification(
            artifact_sha256=artifact_sha256,
            pipeline_fingerprint=pipeline_fingerprint,
            added_frontend_fingerprint=added_fingerprint,
        ),
    )


def config(**overrides: Any) -> Config:
    """A configuration with no environment or file influence."""
    values: dict[str, Any] = {
        "home": None,
        "offline": False,
        "log_level": "WARNING",
        "disable_gpu": False,
        "diagnostics": False,
        "routing_policy": RoutingPolicy.CERTIFIED,
    }
    values.update(overrides)
    return Config(**values)


# ---------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    """The byte-level alphabet used by byte-level BPE artifacts."""
    printable = (
        list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    )
    mapped = list(printable)
    extra = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            mapped.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, mapped, strict=False)}


def byte_level_document(
    added_tokens: Sequence[Mapping[str, Any]] = (),
    *,
    normalizer: Any = None,
    truncation: Any = None,
    padding: Any = None,
) -> dict[str, Any]:
    """A tiny but real byte-level artifact: one token per byte.

    No merges, so every id is a direct function of the exact bytes of a
    span. That makes a difference in extraction spans visible in the
    ids, which is what the frontend tests need.
    """
    alphabet = bytes_to_unicode()
    vocab = {alphabet[value]: value for value in range(256)}
    byte_level = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": True,
    }
    return {
        "version": "1.0",
        "truncation": truncation,
        "padding": padding,
        "added_tokens": [dict(token) for token in added_tokens],
        "normalizer": normalizer,
        "pre_tokenizer": byte_level,
        "post_processor": None,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [],
        },
    }


@dataclass(frozen=True)
class StubArtifact:
    """A verified artifact handle, as the artifacts subsystem yields one.

    Interface alignment note: the real implementation lives in the
    artifacts lane. This stub exists so the reference backend can be
    exercised against the shape it will be handed, including the
    per-file digest map.
    """

    family: str
    root: Path
    artifact_sha256: str
    files: Mapping[str, str] = field(default_factory=dict)

    def path(self, relative_name: str) -> Path:
        """Absolute path of one verified file."""
        return self.root / relative_name


def write_artifact(
    directory: Path,
    document: Mapping[str, Any],
    *,
    family: str = "test_family",
    recorded_sha256: str | None = None,
) -> StubArtifact:
    """Write ``tokenizer.json`` and return a handle for it."""
    directory.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document).encode("utf-8")
    (directory / "tokenizer.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return StubArtifact(
        family=family,
        root=directory,
        artifact_sha256=recorded_sha256 or digest,
        files={"tokenizer.json": recorded_sha256 or digest},
    )


def local_artifact(directory: Path, *, family: str) -> StubArtifact:
    """Handle for an already-present artifact directory."""
    raw = (directory / "tokenizer.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    return StubArtifact(
        family=family,
        root=directory,
        artifact_sha256=digest,
        files={"tokenizer.json": digest},
    )


# ---------------------------------------------------------------------
# fake backends
# ---------------------------------------------------------------------


class FakeBackend:
    """A backend that returns recognizable ids, or raises on demand.

    ``error_type`` defaults to the one exception type the executor
    treats as recoverable; tests hand another type to show that
    anything else propagates.
    """

    def __init__(
        self,
        backend_id: str,
        *,
        fail: bool = False,
        base: int = 0,
        error_type: type[Exception] = BackendExecutionFault,
    ) -> None:
        self._backend_id = backend_id
        self.fail = fail
        self.base = base
        self.error_type = error_type
        self.calls: list[str] = []

    @property
    def backend_id(self) -> str:
        """Backend identifier."""
        return self._backend_id

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode one document."""
        self.calls.append(text)
        if self.fail:
            raise self.error_type(f"{self._backend_id} refused")
        return [self.base + len(text), int(add_special_tokens)]

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Encode a batch."""
        if self.fail:
            self.calls.extend(texts)
            raise self.error_type(f"{self._backend_id} refused")
        return [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]

    def close(self) -> None:
        """No resources to release."""


class FakeScanner:
    """Literal scanner that reports a hit for texts holding a marker."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def scan(self, text: str) -> list[tuple[str, int | None]] | None:
        """Ordered spans, or ``None`` when the marker is absent."""
        if self.marker not in text:
            return None
        head, _, tail = text.partition(self.marker)
        return [(head, None), (self.marker, 99), (tail, None)]
