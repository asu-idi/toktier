"""Construction-time projection for the one-call Rust prebuilt GPU host.

Python remains responsible for verified artifact acquisition and immutable
registry/configuration projection.  This module loads and digest-checks those
facts once, converts the already exported NumPy tables to byte strings, and
hands ownership to :mod:`toktier._native`.  The native runtime then opens the
engine deferred, on the first request that actually routes to the GPU; no
CUDA call happens here.  The module imports neither torch nor the legacy
ctypes launcher; no Python object participates after construction.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...backends.protocol import TOKENIZER_FILE
from ...errors import KernelIncompatible
from ...kernels.bpe_tables import BpeTableStore
from ...kernels.prebuilt import fatbin_digest, fatbin_path, load_manifest
from .class_tables import ClassTableStore
from .families import KernelFamilyTable

if TYPE_CHECKING:
    from ... import _native
    from ...backends.protocol import ArtifactHandle

__all__ = [
    "NativeHostBuildFacts",
    "PreparedNativeGpu",
    "native_host_build_facts",
    "prepare_native_prebuilt_gpu",
]


@dataclass(frozen=True)
class NativeHostBuildFacts:
    """Identity emitted by the extension that executes prebuilt requests.

    Empty fields are an observation failure, never a permissive default. The
    routing planner compares all three fields with the prebuilt delivery row.
    """

    source_digest: str | None = None
    build_flags: tuple[str, ...] = ()
    toolchain: str | None = None


def native_host_build_facts() -> NativeHostBuildFacts:
    """Return validated, build-embedded native-host identity facts."""
    try:
        from ... import _native

        observed: object = _native.native_host_build_facts()
    except (ImportError, RuntimeError, TypeError, ValueError):
        return NativeHostBuildFacts()
    if not isinstance(observed, Mapping):
        return NativeHostBuildFacts()
    source_digest = observed.get("source_digest")
    build_flags = observed.get("build_flags")
    toolchain = observed.get("toolchain")
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
        or not isinstance(build_flags, list)
        or not build_flags
        or not all(isinstance(value, str) for value in build_flags)
        or not isinstance(toolchain, str)
        or not toolchain
    ):
        return NativeHostBuildFacts()
    return NativeHostBuildFacts(
        source_digest=source_digest,
        build_flags=tuple(build_flags),
        toolchain=toolchain,
    )


def _needs_nfc(path: Path) -> bool:
    document = json.loads(path.read_text(encoding="utf-8"))
    normalizer = document.get("normalizer")
    return isinstance(normalizer, dict) and normalizer.get("type") == "NFC"


def _digits_max(family: Any, table: Any) -> int:
    if family.digits_max is not None:
        return int(family.digits_max)
    meta = table.meta or {}
    value = meta.get("digits_max")
    if type(value) is not int:
        raise KernelIncompatible(
            f"family {family.name!r} has no bound digits_max value",
            details={
                "backend": "gpu",
                "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                "family": family.name,
            },
        )
    return value


@dataclass
class PreparedNativeGpu:
    """A deferred prebuilt engine plus the loader facts it publishes.

    ``engine`` owns the verified engine inputs; the native runtime opens it
    on the first request that routes to the GPU.  The identity facts stay
    here so :meth:`publish_loaded` can record the process-wide loaded fact
    once the facade observes that the open happened.
    """

    engine: Any
    manifest: dict[str, Any]
    fatbin_digest: str
    architecture: str
    published: bool = field(default=False)

    def publish_loaded(self) -> None:
        """Record the loaded delivery once; idempotent afterwards."""
        if self.published:
            return
        from .loader import KernelLoader

        KernelLoader.note_native_prebuilt_loaded(
            manifest=self.manifest,
            fatbin_digest=self.fatbin_digest,
            architecture=self.architecture,
        )
        self.published = True


def prepare_native_prebuilt_gpu(
    *,
    artifact: ArtifactHandle,
    family: str,
    cache_dir: Path,
    device_ordinal: int,
    architecture: str,
    reference: _native.ReferenceEngine,
) -> PreparedNativeGpu:
    """Project verified package data into a deferred Rust CUDA engine."""
    from ... import _native

    family_table = KernelFamilyTable.load()
    entry = family_table.get(family)
    band = family_table.band_spec(entry.band)
    if not band.e2e:
        raise KernelIncompatible(
            f"family {family!r} has no end-to-end prebuilt kernel",
            details={"backend": "gpu", "family": family},
        )
    class_store = ClassTableStore(family_table, cache_dir=cache_dir)
    class_table = class_store.load(entry.class_table)
    bpe = BpeTableStore(
        {family: artifact},
        cache_dir,
        # A facade instance owns one verified handle. Export its model
        # directly; cross-family cache reuse is an initialization optimization,
        # not a tokenization premise.
        shared_model={},
    ).load(family)

    manifest = load_manifest()
    image = fatbin_path().read_bytes()
    observed = fatbin_digest(image)
    expected = str(manifest["fatbin"]["digest"])
    if observed != expected:
        raise KernelIncompatible(
            "the shipped prebuilt fatbin does not match its build manifest",
            details={
                "backend": "gpu",
                "reason_code": "R_KERNEL_DIGEST_MISMATCH",
                "expected_digest": expected,
                "observed_digest": observed,
            },
        )
    tokenizer_path = artifact.path(TOKENIZER_FILE)
    unsafe = bpe.unsafe_bits
    engine = _native.NativePrebuiltGpu(
        family,
        artifact.artifact_sha256,
        image,
        expected,
        architecture,
        device_ordinal,
        entry.ruleset,
        _digits_max(entry, class_table),
        entry.contractions,
        _needs_nfc(tokenizer_path),
        int(bpe.ignore_merges),
        dict(manifest["kernels"]),
        class_table.array.tobytes(order="C"),
        bpe.pair_keys.tobytes(order="C"),
        bpe.pair_vals.tobytes(order="C"),
        bpe.byte_id.tobytes(order="C"),
        bpe.vocab_keys.tobytes(order="C"),
        bpe.vocab_vals.tobytes(order="C"),
        bpe.vocab_blob.tobytes(order="C"),
        b"" if unsafe is None else unsafe.tobytes(order="C"),
        int(bpe.pair_keys.size),
        int(bpe.vocab_keys.size),
        reference,
    )
    # Construction returning means the projected shapes were validated and
    # the engine inputs are owned below PyO3; the module itself loads on
    # the first request routed to the GPU.  Reserve the process-wide
    # delivery now, so the single-delivery guard holds through the window
    # before that open; ``PreparedNativeGpu.publish_loaded`` records the
    # loaded fact once the open is observed.  Neither step imports torch
    # or the legacy host.
    from .loader import KernelLoader

    KernelLoader.reserve_native_prebuilt(
        manifest=manifest,
        fatbin_digest=expected,
        architecture=architecture,
    )
    return PreparedNativeGpu(
        engine=engine,
        manifest=dict(manifest),
        fatbin_digest=expected,
        architecture=architecture,
    )
