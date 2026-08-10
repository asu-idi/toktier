"""The GPU engine facade: one loader, one family table, one set of stores.

This is the only place in the package that asks the loader for a kernel
build, and the only place that constructs encoders for families. Both
facts are contract, not tidiness:

- ``docs/contracts/registry.md`` Section 3.2: a ``certified_source``
  certificate covers exactly one kernel build configuration per process.
  Routing every construction through one facade is what makes "one
  loader, one flag set" checkable rather than hoped for.
- ``docs/contracts/registry.md`` Section 3.3: the registry is the only
  data source for family-to-kernel mappings. Dispatch below reads the
  band's declared entry points out of the routing data and resolves them
  through :mod:`toktier.engine.gpu.entry_points`; no band or family name
  appears in this module.

Trust boundary: the engine consumes **verified artifact handles**
(``toktier.backends.protocol.ArtifactHandle``), produced by the artifact
layer after per-file sha256 verification (see
:mod:`toktier.engine.gpu.handles`). It never reads a manifest of its own
and never accepts a bare directory.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...errors import ArtifactNotFound, UncertifiedTokenizer, UnsupportedConfig
from ...kernels.bpe_tables import BpeTableStore
from ...policy import BACKEND_GPU
from .backend import GpuBackend
from .batched import BatchedE2E
from .certify import family_certification, oracle_binding
from .class_tables import ClassTableStore
from .entry_points import encoder_deliveries, pretok_class
from .families import KernelFamilyTable
from .loader import DEFAULT_BUILD_FLAGS, BuildFlags, KernelLoader
from .options import DEFAULT_GPU_OPTIONS, GpuOptions

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...artifacts.store import ArtifactStore
    from ...backends.protocol import ArtifactHandle
    from ...config import Config
    from ...routing.registry_view import ArtifactRecord, RegistryView
    from .encoder import AddedTokenFrontend, GpuTokenizer

__all__ = ["ENCODER_KINDS", "GpuBackend", "GpuEngine", "oracle_binding"]

#: Encoder delivery forms. ``eager`` runs the chain stage by stage;
#: ``fused`` runs it in one call; ``graph`` additionally captures that
#: call per size bucket. All three produce the same ids.
ENCODER_KINDS = ("eager", "fused", "graph")


@dataclass(frozen=True)
class GpuEngine:
    """Everything a process needs to build GPU encoders."""

    ext: Any
    families: KernelFamilyTable
    class_tables: ClassTableStore
    bpe_tables: BpeTableStore
    artifacts: Mapping[str, ArtifactHandle]
    options: GpuOptions = DEFAULT_GPU_OPTIONS
    build_flags: BuildFlags = DEFAULT_BUILD_FLAGS

    # -- construction ---------------------------------------------------

    @classmethod
    def create(
        cls,
        artifacts: Mapping[str, ArtifactHandle],
        *,
        config: Config | None = None,
        cache_dir: Path | None = None,
        options: GpuOptions = DEFAULT_GPU_OPTIONS,
        build_flags: BuildFlags = DEFAULT_BUILD_FLAGS,
        family_table_path: Path | None = None,
        class_table_dir: Path | None = None,
        delivery: str = "auto",
    ) -> GpuEngine:
        """Load the kernel once and build the shared stores.

        Args:
            artifacts: family name to a verified artifact handle, as the
                artifact layer resolves them (``ArtifactStore`` plus
                :func:`toktier.engine.gpu.handles.verified_handle`). A
                handle exists only after every manifest-listed file
                matched its recorded sha256.
            config: resolved configuration, used for the cache directory.
            cache_dir: overrides the configuration's cache directory.
            options: GPU tuning options.
            build_flags: the certificate-bound kernel build flags.
            family_table_path: routing data location, for tests and for
                consuming a freshly regenerated table.
            class_table_dir: extra directory searched for generated
                lookup tables, ahead of the packaged ones.
            delivery: kernel delivery selector handed to the loader
                (``auto`` / ``prebuilt`` / ``jit``; see
                ``KernelLoader.get``).
        """
        if cache_dir is None:
            if config is None:
                from ...config import Config

                config = Config.resolve()
            cache_dir = Path(config.cache_dir)
        ext = KernelLoader.get(
            cache_dir=cache_dir,
            flags=build_flags,
            device=options.device,
            delivery=delivery,
        )
        families = KernelFamilyTable.load(family_table_path)
        return cls(
            ext=ext,
            families=families,
            class_tables=ClassTableStore(
                families, table_dir=class_table_dir, cache_dir=cache_dir
            ),
            bpe_tables=BpeTableStore(
                artifacts, cache_dir, shared_model=families.shared_model_map()
            ),
            artifacts=artifacts,
            options=options,
            build_flags=build_flags,
        )

    @classmethod
    def from_store(
        cls,
        store: ArtifactStore,
        families: Iterable[str],
        **kwargs: Any,
    ) -> GpuEngine:
        """Resolve handles through the artifact store, then create.

        Convenience over :meth:`create` for callers that hold a manifest
        and a store rather than pre-resolved handles. Every family is
        fetched (when a source allows it) and hash-verified before the
        engine sees it.
        """
        from .handles import verified_handles

        return cls.create(verified_handles(store, families), **kwargs)

    # -- artifact access ------------------------------------------------

    def _artifact(self, family: str) -> ArtifactHandle:
        handle = self.artifacts.get(family)
        if handle is None:
            raise ArtifactNotFound(
                f"no verified artifact handle for family {family!r}",
                details={
                    "family": family,
                    "searched": sorted(self.artifacts),
                },
            )
        return handle

    # -- encoders -------------------------------------------------------

    def encoder(
        self,
        family: str,
        *,
        kind: str = "fused",
        frontend: AddedTokenFrontend | None = None,
        options: GpuOptions | None = None,
    ) -> GpuTokenizer:
        """Build an end-to-end encoder for one family.

        Raises:
            UncertifiedTokenizer: the family is not in the routing data,
                or its band has no end-to-end encoder (the split-only
                band). Under the default policy the caller turns this
                into a reference fallback, not a failure.
            UnsupportedConfig: unknown encoder kind.
        """
        if kind not in ENCODER_KINDS:
            raise UnsupportedConfig(
                f"unknown encoder kind {kind!r}",
                details={
                    "option": "kind",
                    "value": kind,
                    "reason": f"expected one of {ENCODER_KINDS}",
                },
            )
        entry = self.families.get(family)
        band = self.families.band_spec(entry.band)
        if not band.e2e or band.encoder is None:
            raise UncertifiedTokenizer(
                f"family {family!r} has a split-layer kernel only; "
                "end-to-end encoding runs on the reference backend",
                details={"family": family, "artifact_sha256": None},
            )
        resolved = options or self.options
        if kind == "fused":
            resolved = resolved.replace(use_cuda_graph=False)
        elif kind == "graph":
            resolved = resolved.replace(use_cuda_graph=True)
        encoder_class = encoder_deliveries(band.encoder)[kind]
        encoder: GpuTokenizer = encoder_class(
            ext=self.ext,
            family=entry,
            artifact_dir=self._artifact(family).root,
            bpe=self.bpe_tables.load(family),
            class_tables=self.class_tables,
            options=resolved,
            frontend=frontend,
        )
        return encoder

    def pretok(
        self, family: str, *, options: GpuOptions | None = None
    ) -> Any:
        """Build only the piece-start (split) layer for one family.

        Every certified band has a split layer, including the one with no
        end-to-end encoder. Exposing it separately is what lets that band
        be used for what it is certified for, without implying it can do
        more.
        """
        from .encoder import GpuTokenizer

        entry = self.families.get(family)
        band = self.families.band_spec(entry.band)
        resolved = options or self.options
        table = self.class_tables.load(entry.class_table)
        digits_max = GpuTokenizer.resolve_digits_max(entry, table)
        return pretok_class(band.pretok).from_family(
            self.ext,
            table,
            family=entry,
            digits_max=digits_max,
            options=resolved,
        )

    def batched(
        self,
        family: str,
        *,
        kind: str = "eager",
        options: GpuOptions | None = None,
    ) -> BatchedE2E:
        """A batched channel for one family.

        The batched channel drives the eager entry points by design: it
        already amortises the launch cost over the batch, and the fused
        entry is a single-request delivery form.
        """
        entry = self.families.get(family)
        band = self.families.band_spec(entry.band)
        resolved = options or self.options
        return BatchedE2E(
            self.encoder(family, kind=kind, options=resolved),
            options=resolved,
            windowed_starts=band.windowed_starts,
        )

    def backend(
        self,
        family: str,
        *,
        kind: str = "fused",
        frontend: AddedTokenFrontend | None = None,
        options: GpuOptions | None = None,
    ) -> GpuBackend:
        """The executor-facing backend for one family.

        Owns a single-request encoder of the given delivery ``kind``
        plus the family's batched channel, behind the one protocol the
        routing executor runs against
        (``toktier.backends.protocol.Backend``).
        """
        return GpuBackend(
            self.encoder(family, kind=kind, frontend=frontend, options=options),
            batched=self.batched(family, options=options),
        )

    # -- certificate support --------------------------------------------

    def binding_set(self) -> dict[str, Any]:
        """The values a ``certified_source`` record for this process binds.

        Includes the loader's single-flag-set status: once a second,
        divergent build has been requested in a process, the certificate
        is void and this reports it rather than hiding it. The family
        routing data is bound by content digest, not by path: a path
        stays the same while its bytes drift, and a drifted routing
        table could select a kernel the certificate never covered.

        The installed oracle is part of the report
        (``docs/contracts/registry.md`` Section 2): every certification
        reading was taken against a specific oracle version, so a
        binding set that omitted the installed version could present a
        judged binary identity for a process whose reference behavior
        the judgment never covered. ``oracle`` carries the installed
        version and the certified set of the records covering this
        engine's artifacts; when the installed version falls outside
        that set, ``uncertified_oracle`` is ``True`` and the certificate
        does not attach to this process.
        """
        binding = KernelLoader.binding_set(
            class_table_digest=self.class_tables.binding_digest(),
            family_table_digest=self.families.content_sha256,
        )
        binding["observed_class_table_digests"] = (
            self.class_tables.observed_digests()
        )
        binding["family_table_path"] = (
            str(self.families.source) if self.families.source else None
        )
        binding["families"] = list(self.families.names())
        registry, records = self._certification_context()
        oracle = oracle_binding(registry, records)
        binding["oracle"] = oracle
        binding["uncertified_oracle"] = not oracle["in_certified_set"]
        return binding

    def explain(self) -> dict[str, Any]:
        """Delivery, certification and oracle state of this engine.

        The explicit-engine counterpart of the facade's ``explain()``:
        everything reported here describes the process this engine runs
        in -- the kernel delivery actually loaded, the shipped prebuilt
        fact (the same answer ``toktier doctor`` gives), the installed
        oracle against the certified set, and one certification verdict
        per family for the delivery, device architecture and oracle in
        effect. The full binding set is included under ``binding_set``,
        so this report subsumes it. Plain data throughout, safe to log.
        """
        from ...kernels.prebuilt import shipped_prebuilt_facts

        binding = self.binding_set()
        delivery = KernelLoader.delivery()
        prebuilt_available, _digest = shipped_prebuilt_facts()
        architecture = self._device_architecture(binding)
        registry, records = self._certification_context()
        families = {
            family: family_certification(
                registry=registry,
                record=records.get(family),
                delivery=delivery,
                architecture=architecture,
                certificate_void=bool(binding.get("certificate_void")),
                jit_toolchain_satisfied=(
                    KernelLoader.jit_toolchain_satisfied()
                ),
            )
            for family in sorted(self.artifacts)
        }
        return {
            "engine": BACKEND_GPU,
            "kernel_delivery": delivery,
            "prebuilt_available": prebuilt_available,
            "device_architecture": architecture,
            "certificate_void": bool(binding.get("certificate_void")),
            "jit_toolchain_satisfied": (
                KernelLoader.jit_toolchain_satisfied()
                if delivery == "jit"
                else None
            ),
            "oracle": binding["oracle"],
            "uncertified_oracle": binding["uncertified_oracle"],
            "families": families,
            "binding_set": binding,
        }

    def _certification_context(
        self,
    ) -> tuple[RegistryView, dict[str, ArtifactRecord | None]]:
        """Shipped-registry records for this engine's artifacts.

        Read-only: consulting the registry grants nothing, it only makes
        the certification statements reportable. Families whose artifact
        digest carries no record map to ``None`` -- an absence the
        verdict reports as such, never as a pass.
        """
        from ...routing.registry_load import shipped_registry

        registry = shipped_registry()
        records: dict[str, ArtifactRecord | None] = {}
        for family, handle in self.artifacts.items():
            match = registry.certification(
                artifact_sha256=handle.artifact_sha256
            )
            records[family] = match.record if match is not None else None
        return registry, records

    @staticmethod
    def _device_architecture(binding: Mapping[str, Any]) -> str | None:
        """The device architecture this process observed, if any."""
        prebuilt = binding.get("prebuilt")
        if isinstance(prebuilt, Mapping):
            value = prebuilt.get("device_architecture")
            if value:
                return str(value)
        toolchain = binding.get("toolchain_facts")
        if isinstance(toolchain, Mapping):
            value = toolchain.get("device_capability")
            if value:
                return str(value)
        return None
