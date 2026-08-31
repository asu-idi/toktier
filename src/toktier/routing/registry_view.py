"""Read-only view of the support registry, for routing decisions.

Contract reference: ``docs/contracts/registry.md`` (three identities,
status vocabulary, oracle version policy, default-closed compositions)
and ``schemas/support_registry.schema.json``, which is normative for the
document shape.

Scope boundary: this module *reads* an already-parsed registry document
and answers the questions the planner asks. Loading, JSON Schema
validation, and root-digest verification belong to the registry
subsystem; failures there raise ``RegistryInvalid``. Splitting it this
way keeps the planner a pure function over plain values and lets the
routing tests build registry documents inline.

Single source of truth (registry.md Section 3.3): nothing else in this
package may carry a second copy of a mapping the registry expresses.
That is why there is no constant table of families, kernels, or device
architectures anywhere in the routing layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypedDict

from ..errors import RegistryInvalid
from ..kernels.bindings import CertifiedSourceBindings

__all__ = [
    "STATUS_CERTIFIED",
    "STATUS_CERTIFIED_SOURCE",
    "STATUS_EXPERIMENTAL",
    "STATUS_UNSUPPORTED",
    "ArtifactRecord",
    "BackendEntry",
    "CertificationMatch",
    "OracleRecord",
    "RegistryView",
    "ValidatedRegistryDocument",
    "empty_registry",
]

#: The judged binary itself is bound by digest.
STATUS_CERTIFIED = "certified"

#: Source digest, build flags, toolchain and class-table digest are
#: bound; the local build product is not bit-identical to the judged
#: build. Reported distinctly from ``certified`` everywhere.
STATUS_CERTIFIED_SOURCE = "certified_source"

#: Present but not certified; reachable only under EXPERIMENTAL policy.
STATUS_EXPERIMENTAL = "experimental"

#: Known not to work; never planned, under any policy.
STATUS_UNSUPPORTED = "unsupported"

#: Statuses a certified-class policy may open.
ELIGIBLE_STATUSES = frozenset({STATUS_CERTIFIED, STATUS_CERTIFIED_SOURCE})


class _BackendDocumentOptional(TypedDict, total=False):
    binary_digest: str
    source_digest: str
    class_table_digest: str
    build_flags: list[str]
    toolchain: str
    host_source_digest: str
    host_build_flags: list[str]
    host_toolchain: str
    devices: list[str]
    devices_experimental: list[str]
    architecture_digests: Mapping[str, str]
    driver_min: str
    deliveries: Mapping[str, _BackendDocument]
    engine: str
    engine_version: str
    engine_delivery: str
    engine_module: str
    engine_unicode_data: str
    patch_sha256: str
    config_id: str
    config_digest: str


class _BackendDocument(_BackendDocumentOptional):
    status: str


class _ReadingsDocument(TypedDict):
    docs: int
    bytes: int
    mismatches: int


class _ArtifactDocumentOptional(TypedDict, total=False):
    aliases: list[str]
    config_added_tokens: Mapping[str, object]
    carryover: Mapping[str, object]


class _ArtifactDocument(_ArtifactDocumentOptional):
    artifact_sha256: str
    family: str
    pipeline_id: str
    added_frontend_id: str
    oracle_id: str
    suite_version: str
    evidence_id: str
    readings: _ReadingsDocument
    backends: Mapping[str, _BackendDocument]


class _OracleDocument(TypedDict):
    oracle_id: str
    package: str
    certified_versions: list[str]
    semantic_id: str


class _PipelineDocument(TypedDict):
    pipeline_id: str
    pipeline_fingerprint: str


class _AddedFrontendDocument(TypedDict):
    added_frontend_id: str
    added_frontend_fingerprint: str


class _CompositionDocument(TypedDict):
    pipeline_id: str
    added_frontend_id: str
    evidence_id: str


class ValidatedRegistryDocument(TypedDict):
    """Schema-v1 registry after validation and digest verification."""

    schema_version: int
    generated_by: Mapping[str, str]
    root_digest: str
    oracles: list[_OracleDocument]
    pipelines: list[_PipelineDocument]
    added_frontends: list[_AddedFrontendDocument]
    artifacts: list[_ArtifactDocument]
    compositions: list[_CompositionDocument]


@dataclass(frozen=True)
class BackendEntry:
    """Per-backend certification entry of one artifact record.

    ``deliveries`` refines the entry per kernel delivery mode
    (``jit`` / ``prebuilt``): each value is itself a ``BackendEntry``
    (without further nesting) carrying the binding set of that delivery.
    The top-level fields remain the JIT-era view for readers that do not
    know about deliveries; the planner verifies against the sub-entry of
    the delivery the process actually runs.
    ``devices_experimental`` lists architectures the delivery ships an
    image for without judgment evidence (EXPERIMENTAL-only, honestly
    labeled); ``architecture_digests`` binds each embedded image.
    """

    status: str
    binary_digest: str | None = None
    source_digest: str | None = None
    class_table_digest: str | None = None
    build_flags: tuple[str, ...] = ()
    toolchain: str | None = None
    #: Source/build identity of the Rust request host paired with a
    #: ``certified`` prebuilt GPU binary. These fields are separate from the
    #: CUDA JIT source binding above.
    host_source_digest: str | None = None
    host_build_flags: tuple[str, ...] = ()
    host_toolchain: str | None = None
    devices: tuple[str, ...] = ()
    devices_experimental: tuple[str, ...] = ()
    architecture_digests: Mapping[str, str] = field(default_factory=dict)
    driver_min: str | None = None
    deliveries: Mapping[str, BackendEntry] = field(default_factory=dict)
    #: External-engine binding fields.  ``engine_version`` and
    #: ``binary_digest`` are observed at runtime; the remaining fields are
    #: provenance transitively bound by those bytes.
    engine: str | None = None
    engine_version: str | None = None
    engine_delivery: str | None = None
    engine_module: str | None = None
    engine_unicode_data: str | None = None
    patch_sha256: str | None = None
    config_id: str | None = None
    config_digest: str | None = None

    def architecture_statuses(self) -> dict[str, str]:
        """Per-architecture status labels of this entry.

        Judged architectures (``devices``) carry the entry's own status;
        architectures shipped without judgment evidence
        (``devices_experimental``) are labeled ``experimental``. The
        honesty distinction is contract, not presentation (registry.md
        Section 3).
        """
        architectures: dict[str, str] = {}
        for device in self.devices:
            architectures[device] = self.status
        for device in self.devices_experimental:
            architectures.setdefault(device, STATUS_EXPERIMENTAL)
        return architectures

    def for_delivery(self, delivery: str | None) -> BackendEntry:
        """The entry for one selected delivery, or this compatibility view.

        Delivery-refined GPU records keep the historical JIT view at the
        backend level. Callers that know which delivery will run must follow
        that delivery's row so a prebuilt binary is not labeled with the JIT
        row beside it. Records without a matching refinement retain their
        backend-level meaning.
        """
        if delivery is None:
            return self
        return self.deliveries.get(delivery, self)


@dataclass(frozen=True)
class OracleRecord:
    """One oracle behavior-equivalence class."""

    oracle_id: str
    package: str
    certified_versions: tuple[str, ...]
    semantic_id: str


@dataclass(frozen=True)
class ArtifactRecord:
    """One artifact's certification record."""

    artifact_sha256: str
    family: str
    pipeline_id: str
    added_frontend_id: str
    oracle_id: str
    suite_version: str
    evidence_id: str
    backends: Mapping[str, BackendEntry]
    aliases: tuple[str, ...] = ()
    docs: int = 0
    bytes_judged: int = 0
    mismatches: int = 0
    #: Declared configuration-side added-token subset of this artifact
    #: (``sha256``, ``count``, ``source``), present only when the loader
    #: face carries added tokens beyond the artifact file. The loading
    #: paths verify the subset they observe against this claim.
    config_added_tokens: Mapping[str, object] | None = None
    #: Corpus-equivalence carry-over annotation: how readings taken under
    #: an earlier judge definition remain valid under the current one,
    #: naming the absence certificate they rest on
    #: (``docs/contracts/evidence-carryover.md`` Section 3).
    carryover: Mapping[str, object] | None = None


@dataclass(frozen=True)
class CertificationMatch:
    """How an artifact obtained a certification identity.

    ``identity`` is ``"exact"`` when the artifact's own sha256 carries a
    record, and ``"composition"`` when eligibility came from a certified
    pipeline capability plus a certified added-frontend capability with
    an explicit composition grant (registry.md Section 1.1).
    """

    record: ArtifactRecord
    identity: str


def _backend_entry(
    raw: _BackendDocument, *, nested: bool = False
) -> BackendEntry:
    # The certified_source binding fields are parsed by the one shared
    # representation the loader also produces, so a record and a
    # process's own binding set are comparable field by field.
    bindings = CertifiedSourceBindings.from_mapping(raw)
    deliveries: dict[str, BackendEntry] = {}
    if not nested:
        for name, sub in (raw.get("deliveries") or {}).items():
            deliveries[str(name)] = _backend_entry(sub, nested=True)
    return BackendEntry(
        status=raw["status"],
        binary_digest=raw.get("binary_digest"),
        source_digest=bindings.source_digest,
        class_table_digest=bindings.class_table_digest,
        build_flags=bindings.build_flags,
        toolchain=bindings.toolchain,
        host_source_digest=raw.get("host_source_digest"),
        host_build_flags=tuple(str(v) for v in raw.get("host_build_flags") or ()),
        host_toolchain=raw.get("host_toolchain"),
        devices=bindings.devices,
        devices_experimental=tuple(
            str(v) for v in raw.get("devices_experimental") or ()
        ),
        architecture_digests={
            str(k): str(v)
            for k, v in (raw.get("architecture_digests") or {}).items()
        },
        driver_min=raw.get("driver_min"),
        deliveries=deliveries,
        engine=raw.get("engine"),
        engine_version=raw.get("engine_version"),
        engine_delivery=raw.get("engine_delivery"),
        engine_module=raw.get("engine_module"),
        engine_unicode_data=raw.get("engine_unicode_data"),
        patch_sha256=raw.get("patch_sha256"),
        config_id=raw.get("config_id"),
        config_digest=raw.get("config_digest"),
    )


class RegistryView:
    """Immutable, read-only lookups over one registry document."""

    def __init__(
        self,
        *,
        artifacts: Sequence[ArtifactRecord] = (),
        oracles: Sequence[OracleRecord] = (),
        pipelines: Mapping[str, str] | None = None,
        added_frontends: Mapping[str, str] | None = None,
        compositions: Sequence[tuple[str, str]] = (),
    ) -> None:
        self._by_sha = {record.artifact_sha256: record for record in artifacts}
        self._oracles = {record.oracle_id: record for record in oracles}
        # fingerprint -> id
        self._pipelines = dict(pipelines or {})
        self._added_frontends = dict(added_frontends or {})
        self._compositions = frozenset(compositions)
        self._by_capability: dict[tuple[str, str], ArtifactRecord] = {}
        for record in artifacts:
            key = (record.pipeline_id, record.added_frontend_id)
            self._by_capability.setdefault(key, record)

    # -- construction ---------------------------------------------------

    @classmethod
    def from_document(
        cls, document: ValidatedRegistryDocument, *, path: str = ""
    ) -> RegistryView:
        """Build a view from a validated registry document (schema v1).

        Schema and digest checks belong to the registry loader. This method
        translates validated fields without coercing or revalidating them.
        A missing schema-required field raises ``RegistryInvalid`` because
        it violates the validated-document boundary.
        """
        try:
            artifacts: list[ArtifactRecord] = []
            for raw in document["artifacts"]:
                readings = raw["readings"]
                backends = {
                    backend_id: _backend_entry(entry)
                    for backend_id, entry in raw["backends"].items()
                }
                artifacts.append(
                    ArtifactRecord(
                        artifact_sha256=raw["artifact_sha256"],
                        family=raw["family"],
                        pipeline_id=raw["pipeline_id"],
                        added_frontend_id=raw["added_frontend_id"],
                        oracle_id=raw["oracle_id"],
                        suite_version=raw["suite_version"],
                        evidence_id=raw["evidence_id"],
                        backends=backends,
                        aliases=tuple(raw.get("aliases", ())),
                        docs=readings["docs"],
                        bytes_judged=readings["bytes"],
                        mismatches=readings["mismatches"],
                        config_added_tokens=raw.get("config_added_tokens"),
                        carryover=raw.get("carryover"),
                    )
                )
            oracles = [
                OracleRecord(
                    oracle_id=raw["oracle_id"],
                    package=raw["package"],
                    certified_versions=tuple(raw["certified_versions"]),
                    semantic_id=raw["semantic_id"],
                )
                for raw in document["oracles"]
            ]
            pipelines = {
                raw["pipeline_fingerprint"]: raw["pipeline_id"]
                for raw in document["pipelines"]
            }
            added_frontends = {
                raw["added_frontend_fingerprint"]: raw["added_frontend_id"]
                for raw in document["added_frontends"]
            }
            compositions = [
                (raw["pipeline_id"], raw["added_frontend_id"])
                for raw in document["compositions"]
            ]
        except KeyError as error:
            field = str(error.args[0]) if error.args else "<unknown>"
            raise RegistryInvalid(
                f"registry is missing required field {field!r}",
                details={
                    "path": path,
                    "failure": f"missing required field: {field}",
                },
            ) from error

        return cls(
            artifacts=artifacts,
            oracles=oracles,
            pipelines=pipelines,
            added_frontends=added_frontends,
            compositions=compositions,
        )

    # -- lookups ---------------------------------------------------------

    def oracle(self, oracle_id: str) -> OracleRecord | None:
        """Oracle record by id."""
        return self._oracles.get(oracle_id)

    def pipeline_id(self, fingerprint: str | None) -> str | None:
        """Certified pipeline capability id for a pipeline fingerprint."""
        if fingerprint is None:
            return None
        return self._pipelines.get(fingerprint)

    def added_frontend_id(self, fingerprint: str | None) -> str | None:
        """Certified added-frontend capability id for a fingerprint."""
        if fingerprint is None:
            return None
        return self._added_frontends.get(fingerprint)

    def composition_allows(self, pipeline_id: str, added_frontend_id: str) -> bool:
        """Whether an explicit composition grant covers this pair."""
        return (pipeline_id, added_frontend_id) in self._compositions

    def certification(
        self,
        *,
        artifact_sha256: str | None,
        pipeline_fingerprint: str | None = None,
        added_frontend_fingerprint: str | None = None,
    ) -> CertificationMatch | None:
        """Resolve an artifact to a certification record.

        Order follows registry.md Section 1.1: the exact artifact
        identity first, then a capability composition -- and only when
        an explicit composition entry grants that pair. Absent both, the
        artifact has no certification identity and runs as reference.
        """
        if artifact_sha256 is not None:
            exact = self._by_sha.get(artifact_sha256)
            if exact is not None:
                return CertificationMatch(record=exact, identity="exact")
        pipeline_id = self.pipeline_id(pipeline_fingerprint)
        added_id = self.added_frontend_id(added_frontend_fingerprint)
        if pipeline_id is None or added_id is None:
            return None
        if not self.composition_allows(pipeline_id, added_id):
            return None
        record = self._by_capability.get((pipeline_id, added_id))
        if record is None:
            return None
        return CertificationMatch(record=record, identity="composition")

    def shared_delivery_architecture_statuses(
        self, backend_id: str, delivery: str
    ) -> dict[str, str]:
        """Architecture labels shared by this delivery's artifact records.

        ``doctor`` has no artifact argument, so it may report only delivery
        facts that agree across the shipped records carrying that backend.
        The current GPU delivery rows are generated from one kernel evidence
        set and therefore share this map. Selection and labeling stay on the
        same :class:`BackendEntry` helpers used by ``explain()``.
        """
        entries: list[BackendEntry] = []
        for record in self._by_sha.values():
            entry = record.backends.get(backend_id)
            if entry is not None:
                entries.append(entry.for_delivery(delivery))
        if not entries:
            return {}
        shared = entries[0].architecture_statuses()
        for entry in entries[1:]:
            statuses = entry.architecture_statuses()
            shared = {
                architecture: status
                for architecture, status in shared.items()
                if statuses.get(architecture) == status
            }
        return shared

    def eligible_entries(self, backend_id: str) -> tuple[BackendEntry, ...]:
        """This backend's entries whose status admits an accelerated route.

        ``doctor`` without a family argument reports what an automatic
        request would use on this installation, and for the CPU fast path
        that depends on whether the installed engine is the one a record
        binds. The question has no artifact to hang on, so it is asked of
        the records that carry an eligible entry at all: an engine that
        matches none of them is an engine no family could take the fast
        path with, whichever family is asked for next.
        """
        return tuple(
            entry
            for record in self._by_sha.values()
            if (entry := record.backends.get(backend_id)) is not None
            and entry.status in ELIGIBLE_STATUSES
        )


def empty_registry() -> RegistryView:
    """A registry that certifies nothing (everything runs as reference)."""
    return RegistryView()
