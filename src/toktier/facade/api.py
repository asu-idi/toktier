"""The facade: load a family once, then use it like a tokenizer.

Contract reference: ``docs/contracts/facade.md`` (0.x surface). The
facade is a thin composition of shipped pieces: artifacts resolve
through the manifest and the verified cache, routing follows the
standing policy semantics (a certified CPU backend is selected only for
an exact artifact-and-engine binding), and the
session/lookup paths run on the entry store, whose streams come from
the same reference oracle. Every path returns ids equal to a
from-scratch reference encode; store layers can only decline to serve,
never answer differently.

Heavy work is deferred: importing this module loads no oracle and no
native store. The oracle loads when the tokenizer is constructed (the
reference backend executes it); the native store loads on the first
session or content-lookup call.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .._oracle import ORACLE_PACKAGE, import_oracle, oracle_version
from ..artifacts import ArtifactManifest, ArtifactStore, HuggingFaceSource
from ..artifacts.tables import ARTIFACT_MANIFEST
from ..backends.fast_cpu import FastCpuBackend
from ..backends.hf import HfBackend
from ..backends.protocol import TOKENIZER_FILE
from ..config import Config
from ..errors import BackendExecutionFault, UnsupportedConfig
from ..policy import (
    BACKEND_FAST_CPU,
    BACKEND_REFERENCE,
    RoutePlan,
    RoutingPolicy,
)
from ..repair.fastokens import FastokensFullRepair
from ..repair.gigatoken import GigatokenRepair, WindowUnsupported
from ..repair.registry import family_spec
from ..routing.execute import RoutedExecutor
from ..routing.explain import build_explanation
from ..routing.plan import assessments_for
from ..routing.plan import plan as build_plan
from ..routing.probe import probe
from ..routing.registry_load import shipped_registry
from ..routing.registry_view import ArtifactRecord
from .store import DEFAULT_CACHE_BUDGET_BYTES, EntryStore

__all__ = ["Encoding", "Tokenizer", "load"]

#: Engine identity bound into the semantic fingerprint. Changes when the
#: session-path engine semantics change; stored sessions then miss and
#: re-encode instead of being reused across meanings.
_ENGINE_ID = "facade-session/v2"

#: Session API semantic version (versioning.md Section 2).
_SESSION_SEMVER = "1"

_LOOKUP_VALUES = (None, "auto", "off")
_REPAIR_BACKENDS = ("auto", "reference", "fastokens")

_SessionRepair = GigatokenRepair | FastokensFullRepair


class _OracleTokenizer(Protocol):
    """The slice of the oracle tokenizer object the facade calls."""

    def encode(self, sequence: str, add_special_tokens: bool = True) -> Any:
        """Encode one sequence; the result carries ``ids`` and ``offsets``."""

    def decode(
        self, ids: Sequence[int], skip_special_tokens: bool = True
    ) -> str:
        """Decode ids back to text."""


@dataclass(frozen=True)
class _ResolvedArtifact:
    """Verified artifact handle over the cache directory."""

    family: str
    root: Path
    artifact_sha256: str
    files: Mapping[str, str]

    def path(self, relative_name: str) -> Path:
        return self.root / relative_name


@dataclass(frozen=True)
class Encoding:
    """Result of one encode. ``ids`` is the token id sequence."""

    ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ids", tuple(self.ids))

    def __len__(self) -> int:
        return len(self.ids)


class Tokenizer:
    """One loaded family behind the facade surface.

    Construction resolves and verifies the artifact, probes the machine
    and fixes an immutable route plan. The optional ``store`` names a
    directory for persistent sessions; without it, session and lookup
    state lives in this process only.
    """

    def __init__(
        self,
        family: str,
        config: Config | None = None,
        *,
        policy: RoutingPolicy | str | None = None,
        store: str | os.PathLike[str] | None = None,
        device: str = "cpu",
        manifest: ArtifactManifest | None = None,
        cache_budget_bytes: int | None = None,
        repair_backend: str = "auto",
    ) -> None:
        if device != "cpu":
            raise UnsupportedConfig(
                f"device {device!r} is not available through the facade; the "
                "GPU engine remains an explicit, separate entry point",
                details={
                    "option": "device",
                    "value": device,
                    "reason": "facade v1 runs on the CPU",
                },
            )
        self._config = config if config is not None else Config.resolve()
        resolved_policy = (
            RoutingPolicy.coerce(policy)
            if policy is not None
            else self._config.routing_policy
        )
        if repair_backend not in _REPAIR_BACKENDS:
            raise ValueError(
                f"repair_backend must be one of {_REPAIR_BACKENDS!r}, "
                f"not {repair_backend!r}"
            )
        if (
            repair_backend == "fastokens"
            and resolved_policy is not RoutingPolicy.EXPERIMENTAL
        ):
            raise UnsupportedConfig(
                "Fastokens is an explicit experimental repair backend and "
                "cannot be selected by a certified policy",
                details={
                    "option": "repair_backend",
                    "value": repair_backend,
                    "required_policy": RoutingPolicy.EXPERIMENTAL.value,
                    "exact_id_guarantee": False,
                },
            )
        self._repair_backend_request = repair_backend
        active_manifest = (
            manifest
            if manifest is not None
            else ArtifactManifest.load(ARTIFACT_MANIFEST)
        )
        entry = active_manifest.get(family)
        verified = ArtifactStore(
            active_manifest, config=self._config, source=HuggingFaceSource()
        ).ensure(family)
        self._artifact_sha256 = entry.file(TOKENIZER_FILE).sha256
        handle = _ResolvedArtifact(
            family=entry.family,
            root=verified.directory,
            artifact_sha256=self._artifact_sha256,
            files={item.name: item.sha256 for item in entry.files},
        )
        self._backend = HfBackend.open(handle)
        self._registry = shipped_registry()
        self._snapshot = probe(
            family=entry.family,
            registry=self._registry,
            artifact_sha256=self._artifact_sha256,
        )
        self._plan = build_plan(
            self._snapshot, resolved_policy, self._registry, self._config
        )
        routed_backends: dict[str, Any] = {BACKEND_REFERENCE: self._backend}
        self._fast_backend: FastCpuBackend | None = None
        if BACKEND_FAST_CPU in self._plan.fallback_chain:
            self._fast_backend = FastCpuBackend.open(handle)
            routed_backends[BACKEND_FAST_CPU] = self._fast_backend
        self._executor = RoutedExecutor(self._plan, routed_backends)
        self._store_directory = (
            Path(os.fspath(store)).expanduser() if store is not None else None
        )
        self._cache_budget = (
            DEFAULT_CACHE_BUDGET_BYTES
            if cache_budget_bytes is None
            else cache_budget_bytes
        )
        self._oracle_handle: _OracleTokenizer | None = None
        self._entry_store: EntryStore | None = None
        self._session_repair: _SessionRepair | None = None
        self._session_repair_initialised = False
        self._session_repair_guard: dict[str, object] | None = None

    # -- identity ------------------------------------------------------

    @property
    def family(self) -> str:
        """Canonical family id this object was loaded for."""
        return self._backend.family

    @property
    def plan(self) -> RoutePlan:
        """The immutable route plan fixed at construction."""
        return self._plan

    # -- encoding ------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        session: str | None = None,
        lookup: str | None = None,
        add_special_tokens: bool = False,
    ) -> Encoding:
        """Encode one document.

        ``session`` names a store entry: when its stored text is a
        prefix of ``text``, only the remainder is appended; otherwise
        the entry is re-encoded whole and overwritten. Without a
        session, content lookup (``lookup="auto"``, the default) asks
        the store whether a stored text is a prefix of this input;
        ``lookup="off"`` skips the store. Every variant returns the ids
        a from-scratch reference encode would return.
        """
        if lookup not in _LOOKUP_VALUES:
            raise ValueError(
                f"lookup must be one of {_LOOKUP_VALUES!r}, not {lookup!r}"
            )
        if session is not None and lookup is not None:
            raise ValueError(
                "a session names the entry directly; lookup does not combine"
            )
        if add_special_tokens and (session is not None or lookup == "auto"):
            raise UnsupportedConfig(
                "store-backed paths hold the core token stream; "
                "add_special_tokens=True cannot be served from them",
                details={
                    "option": "add_special_tokens",
                    "value": True,
                    "reason": "sessions and content lookup store the "
                    "pre-postprocessor core stream",
                },
            )
        ids: list[int] | None = None
        if session is not None:
            ids = self._store().encode_session(session, text)
        elif lookup != "off" and not add_special_tokens:
            ids = self._store().encode_auto(text)
        if ids is None:
            ids = self._executor.encode(
                text, add_special_tokens=add_special_tokens
            )
        return Encoding(ids=tuple(ids))

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = False,
    ) -> list[Encoding]:
        """Encode a batch; row ``i`` equals ``encode(texts[i])``.

        Batches run the plain routed path; per-document content lookup
        belongs to :meth:`encode`.
        """
        rows = self._executor.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )
        return [Encoding(ids=tuple(row)) for row in rows]

    def decode(
        self, ids: Sequence[int], *, skip_special_tokens: bool = True
    ) -> str:
        """Decode token ids back to text through the reference oracle."""
        return self._oracle().decode(
            list(ids), skip_special_tokens=skip_special_tokens
        )

    # -- diagnostics ---------------------------------------------------

    def explain(self) -> dict[str, object]:
        """The active plan, its reasons, and accumulated counters.

        The report is the routing layer's own explanation
        (:func:`toktier.routing.explain.build_explanation`) plus the
        facade keys (``family``, ``store_directory``, ``store``). The
        requested routing policy is reported as ``routing_policy``; the
        ``certification`` block is a separate answer to a separate
        question. The CPU facade plans against the digest-verified shipped
        registry. Device enumeration is not performed on this path: the
        probe summary reports
        ``devices_probed: False`` and the GPU option is recorded as
        ``R_ACCELERATOR_NOT_ADOPTED`` when its modules are importable --
        never as a claim about the machine's hardware.

        "Not adopted" and "not available" are separate statements, and
        the report keeps them separate: the ``kernel_deliveries`` block
        carries the read-only shipped facts (whether a prebuilt fatbin
        and the JIT sources are installed -- the same answer ``toktier
        doctor`` gives) together with the per-delivery,
        per-architecture certification statuses of this artifact's
        record in the shipped support registry. The ``session_repair``
        block separately reports whether store appends use the certified
        corrected-Gigatoken callback or exact HF full re-encoding.
        """
        report = build_explanation(
            route_plan=self._plan,
            snapshot=self._snapshot,
            assessments=assessments_for(
                self._snapshot, self._plan.policy, self._registry, self._config
            ),
            fallback_counts=self._executor.fallback_counts,
            delivery_record=self._shipped_record(),
        )
        report["family"] = self.family
        report["store_directory"] = (
            str(self._store_directory) if self._store_directory else None
        )
        if self._entry_store is not None:
            report["store"] = self._entry_store.stats()
        report["session_repair"] = self._session_repair_report()
        return report

    def close(self) -> None:
        """Release the loaded backend. Idempotent."""
        self._backend.close()
        if self._fast_backend is not None:
            self._fast_backend.close()

    # -- internals -----------------------------------------------------

    def _shipped_record(self) -> ArtifactRecord | None:
        """This artifact's record in the shipped support registry.

        The same digest-verified record drives planning and reporting.
        Returns ``None`` when the artifact has no registry identity.
        """
        match = self._snapshot.certification
        return match.record if match is not None else None

    def _oracle(self) -> _OracleTokenizer:
        if self._oracle_handle is None:
            crate = import_oracle().Tokenizer.from_file(
                str(self._backend.tokenizer_path)
            )
            self._oracle_handle = cast("_OracleTokenizer", crate)
        return self._oracle_handle

    def _reference_encode(
        self, text: str
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Core-stream reference encode with per-token spans."""
        encoded = self._oracle().encode(text, add_special_tokens=False)
        return (
            [int(token_id) for token_id in encoded.ids],
            [(int(a), int(b)) for a, b in encoded.offsets],
        )

    def _store(self) -> EntryStore:
        if self._entry_store is None:
            repair = self._repair_callback()
            self._entry_store = EntryStore(
                fingerprint=self._semantic_fingerprint(repair),
                encode=self._reference_encode,
                append=repair,
                append_stats=repair.stats if repair is not None else None,
                certified_bpe_witness=isinstance(repair, GigatokenRepair),
                directory=self._store_directory,
                cache_budget_bytes=self._cache_budget,
            )
        return self._entry_store

    def _repair_callback(self) -> _SessionRepair | None:
        """Build the certified append callback once; otherwise use HF."""
        if self._session_repair_initialised:
            return self._session_repair
        self._session_repair_initialised = True
        if self._repair_backend_request == "reference":
            return None
        spec = family_spec(self.family, self._artifact_sha256)
        if self._repair_backend_request == "fastokens":
            if spec is None:
                raise UnsupportedConfig(
                    "the experimental Fastokens adapter has no repair-table "
                    f"entry for {self.family!r}",
                    details={
                        "option": "repair_backend",
                        "value": "fastokens",
                        "family": self.family,
                    },
                )
            self._session_repair = FastokensFullRepair.open(
                spec=spec,
                tokenizer_path=self._backend.tokenizer_path,
                hf_tokenizer=cast(Any, self._oracle()),
                reference_encode=self._reference_encode,
            )
            return self._session_repair
        backend = self._fast_backend
        if backend is None or BACKEND_FAST_CPU not in self._plan.fallback_chain:
            return None
        if spec is None:
            self._session_repair_guard = {
                "reason": "artifact_not_in_certified_repair_roster"
            }
            return None
        try:
            engine, hf_tokenizer = backend.repair_components()
            self._session_repair = GigatokenRepair(
                spec=spec,
                engine=engine,
                hf_tokenizer=cast(Any, hf_tokenizer),
                reference_encode=self._reference_encode,
            )
        except (BackendExecutionFault, WindowUnsupported) as error:
            self._session_repair_guard = {
                "reason": "repair_initialisation_guard",
                "error": type(error).__name__,
                "message": str(error),
            }
        return self._session_repair

    def _session_repair_report(self) -> dict[str, object]:
        if self._session_repair is not None:
            return {"status": "active", **self._session_repair.stats()}
        if self._session_repair_guard is not None:
            return {
                "status": "reference_fallback",
                "backend": BACKEND_REFERENCE,
                **self._session_repair_guard,
            }
        if not self._session_repair_initialised:
            requested = self._repair_backend_request
            eligible = (
                "fastokens"
                if requested == "fastokens"
                else (
                    BACKEND_FAST_CPU
                    if BACKEND_FAST_CPU in self._plan.fallback_chain
                    else None
                )
            )
            return {
                "status": "not_initialised" if eligible else "reference_only",
                "eligible_backend": eligible,
            }
        return {"status": "reference_only", "backend": BACKEND_REFERENCE}

    def _semantic_fingerprint(
        self, repair: _SessionRepair | None = None
    ) -> bytes:
        """32-byte identity binding artifact, oracle and engine semantics.

        Any component changing produces a different fingerprint, so
        stored sessions from another meaning miss and re-encode instead
        of being replayed. The preimage is internal to the facade.
        """
        digest = hashlib.sha256(b"toktier.facade.fingerprint.v1\0")
        repair_stats = repair.stats() if repair is not None else {}
        for component in (
            self._artifact_sha256,
            ORACLE_PACKAGE,
            oracle_version() or "",
            _ENGINE_ID,
            _SESSION_SEMVER,
            (
                str(repair_stats.get("backend"))
                if repair is not None
                else BACKEND_REFERENCE
            ),
            repair.config_id if repair is not None else "",
            str(repair_stats.get("engine_version") or ""),
            str(repair_stats.get("engine_digest") or ""),
            self._snapshot.fast_cpu_engine.version or "",
            self._snapshot.fast_cpu_engine.binary_digest or "",
            self._snapshot.fast_cpu_engine.config_digest or "",
        ):
            raw = component.encode("utf-8")
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
        return digest.digest()


def load(
    family: str,
    *,
    store: str | os.PathLike[str] | None = None,
    device: str = "cpu",
    config: Config | None = None,
    policy: RoutingPolicy | str | None = None,
    manifest: ArtifactManifest | None = None,
    cache_budget_bytes: int | None = None,
    repair_backend: str = "auto",
) -> Tokenizer:
    """Load a family and return a ready :class:`Tokenizer`.

    ``store`` names a directory for persistent session state (state, not
    cache: deleting it loses sessions). ``device`` accepts ``"cpu"``;
    the GPU engine remains an explicit, separate entry point. The
    ``repair_backend="fastokens"`` is an explicit experimental full-session
    re-encode path and requires ``policy="experimental"``; it carries no
    exact-ID guarantee. The remaining keywords inject a configuration,
    routing policy, artifact manifest, or in-process cache budget without
    touching process environment.
    """
    return Tokenizer(
        family,
        config,
        policy=policy,
        store=store,
        device=device,
        manifest=manifest,
        cache_budget_bytes=cache_budget_bytes,
        repair_backend=repair_backend,
    )
