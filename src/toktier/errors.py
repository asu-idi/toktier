"""Structured exception hierarchy with stable machine codes.

Contract reference: ``docs/contracts/errors.md``. Every library-domain
exception derives from :class:`ToktierError` and carries a stable
``.code`` identifier plus a machine-readable ``.details`` mapping.
Natural-language messages are for humans and are not a machine
interface; tools must switch on ``.code``.

Codes are append-only: never renamed, reused, or re-meant.

This module is dependency-free (standard library only).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "ERROR_CLASSES_BY_CODE",
    "AliasConflict",
    "ArtifactHashMismatch",
    "ArtifactNotFound",
    "BackendExecutionFault",
    "BackendUnavailable",
    "BundleInvalid",
    "ConfigInvalid",
    "CudaDriverTooOld",
    "KernelIncompatible",
    "OracleVersionUnsupported",
    "RegistryInvalid",
    "SessionRevisionConflict",
    "SessionStateMismatch",
    "StoreCorrupt",
    "StoreFormatUnsupported",
    "ToktierError",
    "UncertifiedTokenizer",
    "UnsupportedConfig",
    "one_line",
]


def one_line(text: str, *, limit: int = 200) -> str:
    """Fold a message from elsewhere into one bounded line.

    The prose command-line report is the single line
    ``error <CODE>: <message>`` (``docs/contracts/errors.md`` Section 4),
    so text that arrives from a library below -- a hub client, the tar
    reader -- is collapsed before it becomes a message. Nothing is lost:
    the caller keeps the text whole in ``details``, where a reader that
    wants it can ask for ``--json``.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


class ToktierError(Exception):
    """Base class for all toktier library-domain errors.

    Attributes:
        code: Stable UPPER_SNAKE identifier from the frozen error code
            table. Subclasses set :attr:`CODE`; instances expose it as
            ``.code``.
        details: Read-only mapping with machine-readable facts
            (expected/observed hashes, paths, remediation hints).
            Unknown keys must be tolerated by consumers.
    """

    #: Stable code for this class; overridden by every concrete subclass.
    CODE: str = "TOKTIER_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self._details: Mapping[str, object] = MappingProxyType(
            dict(details) if details else {}
        )

    @property
    def code(self) -> str:
        """The stable machine code of this error."""
        return type(self).CODE

    @property
    def details(self) -> Mapping[str, object]:
        """Read-only machine-readable payload."""
        return self._details

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(code={self.code!r}, {self.args[0]!r})"


class ArtifactNotFound(ToktierError):
    """The tokenizer artifact cannot be resolved.

    Unknown family, missing local file, offline mode with an empty
    cache, or a configured source that could not deliver the bytes --
    including a failure inside a download client this package does not
    own, which is classified at the fetch boundary so that a caller sees
    a code rather than that client's own exception. Typical details:
    ``family``, ``searched``, ``offline``; for a source that failed,
    also ``cause``, ``cause_message`` and ``remedy``.
    """

    CODE = "ARTIFACT_NOT_FOUND"


class AliasConflict(ToktierError):
    """A cache holds the requested alias with contents that are not it.

    Raised by :func:`toktier.artifacts.import_bundle` when the alias
    directory is already there and the installed tree does not
    authenticate as the bundle being imported: an undeclared or missing
    file, something that is not a regular file, or a byte count or
    digest that differs. An installed tree that does authenticate is not
    an error -- the import is idempotent and returns it. Typical
    details: ``family``, ``searched``, ``path``, ``failure``,
    ``remedy``; for a byte difference also ``expected_sha256``,
    ``observed_sha256`` or ``expected_size``, ``observed_size``.
    """

    CODE = "ALIAS_CONFLICT"


class ArtifactHashMismatch(ToktierError):
    """A fetched or cached artifact failed content-hash verification.

    Online: raised after quarantine and one re-fetch both failed.
    Offline: raised immediately. Typical details: ``expected_sha256``,
    ``observed_sha256``, ``path``, ``remedy``.
    """

    CODE = "ARTIFACT_HASH_MISMATCH"


class UncertifiedTokenizer(ToktierError):
    """REQUIRE_ACCELERATED policy with no eligible certification identity.

    Under CERTIFIED policy this condition is not an error: the plan
    falls back to reference with reason ``R_UNCERTIFIED_ARTIFACT``.
    Typical details: ``artifact_sha256``, ``family``.
    """

    CODE = "UNCERTIFIED_TOKENIZER"


class OracleVersionUnsupported(ToktierError):
    """Reference execution itself is impossible with the installed oracle.

    Raised only when the oracle package is absent or incompatible at
    import level; a mere certification mismatch degrades to
    reference-only with reason ``R_ORACLE_MISMATCH`` instead. Typical
    details: ``package``, ``installed``, ``certified``.
    """

    CODE = "ORACLE_VERSION_UNSUPPORTED"


class BackendUnavailable(ToktierError):
    """A backend demanded by the policy cannot be loaded.

    Typical details: ``backend``, ``missing``.
    """

    CODE = "BACKEND_UNAVAILABLE"


class BackendExecutionFault(ToktierError):
    """A backend failed on an input in a way the routing layer may recover.

    Raised by accelerated backends around expected device and runtime
    failures. The executor re-runs the affected input on the next
    backend in the plan's chain and records ``R_EXEC_FAULT`` for genuine
    failures. Adapters also use ``stage="add_special_tokens"`` internally
    to request a planned reference route; that route is recorded as
    ``R_INPUT_POSTPROCESS_ROUTED`` instead. Any other exception type
    propagates unchanged, because an unexpected error is a defect to
    surface, not a route. Typical details: ``backend``, ``stage``.
    """

    CODE = "BACKEND_EXECUTION_FAULT"


class KernelIncompatible(ToktierError):
    """Kernel constraints failed under a policy that demands the kernel.

    Uncertified device architecture, digest mismatch, or build failure.
    Typical details: ``backend``, ``reason_code``, ``sm``,
    ``expected_digest``, ``observed_digest``.
    """

    CODE = "KERNEL_INCOMPATIBLE"


class CudaDriverTooOld(ToktierError):
    """CUDA driver below the certified minimum, under a demanding policy.

    Typical details: ``installed``, ``required``.
    """

    CODE = "CUDA_DRIVER_TOO_OLD"


class StoreCorrupt(ToktierError):
    """An explicit integrity operation found a store failure.

    Raised by verify/fsck-style APIs only. On the ordinary read path,
    integrity failures degrade to a counted miss instead of raising: we
    prefer a miss over a wrong result. Typical details: ``path``,
    ``record``, ``failure``.
    """

    CODE = "STORE_CORRUPT"


class StoreFormatUnsupported(ToktierError):
    """A store record is well-formed but not decodable by this reader.

    Future format version, unknown mandatory flag bit, or unknown
    witness category. Deliberately distinct from corruption. Typical
    details: ``format_version``, ``flags``, ``witness_category``.
    """

    CODE = "STORE_FORMAT_UNSUPPORTED"


class SessionStateMismatch(ToktierError):
    """Stored session state does not match the current configuration.

    The semantic fingerprint of the stored state differs from the
    fingerprint of the tokenizer/configuration opening it. Typical
    details: ``expected_fingerprint``, ``stored_fingerprint``.
    """

    CODE = "SESSION_STATE_MISMATCH"


class SessionRevisionConflict(ToktierError):
    """Optimistic-concurrency conflict on a session write.

    The write carried an ``expected_revision`` that no longer matches
    the stored revision. Last-writer-wins is not offered. Typical
    details: ``expected_revision``, ``stored_revision``.
    """

    CODE = "SESSION_REVISION_CONFLICT"


class ConfigInvalid(ToktierError):
    """A configuration value cannot be parsed or is out of range.

    Includes strict boolean environment parsing failures and config
    file syntax errors. Typical details: ``field``, ``value``,
    ``source``.
    """

    CODE = "CONFIG_INVALID"


class UnsupportedConfig(ToktierError):
    """A valid option combination is outside the supported envelope.

    Rejected at construction rather than silently ignored (for example
    padding/truncation modes together with sessions). Typical details:
    ``option``, ``value``, ``reason``.
    """

    CODE = "UNSUPPORTED_CONFIG"


class RegistryInvalid(ToktierError):
    """Registry or evidence manifest failed schema or digest checks.

    Typical details: ``path``, ``failure``.
    """

    CODE = "REGISTRY_INVALID"


class BundleInvalid(ToktierError):
    """An air-gap bundle violates the frozen bundle archive format.

    Covers the tar container and the embedded bundle manifest: unreadable
    or truncated archives, unsafe member paths, link members, duplicate
    members, member-set mismatches against the manifest, resource-limit
    violations, and a bundle manifest that fails parsing, schema, or root
    digest verification. Content-hash failures of the artifact files
    themselves raise :class:`ArtifactHashMismatch` instead. Typical
    details: ``path``, ``failure``, ``member``, ``cause``.
    """

    CODE = "BUNDLE_INVALID"


#: Frozen code -> class lookup for machine consumers.
ERROR_CLASSES_BY_CODE: Mapping[str, type[ToktierError]] = MappingProxyType(
    {
        cls.CODE: cls
        for cls in (
            ArtifactNotFound,
            AliasConflict,
            ArtifactHashMismatch,
            UncertifiedTokenizer,
            OracleVersionUnsupported,
            BackendUnavailable,
            BackendExecutionFault,
            KernelIncompatible,
            CudaDriverTooOld,
            StoreCorrupt,
            StoreFormatUnsupported,
            SessionStateMismatch,
            SessionRevisionConflict,
            ConfigInvalid,
            UnsupportedConfig,
            RegistryInvalid,
            BundleInvalid,
        )
    }
)
