"""The ``certified_source`` binding set, in the schema's own spelling.

Contract reference: ``docs/contracts/registry.md`` Section 3 and
``schemas/support_registry.schema.json`` (``backend_entry``), which is
normative for the field names and value shapes. A ``certified_source``
record binds the kernel source digest, the build flags, the toolchain
constraint, the judged device list and the class-table digest.

Why this module exists: the producer of these values (the kernel
loader), the registry reader and the routing probe used to spell them
independently -- a prefixed ``kernel_source_digest`` on one side, a bare
``source_digest`` on the other; a structured flag mapping here, a flat
string list there. Two spellings of one binding set cannot be compared,
and a binding set that cannot be compared verifies nothing. This module
is the one spelling; the loader produces it, the registry parser and the
probe consume it.

Dependency note: standard library only, and importing it must never
import torch. The binding set is exactly the part of the certificate a
machine without a GPU still has to be able to compute and check.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import RegistryInvalid

__all__ = ["CertifiedSourceBindings", "bare_sha256"]

_SHA256_HEX = re.compile(r"\A[0-9a-f]{64}\Z")

#: Prefix used by the in-package digest helpers for readability. The
#: registry schema stores the bare hexadecimal form; both spellings are
#: accepted on input and exactly one is stored.
_DIGEST_PREFIX = "sha256:"


def bare_sha256(value: str) -> str:
    """Normalize a digest to the schema spelling: 64 bare hex digits.

    The in-package digest helpers return ``sha256:<hex>``; the registry
    schema (``$defs.sha256_hex``) stores the bare form. Both are accepted
    here; anything else raises ``ValueError`` rather than flowing on as
    an incomparable string.
    """
    digest = value.removeprefix(_DIGEST_PREFIX)
    if not _SHA256_HEX.match(digest):
        raise ValueError(f"not a sha256 digest: {value!r}")
    return digest


@dataclass(frozen=True)
class CertifiedSourceBindings:
    """The values a ``certified_source`` certificate binds.

    Field names and value shapes follow the registry schema exactly, so
    a producer-side instance (built by the kernel loader) and a
    consumer-side instance (parsed from a registry record) are directly
    comparable field by field. ``toolchain`` and ``devices`` are
    constraints only the registry states; a producer leaves them at
    their defaults because it observes facts, not constraints.
    """

    #: Bare-hex digest of the kernel source set.
    source_digest: str | None = None
    #: Bound build flags in the canonical flat encoding (see
    #: ``BuildFlags.as_binding_flags``). The certified default
    #: configuration encodes as ``("-O3",)``.
    build_flags: tuple[str, ...] = ()
    #: Toolchain constraint expression (registry side only).
    toolchain: str | None = None
    #: Bare-hex digest over the generated class tables.
    class_table_digest: str | None = None
    #: Exact GPU architectures judged (registry side only).
    devices: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, source: str = ""
    ) -> CertifiedSourceBindings:
        """Parse the binding fields of one registry backend entry.

        Absent members stay at their defaults (the planner treats an
        unverifiable value as a failed verification, never a pass); a
        member that is present with an impossible shape raises
        ``RegistryInvalid`` rather than flowing on as an incomparable
        value.
        """
        return cls(
            source_digest=_digest_or_none(
                raw.get("source_digest"), field="source_digest", source=source
            ),
            build_flags=_flags(raw.get("build_flags"), source=source),
            toolchain=_str_or_none(
                raw.get("toolchain"), field="toolchain", source=source
            ),
            class_table_digest=_digest_or_none(
                raw.get("class_table_digest"),
                field="class_table_digest",
                source=source,
            ),
            devices=_flags(raw.get("devices"), source=source),
        )

    def as_mapping(self) -> dict[str, Any]:
        """Mapping form in schema spelling; unset members are omitted.

        Omission mirrors the schema, where these members are optional
        per entry: a ``None`` would not validate, and an absent member
        already means "not bound".
        """
        out: dict[str, Any] = {"build_flags": list(self.build_flags)}
        if self.source_digest is not None:
            out["source_digest"] = self.source_digest
        if self.toolchain is not None:
            out["toolchain"] = self.toolchain
        if self.class_table_digest is not None:
            out["class_table_digest"] = self.class_table_digest
        if self.devices:
            out["devices"] = list(self.devices)
        return out


def _digest_or_none(value: Any, *, field: str, source: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return bare_sha256(value)
        except ValueError:
            pass
    raise RegistryInvalid(
        f"registry field {field!r} must be a sha256 hex digest",
        details={"path": source, "failure": f"{field}: {value!r}"},
    )


def _str_or_none(value: Any, *, field: str, source: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise RegistryInvalid(
        f"registry field {field!r} must be a string",
        details={"path": source, "failure": f"{field}: {type(value).__name__}"},
    )


def _flags(value: Any, *, source: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        raise RegistryInvalid(
            "binding flag lists must be flat arrays of strings",
            details={"path": source, "failure": f"mapping: {value!r}"},
        )
    return tuple(str(item) for item in value)
