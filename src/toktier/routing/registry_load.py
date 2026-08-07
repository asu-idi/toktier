"""Load and verify the shipped support registry document.

Contract reference: ``docs/contracts/registry.md`` Sections 6 and 7. The
registry ships inside the package (``toktier.routing.tables``) so that
an installed wheel can report certification statuses -- the delivery and
per-architecture maps in ``explain()`` and the explicit GPU engine's
reports -- without a source checkout.

Verification split, stated so it cannot be assumed larger than it is:
this loader recomputes and checks the **root digest** (the frozen
construction of registry.md Section 6, over the RFC 8785 canonical form)
and relies on :meth:`RegistryView.from_document` for the structural
checks. Full JSON Schema validation needs the ``jsonschema`` package,
which is a test-side dependency; it runs in the repository checks
(``tools/generate_registry.py --check``) and in CI, not here. Any
failure raises :class:`~toktier.errors.RegistryInvalid` rather than
flowing on as an unverified document.

Reading the registry here grants nothing by itself: routing eligibility still
follows every planner check. The facade plans against this digest-verified view,
so the same record drives both execution and reporting.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import cast

from .._jcs import CanonicalizationError, canonical_json
from ..errors import RegistryInvalid
from .registry_view import RegistryView, ValidatedRegistryDocument
from .tables import SUPPORT_REGISTRY

__all__ = ["load_registry", "load_registry_document", "shipped_registry"]

#: Domain tag of the registry root digest (registry.md Section 6).
_REGISTRY_DOMAIN_TAG = b"toktier.registry.v1\x00"


def load_registry_document(path: Path) -> ValidatedRegistryDocument:
    """Read one registry document and verify its root digest.

    The digest is recomputed after *deleting* the ``root_digest`` member
    (never blanking it), exactly as registry.md Section 6 instructs
    verifiers. A missing file, unparseable JSON, or a digest that does
    not match raises ``RegistryInvalid``.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RegistryInvalid(
            f"registry file cannot be read: {path}",
            details={"path": str(path), "failure": str(error)},
        ) from error
    except ValueError as error:
        raise RegistryInvalid(
            f"registry file is not valid JSON: {path}",
            details={"path": str(path), "failure": str(error)},
        ) from error
    if not isinstance(raw, dict):
        raise RegistryInvalid(
            "registry document must be a JSON object",
            details={"path": str(path), "failure": type(raw).__name__},
        )
    recorded = raw.get("root_digest")
    if not isinstance(recorded, str):
        raise RegistryInvalid(
            "registry document carries no root_digest",
            details={"path": str(path), "failure": "root_digest missing"},
        )
    body = {key: value for key, value in raw.items() if key != "root_digest"}
    try:
        payload = canonical_json(body)
    except CanonicalizationError as error:
        raise RegistryInvalid(
            "registry document cannot be canonicalized",
            details={"path": str(path), "failure": str(error)},
        ) from error
    recomputed = "sha256:" + hashlib.sha256(
        _REGISTRY_DOMAIN_TAG + payload
    ).hexdigest()
    if recomputed != recorded:
        raise RegistryInvalid(
            "registry root digest does not match the document",
            details={
                "path": str(path),
                "failure": f"recorded {recorded}, recomputed {recomputed}",
            },
        )
    return cast("ValidatedRegistryDocument", raw)


def load_registry(path: Path) -> RegistryView:
    """Digest-verified :class:`RegistryView` over one registry file."""
    document = load_registry_document(path)
    return RegistryView.from_document(document, path=str(path))


@lru_cache(maxsize=1)
def shipped_registry() -> RegistryView:
    """The support registry shipped with this installation.

    Cached per process: the packaged file is installation data and does
    not change while the process runs. A missing or tampered file raises
    ``RegistryInvalid`` on first use -- a broken installation is stated,
    never silently treated as an empty registry.
    """
    return load_registry(SUPPORT_REGISTRY)
