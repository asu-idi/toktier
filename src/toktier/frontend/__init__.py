"""Input frontends that run ahead of a backend.

The added-token frontend extracts added-token literals from a document
before the pipeline proper runs, so an accelerated backend only sees the
segments between literals. It is part of the certified pipeline design,
not a correctness incident: routing records it with
``R_INPUT_ADDED_TOKEN``.
"""

from __future__ import annotations

from .added import (
    SEG_ID,
    SEG_TOKEN,
    TABLE_VERSION,
    AddedTokenFrontend,
    AddedTokenFrontendProtocol,
    AddedTokenPlan,
    added_frontend_fingerprint,
    assemble,
    load_table,
    table_content_sha256,
)

__all__ = [
    "SEG_ID",
    "SEG_TOKEN",
    "TABLE_VERSION",
    "AddedTokenFrontend",
    "AddedTokenFrontendProtocol",
    "AddedTokenPlan",
    "added_frontend_fingerprint",
    "assemble",
    "load_table",
    "table_content_sha256",
]
