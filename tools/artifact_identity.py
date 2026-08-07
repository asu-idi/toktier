# Compute the capability fingerprints of a tokenizer artifact.
"""Pipeline fingerprints and added-frontend fingerprints.

The two capability fingerprints follow ``docs/contracts/fingerprint.md``
Sections 5 and 6. They are computed here because the generated tables have to
carry them and the library does not implement them yet; when
``toktier.fingerprint`` lands, this module should call it instead, and the
test that pins the values will catch any difference.

The maintainer tooling that generates the shipped tables imports these
functions so that a certification claim always names exact bytes; the tests in
``tests/registry/test_fingerprints.py`` pin the byte layout to the contract
text.
"""

import struct
from typing import Any

from registry_common import (
    ADDED_FRONTEND_DOMAIN_TAG,
    PIPELINE_DOMAIN_TAG,
    GenerationError,
    canonical_json,
    sha256_of_bytes,
)

#: Pipeline sections, in the order the fingerprint contract lists them.
PIPELINE_SECTIONS = ("decoder", "model", "normalizer", "pre_tokenizer")

#: Added-token attributes, in the order the fingerprint contract lists them.
ADDED_TOKEN_FIELDS = (
    "content",
    "id",
    "special",
    "single_word",
    "lstrip",
    "rstrip",
    "normalized",
)


def _encode_string(value: str) -> bytes:
    return b"\x01" + value.encode("utf-8")


def _encode_bool(value: bool) -> bytes:
    return b"\x01\x01" if value else b"\x01\x00"


def _encode_u64(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise GenerationError(f"value does not fit in a u64: {value}")
    return b"\x01" + struct.pack("<Q", value)


def added_frontend_preimage(added_tokens: list[dict[str, Any]]) -> bytes:
    """Encode the added-token table (fingerprint contract Section 6)."""
    parts = [b"\x01", struct.pack("<I", len(added_tokens))]
    for entry in added_tokens:
        for name in ADDED_TOKEN_FIELDS:
            if name not in entry:
                raise GenerationError(
                    f"added token entry is missing the {name!r} attribute: {entry!r}"
                )
            value = entry[name]
            if name == "content":
                if not isinstance(value, str):
                    raise GenerationError(
                        f"added token content is not a string: {value!r}"
                    )
                parts.append(_encode_string(value))
            elif name == "id":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise GenerationError(
                        f"added token id is not an integer: {value!r}"
                    )
                parts.append(_encode_u64(value))
            else:
                if not isinstance(value, bool):
                    raise GenerationError(
                        f"added token {name!r} is not a boolean: {value!r}"
                    )
                parts.append(_encode_bool(value))
    return b"".join(parts)


def pipeline_fingerprint(artifact: dict[str, Any]) -> str:
    """Digest of the core pipeline (fingerprint contract Section 5)."""
    sections = {name: artifact.get(name) for name in PIPELINE_SECTIONS}
    return sha256_of_bytes(PIPELINE_DOMAIN_TAG + canonical_json(sections))


def added_frontend_fingerprint(artifact: dict[str, Any]) -> str:
    """Digest of the added-token frontend surface (registry contract 1.3)."""
    added_tokens = artifact.get("added_tokens") or []
    if not isinstance(added_tokens, list):
        raise GenerationError("added_tokens is not a list")
    preimage = added_frontend_preimage(added_tokens)
    return sha256_of_bytes(ADDED_FRONTEND_DOMAIN_TAG + preimage)
