"""Read-side decoder for store format v1 records, in pure Python.

Contract reference: ``docs/contracts/store-format-v1.md``. This module
is the importable counterpart of the frozen byte-level contract: it
decodes and verifies a single record without the native store, so a
record can be judged (and its text tail recovered) anywhere the package
imports. It never writes records; writers live in the store crates.

Verification follows the normative decode order of the contract's
Section 5. Failure classification is the explicit-verify split:
structural and integrity failures raise :class:`~toktier.errors.StoreCorrupt`,
well-formed-but-newer records raise
:class:`~toktier.errors.StoreFormatUnsupported`. Callers on a read/lookup
path catch both and treat the record as a miss; we prefer a miss over a
wrong result.

The chain-link fields of one record are verified in isolation here
(genesis rule and ``curr_block_hash`` recomputation); walking a chain of
records is the caller's concern. The semantic fingerprint is exposed as
opaque bytes and never interpreted.

This module is dependency-free (standard library only).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from .errors import StoreCorrupt, StoreFormatUnsupported

__all__ = [
    "FIXED_HEADER_LENGTH",
    "MAGIC",
    "WITNESS_CATEGORIES",
    "RecordView",
    "decode_record",
]

#: Leading magic of every record.
MAGIC = b"TOKTIERS"

#: Size of the fixed header portion.
FIXED_HEADER_LENGTH = 200

#: Upper bound on ``header_length``.
_MAX_HEADER_LENGTH = 4096

#: Witness category values assigned by the format contract (Section 3).
WITNESS_CATEGORIES = frozenset({0x0000, 0x0001, 0x0002, 0x0003})

_ENDIANNESS_MARKER = 0x01
_PAYLOAD_DOMAIN = b"toktier.store.v1.payload\0"
_LINK_DOMAIN = b"toktier.store.v1.link\0"
_RECORD_DOMAIN = b"toktier.store.v1.record\0"

_MAX_FULL_TEXT = 2**40
_MAX_TAIL = 2**31
_MAX_TOKENS = 2**31

#: Fixed header layout: everything after the magic, in offset order.
_HEADER = struct.Struct("<HHIBBHI32sQ32s32sQQQQQ32s")


def _corrupt(failure: str) -> StoreCorrupt:
    return StoreCorrupt(
        f"record failed verification: {failure}",
        details={"failure": failure},
    )


@dataclass(frozen=True)
class RecordView:
    """One decoded and verified format v1 record."""

    format_version: int
    header_length: int
    flags: int
    witness_category: int
    semantic_fingerprint: bytes
    session_revision: int
    prev_block_hash: bytes
    curr_block_hash: bytes
    full_text_byte_length: int
    stable_prefix_byte_length: int
    text_tail_byte_length: int
    token_count: int
    replace_token_offset: int
    payload_checksum: bytes
    #: Token ids of the full core stream, little-endian u32 each.
    ids_bytes: bytes
    #: Raw text tail exactly as stored (verified UTF-8).
    text_tail: str

    @property
    def ids(self) -> list[int]:
        """Token ids decoded from :attr:`ids_bytes`."""
        return [
            int.from_bytes(self.ids_bytes[i : i + 4], "little")
            for i in range(0, len(self.ids_bytes), 4)
        ]


def decode_record(record: bytes) -> RecordView:
    """Decode one record, verifying every contract rule that applies.

    Returns the verified view. Raises :class:`StoreCorrupt` for
    structural or integrity failures and :class:`StoreFormatUnsupported`
    for well-formed records this reader must not interpret (newer
    format version, unknown mandatory flag bit, unknown witness
    category).
    """
    # Step 1: bounds gate.
    if len(record) < FIXED_HEADER_LENGTH:
        raise _corrupt("record shorter than the fixed header")
    if record[0:8] != MAGIC:
        raise _corrupt("bad magic")
    (
        format_version,
        header_length,
        flags,
        endianness,
        reserved0,
        witness_category,
        reserved1,
        fingerprint,
        session_revision,
        prev_block_hash,
        curr_block_hash,
        full_text_len,
        stable_prefix_len,
        tail_len,
        token_count,
        replace_offset,
        payload_checksum,
    ) = _HEADER.unpack_from(record, 8)
    if endianness != _ENDIANNESS_MARKER:
        raise _corrupt("endianness marker mismatch")
    if reserved0 != 0 or reserved1 != 0:
        raise _corrupt("reserved bytes are not zero")
    if format_version != 1:
        raise StoreFormatUnsupported(
            f"record format version {format_version} is newer than this reader",
            details={"format_version": format_version},
        )
    if (
        header_length < FIXED_HEADER_LENGTH
        or header_length > _MAX_HEADER_LENGTH
        or header_length % 8 != 0
        or header_length > len(record)
    ):
        raise _corrupt("header length out of bounds")

    # Step 2: field bounds and the header extension.
    if flags & 0x0000FFFF:
        raise StoreFormatUnsupported(
            "record carries mandatory feature bits unknown to this reader",
            details={"flags": flags},
        )
    if full_text_len > _MAX_FULL_TEXT:
        raise _corrupt("full text length exceeds the format bound")
    if stable_prefix_len > full_text_len:
        raise _corrupt("stable prefix longer than the full text")
    if tail_len > _MAX_TAIL:
        raise _corrupt("text tail length exceeds the format bound")
    if stable_prefix_len + tail_len != full_text_len:
        raise _corrupt("prefix and tail lengths do not close to the full text")
    if token_count > _MAX_TOKENS:
        raise _corrupt("token count exceeds the format bound")
    if replace_offset > token_count:
        raise _corrupt("replace offset exceeds the token count")
    if session_revision == 0 and prev_block_hash != bytes(32):
        raise _corrupt("genesis record with a non-zero predecessor hash")
    _parse_extension(record, header_length)

    # Step 3: size closure.
    if len(record) != header_length + token_count * 4 + tail_len:
        raise _corrupt("record size does not close over header, ids and tail")

    # Step 4: integrity.
    payload = record[header_length:]
    payload_digest = hashlib.sha256(_PAYLOAD_DOMAIN + payload).digest()
    header = bytearray(record[:header_length])
    header[168:200] = bytes(32)
    expected_checksum = hashlib.sha256(
        _RECORD_DOMAIN + bytes(header) + payload_digest
    ).digest()
    if payload_checksum != expected_checksum:
        raise _corrupt("payload checksum mismatch")
    link = hashlib.sha256(
        _LINK_DOMAIN
        + prev_block_hash
        + fingerprint
        + struct.pack(
            "<QQQQQQH",
            session_revision,
            full_text_len,
            stable_prefix_len,
            tail_len,
            token_count,
            replace_offset,
            witness_category,
        )
        + payload_digest
    ).digest()
    if curr_block_hash != link:
        raise _corrupt("chain link hash mismatch")

    # Step 5: semantic checks.
    if witness_category not in WITNESS_CATEGORIES:
        raise StoreFormatUnsupported(
            f"witness category {witness_category} is not assigned",
            details={"witness_category": witness_category},
        )
    if witness_category == 0 and (stable_prefix_len != 0 or replace_offset != 0):
        raise _corrupt("full-reencode record claims a sealed prefix")
    ids_bytes = payload[: token_count * 4]
    try:
        text_tail = payload[token_count * 4 :].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _corrupt("text tail is not valid UTF-8") from exc

    return RecordView(
        format_version=format_version,
        header_length=header_length,
        flags=flags,
        witness_category=witness_category,
        semantic_fingerprint=fingerprint,
        session_revision=session_revision,
        prev_block_hash=prev_block_hash,
        curr_block_hash=curr_block_hash,
        full_text_byte_length=full_text_len,
        stable_prefix_byte_length=stable_prefix_len,
        text_tail_byte_length=tail_len,
        token_count=token_count,
        replace_token_offset=replace_offset,
        payload_checksum=payload_checksum,
        ids_bytes=ids_bytes,
        text_tail=text_tail,
    )


def _parse_extension(record: bytes, header_length: int) -> None:
    """Walk the TLV extension; v1 readers skip every known-optional type."""
    position = FIXED_HEADER_LENGTH
    while position < header_length:
        if header_length - position < 4:
            raise _corrupt("truncated TLV header in the extension")
        length = int.from_bytes(record[position + 2 : position + 4], "little")
        position += 4
        if position + length > header_length:
            raise _corrupt("TLV value extends past the header")
        position += length
