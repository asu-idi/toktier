"""Small integrity binding used to recover sealed facade entries.

Store-format v1 deliberately omits the bytes of a certified stable text
prefix.  A caller that presents the old text again can still restore the
native session, but only after those bytes have been bound to the record
that supplied the token stream.  This private sidecar carries that binding
without storing a second copy of the text.

The sidecar is not a portable store record and never decides token IDs.  A
missing, corrupt, or mismatched sidecar costs a cache hit and the facade
performs a normal full encode.  Its full SHA-256 text binding is distinct
from the shorter, derived checkpoint index: the latter proposes candidates;
this object verifies which historical bytes may be paired with a record.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass

from .index import IndexEntry, entry_for, mark_positions

__all__ = ["RecoveryBinding"]

_MAGIC = b"TKFR"
_VERSION = 1
_DIGEST_BYTES = 16
_CHECKSUM_BYTES = 32
_MAX_MARKS = 64
_TEXT_DOMAIN = b"toktier.facade.v1.recovery-text\0"
_STATE_DOMAIN = b"toktier.facade.v1.recovery-state\0"

# magic, version, record hash, text bytes, text digest, endpoint digest,
# number of geometric checkpoint rows
_HEADER = struct.Struct("<4sH32sQ32s16sI")
_MARK = struct.Struct("<Q16s")


def _text_digest(data: bytes) -> bytes:
    return hashlib.sha256(_TEXT_DOMAIN + data).digest()


def _decode_digest(value: str) -> bytes:
    if len(value) != _DIGEST_BYTES * 2:
        raise ValueError("checkpoint digest has the wrong length")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("checkpoint digest is not hexadecimal") from exc
    if len(result) != _DIGEST_BYTES:
        raise ValueError("checkpoint digest has the wrong length")
    return result


@dataclass(frozen=True)
class RecoveryBinding:
    """Record-bound digest and checkpoint row for one full entry text."""

    record_hash: bytes
    text_digest: bytes
    index_entry: IndexEntry

    @classmethod
    def create(cls, data: bytes, record_hash: bytes) -> RecoveryBinding:
        if len(record_hash) != 32:
            raise ValueError("record hash must be 32 bytes")
        return cls(
            record_hash=record_hash,
            text_digest=_text_digest(data),
            index_entry=entry_for(data),
        )

    @classmethod
    def from_material(
        cls,
        *,
        record_hash: bytes,
        text_byte_length: int,
        text_digest: bytes,
        index_entry: IndexEntry,
    ) -> RecoveryBinding:
        """Build TKFR-v1 from native integrity and an existing index row.

        The facade has already computed ``index_entry`` for content lookup;
        the native session has maintained the full-text digest and byte
        length incrementally. Joining those two results avoids scanning the
        historical text again solely to serialize this private sidecar.
        """
        if len(record_hash) != 32:
            raise ValueError("record hash must be 32 bytes")
        if len(text_digest) != 32:
            raise ValueError("text digest must be 32 bytes")
        if text_byte_length < 0 or text_byte_length != index_entry.byte_length:
            raise ValueError("native text length and entry index disagree")
        return cls(
            record_hash=record_hash,
            text_digest=text_digest,
            index_entry=index_entry,
        )

    def matches_checkpoints(self, data: bytes, record_hash: bytes) -> bool:
        """Whether bytes match the record, length, and checkpoint row.

        The recovery-aware native import separately verifies
        :attr:`text_digest` while initializing its incremental SHA state.
        Splitting the gates avoids hashing the full text twice on recovery.
        """
        return (
            len(data) == self.index_entry.byte_length
            and hmac.compare_digest(self.record_hash, record_hash)
            and self.index_entry == entry_for(data)
        )

    def matches(self, data: bytes, record_hash: bytes) -> bool:
        """Whether ``data`` is the text bound to this exact record."""
        return (
            self.matches_checkpoints(data, record_hash)
            and hmac.compare_digest(self.text_digest, _text_digest(data))
        )

    def to_bytes(self) -> bytes:
        row = self.index_entry
        if len(self.record_hash) != 32 or len(self.text_digest) != 32:
            raise ValueError("recovery hashes must be 32 bytes")
        expected_positions = mark_positions(row.byte_length)
        if tuple(position for position, _digest in row.marks) != expected_positions:
            raise ValueError("checkpoint positions are not canonical")
        if len(row.marks) > _MAX_MARKS:
            raise ValueError("too many recovery checkpoints")
        body = bytearray(
            _HEADER.pack(
                _MAGIC,
                _VERSION,
                self.record_hash,
                row.byte_length,
                self.text_digest,
                _decode_digest(row.end_digest),
                len(row.marks),
            )
        )
        for position, digest in row.marks:
            body.extend(_MARK.pack(position, _decode_digest(digest)))
        body.extend(hashlib.sha256(_STATE_DOMAIN + body).digest())
        return bytes(body)

    @classmethod
    def from_bytes(cls, raw: bytes) -> RecoveryBinding:
        if len(raw) < _HEADER.size + _CHECKSUM_BYTES:
            raise ValueError("recovery binding is truncated")
        body, checksum = raw[:-_CHECKSUM_BYTES], raw[-_CHECKSUM_BYTES:]
        expected = hashlib.sha256(_STATE_DOMAIN + body).digest()
        if not hmac.compare_digest(checksum, expected):
            raise ValueError("recovery binding checksum mismatch")
        (
            magic,
            version,
            record_hash,
            byte_length,
            text_digest,
            endpoint,
            mark_count,
        ) = _HEADER.unpack_from(body)
        if magic != _MAGIC:
            raise ValueError("recovery binding has bad magic")
        if version != _VERSION:
            raise ValueError("recovery binding version is unsupported")
        if mark_count > _MAX_MARKS:
            raise ValueError("recovery binding has too many checkpoints")
        expected_size = _HEADER.size + mark_count * _MARK.size
        if len(body) != expected_size:
            raise ValueError("recovery binding size does not close")
        marks: list[tuple[int, str]] = []
        offset = _HEADER.size
        for _ in range(mark_count):
            position, digest = _MARK.unpack_from(body, offset)
            marks.append((position, digest.hex()))
            offset += _MARK.size
        expected_positions = mark_positions(byte_length)
        if tuple(position for position, _digest in marks) != expected_positions:
            raise ValueError("recovery checkpoint positions are not canonical")
        return cls(
            record_hash=record_hash,
            text_digest=text_digest,
            index_entry=IndexEntry(
                byte_length=byte_length,
                end_digest=endpoint.hex(),
                marks=tuple(marks),
            ),
        )
