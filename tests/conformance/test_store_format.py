"""Pure-Python golden checks for the frozen v1 session-store header."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import pytest

MAGIC = b"TOKTIERS"
FIXED_HEADER_LENGTH = 200
MAXIMUM_HEADER_LENGTH = 4096
LITTLE_ENDIAN_MARKER = 0x01

FIELD_LAYOUT = (
    ("magic", 0, 8),
    ("format_version", 8, 2),
    ("header_length", 10, 2),
    ("flags", 12, 4),
    ("endianness", 16, 1),
    ("reserved0", 17, 1),
    ("witness_category", 18, 2),
    ("reserved1", 20, 4),
    ("semantic_fingerprint", 24, 32),
    ("session_revision", 56, 8),
    ("prev_block_hash", 64, 32),
    ("curr_block_hash", 96, 32),
    ("full_text_byte_length", 128, 8),
    ("stable_prefix_byte_length", 136, 8),
    ("text_tail_byte_length", 144, 8),
    ("token_count", 152, 8),
    ("replace_token_offset", 160, 8),
    ("payload_checksum", 168, 32),
)
FIELDS = {name: (offset, width) for name, offset, width in FIELD_LAYOUT}

ASSIGNED_WITNESS_CATEGORIES = frozenset({0x0000, 0x0001, 0x0002, 0x0003})
PAYLOAD_DOMAIN = b"toktier.store.v1.payload\0"
LINK_DOMAIN = b"toktier.store.v1.link\0"
RECORD_DOMAIN = b"toktier.store.v1.record\0"


def _put_unsigned(header: bytearray, field_name: str, value: int) -> None:
    offset, width = FIELDS[field_name]
    header[offset : offset + width] = value.to_bytes(width, "little")


def _get_unsigned(record: bytes, field_name: str) -> int:
    offset, width = FIELDS[field_name]
    return int.from_bytes(record[offset : offset + width], "little")


def _build_empty_record(*, flags: int = 0, witness_category: int = 0) -> bytes:
    """Build an otherwise-valid genesis record directly from the tables."""

    header = bytearray(FIXED_HEADER_LENGTH)
    header[0:8] = MAGIC
    _put_unsigned(header, "format_version", 1)
    _put_unsigned(header, "header_length", FIXED_HEADER_LENGTH)
    _put_unsigned(header, "flags", flags)
    _put_unsigned(header, "endianness", LITTLE_ENDIAN_MARKER)
    _put_unsigned(header, "witness_category", witness_category)

    semantic_fingerprint = bytes(range(32))
    fingerprint_offset, fingerprint_width = FIELDS["semantic_fingerprint"]
    header[
        fingerprint_offset : fingerprint_offset + fingerprint_width
    ] = semantic_fingerprint

    payload = b""
    payload_digest = hashlib.sha256(PAYLOAD_DOMAIN + payload).digest()
    link_preimage = b"".join(
        (
            LINK_DOMAIN,
            bytes(32),
            semantic_fingerprint,
            struct.pack("<QQQQQQH", 0, 0, 0, 0, 0, 0, witness_category),
            payload_digest,
        )
    )
    current_hash = hashlib.sha256(link_preimage).digest()
    current_offset, current_width = FIELDS["curr_block_hash"]
    header[current_offset : current_offset + current_width] = current_hash

    checksum = hashlib.sha256(RECORD_DOMAIN + bytes(header) + payload_digest).digest()
    checksum_offset, checksum_width = FIELDS["payload_checksum"]
    header[checksum_offset : checksum_offset + checksum_width] = checksum
    return bytes(header) + payload


def _payload_digest(record: bytes) -> bytes:
    header_length = _get_unsigned(record, "header_length")
    return hashlib.sha256(PAYLOAD_DOMAIN + record[header_length:]).digest()


def _record_checksum_is_valid(record: bytes) -> bool:
    header_length = _get_unsigned(record, "header_length")
    header = bytearray(record[:header_length])
    checksum_offset, checksum_width = FIELDS["payload_checksum"]
    observed = bytes(header[checksum_offset : checksum_offset + checksum_width])
    header[checksum_offset : checksum_offset + checksum_width] = bytes(checksum_width)
    expected = hashlib.sha256(
        RECORD_DOMAIN + bytes(header) + _payload_digest(record)
    ).digest()
    return observed == expected


def _link_hash_is_valid(record: bytes) -> bool:
    values = (
        _get_unsigned(record, "session_revision"),
        _get_unsigned(record, "full_text_byte_length"),
        _get_unsigned(record, "stable_prefix_byte_length"),
        _get_unsigned(record, "text_tail_byte_length"),
        _get_unsigned(record, "token_count"),
        _get_unsigned(record, "replace_token_offset"),
    )
    witness_category = _get_unsigned(record, "witness_category")
    previous_offset, previous_width = FIELDS["prev_block_hash"]
    fingerprint_offset, fingerprint_width = FIELDS["semantic_fingerprint"]
    current_offset, current_width = FIELDS["curr_block_hash"]
    preimage = b"".join(
        (
            LINK_DOMAIN,
            record[previous_offset : previous_offset + previous_width],
            record[fingerprint_offset : fingerprint_offset + fingerprint_width],
            struct.pack("<QQQQQQH", *values, witness_category),
            _payload_digest(record),
        )
    )
    expected = hashlib.sha256(preimage).digest()
    return record[current_offset : current_offset + current_width] == expected


def _header_length_is_valid(record: bytes) -> bool:
    if len(record) < FIXED_HEADER_LENGTH:
        return False
    header_length = _get_unsigned(record, "header_length")
    return (
        FIXED_HEADER_LENGTH <= header_length <= MAXIMUM_HEADER_LENGTH
        and header_length % 8 == 0
        and header_length <= len(record)
    )


def _record_with_encoded_header_length(header_length: int, record_size: int) -> bytes:
    record = bytearray(record_size)
    if record_size >= 12:
        record[10:12] = header_length.to_bytes(2, "little")
    return bytes(record)


def _unsupported_feature(record: bytes) -> str | None:
    flags = _get_unsigned(record, "flags")
    if flags & 0x0000FFFF:
        return "mandatory_flag"
    witness_category = _get_unsigned(record, "witness_category")
    if witness_category not in ASSIGNED_WITNESS_CATEGORIES:
        return "witness_category"
    return None


def _verify_with_public_store_reader(
    record: bytes, installed_package: Any
) -> str | None:
    """Judge one record through the public reader; its error code or None."""

    observed = installed_package.json_output(
        """
import json
import sys

import toktier
import toktier.records

record = bytes.fromhex(sys.argv[1])
try:
    toktier.records.decode_record(record)
except toktier.ToktierError as error:
    print(json.dumps({"code": error.code}))
else:
    print(json.dumps({"code": None}))
""",
        record.hex(),
    )
    assert isinstance(observed, dict)
    code = observed["code"]
    assert code is None or isinstance(code, str)
    return code


UNKNOWN_MANDATORY_FLAG_RECORD = _build_empty_record(flags=0x00000001)
UNKNOWN_WITNESS_CATEGORY_RECORD = _build_empty_record(witness_category=0x0004)


def test_fixed_header_field_offsets_widths_and_total_size() -> None:
    expected_next_offset = 0
    for _, offset, width in FIELD_LAYOUT:
        assert offset == expected_next_offset
        assert width > 0
        expected_next_offset = offset + width

    assert expected_next_offset == FIXED_HEADER_LENGTH
    assert len(_build_empty_record()) == FIXED_HEADER_LENGTH


def test_magic_endianness_and_little_endian_integer_bytes() -> None:
    record = _build_empty_record()

    assert record[0:8] == b"TOKTIERS"
    assert record[8:10] == b"\x01\x00"
    assert record[10:12] == b"\xc8\x00"
    assert record[16] == 0x01


@pytest.mark.parametrize(
    ("header_length", "record_size", "expected"),
    (
        (192, 200, False),
        (199, 200, False),
        (200, 200, True),
        (204, 204, False),
        (208, 207, False),
        (208, 208, True),
        (4096, 4096, True),
        (4096, 4095, False),
        (4104, 4104, False),
    ),
)
def test_header_length_bounds_alignment_and_record_size_rule(
    header_length: int,
    record_size: int,
    expected: bool,
) -> None:
    record = _record_with_encoded_header_length(header_length, record_size)
    assert _get_unsigned(record, "header_length") == header_length
    assert _header_length_is_valid(record) is expected


@pytest.mark.parametrize(
    ("record", "expected_feature"),
    (
        (UNKNOWN_MANDATORY_FLAG_RECORD, "mandatory_flag"),
        (UNKNOWN_WITNESS_CATEGORY_RECORD, "witness_category"),
    ),
    ids=("unknown-mandatory-flag", "unknown-witness-category"),
)
def test_unsupported_feature_fixtures_are_well_formed_golden_records(
    record: bytes,
    expected_feature: str,
) -> None:
    assert len(record) == FIXED_HEADER_LENGTH
    assert record[0:8] == MAGIC
    assert _record_checksum_is_valid(record)
    assert _link_hash_is_valid(record)
    assert _unsupported_feature(record) == expected_feature


def test_reader_rejects_unknown_mandatory_flag(installed_package: Any) -> None:
    assert _get_unsigned(UNKNOWN_MANDATORY_FLAG_RECORD, "flags") & 0x0000FFFF
    assert (
        _verify_with_public_store_reader(
            UNKNOWN_MANDATORY_FLAG_RECORD, installed_package
        )
        == "STORE_FORMAT_UNSUPPORTED"
    )


def test_reader_rejects_unknown_witness_category(
    installed_package: Any,
) -> None:
    witness_category = _get_unsigned(
        UNKNOWN_WITNESS_CATEGORY_RECORD, "witness_category"
    )
    assert witness_category not in ASSIGNED_WITNESS_CATEGORIES
    assert (
        _verify_with_public_store_reader(
            UNKNOWN_WITNESS_CATEGORY_RECORD, installed_package
        )
        == "STORE_FORMAT_UNSUPPORTED"
    )


def test_reader_accepts_the_golden_record_and_rejects_corruption(
    installed_package: Any,
) -> None:
    """Positive control plus the corrupt/unsupported distinction."""

    golden = _build_empty_record()
    assert _verify_with_public_store_reader(golden, installed_package) is None

    tampered = bytearray(golden)
    fingerprint_offset, _ = FIELDS["semantic_fingerprint"]
    tampered[fingerprint_offset] ^= 0xFF
    assert (
        _verify_with_public_store_reader(bytes(tampered), installed_package)
        == "STORE_CORRUPT"
    )
