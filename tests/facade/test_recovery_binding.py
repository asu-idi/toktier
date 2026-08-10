"""Integrity and canonical-shape tests for sealed-entry recovery bindings."""

from __future__ import annotations

import pytest

from toktier.facade.index import IndexEntry
from toktier.facade.recovery import RecoveryBinding


def test_recovery_binding_round_trips_and_matches_only_its_text_and_record() -> None:
    data = ("prefix \u4f60\u597d caf\u00e9\n" * 600).encode()
    record_hash = bytes(range(32))
    binding = RecoveryBinding.create(data, record_hash)
    restored = RecoveryBinding.from_bytes(binding.to_bytes())

    assert restored == binding
    assert restored.matches(data, record_hash)
    assert not restored.matches(data + b"!", record_hash)
    assert not restored.matches(data, bytes(reversed(record_hash)))


def test_material_constructor_is_byte_identical_to_reference_constructor() -> None:
    data = ("incremental \u4f60\u597d \U0001f680\n" * 900).encode()
    record_hash = bytes(reversed(range(32)))
    reference = RecoveryBinding.create(data, record_hash)
    optimized = RecoveryBinding.from_material(
        record_hash=record_hash,
        text_byte_length=len(data),
        text_digest=reference.text_digest,
        index_entry=reference.index_entry,
    )

    assert optimized == reference
    assert optimized.to_bytes() == reference.to_bytes()
    assert optimized.matches_checkpoints(data, record_hash)
    assert not optimized.matches_checkpoints(data + b"!", record_hash)

    with pytest.raises(ValueError, match="length"):
        RecoveryBinding.from_material(
            record_hash=record_hash,
            text_byte_length=len(data) + 1,
            text_digest=reference.text_digest,
            index_entry=reference.index_entry,
        )


def test_recovery_binding_rejects_corruption_and_truncation() -> None:
    raw = bytearray(
        RecoveryBinding.create(b"payload" * 1000, b"r" * 32).to_bytes()
    )
    raw[len(raw) // 2] ^= 0x80
    with pytest.raises(ValueError, match="checksum"):
        RecoveryBinding.from_bytes(bytes(raw))
    with pytest.raises(ValueError, match="truncated"):
        RecoveryBinding.from_bytes(b"short")


def test_recovery_binding_writer_rejects_noncanonical_checkpoint_rows() -> None:
    binding = RecoveryBinding(
        record_hash=b"r" * 32,
        text_digest=b"t" * 32,
        index_entry=IndexEntry(
            byte_length=9000,
            end_digest="00" * 16,
            marks=((123, "11" * 16),),
        ),
    )
    with pytest.raises(ValueError, match="positions"):
        binding.to_bytes()
