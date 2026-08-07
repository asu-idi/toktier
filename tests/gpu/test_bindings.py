"""Host tests for the shared certified-source binding representation.

Contract reference: ``schemas/support_registry.schema.json``
(``backend_entry``) and ``docs/contracts/registry.md`` Section 3. One
representation is produced by the kernel loader and consumed by the
registry parser and the probe; these tests pin the spelling both sides
share, because a spelling drift between producer and consumer is
exactly what once made the binding set unverifiable.
"""

from __future__ import annotations

import pytest

from toktier.errors import RegistryInvalid
from toktier.kernels.bindings import CertifiedSourceBindings, bare_sha256

HEX = "ab" * 32


def test_bare_sha256_accepts_both_spellings() -> None:
    assert bare_sha256(HEX) == HEX
    assert bare_sha256(f"sha256:{HEX}") == HEX


@pytest.mark.parametrize(
    "value",
    ["", "sha256:", "0" * 63, "0" * 65, "G" * 64, "SHA256:" + "0" * 64],
)
def test_bare_sha256_refuses_non_digests(value: str) -> None:
    with pytest.raises(ValueError):
        bare_sha256(value)


def test_from_mapping_reads_the_schema_shape() -> None:
    """The exact member names and value shapes of a registry entry."""
    parsed = CertifiedSourceBindings.from_mapping(
        {
            "status": "certified_source",
            "source_digest": HEX,
            "build_flags": ["-O3"],
            "toolchain": ">=12.0",
            "class_table_digest": "cd" * 32,
            "devices": ["sm_89", "sm_120"],
            "driver_min": "560.0",
        }
    )
    assert parsed == CertifiedSourceBindings(
        source_digest=HEX,
        build_flags=("-O3",),
        toolchain=">=12.0",
        class_table_digest="cd" * 32,
        devices=("sm_89", "sm_120"),
    )


def test_absent_members_stay_unbound() -> None:
    """An absent member means "not bound", which never verifies as a pass."""
    parsed = CertifiedSourceBindings.from_mapping({"status": "experimental"})
    assert parsed == CertifiedSourceBindings()


def test_prefixed_digests_are_normalized_on_input() -> None:
    parsed = CertifiedSourceBindings.from_mapping(
        {"source_digest": f"sha256:{HEX}"}
    )
    assert parsed.source_digest == HEX


@pytest.mark.parametrize(
    "raw",
    [
        {"source_digest": 7},
        {"source_digest": "zz" * 32},
        {"class_table_digest": True},
        {"toolchain": 12},
        {"build_flags": {"cuda_cflags": ["-O3"]}},
    ],
)
def test_impossible_shapes_are_refused(raw: dict[str, object]) -> None:
    """A present-but-impossible member raises rather than flowing on."""
    with pytest.raises(RegistryInvalid):
        CertifiedSourceBindings.from_mapping(raw)


def test_as_mapping_round_trips_and_omits_unbound_members() -> None:
    bindings = CertifiedSourceBindings(
        source_digest=HEX, build_flags=("-O3",), class_table_digest="cd" * 32
    )
    mapping = bindings.as_mapping()
    assert "toolchain" not in mapping
    assert "devices" not in mapping
    assert CertifiedSourceBindings.from_mapping(mapping) == bindings
