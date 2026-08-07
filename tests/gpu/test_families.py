"""Host tests for family routing data being the single source of truth.

Contract reference: ``docs/contracts/registry.md`` Section 3.3 -- the
registry is the only data source for family-to-kernel mappings, and
runtime code must not carry a second copy of any mapping it expresses.

The failure this prevents is concrete: when the mapping lived in two
places, adding a family meant editing both, and one release added it to
one place only. A drifted copy routes inputs the certificate never
covered, which is worse than not routing them at all.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from toktier.engine.gpu.families import (
    DEFAULT_FAMILY_TABLE_PATH,
    KernelFamilyTable,
)
from toktier.errors import RegistryInvalid, UncertifiedTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "toktier"


@pytest.fixture(scope="module")
def table() -> KernelFamilyTable:
    return KernelFamilyTable.load()


def test_packaged_routing_data_loads(table: KernelFamilyTable) -> None:
    assert table.source == DEFAULT_FAMILY_TABLE_PATH
    assert table.names()
    assert table.bands()


def test_every_family_resolves_completely(table: KernelFamilyTable) -> None:
    for name in table.names():
        entry = table.get(name)
        assert entry.band in table.bands()
        assert entry.ruleset
        assert entry.digits_max is None or entry.digits_max >= 1
        # The class table it names must be described by the same file.
        table.class_table(entry.class_table)
        # And its band must resolve to a typed dispatch entry.
        band = table.band_spec(entry.band)
        assert band.pretok
        assert band.e2e == (band.encoder is not None)


def test_every_declared_entry_point_is_implemented(
    table: KernelFamilyTable,
) -> None:
    """The routing data may only name entry points the code declares.

    Checked against the engine's entry-point registry, which is
    importable without torch: a band naming a missing implementation
    would otherwise only fail on GPU hardware.
    """
    from toktier.engine.gpu.entry_points import (
        ENCODER_ENTRY_POINTS,
        PRETOK_ENTRY_POINTS,
    )

    for name in table.bands():
        band = table.band_spec(name)
        assert band.pretok in PRETOK_ENTRY_POINTS, band
        if band.encoder is not None:
            assert band.encoder in ENCODER_ENTRY_POINTS, band


def test_content_digest_pins_the_exact_bytes(table: KernelFamilyTable) -> None:
    """The binding-set digest is the digest of the bytes on disk.

    Binding the path alone would let the bytes drift while the binding
    set stays green; the content digest is what makes routing drift
    visible to verification.
    """
    import hashlib

    raw = DEFAULT_FAMILY_TABLE_PATH.read_bytes()
    assert table.content_sha256 == hashlib.sha256(raw).hexdigest()
    in_memory = KernelFamilyTable(json.loads(raw.decode("utf-8")))
    assert in_memory.content_sha256 is None  # unknown, never assumed


def test_digits_max_is_only_omitted_with_table_metadata(
    table: KernelFamilyTable,
) -> None:
    """A family may defer digits-max, but only to the class table.

    Deferring is how the three-splitter families avoid a second copy of a
    value their own pattern already carries; it is not a licence to leave
    it unspecified.
    """
    for name in table.names():
        entry = table.get(name)
        if entry.digits_max is None:
            spec = table.class_table(entry.class_table)
            assert spec.generator, name


def test_unknown_family_is_reported_as_uncertified(
    table: KernelFamilyTable,
) -> None:
    assert table.find("not_a_family") is None
    with pytest.raises(UncertifiedTokenizer) as caught:
        table.get("not_a_family")
    assert caught.value.code == "UNCERTIFIED_TOKENIZER"


def test_split_only_band_is_declared_honestly(table: KernelFamilyTable) -> None:
    """A band with no end-to-end encoder must say so in the data.

    One certified band implements the split layer only. If the routing
    data did not carry that, the engine would report the family as
    GPU-certified end to end, which it is not.
    """
    split_only = [name for name in table.names() if not table.supports_e2e(name)]
    assert split_only, "the split-only band should still be described"
    for name in split_only:
        assert table.get(name).note, name


def test_bad_schema_is_rejected() -> None:
    with pytest.raises(RegistryInvalid):
        KernelFamilyTable({"schema": "something.else"})


def test_unknown_band_reference_is_rejected() -> None:
    document = json.loads(
        DEFAULT_FAMILY_TABLE_PATH.read_text(encoding="utf-8")
    )
    name = next(iter(document["families"]))
    document["families"][name]["band"] = "no_such_band"
    with pytest.raises(RegistryInvalid):
        KernelFamilyTable(document)


def test_unknown_class_table_reference_is_rejected() -> None:
    document = json.loads(
        DEFAULT_FAMILY_TABLE_PATH.read_text(encoding="utf-8")
    )
    name = next(iter(document["families"]))
    document["families"][name]["class_table"] = "no_such_table"
    with pytest.raises(RegistryInvalid):
        KernelFamilyTable(document)


def test_band_without_pretok_is_rejected() -> None:
    """Every band has a split layer; data that says otherwise is refused."""
    document = json.loads(
        DEFAULT_FAMILY_TABLE_PATH.read_text(encoding="utf-8")
    )
    band = next(iter(document["bands"]))
    del document["bands"][band]["pretok"]
    with pytest.raises(RegistryInvalid):
        KernelFamilyTable(document)


def test_e2e_band_without_encoder_is_rejected() -> None:
    document = json.loads(
        DEFAULT_FAMILY_TABLE_PATH.read_text(encoding="utf-8")
    )
    band = next(
        name
        for name, spec in document["bands"].items()
        if spec.get("e2e")
    )
    del document["bands"][band]["encoder"]
    with pytest.raises(RegistryInvalid):
        KernelFamilyTable(document)


def test_split_only_band_with_encoder_is_rejected() -> None:
    """A split-only band claiming an encoder is an incoherent record."""
    document = json.loads(
        DEFAULT_FAMILY_TABLE_PATH.read_text(encoding="utf-8")
    )
    band = next(
        name
        for name, spec in document["bands"].items()
        if not spec.get("e2e")
    )
    document["bands"][band]["encoder"] = "encoder"
    with pytest.raises(RegistryInvalid):
        KernelFamilyTable(document)


def test_no_module_carries_a_second_family_table(
    table: KernelFamilyTable,
) -> None:
    """No family name is hardcoded anywhere in the package.

    The routing data is the only place a family is named. This is checked
    by parsing every module and looking for the names as string literals,
    rather than by grepping, so a name split across a concatenation or
    hidden in a comment is not mistaken for a mapping.
    """
    names = set(table.names())
    offenders: list[tuple[str, str]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in names:
                offenders.append((str(path), str(node.value)))
    assert not offenders, offenders


def test_dispatch_carries_no_band_or_family_literals(
    table: KernelFamilyTable,
) -> None:
    """The engine's dispatch names no band and no family.

    The routing data declares which entry point serves which band; the
    code declares which entry points exist. A band or family literal in
    the dispatch would be a second copy of a registry mapping -- exactly
    the drift-prone shape this data file exists to remove. Read from
    disk rather than imported: ``engine.py`` pulls in torch-dependent
    modules, and this property is decidable without them.
    """
    for module in ("engine.py", "entry_points.py"):
        source = (PACKAGE_ROOT / "engine" / "gpu" / module).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert not literals & set(table.bands()), module
        assert not literals & set(table.names()), module
