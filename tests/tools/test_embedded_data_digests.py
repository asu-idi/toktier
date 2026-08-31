"""The crate's hand-pinned embedded-data digests, checked from Python.

``crates/toktier/src/manifest.rs`` refuses to load its embedded data when
a payload does not hash to the constant beside it.  No generator writes
those constants, so before this check the only thing that noticed a data
file moving without its constant was ``cargo test``: 0.2.8 shipped a
`sibling aliases digest mismatch` that survived three green Python
gates, and 0.2.9 met the same blind spot again.  The synchronizing tool
now recomputes them, and these tests hold it to that.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import sync_rust_package_data as sync  # noqa: E402

MANIFEST = sync.MANIFEST_SOURCE.read_text(encoding="utf-8")


def test_the_shipped_constants_match_the_payloads_they_pin() -> None:
    """The tree under test is consistent, and something was checked."""
    assert sync.embedded_digest_problems() == []
    names = sync.hand_pinned_digest_names(MANIFEST)
    assert names, "no hand-pinned digest constant was found to check"
    # The constant that went stale in 0.2.8 and again in 0.2.9 is one of
    # them; the list is read from the source, so a constant added later
    # is covered without this test being edited.
    assert "SIBLING_ALIASES" in names


def test_a_build_script_value_is_not_counted_as_hand_pinned() -> None:
    """The support registry digest is hashed by the build script itself."""
    assert "SUPPORT_REGISTRY_SHA256" in MANIFEST
    assert "SUPPORT_REGISTRY" not in sync.hand_pinned_digest_names(MANIFEST)


def test_a_stale_constant_is_reported() -> None:
    """The blind spot itself: the payload moved, the constant did not."""
    stale = MANIFEST.replace(
        '"2f6524884f84c87bb6a8ac1a85e81ef4ab151e5cec076bebfb8d8aec1a9a760f"',
        f'"{"0" * 64}"',
        1,
    )
    assert stale != MANIFEST
    issues = sync.embedded_digest_problems(stale)
    assert len(issues) == 1
    assert "SIBLING_ALIASES_SHA256 is stale" in issues[0]
    assert "sibling_aliases.v1.json" in issues[0]


def test_a_constant_with_no_payload_beside_it_is_reported() -> None:
    """A digest that pins nothing checkable is a finding, not a pass."""
    orphaned = MANIFEST + (
        '\nconst INVENTED_SHA256: &str =\n    "' + "a" * 64 + '";\n'
    )
    issues = sync.embedded_digest_problems(orphaned)
    assert len(issues) == 1
    assert "INVENTED_SHA256 has no INVENTED_BYTES payload" in issues[0]


def test_a_value_that_is_neither_hex_nor_a_build_script_value_is_reported() -> None:
    malformed = MANIFEST.replace(
        '"2f6524884f84c87bb6a8ac1a85e81ef4ab151e5cec076bebfb8d8aec1a9a760f"',
        '"not a digest"',
        1,
    )
    issues = sync.embedded_digest_problems(malformed)
    assert len(issues) == 1
    assert "SIBLING_ALIASES_SHA256 is neither" in issues[0]


def test_a_source_that_pins_nothing_is_reported() -> None:
    """A check that passes by finding nothing to do is worth nothing."""
    issues = sync.embedded_digest_problems("// no constants here\n")
    assert len(issues) == 1
    assert "no hand-pinned digest constant found" in issues[0]


def test_a_missing_payload_is_reported(tmp_path: Path) -> None:
    """The constants are read from the source, the bytes from the crate."""
    issues = sync.embedded_digest_problems(MANIFEST, crate=tmp_path)
    assert issues
    assert all("embedded payload is missing" in issue for issue in issues)
