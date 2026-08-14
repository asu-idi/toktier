"""Host tests for the BPE table export helpers.

These need no GPU and no torch: the export path is deliberately
torch-free so that a machine without a device can still build and verify
the tables a certificate binds.

Artifacts reach the store as **verified handles**, built here through
the real chain -- ``ArtifactManifest`` (per-file sha256) ->
``ArtifactStore`` with a local source (fetch, hash-verify, install) ->
``verified_handle`` -- so what these tests exercise is the path the
engine trusts, not a weaker test-only shape.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

# numpy ships with the gpu-jit extra; the table export needs it.
pytest.importorskip("numpy", reason="numpy is part of the gpu-jit extra")

from toktier.artifacts.manifest import ArtifactManifest
from toktier.artifacts.sources import LocalDirectorySource
from toktier.artifacts.store import ArtifactStore
from toktier.config import Config
from toktier.engine.gpu.handles import VerifiedHandle, verified_handle
from toktier.errors import ArtifactHashMismatch
from toktier.kernels import bpe_tables
from toktier.policy import RoutingPolicy


def _store_config(tmp_path: Path) -> Config:
    """A configuration whose caches live under the test directory."""
    return Config(
        home=None,
        cache_dir=tmp_path / "toktier-cache",
        state_dir=tmp_path / "toktier-state",
        offline=False,
        log_level="WARNING",
        disable_gpu=True,
        diagnostics=False,
        routing_policy=RoutingPolicy.CERTIFIED,
    )


def _verified(
    tmp_path: Path, family: str, document: Mapping[str, Any]
) -> VerifiedHandle:
    """One artifact through the real manifest-and-verification path."""
    source_dir = tmp_path / "src" / family
    source_dir.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document).encode("utf-8")
    (source_dir / "tokenizer.json").write_bytes(raw)
    manifest = ArtifactManifest.from_mapping(
        {
            family: {
                "repo_id": f"test/{family}",
                "revision": "0" * 40,
                "local_dir": str(source_dir),
                "files": {
                    "tokenizer.json": {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "size": len(raw),
                    }
                },
            }
        }
    )
    store = ArtifactStore(
        manifest, config=_store_config(tmp_path), source=LocalDirectorySource()
    )
    return verified_handle(store, family)


@pytest.mark.parametrize("mask", [0o022, 0o000, 0o077])
def test_every_kernel_cache_directory_is_owner_only(
    tmp_path: Path, mask: int
) -> None:
    """``config.md`` section 5, on the kernel-table half of the cache.

    The export used to reach its cache with a plain
    ``mkdir(parents=True)``, so a fresh deep cache root arrived at the
    process umask.
    """
    handle = _verified(
        tmp_path, "fam", {"model": {"vocab": {"a": 0}, "merges": []}}
    )
    cache_dir = tmp_path / "new" / "levels" / "kernels"
    store = bpe_tables.BpeTableStore({"fam": handle}, cache_dir)

    previous = os.umask(mask)
    try:
        store.export("fam")
    finally:
        os.umask(previous)

    for created in (cache_dir, cache_dir.parent, cache_dir.parent.parent):
        assert stat.S_IMODE(created.stat().st_mode) == 0o700, created


def test_export_module_does_not_import_torch() -> None:
    """The export path must stay usable without the GPU extra."""
    source = Path(bpe_tables.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch." not in source


def test_byte_alphabet_is_a_bijection() -> None:
    mapping = bpe_tables.bytes_to_unicode()
    assert len(mapping) == 256
    assert len(set(mapping.values())) == 256
    for byte, char in mapping.items():
        assert bpe_tables.token_to_raw(char) == bytes([byte])


def test_token_to_raw_round_trips_multi_byte_tokens() -> None:
    mapping = bpe_tables.bytes_to_unicode()
    raw = bytes([0x00, 0x41, 0xC3, 0xA9, 0xFF])
    token = "".join(mapping[byte] for byte in raw)
    assert bpe_tables.token_to_raw(token) == raw


def test_token_to_raw_rejects_undecodable_tokens() -> None:
    with pytest.raises(KeyError):
        bpe_tables.token_to_raw("\u4e00")


def test_fnv1a64_matches_the_published_constants() -> None:
    """The offset basis and prime the kernel uses, checked end to end."""
    assert bpe_tables.fnv1a64(b"") == 0xCBF29CE484222325
    expected = 0xCBF29CE484222325
    for byte in b"abc":
        expected = ((expected ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    assert bpe_tables.fnv1a64(b"abc") == expected


def test_hash_table_is_deterministic_and_under_one_third_full() -> None:
    keys = list(range(1000))
    values = [key * 7 for key in keys]
    first_keys, first_values = bpe_tables._build_hash(keys, values)
    second_keys, second_values = bpe_tables._build_hash(keys, values)
    assert (first_keys == second_keys).all()
    assert (first_values == second_values).all()
    assert first_keys.size >= 3 * len(keys)
    assert first_keys.size & (first_keys.size - 1) == 0  # power of two


def test_hash_table_refuses_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        bpe_tables._build_hash([5, 5], [1, 2])


# -- verified handles ---------------------------------------------------


def test_second_manifest_reader_is_gone() -> None:
    """The GPU lane has no manifest loader of its own.

    Artifact identity has one reader (``toktier.artifacts.manifest``),
    and it rejects entries without per-file digests. A second, weaker
    loader here would let a digest-free mapping reach the kernel tables.
    """
    assert not hasattr(bpe_tables, "load_manifest")
    source = Path(bpe_tables.__file__).read_text(encoding="utf-8")
    assert "local_dir" not in source


def test_tampered_bytes_are_refused_on_read(tmp_path: Path) -> None:
    """Bytes that move after verification do not reach the export.

    The handle was hash-verified when it was produced; the store
    re-checks the one file it opens, so a post-verification rewrite is
    an ``ArtifactHashMismatch``, never a silently different table.
    """
    handle = _verified(
        tmp_path, "fam", {"model": {"vocab": {"a": 0}, "merges": []}}
    )
    store = bpe_tables.BpeTableStore({"fam": handle}, tmp_path / "cache")
    handle.path("tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {"b": 1}, "merges": []}}),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactHashMismatch):
        store.export("fam")


def test_shared_model_claim_is_verified_not_trusted(tmp_path: Path) -> None:
    """A claimed shared model section is checked before it is used."""
    store = bpe_tables.BpeTableStore(
        {
            "left": _verified(
                tmp_path, "left", {"model": {"vocab": {"a": 0}, "merges": []}}
            ),
            "right": _verified(
                tmp_path, "right", {"model": {"vocab": {"b": 0}, "merges": []}}
            ),
        },
        tmp_path / "cache",
        shared_model={"left": "right"},
    )
    with pytest.raises(ValueError, match="no longer share a model section"):
        store.export("left")


def test_unknown_family_is_reported_by_name(tmp_path: Path) -> None:
    store = bpe_tables.BpeTableStore({}, tmp_path)
    with pytest.raises(KeyError, match="missing_family"):
        store.export("missing_family")


# -- the non-monotone merge guard ---------------------------------------


def _store_with_merges(
    tmp_path: Path, merges: list[Any], vocab: dict[str, int]
) -> bpe_tables.BpeTableStore:
    handle = _verified(
        tmp_path, "fam", {"model": {"vocab": vocab, "merges": merges}}
    )
    return bpe_tables.BpeTableStore({"fam": handle}, tmp_path / "cache")


def test_monotone_merge_table_flags_nothing(tmp_path: Path) -> None:
    """A rank-monotone table yields no bitmap, so the kernel path is unchanged."""
    store = _store_with_merges(
        tmp_path,
        [["a", "b"], ["ab", "c"]],
        {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
    )
    assert store.unsafe_ranks("fam") == []
    assert store.unsafe_bits("fam") is None


def test_non_monotone_rule_is_flagged(tmp_path: Path) -> None:
    """A rule whose result an earlier rule consumes must be flagged.

    Rule 0 uses ``xy`` as a component; rule 1 produces it. A batched
    round for rank 1 could therefore create a pair that rank 0 should
    have merged first, so rank 1 degrades to a single leftmost merge.
    """
    store = _store_with_merges(
        tmp_path,
        [["xy", "z"], ["x", "y"]],
        {"x": 0, "y": 1, "z": 2, "xy": 3, "xyz": 4},
    )
    assert store.unsafe_ranks("fam") == [1]
    bits = store.unsafe_bits("fam")
    assert bits is not None
    assert int(bits[0]) == 0b10


def test_merge_string_form_is_accepted(tmp_path: Path) -> None:
    """Both published encodings of a merge rule parse the same way."""
    store = _store_with_merges(
        tmp_path,
        ["xy z", "x y"],
        {"x": 0, "y": 1, "z": 2, "xy": 3, "xyz": 4},
    )
    assert store.unsafe_ranks("fam") == [1]


def test_unsafe_bits_are_cached_beside_the_tables(tmp_path: Path) -> None:
    store = _store_with_merges(
        tmp_path,
        [["xy", "z"], ["x", "y"]],
        {"x": 0, "y": 1, "z": 2, "xy": 3, "xyz": 4},
    )
    first = store.unsafe_bits("fam")
    cached = list((tmp_path / "cache").glob("bpe_unsafe_fam.*.v1.npy"))
    assert len(cached) == 1
    # The cache identity carries the artifact digest, not just the name.
    assert re.fullmatch(r"[0-9a-f]{16}", cached[0].name.split(".")[1])
    second = store.unsafe_bits("fam")
    assert first is not None and second is not None
    assert (first == second).all()


def test_cache_identity_follows_the_artifact_bytes(tmp_path: Path) -> None:
    """Changing the tokenizer data under one family yields new cache files."""
    first = _verified(
        tmp_path, "fam", {"model": {"vocab": {"a": 0}, "merges": []}}
    )
    store = bpe_tables.BpeTableStore({"fam": first}, tmp_path / "cache")
    first_path = store.export("fam")

    second = _verified(
        tmp_path, "fam", {"model": {"vocab": {"b": 0}, "merges": []}}
    )
    other = bpe_tables.BpeTableStore({"fam": second}, tmp_path / "cache")
    second_path = other.export("fam")
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()


def test_cache_with_a_foreign_identity_is_refused(tmp_path: Path) -> None:
    """A cache file renamed under the wrong identity does not load."""
    handle = _verified(
        tmp_path, "fam", {"model": {"vocab": {"a": 0}, "merges": []}}
    )
    donor = _verified(
        tmp_path, "donor", {"model": {"vocab": {"b": 0}, "merges": []}}
    )
    cache_dir = tmp_path / "cache"
    donor_store = bpe_tables.BpeTableStore({"donor": donor}, cache_dir)
    donor_path = donor_store.export("donor")

    store = bpe_tables.BpeTableStore({"fam": handle}, cache_dir)
    expected = store.export("fam")
    expected.unlink()
    donor_path.rename(expected)
    with pytest.raises(ArtifactHashMismatch, match="recorded identity"):
        store.load("fam")
