"""Fixtures for the facade test tier.

The tier is self-contained: it builds a tiny but real byte-level
artifact (one token per byte, no merges), pins it in a manifest, and
pre-places it in an isolated artifact cache, so every test runs offline
against the real oracle and the real native store. With one token per
byte, any difference in the served ids is visible immediately.

Judgments always compare against a from-scratch encode by the oracle
package itself (the ``reference`` fixture), never against another
facade path.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# Thread hygiene must precede any tokenizer import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _var in (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from toktier import Config, Tokenizer, load  # noqa: E402
from toktier.artifacts.manifest import ArtifactManifest  # noqa: E402
from toktier.paths import artifact_cache_dir  # noqa: E402

FAMILY = "tiny_bytes"
REVISION = "deadbeefcafe0000"


def _bytes_to_unicode() -> dict[int, str]:
    printable = (
        list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    )
    mapped = list(printable)
    extra = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            mapped.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, mapped, strict=False)}


def byte_level_document() -> dict[str, Any]:
    """A tiny real artifact: byte-level BPE with one token per byte."""
    alphabet = _bytes_to_unicode()
    byte_level = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": True,
    }
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": byte_level,
        "post_processor": None,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {alphabet[value]: value for value in range(256)},
            "merges": [],
        },
    }


@dataclass
class Rig:
    """One isolated facade environment over the tiny artifact."""

    family: str
    config: Config
    manifest: ArtifactManifest
    artifact_path: Path
    artifact_sha256: str
    base: Path

    def tokenizer(self, **keywords: Any) -> Tokenizer:
        return load(
            self.family,
            config=self.config,
            manifest=self.manifest,
            **keywords,
        )

    def store_path(self, name: str = "store") -> Path:
        return self.base / name


def build_rig(
    base: Path, document: dict[str, Any] | None = None, family: str = FAMILY
) -> Rig:
    raw = json.dumps(document or byte_level_document()).encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    manifest = ArtifactManifest.from_mapping(
        {
            family: {
                "repo_id": f"toktier-tests/{family}",
                "revision": REVISION,
                "files": {"tokenizer.json": {"sha256": sha, "size": len(raw)}},
            }
        },
        source="tests/facade",
    )
    config = Config(home=base / "home", offline=True)
    directory = artifact_cache_dir(config) / manifest.get(family).directory_name
    directory.mkdir(parents=True)
    artifact_path = directory / "tokenizer.json"
    artifact_path.write_bytes(raw)
    return Rig(
        family=family,
        config=config,
        manifest=manifest,
        artifact_path=artifact_path,
        artifact_sha256=sha,
        base=base,
    )


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return build_rig(tmp_path)


@pytest.fixture
def reference(rig: Rig) -> Callable[[str], list[int]]:
    """From-scratch core-stream encode by the oracle package itself."""
    import tokenizers

    handle = tokenizers.Tokenizer.from_file(str(rig.artifact_path))

    def encode(text: str) -> list[int]:
        return [
            int(token_id)
            for token_id in handle.encode(text, add_special_tokens=False).ids
        ]

    return encode


SpanEncode = Callable[[str], "tuple[list[int], list[tuple[int, int]]]"]


@pytest.fixture
def span_reference(rig: Rig) -> SpanEncode:
    """Reference encode with per-token spans, for direct EntryStore use."""
    import tokenizers

    handle = tokenizers.Tokenizer.from_file(str(rig.artifact_path))

    def encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = handle.encode(text, add_special_tokens=False)
        return (
            [int(token_id) for token_id in encoded.ids],
            [(int(a), int(b)) for a, b in encoded.offsets],
        )

    return encode


TEST_FINGERPRINT = bytes(range(32))
