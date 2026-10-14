"""The loader-face digest the shipped registry records on every artifact.

Contract reference: ``docs/contracts/registry.md`` Section 1
(``loader_face_sha256``: the digest of the loader-face document a
record's readings were taken on) and ``docs/contracts/routing.md``
Section 5.1 (``R_LOADER_FACE_MISMATCH``).

These are shipped-table facts, pinned rather than recomputed, for the
same reason the capability ids beside them are: computing a loader face
needs the pinned loader and the frozen artifact directories, which this
suite deliberately does not read. ``tools/update_loader_face_registry.py``
writes the values under the locked loader,
``tools/check_loader_face_alignment.py --fingerprints`` re-establishes
them against real artifacts on a machine that holds them, and these
assertions keep the shipped file in step with both.

What is pinned here is the 0.3.0 state of the world: every packaged
artifact carries a digest; for the fourteen whose configuration sidecar
contributes no added token the face is the artifact file, so the digest
is the artifact digest; and for ``qwen3_5_08b``, whose face is the
pinned loader's serialization of the live object, the digest is the
value two independent computations under ``transformers`` 4.57.6 and
``tokenizers`` 0.22.2 agreed on (2026-10-13).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "tables" / "support_registry.json"
PACKAGED = (
    ROOT / "src" / "toktier" / "routing" / "tables" / "support_registry.v1.json"
)

#: The face the qwen3_5_08b readings were taken on: the pinned loader's
#: serialization of the live object, which carries the seven
#: configuration-side added tokens the artifact file does not.
QWEN_LOADER_FACE_SHA256 = (
    "a57b5c62cd5f242a06c632b82cc0b4f8239d08cc80fd9347fc1f629be977c6ca"
)


def _shipped() -> dict[str, Any]:
    document = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_every_artifact_record_carries_a_loader_face_digest() -> None:
    document = _shipped()
    assert document["artifacts"]
    for row in document["artifacts"]:
        digest = row.get("loader_face_sha256")
        assert isinstance(digest, str), row["family"]
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")


def test_file_only_faces_record_the_artifact_digest() -> None:
    """Without a configuration-side subset the face is the file itself."""
    document = _shipped()
    for row in document["artifacts"]:
        if "config_added_tokens" in row:
            continue
        assert row["loader_face_sha256"] == row["artifact_sha256"], row["family"]


def test_the_qwen_anchor_records_the_pinned_loaders_face() -> None:
    """The one materialized face differs from its file, and is the judged one."""
    document = _shipped()
    rows = [row for row in document["artifacts"] if row["family"] == "qwen3_5_08b"]
    assert len(rows) == 1
    row = rows[0]
    assert "config_added_tokens" in row
    assert row["loader_face_sha256"] != row["artifact_sha256"]
    assert row["loader_face_sha256"] == QWEN_LOADER_FACE_SHA256


def test_the_packaged_copy_carries_the_same_digests() -> None:
    shipped = _shipped()
    packaged = json.loads(PACKAGED.read_text(encoding="utf-8"))

    def digests(document: dict[str, Any]) -> dict[str, str]:
        rows = document["artifacts"]
        return {row["family"]: row["loader_face_sha256"] for row in rows}

    assert digests(shipped) == digests(packaged)
