"""The capability ids the shipped registry records on the loader face.

Contract reference: ``docs/contracts/registry.md`` Section 1 (both
capability fingerprints are computed on the loader-face document since
0.2.9) and Section 4.1 (the ``equivalent_loader_face`` sibling basis).

These are shipped-table facts, pinned rather than recomputed: computing
a loader face needs the pinned loader and the frozen artifact
directories, which this suite deliberately does not read (it never
touches a developer cache). The maintainer generator computes the values
under the locked loader, ``tools/check_loader_face_alignment.py``
re-establishes them against real artifacts on a machine that holds them,
and these assertions keep the shipped file in step with both.

What is pinned here is the migration's own claim: the qwen pair holds
one pipeline id and one added-frontend id -- the ids the Qwen3.8 sibling
rows resolve onto -- the deepseek pair still shares a pipeline row, and
no capability id is claimed by two different fingerprints.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "tables" / "support_registry.json"
SIBLINGS = (
    ROOT / "src" / "toktier" / "artifacts" / "tables" / "sibling_aliases.v1.json"
)

#: The loader-face ids of the object the Qwen3.8 repositories and the
#: qwen3_5_08b anchor share (release notes v0.2.9, migration table).
QWEN_PIPELINE_ID = "pipeline.244da91e0b85bac5"
QWEN_ADDED_FRONTEND_ID = "added-frontend.6f80a43b9fe6a15f"

#: The anchor those sibling rows execute.
QWEN_ANCHOR_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)


def _shipped() -> dict[str, Any]:
    document = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _row(document: dict[str, Any], family: str) -> dict[str, Any]:
    rows = [row for row in document["artifacts"] if row["family"] == family]
    assert len(rows) == 1, f"expected exactly one {family} record"
    row = rows[0]
    assert isinstance(row, dict)
    return row


def test_the_qwen_anchor_carries_the_shared_loader_face_ids() -> None:
    row = _row(_shipped(), "qwen3_5_08b")
    assert row["artifact_sha256"] == QWEN_ANCHOR_SHA256
    assert row["pipeline_id"] == QWEN_PIPELINE_ID
    assert row["added_frontend_id"] == QWEN_ADDED_FRONTEND_ID


def test_the_added_frontend_row_describes_the_materialized_table() -> None:
    """33 rows, not the 26 the artifact file carries.

    The seven configuration-side added tokens are part of the certified
    subject, so the fingerprinted surface holds them; the description the
    generator writes is the readable form of that.
    """
    document = _shipped()
    rows = [
        row
        for row in document["added_frontends"]
        if row["added_frontend_id"] == QWEN_ADDED_FRONTEND_ID
    ]
    assert len(rows) == 1
    assert rows[0]["description"] == "Added-token table of qwen3_5_08b (33 entries)."
    assert rows[0]["added_frontend_fingerprint"].startswith("6f80a43b9fe6a15f")


def test_the_loader_face_siblings_resolve_onto_that_object() -> None:
    table = json.loads(SIBLINGS.read_text(encoding="utf-8"))
    rows = [
        row for row in table["aliases"] if row["basis"] == "equivalent_loader_face"
    ]
    assert {row["repo_id"] for row in rows} == {
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.8-Flash-Next",
    }
    for row in rows:
        assert row["canonical_family"] == "qwen3_5_08b"
        assert row["canonical_anchor_sha256"] == QWEN_ANCHOR_SHA256
        assert row["canonical_packaged"] is True
        # One and the same tokenizer file, published under two names.
        assert row["source_sha256"] == (
            "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
        )
        assert row["source_size"] == 12809320


def test_the_deepseek_pair_still_shares_one_pipeline_row() -> None:
    """Grouping survives the move: sharing families keep sharing."""
    document = _shipped()
    shared = {
        _row(document, family)["pipeline_id"]
        for family in ("deepseek_v3", "deepseek_v4_flash")
    }
    assert len(shared) == 1
    members = [
        row
        for row in document["pipelines"]
        if row["pipeline_id"] in shared
    ]
    assert len(members) == 1
    assert "deepseek_v3, deepseek_v4_flash" in members[0]["description"]


def test_every_capability_id_names_exactly_one_fingerprint() -> None:
    """No id collides, and every referenced id has a defining row."""
    document = _shipped()
    pipelines = {
        row["pipeline_id"]: row["pipeline_fingerprint"]
        for row in document["pipelines"]
    }
    frontends = {
        row["added_frontend_id"]: row["added_frontend_fingerprint"]
        for row in document["added_frontends"]
    }
    assert len(pipelines) == len(document["pipelines"])
    assert len(frontends) == len(document["added_frontends"])
    assert len(set(pipelines.values())) == len(pipelines)
    assert len(set(frontends.values())) == len(frontends)
    for identifier, digest in {**pipelines, **frontends}.items():
        assert identifier.split(".", 1)[1] == digest[:16]
    for row in document["artifacts"]:
        assert row["pipeline_id"] in pipelines
        assert row["added_frontend_id"] in frontends
    for row in document["compositions"]:
        assert row["pipeline_id"] in pipelines
        assert row["added_frontend_id"] in frontends
    pairs = Counter(
        (row["pipeline_id"], row["added_frontend_id"])
        for row in document["compositions"]
    )
    assert set(pairs.values()) == {1}
