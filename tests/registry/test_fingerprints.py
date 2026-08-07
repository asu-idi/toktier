# Pin the capability fingerprint constructions to the contract text.
"""Tests for the pipeline and added-frontend fingerprints.

These pin the byte layout described in ``docs/contracts/fingerprint.md``
Sections 5 and 6 (and the added-frontend domain tag of
``docs/contracts/registry.md`` Section 1.3). When the library grows its own
fingerprint implementation, running it against these vectors is what shows the
two agree.
"""

import hashlib
import struct
from typing import Any

import pytest
from artifact_identity import (
    added_frontend_fingerprint,
    added_frontend_preimage,
    pipeline_fingerprint,
)
from registry_common import (
    ADDED_FRONTEND_DOMAIN_TAG,
    PIPELINE_DOMAIN_TAG,
    GenerationError,
    canonical_json,
)


def sample_artifact() -> dict[str, Any]:
    return {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": {"b": 1, "a": 0}, "dropout": 0.0},
        "pre_tokenizer": {"type": "ByteLevel"},
        "added_tokens": [
            {
                "id": 0,
                "content": "<s>",
                "special": True,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
        ],
        "post_processor": {"type": "TemplateProcessing"},
    }


def test_pipeline_fingerprint_binds_exactly_the_four_sections() -> None:
    artifact = sample_artifact()
    expected_preimage = PIPELINE_DOMAIN_TAG + canonical_json(
        {
            "decoder": None,
            "model": artifact["model"],
            "normalizer": None,
            "pre_tokenizer": artifact["pre_tokenizer"],
        }
    )
    assert (
        pipeline_fingerprint(artifact)
        == hashlib.sha256(expected_preimage).hexdigest()
    )


def test_pipeline_fingerprint_ignores_added_tokens_and_post_processor() -> None:
    artifact = sample_artifact()
    before = pipeline_fingerprint(artifact)
    artifact["added_tokens"] = []
    artifact["post_processor"] = None
    artifact["version"] = "9.9"
    assert pipeline_fingerprint(artifact) == before


def test_pipeline_fingerprint_follows_the_pipeline() -> None:
    artifact = sample_artifact()
    before = pipeline_fingerprint(artifact)
    artifact["normalizer"] = {"type": "NFC"}
    assert pipeline_fingerprint(artifact) != before


def test_added_token_encoding_matches_the_contract_layout() -> None:
    preimage = added_frontend_preimage(sample_artifact()["added_tokens"])
    expected = (
        b"\x01"
        + struct.pack("<I", 1)
        + b"\x01"
        + b"<s>"
        + b"\x01"
        + struct.pack("<Q", 0)
        + b"\x01\x01"
        + b"\x01\x00" * 4
    )
    assert preimage == expected


def test_added_frontend_fingerprint_is_domain_separated() -> None:
    artifact = sample_artifact()
    preimage = added_frontend_preimage(artifact["added_tokens"])
    assert (
        added_frontend_fingerprint(artifact)
        == hashlib.sha256(ADDED_FRONTEND_DOMAIN_TAG + preimage).hexdigest()
    )
    assert added_frontend_fingerprint(artifact) != hashlib.sha256(preimage).hexdigest()


def test_added_frontend_fingerprint_follows_insertion_order() -> None:
    artifact = sample_artifact()
    second = dict(artifact["added_tokens"][0], id=1, content="</s>")
    artifact["added_tokens"] = [artifact["added_tokens"][0], second]
    forward = added_frontend_fingerprint(artifact)
    artifact["added_tokens"] = list(reversed(artifact["added_tokens"]))
    assert added_frontend_fingerprint(artifact) != forward


def test_added_token_missing_attribute_is_refused() -> None:
    artifact = sample_artifact()
    del artifact["added_tokens"][0]["normalized"]
    with pytest.raises(GenerationError):
        added_frontend_fingerprint(artifact)


def test_added_token_wrong_attribute_type_is_refused() -> None:
    artifact = sample_artifact()
    artifact["added_tokens"][0]["special"] = 1
    with pytest.raises(GenerationError):
        added_frontend_fingerprint(artifact)
