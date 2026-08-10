"""Reference backend: construction guards and parity sampling.

Contract reference: ``docs/contracts/api.md`` Sections 3-4,
``docs/contracts/errors.md``, ``docs/contracts/registry.md`` Section 4.

Parity here means the backend is id-equal to the pinned oracle called
directly, for every document in the pool, in both single and batch form
and with specials on and off. The backend is a thin wrapper by design,
and this is the test that keeps it one: anything it silently added to
the call -- a flag, a normalization, a truncation -- would show up as a
difference.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import _support as support
import pytest

from toktier.backends.hf import HfBackend, oracle_version
from toktier.errors import (
    ArtifactHashMismatch,
    ArtifactNotFound,
    UnsupportedConfig,
)

tokenizers = pytest.importorskip("tokenizers")

#: Parity readings, printed at the end of the session so a run can be
#: reported as family x documents x mismatches. The list lives in the
#: shared helpers so the session hook in conftest can report it.
READINGS = support.PARITY_READINGS


def _artifact(tmp_path: Path, **document_kwargs: Any) -> support.StubArtifact:
    document = support.byte_level_document(**document_kwargs)
    return support.write_artifact(tmp_path / "artifact", document)


# ---------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------


def test_opens_a_verified_artifact(tmp_path: Path) -> None:
    """The backend reports the identity it was opened with."""
    artifact = _artifact(tmp_path)
    backend = HfBackend.open(artifact)
    assert backend.backend_id == "hf"
    assert backend.family == "test_family"
    assert backend.artifact_sha256 == artifact.artifact_sha256
    assert backend.tokenizer_path.name == "tokenizer.json"
    backend.close()


def test_loader_flags_are_refused(tmp_path: Path) -> None:
    """A flag that would rewrite the pipeline is refused, not ignored."""
    artifact = _artifact(tmp_path)
    with pytest.raises(UnsupportedConfig) as caught:
        HfBackend.open(artifact, loader_flags={"fix_mistral_regex": True})
    assert caught.value.code == "UNSUPPORTED_CONFIG"
    assert caught.value.details["option"] == "fix_mistral_regex"


def test_unknown_loader_flags_are_refused_too(tmp_path: Path) -> None:
    """The refusal is by rule, not by a list of known-bad names."""
    artifact = _artifact(tmp_path)
    with pytest.raises(UnsupportedConfig):
        HfBackend.open(artifact, loader_flags={"some_future_flag": True})


@pytest.mark.parametrize("section", ["truncation", "padding"])
def test_output_rewriting_artifacts_are_refused(
    tmp_path: Path, section: str
) -> None:
    """Modes the core stream cannot represent are rejected at construction."""
    settings = {
        "truncation": {"max_length": 8, "strategy": "LongestFirst", "stride": 0},
        "padding": {"strategy": "BatchLongest", "direction": "Right"},
    }
    artifact = _artifact(tmp_path, **{section: settings[section]})
    with pytest.raises(UnsupportedConfig) as caught:
        HfBackend.open(artifact)
    assert caught.value.details["option"] == section


def test_missing_artifact_file_is_reported(tmp_path: Path) -> None:
    """A missing file is an error with the path in it."""
    empty = support.StubArtifact(
        family="test_family", root=tmp_path / "nothing", artifact_sha256="0" * 64
    )
    with pytest.raises(ArtifactNotFound) as caught:
        HfBackend.open(empty)
    assert caught.value.code == "ARTIFACT_NOT_FOUND"
    assert caught.value.details["family"] == "test_family"


def test_digest_drift_is_refused(tmp_path: Path) -> None:
    """Bytes that no longer match the manifest are never accepted."""
    artifact = _artifact(tmp_path)
    wrong = support.StubArtifact(
        family=artifact.family,
        root=artifact.root,
        artifact_sha256="0" * 64,
        files={"tokenizer.json": "0" * 64},
    )
    with pytest.raises(ArtifactHashMismatch) as caught:
        HfBackend.open(wrong)
    assert caught.value.details["expected_sha256"] == "0" * 64
    assert caught.value.details["observed_sha256"] == artifact.artifact_sha256


def test_closed_backend_refuses_to_encode(tmp_path: Path) -> None:
    """Use after close is a programming error, and says so."""
    backend = HfBackend.open(_artifact(tmp_path))
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError):
        backend.encode("x")


def test_oracle_version_is_read_from_metadata() -> None:
    """The installed oracle version is reported without importing it."""
    assert oracle_version() == tokenizers.__version__


# ---------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------


def _parity(
    backend: HfBackend,
    reference: Any,
    documents: list[str],
    *,
    family: str,
    provenance: str,
) -> int:
    """Compare backend against the oracle; returns the mismatch count."""
    mismatches = 0
    for add_special_tokens in (False, True):
        expected = [
            list(
                reference.encode(
                    text, add_special_tokens=add_special_tokens
                ).ids
            )
            for text in documents
        ]
        single = [
            backend.encode(text, add_special_tokens=add_special_tokens)
            for text in documents
        ]
        batched = backend.encode_batch(
            documents, add_special_tokens=add_special_tokens
        )
        mismatches += sum(1 for a, b in zip(single, expected, strict=False) if a != b)
        mismatches += sum(1 for a, b in zip(batched, expected, strict=False) if a != b)
    READINGS.append(
        {
            "family": family,
            "documents": len(documents),
            "mismatches": mismatches,
            "provenance": provenance,
        }
    )
    return mismatches


@pytest.mark.slow
def test_parity_against_the_oracle_synthetic(
    tmp_path: Path, parity_documents: tuple[list[str], str]
) -> None:
    """Parity on a synthetic artifact; always runs, everywhere."""
    documents, provenance = parity_documents
    artifact = _artifact(tmp_path, added_tokens=[
        {
            "content": "<|end|>",
            "id": 256,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    ])
    backend = HfBackend.open(artifact)
    reference = tokenizers.Tokenizer.from_file(str(artifact.path("tokenizer.json")))
    assert (
        _parity(
            backend,
            reference,
            documents,
            family="synthetic_byte_level",
            provenance=provenance,
        )
        == 0
    )
    backend.close()


def _local_artifacts() -> list[tuple[str, Path]]:
    """(family, directory) pairs discovered at collection time."""
    root = os.environ.get("TOKTIER_TEST_ARTIFACTS")
    if not root or not Path(root).is_dir():
        return []
    revision = re.compile(r"-[0-9a-f]{12}$")
    return [
        (revision.sub("", path.name), path)
        for path in sorted(Path(root).iterdir())
        if path.is_dir() and (path / "tokenizer.json").is_file()
    ]


LOCAL_ARTIFACTS = _local_artifacts()


@pytest.mark.slow
@pytest.mark.skipif(
    not LOCAL_ARTIFACTS, reason="no local artifacts (set TOKTIER_TEST_ARTIFACTS)"
)
@pytest.mark.parametrize(
    ("family", "directory"),
    LOCAL_ARTIFACTS,
    ids=[item[0] for item in LOCAL_ARTIFACTS],
)
def test_parity_against_local_artifacts(
    family: str,
    directory: Path,
    parity_documents: tuple[list[str], str],
) -> None:
    """Parity for one locally available frozen artifact."""
    documents, provenance = parity_documents
    backend = HfBackend.open(support.local_artifact(directory, family=family))
    reference = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    mismatches = _parity(
        backend, reference, documents, family=family, provenance=provenance
    )
    backend.close()
    assert mismatches == 0


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("TOKTIER_TEST_REFERENCE_CROSSCHECK") != "1",
    reason="cross-check against the wrapper library is opt-in",
)
def test_parity_against_the_wrapper_library(
    parity_documents: tuple[list[str], str],
) -> None:
    """The backend matches the wrapper caliber the readings were taken with.

    The judged campaigns loaded artifacts through the wrapper library
    with no loader flags at all. This backend goes straight to the
    oracle instead, so the two paths are compared on the same documents
    to show that the delivered reference is the judged one.
    """
    transformers = pytest.importorskip("transformers")
    transformers.logging.set_verbosity_error()
    if not LOCAL_ARTIFACTS:
        pytest.skip("no local artifacts (set TOKTIER_TEST_ARTIFACTS)")
    documents, provenance = parity_documents
    for family, directory in LOCAL_ARTIFACTS:
        try:
            wrapper = transformers.AutoTokenizer.from_pretrained(
                str(directory), use_fast=True, local_files_only=True
            )
        except Exception:
            tokenizer_file = str(directory / "tokenizer.json")
            wrapper = transformers.PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_file
            )
        backend = HfBackend.open(support.local_artifact(directory, family=family))
        mismatches = 0
        for text in documents:
            expected = [
                int(value)
                for value in wrapper(text, add_special_tokens=False)["input_ids"]
            ]
            if backend.encode(text, add_special_tokens=False) != expected:
                mismatches += 1
        READINGS.append(
            {
                "family": f"{family} (wrapper cross-check)",
                "documents": len(documents),
                "mismatches": mismatches,
                "provenance": provenance,
            }
        )
        backend.close()
        assert mismatches == 0, family


def test_batch_is_row_for_row_equal(tmp_path: Path) -> None:
    """encode_batch equals encode, element by element."""
    backend = HfBackend.open(_artifact(tmp_path))
    texts = ["", "a", "hello world", "\u4e2d\u6587", " " * 10]
    assert backend.encode_batch(texts) == [backend.encode(t) for t in texts]
    assert backend.encode_batch([]) == []
    backend.close()


def test_artifact_document_is_unchanged_by_opening(tmp_path: Path) -> None:
    """Opening reads; it does not rewrite the artifact."""
    artifact = _artifact(tmp_path)
    before = artifact.path("tokenizer.json").read_bytes()
    HfBackend.open(artifact).close()
    assert artifact.path("tokenizer.json").read_bytes() == before
    assert json.loads(before.decode("utf-8"))["truncation"] is None
