"""Content-bound model-repository resolution for the public facade."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import toktier
from toktier.artifacts import model_resolution as model_resolution_module
from toktier.artifacts.sibling_aliases import (
    SiblingAliasRecord,
    SiblingAliasRegistry,
)
from toktier.errors import ArtifactNotFound, ToktierError
from toktier.facade import api as facade_api

from .conftest import Rig, byte_level_document

REVISION = "1" * 40
ROOT_DIGEST = "sha256:" + "a" * 64


def _write_tokenizer(path: Path, *, indent: int | None = 2) -> tuple[str, int]:
    raw = json.dumps(byte_level_document(), indent=indent).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _record(
    *,
    source_sha256: str,
    source_size: int,
    canonical_family: str,
    canonical_anchor_sha256: str,
    basis: str = "equivalent_serialisation",
    packaged: bool = True,
    repo_id: str = "example/model",
    source_file: str = "tokenizer.json",
) -> SiblingAliasRecord:
    return SiblingAliasRecord(
        repo_id=repo_id,
        revision=REVISION,
        source_file=source_file,
        source_sha256=source_sha256,
        source_size=source_size,
        canonical_family=canonical_family,
        canonical_anchor_sha256=canonical_anchor_sha256,
        basis=basis,
        canonical_packaged=packaged,
    )


def _install_resolver(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *records: SiblingAliasRecord,
) -> None:
    registry = SiblingAliasRegistry(records=tuple(records), root_digest=ROOT_DIGEST)
    monkeypatch.setattr(facade_api, "shipped_sibling_aliases", lambda: registry)
    monkeypatch.setattr(
        model_resolution_module,
        "_download_model_file",
        lambda *_args, **_kwargs: path,
    )


@pytest.mark.parametrize(
    "basis",
    ["equivalent_serialisation", "equivalent_canonicalisation"],
)
def test_registered_equivalence_uses_the_canonical_artifact(
    rig: Rig,
    reference: Callable[[str], list[int]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    basis: str,
) -> None:
    source = tmp_path / "tokenizer.json"
    source_sha, source_size = _write_tokenizer(source)
    assert source_sha != rig.artifact_sha256
    record = _record(
        source_sha256=source_sha,
        source_size=source_size,
        canonical_family=rig.family,
        canonical_anchor_sha256=rig.artifact_sha256,
        basis=basis,
    )
    _install_resolver(monkeypatch, source, record)

    tokenizer = toktier.from_pretrained(
        record.repo_id,
        config=rig.config,
        manifest=rig.manifest,
        device="cpu",
    )
    try:
        assert tokenizer.family == rig.family
        assert list(tokenizer.encode("hello world").ids) == reference("hello world")
        report = tokenizer.explain()["model_resolution"]
        assert isinstance(report, dict)
        assert report["admitted"] is True
        assert report["basis"] == basis
        assert report["canonical_family"] == rig.family
        assert report["resolved_revision"] == REVISION
        assert report["execution_artifact_sha256"] == rig.artifact_sha256
    finally:
        tokenizer.close()


def test_repository_name_cannot_admit_changed_content(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = tmp_path / "recorded.json"
    recorded_sha, recorded_size = _write_tokenizer(recorded)
    changed = tmp_path / "tokenizer.json"
    changed_raw = json.dumps(byte_level_document(), separators=(",", ":")).encode()
    changed.write_bytes(changed_raw)
    changed_sha = hashlib.sha256(changed_raw).hexdigest()
    assert changed_sha != recorded_sha
    record = _record(
        source_sha256=recorded_sha,
        source_size=recorded_size,
        canonical_family=rig.family,
        canonical_anchor_sha256=rig.artifact_sha256,
    )
    _install_resolver(monkeypatch, changed, record)

    tokenizer = toktier.from_pretrained(
        record.repo_id,
        config=rig.config,
        manifest=rig.manifest,
        device="cpu",
    )
    try:
        assert tokenizer.family == f"external_{changed_sha[:24]}"
        assert tokenizer.plan.backend == "hf"
        report = tokenizer.explain()["model_resolution"]
        assert isinstance(report, dict)
        assert report["admitted"] is False
        assert report["refusal_reason"] == "content_not_registered"
    finally:
        tokenizer.close()


def test_a_different_repository_is_admitted_by_matching_content(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "tokenizer.json"
    source_sha, source_size = _write_tokenizer(source)
    record = _record(
        source_sha256=source_sha,
        source_size=source_size,
        canonical_family=rig.family,
        canonical_anchor_sha256=rig.artifact_sha256,
    )
    _install_resolver(monkeypatch, source, record)

    tokenizer = toktier.from_pretrained(
        "different/fork",
        revision="feature-branch",
        config=rig.config,
        manifest=rig.manifest,
        device="cpu",
    )
    try:
        report = tokenizer.explain()["model_resolution"]
        assert isinstance(report, dict)
        assert tokenizer.family == rig.family
        assert report["requested_repo"] == "different/fork"
        assert report["evidence_repo"] == record.repo_id
        assert report["admitted"] is True
    finally:
        tokenizer.close()


def test_unavailable_canonical_family_stays_reference_only(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "tokenizer.json"
    source_sha, source_size = _write_tokenizer(source)
    record = _record(
        source_sha256=source_sha,
        source_size=source_size,
        canonical_family="bert_cased",
        canonical_anchor_sha256=source_sha,
        basis="equivalent_canonicalisation",
        packaged=False,
    )
    _install_resolver(monkeypatch, source, record)

    tokenizer = toktier.from_pretrained(
        record.repo_id,
        config=rig.config,
        manifest=rig.manifest,
        device="cpu",
    )
    try:
        report = tokenizer.explain()["model_resolution"]
        assert isinstance(report, dict)
        assert tokenizer.plan.backend == "hf"
        assert report["basis"] == "equivalent_canonicalisation"
        assert report["canonical_family"] == "bert_cased"
        assert report["admitted"] is False
        assert report["refusal_reason"] == "canonical_artifact_not_packaged"
    finally:
        tokenizer.close()


def test_exact_anchor_content_is_admitted_without_a_repository_row(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_resolver(monkeypatch, rig.artifact_path)
    tokenizer = toktier.from_pretrained(
        "new/fork",
        config=rig.config,
        manifest=rig.manifest,
        device="cpu",
    )
    try:
        report = tokenizer.explain()["model_resolution"]
        assert isinstance(report, dict)
        assert tokenizer.family == rig.family
        assert report["basis"] == "exact_anchor"
        assert report["admitted"] is True
    finally:
        tokenizer.close()


def test_unregistered_content_honors_require_accelerated(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "tokenizer.json"
    _write_tokenizer(source)
    _install_resolver(monkeypatch, source)
    with pytest.raises(ToktierError):
        toktier.from_pretrained(
            "unknown/model",
            config=rig.config,
            manifest=rig.manifest,
            device="cpu",
            policy="require_accelerated",
        )


def test_source_only_kimi_row_has_an_actionable_error(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "tiktoken.model"
    source.write_bytes(b"tiktoken source fixture")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    record = _record(
        source_sha256=source_sha,
        source_size=source.stat().st_size,
        canonical_family="kimi_k3",
        canonical_anchor_sha256="b" * 64,
        basis="identical_source",
        packaged=False,
        repo_id="moonshotai/Kimi-Test",
        source_file="tiktoken.model",
    )
    _install_resolver(monkeypatch, source, record)

    with pytest.raises(ArtifactNotFound, match="cannot materialize") as caught:
        toktier.from_pretrained(
            record.repo_id,
            config=rig.config,
            manifest=rig.manifest,
            device="cpu",
        )
    assert caught.value.details["source_file"] == "tiktoken.model"
    assert "conversion artifact" in str(caught.value.details["remedy"])
