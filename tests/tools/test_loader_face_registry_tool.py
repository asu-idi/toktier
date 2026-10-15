"""The maintainer tool that records the loader-face digest on each record.

Contract reference: ``docs/contracts/registry.md`` Section 1
(``loader_face_sha256``). What is checked here is the agreement the
field rests on: the text the tool digests for an artifact directory is
the text the reference backend executes for it
(``materialized_tokenizer_json``), on both branches of the rule -- the
file-only face and the face the pinned loader materializes when the
configuration sidecar contributes an added token -- and the document the
tool writes carries the digest with a verifying root digest. The shipped
values themselves are pinned in ``tests/registry/test_loader_face_digests.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for entry in (str(ROOT), str(TOOLS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import update_loader_face_registry as tool  # noqa: E402
from registry_common import (  # noqa: E402
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    root_digest,
    serialise_document,
)

from tests.facade.conftest import Rig, build_rig  # noqa: E402
from toktier.backends.loader_face import loader_face_sha256  # noqa: E402

EXTRA = "<x_extra>"
EXTRA_ID = 256


def _sidecar_rig(tmp_path: Path) -> Rig:
    rig = build_rig(tmp_path)
    sidecar = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "added_tokens_decoder": {
            str(EXTRA_ID): {
                "content": EXTRA,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
        },
    }
    (rig.artifact_path.parent / "tokenizer_config.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return rig


def test_the_recorded_loader_is_the_floor_of_the_package_metadata() -> None:
    """The faces were recorded with one loader release, and the package
    metadata admits no loader older than it: the tool states the release
    and reads the floor, and refuses when the two have moved apart."""
    assert tool.RECORDED_LOADER_VERSION == "4.57.6"
    assert tool.pinned_loader_version() == tool.RECORDED_LOADER_VERSION


def test_a_moved_loader_floor_is_refused(tmp_path: Path) -> None:
    moved = tmp_path / "pyproject.toml"
    moved.write_text(
        'dependencies = [\n    "transformers>=5.0,<6",\n]\n', encoding="utf-8"
    )
    with pytest.raises(GenerationError, match=r"recorded with 4\.57\.6"):
        tool.pinned_loader_version(moved)
    exact = tmp_path / "exact.toml"
    exact.write_text(
        'dependencies = [\n    "transformers==4.57.6",\n]\n', encoding="utf-8"
    )
    with pytest.raises(GenerationError, match="floor"):
        tool.pinned_loader_version(exact)


def test_file_only_face_is_the_artifact_text(tmp_path: Path) -> None:
    rig = build_rig(tmp_path)
    root = rig.artifact_path.parent
    text = tool.loader_face_text(root)
    assert text == rig.artifact_path.read_text(encoding="utf-8")
    assert loader_face_sha256(text) == rig.artifact_sha256
    tokenizer = rig.tokenizer(device="cpu")
    try:
        assert text == tokenizer._backend.materialized_tokenizer_json()
    finally:
        tokenizer.close()


def test_materialized_face_is_the_backends_document(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    rig = _sidecar_rig(tmp_path)
    root = rig.artifact_path.parent
    text = tool.loader_face_text(root)
    assert text != rig.artifact_path.read_text(encoding="utf-8")
    assert loader_face_sha256(text) != rig.artifact_sha256
    tokenizer = rig.tokenizer(device="cpu")
    try:
        assert text == tokenizer._backend.materialized_tokenizer_json()
    finally:
        tokenizer.close()


def test_augmented_document_writes_the_digest_and_a_verifying_root() -> None:
    artifacts: list[dict[str, str]] = [
        {"family": "a", "artifact_sha256": "a" * 64},
        {"family": "b", "artifact_sha256": "b" * 64},
    ]
    registry: dict[str, object] = {
        "schema_version": 1,
        "root_digest": "sha256:" + "0" * 64,
        "artifacts": artifacts,
    }
    observed = {"a": "1" * 64}
    generated = tool.augmented_document(registry, observed)
    rows = {row["family"]: row for row in generated["artifacts"]}
    assert rows["a"]["loader_face_sha256"] == "1" * 64
    assert "loader_face_sha256" not in rows["b"]
    assert generated["root_digest"] == root_digest(generated, REGISTRY_DOMAIN_TAG)
    # The input is left alone.
    assert "loader_face_sha256" not in artifacts[0]


def test_check_names_missing_and_differing_digests() -> None:
    def row(family: str, recorded: str | None) -> dict[str, str]:
        entry = {"family": family, "artifact_sha256": family * 64}
        if recorded is not None:
            entry["loader_face_sha256"] = recorded
        return entry

    registry = {"artifacts": [row("a", "1" * 64), row("b", None), row("c", "3" * 64)]}
    observed = {"a": "1" * 64, "b": "2" * 64, "c": "4" * 64}
    problems = tool.check_problems(registry, observed)
    assert len(problems) == 2
    assert problems[0].startswith("b: no loader_face_sha256")
    assert problems[1].startswith("c: the registry records loader_face_sha256 3")


def test_the_shipped_table_digests_are_sha256_values() -> None:
    document = json.loads(tool.DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    for row in document["artifacts"]:
        digest = row["loader_face_sha256"]
        assert hashlib.sha256(b"").hexdigest() != digest
        assert len(digest) == 64


def test_check_compares_the_copies_before_counting_uncached_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two registry copies need no cached artifact to compare, so a
    machine with only part of the roster cached still gets that answer:
    differing copies fail the check (exit 1) instead of the partial cache
    ending it first with the not-concluded status (exit 3)."""
    registry: dict[str, object] = {
        "schema_version": 1,
        "artifacts": [{"family": "a", "artifact_sha256": "a" * 64}],
    }
    repository = tmp_path / "support_registry.json"
    packaged = tmp_path / "packaged" / "support_registry.json"
    packaged.parent.mkdir()
    for path in (repository, packaged):
        path.write_bytes(serialise_document(registry))
    monkeypatch.setattr(tool, "DEFAULT_REGISTRY", repository)
    monkeypatch.setattr(tool, "PACKAGED_REGISTRY", packaged)
    monkeypatch.setattr(tool, "installed_loader_version", tool.pinned_loader_version)
    monkeypatch.setattr(tool, "observed_loader_faces", lambda _registry: ({}, ["a"]))

    assert tool.copy_problems(registry) == []
    assert tool.main(["--check"]) == tool.ABSENT
    packaged.write_bytes(serialise_document({**registry, "schema_version": 2}))
    assert tool.copy_problems(registry) == [
        f"{packaged} is not the serialised registry"
    ]
    capsys.readouterr()
    assert tool.main(["--check"]) == 1
    captured = capsys.readouterr()
    assert f"{packaged} is not the serialised registry" in captured.err
    reading = json.loads(captured.out)
    assert reading["absent"] == ["a"]
    assert reading["problems"] == [f"{packaged} is not the serialised registry"]
