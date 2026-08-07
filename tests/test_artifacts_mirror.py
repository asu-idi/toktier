"""Mirror artifact source acceptance tests; no test reaches the network."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from toktier.artifacts import ArtifactManifest, ArtifactStore, MirrorSource
from toktier.config import Config
from toktier.errors import ArtifactHashMismatch, ArtifactNotFound

FAMILY = "mirror_demo"
REVISION = "b" * 40
GOOD = b'{"model": "verified"}\n'
BAD = b"tampered\n"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest() -> ArtifactManifest:
    return ArtifactManifest.from_mapping(
        {
            FAMILY: {
                "repo_id": "demo/tokenizer",
                "revision": REVISION,
                "files": {
                    "tokenizer.json": {
                        "sha256": _digest(GOOD),
                        "size": len(GOOD),
                    }
                },
            }
        },
        source="<mirror-test>",
    )


class ScriptedFetcher:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, *, url: str, destination: Path) -> None:
        self.calls.append(url)
        response = self.responses.pop(0)
        destination.write_bytes(response)


def _store(tmp_path: Path, source: MirrorSource) -> ArtifactStore:
    return ArtifactStore(
        _manifest(),
        config=Config(home=tmp_path / "toktier-home"),
        source=source,
    )


def test_mirror_fetches_the_repo_relative_url_and_installs_verified_bytes(
    tmp_path: Path,
) -> None:
    fetcher = ScriptedFetcher([GOOD])
    source = MirrorSource("https://mirror.example/artifacts", fetcher=fetcher)

    artifact = _store(tmp_path, source).ensure(FAMILY)

    assert artifact.path("tokenizer.json").read_bytes() == GOOD
    assert fetcher.calls == [
        "https://mirror.example/artifacts/demo/tokenizer/resolve/"
        f"{REVISION}/tokenizer.json"
    ]


def test_mirror_hash_mismatch_is_quarantined_and_refetched_once(
    tmp_path: Path,
) -> None:
    fetcher = ScriptedFetcher([BAD, GOOD])
    store = _store(
        tmp_path,
        MirrorSource("https://mirror.example", fetcher=fetcher),
    )

    artifact = store.ensure(FAMILY)

    assert artifact.path("tokenizer.json").read_bytes() == GOOD
    assert len(fetcher.calls) == 2
    quarantined = list(
        (store.quarantine_root / _manifest().get(FAMILY).directory_name).iterdir()
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == BAD


def test_mirror_second_hash_mismatch_is_a_hard_error(tmp_path: Path) -> None:
    fetcher = ScriptedFetcher([BAD, BAD])
    store = _store(
        tmp_path,
        MirrorSource("https://mirror.example", fetcher=fetcher),
    )

    with pytest.raises(ArtifactHashMismatch) as caught:
        store.ensure(FAMILY)

    assert caught.value.code == "ARTIFACT_HASH_MISMATCH"
    assert caught.value.details["attempts"] == 2
    assert caught.value.details["expected_sha256"] == _digest(GOOD)
    assert caught.value.details["observed_sha256"] == _digest(BAD)
    assert len(fetcher.calls) == 2
    assert not (store.directory(FAMILY) / "tokenizer.json").exists()


def test_offline_mirror_does_not_fetch_a_missing_file(tmp_path: Path) -> None:
    fetcher = ScriptedFetcher([GOOD])
    store = _store(
        tmp_path,
        MirrorSource(
            "https://mirror.example",
            fetcher=fetcher,
            offline=True,
        ),
    )

    with pytest.raises(ArtifactNotFound) as caught:
        store.ensure(FAMILY)

    assert caught.value.code == "ARTIFACT_NOT_FOUND"
    assert caught.value.details["offline"] is True
    assert fetcher.calls == []


def test_offline_mirror_rejects_a_bad_cached_file_without_refetch(
    tmp_path: Path,
) -> None:
    fetcher = ScriptedFetcher([GOOD])
    store = _store(
        tmp_path,
        MirrorSource(
            "https://mirror.example",
            fetcher=fetcher,
            offline=True,
        ),
    )
    directory = store.directory(FAMILY)
    directory.mkdir(parents=True)
    cached = directory / "tokenizer.json"
    cached.write_bytes(BAD)

    with pytest.raises(ArtifactHashMismatch) as caught:
        store.ensure(FAMILY)

    assert caught.value.details["offline"] is True
    assert caught.value.details["attempts"] == 0
    assert cached.read_bytes() == BAD
    assert fetcher.calls == []
