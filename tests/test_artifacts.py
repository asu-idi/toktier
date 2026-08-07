"""Artifact manifests, sources, and the verified cache.

Acceptance surface: per-file sha256 manifests, the three digest
mismatch states (online re-fetch, offline hard error, quarantine), and
offline behavior. No test reaches the network: the hub source is
exercised through an injected fetcher.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from toktier.artifacts import (
    ArtifactEntry,
    ArtifactFile,
    ArtifactManifest,
    ArtifactStore,
    HuggingFaceSource,
    LocalDirectorySource,
    artifact_cache_dir,
    kernel_cache_dir,
    sha256_file,
    store_state_dir,
)
from toktier.config import Config
from toktier.errors import ArtifactHashMismatch, ArtifactNotFound, RegistryInvalid

FAMILY = "demo_family"
REVISION = "a" * 40
GOOD = b'{"version": "1.0", "model": {}}\n'
BAD = b"corrupted bytes\n"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_mapping(
    payloads: Mapping[str, bytes],
    *,
    with_size: bool = True,
    local_dir: str | None = None,
) -> dict[str, object]:
    files: dict[str, object] = {}
    for name, data in payloads.items():
        spec: dict[str, object] = {"sha256": digest(data)}
        if with_size:
            spec["size"] = len(data)
        files[name] = spec
    entry: dict[str, object] = {
        "repo_id": "demo/demo",
        "revision": REVISION,
        "files": files,
    }
    if local_dir is not None:
        entry["local_dir"] = local_dir
    return {FAMILY: entry}


def make_manifest(payloads: Mapping[str, bytes], **kwargs: object) -> ArtifactManifest:
    return ArtifactManifest.from_mapping(
        manifest_mapping(payloads, **kwargs),  # type: ignore[arg-type]
        source="<test>",
    )


class RecordingSource:
    """Serves scripted payloads and counts the calls it received."""

    name = "recording"

    def __init__(
        self,
        responses: Mapping[str, list[bytes]],
        *,
        offline: bool = False,
    ) -> None:
        self._responses = {name: list(items) for name, items in responses.items()}
        self.offline = offline
        self.calls: list[str] = []

    def fetch(
        self, entry: ArtifactEntry, artifact_file: ArtifactFile, destination: Path
    ) -> None:
        self.calls.append(artifact_file.name)
        try:
            queue = self._responses[artifact_file.name]
        except KeyError as exc:
            raise ArtifactNotFound(
                f"no scripted payload for {artifact_file.name}",
                details={"family": entry.family, "searched": [], "offline": False},
            ) from exc
        data = queue.pop(0) if len(queue) > 1 else queue[0]
        destination.write_bytes(data)


def make_store(
    tmp_path: Path,
    manifest: ArtifactManifest,
    source: object | None,
    *,
    offline: bool = False,
) -> ArtifactStore:
    config = Config(home=tmp_path / "toktier-home", offline=offline)
    return ArtifactStore(
        manifest,
        config=config,
        source=source,  # type: ignore[arg-type]
    )


# -- manifests ---------------------------------------------------------


def test_manifest_parses_per_file_digests() -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    entry = manifest.get(FAMILY)

    assert entry.repo_id == "demo/demo"
    assert entry.revision == REVISION
    assert entry.files == (
        ArtifactFile(name="tokenizer.json", sha256=digest(GOOD), size=len(GOOD)),
    )
    assert entry.directory_name == f"{FAMILY}-{REVISION[:12]}"
    assert manifest.families() == (FAMILY,)


def test_manifest_rejects_a_revision_only_entry() -> None:
    with pytest.raises(RegistryInvalid) as caught:
        ArtifactManifest.from_mapping(
            {FAMILY: {"repo_id": "demo/demo", "revision": REVISION}},
            source="<test>",
        )

    assert caught.value.code == "REGISTRY_INVALID"
    assert "sha256" in str(caught.value.details["failure"])


def test_manifest_rejects_a_malformed_digest() -> None:
    with pytest.raises(RegistryInvalid):
        ArtifactManifest.from_mapping(
            {
                FAMILY: {
                    "repo_id": "demo/demo",
                    "revision": REVISION,
                    "files": {"tokenizer.json": {"sha256": "NOTAHASH"}},
                }
            },
            source="<test>",
        )


@pytest.mark.parametrize("name", ["../escape.json", "/abs.json", ".hidden.json"])
def test_manifest_rejects_unsafe_file_names(name: str) -> None:
    with pytest.raises(RegistryInvalid):
        ArtifactManifest.from_mapping(
            {
                FAMILY: {
                    "repo_id": "demo/demo",
                    "revision": REVISION,
                    "files": {name: {"sha256": digest(GOOD)}},
                }
            },
            source="<test>",
        )


@pytest.mark.parametrize(
    "family",
    ["../escape", "up/../side", "a/b", "a\\b", ".hidden", "..", "Fam", "sp ace"],
)
def test_manifest_rejects_unsafe_family_ids(family: str) -> None:
    """Family ids become cache path components; the grammar is enforced."""
    with pytest.raises(RegistryInvalid):
        ArtifactManifest.from_mapping(
            {
                family: {
                    "repo_id": "demo/demo",
                    "revision": REVISION,
                    "files": {"tokenizer.json": {"sha256": digest(GOOD)}},
                }
            },
            source="<test>",
        )


@pytest.mark.parametrize(
    "revision", ["../rev", "a/b", "a\\b", ".hidden", "..", "re v", ""]
)
def test_manifest_rejects_unsafe_revisions(revision: str) -> None:
    with pytest.raises(RegistryInvalid):
        ArtifactManifest.from_mapping(
            {
                FAMILY: {
                    "repo_id": "demo/demo",
                    "revision": revision,
                    "files": {"tokenizer.json": {"sha256": digest(GOOD)}},
                }
            },
            source="<test>",
        )


def test_directly_constructed_entries_enforce_the_same_grammar() -> None:
    """The value object guards itself; the parser is not the only gate."""
    with pytest.raises(ValueError):
        ArtifactEntry(
            family="../escape",
            repo_id="demo/demo",
            revision=REVISION,
            files=(ArtifactFile(name="tokenizer.json", sha256=digest(GOOD)),),
        )
    with pytest.raises(ValueError):
        ArtifactEntry(
            family=FAMILY,
            repo_id="demo/demo",
            revision="../rev",
            files=(ArtifactFile(name="tokenizer.json", sha256=digest(GOOD)),),
        )


def test_manifest_overlay_only_adds(tmp_path: Path) -> None:
    base = make_manifest({"tokenizer.json": GOOD})
    overlay = ArtifactManifest.from_mapping(
        {
            FAMILY: {
                "repo_id": "impostor/impostor",
                "revision": "b" * 40,
                "files": {"tokenizer.json": {"sha256": digest(BAD)}},
            },
            "other_family": {
                "repo_id": "demo/other",
                "revision": "c" * 40,
                "files": {"tokenizer.json": {"sha256": digest(GOOD)}},
            },
        },
        source="<overlay>",
    )

    merged = base.overlay(overlay)

    assert merged.get(FAMILY).repo_id == "demo/demo"
    assert merged.get("other_family").repo_id == "demo/other"
    assert merged.families() == (FAMILY, "other_family")


def test_manifest_round_trips_through_a_file(tmp_path: Path) -> None:
    import json

    path = tmp_path / "tokenizer_manifest.json"
    path.write_text(json.dumps(manifest_mapping({"tokenizer.json": GOOD})))

    manifest = ArtifactManifest.load(path)

    assert manifest.get(FAMILY).files[0].sha256 == digest(GOOD)
    assert manifest.sources == (str(path),)


def test_unknown_family_reports_where_we_looked() -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})

    with pytest.raises(ArtifactNotFound) as caught:
        manifest.get("no_such_family")

    assert caught.value.code == "ARTIFACT_NOT_FOUND"
    assert caught.value.details["family"] == "no_such_family"
    assert caught.value.details["searched"] == ["<test>"]


def test_aliases_resolve() -> None:
    manifest = ArtifactManifest.from_mapping(
        {
            FAMILY: {
                "repo_id": "demo/demo",
                "revision": REVISION,
                "files": {"tokenizer.json": {"sha256": digest(GOOD)}},
                "aliases": ["Demo-Family"],
            }
        },
        source="<test>",
    )

    assert manifest.get("Demo-Family").family == FAMILY
    assert "Demo-Family" in manifest


# -- fetch, verify, install -------------------------------------------


def test_ensure_fetches_verifies_and_installs(tmp_path: Path) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source)

    artifact = store.ensure(FAMILY)

    installed = artifact.path("tokenizer.json")
    assert installed.read_bytes() == GOOD
    assert sha256_file(installed) == (digest(GOOD), len(GOOD))
    assert source.calls == ["tokenizer.json"]
    assert artifact.directory == store.root / f"{FAMILY}-{REVISION[:12]}"
    assert (artifact.directory / ".toktier-verified.json").is_file()
    assert (installed.stat().st_mode & 0o777) == 0o600
    assert not list(artifact.directory.glob("*.tmp"))


def test_second_ensure_uses_the_verified_marker(tmp_path: Path) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source)

    store.ensure(FAMILY)
    store.ensure(FAMILY)

    assert source.calls == ["tokenizer.json"]


def test_touched_file_is_re_verified(tmp_path: Path) -> None:
    """A marker is an optimization, not a promise: if the file moved on
    disk, the content is hashed again."""
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source)
    artifact = store.ensure(FAMILY)

    installed = artifact.path("tokenizer.json")
    os.utime(installed, ns=(1, 1))

    store.ensure(FAMILY)

    # Re-hashed, found good, and not fetched again.
    assert source.calls == ["tokenizer.json"]


def test_multiple_files_are_each_verified(tmp_path: Path) -> None:
    payloads = {"tokenizer.json": GOOD, "special_tokens_map.json": b"{}\n"}
    manifest = make_manifest(payloads)
    source = RecordingSource({name: [data] for name, data in payloads.items()})
    store = make_store(tmp_path, manifest, source)

    artifact = store.ensure(FAMILY)

    assert sorted(artifact.files) == sorted(payloads)
    assert sorted(source.calls) == sorted(payloads)


# -- digest mismatch: the three states --------------------------------


def test_online_mismatch_quarantines_and_refetches(tmp_path: Path) -> None:
    """State 1: a suspect cached copy is moved aside and fetched again."""
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source)
    directory = store.root / f"{FAMILY}-{REVISION[:12]}"
    directory.mkdir(parents=True)
    (directory / "tokenizer.json").write_bytes(BAD)

    artifact = store.ensure(FAMILY)

    assert artifact.path("tokenizer.json").read_bytes() == GOOD
    assert source.calls == ["tokenizer.json"]
    quarantined = list(
        (store.quarantine_root / f"{FAMILY}-{REVISION[:12]}").iterdir()
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == BAD


def test_online_mismatch_twice_is_a_hard_error(tmp_path: Path) -> None:
    """State 2: one re-fetch, then the call fails; nothing is installed."""
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [BAD]})
    store = make_store(tmp_path, manifest, source)

    with pytest.raises(ArtifactHashMismatch) as caught:
        store.ensure(FAMILY)

    error = caught.value
    assert error.code == "ARTIFACT_HASH_MISMATCH"
    assert error.details["expected_sha256"] == digest(GOOD)
    assert error.details["observed_sha256"] == digest(BAD)
    assert error.details["attempts"] == 2
    assert str(error.details["path"]).endswith("tokenizer.json")
    assert "toktier artifacts fetch" in str(error.details["remedy"])
    assert source.calls == ["tokenizer.json", "tokenizer.json"]

    directory = store.root / f"{FAMILY}-{REVISION[:12]}"
    assert not (directory / "tokenizer.json").exists()
    assert not list(directory.glob("*.tmp"))
    quarantined = list(
        (store.quarantine_root / f"{FAMILY}-{REVISION[:12]}").iterdir()
    )
    assert len(quarantined) == 2


def test_offline_mismatch_fails_immediately(tmp_path: Path) -> None:
    """State 3: offline, a bad cached file raises at once, no fetch."""
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source, offline=True)
    directory = store.root / f"{FAMILY}-{REVISION[:12]}"
    directory.mkdir(parents=True)
    bad_path = directory / "tokenizer.json"
    bad_path.write_bytes(BAD)

    with pytest.raises(ArtifactHashMismatch) as caught:
        store.ensure(FAMILY)

    error = caught.value
    assert error.details["expected_sha256"] == digest(GOOD)
    assert error.details["observed_sha256"] == digest(BAD)
    assert error.details["path"] == str(bad_path)
    assert error.details["offline"] is True
    assert "network access" in str(error.details["remedy"])
    assert source.calls == []
    # Offline never destroys evidence: the file stays where it is.
    assert bad_path.read_bytes() == BAD


def test_declared_size_is_checked(tmp_path: Path) -> None:
    manifest = ArtifactManifest.from_mapping(
        {
            FAMILY: {
                "repo_id": "demo/demo",
                "revision": REVISION,
                "files": {
                    "tokenizer.json": {
                        "sha256": digest(GOOD),
                        "size": len(GOOD) + 1,
                    }
                },
            }
        },
        source="<test>",
    )
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source)

    with pytest.raises(ArtifactHashMismatch) as caught:
        store.ensure(FAMILY)

    assert caught.value.details["expected_size"] == len(GOOD) + 1
    assert caught.value.details["observed_size"] == len(GOOD)


def test_verify_rehashes_a_tampered_file(tmp_path: Path) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source)
    artifact = store.ensure(FAMILY)

    # Tamper without changing size or timestamps the marker recorded.
    path = artifact.path("tokenizer.json")
    stat = path.stat()
    path.write_bytes(b"x" * len(GOOD))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    offline_store = make_store(tmp_path, manifest, source, offline=True)
    with pytest.raises(ArtifactHashMismatch):
        offline_store.verify(FAMILY)


# -- offline behavior --------------------------------------------------


def test_offline_missing_artifact_is_not_found(tmp_path: Path) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    store = make_store(tmp_path, manifest, None)

    assert store.offline is True
    assert store.availability.source_configured is False
    assert store.availability.reasons == ("no_source",)
    with pytest.raises(ArtifactNotFound) as caught:
        store.ensure(FAMILY)

    assert caught.value.details["offline"] is True
    assert caught.value.details["offline_reasons"] == ["no_source"]
    assert caught.value.details["family"] == FAMILY


def test_configuration_offline_beats_a_working_source(tmp_path: Path) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source, offline=True)

    assert store.offline is True
    assert store.availability.configured_offline is True
    assert store.availability.source_offline is False
    assert store.availability.reasons == ("configured_offline",)
    with pytest.raises(ArtifactNotFound) as caught:
        store.ensure(FAMILY)
    assert caught.value.details["offline_reasons"] == ["configured_offline"]
    assert source.calls == []


def test_an_offline_source_makes_the_store_offline(tmp_path: Path) -> None:
    """The case a single flag hides: the configuration allows fetching."""
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]}, offline=True)
    store = make_store(tmp_path, manifest, source)

    assert store.offline is True
    assert store.availability.configured_offline is False
    assert store.availability.source_name == "recording"
    assert store.availability.source_offline is True
    assert store.availability.available is False
    assert store.availability.reasons == ("source_offline",)


def test_availability_reports_every_reason_it_finds(tmp_path: Path) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]}, offline=True)
    store = make_store(tmp_path, manifest, source, offline=True)

    assert store.availability.reasons == ("configured_offline", "source_offline")


def test_availability_is_clear_when_a_working_source_is_configured(
    tmp_path: Path,
) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source)

    assert store.availability.available is True
    assert store.availability.reasons == ()
    assert store.offline is False


def test_hub_source_respects_the_hub_offline_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    source = HuggingFaceSource()

    assert source.offline is True


def test_hub_source_reads_its_environment_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    source = HuggingFaceSource()
    monkeypatch.delenv("HF_HUB_OFFLINE")

    assert source.offline is True


def test_offline_hub_source_refuses_to_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    monkeypatch.setenv("HF_HUB_OFFLINE", "on")
    store = make_store(tmp_path, manifest, HuggingFaceSource())

    with pytest.raises(ArtifactNotFound) as caught:
        store.ensure(FAMILY)

    assert caught.value.details["offline"] is True


# -- sources -----------------------------------------------------------


def test_hub_source_uses_the_injected_fetcher(tmp_path: Path) -> None:
    """The hub path is exercised without importing a network client."""
    calls: list[tuple[str, str, str]] = []

    def fetcher(
        *, repo_id: str, filename: str, revision: str, local_dir: str
    ) -> str:
        calls.append((repo_id, filename, revision))
        landed = Path(local_dir) / filename
        landed.parent.mkdir(parents=True, exist_ok=True)
        landed.write_bytes(GOOD)
        return str(landed)

    manifest = make_manifest({"tokenizer.json": GOOD})
    store = make_store(tmp_path, manifest, HuggingFaceSource(fetcher=fetcher))

    artifact = store.ensure(FAMILY)

    assert artifact.path("tokenizer.json").read_bytes() == GOOD
    assert calls == [("demo/demo", "tokenizer.json", REVISION)]


def test_local_directory_source_copies_from_disk(tmp_path: Path) -> None:
    upstream = tmp_path / "frozen" / f"{FAMILY}-{REVISION[:12]}"
    upstream.mkdir(parents=True)
    (upstream / "tokenizer.json").write_bytes(GOOD)
    manifest = make_manifest({"tokenizer.json": GOOD})
    store = make_store(
        tmp_path, manifest, LocalDirectorySource(root=tmp_path / "frozen")
    )

    artifact = store.ensure(FAMILY)

    assert artifact.path("tokenizer.json").read_bytes() == GOOD
    # The original is untouched; the cache holds an installed copy.
    assert (upstream / "tokenizer.json").read_bytes() == GOOD


def test_local_directory_source_honors_a_relative_local_dir(
    tmp_path: Path,
) -> None:
    upstream = tmp_path / "frozen" / "tokenizers" / "demo"
    upstream.mkdir(parents=True)
    (upstream / "tokenizer.json").write_bytes(GOOD)
    manifest = make_manifest(
        {"tokenizer.json": GOOD}, local_dir="tokenizers/demo"
    )
    store = make_store(
        tmp_path, manifest, LocalDirectorySource(root=tmp_path / "frozen")
    )

    assert store.ensure(FAMILY).path("tokenizer.json").read_bytes() == GOOD


def test_local_directory_source_reports_a_missing_file(tmp_path: Path) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    store = make_store(
        tmp_path, manifest, LocalDirectorySource(root=tmp_path / "frozen")
    )

    with pytest.raises(ArtifactNotFound) as caught:
        store.ensure(FAMILY)

    searched = caught.value.details["searched"]
    assert isinstance(searched, list)
    assert "tokenizer.json" in str(searched[0])


# -- directory layout --------------------------------------------------


def test_cache_and_state_subtrees_are_separate(tmp_path: Path) -> None:
    config = Config(home=tmp_path / "toktier-home")

    assert artifact_cache_dir(config) == config.cache_dir / "artifacts"
    assert kernel_cache_dir(config) == config.cache_dir / "kernels"
    assert store_state_dir(config) == config.state_dir / "store"
    assert not str(store_state_dir(config)).startswith(str(config.cache_dir))


def test_created_directories_are_owner_only(tmp_path: Path) -> None:
    manifest = make_manifest({"tokenizer.json": GOOD})
    source = RecordingSource({"tokenizer.json": [GOOD]})
    store = make_store(tmp_path, manifest, source)

    artifact = store.ensure(FAMILY)

    assert (artifact.directory.stat().st_mode & 0o077) == 0
