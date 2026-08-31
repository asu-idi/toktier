"""Air-gap bundle happy paths and the complete import acceptance list."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from toktier.artifacts import (
    AirgapBundleSource,
    ArtifactManifest,
    ArtifactStore,
    artifact_cache_dir,
    export_bundle,
    import_bundle,
)
from toktier.artifacts.bundle import (
    BUNDLE_MANIFEST_NAME,
    BUNDLE_ROOT_DOMAIN,
    MAX_BUNDLE_MEMBERS,
    MAX_BUNDLE_UNCOMPRESSED_SIZE,
    VERIFIED_MARKER_NAME,
)
from toktier.config import Config
from toktier.errors import (
    AliasConflict,
    ArtifactHashMismatch,
    BundleInvalid,
)

FAMILY = "bundle_demo"
REVISION = "c" * 40
ALIAS = f"{FAMILY}-{REVISION[:12]}"
GOOD = b"verified bundle\n"
TAMPERED = b"tampered bundle\n"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _root_digest(document: dict[str, object]) -> str:
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(BUNDLE_ROOT_DOMAIN + canonical).hexdigest()


def _manifest_document(
    files: list[dict[str, object]],
    *,
    alias: str = ALIAS,
) -> dict[str, object]:
    body: dict[str, object] = {"alias": alias, "files": files}
    return {**body, "root_digest": _root_digest(body)}


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


def _write_bundle(
    path: Path,
    manifest: dict[str, object],
    members: list[tuple[str, bytes]],
) -> Path:
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    with tarfile.open(path, mode="w") as archive:
        _add_bytes(archive, BUNDLE_MANIFEST_NAME, manifest_bytes)
        for name, data in members:
            _add_bytes(archive, name, data)
    return path


def _artifact_manifest(payloads: dict[str, bytes]) -> ArtifactManifest:
    return ArtifactManifest.from_mapping(
        {
            FAMILY: {
                "repo_id": "demo/bundle",
                "revision": REVISION,
                "files": {
                    name: {"sha256": _digest(data), "size": len(data)}
                    for name, data in payloads.items()
                },
            }
        },
        source="<bundle-test>",
    )


def _assert_no_trace(cache: Path) -> None:
    assert not cache.exists() or list(cache.iterdir()) == []


def _exported_bundle(tmp_path: Path) -> Path:
    source_directory = tmp_path / "verified"
    source_directory.mkdir()
    (source_directory / "tokenizer.json").write_bytes(GOOD)
    return export_bundle(
        tmp_path / "artifact.tar",
        ALIAS,
        {"tokenizer.json": source_directory / "tokenizer.json"},
    )


def test_a_second_import_of_the_same_bundle_is_idempotent(
    tmp_path: Path,
) -> None:
    """Both faces now answer the same way, on the same condition.

    The Rust face has always said re-import is idempotent while the
    visible tree still authenticates exactly; the Python face used to
    refuse every alias it already held. Running the printed recipe twice
    against one cache reaches this path.
    """
    bundle = _exported_bundle(tmp_path)
    cache = artifact_cache_dir(Config(home=tmp_path / "home", offline=True))

    installed = import_bundle(bundle, cache)
    assert installed == cache / ALIAS
    before = (installed / "tokenizer.json").stat()

    assert import_bundle(bundle, cache) == installed

    # The installed bytes were re-read, not rewritten.
    after = (installed / "tokenizer.json").stat()
    assert (installed / "tokenizer.json").read_bytes() == GOOD
    assert (after.st_mtime_ns, after.st_ino) == (before.st_mtime_ns, before.st_ino)
    assert not list(cache.glob(".toktier-bundle-import-*"))


def test_a_reimport_after_the_cache_verified_the_tree_is_idempotent(
    tmp_path: Path,
) -> None:
    """import, verify, import: the order a reader actually uses it in.

    Using an imported artifact makes the store re-hash the directory and
    write its verified marker beside the files. That marker is toktier's
    own sidecar rather than bundle content, so a tree carrying it is
    still exactly this bundle and the next import returns it.
    """
    bundle = _exported_bundle(tmp_path)
    config = Config(home=tmp_path / "home", offline=True)
    cache = artifact_cache_dir(config)
    installed = import_bundle(bundle, cache)

    store = ArtifactStore(
        _artifact_manifest({"tokenizer.json": GOOD}),
        config=config,
        source=None,
    )
    assert store.ensure(FAMILY).directory == installed
    marker = installed / VERIFIED_MARKER_NAME
    assert marker.is_file()
    before = (installed / "tokenizer.json").stat()
    written = marker.read_text(encoding="utf-8")

    assert import_bundle(bundle, cache) == installed

    after = (installed / "tokenizer.json").stat()
    assert (after.st_mtime_ns, after.st_ino) == (before.st_mtime_ns, before.st_ino)
    # The marker is neither read nor rewritten by the import.
    assert marker.read_text(encoding="utf-8") == written
    assert not list(cache.glob(".toktier-bundle-import-*"))


def test_a_changed_marker_is_not_a_conflict_and_is_left_alone(
    tmp_path: Path,
) -> None:
    """The marker's contents are the store's business, not the bundle's.

    An import asks whether the tree is this bundle. The marker is not
    part of that question, so editing it neither refuses the import nor
    provokes a rewrite; the next verification is where it is judged.
    """
    bundle = _exported_bundle(tmp_path)
    cache = artifact_cache_dir(Config(home=tmp_path / "home", offline=True))
    installed = import_bundle(bundle, cache)
    marker = installed / VERIFIED_MARKER_NAME
    marker.write_text("not a marker this reader understands\n", encoding="utf-8")

    assert import_bundle(bundle, cache) == installed

    assert marker.read_text(encoding="utf-8") == (
        "not a marker this reader understands\n"
    )


def test_an_alias_holding_other_bytes_is_a_conflict(tmp_path: Path) -> None:
    """One changed byte is the difference between idempotent and refused."""
    bundle = _exported_bundle(tmp_path)
    cache = artifact_cache_dir(Config(home=tmp_path / "home", offline=True))
    installed = import_bundle(bundle, cache)
    (installed / "tokenizer.json").write_bytes(TAMPERED)

    with pytest.raises(AliasConflict) as caught:
        import_bundle(bundle, cache)

    details = caught.value.details
    assert caught.value.code == "ALIAS_CONFLICT"
    assert details["cause"] == "alias_conflict"
    assert details["failure"] == "hash_mismatch"
    assert details["path"] == str(installed / "tokenizer.json")
    assert details["observed_sha256"] == hashlib.sha256(TAMPERED).hexdigest()
    assert str(installed) in str(details["remedy"])
    # The refusal leaves the cache and the staging area as they were.
    assert (installed / "tokenizer.json").read_bytes() == TAMPERED
    assert not list(cache.glob(".toktier-bundle-import-*"))


def test_an_alias_holding_an_undeclared_file_is_a_conflict(
    tmp_path: Path,
) -> None:
    """A tree with something extra in it is not the bundle either."""
    bundle = _exported_bundle(tmp_path)
    cache = artifact_cache_dir(Config(home=tmp_path / "home", offline=True))
    installed = import_bundle(bundle, cache)
    (installed / "extra.json").write_bytes(b"{}\n")

    with pytest.raises(AliasConflict) as caught:
        import_bundle(bundle, cache)

    assert caught.value.details["failure"] == "undeclared_file"
    assert caught.value.details["path"] == str(installed / "extra.json")
    (installed / "extra.json").unlink()

    # The one file the check passes over is the marker itself, by that
    # exact name at the top of the tree. A file of that name deeper in
    # the tree, and a leftover of some other write, are undeclared like
    # anything else -- including one whose name is close to the marker's
    # own temporary shapes without being one of them.
    for extra in (
        installed / "nested" / VERIFIED_MARKER_NAME,
        installed / f"{VERIFIED_MARKER_NAME}.tmp",
        installed / f"{VERIFIED_MARKER_NAME}.4321.part",
        installed / f"{VERIFIED_MARKER_NAME}.later.tmp",
        installed / "nested" / f"{VERIFIED_MARKER_NAME}.4321.tmp",
        installed / "other.4321.tmp",
    ):
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(b"{}\n")
        with pytest.raises(AliasConflict) as caught:
            import_bundle(bundle, cache)
        assert caught.value.details["failure"] == "undeclared_file"
        assert caught.value.details["path"] == str(extra)
        extra.unlink()


def test_a_leftover_of_the_markers_own_write_is_cleared_by_an_import(
    tmp_path: Path,
) -> None:
    """The friendly side of a refusal that was protecting nothing.

    A process that stops between writing the verified marker and renaming
    it into place leaves its temporary at the top of the alias. Nothing
    reads that file, and until 0.2.9 the next import of the same bundle
    refused the alias over it; the answer was to delete it by hand. Both
    caches write the same marker, so both of their temporary shapes are
    cleared, and the import goes on to be idempotent as usual.
    """
    bundle = _exported_bundle(tmp_path)
    cache = artifact_cache_dir(Config(home=tmp_path / "home", offline=True))
    installed = import_bundle(bundle, cache)
    leftovers = [
        installed / f"{VERIFIED_MARKER_NAME}.4321.tmp",
        installed / f".{VERIFIED_MARKER_NAME}.4321.7.part",
    ]
    for leftover in leftovers:
        leftover.write_bytes(b"{}\n")
    declared = (installed / "tokenizer.json").stat()

    assert import_bundle(bundle, cache) == installed

    assert not any(leftover.exists() for leftover in leftovers)
    # Clearing them is the whole of the change: the declared bytes are
    # still the ones that were there, unread and unwritten.
    assert (installed / "tokenizer.json").stat().st_ino == declared.st_ino
    assert (installed / "tokenizer.json").stat().st_mtime_ns == declared.st_mtime_ns
    assert not list(cache.glob(".toktier-bundle-import-*"))


def test_the_leftover_pass_does_not_reach_the_marker_or_a_directory(
    tmp_path: Path,
) -> None:
    """What the prune must not take with it.

    The marker itself is not a leftover, and neither is a directory that
    happens to carry a leftover's name; a tree still has to be judged as
    the bundle it claims to be after the pass.
    """
    bundle = _exported_bundle(tmp_path)
    cache = artifact_cache_dir(Config(home=tmp_path / "home", offline=True))
    installed = import_bundle(bundle, cache)
    marker = installed / VERIFIED_MARKER_NAME
    marker.write_bytes(b"{}\n")
    shaped_directory = installed / f"{VERIFIED_MARKER_NAME}.99.tmp"
    shaped_directory.mkdir()

    # An empty directory is not a file, so the walk finds nothing
    # undeclared in it and the import is idempotent; the point here is
    # that the prune did not remove the directory.
    assert import_bundle(bundle, cache) == installed
    assert shaped_directory.is_dir()
    assert marker.read_bytes() == b"{}\n"

    (shaped_directory / "inside.json").write_bytes(b"{}\n")
    with pytest.raises(AliasConflict) as caught:
        import_bundle(bundle, cache)
    assert caught.value.details["failure"] == "undeclared_file"
    assert caught.value.details["path"] == str(shaped_directory / "inside.json")


def test_an_alias_missing_a_declared_file_is_a_conflict(
    tmp_path: Path,
) -> None:
    """So is a tree the bundle would fill in."""
    bundle = _exported_bundle(tmp_path)
    cache = artifact_cache_dir(Config(home=tmp_path / "home", offline=True))
    installed = import_bundle(bundle, cache)
    (installed / "tokenizer.json").unlink()

    with pytest.raises(AliasConflict) as caught:
        import_bundle(bundle, cache)

    assert caught.value.details["failure"] == "missing_file"
    assert caught.value.details["path"] == str(installed / "tokenizer.json")


def test_export_import_and_airgap_source_happy_paths(tmp_path: Path) -> None:
    source_directory = tmp_path / "verified"
    source_directory.mkdir()
    tokenizer = source_directory / "tokenizer.json"
    metadata = source_directory / "metadata.json"
    tokenizer.write_bytes(GOOD)
    metadata.write_bytes(b"{}\n")
    payloads = {"tokenizer.json": GOOD, "metadata.json": b"{}\n"}
    bundle = export_bundle(
        tmp_path / "artifact.tar",
        ALIAS,
        {name: source_directory / name for name in payloads},
    )

    with tarfile.open(bundle, mode="r") as archive:
        assert sorted(archive.getnames()) == [
            BUNDLE_MANIFEST_NAME,
            "metadata.json",
            "tokenizer.json",
        ]
        extracted = archive.extractfile(BUNDLE_MANIFEST_NAME)
        assert extracted is not None
        manifest = json.loads(extracted.read())
    digest_input = dict(manifest)
    del digest_input["root_digest"]
    assert manifest["root_digest"] == _root_digest(digest_input)

    offline_config = Config(home=tmp_path / "offline-home", offline=True)
    cache = artifact_cache_dir(offline_config)
    installed = import_bundle(bundle, cache)

    assert installed == cache / ALIAS
    assert installed.joinpath("tokenizer.json").read_bytes() == GOOD
    assert installed.joinpath("metadata.json").read_bytes() == b"{}\n"
    assert not list(cache.glob(".toktier-bundle-import-*"))

    # The imported directory is already usable by an offline store.  The
    # store re-hashes it once and writes its normal verified marker.
    offline_store = ArtifactStore(
        _artifact_manifest(payloads),
        config=offline_config,
        source=None,
    )
    assert offline_store.ensure(FAMILY).path("tokenizer.json").read_bytes() == GOOD

    # The same archive also implements the ordinary ArtifactSource protocol.
    source_store = ArtifactStore(
        _artifact_manifest(payloads),
        config=Config(home=tmp_path / "source-home"),
        source=AirgapBundleSource(bundle),
    )
    assert source_store.ensure(FAMILY).path("metadata.json").read_bytes() == b"{}\n"


def test_import_reports_an_unreadable_archive_with_its_cause(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "not-a-tar.bin"
    bundle.write_bytes(b"this is not a tar archive")
    cache = tmp_path / "cache"

    with pytest.raises(BundleInvalid) as caught:
        import_bundle(bundle, cache)

    assert caught.value.code == "BUNDLE_INVALID"
    assert caught.value.details["failure"] == "cannot read tar archive"
    assert caught.value.details["path"] == str(bundle)
    assert caught.value.details["cause"]
    # The reader's per-method report is several lines; the message that
    # becomes the prose command-line report is one.
    assert "\n" in str(caught.value.details["cause"])
    assert "\n" not in str(caught.value)
    _assert_no_trace(cache)


@pytest.mark.parametrize("unsafe_path", ["../escape.json", "/absolute.json"])
def test_import_rejects_path_traversal_and_absolute_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    bundle = tmp_path / "unsafe-path.tar"
    with tarfile.open(bundle, mode="w") as archive:
        _add_bytes(archive, unsafe_path, b"")
    cache = tmp_path / "cache"

    with pytest.raises(BundleInvalid) as caught:
        import_bundle(bundle, cache)

    assert caught.value.code == "BUNDLE_INVALID"
    assert "traverse" in str(caught.value.details["failure"])
    assert caught.value.details["member"] == unsafe_path
    _assert_no_trace(cache)


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_import_rejects_symbolic_and_hard_links(
    tmp_path: Path,
    link_type: bytes,
) -> None:
    bundle = tmp_path / "link.tar"
    with tarfile.open(bundle, mode="w") as archive:
        member = tarfile.TarInfo("linked.json")
        member.type = link_type
        member.linkname = "tokenizer.json"
        archive.addfile(member)
    cache = tmp_path / "cache"

    with pytest.raises(BundleInvalid) as caught:
        import_bundle(bundle, cache)

    assert caught.value.code == "BUNDLE_INVALID"
    assert "link member" in str(caught.value.details["failure"])
    assert caught.value.details["member"] == "linked.json"
    _assert_no_trace(cache)


def test_import_caps_member_count_at_4096(tmp_path: Path) -> None:
    bundle = tmp_path / "too-many-members.tar"
    with tarfile.open(bundle, mode="w") as archive:
        for index in range(MAX_BUNDLE_MEMBERS + 1):
            member = tarfile.TarInfo(f"directory-{index}/")
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
    cache = tmp_path / "cache"

    with pytest.raises(BundleInvalid) as caught:
        import_bundle(bundle, cache)

    assert caught.value.code == "BUNDLE_INVALID"
    assert "4096" in str(caught.value.details["failure"])
    _assert_no_trace(cache)


def test_import_caps_total_uncompressed_size_at_8_gib(tmp_path: Path) -> None:
    bundle = tmp_path / "too-large.tar"
    member = tarfile.TarInfo("oversized.bin")
    member.size = MAX_BUNDLE_UNCOMPRESSED_SIZE + 1
    # A GNU base-256 size field lets the scanner observe the declared size
    # without allocating or writing an 8 GiB fixture.
    bundle.write_bytes(member.tobuf(format=tarfile.GNU_FORMAT) + (b"\0" * 1024))
    cache = tmp_path / "cache"

    with pytest.raises(BundleInvalid) as caught:
        import_bundle(bundle, cache)

    assert caught.value.code == "BUNDLE_INVALID"
    assert "8 GiB" in str(caught.value.details["failure"])
    _assert_no_trace(cache)


def test_import_rejects_duplicate_paths(tmp_path: Path) -> None:
    bundle = tmp_path / "duplicate.tar"
    with tarfile.open(bundle, mode="w") as archive:
        _add_bytes(archive, "same.json", b"first")
        _add_bytes(archive, "same.json", b"second")
    cache = tmp_path / "cache"

    with pytest.raises(BundleInvalid) as caught:
        import_bundle(bundle, cache)

    assert caught.value.code == "BUNDLE_INVALID"
    assert "duplicate path" in str(caught.value.details["failure"])
    assert caught.value.details["member"] == "same.json"
    _assert_no_trace(cache)


def test_import_verifies_every_file_sha256(tmp_path: Path) -> None:
    files = [
        {"path": "tokenizer.json", "sha256": _digest(GOOD), "size": len(GOOD)}
    ]
    bundle = _write_bundle(
        tmp_path / "bad-file-digest.tar",
        _manifest_document(files),
        [("tokenizer.json", TAMPERED)],
    )
    cache = tmp_path / "cache"

    with pytest.raises(ArtifactHashMismatch) as caught:
        import_bundle(bundle, cache)

    assert caught.value.code == "ARTIFACT_HASH_MISMATCH"
    assert caught.value.details["expected_sha256"] == _digest(GOOD)
    assert caught.value.details["observed_sha256"] == _digest(TAMPERED)
    _assert_no_trace(cache)


def test_import_verifies_manifest_root_digest(tmp_path: Path) -> None:
    files = [
        {"path": "tokenizer.json", "sha256": _digest(GOOD), "size": len(GOOD)}
    ]
    manifest = _manifest_document(files)
    manifest["root_digest"] = "sha256:" + ("0" * 64)
    bundle = _write_bundle(
        tmp_path / "bad-root.tar",
        manifest,
        [("tokenizer.json", GOOD)],
    )
    cache = tmp_path / "cache"

    with pytest.raises(BundleInvalid) as caught:
        import_bundle(bundle, cache)

    assert caught.value.code == "BUNDLE_INVALID"
    assert "root digest mismatch" in str(caught.value.details["failure"])
    assert caught.value.details["member"] == BUNDLE_MANIFEST_NAME
    _assert_no_trace(cache)


def test_failed_import_after_an_extracted_file_leaves_no_trace(
    tmp_path: Path,
) -> None:
    files = [
        {"path": "first.json", "sha256": _digest(GOOD), "size": len(GOOD)},
        {"path": "second.json", "sha256": _digest(GOOD), "size": len(GOOD)},
    ]
    bundle = _write_bundle(
        tmp_path / "partial.tar",
        _manifest_document(files),
        [("first.json", GOOD), ("second.json", TAMPERED)],
    )
    cache = tmp_path / "cache"
    cache.mkdir()

    with pytest.raises(ArtifactHashMismatch):
        import_bundle(bundle, cache)

    assert list(cache.iterdir()) == []
