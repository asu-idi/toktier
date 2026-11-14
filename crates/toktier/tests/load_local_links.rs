//! `Runtime::load_local` on paths that are, or lead through, symbolic
//! links. The tests load the shipped `qwen3_8b` artifact (from
//! `TOKTIER_TEST_ARTIFACTS` or the default cache, as the lifecycle tests
//! do) and skip when it is absent.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use toktier::{Device, ErrorCode, Runtime};

const FAMILY: &str = "qwen3_8b";
const ENTRY: &str = "qwen3_8b-b968826d9c46";

fn artifact_file() -> Option<PathBuf> {
    let file = std::env::var_os("TOKTIER_TEST_ARTIFACTS")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".cache/toktier/artifacts"))
        })
        .map(|root| root.join(ENTRY).join("tokenizer.json"))
        .filter(|file| file.is_file());
    if file.is_none() {
        eprintln!("skip: no {FAMILY} artifact under TOKTIER_TEST_ARTIFACTS or the cache");
    }
    file
}

/// A runtime whose default cache is an empty directory, so that only the
/// explicit paths under test can serve the family.
fn runtime(scratch: &Path) -> Runtime {
    let cache = scratch.join("empty-cache");
    fs::create_dir_all(&cache).unwrap();
    Runtime::builder()
        .artifact_cache(&cache)
        .device(Device::Cpu)
        .build()
        .unwrap()
}

fn ids(tokenizer: &toktier::Tokenizer) -> Vec<u32> {
    tokenizer
        .encode("Symbolic links lead to the same bytes: 0123456789, caf\u{e9}.")
        .unwrap()
        .ids()
        .to_vec()
}

fn refused(runtime: &Runtime, path: &Path) -> toktier::Error {
    match runtime.load_local(FAMILY, path) {
        Ok(_) => panic!("{} was admitted", path.display()),
        Err(error) => error,
    }
}

#[test]
fn a_link_to_the_tokenizer_file_loads_the_verified_bytes() {
    let Some(real) = artifact_file() else {
        return;
    };
    let scratch = tempfile::tempdir().unwrap();
    let runtime = runtime(scratch.path());
    let expected = ids(&runtime.load_local(FAMILY, real.parent().unwrap()).unwrap());
    let link = scratch.path().join("linked-tokenizer.json");
    symlink(&real, &link).unwrap();
    let tokenizer = runtime.load_local(FAMILY, &link).unwrap();
    assert_eq!(tokenizer.artifact().tokenizer_json(), link.as_path());
    assert_eq!(tokenizer.artifact().directory(), scratch.path());
    assert_eq!(ids(&tokenizer), expected);
}

#[test]
fn a_hub_snapshot_directory_with_relative_links_loads() {
    let Some(real) = artifact_file() else {
        return;
    };
    let scratch = tempfile::tempdir().unwrap();
    let runtime = runtime(scratch.path());
    let expected = ids(&runtime.load_local(FAMILY, real.parent().unwrap()).unwrap());
    // The hub layout: snapshots/<revision>/tokenizer.json is a relative
    // link into blobs/, and the snapshot keeps its sibling files.
    let repository = scratch.path().join("models--example");
    let blobs = repository.join("blobs");
    let snapshot = repository.join("snapshots").join("0123abcd");
    fs::create_dir_all(&blobs).unwrap();
    fs::create_dir_all(&snapshot).unwrap();
    fs::copy(&real, blobs.join("blob-tokenizer")).unwrap();
    fs::write(blobs.join("blob-config"), b"{}").unwrap();
    symlink(
        "../../blobs/blob-tokenizer",
        snapshot.join("tokenizer.json"),
    )
    .unwrap();
    symlink("../../blobs/blob-config", snapshot.join("config.json")).unwrap();
    let tokenizer = runtime.load_local(FAMILY, &snapshot).unwrap();
    assert_eq!(tokenizer.artifact().directory(), snapshot.as_path());
    assert_eq!(
        tokenizer.artifact().tokenizer_json(),
        snapshot.join("tokenizer.json").as_path()
    );
    assert!(tokenizer
        .artifact()
        .directory()
        .join("config.json")
        .is_file());
    assert_eq!(ids(&tokenizer), expected);

    // The snapshot directory reached through a link of its own.
    let linked_snapshot = scratch.path().join("current");
    symlink(&snapshot, &linked_snapshot).unwrap();
    let tokenizer = runtime.load_local(FAMILY, &linked_snapshot).unwrap();
    assert_eq!(tokenizer.artifact().directory(), linked_snapshot.as_path());
    assert_eq!(ids(&tokenizer), expected);
}

#[test]
fn dangling_links_and_link_loops_are_not_found_under_the_given_path() {
    if artifact_file().is_none() {
        return;
    }
    let scratch = tempfile::tempdir().unwrap();
    let runtime = runtime(scratch.path());
    let dangling = scratch.path().join("dangling.json");
    symlink(scratch.path().join("absent.json"), &dangling).unwrap();
    let first = scratch.path().join("loop-a");
    let second = scratch.path().join("loop-b");
    symlink(&second, &first).unwrap();
    symlink(&first, &second).unwrap();
    let directory = scratch.path().join("snapshot");
    fs::create_dir_all(&directory).unwrap();
    symlink("absent-blob", directory.join("tokenizer.json")).unwrap();
    for (path, named) in [
        (dangling.clone(), dangling.clone()),
        (first.clone(), first.clone()),
        (directory.clone(), directory.join("tokenizer.json")),
    ] {
        let error = refused(&runtime, &path);
        assert_eq!(error.code(), ErrorCode::ArtifactNotFound, "{error}");
        assert_eq!(error.path(), Some(named.as_path()), "{error}");
    }
}

#[test]
fn a_link_to_a_directory_is_refused() {
    if artifact_file().is_none() {
        return;
    }
    let scratch = tempfile::tempdir().unwrap();
    let runtime = runtime(scratch.path());
    let target = scratch.path().join("a-directory");
    fs::create_dir_all(&target).unwrap();
    let snapshot = scratch.path().join("snapshot");
    fs::create_dir_all(&snapshot).unwrap();
    symlink(&target, snapshot.join("tokenizer.json")).unwrap();
    let error = refused(&runtime, &snapshot);
    assert_eq!(error.code(), ErrorCode::ArtifactNotFound, "{error}");
    assert_eq!(
        error.path(),
        Some(snapshot.join("tokenizer.json").as_path()),
        "{error}"
    );
}

#[test]
fn a_link_to_other_bytes_is_refused_as_a_regular_file_is() {
    let Some(real) = artifact_file() else {
        return;
    };
    let scratch = tempfile::tempdir().unwrap();
    let runtime = runtime(scratch.path());
    let bytes = fs::read(&real).unwrap();
    let mut changed = bytes.clone();
    let last = changed.len() - 2;
    changed[last] = if changed[last] == b' ' { b'\n' } else { b' ' };
    let cases = [
        ("changed", changed, ErrorCode::ArtifactHashMismatch),
        (
            "short",
            bytes[..bytes.len() - 1].to_vec(),
            ErrorCode::ArtifactSizeMismatch,
        ),
    ];
    for (name, content, code) in cases {
        let file = scratch.path().join(format!("{name}.json"));
        fs::write(&file, &content).unwrap();
        assert_eq!(
            refused(&runtime, &file).code(),
            code,
            "{name}, regular file"
        );
        let link = scratch.path().join(format!("{name}-link.json"));
        symlink(&file, &link).unwrap();
        let error = refused(&runtime, &link);
        assert_eq!(error.code(), code, "{name}, link: {error}");
        assert_eq!(error.path(), Some(link.as_path()), "{error}");
    }
}

#[test]
fn the_default_cache_entry_as_a_link_is_still_refused() {
    let Some(real) = artifact_file() else {
        return;
    };
    let scratch = tempfile::tempdir().unwrap();
    let cache = scratch.path().join("cache");
    fs::create_dir_all(cache.join(ENTRY)).unwrap();
    symlink(&real, cache.join(ENTRY).join("tokenizer.json")).unwrap();
    let runtime = Runtime::builder()
        .artifact_cache(&cache)
        .device(Device::Cpu)
        .build()
        .unwrap();
    match runtime.load(FAMILY) {
        Ok(_) => panic!("a linked cache entry was admitted"),
        Err(error) => eprintln!("refused as expected: {:?}: {error}", error.code()),
    }
}
