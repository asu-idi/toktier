use std::collections::BTreeMap;
#[cfg(feature = "serving")]
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "serving")]
use std::time::Duration;

use toktier::{
    export_bundle, import_bundle, inspect_bundle, ArtifactManager, ArtifactSource, Device,
    ErrorCode, Revision, Runtime,
};
#[cfg(feature = "serving")]
use toktier::{ServingLimits, ServingPool};

fn existing_artifact_root() -> Option<PathBuf> {
    std::env::var_os("TOKTIER_TEST_ARTIFACTS")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".cache/toktier/artifacts"))
        })
        .filter(|root| root.join("qwen3_8b-b968826d9c46/tokenizer.json").is_file())
}

#[cfg(feature = "serving")]
fn runtime_from(root: &Path) -> toktier::Result<Runtime> {
    Runtime::builder()
        .artifact_cache(root)
        .device(Device::Cpu)
        .build()
}

#[test]
fn canonical_bundle_is_deterministic_and_idempotent() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("tokenizer.json");
    std::fs::write(&source, b"verified bytes\n").unwrap();
    let files = BTreeMap::from([("tokenizer.json".to_owned(), source)]);
    let first = temporary.path().join("first.tar");
    let second = temporary.path().join("second.tar");
    let facts = export_bundle(&first, "demo-deadbeef0000", &files).unwrap();
    export_bundle(&second, "demo-deadbeef0000", &files).unwrap();
    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap()
    );
    assert_eq!(inspect_bundle(&first).unwrap(), facts);
    let cache = temporary.path().join("cache");
    let installed = import_bundle(&first, &cache).unwrap();
    assert_eq!(
        std::fs::read(installed.join("tokenizer.json")).unwrap(),
        b"verified bytes\n"
    );
    assert_eq!(import_bundle(&first, &cache).unwrap(), installed);

    // An alias the cache holds with other contents is its own condition and
    // has carried its own code since 0.2.8; the Python facade reports the
    // same one for the same tree.
    std::fs::write(installed.join("undeclared.txt"), b"foreign").unwrap();
    assert_eq!(
        import_bundle(&first, &cache).unwrap_err().code(),
        ErrorCode::AliasConflict
    );
    std::fs::remove_file(installed.join("undeclared.txt")).unwrap();
    assert_eq!(import_bundle(&first, &cache).unwrap(), installed);
    std::fs::write(installed.join("tokenizer.json"), b"other bytes\n").unwrap();
    assert_eq!(
        import_bundle(&first, &cache).unwrap_err().code(),
        ErrorCode::AliasConflict
    );
    std::fs::write(installed.join("tokenizer.json"), b"verified bytes\n").unwrap();
    assert_eq!(import_bundle(&first, &cache).unwrap(), installed);

    let corrupt = temporary.path().join("corrupt.tar");
    let mut bytes = std::fs::read(&first).unwrap();
    let offset = bytes
        .windows(b"verified bytes\n".len())
        .position(|window| window == b"verified bytes\n")
        .expect("payload is present in deterministic tar");
    bytes[offset] ^= 0x01;
    std::fs::write(&corrupt, bytes).unwrap();
    assert_eq!(
        inspect_bundle(&corrupt).unwrap_err().code(),
        ErrorCode::ArtifactHashMismatch
    );
    let corrupt_cache = temporary.path().join("corrupt-cache");
    assert_eq!(
        import_bundle(&corrupt, &corrupt_cache).unwrap_err().code(),
        ErrorCode::ArtifactHashMismatch
    );
    assert!(!corrupt_cache.join("demo-deadbeef0000").exists());
}

#[test]
fn reimport_is_idempotent_across_the_cache_marker() {
    // import, verify, import: the order the cache is actually used in.
    // Verifying an installed artifact writes `.toktier-verified.json`
    // beside its files. That marker is toktier's own sidecar rather than
    // bundle content, so a tree carrying it -- or carrying an edited one
    // -- is still exactly this bundle, and the next import returns it
    // without reading or rewriting the marker. Everything else in the
    // tree is judged as before.
    const MARKER: &str = ".toktier-verified.json";
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("tokenizer.json");
    std::fs::write(&source, b"verified bytes\n").unwrap();
    let files = BTreeMap::from([("tokenizer.json".to_owned(), source)]);
    let bundle = temporary.path().join("marker.tar");
    export_bundle(&bundle, "demo-deadbeef0001", &files).unwrap();
    let cache = temporary.path().join("cache");
    let installed = import_bundle(&bundle, &cache).unwrap();

    let marker = installed.join(MARKER);
    std::fs::write(&marker, b"{\"format\":1}\n").unwrap();
    assert_eq!(import_bundle(&bundle, &cache).unwrap(), installed);
    assert_eq!(std::fs::read(&marker).unwrap(), b"{\"format\":1}\n");

    // A marker this reader does not understand is the next verification's
    // business, not the import's.
    std::fs::write(&marker, b"not a marker\n").unwrap();
    assert_eq!(import_bundle(&bundle, &cache).unwrap(), installed);
    assert_eq!(std::fs::read(&marker).unwrap(), b"not a marker\n");
    assert_eq!(
        std::fs::read(installed.join("tokenizer.json")).unwrap(),
        b"verified bytes\n"
    );

    // The pass is exactly that name at the top of the tree: a leftover of
    // the marker's own write, and a file of that name further down, are
    // undeclared like anything else.
    let leftover = installed.join(format!("{MARKER}.4321.tmp"));
    std::fs::write(&leftover, b"{}\n").unwrap();
    assert_eq!(
        import_bundle(&bundle, &cache).unwrap_err().code(),
        ErrorCode::AliasConflict
    );
    std::fs::remove_file(&leftover).unwrap();
    let nested = installed.join("nested");
    std::fs::create_dir(&nested).unwrap();
    std::fs::write(nested.join(MARKER), b"{}\n").unwrap();
    assert_eq!(
        import_bundle(&bundle, &cache).unwrap_err().code(),
        ErrorCode::AliasConflict
    );
    std::fs::remove_dir_all(&nested).unwrap();
    assert_eq!(import_bundle(&bundle, &cache).unwrap(), installed);
}

#[test]
#[cfg(unix)]
fn bundle_rejects_symlink_sources_and_cache_roots() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    let linked_source = temporary.path().join("linked-source");
    std::fs::write(&source, b"bytes").unwrap();
    symlink(&source, &linked_source).unwrap();
    let files = BTreeMap::from([("tokenizer.json".to_owned(), linked_source)]);
    assert_eq!(
        export_bundle(
            temporary.path().join("symlink-source.tar"),
            "demo-deadbeef0000",
            &files,
        )
        .unwrap_err()
        .code(),
        ErrorCode::ArtifactNotFound
    );

    let files = BTreeMap::from([("tokenizer.json".to_owned(), source)]);
    let bundle = temporary.path().join("bundle.tar");
    export_bundle(&bundle, "demo-deadbeef0000", &files).unwrap();
    let real_cache = temporary.path().join("real-cache");
    std::fs::create_dir(&real_cache).unwrap();
    let linked_cache = temporary.path().join("linked-cache");
    symlink(&real_cache, &linked_cache).unwrap();
    // A destination the crate refuses to write into is a statement
    // about the configured location, not about an attempted write.
    assert_eq!(
        import_bundle(&bundle, &linked_cache).unwrap_err().code(),
        ErrorCode::ConfigInvalid
    );
    assert!(!real_cache.join("demo-deadbeef0000").exists());

    let parent_relative = temporary.path().join("cache/../escaped-cache");
    let error = import_bundle(&bundle, &parent_relative).unwrap_err();
    assert_eq!(error.code(), ErrorCode::ConfigInvalid);
    assert!(
        error
            .message()
            .contains("may not contain parent-directory components"),
        "{error}"
    );

    // Three refusals about the path leading to the destination, and one
    // about the destination itself. All four are decisions about a
    // configured location, so all four report the same code.
    let occupied = temporary.path().join("occupied");
    std::fs::write(&occupied, b"not a directory").unwrap();
    assert_eq!(
        import_bundle(&bundle, occupied.join("below"))
            .unwrap_err()
            .code(),
        ErrorCode::ConfigInvalid
    );

    let final_is_a_file = temporary.path().join("cache-is-a-file");
    std::fs::write(&final_is_a_file, b"not a directory").unwrap();
    let error = import_bundle(&bundle, &final_is_a_file).unwrap_err();
    assert_eq!(error.code(), ErrorCode::ConfigInvalid);
    assert!(error.message().contains("directory"), "{error}");
    // The file is left exactly as it was; refusing is not repairing.
    assert_eq!(
        std::fs::read(&final_is_a_file).unwrap(),
        b"not a directory".to_vec()
    );
}

#[test]
fn artifact_lifecycle_is_concurrent_offline_and_rust_only() {
    let Some(source_root) = existing_artifact_root() else {
        return;
    };
    let temporary = tempfile::tempdir().unwrap();
    let manager = Arc::new(
        ArtifactManager::builder()
            .cache(temporary.path().join("cache"))
            .source(ArtifactSource::LocalDirectory { root: source_root })
            .build()
            .unwrap(),
    );
    let workers = (0..8)
        .map(|_| {
            let manager = Arc::clone(&manager);
            std::thread::spawn(move || manager.fetch("qwen3_8b").unwrap())
        })
        .collect::<Vec<_>>();
    let rows = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    assert!(rows.iter().all(|row| row == &rows[0]));

    let mirror_root = temporary.path().join("mirror");
    manager.mirror("qwen3_8b", &mirror_root).unwrap();
    let mirrored = ArtifactManager::builder()
        .cache(temporary.path().join("mirror-cache"))
        .source(ArtifactSource::LocalDirectory { root: mirror_root })
        .build()
        .unwrap()
        .fetch("qwen3_8b")
        .unwrap();
    assert_eq!(mirrored.tokenizer_sha256, rows[0].tokenizer_sha256);

    let bundle = temporary.path().join("qwen.tar");
    let exported = manager.export("qwen3_8b", &bundle).unwrap();
    assert!(exported.files.len() > 20);
    let imported_cache = temporary.path().join("imported");
    let offline = ArtifactManager::builder()
        .cache(&imported_cache)
        .source(ArtifactSource::None)
        .offline(true)
        .build()
        .unwrap();
    offline.import(&bundle).unwrap();
    // The manager writes its verified marker into the alias it just
    // installed, so running the documented import twice is the shape a
    // reader meets first.
    offline.import(&bundle).unwrap();
    let runtime = Runtime::builder()
        .artifacts(offline)
        .device(Device::Cpu)
        .build()
        .unwrap();
    let tokenizer = runtime
        .from_pretrained(
            "Qwen/Qwen3-8B",
            Revision::commit("b968826d9c46dd6066d109eabc6255188de91218").unwrap(),
        )
        .unwrap();
    assert!(!tokenizer
        .encode("offline Rust lifecycle")
        .unwrap()
        .ids()
        .is_empty());
}

#[test]
fn artifact_hash_failure_never_publishes_a_verified_handle() {
    let temporary = tempfile::tempdir().unwrap();
    let source_root = temporary.path().join("source");
    let source = source_root
        .join("qwen3_8b-b968826d9c46")
        .join("tokenizer.json");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    // The shipped qwen3 tokenizer is 11,422,654 bytes. Equal size with the
    // wrong content distinguishes digest refusal from a short-read refusal.
    std::fs::write(&source, vec![0u8; 11_422_654]).unwrap();
    let cache = temporary.path().join("cache");
    let manager = ArtifactManager::builder()
        .cache(&cache)
        .source(ArtifactSource::LocalDirectory { root: source_root })
        .build()
        .unwrap();
    assert_eq!(
        manager.fetch("qwen3_8b").unwrap_err().code(),
        ErrorCode::ArtifactHashMismatch
    );
    let visible = cache.join("qwen3_8b-b968826d9c46");
    assert!(!visible.join("tokenizer.json").exists());
    assert!(!visible.join(".toktier-verified.json").exists());
}

#[test]
#[cfg(feature = "serving")]
fn bounded_serving_matches_serial_and_sessions() {
    let Some(root) = existing_artifact_root() else {
        return;
    };
    let tokenizer = runtime_from(&root).unwrap().load("qwen3_8b").unwrap();
    let limits = ServingLimits {
        max_queued_requests: 128,
        max_queued_bytes: 1024 * 1024,
        max_batch_rows: 32,
        max_batch_bytes: 64 * 1024,
        max_session_requests: 32,
        batch_window: Duration::from_millis(2),
        worker_threads: 2,
    };
    let pool = ServingPool::builder(tokenizer.clone())
        .limits(limits)
        .build()
        .unwrap();
    let documents = (0..64)
        .map(|index| format!("concurrent exact row {index} — 你好"))
        .collect::<Vec<_>>();
    let requests = documents
        .iter()
        .map(|text| pool.submit(text).unwrap())
        .collect::<Vec<_>>();
    for (text, request) in documents.iter().zip(requests) {
        let response = request.wait().unwrap();
        assert_eq!(
            response.value.ids(),
            tokenizer.encode(text).unwrap().ids(),
            "queued output diverged"
        );
        assert!(response.timings.total >= response.timings.engine);
        assert!((1..=32).contains(&response.timings.batch_rows));
    }

    let session = pool.open_session("bounded-session").unwrap();
    session.submit_seed("seed: ").unwrap().wait().unwrap();
    let first = session.submit_append("hello").unwrap().wait().unwrap();
    let second = session.submit_append(" world").unwrap().wait().unwrap();
    assert_eq!(first.value.revision(), 1);
    assert_eq!(second.value.revision(), 2);

    // Several workers may dequeue one session concurrently, but execution
    // and revisions must retain submission order exactly.
    let ordered = pool.open_session("ordered-session").unwrap();
    let seed = ordered.submit_seed("ordered seed").unwrap();
    let suffixes = (0..24)
        .map(|index| format!(" / append-{index}"))
        .collect::<Vec<_>>();
    let appends = suffixes
        .iter()
        .map(|suffix| ordered.submit_append(suffix).unwrap())
        .collect::<Vec<_>>();
    let mut ids = seed.wait().unwrap().value.ids().to_vec();
    for (index, request) in appends.into_iter().enumerate() {
        let patch = request.wait().unwrap().value;
        assert_eq!(patch.revision(), index as u64 + 1);
        ids.truncate(patch.keep_tokens() as usize);
        ids.extend_from_slice(patch.replacement_ids());
    }
    let complete = format!("ordered seed{}", suffixes.concat());
    assert_eq!(ids, tokenizer.encode(&complete).unwrap().ids());

    // A request rejected before enqueue still consumes and then explicitly
    // skips its FIFO ticket; the next valid append must not wait forever.
    assert_eq!(
        ordered
            .submit_append("x".repeat(64 * 1024 + 1))
            .unwrap_err()
            .code(),
        ErrorCode::QueueFull
    );
    assert_eq!(
        ordered
            .submit_append(" / after-rejection")
            .unwrap()
            .wait_timeout(Duration::from_secs(5))
            .unwrap()
            .value
            .revision(),
        25
    );

    let oversized = "x".repeat(64 * 1024 + 1);
    assert_eq!(
        pool.submit(oversized).unwrap_err().code(),
        ErrorCode::QueueFull
    );
    let cancelled = pool.submit("cancel before observation").unwrap();
    if cancelled.cancel() {
        assert_eq!(
            cancelled.wait().unwrap_err().code(),
            ErrorCode::RequestCancelled
        );
    }
    pool.shutdown();

    let multi = ServingPool::builder(tokenizer.clone())
        .device(tokenizer)
        .limits(ServingLimits {
            max_queued_requests: 16,
            max_queued_bytes: 1024 * 1024,
            max_batch_rows: 8,
            max_batch_bytes: 64 * 1024,
            max_session_requests: 2,
            batch_window: Duration::ZERO,
            worker_threads: 1,
        })
        .build()
        .unwrap();
    assert_eq!(multi.worker_threads(), 1);
    let requests = (0..4)
        .map(|index| multi.submit(format!("device row {index}")).unwrap())
        .collect::<Vec<_>>();
    let devices = requests
        .into_iter()
        .map(|request| request.wait().unwrap().timings.device_index)
        .collect::<Vec<_>>();
    assert_eq!(devices, vec![0, 1, 0, 1]);
    multi.shutdown();
}

/// PLAN/162 WP8: a durable session state must reopen bit-exactly in a
/// completely fresh `Runtime`, and continue appending against the frozen
/// reference, now that its in-memory form uses shared blocks and lazy
/// checkpointed spans.
#[test]
fn durable_sessions_reopen_bit_exactly_in_a_fresh_runtime() {
    let Some(root) = existing_artifact_root() else {
        return;
    };
    let temporary = tempfile::tempdir().unwrap();
    let home = temporary.path().join("runtime-home");
    let build = || {
        Runtime::builder()
            .artifact_cache(&root)
            .device(Device::Cpu)
            .home(&home)
            .build()
            .unwrap()
    };
    let text = "durable: seed text with several words 123\n";
    let deltas = [" turn one", " turn two with more words", " third"];
    let (pre_save_ids, pre_save_revision) = {
        let runtime = build();
        let tokenizer = runtime.load("qwen3_8b").unwrap();
        let mut session = tokenizer.open_session("wp8-agent").unwrap();
        session.seed(text).unwrap();
        for delta in deltas {
            session.append(delta).unwrap();
        }
        let snapshot = session.snapshot().unwrap();
        let revision = session.revision();
        session.close().unwrap();
        (snapshot.ids().to_vec(), revision)
    };

    // A brand-new Runtime over the same home restores the exact state.
    let runtime = build();
    let tokenizer = runtime.load("qwen3_8b").unwrap();
    let complete = format!("{text}{}", deltas.concat());
    assert_eq!(
        pre_save_ids,
        tokenizer.encode(&complete).unwrap().ids(),
        "pre-save stream deviates from the frozen reference"
    );
    let mut session = tokenizer.open_session("wp8-agent").unwrap();
    assert_eq!(session.revision(), pre_save_revision);
    let reopened = session.snapshot().unwrap();
    assert_eq!(reopened.ids(), &pre_save_ids[..]);

    // The next append after restart matches a fresh reference encode.
    let patch = session.append(" post-restart").unwrap();
    let extended = format!("{complete} post-restart");
    let mut rebuilt = pre_save_ids[..patch.keep_tokens() as usize].to_vec();
    rebuilt.extend_from_slice(patch.replacement_ids());
    assert_eq!(rebuilt, tokenizer.encode(&extended).unwrap().ids());
    assert_eq!(session.snapshot().unwrap().ids(), &rebuilt[..]);
}
