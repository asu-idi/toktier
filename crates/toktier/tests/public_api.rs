use toktier::{Device, ErrorCode, GpuDelivery, Runtime, TokenBuffer};

#[test]
fn token_buffers_are_shared_and_contiguous() {
    let buffer = TokenBuffer::from(vec![1, 2, 3, 5]);
    let cloned = buffer.clone();
    assert_eq!(buffer.as_slice(), &[1, 2, 3, 5]);
    assert_eq!(cloned.as_ptr(), buffer.as_ptr());
    assert_eq!(cloned.into_vec(), vec![1, 2, 3, 5]);
}

#[test]
fn public_owners_are_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<toktier::Runtime>();
    send_sync::<toktier::Tokenizer>();
    send_sync::<TokenBuffer>();
}

#[test]
fn stable_error_codes_are_not_display_strings() {
    assert_eq!(
        ErrorCode::ArtifactHashMismatch.as_str(),
        "ARTIFACT_HASH_MISMATCH"
    );
    assert_eq!(
        ErrorCode::SessionRevisionConflict.as_str(),
        "SESSION_REVISION_CONFLICT"
    );
}

#[test]
fn builder_validates_before_engine_construction() {
    let error = Runtime::builder()
        .device(Device::Cpu)
        .gpu_delivery(GpuDelivery::Prebuilt)
        .build()
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::ConfigInvalid);
}

#[test]
fn doctor_is_typed_and_python_free() {
    let facts = Runtime::builder()
        .device(Device::Cpu)
        .build()
        .unwrap()
        .doctor();
    assert!(facts.registry_verified);
    assert!(!facts.python_required);
    assert!(facts.cuda.is_none());
    if cfg!(debug_assertions) {
        assert!(!facts.runtime_build.certified);
    } else {
        // A release-profile public facade is an accelerated execution host;
        // source or build-flag drift must make this assertion fail closed
        // until fresh matrix evidence updates the shipped registry.
        assert!(facts.runtime_build.certified);
    }
    assert_eq!(facts.runtime_build.source_digest.len(), 64);
    assert_eq!(facts.runtime_build.fast_cpu_source_digest.len(), 64);
    assert_eq!(facts.runtime_build.native_host_source_digest.len(), 64);
    assert_eq!(facts.oracle, toktier::ORACLE);
}

/// The verified local qwen3 artifact, when this host has one.
fn artifact_root() -> Option<std::path::PathBuf> {
    std::env::var_os("TOKTIER_TEST_ARTIFACTS")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .map(|home| home.join(".cache/toktier/artifacts"))
        })
        .filter(|root| root.join("qwen3_8b-b968826d9c46/tokenizer.json").is_file())
}

#[test]
fn session_encodings_share_the_store_row_and_survive_mutation_and_drop() {
    let Some(root) = artifact_root() else {
        return;
    };
    let runtime = Runtime::builder()
        .artifact_cache(&root)
        .device(Device::Cpu)
        .build()
        .unwrap();
    let tokenizer = runtime.load("qwen3_8b").unwrap();
    let mut session = tokenizer.open_session("wp7-sharing").unwrap();
    let text = "user: shared-buffer ownership check 123\n";
    let seed = session.seed(text).unwrap();
    assert_eq!(seed.ids(), tokenizer.encode(text).unwrap().ids());

    // Snapshots of the unchanged session are the same immutable row the
    // seed returned: one allocation, observed twice.
    let first_snapshot = session.snapshot().unwrap();
    assert_eq!(seed.ids().as_ptr(), first_snapshot.ids().as_ptr());

    // Mutation never changes an already returned Encoding; the patch is
    // delta-only and rebuilds the exact new stream.
    let frozen = seed.ids().to_vec();
    let delta = "assistant: still exact\n";
    let patch = session.append(delta).unwrap();
    assert_eq!(seed.ids(), &frozen[..], "retained Encoding changed");
    let second_snapshot = session.snapshot().unwrap();
    assert_ne!(
        second_snapshot.ids().as_ptr(),
        first_snapshot.ids().as_ptr()
    );
    let mut rebuilt = first_snapshot.ids()[..patch.keep_tokens() as usize].to_vec();
    rebuilt.extend_from_slice(patch.replacement_ids());
    assert_eq!(rebuilt, second_snapshot.ids());
    let complete = format!("{text}{delta}");
    assert_eq!(
        second_snapshot.ids(),
        tokenizer.encode(&complete).unwrap().ids()
    );

    // Concurrent readers of retained encodings while the session mutates.
    std::thread::scope(|scope| {
        let readers = (0..4)
            .map(|_| {
                let seed = &seed;
                let snapshot = &second_snapshot;
                let frozen = &frozen;
                scope.spawn(move || {
                    for _ in 0..200 {
                        assert_eq!(seed.ids().len(), frozen.len());
                        assert!(!snapshot.ids().is_empty());
                    }
                })
            })
            .collect::<Vec<_>>();
        for round in 0..8 {
            session.append(&format!(" turn {round}")).unwrap();
        }
        for reader in readers {
            reader.join().unwrap();
        }
    });
    assert_eq!(seed.ids(), &frozen[..]);

    // Encodings outlive session close and runtime drop.
    let final_snapshot = session.snapshot().unwrap();
    session.close().unwrap();
    drop(tokenizer);
    runtime.shutdown();
    assert_eq!(seed.ids(), &frozen[..]);
    assert!(!final_snapshot.ids().is_empty());
    assert_eq!(&final_snapshot.ids()[..frozen.len()], &frozen[..]);
}

#[test]
fn lookup_encodings_outlive_the_evicted_lookup_session() {
    let Some(root) = artifact_root() else {
        return;
    };
    let runtime = Runtime::builder()
        .artifact_cache(&root)
        .device(Device::Cpu)
        .build()
        .unwrap();
    let tokenizer = runtime.load("qwen3_8b").unwrap();
    // Content lookup needs sealed chain nodes; a session long enough to
    // cross several 4096-character blocks feeds the prefix cache.
    let text = "content lookup body with words 0123456789 ".repeat(400);
    let mut session = tokenizer.open_session("wp7-lookup-source").unwrap();
    session.seed(&text).unwrap();
    session.close().unwrap();
    let Some(hit) = tokenizer.lookup(&text).unwrap() else {
        // A miss is legal (no certified boundary sealed a block); the
        // lifetime property below is then covered by the session tests.
        return;
    };
    // The lookup's temporary session was evicted before returning; the
    // shared row must remain readable and exact regardless.
    assert_eq!(hit.ids(), tokenizer.encode(&text).unwrap().ids());
    drop(tokenizer);
    runtime.shutdown();
    assert!(!hit.ids().is_empty());
}

/// PLAN/162 WP5/WP6: with `seed_digest_overlap(true)` the seed digest
/// scan joins the seed encode on the bounded pool. The public results
/// must be indistinguishable from the serial default: identical rows,
/// the same shared-allocation behavior, and a working content-digest
/// consumer (`encode_transcript` proves the stored prefix digest).
#[test]
fn seed_digest_overlap_matches_the_serial_sessions_exactly() {
    let Some(root) = artifact_root() else {
        return;
    };
    let build = |overlap: bool| {
        Runtime::builder()
            .artifact_cache(&root)
            .device(Device::Cpu)
            .seed_digest_overlap(overlap)
            .build()
            .unwrap()
    };
    let text = "user: overlap parity check over several words 0123456789\n".repeat(64);
    let delta = "assistant: still bit-exact\n";

    let serial_runtime = build(false);
    let serial_tokenizer = serial_runtime.load("qwen3_8b").unwrap();
    let mut serial_session = serial_tokenizer.open_session("w4c-serial").unwrap();
    let serial_seed = serial_session.seed(&text).unwrap();

    let overlap_runtime = build(true);
    let overlap_tokenizer = overlap_runtime.load("qwen3_8b").unwrap();
    let mut overlap_session = overlap_tokenizer.open_session("w4c-overlap").unwrap();
    let overlap_seed = overlap_session.seed(&text).unwrap();

    assert_eq!(serial_seed.ids(), overlap_seed.ids());
    assert_eq!(
        overlap_seed.ids(),
        overlap_tokenizer.encode(&text).unwrap().ids()
    );
    // The seed still shares the store's adopted row under overlap.
    let snapshot = overlap_session.snapshot().unwrap();
    assert_eq!(overlap_seed.ids().as_ptr(), snapshot.ids().as_ptr());

    // Appends evolve both sessions identically.
    let serial_patch = serial_session.append(delta).unwrap();
    let overlap_patch = overlap_session.append(delta).unwrap();
    assert_eq!(serial_patch.keep_tokens(), overlap_patch.keep_tokens());
    assert_eq!(
        serial_patch.replacement_ids(),
        overlap_patch.replacement_ids()
    );

    // The content digest the overlap path computed proves the stored
    // prefix for the transcript-compatibility operation.
    let complete = format!("{text}{delta}");
    let transcript = overlap_session.encode_transcript(&complete).unwrap();
    assert_eq!(
        transcript.ids(),
        overlap_tokenizer.encode(&complete).unwrap().ids()
    );
}

/// Durable tier under overlap: the TKFR-v1 binding written from the
/// overlap-computed content checkpoints must verify bit-exactly in a
/// completely fresh `Runtime` (a diverging digest byte would fail the
/// restore), and recovery hashing stays serial in the durable tier.
#[test]
fn seed_digest_overlap_durable_sessions_reopen_bit_exactly() {
    let Some(root) = artifact_root() else {
        return;
    };
    let temporary = tempfile::tempdir().unwrap();
    let home = temporary.path().join("runtime-home");
    let build = || {
        Runtime::builder()
            .artifact_cache(&root)
            .device(Device::Cpu)
            .home(&home)
            .seed_digest_overlap(true)
            .build()
            .unwrap()
    };
    let text = "durable overlap: seed text with several words 123\n";
    let deltas = [" turn one", " turn two with more words"];
    let pre_save_ids = {
        let runtime = build();
        let tokenizer = runtime.load("qwen3_8b").unwrap();
        let mut session = tokenizer.open_session("w4c-durable").unwrap();
        session.seed(text).unwrap();
        for delta in deltas {
            session.append(delta).unwrap();
        }
        let snapshot = session.snapshot().unwrap();
        session.close().unwrap();
        snapshot.ids().to_vec()
    };
    let runtime = build();
    let tokenizer = runtime.load("qwen3_8b").unwrap();
    let complete = format!("{text}{}", deltas.concat());
    assert_eq!(pre_save_ids, tokenizer.encode(&complete).unwrap().ids());
    let mut session = tokenizer.open_session("w4c-durable").unwrap();
    let reopened = session.snapshot().unwrap();
    assert_eq!(reopened.ids(), &pre_save_ids[..]);
    let patch = session.append(" post-restart").unwrap();
    let extended = format!("{complete} post-restart");
    let mut rebuilt = pre_save_ids[..patch.keep_tokens() as usize].to_vec();
    rebuilt.extend_from_slice(patch.replacement_ids());
    assert_eq!(rebuilt, tokenizer.encode(&extended).unwrap().ids());
}

/// Concurrent named sessions on one overlap-enabled tokenizer: every
/// lane's stream stays bit-identical to a fresh reference encode, and
/// the run completes without deadlock across the shared bounded pool.
#[test]
fn seed_digest_overlap_concurrent_sessions_stay_exact() {
    let Some(root) = artifact_root() else {
        return;
    };
    let runtime = Runtime::builder()
        .artifact_cache(&root)
        .device(Device::Cpu)
        .seed_digest_overlap(true)
        .build()
        .unwrap();
    let tokenizer = runtime.load("qwen3_8b").unwrap();
    std::thread::scope(|scope| {
        for lane in 0..4usize {
            let tokenizer = &tokenizer;
            scope.spawn(move || {
                let mut session = tokenizer.open_session(format!("w4c-lane-{lane}")).unwrap();
                let mut text = format!("lane {lane}: words to seed 0123456789\n").repeat(200);
                session.seed(&text).unwrap();
                for round in 0..3 {
                    let delta = format!(" lane {lane} turn {round}");
                    session.append(&delta).unwrap();
                    text.push_str(&delta);
                }
                assert_eq!(
                    session.snapshot().unwrap().ids(),
                    tokenizer.encode(&text).unwrap().ids(),
                    "lane {lane} diverged from the fresh reference encode"
                );
            });
        }
    });
}
