//! Store behavior battery (hermetic tier).
//!
//! This is the Rust-tier port of the pre-release prototype store v1 test
//! battery, driven by the deterministic mock encoder so it runs with no
//! tokenizer dependency. The real-tokenizer tier of the same battery
//! (plus the cross-implementation equivalence run) was exercised
//! against the prototype before this port was adopted.

use sha2::{Digest, Sha256};
use toktier_store_core::testing::{fp, MockEncoder};
use toktier_store_core::{
    AppendOutcome, RecoveryMaterial, SessionEncoder, SessionRecordV1, SessionStore, StoreConfig,
    StoreError, WitnessCategory,
};

fn cfg(block_chars: u64) -> StoreConfig {
    StoreConfig {
        block_chars,
        ..StoreConfig::default()
    }
}

fn judge(enc: &MockEncoder, text: &str) -> Vec<u32> {
    enc.encode(text).unwrap().ids
}

fn recovery_digest(text: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"toktier.facade.v1.recovery-text\0");
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

/// Append with the frozen outcome invariant checked:
/// `all_ids == old_all[..replace_from] ++ replacement_ids`.
fn checked_append(
    store: &mut SessionStore,
    handle: toktier_store_core::SessionHandle,
    delta: &str,
    revision: u64,
    enc: &MockEncoder,
    old_all: &[u32],
) -> AppendOutcome {
    let out = store.append(handle, delta, revision, enc).unwrap();
    let rf = usize::try_from(out.replace_from).unwrap();
    assert!(rf <= old_all.len(), "replace_from beyond old stream");
    assert_eq!(&out.all_ids[..rf], &old_all[..rf], "kept prefix changed");
    let mut rebuilt = old_all[..rf].to_vec();
    rebuilt.extend_from_slice(&out.replacement_ids);
    assert_eq!(rebuilt, out.all_ids, "append invariant broken");
    out
}

const DELTAS: [&str; 5] = [
    "alization proposal.\n",
    "It was accepted by all members ",
    "after a long debate. after a long debate. after a long debate. ",
    "\u{4f60}\u{597d}, world. CRLF\r\nmixed emoji \u{1f642} combining: cafe\u{301} done. ",
    "tail",
];

#[test]
fn put_append_matches_reference() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(64)).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let mut acc = String::from("The committee discussed the internation");
    let put = store.put(kid, &acc, &enc).unwrap();
    assert_eq!(store.all_ids(put.handle).unwrap(), judge(&enc, &acc));
    let mut rev = put.revision;
    for delta in DELTAS {
        let old_all = store.all_ids(put.handle).unwrap();
        let out = checked_append(&mut store, put.handle, delta, rev, &enc, &old_all);
        rev = out.revision;
        acc.push_str(delta);
        assert_eq!(out.all_ids, judge(&enc, &acc), "path={}", out.path);
        assert_eq!(store.all_ids(put.handle).unwrap(), judge(&enc, &acc));
    }
    let st = store.stats();
    assert_eq!(st.format, "toktier.store.v1");
    assert!(
        st.seals > 0,
        "prose text should advance certified seal points"
    );
}

#[test]
fn patch_only_append_reconstructs_without_requesting_a_snapshot() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(32)).unwrap();
    let kid = store.register_fingerprint(fp(9), 0).unwrap();
    let mut text = "A long agent transcript ending in an internation".repeat(4);
    let put = store.put(kid, &text, &enc).unwrap();
    let mut downstream = judge(&enc, &text);
    let mut revision = put.revision;

    for delta in ["al reply", " with Unicode 你好", " and a final turn."] {
        let patch = store
            .append_patch(put.handle, delta, revision, &enc)
            .unwrap();
        let keep = usize::try_from(patch.replace_from).unwrap();
        assert!(keep <= downstream.len());
        downstream.truncate(keep);
        downstream.extend_from_slice(&patch.replacement_ids);
        text.push_str(delta);
        assert_eq!(downstream, judge(&enc, &text));
        assert_eq!(patch.token_count, downstream.len() as u64);
        revision = patch.revision;
    }
}

#[test]
fn noop_and_empty_put() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::with_defaults();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let put = store.put(kid, "", &enc).unwrap();
    assert_eq!(put.token_count, 0);
    assert!(store.all_ids(put.handle).unwrap().is_empty());
    let out = store.append(put.handle, "", 0, &enc).unwrap();
    assert_eq!(out.path, "noop");
    assert!(out.all_ids.is_empty());
    assert!(out.replacement_ids.is_empty());
    assert_eq!(out.replace_from, 0);
    let out2 = store
        .append(put.handle, "hello world", out.revision, &enc)
        .unwrap();
    assert_eq!(out2.all_ids, judge(&enc, "hello world"));
}

#[test]
fn wrong_fingerprint_must_miss() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(32)).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let text = "shared system prompt text, long enough for blocks. ".repeat(8);
    store.put(kid, &text, &enc).unwrap();
    assert!(store.lookup(kid, &text, &enc).unwrap().is_some());
    for tag in [2u8, 3, 4] {
        let kw = store.register_fingerprint(fp(tag), 0).unwrap();
        assert!(store.lookup(kw, &text, &enc).unwrap().is_none());
    }
    assert_eq!(store.stats().lookup_misses, 3);
}

#[test]
fn corrupted_node_must_miss_and_count() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(32)).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let text = "shared system prompt text, long enough for blocks. ".repeat(8);
    store.put(kid, &text, &enc).unwrap();
    assert!(store.lookup(kid, &text, &enc).unwrap().is_some());
    let items = store.export_node_items().unwrap();
    assert!(!items.is_empty());
    for (node_key, _rec) in &items {
        assert!(store.corrupt_node_for_tests(node_key).unwrap());
    }
    let before = store.stats().checksum_rejects;
    assert!(store.lookup(kid, &text, &enc).unwrap().is_none());
    let st = store.stats();
    assert!(st.checksum_rejects > before);
    assert!(st.lookup_misses >= 1);
}

#[test]
fn lookup_materialize_then_append() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(64)).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let prefix = "A shared prefix used by many sessions. ".repeat(12);
    store.put(kid, &prefix, &enc).unwrap();
    let query =
        format!("{prefix}and a divergent continuation with \u{4e2d}\u{6587} and\r\nnewlines.");
    let hit = store.lookup(kid, &query, &enc).unwrap().expect("must hit");
    assert!(hit.matched_chars >= 64 && hit.matched_chars.is_multiple_of(64));
    let prefix_chars = prefix.chars().count() as u64;
    assert!(hit.matched_chars <= prefix_chars);
    let rest: String = query
        .chars()
        .skip(usize::try_from(hit.matched_chars).unwrap())
        .collect();
    let old_all = store.all_ids(hit.handle).unwrap();
    let out = checked_append(&mut store, hit.handle, &rest, hit.revision, &enc, &old_all);
    assert_eq!(out.all_ids, judge(&enc, &query));
    assert_eq!(store.stats().hit_rate, Some(1.0));
}

#[test]
fn fork_shares_nodes_and_diverges() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(64)).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let base = "Base conversation content here. ".repeat(10);
    let h1 = store.put(kid, &base, &enc).unwrap().handle;
    let nodes_before = store.stats().node_count;
    let h2 = store.fork(h1).unwrap();
    assert_eq!(
        store.stats().node_count,
        nodes_before,
        "fork must not copy nodes"
    );
    store.append(h1, " branch one.", 0, &enc).unwrap();
    store
        .append(h2, " branch two, different.", 0, &enc)
        .unwrap();
    assert_eq!(
        store.all_ids(h1).unwrap(),
        judge(&enc, &format!("{base} branch one."))
    );
    assert_eq!(
        store.all_ids(h2).unwrap(),
        judge(&enc, &format!("{base} branch two, different."))
    );
}

#[test]
fn session_lru_eviction() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(StoreConfig {
        max_sessions: 2,
        ..StoreConfig::default()
    })
    .unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let h1 = store.put(kid, "first session", &enc).unwrap().handle;
    let h2 = store.put(kid, "second session", &enc).unwrap().handle;
    store.append(h1, " touch", 0, &enc).unwrap(); // h2 becomes LRU
    let h3 = store.put(kid, "third session", &enc).unwrap().handle;
    assert_eq!(store.stats().sessions_evicted, 1);
    let mut expect = vec![h1, h3];
    expect.sort();
    assert_eq!(store.list_handles(), expect);
    assert!(matches!(
        store.append(h2, "gone", 0, &enc),
        Err(StoreError::UnknownSession(_))
    ));
    assert!(store.evict(h1));
    assert!(!store.evict(h1));
}

#[test]
fn soft_cap_overflow_counted_correctness_kept() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(StoreConfig {
        tail_soft_cap_bytes: 256,
        tail_hard_cap_bytes: 1 << 20,
        ..StoreConfig::default()
    })
    .unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let run = "x".repeat(400); // single-class run: no certified boundary
    let h = store.put(kid, &run, &enc).unwrap().handle;
    store.append(h, &"y".repeat(200), 0, &enc).unwrap();
    let st = store.stats();
    assert!(st.k_cap_overflows >= 1);
    let full = format!("{}{}", "x".repeat(400), "y".repeat(200));
    assert_eq!(store.all_ids(h).unwrap(), judge(&enc, &full));
}

#[test]
fn hard_cap_degrades_counted_correctness_kept() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(StoreConfig {
        tail_soft_cap_bytes: 128,
        tail_hard_cap_bytes: 512,
        ..StoreConfig::default()
    })
    .unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let mut acc = "z".repeat(600); // tail already over the hard cap
    let h = store.put(kid, &acc, &enc).unwrap().handle;
    let out = store.append(h, &"q".repeat(40), 0, &enc).unwrap();
    assert_eq!(out.path, "degraded_full_reencode");
    acc.push_str(&"q".repeat(40));
    let st = store.stats();
    assert_eq!(st.hard_cap_degrades, 1);
    assert_eq!(store.all_ids(h).unwrap(), judge(&enc, &acc));
    // Normal text afterwards: seals recover the tail, degradation stops.
    let words = " normal words return here. ".repeat(30);
    let out2 = store.append(h, &words, out.revision, &enc).unwrap();
    acc.push_str(&words);
    let out3 = store.append(h, " and more.", out2.revision, &enc).unwrap();
    acc.push_str(" and more.");
    assert_ne!(out3.path, "degraded_full_reencode");
    assert_eq!(store.all_ids(h).unwrap(), judge(&enc, &acc));
}

#[test]
fn uncertified_engine_never_seals_still_correct() {
    let enc = MockEncoder {
        certify: false,
        ..MockEncoder::default()
    };
    let mut store = SessionStore::new(cfg(64)).unwrap();
    let kid = store.register_fingerprint(fp(9), 0).unwrap();
    let acc = "words and spaces everywhere ".repeat(40);
    let h = store.put(kid, &acc, &enc).unwrap().handle;
    store.append(h, "more words", 0, &enc).unwrap();
    let st = store.stats();
    assert_eq!(st.seals, 0);
    let full = format!("{acc}more words");
    assert_eq!(store.all_ids(h).unwrap(), judge(&enc, &full));
    assert_eq!(
        store.session_info(h).unwrap().witness,
        WitnessCategory::NoneFullReencode
    );
}

#[test]
fn seal_guard_keeps_distance_from_tail_end() {
    let enc = MockEncoder::default();
    // Guard larger than any text: no seal may ever happen.
    let mut store = SessionStore::new(StoreConfig {
        block_chars: 64,
        tail_soft_cap_bytes: 64,
        ..StoreConfig::default()
    })
    .unwrap();
    let kid = store.register_fingerprint(fp(1), 1_000_000).unwrap();
    let h = store
        .put(kid, &"many words with boundaries ".repeat(30), &enc)
        .unwrap()
        .handle;
    store
        .append(h, &"and more words here ".repeat(10), 0, &enc)
        .unwrap();
    assert_eq!(store.stats().seals, 0, "guard must suppress all seals");

    // Moderate guard: every seal stays at least guard chars away from
    // the then-current tail end.
    let guard = 5u64;
    let mut store2 = SessionStore::new(StoreConfig {
        block_chars: 64,
        tail_soft_cap_bytes: 64,
        ..StoreConfig::default()
    })
    .unwrap();
    let kid2 = store2.register_fingerprint(fp(2), guard).unwrap();
    let mut acc = String::new();
    let mut rev = 0u64;
    let mut h2 = None;
    for step in 0..12 {
        let delta = format!("chunk {step} with words and 123 numbers ");
        match h2 {
            None => {
                let put = store2.put(kid2, &delta, &enc).unwrap();
                h2 = Some(put.handle);
                rev = put.revision;
            }
            Some(h) => {
                rev = store2.append(h, &delta, rev, &enc).unwrap().revision;
            }
        }
        acc.push_str(&delta);
        let info = store2.session_info(h2.unwrap()).unwrap();
        assert!(
            info.safe_char + guard <= info.total_chars || info.safe_char == 0,
            "seal landed inside the guard window"
        );
        assert_eq!(store2.all_ids(h2.unwrap()).unwrap(), judge(&enc, &acc));
    }
}

#[test]
fn revision_conflicts_are_rejected_and_counted() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::with_defaults();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let put = store.put(kid, "base text ", &enc).unwrap();
    let h = put.handle;
    let out = store
        .append(h, "first append ", put.revision, &enc)
        .unwrap();
    // Stale expected_revision: must fail, must not mutate.
    let ids_before = store.all_ids(h).unwrap();
    let err = store
        .append(h, "conflicting append", put.revision, &enc)
        .unwrap_err();
    assert!(matches!(
        err,
        StoreError::RevisionConflict {
            expected: 0,
            actual: 1
        }
    ));
    assert_eq!(err.code(), "SESSION_REVISION_CONFLICT");
    assert_eq!(store.all_ids(h).unwrap(), ids_before);
    assert_eq!(store.revision(h).unwrap(), out.revision);
    assert_eq!(store.stats().revision_conflicts, 1);
    // The current revision still works.
    let out2 = store
        .append(h, "second append", out.revision, &enc)
        .unwrap();
    assert_eq!(
        out2.all_ids,
        judge(&enc, "base text first append second append")
    );
}

#[test]
fn witness_category_mismatch_is_rejected() {
    let certified = MockEncoder::default();
    let uncertified = MockEncoder {
        certify: false,
        ..MockEncoder::default()
    };
    let mut store = SessionStore::with_defaults();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let h = store.put(kid, "some text", &certified).unwrap().handle;
    let err = store.append(h, "delta", 0, &uncertified).unwrap_err();
    assert_eq!(err.code(), "SESSION_STATE_MISMATCH");
    assert!(matches!(err, StoreError::WitnessCategoryMismatch { .. }));
}

#[test]
fn invalid_inputs_raise() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::with_defaults();
    assert!(matches!(
        store.put(toktier_store_core::KeyId(99), "text", &enc),
        Err(StoreError::UnknownKey(99))
    ));
    assert!(matches!(
        store.append(toktier_store_core::SessionHandle(42), "d", 1, &enc),
        Err(StoreError::UnknownSession(42))
    ));
    let kid = store.register_fingerprint(fp(1), 3).unwrap();
    assert_eq!(store.register_fingerprint(fp(1), 3).unwrap(), kid);
    assert!(matches!(
        store.register_fingerprint(fp(1), 4),
        Err(StoreError::GuardMismatch)
    ));
    assert!(SessionStore::new(StoreConfig {
        block_chars: 0,
        ..StoreConfig::default()
    })
    .is_err());
    assert!(SessionStore::new(StoreConfig {
        tail_soft_cap_bytes: 2,
        tail_hard_cap_bytes: 1,
        ..StoreConfig::default()
    })
    .is_err());
}

// ------------------------------------------------- record transport ----

#[test]
fn recovery_tracking_is_opt_in_and_must_precede_sessions() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(32)).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let handle = store
        .put(kid, "ordinary in-memory session", &enc)
        .unwrap()
        .handle;
    assert!(store.recovery_material(handle).unwrap().is_none());
    assert!(matches!(
        store.enable_recovery_tracking(),
        Err(StoreError::InvalidInput(_))
    ));
}

#[test]
fn recovery_material_tracks_append_fork_and_lookup_text() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(32)).unwrap();
    store.enable_recovery_tracking().unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let mut text = "prefix \u{4f60}\u{597d} cafe\u{301} \u{1f642} ".repeat(20);
    let put = store.put(kid, &text, &enc).unwrap();

    let initial = store.recovery_material(put.handle).unwrap().unwrap();
    let record = SessionRecordV1::from_bytes(&store.export_session(put.handle).unwrap()).unwrap();
    assert_eq!(initial.record_hash, record.curr_block_hash);
    assert_eq!(initial.text_bytes, text.len() as u64);
    assert_eq!(initial.text_digest, recovery_digest(&text));

    let delta = " append \u{1f680}\r\n";
    let out = store.append(put.handle, delta, put.revision, &enc).unwrap();
    text.push_str(delta);
    let grown = store.recovery_material(put.handle).unwrap().unwrap();
    assert_eq!(grown.text_bytes, text.len() as u64);
    assert_eq!(grown.text_digest, recovery_digest(&text));

    let fork = store.fork(put.handle).unwrap();
    assert_eq!(
        store.recovery_material(fork).unwrap().unwrap().text_digest,
        grown.text_digest
    );
    store.append(fork, " fork only", 0, &enc).unwrap();
    assert_eq!(
        store.recovery_material(put.handle).unwrap().unwrap(),
        grown,
        "fork updates must not mutate their source"
    );

    let query = format!("{text} lookup continuation");
    let hit = store.lookup(kid, &query, &enc).unwrap().unwrap();
    let matched: String = query.chars().take(hit.matched_chars as usize).collect();
    let lookup_material = store.recovery_material(hit.handle).unwrap().unwrap();
    assert_eq!(lookup_material.text_bytes, matched.len() as u64);
    assert_eq!(lookup_material.text_digest, recovery_digest(&matched));

    let noop = store.append(put.handle, "", out.revision, &enc).unwrap();
    let after_noop = store.recovery_material(put.handle).unwrap().unwrap();
    assert_eq!(after_noop.text_bytes, grown.text_bytes);
    assert_eq!(after_noop.text_digest, grown.text_digest);
    assert_ne!(after_noop.record_hash, grown.record_hash);
    assert_eq!(noop.revision, out.revision + 1);
}

#[test]
fn recovery_aware_import_requires_exact_historical_binding() {
    let enc = MockEncoder::default();
    let mut source = SessionStore::new(cfg(32)).unwrap();
    source.enable_recovery_tracking().unwrap();
    let kid = source.register_fingerprint(fp(1), 0).unwrap();
    let text = "sealed recovery \u{4f60}\u{597d} data ".repeat(80);
    let handle = source.put(kid, &text, &enc).unwrap().handle;
    assert!(source.session_info(handle).unwrap().stable_prefix_bytes > 0);
    let record = source.export_session(handle).unwrap();
    let expected = source.recovery_material(handle).unwrap().unwrap();

    let mut bare = SessionStore::new(cfg(32)).unwrap();
    bare.enable_recovery_tracking().unwrap();
    let bare_key = bare.register_fingerprint(fp(1), 0).unwrap();
    let bare_handle = bare.import_session(bare_key, &record, &enc).unwrap();
    assert!(bare.recovery_material(bare_handle).unwrap().is_none());

    let mut restored = SessionStore::new(cfg(32)).unwrap();
    restored.enable_recovery_tracking().unwrap();
    let restored_key = restored.register_fingerprint(fp(1), 0).unwrap();
    let restored_handle = restored
        .import_session_with_recovery(restored_key, &record, &text, &expected, &enc)
        .unwrap();
    assert_eq!(
        restored
            .recovery_material(restored_handle)
            .unwrap()
            .unwrap(),
        expected
    );

    let mut wrong_hash = expected.record_hash;
    wrong_hash[0] ^= 1;
    let mut wrong_digest = expected.text_digest;
    wrong_digest[31] ^= 1;
    let wrong_hash_material = RecoveryMaterial {
        record_hash: wrong_hash,
        ..expected.clone()
    };
    let wrong_length_material = RecoveryMaterial {
        text_bytes: expected.text_bytes + 1,
        ..expected.clone()
    };
    let wrong_digest_material = RecoveryMaterial {
        text_digest: wrong_digest,
        ..expected.clone()
    };
    for result in [
        restored.import_session_with_recovery(
            restored_key,
            &record,
            &text,
            &wrong_hash_material,
            &enc,
        ),
        restored.import_session_with_recovery(
            restored_key,
            &record,
            &text,
            &wrong_length_material,
            &enc,
        ),
        restored.import_session_with_recovery(
            restored_key,
            &record,
            &text,
            &wrong_digest_material,
            &enc,
        ),
        restored.import_session_with_recovery(
            restored_key,
            &record,
            &(text.clone() + "!"),
            &expected,
            &enc,
        ),
    ] {
        assert!(matches!(result, Err(StoreError::MalformedRecord(_))));
    }

    let short_text = "unsealed \u{4f60}\u{597d}";
    let short = source.put(kid, short_text, &enc).unwrap().handle;
    assert_eq!(source.session_info(short).unwrap().stable_prefix_bytes, 0);
    let short_record = source.export_session(short).unwrap();
    let short_import = restored
        .import_session(restored_key, &short_record, &enc)
        .unwrap();
    let short_material = restored.recovery_material(short_import).unwrap().unwrap();
    assert_eq!(short_material.text_digest, recovery_digest(short_text));
}

#[test]
fn sidecar_import_can_restore_incremental_tracking_from_tkfr() {
    let enc = MockEncoder::default();
    let text = "restart-safe multilingual history 中🙂. ".repeat(40);
    let mut original = SessionStore::new(cfg(64)).unwrap();
    original.enable_recovery_tracking().unwrap();
    original.enable_content_tracking().unwrap();
    let key = original.register_fingerprint(fp(11), 0).unwrap();
    let put = original.put(key, &text, &enc).unwrap();
    let record = original.export_session(put.handle).unwrap();
    let sidecar = original.export_session_sidecar(put.handle).unwrap();
    let binding = original
        .export_recovery_binding(put.handle)
        .unwrap()
        .unwrap();

    let mut restored = SessionStore::new(cfg(64)).unwrap();
    restored.enable_recovery_tracking().unwrap();
    restored.enable_content_tracking().unwrap();
    let restored_key = restored.register_fingerprint(fp(11), 0).unwrap();
    for (node_key, node) in original.export_node_items().unwrap() {
        assert!(restored.import_node_item(&node_key, &node));
    }
    let handle = restored
        .import_session_with_sidecar(restored_key, &record, &sidecar, &enc)
        .unwrap();
    assert!(restored.content_index_entry(handle).unwrap().is_none());
    restored
        .restore_tracking_with_binding(handle, &text, &binding)
        .unwrap();
    assert!(restored.content_index_entry(handle).unwrap().is_some());

    let delta = " appended after restart.";
    let patch = restored
        .append_patch(handle, delta, put.revision, &enc)
        .unwrap();
    let mut expected_text = text;
    expected_text.push_str(delta);
    assert_eq!(
        restored.all_ids(handle).unwrap(),
        judge(&enc, &expected_text)
    );
    assert_eq!(
        patch.token_count as usize,
        judge(&enc, &expected_text).len()
    );

    let mut changed = expected_text;
    changed.push('!');
    assert!(restored
        .verify_recovery_binding(handle, &changed, &binding)
        .is_err());
}

#[test]
fn session_record_roundtrip_and_post_import_append() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(64)).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let mut acc = "Persistent session content. ".repeat(15);
    let put = store.put(kid, &acc, &enc).unwrap();
    let out = store
        .append(
            put.handle,
            " with a tail delta \u{1f642}",
            put.revision,
            &enc,
        )
        .unwrap();
    acc.push_str(" with a tail delta \u{1f642}");

    let rec = store.export_session(put.handle).unwrap();
    let mut store2 = SessionStore::new(cfg(64)).unwrap();
    let kid2 = store2.register_fingerprint(fp(1), 0).unwrap();
    for (k, r) in store.export_node_items().unwrap() {
        assert!(store2.import_node_item(&k, &r));
    }
    let h2 = store2.import_session(kid2, &rec, &enc).unwrap();
    assert_eq!(
        store2.all_ids(h2).unwrap(),
        store.all_ids(put.handle).unwrap()
    );
    assert_eq!(store2.revision(h2).unwrap(), out.revision);
    assert_eq!(
        store2.export_node_items().unwrap(),
        store.export_node_items().unwrap()
    );
    let out2 = store2
        .append(h2, " post-load delta.", out.revision, &enc)
        .unwrap();
    acc.push_str(" post-load delta.");
    assert_eq!(out2.all_ids, judge(&enc, &acc));
}

#[test]
fn corrupted_records_are_rejected_with_contract_codes() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(32)).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let text = "prompt text for corruption test. ".repeat(8);
    let h = store.put(kid, &text, &enc).unwrap().handle;
    let rec = store.export_session(h).unwrap();
    let nodes = store.export_node_items().unwrap();
    assert!(!nodes.is_empty());

    let mut fresh = SessionStore::new(cfg(32)).unwrap();
    let kf = fresh.register_fingerprint(fp(1), 0).unwrap();

    // Bit flips anywhere in a session record: loud, corruption-coded.
    for ix in (0..rec.len()).step_by(11) {
        let mut bad = rec.clone();
        bad[ix] ^= 1;
        let err = fresh.import_session(kf, &bad, &enc).unwrap_err();
        assert!(
            err.is_rejection(),
            "flip at {ix} produced non-rejection error {err:?}"
        );
    }
    // Truncations: loud, corruption-coded.
    for cut in [0usize, 5, 100, rec.len() - 1] {
        let err = fresh.import_session(kf, &rec[..cut], &enc).unwrap_err();
        assert!(err.is_rejection(), "truncation at {cut} not rejected");
    }
    // Wrong key: fingerprint mismatch is loud on explicit import.
    let kw = fresh.register_fingerprint(fp(7), 0).unwrap();
    assert!(matches!(
        fresh.import_session(kw, &rec, &enc).unwrap_err(),
        StoreError::FingerprintMismatch
    ));
    // Witness mismatch on import: rejected.
    let uncertified = MockEncoder {
        certify: false,
        ..MockEncoder::default()
    };
    assert!(matches!(
        fresh.import_session(kf, &rec, &uncertified).unwrap_err(),
        StoreError::WitnessCategoryMismatch { .. }
    ));
    // Node records: corrupt imports are silent misses, counted.
    let before = fresh.stats().import_rejects;
    let (nk, nrec) = &nodes[0];
    let mut bad = nrec.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 1;
    assert!(!fresh.import_node_item(nk, &bad));
    assert!(fresh.stats().import_rejects > before);
    // Good imports still work afterwards.
    for (k, r) in &nodes {
        assert!(fresh.import_node_item(k, r));
    }
    let h2 = fresh.import_session(kf, &rec, &enc).unwrap();
    assert_eq!(fresh.all_ids(h2).unwrap(), store.all_ids(h).unwrap());
    assert!(fresh.lookup(kf, &text, &enc).unwrap().is_some());
}

// ------------------------------------------------- mini differential ---

/// Small linear congruential generator (deterministic, dependency-free).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next() as usize) % items.len()]
    }
}

const PIECES: [&str; 10] = [
    "hello world ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "\r\n",
    "\u{4f60}\u{597d}\u{4e16}\u{754c}",
    "words, punctuation! and numbers 12345 ",
    " ",
    "e\u{301}e\u{301}e\u{301} combining seams ",
    "\u{1f642}\u{1f680}\u{1f9ea}",
    "short",
    "The quick brown fox jumps over the lazy dog. ",
];

/// Randomized sequences of put/append/fork/lookup against the
/// full-re-encode judge; every step must be bit-exact and every hit
/// must verify (this is the hermetic miniature of the differential
/// battery; the full-scale run against the prototype implementation
/// was performed before this port was adopted).
#[test]
fn randomized_sessions_match_reference() {
    let enc = MockEncoder::default();
    for seed in 0..40u64 {
        let mut rng = Lcg(seed.wrapping_mul(0x9e3779b97f4a7c15) + 1);
        let mut store = SessionStore::new(StoreConfig {
            block_chars: 32,
            tail_soft_cap_bytes: 128,
            tail_hard_cap_bytes: 4096,
            node_tail_cap_bytes: 4096,
            max_sessions: 64,
        })
        .unwrap();
        let kid = store.register_fingerprint(fp(1), 0).unwrap();
        // (handle, accumulated text, revision)
        let mut live: Vec<(toktier_store_core::SessionHandle, String, u64)> = Vec::new();
        for _step in 0..60 {
            match rng.next() % 5 {
                0 => {
                    let text = PIECES[..(1 + rng.next() as usize % 4)].join("");
                    let put = store.put(kid, &text, &enc).unwrap();
                    assert_eq!(store.all_ids(put.handle).unwrap(), judge(&enc, &text));
                    live.push((put.handle, text, put.revision));
                }
                1 | 2 if !live.is_empty() => {
                    let ix = rng.next() as usize % live.len();
                    let delta = (*rng.pick(&PIECES)).to_string();
                    let (h, acc, rev) = live[ix].clone();
                    let old_all = store.all_ids(h).unwrap();
                    let out = store.append(h, &delta, rev, &enc).unwrap();
                    let rf = usize::try_from(out.replace_from).unwrap();
                    assert_eq!(&out.all_ids[..rf], &old_all[..rf]);
                    let mut rebuilt = old_all[..rf].to_vec();
                    rebuilt.extend_from_slice(&out.replacement_ids);
                    assert_eq!(rebuilt, out.all_ids);
                    let acc2 = format!("{acc}{delta}");
                    assert_eq!(out.all_ids, judge(&enc, &acc2), "seed={seed}");
                    live[ix] = (h, acc2, out.revision);
                }
                3 if !live.is_empty() => {
                    let ix = rng.next() as usize % live.len();
                    let (h, acc, _rev) = live[ix].clone();
                    let h2 = store.fork(h).unwrap();
                    assert_eq!(store.all_ids(h2).unwrap(), judge(&enc, &acc));
                    live.push((h2, acc, 0));
                }
                4 if !live.is_empty() => {
                    let ix = rng.next() as usize % live.len();
                    let query = live[ix].1.clone();
                    if let Some(hit) = store.lookup(kid, &query, &enc).unwrap() {
                        let matched = usize::try_from(hit.matched_chars).unwrap();
                        let rest: String = query.chars().skip(matched).collect();
                        let out = store.append(hit.handle, &rest, hit.revision, &enc).unwrap();
                        assert_eq!(out.all_ids, judge(&enc, &query), "seed={seed}");
                        live.push((hit.handle, query, out.revision));
                    }
                }
                _ => {}
            }
        }
    }
}

#[test]
fn sidecar_import_restores_exact_chain_behavior() {
    // Two stores driven identically; one round-trips through
    // record + sidecar mid-way. Afterwards both must keep producing
    // identical node tables and identical lookup behavior.
    let enc = MockEncoder::default();
    let drive = |store: &mut SessionStore, roundtrip: bool| -> (Vec<u32>, Vec<u8>) {
        let kid = store.register_fingerprint(fp(1), 0).unwrap();
        let mut acc = "A prefix shared and sealed across blocks. ".repeat(6);
        let put = store.put(kid, &acc, &enc).unwrap();
        let mut handle = put.handle;
        let mut rev = put.revision;
        let out = store
            .append(handle, " middle words with boundaries. ", rev, &enc)
            .unwrap();
        acc.push_str(" middle words with boundaries. ");
        rev = out.revision;
        if roundtrip {
            let rec = store.export_session(handle).unwrap();
            let sc = store.export_session_sidecar(handle).unwrap();
            assert!(store.evict(handle));
            handle = store
                .import_session_with_sidecar(kid, &rec, &sc, &enc)
                .unwrap();
            assert_eq!(store.revision(handle).unwrap(), rev);
            assert!(store.session_info(handle).unwrap().chain_ok);
        }
        // Enough further text to complete more blocks post-import.
        let more = "continuing text that completes additional blocks. ".repeat(4);
        let out2 = store.append(handle, &more, rev, &enc).unwrap();
        acc.push_str(&more);
        assert_eq!(out2.all_ids, judge(&enc, &acc));
        let node_bytes: Vec<u8> = store
            .export_node_items()
            .unwrap()
            .into_iter()
            .flat_map(|(k, r)| k.into_iter().chain(r))
            .collect();
        // Lookup of the full accumulated text must hit identically.
        let hit = store.lookup(kid, &acc, &enc).unwrap();
        let matched = hit.map(|h| h.matched_chars);
        (
            vec![matched.unwrap_or(0) as u32, store.stats().node_count as u32],
            node_bytes,
        )
    };
    let mut plain = SessionStore::new(cfg(64)).unwrap();
    let mut tripped = SessionStore::new(cfg(64)).unwrap();
    let (a_meta, a_nodes) = drive(&mut plain, false);
    let (b_meta, b_nodes) = drive(&mut tripped, true);
    assert_eq!(a_meta, b_meta, "matched chars / node counts diverged");
    assert_eq!(
        a_nodes, b_nodes,
        "node tables diverged after sidecar import"
    );
}

#[test]
fn bare_record_import_is_conservative_but_correct() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(64)).unwrap();
    let kid = store.register_fingerprint(fp(1), 0).unwrap();
    let mut acc = "Sealed content across several blocks here. ".repeat(8);
    let put = store.put(kid, &acc, &enc).unwrap();
    let out = store
        .append(put.handle, " and a tail.", put.revision, &enc)
        .unwrap();
    acc.push_str(" and a tail.");
    let rec = store.export_session(put.handle).unwrap();

    let mut fresh = SessionStore::new(cfg(64)).unwrap();
    let kf = fresh.register_fingerprint(fp(1), 0).unwrap();
    let h = fresh.import_session(kf, &rec, &enc).unwrap();
    // Full stream and revision survive; chain is detached.
    assert_eq!(fresh.all_ids(h).unwrap(), judge(&enc, &acc));
    assert_eq!(fresh.revision(h).unwrap(), out.revision);
    let info = fresh.session_info(h).unwrap();
    assert!(!info.chain_ok);
    assert_eq!(
        info.stable_prefix_bytes,
        store.session_info(put.handle).unwrap().stable_prefix_bytes
    );
    // Appends after a conservative import stay bit-exact.
    let out2 = fresh
        .append(h, " more words after import.", out.revision, &enc)
        .unwrap();
    acc.push_str(" more words after import.");
    assert_eq!(out2.all_ids, judge(&enc, &acc));
}

/// A1 oracle: the session's incrementally maintained record hash must equal
/// a from-scratch recomputation over the complete exported ID stream
/// (`compute_curr` rehashes every ID through `payload_digest_parts`).
/// Returns the record hash so callers can also check chain linkage.
fn assert_commit_hash_matches_full_recompute(
    store: &SessionStore,
    handle: toktier_store_core::SessionHandle,
) -> ([u8; 32], [u8; 32]) {
    let material = store
        .recovery_material(handle)
        .unwrap()
        .expect("oracle sessions carry recovery material");
    let record = SessionRecordV1::from_bytes(&store.export_session(handle).unwrap()).unwrap();
    assert_eq!(
        material.record_hash,
        record.compute_curr(),
        "incremental sealed-prefix digest diverged from the full recomputation"
    );
    (material.record_hash, record.prev_block_hash)
}

#[test]
fn incremental_commit_hash_survives_seed_seal_fork_overwrite_and_lookup() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(32)).unwrap();
    store.enable_recovery_tracking().unwrap();
    store.enable_content_tracking().unwrap();
    let kid = store.register_fingerprint(fp(3), 0).unwrap();

    // Seed with a multi-block prose text so seals and chain nodes exist.
    let mut acc = "The committee discussed the internationalization proposal in depth. ".repeat(6);
    let put = store.put(kid, &acc, &enc).unwrap();
    let (mut last_hash, _) = assert_commit_hash_matches_full_recompute(&store, put.handle);

    // Appends that advance the certified seal point; every revision must
    // chain onto the previous incremental hash and match the full oracle.
    let mut revision = put.revision;
    for delta in [
        "Another paragraph of ordinary prose follows the discussion. ",
        "\u{4f60}\u{597d}, mixed \u{1f642} content with CRLF\r\nand cafe\u{301} marks. ",
        "tail",
        "",
    ] {
        let out = store.append(put.handle, delta, revision, &enc).unwrap();
        revision = out.revision;
        acc.push_str(delta);
        let (hash, prev) = assert_commit_hash_matches_full_recompute(&store, put.handle);
        assert_eq!(
            prev, last_hash,
            "revision chain must link the previous commit"
        );
        last_hash = hash;
    }
    assert!(
        store.stats().seals > 0,
        "the lineage must advance seal points"
    );

    // Fork: the cloned hasher state must keep producing oracle-equal hashes
    // on both lineages without cross-talk.
    let fork = store.fork(put.handle).unwrap();
    assert_commit_hash_matches_full_recompute(&store, fork);
    store.append(fork, " fork-only growth. ", 0, &enc).unwrap();
    assert_commit_hash_matches_full_recompute(&store, fork);
    let (unchanged, _) = assert_commit_hash_matches_full_recompute(&store, put.handle);
    assert_eq!(
        unchanged, last_hash,
        "fork appends must not move the source"
    );

    // Overwrite shape (evict + fresh genesis put under the same name).
    store.evict(put.handle);
    let replaced = store
        .put(kid, "a fresh genesis stream after overwrite", &enc)
        .unwrap();
    assert_commit_hash_matches_full_recompute(&store, replaced.handle);

    // Lookup materialization rebuilds the sealed prefix from chain nodes;
    // its rebuilt hasher state must agree with the oracle before and after
    // a subsequent append.
    let query = format!("{acc} lookup continuation beyond the sealed blocks.");
    let hit = store.lookup(kid, &query, &enc).unwrap().expect("must hit");
    assert_commit_hash_matches_full_recompute(&store, hit.handle);
    store
        .append(hit.handle, " appended after lookup.", hit.revision, &enc)
        .unwrap();
    assert_commit_hash_matches_full_recompute(&store, hit.handle);
}

#[test]
fn incremental_commit_hash_survives_import_paths() {
    let enc = MockEncoder::default();
    let mut source = SessionStore::new(cfg(32)).unwrap();
    source.enable_recovery_tracking().unwrap();
    source.enable_content_tracking().unwrap();
    let kid = source.register_fingerprint(fp(4), 0).unwrap();
    let mut text = "Sealed content across several blocks for the import oracle. ".repeat(6);
    let put = source.put(kid, &text, &enc).unwrap();
    let out = source.append(put.handle, "and a tail.", 0, &enc).unwrap();
    text.push_str("and a tail.");
    let record = source.export_session(put.handle).unwrap();
    let sidecar = source.export_session_sidecar(put.handle).unwrap();
    let expected = source.recovery_material(put.handle).unwrap().unwrap();
    let binding = source
        .export_recovery_binding(put.handle)
        .unwrap()
        .expect("tracked session exports a binding");

    // Recovery-aware record import: the rebuilt sealed prefix state must
    // hash oracle-equal on the next commits.
    let mut restored = SessionStore::new(cfg(32)).unwrap();
    restored.enable_recovery_tracking().unwrap();
    restored.enable_content_tracking().unwrap();
    let rk = restored.register_fingerprint(fp(4), 0).unwrap();
    let handle = restored
        .import_session_with_recovery(rk, &record, &text, &expected, &enc)
        .unwrap();
    let (hash, _) = assert_commit_hash_matches_full_recompute(&restored, handle);
    assert_eq!(
        hash, expected.record_hash,
        "import must reproduce the record hash"
    );
    let grown = restored
        .append(
            handle,
            " appended after recovery import.",
            out.revision,
            &enc,
        )
        .unwrap();
    assert!(grown.revision > out.revision);
    assert_commit_hash_matches_full_recompute(&restored, handle);

    // Sidecar import plus TKFR binding restoration: same obligation.
    let mut exact = SessionStore::new(cfg(32)).unwrap();
    exact.enable_recovery_tracking().unwrap();
    exact.enable_content_tracking().unwrap();
    let ek = exact.register_fingerprint(fp(4), 0).unwrap();
    let eh = exact
        .import_session_with_sidecar(ek, &record, &sidecar, &enc)
        .unwrap();
    exact
        .restore_tracking_with_binding(eh, &text, &binding)
        .unwrap();
    let (ehash, _) = assert_commit_hash_matches_full_recompute(&exact, eh);
    assert_eq!(ehash, expected.record_hash);
    exact
        .append(eh, " appended after sidecar import.", out.revision, &enc)
        .unwrap();
    assert_commit_hash_matches_full_recompute(&exact, eh);
}

// ------------------------------------------------------------------
// Shared-buffer ownership (PLAN/162 WP2/WP4): engine allocation adoption,
// generation-keyed snapshots, and copy-behavior counters.
// ------------------------------------------------------------------

use std::cell::Cell;
use toktier_store_core::{AppendReport, BoundaryCut, Encoding, EngineError, SharedIds, TailState};

/// A mock engine whose cold seed adopts a shared ID allocation, modeling
/// the native router's zero-copy state seed. Later appends and boundary
/// probes delegate to the ordinary mock encoder.
struct SharedSeedEncoder {
    inner: MockEncoder,
    /// Address of the last seed allocation this engine produced, so a
    /// test can prove the store and snapshot observe the same memory.
    last_seed_ptr: Cell<usize>,
}

impl SharedSeedEncoder {
    fn new() -> SharedSeedEncoder {
        SharedSeedEncoder {
            inner: MockEncoder::default(),
            last_seed_ptr: Cell::new(0),
        }
    }
}

impl SessionEncoder for SharedSeedEncoder {
    fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
        self.inner.encode(text)
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        if !tail.text().is_empty() {
            return self.inner.append(tail, delta);
        }
        let enc = self.inner.encode(delta)?;
        let (starts, ends): (Vec<u32>, Vec<u32>) = enc.spans.iter().copied().unzip();
        let shared = SharedIds::from_vec(enc.ids);
        self.last_seed_ptr.set(shared.as_slice().as_ptr() as usize);
        tail.fill_shared(delta, shared, starts, ends)
            .map_err(|error| EngineError(error.to_string()))?;
        Ok(AppendReport {
            path: "cold_full_shared".to_owned(),
            kept_tokens: 0,
        })
    }

    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        self.inner
            .last_certified_boundary(tail, floor_char, ceil_char)
    }

    fn witness_category(&self) -> WitnessCategory {
        self.inner.witness_category()
    }
}

#[test]
fn shared_seed_seal_and_snapshot_observe_one_allocation() {
    let enc = SharedSeedEncoder::new();
    let mut store = SessionStore::new(cfg(8)).unwrap();
    let key = store.register_fingerprint(fp(41), 0).unwrap();
    let text = "the quick brown fox jumps over the lazy dog 1234 end";
    let put = store.put(key, text, &enc).unwrap();
    // The seed sealed a stable prefix and left a residual tail.
    let stats = store.stats();
    assert!(stats.seals > 0, "expected the seed to seal");
    let expected = judge(&enc.inner, text);
    let before = store.ids_materialization_count();
    let snapshot = store.shared_all_ids(put.handle).unwrap();
    assert_eq!(snapshot.as_slice(), &expected[..]);
    // The complete row is the engine's own seed allocation: sealed head
    // and mutable tail are two adjacent ranges of it, so the snapshot
    // joined them without materializing anything.
    assert_eq!(
        snapshot.as_slice().as_ptr() as usize,
        enc.last_seed_ptr.get(),
        "snapshot does not share the engine allocation"
    );
    assert_eq!(store.ids_materialization_count(), before);
    // A repeated snapshot of the unchanged stream is the cached row.
    let again = store.shared_all_ids(put.handle).unwrap();
    assert!(again.same_allocation(&snapshot));
    assert_eq!(store.ids_materialization_count(), before);
}

#[test]
fn snapshots_are_generation_keyed_and_stable_across_mutation() {
    let enc = SharedSeedEncoder::new();
    let mut store = SessionStore::new(cfg(8)).unwrap();
    let key = store.register_fingerprint(fp(42), 0).unwrap();
    let text = "alpha beta gamma delta 42 epsilon zeta";
    let put = store.put(key, text, &enc).unwrap();
    let old_row = judge(&enc.inner, text);
    let old_snapshot = store.shared_all_ids(put.handle).unwrap();
    assert_eq!(old_snapshot.as_slice(), &old_row[..]);

    // Mutate the session; the retained snapshot must not change.
    let patch = store
        .append_patch(put.handle, " and more words follow", 0, &enc)
        .unwrap();
    assert_eq!(old_snapshot.as_slice(), &old_row[..], "stale row mutated");
    let new_row = judge(
        &enc.inner,
        "alpha beta gamma delta 42 epsilon zeta and more words follow",
    );
    let new_snapshot = store.shared_all_ids(put.handle).unwrap();
    assert_eq!(new_snapshot.as_slice(), &new_row[..]);
    assert!(!new_snapshot.same_allocation(&old_snapshot));
    // Patch invariant against the two snapshots.
    let rf = usize::try_from(patch.replace_from).unwrap();
    let mut rebuilt = old_row[..rf].to_vec();
    rebuilt.extend_from_slice(&patch.replacement_ids);
    assert_eq!(rebuilt, new_row);

    // The old snapshot outlives even session eviction.
    assert!(store.evict(put.handle));
    assert_eq!(old_snapshot.as_slice(), &old_row[..]);
}

#[test]
fn noop_append_keeps_the_snapshot_and_fork_shares_it() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(8)).unwrap();
    let key = store.register_fingerprint(fp(43), 0).unwrap();
    let put = store.put(key, "shared snapshot text 99", &enc).unwrap();
    let snapshot = store.shared_all_ids(put.handle).unwrap();
    // A noop append does not change content; the cache stays valid.
    store.append_patch(put.handle, "", 0, &enc).unwrap();
    let after_noop = store.shared_all_ids(put.handle).unwrap();
    assert!(after_noop.same_allocation(&snapshot));
    // A fork carries the still-valid cache; divergence separates them.
    let forked = store.fork(put.handle).unwrap();
    let fork_snapshot = store.shared_all_ids(forked).unwrap();
    assert!(fork_snapshot.same_allocation(&snapshot));
    store.append_patch(forked, " diverged", 0, &enc).unwrap();
    let diverged = store.shared_all_ids(forked).unwrap();
    assert!(!diverged.same_allocation(&snapshot));
    assert_eq!(
        store.shared_all_ids(put.handle).unwrap().as_slice(),
        snapshot.as_slice(),
        "source stream disturbed by fork divergence"
    );
}

#[test]
fn short_appends_do_not_materialize_the_full_row() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(8)).unwrap();
    let key = store.register_fingerprint(fp(44), 0).unwrap();
    let put = store.put(key, "a long enough seed text 123", &enc).unwrap();
    let baseline = store.ids_materialization_count();
    let mut revision = put.revision;
    for delta in DELTAS {
        let patch = store
            .append_patch(put.handle, delta, revision, &enc)
            .unwrap();
        revision = patch.revision;
    }
    assert_eq!(
        store.ids_materialization_count(),
        baseline,
        "a patch-only append materialized the historical row"
    );
    // Explicit materialization is counted.
    store.all_ids(put.handle).unwrap();
    assert_eq!(store.ids_materialization_count(), baseline + 1);
}

#[test]
fn shared_all_ids_agrees_with_all_ids_everywhere() {
    let enc = MockEncoder::default();
    let mut store = SessionStore::new(cfg(8)).unwrap();
    let key = store.register_fingerprint(fp(45), 0).unwrap();
    let put = store
        .put(key, "cross check body with words 5678", &enc)
        .unwrap();
    let mut revision = put.revision;
    for delta in DELTAS {
        assert_eq!(
            store.shared_all_ids(put.handle).unwrap().as_slice(),
            &store.all_ids(put.handle).unwrap()[..]
        );
        revision = store
            .append_patch(put.handle, delta, revision, &enc)
            .unwrap()
            .revision;
    }
    assert_eq!(
        store.shared_all_ids(put.handle).unwrap().as_slice(),
        &store.all_ids(put.handle).unwrap()[..]
    );
}

// ------------------------------------------------------------------
// Lazy spans (PLAN/162 WP1.5): sparse checkpoints through seed, seal,
// snapshot, and failure injection.
// ------------------------------------------------------------------

use std::sync::Arc;

/// One token per byte over ASCII text (id == byte value), adopting the
/// shared allocation and lazy checkpointed spans on a cold seed. This
/// models the native router's lazy state seed with a dense toy table.
struct LazyByteEncoder {
    table: Arc<[usize]>,
    last_seed_ptr: Cell<usize>,
    /// When set, the boundary probe reports this injected failure.
    fail_boundary: bool,
}

impl LazyByteEncoder {
    fn new() -> LazyByteEncoder {
        LazyByteEncoder {
            table: vec![1usize; 256].into(),
            last_seed_ptr: Cell::new(0),
            fail_boundary: false,
        }
    }

    fn byte_ids(text: &str) -> Vec<u32> {
        assert!(text.is_ascii(), "LazyByteEncoder is ASCII-only");
        text.bytes().map(u32::from).collect()
    }
}

impl SessionEncoder for LazyByteEncoder {
    fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
        let ids = LazyByteEncoder::byte_ids(text);
        let spans = (0..ids.len() as u32).map(|i| (i, i + 1)).collect();
        Ok(Encoding { ids, spans })
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        if tail.text().is_empty() {
            let shared = SharedIds::from_vec(LazyByteEncoder::byte_ids(delta));
            self.last_seed_ptr.set(shared.as_slice().as_ptr() as usize);
            tail.fill_lazy(delta, shared, Arc::clone(&self.table))
                .map_err(|error| EngineError(error.to_string()))?;
            return Ok(AppendReport {
                path: "cold_full_lazy".to_owned(),
                kept_tokens: 0,
            });
        }
        let kept = tail.n_tokens();
        let mut full = String::with_capacity(tail.text_bytes() + delta.len());
        full.push_str(tail.text());
        full.push_str(delta);
        let enc = self.encode(&full)?;
        tail.fill(&full, enc)
            .map_err(|error| EngineError(error.to_string()))?;
        Ok(AppendReport {
            path: "byte_full_reencode".to_owned(),
            kept_tokens: kept,
        })
    }

    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        if self.fail_boundary {
            return Err(EngineError("injected boundary failure".to_owned()));
        }
        // One token per character: a space-to-nonspace transition is a
        // sound cut. Scan backward through a small span window per step
        // so the probe never forces whole-row materialization.
        let text = tail.text().as_bytes();
        let n = tail.n_tokens();
        let ceiling = usize::try_from(ceil_char.min(text.len() as u64)).unwrap();
        let floor = usize::try_from(floor_char).unwrap();
        for boundary in (floor + 1..=ceiling.min(n.saturating_sub(1))).rev() {
            if text[boundary - 1] == b' ' && text[boundary] != b' ' {
                // Cross-check the window API against the per-byte layout.
                let (starts, ends) = tail
                    .span_window(boundary - 1, boundary + 1)
                    .map_err(|error| EngineError(error.to_string()))?;
                assert_eq!(starts[1] as usize, boundary);
                assert_eq!(ends[0] as usize, boundary);
                return Ok(Some(BoundaryCut {
                    cut_tokens: boundary,
                    cut_char: boundary as u64,
                }));
            }
        }
        Ok(None)
    }

    fn witness_category(&self) -> WitnessCategory {
        WitnessCategory::BpeSyncTransition
    }
}

#[test]
fn lazy_seed_seals_and_shares_the_engine_allocation_end_to_end() {
    let enc = LazyByteEncoder::new();
    let mut store = SessionStore::new(cfg(8)).unwrap();
    let key = store.register_fingerprint(fp(50), 0).unwrap();
    let text = "lazy checkpointed seed with several words and a tail";
    let put = store.put(key, text, &enc).unwrap();
    assert!(store.stats().seals > 0, "expected the lazy seed to seal");
    let expected = LazyByteEncoder::byte_ids(text);
    let before = store.ids_materialization_count();
    let snapshot = store.shared_all_ids(put.handle).unwrap();
    assert_eq!(snapshot.as_slice(), &expected[..]);
    assert_eq!(
        snapshot.as_slice().as_ptr() as usize,
        enc.last_seed_ptr.get(),
        "lazy seed snapshot does not share the engine allocation"
    );
    assert_eq!(store.ids_materialization_count(), before);
    // Appends after the seal keep the exact stream.
    let patch = store
        .append_patch(put.handle, " more words", 0, &enc)
        .unwrap();
    let full = LazyByteEncoder::byte_ids(
        "lazy checkpointed seed with several words and a tail more words",
    );
    let rf = usize::try_from(patch.replace_from).unwrap();
    let mut rebuilt = expected[..rf].to_vec();
    rebuilt.extend_from_slice(&patch.replacement_ids);
    assert_eq!(rebuilt, full);
    assert_eq!(
        store.shared_all_ids(put.handle).unwrap().as_slice(),
        &full[..]
    );
    // The pre-append snapshot is untouched.
    assert_eq!(snapshot.as_slice(), &expected[..]);
}

#[test]
fn injected_boundary_failure_leaves_no_partially_visible_session() {
    let mut enc = LazyByteEncoder::new();
    enc.fail_boundary = true;
    let mut store = SessionStore::new(cfg(8)).unwrap();
    let key = store.register_fingerprint(fp(51), 0).unwrap();
    let result = store.put(key, "words that would otherwise seal here", &enc);
    assert!(result.is_err(), "injected failure must surface");
    assert!(store.list_handles().is_empty(), "partial session visible");
    // The store remains fully usable afterwards.
    enc.fail_boundary = false;
    let put = store
        .put(key, "words that would otherwise seal here", &enc)
        .unwrap();
    assert_eq!(
        store.shared_all_ids(put.handle).unwrap().as_slice(),
        &LazyByteEncoder::byte_ids("words that would otherwise seal here")[..]
    );
}

#[test]
fn non_closing_lazy_payload_fails_before_any_state_is_visible() {
    /// An engine whose seed lies about closure: the lazy fill must
    /// reject it inside `append`, and `put` must not insert a session.
    struct LyingEncoder(LazyByteEncoder);
    impl SessionEncoder for LyingEncoder {
        fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
            self.0.encode(text)
        }
        fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
            let mut ids = LazyByteEncoder::byte_ids(delta);
            ids.pop(); // no longer closes over delta
            tail.fill_lazy(delta, SharedIds::from_vec(ids), Arc::clone(&self.0.table))
                .map_err(|error| EngineError(error.to_string()))?;
            Ok(AppendReport {
                path: "lying".to_owned(),
                kept_tokens: 0,
            })
        }
        fn last_certified_boundary(
            &self,
            tail: &TailState,
            floor_char: u64,
            ceil_char: u64,
        ) -> Result<Option<BoundaryCut>, EngineError> {
            self.0.last_certified_boundary(tail, floor_char, ceil_char)
        }
        fn witness_category(&self) -> WitnessCategory {
            self.0.witness_category()
        }
    }
    let enc = LyingEncoder(LazyByteEncoder::new());
    let mut store = SessionStore::new(cfg(8)).unwrap();
    let key = store.register_fingerprint(fp(52), 0).unwrap();
    let result = store.put(key, "text the row does not close over", &enc);
    assert!(result.is_err());
    assert!(store.list_handles().is_empty());
}

#[test]
fn lazy_and_materialized_seeds_produce_identical_sessions() {
    let lazy_enc = LazyByteEncoder::new();
    let text = "twin sessions must agree exactly word for word 12345";
    let mut lazy_store = SessionStore::new(cfg(8)).unwrap();
    let lazy_key = lazy_store.register_fingerprint(fp(53), 0).unwrap();
    let lazy_put = lazy_store.put(lazy_key, text, &lazy_enc).unwrap();

    /// The same toy tokenizer, but filling materialized SoA spans.
    struct SoaByteEncoder(LazyByteEncoder);
    impl SessionEncoder for SoaByteEncoder {
        fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
            self.0.encode(text)
        }
        fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
            let enc = self.0.encode(delta)?;
            if tail.text().is_empty() {
                let (starts, ends) = enc.spans.iter().copied().unzip();
                tail.fill_soa(
                    delta,
                    toktier_store_core::SoaEncoding {
                        ids: enc.ids,
                        span_starts: starts,
                        span_ends: ends,
                    },
                )
                .map_err(|error| EngineError(error.to_string()))?;
                return Ok(AppendReport {
                    path: "cold_full_soa".to_owned(),
                    kept_tokens: 0,
                });
            }
            self.0.append(tail, delta)
        }
        fn last_certified_boundary(
            &self,
            tail: &TailState,
            floor_char: u64,
            ceil_char: u64,
        ) -> Result<Option<BoundaryCut>, EngineError> {
            self.0.last_certified_boundary(tail, floor_char, ceil_char)
        }
        fn witness_category(&self) -> WitnessCategory {
            self.0.witness_category()
        }
    }
    let soa_enc = SoaByteEncoder(LazyByteEncoder::new());
    let mut soa_store = SessionStore::new(cfg(8)).unwrap();
    let soa_key = soa_store.register_fingerprint(fp(53), 0).unwrap();
    let soa_put = soa_store.put(soa_key, text, &soa_enc).unwrap();

    // Same stream, same record bytes, same sidecar, same seal counters.
    assert_eq!(
        lazy_store.all_ids(lazy_put.handle).unwrap(),
        soa_store.all_ids(soa_put.handle).unwrap()
    );
    assert_eq!(
        lazy_store.export_session(lazy_put.handle).unwrap(),
        soa_store.export_session(soa_put.handle).unwrap()
    );
    assert_eq!(
        lazy_store.export_session_sidecar(lazy_put.handle).unwrap(),
        soa_store.export_session_sidecar(soa_put.handle).unwrap()
    );
    assert_eq!(lazy_store.stats().seals, soa_store.stats().seals);
    // And the two evolve identically under the same appends.
    for (index, delta) in [" one", " two words", " three more words"]
        .iter()
        .enumerate()
    {
        let a = lazy_store
            .append_patch(lazy_put.handle, delta, index as u64, &lazy_enc)
            .unwrap();
        let b = soa_store
            .append_patch(soa_put.handle, delta, index as u64, &soa_enc)
            .unwrap();
        assert_eq!(a.replace_from, b.replace_from);
        assert_eq!(a.replacement_ids, b.replacement_ids);
        assert_eq!(
            lazy_store.export_session(lazy_put.handle).unwrap(),
            soa_store.export_session(soa_put.handle).unwrap()
        );
    }
}

// ------------------------------------------------------------------
// Seed overlap (PLAN/162 WP5/WP6): the content-digest scan joins the
// seed encode on a bounded worker. The digest bytes, the failure
// ordering, and the atomicity of the insertion must be identical to
// the serial path.
// ------------------------------------------------------------------

use toktier_store_core::{ContentDigest, OverlapRunner};

/// Test overlap runner: one scoped thread per call. It is a test-only
/// stand-in for the production bounded pool; the store contract only
/// needs the join semantics.
struct ScopedOverlap;

impl OverlapRunner for ScopedOverlap {
    fn run_joined(&self, background: &mut (dyn FnMut() + Send), foreground: &mut dyn FnMut()) {
        std::thread::scope(|scope| {
            scope.spawn(background);
            foreground();
        });
    }

    fn worker_count(&self) -> usize {
        1
    }
}

/// A runner that skips one of the closures, for fail-closed coverage.
struct SkippingOverlap {
    run_background: bool,
    run_foreground: bool,
}

impl OverlapRunner for SkippingOverlap {
    fn run_joined(&self, background: &mut (dyn FnMut() + Send), foreground: &mut dyn FnMut()) {
        if self.run_background {
            background();
        }
        if self.run_foreground {
            foreground();
        }
    }

    fn worker_count(&self) -> usize {
        0
    }
}

fn content_store(block_chars: u64, overlap: bool) -> SessionStore {
    let mut store = SessionStore::new(cfg(block_chars)).unwrap();
    store.enable_content_tracking().unwrap();
    if overlap {
        store.set_seed_overlap(Some(Arc::new(ScopedOverlap)));
    }
    store
}

/// Exactly 4 MiB of ASCII text with word transitions, so the lazy
/// production seed shape (fill_lazy plus sealing) is exercised at the
/// frozen payload size.
fn four_mib_ascii() -> String {
    let mut text = "overlap seed words ".repeat((4 * 1024 * 1024) / 19 + 1);
    text.truncate(4 * 1024 * 1024);
    text
}

#[test]
fn overlap_seed_digest_bytes_match_the_serial_path() {
    // Unicode and boundary shapes through the deterministic mock
    // encoder: empty, single char, single token, ASCII, CRLF, emoji
    // ZWJ, combining marks, CJK, RTL.
    let cases = [
        "",
        "a",
        "abc",
        "plain ascii words over several pieces",
        "CRLF\r\nline",
        "emoji \u{1f469}\u{200d}\u{1f680} joined",
        "combining cafe\u{301} mark",
        "\u{4f60}\u{597d}\u{4e16}\u{754c} cjk",
        "rtl \u{05e9}\u{05dc}\u{05d5}\u{05dd} text",
    ];
    let enc = MockEncoder::default();
    for text in cases {
        let mut serial = content_store(8, false);
        let serial_key = serial.register_fingerprint(fp(60), 0).unwrap();
        let serial_put = serial.put(serial_key, text, &enc).unwrap();
        let mut overlap = content_store(8, true);
        let overlap_key = overlap.register_fingerprint(fp(60), 0).unwrap();
        let overlap_put = overlap.put(overlap_key, text, &enc).unwrap();
        assert_eq!(
            serial.content_index_entry(serial_put.handle).unwrap(),
            overlap.content_index_entry(overlap_put.handle).unwrap(),
            "content digest diverged for {text:?}"
        );
        assert_eq!(
            serial.content_index_entry(serial_put.handle).unwrap(),
            Some(ContentDigest::from_bytes(text.as_bytes()).unwrap().entry()),
            "content digest diverged from the direct scan for {text:?}"
        );
        assert_eq!(
            serial.all_ids(serial_put.handle).unwrap(),
            overlap.all_ids(overlap_put.handle).unwrap()
        );
        assert_eq!(
            serial.export_session(serial_put.handle).unwrap(),
            overlap.export_session(overlap_put.handle).unwrap()
        );
    }
}

#[test]
fn overlap_seed_digest_matches_serial_at_four_mib_with_ten_marks() {
    let text = four_mib_ascii();
    let serial_enc = LazyByteEncoder::new();
    let mut serial = SessionStore::new(StoreConfig::default()).unwrap();
    serial.enable_content_tracking().unwrap();
    let serial_key = serial.register_fingerprint(fp(61), 0).unwrap();
    let serial_put = serial.put(serial_key, &text, &serial_enc).unwrap();

    let overlap_enc = LazyByteEncoder::new();
    let mut overlap = SessionStore::new(StoreConfig::default()).unwrap();
    overlap.enable_content_tracking().unwrap();
    overlap.set_seed_overlap(Some(Arc::new(ScopedOverlap)));
    let overlap_key = overlap.register_fingerprint(fp(61), 0).unwrap();
    let overlap_put = overlap.put(overlap_key, &text, &overlap_enc).unwrap();

    let serial_entry = serial
        .content_index_entry(serial_put.handle)
        .unwrap()
        .unwrap();
    let overlap_entry = overlap
        .content_index_entry(overlap_put.handle)
        .unwrap()
        .unwrap();
    assert_eq!(serial_entry, overlap_entry);
    assert_eq!(
        serial_entry,
        ContentDigest::from_bytes(text.as_bytes()).unwrap().entry()
    );
    // The frozen 4 MiB payload carries the endpoint plus exactly ten
    // geometric checkpoints (4096 * 2^k below 4 MiB).
    assert_eq!(serial_entry.byte_length, 4 * 1024 * 1024);
    assert_eq!(serial_entry.marks.len(), 10);
    assert!(serial.stats().seals > 0, "expected the 4 MiB seed to seal");
    assert_eq!(serial.stats().seals, overlap.stats().seals);
    assert_eq!(
        serial.export_session(serial_put.handle).unwrap(),
        overlap.export_session(overlap_put.handle).unwrap()
    );
}

#[test]
fn overlap_seed_preserves_the_shared_allocation_and_seal() {
    let enc = LazyByteEncoder::new();
    let mut store = content_store(8, true);
    let key = store.register_fingerprint(fp(62), 0).unwrap();
    let text = "overlap keeps the zero copy adoption path fully intact";
    let put = store.put(key, text, &enc).unwrap();
    assert!(store.stats().seals > 0, "expected the seed to seal");
    let before = store.ids_materialization_count();
    let snapshot = store.shared_all_ids(put.handle).unwrap();
    assert_eq!(
        snapshot.as_slice().as_ptr() as usize,
        enc.last_seed_ptr.get(),
        "overlap seed snapshot does not share the engine allocation"
    );
    assert_eq!(store.ids_materialization_count(), before);
    assert_eq!(
        snapshot.as_slice(),
        &LazyByteEncoder::byte_ids(text)[..],
        "overlap seed stream diverged"
    );
}

/// An encoder whose seed encode fails after touching nothing.
struct FailingSeedEncoder(LazyByteEncoder);

impl SessionEncoder for FailingSeedEncoder {
    fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
        self.0.encode(text)
    }
    fn append(&self, _tail: &mut TailState, _delta: &str) -> Result<AppendReport, EngineError> {
        Err(EngineError("injected seed encode failure".to_owned()))
    }
    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        self.0.last_certified_boundary(tail, floor_char, ceil_char)
    }
    fn witness_category(&self) -> WitnessCategory {
        self.0.witness_category()
    }
}

#[test]
fn overlap_encode_failure_matches_serial_and_leaves_no_session() {
    let enc = FailingSeedEncoder(LazyByteEncoder::new());
    let text = "the encode side reports the failure";
    let mut serial = content_store(8, false);
    let serial_key = serial.register_fingerprint(fp(63), 0).unwrap();
    let serial_err = serial.put(serial_key, text, &enc).unwrap_err();
    let mut overlap = content_store(8, true);
    let overlap_key = overlap.register_fingerprint(fp(63), 0).unwrap();
    let overlap_err = overlap.put(overlap_key, text, &enc).unwrap_err();
    assert_eq!(serial_err, overlap_err, "error identity diverged");
    assert!(serial.list_handles().is_empty());
    assert!(overlap.list_handles().is_empty());
    // Both stores stay fully usable afterwards.
    let good = LazyByteEncoder::new();
    let retry = overlap.put(overlap_key, text, &good).unwrap();
    assert_eq!(
        overlap.shared_all_ids(retry.handle).unwrap().as_slice(),
        &LazyByteEncoder::byte_ids(text)[..]
    );
}

#[test]
fn overlap_digest_failure_matches_serial_and_leaves_no_session() {
    let text = "the digest side reports the failure";
    let mut serial = content_store(8, false);
    serial.inject_content_digest_fault(Some("injected digest failure".to_owned()));
    let serial_key = serial.register_fingerprint(fp(64), 0).unwrap();
    let serial_err = serial
        .put(serial_key, text, &LazyByteEncoder::new())
        .unwrap_err();
    let mut overlap = content_store(8, true);
    overlap.inject_content_digest_fault(Some("injected digest failure".to_owned()));
    let overlap_key = overlap.register_fingerprint(fp(64), 0).unwrap();
    let overlap_err = overlap
        .put(overlap_key, text, &LazyByteEncoder::new())
        .unwrap_err();
    assert_eq!(serial_err, overlap_err, "error identity diverged");
    assert!(serial.list_handles().is_empty(), "partial session visible");
    assert!(overlap.list_handles().is_empty(), "partial session visible");
    // Clearing the fault restores service on the same stores.
    overlap.inject_content_digest_fault(None);
    let retry = overlap
        .put(overlap_key, text, &LazyByteEncoder::new())
        .unwrap();
    assert_eq!(
        overlap.content_index_entry(retry.handle).unwrap(),
        Some(ContentDigest::from_bytes(text.as_bytes()).unwrap().entry())
    );
}

#[test]
fn overlap_double_failure_prefers_the_encode_error_like_serial() {
    let enc = FailingSeedEncoder(LazyByteEncoder::new());
    let text = "both sides fail; the encode error wins in both modes";
    let mut serial = content_store(8, false);
    serial.inject_content_digest_fault(Some("injected digest failure".to_owned()));
    let serial_key = serial.register_fingerprint(fp(65), 0).unwrap();
    let serial_err = serial.put(serial_key, text, &enc).unwrap_err();
    let mut overlap = content_store(8, true);
    overlap.inject_content_digest_fault(Some("injected digest failure".to_owned()));
    let overlap_key = overlap.register_fingerprint(fp(65), 0).unwrap();
    let overlap_err = overlap.put(overlap_key, text, &enc).unwrap_err();
    assert_eq!(
        serial_err,
        StoreError::Engine("injected seed encode failure".to_owned())
    );
    assert_eq!(serial_err, overlap_err, "failure precedence diverged");
    assert!(serial.list_handles().is_empty());
    assert!(overlap.list_handles().is_empty());
}

#[test]
fn overlap_boundary_failure_after_the_join_leaves_no_session() {
    let mut enc = LazyByteEncoder::new();
    enc.fail_boundary = true;
    let mut store = content_store(8, true);
    let key = store.register_fingerprint(fp(66), 0).unwrap();
    let result = store.put(key, "words that would otherwise seal here", &enc);
    assert!(result.is_err(), "injected failure must surface");
    assert!(store.list_handles().is_empty(), "partial session visible");
    enc.fail_boundary = false;
    let put = store
        .put(key, "words that would otherwise seal here", &enc)
        .unwrap();
    assert_eq!(
        store.content_index_entry(put.handle).unwrap(),
        Some(
            ContentDigest::from_bytes(b"words that would otherwise seal here")
                .unwrap()
                .entry()
        )
    );
}

#[test]
fn overlap_non_closing_lazy_payload_still_fails_before_any_state_is_visible() {
    /// Same lying-closure shape as the serial test: the lazy fill must
    /// reject inside the joined encode, and `put` must not insert.
    struct LyingEncoder(LazyByteEncoder);
    impl SessionEncoder for LyingEncoder {
        fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
            self.0.encode(text)
        }
        fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
            let mut ids = LazyByteEncoder::byte_ids(delta);
            ids.pop();
            tail.fill_lazy(delta, SharedIds::from_vec(ids), Arc::clone(&self.0.table))
                .map_err(|error| EngineError(error.to_string()))?;
            Ok(AppendReport {
                path: "lying".to_owned(),
                kept_tokens: 0,
            })
        }
        fn last_certified_boundary(
            &self,
            tail: &TailState,
            floor_char: u64,
            ceil_char: u64,
        ) -> Result<Option<BoundaryCut>, EngineError> {
            self.0.last_certified_boundary(tail, floor_char, ceil_char)
        }
        fn witness_category(&self) -> WitnessCategory {
            self.0.witness_category()
        }
    }
    let mut store = content_store(8, true);
    let key = store.register_fingerprint(fp(67), 0).unwrap();
    let result = store.put(key, "close over me", &LyingEncoder(LazyByteEncoder::new()));
    assert!(result.is_err(), "non-closing payload must be rejected");
    assert!(store.list_handles().is_empty(), "partial session visible");
}

#[test]
fn overlap_runner_that_skips_the_encode_fails_closed() {
    let mut store = content_store(8, false);
    store.set_seed_overlap(Some(Arc::new(SkippingOverlap {
        run_background: true,
        run_foreground: false,
    })));
    let key = store.register_fingerprint(fp(68), 0).unwrap();
    let err = store
        .put(key, "some text", &LazyByteEncoder::new())
        .unwrap_err();
    assert!(
        err.to_string().contains("skipped the foreground encode"),
        "unexpected error: {err}"
    );
    assert!(store.list_handles().is_empty(), "partial session visible");
}

#[test]
fn overlap_runner_that_skips_the_digest_falls_back_serially() {
    let mut store = content_store(8, false);
    store.set_seed_overlap(Some(Arc::new(SkippingOverlap {
        run_background: false,
        run_foreground: true,
    })));
    let key = store.register_fingerprint(fp(69), 0).unwrap();
    let text = "the digest is recomputed serially";
    let put = store.put(key, text, &LazyByteEncoder::new()).unwrap();
    assert_eq!(
        store.content_index_entry(put.handle).unwrap(),
        Some(ContentDigest::from_bytes(text.as_bytes()).unwrap().entry())
    );
}

#[test]
fn overlap_seed_with_empty_text_matches_serial() {
    let enc = MockEncoder::default();
    let mut serial = content_store(8, false);
    let serial_key = serial.register_fingerprint(fp(70), 0).unwrap();
    let serial_put = serial.put(serial_key, "", &enc).unwrap();
    let mut overlap = content_store(8, true);
    let overlap_key = overlap.register_fingerprint(fp(70), 0).unwrap();
    let overlap_put = overlap.put(overlap_key, "", &enc).unwrap();
    assert_eq!(
        serial.content_index_entry(serial_put.handle).unwrap(),
        overlap.content_index_entry(overlap_put.handle).unwrap()
    );
    assert_eq!(
        serial.export_session(serial_put.handle).unwrap(),
        overlap.export_session(overlap_put.handle).unwrap()
    );
}

#[test]
fn overlap_stores_run_from_many_threads_with_one_shared_runner() {
    let runner: Arc<dyn OverlapRunner> = Arc::new(ScopedOverlap);
    std::thread::scope(|scope| {
        for lane in 0..4u8 {
            let runner = Arc::clone(&runner);
            scope.spawn(move || {
                let enc = LazyByteEncoder::new();
                let mut store = SessionStore::new(cfg(8)).unwrap();
                store.enable_content_tracking().unwrap();
                store.set_seed_overlap(Some(runner));
                let key = store.register_fingerprint(fp(80 + lane), 0).unwrap();
                for round in 0..8u8 {
                    let text = format!("lane {lane} round {round} words to seal and digest");
                    let put = store.put(key, &text, &enc).unwrap();
                    assert_eq!(
                        store.content_index_entry(put.handle).unwrap(),
                        Some(ContentDigest::from_bytes(text.as_bytes()).unwrap().entry())
                    );
                    assert_eq!(
                        store.shared_all_ids(put.handle).unwrap().as_slice(),
                        &LazyByteEncoder::byte_ids(&text)[..]
                    );
                }
            });
        }
    });
}
