//! Store behavior battery (hermetic tier).
//!
//! This is the Rust-tier port of the pre-release prototype store v1 test
//! battery, driven by the deterministic mock encoder so it runs with no
//! tokenizer dependency. The real-tokenizer tier of the same battery
//! (plus the cross-implementation equivalence run) was exercised
//! against the prototype before this port was adopted.

use toktier_store_core::testing::MockEncoder;
use toktier_store_core::{
    AppendOutcome, SessionEncoder, SessionStore, StoreConfig, StoreError, WitnessCategory,
};

fn fp(tag: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = tag;
    out[31] = tag.wrapping_add(1);
    out
}

fn cfg(block_chars: u64) -> StoreConfig {
    StoreConfig {
        block_chars,
        ..StoreConfig::default()
    }
}

fn judge(enc: &MockEncoder, text: &str) -> Vec<u32> {
    enc.encode(text).unwrap().ids
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
