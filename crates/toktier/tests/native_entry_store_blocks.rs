//! The native entry store (the facade's request runtime on CPU and
//! prebuilt plans) finds stored prefixes through its content index and
//! never reads the store's per-block records. These tests drive mixed
//! traffic through `NativeEntryStore` and check three things: no block
//! record is kept, every returned id stream equals the reference
//! engine's encode of the same text, and the entry, seal and append-path
//! counters are the values this traffic produced while the store still
//! wrote block records (pinned below).
//!
//! The small tokenizer case always runs; the `qwen3_8b` case runs on the
//! fast CPU route with repair when the verified artifact is present
//! (`TOKTIER_TEST_ARTIFACTS` or the user cache, as in `public_api.rs`).
//! A store reopened over a directory starts with a detached chain, so
//! the block-record check bites on the first store of each run.
//!
//! The file sits outside every source-identity path set, so its wording
//! can change without moving an identity.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use toktier_routing_core::{
    FastCpuEngine, FastRepairSpec, NativeEntryStore, NativeRouter, ReferenceEngine,
    BACKEND_FAST_CPU, BACKEND_REFERENCE, DEFAULT_CACHE_BUDGET_BYTES,
};
use toktier_store_core::StatsSnapshot;

/// A small BPE over five letters and the space, with a few merges,
/// split before each space. It needs no artifact, so this case always
/// runs.
const SMALL_TOKENIZER: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithNext", "invert": false},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
    "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
    "vocab": {" ": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, " a": 6, "ab": 7, "cd": 8, " ab": 9, "ee": 10},
    "merges": [[" ", "a"], ["a", "b"], ["c", "d"], [" a", "b"], ["e", "e"]]
  }
}"#;

/// How the store's router is built for one case.
enum Route {
    /// The reference engine alone.
    Reference,
    /// The fast CPU engine with repair in front of the reference, as on
    /// a CPU plan, with the family's repair parameters from the frozen
    /// table in this tree.
    FastCpu { family: &'static str },
}

fn repair_tables() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/toktier/repair/tables")
}

fn fast_engine(json: &[u8], reference: Arc<ReferenceEngine>, family: &str) -> FastCpuEngine {
    let table: serde_json::Value = serde_json::from_slice(
        &std::fs::read(repair_tables().join("fast_repair_families.v1.json")).unwrap(),
    )
    .unwrap();
    let row = table["families"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["family"] == family)
        .unwrap();
    let spec = FastRepairSpec::new(
        family.to_owned(),
        row["artifact_sha256"].as_str().unwrap().to_owned(),
        row["margin"].as_u64().unwrap() as usize,
        row["effective_l_max"].as_u64().unwrap() as usize,
        row["has_normalizer"].as_bool().unwrap(),
    );
    let compressed = std::fs::read(repair_tables().join("repair_pclass.v1.zlib")).unwrap();
    let mut pclass = Vec::new();
    flate2::read::ZlibDecoder::new(&compressed[..])
        .read_to_end(&mut pclass)
        .unwrap();
    FastCpuEngine::from_reference(json, reference, spec, pclass).unwrap()
}

/// The router and the seal end guard the facade would use for it.
fn router_for(json: &[u8], route: &Route) -> (Arc<NativeRouter>, u64) {
    let reference = Arc::new(ReferenceEngine::from_bytes(json).unwrap());
    let guard = longest_added_literal(&reference);
    let (chain, fast) = match route {
        Route::Reference => (vec![BACKEND_REFERENCE.to_owned()], None),
        Route::FastCpu { family } => (
            vec![BACKEND_FAST_CPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            Some(Arc::new(fast_engine(json, Arc::clone(&reference), family))),
        ),
    };
    let guard = match &fast {
        Some(engine) => guard.max(engine.minimum_seal_tail_chars() as u64),
        None => guard,
    };
    let thresholds = vec![0; chain.len()];
    let repair = fast.is_some();
    let router = NativeRouter::new(
        chain, thresholds, reference, fast, repair, None, false, false,
    )
    .unwrap();
    (Arc::new(router), guard)
}

/// Deterministic text from a word list (a small linear congruential
/// generator, so every run sees the same bytes).
struct Words {
    state: u64,
    words: &'static [&'static str],
}

impl Words {
    fn new(seed: u64, words: &'static [&'static str]) -> Self {
        Self {
            state: seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1),
            words,
        }
    }

    /// At least `chars` characters, cut at a character boundary.
    fn text(&mut self, chars: usize) -> String {
        let mut out = String::new();
        while out.chars().count() < chars {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let pick = (self.state >> 33) as usize % self.words.len();
            out.push_str(self.words[pick]);
        }
        out.chars().take(chars).collect()
    }
}

const SMALL_WORDS: &[&str] = &[
    " ab", " a", "cd", " abe", "ee", " cdab", "b", " ", "dc", " eab",
];

const NATURAL_WORDS: &[&str] = &[
    " the",
    " store",
    " keeps",
    " text",
    ",",
    ".",
    "\n",
    " 2048",
    " caf\u{e9}",
    " na\u{ef}ve",
    " \u{6771}\u{4eac}",
    " \u{30c6}\u{30b9}\u{30c8}",
    " \u{c548}\u{b155}",
    " r\u{e9}sum\u{e9}",
    " data",
    "  ",
    "\n\n",
    " (x + y)",
    " fn",
    " {",
    "}",
    " -- ",
    " 17",
    " \u{e9}t\u{e9}",
];

/// What one run of the traffic below observed.
struct Observed {
    unequal: Vec<String>,
    compared: usize,
    stats_before_reopen: toktier_routing_core::EntryStoreStats,
    native_before_reopen: StatsSnapshot,
    stats_after_reopen: toktier_routing_core::EntryStoreStats,
    native_after_reopen: StatsSnapshot,
    longest_text_chars: usize,
}

/// Mixed traffic: sessions that grow by appends of 17 to 9,000
/// characters, default-style calls as first call, exact repeat and
/// extension, all over a directory-backed store; then the store is
/// dropped and a second store over the same directory continues the
/// sessions and repeats the default calls.
fn drive(json: &[u8], route: Route, words: &'static [&'static str]) -> Observed {
    let directory = tempfile::tempdir().unwrap();
    let (router, seal_end_guard_chars) = router_for(json, &route);
    let check = ReferenceEngine::from_bytes(json).unwrap();
    let open = |router: &Arc<NativeRouter>, dir: &Path| {
        NativeEntryStore::open(
            [5; 32],
            Arc::clone(router),
            Some(dir.to_owned()),
            DEFAULT_CACHE_BUDGET_BYTES,
            seal_end_guard_chars,
        )
        .unwrap()
    };

    let mut unequal = Vec::new();
    let mut compared = 0usize;
    let mut longest = 0usize;
    let mut compare = |what: String, text: &str, got: Option<Vec<u32>>| {
        let want = check.encode_ids(text, false).unwrap();
        compared += 1;
        longest = longest.max(text.chars().count());
        if got.as_deref() != Some(&want[..]) {
            unequal.push(what);
        }
    };

    let appends = [17usize, 500, 4_100, 9_000, 1_200, 33, 2_048, 6_000];
    let mut sessions: Vec<(String, String)> = Vec::new();
    let mut store = open(&router, directory.path());
    for index in 0..3u64 {
        let mut words_for = Words::new(100 + index, words);
        let name = format!("chat-{index}");
        let mut text = words_for.text(3_000);
        compare(
            format!("{name} seed"),
            &text,
            store.encode_session(&name, &text),
        );
        for (step, size) in appends.iter().enumerate() {
            text.push_str(&words_for.text(*size));
            compare(
                format!("{name} append {step}"),
                &text,
                store.encode_session(&name, &text),
            );
        }
        // The same transcript again is served from the entry.
        compare(
            format!("{name} repeat"),
            &text,
            store.encode_session(&name, &text),
        );
        sessions.push((name, text));
    }

    let mut defaults: Vec<String> = Vec::new();
    for (index, size) in [6_000usize, 24_000, 70_000].into_iter().enumerate() {
        let mut words_for = Words::new(200 + index as u64, words);
        let first = words_for.text(size);
        compare(
            format!("default {size} first"),
            &first,
            store.encode_auto(&first),
        );
        compare(
            format!("default {size} repeat"),
            &first,
            store.encode_auto(&first),
        );
        let extended = format!("{first}{}", words_for.text(5_000));
        compare(
            format!("default {size} extension"),
            &extended,
            store.encode_auto(&extended),
        );
        defaults.push(extended);
    }
    let stats_before_reopen = store.stats().clone();
    let native_before_reopen = store.native_stats();
    drop(store);

    let mut reopened = open(&router, directory.path());
    for (index, (name, text)) in sessions.iter_mut().enumerate() {
        compare(
            format!("{name} reopened"),
            text,
            reopened.encode_session(name, text),
        );
        let mut words_for = Words::new(300 + index as u64, words);
        text.push_str(&words_for.text(4_500));
        compare(
            format!("{name} reopened append"),
            text,
            reopened.encode_session(name, text),
        );
    }
    for text in &defaults {
        compare(
            format!("default {} chars after reopen", text.chars().count()),
            text,
            reopened.encode_auto(text),
        );
    }
    Observed {
        unequal,
        compared,
        stats_before_reopen,
        native_before_reopen,
        stats_after_reopen: reopened.stats().clone(),
        native_after_reopen: reopened.native_stats(),
        longest_text_chars: longest,
    }
}

fn report(label: &str, observed: &Observed) {
    eprintln!(
        "{label}: compared {} unequal {} longest {} chars",
        observed.compared,
        observed.unequal.len(),
        observed.longest_text_chars
    );
    for (when, stats, native) in [
        (
            "before reopen",
            &observed.stats_before_reopen,
            &observed.native_before_reopen,
        ),
        (
            "after reopen",
            &observed.stats_after_reopen,
            &observed.native_after_reopen,
        ),
    ] {
        eprintln!("{label} {when}: entry {stats:?}");
        eprintln!(
            "{label} {when}: native node_count {} seals {} sealed_tokens {} puts {} extends {} \
             chain_detaches {} nodes_skipped_tail_cap {} path_counts {:?}",
            native.node_count,
            native.seals,
            native.sealed_tokens,
            native.puts,
            native.extends,
            native.chain_detaches,
            native.nodes_skipped_tail_cap,
            native.path_counts
        );
    }
}

/// The verified local qwen3 artifact, when this host has one (the same
/// lookup as `public_api.rs`).
fn qwen3_tokenizer_json() -> Option<Vec<u8>> {
    std::env::var_os("TOKTIER_TEST_ARTIFACTS")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .map(|home| home.join(".cache/toktier/artifacts"))
        })
        .map(|root| root.join("qwen3_8b-b968826d9c46/tokenizer.json"))
        .and_then(|path| std::fs::read(path).ok())
}

/// The longest added-token literal, in characters (the facade's lower
/// bound for the seal end guard).
fn longest_added_literal(reference: &ReferenceEngine) -> u64 {
    reference
        .added_tokens()
        .iter()
        .map(|(_, content, _)| content.chars().count() as u64)
        .max()
        .unwrap_or(0)
}

/// Counters pinned from the same traffic on the store as it was when it
/// still wrote block records, before and after the reopen.
struct Pinned {
    /// session hits, appends, overwrites, misses; auto hits, appends, misses
    entry: [[u64; 7]; 2],
    seals: [u64; 2],
    sealed_tokens: [u64; 2],
    path_counts: [&'static [(&'static str, u64)]; 2],
}

fn entry_counters(stats: &toktier_routing_core::EntryStoreStats) -> [u64; 7] {
    [
        stats.session_hits,
        stats.session_appends,
        stats.session_overwrites,
        stats.session_misses,
        stats.auto_hits,
        stats.auto_appends,
        stats.auto_misses,
    ]
}

fn assert_traffic(label: &str, observed: &Observed, pinned: &Pinned) {
    report(label, observed);
    assert!(
        observed.unequal.is_empty(),
        "{label}: unequal ids at {:?}",
        observed.unequal
    );
    assert!(
        observed.longest_text_chars > 16 * 4_096,
        "{label}: the traffic crosses many blocks"
    );
    for (when, stats) in [
        ("before reopen", &observed.stats_before_reopen),
        ("after reopen", &observed.stats_after_reopen),
    ] {
        assert_eq!(stats.degraded, 0, "{label} {when}");
        assert_eq!(stats.collision_rejects, 0, "{label} {when}");
    }
    let natives = [
        &observed.native_before_reopen,
        &observed.native_after_reopen,
    ];
    let entries = [&observed.stats_before_reopen, &observed.stats_after_reopen];
    for (index, when) in ["before reopen", "after reopen"].into_iter().enumerate() {
        assert_eq!(
            entry_counters(entries[index]),
            pinned.entry[index],
            "{label} {when}: entry"
        );
        assert_eq!(
            natives[index].seals, pinned.seals[index],
            "{label} {when}: seals"
        );
        assert_eq!(
            natives[index].sealed_tokens, pinned.sealed_tokens[index],
            "{label} {when}: sealed tokens"
        );
        let paths: Vec<(&str, u64)> = natives[index]
            .path_counts
            .iter()
            .map(|(name, count)| (name.as_str(), *count))
            .collect();
        assert_eq!(
            paths, pinned.path_counts[index],
            "{label} {when}: append paths"
        );
    }
}

fn assert_no_block_records(label: &str, observed: &Observed) {
    report(label, observed);
    assert_eq!(
        observed.native_before_reopen.node_count, 0,
        "{label}: block records kept before reopen"
    );
    assert_eq!(
        observed.native_after_reopen.node_count, 0,
        "{label}: block records kept after reopen"
    );
}

fn small() -> Observed {
    drive(SMALL_TOKENIZER.as_bytes(), Route::Reference, SMALL_WORDS)
}

fn qwen3() -> Option<Observed> {
    let json = qwen3_tokenizer_json()?;
    Some(drive(
        &json,
        Route::FastCpu { family: "qwen3_8b" },
        NATURAL_WORDS,
    ))
}

#[test]
fn the_native_entry_store_keeps_no_block_records() {
    assert_no_block_records("small", &small());
}

#[test]
fn the_native_entry_store_keeps_no_block_records_on_a_fast_cpu_route() {
    if let Some(observed) = qwen3() {
        assert_no_block_records("qwen3_8b", &observed);
    }
}

#[test]
fn ids_and_counters_are_those_of_the_traffic_with_block_records() {
    assert_traffic("small", &small(), &PINNED_SMALL);
}

#[test]
fn ids_and_counters_are_those_of_the_traffic_with_block_records_on_a_fast_cpu_route() {
    if let Some(observed) = qwen3() {
        assert_traffic("qwen3_8b", &observed, &PINNED_QWEN3);
    }
}

const PINNED_SMALL: Pinned = Pinned {
    entry: [[3, 24, 0, 3, 3, 3, 3], [3, 3, 0, 0, 3, 0, 0]],
    seals: [0, 0],
    sealed_tokens: [0, 0],
    path_counts: [
        &[
            ("hf_no_certified_span_bridge", 6),
            ("native_hf_full_reencode", 27),
        ],
        &[("native_hf_full_reencode", 3)],
    ],
};

const PINNED_QWEN3: Pinned = Pinned {
    entry: [[3, 24, 0, 3, 3, 3, 3], [3, 3, 0, 0, 3, 0, 0]],
    seals: [15, 3],
    sealed_tokens: [86_546, 6_230],
    path_counts: [
        &[("gigatoken_full", 6), ("gigatoken_repair", 27)],
        &[("gigatoken_repair", 3)],
    ],
};
