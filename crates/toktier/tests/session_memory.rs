//! `Tokenizer::lookup` over the sessions a store holds, through the public
//! API: a hit returns exactly the ids `Tokenizer::encode` returns, reusing a
//! stored session's sealed ids where it can, and the size of the lookup index
//! can be limited or the index turned off without changing any id.
//!
//! Artifact tier: the tests load `qwen3_8b` and `llama_3_1_8b` from
//! `TOKTIER_TEST_ARTIFACTS` (or the default cache) and are skipped when they
//! are not there. They run the fast CPU engine (`Policy::Experimental`), which
//! is what gives sessions seal points to reuse.

use std::path::PathBuf;

use toktier::{Backend, Device, Policy, Runtime, RuntimeBuilder, Tokenizer};

// The two helpers that read what the lookup index added, in one place.

fn reused_ids(tokenizer: &Tokenizer) -> u64 {
    tokenizer.store_stats().lookup_reused_ids
}

fn blocks_declined(tokenizer: &Tokenizer) -> u64 {
    tokenizer.store_stats().index_blocks_declined
}

fn with_index_blocks(builder: RuntimeBuilder, blocks: usize) -> RuntimeBuilder {
    builder.lookup_index_blocks(blocks)
}

fn artifact_root(family: &str) -> Option<PathBuf> {
    let directory = match family {
        "qwen3_8b" => "qwen3_8b-b968826d9c46",
        "llama_3_1_8b" => "llama_3_1_8b-d04e592bb4f6",
        _ => unreachable!(),
    };
    let root = std::env::var_os("TOKTIER_TEST_ARTIFACTS")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".cache/toktier/artifacts"))
        })
        .filter(|root| root.join(directory).join("tokenizer.json").is_file());
    if root.is_none() {
        eprintln!("skip: no {family} artifact under TOKTIER_TEST_ARTIFACTS or the cache");
    }
    root
}

fn load(family: &str, index_blocks: Option<usize>) -> Option<(Runtime, Tokenizer)> {
    let root = artifact_root(family)?;
    let mut builder = Runtime::builder()
        .artifact_cache(&root)
        .device(Device::Cpu)
        .policy(Policy::Experimental);
    if let Some(blocks) = index_blocks {
        builder = with_index_blocks(builder, blocks);
    }
    let runtime = builder.build().unwrap();
    let tokenizer = runtime.load(family).unwrap();
    assert!(tokenizer.plan().backends.contains(&Backend::FastCpu));
    Some((runtime, tokenizer))
}

/// A small deterministic generator (SplitMix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound.max(1) as u64) as usize
    }
}

/// `chars` characters of word-like text built from `pieces`.
fn prose(seed: u64, chars: usize, pieces: &[&str]) -> String {
    let mut rng = Rng(seed ^ 0x0BAD_5EED);
    let mut out = String::new();
    let mut count = 0;
    while count < chars {
        let piece = pieces[rng.below(pieces.len())];
        out.push_str(piece);
        count += piece.chars().count();
    }
    out.chars().take(chars).collect()
}

const ASCII: &[&str] = &[
    "river ",
    "stone ",
    "maple, ",
    "cloud ",
    "lantern. ",
    "harbor ",
    "meadow ",
    "copper ",
    "17 ",
    "\n",
];
const MULTIBYTE: &[&str] = &[
    "river ",
    "caf\u{e9} ",
    "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{6587}\u{7ae0} ",
    "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442} ",
    "\u{1f642} ",
    "stone ",
];
/// Text that NFC normalization changes: decomposed accents, conjoining
/// Hangul jamo, the Angstrom sign.
const NFC: &[&str] = &[
    "river ",
    "e\u{301}te\u{301} ",
    "A\u{30a}ngstro\u{308}m ",
    "\u{1100}\u{1161}\u{11a8} ",
    "\u{212b} ",
    "maple ",
    "\u{65e5}\u{672c} ",
];

fn take_chars(text: &str, chars: usize) -> String {
    text.chars().take(chars).collect()
}

fn ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text).unwrap().ids().to_vec()
}

fn lookup(tokenizer: &Tokenizer, query: &str) -> Option<Vec<u32>> {
    let found = tokenizer.lookup(query).unwrap()?;
    assert_eq!(found.ids(), ids(tokenizer, query), "lookup ids");
    Some(found.ids().to_vec())
}

/// Seed `name` and grow it by appends; every snapshot equals `encode`.
fn grow(tokenizer: &Tokenizer, name: &str, seed: u64, pieces: &[&str], turns: usize) -> String {
    let mut text = prose(seed, 2_000, pieces);
    let mut session = tokenizer.open_session(name).unwrap();
    assert_eq!(session.seed(&text).unwrap().ids(), ids(tokenizer, &text));
    for turn in 0..turns {
        let delta = prose(seed * 100 + turn as u64, 1_500, pieces);
        session.append(&delta).unwrap();
        text.push_str(&delta);
        assert_eq!(
            session.snapshot().unwrap().ids(),
            ids(tokenizer, &text),
            "{name} turn {turn}"
        );
    }
    session.close().unwrap();
    text
}

/// Lookups on sessions grown past several blocks: extensions, cuts inside a
/// block, texts that leave the session after block k, a fork's text, and an
/// unrelated text.
fn exact_with_reuse(family: &str, pieces: &[&str], salt: u64) {
    let Some((_runtime, tokenizer)) = load(family, None) else {
        return;
    };
    let text = grow(&tokenizer, "grown", salt, pieces, 20);
    let chars = text.chars().count();
    assert!(chars > 7 * 4_096);
    let other = grow(&tokenizer, "other", salt + 1, pieces, 6);

    let tail = prose(salt + 2, 2_500, pieces);
    assert!(lookup(&tokenizer, &format!("{text}{tail}")).is_some());
    assert!(lookup(&tokenizer, &text).is_some());
    for k in 1..=(chars / 4_096) {
        // A cut inside block k + 1, and a text that leaves the session
        // right after block k.
        let cut = take_chars(&text, k * 4_096 + 1_000.min(chars - k * 4_096));
        assert!(lookup(&tokenizer, &cut).is_some(), "cut after block {k}");
        let left = format!("{}{tail}", take_chars(&text, k * 4_096 + 3));
        assert!(
            lookup(&tokenizer, &left).is_some(),
            "leaves after block {k}"
        );
    }
    assert!(lookup(&tokenizer, &format!("{other} and more")).is_some());

    // A fork that goes its own way.
    let source = tokenizer.open_session("grown").unwrap();
    let mut fork = source.fork("fork").unwrap();
    source.close().unwrap();
    let delta = prose(salt + 3, 9_000, pieces);
    fork.append(&delta).unwrap();
    let forked = format!("{text}{delta}");
    assert_eq!(fork.snapshot().unwrap().ids(), ids(&tokenizer, &forked));
    fork.close().unwrap();
    assert!(lookup(&tokenizer, &format!("{forked} and more")).is_some());

    // Text no session has.
    assert!(lookup(&tokenizer, &prose(salt + 4, 9_000, ASCII)).is_none());
    assert!(lookup(&tokenizer, "short").is_none());
    assert!(
        reused_ids(&tokenizer) > 0,
        "{family}: lookups reused sealed ids"
    );
}

#[test]
fn lookups_reuse_sealed_ids_and_return_exact_ids_ascii() {
    exact_with_reuse("qwen3_8b", ASCII, 1);
    exact_with_reuse("llama_3_1_8b", ASCII, 11);
}

#[test]
fn lookups_reuse_sealed_ids_and_return_exact_ids_multibyte() {
    exact_with_reuse("qwen3_8b", MULTIBYTE, 21);
    exact_with_reuse("llama_3_1_8b", MULTIBYTE, 31);
}

#[test]
fn lookups_reuse_sealed_ids_and_return_exact_ids_nfc() {
    exact_with_reuse("qwen3_8b", NFC, 41);
}

#[test]
fn the_index_limit_changes_hits_and_never_ids() {
    // Off: every lookup misses; sessions are as before.
    if let Some((_runtime, tokenizer)) = load("qwen3_8b", Some(0)) {
        let text = grow(&tokenizer, "s", 51, ASCII, 8);
        assert!(tokenizer.lookup(&text).unwrap().is_none());
        assert!(tokenizer.lookup(&format!("{text} more")).unwrap().is_none());
        let stats = tokenizer.store_stats();
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.lookup_misses, 2);
        let mut session = tokenizer.open_session("s").unwrap();
        let delta = prose(52, 3_000, ASCII);
        session.append(&delta).unwrap();
        assert_eq!(
            session.snapshot().unwrap().ids(),
            ids(&tokenizer, &format!("{text}{delta}"))
        );
        session.close().unwrap();
    }

    // A small limit: blocks past it are declined; ids stay exact.
    if let Some((_runtime, tokenizer)) = load("qwen3_8b", Some(3)) {
        let first = grow(&tokenizer, "a", 61, ASCII, 8);
        let second = grow(&tokenizer, "b", 62, NFC, 8);
        assert!(blocks_declined(&tokenizer) > 0);
        assert_eq!(tokenizer.store_stats().node_count, 3);
        assert!(lookup(&tokenizer, &format!("{first} and more")).is_some());
        // `b` found no room; a lookup of it is a miss.
        assert!(lookup(&tokenizer, &format!("{second} and more")).is_none());
        tokenizer.open_session("a").unwrap().delete().unwrap();
        assert_eq!(tokenizer.store_stats().node_count, 0);
        let third = grow(&tokenizer, "c", 63, MULTIBYTE, 3);
        assert!(lookup(&tokenizer, &format!("{third} and more")).is_some());
    }
}
