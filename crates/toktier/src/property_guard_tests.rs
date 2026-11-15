//! The property-data guard through the public calls.
//!
//! In this workspace the build's own standing is judged, so these tests
//! force a standing through the runtime builder's test seam
//! (`RuntimeBuilder::force_property_standing`, test builds only). In the
//! workspace the judged data makes the fast engine and the reference
//! engine agree on every text, so what a test of the routes can observe
//! here is where a call ran (backend and reason), whether a seal moved,
//! and that the ids equal the reference engine's; the ids themselves
//! differ only on a build that resolves newer property data, which the
//! consumer builds of the evidence lane cover.

use std::path::{Path, PathBuf};

use super::{LocatedBy, Standing, MOVED_CODE_POINTS};
use crate::{Backend, Device, ErrorCode, Policy, ReasonCode, Runtime, RuntimeBuilder, Tokenizer};

/// The texts of the review that found the difference, and one for the
/// already-assigned code point whose category moved (U+0295).
const HITS: &[&str] = &[
    "1\u{11DE1}2345",
    "it's\u{323B0}'s \u{323B0}'ll",
    "abc \u{20C1}def ghi",
    "Hello \u{0295}\u{2022}\u{1D25}\u{2022}\u{0294} World a\u{0295}b A\u{0295}B",
];

/// Texts no admitted pair classifies differently, which the fast CPU
/// route serves.
const CLEAN: &[&str] = &[
    "plain ascii text 12345",
    "caf\u{e9} na\u{ef}ve \u{fc}ber stra\u{df}e",
    "\u{4e2d}\u{6587}\u{6d4b}\u{8bd5} 12345 \u{30c6}\u{30b9}\u{30c8}",
];

const FAMILIES: &[&str] = &["qwen3_8b", "llama_3_1_8b", "deepseek_v3"];

fn guarded() -> Standing {
    Standing::Guarded {
        resolved: "icu_properties 2.3.0, icu_properties_data 2.3.0".to_owned(),
    }
}

fn untabled() -> Standing {
    Standing::Untabled {
        reason: "icu_properties_data 2.9.0 (checksum 00) is not among the property-data \
                 copies this release was judged against"
            .to_owned(),
        remedy: "align the lockfile with the judged property data".to_owned(),
    }
}

fn unread(located_by: LocatedBy) -> Standing {
    Standing::Unread { located_by }
}

/// The artifact root of the existing tests (`TOKTIER_TEST_ARTIFACTS`, or
/// the default cache), when it holds `family`.
fn artifact_root(family: &str) -> Option<PathBuf> {
    let root = std::env::var_os("TOKTIER_TEST_ARTIFACTS")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".cache/toktier/artifacts"))
        })?;
    let present = std::fs::read_dir(&root).ok()?.flatten().any(|entry| {
        entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with(&format!("{family}-")))
            && entry.path().join("tokenizer.json").is_file()
    });
    if !present {
        eprintln!("skip: no {family} artifact under TOKTIER_TEST_ARTIFACTS or the cache");
        return None;
    }
    Some(root)
}

fn builder(root: &Path, policy: Policy) -> RuntimeBuilder {
    Runtime::builder()
        .artifact_cache(root)
        .device(Device::Cpu)
        .policy(policy)
}

fn load(builder: RuntimeBuilder, family: &str) -> Tokenizer {
    builder.build().unwrap().load(family).unwrap()
}

/// A tokenizer under `standing` with the fast CPU route (the experimental
/// policy admits the route on a tree the shipped register does not
/// record, as the session tests do).
fn forced(root: &Path, family: &str, standing: Standing) -> Tokenizer {
    let tokenizer = load(
        builder(root, Policy::Experimental).force_property_standing(standing),
        family,
    );
    assert!(
        tokenizer.plan().backends.contains(&Backend::FastCpu),
        "{family}: the experimental plan runs the fast CPU engine"
    );
    tokenizer
}

fn reference(root: &Path, family: &str) -> Tokenizer {
    load(builder(root, Policy::Reference), family)
}

fn ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text).unwrap().ids().to_vec()
}

fn assert_guard_routed(facts: &crate::ExecutionFacts, what: &str) {
    assert_eq!(facts.backend, Backend::HuggingFace, "{what}: {facts:?}");
    assert_eq!(
        facts.reason,
        Some(ReasonCode::InputGuardRouted),
        "{what}: {facts:?}"
    );
}

fn assert_fast(facts: &crate::ExecutionFacts, what: &str) {
    assert_eq!(facts.backend, Backend::FastCpu, "{what}: {facts:?}");
    assert_ne!(
        facts.reason,
        Some(ReasonCode::InputGuardRouted),
        "{what}: {facts:?}"
    );
}

/// `chars` characters of word-like text in several scripts, none of them
/// a listed code point.
fn prose(seed: u64, chars: usize) -> String {
    const PIECES: [&str; 12] = [
        "river ",
        "stone ",
        "maple, ",
        "lantern. ",
        "42 ",
        "\n",
        "caf\u{e9} ",
        "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{6587}\u{7ae0} ",
        "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442} ",
        "\u{1f642} ",
        "it's ",
        "\u{915}\u{93f}\u{924}\u{93e}\u{92c} ",
    ];
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = String::new();
    let mut count = 0;
    while count < chars {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let piece = PIECES[(state >> 33) as usize % PIECES.len()];
        out.push_str(piece);
        count += piece.chars().count();
    }
    let out: String = out.chars().take(chars).collect();
    assert!(!super::holds_moved(&out));
    out
}

#[test]
fn guarded_one_shot_batch_lookup_and_chunked_calls_send_hits_to_the_reference_engine() {
    for family in FAMILIES {
        let Some(root) = artifact_root(family) else {
            continue;
        };
        let tokenizer = forced(&root, family, guarded());
        let judge = reference(&root, family);
        assert!(tokenizer.inner.guard_router.is_some(), "{family}");

        // One-shot, with and without special tokens.
        for text in HITS {
            let encoded = tokenizer.encode(text).unwrap();
            assert_guard_routed(encoded.execution(), &format!("{family} {text:?}"));
            assert_eq!(encoded.ids(), ids(&judge, text), "{family} {text:?}");
            let options = crate::EncodeOptions {
                add_special_tokens: true,
                offsets: false,
            };
            let special = tokenizer.encode_with(text, options).unwrap();
            assert_guard_routed(special.execution(), &format!("{family} {text:?} special"));
            assert_eq!(
                special.ids(),
                judge.encode_with(text, options).unwrap().ids(),
                "{family} {text:?} special"
            );
        }
        for text in CLEAN {
            let encoded = tokenizer.encode(text).unwrap();
            assert_fast(encoded.execution(), &format!("{family} {text:?}"));
            assert_eq!(encoded.ids(), ids(&judge, text), "{family} {text:?}");
        }

        // Batch: mixed rows keep their order; all-clean and all-hit too.
        let mixed: Vec<&str> = vec![
            CLEAN[0], HITS[0], HITS[1], CLEAN[2], HITS[2], CLEAN[1], HITS[3], CLEAN[0],
        ];
        for rows in [mixed.clone(), CLEAN.to_vec(), HITS.to_vec()] {
            let batch = tokenizer.encode_batch(&rows).unwrap();
            let expected = judge.encode_batch(&rows).unwrap();
            assert_eq!(batch.len(), rows.len());
            for (index, text) in rows.iter().enumerate() {
                assert_eq!(
                    batch.row(index).unwrap(),
                    expected.row(index).unwrap(),
                    "{family} row {index} {text:?}"
                );
                let facts = &batch.executions()[index];
                if super::holds_moved(text) {
                    assert_guard_routed(facts, &format!("{family} row {index}"));
                } else {
                    assert_fast(facts, &format!("{family} row {index}"));
                }
            }
        }

        // Lookup: a stored session with a clean first block, asked about
        // that text followed by a hit and by clean text.
        let stored = prose(7, 9_000);
        let mut session = tokenizer.open_session("lookup-source").unwrap();
        session.seed(&stored).unwrap();
        session.close().unwrap();
        for (suffix, hit) in [(HITS[1], true), (" and more clean words", false)] {
            let query = format!("{stored}{suffix}");
            let found = tokenizer.lookup(&query).unwrap().expect("a lookup hit");
            assert_eq!(found.ids(), ids(&judge, &query), "{family} lookup {hit}");
            if hit {
                assert_guard_routed(found.execution(), &format!("{family} lookup"));
            } else {
                assert_eq!(found.execution().backend, Backend::FastCpu);
                assert_eq!(found.execution().reason, None);
            }
        }

        // Chunked: a long document with a hit is the serial encode on the
        // reference engine; a clean one is not routed by the guard.
        let chunked = tokenizer.chunked();
        let options = crate::chunked::ChunkOptions::new();
        for (text, hit) in [
            (
                format!("{}{}{}", prose(11, 12_000), HITS[0], prose(12, 12_000)),
                true,
            ),
            (prose(13, 24_000), false),
        ] {
            let encoded = chunked.encode(&text, &options).unwrap();
            assert_eq!(encoded.ids(), ids(&judge, &text), "{family} chunked {hit}");
            if hit {
                assert_eq!(
                    encoded.serial_reason,
                    Some(ReasonCode::ChunkRouteNotFastCpu),
                    "{family}"
                );
                assert_eq!(encoded.chunks, 1);
            } else {
                assert_ne!(
                    encoded.serial_reason,
                    Some(ReasonCode::ChunkRouteNotFastCpu),
                    "{family}"
                );
            }
        }
        // A warm-up text with a hit is not run on the workers.
        chunked.warm(Some(1), Some(HITS[0])).unwrap();
    }
}

#[test]
fn guarded_sessions_never_seal_past_a_hit_and_stay_exact() {
    let family = "qwen3_8b";
    let Some(root) = artifact_root(family) else {
        return;
    };
    let tokenizer = forced(&root, family, guarded());
    let judge = reference(&root, family);

    // A hit in the seed: the seed runs on the reference engine and the
    // session offers no seal while the hit is in its tail.
    let mut text = format!("{}{}{}", prose(1, 6_000), HITS[0], prose(2, 200));
    let mut seeded = tokenizer.open_session("hit-seed").unwrap();
    let encoded = seeded.seed(&text).unwrap();
    assert_guard_routed(encoded.execution(), "seed with a hit");
    assert_eq!(encoded.ids(), ids(&judge, &text));
    assert_eq!(seeded.seal_for_test(), (0, 0), "no seal over a hit");
    for turn in 0..4 {
        let delta = prose(100 + turn, 1_500);
        let patch = seeded.append(&delta).unwrap();
        assert_guard_routed(patch.execution(), "append behind a hit in the tail");
        text.push_str(&delta);
        assert_eq!(seeded.snapshot().unwrap().ids(), ids(&judge, &text));
        assert_eq!(seeded.seal_for_test(), (0, 0), "turn {turn}");
    }

    // A hit arriving in a later delta: the seal moves before it, never
    // after it.
    let mut text = prose(3, 6_000);
    let mut session = tokenizer.open_session("hit-later").unwrap();
    assert_fast(session.seed(&text).unwrap().execution(), "clean seed");
    for turn in 0..3 {
        let delta = prose(200 + turn, 1_500);
        assert_fast(session.append(&delta).unwrap().execution(), "clean append");
        text.push_str(&delta);
        assert_eq!(session.snapshot().unwrap().ids(), ids(&judge, &text));
    }
    let before_hit = session.seal_for_test();
    assert!(before_hit.1 > 0, "clean appends seal: {before_hit:?}");
    let delta = format!(" {} ", HITS[1]);
    assert_guard_routed(session.append(&delta).unwrap().execution(), "hit delta");
    text.push_str(&delta);
    assert_eq!(session.snapshot().unwrap().ids(), ids(&judge, &text));
    let at_hit = session.seal_for_test();
    assert_eq!(at_hit, before_hit, "the hit's own append seals nothing");
    for turn in 0..4 {
        let delta = prose(300 + turn, 1_500);
        let patch = session.append(&delta).unwrap();
        assert_guard_routed(patch.execution(), "clean append behind the hit");
        text.push_str(&delta);
        assert_eq!(
            session.snapshot().unwrap().ids(),
            ids(&judge, &text),
            "after clean turn {turn}"
        );
        assert_eq!(
            session.seal_for_test(),
            at_hit,
            "turn {turn}: the seal stays"
        );
    }

    // A fork of such a session keeps the rule.
    let mut fork = session.fork("hit-later-fork").unwrap();
    let mut fork_text = text.clone();
    for turn in 0..2 {
        let delta = prose(400 + turn, 1_500);
        assert_guard_routed(fork.append(&delta).unwrap().execution(), "fork append");
        fork_text.push_str(&delta);
        assert_eq!(fork.snapshot().unwrap().ids(), ids(&judge, &fork_text));
        assert_eq!(fork.seal_for_test(), at_hit, "fork turn {turn}");
    }

    // An overwrite with a hit runs on the reference engine.
    let replaced = format!("{}{}", prose(5, 3_000), HITS[2]);
    let encoded = session.overwrite(&replaced).unwrap();
    assert_guard_routed(encoded.execution(), "overwrite with a hit");
    assert_eq!(encoded.ids(), ids(&judge, &replaced));
    // A clean overwrite does not.
    let clean = prose(6, 3_000);
    assert_fast(
        session.overwrite(&clean).unwrap().execution(),
        "clean overwrite",
    );
}

#[cfg(feature = "sqlite")]
#[test]
fn a_durable_store_reopened_with_a_hit_in_a_tail_stays_exact() {
    let family = "qwen3_8b";
    let Some(root) = artifact_root(family) else {
        return;
    };
    let home = tempfile::tempdir().unwrap();
    let durable = || {
        load(
            builder(&root, Policy::Experimental)
                .home(home.path())
                .force_property_standing(guarded()),
            family,
        )
    };
    let judge = reference(&root, family);
    let mut text = prose(21, 6_000);
    let sealed_before;
    {
        let tokenizer = durable();
        let mut session = tokenizer.open_session("durable").unwrap();
        session.seed(&text).unwrap();
        let delta = prose(22, 1_500);
        session.append(&delta).unwrap();
        text.push_str(&delta);
        sealed_before = session.seal_for_test();
        let delta = format!("{} tail", HITS[0]);
        assert_guard_routed(session.append(&delta).unwrap().execution(), "hit delta");
        text.push_str(&delta);
        assert_eq!(session.seal_for_test(), sealed_before);
        session.close().unwrap();
    }
    let tokenizer = durable();
    let mut session = tokenizer.open_session("durable").unwrap();
    assert_eq!(session.snapshot().unwrap().ids(), ids(&judge, &text));
    assert_eq!(session.seal_for_test(), sealed_before, "after reopening");
    for turn in 0..3 {
        let delta = prose(30 + turn, 1_500);
        assert_guard_routed(
            session.append(&delta).unwrap().execution(),
            "after reopening",
        );
        text.push_str(&delta);
        assert_eq!(session.snapshot().unwrap().ids(), ids(&judge, &text));
        assert_eq!(session.seal_for_test(), sealed_before, "turn {turn}");
    }
}

#[test]
fn untabled_and_unread_builds_decline_the_fast_cpu_route() {
    let family = "qwen3_8b";
    let Some(root) = artifact_root(family) else {
        return;
    };
    let judge = reference(&root, family);
    // The control: on sources read as judged, a judged standing has the
    // route under the ordinary policies, so its absence below is the
    // guard's doing.
    for policy in [Policy::Supported, Policy::RequireAccelerated] {
        let tokenizer = load(
            builder(&root, policy)
                .force_sources_judged()
                .force_property_standing(Standing::Judged),
            family,
        );
        assert!(
            tokenizer.plan().backends.contains(&Backend::FastCpu),
            "{policy:?}: {:?}",
            tokenizer.plan()
        );
    }
    for (standing, reason) in [
        (untabled(), ReasonCode::PropertyDataUntabled),
        (unread(LocatedBy::None), ReasonCode::PropertyDataUnread),
        (unread(LocatedBy::Manifest), ReasonCode::PropertyDataUnread),
    ] {
        let tokenizer = load(
            builder(&root, Policy::Supported)
                .force_sources_judged()
                .force_property_standing(standing.clone()),
            family,
        );
        let plan = tokenizer.plan();
        assert!(!plan.backends.contains(&Backend::FastCpu), "{plan:?}");
        assert!(plan.reasons.contains(&reason), "{plan:?}");
        assert!(tokenizer.inner.guard_router.is_none());
        for text in HITS.iter().chain(CLEAN) {
            let encoded = tokenizer.encode(text).unwrap();
            assert_eq!(encoded.execution().backend, Backend::HuggingFace);
            assert_eq!(encoded.ids(), ids(&judge, text));
        }
        // Without an accelerated route, `RequireAccelerated` refuses the
        // tokenizer as it does for any build without one.
        let refused = builder(&root, Policy::RequireAccelerated)
            .force_sources_judged()
            .force_property_standing(standing.clone())
            .build()
            .unwrap()
            .load(family)
            .unwrap_err();
        assert_eq!(refused.code(), ErrorCode::UncertifiedTokenizer, "{refused}");
        // `Policy::Experimental` runs the route guarded.
        let experimental = forced(&root, family, standing.clone());
        assert!(experimental.inner.guard_router.is_some());
        assert!(!experimental.plan().reasons.contains(&reason));
        for text in HITS {
            let encoded = experimental.encode(text).unwrap();
            assert_guard_routed(encoded.execution(), "experimental");
            assert_eq!(encoded.ids(), ids(&judge, text));
        }
        assert_fast(experimental.encode(CLEAN[0]).unwrap().execution(), "clean");
        // `Policy::Reference` records no property-data reason.
        let reference_only = load(
            builder(&root, Policy::Reference).force_property_standing(standing),
            family,
        );
        assert!(!reference_only.plan().reasons.contains(&reason));
    }
}

#[test]
fn a_judged_build_builds_no_second_router_and_routes_as_before() {
    let family = "qwen3_8b";
    let Some(root) = artifact_root(family) else {
        return;
    };
    let judge = reference(&root, family);
    let tokenizer = forced(&root, family, Standing::Judged);
    assert!(tokenizer.inner.guard_router.is_none());
    for text in HITS.iter().chain(CLEAN) {
        let encoded = tokenizer.encode(text).unwrap();
        assert_ne!(
            encoded.execution().reason,
            Some(ReasonCode::InputGuardRouted),
            "{text:?}"
        );
        assert_eq!(encoded.ids(), ids(&judge, text), "{text:?}");
    }
    let mut session = tokenizer.open_session("judged").unwrap();
    let seeded = session
        .seed(&format!("{} {}", prose(1, 500), HITS[0]))
        .unwrap();
    assert_eq!(seeded.execution().backend, Backend::FastCpu);
    assert_eq!(seeded.execution().reason, None);
    // The unforced tokenizer of this workspace is the same.
    let own = load(builder(&root, Policy::Experimental), family);
    assert!(own.inner.guard_router.is_none());
    assert_fast(own.encode(HITS[0]).unwrap().execution(), "own standing");
}

#[test]
fn the_workspace_build_reads_the_judged_property_data() {
    let standing = super::standing();
    assert_eq!(
        standing,
        &Standing::Judged,
        "this build located its lockfile by {:?}; a build of this workspace with a target \
         directory outside the tree reads Unread unless TOKTIER_CARGO_LOCK names the \
         workspace Cargo.lock: {standing}",
        super::LOCATED_BY_RECORD
    );
    assert!(
        matches!(super::LOCATED_BY_RECORD, "out_dir" | "named"),
        "{}",
        super::LOCATED_BY_RECORD
    );
    let doctor = Runtime::builder()
        .device(Device::Cpu)
        .build()
        .unwrap()
        .doctor();
    assert_eq!(doctor.runtime_build.property_data, "judged");
    assert_eq!(
        doctor.runtime_build.property_lock_located_by,
        super::LOCATED_BY_RECORD
    );
    // The record names exactly the judged copies of the two crates.
    let copies = super::parse_copies(super::PROPERTY_DATA_RECORD).unwrap();
    assert_eq!(copies.len(), 2, "{copies:?}");
    // The guarded line says what the guard does.
    assert!(guarded()
        .to_string()
        .contains(&MOVED_CODE_POINTS.to_string()));
}

/// A lookup's ids are a holder's sealed prefix and the encode of the
/// rest. The rest is scanned like any text; the prefix is not, because a
/// seal on a guarded build is only certified on a tail without a listed
/// code point, and a seal a judged build certified was certified on the
/// judged classes, which are the reference engine's. So (a) a listed code
/// point inside a reused prefix is served from that prefix, and (b) one in
/// the rest sends the rest to the reference engine.
#[cfg(feature = "sqlite")]
#[test]
fn a_lookup_reuses_sealed_prefixes_and_scans_the_rest() {
    let family = "qwen3_8b";
    let Some(root) = artifact_root(family) else {
        return;
    };
    let judge = reference(&root, family);
    let reused = |tokenizer: &Tokenizer| tokenizer.store_stats().lookup_reused_ids;

    // (b) A guarded session sealed before a hit; the lookup reuses the
    // prefix before the seal and the rest, which holds the hit, runs on
    // the reference engine.
    let tokenizer = forced(&root, family, guarded());
    let mut text = prose(41, 6_000);
    let mut session = tokenizer.open_session("rest").unwrap();
    session.seed(&text).unwrap();
    for turn in 0..3 {
        let delta = prose(42 + turn, 1_500);
        session.append(&delta).unwrap();
        text.push_str(&delta);
    }
    let seal = session.seal_for_test();
    assert!(seal.1 > 0, "{seal:?}");
    let delta = format!(" {} ", HITS[1]);
    session.append(&delta).unwrap();
    text.push_str(&delta);
    let delta = prose(50, 5_000);
    session.append(&delta).unwrap();
    text.push_str(&delta);
    session.close().unwrap();
    let before = reused(&tokenizer);
    let query = format!("{text} and a clean question");
    let found = tokenizer.lookup(&query).unwrap().expect("a lookup hit");
    assert_eq!(found.ids(), ids(&judge, &query));
    assert!(reused(&tokenizer) > before, "the sealed prefix was reused");
    assert_guard_routed(found.execution(), "a rest that holds a hit");

    // (a) A session a judged build sealed past a hit, read by a guarded
    // build from the durable store: the reused prefix holds the hit, the
    // rest is clean and runs on the fast engine.
    let home = tempfile::tempdir().unwrap();
    let durable = |standing: Standing| {
        load(
            builder(&root, Policy::Experimental)
                .home(home.path())
                .force_property_standing(standing),
            family,
        )
    };
    let mut text = format!("{} {} ", prose(61, 6_000), HITS[0]);
    let hit_chars = text.chars().count() as u64;
    {
        let judged = durable(Standing::Judged);
        let mut session = judged.open_session("prefix").unwrap();
        session.seed(&text).unwrap();
        for turn in 0..4 {
            let delta = prose(62 + turn, 1_500);
            session.append(&delta).unwrap();
            text.push_str(&delta);
        }
        let seal = session.seal_for_test();
        assert!(
            seal.0 > hit_chars,
            "the judged build sealed past the hit: {seal:?}"
        );
        session.close().unwrap();
    }
    let guarded_tokenizer = durable(guarded());
    let mut session = guarded_tokenizer.open_session("prefix").unwrap();
    // Blocks completed after the reopening carry the session's seal.
    for turn in 0..3 {
        let delta = prose(70 + turn, 4_096);
        assert_fast(session.append(&delta).unwrap().execution(), "clean tail");
        text.push_str(&delta);
    }
    assert!(session.seal_for_test().0 > hit_chars);
    session.close().unwrap();
    let before = reused(&guarded_tokenizer);
    let query = format!("{text} and a clean question");
    let found = guarded_tokenizer
        .lookup(&query)
        .unwrap()
        .expect("a lookup hit");
    assert_eq!(found.ids(), ids(&judge, &query));
    assert!(
        reused(&guarded_tokenizer) > before,
        "the prefix with the hit was reused"
    );
    assert_eq!(found.execution().backend, Backend::FastCpu);
    assert_eq!(found.execution().reason, None);
}
