//! The chunk-parallel surface of the `toktier` crate on real artifacts.
//!
//! Artifact tier: every shipped family found under `TOKTIER_TEST_ARTIFACTS`
//! (or the default artifact cache) is loaded, its admission is compared
//! with the roster and the shipped chunked admission table (admitted by
//! the row's status, the native path's evidence reported as a label), a
//! refused tokenizer is shown to plan nothing and encode serially, an
//! admitted one under the default policy is encoded on whichever branch
//! its route takes (chunked on the corrected CPU engine, or serially
//! under `ChunkRouteNotFastCpu` on a route without it), and under
//! `Policy::Experimental`, which lifts the evidence step and admits the
//! engine of this build, the chunked ids are compared with the serial
//! encode of the same route and with the reference engine. A family whose
//! artifact is absent is skipped by name; a host with none skips the
//! tier.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use toktier::chunked::{default_chunks, AdmissionGate, ChunkOptions};
use toktier::{Backend, Certification, Device, ErrorCode, Policy, ReasonCode, Runtime, Tokenizer};

const SHIPPED: [&str; 15] = [
    "deepseek_v3",
    "deepseek_v4_flash",
    "glm_5_2",
    "gpt_oss_120b",
    "hy3",
    "kimi_k3",
    "laguna_s_2_1",
    "ling_3_0_flash",
    "llama_3_1_8b",
    "minimax_m3",
    "ministral_3_8b",
    "nemotron_3_nano_4b",
    "olmo_3_7b",
    "qwen3_5_08b",
    "qwen3_8b",
];

/// The shipped families outside the certified repair roster.
const NON_ROSTER: [&str; 4] = ["hy3", "kimi_k3", "laguna_s_2_1", "ling_3_0_flash"];

fn artifact_root() -> Option<PathBuf> {
    std::env::var_os("TOKTIER_TEST_ARTIFACTS")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".cache/toktier/artifacts"))
        })
        .filter(|root| root.is_dir())
}

fn artifact_directory(root: &Path, family: &str) -> Option<PathBuf> {
    let prefix = format!("{family}-");
    let mut named = std::fs::read_dir(root)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix))
                && path.join("tokenizer.json").is_file()
        })
        .collect::<Vec<_>>();
    named.sort();
    named
        .into_iter()
        .next()
        .or_else(|| Some(root.join(family)).filter(|path| path.join("tokenizer.json").is_file()))
}

/// `family` under `policy`, or `None` (with a note) when this host has
/// no verifiable artifact for it.
fn load(family: &str, policy: Policy) -> Option<Tokenizer> {
    let root = artifact_root()?;
    let directory = artifact_directory(&root, family)?;
    let runtime = Runtime::builder()
        .device(Device::Cpu)
        .policy(policy)
        .build()
        .unwrap();
    match runtime.load_local(family, &directory) {
        Ok(tokenizer) => Some(tokenizer),
        Err(error) => {
            eprintln!("skip {family}: {} ({:?})", error.message(), error.code());
            None
        }
    }
}

/// The shipped admission table, as the crate embeds it.
fn table() -> BTreeMap<String, serde_json::Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/src/toktier/routing/tables/chunked_admission.v1.json");
    let document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    document["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| (row["family"].as_str().unwrap().to_owned(), row.clone()))
        .collect()
}

fn strings(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// The refusal the documented rule gives a roster artifact whose
/// structural step passed, from its table row: no row, then the row's
/// status (the one the Python face admits by), then the loader face.
fn expected_table_refusal(row: Option<&serde_json::Value>, sha256: &str) -> Option<String> {
    let Some(row) = row else {
        return Some("artifact_without_chunked_evidence".to_owned());
    };
    let status = row["status"].as_str().unwrap();
    if status != "certified" {
        return Some(format!("chunked_evidence_incomplete:{status}"));
    }
    row["loader_face_sha256"]
        .as_str()
        .filter(|face| *face != sha256)
        .map(|_| {
            format!(
                "loader_face_differs_from_chunked_evidence:{}",
                &sha256[..12]
            )
        })
}

/// The label the documented rule gives an admitted artifact from its
/// row's native-path levels: none with both, the status they spell
/// otherwise.
fn expected_label(row: Option<&serde_json::Value>) -> Option<String> {
    let levels = row
        .map(|row| strings(&row["evidence"]["native"]))
        .unwrap_or_default();
    let adversarial = levels.iter().any(|level| level == "adversarial");
    let corpus = levels.iter().any(|level| level == "corpus");
    match (adversarial, corpus) {
        (true, true) => None,
        (true, false) => Some("native_evidence:adversarial_only".to_owned()),
        (false, true) => Some("native_evidence:corpus_only".to_owned()),
        (false, false) => Some("native_evidence:none".to_owned()),
    }
}

/// Whether the shipped row certifies the artifact for the Python face.
fn certified_row(row: Option<&serde_json::Value>) -> bool {
    row.is_some_and(|row| row["status"].as_str() == Some("certified"))
}

/// A deterministic long text: prose with punctuation, numbers, CJK,
/// line breaks and special-token literals.
fn generated(chars: usize, seed: u64) -> String {
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dogs",
        "1990s",
        "42",
        "don't",
        "caf\u{e9}",
        "\u{4e2d}\u{6587}",
        "x.y",
        "a,b",
        "tokenization",
        "is",
        "a",
        "solved",
        "problem,",
        "nobody",
        "agrees.",
    ];
    let mut state = seed;
    let mut out = String::new();
    let mut count = 0;
    while count < chars {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // A special-token literal every hundred thousand characters or
        // so: a chunk that holds one is encoded by the reference whole,
        // so most chunks stay on the corrected engine and a few meet the
        // literal guard.
        let word = if (state >> 40).is_multiple_of(20_011) {
            "<|im_start|>"
        } else if (state >> 40).is_multiple_of(20_021) {
            "<|endoftext|>"
        } else {
            words[(state >> 33) as usize % words.len()]
        };
        out.push_str(word);
        count += word.chars().count() + 1;
        out.push(if (state >> 20).is_multiple_of(11) {
            '\n'
        } else {
            ' '
        });
    }
    out
}

#[test]
fn admission_follows_the_roster_the_structural_rule_and_the_shipped_table() {
    let rows = table();
    let mut checked = 0;
    for family in SHIPPED {
        let Some(tokenizer) = load(family, Policy::default()) else {
            continue;
        };
        checked += 1;
        let admission = tokenizer.chunked().admission().unwrap();
        assert_eq!(admission.family, family);
        assert_eq!(
            admission.artifact_sha256,
            tokenizer.artifact().identity().tokenizer_sha256
        );
        assert_eq!(admission.gate, AdmissionGate::EvidenceTable);
        assert_eq!(admission.admitted, admission.refusal.is_none());
        assert_eq!(
            admission.cpu_route,
            tokenizer.plan().backends.contains(&Backend::FastCpu),
            "{family}"
        );
        let row = rows.get(family);
        if NON_ROSTER.contains(&family) {
            assert_eq!(
                admission.refusal.as_deref(),
                Some("artifact_not_in_certified_repair_roster"),
                "{family}"
            );
            assert_eq!(admission.normalizer, None);
            assert_eq!(admission.literals, 0);
            assert!(
                row.is_none(),
                "{family} is not on the roster and has no row"
            );
            assert_eq!(admission.evidence_id, None);
            assert_eq!(admission.evidence_status, None);
            assert!(admission.native_evidence.is_empty());
            // No row: the label says the native path holds no evidence.
            assert_eq!(
                admission.evidence_label.as_deref(),
                Some("native_evidence:none")
            );
            continue;
        }
        // Every roster artifact this release ships passes the structural
        // rule (no packaged artifact sets lstrip, rstrip or single_word,
        // and the normalizers are none or NFC), so the table decides.
        let row = row.unwrap_or_else(|| panic!("{family}: a roster family with no row"));
        assert_eq!(
            admission.normalizer.as_deref(),
            row["normalizer"].as_str(),
            "{family}"
        );
        assert!(admission.literals > 0, "{family} has added tokens");
        assert_eq!(
            admission.evidence_id.as_deref(),
            row["evidence_id"].as_str(),
            "{family}"
        );
        assert_eq!(
            admission.native_evidence,
            strings(&row["evidence"]["native"]),
            "{family}"
        );
        assert_eq!(
            admission.refusal,
            expected_table_refusal(Some(row), &admission.artifact_sha256),
            "{family}"
        );
        // The Python face's status decides; the native levels only label,
        // admitted or not. In this release every row holds the battery
        // alone for the native path, so the label reads adversarial_only
        // on all of them, and the uncertified rows answer with the Python
        // face's string.
        assert_eq!(
            admission.evidence_label,
            expected_label(Some(row)),
            "{family}"
        );
        if strings(&row["evidence"]["native"]) == ["adversarial"] {
            assert_eq!(
                admission.evidence_status.as_deref(),
                Some("adversarial_only"),
                "{family}"
            );
            assert_eq!(
                admission.evidence_label.as_deref(),
                Some("native_evidence:adversarial_only"),
                "{family}"
            );
        }
        let same_face = row["loader_face_sha256"]
            .as_str()
            .is_none_or(|face| face == admission.artifact_sha256);
        eprintln!(
            "{family}: default policy admitted={} refusal={:?} evidence_label={:?} cpu_route={}",
            admission.admitted, admission.refusal, admission.evidence_label, admission.cpu_route
        );
        if certified_row(Some(row)) && same_face {
            assert!(admission.admitted, "{family}: {:?}", admission.refusal);
            assert_eq!(admission.refusal, None, "{family}");
        } else {
            assert!(!admission.admitted, "{family}");
            if !certified_row(Some(row)) {
                assert_eq!(
                    admission.refusal,
                    Some(format!(
                        "chunked_evidence_incomplete:{}",
                        row["status"].as_str().unwrap()
                    )),
                    "{family}"
                );
            }
        }
        // The same decision, decided once: a second handle answers alike.
        assert_eq!(tokenizer.chunked().admission().unwrap(), admission);
    }
    if checked == 0 {
        eprintln!("skip: no shipped artifact under TOKTIER_TEST_ARTIFACTS or the cache");
    }
}

#[test]
fn the_default_policy_admits_the_certified_rows_and_labels_the_native_evidence() {
    // One encode per admitted family on whichever branch its route takes:
    // the corrected CPU engine cuts the text into the requested chunks;
    // a route without it encodes serially and says so. Both are exact.
    let rows = table();
    let text = generated(300_000, 7);
    let mut branches = BTreeMap::new();
    for family in SHIPPED {
        let row = rows.get(family);
        if !certified_row(row) {
            continue;
        }
        let Some(tokenizer) = load(family, Policy::default()) else {
            continue;
        };
        let chunked = tokenizer.chunked();
        let admission = chunked.admission().unwrap();
        assert_eq!(admission.gate, AdmissionGate::EvidenceTable);
        assert!(admission.admitted, "{family}: {:?}", admission.refusal);
        assert_eq!(admission.refusal, None);
        let label = expected_label(row);
        assert_eq!(admission.evidence_label, label, "{family}");
        let options = ChunkOptions::default().k(8);
        let encoded = chunked.encode(&text, &options).unwrap();
        let serial = tokenizer.encode(&text).unwrap();
        assert_eq!(encoded.ids(), serial.ids(), "{family}");
        assert_eq!(encoded.certification, tokenizer.plan().certification);
        assert_eq!(encoded.k_requested, 8);
        assert!(!encoded.below_threshold);
        let stats = chunked.stats();
        assert_eq!(stats.evidence_label, label, "{family}");
        assert_eq!(stats.calls, 1);
        if admission.cpu_route {
            branches.insert(family, "chunked on the corrected CPU engine");
            assert_eq!(encoded.serial_reason, None, "{family}");
            assert_eq!(encoded.chunks, 8, "{family}");
            assert_eq!(encoded.cuts.len(), 7, "{family}");
            assert_eq!(encoded.evidence_label, label, "{family}");
            assert_eq!((stats.applied, stats.serial), (1, 0));
            assert!(stats.fallback_counts.is_empty());
            let plan = chunked.plan(&text, &options).unwrap();
            assert_eq!(plan.cuts(), encoded.cuts, "{family}");
            if !cfg!(debug_assertions) {
                let reference = load(family, Policy::Reference).unwrap();
                assert_eq!(
                    encoded.ids(),
                    reference.encode(&text).unwrap().ids(),
                    "{family}: the reference engine"
                );
            }
        } else {
            branches.insert(family, "serial under ChunkRouteNotFastCpu");
            assert_eq!(
                encoded.serial_reason,
                Some(ReasonCode::ChunkRouteNotFastCpu),
                "{family}"
            );
            assert_eq!(encoded.chunks, 1);
            assert!(encoded.cuts.is_empty());
            assert_eq!(encoded.evidence_label, None, "{family}");
            assert_eq!((stats.applied, stats.serial), (0, 1));
            assert_eq!(
                stats.fallback_counts,
                BTreeMap::from([("R_CHUNK_ROUTE_NOT_FAST_CPU".to_owned(), 1)])
            );
            let error = chunked.plan(&text, &options).unwrap_err();
            assert_eq!(error.code(), ErrorCode::UncertifiedTokenizer, "{family}");
            let error = chunked.warm(Some(2), None).unwrap_err();
            assert_eq!(error.code(), ErrorCode::UncertifiedTokenizer, "{family}");
        }
    }
    for (family, branch) in &branches {
        eprintln!("{family}: default policy, {branch}");
    }
    if branches.is_empty() {
        eprintln!("skip: no artifact with a certified row on this host");
    }
}

#[test]
fn a_refused_tokenizer_plans_nothing_and_encodes_serially() {
    let text = generated(300_000, 3);
    let mut checked = 0;
    for family in SHIPPED {
        let Some(tokenizer) = load(family, Policy::default()) else {
            continue;
        };
        let chunked = tokenizer.chunked();
        let admission = chunked.admission().unwrap();
        let Some(refusal) = admission.refusal.clone() else {
            continue;
        };
        checked += 1;
        let error = chunked
            .plan(&text, &ChunkOptions::default().k(4))
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::UncertifiedTokenizer, "{family}");
        assert!(
            error.message().contains(&refusal),
            "{family}: {}",
            error.message()
        );
        assert_eq!(error.family(), Some(family));
        let encoded = chunked
            .encode(&text, &ChunkOptions::default().k(4))
            .unwrap();
        assert_eq!(encoded.chunks, 1);
        assert_eq!(encoded.k_requested, 4);
        assert!(encoded.cuts.is_empty());
        assert_eq!(
            encoded.serial_reason,
            Some(ReasonCode::ChunkFamilyUnsupported)
        );
        assert!(!encoded.below_threshold);
        assert_eq!(
            encoded.ids(),
            tokenizer.encode(&text).unwrap().ids(),
            "{family}"
        );
        assert_eq!(encoded.certification, tokenizer.plan().certification);
        // A serial result carries no label; the counters carry the row's.
        assert_eq!(encoded.evidence_label, None, "{family}");
        let stats = chunked.stats();
        assert_eq!((stats.calls, stats.serial, stats.applied), (1, 1, 0));
        assert_eq!(
            stats.fallback_counts,
            BTreeMap::from([("R_CHUNK_FAMILY_UNSUPPORTED".to_owned(), 1)])
        );
        assert_eq!(stats.evidence_label, admission.evidence_label, "{family}");
        if !admission.cpu_route {
            let error = chunked.warm(Some(2), None).unwrap_err();
            assert_eq!(error.code(), ErrorCode::UncertifiedTokenizer);
        }
        // Malformed options are refused before anything is encoded.
        for options in [
            ChunkOptions::default().k(0),
            ChunkOptions::default().scan_budget(0),
        ] {
            let error = chunked.encode(&text, &options).unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidArgument);
            let error = chunked.plan(&text, &options).unwrap_err();
            assert_eq!(error.code(), ErrorCode::InvalidArgument);
        }
        assert_eq!(chunked.stats().calls, 1);
        if checked == 2 {
            break;
        }
    }
    if checked == 0 {
        eprintln!("skip: no refused shipped artifact on this host");
    }
}

#[test]
fn experimental_lifts_the_table_and_chunked_ids_equal_the_serial_encode() {
    let rows = table();
    // The release profile is where the engine runs at speed; the
    // development profile keeps the same shapes at a third of the size,
    // every text still above the chunk table's first step.
    let long = if cfg!(debug_assertions) {
        400_000
    } else {
        1_200_000
    };
    let texts = [
        generated(300_000, 1),
        generated(long, 2),
        format!(
            "{}\n{}",
            generated(40_000, 4),
            "x = f(y)\r\n".repeat(30_000)
        ),
    ];
    let mut checked = 0;
    for family in SHIPPED {
        if NON_ROSTER.contains(&family) {
            continue;
        }
        let Some(tokenizer) = load(family, Policy::Experimental) else {
            continue;
        };
        let chunked = tokenizer.chunked();
        let admission = chunked.admission().unwrap();
        assert_eq!(admission.gate, AdmissionGate::LiftedByExperimentalPolicy);
        assert!(admission.admitted, "{family}: {:?}", admission.refusal);
        let label = expected_label(rows.get(family));
        assert_eq!(admission.evidence_label, label, "{family}");
        if !admission.cpu_route {
            eprintln!("skip {family}: no corrected CPU route under Policy::Experimental");
            continue;
        }
        checked += 1;
        assert_eq!(tokenizer.plan().certification, Certification::Experimental);
        assert!(chunked.warm(Some(3), Some(&texts[0][..2_000])).unwrap() >= 3);
        // The whole matrix against the reference engine on every family
        // in the release profile; in the development profile on three
        // families (both normalizer shapes), and the first text at one
        // chunk count against the route's own serial encode on the rest.
        let full = !cfg!(debug_assertions)
            || matches!(family, "qwen3_8b" | "llama_3_1_8b" | "deepseek_v3");
        let reference = full.then(|| load(family, Policy::Reference).unwrap());
        let (texts, ks): (&[String], &[Option<usize>]) = if full {
            (&texts, &[None, Some(4), Some(32)])
        } else {
            (&texts[..1], &[Some(4)])
        };
        let encode_reference = |text: &str| match &reference {
            Some(reference) => reference.encode(text).unwrap(),
            None => tokenizer.encode(text).unwrap(),
        };
        let mut calls = 0u64;
        for text in texts {
            let expected = encode_reference(text);
            let serial = tokenizer.encode(text).unwrap();
            assert_eq!(serial.ids(), expected.ids(), "{family}: the route itself");
            for &k in ks {
                calls += 1;
                let mut options = ChunkOptions::default();
                options.k = k;
                let encoded = chunked.encode(text, &options).unwrap();
                assert_eq!(encoded.ids(), expected.ids(), "{family} k={k:?}");
                assert_eq!(encoded.serial_reason, None, "{family} k={k:?}");
                assert_eq!(encoded.evidence_label, label, "{family} k={k:?}");
                assert!(!encoded.below_threshold);
                assert_eq!(encoded.certification, Certification::Experimental);
                let requested = k.unwrap_or_else(|| default_chunks(text.chars().count()));
                assert_eq!(encoded.k_requested, requested);
                assert!(encoded.chunks >= 2, "{family} k={k:?}");
                assert!(encoded.chunks <= requested);
                assert_eq!(encoded.cuts.len() + 1, encoded.chunks);
                let sources = encoded.sources;
                assert_eq!(
                    sources.gigatoken + sources.hf_added_token + sources.hf_engine_guard,
                    encoded.chunks
                );
                let plan = chunked.plan(text, &options).unwrap();
                assert_eq!(plan.cuts(), encoded.cuts, "{family} k={k:?}");
                assert_eq!(plan.spans.len(), encoded.chunks);
                assert_eq!(plan.spans[0].0, 0);
                assert_eq!(plan.spans.last().unwrap().1, plan.chars);
                assert_eq!(plan.chars, text.chars().count());
                assert_eq!(plan.byte_spans.last().unwrap().1, text.len());
                assert!(plan.spans.windows(2).all(|pair| pair[0].1 == pair[1].0));
                assert_eq!(
                    (plan.cut_misses, plan.literal_rejects, plan.norm_rejects),
                    (
                        encoded.cut_misses,
                        encoded.literal_rejects,
                        encoded.norm_rejects
                    )
                );
            }
        }
        // A text without a literal keeps every row on the corrected
        // engine, which is what the parallel section is for.
        let plain = generated(300_000, 6)
            .replace("<|im_start|>", "")
            .replace("<|endoftext|>", "");
        let encoded = chunked
            .encode(&plain, &ChunkOptions::default().k(4))
            .unwrap();
        assert_eq!(encoded.ids(), encode_reference(&plain).ids());
        assert_eq!(encoded.sources.gigatoken, encoded.chunks, "{family}");
        // Below the threshold and k = 1 are serial without a reason; a
        // document with no certified cut says so.
        let short = chunked
            .encode("tiny text", &ChunkOptions::default().k(4))
            .unwrap();
        assert!(short.below_threshold && short.serial_reason.is_none() && short.chunks == 1);
        assert_eq!(short.evidence_label, None);
        assert_eq!(short.ids(), encode_reference("tiny text").ids());
        let one = chunked
            .encode(&texts[0], &ChunkOptions::default().k(1))
            .unwrap();
        assert!(one.below_threshold && one.chunks == 1);
        let cjk = "\u{4e2d}".repeat(40_000);
        let uncut = chunked.encode(&cjk, &ChunkOptions::default().k(4)).unwrap();
        assert_eq!(
            uncut.serial_reason,
            Some(ReasonCode::ChunkNoCut),
            "{family}"
        );
        assert_eq!((uncut.chunks, uncut.cut_misses), (1, 3));
        assert_eq!(uncut.evidence_label, None);
        assert_eq!(uncut.ids(), encode_reference(&cjk).ids());
        let stats = chunked.stats();
        assert_eq!(stats.calls, calls + 4);
        assert_eq!(stats.applied, calls + 1);
        assert_eq!(stats.below_threshold, 2);
        assert_eq!(stats.serial, 1);
        assert_eq!(
            stats.fallback_counts,
            BTreeMap::from([("R_CHUNK_NO_CUT".to_owned(), 1)])
        );
        assert!(stats.workers >= 3);
        assert!(stats.default_workers >= 1);
        assert_eq!(stats.evidence_label, label);
    }
    if checked == 0 {
        eprintln!("skip: no roster artifact reached the corrected CPU route on this host");
    }
}

#[test]
fn a_reference_route_encodes_serially_and_names_the_missing_engine() {
    let Some(tokenizer) = load("qwen3_8b", Policy::Reference) else {
        return;
    };
    let chunked = tokenizer.chunked();
    let admission = chunked.admission().unwrap();
    // The table still decides under Policy::Reference; what this route
    // lacks is the engine, which is reported apart from the refusal.
    assert_eq!(admission.gate, AdmissionGate::EvidenceTable);
    assert_eq!(admission.admitted, certified_row(table().get("qwen3_8b")));
    assert!(!admission.cpu_route);
    assert_eq!(tokenizer.plan().backends, vec![Backend::HuggingFace]);
    let text = generated(300_000, 5);
    let encoded = chunked
        .encode(&text, &ChunkOptions::default().k(4))
        .unwrap();
    assert_eq!(encoded.chunks, 1);
    assert_eq!(encoded.ids(), tokenizer.encode(&text).unwrap().ids());
    assert_eq!(encoded.evidence_label, None);
    let reason = if admission.admitted {
        ReasonCode::ChunkRouteNotFastCpu
    } else {
        ReasonCode::ChunkFamilyUnsupported
    };
    assert_eq!(encoded.serial_reason, Some(reason));
    let error = chunked
        .plan(&text, &ChunkOptions::default().k(4))
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::UncertifiedTokenizer);
}
