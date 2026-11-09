//! `add_special_tokens` on the plain route, on real artifacts.
//!
//! Artifact tier: every family found under `TOKTIER_TEST_ARTIFACTS` (or
//! the default artifact cache) is loaded, its post-processor class is
//! compared with the shape its document carries, and a request with
//! special tokens is compared with the reference engine's own answer for
//! the same document, single and batched, on probe texts and on texts
//! dense in the artifact's added-token literals. Where the class is
//! `identity` or `prefix` the request must have been served on the plain
//! route, not by the reference's post-processed path; where it is
//! `reference` it must have been. A host with no artifact skips the tier.
//!
//! `TOKTIER_TEST_ARTIFACT_FAMILIES`, when set, is the number of families
//! the tier has to check (the release gate points `TOKTIER_TEST_ARTIFACTS`
//! at the 21-family seed and sets it to 21), so a short cache cannot pass
//! by checking fewer faces.

use std::path::{Path, PathBuf};

use toktier::{Device, EncodeOptions, ErrorCode, Policy, ReasonCode, Runtime};
use toktier_routing_core::ReferenceEngine;

/// The four shipped artifacts whose template places one special token
/// before the sequence, with that token's id.
const PREFIX: [(&str, u32); 4] = [
    ("laguna_s_2_1", 2),
    ("llama_3_1_8b", 128_000),
    ("ministral_3_8b", 1),
    ("step5_preview", 0),
];

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

/// `(family, directory)` for every `<family>-<revision>` directory that
/// holds a `tokenizer.json`, in name order.
fn artifacts(root: &Path) -> Vec<(String, PathBuf)> {
    let mut found = std::fs::read_dir(root)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.join("tokenizer.json").is_file())
                .filter_map(|path| {
                    let name = path.file_name()?.to_str()?.to_owned();
                    let (family, _) = name.rsplit_once('-')?;
                    Some((family.to_owned(), path))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    found.sort();
    found
}

/// Probe texts and texts dense in the artifact's own added-token
/// literals: alone, doubled, at both ends, between letters, digits,
/// whitespace and non-ASCII text.
fn texts(reference: &ReferenceEngine) -> Vec<String> {
    let mut out = vec![
        String::new(),
        "a".to_owned(),
        " ".to_owned(),
        "\n\n".to_owned(),
        "hello world".to_owned(),
        "The quick brown fox jumps over the lazy dog, 1990s-style; don't.".to_owned(),
        "caf\u{e9} \u{4e2d}\u{6587} \u{1f600} na\u{ef}ve e\u{301}".to_owned(),
        "fn main() {\n    println!(\"{}\", 42);\n}\n".to_owned(),
    ];
    let literals = reference
        .added_tokens()
        .into_iter()
        .map(|(_, content, _)| content)
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>();
    let neighbours = ["", "a", "1", " ", "\n", ".", "\u{4e2d}", "\u{e9}"];
    for (index, literal) in literals.iter().enumerate().take(32) {
        let left = neighbours[index % neighbours.len()];
        let right = neighbours[(index / neighbours.len() + index) % neighbours.len()];
        out.push(literal.clone());
        out.push(format!("{literal}{literal}"));
        out.push(format!("{left}{literal}{right}"));
        out.push(format!("{literal} user text {index} {literal}"));
    }
    if literals.len() > 1 {
        let chat = literals
            .iter()
            .take(16)
            .enumerate()
            .map(|(turn, literal)| format!("{literal}turn {turn} says hello\n"))
            .collect::<String>();
        out.push(chat);
    }
    out
}

/// The number of families the tier has to check, when the gate names it.
fn expected_families() -> Option<usize> {
    std::env::var("TOKTIER_TEST_ARTIFACT_FAMILIES")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("TOKTIER_TEST_ARTIFACT_FAMILIES is a count")
        })
}

fn with_special_tokens() -> EncodeOptions {
    EncodeOptions {
        add_special_tokens: true,
        offsets: false,
    }
}

#[test]
fn special_tokens_follow_the_reference_and_run_the_plain_route_where_the_class_allows() {
    let Some(root) = artifact_root() else {
        eprintln!("skip: no artifact root on this host");
        return;
    };
    let mut checked = 0;
    let mut classes = std::collections::BTreeMap::new();
    for (family, directory) in artifacts(&root) {
        let reference = ReferenceEngine::from_file(directory.join("tokenizer.json")).unwrap();
        let texts = texts(&reference);
        // The experimental policy admits the corrected CPU engine of this
        // build whatever the shipped register says about its sources, so
        // the plain route is an accelerated one wherever the artifact has
        // one; the default policy is checked too, on whatever it admits.
        for policy in [Policy::Experimental, Policy::default()] {
            let runtime = Runtime::builder()
                .device(Device::Cpu)
                .policy(policy)
                .build()
                .unwrap();
            let tokenizer = match runtime.load_local(&family, &directory) {
                Ok(tokenizer) => tokenizer,
                Err(error) => {
                    eprintln!("skip {family}: {} ({:?})", error.message(), error.code());
                    break;
                }
            };
            let plan = tokenizer.plan();
            assert_eq!(plan.postprocessor_self_check, None, "{family}");
            let expected_prefix = PREFIX
                .iter()
                .find(|(name, _)| *name == family)
                .map(|(_, id)| vec![*id]);
            match plan.postprocessor_class.as_str() {
                "prefix" => assert_eq!(
                    Some(plan.special_token_prefix.clone()),
                    expected_prefix,
                    "{family}"
                ),
                "identity" => {
                    assert_eq!(expected_prefix, None, "{family}");
                    assert!(plan.special_token_prefix.is_empty(), "{family}");
                }
                "reference" => {
                    assert_eq!(expected_prefix, None, "{family}");
                    assert!(plan.special_token_prefix.is_empty(), "{family}");
                }
                other => panic!("{family}: unknown class {other}"),
            }
            classes.insert(family.clone(), plan.postprocessor_class.clone());
            let served_plainly = plan.postprocessor_class != "reference";

            for text in &texts {
                let expected = reference.encode_ids(text, true).unwrap();
                let encoded = tokenizer.encode_with(text, with_special_tokens()).unwrap();
                assert_eq!(encoded.ids(), &expected[..], "{family} {policy:?} {text:?}");
                let facts = encoded.execution();
                if served_plainly {
                    assert_ne!(facts.path, "hf_postprocessed", "{family} {text:?}");
                    assert_ne!(
                        facts.reason,
                        Some(ReasonCode::InputPostprocessRouted),
                        "{family} {text:?}"
                    );
                } else {
                    assert_eq!(facts.path, "hf_postprocessed", "{family} {text:?}");
                }
                // Without special tokens the answer is the core stream, as
                // before.
                assert_eq!(
                    tokenizer.encode(text).unwrap().ids(),
                    &reference.encode_ids(text, false).unwrap()[..],
                    "{family} {text:?}"
                );
            }
            let batch = tokenizer
                .encode_batch_with(&texts, with_special_tokens())
                .unwrap();
            assert_eq!(batch.len(), texts.len(), "{family}");
            for (index, text) in texts.iter().enumerate() {
                assert_eq!(
                    batch.row(index).unwrap(),
                    &reference.encode_ids(text, true).unwrap()[..],
                    "{family} batch row {index}"
                );
            }
            if served_plainly {
                for execution in batch.executions() {
                    assert_ne!(execution.path, "hf_postprocessed", "{family}");
                }
            }
            checked += 1;
        }
    }
    eprintln!("checked {checked} (family, policy) pairs: {classes:?}");
    if let Some(expected) = expected_families() {
        assert_eq!(
            classes.len(),
            expected,
            "the artifact tier checked {} families, {expected} expected: {:?}",
            classes.len(),
            classes.keys().collect::<Vec<_>>()
        );
        // Every family on both policies.
        assert_eq!(checked, 2 * expected);
    }
    for (family, _) in PREFIX {
        if let Some(class) = classes.get(family) {
            assert_eq!(class, "prefix", "{family}");
        }
    }
    for family in ["minimax_m3", "olmo_3_7b", "qwen3_8b", "deepseek_v3"] {
        if let Some(class) = classes.get(family) {
            assert_eq!(class, "identity", "{family}");
        }
    }
}

/// Offsets with special tokens stay outside the single-encode surface on
/// every class.
#[test]
fn offsets_with_special_tokens_are_still_refused() {
    let Some(root) = artifact_root() else {
        return;
    };
    let runtime = Runtime::builder().device(Device::Cpu).build().unwrap();
    for (family, directory) in artifacts(&root).into_iter().take(4) {
        let Ok(tokenizer) = runtime.load_local(&family, &directory) else {
            continue;
        };
        let error = tokenizer
            .encode_with(
                "offsets and special tokens",
                EncodeOptions {
                    add_special_tokens: true,
                    offsets: true,
                },
            )
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidArgument, "{family}");
    }
}

/// `deepseek_v3` keeps the reference document's meaning of
/// `add_special_tokens`: its post-processor adds nothing, so the class is
/// `identity` and the ids equal the core stream (a 4.57.x `transformers`
/// loader would add a leading BOS; 5.x loaders do not). The Rust face has
/// no `final_ids`; the session's seeded ids stand in for it, since an
/// `identity` face adds nothing at read time.
#[test]
fn deepseek_v3_special_tokens_keep_the_reference_document_meaning() {
    let Some(root) = artifact_root() else {
        eprintln!("skip: no artifact root on this host");
        return;
    };
    let Some((family, directory)) = artifacts(&root)
        .into_iter()
        .find(|(family, _)| family == "deepseek_v3")
    else {
        eprintln!("skip: deepseek_v3 is not under {}", root.display());
        return;
    };
    let runtime = Runtime::builder().device(Device::Cpu).build().unwrap();
    let tokenizer = runtime.load_local(&family, &directory).unwrap();
    let plan = tokenizer.plan();
    assert_eq!(plan.postprocessor_class, "identity");
    assert!(plan.special_token_prefix.is_empty());
    let encoded = tokenizer
        .encode_with("hello world", with_special_tokens())
        .unwrap();
    assert_eq!(encoded.ids(), &[33_310, 2_058]);
    let batch = tokenizer
        .encode_batch_with(&["hello world"], with_special_tokens())
        .unwrap();
    assert_eq!(batch.row(0).unwrap(), &[33_310, 2_058]);
    let mut session = tokenizer.open_session("pin").unwrap();
    assert_eq!(session.seed("hello world").unwrap().ids(), encoded.ids());
}
