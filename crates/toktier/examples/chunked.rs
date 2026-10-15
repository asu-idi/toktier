//! Chunk-parallel encode of one long request on the Rust surface.
//!
//! Loads one family, reports its chunked admission, plans the cuts of a
//! long text, encodes it chunk-parallel and serially, and checks that
//! the two id streams are equal. With `--json` the same facts are
//! printed as one JSON object, which is what the Python cross-face test
//! (`tests/test_chunked_rust.py`) compares with `toktier.chunked`.
//!
//! ```text
//! cargo run --release -p toktier --example chunked -- \
//!     --family qwen3_8b --policy experimental --k 8 --text my-request.txt
//! ```
//!
//! Options: `--family <id>` (default `TOKTIER_FAMILY` or `qwen3_8b`);
//! `--artifacts <dir>` (a directory holding `<family>-<revision>/` or
//! `<family>/` artifact directories, default `TOKTIER_TEST_ARTIFACTS`;
//! without it the family is loaded from the artifact cache);
//! `--policy supported|certified|experimental|reference`; `--k <n>`;
//! `--min-chunk <n>`; `--scan-budget <n>`; `--warm <workers>`;
//! `--text <file>` (default: a generated text of `--chars`, default
//! 1,000,000); `--json`; `--docs <file.jsonl>` (one JSON object per line
//! with a `text` field: every document is encoded chunk-parallel and
//! serially in one process, and the per-document facts are printed as
//! one JSON object, which the cross-face probe reads).
//!
//! Run with `--release`: a development build is not in the shipped
//! runtime-build registry, so under every policy but `experimental` it
//! has no corrected CPU route to chunk on.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sha2::{Digest, Sha256};
use toktier::chunked::{ChunkOptions, ChunkedEncoding};
use toktier::{Device, Policy, ReasonCode, Runtime, Tokenizer};

struct Arguments {
    family: String,
    artifacts: Option<PathBuf>,
    policy: Policy,
    options: ChunkOptions,
    warm: Option<usize>,
    text: Option<PathBuf>,
    docs: Option<PathBuf>,
    chars: usize,
    json: bool,
}

fn parse() -> Result<Arguments, String> {
    let mut arguments = Arguments {
        family: std::env::var("TOKTIER_FAMILY").unwrap_or_else(|_| "qwen3_8b".to_owned()),
        artifacts: std::env::var_os("TOKTIER_TEST_ARTIFACTS").map(PathBuf::from),
        policy: Policy::default(),
        options: ChunkOptions::default(),
        warm: None,
        text: None,
        docs: None,
        chars: 1_000_000,
        json: false,
    };
    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--family" => arguments.family = value()?,
            "--artifacts" => arguments.artifacts = Some(PathBuf::from(value()?)),
            "--policy" => {
                arguments.policy = match value()?.as_str() {
                    "supported" => Policy::Supported,
                    "certified" => Policy::Certified,
                    "experimental" => Policy::Experimental,
                    "reference" => Policy::Reference,
                    other => return Err(format!("unknown policy {other}")),
                }
            }
            "--k" => arguments.options.k = Some(number(&value()?)?),
            "--min-chunk" => arguments.options.min_chunk = Some(number(&value()?)?),
            "--scan-budget" => arguments.options.scan_budget = number(&value()?)?,
            "--warm" => arguments.warm = Some(number(&value()?)?),
            "--text" => arguments.text = Some(PathBuf::from(value()?)),
            "--docs" => arguments.docs = Some(PathBuf::from(value()?)),
            "--chars" => arguments.chars = number(&value()?)?,
            "--json" => arguments.json = true,
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(arguments)
}

fn number(value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{value} is not a non-negative integer"))
}

/// The artifact directory of `family` under `root`: `<family>-<revision>`
/// as the artifact cache names it, or `<family>`.
fn artifact_directory(root: &Path, family: &str) -> Option<PathBuf> {
    let prefix = format!("{family}-");
    let mut named = fs::read_dir(root)
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

fn load(arguments: &Arguments) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let runtime = Runtime::builder()
        .device(Device::Cpu)
        .policy(arguments.policy)
        .build()?;
    match &arguments.artifacts {
        Some(root) => {
            let directory = artifact_directory(root, &arguments.family).ok_or_else(|| {
                format!(
                    "no artifact directory for {} under {}",
                    arguments.family,
                    root.display()
                )
            })?;
            Ok(runtime.load_local(&arguments.family, directory)?)
        }
        None => Ok(runtime.load(&arguments.family)?),
    }
}

/// A deterministic long text: prose with punctuation, numbers, a little
/// CJK, line breaks and a few special-token literals.
fn generated(chars: usize) -> String {
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
    let mut state = 0x5EED_u64;
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

fn sha256_hex(ids: &[u32]) -> String {
    let mut digest = Sha256::new();
    for id in ids {
        digest.update(id.to_le_bytes());
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The `R_*` spelling of a serial reason, the one the Python module
/// reports, so the two faces can be compared on it.
fn reason(encoded: &ChunkedEncoding) -> Option<String> {
    encoded.serial_reason.as_ref().map(|reason| match reason {
        ReasonCode::ChunkNoCut => "R_CHUNK_NO_CUT".to_owned(),
        ReasonCode::ChunkFamilyUnsupported => "R_CHUNK_FAMILY_UNSUPPORTED".to_owned(),
        ReasonCode::ChunkRouteNotFastCpu => "R_CHUNK_ROUTE_NOT_FAST_CPU".to_owned(),
        ReasonCode::Other(code) => code.clone(),
        other => format!("{other:?}"),
    })
}

/// `--docs`: every document of a JSONL file, chunk-parallel and serial,
/// in one process.
fn run_documents(
    arguments: &Arguments,
    tokenizer: &Tokenizer,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let chunked = tokenizer.chunked();
    let admission = chunked.admission()?;
    if let Some(workers) = arguments.warm {
        if admission.cpu_route {
            chunked.warm(Some(workers), None)?;
        }
    }
    let mut documents = Vec::new();
    let mut unequal = 0usize;
    for (index, line) in fs::read_to_string(path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(line)?;
        let text = record["text"]
            .as_str()
            .ok_or_else(|| format!("line {index}: no text field"))?;
        let started = Instant::now();
        let encoded = chunked.encode(text, &arguments.options)?;
        let chunked_us = started.elapsed().as_micros() as u64;
        let started = Instant::now();
        let serial = tokenizer.encode(text)?;
        let serial_us = started.elapsed().as_micros() as u64;
        let equal = encoded.ids() == serial.ids();
        unequal += usize::from(!equal);
        documents.push(serde_json::json!({
            "index": index,
            "n": record.get("n").cloned().unwrap_or(serde_json::Value::Null),
            "chars": text.chars().count(),
            "bytes": text.len(),
            "chunks": encoded.chunks,
            "k_requested": encoded.k_requested,
            "cuts": encoded.cuts,
            "cut_misses": encoded.cut_misses,
            "literal_rejects": encoded.literal_rejects,
            "norm_rejects": encoded.norm_rejects,
            "serial_reason": reason(&encoded),
            "below_threshold": encoded.below_threshold,
            "sources": {
                "gigatoken": encoded.sources.gigatoken,
                "hf_added_token": encoded.sources.hf_added_token,
                "hf_engine_guard": encoded.sources.hf_engine_guard,
            },
            "ids_len": encoded.ids().len(),
            "ids_sha256": sha256_hex(encoded.ids()),
            "serial_ids_sha256": sha256_hex(serial.ids()),
            "equal_to_serial": equal,
            "chunked_us": chunked_us,
            "serial_us": serial_us,
        }));
    }
    let stats = chunked.stats();
    let report = serde_json::json!({
        "family": tokenizer.family(),
        "policy": format!("{:?}", arguments.policy),
        "route": format!("{:?}", tokenizer.plan().backends),
        "certification": format!("{:?}", tokenizer.plan().certification),
        "admission": {
            "admitted": admission.admitted,
            "refusal": admission.refusal,
            "cpu_route": admission.cpu_route,
            "gate": format!("{:?}", admission.gate),
        },
        "options": {
            "k": arguments.options.k,
            "min_chunk": arguments.options.min_chunk,
            "scan_budget": arguments.options.scan_budget,
        },
        "documents": documents,
        "unequal_to_serial": unequal,
        "stats": {
            "calls": stats.calls,
            "applied": stats.applied,
            "serial": stats.serial,
            "below_threshold": stats.below_threshold,
            "chunks_total": stats.chunks_total,
            "fallback_counts": stats.fallback_counts,
            "workers": stats.workers,
            "default_workers": stats.default_workers,
        },
    });
    println!("{}", serde_json::to_string(&report)?);
    if unequal > 0 {
        return Err(format!("{unequal} documents differ from the serial encode").into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = parse()?;
    let tokenizer = load(&arguments)?;
    if let Some(docs) = &arguments.docs {
        return run_documents(&arguments, &tokenizer, docs);
    }
    let text = match &arguments.text {
        Some(path) => fs::read_to_string(path)?,
        None => generated(arguments.chars),
    };
    let chunked = tokenizer.chunked();
    let admission = chunked.admission()?;
    if let Some(workers) = arguments.warm {
        if admission.cpu_route {
            chunked.warm(Some(workers), Some(&text[..text.len().min(20_000)]))?;
        }
    }
    let plan = chunked.plan(&text, &arguments.options);
    let started = Instant::now();
    let encoded = chunked.encode(&text, &arguments.options)?;
    let chunked_elapsed = started.elapsed();
    let started = Instant::now();
    let serial = tokenizer.encode(&text)?;
    let serial_elapsed = started.elapsed();
    let equal = encoded.ids() == serial.ids();
    let stats = chunked.stats();

    if arguments.json {
        let plan_value = match &plan {
            Ok(plan) => serde_json::json!({
                "spans": plan.spans,
                "byte_spans": plan.byte_spans,
                "cuts": plan.cuts(),
                "k": plan.k,
                "chars": plan.chars,
                "cut_misses": plan.cut_misses,
                "literal_rejects": plan.literal_rejects,
                "norm_rejects": plan.norm_rejects,
            }),
            Err(_) => serde_json::Value::Null,
        };
        let report = serde_json::json!({
            "family": tokenizer.family(),
            "policy": format!("{:?}", arguments.policy),
            "route": format!("{:?}", tokenizer.plan().backends),
            "certification": format!("{:?}", tokenizer.plan().certification),
            "admission": {
                "admitted": admission.admitted,
                "refusal": admission.refusal,
                "family": admission.family,
                "artifact_sha256": admission.artifact_sha256,
                "cpu_route": admission.cpu_route,
                "normalizer": admission.normalizer,
                "literals": admission.literals,
                "evidence_id": admission.evidence_id,
                "evidence_status": admission.evidence_status,
                "native_evidence": admission.native_evidence,
                "gate": format!("{:?}", admission.gate),
            },
            "plan": plan_value,
            "plan_error": plan.as_ref().err().map(|error| error.message().to_owned()),
            "encode": {
                "chunks": encoded.chunks,
                "k_requested": encoded.k_requested,
                "cuts": encoded.cuts,
                "cut_misses": encoded.cut_misses,
                "literal_rejects": encoded.literal_rejects,
                "norm_rejects": encoded.norm_rejects,
                "serial_reason": reason(&encoded),
                "below_threshold": encoded.below_threshold,
                "sources": {
                    "gigatoken": encoded.sources.gigatoken,
                    "hf_added_token": encoded.sources.hf_added_token,
                    "hf_engine_guard": encoded.sources.hf_engine_guard,
                },
                "certification": format!("{:?}", encoded.certification),
                "ids_len": encoded.ids().len(),
                "ids_sha256": sha256_hex(encoded.ids()),
                "equal_to_serial": equal,
                "chunked_us": chunked_elapsed.as_micros() as u64,
                "serial_us": serial_elapsed.as_micros() as u64,
            },
            "stats": {
                "calls": stats.calls,
                "applied": stats.applied,
                "serial": stats.serial,
                "below_threshold": stats.below_threshold,
                "fallback_counts": stats.fallback_counts,
                "workers": stats.workers,
                "default_workers": stats.default_workers,
            },
            "text_chars": text.chars().count(),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "family={} policy={:?} route={:?} admitted={} refusal={:?} cpu_route={} \
             normalizer={:?} literals={} evidence_status={:?} gate={:?}",
            tokenizer.family(),
            arguments.policy,
            tokenizer.plan().backends,
            admission.admitted,
            admission.refusal,
            admission.cpu_route,
            admission.normalizer,
            admission.literals,
            admission.evidence_status,
            admission.gate,
        );
        match &plan {
            Ok(plan) => println!(
                "plan: k={} chars={} chunks={} cuts={:?} cut_misses={} literal_rejects={} norm_rejects={}",
                plan.k,
                plan.chars,
                plan.spans.len(),
                plan.cuts(),
                plan.cut_misses,
                plan.literal_rejects,
                plan.norm_rejects
            ),
            Err(error) => println!("plan: refused ({:?}): {}", error.code(), error.message()),
        }
        println!(
            "encode: chunks={} k_requested={} serial_reason={:?} below_threshold={} \
             sources={:?} certification={:?} chunked_us={} serial_us={} ids={} equal_to_serial={}",
            encoded.chunks,
            encoded.k_requested,
            reason(&encoded),
            encoded.below_threshold,
            encoded.sources,
            encoded.certification,
            chunked_elapsed.as_micros(),
            serial_elapsed.as_micros(),
            encoded.ids().len(),
            equal,
        );
        println!(
            "stats: calls={} applied={} serial={} below_threshold={} fallback_counts={:?} workers={} default_workers={}",
            stats.calls,
            stats.applied,
            stats.serial,
            stats.below_threshold,
            stats.fallback_counts,
            stats.workers,
            stats.default_workers
        );
    }
    if !equal {
        return Err("chunked ids differ from the serial encode".into());
    }
    Ok(())
}
