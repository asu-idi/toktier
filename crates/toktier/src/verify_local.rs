//! A local check of an accelerated route against this binary's reference
//! engine, and the record it leaves.
//!
//! Since 0.2.6 a device or compiler toolchain no campaign has judged runs
//! by default, labelled `supported_untested` (`docs/rust-api.md`). This
//! module is the other half of that: a way for the person running it to
//! measure the combination on their own machine, on their own text, and
//! to have the answer remembered until something it depended on changes.
//!
//! What it is not: a certificate. The record says who compared what, on
//! which device, and over how many documents; it never says `certified`,
//! it is written by whoever ran the command, and it expires when the
//! driver, the toolchain, the kernel, the source identity or the family
//! artifact changes. Nothing runs it automatically -- a first-run check
//! would be a default behaviour, and this release deliberately has none.
//!
//! The inputs are the caller's. `--synthetic` generates documents from
//! rules rather than from a corpus: no text ships with this crate, and
//! none is downloaded, so a check costs no license question and no
//! network.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::fsutil::ensure_private_directory;
use crate::manifest::{domain_sha256_hex, sha256_hex};
use crate::{Error, ErrorCode, Result};

/// The record format. A record of another schema is not read.
const RECORD_SCHEMA: &str = "toktier.local_verification.v1";

/// The version of the comparison itself. A change to what the check does
/// -- which cases it generates, what it compares -- invalidates records
/// taken under the older one, because they answer a different question.
const TOOL_VERSION: &str = "1";

const KEY_DOMAIN: &[u8] = b"toktier.local_verification_key.v1\0";

/// Everything a record is about. Any of it changing makes the record
/// describe another combination, so the record is filed under all of it
/// and read back only when every field still matches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct VerificationKey {
    /// `cpu` or `gpu`: which accelerated route was compared.
    pub engine: String,
    /// The device architecture, for a GPU route.
    pub architecture: Option<String>,
    /// `prebuilt` or `jit`, for a GPU route.
    pub delivery: Option<String>,
    /// The digest of the image that ran: the shipped fatbin, or the JIT
    /// product.
    pub image_digest: Option<String>,
    /// The compiler triple a JIT product was built with.
    pub toolchain: Option<String>,
    /// The CUDA driver API version the device was opened with.
    pub driver_api_version: Option<i32>,
    /// The two source identities that decide what code ran.
    pub native_host_source_digest: String,
    pub rust_api_source_digest: String,
    /// The family and the exact artifact bytes that were tokenized.
    pub family: String,
    pub artifact_sha256: String,
    /// The version of the check itself.
    pub tool_version: String,
}

impl VerificationKey {
    /// The file this key is filed under.
    fn digest(&self) -> String {
        let rendered = serde_json::to_vec(self).unwrap_or_default();
        domain_sha256_hex(KEY_DOMAIN, &rendered)
    }
}

/// What one local check found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct VerificationRecord {
    pub schema: String,
    pub key: VerificationKey,
    /// `passed` or `failed`. A failed record is kept: a reader is better
    /// served by "this was measured and it did not agree" than by
    /// silence, and the route is labelled as if no check had run.
    pub status: String,
    pub documents: u64,
    pub bytes: u64,
    pub mismatches: u64,
    /// The first document that disagreed, and the token index where it
    /// first did.
    pub first_mismatch: Option<(u64, u64)>,
    /// `your text` or `generated`, so a reader knows what was compared.
    pub input: String,
    /// A digest of the documents themselves, so two runs over different
    /// text are told apart without the text being stored.
    pub input_digest: String,
    /// Seconds since the Unix epoch, for the sentence `doctor` prints.
    pub taken_at: u64,
}

/// Where records live: beside the other caches this crate keeps, in a
/// directory of their own, owner-only like the rest.
pub(crate) fn record_root() -> PathBuf {
    crate::fsutil::default_cache_directory("TOKTIER_VERIFY_CACHE", "device-verify")
}

fn record_path(root: &Path, key: &VerificationKey) -> PathBuf {
    root.join(format!("{}.json", key.digest()))
}

/// The record for one combination, when one is on disk and still
/// describes it.
///
/// "Still describes it" is the whole of the expiry rule: the key is the
/// file name, and a driver, toolchain, kernel, source identity or
/// artifact that moved is a different key. Nothing has to be swept.
pub(crate) fn read_record_in(root: &Path, key: &VerificationKey) -> Option<VerificationRecord> {
    let path = record_path(root, key);
    let bytes = fs::read(&path).ok()?;
    let record: VerificationRecord = serde_json::from_slice(&bytes).ok()?;
    (record.schema == RECORD_SCHEMA && &record.key == key).then_some(record)
}

/// Whether this combination carries a check that passed.
///
/// A check that ran and disagreed leaves a record too, and this answers
/// `false` for it: the route keeps the label it would have had without
/// any check, and the report says what was found. Running a check never
/// makes a combination more restricted than not running one.
pub(crate) fn is_locally_verified_in(root: &Path, key: &VerificationKey) -> bool {
    read_record_in(root, key).is_some_and(|record| record.status == "passed")
}

pub(crate) fn is_locally_verified(key: &VerificationKey) -> bool {
    is_locally_verified_in(&record_root(), key)
}

/// Write one record, replacing whatever this combination had before.
pub(crate) fn write_record_in(root: &Path, record: &VerificationRecord) -> Result<PathBuf> {
    ensure_private_directory(
        root,
        "local verification record directory",
        "the local verification record directory is not a directory",
    )?;
    let path = record_path(root, &record.key);
    let mut bytes = serde_json::to_vec_pretty(record)
        .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?;
    bytes.push(b'\n');
    crate::fsutil::write_private_file(&path, &bytes)?;
    Ok(path)
}

pub(crate) fn write_record(record: &VerificationRecord) -> Result<PathBuf> {
    write_record_in(&record_root(), record)
}

/// Forget one combination's record.
pub(crate) fn forget_record_in(root: &Path, key: &VerificationKey) -> Result<bool> {
    let path = record_path(root, key);
    match fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(Error::new(ErrorCode::Io, error.to_string()).with_path(&path)),
    }
}

pub(crate) fn forget_record(key: &VerificationKey) -> Result<bool> {
    forget_record_in(&record_root(), key)
}

/// Seconds since the Unix epoch, or zero when the clock is before it.
pub(crate) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------
// The generated documents.
// ---------------------------------------------------------------------

/// A deterministic stream of bits from a seed, so the same command
/// generates the same documents on any machine.
struct Sequence(u64);

impl Sequence {
    fn new(seed: u64) -> Self {
        // Any odd constant works; this is splitmix64's.
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn pick<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        &choices[(self.next() % choices.len() as u64) as usize]
    }
}

/// The fragments the generator assembles documents from.
///
/// Every one of them is written here rather than taken from a corpus, and
/// each is a shape a fallback or a divergence has actually been found on:
/// variation selectors after an emoji or a Han character; combining marks
/// in non-canonical order; seams between scripts; newline and punctuation
/// seams; the whitespace variants a regex dialect disagrees about; long
/// repeats that cross a chunk boundary; and the three code points
/// FINDING 044 measured, which separate a Unicode 16 table from a
/// Unicode 17 one.
const FRAGMENTS: &[&str] = &[
    "the quick brown fox jumps over the lazy dog",
    "Cargo builds, tests, and packages 148 crates.",
    "{\"key\": [1, 2.5, true, null], \"nested\": {\"a\": \"b\"}}",
    "fn main() {\n\tlet total = 0usize;\r\n\tprintln!(\"{total}\");\n}",
    "0123456789 3.14159 -42 1e-9 0x2A",
    "-- ... --- ,,, ;;; ??? !!! (((())))",
    "\u{1f600}\u{fe0f}\u{1f601}\u{fe0e} emoji with selectors",
    "\u{4e2d}\u{6587}\u{fe0f}\u{6df7}\u{6392}mixed with ASCII",
    "e\u{301}a\u{34d}\u{8cb} combining marks out of order",
    "spaces\u{00a0}and\u{3000}separators\u{180e}between words",
    "\u{10940}\u{10941} sidetic letters after 16.0",
    "x\u{323b0}\u{323b1}y extension J han",
    "\u{295}Bear \u{294}Bear pharyngeal letters",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "\n\n   \t\t  \r\n \r\n",
    "https://example.invalid/path?query=1&other=2#fragment",
];

/// `count` documents of at most `max_bytes` each, from `seed`.
///
/// Rules only: no text of anyone else's travels with this crate, and
/// nothing is fetched.
pub(crate) fn generate(count: u64, max_bytes: usize, seed: u64) -> Vec<String> {
    let mut sequence = Sequence::new(seed);
    let mut documents = Vec::with_capacity(count as usize);
    for index in 0..count {
        let mut document = String::new();
        // Each document opens on a different fragment, so a small run
        // still reaches every shape above.
        let opening = (index as usize) % FRAGMENTS.len();
        document.push_str(FRAGMENTS[opening]);
        while document.len() < max_bytes {
            let separator = sequence.pick(&[" ", "\n", "", "\t", ", ", "\r\n"]);
            let fragment = sequence.pick(FRAGMENTS);
            if document.len() + separator.len() + fragment.len() > max_bytes {
                break;
            }
            document.push_str(separator);
            document.push_str(fragment);
        }
        documents.push(document);
    }
    documents
}

/// The documents a caller supplied: one per line of a file, or the whole
/// of standard input as one document when it holds no newline.
pub(crate) fn read_input(path: &Path) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    Ok(split_documents(&text))
}

pub(crate) fn split_documents(text: &str) -> Vec<String> {
    let lines = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if lines.is_empty() && !text.is_empty() {
        return vec![text.to_owned()];
    }
    lines
}

/// The digest of the input a record was taken over, so a reader can tell
/// two runs apart without the text being stored.
pub(crate) fn input_digest(documents: &[String]) -> String {
    let mut joined = Vec::new();
    for document in documents {
        joined.extend_from_slice(&(document.len() as u64).to_le_bytes());
        joined.extend_from_slice(document.as_bytes());
    }
    sha256_hex(&joined)
}

/// The key one built GPU route is filed under.
#[cfg(feature = "prebuilt-gpu")]
pub(crate) fn gpu_key(
    built: &crate::gpu_data::BuiltGpu,
    family: &str,
    artifact_sha256: &str,
) -> VerificationKey {
    VerificationKey {
        engine: "gpu".to_owned(),
        architecture: Some(built.architecture.clone()),
        delivery: Some(built.delivery.to_owned()),
        image_digest: Some(built.image_digest.clone()),
        // A JIT product is filed under the compiler that built it as
        // well as under its source digest: the same source through
        // another compiler is another product.
        toolchain: built.toolchain.clone(),
        driver_api_version: Some(built.driver_api_version),
        native_host_source_digest: env!("TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256").to_owned(),
        rust_api_source_digest: env!("TOKTIER_RUST_API_SOURCE_SHA256").to_owned(),
        family: family.to_owned(),
        artifact_sha256: artifact_sha256.to_owned(),
        tool_version: TOOL_VERSION.to_owned(),
    }
}

/// The key the CPU fast path is filed under on this machine.
pub(crate) fn cpu_key(family: &str, artifact_sha256: &str) -> VerificationKey {
    VerificationKey {
        engine: "cpu".to_owned(),
        architecture: None,
        delivery: None,
        image_digest: None,
        toolchain: None,
        driver_api_version: None,
        native_host_source_digest: env!("TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256").to_owned(),
        rust_api_source_digest: env!("TOKTIER_RUST_API_SOURCE_SHA256").to_owned(),
        family: family.to_owned(),
        artifact_sha256: artifact_sha256.to_owned(),
        tool_version: TOOL_VERSION.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        generate, input_digest, split_documents, uncovered_note, VerificationKey, TOOL_VERSION,
    };

    /// A run that served part of its documents measured that part, and
    /// the sentence says so rather than calling it nothing.
    #[test]
    fn a_partly_served_run_says_what_it_did_compare() {
        let none = uncovered_note("cpu", 0, 3, &[("R_INPUT_ADDED_TOKEN".to_owned(), 3)]);
        assert!(
            none.contains("was admitted and served none of the 3 documents")
                && none.contains("per-input reason (R_INPUT_ADDED_TOKEN x3)")
                && none.contains("measured nothing about it")
                && none.contains("`ExecutionFacts::reason`")
                && none.contains("`Tokenizer::plan().reasons`"),
            "{none}"
        );
        // The plan admitted the route, or the run would have been
        // answered before any document was encoded; what `doctor` says
        // about the plan is not what kept these documents off it.
        assert!(!none.contains("says why the route did not run"), "{none}");
        let unrecorded = uncovered_note("cpu", 0, 3, &[]);
        assert!(
            unrecorded.contains("(no reason was recorded)"),
            "{unrecorded}"
        );

        let some = uncovered_note("cpu", 2, 3, &[("R_INPUT_ADDED_TOKEN".to_owned(), 1)]);
        assert!(
            some.contains("served 2 of 3 documents")
                && some.contains("the served ones compared equal"),
            "{some}"
        );
        // The partial state is neither a pass nor "nothing measured",
        // and `doctor` answers about the plan rather than about the
        // documents that took the reference path.
        for word in ["measured nothing", "doctor", "locally_verified"] {
            assert!(!some.contains(word), "{some}");
        }
        assert!(some.contains("no record was written"), "{some}");
    }

    /// This crate has no `explain()`, so no message may send a reader
    /// to one: a consumer who follows that sentence gets a compile
    /// error, not an answer.
    #[test]
    fn no_note_points_at_a_python_only_surface() {
        for note in [
            uncovered_note("cpu", 0, 3, &[("R_INPUT_ADDED_TOKEN".to_owned(), 3)]),
            uncovered_note("cpu", 0, 3, &[]),
            uncovered_note("gpu", 2, 3, &[("R_EXEC_FAULT".to_owned(), 1)]),
        ] {
            assert!(!note.contains("explain()"), "{note}");
        }
    }

    /// A reader at a prompt gets wrapped lines that end in a full stop,
    /// not one physical line of several hundred characters.
    #[test]
    fn the_uncovered_notes_are_wrapped_and_punctuated() {
        for note in [
            uncovered_note("cpu", 0, 3, &[("R_INPUT_ADDED_TOKEN".to_owned(), 3)]),
            uncovered_note("gpu", 2, 3, &[]),
        ] {
            assert!(note.ends_with('.'), "{note}");
            for line in note.lines() {
                assert!(line.chars().count() <= 100, "{line}");
            }
        }
    }

    fn key() -> VerificationKey {
        VerificationKey {
            engine: "gpu".to_owned(),
            architecture: Some("sm_100".to_owned()),
            delivery: Some("prebuilt".to_owned()),
            image_digest: Some("aaaa".to_owned()),
            toolchain: None,
            driver_api_version: Some(13_000),
            native_host_source_digest: "bbbb".to_owned(),
            rust_api_source_digest: "cccc".to_owned(),
            family: "qwen3_8b".to_owned(),
            artifact_sha256: "dddd".to_owned(),
            tool_version: TOOL_VERSION.to_owned(),
        }
    }

    /// Every field of the key is part of it: a record is about one
    /// combination, and a changed driver, kernel, source identity or
    /// artifact makes it a different one.
    #[test]
    fn every_field_of_the_key_changes_where_the_record_is_filed() {
        let base = key().digest();
        type Mutation = Box<dyn Fn(&mut VerificationKey)>;
        let mutations: Vec<Mutation> = vec![
            Box::new(|key| key.engine = "cpu".to_owned()),
            Box::new(|key| key.architecture = Some("sm_90".to_owned())),
            Box::new(|key| key.delivery = Some("jit".to_owned())),
            Box::new(|key| key.image_digest = Some("ffff".to_owned())),
            Box::new(|key| key.toolchain = Some("13.0".to_owned())),
            Box::new(|key| key.driver_api_version = Some(13_010)),
            Box::new(|key| key.native_host_source_digest = "0000".to_owned()),
            Box::new(|key| key.rust_api_source_digest = "0000".to_owned()),
            Box::new(|key| key.family = "kimi_k3".to_owned()),
            Box::new(|key| key.artifact_sha256 = "0000".to_owned()),
            Box::new(|key| key.tool_version = "0".to_owned()),
        ];
        for mutate in mutations {
            let mut moved = key();
            mutate(&mut moved);
            assert_ne!(moved.digest(), base);
        }
    }

    /// The generator is a function of its seed and nothing else.
    #[test]
    fn the_same_seed_generates_the_same_documents() {
        assert_eq!(generate(16, 512, 7), generate(16, 512, 7));
        assert_ne!(generate(16, 512, 7), generate(16, 512, 8));
    }

    /// Every shape the generator is for appears in a small run, the three
    /// code points FINDING 044 measured among them.
    #[test]
    fn a_small_run_reaches_every_shape_the_generator_is_for() {
        let documents = generate(32, 1024, 1);
        let all = documents.concat();
        for sentinel in ['\u{10940}', '\u{323b0}', '\u{295}'] {
            assert!(
                all.contains(sentinel),
                "no {sentinel:?} in the generated run"
            );
        }
        for shape in [
            "\u{fe0f}", "\u{034d}", "\u{00a0}", "\u{3000}", "\u{180e}", "\r\n",
        ] {
            assert!(all.contains(shape), "no {shape:?} in the generated run");
        }
        assert!(documents.iter().all(|document| document.len() <= 1024));
    }

    fn record(status: &str) -> super::VerificationRecord {
        super::VerificationRecord {
            schema: super::RECORD_SCHEMA.to_owned(),
            key: key(),
            status: status.to_owned(),
            documents: 2_000,
            bytes: 4_000_000,
            mismatches: if status == "passed" { 0 } else { 3 },
            first_mismatch: (status != "passed").then_some((417, 88)),
            input: "your text".to_owned(),
            input_digest: "eeee".to_owned(),
            taken_at: 1_760_000_000,
        }
    }

    /// A record is written, read back for the combination it is about,
    /// and forgotten on request.
    #[test]
    fn a_record_is_written_read_and_forgotten() {
        let root = tempfile::tempdir().unwrap();
        assert!(!super::is_locally_verified_in(root.path(), &key()));
        super::write_record_in(root.path(), &record("passed")).unwrap();
        assert!(super::is_locally_verified_in(root.path(), &key()));
        assert_eq!(
            super::read_record_in(root.path(), &key())
                .unwrap()
                .documents,
            2_000
        );
        assert!(super::forget_record_in(root.path(), &key()).unwrap());
        assert!(!super::is_locally_verified_in(root.path(), &key()));
        assert!(!super::forget_record_in(root.path(), &key()).unwrap());
    }

    /// A check that ran and disagreed leaves the route where it was.
    /// Running the check must never make a combination more restricted
    /// than never running it, or a careful operator is punished for
    /// looking.
    #[test]
    fn a_failed_check_does_not_verify_and_does_not_restrict() {
        let root = tempfile::tempdir().unwrap();
        super::write_record_in(root.path(), &record("failed")).unwrap();
        assert!(!super::is_locally_verified_in(root.path(), &key()));
        let stored = super::read_record_in(root.path(), &key()).unwrap();
        assert_eq!(stored.mismatches, 3);
        assert_eq!(stored.first_mismatch, Some((417, 88)));
    }

    /// A record for another combination is not this one's. The key is
    /// the file name, so an expired record is one nothing looks for.
    #[test]
    fn a_record_for_another_combination_is_not_read() {
        let root = tempfile::tempdir().unwrap();
        super::write_record_in(root.path(), &record("passed")).unwrap();
        let mut moved = key();
        moved.driver_api_version = Some(13_010);
        assert!(!super::is_locally_verified_in(root.path(), &moved));
        let mut rebuilt = key();
        rebuilt.image_digest = Some("ffff".to_owned());
        assert!(!super::is_locally_verified_in(root.path(), &rebuilt));
    }

    /// Documents are lines, and a file with no newline is one document.
    #[test]
    fn input_is_split_into_documents_by_line() {
        assert_eq!(split_documents("a\nb\n\nc\n"), vec!["a", "b", "c"]);
        assert_eq!(split_documents("only one"), vec!["only one"]);
        assert!(split_documents("").is_empty());
        assert_ne!(
            input_digest(&split_documents("a\nb")),
            input_digest(&split_documents("ab"))
        );
    }
}

// ---------------------------------------------------------------------
// Running the check.
// ---------------------------------------------------------------------

/// Which accelerated route to compare against the reference engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Engine {
    Cpu,
    Gpu,
}

impl Engine {
    fn word(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// What one run was asked to do.
pub(crate) struct Request {
    pub engines: Vec<Engine>,
    pub families: Vec<String>,
    pub documents: Vec<String>,
    pub input_label: String,
    pub delivery: crate::GpuDelivery,
    pub device: u32,
    pub forget: bool,
}

/// What one comparison found, for one engine and one family.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Comparison {
    pub engine: String,
    pub family: String,
    /// The route the accelerated runtime admitted, as its label.
    pub route: String,
    /// Documents compared, and how many of them the accelerated backend
    /// actually ran. A document a per-input guard sent to the reference
    /// engine is not a disagreement: it is the fallback working, and it
    /// is counted rather than hidden.
    pub documents: u64,
    pub accelerated_documents: u64,
    pub bytes: u64,
    pub mismatches: u64,
    pub first_mismatch: Option<(u64, u64)>,
    /// Whether a record was written, and where.
    pub record: Option<String>,
    /// Why no record was written, when none was.
    pub note: Option<String>,
}

/// The whole command, from its arguments to its report.
///
/// Kept in the library rather than in the binary so that the rules it
/// applies -- what counts as a comparison, what a missing route means,
/// what is written down -- are tested with the rest of the crate.
pub fn verify_local_command(arguments: &[String]) -> Result<String> {
    let engines = match flag(arguments, "--engine").as_deref() {
        None | Some("both") => vec![Engine::Cpu, Engine::Gpu],
        Some("cpu") => vec![Engine::Cpu],
        Some("gpu") => vec![Engine::Gpu],
        Some(other) => {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                format!("--engine takes cpu, gpu or both, not {other:?}"),
            ))
        }
    };
    let families = match flag(arguments, "--family") {
        Some(family) => vec![family],
        None => {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "name the family to compare with --family",
            ))
        }
    };
    let synthetic = match flag(arguments, "--synthetic") {
        Some(value) => value
            .parse::<u64>()
            .map_err(|_| Error::new(ErrorCode::InvalidArgument, "--synthetic takes a count"))?,
        None => 0,
    };
    let max_bytes = match flag(arguments, "--max-bytes") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| Error::new(ErrorCode::InvalidArgument, "--max-bytes takes a size"))?,
        None => 4096,
    };
    let seed = match flag(arguments, "--seed") {
        Some(value) => value
            .parse::<u64>()
            .map_err(|_| Error::new(ErrorCode::InvalidArgument, "--seed takes a number"))?,
        None => 1,
    };
    let forget = arguments.iter().any(|argument| argument == "--forget");
    let (documents, input_label) = match flag(arguments, "--input") {
        Some(path) if path == "-" => {
            let mut text = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)
                .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
            (split_documents(&text), "your text".to_owned())
        }
        Some(path) => (read_input(Path::new(&path))?, "your text".to_owned()),
        None if synthetic > 0 => (generate(synthetic, max_bytes, seed), "generated".to_owned()),
        None if forget => (Vec::new(), "none".to_owned()),
        None => {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "give the text to compare with --input PATH (or - for standard input), or ask \
                 for generated documents with --synthetic N",
            ))
        }
    };
    if documents.is_empty() && !forget {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "no documents to compare",
        ));
    }
    let delivery = match flag(arguments, "--delivery").as_deref() {
        None | Some("auto") => crate::GpuDelivery::Auto,
        Some("prebuilt") => crate::GpuDelivery::Prebuilt,
        Some("jit") => crate::GpuDelivery::Jit,
        Some(other) => {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                format!("--delivery takes prebuilt, jit or auto, not {other:?}"),
            ))
        }
    };
    let device = match flag(arguments, "--device") {
        Some(value) => value
            .parse::<u32>()
            .map_err(|_| Error::new(ErrorCode::InvalidArgument, "--device takes an ordinal"))?,
        None => 0,
    };
    let request = Request {
        engines,
        families,
        documents,
        input_label,
        delivery,
        device,
        forget,
    };
    let comparisons = run(&request)?;
    if arguments.iter().any(|argument| argument == "--json") {
        return serde_json::to_string_pretty(&comparisons)
            .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()));
    }
    Ok(render(&comparisons, &request))
}

fn flag(arguments: &[String], name: &str) -> Option<String> {
    let position = arguments.iter().position(|argument| argument == name)?;
    arguments.get(position + 1).cloned()
}

/// The report a person reads. It says what was compared, against what,
/// and what the answer does and does not mean.
fn render(comparisons: &[Comparison], request: &Request) -> String {
    let mut lines = Vec::new();
    for comparison in comparisons {
        if let Some(note) = &comparison.note {
            lines.push(format!(
                "{} {}: {note}",
                comparison.engine, comparison.family
            ));
            continue;
        }
        if comparison.mismatches == 0 {
            lines.push(format!(
                "{} {}: {} documents, {} bytes, no disagreement with the reference engine in \
                 this binary ({} ran on the {} route, {} took the reference route on a guard). \
                 Recorded as locally verified on this device; this is a local check by you, not \
                 a certificate, and it expires when the driver, toolchain, kernel, source \
                 identity or family artifact changes",
                comparison.engine,
                comparison.family,
                comparison.documents,
                comparison.bytes,
                comparison.accelerated_documents,
                comparison.engine,
                comparison.documents - comparison.accelerated_documents,
            ));
        } else {
            let (document, token) = comparison.first_mismatch.unwrap_or((0, 0));
            lines.push(format!(
                "{} {}: local verification failed on {} of {} documents (first: document {} at \
                 token {}). The {} route on this device does not match the reference engine for \
                 those inputs; select Policy::Certified to keep this combination on the \
                 reference route. Nothing was changed automatically",
                comparison.engine,
                comparison.family,
                comparison.mismatches,
                comparison.documents,
                document,
                token,
                comparison.engine,
            ));
        }
    }
    if !request.forget {
        lines.push(format!(
            "input: {} ({} documents)",
            request.input_label,
            request.documents.len()
        ));
    }
    lines.join("\n")
}

/// Compare an accelerated route with this binary's reference engine, on
/// the caller's documents, and record the answer.
///
/// Nothing here changes a default. A comparison that fails writes a
/// failed record and says which documents disagreed; selecting
/// `Policy::Certified` is what keeps such a combination on the reference
/// route, and that is the caller's decision to make.
pub(crate) fn run(request: &Request) -> Result<Vec<Comparison>> {
    let mut comparisons = Vec::new();
    let asked_for_both = request.engines.len() > 1;
    for engine in &request.engines {
        for family in &request.families {
            match compare_one(*engine, family, request) {
                Ok(comparison) => comparisons.push(comparison),
                // `--engine both` on a machine with no usable device is
                // the ordinary case rather than a failure of the
                // command: the engine that could not be opened is named
                // and the other one still runs. A single explicit
                // `--engine` has nothing to fall back to, so its error
                // travels to the caller unchanged.
                Err(error) if asked_for_both => comparisons.push(unavailable(
                    *engine,
                    family,
                    format!(
                        "this machine could not open the {} route: {error}",
                        engine.word()
                    ),
                )),
                Err(error) => return Err(error),
            }
        }
    }
    Ok(comparisons)
}

/// A comparison that did not happen, and why.
fn unavailable(engine: Engine, family: &str, note: String) -> Comparison {
    Comparison {
        engine: engine.word().to_owned(),
        family: family.to_owned(),
        route: "not opened".to_owned(),
        documents: 0,
        accelerated_documents: 0,
        bytes: 0,
        mismatches: 0,
        first_mismatch: None,
        record: None,
        note: Some(note),
    }
}

/// The ledger's own token for a routing reason, so a note names the
/// code the router recorded rather than a second reading of the path.
/// Codes this release has no frozen name for pass through as the router
/// wrote them, and the plan-time variants render as this crate's own
/// names for them.
fn reason_code(reason: &crate::ReasonCode) -> String {
    use crate::ReasonCode;
    match reason {
        ReasonCode::InputBelowGpuThreshold => {
            toktier_routing_core::R_INPUT_BELOW_GPU_THRESHOLD.to_owned()
        }
        ReasonCode::InputAddedToken => toktier_routing_core::R_INPUT_ADDED_TOKEN.to_owned(),
        ReasonCode::InputGuardRouted => toktier_routing_core::R_INPUT_GUARD_ROUTED.to_owned(),
        ReasonCode::ExecutionFault => toktier_routing_core::R_EXEC_FAULT.to_owned(),
        ReasonCode::InputPostprocessRouted => {
            toktier_routing_core::R_INPUT_POSTPROCESS_ROUTED.to_owned()
        }
        ReasonCode::Other(code) => code.clone(),
        other => format!("{other:?}"),
    }
}

/// What to say when the route did not serve every document.
///
/// Two states share this branch and a reader is owed the difference. A
/// route that served nothing compared the reference engine with itself,
/// so the run says nothing about it at all. A route that served some of
/// them did measure those, and they agreed: what the run lacks is
/// coverage, not a result, and calling that "measured nothing" would be
/// less than the truth. Neither writes a record, and neither is sent to
/// `doctor`: the plan admitted the route (a plan that did not is
/// answered before any document is encoded, and that answer carries the
/// plan's own reasons), so what kept every document off it is a per-input
/// reason. `reasons` carries the codes the ledger recorded for the
/// documents the route did not serve, so the sentence can name them,
/// and it points at the Rust surfaces that carry the same answer:
/// `ExecutionFacts::reason` per encode, `Tokenizer::plan().reasons` for
/// the plan. This crate has no `explain()`, so it does not send anyone
/// there.
///
/// The sentence is wrapped at a terminal width and ends in a full stop:
/// it is read by a person at a prompt, and one 300-character physical
/// line is not.
fn uncovered_note(engine: &str, served: u64, documents: u64, reasons: &[(String, u64)]) -> String {
    if served == 0 {
        let recorded = if reasons.is_empty() {
            "no reason was recorded".to_owned()
        } else {
            reasons
                .iter()
                .map(|(code, count)| format!("{code} x{count}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "the {engine} route was admitted and served none of the {documents} documents:\n\
             each one left it for a per-input reason ({recorded}).\n\
             This run measured nothing about it and no record was written.\n\
             Those codes are what each document's `ExecutionFacts::reason` carried;\n\
             `Tokenizer::plan().reasons` says why the admitted route is what it is, and\n\
             `toktier-rust doctor` answers about this build rather than about one input."
        )
    } else {
        format!(
            "the {engine} route served {served} of {documents} documents; \
             the served ones compared equal,\n\
             but the run does not cover the route and no record was written.\n\
             The rest went to the reference path document by document, so a record needs\n\
             an input the route serves throughout."
        )
    }
}

fn compare_one(engine: Engine, family: &str, request: &Request) -> Result<Comparison> {
    let device = match engine {
        Engine::Cpu => crate::Device::Cpu,
        Engine::Gpu => crate::Device::Cuda(request.device),
    };
    let accelerated = crate::Runtime::builder()
        .device(device)
        .gpu_delivery(match engine {
            Engine::Cpu => crate::GpuDelivery::Disabled,
            Engine::Gpu => request.delivery,
        })
        // Every document goes to the accelerated backend, whatever its
        // size: this is a comparison, not a routing decision.
        .gpu_min_bytes(0)
        .build()?
        .load(family)?;
    let reference = crate::Runtime::builder()
        .device(crate::Device::Cpu)
        .policy(crate::Policy::Reference)
        .build()?
        .load(family)?;
    let plan = accelerated.plan().clone();
    let route = format!("{:?}", plan.certification);
    let key = match engine {
        Engine::Cpu => cpu_key(family, &plan.artifact_sha256),
        Engine::Gpu => match accelerated.verification_key() {
            Some(key) => key.clone(),
            None => {
                return Ok(Comparison {
                    engine: engine.word().to_owned(),
                    family: family.to_owned(),
                    route,
                    documents: 0,
                    accelerated_documents: 0,
                    bytes: 0,
                    mismatches: 0,
                    first_mismatch: None,
                    record: None,
                    note: Some(
                        "this device and delivery are in the judged list, so there is nothing \
                         here for a local check to add; run it where the route is labelled \
                         supported"
                            .to_owned(),
                    ),
                })
            }
        },
    };
    if request.forget {
        let existed = forget_record(&key)?;
        return Ok(Comparison {
            engine: engine.word().to_owned(),
            family: family.to_owned(),
            route,
            documents: 0,
            accelerated_documents: 0,
            bytes: 0,
            mismatches: 0,
            first_mismatch: None,
            record: None,
            note: Some(if existed {
                "the record for this combination was forgotten".to_owned()
            } else {
                "this combination had no record".to_owned()
            }),
        });
    }
    let expected_backend = match engine {
        Engine::Cpu => crate::Backend::FastCpu,
        Engine::Gpu => crate::Backend::Gpu,
    };
    if !plan.backends.contains(&expected_backend) {
        // The plan already recorded why, so the note carries those codes
        // rather than sending the reader to a command that answers a
        // different question. `Tokenizer::plan().reasons` is the same
        // list in typed form, and `toktier-rust doctor` reports the
        // build facts the plan-time codes rest on.
        let recorded = if plan.reasons.is_empty() {
            "the plan recorded no reason".to_owned()
        } else {
            format!(
                "the plan recorded {}",
                plan.reasons
                    .iter()
                    .map(reason_code)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        return Ok(Comparison {
            engine: engine.word().to_owned(),
            family: family.to_owned(),
            route,
            documents: 0,
            accelerated_documents: 0,
            bytes: 0,
            mismatches: 0,
            first_mismatch: None,
            record: None,
            note: Some(format!(
                "this build admitted no {} route for {family}, so there is nothing to compare.\n\
                 {recorded}.\n\
                 `Tokenizer::plan().reasons` carries the same codes in typed form, and\n\
                 `toktier-rust doctor` reports the build facts they rest on.",
                engine.word()
            )),
        });
    }
    let mut bytes = 0u64;
    let mut mismatches = 0u64;
    let mut accelerated_documents = 0u64;
    let mut first_mismatch = None;
    let mut unserved_reasons: Vec<(String, u64)> = Vec::new();
    for (index, document) in request.documents.iter().enumerate() {
        let observed = accelerated.encode(document)?;
        let judged = reference.encode(document)?;
        bytes += document.len() as u64;
        if observed.execution().backend == expected_backend {
            accelerated_documents += 1;
        } else if let Some(reason) = observed.execution().reason.as_ref() {
            let code = reason_code(reason);
            match unserved_reasons.iter_mut().find(|(seen, _)| *seen == code) {
                Some((_, count)) => *count += 1,
                None => unserved_reasons.push((code, 1)),
            }
        }
        if observed.ids() != judged.ids() {
            mismatches += 1;
            if first_mismatch.is_none() {
                let position = observed
                    .ids()
                    .iter()
                    .zip(judged.ids())
                    .position(|(left, right)| left != right)
                    .unwrap_or_else(|| observed.ids().len().min(judged.ids().len()));
                first_mismatch = Some((index as u64, position as u64));
            }
        }
    }
    // A run the route did not serve in full does not cover it, so no
    // record is written. Two different things reach here and the
    // sentence says which: a route that served nothing compared the
    // reference engine with itself and measured nothing at all, while a
    // route that served some of the documents did measure those, and
    // they agreed -- what is missing is coverage, not a result.
    // Disagreement is different again: ids that differ are evidence
    // whoever served them, and that path writes its record above.
    if mismatches == 0 && accelerated_documents < request.documents.len() as u64 {
        let note = uncovered_note(
            engine.word(),
            accelerated_documents,
            request.documents.len() as u64,
            &unserved_reasons,
        );
        return Ok(Comparison {
            engine: engine.word().to_owned(),
            family: family.to_owned(),
            route,
            documents: request.documents.len() as u64,
            accelerated_documents,
            bytes,
            mismatches,
            first_mismatch,
            record: None,
            note: Some(note),
        });
    }
    let record = VerificationRecord {
        schema: RECORD_SCHEMA.to_owned(),
        key,
        status: if mismatches == 0 { "passed" } else { "failed" }.to_owned(),
        documents: request.documents.len() as u64,
        bytes,
        mismatches,
        first_mismatch,
        input: request.input_label.clone(),
        input_digest: input_digest(&request.documents),
        taken_at: now(),
    };
    let path = write_record(&record)?;
    Ok(Comparison {
        engine: engine.word().to_owned(),
        family: family.to_owned(),
        route,
        documents: record.documents,
        accelerated_documents,
        bytes,
        mismatches,
        first_mismatch,
        record: Some(path.display().to_string()),
        note: None,
    })
}
