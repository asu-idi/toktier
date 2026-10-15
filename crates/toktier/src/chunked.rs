//! Chunk-parallel encode of one long request on the corrected CPU route.
//!
//! A single encode runs on one core. [`Chunked::encode`] cuts a long
//! document at *certified stable boundaries*, encodes the chunks in
//! parallel on a persistent pool of corrected-CPU workers, and
//! concatenates the id rows. The result equals the serial encode of the
//! same route by construction: a cut is placed only where the
//! pre-tokenizer of the certified byte-level BPE families cannot carry a
//! pre-token across it (the synchronizing-transition rule the session
//! repair already uses), never inside an added-token literal, and, for a
//! family whose artifact normalizes its input, only where the cut is also
//! a certified cut of the normalized text. A position where no certified
//! cut can be found is simply not cut: fewer chunks, never a different id.
//!
//! This is the Rust face of the Python module `toktier.chunked`, over the
//! same native chunk encoder. The plan is the same, position for
//! position, the chunk-count table is the same, and the admission
//! vocabulary is the same: the strings [`Admission::refusal`] carries are
//! the ones `toktier.chunked.admission(tokenizer)["refusal"]` carries for
//! the same artifact. Where the two faces differ it is because this crate
//! has one path where the Python facade has two.
//!
//! # Admission
//!
//! A tokenizer may be chunked when four things hold, read in this order
//! and reported by [`Chunked::admission`]:
//!
//! 1. Its family and artifact digest are in the certified repair roster
//!    (refusal `artifact_not_in_certified_repair_roster`).
//! 2. Its artifact document has a normalizer shape the cut rule reasons
//!    about -- none, the empty sequence, or NFC with every added token
//!    matched on the input text -- and no added token that sets
//!    `lstrip`, `rstrip` or `single_word` (refusals
//!    `normalizer_not_supported[:<type>]`,
//!    `nfc_with_normalized_added_tokens`,
//!    `added_token_strip_or_single_word`). This is the native chunk
//!    encoder's own structural rule, read from the same artifact bytes
//!    this crate executes.
//! 3. The chunked admission table that ships with the crate records both
//!    levels of evidence, the adversarial battery and the corpus cut
//!    comparison, for the **native** path on this artifact (refusals
//!    `artifact_without_chunked_evidence` when it records none,
//!    `chunked_evidence_incomplete:<status>` when it records one level,
//!    `loader_face_differs_from_chunked_evidence:<digest>` when the row's
//!    readings were taken on a loader face other than the artifact file
//!    this crate executes).
//! 4. The admitted route carries the corrected CPU engine
//!    ([`Admission::cpu_route`]). Without it there is nothing to chunk on:
//!    [`Chunked::encode`] encodes serially through the normal route and
//!    says so with [`ReasonCode::ChunkRouteNotFastCpu`].
//!
//! The Python facade reads the same table for its own path, so the two
//! faces answer alike about a roster artifact whose evidence is
//! incomplete. One step has no Rust counterpart: the Python facade also
//! refuses when the installed reference release is outside the record's
//! certified set. On this crate the embedded reference is judged at
//! build time against its certified reference versions, and a build that
//! resolves another version has no corrected CPU route at all, which the
//! fourth step reports.
//!
//! [`Policy::Experimental`] lifts the third step, the way the campaign
//! tool lifts it to collect the evidence a row is made of;
//! [`Admission::gate`] says when that happened, and every result is then
//! labelled [`Certification::Experimental`] like the rest of the route.
//!
//! # Threads and workers
//!
//! Two pools take part, and they are not the same thing. The chunks are
//! encoded by corrected-CPU *workers*, forked from the engine on first
//! use and kept on a free list; [`Chunked::warm`] forks them ahead of the
//! first request. The chunk encodes run as one parallel iteration on the
//! process-wide Rayon pool the batch path and the native request path
//! already share, sized by `RAYON_NUM_THREADS` or the CPU count, so one
//! setting governs every parallel section of the process and nothing is
//! spawned per request. A host that wants the chunks on a pool of its own
//! runs the call inside that pool's `install`: Rayon executes a parallel
//! iteration started from one of a pool's threads on that pool, which is
//! the documented contract of `rayon::ThreadPool::install`. This crate
//! takes no `rayon` edge of its own for it, so the host names the crate
//! and the snippet is not compiled here:
//!
//! ```rust,ignore
//! let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
//! let encoded = pool.install(|| tokenizer.chunked().encode(&text, &ChunkOptions::default()))?;
//! ```
//!
//! With one thread the chunks run one after another and stay exact.
//!
//! # Example
//!
//! ```rust,no_run
//! use toktier::chunked::ChunkOptions;
//! use toktier::{Device, Runtime};
//!
//! let runtime = Runtime::builder().device(Device::Cpu).build()?;
//! let tokenizer = runtime.load("qwen3_8b")?;
//! let text = "a long request ".repeat(100_000);
//! let admission = tokenizer.chunked().admission()?;
//! println!("admitted={} refusal={:?}", admission.admitted, admission.refusal);
//! let encoded = tokenizer.chunked().encode(&text, &ChunkOptions::default())?;
//! assert_eq!(encoded.ids(), tokenizer.encode(&text)?.ids());
//! println!("chunks={} serial_reason={:?}", encoded.chunks, encoded.serial_reason);
//! # Ok::<(), toktier::Error>(())
//! ```

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use toktier_routing_core::{
    default_chunk_workers, BpeSyncBoundary, ChunkEncoder, ChunkedIds, FastEncodeSource,
    CHUNKED_DEFAULT_SCAN_BUDGET, R_CHUNK_FAMILY_UNSUPPORTED,
};

use crate::session::TokenizerInner;
use crate::{Backend, Certification, Error, ErrorCode, Policy, ReasonCode, Result, TokenBuffer};

/// How far below each target position the cut search looks before it
/// gives that cut up, in characters (`toktier.chunked.DEFAULT_SCAN_BUDGET`).
pub const DEFAULT_SCAN_BUDGET: usize = CHUNKED_DEFAULT_SCAN_BUDGET;

/// Characters below which [`default_chunks`] answers one chunk
/// (`toktier.chunked.MIN_CHARS`).
pub const MIN_CHARS: usize = 256_000;

/// The payload one worker is worth: [`default_chunks`] never asks for
/// more chunks than one per started slice of this many characters
/// (`toktier.chunked.MIN_BYTES_PER_WORKER`, counted in characters, which
/// never overstates the byte length).
pub const MIN_BYTES_PER_WORKER: usize = 64 * 1024;

/// The chunk-count staircase of `toktier.chunked.CHUNK_TABLE`, as
/// `(characters below, chunks)`, read top down; the last row has no bound.
const CHUNK_TABLE: [(Option<usize>, usize); 4] = [
    (Some(MIN_CHARS), 1),
    (Some(1_000_000), 4),
    (Some(4_000_000), 16),
    (None, 32),
];

/// The evidence levels a chunked admission row records per path, as
/// `tools/update_chunked_admission.py` names them.
const EVIDENCE_ADVERSARIAL: &str = "adversarial";
const EVIDENCE_CORPUS: &str = "corpus";

/// The Python facade's refusal strings that this face answers with as
/// well. The structural ones come from the native chunk encoder itself.
const REFUSAL_ROSTER: &str = "artifact_not_in_certified_repair_roster";
const REFUSAL_NO_EVIDENCE: &str = "artifact_without_chunked_evidence";
const REFUSAL_INCOMPLETE: &str = "chunked_evidence_incomplete";
const REFUSAL_FACE: &str = "loader_face_differs_from_chunked_evidence";

/// The chunk count `toktier.chunked.default_chunks` selects for a
/// document of `chars` Unicode scalar values: one below 256 K, 4 up to
/// 1 M, 16 up to 4 M and 32 above, capped by one chunk per started
/// [`MIN_BYTES_PER_WORKER`] characters.
pub fn default_chunks(chars: usize) -> usize {
    let mut selected = 1;
    for (bound, chunks) in CHUNK_TABLE {
        if bound.is_none_or(|bound| chars < bound) {
            selected = chunks;
            break;
        }
    }
    let useful = chars.div_ceil(MIN_BYTES_PER_WORKER);
    selected.min(useful).max(1)
}

/// The bounds of one chunked call, the keyword arguments of
/// `toktier.chunked.encode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChunkOptions {
    /// The requested chunk count; `None` reads the chunk table for the
    /// document length ([`default_chunks`]). Zero is refused.
    pub k: Option<usize>,
    /// The smallest chunk the plan may produce, first and last included;
    /// `None` selects `max(16, min(1024, len / (4 k)))`.
    pub min_chunk: Option<usize>,
    /// How far below each target the cut search looks
    /// ([`DEFAULT_SCAN_BUDGET`]). Zero is refused.
    pub scan_budget: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            k: None,
            min_chunk: None,
            scan_budget: DEFAULT_SCAN_BUDGET,
        }
    }
}

impl ChunkOptions {
    /// The table-driven defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request `k` chunks.
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Bind the smallest chunk the plan may produce.
    pub fn min_chunk(mut self, chars: usize) -> Self {
        self.min_chunk = Some(chars);
        self
    }

    /// Bound the cut search below each target.
    pub fn scan_budget(mut self, chars: usize) -> Self {
        self.scan_budget = chars;
        self
    }
}

/// Where a document would be cut, and what the search had to give up
/// (`toktier.chunked.ChunkPlan`, with the byte spans the native planner
/// also reports).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChunkPlan {
    /// Half-open character spans, in order, covering the document.
    pub spans: Vec<(usize, usize)>,
    /// The same spans as UTF-8 byte ranges of the input.
    pub byte_spans: Vec<(usize, usize)>,
    /// The requested chunk count.
    pub k: usize,
    /// Characters in the document.
    pub chars: usize,
    /// Target positions for which no certified cut was found.
    pub cut_misses: usize,
    /// Candidate positions rejected because they fell inside a literal.
    pub literal_rejects: usize,
    /// Candidate positions rejected by the normalization-stability
    /// condition (NFC families only).
    pub norm_rejects: usize,
}

impl ChunkPlan {
    /// Character positions of the cuts.
    pub fn cuts(&self) -> Vec<usize> {
        self.spans.iter().skip(1).map(|span| span.0).collect()
    }
}

/// How many rows each implementation produced in one chunked call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RowSources {
    /// Rows the corrected engine encoded.
    pub gigatoken: usize,
    /// Rows the reference encoded because they hold an added-token
    /// literal.
    pub hf_added_token: usize,
    /// Rows the reference encoded because the engine's guard could not be
    /// proved for them.
    pub hf_engine_guard: usize,
}

/// The ids of one chunked call and how they were produced
/// (`toktier.chunked.ChunkedEncoding` with its `native_facts`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ChunkedEncoding {
    ids: TokenBuffer,
    /// Rows that ran: 1 for a serial result.
    pub chunks: usize,
    /// The chunk count that was requested.
    pub k_requested: usize,
    /// Character positions of the cuts; empty for a serial result.
    pub cuts: Vec<usize>,
    /// Target positions for which no certified cut was found.
    pub cut_misses: usize,
    /// Candidate positions rejected because they fell inside a literal.
    pub literal_rejects: usize,
    /// Candidate positions rejected by the normalization-stability
    /// condition.
    pub norm_rejects: usize,
    /// Why the document was encoded serially instead, when it was:
    /// [`ReasonCode::ChunkNoCut`], [`ReasonCode::ChunkFamilyUnsupported`]
    /// or [`ReasonCode::ChunkRouteNotFastCpu`]. `None` for a chunked
    /// result and for a document below the threshold.
    pub serial_reason: Option<ReasonCode>,
    /// Whether the document was encoded serially because it was shorter
    /// than 256 characters or one chunk was requested.
    pub below_threshold: bool,
    /// Which implementation produced each row.
    pub sources: RowSources,
    /// The label of the route the rows ran on.
    pub certification: Certification,
    /// Time spent planning the cuts.
    pub plan_time: Duration,
    /// Time spent in the parallel section, or in the serial encode.
    pub encode_time: Duration,
    /// Time spent concatenating the rows.
    pub concat_time: Duration,
    /// Workers forked during this call because the free list was empty.
    pub cold_forks: usize,
}

impl ChunkedEncoding {
    /// The core-stream ids, exactly those of the serial encode.
    pub fn ids(&self) -> &[u32] {
        self.ids.as_slice()
    }

    /// Borrow the ids as a shared buffer.
    pub fn token_buffer(&self) -> &TokenBuffer {
        &self.ids
    }

    /// Take the ids as a shared buffer.
    pub fn into_token_buffer(self) -> TokenBuffer {
        self.ids
    }

    fn serial(
        ids: Vec<u32>,
        k_requested: usize,
        reason: ReasonCode,
        certification: Certification,
        encode_time: Duration,
    ) -> Self {
        Self {
            ids: TokenBuffer::from(ids),
            chunks: 1,
            k_requested,
            cuts: Vec::new(),
            cut_misses: 0,
            literal_rejects: 0,
            norm_rejects: 0,
            serial_reason: Some(reason),
            below_threshold: false,
            sources: RowSources::default(),
            certification,
            plan_time: Duration::ZERO,
            encode_time,
            concat_time: Duration::ZERO,
            cold_forks: 0,
        }
    }

    fn from_native(encoded: ChunkedIds, certification: Certification) -> Self {
        let mut sources = RowSources::default();
        for source in &encoded.sources {
            match source {
                FastEncodeSource::Gigatoken => sources.gigatoken += 1,
                FastEncodeSource::ReferenceAddedToken => sources.hf_added_token += 1,
                FastEncodeSource::ReferenceEngineGuard => sources.hf_engine_guard += 1,
            }
        }
        Self {
            ids: TokenBuffer::from(encoded.ids),
            chunks: encoded.chunks,
            k_requested: encoded.k_requested,
            cuts: encoded.cuts,
            cut_misses: encoded.cut_misses,
            literal_rejects: encoded.literal_rejects,
            norm_rejects: encoded.norm_rejects,
            serial_reason: encoded.serial_reason.map(ReasonCode::from_ledger_code),
            below_threshold: encoded.below_threshold,
            sources,
            certification,
            plan_time: Duration::from_nanos(encoded.plan_ns),
            encode_time: Duration::from_nanos(encoded.encode_ns),
            concat_time: Duration::from_nanos(encoded.concat_ns),
            cold_forks: encoded.cold_forks,
        }
    }
}

/// Which gate decided the evidence step of the admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AdmissionGate {
    /// The shipped chunked admission table was read, and a roster
    /// artifact without complete native-path evidence is refused.
    EvidenceTable,
    /// [`Policy::Experimental`] lifted the table step; the structural
    /// rule and the roster still decide, and every result is labelled
    /// experimental.
    LiftedByExperimentalPolicy,
}

/// The admission decision for one tokenizer and what it rests on
/// (`toktier.chunked.admission(tokenizer)`, the keys this face has).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Admission {
    /// Whether [`Chunked::encode`] cuts documents of this tokenizer.
    /// Equivalent to `refusal.is_none()`; the route is reported apart,
    /// in [`Self::cpu_route`].
    pub admitted: bool,
    /// Why not, in the Python facade's vocabulary (module documentation).
    pub refusal: Option<String>,
    /// The family this decision is for.
    pub family: String,
    /// The verified artifact the decision is bound to.
    pub artifact_sha256: String,
    /// Whether the admitted route carries the corrected CPU engine the
    /// chunk encoder runs on. `false` on a reference-only route, where
    /// [`Chunked::encode`] encodes serially under
    /// [`ReasonCode::ChunkRouteNotFastCpu`] and [`Chunked::plan`] has no
    /// planner to ask.
    pub cpu_route: bool,
    /// The normalizer shape the cut rule reasons about for this artifact,
    /// `"none"` or `"nfc"`; `None` when the artifact was refused before
    /// or by that step.
    pub normalizer: Option<String>,
    /// Unique non-empty added-token literals the cut guard knows.
    pub literals: usize,
    /// The evidence id of the admission row, when the table has one for
    /// this artifact.
    pub evidence_id: Option<String>,
    /// What the row records for the native path: `"certified"` when both
    /// evidence levels are recorded, `"adversarial_only"` or
    /// `"corpus_only"` when one is, `None` when the table has no row or
    /// the row records nothing for this path.
    pub evidence_status: Option<String>,
    /// The evidence levels the row records for the native path.
    pub native_evidence: Vec<String>,
    /// Which gate decided the evidence step.
    pub gate: AdmissionGate,
}

/// The counters this surface keeps per tokenizer
/// (`toktier.chunked.stats(tokenizer)`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ChunkedStats {
    /// Calls to [`Chunked::encode`].
    pub calls: u64,
    /// Calls that cut the document.
    pub applied: u64,
    /// Calls encoded serially for a reason (`fallback_counts` says which).
    pub serial: u64,
    /// Calls encoded serially because the document was too short or one
    /// chunk was requested.
    pub below_threshold: u64,
    /// Rows run over every call.
    pub chunks_total: u64,
    /// Target positions given up over every call.
    pub cut_misses: u64,
    /// Candidates rejected by the literal guard over every call.
    pub literal_rejects: u64,
    /// Candidates rejected by the normalization condition over every call.
    pub norm_rejects: u64,
    /// Serial calls per reason code, by the code's `R_*` spelling.
    pub fallback_counts: BTreeMap<String, u64>,
    /// Corrected-CPU workers forked so far for this tokenizer.
    pub workers: usize,
    /// The worker count [`Chunked::warm`] forks when none is named: the
    /// Rayon thread count.
    pub default_workers: usize,
}

#[derive(Debug, Default)]
struct Counters {
    calls: u64,
    applied: u64,
    serial: u64,
    below_threshold: u64,
    chunks_total: u64,
    cut_misses: u64,
    literal_rejects: u64,
    norm_rejects: u64,
    fallback_counts: BTreeMap<String, u64>,
}

impl Counters {
    fn record(&mut self, encoded: &ChunkedEncoding) {
        self.calls += 1;
        if let Some(reason) = &encoded.serial_reason {
            self.serial += 1;
            *self
                .fallback_counts
                .entry(reason_spelling(reason))
                .or_insert(0) += 1;
        } else if encoded.below_threshold {
            self.below_threshold += 1;
        } else {
            self.applied += 1;
        }
        self.chunks_total += encoded.chunks as u64;
        self.cut_misses += encoded.cut_misses as u64;
        self.literal_rejects += encoded.literal_rejects as u64;
        self.norm_rejects += encoded.norm_rejects as u64;
    }
}

/// The `R_*` spelling of a chunked serial reason, for the counters.
fn reason_spelling(reason: &ReasonCode) -> String {
    match reason {
        ReasonCode::ChunkNoCut => toktier_routing_core::R_CHUNK_NO_CUT.to_owned(),
        ReasonCode::ChunkFamilyUnsupported => R_CHUNK_FAMILY_UNSUPPORTED.to_owned(),
        ReasonCode::ChunkRouteNotFastCpu => {
            toktier_routing_core::R_CHUNK_ROUTE_NOT_FAST_CPU.to_owned()
        }
        ReasonCode::Other(code) => code.clone(),
        other => format!("{other:?}"),
    }
}

/// Per-tokenizer chunked state: the admission, decided once, and the
/// counters.
#[derive(Debug, Default)]
pub(crate) struct State {
    admission: OnceLock<std::result::Result<Admission, (ErrorCode, String)>>,
    counters: Mutex<Counters>,
}

/// The chunk-parallel surface of one tokenizer, from
/// [`Tokenizer::chunked`](crate::Tokenizer::chunked).
#[derive(Debug, Clone, Copy)]
pub struct Chunked<'a> {
    inner: &'a Arc<TokenizerInner>,
}

impl<'a> Chunked<'a> {
    pub(crate) fn new(inner: &'a Arc<TokenizerInner>) -> Self {
        Self { inner }
    }

    /// The admission decision and what it rests on, decided once per
    /// tokenizer on first use.
    pub fn admission(&self) -> Result<Admission> {
        self.inner
            .chunked
            .admission
            .get_or_init(|| {
                decide(self.inner).map_err(|error| (error.code(), error.message().to_owned()))
            })
            .clone()
            .map_err(|(code, message)| {
                Error::new(code, message).with_family(&self.inner.artifact.identity().family)
            })
    }

    /// The cut plan [`Self::encode`] would use for `text`, without
    /// encoding.
    ///
    /// Where `encode` prefers to fall back, this prefers to say so: a
    /// tokenizer outside the admitted set answers
    /// [`ErrorCode::UncertifiedTokenizer`] naming the refusal, and so
    /// does a route without the corrected CPU engine.
    pub fn plan(&self, text: &str, options: &ChunkOptions) -> Result<ChunkPlan> {
        let (k, budget) = resolve(options, text)?;
        let admission = self.admission()?;
        if let Some(refusal) = &admission.refusal {
            return Err(Error::new(
                ErrorCode::UncertifiedTokenizer,
                format!(
                    "{} has no certified chunk-cut rule: {refusal}",
                    admission.family
                ),
            )
            .with_family(&admission.family));
        }
        if !admission.cpu_route {
            return Err(no_cpu_route(&admission.family));
        }
        let plan = self
            .inner
            .router
            .chunked_plan(text, k, options.min_chunk, budget)
            .map_err(|error| Error::new(ErrorCode::BackendExecutionFault, error.to_string()))?;
        Ok(ChunkPlan {
            spans: plan.spans,
            byte_spans: plan.byte_spans,
            k: plan.k,
            chars: plan.chars,
            cut_misses: plan.cut_misses,
            literal_rejects: plan.literal_rejects,
            norm_rejects: plan.norm_rejects,
        })
    }

    /// Encode `text` in certified chunks. The ids equal
    /// [`Tokenizer::encode`](crate::Tokenizer::encode) of the same text
    /// with default options, the core stream without post-processor
    /// special tokens.
    ///
    /// A tokenizer outside the admitted set, a document with no certified
    /// cut and an input whose route does not start at the corrected CPU
    /// engine are encoded serially; the result says which in
    /// [`ChunkedEncoding::serial_reason`], and [`Self::stats`] counts it.
    /// Only a malformed option ([`ErrorCode::InvalidArgument`]) and an
    /// engine fault are errors.
    pub fn encode(&self, text: &str, options: &ChunkOptions) -> Result<ChunkedEncoding> {
        let (k, budget) = resolve(options, text)?;
        let admission = self.admission()?;
        let certification = self.inner.plan.certification;
        let encoded = if admission.admitted {
            let encoded = self
                .inner
                .router
                .chunked_encode_ids(text, k, options.min_chunk, budget)
                .map_err(|error| Error::new(ErrorCode::BackendExecutionFault, error.to_string()))?;
            ChunkedEncoding::from_native(encoded, certification)
        } else {
            // The plain route, as the Python facade takes it for a
            // refused tokenizer: the same guarded serial encode the
            // chunks would have used.
            let started = Instant::now();
            let routed =
                self.inner.router.encode_ids(text, false).map_err(|error| {
                    Error::new(ErrorCode::BackendExecutionFault, error.to_string())
                })?;
            ChunkedEncoding::serial(
                routed.ids,
                k,
                ReasonCode::ChunkFamilyUnsupported,
                certification,
                started.elapsed(),
            )
        };
        self.inner
            .chunked
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .record(&encoded);
        Ok(encoded)
    }

    /// Fork the corrected-CPU worker pool ahead of the first request, up
    /// to `workers` workers (the Rayon thread count when `None`), running
    /// each new worker once over `text` when one is given so its caches
    /// are warm. Returns the pool size. Without the call the pool is
    /// forked on first use.
    pub fn warm(&self, workers: Option<usize>, text: Option<&str>) -> Result<usize> {
        let admission = self.admission()?;
        if !admission.cpu_route {
            return Err(no_cpu_route(&admission.family));
        }
        self.inner
            .router
            .chunked_warm(workers.unwrap_or_else(default_chunk_workers), text)
            .map_err(|error| Error::new(ErrorCode::BackendExecutionFault, error.to_string()))
    }

    /// The counters this surface keeps for the tokenizer.
    pub fn stats(&self) -> ChunkedStats {
        let counters = self
            .inner
            .chunked
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ChunkedStats {
            calls: counters.calls,
            applied: counters.applied,
            serial: counters.serial,
            below_threshold: counters.below_threshold,
            chunks_total: counters.chunks_total,
            cut_misses: counters.cut_misses,
            literal_rejects: counters.literal_rejects,
            norm_rejects: counters.norm_rejects,
            fallback_counts: counters.fallback_counts.clone(),
            workers: self
                .inner
                .router
                .chunk_encoder()
                .map(ChunkEncoder::worker_count)
                .unwrap_or(0),
            default_workers: default_chunk_workers(),
        }
    }
}

fn no_cpu_route(family: &str) -> Error {
    Error::new(
        ErrorCode::UncertifiedTokenizer,
        format!("family {family} has no corrected CPU route on this runtime to chunk on"),
    )
    .with_family(family)
}

/// The chunk count and scan budget of one call, checked the way
/// `toktier.chunked` checks its keyword arguments.
fn resolve(options: &ChunkOptions, text: &str) -> Result<(usize, usize)> {
    if options.scan_budget == 0 {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "scan_budget must be a positive integer",
        ));
    }
    let k = match options.k {
        Some(0) => {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "k must be a positive integer or None",
            ))
        }
        Some(k) => k,
        None => default_chunks(text.chars().count()),
    };
    Ok((k, options.scan_budget))
}

/// The native-path status a row's evidence levels spell, as the table
/// generator spells the Python path's: both levels, one, or none.
fn native_status(levels: &[String]) -> Option<String> {
    let adversarial = levels.iter().any(|level| level == EVIDENCE_ADVERSARIAL);
    let corpus = levels.iter().any(|level| level == EVIDENCE_CORPUS);
    match (adversarial, corpus) {
        (true, true) => Some("certified".to_owned()),
        (true, false) => Some(format!("{EVIDENCE_ADVERSARIAL}_only")),
        (false, true) => Some(format!("{EVIDENCE_CORPUS}_only")),
        (false, false) => None,
    }
}

/// The admission decision, in the order the Python planner makes it:
/// the roster, the structural rule of the native encoder, the evidence
/// table (unless lifted), with the route reported beside.
fn decide(inner: &Arc<TokenizerInner>) -> Result<Admission> {
    let identity = inner.artifact.identity();
    let family = identity.family.clone();
    let artifact_sha256 = identity.tokenizer_sha256.clone();
    let cpu_route = inner.plan.backends.contains(&Backend::FastCpu);
    let gate = if inner.policy == Policy::Experimental {
        AdmissionGate::LiftedByExperimentalPolicy
    } else {
        AdmissionGate::EvidenceTable
    };
    let roster = inner
        .registry
        .repair(&family)
        .is_some_and(|row| row.artifact_sha256 == artifact_sha256);
    let (normalizer, literals, mut refusal) = if !roster {
        (None, 0, Some(REFUSAL_ROSTER.to_owned()))
    } else {
        // The structural facts come from the native encoder itself. On a
        // route that carries the engine it is the engine's own planner;
        // on any other route the same planner is built once over the
        // same artifact bytes for its facts alone.
        match inner.router.chunk_encoder() {
            Ok(encoder) => structural(encoder),
            Err(_) => {
                let boundary = BpeSyncBoundary::new(inner.registry.pclass())
                    .map_err(|error| Error::new(ErrorCode::RegistryInvalid, error.to_string()))?;
                let encoder = ChunkEncoder::new(
                    &inner.artifact.bytes(),
                    boundary,
                    Arc::clone(&inner.reference),
                )
                .map_err(|error| {
                    Error::new(
                        ErrorCode::Internal,
                        format!("chunk planner construction failed: {error}"),
                    )
                })?;
                structural(&encoder)
            }
        }
    };

    let row = inner
        .registry
        .chunked_admission(&family)
        .filter(|row| row.artifact_sha256 == artifact_sha256);
    let evidence_id = row.map(|row| row.evidence_id.clone());
    let native_evidence = row
        .map(|row| row.evidence.native.clone())
        .unwrap_or_default();
    let evidence_status = native_status(&native_evidence);
    if refusal.is_none() && gate == AdmissionGate::EvidenceTable {
        match (row, evidence_status.as_deref()) {
            (None, _) | (Some(_), None) => refusal = Some(REFUSAL_NO_EVIDENCE.to_owned()),
            (Some(_), Some(status)) if status != "certified" => {
                refusal = Some(format!("{REFUSAL_INCOMPLETE}:{status}"));
            }
            (Some(row), Some(_)) => {
                // The row's readings were taken on a loader face; this
                // crate executes the artifact file, which is that face
                // when the loader did not extend it.
                if row
                    .loader_face_sha256
                    .as_deref()
                    .is_some_and(|face| face != artifact_sha256)
                {
                    refusal = Some(format!(
                        "{REFUSAL_FACE}:{}",
                        &artifact_sha256[..artifact_sha256.len().min(12)]
                    ));
                }
            }
        }
    }

    Ok(Admission {
        admitted: refusal.is_none(),
        refusal,
        family,
        artifact_sha256,
        cpu_route,
        normalizer,
        literals,
        evidence_id,
        evidence_status,
        native_evidence,
        gate,
    })
}

/// The structural facts of a chunk encoder: normalizer shape, literal
/// count and refusal.
fn structural(encoder: &ChunkEncoder) -> (Option<String>, usize, Option<String>) {
    (
        encoder.normalizer().map(|mode| mode.as_str().to_owned()),
        encoder.literal_count(),
        encoder.refusal().map(str::to_owned),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_chunks_follows_the_python_table() {
        // The staircase, and the one-worker-per-64-KiB cap below it.
        assert_eq!(default_chunks(0), 1);
        assert_eq!(default_chunks(255_999), 1);
        assert_eq!(default_chunks(256_000), 4);
        assert_eq!(default_chunks(999_999), 4);
        assert_eq!(default_chunks(1_000_000), 16);
        assert_eq!(default_chunks(3_999_999), 16);
        assert_eq!(default_chunks(4_000_000), 32);
        assert_eq!(default_chunks(100_000_000), 32);
        // 256_000 characters are worth four workers (ceil(256000/65536)),
        // so the cap does not bind there; 262_144 is exactly four.
        assert_eq!(default_chunks(262_144), 4);
        assert_eq!(
            default_chunks(1_000_000),
            16.min(1_000_000usize.div_ceil(65_536))
        );
    }

    #[test]
    fn options_refuse_zero_bounds() {
        let text = "x".repeat(300);
        let zero_budget = ChunkOptions::default().scan_budget(0);
        assert_eq!(
            resolve(&zero_budget, &text).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        let zero_k = ChunkOptions::default().k(0);
        assert_eq!(
            resolve(&zero_k, &text).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            resolve(&ChunkOptions::default(), &text).unwrap(),
            (1, DEFAULT_SCAN_BUDGET)
        );
        assert_eq!(
            resolve(&ChunkOptions::default().k(7).scan_budget(40), &text).unwrap(),
            (7, 40)
        );
        // Characters, not bytes: a 4-byte scalar counts once.
        let long = "\u{1f642}".repeat(300_000);
        assert_eq!(resolve(&ChunkOptions::default(), &long).unwrap().0, 4);
    }

    #[test]
    fn native_status_spells_the_levels_like_the_table_generator() {
        let levels = |items: &[&str]| items.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert_eq!(native_status(&levels(&[])), None);
        assert_eq!(
            native_status(&levels(&["adversarial"])).as_deref(),
            Some("adversarial_only")
        );
        assert_eq!(
            native_status(&levels(&["corpus"])).as_deref(),
            Some("corpus_only")
        );
        assert_eq!(
            native_status(&levels(&["adversarial", "corpus"])).as_deref(),
            Some("certified")
        );
        assert_eq!(
            native_status(&levels(&["corpus", "adversarial"])).as_deref(),
            Some("certified")
        );
        // A level this release does not know is neither.
        assert_eq!(native_status(&levels(&["later"])), None);
    }

    #[test]
    fn counters_follow_the_python_record_rule() {
        let mut counters = Counters::default();
        let chunked = ChunkedEncoding {
            ids: TokenBuffer::from(vec![1, 2]),
            chunks: 3,
            k_requested: 4,
            cuts: vec![10, 20],
            cut_misses: 1,
            literal_rejects: 2,
            norm_rejects: 3,
            serial_reason: None,
            below_threshold: false,
            sources: RowSources::default(),
            certification: Certification::Reference,
            plan_time: Duration::ZERO,
            encode_time: Duration::ZERO,
            concat_time: Duration::ZERO,
            cold_forks: 0,
        };
        counters.record(&chunked);
        let mut short = chunked.clone();
        short.chunks = 1;
        short.below_threshold = true;
        counters.record(&short);
        let serial = ChunkedEncoding::serial(
            vec![1],
            2,
            ReasonCode::ChunkFamilyUnsupported,
            Certification::Reference,
            Duration::ZERO,
        );
        counters.record(&serial);
        let mut no_cut = chunked.clone();
        no_cut.chunks = 1;
        no_cut.serial_reason = Some(ReasonCode::ChunkNoCut);
        counters.record(&no_cut);
        assert_eq!(counters.calls, 4);
        assert_eq!(counters.applied, 1);
        assert_eq!(counters.below_threshold, 1);
        assert_eq!(counters.serial, 2);
        assert_eq!(counters.chunks_total, 3 + 1 + 1 + 1);
        assert_eq!(counters.cut_misses, 3);
        assert_eq!(
            counters.fallback_counts,
            BTreeMap::from([
                ("R_CHUNK_FAMILY_UNSUPPORTED".to_owned(), 1),
                ("R_CHUNK_NO_CUT".to_owned(), 1),
            ])
        );
    }
}
