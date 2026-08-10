//! Corrected Gigatoken full encode and certified append repair in Rust.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use aho_corasick::{AhoCorasick, MatchKind};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use toktier_gigatoken_core::load_tokenizer::hf::{load_hf_slice, HfTokenizer};
use toktier_gigatoken_core::Tokenizer as Gigatoken;
use toktier_store_core::{
    AppendReport, BoundaryCut, Encoding, EngineError, SessionEncoder, SoaEncoding, TailState,
    WitnessCategory,
};

use crate::{BpeSyncBoundary, ReferenceEngine, ReferenceEngineError};

/// Frozen artifact-specific repair parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastRepairSpec {
    pub family: String,
    pub artifact_sha256: String,
    pub margin: usize,
    pub effective_l_max: usize,
    pub has_normalizer: bool,
    pub window_chars: usize,
    pub max_retries: usize,
    pub min_match_tokens: usize,
}

impl FastRepairSpec {
    pub fn new(
        family: String,
        artifact_sha256: String,
        margin: usize,
        effective_l_max: usize,
        has_normalizer: bool,
    ) -> Self {
        Self {
            family,
            artifact_sha256,
            margin,
            effective_l_max,
            has_normalizer,
            window_chars: 512,
            max_retries: 5,
            min_match_tokens: 2,
        }
    }
}

/// Construction failure for the native corrected CPU engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastCpuEngineError(String);

impl FastCpuEngineError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for FastCpuEngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FastCpuEngineError {}

/// Request-path counters retained inside the native engine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FastRepairStats {
    pub path_counts: HashMap<String, u64>,
    pub window_calls: u64,
    pub window_chars: u64,
    pub last_path: Option<String>,
    pub last_reason: Option<String>,
    pub last_kept_tokens: usize,
    pub last_window_chars: usize,
    pub last_retries: usize,
}

/// Which implementation produced one native full-encode result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastEncodeSource {
    Gigatoken,
    ReferenceAddedToken,
    ReferenceEngineGuard,
}

/// Full core-stream result plus the honest implementation identity needed by
/// the native routing ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastEncodeOutcome {
    pub encoding: Encoding,
    pub source: FastEncodeSource,
}

/// State-seed payload of the corrected CPU route.
///
/// The certified Gigatoken row returns closure-verified IDs only, so a
/// session seed can adopt them with lazy checkpointed spans; the
/// reference rows return their own exact character offsets (they hold
/// even when the input is not normalization-stable, which the known-ID
/// reconstruction premise would not cover).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FastSeedPayload {
    /// Exact closure-verified IDs from corrected Gigatoken.
    GigatokenIds(Vec<u32>),
    /// Reference encode with materialized spans.
    Reference(SoaEncoding),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Record {
    original_index: usize,
    global_start: usize,
    global_end: usize,
    token_id: u32,
    same_span_ordinal: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Match {
    left_start: usize,
    right_start: usize,
    length: usize,
    covered_chars: usize,
}

struct AddedTokenGate {
    matcher: Option<AhoCorasick>,
    route_every_input: bool,
}

impl fmt::Debug for AddedTokenGate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AddedTokenGate")
            .field("has_matcher", &self.matcher.is_some())
            .field("route_every_input", &self.route_every_input)
            .finish()
    }
}

impl AddedTokenGate {
    fn from_reference(reference: &ReferenceEngine) -> Result<Self, FastCpuEngineError> {
        let added = reference.added_tokens();
        let route_every_input = reference.has_effective_normalizer()
            && added
                .iter()
                .any(|(_, content, normalized)| *normalized || content.is_empty());
        let patterns = added
            .into_iter()
            .filter_map(|(_, content, normalized)| {
                (!content.is_empty() && (!normalized || !reference.has_effective_normalizer()))
                    .then_some(content)
            })
            .collect::<Vec<_>>();
        let matcher = if patterns.is_empty() {
            None
        } else {
            Some(
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(patterns)
                    .map_err(|error| {
                        FastCpuEngineError::new(format!(
                            "failed to build the added-token gate: {error}"
                        ))
                    })?,
            )
        };
        Ok(Self {
            matcher,
            route_every_input,
        })
    }

    fn may_match(&self, text: &str) -> bool {
        self.route_every_input
            || self
                .matcher
                .as_ref()
                .is_some_and(|matcher| matcher.is_match(text))
    }
}

/// Corrected-Gigatoken state with payload-sized lazy batch workers.
///
/// The primary cache is initialized once at runtime construction so the first
/// agent append cannot inherit parser/setup latency. Forked batch workers stay
/// lazy and grow only when the observed payload can use them; this avoids the
/// former eager per-Rayon-worker startup and memory cost.
struct FastCpuCore {
    gigatoken: Mutex<Gigatoken>,
    workers: RwLock<Vec<Arc<Mutex<Gigatoken>>>>,
}

impl FastCpuCore {
    fn batch_workers(
        &self,
        documents: usize,
        total_bytes: usize,
    ) -> Result<Vec<Arc<Mutex<Gigatoken>>>, EngineError> {
        const MIN_BYTES_PER_WORKER: usize = 64 * 1024;
        let useful_for_payload = total_bytes.div_ceil(MIN_BYTES_PER_WORKER).max(1);
        let required = rayon::current_num_threads()
            .max(1)
            .min(documents)
            .min(useful_for_payload);
        {
            let workers = self
                .workers
                .read()
                .map_err(|_| EngineError("Gigatoken worker pool is poisoned".to_owned()))?;
            if workers.len() >= required {
                return Ok(workers[..required].to_vec());
            }
        }
        let mut workers = self
            .workers
            .write()
            .map_err(|_| EngineError("Gigatoken worker pool is poisoned".to_owned()))?;
        if workers.len() < required {
            let engine = self
                .gigatoken
                .lock()
                .map_err(|_| EngineError("Gigatoken cache mutex is poisoned".to_owned()))?;
            let existing = workers.len();
            workers.extend((existing..required).map(|_| Arc::new(Mutex::new(engine.fork()))));
        }
        Ok(workers[..required].to_vec())
    }
}

/// One immutable reference configuration plus an initialized,
/// mutex-protected corrected Gigatoken cache and lazy worker pool. All
/// tokenizer work is native; the mutex protects only upstream memoization
/// tables and is never held while the HF fallback runs.
pub struct FastCpuEngine {
    spec: FastRepairSpec,
    reference: Arc<ReferenceEngine>,
    tokenizer_json: Box<[u8]>,
    core: OnceLock<Result<FastCpuCore, FastCpuEngineError>>,
    boundary: BpeSyncBoundary,
    added_gate: AddedTokenGate,
    stats: Mutex<FastRepairStats>,
}

impl fmt::Debug for FastCpuEngine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FastCpuEngine")
            .field("family", &self.spec.family)
            .field("artifact_sha256", &self.spec.artifact_sha256)
            .field("config_id", &"toktier-fast-repair-v1")
            .finish_non_exhaustive()
    }
}

impl FastCpuEngine {
    /// Construct both native engines from the same materialized tokenizer
    /// JSON and verify every repair premise before accepting the object.
    pub fn from_json(
        tokenizer_json: &[u8],
        spec: FastRepairSpec,
        pclass: Vec<u8>,
    ) -> Result<Self, FastCpuEngineError> {
        let reference = Arc::new(
            ReferenceEngine::from_bytes(tokenizer_json)
                .map_err(|error| FastCpuEngineError::new(error.to_string()))?,
        );
        Self::build(tokenizer_json, reference, spec, pclass)
    }

    /// Construct over a reference engine already parsed from the same
    /// verified artifact. The Python facade only uses this entry point when
    /// no configuration sidecar contributes additional live tokens.
    pub fn from_reference(
        tokenizer_json: &[u8],
        reference: Arc<ReferenceEngine>,
        spec: FastRepairSpec,
        pclass: Vec<u8>,
    ) -> Result<Self, FastCpuEngineError> {
        let observed: [u8; 32] = Sha256::digest(tokenizer_json).into();
        if &observed != reference.artifact_sha256() {
            return Err(FastCpuEngineError::new(
                "shared reference engine does not belong to tokenizer_json",
            ));
        }
        let observed_hex = observed
            .iter()
            .map(|value| format!("{value:02x}"))
            .collect::<String>();
        if observed_hex != spec.artifact_sha256 {
            return Err(FastCpuEngineError::new(
                "shared reference/tokenizer bytes do not match the certified artifact digest",
            ));
        }
        Self::build(tokenizer_json, reference, spec, pclass)
    }

    fn build(
        tokenizer_json: &[u8],
        reference: Arc<ReferenceEngine>,
        spec: FastRepairSpec,
        pclass: Vec<u8>,
    ) -> Result<Self, FastCpuEngineError> {
        if reference.has_normalizer() != spec.has_normalizer {
            return Err(FastCpuEngineError::new(format!(
                "{} normalizer premise differs: registry={}, artifact={}",
                spec.family,
                spec.has_normalizer,
                reference.has_normalizer()
            )));
        }
        let boundary = BpeSyncBoundary::new(pclass)
            .map_err(|error| FastCpuEngineError::new(error.to_string()))?;
        let added_gate = AddedTokenGate::from_reference(&reference)?;
        let engine = Self {
            spec,
            reference,
            tokenizer_json: tokenizer_json.to_vec().into_boxed_slice(),
            core: OnceLock::new(),
            boundary,
            added_gate,
            stats: Mutex::new(FastRepairStats::default()),
        };
        // Pay the single-core parse/verification cost at construction. This
        // keeps the first session append bounded while preserving lazy,
        // payload-sized worker forks for batch traffic.
        engine.core()?;
        Ok(engine)
    }

    fn core(&self) -> Result<&FastCpuCore, FastCpuEngineError> {
        self.core
            .get_or_init(|| {
                let gigatoken = match load_hf_slice(&self.tokenizer_json)
                    .map_err(|error| FastCpuEngineError::new(error.to_string()))?
                {
                    HfTokenizer::Bpe(tokenizer) => tokenizer,
                    HfTokenizer::SentencePiece(_) => {
                        return Err(FastCpuEngineError::new(
                            "the certified repair route requires a ByteLevel BPE artifact",
                        ));
                    }
                };
                let expected = self
                    .reference
                    .raw_byte_lengths()
                    .map_err(|error| FastCpuEngineError::new(error.to_string()))?;
                let observed = verified_byte_lengths(&gigatoken, expected.len())?;
                if observed != expected {
                    let first = observed
                        .iter()
                        .zip(expected.iter())
                        .position(|(left, right)| left != right)
                        .unwrap_or(0);
                    return Err(FastCpuEngineError::new(format!(
                        "HF/Gigatoken byte-length tables differ at id {first}: {} != {}",
                        expected[first], observed[first]
                    )));
                }
                Ok(FastCpuCore {
                    gigatoken: Mutex::new(gigatoken),
                    workers: RwLock::new(Vec::new()),
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub fn spec(&self) -> &FastRepairSpec {
        &self.spec
    }

    /// Whether the corrected-Gigatoken core finished construction.
    pub fn is_initialized(&self) -> bool {
        self.core.get().is_some()
    }

    pub fn batch_worker_count(&self) -> usize {
        self.core
            .get()
            .and_then(|result| result.as_ref().ok())
            .and_then(|core| core.workers.read().ok().map(|workers| workers.len()))
            .unwrap_or(0)
    }

    pub fn minimum_seal_tail_chars(&self) -> usize {
        let mut window = self.spec.window_chars;
        while window.saturating_sub(self.spec.margin) <= self.spec.effective_l_max {
            window = window.saturating_mul(2);
        }
        window.saturating_add(1)
    }

    pub fn stats(&self) -> FastRepairStats {
        self.stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn reference(&self) -> &ReferenceEngine {
        &self.reference
    }

    pub fn reference_arc(&self) -> Arc<ReferenceEngine> {
        Arc::clone(&self.reference)
    }

    pub fn vocab_size(&self) -> Result<usize, FastCpuEngineError> {
        self.core()?
            .gigatoken
            .lock()
            .map(|engine| engine.vocab_size())
            .map_err(|_| FastCpuEngineError::new("Gigatoken cache mutex is poisoned"))
    }

    pub fn vocab_entries(&self) -> Result<Vec<(u32, Vec<u8>)>, FastCpuEngineError> {
        self.core()?
            .gigatoken
            .lock()
            .map(|engine| {
                engine
                    .vocab_entries()
                    .map(|(id, bytes)| (id, bytes.to_vec()))
                    .collect()
            })
            .map_err(|_| FastCpuEngineError::new("Gigatoken cache mutex is poisoned"))
    }

    fn locked_primary(&self) -> Result<std::sync::MutexGuard<'_, Gigatoken>, EngineError> {
        self.core()
            .map_err(|error| EngineError(error.to_string()))?
            .gigatoken
            .lock()
            .map_err(|_| EngineError("Gigatoken cache mutex is poisoned".to_owned()))
    }

    /// Exact core-stream IDs through the same native route used to seed a
    /// session, without constructing character spans the caller did not
    /// request. The row's raw-byte closure over the input is still
    /// verified, so the acceptance decision matches the span surfaces.
    pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        self.encode_ids_with_source(text).map(|(ids, _source)| ids)
    }

    /// Exact full encode with an explicit source identity for the router.
    /// This retained pair-based surface reconstructs spans through the
    /// original pair bridge; the state route uses
    /// [`Self::encode_state_with_source`].
    pub fn encode_with_source(&self, text: &str) -> Result<FastEncodeOutcome, EngineError> {
        let mut engine = self.locked_primary()?;
        self.full_encoding_with_source(text, &mut engine)
    }

    /// Exact full encode in the store's structure-of-arrays layout with
    /// the implementation identity for the router's state route.
    pub fn encode_state_with_source(
        &self,
        text: &str,
    ) -> Result<(SoaEncoding, FastEncodeSource), EngineError> {
        let mut engine = self.locked_primary()?;
        self.full_state_with_source(text, &mut engine)
    }

    /// ID-only counterpart of [`Self::encode_with_source`] for stateless
    /// callers.
    pub fn encode_ids_with_source(
        &self,
        text: &str,
    ) -> Result<(Vec<u32>, FastEncodeSource), EngineError> {
        let mut engine = self.locked_primary()?;
        self.full_ids_with_source(text, &mut engine)
    }

    /// State-seed encode through the single guarded gate: the Gigatoken
    /// row yields closure-verified IDs for lazy span adoption, and the
    /// reference rows yield materialized structure-of-arrays spans. The
    /// acceptance decision, fallback order, and path accounting are the
    /// gate's and cannot drift from the other surfaces.
    pub fn encode_seed_with_source(
        &self,
        text: &str,
    ) -> Result<(FastSeedPayload, FastEncodeSource), EngineError> {
        let mut engine = self.locked_primary()?;
        self.guarded_full_with(
            text,
            &mut engine,
            |reference, text| {
                reference
                    .encode_state_soa(text)
                    .map(FastSeedPayload::Reference)
            },
            |reference, text| {
                reference
                    .encode_state_soa_with_added_flag(text)
                    .map(|(encoding, added)| (FastSeedPayload::Reference(encoding), added))
            },
            |this, text, engine| {
                this.fast_ids_with(text, engine)
                    .map(FastSeedPayload::GigatokenIds)
            },
        )
    }

    /// Preserve row order while using persistent per-worker Gigatoken
    /// caches. Each Rayon job owns one worker mutex for its entire chunk, so
    /// concurrent public calls remain safe without sharing mutable caches.
    pub fn encode_batch_ids(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, EngineError> {
        self.encode_batch_ids_with_source(texts)
            .map(|rows| rows.into_iter().map(|(ids, _source)| ids).collect())
    }

    /// ID-only batch encodes with the implementation identity retained for
    /// the router's per-row fallback ledger; no row constructs spans.
    pub fn encode_batch_ids_with_source(
        &self,
        texts: &[&str],
    ) -> Result<Vec<(Vec<u32>, FastEncodeSource)>, EngineError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let core = self
            .core()
            .map_err(|error| EngineError(error.to_string()))?;
        let total_bytes = texts
            .iter()
            .fold(0usize, |total, text| total.saturating_add(text.len()));
        let workers = core.batch_workers(texts.len(), total_bytes)?;
        let chunks = workers.len();
        let chunk_size = texts.len().div_ceil(chunks);
        let rows = texts
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(worker_index, chunk)| {
                let mut worker = workers[worker_index]
                    .lock()
                    .map_err(|_| EngineError("Gigatoken worker mutex is poisoned".to_owned()))?;
                chunk
                    .iter()
                    .map(|text| self.full_ids_with_source(text, &mut worker))
                    .collect::<Result<Vec<_>, EngineError>>()
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        Ok(rows.into_iter().flatten().collect())
    }

    fn count(&self, path: &str, reason: Option<&str>, kept: usize, window: usize, retries: usize) {
        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *stats.path_counts.entry(path.to_owned()).or_default() += 1;
        stats.last_path = Some(path.to_owned());
        stats.last_reason = reason.map(str::to_owned);
        stats.last_kept_tokens = kept;
        stats.last_window_chars = window;
        stats.last_retries = retries;
    }

    fn reference_full(&self, text: &str, reason: &str) -> Result<Encoding, EngineError> {
        let encoded = self
            .reference
            .encode_core(text)
            .map_err(|error| EngineError(error.to_string()))?;
        self.count(&format!("hf_full_{reason}"), Some(reason), 0, 0, 0);
        Ok(encoded)
    }

    fn require_byte_identity(&self, text: &str) -> Result<(), FastCpuEngineError> {
        if self.spec.has_normalizer && !text.is_ascii() {
            let normalized = self
                .reference
                .normalize(text)
                .map_err(|error| FastCpuEngineError::new(error.to_string()))?;
            if normalized != text {
                return Err(FastCpuEngineError::new(
                    "the repair window is not invariant under the certified normalizer",
                ));
            }
        }
        Ok(())
    }

    fn fast_encoding(&self, text: &str) -> Result<Encoding, FastCpuEngineError> {
        let core = self.core()?;
        self.require_byte_identity(text)?;
        let mut engine = core
            .gigatoken
            .lock()
            .map_err(|_| FastCpuEngineError::new("Gigatoken cache mutex is poisoned"))?;
        self.fast_encoding_with(text, &mut engine)
    }

    fn fast_encoding_with(
        &self,
        text: &str,
        engine: &mut Gigatoken,
    ) -> Result<Encoding, FastCpuEngineError> {
        self.require_byte_identity(text)?;
        let mut ids = Vec::new();
        engine.encode_with_added_tokens_flat(text.as_bytes(), &mut ids);
        let spans = self
            .reference
            .spans_for_ids(text, &ids)
            .map_err(|error| FastCpuEngineError::new(error.to_string()))?;
        Ok(Encoding { ids, spans })
    }

    /// Structure-of-arrays counterpart of [`Self::fast_encoding_with`]:
    /// spans come from the one-pass converter and are adopted by the tail
    /// without pair formation.
    fn fast_state_encoding_with(
        &self,
        text: &str,
        engine: &mut Gigatoken,
    ) -> Result<SoaEncoding, FastCpuEngineError> {
        self.require_byte_identity(text)?;
        let mut ids = Vec::new();
        engine.encode_with_added_tokens_flat(text.as_bytes(), &mut ids);
        let (span_starts, span_ends) = self
            .reference
            .spans_soa_for_ids(text, &ids)
            .map_err(|error| FastCpuEngineError::new(error.to_string()))?;
        Ok(SoaEncoding {
            ids,
            span_starts,
            span_ends,
        })
    }

    /// ID-only counterpart of [`Self::fast_encoding_with`]. The
    /// allocation-free closure check keeps the span surfaces' fail-closed
    /// acceptance decision without materializing offsets.
    fn fast_ids_with(
        &self,
        text: &str,
        engine: &mut Gigatoken,
    ) -> Result<Vec<u32>, FastCpuEngineError> {
        self.require_byte_identity(text)?;
        let mut ids = Vec::new();
        engine.encode_with_added_tokens_flat(text.as_bytes(), &mut ids);
        self.reference
            .verify_ids_close(text, &ids)
            .map_err(|error| FastCpuEngineError::new(error.to_string()))?;
        Ok(ids)
    }

    fn window_encoding(&self, text: &str) -> Result<Encoding, FastCpuEngineError> {
        let encoded = self.fast_encoding(text)?;
        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        stats.window_calls += 1;
        stats.window_chars += text.chars().count() as u64;
        Ok(encoded)
    }

    /// Route one full encode through the added-token gate, the corrected
    /// Gigatoken engine, and the reference guard, generically over the
    /// result carrier. The pair-based, structure-of-arrays, and ID-only
    /// surfaces instantiate this single gate, so their acceptance
    /// decisions, fallback order, and path accounting cannot drift apart.
    fn guarded_full_with<T>(
        &self,
        text: &str,
        engine: &mut Gigatoken,
        reference_plain: impl Fn(&ReferenceEngine, &str) -> Result<T, ReferenceEngineError>,
        reference_with_flag: impl Fn(&ReferenceEngine, &str) -> Result<(T, bool), ReferenceEngineError>,
        gigatoken_encode: impl FnOnce(&Self, &str, &mut Gigatoken) -> Result<T, FastCpuEngineError>,
    ) -> Result<(T, FastEncodeSource), EngineError> {
        let mut candidate_reference = None;
        if self.added_gate.route_every_input {
            let encoded = reference_plain(&self.reference, text)
                .map_err(|error| EngineError(error.to_string()))?;
            self.count("hf_full_added_token", Some("added_token"), 0, 0, 0);
            return Ok((encoded, FastEncodeSource::ReferenceAddedToken));
        }
        if self.added_gate.may_match(text) {
            let (encoded, has_added) = reference_with_flag(&self.reference, text)
                .map_err(|error| EngineError(error.to_string()))?;
            if has_added {
                self.count("hf_full_added_token", Some("added_token"), 0, 0, 0);
                return Ok((encoded, FastEncodeSource::ReferenceAddedToken));
            }
            candidate_reference = Some(encoded);
        }
        match gigatoken_encode(self, text, engine) {
            Ok(encoded) => {
                self.count("gigatoken_full", None, 0, 0, 0);
                Ok((encoded, FastEncodeSource::Gigatoken))
            }
            Err(_) => {
                let encoded = match candidate_reference {
                    Some(encoded) => encoded,
                    None => reference_plain(&self.reference, text)
                        .map_err(|error| EngineError(error.to_string()))?,
                };
                self.count("hf_full_engine_guard", Some("engine_guard"), 0, 0, 0);
                Ok((encoded, FastEncodeSource::ReferenceEngineGuard))
            }
        }
    }

    fn full_encoding_with_source(
        &self,
        text: &str,
        engine: &mut Gigatoken,
    ) -> Result<FastEncodeOutcome, EngineError> {
        self.guarded_full_with(
            text,
            engine,
            |reference, text| reference.encode_core(text),
            |reference, text| reference.encode_core_with_added_flag(text),
            |this, text, engine| this.fast_encoding_with(text, engine),
        )
        .map(|(encoding, source)| FastEncodeOutcome { encoding, source })
    }

    fn full_state_with_source(
        &self,
        text: &str,
        engine: &mut Gigatoken,
    ) -> Result<(SoaEncoding, FastEncodeSource), EngineError> {
        self.guarded_full_with(
            text,
            engine,
            |reference, text| reference.encode_state_soa(text),
            |reference, text| reference.encode_state_soa_with_added_flag(text),
            |this, text, engine| this.fast_state_encoding_with(text, engine),
        )
    }

    fn full_ids_with_source(
        &self,
        text: &str,
        engine: &mut Gigatoken,
    ) -> Result<(Vec<u32>, FastEncodeSource), EngineError> {
        self.guarded_full_with(
            text,
            engine,
            |reference, text| reference.encode_ids(text, false),
            |reference, text| reference.encode_ids_with_added_flag(text),
            |this, text, engine| this.fast_ids_with(text, engine),
        )
    }

    /// Reconstruct character spans for an externally produced exact ID stream
    /// (for example the certified prebuilt GPU path).
    pub fn spans_for_ids(
        &self,
        text: &str,
        ids: &[u32],
    ) -> Result<Vec<(u32, u32)>, FastCpuEngineError> {
        self.require_byte_identity(text)?;
        self.reference
            .spans_for_ids(text, ids)
            .map_err(|error| FastCpuEngineError::new(error.to_string()))
    }

    fn accepted(&self, matched: Match, spans: &[(u32, u32)], text: &str) -> bool {
        if matched.length < self.spec.min_match_tokens {
            return false;
        }
        let covered = if self.spec.has_normalizer {
            let start = spans[matched.left_start].0 as usize;
            let end = spans[matched.left_start + matched.length - 1].1 as usize;
            let Some(slice) = slice_chars(text, start, end) else {
                return false;
            };
            let Ok(normalized) = self.reference.normalize(slice) else {
                return false;
            };
            normalized.chars().count()
        } else {
            matched.covered_chars
        };
        covered > self.spec.effective_l_max && self.has_sync_witness(matched, spans, text)
    }

    fn has_sync_witness(&self, matched: Match, spans: &[(u32, u32)], text: &str) -> bool {
        let chars = text.chars().collect::<Vec<_>>();
        let end = matched.left_start + matched.length.saturating_sub(1);
        for index in matched.left_start..end {
            let boundary = spans[index + 1].0 as usize;
            if boundary == 0 || boundary >= chars.len() {
                continue;
            }
            if self.boundary.accepts(chars[boundary - 1], chars[boundary]) {
                return true;
            }
        }
        false
    }

    fn repair_append(
        &self,
        tail: &mut TailState,
        delta: &str,
    ) -> Result<AppendReport, EngineError> {
        if delta.is_empty() {
            let kept = tail.n_tokens();
            self.count("gigatoken_repair_noop", None, kept, 0, 0);
            return Ok(AppendReport {
                path: "gigatoken_repair_noop".to_owned(),
                kept_tokens: kept,
            });
        }
        let mut full = String::with_capacity(tail.text_bytes() + delta.len());
        full.push_str(tail.text());
        full.push_str(delta);
        let previous_chars = tail.text_chars() as usize;
        let prior_spans = tail.spans();
        if tail.ids().len() != prior_spans.len() {
            let encoded = self.reference_full(&full, "invalid_prior_state")?;
            tail.fill(&full, encoded)
                .map_err(|error| EngineError(error.to_string()))?;
            return Ok(AppendReport {
                path: "hf_full_invalid_prior_state".to_owned(),
                kept_tokens: 0,
            });
        }

        let mut window = self.spec.window_chars;
        let mut retries = 0usize;
        while window < previous_chars {
            let window_start = previous_chars - window;
            let Some(window_text) = slice_chars(&full, window_start, full.chars().count()) else {
                break;
            };
            match self.window_encoding(window_text) {
                Ok(encoded) => {
                    let overlap_start = window_start + self.spec.margin;
                    let left =
                        build_records(tail.ids(), &prior_spans, 0, overlap_start, previous_chars)?;
                    let right = build_records(
                        &encoded.ids,
                        &encoded.spans,
                        window_start,
                        overlap_start,
                        previous_chars,
                    )?;
                    if let Some(matched) = find_match(&left, &right) {
                        if self.accepted(matched, &prior_spans, &full) {
                            let left_end = matched.left_start + matched.length;
                            let right_end = matched.right_start + matched.length;
                            let mut ids = tail.ids()[..left_end].to_vec();
                            ids.extend_from_slice(&encoded.ids[right_end..]);
                            let mut spans = prior_spans[..left_end].to_vec();
                            spans.extend(encoded.spans[right_end..].iter().map(|&(start, end)| {
                                (start + window_start as u32, end + window_start as u32)
                            }));
                            tail.fill(&full, Encoding { ids, spans })
                                .map_err(|error| EngineError(error.to_string()))?;
                            self.count("gigatoken_repair", None, left_end, window, retries);
                            return Ok(AppendReport {
                                path: "gigatoken_repair".to_owned(),
                                kept_tokens: left_end,
                            });
                        }
                    }
                }
                Err(_) => {
                    let encoded = self.reference_full(&full, "engine_guard")?;
                    tail.fill(&full, encoded)
                        .map_err(|error| EngineError(error.to_string()))?;
                    return Ok(AppendReport {
                        path: "hf_full_engine_guard".to_owned(),
                        kept_tokens: 0,
                    });
                }
            }
            if retries >= self.spec.max_retries {
                let encoded = self.reference_full(&full, "no_safe_cut")?;
                tail.fill(&full, encoded)
                    .map_err(|error| EngineError(error.to_string()))?;
                return Ok(AppendReport {
                    path: "hf_full_no_safe_cut".to_owned(),
                    kept_tokens: 0,
                });
            }
            retries += 1;
            window = window.saturating_mul(2);
        }
        let encoded = self.reference_full(&full, "window_covers_all")?;
        tail.fill(&full, encoded)
            .map_err(|error| EngineError(error.to_string()))?;
        Ok(AppendReport {
            path: "hf_full_window_covers_all".to_owned(),
            kept_tokens: 0,
        })
    }
}

impl SessionEncoder for FastCpuEngine {
    fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
        self.encode_state_with_source(text)
            .map(|(encoding, _source)| encoding.into_pairs())
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        if tail.text().is_empty() {
            let (encoded, _source) = self.encode_state_with_source(delta)?;
            tail.fill_soa(delta, encoded)
                .map_err(|error| EngineError(error.to_string()))?;
            return Ok(AppendReport {
                path: "cold_full".to_owned(),
                kept_tokens: 0,
            });
        }
        self.repair_append(tail, delta)
    }

    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        if !tail.spans_materialized() {
            // Lazy checkpointed tail: search over windows rebuilt on
            // demand so a cut near the tail end never materializes the
            // whole span row. Values equal the materialized search
            // element for element (windowed-vs-full property tests).
            return self
                .boundary
                .last_boundary_windowed(
                    tail.text(),
                    tail.text_chars(),
                    tail.n_tokens(),
                    floor_char,
                    ceil_char,
                    toktier_store_core::SPAN_CHECKPOINT_STRIDE,
                    |lo, hi| tail.span_window(lo, hi),
                )
                .map(|found| {
                    found.map(|cut| BoundaryCut {
                        cut_tokens: cut.cut_tokens,
                        cut_char: cut.cut_char,
                    })
                })
                .map_err(|error| EngineError(error.to_string()));
        }
        Ok(self
            .boundary
            .last_boundary(
                tail.text(),
                tail.text_chars(),
                tail.span_starts(),
                tail.span_ends(),
                floor_char,
                ceil_char,
            )
            .map(|cut| BoundaryCut {
                cut_tokens: cut.cut_tokens,
                cut_char: cut.cut_char,
            }))
    }

    fn witness_category(&self) -> WitnessCategory {
        WitnessCategory::BpeSyncTransition
    }
}

fn verified_byte_lengths(
    engine: &Gigatoken,
    expected_size: usize,
) -> Result<Vec<usize>, FastCpuEngineError> {
    if engine.vocab_size() != expected_size {
        return Err(FastCpuEngineError::new(format!(
            "vocabulary sizes differ: HF={expected_size}, Gigatoken={}",
            engine.vocab_size()
        )));
    }
    let mut lengths = vec![0usize; expected_size];
    let mut seen = vec![false; expected_size];
    for (raw_id, token_bytes) in engine.vocab_entries() {
        let id = raw_id as usize;
        if id >= expected_size {
            return Err(FastCpuEngineError::new(format!(
                "Gigatoken vocabulary id {id} is invalid"
            )));
        }
        lengths[id] = token_bytes.len();
        seen[id] = true;
    }
    if let Some(missing) = seen.iter().position(|present| !present) {
        return Err(FastCpuEngineError::new(format!(
            "Gigatoken has no raw vocabulary entry for id {missing}"
        )));
    }
    Ok(lengths)
}

fn build_records(
    ids: &[u32],
    spans: &[(u32, u32)],
    base: usize,
    overlap_start: usize,
    overlap_end: usize,
) -> Result<Vec<Record>, EngineError> {
    if ids.len() != spans.len() {
        return Err(EngineError(format!(
            "ids/spans length mismatch: {} != {}",
            ids.len(),
            spans.len()
        )));
    }
    let mut records = Vec::new();
    let mut ordinals = HashMap::<(usize, usize), usize>::new();
    for (index, (&id, &(local_start, local_end))) in ids.iter().zip(spans).enumerate() {
        let start = base + local_start as usize;
        let end = base + local_end as usize;
        if start == end || start < overlap_start || end > overlap_end {
            continue;
        }
        let ordinal = ordinals.entry((start, end)).or_default();
        records.push(Record {
            original_index: index,
            global_start: start,
            global_end: end,
            token_id: id,
            same_span_ordinal: *ordinal,
        });
        *ordinal += 1;
    }
    Ok(records)
}

fn record_key(record: Record) -> (usize, usize, usize, u32) {
    (
        record.global_start,
        record.global_end,
        record.same_span_ordinal,
        record.token_id,
    )
}

fn find_match(left: &[Record], right: &[Record]) -> Option<Match> {
    let mut right_index = HashMap::<(usize, usize, usize, u32), Vec<usize>>::new();
    for (index, &record) in right.iter().enumerate() {
        right_index
            .entry(record_key(record))
            .or_default()
            .push(index);
    }
    let mut best = None;
    let mut covered = HashSet::new();
    for (left_index, &left_record) in left.iter().enumerate() {
        let Some(right_positions) = right_index.get(&record_key(left_record)) else {
            continue;
        };
        for &right_index_value in right_positions {
            if covered.contains(&(left_index, right_index_value)) {
                continue;
            }
            let right_record = right[right_index_value];
            let mut length = 1usize;
            while left_index + length < left.len()
                && right_index_value + length < right.len()
                && left[left_index + length].original_index == left_record.original_index + length
                && right[right_index_value + length].original_index
                    == right_record.original_index + length
                && record_key(left[left_index + length])
                    == record_key(right[right_index_value + length])
            {
                covered.insert((left_index + length, right_index_value + length));
                length += 1;
            }
            let candidate = Match {
                left_start: left_record.original_index,
                right_start: right_record.original_index,
                length,
                covered_chars: left[left_index + length - 1].global_end - left_record.global_start,
            };
            if best.is_none_or(|current: Match| {
                (candidate.length, candidate.covered_chars)
                    > (current.length, current.covered_chars)
            }) {
                best = Some(candidate);
            }
        }
    }
    best
}

fn slice_chars(text: &str, start: usize, end: usize) -> Option<&str> {
    if start > end {
        return None;
    }
    let total = text.chars().count();
    if end > total {
        return None;
    }
    if text.is_ascii() {
        return text.get(start..end);
    }
    let start_byte = if start == total {
        text.len()
    } else {
        text.char_indices().nth(start)?.0
    };
    let end_byte = if end == total {
        text.len()
    } else {
        text.char_indices().nth(end)?.0
    };
    text.get(start_byte..end_byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_match_preserves_same_span_ordinals() {
        let ids = vec![1, 2, 3, 4];
        let spans = vec![(0, 1), (1, 2), (1, 2), (2, 3)];
        let left = build_records(&ids, &spans, 0, 0, 3).unwrap();
        let right = build_records(&ids, &spans, 0, 0, 3).unwrap();
        assert_eq!(find_match(&left, &right).unwrap().length, 4);
    }

    #[test]
    fn char_slices_are_unicode_scalar_indexed() {
        assert_eq!(slice_chars("a你🙂z", 1, 3), Some("你🙂"));
        assert_eq!(slice_chars("ascii", 1, 4), Some("sci"));
    }
}
