//! One-call native routing, fallback accounting, and store-engine adapter.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use aho_corasick::{AhoCorasick, MatchKind};
use serde_json::{json, Value};
use toktier_store_core::{
    AppendReport, BoundaryCut, Encoding, EngineError, SessionEncoder, SharedIds, SoaEncoding,
    TailState, WitnessCategory,
};

use crate::fast_cpu::SegmentFacts;
use crate::reference::{LiteralPiece, LiteralSegmenter};
use crate::{
    ChunkEncoder, ChunkPlan, ChunkedIds, FastCpuEngine, FastEncodeSource, FastSeedPayload,
    ReferenceEngine, R_CHUNK_ROUTE_NOT_FAST_CPU,
};

pub const BACKEND_GPU: &str = "gpu";
pub const BACKEND_FAST_CPU: &str = "fast_cpu";
pub const BACKEND_REFERENCE: &str = "hf";

pub const R_INPUT_ADDED_TOKEN: &str = "R_INPUT_ADDED_TOKEN";
pub const R_INPUT_BELOW_GPU_THRESHOLD: &str = "R_INPUT_BELOW_GPU_THRESHOLD";
pub const R_INPUT_GUARD_ROUTED: &str = "R_INPUT_GUARD_ROUTED";
pub const R_EXEC_FAULT: &str = "R_EXEC_FAULT";
pub const R_INPUT_POSTPROCESS_ROUTED: &str = "R_INPUT_POSTPROCESS_ROUTED";

const STATE_ENCODE_RECONSTRUCTED_SPANS: &str = "accelerated_with_reconstructed_spans";
const STATE_ENCODE_LAZY_SPAN_CHECKPOINTS: &str = "accelerated_with_lazy_span_checkpoints";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendKind {
    Gpu,
    FastCpu,
    Reference,
}

impl BackendKind {
    fn parse(value: &str) -> Result<Self, NativeRuntimeError> {
        match value {
            BACKEND_GPU => Ok(Self::Gpu),
            BACKEND_FAST_CPU => Ok(Self::FastCpu),
            BACKEND_REFERENCE => Ok(Self::Reference),
            other => Err(NativeRuntimeError::new(format!(
                "unknown backend in native route: {other:?}"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => BACKEND_GPU,
            Self::FastCpu => BACKEND_FAST_CPU,
            Self::Reference => BACKEND_REFERENCE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRuntimeError(String);

impl NativeRuntimeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for NativeRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for NativeRuntimeError {}

/// Raw prebuilt-GPU request surface. Implementations own contexts, streams,
/// tables, and the verified module; no Python callback participates.
pub trait NativeGpuEngine: Send + Sync {
    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, NativeRuntimeError>;
    fn encode_batch_ids(&self, texts: &[&str]) -> Vec<Result<Vec<u32>, NativeRuntimeError>> {
        texts.iter().map(|text| self.encode_ids(text)).collect()
    }
    fn delivery(&self) -> &str;
}

/// One-shot constructor for a deferred native GPU engine.
///
/// The opener owns every input the engine needs -- configuration and host
/// copies of the device tables -- so running it performs no callback into an
/// embedding interpreter and assumes no held interpreter lock. The router
/// runs it at most once, on the first request thread that actually routes to
/// the GPU backend (a thread that has already released the GIL when the
/// router is driven through the Python bindings).
pub type NativeGpuOpener =
    Box<dyn FnOnce() -> Result<Arc<dyn NativeGpuEngine>, NativeRuntimeError> + Send + Sync>;

/// How a router receives its GPU engine.
///
/// `Open` adopts an engine whose device resources already exist (tests and
/// embedders that manage the device lifecycle themselves). `Deferred`
/// carries the opener and leaves the device untouched until the first
/// request the router actually routes to the GPU backend.
pub enum NativeGpuSource {
    Open(Arc<dyn NativeGpuEngine>),
    Deferred(NativeGpuOpener),
}

/// Lazily opened GPU engine slot.
///
/// The first open outcome -- success or failure -- is fixed for the slot's
/// lifetime: concurrent first requests block on one initialization and
/// share its engine, and a failed open is latched so later GPU-routed
/// requests take the ordered fallback chain instead of retrying the device.
struct GpuCell {
    opener: Mutex<Option<NativeGpuOpener>>,
    engine: OnceLock<Result<Arc<dyn NativeGpuEngine>, NativeRuntimeError>>,
}

impl GpuCell {
    fn new(source: NativeGpuSource) -> Self {
        let cell = Self {
            opener: Mutex::new(None),
            engine: OnceLock::new(),
        };
        match source {
            NativeGpuSource::Open(engine) => {
                let _ = cell.engine.set(Ok(engine));
            }
            NativeGpuSource::Deferred(opener) => {
                *cell
                    .opener
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(opener);
            }
        }
        cell
    }

    /// The engine, opening it on first use.
    fn engine(&self) -> Result<Arc<dyn NativeGpuEngine>, NativeRuntimeError> {
        self.engine
            .get_or_init(|| {
                let opener = self
                    .opener
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                match opener {
                    Some(open) => open(),
                    // Reachable only if a previous open panicked between
                    // consuming the opener and publishing an outcome; the
                    // request then falls back like any other open failure.
                    None => Err(NativeRuntimeError::new(
                        "the deferred native GPU opener was already consumed",
                    )),
                }
            })
            .clone()
    }

    /// Whether a routed request has already opened the engine.
    /// Reads the once-cell only; it never triggers an open.
    fn opened(&self) -> bool {
        matches!(self.engine.get(), Some(Ok(_)))
    }

    /// The latched message of a failed open, if one happened.
    fn open_error(&self) -> Option<String> {
        match self.engine.get() {
            Some(Err(error)) => Some(error.to_string()),
            _ => None,
        }
    }
}

/// Bounded seed-overlap runner on the process-wide Rayon pool -- the same
/// bounded pool the batch encode path already shares (PLAN/162 WP5/WP6).
///
/// `rayon::in_place_scope` keeps the foreground encode on the calling
/// thread and queues only the digest scan to the pool, so concurrent
/// requests can never occupy more host threads than their own callers
/// plus the fixed pool; no thread is spawned per request. With a
/// one-thread pool a single request still overlaps (caller plus one
/// worker), and under saturation queued digest scans simply drain in
/// pool order, so total work matches the serial path.
#[derive(Debug, Default, Clone, Copy)]
pub struct RayonSeedOverlap;

impl toktier_store_core::OverlapRunner for RayonSeedOverlap {
    fn run_joined(&self, background: &mut (dyn FnMut() + Send), foreground: &mut dyn FnMut()) {
        rayon::in_place_scope(|scope| {
            scope.spawn(|_| background());
            foreground();
        });
    }

    fn worker_count(&self) -> usize {
        rayon::current_num_threads()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeEvent {
    pub code: String,
    pub backend: String,
    pub target: String,
    pub detail: Value,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeStats {
    pub fallback_counts: BTreeMap<String, u64>,
    pub execution_counts: BTreeMap<String, u64>,
    pub events: Vec<RuntimeEvent>,
    pub last_execution: Option<Value>,
    pub state_encode_counts: BTreeMap<String, u64>,
    pub last_state_encode: Option<Value>,
}

#[derive(Debug)]
struct AddedGate {
    matcher: Option<AhoCorasick>,
    route_every_input: bool,
}

impl AddedGate {
    fn new(reference: &ReferenceEngine) -> Result<Self, NativeRuntimeError> {
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
                        NativeRuntimeError::new(format!(
                            "failed to build native added-token gate: {error}"
                        ))
                    })?,
            )
        };
        Ok(Self {
            matcher,
            route_every_input,
        })
    }

    fn candidate(&self, text: &str) -> bool {
        self.route_every_input
            || self
                .matcher
                .as_ref()
                .is_some_and(|matcher| matcher.is_match(text))
    }

    /// Run-time closure check over a plan, with the gate's own matcher.
    fn text_pieces_closed(&self, text: &str, pieces: &[LiteralPiece]) -> bool {
        pieces.iter().all(|piece| match *piece {
            LiteralPiece::Literal { .. } => true,
            LiteralPiece::Text { start, end } => !self.candidate(&text[start..end]),
        })
    }
}

/// The segment work of one GPU-headed, literal-bearing document.
enum GpuCompose {
    /// Every text segment was encoded; at least one ran on the GPU.
    Ids {
        ids: Vec<u32>,
        facts: SegmentFacts,
        delivery: String,
        reason: Option<&'static str>,
    },
    /// The whole document goes to the reference because of its literals.
    AddedTokenReference,
    /// The engine guard refused a segment (its event is recorded): the
    /// whole document goes to the reference.
    EngineGuard {
        path: &'static str,
        code: &'static str,
    },
    /// Continue the ordered chain from the backend after the GPU.
    Continue(Option<&'static str>),
}

/// What the GPU-headed segmented route decided for one document.
enum SegmentedGpu {
    /// The document is answered; the caller records its execution.
    Routed(RoutedIds, Option<SegmentFacts>),
    /// Continue the ordered chain for the whole document from the backend
    /// after the GPU, carrying this routing reason.
    Continue(Option<&'static str>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedIds {
    pub ids: Vec<u32>,
    pub backend: String,
    pub source: Option<String>,
    pub path: Option<String>,
    /// The routing decision that put this input on this backend, as the
    /// `R_*` code the ledger recorded for it, or `None` when the first
    /// admitted backend ran it and there was nothing to record. This is
    /// the recorded code itself rather than a second reading of `path`,
    /// so the two cannot come to disagree.
    pub reason: Option<&'static str>,
}

/// State-route payload: the session state either adopts materialized
/// structure-of-arrays spans, or -- on the accelerated seed rows -- the
/// closure-verified ID row itself plus the frozen byte-length table for
/// lazy checkpointed spans. Ownership moves with the payload; the state
/// route never clones a full ID row for bookkeeping.
enum StatePayload {
    Soa(SoaEncoding),
    Lazy {
        ids: Vec<u32>,
        table: std::sync::Arc<[usize]>,
    },
}

impl StatePayload {
    fn accelerated_state_encode_path(&self) -> &'static str {
        match self {
            Self::Soa(_) => STATE_ENCODE_RECONSTRUCTED_SPANS,
            Self::Lazy { .. } => STATE_ENCODE_LAZY_SPAN_CHECKPOINTS,
        }
    }
}

/// How the state route may deliver its payload.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StateMode {
    /// Materialized spans (the retained pair/SoA surfaces).
    Materialized,
    /// Lazy checkpointed spans on the accelerated rows (session seeds).
    LazySeed,
}

/// Immutable native request router. It also implements `SessionEncoder`: a
/// cold store seed follows the normal full-encode route, while subsequent
/// appends stay on corrected Gigatoken repair (or exact HF when no certified
/// repair engine exists).
pub struct NativeRouter {
    chain: Box<[BackendKind]>,
    thresholds: Box<[u64]>,
    reference: Arc<ReferenceEngine>,
    fast_cpu: Option<Arc<FastCpuEngine>>,
    repair_fast_cpu: bool,
    gpu: Option<GpuCell>,
    added: AddedGate,
    postprocessor_adds_tokens: bool,
    diagnostics: bool,
    stats: Mutex<RuntimeStats>,
}

impl fmt::Debug for NativeRouter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeRouter")
            .field(
                "chain",
                &self
                    .chain
                    .iter()
                    .map(|backend| backend.as_str())
                    .collect::<Vec<_>>(),
            )
            .field("has_fast_cpu", &self.fast_cpu.is_some())
            .field("has_gpu", &self.gpu.is_some())
            .finish_non_exhaustive()
    }
}

impl NativeRouter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chain: Vec<String>,
        thresholds: Vec<u64>,
        reference: Arc<ReferenceEngine>,
        fast_cpu: Option<Arc<FastCpuEngine>>,
        repair_fast_cpu: bool,
        gpu: Option<NativeGpuSource>,
        postprocessor_adds_tokens: bool,
        diagnostics: bool,
    ) -> Result<Self, NativeRuntimeError> {
        if chain.is_empty() || chain.len() != thresholds.len() {
            return Err(NativeRuntimeError::new(
                "native route chain and thresholds must be non-empty and equal-length",
            ));
        }
        let chain = chain
            .iter()
            .map(|backend| BackendKind::parse(backend))
            .collect::<Result<Vec<_>, _>>()?;
        if chain.last() != Some(&BackendKind::Reference) {
            return Err(NativeRuntimeError::new(
                "native fallback chain must end with hf",
            ));
        }
        let mut seen = HashMap::new();
        for backend in &chain {
            if seen.insert(*backend as u8, ()).is_some() {
                return Err(NativeRuntimeError::new(
                    "native fallback chain repeats a backend",
                ));
            }
            match backend {
                BackendKind::Gpu if gpu.is_none() => {
                    return Err(NativeRuntimeError::new(
                        "native route names gpu but no native GPU engine source was provided",
                    ));
                }
                BackendKind::FastCpu if fast_cpu.is_none() => {
                    return Err(NativeRuntimeError::new(
                        "native route names fast_cpu but no native CPU engine was constructed",
                    ));
                }
                _ => {}
            }
        }
        if thresholds.last().copied() != Some(0) {
            return Err(NativeRuntimeError::new(
                "the native hf fallback threshold must be zero",
            ));
        }
        let added = AddedGate::new(&reference)?;
        if chain
            .iter()
            .any(|backend| *backend != BackendKind::Reference)
        {
            // Build the literal class, with its segmenter self-check, at
            // construction rather than inside the first request.
            let _ = reference.literal_class_reason();
        }
        // Session repair runs the engine core's one-pass
        // added-token matcher over the repair window; on a face where that
        // matcher could disagree with the reference's two passes, repair
        // is disabled and appends re-encode through the reference.
        let repair_fast_cpu = repair_fast_cpu
            && fast_cpu
                .as_ref()
                .is_none_or(|engine| engine.reference().engine_matcher_agrees());
        if gpu.is_some() {
            // The GPU state route reconstructs spans from the frozen raw
            // byte-length table; build and freeze it during construction so
            // the first stateful seed does not silently pay (or hide) table
            // initialization. A failure is recorded by the once-cell and
            // replays at first use, which keeps the existing span-guard
            // fallback semantics.
            let _ = reference.prewarm_raw_byte_lengths();
        }
        Ok(Self {
            chain: chain.into_boxed_slice(),
            thresholds: thresholds.into_boxed_slice(),
            reference,
            fast_cpu,
            repair_fast_cpu,
            gpu: gpu.map(GpuCell::new),
            added,
            postprocessor_adds_tokens,
            diagnostics,
            stats: Mutex::new(RuntimeStats::default()),
        })
    }

    pub fn stats(&self) -> RuntimeStats {
        self.stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn chain(&self) -> Vec<&'static str> {
        self.chain.iter().map(|backend| backend.as_str()).collect()
    }

    /// Whether a routed request has already opened the GPU engine.
    /// Reads the once-cell only; it never triggers an open.
    pub fn gpu_engine_opened(&self) -> bool {
        self.gpu.as_ref().is_some_and(GpuCell::opened)
    }

    /// The latched message of a failed deferred GPU open, if one happened.
    pub fn gpu_engine_open_error(&self) -> Option<String> {
        self.gpu.as_ref().and_then(GpuCell::open_error)
    }

    fn start_index(&self, input_bytes: u64) -> usize {
        self.thresholds
            .iter()
            .position(|threshold| input_bytes >= *threshold)
            .unwrap_or(self.chain.len() - 1)
    }

    fn record_event(&self, code: &str, backend: &str, target: &str, detail: Value) {
        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *stats.fallback_counts.entry(code.to_owned()).or_default() += 1;
        if self.diagnostics {
            stats.events.push(RuntimeEvent {
                code: code.to_owned(),
                backend: backend.to_owned(),
                target: target.to_owned(),
                detail,
            });
        }
    }

    fn record_execution(
        &self,
        backend: &str,
        input_bytes: u64,
        selected_start: usize,
        source: Option<&str>,
        path: Option<&str>,
    ) {
        self.record_execution_from(
            backend,
            input_bytes,
            self.chain[selected_start].as_str(),
            source,
            path,
        );
    }

    fn record_execution_from(
        &self,
        backend: &str,
        input_bytes: u64,
        selected_start: &str,
        source: Option<&str>,
        path: Option<&str>,
    ) {
        self.record_execution_detail(backend, input_bytes, selected_start, source, path, None);
    }

    /// One routed result, with the segment facts of a literal-segmented
    /// document under `segments` (absent for every other document, so the
    /// record of a literal-free input keeps its earlier shape).
    fn record_execution_detail(
        &self,
        backend: &str,
        input_bytes: u64,
        selected_start: &str,
        source: Option<&str>,
        path: Option<&str>,
        segments: Option<SegmentFacts>,
    ) {
        let mut last = json!({
            "input_bytes": input_bytes,
            "selected_start": selected_start,
            "executed_backend": backend,
        });
        if let Some(source) = source {
            last["source"] = Value::String(source.to_owned());
        }
        if let Some(path) = path {
            last["path"] = Value::String(path.to_owned());
        }
        if let Some(segments) = segments {
            last["segments"] = segments.to_json();
        }
        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *stats
            .execution_counts
            .entry(backend.to_owned())
            .or_default() += 1;
        stats.last_execution = Some(last);
    }

    fn record_state_encode(&self, path: &str, detail: Value) {
        let mut stats = self
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *stats
            .state_encode_counts
            .entry(path.to_owned())
            .or_default() += 1;
        stats.last_state_encode = Some(detail);
    }

    fn exact_added_reference_ids(&self, text: &str) -> Result<Option<Vec<u32>>, EngineError> {
        if !self.added.candidate(text) {
            return Ok(None);
        }
        let (ids, has_added) = self
            .reference
            .encode_ids_with_added_flag(text)
            .map_err(|error| EngineError(error.to_string()))?;
        // Outside the served class a gate match is answered
        // by the reference whether or not its row holds an added id.
        Ok((has_added || !self.reference.literal_segmentation_served()).then_some(ids))
    }

    fn exact_added_reference_state(&self, text: &str) -> Result<Option<SoaEncoding>, EngineError> {
        if !self.added.candidate(text) {
            return Ok(None);
        }
        let (encoding, has_added) = self
            .reference
            .encode_state_soa_with_added_flag(text)
            .map_err(|error| EngineError(error.to_string()))?;
        // Outside the served class every gate match is answered by the
        // reference, as on the ID route.
        Ok((has_added || !self.reference.literal_segmentation_served()).then_some(encoding))
    }

    fn reference_state(&self, text: &str) -> Result<SoaEncoding, EngineError> {
        self.reference
            .encode_state_soa(text)
            .map_err(|error| EngineError(error.to_string()))
    }

    /// Records the crossover decision and returns the code it recorded,
    /// so a caller can carry it as this input's routing reason.
    fn below_gpu_threshold_event(&self, input_bytes: u64, start: usize) -> Option<&'static str> {
        if self.chain.first() == Some(&BackendKind::Gpu) && start > 0 {
            self.record_event(
                R_INPUT_BELOW_GPU_THRESHOLD,
                BACKEND_GPU,
                self.chain[start].as_str(),
                json!({
                    "input_bytes": input_bytes,
                    "threshold_bytes": self.thresholds[0],
                }),
            );
            return Some(R_INPUT_BELOW_GPU_THRESHOLD);
        }
        None
    }

    fn gpu_fault_event(&self, index: usize, error: &NativeRuntimeError) {
        self.record_event(
            R_EXEC_FAULT,
            BACKEND_GPU,
            self.chain[index + 1].as_str(),
            json!({
                "error": "NativeGpuError",
                "message": error.to_string(),
                "scope": "input",
            }),
        );
    }

    /// A deferred engine open that failed. The outcome is latched by the
    /// once-cell, so every later GPU-routed request records this event and
    /// takes the ordered fallback chain instead of retrying the device.
    fn gpu_open_fault_event(&self, index: usize, error: &NativeRuntimeError) {
        self.record_event(
            R_EXEC_FAULT,
            BACKEND_GPU,
            self.chain[index + 1].as_str(),
            json!({
                "error": "NativeGpuOpenError",
                "message": error.to_string(),
                "scope": "engine",
                "stage": "engine_open",
            }),
        );
    }

    fn fast_cpu_fault_event(&self, index: usize, error: &EngineError) {
        self.record_event(
            R_EXEC_FAULT,
            BACKEND_FAST_CPU,
            self.chain[index + 1].as_str(),
            json!({
                "error": "FastCpuEngineError",
                "message": error.to_string(),
                "scope": "input",
            }),
        );
    }

    /// The routed-to-reference ledger facts for a fast-CPU outcome that the
    /// reference engine actually produced.
    fn fast_cpu_reference_path(&self, source: FastEncodeSource) -> (&'static str, &'static str) {
        let (code, path, detail) = match source {
            FastEncodeSource::ReferenceAddedToken => {
                (R_INPUT_ADDED_TOKEN, "hf_added_token", json!({}))
            }
            FastEncodeSource::ReferenceEngineGuard => (
                R_INPUT_GUARD_ROUTED,
                "hf_engine_guard",
                json!({"stage": "engine_guard"}),
            ),
            FastEncodeSource::Gigatoken => unreachable!(),
        };
        self.record_event(code, BACKEND_FAST_CPU, BACKEND_REFERENCE, detail);
        (path, code)
    }

    /// Stateless ID route: no span array, byte-to-character map, or pair
    /// vector is constructed anywhere on this path. Ledger events, counters,
    /// and diagnostic shapes are identical to the historical combined route
    /// with `state_encode=false` for every input that holds no added-token
    /// literal; a literal-bearing input of a served face is segmented
    /// and its execution record names the segments.
    fn route_ids_internal(&self, text: &str) -> Result<RoutedIds, EngineError> {
        let (routed, start, segments) = self.route_ids_unrecorded(text)?;
        self.record_execution_detail(
            &routed.backend,
            text.len() as u64,
            self.chain[start].as_str(),
            routed.source.as_deref(),
            routed.path.as_deref(),
            segments,
        );
        Ok(routed)
    }

    /// The literal segmenter when this router serves the face.
    fn served_segmenter(&self) -> Option<&LiteralSegmenter> {
        self.reference.literal_segmenter()
    }

    fn added_token_reference(&self, text: &str, start: usize) -> Result<RoutedIds, EngineError> {
        let ids = self
            .reference
            .encode_ids(text, false)
            .map_err(|error| EngineError(error.to_string()))?;
        self.record_event(
            R_INPUT_ADDED_TOKEN,
            self.chain[start].as_str(),
            BACKEND_REFERENCE,
            json!({}),
        );
        Ok(RoutedIds {
            ids,
            backend: BACKEND_REFERENCE.to_owned(),
            source: None,
            path: Some("hf_added_token".to_owned()),
            reason: Some(R_INPUT_ADDED_TOKEN),
        })
    }

    /// The ID route without its final execution record: the routed result,
    /// the chain index the input started at, and the segment facts of a
    /// literal-segmented result. Every event is recorded here, in order.
    fn route_ids_unrecorded(
        &self,
        text: &str,
    ) -> Result<(RoutedIds, usize, Option<SegmentFacts>), EngineError> {
        let input_bytes = text.len() as u64;
        let start = self.start_index(input_bytes);
        let mut reason = self.below_gpu_threshold_event(input_bytes, start);
        let mut first = start;
        if self.chain[start] != BackendKind::Reference {
            if let Some(segmenter) = self.served_segmenter() {
                if self.chain[start] == BackendKind::Gpu && self.added.candidate(text) {
                    match self.route_segmented_gpu(text, start, segmenter)? {
                        SegmentedGpu::Routed(routed, segments) => {
                            return Ok((routed, start, segments));
                        }
                        SegmentedGpu::Continue(carried) => {
                            reason = carried;
                            first = start + 1;
                        }
                    }
                }
                // A fast-CPU start segments inside the engine.
            } else if let Some(ids) = self.exact_added_reference_ids(text)? {
                self.record_event(
                    R_INPUT_ADDED_TOKEN,
                    self.chain[start].as_str(),
                    BACKEND_REFERENCE,
                    json!({}),
                );
                return Ok((
                    RoutedIds {
                        ids,
                        backend: BACKEND_REFERENCE.to_owned(),
                        source: None,
                        path: Some("hf_added_token".to_owned()),
                        reason: Some(R_INPUT_ADDED_TOKEN),
                    },
                    start,
                    None,
                ));
            }
        }
        for index in first..self.chain.len() {
            match self.chain[index] {
                BackendKind::Gpu => {
                    let cell = self.gpu.as_ref().expect("constructor gate");
                    match cell.engine() {
                        Err(error) => {
                            self.gpu_open_fault_event(index, &error);
                            reason = Some(R_EXEC_FAULT);
                        }
                        Ok(gpu) => match gpu.encode_ids(text) {
                            Ok(ids) => {
                                return Ok((
                                    RoutedIds {
                                        ids,
                                        backend: BACKEND_GPU.to_owned(),
                                        source: Some(gpu.delivery().to_owned()),
                                        path: Some("gpu_full".to_owned()),
                                        reason,
                                    },
                                    start,
                                    None,
                                ));
                            }
                            Err(error) => {
                                self.gpu_fault_event(index, &error);
                                reason = Some(R_EXEC_FAULT);
                            }
                        },
                    }
                }
                BackendKind::FastCpu => {
                    let fast = self.fast_cpu.as_ref().expect("constructor gate");
                    match fast.encode_ids_with_segments(text) {
                        Err(error) => {
                            self.fast_cpu_fault_event(index, &error);
                            reason = Some(R_EXEC_FAULT);
                        }
                        Ok((ids, FastEncodeSource::Gigatoken, segments)) => {
                            let path = if segments.is_some() {
                                "gigatoken_segmented"
                            } else {
                                "gigatoken_full"
                            };
                            return Ok((
                                RoutedIds {
                                    ids,
                                    backend: BACKEND_FAST_CPU.to_owned(),
                                    source: Some("gigatoken".to_owned()),
                                    path: Some(path.to_owned()),
                                    reason,
                                },
                                start,
                                segments,
                            ));
                        }
                        Ok((ids, source, _segments)) => {
                            let (path, code) = self.fast_cpu_reference_path(source);
                            return Ok((
                                RoutedIds {
                                    ids,
                                    backend: BACKEND_REFERENCE.to_owned(),
                                    source: None,
                                    path: Some(path.to_owned()),
                                    reason: Some(code),
                                },
                                start,
                                None,
                            ));
                        }
                    }
                }
                BackendKind::Reference => {
                    let ids = self
                        .reference
                        .encode_ids(text, false)
                        .map_err(|error| EngineError(error.to_string()))?;
                    return Ok((
                        RoutedIds {
                            ids,
                            backend: BACKEND_REFERENCE.to_owned(),
                            source: None,
                            path: Some("hf_full".to_owned()),
                            reason,
                        },
                        start,
                        None,
                    ));
                }
            }
        }
        Err(EngineError(
            "native fallback chain terminated without hf".to_owned(),
        ))
    }

    /// A literal-bearing document of a served face whose route starts on
    /// the GPU. Each text segment is routed as a document
    /// of its own size: segments at or above the crossover run on the GPU
    /// in one batch call, the others (and any segment the device faults on)
    /// on the backend after the GPU, sequentially on this thread. No
    /// `R_INPUT_BELOW_GPU_THRESHOLD` is recorded for the document; the
    /// execution record carries
    /// `segments.largest_text_bytes`. A document with no segment at the
    /// crossover continues on the next backend as a whole.
    fn route_segmented_gpu(
        &self,
        text: &str,
        start: usize,
        segmenter: &LiteralSegmenter,
    ) -> Result<SegmentedGpu, EngineError> {
        Ok(match self.compose_segmented_gpu(text, start, segmenter)? {
            GpuCompose::Ids {
                ids,
                facts,
                delivery,
                reason,
            } => SegmentedGpu::Routed(
                RoutedIds {
                    ids,
                    backend: BACKEND_GPU.to_owned(),
                    source: Some(delivery),
                    path: Some("gpu_segmented".to_owned()),
                    reason,
                },
                Some(facts),
            ),
            GpuCompose::AddedTokenReference => {
                SegmentedGpu::Routed(self.added_token_reference(text, start)?, None)
            }
            GpuCompose::EngineGuard { path, code } => SegmentedGpu::Routed(
                RoutedIds {
                    ids: self
                        .reference
                        .encode_ids(text, false)
                        .map_err(|error| EngineError(error.to_string()))?,
                    backend: BACKEND_REFERENCE.to_owned(),
                    source: None,
                    path: Some(path.to_owned()),
                    reason: Some(code),
                },
                None,
            ),
            GpuCompose::Continue(reason) => SegmentedGpu::Continue(reason),
        })
    }

    /// The segment work of a literal-bearing, GPU-headed document, shared
    /// by the ID route and the state route; each wraps the outcome in its
    /// own carrier.
    fn compose_segmented_gpu(
        &self,
        text: &str,
        start: usize,
        segmenter: &LiteralSegmenter,
    ) -> Result<GpuCompose, EngineError> {
        let pieces = segmenter.plan(text);
        let next = start + 1;
        if !self.added.text_pieces_closed(text, &pieces) {
            // A text segment holds a literal: the whole document goes to the reference.
            return Ok(GpuCompose::AddedTokenReference);
        }
        let crossover = self.thresholds[start];
        let on_gpu = |piece: &LiteralPiece| match *piece {
            LiteralPiece::Text { start, end } => (end - start) as u64 >= crossover,
            LiteralPiece::Literal { .. } => false,
        };
        if !pieces.iter().any(on_gpu) {
            return Ok(match self.chain[next] {
                BackendKind::FastCpu => GpuCompose::Continue(None),
                _ => GpuCompose::AddedTokenReference,
            });
        }
        let cell = self.gpu.as_ref().expect("constructor gate");
        let gpu = match cell.engine() {
            Ok(gpu) => gpu,
            Err(error) => {
                self.gpu_open_fault_event(start, &error);
                return Ok(GpuCompose::Continue(Some(R_EXEC_FAULT)));
            }
        };
        let gpu_pieces = pieces
            .iter()
            .enumerate()
            .filter(|(_, piece)| on_gpu(piece))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let batch = gpu_pieces
            .iter()
            .map(|&index| piece_text(text, pieces[index]))
            .collect::<Vec<_>>();
        let results = if batch.len() == 1 {
            vec![gpu.encode_ids(batch[0])]
        } else {
            gpu.encode_batch_ids(&batch)
        };
        if results.len() != batch.len() {
            return Err(EngineError(
                "native GPU batch returned the wrong row count".to_owned(),
            ));
        }
        let mut rows: Vec<Option<Vec<u32>>> = vec![None; pieces.len()];
        let mut reason = None;
        let mut facts = SegmentFacts::default();
        for (&index, result) in gpu_pieces.iter().zip(results) {
            match result {
                Ok(ids) => {
                    facts.gpu_bytes += piece_text(text, pieces[index]).len();
                    rows[index] = Some(ids);
                }
                Err(error) => {
                    self.record_event(
                        R_EXEC_FAULT,
                        BACKEND_GPU,
                        self.chain[next].as_str(),
                        json!({
                            "error": "NativeGpuError",
                            "message": error.to_string(),
                            "scope": "segment",
                        }),
                    );
                    reason = Some(R_EXEC_FAULT);
                }
            }
        }
        if facts.gpu_bytes == 0 {
            return Ok(GpuCompose::Continue(reason));
        }
        let rest = pieces
            .iter()
            .enumerate()
            .filter(|(index, piece)| {
                rows[*index].is_none() && matches!(piece, LiteralPiece::Text { .. })
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let rest_text = rest
            .iter()
            .map(|&index| piece_text(text, pieces[index]))
            .collect::<Vec<_>>();
        let mut on_reference = rest.clone();
        if !rest.is_empty() && self.chain[next] == BackendKind::FastCpu {
            let fast = self.fast_cpu.as_ref().expect("constructor gate");
            match fast.encode_literal_free_segments(&rest_text) {
                Ok(Some(encoded)) => {
                    for (&index, ids) in rest.iter().zip(encoded) {
                        facts.fast_cpu_bytes += piece_text(text, pieces[index]).len();
                        rows[index] = Some(ids);
                    }
                    on_reference.clear();
                }
                Ok(None) => {
                    // The engine guard refused a segment: the whole
                    // document is answered by the reference, as a guarded
                    // full encode is.
                    let (path, code) =
                        self.fast_cpu_reference_path(FastEncodeSource::ReferenceEngineGuard);
                    return Ok(GpuCompose::EngineGuard { path, code });
                }
                Err(error) => {
                    self.record_event(
                        R_EXEC_FAULT,
                        BACKEND_FAST_CPU,
                        BACKEND_REFERENCE,
                        json!({
                            "error": "FastCpuEngineError",
                            "message": error.to_string(),
                            "scope": "segment",
                        }),
                    );
                    reason = Some(R_EXEC_FAULT);
                }
            }
        }
        for index in on_reference {
            let segment = piece_text(text, pieces[index]);
            rows[index] = Some(
                self.reference
                    .encode_ids(segment, false)
                    .map_err(|error| EngineError(error.to_string()))?,
            );
            facts.hf_bytes += segment.len();
        }
        let mut ids = Vec::new();
        for (index, piece) in pieces.iter().enumerate() {
            match *piece {
                LiteralPiece::Literal { id, start, end } => {
                    ids.push(id);
                    facts.add_literal(end - start);
                }
                LiteralPiece::Text { start, end } => {
                    ids.extend(rows[index].take().ok_or_else(|| {
                        EngineError("a text segment was left unencoded".to_owned())
                    })?);
                    facts.add_text(end - start);
                }
            }
        }
        Ok(GpuCompose::Ids {
            ids,
            facts,
            delivery: gpu.delivery().to_owned(),
            reason,
        })
    }

    /// The state-route acceptance of a GPU-produced ID row: both modes
    /// share one fail-closed decision over the exact row; only the
    /// successful payload representation differs. A rejected row is
    /// re-encoded by the reference (`hf_span_guard`).
    ///
    /// The device normalizes its input itself, so its row is the row of
    /// the normalized text. The byte-length closure alone does not show
    /// that this is the row of `text`: on a text whose normalizer length
    /// changes cancel, the row closes and its spans are shifted. A row is
    /// therefore adopted only when `text` is also stable under the face's
    /// normalizer (checked after the closure, so a row the closure rejects
    /// keeps its diagnostic).
    #[allow(clippy::too_many_arguments)]
    fn accept_gpu_state(
        &self,
        text: &str,
        ids: Vec<u32>,
        mode: StateMode,
        start: usize,
        delivery: &str,
        path: &'static str,
        segments: Option<SegmentFacts>,
    ) -> Result<(StatePayload, String), EngineError> {
        let input_bytes = text.len() as u64;
        let accepted = match mode {
            StateMode::LazySeed => self
                .reference
                .verify_ids_close(text, &ids)
                .and_then(|()| self.reference.raw_byte_lengths_arc())
                .map(|table| StatePayload::Lazy { ids, table }),
            StateMode::Materialized => {
                self.reference
                    .spans_soa_for_ids(text, &ids)
                    .map(|(span_starts, span_ends)| {
                        StatePayload::Soa(SoaEncoding {
                            ids,
                            span_starts,
                            span_ends,
                        })
                    })
            }
        }
        .and_then(|payload| {
            self.reference
                .require_normalization_stable(text)
                .map(|()| payload)
        });
        match accepted {
            Ok(payload) => {
                let state_encode_path = payload.accelerated_state_encode_path();
                self.record_execution_detail(
                    BACKEND_GPU,
                    input_bytes,
                    self.chain[start].as_str(),
                    Some(delivery),
                    Some(path),
                    segments,
                );
                self.record_state_encode(
                    state_encode_path,
                    json!({
                        "path": state_encode_path,
                        "backend": BACKEND_GPU,
                        "input_bytes": input_bytes,
                    }),
                );
                Ok((payload, path.to_owned()))
            }
            Err(error) => {
                self.record_event(
                    R_INPUT_GUARD_ROUTED,
                    BACKEND_GPU,
                    BACKEND_REFERENCE,
                    json!({
                        "stage": "span_bridge",
                        "message": error.to_string(),
                    }),
                );
                let reference = self.reference_state(text)?;
                self.record_execution(
                    BACKEND_REFERENCE,
                    input_bytes,
                    start,
                    Some("state_encode"),
                    Some("hf_span_guard"),
                );
                self.record_state_encode(
                    "hf_span_guard",
                    json!({
                        "path": "hf_span_guard",
                        "error": "ReferenceEngineError",
                        "message": error.to_string(),
                    }),
                );
                Ok((StatePayload::Soa(reference), "hf_span_guard".to_owned()))
            }
        }
    }

    /// The whole input encoded by the reference because of its literals,
    /// on the state route.
    fn added_token_state(
        &self,
        encoding: SoaEncoding,
        input_bytes: u64,
        start: usize,
    ) -> (StatePayload, String) {
        self.record_event(
            R_INPUT_ADDED_TOKEN,
            self.chain[start].as_str(),
            BACKEND_REFERENCE,
            json!({}),
        );
        self.record_execution(
            BACKEND_REFERENCE,
            input_bytes,
            start,
            Some("state_encode"),
            Some("hf_added_token"),
        );
        self.record_state_encode("hf_added_token", json!({"path": "hf_added_token"}));
        (StatePayload::Soa(encoding), "hf_added_token".to_owned())
    }

    /// State route: session state adopts the payload without any
    /// bookkeeping clone of the ID row. In `Materialized` mode spans are
    /// produced directly in the store's structure-of-arrays layout (GPU
    /// IDs through the one-pass known-ID converter; CPU engines through
    /// their SoA surfaces). In `LazySeed` mode the accelerated rows
    /// return the closure-verified ID row itself plus the frozen
    /// byte-length table, and the session tail keeps only sparse span
    /// checkpoints; the reference rows keep materialized spans (exact
    /// under active normalizers). The closure check and the known-ID
    /// converter share the same acceptance decision, so ledger events,
    /// counters, and diagnostic shapes are identical in both modes and
    /// to the historical combined route with `state_encode=true`.
    fn route_state_internal(
        &self,
        text: &str,
        mode: StateMode,
    ) -> Result<(StatePayload, String), EngineError> {
        let input_bytes = text.len() as u64;
        let start = self.start_index(input_bytes);
        self.below_gpu_threshold_event(input_bytes, start);
        let mut first = start;
        if self.chain[start] != BackendKind::Reference {
            if let Some(segmenter) = self.served_segmenter() {
                if self.chain[start] == BackendKind::Gpu && self.added.candidate(text) {
                    match self.compose_segmented_gpu(text, start, segmenter)? {
                        GpuCompose::Ids {
                            ids,
                            facts,
                            delivery,
                            reason: _,
                        } => {
                            return self.accept_gpu_state(
                                text,
                                ids,
                                mode,
                                start,
                                &delivery,
                                "gpu_segmented",
                                Some(facts),
                            );
                        }
                        GpuCompose::AddedTokenReference => {
                            let encoding = self.reference_state(text)?;
                            return Ok(self.added_token_state(encoding, input_bytes, start));
                        }
                        GpuCompose::EngineGuard { path, code: _ } => {
                            let encoding = self.reference_state(text)?;
                            let payload = StatePayload::Soa(encoding);
                            let state_encode_path = payload.accelerated_state_encode_path();
                            self.record_execution(
                                BACKEND_REFERENCE,
                                input_bytes,
                                start,
                                None,
                                Some(path),
                            );
                            self.record_state_encode(
                                state_encode_path,
                                json!({
                                    "path": state_encode_path,
                                    "backend": BACKEND_REFERENCE,
                                    "input_bytes": input_bytes,
                                }),
                            );
                            return Ok((payload, path.to_owned()));
                        }
                        GpuCompose::Continue(_) => first = start + 1,
                    }
                }
                // A fast-CPU start segments inside the engine.
            } else if let Some(encoding) = self.exact_added_reference_state(text)? {
                return Ok(self.added_token_state(encoding, input_bytes, start));
            }
        }
        for index in first..self.chain.len() {
            match self.chain[index] {
                BackendKind::Gpu => {
                    let cell = self.gpu.as_ref().expect("constructor gate");
                    let gpu = match cell.engine() {
                        Ok(gpu) => gpu,
                        Err(error) => {
                            self.gpu_open_fault_event(index, &error);
                            continue;
                        }
                    };
                    match gpu.encode_ids(text) {
                        Ok(ids) => {
                            return self.accept_gpu_state(
                                text,
                                ids,
                                mode,
                                start,
                                gpu.delivery(),
                                "gpu_full",
                                None,
                            );
                        }
                        Err(error) => self.gpu_fault_event(index, &error),
                    }
                }
                BackendKind::FastCpu => {
                    let fast = self.fast_cpu.as_ref().expect("constructor gate");
                    let outcome = match mode {
                        StateMode::LazySeed => fast.encode_seed_with_segments(text),
                        StateMode::Materialized => fast.encode_state_with_segments(text).map(
                            |(encoding, source, segments)| {
                                (FastSeedPayload::Reference(encoding), source, segments)
                            },
                        ),
                    };
                    match outcome {
                        Err(error) => self.fast_cpu_fault_event(index, &error),
                        Ok((payload, FastEncodeSource::Gigatoken, segments)) => {
                            let payload = match payload {
                                FastSeedPayload::GigatokenIds(ids) => StatePayload::Lazy {
                                    ids,
                                    // The construction-time equality check
                                    // binds the Gigatoken byte lengths to
                                    // this exact table, and the row was
                                    // closure-verified against it.
                                    table: self
                                        .reference
                                        .raw_byte_lengths_arc()
                                        .map_err(|error| EngineError(error.to_string()))?,
                                },
                                FastSeedPayload::Reference(encoding) => StatePayload::Soa(encoding),
                            };
                            let state_encode_path = payload.accelerated_state_encode_path();
                            let path = if segments.is_some() {
                                "gigatoken_segmented"
                            } else {
                                "gigatoken_full"
                            };
                            self.record_execution_detail(
                                BACKEND_FAST_CPU,
                                input_bytes,
                                self.chain[start].as_str(),
                                Some("gigatoken"),
                                Some(path),
                                segments,
                            );
                            self.record_state_encode(
                                state_encode_path,
                                json!({
                                    "path": state_encode_path,
                                    "backend": BACKEND_FAST_CPU,
                                    "input_bytes": input_bytes,
                                }),
                            );
                            return Ok((payload, path.to_owned()));
                        }
                        Ok((payload, source, _segments)) => {
                            let FastSeedPayload::Reference(encoding) = payload else {
                                return Err(EngineError(
                                    "reference-routed fast encode returned an ID-only payload"
                                        .to_owned(),
                                ));
                            };
                            let payload = StatePayload::Soa(encoding);
                            let state_encode_path = payload.accelerated_state_encode_path();
                            let (path, _code) = self.fast_cpu_reference_path(source);
                            self.record_execution(
                                BACKEND_REFERENCE,
                                input_bytes,
                                start,
                                None,
                                Some(path),
                            );
                            self.record_state_encode(
                                state_encode_path,
                                json!({
                                    "path": state_encode_path,
                                    "backend": BACKEND_REFERENCE,
                                    "input_bytes": input_bytes,
                                }),
                            );
                            return Ok((payload, path.to_owned()));
                        }
                    }
                }
                BackendKind::Reference => {
                    let encoding = self.reference_state(text)?;
                    let (source, path) = if !self.repair_fast_cpu {
                        (Some("state_encode"), "hf_no_certified_span_bridge")
                    } else {
                        (None, "hf_full")
                    };
                    self.record_execution(
                        BACKEND_REFERENCE,
                        input_bytes,
                        start,
                        source,
                        Some(path),
                    );
                    self.record_state_encode(path, json!({"path": path}));
                    return Ok((StatePayload::Soa(encoding), path.to_owned()));
                }
            }
        }
        Err(EngineError(
            "native fallback chain terminated without hf".to_owned(),
        ))
    }

    pub fn encode_ids(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<RoutedIds, EngineError> {
        if add_special_tokens && self.postprocessor_adds_tokens {
            let input_bytes = text.len() as u64;
            let start = self.start_index(input_bytes);
            for index in start..self.chain.len() - 1 {
                self.record_event(
                    R_INPUT_POSTPROCESS_ROUTED,
                    self.chain[index].as_str(),
                    self.chain[index + 1].as_str(),
                    json!({
                        "error": "BackendExecutionFault",
                        "message": "the accelerated path produces the core stream",
                        "scope": "input",
                        "stage": "add_special_tokens",
                    }),
                );
            }
            let ids = self
                .reference
                .encode_ids(text, true)
                .map_err(|error| EngineError(error.to_string()))?;
            self.record_execution(
                BACKEND_REFERENCE,
                input_bytes,
                start,
                None,
                Some("hf_postprocessed"),
            );
            return Ok(RoutedIds {
                ids,
                backend: BACKEND_REFERENCE.to_owned(),
                source: None,
                path: Some("hf_postprocessed".to_owned()),
                reason: Some(R_INPUT_POSTPROCESS_ROUTED),
            });
        }
        self.route_ids_internal(text)
    }

    pub fn encode_batch_ids(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<RoutedIds>, EngineError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if add_special_tokens && self.postprocessor_adds_tokens {
            let rows = self
                .reference
                .encode_batch_ids(texts, true)
                .map_err(|error| EngineError(error.to_string()))?;
            let mut output = Vec::with_capacity(rows.len());
            for (text, ids) in texts.iter().zip(rows) {
                let input_bytes = text.len() as u64;
                let start = self.start_index(input_bytes);
                for index in start..self.chain.len() - 1 {
                    self.record_event(
                        R_INPUT_POSTPROCESS_ROUTED,
                        self.chain[index].as_str(),
                        self.chain[index + 1].as_str(),
                        json!({
                            "error": "BackendExecutionFault",
                            "message": "the accelerated path produces the core stream",
                            "scope": "input",
                            "stage": "add_special_tokens",
                        }),
                    );
                }
                self.record_execution(
                    BACKEND_REFERENCE,
                    input_bytes,
                    start,
                    None,
                    Some("hf_postprocessed"),
                );
                output.push(RoutedIds {
                    ids,
                    backend: BACKEND_REFERENCE.to_owned(),
                    source: None,
                    path: Some("hf_postprocessed".to_owned()),
                    reason: Some(R_INPUT_POSTPROCESS_ROUTED),
                });
            }
            return Ok(output);
        }

        let starts = texts
            .iter()
            .map(|text| self.start_index(text.len() as u64))
            .collect::<Vec<_>>();
        let mut next = starts.clone();
        let mut output = vec![None; texts.len()];
        // The routing decision recorded for each row, carried alongside
        // it so the returned facts quote the ledger rather than re-read
        // the path string.
        let mut reasons: Vec<Option<&'static str>> = vec![None; texts.len()];
        let mut segments: Vec<Option<SegmentFacts>> = vec![None; texts.len()];
        let served = self.served_segmenter().is_some();

        // Resolve the exact added-token frontend before grouping rows.  The
        // native matcher is only a cheap candidate gate; the reference
        // engine makes the final decision and its already-produced IDs are
        // reused when an added token really occurs.
        for (row, text) in texts.iter().enumerate() {
            if self.chain.first() == Some(&BackendKind::Gpu) && starts[row] > 0 {
                self.record_event(
                    R_INPUT_BELOW_GPU_THRESHOLD,
                    BACKEND_GPU,
                    self.chain[starts[row]].as_str(),
                    json!({
                        "input_bytes": text.len(),
                        "threshold_bytes": self.thresholds[0],
                    }),
                );
                reasons[row] = Some(R_INPUT_BELOW_GPU_THRESHOLD);
            }
            if self.chain[starts[row]] == BackendKind::Reference {
                continue;
            }
            if served {
                // A literal-bearing row whose route starts on the GPU takes
                // the single-document path; a fast-CPU-stage row is
                // segmented inside its batch worker.
                if self.chain[starts[row]] == BackendKind::Gpu && self.added.candidate(text) {
                    let (routed, _start, facts) = self.route_ids_unrecorded(text)?;
                    reasons[row] = routed.reason;
                    segments[row] = facts;
                    output[row] = Some(routed);
                }
                continue;
            }
            if let Some(ids) = self.exact_added_reference_ids(text)? {
                self.record_event(
                    R_INPUT_ADDED_TOKEN,
                    self.chain[starts[row]].as_str(),
                    BACKEND_REFERENCE,
                    json!({}),
                );
                reasons[row] = Some(R_INPUT_ADDED_TOKEN);
                output[row] = Some(RoutedIds {
                    ids,
                    backend: BACKEND_REFERENCE.to_owned(),
                    source: None,
                    path: Some("hf_added_token".to_owned()),
                    reason: reasons[row],
                });
            }
        }

        for stage in 0..self.chain.len() {
            let indices = (0..texts.len())
                .filter(|row| output[*row].is_none() && next[*row] == stage)
                .collect::<Vec<_>>();
            if indices.is_empty() {
                continue;
            }
            let batch = indices.iter().map(|row| texts[*row]).collect::<Vec<_>>();
            match self.chain[stage] {
                BackendKind::Gpu => {
                    let cell = self.gpu.as_ref().expect("constructor gate");
                    let gpu = match cell.engine() {
                        Ok(gpu) => gpu,
                        Err(error) => {
                            for row in indices {
                                self.gpu_open_fault_event(stage, &error);
                                reasons[row] = Some(R_EXEC_FAULT);
                                next[row] += 1;
                            }
                            continue;
                        }
                    };
                    let rows = gpu.encode_batch_ids(&batch);
                    if rows.len() != indices.len() {
                        return Err(EngineError(
                            "native GPU batch returned the wrong row count".to_owned(),
                        ));
                    }
                    for (row, result) in indices.into_iter().zip(rows) {
                        match result {
                            Ok(ids) => {
                                output[row] = Some(RoutedIds {
                                    ids,
                                    backend: BACKEND_GPU.to_owned(),
                                    source: Some(gpu.delivery().to_owned()),
                                    path: Some("gpu_full".to_owned()),
                                    reason: reasons[row],
                                });
                            }
                            Err(error) => {
                                self.record_event(
                                    R_EXEC_FAULT,
                                    BACKEND_GPU,
                                    self.chain[stage + 1].as_str(),
                                    json!({
                                        "error": "NativeGpuError",
                                        "message": error.to_string(),
                                        "scope": "input",
                                    }),
                                );
                                reasons[row] = Some(R_EXEC_FAULT);
                                next[row] += 1;
                            }
                        }
                    }
                }
                BackendKind::FastCpu => {
                    let fast = self.fast_cpu.as_ref().expect("constructor gate");
                    match fast.encode_batch_ids_with_segments(&batch) {
                        Ok(rows) => {
                            for (row, (ids, source, facts)) in indices.into_iter().zip(rows) {
                                let routed = match source {
                                    FastEncodeSource::Gigatoken => RoutedIds {
                                        ids,
                                        backend: BACKEND_FAST_CPU.to_owned(),
                                        source: Some("gigatoken".to_owned()),
                                        path: Some(
                                            if facts.is_some() {
                                                "gigatoken_segmented"
                                            } else {
                                                "gigatoken_full"
                                            }
                                            .to_owned(),
                                        ),
                                        reason: reasons[row],
                                    },
                                    source @ (FastEncodeSource::ReferenceAddedToken
                                    | FastEncodeSource::ReferenceEngineGuard) => {
                                        let (path, code) = self.fast_cpu_reference_path(source);
                                        reasons[row] = Some(code);
                                        RoutedIds {
                                            ids,
                                            backend: BACKEND_REFERENCE.to_owned(),
                                            source: None,
                                            path: Some(path.to_owned()),
                                            reason: Some(code),
                                        }
                                    }
                                };
                                if routed.backend == BACKEND_FAST_CPU {
                                    segments[row] = facts;
                                }
                                output[row] = Some(routed);
                            }
                        }
                        Err(error) => {
                            for row in indices {
                                self.record_event(
                                    R_EXEC_FAULT,
                                    BACKEND_FAST_CPU,
                                    self.chain[stage + 1].as_str(),
                                    json!({
                                        "error": "FastCpuEngineError",
                                        "message": error.to_string(),
                                        "scope": "input",
                                    }),
                                );
                                reasons[row] = Some(R_EXEC_FAULT);
                                next[row] += 1;
                            }
                        }
                    }
                }
                BackendKind::Reference => {
                    let rows = self
                        .reference
                        .encode_batch_ids(&batch, false)
                        .map_err(|error| EngineError(error.to_string()))?;
                    for (row, ids) in indices.into_iter().zip(rows) {
                        output[row] = Some(RoutedIds {
                            ids,
                            backend: BACKEND_REFERENCE.to_owned(),
                            source: None,
                            path: Some("hf_full".to_owned()),
                            reason: reasons[row],
                        });
                    }
                }
            }
        }

        let output = output
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| EngineError("native batch route did not close".to_owned()))?;
        // Processing is grouped by implementation, but the public ledger is
        // recorded in caller order so `last_execution` and counters retain
        // the same observable semantics as row-by-row encoding.
        for (row, routed) in output.iter().enumerate() {
            self.record_execution_detail(
                &routed.backend,
                texts[row].len() as u64,
                self.chain[starts[row]].as_str(),
                routed.source.as_deref(),
                routed.path.as_deref(),
                segments[row],
            );
        }
        Ok(output)
    }
}

impl NativeRouter {
    /// The certified chunk-parallel planner of the corrected CPU engine,
    /// when this route has one.
    pub fn chunk_encoder(&self) -> Result<&ChunkEncoder, EngineError> {
        self.fast_cpu
            .as_ref()
            .ok_or_else(|| {
                EngineError("this route has no corrected CPU engine to chunk on".to_owned())
            })?
            .chunk_encoder()
    }

    /// The cut plan the chunked route would use for `text`.
    pub fn chunked_plan(
        &self,
        text: &str,
        k: usize,
        min_chunk: Option<usize>,
        scan_budget: usize,
    ) -> Result<ChunkPlan, EngineError> {
        self.fast_cpu
            .as_ref()
            .ok_or_else(|| {
                EngineError("this route has no corrected CPU engine to chunk on".to_owned())
            })?
            .chunked_plan(text, k, min_chunk, scan_budget)
    }

    /// Fork (and optionally pre-warm) the chunk worker pool.
    pub fn chunked_warm(&self, workers: usize, text: Option<&str>) -> Result<usize, EngineError> {
        self.fast_cpu
            .as_ref()
            .ok_or_else(|| {
                EngineError("this route has no corrected CPU engine to chunk on".to_owned())
            })?
            .chunked_warm(workers, text)
    }

    /// Core-stream ids of `text` through the certified chunk-parallel
    /// encoder, with every row recorded in the execution ledger exactly
    /// as a batch row is. An input whose starting backend is not the
    /// corrected CPU engine (a GPU-headed plan above its crossover) is
    /// encoded serially through the normal route and says so.
    pub fn chunked_encode_ids(
        &self,
        text: &str,
        k: usize,
        min_chunk: Option<usize>,
        scan_budget: usize,
    ) -> Result<ChunkedIds, EngineError> {
        let input_bytes = text.len() as u64;
        let start = self.start_index(input_bytes);
        let fast = match &self.fast_cpu {
            Some(fast) if self.chain[start] == BackendKind::FastCpu => fast,
            _ => {
                let started = std::time::Instant::now();
                let routed = self.route_ids_internal(text)?;
                let encode_ns = started.elapsed().as_nanos() as u64;
                return Ok(ChunkedIds {
                    ids: routed.ids,
                    chunks: 1,
                    k_requested: k,
                    cuts: Vec::new(),
                    cut_misses: 0,
                    literal_rejects: 0,
                    norm_rejects: 0,
                    serial_reason: Some(R_CHUNK_ROUTE_NOT_FAST_CPU),
                    below_threshold: false,
                    sources: Vec::new(),
                    row_bytes: vec![text.len()],
                    plan_ns: 0,
                    encode_ns,
                    concat_ns: 0,
                    worker_ns_max: encode_ns,
                    worker_ns_min: encode_ns,
                    cold_forks: 0,
                });
            }
        };
        let encoded = fast.chunked_encode_ids(text, k, min_chunk, scan_budget)?;
        let served = self.served_segmenter().is_some();
        let mut row_start = 0usize;
        for (source, bytes) in encoded.sources.iter().zip(&encoded.row_bytes) {
            let row = text.get(row_start..row_start + *bytes).unwrap_or("");
            row_start += *bytes;
            match source {
                // A chunk row that holds a literal was segmented inside its
                // worker.
                FastEncodeSource::Gigatoken => self.record_execution(
                    BACKEND_FAST_CPU,
                    *bytes as u64,
                    start,
                    Some("gigatoken"),
                    Some(if served && self.added.candidate(row) {
                        "gigatoken_segmented"
                    } else {
                        "gigatoken_full"
                    }),
                ),
                source => {
                    let (path, _code) = self.fast_cpu_reference_path(*source);
                    self.record_execution(
                        BACKEND_REFERENCE,
                        *bytes as u64,
                        start,
                        None,
                        Some(path),
                    );
                }
            }
        }
        Ok(encoded)
    }
}

impl SessionEncoder for NativeRouter {
    fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
        match self.route_state_internal(text, StateMode::Materialized)? {
            (StatePayload::Soa(encoding), _path) => Ok(encoding.into_pairs()),
            (StatePayload::Lazy { .. }, _path) => Err(EngineError(
                "the materialized state route returned a lazy payload".to_owned(),
            )),
        }
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        // `SessionStore::put` seeds a fresh session by appending into an empty
        // tail.  Treat that shape as a full routed request: it must retain the
        // GPU crossover, the ordered fallback chain, and the same execution
        // ledger as a non-store encode.  Only an actual continuation goes
        // directly to the certified repair engine.  The payload moves into
        // the tail without pair materialization or a bookkeeping ID clone:
        // accelerated rows adopt the closure-verified ID allocation itself
        // with lazy checkpointed spans, reference rows adopt their
        // structure-of-arrays offsets.
        if tail.text().is_empty() {
            let (payload, path) = self.route_state_internal(delta, StateMode::LazySeed)?;
            match payload {
                StatePayload::Soa(encoding) => tail.fill_soa(delta, encoding),
                StatePayload::Lazy { ids, table } => {
                    tail.fill_lazy(delta, SharedIds::from_vec(ids), table)
                }
            }
            .map_err(|error| EngineError(error.to_string()))?;
            return Ok(AppendReport {
                path,
                kept_tokens: 0,
            });
        }
        match (&self.fast_cpu, self.repair_fast_cpu) {
            (Some(engine), true) => {
                let grown_bytes = tail.text_bytes() as u64 + delta.len() as u64;
                let (report, repair_input_bytes) = engine.append_with_repair_input(tail, delta)?;
                if let Some(input_bytes) = repair_input_bytes {
                    self.record_execution_from(
                        BACKEND_FAST_CPU,
                        input_bytes,
                        BACKEND_FAST_CPU,
                        Some("gigatoken"),
                        Some(&report.path),
                    );
                } else if report.path.starts_with("hf_full_") {
                    // The bounded window could not find a safe cut, so the
                    // whole grown text was re-encoded on the reference
                    // engine. That ran, and the ledger says so: without
                    // this the answer to "what ran most recently" stayed
                    // on the seed encode while the reference path was the
                    // one that produced what was returned. A no-op append
                    // records nothing, because nothing ran.
                    self.record_execution_from(
                        BACKEND_REFERENCE,
                        grown_bytes,
                        BACKEND_REFERENCE,
                        Some("gigatoken"),
                        Some(&report.path),
                    );
                }
                Ok(report)
            }
            _ => {
                let input_bytes = tail.text_bytes() as u64 + delta.len() as u64;
                let report = self.reference.append(tail, delta)?;
                self.record_execution_from(
                    BACKEND_REFERENCE,
                    input_bytes,
                    BACKEND_REFERENCE,
                    None,
                    Some(&report.path),
                );
                Ok(report)
            }
        }
    }

    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        match (&self.fast_cpu, self.repair_fast_cpu) {
            (Some(engine), true) => engine.last_certified_boundary(tail, floor_char, ceil_char),
            _ => Ok(None),
        }
    }

    fn witness_category(&self) -> WitnessCategory {
        match (&self.fast_cpu, self.repair_fast_cpu) {
            (Some(engine), true) => engine.witness_category(),
            _ => WitnessCategory::NoneFullReencode,
        }
    }
}

fn piece_text(text: &str, piece: LiteralPiece) -> &str {
    match piece {
        LiteralPiece::Literal { start, end, .. } | LiteralPiece::Text { start, end } => {
            &text[start..end]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use tokenizers::models::bpe::{Vocab, BPE};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::Tokenizer;

    #[derive(Debug)]
    struct MockGpu;

    impl NativeGpuEngine for MockGpu {
        fn encode_ids(&self, text: &str) -> Result<Vec<u32>, NativeRuntimeError> {
            if text.contains('x') {
                Err(NativeRuntimeError::new("injected device fault"))
            } else {
                Ok(vec![99])
            }
        }

        fn delivery(&self) -> &'static str {
            "prebuilt"
        }
    }

    fn reference() -> Arc<ReferenceEngine> {
        let model = BPE::builder()
            .vocab_and_merges([("a".to_owned(), 0)], Vec::new())
            .build()
            .unwrap();
        let tokenizer = Tokenizer::new(model);
        Arc::new(
            ReferenceEngine::from_bytes(tokenizer.to_string(false).unwrap().as_bytes()).unwrap(),
        )
    }

    fn assert_guard_events_have_stage(stats: &RuntimeStats) {
        let guard_events = stats
            .events
            .iter()
            .filter(|event| event.code == R_INPUT_GUARD_ROUTED)
            .collect::<Vec<_>>();
        assert!(!guard_events.is_empty(), "expected a guard-routing event");
        for event in guard_events {
            assert!(
                event.detail.get("stage").is_some(),
                "guard event has no stage: {event:?}"
            );
        }
    }

    fn byte_level_tokenizer_json() -> Vec<u8> {
        let visible = (33u32..=126)
            .chain(161..=172)
            .chain(174..=255)
            .collect::<Vec<_>>();
        let mut extra = 0u32;
        let mut vocab = Vocab::default();
        for byte in 0u32..=255 {
            let codepoint = if visible.contains(&byte) {
                byte
            } else {
                let mapped = 256 + extra;
                extra += 1;
                mapped
            };
            vocab.insert(char::from_u32(codepoint).unwrap().to_string(), byte);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        let byte_level = ByteLevel::new(false, true, true);
        tokenizer.with_pre_tokenizer(Some(byte_level));
        tokenizer.with_decoder(Some(byte_level));
        tokenizer.to_string(false).unwrap().into_bytes()
    }

    fn repair_router(gpu: Arc<dyn NativeGpuEngine>) -> NativeRouter {
        let tokenizer_json = byte_level_tokenizer_json();
        let reference = Arc::new(ReferenceEngine::from_bytes(&tokenizer_json).unwrap());
        let digest = Sha256::digest(&tokenizer_json)
            .iter()
            .map(|value| format!("{value:02x}"))
            .collect::<String>();
        let mut spec = crate::FastRepairSpec::new("tiny_bytes".to_owned(), digest, 4, 2, false);
        spec.window_chars = 64;
        spec.max_retries = 2;
        let mut pclass = vec![0u8; crate::N_CODEPOINTS];
        for value in b'a'..=b'z' {
            pclass[value as usize] = 2;
        }
        for value in b'A'..=b'Z' {
            pclass[value as usize] = 2;
        }
        for value in b'0'..=b'9' {
            pclass[value as usize] = 3;
        }
        pclass[b' ' as usize] = 1;
        let fast = Arc::new(
            FastCpuEngine::from_reference(&tokenizer_json, Arc::clone(&reference), spec, pclass)
                .unwrap(),
        );
        NativeRouter::new(
            vec![
                BACKEND_GPU.to_owned(),
                BACKEND_FAST_CPU.to_owned(),
                BACKEND_REFERENCE.to_owned(),
            ],
            vec![1024, 0, 0],
            reference,
            Some(fast),
            true,
            Some(NativeGpuSource::Open(gpu)),
            false,
            true,
        )
        .unwrap()
    }

    fn reference_reencode_router(gpu: Arc<dyn NativeGpuEngine>) -> NativeRouter {
        let tokenizer_json = byte_level_tokenizer_json();
        let reference = Arc::new(ReferenceEngine::from_bytes(&tokenizer_json).unwrap());
        NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![1, 0],
            reference,
            None,
            false,
            Some(NativeGpuSource::Open(gpu)),
            false,
            true,
        )
        .unwrap()
    }

    #[test]
    fn native_batch_groups_rows_and_preserves_ordered_fallbacks() {
        let router = NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![4, 0],
            reference(),
            None,
            false,
            Some(NativeGpuSource::Open(Arc::new(MockGpu))),
            false,
            true,
        )
        .unwrap();
        let rows = router
            .encode_batch_ids(&["a", "aaaa", "xxxx"], false)
            .unwrap();

        assert_eq!(rows[0].backend, BACKEND_REFERENCE);
        assert_eq!(rows[0].ids, vec![0]);
        assert_eq!(rows[1].backend, BACKEND_GPU);
        assert_eq!(rows[1].ids, vec![99]);
        assert_eq!(rows[2].backend, BACKEND_REFERENCE);
        assert!(rows[2].ids.is_empty());
        // Each row carries the decision that moved it, and the row that
        // was not moved carries nothing.
        assert_eq!(rows[0].reason, Some(R_INPUT_BELOW_GPU_THRESHOLD));
        assert_eq!(rows[1].reason, None);
        assert_eq!(rows[2].reason, Some(R_EXEC_FAULT));
        let stats = router.stats();
        assert_eq!(stats.execution_counts.get(BACKEND_GPU), Some(&1));
        assert_eq!(stats.execution_counts.get(BACKEND_REFERENCE), Some(&2));
        assert_eq!(stats.fallback_counts.get(R_EXEC_FAULT), Some(&1));
        assert_eq!(
            stats.fallback_counts.get(R_INPUT_BELOW_GPU_THRESHOLD),
            Some(&1)
        );
        assert_eq!(
            stats.last_execution.as_ref().unwrap()["executed_backend"],
            BACKEND_REFERENCE
        );
    }

    /// The condition the bisecting replay rests on, pinned where it can
    /// be checked without a device.
    ///
    /// Locating a failing row by halving a batch and adopting the halves
    /// that succeed is only equal to redoing every row if a document's
    /// ids do not depend on which other documents share its batch. The
    /// batch surface is required to be row-for-row equal to encoding each
    /// document on its own (`docs/contracts/api.md` Section 4); this asks
    /// for that across the batch shapes a halving search actually
    /// produces, and with neighbours changed underneath a fixed document.
    #[test]
    fn a_documents_ids_do_not_depend_on_its_neighbours() {
        let router = NativeRouter::new(
            vec![BACKEND_REFERENCE.to_owned()],
            vec![0],
            reference(),
            None,
            false,
            None,
            false,
            true,
        )
        .unwrap();

        let corpus = (0..64)
            .map(|index| "a".repeat(index % 7 + 1))
            .collect::<Vec<_>>();
        let documents = corpus.iter().map(String::as_str).collect::<Vec<_>>();

        let alone = documents
            .iter()
            .map(|text| router.encode_ids(text, false).unwrap().ids)
            .collect::<Vec<_>>();

        // Every halving a search can reach, down to batches of one.
        for size in [64usize, 32, 16, 8, 4, 2, 1] {
            let mut produced = Vec::new();
            for chunk in documents.chunks(size) {
                for row in router.encode_batch_ids(chunk, false).unwrap() {
                    produced.push(row.ids);
                }
            }
            assert_eq!(produced, alone, "batch size {size}");
        }

        // And with the neighbours reversed under each document.
        let reversed = documents.iter().rev().copied().collect::<Vec<_>>();
        let rows = router.encode_batch_ids(&reversed, false).unwrap();
        for (offset, row) in rows.into_iter().enumerate() {
            assert_eq!(row.ids, alone[documents.len() - 1 - offset], "row {offset}");
        }
    }

    /// A fallback is counted once per document that ends on the
    /// reference path, whatever the accelerated backend had to do
    /// internally to work out which document that was.
    ///
    /// The prebuilt engine locates a failing row inside a failed batch by
    /// halving the batch and re-attempting the halves. Those attempts are
    /// internal probes: they are below this boundary, they record
    /// nothing, and a tree of any depth still hands one result back per
    /// row. This pins the arithmetic the ledger promises -- one faulting
    /// document counts one, and n count n -- against any future change to
    /// how the search is shaped.
    #[test]
    fn a_fallback_is_counted_once_per_document_not_once_per_attempt() {
        let router = NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![0, 0],
            reference(),
            None,
            false,
            Some(NativeGpuSource::Open(Arc::new(MockGpu))),
            false,
            true,
        )
        .unwrap();

        // One faulting document among many clean ones counts exactly one.
        let mut documents = vec!["aaaa"; 127];
        documents.insert(70, "xxxx");
        let rows = router.encode_batch_ids(&documents, false).unwrap();
        assert_eq!(rows.len(), 128);
        assert_eq!(
            rows.iter()
                .filter(|row| row.backend == BACKEND_REFERENCE)
                .count(),
            1
        );
        assert_eq!(rows[70].backend, BACKEND_REFERENCE);
        assert_eq!(rows[70].reason, Some(R_EXEC_FAULT));
        assert_eq!(router.stats().fallback_counts.get(R_EXEC_FAULT), Some(&1));

        // And n faulting documents in one call count exactly n.
        let mut documents = vec!["aaaa"; 128];
        let faults = [0usize, 1, 63, 64, 65, 127];
        for position in faults {
            documents[position] = "xxxx";
        }
        let rows = router.encode_batch_ids(&documents, false).unwrap();
        let reported = (0..rows.len())
            .filter(|index| rows[*index].reason == Some(R_EXEC_FAULT))
            .collect::<Vec<_>>();
        assert_eq!(reported, faults.to_vec());
        assert_eq!(
            router.stats().fallback_counts.get(R_EXEC_FAULT),
            Some(&(1 + faults.len() as u64))
        );
    }

    #[test]
    fn fast_cpu_guard_events_name_the_engine_guard_stage() {
        let router = NativeRouter::new(
            vec![BACKEND_REFERENCE.to_owned()],
            vec![0],
            reference(),
            None,
            false,
            None,
            false,
            true,
        )
        .unwrap();

        assert_eq!(
            router.fast_cpu_reference_path(FastEncodeSource::ReferenceEngineGuard),
            ("hf_engine_guard", R_INPUT_GUARD_ROUTED)
        );
        let stats = router.stats();
        assert_guard_events_have_stage(&stats);
        assert_eq!(stats.events[0].detail, json!({"stage": "engine_guard"}));
    }

    /// A routed result carries the same code the ledger counted for it,
    /// and carries none when the first admitted backend ran the input.
    #[test]
    fn a_routed_result_quotes_the_reason_the_ledger_recorded() {
        let router = NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![4, 0],
            reference(),
            None,
            false,
            Some(NativeGpuSource::Open(Arc::new(MockGpu))),
            false,
            true,
        )
        .unwrap();

        // Long enough for the GPU, and the mock answers it.
        let admitted = router.encode_ids("aaaa", false).unwrap();
        assert_eq!(admitted.backend, BACKEND_GPU);
        assert_eq!(admitted.reason, None);

        // Below the crossover: the same plan starts one backend later.
        let below = router.encode_ids("a", false).unwrap();
        assert_eq!(below.backend, BACKEND_REFERENCE);
        assert_eq!(below.reason, Some(R_INPUT_BELOW_GPU_THRESHOLD));

        // Long enough for the GPU, which faults on it.
        let faulted = router.encode_ids("xxxx", false).unwrap();
        assert_eq!(faulted.backend, BACKEND_REFERENCE);
        assert_eq!(faulted.reason, Some(R_EXEC_FAULT));

        // Every reason a result reported was also counted in the ledger.
        let stats = router.stats();
        for code in [R_INPUT_BELOW_GPU_THRESHOLD, R_EXEC_FAULT] {
            assert_eq!(stats.fallback_counts.get(code), Some(&1), "for {code}");
        }
    }

    #[test]
    fn postprocessing_bypass_has_its_own_reason_code() {
        let router = NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![0, 0],
            reference(),
            None,
            false,
            Some(NativeGpuSource::Open(Arc::new(MockGpu))),
            true,
            true,
        )
        .unwrap();

        let single = router.encode_ids("a", true).unwrap();
        assert_eq!(single.backend, BACKEND_REFERENCE);
        assert_eq!(single.reason, Some(R_INPUT_POSTPROCESS_ROUTED));
        let batch = router.encode_batch_ids(&["a", "aa"], true).unwrap();
        assert!(batch
            .iter()
            .all(|outcome| outcome.backend == BACKEND_REFERENCE));
        assert!(batch
            .iter()
            .all(|outcome| outcome.reason == Some(R_INPUT_POSTPROCESS_ROUTED)));

        let stats = router.stats();
        assert_eq!(
            stats.fallback_counts.get(R_INPUT_POSTPROCESS_ROUTED),
            Some(&3)
        );
        assert_eq!(stats.fallback_counts.get(R_EXEC_FAULT), None);
        assert_eq!(stats.execution_counts.get(BACKEND_REFERENCE), Some(&3));
        assert_eq!(stats.events.len(), 3);
        assert!(stats
            .events
            .iter()
            .all(|event| event.code == R_INPUT_POSTPROCESS_ROUTED));
    }

    /// GPU whose IDs are the exact core stream for pure-"a" inputs, so the
    /// known-ID span bridge closes and the state seed stays on the GPU row.
    /// The address of the last produced ID allocation is recorded so tests
    /// can prove the store adopted it rather than copying it.
    #[derive(Debug, Default)]
    struct ClosingGpu {
        last_ids_ptr: std::sync::atomic::AtomicUsize,
    }

    #[derive(Debug)]
    struct ByteGpu;

    impl NativeGpuEngine for ByteGpu {
        fn encode_ids(&self, text: &str) -> Result<Vec<u32>, NativeRuntimeError> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn delivery(&self) -> &'static str {
            "prebuilt"
        }
    }

    #[test]
    fn native_repair_replaces_the_seed_ledger_and_noop_leaves_it_unchanged() {
        let router = repair_router(Arc::new(ByteGpu));
        let mut store = toktier_store_core::SessionStore::with_defaults();
        let key = store.register_fingerprint([6u8; 32], 0).unwrap();
        let base = ("alpha 123 beta 456 ".repeat(60)) + "tail";
        let delta = " appended text 789";
        let put = store.put(key, &base, &router).unwrap();
        assert_eq!(
            router.stats().last_execution.unwrap()["executed_backend"],
            BACKEND_GPU
        );

        let appended = store
            .append_patch(put.handle, delta, put.revision, &router)
            .unwrap();

        assert_eq!(appended.path, "gigatoken_repair");
        let stats = router.stats();
        assert_eq!(stats.execution_counts.get(BACKEND_GPU), Some(&1));
        assert_eq!(stats.execution_counts.get(BACKEND_FAST_CPU), Some(&1));
        assert_eq!(
            stats.last_execution,
            Some(json!({
                "input_bytes": 64 + delta.len(),
                "selected_start": BACKEND_FAST_CPU,
                "executed_backend": BACKEND_FAST_CPU,
                "source": "gigatoken",
                "path": "gigatoken_repair",
            }))
        );

        let noop = store
            .append_patch(put.handle, "", appended.revision, &router)
            .unwrap();
        assert_eq!(noop.path, "noop");
        let after_noop = router.stats();
        assert_eq!(after_noop.execution_counts, stats.execution_counts);
        assert_eq!(after_noop.last_execution, stats.last_execution);
    }

    /// A session short enough that the repair window covers the whole
    /// text has no safe cut to find, so the grown text is re-encoded on
    /// the reference engine. That is the execution the ledger must
    /// report: it used to keep reporting the seed encode, so a session
    /// whose latest answer came from the reference path headlined the
    /// accelerated one.
    #[test]
    fn a_repair_that_falls_back_headlines_the_reference_execution() {
        let router = repair_router(Arc::new(ByteGpu));
        let mut store = toktier_store_core::SessionStore::with_defaults();
        let key = store.register_fingerprint([9u8; 32], 0).unwrap();
        let base = "alpha 123 beta";
        let delta = " gamma 456";
        assert!(base.len() + delta.len() < 64, "inside the repair window");
        let put = store.put(key, base, &router).unwrap();
        let seeded = router.stats().last_execution.unwrap();

        let appended = store
            .append_patch(put.handle, delta, put.revision, &router)
            .unwrap();

        assert_eq!(appended.path, "hf_full_window_covers_all");
        let stats = router.stats();
        assert_ne!(stats.last_execution.as_ref().unwrap(), &seeded);
        assert_eq!(
            stats.last_execution,
            Some(json!({
                "input_bytes": (base.len() + delta.len()) as u64,
                "selected_start": BACKEND_REFERENCE,
                "executed_backend": BACKEND_REFERENCE,
                "source": "gigatoken",
                "path": "hf_full_window_covers_all",
            }))
        );

        let noop = store
            .append_patch(put.handle, "", appended.revision, &router)
            .unwrap();
        assert_eq!(noop.path, "noop");
        assert_eq!(router.stats().last_execution, stats.last_execution);
    }

    #[test]
    fn native_reference_reencode_replaces_the_seed_ledger_and_noop_leaves_it_unchanged() {
        let router = reference_reencode_router(Arc::new(ByteGpu));
        let mut store = toktier_store_core::SessionStore::with_defaults();
        let key = store.register_fingerprint([7u8; 32], 0).unwrap();
        let base = "alpha βeta 123";
        let delta = " appended café 中";
        let put = store.put(key, base, &router).unwrap();
        assert_eq!(
            router.stats().last_execution.unwrap()["executed_backend"],
            BACKEND_GPU
        );

        let appended = store
            .append_patch(put.handle, delta, put.revision, &router)
            .unwrap();

        assert_eq!(appended.path, "native_hf_full_reencode");
        let stats = router.stats();
        assert_eq!(stats.execution_counts.get(BACKEND_GPU), Some(&1));
        assert_eq!(stats.execution_counts.get(BACKEND_REFERENCE), Some(&1));
        assert_eq!(
            stats.last_execution,
            Some(json!({
                "input_bytes": (base.len() + delta.len()) as u64,
                "selected_start": BACKEND_REFERENCE,
                "executed_backend": BACKEND_REFERENCE,
                "path": "native_hf_full_reencode",
            }))
        );

        let noop = store
            .append_patch(put.handle, "", appended.revision, &router)
            .unwrap();
        assert_eq!(noop.path, "noop");
        let after_noop = router.stats();
        assert_eq!(after_noop.execution_counts, stats.execution_counts);
        assert_eq!(after_noop.last_execution, stats.last_execution);
    }

    impl NativeGpuEngine for ClosingGpu {
        fn encode_ids(&self, text: &str) -> Result<Vec<u32>, NativeRuntimeError> {
            let ids = vec![0; text.len()];
            self.last_ids_ptr
                .store(ids.as_ptr() as usize, std::sync::atomic::Ordering::Relaxed);
            Ok(ids)
        }

        fn delivery(&self) -> &'static str {
            "prebuilt"
        }
    }

    #[test]
    fn gpu_state_seed_adopts_the_device_row_without_copying_it() {
        let gpu = Arc::new(ClosingGpu::default());
        let router = NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![1, 0],
            reference(),
            None,
            false,
            Some(NativeGpuSource::Open(
                Arc::clone(&gpu) as Arc<dyn NativeGpuEngine>
            )),
            false,
            true,
        )
        .unwrap();
        let mut store = toktier_store_core::SessionStore::with_defaults();
        let key = store.register_fingerprint([9u8; 32], 0).unwrap();
        let put = store.put(key, "aaaaaaa", &router).unwrap();
        // The complete-row snapshot is the engine's own allocation: the
        // router moved ownership into the store instead of cloning it.
        let snapshot = store.shared_all_ids(put.handle).unwrap();
        assert_eq!(snapshot.as_slice(), &[0u32; 7][..]);
        assert_eq!(
            snapshot.as_slice().as_ptr() as usize,
            gpu.last_ids_ptr.load(std::sync::atomic::Ordering::Relaxed),
            "state seed copied the device ID row"
        );
        assert_eq!(store.ids_materialization_count(), 0);
    }

    #[test]
    fn gpu_state_seed_names_lazy_checkpoints_and_materialized_spans_separately() {
        let router = NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![1, 0],
            reference(),
            None,
            false,
            Some(NativeGpuSource::Open(Arc::new(ClosingGpu::default()))),
            false,
            true,
        )
        .unwrap();
        let mut store = toktier_store_core::SessionStore::with_defaults();
        let key = store.register_fingerprint([7u8; 32], 0).unwrap();
        let put = store.put(key, "aaaa", &router).unwrap();
        assert_eq!(store.all_ids(put.handle).unwrap(), vec![0, 0, 0, 0]);
        let stats = router.stats();
        assert_eq!(stats.execution_counts.get(BACKEND_GPU), Some(&1));
        assert_eq!(
            stats
                .state_encode_counts
                .get("accelerated_with_lazy_span_checkpoints"),
            Some(&1)
        );
        assert_eq!(
            stats
                .state_encode_counts
                .get("accelerated_with_reconstructed_spans"),
            None
        );
        assert_eq!(
            stats.last_state_encode,
            Some(json!({
                "path": "accelerated_with_lazy_span_checkpoints",
                "backend": BACKEND_GPU,
                "input_bytes": 4,
            }))
        );
        assert_eq!(
            stats.last_execution,
            Some(json!({
                "input_bytes": 4,
                "selected_start": BACKEND_GPU,
                "executed_backend": BACKEND_GPU,
                "source": "prebuilt",
                "path": "gpu_full",
            }))
        );
        // The state route and the pair-based trait encode agree exactly.
        let pairs = <NativeRouter as SessionEncoder>::encode(&router, "aaaa").unwrap();
        assert_eq!(pairs.ids, vec![0, 0, 0, 0]);
        assert_eq!(pairs.spans, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        let stats = router.stats();
        assert_eq!(
            stats
                .state_encode_counts
                .get("accelerated_with_lazy_span_checkpoints"),
            Some(&1)
        );
        assert_eq!(
            stats
                .state_encode_counts
                .get("accelerated_with_reconstructed_spans"),
            Some(&1)
        );
        assert_eq!(
            stats.last_state_encode,
            Some(json!({
                "path": "accelerated_with_reconstructed_spans",
                "backend": BACKEND_GPU,
                "input_bytes": 4,
            }))
        );
    }

    #[test]
    fn gpu_state_seed_falls_back_when_the_span_bridge_rejects() {
        // MockGpu returns id 99, which is unknown to the reference table,
        // so the span bridge must reject and the seed must fall to the
        // reference row with the frozen hf_span_guard diagnostics.
        let router = NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![1, 0],
            reference(),
            None,
            false,
            Some(NativeGpuSource::Open(Arc::new(MockGpu))),
            false,
            true,
        )
        .unwrap();
        let mut store = toktier_store_core::SessionStore::with_defaults();
        let key = store.register_fingerprint([8u8; 32], 0).unwrap();
        let put = store.put(key, "aa", &router).unwrap();
        assert_eq!(store.all_ids(put.handle).unwrap(), vec![0, 0]);
        let stats = router.stats();
        assert_eq!(stats.fallback_counts.get(R_INPUT_GUARD_ROUTED), Some(&1));
        assert_guard_events_have_stage(&stats);
        assert_eq!(stats.events[0].detail["stage"], "span_bridge");
        assert_eq!(stats.state_encode_counts.get("hf_span_guard"), Some(&1));
        assert_eq!(
            stats.last_state_encode,
            Some(json!({
                "path": "hf_span_guard",
                "error": "ReferenceEngineError",
                "message": "tokenizer returned unknown id 99",
            }))
        );
    }

    #[test]
    fn reference_state_seed_keeps_the_frozen_diagnostic_shape() {
        let router = NativeRouter::new(
            vec![BACKEND_REFERENCE.to_owned()],
            vec![0],
            reference(),
            None,
            false,
            None,
            false,
            true,
        )
        .unwrap();
        let encoding = <NativeRouter as SessionEncoder>::encode(&router, "a").unwrap();
        assert_eq!(encoding.ids, vec![0]);
        let stats = router.stats();
        assert_eq!(
            stats.last_execution,
            Some(json!({
                "input_bytes": 1,
                "selected_start": BACKEND_REFERENCE,
                "executed_backend": BACKEND_REFERENCE,
                "source": "state_encode",
                "path": "hf_no_certified_span_bridge",
            }))
        );
        assert_eq!(
            stats.last_state_encode,
            Some(json!({"path": "hf_no_certified_span_bridge"}))
        );
    }

    // --------------------------------------------------------------
    // Seed overlap (PLAN/162 WP5): the digest scan joins the routed
    // seed encode through the bounded Rayon runner. The mock GPU
    // stands in for the device engine; the real-hardware pass stays
    // with the recertification batch.
    // --------------------------------------------------------------

    use std::sync::atomic::{AtomicBool, Ordering};
    use toktier_store_core::{ContentDigest, OverlapRunner, SessionStore};

    fn overlap_store(overlap: bool) -> SessionStore {
        let mut store = SessionStore::with_defaults();
        store.enable_content_tracking().unwrap();
        if overlap {
            store.set_seed_overlap(Some(Arc::new(RayonSeedOverlap)));
        }
        store
    }

    /// Wraps the production runner and flags when the digest scan has
    /// started, so a test can prove the scan runs while the device
    /// encode is still in flight.
    struct FlaggingOverlap {
        digest_started: Arc<AtomicBool>,
    }

    impl OverlapRunner for FlaggingOverlap {
        fn run_joined(&self, background: &mut (dyn FnMut() + Send), foreground: &mut dyn FnMut()) {
            let flag = Arc::clone(&self.digest_started);
            RayonSeedOverlap.run_joined(
                &mut || {
                    flag.store(true, Ordering::SeqCst);
                    background();
                },
                foreground,
            );
        }

        fn worker_count(&self) -> usize {
            RayonSeedOverlap.worker_count()
        }
    }

    /// A closing GPU whose encode does not return until the digest
    /// scan has observably started. If the overlap machinery ever ran
    /// the two sides sequentially, this encode would time out and the
    /// test would fail loudly instead of passing by accident.
    #[derive(Debug)]
    struct WaitingGpu {
        inner: ClosingGpu,
        digest_started: Arc<AtomicBool>,
    }

    impl NativeGpuEngine for WaitingGpu {
        fn encode_ids(&self, text: &str) -> Result<Vec<u32>, NativeRuntimeError> {
            let started = std::time::Instant::now();
            while !self.digest_started.load(Ordering::SeqCst) {
                if started.elapsed() > std::time::Duration::from_secs(10) {
                    return Err(NativeRuntimeError::new(
                        "the digest scan never started while the device encode was in flight",
                    ));
                }
                std::thread::yield_now();
            }
            self.inner.encode_ids(text)
        }

        fn delivery(&self) -> &'static str {
            "prebuilt"
        }
    }

    /// A GPU that always reports a device fault, for fallback parity.
    #[derive(Debug)]
    struct FaultingGpu;

    impl NativeGpuEngine for FaultingGpu {
        fn encode_ids(&self, _text: &str) -> Result<Vec<u32>, NativeRuntimeError> {
            Err(NativeRuntimeError::new("injected device fault"))
        }

        fn delivery(&self) -> &'static str {
            "prebuilt"
        }
    }

    fn gpu_router(gpu: Arc<dyn NativeGpuEngine>) -> NativeRouter {
        NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![1, 0],
            reference(),
            None,
            false,
            Some(NativeGpuSource::Open(gpu)),
            false,
            true,
        )
        .unwrap()
    }

    #[test]
    fn overlap_gpu_seed_scans_the_digest_while_the_device_encode_is_in_flight() {
        let digest_started = Arc::new(AtomicBool::new(false));
        let gpu = Arc::new(WaitingGpu {
            inner: ClosingGpu::default(),
            digest_started: Arc::clone(&digest_started),
        });
        let router = gpu_router(Arc::clone(&gpu) as Arc<dyn NativeGpuEngine>);
        let mut store = SessionStore::with_defaults();
        store.enable_content_tracking().unwrap();
        store.set_seed_overlap(Some(Arc::new(FlaggingOverlap { digest_started })));
        let key = store.register_fingerprint([20u8; 32], 0).unwrap();
        let text = "aaaaaaa";
        let put = store.put(key, text, &router).unwrap();
        // The seed stayed on the GPU row: the wait did not turn into a
        // fault-and-fallback.
        let stats = router.stats();
        assert_eq!(stats.execution_counts.get(BACKEND_GPU), Some(&1));
        assert_eq!(stats.fallback_counts.get(R_EXEC_FAULT), None);
        // Zero-copy adoption is preserved under overlap.
        let snapshot = store.shared_all_ids(put.handle).unwrap();
        assert_eq!(
            snapshot.as_slice().as_ptr() as usize,
            gpu.inner
                .last_ids_ptr
                .load(std::sync::atomic::Ordering::Relaxed),
            "overlap seed copied the device ID row"
        );
        assert_eq!(store.ids_materialization_count(), 0);
        // The digest joined with byte-identical results.
        assert_eq!(
            store.content_index_entry(put.handle).unwrap(),
            Some(ContentDigest::from_bytes(text.as_bytes()).unwrap().entry())
        );
    }

    #[test]
    fn overlap_gpu_fault_falls_back_identically_to_serial() {
        let text = "aaaa";
        let mut outcomes = Vec::new();
        for overlap in [false, true] {
            let router = gpu_router(Arc::new(FaultingGpu));
            let mut store = overlap_store(overlap);
            let key = store.register_fingerprint([21u8; 32], 0).unwrap();
            let put = store.put(key, text, &router).unwrap();
            let stats = router.stats();
            assert_eq!(stats.fallback_counts.get(R_EXEC_FAULT), Some(&1));
            assert_eq!(stats.execution_counts.get(BACKEND_REFERENCE), Some(&1));
            outcomes.push((
                store.all_ids(put.handle).unwrap(),
                store.content_index_entry(put.handle).unwrap(),
                store.export_session(put.handle).unwrap(),
                stats.last_execution,
                stats.last_state_encode,
            ));
        }
        assert_eq!(outcomes[0], outcomes[1], "fallback outcomes diverged");
    }

    #[test]
    fn overlap_span_bridge_rejection_falls_back_identically_to_serial() {
        let text = "aa";
        let mut outcomes = Vec::new();
        for overlap in [false, true] {
            let router = gpu_router(Arc::new(MockGpu));
            let mut store = overlap_store(overlap);
            let key = store.register_fingerprint([22u8; 32], 0).unwrap();
            let put = store.put(key, text, &router).unwrap();
            let stats = router.stats();
            assert_eq!(stats.fallback_counts.get(R_INPUT_GUARD_ROUTED), Some(&1));
            assert_guard_events_have_stage(&stats);
            assert_eq!(stats.state_encode_counts.get("hf_span_guard"), Some(&1));
            outcomes.push((
                store.all_ids(put.handle).unwrap(),
                store.content_index_entry(put.handle).unwrap(),
                store.export_session(put.handle).unwrap(),
                stats.last_state_encode,
            ));
        }
        assert_eq!(outcomes[0], outcomes[1], "guard outcomes diverged");
    }

    #[test]
    fn overlap_digest_fault_after_gpu_success_matches_serial_and_inserts_nothing() {
        let text = "aaaaaaa";
        let mut errors = Vec::new();
        for overlap in [false, true] {
            let router = gpu_router(Arc::new(ClosingGpu::default()));
            let mut store = overlap_store(overlap);
            store.inject_content_digest_fault(Some("injected digest failure".to_owned()));
            let key = store.register_fingerprint([23u8; 32], 0).unwrap();
            let error = store.put(key, text, &router).unwrap_err();
            assert!(store.list_handles().is_empty(), "partial session visible");
            // The store recovers on the same key once the fault clears.
            store.inject_content_digest_fault(None);
            let retry = store.put(key, text, &router).unwrap();
            assert_eq!(
                store.content_index_entry(retry.handle).unwrap(),
                Some(ContentDigest::from_bytes(text.as_bytes()).unwrap().entry())
            );
            errors.push(error);
        }
        assert_eq!(errors[0], errors[1], "digest fault identity diverged");
    }

    #[test]
    fn overlap_gpu_fault_and_digest_fault_together_match_serial() {
        // The device faults, the fallback row still encodes, and the
        // digest fault therefore decides the outcome in both modes.
        let text = "aaaa";
        let mut errors = Vec::new();
        for overlap in [false, true] {
            let router = gpu_router(Arc::new(FaultingGpu));
            let mut store = overlap_store(overlap);
            store.inject_content_digest_fault(Some("injected digest failure".to_owned()));
            let key = store.register_fingerprint([24u8; 32], 0).unwrap();
            let error = store.put(key, text, &router).unwrap_err();
            assert!(store.list_handles().is_empty(), "partial session visible");
            errors.push(error);
        }
        assert_eq!(errors[0], errors[1], "double-fault identity diverged");
    }

    #[test]
    fn overlap_concurrent_seeds_and_appends_share_the_bounded_pool() {
        let router = Arc::new(gpu_router(Arc::new(ClosingGpu::default())));
        let workers_before = rayon::current_num_threads();
        std::thread::scope(|scope| {
            for lane in 0..4usize {
                let router = Arc::clone(&router);
                scope.spawn(move || {
                    let mut store = overlap_store(true);
                    let key = store
                        .register_fingerprint([30 + lane as u8; 32], 0)
                        .unwrap();
                    for round in 0..6usize {
                        // Mixed traffic: larger seeds on even rounds,
                        // seed-plus-small-appends on odd rounds.
                        let seed = "a".repeat(512 * (lane + 1) + round);
                        let put = store.put(key, &seed, router.as_ref()).unwrap();
                        let mut text = seed.clone();
                        let mut revision = 0;
                        if round % 2 == 1 {
                            for _ in 0..3 {
                                let outcome = store
                                    .append_patch(put.handle, "aa", revision, router.as_ref())
                                    .unwrap();
                                revision = outcome.revision;
                                text.push_str("aa");
                            }
                        }
                        assert_eq!(
                            store.shared_all_ids(put.handle).unwrap().as_slice(),
                            &vec![0u32; text.len()][..],
                            "lane {lane} round {round} stream diverged"
                        );
                        assert_eq!(
                            store.content_index_entry(put.handle).unwrap(),
                            Some(ContentDigest::from_bytes(text.as_bytes()).unwrap().entry()),
                            "lane {lane} round {round} digest diverged"
                        );
                    }
                });
            }
        });
        assert_eq!(
            rayon::current_num_threads(),
            workers_before,
            "the bounded pool size changed under concurrent seeds"
        );
    }

    // --------------------------------------------------------------
    // Deferred engine opening: the opener runs on the first request
    // that actually routes to the GPU backend, exactly once, and a
    // failed open is latched with per-request fallback accounting.
    // --------------------------------------------------------------

    use std::sync::atomic::AtomicUsize;

    fn counting_deferred_router(
        opens: &Arc<AtomicUsize>,
        outcome: Result<(), &'static str>,
    ) -> NativeRouter {
        let opens = Arc::clone(opens);
        let opener: NativeGpuOpener = Box::new(move || {
            opens.fetch_add(1, Ordering::SeqCst);
            match outcome {
                Ok(()) => Ok(Arc::new(ClosingGpu::default()) as Arc<dyn NativeGpuEngine>),
                Err(message) => Err(NativeRuntimeError::new(message)),
            }
        });
        NativeRouter::new(
            vec![BACKEND_GPU.to_owned(), BACKEND_REFERENCE.to_owned()],
            vec![4, 0],
            reference(),
            None,
            false,
            Some(NativeGpuSource::Deferred(opener)),
            false,
            true,
        )
        .unwrap()
    }

    #[test]
    fn deferred_gpu_opens_on_the_first_routed_request_and_only_once() {
        let opens = Arc::new(AtomicUsize::new(0));
        let router = counting_deferred_router(&opens, Ok(()));
        assert!(!router.gpu_engine_opened());
        assert_eq!(router.gpu_engine_open_error(), None);

        // Below the crossover: the reference row serves and the device
        // stays untouched.
        let below = router.encode_ids("a", false).unwrap();
        assert_eq!(below.backend, BACKEND_REFERENCE);
        assert!(!router.gpu_engine_opened());
        assert_eq!(opens.load(Ordering::SeqCst), 0);

        // At the crossover: the first routed request opens the engine.
        let routed = router.encode_ids("aaaa", false).unwrap();
        assert_eq!(routed.backend, BACKEND_GPU);
        assert!(router.gpu_engine_opened());
        assert_eq!(opens.load(Ordering::SeqCst), 1);

        // Later routed requests reuse the one opened engine.
        let again = router.encode_ids("aaaaa", false).unwrap();
        assert_eq!(again.backend, BACKEND_GPU);
        assert_eq!(opens.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn deferred_gpu_batch_opens_only_when_a_row_routes_to_the_gpu() {
        let opens = Arc::new(AtomicUsize::new(0));
        let router = counting_deferred_router(&opens, Ok(()));

        let rows = router.encode_batch_ids(&["a", "aa"], false).unwrap();
        assert!(rows.iter().all(|row| row.backend == BACKEND_REFERENCE));
        assert!(!router.gpu_engine_opened());
        assert_eq!(opens.load(Ordering::SeqCst), 0);

        let rows = router.encode_batch_ids(&["a", "aaaa"], false).unwrap();
        assert_eq!(rows[0].backend, BACKEND_REFERENCE);
        assert_eq!(rows[1].backend, BACKEND_GPU);
        assert!(router.gpu_engine_opened());
        assert_eq!(opens.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn deferred_gpu_state_seed_opens_on_the_first_routed_seed() {
        let opens = Arc::new(AtomicUsize::new(0));
        let router = counting_deferred_router(&opens, Ok(()));
        let mut store = toktier_store_core::SessionStore::with_defaults();
        let key = store.register_fingerprint([40u8; 32], 0).unwrap();

        let short = store.put(key, "a", &router).unwrap();
        assert_eq!(store.all_ids(short.handle).unwrap(), vec![0]);
        assert!(!router.gpu_engine_opened());

        let routed = store.put(key, "aaaaaaa", &router).unwrap();
        assert_eq!(store.all_ids(routed.handle).unwrap(), vec![0u32; 7]);
        assert!(router.gpu_engine_opened());
        assert_eq!(opens.load(Ordering::SeqCst), 1);
        assert_eq!(
            router.stats().execution_counts.get(BACKEND_GPU),
            Some(&1),
            "the routed seed did not execute on the opened engine"
        );
    }

    #[test]
    fn concurrent_first_requests_share_one_deferred_open() {
        let opens = Arc::new(AtomicUsize::new(0));
        let router = Arc::new(counting_deferred_router(&opens, Ok(())));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let router = Arc::clone(&router);
                scope.spawn(move || {
                    let routed = router.encode_ids("aaaa", false).unwrap();
                    assert_eq!(routed.backend, BACKEND_GPU);
                });
            }
        });
        assert_eq!(opens.load(Ordering::SeqCst), 1, "the open was not shared");
        assert!(router.gpu_engine_opened());
        assert_eq!(
            router.stats().execution_counts.get(BACKEND_GPU),
            Some(&8),
            "a concurrent first request missed the shared engine"
        );
    }

    #[test]
    fn deferred_gpu_open_failure_is_latched_and_falls_back_per_request() {
        let opens = Arc::new(AtomicUsize::new(0));
        let router = counting_deferred_router(&opens, Err("injected open failure"));

        for round in 1..=2u64 {
            let routed = router.encode_ids("aaaa", false).unwrap();
            assert_eq!(routed.backend, BACKEND_REFERENCE);
            assert_eq!(routed.ids, vec![0u32; 4]);
            let stats = router.stats();
            assert_eq!(stats.fallback_counts.get(R_EXEC_FAULT), Some(&round));
        }
        // The failed open ran once; later requests reuse the latched
        // outcome instead of retrying the device.
        assert_eq!(opens.load(Ordering::SeqCst), 1);
        assert!(!router.gpu_engine_opened());
        assert_eq!(
            router.gpu_engine_open_error(),
            Some("injected open failure".to_owned())
        );
        let stats = router.stats();
        assert_eq!(stats.execution_counts.get(BACKEND_REFERENCE), Some(&2));
        let event = stats
            .events
            .iter()
            .find(|event| event.code == R_EXEC_FAULT)
            .expect("open failure event");
        assert_eq!(event.backend, BACKEND_GPU);
        assert_eq!(event.target, BACKEND_REFERENCE);
        assert_eq!(
            event.detail,
            json!({
                "error": "NativeGpuOpenError",
                "message": "injected open failure",
                "scope": "engine",
                "stage": "engine_open",
            })
        );
    }

    #[test]
    fn deferred_gpu_open_failure_state_seed_falls_back_identically() {
        let opens = Arc::new(AtomicUsize::new(0));
        let router = counting_deferred_router(&opens, Err("injected open failure"));
        let mut store = toktier_store_core::SessionStore::with_defaults();
        let key = store.register_fingerprint([41u8; 32], 0).unwrap();
        let put = store.put(key, "aaaa", &router).unwrap();
        assert_eq!(store.all_ids(put.handle).unwrap(), vec![0u32; 4]);
        let stats = router.stats();
        assert_eq!(stats.fallback_counts.get(R_EXEC_FAULT), Some(&1));
        assert_eq!(stats.execution_counts.get(BACKEND_REFERENCE), Some(&1));
        assert_eq!(opens.load(Ordering::SeqCst), 1);
        assert!(!router.gpu_engine_opened());
    }

    #[test]
    fn an_open_source_reports_opened_without_any_request() {
        let router = gpu_router(Arc::new(ClosingGpu::default()));
        assert!(router.gpu_engine_opened());
        assert_eq!(router.gpu_engine_open_error(), None);
    }
}

/// Literal segmentation on the router's ID carrier. The faces are small
/// byte-level faces;
/// the GPU stand-in encodes through the reference and refuses any input
/// that holds a literal, so a passing test also shows that only text
/// segments reach the device.
#[cfg(test)]
mod literal_segment_tests {
    use super::*;
    use crate::reference::test_support::{face_json, plain, TestNormalizer};
    use sha2::{Digest, Sha256};
    use tokenizers::AddedToken;

    const CHAT: [(&str, bool); 4] = [
        ("<|im_start|>", false),
        ("<|im_end|>", false),
        ("<think>", true),
        ("</think>", true),
    ];

    struct RefGpu {
        reference: Arc<ReferenceEngine>,
        literals: Vec<String>,
        fault_marker: Option<&'static str>,
        inputs: Mutex<Vec<String>>,
    }

    impl NativeGpuEngine for RefGpu {
        fn encode_ids(&self, text: &str) -> Result<Vec<u32>, NativeRuntimeError> {
            assert!(
                self.literals
                    .iter()
                    .all(|literal| !text.contains(literal.as_str())),
                "a literal reached the device: {text:?}"
            );
            self.inputs.lock().unwrap().push(text.to_owned());
            if self
                .fault_marker
                .is_some_and(|marker| text.contains(marker))
            {
                return Err(NativeRuntimeError::new("injected device fault"));
            }
            self.reference
                .encode_ids(text, false)
                .map_err(|error| NativeRuntimeError::new(error.to_string()))
        }

        fn delivery(&self) -> &'static str {
            "prebuilt"
        }
    }

    fn pclass() -> Vec<u8> {
        let mut pclass = vec![0u8; crate::N_CODEPOINTS];
        for value in b'a'..=b'z' {
            pclass[value as usize] = 2;
        }
        for value in b'A'..=b'Z' {
            pclass[value as usize] = 2;
        }
        for value in b'0'..=b'9' {
            pclass[value as usize] = 3;
        }
        pclass[b' ' as usize] = 1;
        pclass
    }

    fn chat_json() -> Vec<u8> {
        let tokens = CHAT
            .iter()
            .map(|(content, normalized)| plain(content, *normalized))
            .collect::<Vec<_>>();
        face_json(TestNormalizer::EmptySequence, &tokens)
    }

    fn fast_engine(json: &[u8], reference: &Arc<ReferenceEngine>) -> Arc<FastCpuEngine> {
        let digest = Sha256::digest(json)
            .iter()
            .map(|value| format!("{value:02x}"))
            .collect::<String>();
        let mut spec = crate::FastRepairSpec::new(
            "tiny_bytes".to_owned(),
            digest,
            4,
            2,
            reference.has_normalizer(),
        );
        spec.window_chars = 64;
        spec.max_retries = 2;
        Arc::new(
            FastCpuEngine::from_reference(json, Arc::clone(reference), spec, pclass()).unwrap(),
        )
    }

    struct Setup {
        router: NativeRouter,
        reference: Arc<ReferenceEngine>,
        gpu: Option<Arc<RefGpu>>,
    }

    fn setup(
        json: &[u8],
        chain: &[&str],
        thresholds: &[u64],
        fault: Option<&'static str>,
    ) -> Setup {
        let reference = Arc::new(ReferenceEngine::from_bytes(json).unwrap());
        let fast = chain
            .contains(&BACKEND_FAST_CPU)
            .then(|| fast_engine(json, &reference));
        let literals = reference
            .added_tokens()
            .into_iter()
            .map(|(_, content, _)| content)
            .collect::<Vec<_>>();
        let gpu = chain.contains(&BACKEND_GPU).then(|| {
            Arc::new(RefGpu {
                reference: Arc::clone(&reference),
                literals,
                fault_marker: fault,
                inputs: Mutex::new(Vec::new()),
            })
        });
        let router = NativeRouter::new(
            chain.iter().map(|name| (*name).to_owned()).collect(),
            thresholds.to_vec(),
            Arc::clone(&reference),
            fast,
            true,
            gpu.clone()
                .map(|gpu| NativeGpuSource::Open(gpu as Arc<dyn NativeGpuEngine>)),
            false,
            true,
        )
        .unwrap();
        Setup {
            router,
            reference,
            gpu,
        }
    }

    fn chat(turns: &[&str]) -> String {
        let mut text = String::new();
        for (index, turn) in turns.iter().enumerate() {
            let role = if index % 2 == 0 { "user" } else { "assistant" };
            text.push_str(&format!("<|im_start|>{role}\n"));
            if index % 2 == 1 {
                text.push_str("<think>\nreasoning here\n</think>\n");
            }
            text.push_str(turn);
            text.push_str("<|im_end|>\n");
        }
        text
    }

    fn segments(stats: &RuntimeStats) -> Value {
        stats.last_execution.as_ref().unwrap()["segments"].clone()
    }

    fn codes(stats: &RuntimeStats) -> Vec<String> {
        stats
            .events
            .iter()
            .map(|event| event.code.clone())
            .collect()
    }

    #[test]
    fn a_literal_bearing_document_is_segmented_on_the_fast_cpu_route() {
        let setup = setup(
            &chat_json(),
            &[BACKEND_FAST_CPU, BACKEND_REFERENCE],
            &[0, 0],
            None,
        );
        assert!(setup.reference.literal_segmentation_served());
        let text = chat(&[
            "hello there, caf\u{e9} \u{4e2d}\u{6587}",
            "a reply with 42 words",
        ]);
        let routed = setup.router.encode_ids(&text, false).unwrap();
        assert_eq!(
            routed.ids,
            setup.reference.encode_ids(&text, false).unwrap()
        );
        assert_eq!(routed.backend, BACKEND_FAST_CPU);
        assert_eq!(routed.path.as_deref(), Some("gigatoken_segmented"));
        assert_eq!(routed.reason, None);
        let stats = setup.router.stats();
        assert!(codes(&stats).is_empty(), "{:?}", stats.events);
        let facts = segments(&stats);
        assert_eq!(facts["literals"], 6);
        assert_eq!(facts["bytes"]["hf"], 0);
        assert_eq!(facts["bytes"]["gpu"], 0);
        assert!(facts["bytes"]["fast_cpu"].as_u64().unwrap() > 0);
        assert_eq!(
            facts["bytes"]["fast_cpu"].as_u64().unwrap() + facts["literal_bytes"].as_u64().unwrap(),
            text.len() as u64
        );
        let fast_stats = setup.router.fast_cpu.as_ref().unwrap().stats();
        assert_eq!(fast_stats.path_counts.get("gigatoken_segmented"), Some(&1));
        // A literal at offset 0, at the end, and adjacent literals.
        for text in [
            "<|im_start|>",
            "<|im_end|><|im_start|>x",
            "a<think></think>",
            "</think>",
        ] {
            let routed = setup.router.encode_ids(text, false).unwrap();
            assert_eq!(
                routed.ids,
                setup.reference.encode_ids(text, false).unwrap(),
                "{text:?}"
            );
            assert_eq!(routed.path.as_deref(), Some("gigatoken_segmented"));
        }
    }

    #[test]
    fn a_literal_free_document_keeps_its_unsegmented_record() {
        let setup = setup(
            &chat_json(),
            &[BACKEND_FAST_CPU, BACKEND_REFERENCE],
            &[0, 0],
            None,
        );
        let text = "plain text with <|im_star and think> fragments";
        let routed = setup.router.encode_ids(text, false).unwrap();
        assert_eq!(routed.path.as_deref(), Some("gigatoken_full"));
        let stats = setup.router.stats();
        assert_eq!(
            stats.last_execution,
            Some(json!({
                "input_bytes": text.len(),
                "selected_start": BACKEND_FAST_CPU,
                "executed_backend": BACKEND_FAST_CPU,
                "source": "gigatoken",
                "path": "gigatoken_full",
            }))
        );
        assert!(stats.events.is_empty());
    }

    #[test]
    fn batch_rows_holding_literals_are_segmented_in_their_workers() {
        let setup = setup(
            &chat_json(),
            &[BACKEND_FAST_CPU, BACKEND_REFERENCE],
            &[0, 0],
            None,
        );
        let texts = [
            chat(&["first"]),
            "no literal at all".to_owned(),
            chat(&["second", "third"]),
            String::new(),
        ];
        let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        let rows = setup.router.encode_batch_ids(&refs, false).unwrap();
        let paths = rows
            .iter()
            .map(|row| row.path.clone().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            [
                "gigatoken_segmented",
                "gigatoken_full",
                "gigatoken_segmented",
                "gigatoken_full"
            ]
        );
        for (row, text) in rows.iter().zip(&texts) {
            assert_eq!(row.ids, setup.reference.encode_ids(text, false).unwrap());
        }
        let stats = setup.router.stats();
        assert_eq!(stats.execution_counts.get(BACKEND_FAST_CPU), Some(&4));
        assert!(stats.events.is_empty());
    }

    #[test]
    fn gpu_headed_documents_route_each_segment_by_its_own_size() {
        let crossover = 64u64;
        let setup = setup(
            &chat_json(),
            &[BACKEND_GPU, BACKEND_FAST_CPU, BACKEND_REFERENCE],
            &[crossover, 0, 0],
            None,
        );
        let long = "the quick brown fox jumps over the lazy dog ".repeat(4);
        let text = chat(&[&long, "short"]);
        let routed = setup.router.encode_ids(&text, false).unwrap();
        assert_eq!(
            routed.ids,
            setup.reference.encode_ids(&text, false).unwrap()
        );
        assert_eq!(routed.backend, BACKEND_GPU);
        assert_eq!(routed.path.as_deref(), Some("gpu_segmented"));
        assert_eq!(routed.reason, None);
        let stats = setup.router.stats();
        assert!(!codes(&stats).contains(&R_INPUT_BELOW_GPU_THRESHOLD.to_owned()));
        let facts = segments(&stats);
        assert!(facts["bytes"]["gpu"].as_u64().unwrap() >= crossover);
        assert!(facts["bytes"]["fast_cpu"].as_u64().unwrap() > 0);
        assert_eq!(facts["bytes"]["hf"], 0);
        let inputs = setup.gpu.as_ref().unwrap().inputs.lock().unwrap().clone();
        assert!(inputs.iter().all(|input| input.len() as u64 >= crossover));

        // At or above the crossover, every segment below it: the corrected
        // CPU engine runs the document, and no below-threshold code is
        // recorded for it.
        let small = chat(&["one small turn", "and another", "a third one"]);
        assert!(small.len() as u64 >= crossover);
        let routed = setup.router.encode_ids(&small, false).unwrap();
        assert_eq!(
            routed.ids,
            setup.reference.encode_ids(&small, false).unwrap()
        );
        assert_eq!(routed.path.as_deref(), Some("gigatoken_segmented"));
        assert_eq!(routed.reason, None);
        let stats = setup.router.stats();
        assert!(!codes(&stats).contains(&R_INPUT_BELOW_GPU_THRESHOLD.to_owned()));
        assert!(segments(&stats)["largest_text_bytes"].as_u64().unwrap() < crossover);
        assert_eq!(
            stats.last_execution.as_ref().unwrap()["selected_start"],
            BACKEND_GPU
        );

        // Below the crossover the document-level decision is recorded as
        // before, then the engine segments it.
        let tiny = "<|im_start|>hi";
        let routed = setup.router.encode_ids(tiny, false).unwrap();
        assert_eq!(routed.reason, Some(R_INPUT_BELOW_GPU_THRESHOLD));
        assert_eq!(routed.path.as_deref(), Some("gigatoken_segmented"));
    }

    #[test]
    fn a_segment_the_device_faults_on_falls_back_alone() {
        let setup = setup(
            &chat_json(),
            &[BACKEND_GPU, BACKEND_FAST_CPU, BACKEND_REFERENCE],
            &[32, 0, 0],
            Some("FAULT"),
        );
        let text = chat(&[
            "a long enough first turn for the device to run",
            "a long enough FAULT turn that the device refuses",
        ]);
        let routed = setup.router.encode_ids(&text, false).unwrap();
        assert_eq!(
            routed.ids,
            setup.reference.encode_ids(&text, false).unwrap()
        );
        assert_eq!(routed.path.as_deref(), Some("gpu_segmented"));
        assert_eq!(routed.reason, Some(R_EXEC_FAULT));
        let stats = setup.router.stats();
        let fault = stats
            .events
            .iter()
            .find(|event| event.code == R_EXEC_FAULT)
            .unwrap();
        assert_eq!(fault.detail["scope"], "segment");
        // Every device segment faulting: the whole document continues on
        // the next backend, segmented there.
        let all_fault = chat(&[
            "FAULT on a long enough first turn",
            "FAULT again, long enough here",
        ]);
        let routed = setup.router.encode_ids(&all_fault, false).unwrap();
        assert_eq!(
            routed.ids,
            setup.reference.encode_ids(&all_fault, false).unwrap()
        );
        assert_eq!(routed.path.as_deref(), Some("gigatoken_segmented"));
        assert_eq!(routed.reason, Some(R_EXEC_FAULT));
    }

    #[test]
    fn a_gpu_reference_chain_answers_small_segments_on_the_reference() {
        let setup = setup(
            &chat_json(),
            &[BACKEND_GPU, BACKEND_REFERENCE],
            &[48, 0],
            None,
        );
        let small = chat(&["one small turn", "and another one"]);
        assert!(small.len() >= 48);
        let routed = setup.router.encode_ids(&small, false).unwrap();
        assert_eq!(
            routed.ids,
            setup.reference.encode_ids(&small, false).unwrap()
        );
        assert_eq!(routed.path.as_deref(), Some("hf_added_token"));
        assert_eq!(routed.reason, Some(R_INPUT_ADDED_TOKEN));
        let long = "the quick brown fox jumps over the lazy dog ".repeat(2);
        let mixed = chat(&[&long, "short"]);
        let routed = setup.router.encode_ids(&mixed, false).unwrap();
        assert_eq!(
            routed.ids,
            setup.reference.encode_ids(&mixed, false).unwrap()
        );
        assert_eq!(routed.path.as_deref(), Some("gpu_segmented"));
        let facts = segments(&setup.router.stats());
        assert!(facts["bytes"]["gpu"].as_u64().unwrap() > 0);
        assert!(facts["bytes"]["hf"].as_u64().unwrap() > 0);
    }

    #[test]
    fn gpu_headed_batch_rows_with_literals_take_the_single_document_path() {
        let setup = setup(
            &chat_json(),
            &[BACKEND_GPU, BACKEND_FAST_CPU, BACKEND_REFERENCE],
            &[64, 0, 0],
            None,
        );
        let long = "the quick brown fox jumps over the lazy dog ".repeat(4);
        let texts = [
            chat(&[&long]),
            long.clone(),
            "<|im_start|>x".to_owned(),
            chat(&["a", "b"]),
        ];
        let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        let rows = setup.router.encode_batch_ids(&refs, false).unwrap();
        let paths = rows
            .iter()
            .map(|row| row.path.clone().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            [
                "gpu_segmented",
                "gpu_full",
                "gigatoken_segmented",
                "gigatoken_segmented"
            ]
        );
        for (row, text) in rows.iter().zip(&texts) {
            assert_eq!(row.ids, setup.reference.encode_ids(text, false).unwrap());
        }
        let stats = setup.router.stats();
        let executions = stats.execution_counts.values().sum::<u64>();
        assert_eq!(executions, 4);
    }

    /// On a face outside the served class a gate match goes to the
    /// reference even when the reference declines the literal there.
    #[test]
    fn an_unserved_face_answers_every_gate_match_on_the_reference() {
        let json = face_json(
            TestNormalizer::None,
            &[AddedToken::from("xq", true)
                .single_word(true)
                .normalized(false)],
        );
        let setup = setup(&json, &[BACKEND_FAST_CPU, BACKEND_REFERENCE], &[0, 0], None);
        assert!(!setup.reference.literal_segmentation_served());
        for text in ["axq", "a xq b", "xqxq"] {
            let routed = setup.router.encode_ids(text, false).unwrap();
            assert_eq!(
                routed.ids,
                setup.reference.encode_ids(text, false).unwrap(),
                "{text:?}"
            );
            assert_eq!(routed.path.as_deref(), Some("hf_added_token"));
            let rows = setup.router.encode_batch_ids(&[text], false).unwrap();
            assert_eq!(rows[0].ids, routed.ids);
            let fast = setup.router.fast_cpu.as_ref().unwrap();
            assert_eq!(fast.encode_ids(text).unwrap(), routed.ids);
        }
        // The engine core's matcher ignores the flag, so repair is off.
        assert_eq!(
            setup.router.witness_category(),
            WitnessCategory::NoneFullReencode
        );
    }

    fn seed_then_append(
        encoder: &dyn SessionEncoder,
        seed: &str,
        delta: &str,
    ) -> (Vec<u32>, String) {
        let mut tail = TailState::new();
        encoder.append(&mut tail, seed).unwrap();
        let report = encoder.append(&mut tail, delta).unwrap();
        (tail.ids().to_vec(), report.path)
    }

    /// The append path on cross-phase overlap faces: a session seeded
    /// with literal-free text and extended by overlapping literals.
    #[test]
    fn a_face_whose_engine_matcher_disagrees_never_repairs_over_literals() {
        for later in ["x<zq>y", "kk<z"] {
            let json = face_json(
                TestNormalizer::EmptySequence,
                &[plain("<zq>", false), plain(later, true)],
            );
            let setup = setup(&json, &[BACKEND_FAST_CPU, BACKEND_REFERENCE], &[0, 0], None);
            assert!(setup.reference.literal_segmentation_served());
            assert!(!setup.reference.engine_matcher_agrees());
            assert_eq!(
                setup.router.witness_category(),
                WitnessCategory::NoneFullReencode
            );
            let seed = "lorem ipsum dolor sit amet ".repeat(70);
            let delta = format!("note {later}<zq>y end kk<zq> and x<zq>y again");
            let full = format!("{seed}{delta}");
            let expected = setup.reference.encode_ids(&full, false).unwrap();
            let (ids, _path) = seed_then_append(&setup.router, &seed, &delta);
            assert_eq!(ids, expected, "router {later}");
            let fast = setup.router.fast_cpu.as_ref().unwrap();
            let (ids, path) = seed_then_append(fast.as_ref(), &seed, &delta);
            assert_eq!(ids, expected, "engine {later}");
            assert_eq!(path, "hf_full_added_token");
            // Full encodes are segmented and exact.
            let routed = setup.router.encode_ids(&full, false).unwrap();
            assert_eq!(routed.ids, expected);
            assert_eq!(routed.path.as_deref(), Some("gigatoken_segmented"));
        }
        // The harmless direction keeps repair, and it is exact.
        let json = face_json(
            TestNormalizer::EmptySequence,
            &[
                plain("<\u{ff5c}/table>\u{ff5c}", false),
                plain("\u{ff5c}DSML\u{ff5c}", true),
            ],
        );
        let setup = setup(&json, &[BACKEND_FAST_CPU, BACKEND_REFERENCE], &[0, 0], None);
        assert!(setup.reference.engine_matcher_agrees());
        assert_eq!(
            setup.router.witness_category(),
            WitnessCategory::BpeSyncTransition
        );
        let seed = "lorem ipsum dolor sit amet ".repeat(70);
        let delta = " a <\u{ff5c}/table>\u{ff5c}DSML\u{ff5c} b \u{ff5c}DSML\u{ff5c} c".to_owned();
        let expected = setup
            .reference
            .encode_ids(&format!("{seed}{delta}"), false)
            .unwrap();
        let (ids, _path) = seed_then_append(&setup.router, &seed, &delta);
        assert_eq!(ids, expected);
    }
}

/// Literal segmentation on the span-carrying carriers and the state route:
/// session seeds, materialized state encodes, and appends after a
/// segmented seed, against the reference's ids and character offsets.
#[cfg(test)]
mod literal_state_tests {
    use super::*;
    use crate::reference::test_support::{face_json, plain, TestNormalizer};
    use sha2::{Digest, Sha256};

    struct RefGpu(Arc<ReferenceEngine>, Vec<String>);

    impl NativeGpuEngine for RefGpu {
        fn encode_ids(&self, text: &str) -> Result<Vec<u32>, NativeRuntimeError> {
            assert!(
                self.1
                    .iter()
                    .all(|literal| !text.contains(literal.as_str())),
                "a literal reached the device: {text:?}"
            );
            self.0
                .encode_ids(text, false)
                .map_err(|error| NativeRuntimeError::new(error.to_string()))
        }

        fn delivery(&self) -> &'static str {
            "prebuilt"
        }
    }

    fn router(normalizer: TestNormalizer, gpu: bool) -> (NativeRouter, Arc<ReferenceEngine>) {
        let tokens = [
            plain("<|im_start|>", false),
            plain("<|im_end|>", false),
            plain("<think>", true),
            plain("\u{e9}t\u{e9}", false),
        ];
        let tokens = match normalizer {
            // A normalized token under NFC is outside the served class.
            TestNormalizer::Nfc => tokens[..2].to_vec(),
            _ => tokens.to_vec(),
        };
        let json = face_json(normalizer, &tokens);
        let reference = Arc::new(ReferenceEngine::from_bytes(&json).unwrap());
        let digest = Sha256::digest(&json)
            .iter()
            .map(|value| format!("{value:02x}"))
            .collect::<String>();
        let mut spec = crate::FastRepairSpec::new(
            "tiny_bytes".to_owned(),
            digest,
            4,
            2,
            reference.has_normalizer(),
        );
        spec.window_chars = 64;
        spec.max_retries = 2;
        let mut pclass = vec![0u8; crate::N_CODEPOINTS];
        for value in b'a'..=b'z' {
            pclass[value as usize] = 2;
        }
        pclass[b' ' as usize] = 1;
        let fast = Arc::new(
            FastCpuEngine::from_reference(&json, Arc::clone(&reference), spec, pclass).unwrap(),
        );
        let literals = tokens.iter().map(|token| token.content.clone()).collect();
        let (chain, thresholds, source) = if gpu {
            (
                vec![BACKEND_GPU, BACKEND_FAST_CPU, BACKEND_REFERENCE],
                vec![48, 0, 0],
                Some(NativeGpuSource::Open(Arc::new(RefGpu(
                    Arc::clone(&reference),
                    literals,
                )))),
            )
        } else {
            (vec![BACKEND_FAST_CPU, BACKEND_REFERENCE], vec![0, 0], None)
        };
        let router = NativeRouter::new(
            chain.into_iter().map(str::to_owned).collect(),
            thresholds,
            Arc::clone(&reference),
            Some(fast),
            true,
            source,
            false,
            true,
        )
        .unwrap();
        (router, reference)
    }

    fn chat(seed: u64) -> String {
        let bodies = [
            "hello there, a fairly long user turn with words",
            "short",
            "caf\u{e9} \u{4e2d}\u{6587} mixed text that runs long enough",
            "x",
        ];
        let mut text = String::new();
        for turn in 0..4u64 {
            let body = bodies[((seed + turn) % 4) as usize];
            text.push_str("<|im_start|>user\n");
            if turn % 2 == 1 {
                text.push_str("<think>plan</think>\u{e9}t\u{e9}");
            }
            text.push_str(body);
            text.push_str("<|im_end|>\n");
        }
        text
    }

    fn assert_state(tail: &TailState, reference: &ReferenceEngine, text: &str) {
        let expected = reference.encode_core(text).unwrap();
        assert_eq!(tail.ids(), expected.ids.as_slice(), "{text:?}");
        assert_eq!(tail.spans(), expected.spans, "{text:?}");
    }

    #[test]
    fn segmented_seeds_and_appends_equal_the_reference_state() {
        for normalizer in [
            TestNormalizer::None,
            TestNormalizer::EmptySequence,
            TestNormalizer::Nfc,
        ] {
            for gpu in [false, true] {
                let (router, reference) = router(normalizer, gpu);
                for seed in 0..4u64 {
                    let text = chat(seed);
                    // Lazy seed (append on an empty tail), then appends that
                    // carry literals, one straddling the boundary.
                    let mut tail = TailState::new();
                    let report = router.append(&mut tail, &text).unwrap();
                    assert!(
                        report.path == "gigatoken_segmented" || report.path == "gpu_segmented",
                        "{}",
                        report.path
                    );
                    assert_state(&tail, &reference, &text);
                    let mut full = text.clone();
                    for delta in [
                        "<|im_start|>assistant\n",
                        "<thi",
                        "nk>ok</think> done<|im_end|>",
                    ] {
                        router.append(&mut tail, delta).unwrap();
                        full.push_str(delta);
                        assert_state(&tail, &reference, &full);
                    }
                    // Materialized state encode.
                    let encoded = router.encode(&text).unwrap();
                    let expected = reference.encode_core(&text).unwrap();
                    assert_eq!(encoded, expected);
                    let stats = router.stats();
                    let last = stats.last_execution.as_ref().unwrap();
                    assert!(last.get("segments").is_some(), "{last:?}");
                    assert!(
                        !stats
                            .events
                            .iter()
                            .any(|event| event.code == R_INPUT_ADDED_TOKEN),
                        "{:?}",
                        stats.events
                    );
                }
            }
        }
    }

    /// A text that is not stable under NFC although its byte length is:
    /// three decomposed accents early (one byte shorter each after NFC)
    /// and one U+0958 late (three bytes longer after NFC). A device row
    /// of the normalized text closes over it in bytes, and its spans
    /// would be shifted between the two.
    fn nfc_balanced(body: &str) -> String {
        format!("cafe\u{301} re\u{301}sume\u{301} {body} {body} {body} \u{958} end of the text")
    }

    #[test]
    fn a_device_row_of_a_text_unstable_under_the_normalizer_is_not_adopted() {
        let (router, reference) = router(TestNormalizer::Nfc, true);
        let body = "the committee met on tuesday and agreed to the plan";
        for text in [
            nfc_balanced(body),
            format!("<|im_start|>user\n{}<|im_end|>\n", nfc_balanced(body)),
        ] {
            assert_ne!(reference.normalize(&text).unwrap(), text);
            assert_eq!(reference.normalize(&text).unwrap().len(), text.len());
            // Lazy seed, then an append.
            let mut tail = TailState::new();
            let report = router.append(&mut tail, &text).unwrap();
            assert_eq!(report.path, "hf_span_guard", "{text:?}");
            assert_state(&tail, &reference, &text);
            let delta = " one more sentence arrives after the seed";
            router.append(&mut tail, delta).unwrap();
            assert_state(&tail, &reference, &format!("{text}{delta}"));
            // Materialized state encode.
            let encoded = router.encode(&text).unwrap();
            assert_eq!(encoded, reference.encode_core(&text).unwrap());
            let stats = router.stats();
            assert_eq!(
                stats.last_state_encode.as_ref().unwrap()["path"],
                "hf_span_guard"
            );
            assert_eq!(stats.events.last().unwrap().detail["stage"], "span_bridge");
        }
        // The NFC-stable control is still adopted from the device.
        let stable = format!("caf\u{e9} r\u{e9}sum\u{e9} {body} {body} {body} end");
        let mut tail = TailState::new();
        let report = router.append(&mut tail, &stable).unwrap();
        assert_eq!(report.path, "gpu_full");
        assert_state(&tail, &reference, &stable);
    }

    #[test]
    fn the_engine_session_surface_segments_its_seed() {
        let (router, reference) = router(TestNormalizer::EmptySequence, false);
        let fast = router.fast_cpu.as_ref().unwrap();
        let text = chat(1);
        let mut tail = TailState::new();
        let report = fast.append(&mut tail, &text).unwrap();
        assert_eq!(report.path, "cold_full");
        assert_state(&tail, &reference, &text);
        assert_eq!(
            fast.encode(&text).unwrap(),
            reference.encode_core(&text).unwrap()
        );
        assert_eq!(
            fast.stats().path_counts.get("gigatoken_segmented"),
            Some(&2)
        );
    }
}
