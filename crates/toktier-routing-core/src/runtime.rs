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

use crate::{FastCpuEngine, FastEncodeSource, FastSeedPayload, ReferenceEngine};

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedIds {
    pub ids: Vec<u32>,
    pub backend: String,
    pub source: Option<String>,
    pub path: Option<String>,
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
        Ok(has_added.then_some(ids))
    }

    fn exact_added_reference_state(&self, text: &str) -> Result<Option<SoaEncoding>, EngineError> {
        if !self.added.candidate(text) {
            return Ok(None);
        }
        let (encoding, has_added) = self
            .reference
            .encode_state_soa_with_added_flag(text)
            .map_err(|error| EngineError(error.to_string()))?;
        Ok(has_added.then_some(encoding))
    }

    fn reference_state(&self, text: &str) -> Result<SoaEncoding, EngineError> {
        self.reference
            .encode_state_soa(text)
            .map_err(|error| EngineError(error.to_string()))
    }

    fn below_gpu_threshold_event(&self, input_bytes: u64, start: usize) {
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
        }
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
    fn fast_cpu_reference_path(&self, source: FastEncodeSource) -> &'static str {
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
        path
    }

    /// Stateless ID route: no span array, byte-to-character map, or pair
    /// vector is constructed anywhere on this path. Ledger events, counters,
    /// and diagnostic shapes are identical to the historical combined route
    /// with `state_encode=false`.
    fn route_ids_internal(&self, text: &str) -> Result<RoutedIds, EngineError> {
        let input_bytes = text.len() as u64;
        let start = self.start_index(input_bytes);
        self.below_gpu_threshold_event(input_bytes, start);
        if self.chain[start] != BackendKind::Reference {
            if let Some(ids) = self.exact_added_reference_ids(text)? {
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
                    None,
                    Some("hf_added_token"),
                );
                return Ok(RoutedIds {
                    ids,
                    backend: BACKEND_REFERENCE.to_owned(),
                    source: None,
                    path: Some("hf_added_token".to_owned()),
                });
            }
        }
        for index in start..self.chain.len() {
            match self.chain[index] {
                BackendKind::Gpu => {
                    let cell = self.gpu.as_ref().expect("constructor gate");
                    match cell.engine() {
                        Err(error) => self.gpu_open_fault_event(index, &error),
                        Ok(gpu) => match gpu.encode_ids(text) {
                            Ok(ids) => {
                                self.record_execution(
                                    BACKEND_GPU,
                                    input_bytes,
                                    start,
                                    Some(gpu.delivery()),
                                    Some("gpu_full"),
                                );
                                return Ok(RoutedIds {
                                    ids,
                                    backend: BACKEND_GPU.to_owned(),
                                    source: Some(gpu.delivery().to_owned()),
                                    path: Some("gpu_full".to_owned()),
                                });
                            }
                            Err(error) => self.gpu_fault_event(index, &error),
                        },
                    }
                }
                BackendKind::FastCpu => {
                    let fast = self.fast_cpu.as_ref().expect("constructor gate");
                    match fast.encode_ids_with_source(text) {
                        Err(error) => self.fast_cpu_fault_event(index, &error),
                        Ok((ids, FastEncodeSource::Gigatoken)) => {
                            self.record_execution(
                                BACKEND_FAST_CPU,
                                input_bytes,
                                start,
                                Some("gigatoken"),
                                Some("gigatoken_full"),
                            );
                            return Ok(RoutedIds {
                                ids,
                                backend: BACKEND_FAST_CPU.to_owned(),
                                source: Some("gigatoken".to_owned()),
                                path: Some("gigatoken_full".to_owned()),
                            });
                        }
                        Ok((ids, source)) => {
                            let path = self.fast_cpu_reference_path(source);
                            self.record_execution(
                                BACKEND_REFERENCE,
                                input_bytes,
                                start,
                                None,
                                Some(path),
                            );
                            return Ok(RoutedIds {
                                ids,
                                backend: BACKEND_REFERENCE.to_owned(),
                                source: None,
                                path: Some(path.to_owned()),
                            });
                        }
                    }
                }
                BackendKind::Reference => {
                    let ids = self
                        .reference
                        .encode_ids(text, false)
                        .map_err(|error| EngineError(error.to_string()))?;
                    self.record_execution(
                        BACKEND_REFERENCE,
                        input_bytes,
                        start,
                        None,
                        Some("hf_full"),
                    );
                    return Ok(RoutedIds {
                        ids,
                        backend: BACKEND_REFERENCE.to_owned(),
                        source: None,
                        path: Some("hf_full".to_owned()),
                    });
                }
            }
        }
        Err(EngineError(
            "native fallback chain terminated without hf".to_owned(),
        ))
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
        if self.chain[start] != BackendKind::Reference {
            if let Some(encoding) = self.exact_added_reference_state(text)? {
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
                return Ok((StatePayload::Soa(encoding), "hf_added_token".to_owned()));
            }
        }
        for index in start..self.chain.len() {
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
                            // Both modes share one fail-closed acceptance
                            // decision over the exact ID row; only the
                            // successful payload representation differs.
                            let accepted = match mode {
                                StateMode::LazySeed => self
                                    .reference
                                    .verify_ids_close(text, &ids)
                                    .and_then(|()| self.reference.raw_byte_lengths_arc())
                                    .map(|table| StatePayload::Lazy { ids, table }),
                                StateMode::Materialized => self
                                    .reference
                                    .spans_soa_for_ids(text, &ids)
                                    .map(|(span_starts, span_ends)| {
                                        StatePayload::Soa(SoaEncoding {
                                            ids,
                                            span_starts,
                                            span_ends,
                                        })
                                    }),
                            };
                            match accepted {
                                Ok(payload) => {
                                    let state_encode_path = payload.accelerated_state_encode_path();
                                    self.record_execution(
                                        BACKEND_GPU,
                                        input_bytes,
                                        start,
                                        Some(gpu.delivery()),
                                        Some("gpu_full"),
                                    );
                                    self.record_state_encode(
                                        state_encode_path,
                                        json!({
                                            "path": state_encode_path,
                                            "backend": BACKEND_GPU,
                                            "input_bytes": input_bytes,
                                        }),
                                    );
                                    return Ok((payload, "gpu_full".to_owned()));
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
                                    return Ok((
                                        StatePayload::Soa(reference),
                                        "hf_span_guard".to_owned(),
                                    ));
                                }
                            }
                        }
                        Err(error) => self.gpu_fault_event(index, &error),
                    }
                }
                BackendKind::FastCpu => {
                    let fast = self.fast_cpu.as_ref().expect("constructor gate");
                    let outcome = match mode {
                        StateMode::LazySeed => fast.encode_seed_with_source(text),
                        StateMode::Materialized => {
                            fast.encode_state_with_source(text)
                                .map(|(encoding, source)| {
                                    (FastSeedPayload::Reference(encoding), source)
                                })
                        }
                    };
                    match outcome {
                        Err(error) => self.fast_cpu_fault_event(index, &error),
                        Ok((payload, FastEncodeSource::Gigatoken)) => {
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
                            self.record_execution(
                                BACKEND_FAST_CPU,
                                input_bytes,
                                start,
                                Some("gigatoken"),
                                Some("gigatoken_full"),
                            );
                            self.record_state_encode(
                                state_encode_path,
                                json!({
                                    "path": state_encode_path,
                                    "backend": BACKEND_FAST_CPU,
                                    "input_bytes": input_bytes,
                                }),
                            );
                            return Ok((payload, "gigatoken_full".to_owned()));
                        }
                        Ok((payload, source)) => {
                            let FastSeedPayload::Reference(encoding) = payload else {
                                return Err(EngineError(
                                    "reference-routed fast encode returned an ID-only payload"
                                        .to_owned(),
                                ));
                            };
                            let payload = StatePayload::Soa(encoding);
                            let state_encode_path = payload.accelerated_state_encode_path();
                            let path = self.fast_cpu_reference_path(source);
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
            }
            if self.chain[starts[row]] == BackendKind::Reference {
                continue;
            }
            if let Some(ids) = self.exact_added_reference_ids(text)? {
                self.record_event(
                    R_INPUT_ADDED_TOKEN,
                    self.chain[starts[row]].as_str(),
                    BACKEND_REFERENCE,
                    json!({}),
                );
                output[row] = Some(RoutedIds {
                    ids,
                    backend: BACKEND_REFERENCE.to_owned(),
                    source: None,
                    path: Some("hf_added_token".to_owned()),
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
                                next[row] += 1;
                            }
                        }
                    }
                }
                BackendKind::FastCpu => {
                    let fast = self.fast_cpu.as_ref().expect("constructor gate");
                    match fast.encode_batch_ids_with_source(&batch) {
                        Ok(rows) => {
                            for (row, (ids, source)) in indices.into_iter().zip(rows) {
                                let routed = match source {
                                    FastEncodeSource::Gigatoken => RoutedIds {
                                        ids,
                                        backend: BACKEND_FAST_CPU.to_owned(),
                                        source: Some("gigatoken".to_owned()),
                                        path: Some("gigatoken_full".to_owned()),
                                    },
                                    source @ (FastEncodeSource::ReferenceAddedToken
                                    | FastEncodeSource::ReferenceEngineGuard) => {
                                        let path = self.fast_cpu_reference_path(source);
                                        RoutedIds {
                                            ids,
                                            backend: BACKEND_REFERENCE.to_owned(),
                                            source: None,
                                            path: Some(path.to_owned()),
                                        }
                                    }
                                };
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
            self.record_execution(
                &routed.backend,
                texts[row].len() as u64,
                starts[row],
                routed.source.as_deref(),
                routed.path.as_deref(),
            );
        }
        Ok(output)
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
            "hf_engine_guard"
        );
        let stats = router.stats();
        assert_guard_events_have_stage(&stats);
        assert_eq!(stats.events[0].detail, json!({"stage": "engine_guard"}));
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
        let batch = router.encode_batch_ids(&["a", "aa"], true).unwrap();
        assert!(batch
            .iter()
            .all(|outcome| outcome.backend == BACKEND_REFERENCE));

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
