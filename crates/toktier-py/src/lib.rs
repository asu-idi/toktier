//! Thin Python facade over the toktier session store.
//!
//! This crate adds no store logic: it converts arguments, adapts a
//! Python-callable encoder onto the core [`SessionEncoder`] trait, maps
//! [`StoreError`] onto the structured exception contract
//! (`docs/contracts/errors.md`: every library-domain exception carries a
//! stable `.code` and a `.details` mapping; plain argument misuse stays
//! a plain `ValueError`/`KeyError`), and exposes the SQLite tier.
//!
//! The extension defines no public exception hierarchy of its own
//! (decision 0004). At the binding boundary it instantiates the classes
//! from `toktier.errors`, so a native failure is caught by
//! `toktier.ToktierError`, carries the frozen `.code`, and exposes
//! `.details` as a read-only mapping -- identical to a Python-raised
//! error. The names re-exported on `toktier._native` are the same class
//! objects. Only when the extension is loaded standalone, without the
//! `toktier` package importable, does a private contract-equivalent shim
//! stand in.

#![deny(unsafe_code)]

use std::ffi::CStr;

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyBytes, PyDict, PyString, PyStringMethods};

use toktier_routing_core::{
    BpeSyncBoundary, LiteralMode, LiteralPrefix, RouteSelector as CoreRouteSelector,
};
use toktier_store_core::{
    AppendReport, BoundaryCut, Encoding, EngineError, KeyId, SemanticFingerprint, SessionEncoder,
    SessionHandle, StoreConfig, StoreError, TailState, WitnessCategory,
};
use toktier_store_sqlite::{SingleEngine, StoreDb};

/// Fallback mirror of the error classes this boundary raises, used only
/// when `toktier.errors` is not importable (standalone `.so` loading).
/// `toktier.errors` stays authoritative; the shim reproduces the frozen
/// shape -- stable `.code`, read-only `.details` -- so the boundary
/// behaves identically either way, and it is never used when the public
/// classes can be imported.
const STANDALONE_ERRORS_SHIM: &CStr = cr#"
"""Standalone stand-in for ``toktier.errors`` (which is authoritative).

Loaded by ``toktier._native`` only when the ``toktier`` package is not
importable; see ``crates/toktier-py/src/lib.rs``.
"""
from types import MappingProxyType


class ToktierError(Exception):
    CODE = "TOKTIER_ERROR"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self._details = MappingProxyType(dict(details) if details else {})

    @property
    def code(self):
        return type(self).CODE

    @property
    def details(self):
        return self._details


class StoreCorrupt(ToktierError):
    CODE = "STORE_CORRUPT"


class StoreFormatUnsupported(ToktierError):
    CODE = "STORE_FORMAT_UNSUPPORTED"


class SessionStateMismatch(ToktierError):
    CODE = "SESSION_STATE_MISMATCH"


class SessionRevisionConflict(ToktierError):
    CODE = "SESSION_REVISION_CONFLICT"


class ConfigInvalid(ToktierError):
    CODE = "CONFIG_INVALID"
"#;

/// The exception classes this boundary raises, resolved once per
/// process: the public `toktier.errors` classes when importable, the
/// standalone shim otherwise.
struct ErrorClasses {
    base: Py<PyAny>,
    store_corrupt: Py<PyAny>,
    store_format_unsupported: Py<PyAny>,
    session_state_mismatch: Py<PyAny>,
    session_revision_conflict: Py<PyAny>,
    config_invalid: Py<PyAny>,
}

static ERROR_CLASSES: GILOnceCell<ErrorClasses> = GILOnceCell::new();

fn error_classes(py: Python<'_>) -> PyResult<&'static ErrorClasses> {
    ERROR_CLASSES.get_or_try_init(py, || {
        let source = match py.import("toktier.errors") {
            Ok(module) => module.into_any(),
            Err(_) => PyModule::from_code(
                py,
                STANDALONE_ERRORS_SHIM,
                c"toktier/_native_errors_shim.py",
                c"toktier._native_errors_shim",
            )?
            .into_any(),
        };
        Ok(ErrorClasses {
            base: source.getattr("ToktierError")?.unbind(),
            store_corrupt: source.getattr("StoreCorrupt")?.unbind(),
            store_format_unsupported: source.getattr("StoreFormatUnsupported")?.unbind(),
            session_state_mismatch: source.getattr("SessionStateMismatch")?.unbind(),
            session_revision_conflict: source.getattr("SessionRevisionConflict")?.unbind(),
            config_invalid: source.getattr("ConfigInvalid")?.unbind(),
        })
    })
}

/// Instantiate a structured exception class with a message and a
/// `details` mapping. The class constructor owns the contract shape:
/// it wraps `details` read-only and exposes the class `.code`.
fn structured(
    py: Python<'_>,
    class: &Py<PyAny>,
    message: &str,
    details: &Bound<'_, PyDict>,
) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("details", details)?;
        let exc = class.bind(py).call((message,), Some(&kwargs))?;
        Ok(PyErr::from_value(exc))
    };
    build().unwrap_or_else(|e| e)
}

fn err_to_py(py: Python<'_>, err: StoreError) -> PyErr {
    let classes = match error_classes(py) {
        Ok(classes) => classes,
        Err(resolve_err) => return resolve_err,
    };
    let details = PyDict::new(py);
    match &err {
        StoreError::RevisionConflict { expected, actual } => {
            if let Err(item_err) = details
                .set_item("expected_revision", *expected)
                .and_then(|()| details.set_item("stored_revision", *actual))
            {
                return item_err;
            }
            structured(
                py,
                &classes.session_revision_conflict,
                &err.to_string(),
                &details,
            )
        }
        StoreError::UnknownSession(h) => PyKeyError::new_err(format!("unknown session handle {h}")),
        StoreError::UnknownKey(_) | StoreError::GuardMismatch | StoreError::InvalidInput(_) => {
            PyValueError::new_err(err.to_string())
        }
        StoreError::Engine(msg) => PyRuntimeError::new_err(format!("encoder error: {msg}")),
        StoreError::Internal(_) => PyRuntimeError::new_err(err.to_string()),
        _ => {
            let class = match err.code() {
                "STORE_CORRUPT" => &classes.store_corrupt,
                "STORE_FORMAT_UNSUPPORTED" => &classes.store_format_unsupported,
                "SESSION_STATE_MISMATCH" => &classes.session_state_mismatch,
                "CONFIG_INVALID" => &classes.config_invalid,
                other => {
                    // Defensive only: every current code is mapped above.
                    // Preserve an unmapped native code in the details so
                    // nothing is lost if the core grows one.
                    if let Err(item_err) = details.set_item("native_code", other) {
                        return item_err;
                    }
                    &classes.base
                }
            };
            structured(py, class, &err.to_string(), &details)
        }
    }
}

fn ids_to_bytes<'py>(py: Python<'py>, ids: &[u32]) -> Bound<'py, PyBytes> {
    let mut raw = Vec::with_capacity(ids.len() * 4);
    for &v in ids {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    PyBytes::new(py, &raw)
}

fn fingerprint_of(raw: &[u8]) -> PyResult<SemanticFingerprint> {
    raw.try_into()
        .map_err(|_| PyValueError::new_err("fingerprint must be exactly 32 bytes"))
}

// --------------------------------------------------------------- route --

/// Allocation-free per-input selector over one immutable fallback chain.
///
/// Python owns the public RoutePlan and diagnostics. This private helper reads
/// CPython's cached UTF-8 view without creating a `bytes` object, applies the
/// byte crossover, and runs the exact frontend's necessary-condition literal
/// gate in the same pass.
#[pyclass(name = "RouteSelector", module = "toktier._native")]
struct NativeRouteSelector {
    inner: CoreRouteSelector,
}

#[pymethods]
impl NativeRouteSelector {
    #[new]
    #[pyo3(signature = (
        thresholds,
        reference_index,
        gpu_head,
        literal_mode = 0,
        literal_prefixes = Vec::new()
    ))]
    fn new(
        thresholds: Vec<u64>,
        reference_index: usize,
        gpu_head: bool,
        literal_mode: u8,
        literal_prefixes: Vec<(u8, i16)>,
    ) -> PyResult<Self> {
        let mode = match literal_mode {
            0 => LiteralMode::Disabled,
            1 => LiteralMode::AlwaysCandidate,
            2 => LiteralMode::Prefixes,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown literal prefilter mode {other}"
                )))
            }
        };
        let prefixes = literal_prefixes
            .into_iter()
            .map(|(first, raw_second)| {
                let second = match raw_second {
                    -1 => None,
                    0..=255 => Some(raw_second as u8),
                    _ => {
                        return Err(PyValueError::new_err(
                            "literal prefix second byte must be -1 or 0..255",
                        ))
                    }
                };
                Ok(LiteralPrefix { first, second })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let inner = CoreRouteSelector::new(thresholds, reference_index, gpu_head, mode, prefixes)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    /// `(input_bytes, start_index, below_gpu_threshold, literal_candidate)`.
    fn route(&self, text: &Bound<'_, PyString>) -> (Option<u64>, usize, bool, bool) {
        // `to_str()` borrows CPython's cached UTF-8 representation. A lone
        // surrogate has no valid UTF-8 view; matching the Python executor's
        // historical behavior, route it to the reference backend so that the
        // oracle raises the original user-facing error.
        let input = text.to_str().ok().map(str::as_bytes);
        let decision = self.inner.decide(input);
        (
            decision.input_bytes,
            decision.start_index,
            decision.below_gpu_threshold,
            decision.literal_candidate,
        )
    }

    #[getter]
    fn reference_index(&self) -> usize {
        self.inner.reference_index()
    }
}

// -------------------------------------------------------------- encoder --

/// Python-callable encoder adapter.
///
/// * `encode_cb(text) -> (ids, spans)` -- reference encode with
///   character-unit spans (required).
/// * `append_cb(tail_text, tail_ids, tail_spans, delta) -> (ids, spans,
///   kept_tokens, path)` -- incremental append over the whole tail;
///   optional, defaults to a full re-encode through `encode_cb`.
/// * `boundary_cb(tail_text, tail_ids, tail_spans, floor_char,
///   ceil_char) -> (cut_tokens, cut_char) | None` -- certified boundary
///   probe; optional, defaults to never sealing.
/// * `bpe_sync_pclass` -- the frozen 0x110000-byte O/S/L/N/M property
///   table. When present, the boundary probe runs directly in Rust and
///   `boundary_cb` is only a compatibility fallback.
/// * `witness_category` -- frozen registry value (u16) matching the
///   certificates the callbacks implement.
#[pyclass(module = "toktier._native")]
struct CallbackEncoder {
    witness: WitnessCategory,
    encode_cb: Py<PyAny>,
    append_cb: Option<Py<PyAny>>,
    boundary_cb: Option<Py<PyAny>>,
    bpe_sync: Option<BpeSyncBoundary>,
}

#[pymethods]
impl CallbackEncoder {
    #[new]
    #[pyo3(signature = (
        witness_category,
        encode_cb,
        append_cb = None,
        boundary_cb = None,
        bpe_sync_pclass = None
    ))]
    fn new(
        witness_category: u16,
        encode_cb: Py<PyAny>,
        append_cb: Option<Py<PyAny>>,
        boundary_cb: Option<Py<PyAny>>,
        bpe_sync_pclass: Option<Bound<'_, PyBytes>>,
    ) -> PyResult<Self> {
        let witness = WitnessCategory::from_u16(witness_category)
            .map_err(|_| PyValueError::new_err("unknown witness category"))?;
        let bpe_sync = bpe_sync_pclass
            .map(|raw| {
                BpeSyncBoundary::new(raw.as_bytes().to_vec())
                    .map_err(|error| PyValueError::new_err(error.to_string()))
            })
            .transpose()?;
        if bpe_sync.is_some() && witness != WitnessCategory::BpeSyncTransition {
            return Err(PyValueError::new_err(
                "a BPE sync table requires the BPE sync witness category",
            ));
        }
        Ok(CallbackEncoder {
            witness,
            encode_cb,
            append_cb,
            boundary_cb,
            bpe_sync,
        })
    }

    #[getter]
    fn witness_category(&self) -> u16 {
        self.witness.as_u16()
    }
}

fn engine_err(e: PyErr) -> EngineError {
    EngineError(format!("python callback failed: {e}"))
}

impl CallbackEncoder {
    fn call_encode(&self, text: &str) -> Result<Encoding, EngineError> {
        Python::with_gil(|py| {
            let out = self.encode_cb.call1(py, (text,)).map_err(engine_err)?;
            let (ids, spans): (Vec<u32>, Vec<(u32, u32)>) = out.extract(py).map_err(engine_err)?;
            if ids.len() != spans.len() {
                return Err(EngineError(format!(
                    "encode callback returned {} ids but {} spans",
                    ids.len(),
                    spans.len()
                )));
            }
            Ok(Encoding { ids, spans })
        })
    }
}

impl SessionEncoder for CallbackEncoder {
    fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
        self.call_encode(text)
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        let was_empty = tail.text().is_empty();
        let mut full = String::with_capacity(tail.text_bytes() + delta.len());
        full.push_str(tail.text());
        full.push_str(delta);
        // A fresh session is a full encode, not an append repair. Keep it
        // on `encode_cb` even when a repair callback exists so the facade's
        // normal CPU/GPU full-encode router can seed state once; only later
        // strict appends enter the CPU repair callback.
        let (enc, kept, path) = if was_empty {
            (self.call_encode(&full)?, 0, "cold_full".to_string())
        } else {
            match &self.append_cb {
                Some(cb) => Python::with_gil(|py| {
                    let args = (tail.text(), tail.ids().to_vec(), tail.spans(), delta);
                    let out = cb.call1(py, args).map_err(engine_err)?;
                    let (ids, spans, kept, path): (Vec<u32>, Vec<(u32, u32)>, usize, String) =
                        out.extract(py).map_err(engine_err)?;
                    if ids.len() != spans.len() {
                        return Err(EngineError(format!(
                            "append callback returned {} ids but {} spans",
                            ids.len(),
                            spans.len()
                        )));
                    }
                    Ok((Encoding { ids, spans }, kept, path))
                })?,
                None => (self.call_encode(&full)?, 0, "cb_full_reencode".to_string()),
            }
        };
        tail.fill(&full, enc)
            .map_err(|e| EngineError(format!("tail fill failed: {e}")))?;
        Ok(AppendReport {
            path,
            kept_tokens: kept,
        })
    }

    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        if let Some(predicate) = &self.bpe_sync {
            return Ok(predicate
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
                }));
        }
        let Some(cb) = &self.boundary_cb else {
            return Ok(None);
        };
        Python::with_gil(|py| {
            let args = (
                tail.text(),
                tail.ids().to_vec(),
                tail.spans(),
                floor_char,
                ceil_char,
            );
            let out = cb.call1(py, args).map_err(engine_err)?;
            let cut: Option<(usize, u64)> = out.extract(py).map_err(engine_err)?;
            Ok(cut.map(|(cut_tokens, cut_char)| BoundaryCut {
                cut_tokens,
                cut_char,
            }))
        })
    }

    fn witness_category(&self) -> WitnessCategory {
        self.witness
    }
}

// ---------------------------------------------------------------- store --

/// Session store (thin facade over the Rust core; see the core crate
/// for semantics).
#[pyclass(module = "toktier._native")]
struct SessionStore {
    inner: toktier_store_core::SessionStore,
}

fn map<T>(py: Python<'_>, r: Result<T, StoreError>) -> PyResult<T> {
    r.map_err(|e| err_to_py(py, e))
}

#[pymethods]
impl SessionStore {
    #[new]
    #[pyo3(signature = (block_chars = 4096, tail_soft_cap_bytes = 65536,
                        tail_hard_cap_bytes = 1048576, node_tail_cap_bytes = 65536,
                        max_sessions = 1024))]
    fn new(
        py: Python<'_>,
        block_chars: u64,
        tail_soft_cap_bytes: usize,
        tail_hard_cap_bytes: usize,
        node_tail_cap_bytes: usize,
        max_sessions: usize,
    ) -> PyResult<Self> {
        let cfg = StoreConfig {
            block_chars,
            tail_soft_cap_bytes,
            tail_hard_cap_bytes,
            node_tail_cap_bytes,
            max_sessions,
        };
        // Config errors are argument misuse at this surface: ValueError
        // (prototype battery parity), code available via message.
        toktier_store_core::SessionStore::new(cfg)
            .map(|inner| SessionStore { inner })
            .map_err(|e| {
                let _ = py;
                PyValueError::new_err(e.to_string())
            })
    }

    /// Intern a 32-byte semantic fingerprint; returns a stable key id.
    fn register_fingerprint(
        &mut self,
        py: Python<'_>,
        fingerprint: &[u8],
        seal_end_guard_chars: u64,
    ) -> PyResult<u32> {
        let fp = fingerprint_of(fingerprint)?;
        map(
            py,
            self.inner.register_fingerprint(fp, seal_end_guard_chars),
        )
        .map(|k| k.0)
    }

    /// Full encode of `text` into a new session; returns
    /// `(handle, revision, token_count)`.
    fn put(
        &mut self,
        py: Python<'_>,
        key_id: u32,
        text: &str,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<(u64, u64, u64)> {
        let out = map(py, self.inner.put(KeyId(key_id), text, &*engine))?;
        Ok((out.handle.0, out.revision, out.token_count))
    }

    /// Certified append under optimistic concurrency. Returns a dict:
    /// `path`, `revision`, `replace_from`, `replacement_ids` (bytes,
    /// u32 LE), `all_ids` (bytes, u32 LE), `n_ids`.
    fn append<'py>(
        &mut self,
        py: Python<'py>,
        handle: u64,
        delta: &str,
        expected_revision: u64,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = map(
            py,
            self.inner
                .append(SessionHandle(handle), delta, expected_revision, &*engine),
        )?;
        let d = PyDict::new(py);
        d.set_item("path", &out.path)?;
        d.set_item("revision", out.revision)?;
        d.set_item("replace_from", out.replace_from)?;
        d.set_item("replacement_ids", ids_to_bytes(py, &out.replacement_ids))?;
        d.set_item("all_ids", ids_to_bytes(py, &out.all_ids))?;
        d.set_item("n_ids", out.all_ids.len())?;
        Ok(d)
    }

    /// Longest block-prefix hit; `(handle, matched_chars, revision)` or
    /// `None`.
    fn lookup(
        &mut self,
        py: Python<'_>,
        key_id: u32,
        text: &str,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<Option<(u64, u64, u64)>> {
        let hit = map(py, self.inner.lookup(KeyId(key_id), text, &*engine))?;
        Ok(hit.map(|h| (h.handle.0, h.matched_chars, h.revision)))
    }

    fn fork(&mut self, py: Python<'_>, handle: u64) -> PyResult<u64> {
        map(py, self.inner.fork(SessionHandle(handle))).map(|h| h.0)
    }

    fn evict(&mut self, handle: u64) -> bool {
        self.inner.evict(SessionHandle(handle))
    }

    /// Full token stream as raw little-endian u32 bytes.
    fn ids_bytes<'py>(&mut self, py: Python<'py>, handle: u64) -> PyResult<Bound<'py, PyBytes>> {
        let ids = map(py, self.inner.all_ids(SessionHandle(handle)))?;
        Ok(ids_to_bytes(py, &ids))
    }

    fn revision(&self, py: Python<'_>, handle: u64) -> PyResult<u64> {
        map(py, self.inner.revision(SessionHandle(handle)))
    }

    fn session_info<'py>(&self, py: Python<'py>, handle: u64) -> PyResult<Bound<'py, PyDict>> {
        let info = map(py, self.inner.session_info(SessionHandle(handle)))?;
        let d = PyDict::new(py);
        d.set_item("key_id", info.key_id.0)?;
        d.set_item("witness_category", info.witness.as_u16())?;
        d.set_item("revision", info.revision)?;
        d.set_item("total_chars", info.total_chars)?;
        d.set_item("safe_char", info.safe_char)?;
        d.set_item("stable_prefix_bytes", info.stable_prefix_bytes)?;
        d.set_item("n_ids", info.token_count)?;
        d.set_item("sealed_tokens", info.sealed_tokens)?;
        d.set_item("tail_chars", info.tail_chars)?;
        d.set_item("tail_bytes", info.tail_bytes)?;
        d.set_item("buf_bytes", info.buf_bytes)?;
        d.set_item("blocks_end", info.blocks_end)?;
        d.set_item("chain_ok", info.chain_ok)?;
        d.set_item("last_replace_from", info.last_replace_from)?;
        d.set_item("approx_bytes", info.approx_bytes)?;
        Ok(d)
    }

    fn list_handles(&self) -> Vec<u64> {
        self.inner.list_handles().into_iter().map(|h| h.0).collect()
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let d = PyDict::new(py);
        d.set_item("format", s.format)?;
        d.set_item("block_chars", s.block_chars)?;
        d.set_item("tail_soft_cap_bytes", s.tail_soft_cap_bytes)?;
        d.set_item("tail_hard_cap_bytes", s.tail_hard_cap_bytes)?;
        d.set_item("max_sessions", s.max_sessions)?;
        d.set_item("session_count", s.session_count)?;
        d.set_item("node_count", s.node_count)?;
        d.set_item("puts", s.puts)?;
        d.set_item("extends", s.extends)?;
        d.set_item("forks", s.forks)?;
        d.set_item("lookups", s.lookups)?;
        d.set_item("lookup_hits", s.lookup_hits)?;
        d.set_item("lookup_misses", s.lookup_misses)?;
        d.set_item("hit_rate", s.hit_rate)?;
        d.set_item("checksum_rejects", s.checksum_rejects)?;
        d.set_item("k_cap_overflows", s.k_cap_overflows)?;
        d.set_item("hard_cap_degrades", s.hard_cap_degrades)?;
        d.set_item("seals", s.seals)?;
        d.set_item("sealed_tokens", s.sealed_tokens)?;
        d.set_item("sessions_evicted", s.sessions_evicted)?;
        d.set_item("nodes_skipped_tail_cap", s.nodes_skipped_tail_cap)?;
        d.set_item("chain_detaches", s.chain_detaches)?;
        d.set_item("import_rejects", s.import_rejects)?;
        d.set_item("revision_conflicts", s.revision_conflicts)?;
        let paths = PyDict::new(py);
        for (k, v) in &s.path_counts {
            paths.set_item(k, v)?;
        }
        d.set_item("path_counts", paths)?;
        Ok(d)
    }

    fn export_fingerprints<'py>(&self, py: Python<'py>) -> Vec<(u32, Bound<'py, PyBytes>, u64)> {
        self.inner
            .export_fingerprints()
            .into_iter()
            .map(|(id, fp, guard)| (id, PyBytes::new(py, &fp), guard))
            .collect()
    }

    /// Serialize one session as a store format v1 record.
    fn export_session<'py>(&self, py: Python<'py>, handle: u64) -> PyResult<Bound<'py, PyBytes>> {
        let rec = map(py, self.inner.export_session(SessionHandle(handle)))?;
        Ok(PyBytes::new(py, &rec))
    }

    /// Serialize the internal bookkeeping sidecar of one session.
    fn export_session_sidecar<'py>(
        &self,
        py: Python<'py>,
        handle: u64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let sc = map(py, self.inner.export_session_sidecar(SessionHandle(handle)))?;
        Ok(PyBytes::new(py, &sc))
    }

    /// Import a session from a bare format v1 record (conservative:
    /// detached chain; ids and revision chain fully restored).
    fn import_session(
        &mut self,
        py: Python<'_>,
        key_id: u32,
        rec: &[u8],
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<u64> {
        map(py, self.inner.import_session(KeyId(key_id), rec, &*engine)).map(|h| h.0)
    }

    /// Import a session from a record plus its sidecar (exact restore).
    fn import_session_with_sidecar(
        &mut self,
        py: Python<'_>,
        key_id: u32,
        rec: &[u8],
        sidecar: &[u8],
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<u64> {
        map(
            py,
            self.inner
                .import_session_with_sidecar(KeyId(key_id), rec, sidecar, &*engine),
        )
        .map(|h| h.0)
    }

    fn export_node_items<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>> {
        let items = map(py, self.inner.export_node_items())?;
        Ok(items
            .into_iter()
            .map(|(k, r)| (PyBytes::new(py, &k), PyBytes::new(py, &r)))
            .collect())
    }

    fn import_node_item(&mut self, node_key: &[u8], rec: &[u8]) -> bool {
        self.inner.import_node_item(node_key, rec)
    }

    /// Test support only: corrupt one stored node in memory so the next
    /// lookup must reject it. Compiled only into `testing` builds.
    #[cfg(feature = "testing")]
    fn corrupt_node_for_tests(&mut self, py: Python<'_>, node_key: &[u8]) -> PyResult<bool> {
        map(py, self.inner.corrupt_node_for_tests(node_key))
    }

    /// Persist the full store into an exclusively-owned SQLite file.
    fn save_sqlite(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let mut db = StoreDb::open(std::path::Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        db.save(&self.inner).map_err(|e| match e {
            toktier_store_sqlite::DbError::Store(se) => err_to_py(py, se),
            other => PyRuntimeError::new_err(other.to_string()),
        })
    }

    /// Rebuild a store from a SQLite file; every session tail is
    /// re-encoded through `engine` and verified. Returns
    /// `(store, handle_map)` where `handle_map` maps saved session ids
    /// to fresh handles.
    #[staticmethod]
    fn load_sqlite(
        py: Python<'_>,
        path: &str,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<(SessionStore, std::collections::HashMap<i64, u64>)> {
        let db = StoreDb::open(std::path::Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let (inner, hmap) = db.load(&SingleEngine(&*engine)).map_err(|e| match e {
            toktier_store_sqlite::DbError::Store(se) => err_to_py(py, se),
            other => PyRuntimeError::new_err(other.to_string()),
        })?;
        Ok((
            SessionStore { inner },
            hmap.into_iter().map(|(sid, h)| (sid, h.0)).collect(),
        ))
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stats();
        format!(
            "SessionStore(format={}, sessions={}, nodes={})",
            s.format, s.session_count, s.node_count
        )
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<SessionStore>()?;
    m.add_class::<CallbackEncoder>()?;
    m.add_class::<NativeRouteSelector>()?;
    // Convenience re-exports of the classes this boundary raises. These
    // are the public `toktier.errors` objects themselves (or, only under
    // standalone loading, the private shim); the extension defines no
    // exception types of its own.
    let classes = error_classes(py)?;
    m.add("ToktierError", classes.base.bind(py))?;
    m.add("StoreCorrupt", classes.store_corrupt.bind(py))?;
    m.add(
        "StoreFormatUnsupported",
        classes.store_format_unsupported.bind(py),
    )?;
    m.add(
        "SessionStateMismatch",
        classes.session_state_mismatch.bind(py),
    )?;
    m.add(
        "SessionRevisionConflict",
        classes.session_revision_conflict.bind(py),
    )?;
    m.add("ConfigInvalid", classes.config_invalid.bind(py))?;
    m.add("FORMAT_NAME", toktier_store_core::FORMAT_NAME)?;
    m.add("WITNESS_NONE_FULL_REENCODE", 0u16)?;
    m.add("WITNESS_BPE_SYNC_TRANSITION", 1u16)?;
    m.add("WITNESS_WORDPIECE_CONTINUATION", 2u16)?;
    m.add("WITNESS_METASPACE_WORD_START", 3u16)?;
    Ok(())
}
