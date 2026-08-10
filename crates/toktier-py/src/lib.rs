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

use std::collections::BTreeMap;
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyStringMethods};

use toktier_routing_core::{
    BpeSyncBoundary, EntryStoreOpenError, FastCpuEngine, FastRepairSpec, LiteralMode,
    LiteralPrefix, NativeEntryStore, NativePrebuiltGpu, NativePrebuiltGpuConfig, NativeRouter,
    ReferenceEngine, RouteSelector as CoreRouteSelector,
};
use toktier_store_core::{
    AppendReport, BoundaryCut, Encoding, EngineError, KeyId, RecoveryMaterial, SemanticFingerprint,
    SessionEncoder, SessionHandle, StoreConfig, StoreError, TailState, WitnessCategory,
};
use toktier_store_sqlite::{SingleEngine, StoreDb};

const FAST_CPU_ENGINE_VERSION: &str = "0.10.0+toktier.pinned.1";
type PyIdsWithOffsets = (Vec<u32>, Vec<(u32, u32)>);
type PyContentIndexEntry = (u64, String, Vec<(u64, String)>);
const FAST_CPU_ENGINE_MODULE: &str = "toktier._native";
const FAST_CPU_ENGINE_DELIVERY: &str = "integrated";

/// Build-time identity of the corrected Gigatoken implementation that is
/// actually linked into this extension.  The planner compares these facts to
/// the checked registry before admitting the CPU-fast route.
#[pyfunction]
fn fast_cpu_build_facts<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("engine", "gigatoken")?;
    output.set_item("engine_version", FAST_CPU_ENGINE_VERSION)?;
    output.set_item("engine_delivery", FAST_CPU_ENGINE_DELIVERY)?;
    output.set_item("engine_module", FAST_CPU_ENGINE_MODULE)?;
    output.set_item("source_digest", env!("TOKTIER_FAST_CPU_SOURCE_SHA256"))?;
    output.set_item(
        "build_flags",
        env!("TOKTIER_FAST_CPU_BUILD_FLAGS")
            .split('\x1f')
            .collect::<Vec<_>>(),
    )?;
    output.set_item("toolchain", env!("TOKTIER_FAST_CPU_TOOLCHAIN"))?;
    Ok(output)
}

/// Build-time identity of the Rust request host paired with the shipped
/// prebuilt CUDA binary.  The prebuilt certificate binds these facts in
/// addition to the fatbin and per-architecture image digests.
#[pyfunction]
fn native_host_build_facts<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("source_digest", env!("TOKTIER_NATIVE_HOST_SOURCE_SHA256"))?;
    output.set_item(
        "build_flags",
        env!("TOKTIER_NATIVE_HOST_BUILD_FLAGS")
            .split('\x1f')
            .collect::<Vec<_>>(),
    )?;
    output.set_item("toolchain", env!("TOKTIER_NATIVE_HOST_TOOLCHAIN"))?;
    Ok(output)
}

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

fn digest32_of(raw: &[u8], name: &str) -> PyResult<[u8; 32]> {
    raw.try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be exactly 32 bytes")))
}

fn hex(raw: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(raw.len() * 2);
    for &byte in raw {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    Ok(match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(value) => value.into_pyobject(py)?.to_owned().unbind().into_any(),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                value.into_pyobject(py)?.to_owned().unbind().into_any()
            } else if let Some(value) = value.as_u64() {
                value.into_pyobject(py)?.to_owned().unbind().into_any()
            } else {
                value
                    .as_f64()
                    .unwrap_or_default()
                    .into_pyobject(py)?
                    .to_owned()
                    .unbind()
                    .into_any()
            }
        }
        serde_json::Value::String(value) => PyString::new(py, value).unbind().into_any(),
        serde_json::Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(json_to_py(py, value)?)?;
            }
            list.unbind().into_any()
        }
        serde_json::Value::Object(values) => {
            let mapping = PyDict::new(py);
            for (key, value) in values {
                mapping.set_item(key, json_to_py(py, value)?)?;
            }
            mapping.unbind().into_any()
        }
    })
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

// ----------------------------------------------------------- reference --

/// Frozen Hugging Face reference engine owned entirely by Rust.
///
/// The Python facade verifies the artifact before construction.  Every method
/// then borrows Python's immutable UTF-8 storage, releases the GIL for the
/// complete tokenizer operation, and converts only the final result.
#[pyclass(name = "ReferenceEngine", module = "toktier._native")]
struct NativeReferenceEngine {
    inner: Arc<ReferenceEngine>,
}

fn reference_err(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("native reference engine failed: {error}"))
}

#[pymethods]
impl NativeReferenceEngine {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = ReferenceEngine::from_file(path).map_err(reference_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[pyo3(signature = (text, add_special_tokens = true))]
    fn encode(
        &self,
        py: Python<'_>,
        text: PyBackedStr,
        add_special_tokens: bool,
    ) -> PyResult<Vec<u32>> {
        let engine = Arc::clone(&self.inner);
        py.allow_threads(move || engine.encode_ids(&text, add_special_tokens))
            .map_err(reference_err)
    }

    fn encode_with_offsets(&self, py: Python<'_>, text: PyBackedStr) -> PyResult<PyIdsWithOffsets> {
        let engine = Arc::clone(&self.inner);
        let encoded = py
            .allow_threads(move || engine.encode_core(&text))
            .map_err(reference_err)?;
        Ok((encoded.ids, encoded.spans))
    }

    #[pyo3(signature = (texts, add_special_tokens = true))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<PyBackedStr>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        let engine = Arc::clone(&self.inner);
        py.allow_threads(move || {
            let rows = texts.iter().map(|text| text.as_ref()).collect::<Vec<_>>();
            engine.encode_batch_ids(&rows, add_special_tokens)
        })
        .map_err(reference_err)
    }

    #[pyo3(signature = (ids, skip_special_tokens = true))]
    fn decode(&self, py: Python<'_>, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        let engine = Arc::clone(&self.inner);
        py.allow_threads(move || engine.decode(&ids, skip_special_tokens))
            .map_err(reference_err)
    }

    #[getter]
    fn oracle_version(&self) -> &'static str {
        "tokenizers==0.22.2"
    }
}

// ---------------------------------------------------------- prebuilt GPU --

/// Manifest-bound CUDA Driver host. Construction performs all Python-owned
/// artifact/table projection; request execution thereafter is entirely Rust.
#[pyclass(name = "NativePrebuiltGpu", module = "toktier._native")]
struct PyNativePrebuiltGpu {
    inner: Arc<NativePrebuiltGpu>,
}

#[pymethods]
impl PyNativePrebuiltGpu {
    #[new]
    #[pyo3(signature = (
        family,
        artifact_sha256,
        fatbin,
        expected_fatbin_sha256,
        expected_architecture,
        device_ordinal,
        ruleset,
        digits_max,
        contractions,
        needs_nfc,
        ignore_merges,
        symbols,
        class_table,
        pair_keys,
        pair_vals,
        byte_id,
        vocab_keys,
        vocab_vals,
        vocab_blob,
        unsafe_bits,
        pair_count,
        vocab_count,
        reference
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        family: String,
        artifact_sha256: String,
        fatbin: &[u8],
        expected_fatbin_sha256: String,
        expected_architecture: String,
        device_ordinal: i32,
        ruleset: String,
        digits_max: i32,
        contractions: bool,
        needs_nfc: bool,
        ignore_merges: i32,
        symbols: BTreeMap<String, String>,
        class_table: &[u8],
        pair_keys: &[u8],
        pair_vals: &[u8],
        byte_id: &[u8],
        vocab_keys: &[u8],
        vocab_vals: &[u8],
        vocab_blob: &[u8],
        unsafe_bits: &[u8],
        pair_count: usize,
        vocab_count: usize,
        reference: PyRef<'_, NativeReferenceEngine>,
    ) -> PyResult<Self> {
        let config = NativePrebuiltGpuConfig {
            family,
            artifact_sha256,
            expected_fatbin_sha256,
            expected_architecture,
            device_ordinal,
            ruleset,
            digits_max,
            contractions,
            needs_nfc,
            ignore_merges,
            pair_count,
            vocab_count,
            delivery: "prebuilt".to_owned(),
        };
        let reference = reference.native();
        let engine = py
            .allow_threads(|| {
                NativePrebuiltGpu::new(
                    config,
                    (*reference).clone(),
                    fatbin,
                    symbols,
                    class_table,
                    pair_keys,
                    pair_vals,
                    byte_id,
                    vocab_keys,
                    vocab_vals,
                    vocab_blob,
                    unsafe_bits,
                )
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(engine),
        })
    }
}

impl NativeReferenceEngine {
    fn native(&self) -> Arc<ReferenceEngine> {
        Arc::clone(&self.inner)
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
    inner: EncoderImpl,
}

enum EncoderImpl {
    Python {
        witness: WitnessCategory,
        encode_cb: Py<PyAny>,
        append_cb: Option<Py<PyAny>>,
        boundary_cb: Option<Py<PyAny>>,
        bpe_sync: Option<BpeSyncBoundary>,
    },
    NativeFastCpu(Arc<FastCpuEngine>),
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
            inner: EncoderImpl::Python {
                witness,
                encode_cb,
                append_cb,
                boundary_cb,
                bpe_sync,
            },
        })
    }

    /// Construct the corrected CPU/session engine entirely inside Rust.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        tokenizer_json,
        family,
        artifact_sha256,
        margin,
        effective_l_max,
        has_normalizer,
        bpe_sync_pclass,
        reference = None
    ))]
    fn native_fast_cpu(
        tokenizer_json: &[u8],
        family: String,
        artifact_sha256: String,
        margin: usize,
        effective_l_max: usize,
        has_normalizer: bool,
        bpe_sync_pclass: &[u8],
        reference: Option<PyRef<'_, NativeReferenceEngine>>,
    ) -> PyResult<Self> {
        let spec = FastRepairSpec::new(
            family,
            artifact_sha256,
            margin,
            effective_l_max,
            has_normalizer,
        );
        let engine = match reference {
            Some(reference) => FastCpuEngine::from_reference(
                tokenizer_json,
                reference.native(),
                spec,
                bpe_sync_pclass.to_vec(),
            ),
            None => FastCpuEngine::from_json(tokenizer_json, spec, bpe_sync_pclass.to_vec()),
        }
        .map_err(|error| {
            PyValueError::new_err(format!(
                "native fast CPU engine rejected its inputs: {error}"
            ))
        })?;
        Ok(Self {
            inner: EncoderImpl::NativeFastCpu(Arc::new(engine)),
        })
    }

    #[getter]
    fn witness_category(&self) -> u16 {
        self.witness().as_u16()
    }

    #[getter]
    fn native_request_path(&self) -> bool {
        matches!(self.inner, EncoderImpl::NativeFastCpu(_))
    }

    #[getter]
    fn engine_initialized(&self) -> bool {
        match &self.inner {
            EncoderImpl::NativeFastCpu(engine) => engine.is_initialized(),
            EncoderImpl::Python { .. } => false,
        }
    }

    #[getter]
    fn batch_worker_count(&self) -> usize {
        match &self.inner {
            EncoderImpl::NativeFastCpu(engine) => engine.batch_worker_count(),
            EncoderImpl::Python { .. } => 0,
        }
    }

    #[getter]
    fn minimum_seal_tail_chars(&self) -> usize {
        match &self.inner {
            EncoderImpl::NativeFastCpu(engine) => engine.minimum_seal_tail_chars(),
            EncoderImpl::Python { .. } => 0,
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        match &self.inner {
            EncoderImpl::NativeFastCpu(engine) => {
                let stats = engine.stats();
                output.set_item("backend", "fast_cpu")?;
                output.set_item("engine", "gigatoken")?;
                output.set_item("config_id", "toktier-fast-repair-v1")?;
                output.set_item("family", &engine.spec().family)?;
                output.set_item("artifact_sha256", &engine.spec().artifact_sha256)?;
                output.set_item("window_calls", stats.window_calls)?;
                output.set_item("window_chars", stats.window_chars)?;
                let paths = PyDict::new(py);
                for (name, value) in stats.path_counts {
                    paths.set_item(name, value)?;
                }
                output.set_item("path_counts", paths)?;
                let last = PyDict::new(py);
                if let Some(path) = stats.last_path {
                    last.set_item("path", path)?;
                    last.set_item("reason", stats.last_reason)?;
                    last.set_item("kept_tokens", stats.last_kept_tokens)?;
                    last.set_item("window_chars", stats.last_window_chars)?;
                    last.set_item("retries", stats.last_retries)?;
                    output.set_item("last", last)?;
                } else {
                    output.set_item("last", py.None())?;
                }
            }
            EncoderImpl::Python { .. } => {
                output.set_item("backend", "python_callback")?;
            }
        }
        Ok(output)
    }

    /// Core-stream IDs through the native corrected-CPU route.
    fn encode(&self, py: Python<'_>, text: PyBackedStr) -> PyResult<Vec<u32>> {
        match &self.inner {
            EncoderImpl::NativeFastCpu(engine) => {
                let engine = Arc::clone(engine);
                py.allow_threads(move || engine.encode_ids(&text))
                    .map_err(reference_err)
            }
            EncoderImpl::Python { .. } => self
                .call_encode(&text)
                .map(|encoding| encoding.ids)
                .map_err(reference_err),
        }
    }

    /// Batch core-stream IDs with persistent native worker caches.
    fn encode_batch(&self, py: Python<'_>, texts: Vec<PyBackedStr>) -> PyResult<Vec<Vec<u32>>> {
        match &self.inner {
            EncoderImpl::NativeFastCpu(engine) => {
                let engine = Arc::clone(engine);
                py.allow_threads(move || {
                    let rows = texts.iter().map(|text| text.as_ref()).collect::<Vec<_>>();
                    engine.encode_batch_ids(&rows)
                })
                .map_err(reference_err)
            }
            EncoderImpl::Python { .. } => texts
                .iter()
                .map(|text| self.call_encode(text).map(|encoding| encoding.ids))
                .collect::<Result<Vec<_>, _>>()
                .map_err(reference_err),
        }
    }

    #[getter]
    fn vocab_size(&self) -> PyResult<usize> {
        match &self.inner {
            EncoderImpl::NativeFastCpu(engine) => engine.vocab_size().map_err(reference_err),
            EncoderImpl::Python { .. } => Err(PyRuntimeError::new_err(
                "a Python callback encoder has no vocabulary surface",
            )),
        }
    }

    #[getter]
    fn vocab<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let EncoderImpl::NativeFastCpu(engine) = &self.inner else {
            return Err(PyRuntimeError::new_err(
                "a Python callback encoder has no vocabulary surface",
            ));
        };
        let rows = engine.vocab_entries().map_err(reference_err)?;
        let output = PyDict::new(py);
        for (id, bytes) in rows {
            output.set_item(id, PyBytes::new(py, &bytes))?;
        }
        Ok(output)
    }
}

fn engine_err(e: PyErr) -> EngineError {
    EngineError(format!("python callback failed: {e}"))
}

impl CallbackEncoder {
    fn witness(&self) -> WitnessCategory {
        match &self.inner {
            EncoderImpl::Python { witness, .. } => *witness,
            EncoderImpl::NativeFastCpu(engine) => engine.witness_category(),
        }
    }

    fn native(&self) -> Option<Arc<FastCpuEngine>> {
        match &self.inner {
            EncoderImpl::NativeFastCpu(engine) => Some(Arc::clone(engine)),
            EncoderImpl::Python { .. } => None,
        }
    }

    fn call_encode(&self, text: &str) -> Result<Encoding, EngineError> {
        match &self.inner {
            EncoderImpl::NativeFastCpu(engine) => engine.encode(text),
            EncoderImpl::Python { encode_cb, .. } => Python::with_gil(|py| {
                let out = encode_cb.call1(py, (text,)).map_err(engine_err)?;
                let (ids, spans): (Vec<u32>, Vec<(u32, u32)>) =
                    out.extract(py).map_err(engine_err)?;
                if ids.len() != spans.len() {
                    return Err(EngineError(format!(
                        "encode callback returned {} ids but {} spans",
                        ids.len(),
                        spans.len()
                    )));
                }
                Ok(Encoding { ids, spans })
            }),
        }
    }
}

impl SessionEncoder for CallbackEncoder {
    fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
        self.call_encode(text)
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        if let EncoderImpl::NativeFastCpu(engine) = &self.inner {
            return engine.append(tail, delta);
        }
        let EncoderImpl::Python { append_cb, .. } = &self.inner else {
            unreachable!()
        };
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
            match append_cb {
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
        if let EncoderImpl::NativeFastCpu(engine) = &self.inner {
            return engine.last_certified_boundary(tail, floor_char, ceil_char);
        }
        let EncoderImpl::Python {
            boundary_cb,
            bpe_sync,
            ..
        } = &self.inner
        else {
            unreachable!()
        };
        if let Some(predicate) = bpe_sync {
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
        let Some(cb) = boundary_cb else {
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
        self.witness()
    }
}

// ------------------------------------------------------ native runtime --

/// Complete CPU/store request path behind one GIL-released call.
#[pyclass(name = "NativeRuntime", module = "toktier._native")]
struct PyNativeRuntime {
    router: Arc<NativeRouter>,
    store: Mutex<NativeEntryStore>,
    calls: std::sync::atomic::AtomicU64,
}

fn entry_open_err(py: Python<'_>, error: EntryStoreOpenError) -> PyErr {
    match error {
        EntryStoreOpenError::StateMismatch(message) => {
            let classes = match error_classes(py) {
                Ok(classes) => classes,
                Err(error) => return error,
            };
            structured(
                py,
                &classes.session_state_mismatch,
                &message,
                &PyDict::new(py),
            )
        }
        EntryStoreOpenError::Store(error) => err_to_py(py, error),
        EntryStoreOpenError::Io(error) => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pymethods]
impl PyNativeRuntime {
    #[new]
    #[pyo3(signature = (
        fallback_chain,
        minimum_input_bytes,
        reference,
        fast_encoder,
        gpu_encoder,
        repair_fast_cpu,
        fingerprint,
        seal_end_guard_chars,
        postprocessor_adds_tokens,
        diagnostics = false,
        store_directory = None,
        cache_budget_bytes = toktier_routing_core::DEFAULT_CACHE_BUDGET_BYTES
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        fallback_chain: Vec<String>,
        minimum_input_bytes: Vec<u64>,
        reference: PyRef<'_, NativeReferenceEngine>,
        fast_encoder: Option<PyRef<'_, CallbackEncoder>>,
        gpu_encoder: Option<PyRef<'_, PyNativePrebuiltGpu>>,
        repair_fast_cpu: bool,
        fingerprint: &[u8],
        seal_end_guard_chars: u64,
        postprocessor_adds_tokens: bool,
        diagnostics: bool,
        store_directory: Option<String>,
        cache_budget_bytes: usize,
    ) -> PyResult<Self> {
        let fingerprint = fingerprint_of(fingerprint)?;
        let fast = fast_encoder.as_ref().and_then(|encoder| encoder.native());
        if fast_encoder.is_some() && fast.is_none() {
            return Err(PyValueError::new_err(
                "fast_encoder must be a native corrected-CPU engine",
            ));
        }
        let reference = fast
            .as_ref()
            .map(|engine| engine.reference_arc())
            .unwrap_or_else(|| reference.native());
        let gpu = gpu_encoder.map(|engine| Arc::clone(&engine.inner));
        let router = Arc::new(
            NativeRouter::new(
                fallback_chain,
                minimum_input_bytes,
                reference,
                fast,
                repair_fast_cpu,
                gpu.map(|engine| engine as Arc<dyn toktier_routing_core::NativeGpuEngine>),
                postprocessor_adds_tokens,
                diagnostics,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        );
        let store = NativeEntryStore::open(
            fingerprint,
            Arc::clone(&router),
            store_directory.map(PathBuf::from),
            cache_budget_bytes,
            seal_end_guard_chars,
        )
        .map_err(|error| entry_open_err(py, error))?;
        Ok(Self {
            router,
            store: Mutex::new(store),
            calls: std::sync::atomic::AtomicU64::new(0),
        })
    }

    #[pyo3(signature = (
        text,
        session = None,
        lookup_auto = true,
        add_special_tokens = false
    ))]
    fn encode(
        &self,
        py: Python<'_>,
        text: PyBackedStr,
        session: Option<String>,
        lookup_auto: bool,
        add_special_tokens: bool,
    ) -> PyResult<Vec<u32>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        py.allow_threads(|| {
            let stored = if let Some(session) = session.as_deref() {
                self.store
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .encode_session(session, &text)
            } else if lookup_auto && !add_special_tokens {
                self.store
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .encode_auto(&text)
            } else {
                None
            };
            match stored {
                Some(ids) => Ok(ids),
                None => self
                    .router
                    .encode_ids(&text, add_special_tokens)
                    .map(|outcome| outcome.ids),
            }
        })
        .map_err(reference_err)
    }

    #[pyo3(signature = (texts, add_special_tokens = false))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<PyBackedStr>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        py.allow_threads(|| {
            let borrowed = texts.iter().map(|text| text.as_ref()).collect::<Vec<_>>();
            self.router
                .encode_batch_ids(&borrowed, add_special_tokens)
                .map(|rows| rows.into_iter().map(|outcome| outcome.ids).collect())
        })
        .map_err(reference_err)
    }

    fn runtime_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.router.stats();
        let output = PyDict::new(py);
        output.set_item("fallback_counts", stats.fallback_counts)?;
        output.set_item("execution_counts", stats.execution_counts)?;
        output.set_item(
            "last_execution",
            match &stats.last_execution {
                Some(value) => json_to_py(py, value)?,
                None => py.None(),
            },
        )?;
        let events = PyList::empty(py);
        for event in stats.events {
            let item = PyDict::new(py);
            item.set_item("code", event.code)?;
            item.set_item("backend", event.backend)?;
            item.set_item("target", event.target)?;
            item.set_item("detail", json_to_py(py, &event.detail)?)?;
            events.append(item)?;
        }
        output.set_item("events", events)?;
        output.set_item("state_encode_counts", stats.state_encode_counts)?;
        output.set_item(
            "last_state_encode",
            match &stats.last_state_encode {
                Some(value) => json_to_py(py, value)?,
                None => py.None(),
            },
        )?;
        output.set_item(
            "python_to_native_calls",
            self.calls.load(std::sync::atomic::Ordering::Relaxed),
        )?;
        Ok(output)
    }

    fn store_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let store = self
            .store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let stats = store.stats();
        let output = PyDict::new(py);
        output.set_item("session_hits", stats.session_hits)?;
        output.set_item("session_appends", stats.session_appends)?;
        output.set_item("session_overwrites", stats.session_overwrites)?;
        output.set_item("session_misses", stats.session_misses)?;
        output.set_item("auto_hits", stats.auto_hits)?;
        output.set_item("auto_appends", stats.auto_appends)?;
        output.set_item("auto_misses", stats.auto_misses)?;
        output.set_item("collision_rejects", stats.collision_rejects)?;
        output.set_item("degraded", stats.degraded)?;
        output.set_item("index_rebuilds", stats.index_rebuilds)?;
        output.set_item("entries_evicted", stats.entries_evicted)?;
        for (name, value) in &stats.extra {
            output.set_item(name, value)?;
        }
        output.set_item("entries", store.entries_len())?;
        output.set_item("resident_bytes", store.resident_bytes())?;
        let native = store.native_stats();
        output.set_item("append_paths", native.path_counts)?;
        Ok(output)
    }

    #[getter]
    fn fallback_chain(&self) -> Vec<&'static str> {
        self.router.chain()
    }
}

// ---------------------------------------------------------------- store --

/// Session store (thin facade over the Rust core; see the core crate
/// for semantics).
#[pyclass(module = "toktier._native")]
struct SessionStore {
    inner: toktier_store_core::SessionStore,
}

type PyRecoveryMaterial<'py> = (Bound<'py, PyBytes>, u64, Bound<'py, PyBytes>);

fn map<T>(py: Python<'_>, r: Result<T, StoreError>) -> PyResult<T> {
    r.map_err(|e| err_to_py(py, e))
}

#[pymethods]
impl SessionStore {
    #[new]
    #[pyo3(signature = (block_chars = 4096, tail_soft_cap_bytes = 65536,
                        tail_hard_cap_bytes = 1048576, node_tail_cap_bytes = 65536,
                        max_sessions = 1024, track_recovery = false,
                        track_content_index = false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        block_chars: u64,
        tail_soft_cap_bytes: usize,
        tail_hard_cap_bytes: usize,
        node_tail_cap_bytes: usize,
        max_sessions: usize,
        track_recovery: bool,
        track_content_index: bool,
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
        let mut inner = toktier_store_core::SessionStore::new(cfg).map_err(|e| {
            let _ = py;
            PyValueError::new_err(e.to_string())
        })?;
        if track_recovery {
            inner
                .enable_recovery_tracking()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        if track_content_index {
            inner
                .enable_content_tracking()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(SessionStore { inner })
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
        text: PyBackedStr,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<(u64, u64, u64)> {
        let result = match engine.native() {
            Some(native) => py.allow_threads(|| self.inner.put(KeyId(key_id), &text, &*native)),
            None => self.inner.put(KeyId(key_id), &text, &*engine),
        };
        let out = map(py, result)?;
        Ok((out.handle.0, out.revision, out.token_count))
    }

    /// Certified append under optimistic concurrency. Returns a dict:
    /// `path`, `revision`, `replace_from`, `replacement_ids` (bytes,
    /// u32 LE), `all_ids` (bytes, u32 LE), `n_ids`.
    fn append<'py>(
        &mut self,
        py: Python<'py>,
        handle: u64,
        delta: PyBackedStr,
        expected_revision: u64,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = match engine.native() {
            Some(native) => py.allow_threads(|| {
                self.inner
                    .append(SessionHandle(handle), &delta, expected_revision, &*native)
            }),
            None => self
                .inner
                .append(SessionHandle(handle), &delta, expected_revision, &*engine),
        };
        let out = map(py, result)?;
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
        text: PyBackedStr,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<Option<(u64, u64, u64)>> {
        let result = match engine.native() {
            Some(native) => py.allow_threads(|| self.inner.lookup(KeyId(key_id), &text, &*native)),
            None => self.inner.lookup(KeyId(key_id), &text, &*engine),
        };
        let hit = map(py, result)?;
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

    /// Private facade recovery material. `None` means the native session
    /// does not possess every historical text byte needed to bind it.
    fn recovery_material<'py>(
        &self,
        py: Python<'py>,
        handle: u64,
    ) -> PyResult<Option<PyRecoveryMaterial<'py>>> {
        let material = map(py, self.inner.recovery_material(SessionHandle(handle)))?;
        Ok(material.map(|value| {
            (
                PyBytes::new(py, &value.record_hash),
                value.text_bytes,
                PyBytes::new(py, &value.text_digest),
            )
        }))
    }

    /// Native personalized-BLAKE2b endpoint and geometric checkpoints.
    fn content_index_entry(
        &self,
        py: Python<'_>,
        handle: u64,
    ) -> PyResult<Option<PyContentIndexEntry>> {
        let entry = map(py, self.inner.content_index_entry(SessionHandle(handle)))?;
        Ok(entry.map(|row| {
            (
                row.byte_length,
                hex(&row.end_digest),
                row.marks
                    .into_iter()
                    .map(|(position, digest)| (position, hex(&digest)))
                    .collect(),
            )
        }))
    }

    /// Canonical TKFR-v1 bytes assembled from resident incremental states.
    fn export_recovery_binding<'py>(
        &self,
        py: Python<'py>,
        handle: u64,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let raw = map(
            py,
            self.inner.export_recovery_binding(SessionHandle(handle)),
        )?;
        Ok(raw.map(|bytes| PyBytes::new(py, &bytes)))
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
        rec: PyBackedBytes,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<u64> {
        let result = match engine.native() {
            Some(native) => {
                py.allow_threads(|| self.inner.import_session(KeyId(key_id), &rec, &*native))
            }
            None => self.inner.import_session(KeyId(key_id), &rec, &*engine),
        };
        map(py, result).map(|h| h.0)
    }

    /// Import a session only after caller-presented historical text is
    /// bound to this exact record and private recovery digest.
    fn import_session_with_recovery(
        &mut self,
        py: Python<'_>,
        key_id: u32,
        rec: PyBackedBytes,
        historical_text: PyBackedStr,
        expected_material: (Vec<u8>, u64, Vec<u8>),
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<u64> {
        let (expected_record_hash, expected_text_bytes, expected_text_digest) = expected_material;
        let record_hash = digest32_of(&expected_record_hash, "expected_record_hash")?;
        let text_digest = digest32_of(&expected_text_digest, "expected_text_digest")?;
        let expected = RecoveryMaterial {
            record_hash,
            text_bytes: expected_text_bytes,
            text_digest,
        };
        let result = match engine.native() {
            Some(native) => py.allow_threads(|| {
                self.inner.import_session_with_recovery(
                    KeyId(key_id),
                    &rec,
                    &historical_text,
                    &expected,
                    &*native,
                )
            }),
            None => self.inner.import_session_with_recovery(
                KeyId(key_id),
                &rec,
                &historical_text,
                &expected,
                &*engine,
            ),
        };
        map(py, result).map(|h| h.0)
    }

    /// Import a session from a record plus its sidecar (exact restore).
    fn import_session_with_sidecar(
        &mut self,
        py: Python<'_>,
        key_id: u32,
        rec: PyBackedBytes,
        sidecar: PyBackedBytes,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<u64> {
        let result = match engine.native() {
            Some(native) => py.allow_threads(|| {
                self.inner
                    .import_session_with_sidecar(KeyId(key_id), &rec, &sidecar, &*native)
            }),
            None => self
                .inner
                .import_session_with_sidecar(KeyId(key_id), &rec, &sidecar, &*engine),
        };
        map(py, result).map(|h| h.0)
    }

    /// Import with TKFR-v1, slicing the historical prefix from the caller's
    /// complete transcript under the released GIL.
    fn import_session_with_binding(
        &mut self,
        py: Python<'_>,
        key_id: u32,
        rec: PyBackedBytes,
        candidate_text: PyBackedStr,
        binding: PyBackedBytes,
        engine: PyRef<'_, CallbackEncoder>,
    ) -> PyResult<(u64, usize)> {
        let result = match engine.native() {
            Some(native) => py.allow_threads(|| {
                self.inner.import_session_with_binding_candidate(
                    KeyId(key_id),
                    &rec,
                    &candidate_text,
                    &binding,
                    &*native,
                )
            }),
            None => self.inner.import_session_with_binding_candidate(
                KeyId(key_id),
                &rec,
                &candidate_text,
                &binding,
                &*engine,
            ),
        };
        map(py, result).map(|(handle, historical_chars)| (handle.0, historical_chars))
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
    m.add_class::<NativeReferenceEngine>()?;
    m.add_class::<PyNativePrebuiltGpu>()?;
    m.add_class::<PyNativeRuntime>()?;
    m.add_function(wrap_pyfunction!(fast_cpu_build_facts, m)?)?;
    m.add_function(wrap_pyfunction!(native_host_build_facts, m)?)?;
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
