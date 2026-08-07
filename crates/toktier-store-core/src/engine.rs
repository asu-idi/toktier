//! The encoder abstraction the store is written against.
//!
//! The store owns bookkeeping only. All tokenization semantics -- full
//! encodes, certified appends, and the boundary certificates that allow
//! sealing -- are delegated to a [`SessionEncoder`] supplied by the
//! caller. The store contains no tokenizer-family logic and no opinion
//! about how the encoder produces its results; it only enforces the
//! structural contract stated on each trait method and rejects encoders
//! that break it.

use crate::error::StoreError;
use crate::tail::TailState;

/// A full encode: token ids with character-unit spans.
///
/// Spans are `(start_char, end_char)` pairs over the encoded text, in
/// Unicode scalar values (the `len(str)` convention of CPython). Byte
/// fallback tokenizers may emit several tokens sharing one character
/// span; the store handles those groups and never splits them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub spans: Vec<(u32, u32)>,
}

impl Encoding {
    /// Check the structural facts every consumer of an encoding relies
    /// on: `ids` and `spans` have equal length, every span is ordered,
    /// and no span reaches past `text_chars`. Called at every engine
    /// boundary before state mutation, so a malformed encoder result
    /// fails here instead of surfacing later as span arithmetic gone
    /// wrong. Ordering *between* spans is deliberately not enforced:
    /// byte-fallback encoders emit token groups sharing one span.
    pub fn validate(&self, text_chars: u32) -> Result<(), StoreError> {
        if self.ids.len() != self.spans.len() {
            return Err(StoreError::InvalidInput(format!(
                "ids/spans length mismatch: {} != {}",
                self.ids.len(),
                self.spans.len()
            )));
        }
        for (index, &(start, end)) in self.spans.iter().enumerate() {
            if start > end || end > text_chars {
                return Err(StoreError::InvalidInput(format!(
                    "span {index} ({start}, {end}) is reversed or exceeds \
                     the text length {text_chars}"
                )));
            }
        }
        Ok(())
    }
}

/// What an append did to the tail state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendReport {
    /// Encoder-defined path label (for example `cold_full`, `healed`,
    /// `fallback_full`). Recorded verbatim in the store statistics.
    pub path: String,
    /// Number of leading tokens of the previous tail encoding that are
    /// unchanged. The store verifies this claim against a snapshot and
    /// uses it to compute the `replace_from` index of the append result.
    pub kept_tokens: usize,
}

/// A certified split point inside a tail: tokens `[0, cut_tokens)` cover
/// characters `[0, cut_char)`, and `cut_char` is a certified boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryCut {
    pub cut_tokens: usize,
    pub cut_char: u64,
}

/// Witness category registry (store format v1, frozen; u16,
/// append-only, values assigned in `docs/contracts/store-format-v1.md`
/// Section 3).
///
/// The category records which class of certification predicate proved
/// the current safe cut point. Decoders must reject any value outside
/// this registry (`STORE_FORMAT_UNSUPPORTED`): a stable prefix
/// certified by an unknown predicate class cannot be trusted.
/// Predicate profile parameters (including family-specific sync
/// profiles) are bound inside the semantic fingerprint; the category
/// records only the predicate class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum WitnessCategory {
    /// No certified incremental cut backs the record; the stream came
    /// from a from-scratch encode. Cross-field invariant (frozen):
    /// records under this category have `stable_prefix_byte_length == 0`
    /// and `replace_token_offset == 0`. Engines without certificates
    /// use this category and never seal.
    NoneFullReencode = 0x0000,
    /// Byte-level BPE synchronizing-transition predicate.
    BpeSyncTransition = 0x0001,
    /// WordPiece continuation witness-anchor predicate.
    WordpieceContinuation = 0x0002,
    /// Metaspace / word-start-marker predicate.
    MetaspaceWordStart = 0x0003,
}

impl WitnessCategory {
    /// Parse the on-disk value. Unknown values are a format rejection.
    pub fn from_u16(v: u16) -> Result<WitnessCategory, StoreError> {
        Ok(match v {
            0x0000 => WitnessCategory::NoneFullReencode,
            0x0001 => WitnessCategory::BpeSyncTransition,
            0x0002 => WitnessCategory::WordpieceContinuation,
            0x0003 => WitnessCategory::MetaspaceWordStart,
            other => return Err(StoreError::UnknownWitnessCategory(other)),
        })
    }

    pub fn as_u16(self) -> u16 {
        self as u16
    }
}

/// Error type reported by encoders. The store wraps it into
/// [`StoreError::Engine`] without interpretation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for EngineError {}

impl From<EngineError> for StoreError {
    fn from(e: EngineError) -> StoreError {
        StoreError::Engine(e.0)
    }
}

/// The tokenization engine a store operation runs against.
///
/// Correctness contract (the store's guarantees are conditional on it):
///
/// * [`encode`](Self::encode) returns the reference encoding of `text`
///   as a pre-postprocessor core stream (no special-token wrapping),
///   with character-unit spans, `ids.len() == spans.len()`.
/// * [`append`](Self::append) extends the tail in place so that
///   afterwards `tail.text() == old_text + delta` and the tail holds the
///   reference encoding of that text; the first
///   [`kept_tokens`](AppendReport::kept_tokens) tokens must be unchanged
///   from the previous encoding. The store snapshots the previous ids
///   and verifies the kept prefix; a violation is reported as an engine
///   error, never silently accepted.
/// * [`last_certified_boundary`](Self::last_certified_boundary) may only
///   return cuts that are certified safe: re-encoding the text on either
///   side of the cut independently must reproduce the exact token
///   stream. Engines without certificates return `None`; the store then
///   simply never seals, which is correct (the tail grows and the caps
///   count it).
/// * [`witness_category`](Self::witness_category) names the predicate
///   class behind those certificates and is stamped into every record.
pub trait SessionEncoder {
    /// Reference encoding of `text` (pre-postprocessor core stream).
    fn encode(&self, text: &str) -> Result<Encoding, EngineError>;

    /// Append `delta` to the tail in place; see the trait contract.
    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError>;

    /// Last certified split point with tail-local `cut_char` in
    /// `(floor_char, ceil_char]`, or `None` when no certified boundary
    /// exists in that range (or the engine issues no certificates).
    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError>;

    /// The witness predicate category backing this engine's certificates.
    fn witness_category(&self) -> WitnessCategory;
}
