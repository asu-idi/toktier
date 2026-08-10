//! The encoder abstraction the store is written against.
//!
//! The store owns bookkeeping only. All tokenization semantics -- full
//! encodes, certified appends, and the boundary certificates that allow
//! sealing -- are delegated to a [`SessionEncoder`] supplied by the
//! caller. The store contains no tokenizer-family logic and no opinion
//! about how the encoder produces its results; it only enforces the
//! structural contract stated on each trait method and rejects encoders
//! that break it.

use std::sync::Arc;

use crate::error::StoreError;
use crate::tail::TailState;

/// Immutable shared ownership of one contiguous token-ID allocation.
///
/// The buffer has exactly one owning allocation (the `Arc`ed vector) and
/// any number of cheap immutable range views into it. Adopting an
/// engine-produced `Vec<u32>` moves the allocation without copying it,
/// so the router, the session store, and the public result type can all
/// observe the same memory. Nothing in this crate ever writes through a
/// `SharedIds`; mutation always happens in separately owned storage
/// (copy-on-write at the mutable tail), so a view handed out earlier can
/// never change underneath its holder.
#[derive(Debug, Clone)]
pub struct SharedIds {
    buf: Arc<Vec<u32>>,
    start: usize,
    end: usize,
}

impl SharedIds {
    /// Adopt an owned row as the single shared allocation (no copy).
    pub fn from_vec(ids: Vec<u32>) -> SharedIds {
        let end = ids.len();
        SharedIds {
            buf: Arc::new(ids),
            start: 0,
            end,
        }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.buf[self.start..self.end]
    }

    /// A sub-view `[lo, hi)` of this view (indices are view-relative).
    pub fn slice(&self, lo: usize, hi: usize) -> Result<SharedIds, StoreError> {
        if lo > hi || hi > self.len() {
            return Err(StoreError::InvalidInput(format!(
                "shared id slice [{lo}, {hi}) is outside its parent of length {}",
                self.len()
            )));
        }
        Ok(SharedIds {
            buf: Arc::clone(&self.buf),
            start: self.start + lo,
            end: self.start + hi,
        })
    }

    /// Whether both views share one owning allocation.
    pub fn same_allocation(&self, other: &SharedIds) -> bool {
        Arc::ptr_eq(&self.buf, &other.buf)
    }

    /// Join two views into one when they are adjacent ranges of the same
    /// allocation; `None` otherwise.
    pub fn join_adjacent(&self, other: &SharedIds) -> Option<SharedIds> {
        (self.same_allocation(other) && self.end == other.start).then(|| SharedIds {
            buf: Arc::clone(&self.buf),
            start: self.start,
            end: other.end,
        })
    }

    /// The owning allocation and this view's absolute bounds inside it.
    /// Consumers (for example a public buffer type) may retain the `Arc`
    /// to keep observing the same immutable memory.
    pub fn into_parts(self) -> (Arc<Vec<u32>>, usize, usize) {
        (self.buf, self.start, self.end)
    }
}

impl PartialEq for SharedIds {
    fn eq(&self, other: &SharedIds) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for SharedIds {}

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

/// A full encode carried in the store's structure-of-arrays span layout.
///
/// Semantics match [`Encoding`]: `span_starts[i]..span_ends[i]` is the
/// character-unit span of token `i`, and byte-fallback token groups may
/// share one span. Carrying the two final `u32` arrays directly lets an
/// engine result be adopted by [`TailState::fill_soa`] without forming
/// pair tuples that the tail would immediately split again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoaEncoding {
    pub ids: Vec<u32>,
    pub span_starts: Vec<u32>,
    pub span_ends: Vec<u32>,
}

impl SoaEncoding {
    /// The structural checks of [`Encoding::validate`] over the
    /// structure-of-arrays layout: all three arrays have equal length,
    /// every span is ordered, and no span reaches past `text_chars`.
    /// Ordering *between* spans is deliberately not enforced (byte
    /// fallback).
    pub fn validate(&self, text_chars: u32) -> Result<(), StoreError> {
        if self.span_starts.len() != self.ids.len() || self.span_ends.len() != self.ids.len() {
            return Err(StoreError::InvalidInput(format!(
                "ids/spans length mismatch: {} != {}/{}",
                self.ids.len(),
                self.span_starts.len(),
                self.span_ends.len()
            )));
        }
        validate_span_arrays(&self.span_starts, &self.span_ends, text_chars)
    }

    /// Materialize the pair-based [`Encoding`] view (used by the retained
    /// pair-based surfaces; the state path adopts the arrays directly).
    pub fn into_pairs(self) -> Encoding {
        let spans = self
            .span_starts
            .into_iter()
            .zip(self.span_ends)
            .collect::<Vec<_>>();
        Encoding {
            ids: self.ids,
            spans,
        }
    }
}

/// Per-span structural checks shared by every span-array carrier: each
/// span ordered and no span past `text_chars`. Ordering *between* spans
/// is deliberately not enforced (byte fallback). One definition serves
/// [`SoaEncoding::validate`] and the tail's shared-range fill so the
/// rejection text cannot drift.
pub(crate) fn validate_span_arrays(
    starts: &[u32],
    ends: &[u32],
    text_chars: u32,
) -> Result<(), StoreError> {
    for (index, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        if start > end || end > text_chars {
            return Err(StoreError::InvalidInput(format!(
                "span {index} ({start}, {end}) is reversed or exceeds \
                 the text length {text_chars}"
            )));
        }
    }
    Ok(())
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

/// Joins one background integrity job with one foreground encode so
/// that both have finished when the call returns (PLAN/162 WP5/WP6
/// seed overlap).
///
/// The store hands `background` the seed content-digest scan over the
/// already-host-resident input and keeps `foreground` (the routed
/// encode) for the calling thread. Contract:
///
/// * `foreground` runs exactly once, on the calling thread. The type
///   system supports this: only `background` carries a `Send` bound.
/// * `background` should run exactly once, preferably on a bounded
///   runtime worker; implementations must not spawn an unbounded
///   thread per call.
/// * Both closures have returned before `run_joined` returns. Safe
///   implementations cannot retain either closure past the call, since
///   both borrow from the caller's stack frame.
/// * Neither closure blocks on locks: each one only reads the input
///   text and writes its own result slot.
///
/// The store fails closed around an implementation that skips a
/// closure: a skipped `foreground` surfaces as an internal error, and
/// a skipped `background` falls back to the serial digest computation.
pub trait OverlapRunner: Send + Sync {
    /// Execute both closures; return only after both have completed.
    fn run_joined(&self, background: &mut (dyn FnMut() + Send), foreground: &mut dyn FnMut());

    /// How many bounded workers may serve `background`. Observability
    /// surfaces (for example the seed profiler's environment records)
    /// report this next to overlap on/off readings.
    fn worker_count(&self) -> usize;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_ids_adopt_slice_and_join_share_one_allocation() {
        let row: Vec<u32> = (0..100).collect();
        let expected_ptr = row.as_ptr();
        let whole = SharedIds::from_vec(row);
        // Adoption moves the allocation; it does not copy it.
        assert_eq!(whole.as_slice().as_ptr(), expected_ptr);
        let head = whole.slice(0, 60).unwrap();
        let tail = whole.slice(60, 100).unwrap();
        assert!(head.same_allocation(&tail));
        assert_eq!(head.as_slice(), &(0..60).collect::<Vec<u32>>()[..]);
        assert_eq!(tail.as_slice(), &(60..100).collect::<Vec<u32>>()[..]);
        // Nested slicing stays view-relative.
        let mid = tail.slice(10, 20).unwrap();
        assert_eq!(mid.as_slice(), &(70..80).collect::<Vec<u32>>()[..]);
        // Adjacent views of one allocation join without copying.
        let joined = head.join_adjacent(&tail).unwrap();
        assert_eq!(joined, whole);
        assert_eq!(joined.as_slice().as_ptr(), expected_ptr);
        // Non-adjacent or foreign views do not join.
        assert!(tail.join_adjacent(&head).is_none());
        assert!(head
            .join_adjacent(&SharedIds::from_vec(vec![1, 2]))
            .is_none());
    }

    #[test]
    fn shared_ids_out_of_range_slices_are_rejected() {
        let ids = SharedIds::from_vec(vec![1, 2, 3]);
        assert!(ids.slice(2, 1).is_err());
        assert!(ids.slice(0, 4).is_err());
        assert!(ids.slice(4, 4).is_err());
        // An empty in-range slice is fine.
        assert!(ids.slice(3, 3).unwrap().is_empty());
    }

    #[test]
    fn shared_ids_equality_is_by_content() {
        let a = SharedIds::from_vec(vec![5, 6, 7]);
        let b = SharedIds::from_vec(vec![5, 6, 7]);
        assert_eq!(a, b);
        assert!(!a.same_allocation(&b));
        assert_ne!(a, SharedIds::from_vec(vec![5, 6]));
    }
}
