//! Mutable tail state of a session: the text past the last certified
//! seal point together with its standalone encoding.
//!
//! Layout follows the pre-release prototype's session state: UTF-8 text plus
//! structure-of-arrays token storage (`ids`, `span_start`, `span_end`,
//! spans in character units). All mutation goes through validating
//! methods, so an ill-behaved encoder cannot leave the state with
//! mismatched lengths; illegal states are not representable through the
//! public API.

use std::sync::{Arc, OnceLock};

use crate::engine::{validate_span_arrays, Encoding, SharedIds, SoaEncoding};
use crate::error::StoreError;

/// Token stride between sparse span checkpoints on the lazy span path.
pub const SPAN_CHECKPOINT_STRIDE: usize = 4096;

/// One sparse span checkpoint: the position of token index
/// `k * SPAN_CHECKPOINT_STRIDE` inside the tail text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpanCheckpoint {
    /// Cumulative token byte offset (start byte of the checkpoint token).
    byte: usize,
    /// Byte offset of the start of the character containing `byte`
    /// (equal to `byte` when it falls on a character boundary).
    anchor_byte: usize,
    /// Character index corresponding to `anchor_byte`.
    anchor_char: u32,
}

/// Span storage of a tail: materialized arrays, or the lazy sparse-
/// checkpoint form adopted from a closure-verified engine payload.
///
/// In the lazy form no per-token span exists in memory; any region is
/// rebuilt on demand from the nearest checkpoint, and the values are
/// element-identical to what a full structure-of-arrays conversion of
/// the same row would have produced (the fill-time fused scan proves
/// every precondition, so a rebuild cannot fail on committed state).
#[derive(Debug, Clone)]
enum TailSpans {
    Materialized {
        start: Vec<u32>,
        end: Vec<u32>,
    },
    Lazy {
        /// Raw byte length of every vocabulary ID (the frozen table the
        /// engine derives spans from; opaque bytes to the store).
        table: Arc<[usize]>,
        checkpoints: Box<[SpanCheckpoint]>,
        /// Whole-row materialization cache for the slice accessors.
        cache: OnceLock<(Vec<u32>, Vec<u32>)>,
    },
}

impl Default for TailSpans {
    fn default() -> TailSpans {
        TailSpans::Materialized {
            start: Vec::new(),
            end: Vec::new(),
        }
    }
}

/// Raw byte length of `id` under the frozen table, with the lazy path's
/// unknown-ID and zero-length rejections.
fn lazy_token_len(table: &[usize], id: u32) -> Result<usize, StoreError> {
    match table.get(id as usize).copied() {
        Some(length) if length > 0 => Ok(length),
        Some(_) => Err(StoreError::InvalidInput(format!(
            "lazy span table maps id {id} to zero bytes"
        ))),
        None => Err(StoreError::InvalidInput(format!(
            "lazy span table has no entry for id {id}"
        ))),
    }
}

/// Forward character walker used to anchor checkpoints during the
/// fill-time fused scan. Queries must be monotonically nondecreasing
/// byte offsets strictly inside the text.
struct CharWalker<'a> {
    iter: std::str::CharIndices<'a>,
    start: usize,
    end: usize,
    index: u32,
}

impl<'a> CharWalker<'a> {
    fn new(text: &'a str) -> CharWalker<'a> {
        let mut iter = text.char_indices();
        let end = iter.next().map_or(0, |(_, value)| value.len_utf8());
        CharWalker {
            iter,
            start: 0,
            end,
            index: 0,
        }
    }

    /// `(char start byte, char index)` of the character containing `byte`.
    fn anchor_at(&mut self, byte: usize) -> (usize, u32) {
        while self.end <= byte {
            let Some((offset, value)) = self.iter.next() else {
                break;
            };
            self.start = offset;
            self.end = offset + value.len_utf8();
            self.index += 1;
        }
        (self.start, self.index)
    }
}

/// Streaming byte-to-character cursor anchored at a checkpoint: the
/// character containing absolute byte offset `base_byte` has index
/// `base_char` and starts exactly at `base_byte`. Queries must be
/// monotonically nondecreasing, which token endpoint order guarantees.
struct AnchoredCharCursor<'a> {
    rest: std::str::CharIndices<'a>,
    base_byte: usize,
    index: u32,
    start: usize,
    end: usize,
    exhausted: bool,
}

impl<'a> AnchoredCharCursor<'a> {
    fn new(text: &'a str, base_byte: usize, base_char: u32) -> AnchoredCharCursor<'a> {
        let mut rest = text[base_byte..].char_indices();
        match rest.next() {
            Some((_, first)) => AnchoredCharCursor {
                rest,
                base_byte,
                index: base_char,
                start: base_byte,
                end: base_byte + first.len_utf8(),
                exhausted: false,
            },
            None => AnchoredCharCursor {
                rest,
                base_byte,
                index: base_char,
                start: base_byte,
                end: base_byte,
                exhausted: true,
            },
        }
    }

    /// Index of the character containing absolute byte offset `byte`, or
    /// `None` when the offset lies outside the anchored suffix.
    fn char_containing(&mut self, byte: usize) -> Option<u32> {
        if self.exhausted || byte < self.start {
            return None;
        }
        while self.end <= byte {
            let (offset, value) = self.rest.next()?;
            self.index += 1;
            self.start = self.base_byte + offset;
            self.end = self.start + value.len_utf8();
        }
        Some(self.index)
    }
}

/// Fused validation and checkpoint construction: one pass over `ids`
/// proves every ID has a nonzero table byte length, the byte cursor
/// cannot overflow, and the row closes exactly over `text`, while a
/// checkpoint is recorded every [`SPAN_CHECKPOINT_STRIDE`] tokens. After
/// this scan succeeds, any window rebuild over the same inputs is total.
fn build_span_checkpoints(
    text: &str,
    text_chars: u32,
    ids: &[u32],
    table: &[usize],
) -> Result<Box<[SpanCheckpoint]>, StoreError> {
    let ascii = text.len() == text_chars as usize;
    let mut checkpoints = Vec::with_capacity(ids.len() / SPAN_CHECKPOINT_STRIDE + 1);
    let mut walker = CharWalker::new(text);
    let mut byte = 0usize;
    for (index, &id) in ids.iter().enumerate() {
        if index.is_multiple_of(SPAN_CHECKPOINT_STRIDE) {
            let (anchor_byte, anchor_char) = if ascii {
                (byte, byte as u32)
            } else {
                walker.anchor_at(byte)
            };
            checkpoints.push(SpanCheckpoint {
                byte,
                anchor_byte,
                anchor_char,
            });
        }
        byte = byte
            .checked_add(lazy_token_len(table, id)?)
            .ok_or_else(|| StoreError::InvalidInput("lazy span byte cursor overflow".into()))?;
    }
    if byte != text.len() {
        return Err(StoreError::InvalidInput(format!(
            "lazy span payload does not close over the text: tokens={byte}, text={}",
            text.len()
        )));
    }
    Ok(checkpoints.into_boxed_slice())
}

/// Rebuild the character spans of tokens `[lo, hi)` from the nearest
/// checkpoint. Values are element-identical to a full conversion of the
/// row; on state committed through `fill_lazy` this cannot fail.
fn rebuild_span_window(
    text: &str,
    text_chars: u32,
    table: &[usize],
    checkpoints: &[SpanCheckpoint],
    ids: &[u32],
    lo: usize,
    hi: usize,
) -> Result<(Vec<u32>, Vec<u32>), StoreError> {
    debug_assert!(lo <= hi && hi <= ids.len());
    if lo == hi {
        return Ok((Vec::new(), Vec::new()));
    }
    let cp_index = lo / SPAN_CHECKPOINT_STRIDE;
    let cp = checkpoints.get(cp_index).copied().ok_or_else(|| {
        StoreError::Internal("lazy span state is missing a required checkpoint".into())
    })?;
    let outside = || StoreError::Internal("lazy span rebuild left the validated text".into());
    let ascii = text.len() == text_chars as usize;
    let mut byte = cp.byte;
    for &id in &ids[cp_index * SPAN_CHECKPOINT_STRIDE..lo] {
        byte = byte
            .checked_add(lazy_token_len(table, id)?)
            .ok_or_else(outside)?;
    }
    let mut starts = Vec::with_capacity(hi - lo);
    let mut ends = Vec::with_capacity(hi - lo);
    if ascii {
        for &id in &ids[lo..hi] {
            let start = byte;
            byte = byte
                .checked_add(lazy_token_len(table, id)?)
                .ok_or_else(outside)?;
            if byte > text.len() {
                return Err(outside());
            }
            starts.push(start as u32);
            ends.push(byte as u32);
        }
    } else {
        let mut cursor = AnchoredCharCursor::new(text, cp.anchor_byte, cp.anchor_char);
        for &id in &ids[lo..hi] {
            let start_byte = byte;
            byte = byte
                .checked_add(lazy_token_len(table, id)?)
                .ok_or_else(outside)?;
            let start_char = cursor.char_containing(start_byte).ok_or_else(outside)?;
            let end_char = cursor
                .char_containing(byte - 1)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(outside)?;
            starts.push(start_char);
            ends.push(end_char);
        }
    }
    Ok((starts, ends))
}

/// Tail ID storage: either an owned mutable row or an immutable shared
/// range (typically the residual of an adopted engine allocation).
///
/// A shared range is never written through; the first mutating
/// operation copies the (small) view into owned storage, so any other
/// holder of the same allocation keeps observing unchanged memory.
#[derive(Debug, Clone)]
enum TailIds {
    Owned(Vec<u32>),
    Shared(SharedIds),
}

impl Default for TailIds {
    fn default() -> TailIds {
        TailIds::Owned(Vec::new())
    }
}

impl TailIds {
    fn as_slice(&self) -> &[u32] {
        match self {
            TailIds::Owned(ids) => ids,
            TailIds::Shared(view) => view.as_slice(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Copy-on-write access for mutation.
    fn to_mut(&mut self) -> &mut Vec<u32> {
        if let TailIds::Shared(view) = self {
            *self = TailIds::Owned(view.as_slice().to_vec());
        }
        match self {
            TailIds::Owned(ids) => ids,
            TailIds::Shared(_) => unreachable!("copy-on-write just produced owned storage"),
        }
    }
}

/// Tail text and its encoding. Spans are character-unit `(start, end)`
/// pairs relative to the tail origin.
#[derive(Debug, Clone, Default)]
pub struct TailState {
    text: String,
    text_chars: u32,
    ids: TailIds,
    spans: TailSpans,
}

fn chars_u32(text: &str) -> Result<u32, StoreError> {
    u32::try_from(text.chars().count())
        .map_err(|_| StoreError::InvalidInput("text exceeds u32 character capacity".into()))
}

impl TailState {
    /// Empty tail.
    pub fn new() -> TailState {
        TailState::default()
    }

    /// Replace the whole state with `text` and its encoding.
    pub fn fill(&mut self, text: &str, enc: Encoding) -> Result<(), StoreError> {
        let chars = chars_u32(text)?;
        enc.validate(chars)?;
        self.text.clear();
        self.text.push_str(text);
        self.text_chars = chars;
        self.ids = TailIds::Owned(enc.ids);
        let mut start = Vec::with_capacity(enc.spans.len());
        let mut end = Vec::with_capacity(enc.spans.len());
        for (a, b) in enc.spans {
            start.push(a);
            end.push(b);
        }
        self.spans = TailSpans::Materialized { start, end };
        Ok(())
    }

    /// Replace the whole state with `text` and its structure-of-arrays
    /// encoding. The validated arrays are adopted as-is: no pair vector
    /// is formed and no per-token copy runs.
    pub fn fill_soa(&mut self, text: &str, enc: SoaEncoding) -> Result<(), StoreError> {
        let chars = chars_u32(text)?;
        enc.validate(chars)?;
        self.text.clear();
        self.text.push_str(text);
        self.text_chars = chars;
        self.ids = TailIds::Owned(enc.ids);
        self.spans = TailSpans::Materialized {
            start: enc.span_starts,
            end: enc.span_ends,
        };
        Ok(())
    }

    /// Replace the whole state with `text`, an immutable shared ID range,
    /// and already-materialized span arrays. Validation matches
    /// [`Self::fill_soa`]; the shared range is adopted without copying and
    /// is only ever copied out again if a later splice must mutate it.
    pub fn fill_shared(
        &mut self,
        text: &str,
        ids: SharedIds,
        span_starts: Vec<u32>,
        span_ends: Vec<u32>,
    ) -> Result<(), StoreError> {
        let chars = chars_u32(text)?;
        if span_starts.len() != ids.len() || span_ends.len() != ids.len() {
            return Err(StoreError::InvalidInput(format!(
                "ids/spans length mismatch: {} != {}/{}",
                ids.len(),
                span_starts.len(),
                span_ends.len()
            )));
        }
        validate_span_arrays(&span_starts, &span_ends, chars)?;
        self.text.clear();
        self.text.push_str(text);
        self.text_chars = chars;
        self.ids = TailIds::Shared(ids);
        self.spans = TailSpans::Materialized {
            start: span_starts,
            end: span_ends,
        };
        Ok(())
    }

    /// Replace the whole state with `text`, an immutable shared ID range,
    /// and lazily rebuildable spans (sparse checkpoints instead of
    /// per-token arrays).
    ///
    /// `token_byte_lengths` is the frozen raw byte length of every
    /// vocabulary ID, exactly the table the engine's span converters
    /// consume. The fill runs one fused scan that validates the payload
    /// (unknown ID, zero-length ID, byte-cursor overflow, and exact byte
    /// closure over `text` are rejected before any state changes) and
    /// records a checkpoint every [`SPAN_CHECKPOINT_STRIDE`] tokens, so
    /// later region rebuilds are total and element-identical to a full
    /// conversion. This is deliberately stricter than [`Self::fill_soa`]:
    /// a non-closing row cannot be represented lazily at all.
    pub fn fill_lazy(
        &mut self,
        text: &str,
        ids: SharedIds,
        token_byte_lengths: Arc<[usize]>,
    ) -> Result<(), StoreError> {
        let chars = chars_u32(text)?;
        let checkpoints = build_span_checkpoints(text, chars, ids.as_slice(), &token_byte_lengths)?;
        self.text.clear();
        self.text.push_str(text);
        self.text_chars = chars;
        self.ids = TailIds::Shared(ids);
        self.spans = TailSpans::Lazy {
            table: token_byte_lengths,
            checkpoints,
            cache: OnceLock::new(),
        };
        Ok(())
    }

    /// The tail's ID storage as an immutable shared range, when it still
    /// is one (before any copy-on-write mutation).
    pub fn shared_ids(&self) -> Option<&SharedIds> {
        match &self.ids {
            TailIds::Shared(view) => Some(view),
            TailIds::Owned(_) => None,
        }
    }

    /// Truncate the token stream to `cut_idx` tokens, then append
    /// `delta_text` to the text and the new tokens to the stream.
    pub fn splice(
        &mut self,
        cut_idx: usize,
        delta_text: &str,
        new: Encoding,
    ) -> Result<(), StoreError> {
        if cut_idx > self.ids.len() {
            return Err(StoreError::InvalidInput(format!(
                "splice cut index {cut_idx} exceeds token count {}",
                self.ids.len()
            )));
        }
        let added = u32::try_from(delta_text.chars().count())
            .ok()
            .and_then(|d| self.text_chars.checked_add(d))
            .ok_or_else(|| {
                StoreError::InvalidInput("text exceeds u32 character capacity".into())
            })?;
        new.validate(added)?;
        self.materialize_spans();
        let ids = self.ids.to_mut();
        ids.truncate(cut_idx);
        ids.extend_from_slice(&new.ids);
        let TailSpans::Materialized { start, end } = &mut self.spans else {
            unreachable!("spans were just materialized");
        };
        start.truncate(cut_idx);
        end.truncate(cut_idx);
        start.reserve(new.spans.len());
        end.reserve(new.spans.len());
        for (a, b) in new.spans {
            start.push(a);
            end.push(b);
        }
        self.text.push_str(delta_text);
        self.text_chars = added;
        Ok(())
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn text_chars(&self) -> u32 {
        self.text_chars
    }

    pub fn text_bytes(&self) -> usize {
        self.text.len()
    }

    pub fn ids(&self) -> &[u32] {
        self.ids.as_slice()
    }

    pub fn n_tokens(&self) -> usize {
        self.ids.len()
    }

    pub fn span_starts(&self) -> &[u32] {
        self.forced_spans().0
    }

    pub fn span_ends(&self) -> &[u32] {
        self.forced_spans().1
    }

    /// Spans materialized as pairs (convenience for encoder adapters).
    pub fn spans(&self) -> Vec<(u32, u32)> {
        let (start, end) = self.forced_spans();
        start
            .iter()
            .zip(end.iter())
            .map(|(&a, &b)| (a, b))
            .collect()
    }

    /// Whether per-token spans currently exist in memory. On the lazy
    /// path they do not, and region consumers should prefer
    /// [`Self::span_window`] over the whole-row slice accessors.
    pub fn spans_materialized(&self) -> bool {
        match &self.spans {
            TailSpans::Materialized { .. } => true,
            TailSpans::Lazy { cache, .. } => cache.get().is_some(),
        }
    }

    /// Character spans of tokens `[lo, hi)` as owned arrays. On the lazy
    /// path the region is rebuilt from its nearest sparse checkpoint;
    /// values are element-identical to the whole-row materialization.
    pub fn span_window(&self, lo: usize, hi: usize) -> Result<(Vec<u32>, Vec<u32>), StoreError> {
        let n_tokens = self.ids.len();
        if lo > hi || hi > n_tokens {
            return Err(StoreError::InvalidInput(format!(
                "span window [{lo}, {hi}) is outside the token count {n_tokens}"
            )));
        }
        match &self.spans {
            TailSpans::Materialized { start, end } => {
                Ok((start[lo..hi].to_vec(), end[lo..hi].to_vec()))
            }
            TailSpans::Lazy {
                table,
                checkpoints,
                cache,
            } => {
                if let Some((start, end)) = cache.get() {
                    return Ok((start[lo..hi].to_vec(), end[lo..hi].to_vec()));
                }
                rebuild_span_window(
                    &self.text,
                    self.text_chars,
                    table,
                    checkpoints,
                    self.ids.as_slice(),
                    lo,
                    hi,
                )
            }
        }
    }

    /// Whole-row span slices, rebuilding and caching them on the lazy
    /// path. Committed lazy state was validated by the fill-time fused
    /// scan, so the rebuild cannot fail.
    fn forced_spans(&self) -> (&[u32], &[u32]) {
        match &self.spans {
            TailSpans::Materialized { start, end } => (start, end),
            TailSpans::Lazy {
                table,
                checkpoints,
                cache,
            } => {
                let (start, end) = cache.get_or_init(|| {
                    rebuild_span_window(
                        &self.text,
                        self.text_chars,
                        table,
                        checkpoints,
                        self.ids.as_slice(),
                        0,
                        self.ids.len(),
                    )
                    .expect("lazy span state was validated when it was committed")
                });
                (start, end)
            }
        }
    }

    /// Convert lazy span state into materialized arrays before mutation.
    fn materialize_spans(&mut self) {
        if let TailSpans::Lazy { .. } = &self.spans {
            let (start, end) = {
                let (start, end) = self.forced_spans();
                (start.to_vec(), end.to_vec())
            };
            self.spans = TailSpans::Materialized { start, end };
        }
    }

    /// Byte offset of character index `char_ix` (`text.len()` past the
    /// end). Scans from whichever end is closer in the multibyte case;
    /// O(1) for pure-ASCII text.
    pub fn byte_ix_of_char(&self, char_ix: u32) -> usize {
        byte_ix_of_char(&self.text, self.text_chars, char_ix)
    }
}

/// Byte offset of character index `char_ix` in `s`, where `s_chars` is
/// the total character count of `s`.
pub(crate) fn byte_ix_of_char(s: &str, s_chars: u32, char_ix: u32) -> usize {
    if char_ix == 0 {
        return 0;
    }
    if char_ix >= s_chars {
        return s.len();
    }
    if s.len() == s_chars as usize {
        return char_ix as usize; // pure ASCII: byte == char
    }
    let bytes = s.as_bytes();
    let tail_chars = (s_chars - char_ix) as usize;
    let mut cnt = 0usize;
    for (i, &b) in bytes.iter().enumerate().rev() {
        if (b & 0xC0) != 0x80 {
            cnt += 1;
            if cnt == tail_chars {
                return i;
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_ix_handles_multibyte() {
        let s = "a\u{4f60}b\u{1f642}c"; // "a", a CJK ideograph, "b", an emoji, "c"
        let n = s.chars().count() as u32;
        assert_eq!(byte_ix_of_char(s, n, 0), 0);
        assert_eq!(byte_ix_of_char(s, n, 1), 1);
        assert_eq!(byte_ix_of_char(s, n, 2), 4);
        assert_eq!(byte_ix_of_char(s, n, 3), 5);
        assert_eq!(byte_ix_of_char(s, n, 4), 9);
        assert_eq!(byte_ix_of_char(s, n, 5), 10);
        assert_eq!(byte_ix_of_char(s, n, 99), 10);
    }

    #[test]
    fn fill_and_splice_validate_lengths() {
        let mut t = TailState::new();
        let bad = Encoding {
            ids: vec![1, 2],
            spans: vec![(0, 1)],
        };
        assert!(t.fill("ab", bad).is_err());
        let ok = Encoding {
            ids: vec![1, 2],
            spans: vec![(0, 1), (1, 2)],
        };
        t.fill("ab", ok).unwrap();
        assert_eq!(t.text(), "ab");
        assert_eq!(t.n_tokens(), 2);
        let over = Encoding {
            ids: vec![],
            spans: vec![],
        };
        assert!(t.splice(3, "c", over).is_err());
        t.splice(
            1,
            "c",
            Encoding {
                ids: vec![7],
                spans: vec![(1, 3)],
            },
        )
        .unwrap();
        assert_eq!(t.text(), "abc");
        assert_eq!(t.ids(), &[1, 7]);
        assert_eq!(t.text_chars(), 3);
    }

    #[test]
    fn fill_soa_adopts_arrays_and_matches_the_pair_path() {
        let text = "a\u{4f60}b"; // 3 chars, multibyte in the middle
        let pairs = Encoding {
            ids: vec![5, 6, 7],
            spans: vec![(0, 1), (1, 2), (1, 3)],
        };
        let soa = SoaEncoding {
            ids: vec![5, 6, 7],
            span_starts: vec![0, 1, 1],
            span_ends: vec![1, 2, 3],
        };
        let mut by_pairs = TailState::new();
        by_pairs.fill(text, pairs).unwrap();
        let mut by_soa = TailState::new();
        by_soa.fill_soa(text, soa.clone()).unwrap();
        assert_eq!(by_pairs.text(), by_soa.text());
        assert_eq!(by_pairs.text_chars(), by_soa.text_chars());
        assert_eq!(by_pairs.ids(), by_soa.ids());
        assert_eq!(by_pairs.span_starts(), by_soa.span_starts());
        assert_eq!(by_pairs.span_ends(), by_soa.span_ends());
        assert_eq!(soa.into_pairs().spans, by_pairs.spans());
    }

    #[test]
    fn fill_soa_rejects_the_same_malformed_shapes_as_fill() {
        let mut t = TailState::new();
        // Length mismatch across the three arrays.
        assert!(t
            .fill_soa(
                "ab",
                SoaEncoding {
                    ids: vec![1, 2],
                    span_starts: vec![0],
                    span_ends: vec![1, 2],
                }
            )
            .is_err());
        // Reversed span.
        assert!(t
            .fill_soa(
                "ab",
                SoaEncoding {
                    ids: vec![1],
                    span_starts: vec![2],
                    span_ends: vec![1],
                }
            )
            .is_err());
        // Span past the end of the text.
        assert!(t
            .fill_soa(
                "ab",
                SoaEncoding {
                    ids: vec![1],
                    span_starts: vec![0],
                    span_ends: vec![3],
                }
            )
            .is_err());
        // A failed fill must not have mutated the state.
        assert_eq!(t.text(), "");
        assert_eq!(t.n_tokens(), 0);
    }

    /// Reference span computation with an explicit byte-to-character
    /// map, mirroring the original pair-bridge semantics: a token's span
    /// is [char containing its first byte, char containing its last byte
    /// + 1), so byte-fallback tokens inside one code point share spans.
    fn naive_spans(text: &str, ids: &[u32], table: &[usize]) -> (Vec<u32>, Vec<u32>) {
        let mut char_of_byte = Vec::with_capacity(text.len());
        for (index, value) in text.chars().enumerate() {
            for _ in 0..value.len_utf8() {
                char_of_byte.push(index as u32);
            }
        }
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut cursor = 0usize;
        for &id in ids {
            let length = table[id as usize];
            if text.is_ascii() {
                starts.push(cursor as u32);
                ends.push((cursor + length) as u32);
            } else {
                starts.push(char_of_byte[cursor]);
                ends.push(char_of_byte[cursor + length - 1] + 1);
            }
            cursor += length;
        }
        assert_eq!(cursor, text.len(), "test row must close");
        (starts, ends)
    }

    /// Assert that a lazily filled tail reproduces the reference spans
    /// for the whole row and for every window.
    fn assert_lazy_matches(text: &str, ids: &[u32], table: &[usize]) {
        let (want_starts, want_ends) = naive_spans(text, ids, table);
        let shared_table: Arc<[usize]> = table.to_vec().into();
        let mut lazy = TailState::new();
        lazy.fill_lazy(
            text,
            SharedIds::from_vec(ids.to_vec()),
            Arc::clone(&shared_table),
        )
        .unwrap();
        assert!(!lazy.spans_materialized());
        // Window rebuilds run before the whole-row cache exists.
        let n = ids.len();
        let probes: Vec<(usize, usize)> = vec![
            (0, n),
            (0, n / 2),
            (n / 2, n),
            (n / 3, 2 * n / 3),
            (n.saturating_sub(1), n),
            (n, n),
        ];
        for &(lo, hi) in &probes {
            let (starts, ends) = lazy.span_window(lo, hi).unwrap();
            assert_eq!(starts, want_starts[lo..hi], "starts [{lo}, {hi})");
            assert_eq!(ends, want_ends[lo..hi], "ends [{lo}, {hi})");
        }
        assert!(!lazy.spans_materialized());
        // The whole-row accessors force one cached materialization.
        assert_eq!(lazy.span_starts(), &want_starts[..]);
        assert_eq!(lazy.span_ends(), &want_ends[..]);
        assert!(lazy.spans_materialized());
        // Windows served from the cache agree as well.
        for &(lo, hi) in &probes {
            let (starts, ends) = lazy.span_window(lo, hi).unwrap();
            assert_eq!(starts, want_starts[lo..hi]);
            assert_eq!(ends, want_ends[lo..hi]);
        }
    }

    #[test]
    fn lazy_fill_matches_the_reference_conversion_on_boundary_shapes() {
        // Table: id i has raw byte length i for 1..=8.
        let table: Vec<usize> = (0..9).collect();
        let t = &table[..];
        // ASCII shapes.
        assert_lazy_matches("abc", &[3], t);
        assert_lazy_matches("abcd", &[1, 3], t);
        assert_lazy_matches("12345678", &[8], t);
        // Every UTF-8 width, exact-width tokens.
        assert_lazy_matches("\u{e9}", &[2], t);
        assert_lazy_matches("\u{4f60}", &[3], t);
        assert_lazy_matches("\u{1f642}", &[4], t);
        // Byte-fallback groups sharing one character span.
        assert_lazy_matches("\u{e9}", &[1, 1], t);
        assert_lazy_matches("\u{4f60}", &[1, 1, 1], t);
        assert_lazy_matches("\u{1f642}", &[1, 1, 1, 1], t);
        assert_lazy_matches("\u{1f642}", &[2, 2], t);
        assert_lazy_matches("\u{4f60}", &[1, 2], t);
        assert_lazy_matches("\u{4f60}", &[2, 1], t);
        // Tokens straddling character boundaries.
        assert_lazy_matches("a\u{4f60}b", &[2, 3], t);
        assert_lazy_matches("a\u{4f60}b", &[4, 1], t);
        assert_lazy_matches("a\u{4f60}b", &[1, 3, 1], t);
        // Combining marks, ZWJ emoji, CJK, RTL, CRLF, mixed.
        assert_lazy_matches("q\u{301}", &[1, 2], t);
        assert_lazy_matches("q\u{301}", &[1, 1, 1], t);
        assert_lazy_matches("\u{1f469}\u{200d}\u{1f4bb}", &[4, 3, 4], t);
        assert_lazy_matches("\u{1f469}\u{200d}\u{1f4bb}", &[5, 6], t);
        assert_lazy_matches("\u{4f60}\u{597d}", &[3, 3], t);
        assert_lazy_matches("\u{5d0}\u{5d1}\u{5d2}", &[2, 2, 2], t);
        assert_lazy_matches("\r\n", &[1, 1], t);
        assert_lazy_matches(" ab\u{4f60}\u{1f642}\u{e9}", &[3, 3, 4, 2], t);
        // Empty row over empty text.
        assert_lazy_matches("", &[], t);
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }
    }

    #[test]
    fn lazy_windows_cross_checkpoints_exactly() {
        // Long rows exercise multiple checkpoints; byte-fallback splits
        // of multi-byte characters land on both sides of the stride.
        let pool = [
            'a',
            '7',
            ' ',
            '\u{e9}',
            '\u{4f60}',
            '\u{1f642}',
            '\u{200d}',
            '\u{301}',
            '\r',
            '\n',
        ];
        let table: Vec<usize> = (0..9).collect();
        let by_length = |target: usize| -> u32 { target as u32 };
        let mut rng = Lcg(0x77aa_1122_3344_55ee);
        for round in 0..6 {
            let ascii_only = round % 2 == 0;
            let mut text = String::new();
            // Enough characters that the row comfortably crosses several
            // 4096-token checkpoints when split into 1..=4-byte tokens.
            let chars = 12_000 + rng.below(4_000);
            for _ in 0..chars {
                let value = if ascii_only {
                    pool[rng.below(3)]
                } else {
                    pool[rng.below(pool.len())]
                };
                text.push(value);
            }
            let mut ids = Vec::new();
            let mut remaining = text.len();
            while remaining > 0 {
                let width = 1 + rng.below(4.min(remaining));
                ids.push(by_length(width));
                remaining -= width;
            }
            assert!(ids.len() > SPAN_CHECKPOINT_STRIDE, "row too short to test");
            let (want_starts, want_ends) = naive_spans(&text, &ids, &table);
            let shared_table: Arc<[usize]> = table.clone().into();
            let mut lazy = TailState::new();
            lazy.fill_lazy(
                &text,
                SharedIds::from_vec(ids.clone()),
                Arc::clone(&shared_table),
            )
            .unwrap();
            let n = ids.len();
            // Deliberate probes around every checkpoint boundary plus
            // random windows; all before any full materialization.
            let mut probes = Vec::new();
            let mut mark = SPAN_CHECKPOINT_STRIDE;
            while mark < n {
                probes.push((mark - 3, (mark + 3).min(n)));
                probes.push((mark, (mark + 17).min(n)));
                mark += SPAN_CHECKPOINT_STRIDE;
            }
            for _ in 0..24 {
                let lo = rng.below(n);
                let hi = lo + 1 + rng.below(n - lo);
                probes.push((lo, hi));
            }
            probes.push((0, n));
            for (lo, hi) in probes {
                let (starts, ends) = lazy.span_window(lo, hi).unwrap();
                assert_eq!(starts, want_starts[lo..hi], "starts [{lo}, {hi})");
                assert_eq!(ends, want_ends[lo..hi], "ends [{lo}, {hi})");
            }
            assert!(!lazy.spans_materialized());
        }
    }

    #[test]
    fn lazy_fill_rejects_malformed_payloads_without_mutation() {
        let table: Arc<[usize]> = vec![0usize, 1, 2, 3, 4].into();
        let mut tail = TailState::new();
        tail.fill_soa(
            "keep",
            SoaEncoding {
                ids: vec![1],
                span_starts: vec![0],
                span_ends: vec![4],
            },
        )
        .unwrap();
        let cases: Vec<(&str, Vec<u32>)> = vec![
            // Unknown ID (past the table).
            ("ab", vec![1, 9]),
            // Zero-byte table entry.
            ("ab", vec![1, 0]),
            // Undershoot and overshoot.
            ("abc", vec![1, 1]),
            ("abc", vec![2, 2]),
            // Empty row over non-empty text.
            ("abc", vec![]),
        ];
        for (text, ids) in cases {
            let result = tail.fill_lazy(text, SharedIds::from_vec(ids.clone()), Arc::clone(&table));
            assert!(result.is_err(), "{text:?}/{ids:?} accepted");
            assert_eq!(tail.text(), "keep", "failed fill mutated the tail");
            assert_eq!(tail.ids(), &[1]);
        }
        // Byte-cursor overflow.
        let huge: Arc<[usize]> = vec![usize::MAX, usize::MAX].into();
        let overflow = tail.fill_lazy("ab", SharedIds::from_vec(vec![0, 1]), huge);
        assert!(overflow.is_err());
        assert_eq!(tail.text(), "keep");
    }

    #[test]
    fn splice_on_a_lazy_tail_matches_the_materialized_path() {
        let table: Arc<[usize]> = vec![0usize, 1, 2, 3, 4].into();
        let text = "ab\u{4f60}cd";
        let ids = vec![1, 1, 3, 1, 1];
        let (starts, ends) = naive_spans(text, &ids, &table);
        let mut lazy = TailState::new();
        lazy.fill_lazy(text, SharedIds::from_vec(ids.clone()), Arc::clone(&table))
            .unwrap();
        let mut soa = TailState::new();
        soa.fill_soa(
            text,
            SoaEncoding {
                ids: ids.clone(),
                span_starts: starts,
                span_ends: ends,
            },
        )
        .unwrap();
        let delta = Encoding {
            ids: vec![2, 1],
            spans: vec![(3, 6), (6, 7)],
        };
        lazy.splice(3, "xy", delta.clone()).unwrap();
        soa.splice(3, "xy", delta).unwrap();
        assert_eq!(lazy.text(), soa.text());
        assert_eq!(lazy.ids(), soa.ids());
        assert_eq!(lazy.span_starts(), soa.span_starts());
        assert_eq!(lazy.span_ends(), soa.span_ends());
        assert!(lazy.shared_ids().is_none(), "splice must copy out");
    }

    #[test]
    fn span_window_rejects_out_of_range_regions() {
        let table: Arc<[usize]> = vec![0usize, 1].into();
        let mut tail = TailState::new();
        tail.fill_lazy("aa", SharedIds::from_vec(vec![1, 1]), table)
            .unwrap();
        assert!(tail.span_window(2, 1).is_err());
        assert!(tail.span_window(0, 3).is_err());
        assert!(tail.span_window(3, 3).is_err());
        assert_eq!(tail.span_window(2, 2).unwrap().0.len(), 0);
    }

    #[test]
    fn fill_shared_adopts_the_allocation_and_splices_copy_on_write() {
        let row: Vec<u32> = vec![10, 11, 12];
        let row_ptr = row.as_ptr();
        let shared = SharedIds::from_vec(row);
        let observer = shared.clone();
        let mut tail = TailState::new();
        tail.fill_shared("abc", shared, vec![0, 1, 2], vec![1, 2, 3])
            .unwrap();
        // The tail observes the adopted allocation without copying it.
        assert_eq!(tail.ids().as_ptr(), row_ptr);
        assert_eq!(tail.shared_ids().unwrap().as_slice(), &[10, 11, 12]);
        // Mutation copies out; the shared allocation never changes.
        tail.splice(
            1,
            "d",
            Encoding {
                ids: vec![77, 78],
                spans: vec![(1, 3), (3, 4)],
            },
        )
        .unwrap();
        assert!(tail.shared_ids().is_none());
        assert_eq!(tail.ids(), &[10, 77, 78]);
        assert_eq!(observer.as_slice(), &[10, 11, 12]);
        assert_eq!(observer.as_slice().as_ptr(), row_ptr);
    }

    #[test]
    fn fill_shared_rejects_the_same_malformed_shapes_as_fill_soa() {
        let mut tail = TailState::new();
        // Length mismatch.
        let mismatch = tail.fill_shared("ab", SharedIds::from_vec(vec![1, 2]), vec![0], vec![1, 2]);
        assert!(mismatch.is_err());
        // Reversed span.
        let reversed = tail.fill_shared("ab", SharedIds::from_vec(vec![1]), vec![2], vec![1]);
        // Span past the end.
        let past_end = tail.fill_shared("ab", SharedIds::from_vec(vec![1]), vec![0], vec![3]);
        // The rejection text matches the SoA fill exactly.
        let mut soa = TailState::new();
        let soa_reversed = soa.fill_soa(
            "ab",
            SoaEncoding {
                ids: vec![1],
                span_starts: vec![2],
                span_ends: vec![1],
            },
        );
        assert_eq!(
            reversed.unwrap_err().to_string(),
            soa_reversed.unwrap_err().to_string()
        );
        assert!(past_end.is_err());
        // A failed fill must not have mutated the state.
        assert_eq!(tail.text(), "");
        assert_eq!(tail.n_tokens(), 0);
        assert!(tail.shared_ids().is_none());
    }

    #[test]
    fn fill_and_splice_validate_spans() {
        let mut t = TailState::new();
        let reversed = Encoding {
            ids: vec![1],
            spans: vec![(2, 1)],
        };
        assert!(t.fill("ab", reversed).is_err());
        let past_end = Encoding {
            ids: vec![1],
            spans: vec![(0, 3)],
        };
        assert!(t.fill("ab", past_end).is_err());
        t.fill(
            "ab",
            Encoding {
                ids: vec![1],
                spans: vec![(0, 2)],
            },
        )
        .unwrap();
        let over = Encoding {
            ids: vec![7],
            spans: vec![(1, 4)],
        };
        assert!(t.splice(1, "c", over).is_err());
    }
}
