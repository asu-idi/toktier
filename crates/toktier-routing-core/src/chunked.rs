//! Certified chunk-parallel encode on the corrected CPU route.
//!
//! Like the rest of this internal supporting crate, the items exported from
//! this module carry no independent API stability promise and may change in
//! any release; the supported entry point is the Python module
//! `toktier.chunked`.
//!
//! A long document is cut at *certified stable boundaries*, the chunks are
//! encoded in parallel on a persistent pool of corrected-Gigatoken workers,
//! and the id rows are concatenated. There is no overlap, no seam matching
//! and no retry: a cut is placed only where the pre-tokenizer of the
//! certified families cannot carry a pre-token across it -- the
//! synchronizing-transition predicate of the session-repair witness
//! (`BpeSyncBoundary::accepts`, eight class pairs over the frozen
//! O/S/L/N/M table minus two carve-outs) -- with two further conditions:
//! no cut falls strictly inside an occurrence of an added-token literal,
//! and for a family whose artifact normalizes its input (NFC) a cut is
//! also a certified cut of the *normalized* text. A cut that cannot be
//! certified is simply not made: fewer chunks, never a different id.
//!
//! The plan semantics are those of the Python module `toktier.chunked`
//! (`_Planner.plan`, `_LiteralGuard`, `_NfcRule`), position for position:
//! every index below is a Unicode scalar-value index, targets are
//! `len * i / k`, the quick scan looks 48 characters below a target and
//! the extended scan up to `scan_budget`, the chunk floor is
//! `max(16, min(1024, len / (4 k)))` unless a `min_chunk` is given, and
//! the normalization-stability check anchors the cut at the nearest
//! stable split point on each side (at most 256 characters away), asks
//! the artifact's own normalizer whether the anchored window normalizes
//! to the concatenation of its halves, and decides the predicate on the
//! two normalized characters that meet at the cut. The character classes
//! are read from the frozen property table; the normalizer is used for
//! equality only, and its per-code-point answers are memoized.
//!
//! Admission is decided from the artifact document exactly as the Python
//! module decides it: no `normalizer` section, or the empty `Sequence`,
//! or `NFC` with every added token matched on the input text
//! (`normalized: false`). Anything else is encoded serially under
//! `R_CHUNK_FAMILY_UNSUPPORTED`; a document without a certified cut is
//! encoded serially under `R_CHUNK_NO_CUT`.
//!
//! Each chunk runs through the engine's own guarded full-encode gate, so
//! the added-token gate, the corrected-Gigatoken closure check and the
//! reference guard apply to a chunk exactly as they apply to a whole
//! document; a chunk that trips a guard is encoded by the reference on
//! its own, which localizes that cost to one chunk. Workers are checked
//! out of a free list per chunk (no prefix interlock between concurrent
//! calls, and no coupling between the worker count and the chunk count);
//! the pool is forked and optionally pre-warmed once, on request or at
//! first use.
//!
//! Source text in this repository is ASCII, so every non-ASCII character
//! named below is written as an escape.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use aho_corasick::{AhoCorasick, MatchKind};
use rayon::prelude::*;
use serde_json::Value;
use toktier_gigatoken_core::Tokenizer as Gigatoken;
use toktier_store_core::EngineError;

use crate::fast_cpu::{FastCpuEngine, FastEncodeSource};
use crate::{BpeSyncBoundary, ReferenceEngine, N_CODEPOINTS};

/// Reason code of a document in which no certified cut was found.
pub const R_CHUNK_NO_CUT: &str = "R_CHUNK_NO_CUT";
/// Reason code of a tokenizer outside the admitted set.
pub const R_CHUNK_FAMILY_UNSUPPORTED: &str = "R_CHUNK_FAMILY_UNSUPPORTED";
/// Reason code of an input whose starting backend is not the corrected
/// CPU engine (chunking is CPU-route only).
pub const R_CHUNK_ROUTE_NOT_FAST_CPU: &str = "R_CHUNK_ROUTE_NOT_FAST_CPU";
/// How far below a target position the cut search looks before it gives
/// that cut up (characters).
pub const CHUNKED_DEFAULT_SCAN_BUDGET: usize = 4096;

/// The first, cheap part of every cut search.
const QUICK_SCAN: usize = 48;
/// Documents shorter than this are never cut, whatever `k` says.
const MIN_TEXT_CHARS: usize = 256;
/// How far the normalization-stability check looks for a stable split
/// point on each side of a candidate cut.
const ANCHOR_SCAN: usize = 256;
/// Probes of the stable-split-point test: a combining mark of canonical
/// class 230 placed before the character and one of class 1 after it, and
/// a leading Hangul consonant jamo and syllable.
const PROBE_BEFORE: char = '\u{0301}';
const PROBE_AFTER: char = '\u{0335}';
const PROBE_JAMO_L: char = '\u{1100}';
const PROBE_SYLLABLE: char = '\u{AC00}';

/// Class index of combining marks in the frozen O/S/L/N/M table.
const CLASS_M: u8 = 4;

const STABLE_UNKNOWN: u8 = 0;
const STABLE_YES: u8 = 1;
const STABLE_NO: u8 = 2;

/// Loader-face normalizer shapes the planner reasons about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizerMode {
    /// No normalizer, or the empty `Sequence`: the input is the text the
    /// pre-tokenizer sees.
    None,
    /// `NFC` with every added token matched on the input text.
    Nfc,
}

impl NormalizerMode {
    pub fn as_str(self) -> &'static str {
        match self {
            NormalizerMode::None => "none",
            NormalizerMode::Nfc => "nfc",
        }
    }
}

/// Where a document was cut, and what the search had to give up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPlan {
    /// Half-open character spans, in order, covering the document.
    pub spans: Vec<(usize, usize)>,
    /// The same spans as UTF-8 byte ranges of the input.
    pub byte_spans: Vec<(usize, usize)>,
    /// Requested chunk count.
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
    fn single(chars: usize, bytes: usize, k: usize) -> Self {
        ChunkPlan {
            spans: vec![(0, chars)],
            byte_spans: vec![(0, bytes)],
            k,
            chars,
            cut_misses: 0,
            literal_rejects: 0,
            norm_rejects: 0,
        }
    }

    /// Character positions of the cuts.
    pub fn cuts(&self) -> Vec<usize> {
        self.spans.iter().skip(1).map(|span| span.0).collect()
    }
}

/// The ids of a chunk-parallel encode plus how they were produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkedIds {
    pub ids: Vec<u32>,
    /// Number of rows that ran (1 for a serial result).
    pub chunks: usize,
    pub k_requested: usize,
    pub cuts: Vec<usize>,
    pub cut_misses: usize,
    pub literal_rejects: usize,
    pub norm_rejects: usize,
    /// The reason code when the document was encoded serially instead;
    /// `None` for a chunked result and for a document below the threshold.
    pub serial_reason: Option<&'static str>,
    /// Whether the document was encoded serially because it was too
    /// short or one chunk was requested.
    pub below_threshold: bool,
    /// Which implementation produced each row, in row order.
    pub sources: Vec<FastEncodeSource>,
    /// Bytes of each row, in row order (the ledger records them).
    pub row_bytes: Vec<usize>,
    pub plan_ns: u64,
    pub encode_ns: u64,
    pub concat_ns: u64,
    /// Longest and shortest per-row worker time inside the parallel
    /// section (checkout to checkin).
    pub worker_ns_max: u64,
    pub worker_ns_min: u64,
    /// Workers forked during this call because the free list was empty.
    pub cold_forks: usize,
}

/// `(mode, refusal)` for the loader face's normalizer section, decided
/// as `toktier.chunked._normalizer_mode` decides it.
pub fn normalizer_mode(document: &Value) -> Result<NormalizerMode, String> {
    let Some(section) = document.get("normalizer") else {
        return Ok(NormalizerMode::None);
    };
    if section.is_null() {
        return Ok(NormalizerMode::None);
    }
    let Some(object) = section.as_object() else {
        return Err("normalizer_not_supported".to_owned());
    };
    let kind = object.get("type").and_then(Value::as_str);
    if kind == Some("Sequence") && !truthy(object.get("normalizers")) {
        return Ok(NormalizerMode::None);
    }
    if kind == Some("NFC") {
        let rows = document.get("added_tokens");
        if let Some(rows) = rows.and_then(Value::as_array) {
            if rows
                .iter()
                .any(|row| row.is_object() && truthy(row.get("normalized")))
            {
                return Err("nfc_with_normalized_added_tokens".to_owned());
            }
        }
        return Ok(NormalizerMode::Nfc);
    }
    Err(format!(
        "normalizer_not_supported:{}",
        kind.unwrap_or("None")
    ))
}

/// Python truthiness of a JSON value.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|value| value != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(items)) => !items.is_empty(),
    }
}

/// The added-token literals of the loader face, in document order, as
/// `toktier.chunked._Planner` reads them.
pub fn face_literals(document: &Value) -> Vec<String> {
    document
        .get("added_tokens")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.as_object())
                .map(|row| match row.get("content") {
                    None | Some(Value::Null) => String::new(),
                    Some(Value::String(text)) => text.clone(),
                    Some(other) => other.to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------- text --

/// Number of Unicode scalar values in a UTF-8 string.
fn count_chars(text: &str) -> usize {
    if text.is_ascii() {
        return text.len();
    }
    text.as_bytes()
        .iter()
        .filter(|&&byte| (byte as i8) >= -0x40)
        .count()
}

#[inline]
fn utf8_len(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead < 0xE0 {
        2
    } else if lead < 0xF0 {
        3
    } else {
        4
    }
}

/// A `(character index, byte offset)` cursor over a UTF-8 string that
/// moves forward in blocks and backward byte by byte.
struct CharCursor<'a> {
    text: &'a str,
    ascii: bool,
    char_index: usize,
    byte_index: usize,
}

impl<'a> CharCursor<'a> {
    fn new(text: &'a str) -> Self {
        CharCursor {
            text,
            ascii: text.is_ascii(),
            char_index: 0,
            byte_index: 0,
        }
    }

    /// Byte offset of character `target`, which may be the text length.
    fn seek(&mut self, target: usize) -> usize {
        if self.ascii {
            self.char_index = target;
            self.byte_index = target;
            return target;
        }
        let bytes = self.text.as_bytes();
        if target >= self.char_index {
            let mut remaining = target - self.char_index;
            let mut index = self.byte_index;
            const BLOCK: usize = 64;
            while remaining >= BLOCK && index + BLOCK <= bytes.len() {
                let starts = bytes[index..index + BLOCK]
                    .iter()
                    .filter(|&&byte| (byte as i8) >= -0x40)
                    .count();
                if starts > remaining {
                    break;
                }
                // The block may end inside a character; the byte-wise
                // loop below resolves the tail.
                remaining -= starts;
                index += BLOCK;
            }
            while (bytes.get(index).copied().unwrap_or(0) & 0xC0) == 0x80 {
                index += 1;
            }
            while remaining > 0 {
                index += utf8_len(bytes[index]);
                remaining -= 1;
            }
            self.char_index = target;
            self.byte_index = index;
        } else {
            let mut remaining = self.char_index - target;
            let mut index = self.byte_index;
            while remaining > 0 {
                index -= 1;
                while (bytes[index] & 0xC0) == 0x80 {
                    index -= 1;
                }
                remaining -= 1;
            }
            self.char_index = target;
            self.byte_index = index;
        }
        self.byte_index
    }
}

/// A decoded window of the text in global character coordinates.
struct Window<'a> {
    text: &'a str,
    /// Global character index of `chars[0]`.
    base: usize,
    chars: Vec<char>,
    /// Global byte offset of each window character, plus one past the end.
    byte_starts: Vec<usize>,
    /// Characters in the whole text.
    total: usize,
}

impl<'a> Window<'a> {
    fn extract(cursor: &mut CharCursor<'a>, total: usize, low: usize, high: usize) -> Self {
        let high = high.min(total);
        let low = low.min(high);
        let start = cursor.seek(low);
        let count = high - low;
        let mut chars = Vec::with_capacity(count);
        let mut byte_starts = Vec::with_capacity(count + 1);
        for (offset, value) in cursor.text[start..].char_indices().take(count) {
            chars.push(value);
            byte_starts.push(start + offset);
        }
        let end = byte_starts
            .last()
            .map(|&last| last + cursor.text[last..].chars().next().map_or(0, char::len_utf8))
            .unwrap_or(start);
        byte_starts.push(end);
        cursor.char_index = high;
        cursor.byte_index = end;
        Window {
            text: cursor.text,
            base: low,
            chars,
            byte_starts,
            total,
        }
    }

    #[inline]
    fn char_at(&self, index: usize) -> char {
        self.chars[index - self.base]
    }

    #[inline]
    fn byte_of(&self, index: usize) -> usize {
        self.byte_starts[index - self.base]
    }

    fn slice(&self, low: usize, high: usize) -> &'a str {
        &self.text[self.byte_of(low)..self.byte_of(high)]
    }

    /// Global character index of a global byte offset inside the window.
    fn char_of_byte(&self, byte: usize) -> usize {
        self.base + self.byte_starts.partition_point(|&start| start < byte)
    }
}

// --------------------------------------------------------------- guard --

/// Cut positions strictly inside an added-token literal are forbidden.
///
/// Conservative on purpose: every occurrence of every literal counts,
/// overlapping ones included. That is a superset of what any
/// leftmost-longest extraction would actually consume, so a cut the guard
/// allows never splits a literal the reference would have matched, and a
/// window the guard reports untouched holds no literal the reference
/// would extract.
struct LiteralGuard {
    matcher: Option<AhoCorasick>,
    /// Character length of each pattern.
    lengths: Vec<usize>,
    count: usize,
    /// Longest literal, in characters.
    lmax: usize,
}

impl LiteralGuard {
    fn new(literals: &[String]) -> Result<Self, EngineError> {
        let mut unique = literals
            .iter()
            .filter(|item| !item.is_empty())
            .cloned()
            .collect::<Vec<_>>();
        unique.sort();
        unique.dedup();
        let lengths = unique
            .iter()
            .map(|item| item.chars().count())
            .collect::<Vec<_>>();
        let lmax = lengths.iter().copied().max().unwrap_or(0);
        let matcher = if unique.is_empty() {
            None
        } else {
            Some(
                AhoCorasick::builder()
                    .match_kind(MatchKind::Standard)
                    .build(&unique)
                    .map_err(|error| {
                        EngineError(format!("failed to build the literal guard: {error}"))
                    })?,
            )
        };
        Ok(LiteralGuard {
            matcher,
            lengths,
            count: unique.len(),
            lmax,
        })
    }

    /// Occurrences `(first, last)` in global character coordinates whose
    /// start lies in `[start, end)`.
    fn occurrences(&self, window: &Window<'_>, start: usize, end: usize) -> Vec<(usize, usize)> {
        let Some(matcher) = &self.matcher else {
            return Vec::new();
        };
        if start >= end {
            return Vec::new();
        }
        let haystack = window.slice(start, end);
        let offset = window.byte_of(start);
        matcher
            .find_overlapping_iter(haystack)
            .map(|found| {
                let first = window.char_of_byte(offset + found.start());
                (first, first + self.lengths[found.pattern().as_usize()])
            })
            .collect()
    }

    /// For each position in `[low, high]`, whether it falls inside some
    /// occurrence.
    fn forbidden(&self, window: &Window<'_>, low: usize, high: usize) -> Vec<bool> {
        let mut out = vec![false; high + 1 - low];
        if self.matcher.is_none() || self.lmax < 2 {
            return out;
        }
        let start = low.saturating_sub(self.lmax - 1);
        let end = window.total.min(high + self.lmax - 1);
        for (first, last) in self.occurrences(window, start, end) {
            let from = (first + 1).max(low);
            let to = last.min(high + 1);
            for position in from..to {
                out[position - low] = true;
            }
        }
        out
    }

    /// Whether some occurrence intersects the span `[low, high)`.
    fn touches(&self, window: &Window<'_>, low: usize, high: usize) -> bool {
        if self.matcher.is_none() {
            return false;
        }
        let start = low.saturating_sub(self.lmax.saturating_sub(1));
        let end = window.total.min(high + self.lmax.saturating_sub(1));
        self.occurrences(window, start, end)
            .into_iter()
            .any(|(first, last)| first < high && last > low)
    }
}

// ------------------------------------------------------------ nfc rule --

/// The normalization-stability condition of an NFC family.
struct NfcRule {
    reference: Arc<ReferenceEngine>,
    stable: Box<[AtomicU8]>,
}

impl NfcRule {
    fn new(reference: Arc<ReferenceEngine>) -> Self {
        let stable = (0..N_CODEPOINTS)
            .map(|_| AtomicU8::new(STABLE_UNKNOWN))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        NfcRule { reference, stable }
    }

    fn normalize(&self, text: &str) -> Result<String, EngineError> {
        self.reference
            .normalize(text)
            .map_err(|error| EngineError(error.to_string()))
    }

    fn probe(
        &self,
        prefix: Option<char>,
        value: char,
        suffix: Option<char>,
    ) -> Result<bool, EngineError> {
        let mut text = String::with_capacity(12);
        if let Some(prefix) = prefix {
            text.push(prefix);
        }
        text.push(value);
        if let Some(suffix) = suffix {
            text.push(suffix);
        }
        Ok(self.normalize(&text)? == text)
    }

    /// Whether `value` is a stable split point (module documentation).
    fn stable(&self, boundary: &BpeSyncBoundary, value: char) -> Result<bool, EngineError> {
        if value.is_ascii() {
            return Ok(true);
        }
        let slot = &self.stable[value as usize];
        match slot.load(Ordering::Relaxed) {
            STABLE_YES => return Ok(true),
            STABLE_NO => return Ok(false),
            _ => {}
        }
        let verdict = boundary.class(value) != CLASS_M
            && self.probe(None, value, None)?
            && self.probe(Some(PROBE_BEFORE), value, None)?
            && self.probe(None, value, Some(PROBE_AFTER))?
            && self.probe(Some(PROBE_JAMO_L), value, None)?
            && self.probe(Some(PROBE_SYLLABLE), value, None)?;
        slot.store(
            if verdict { STABLE_YES } else { STABLE_NO },
            Ordering::Relaxed,
        );
        Ok(verdict)
    }

    /// Whether the cut at `position` is also a certified cut of the
    /// normalized text.
    fn accepts(
        &self,
        boundary: &BpeSyncBoundary,
        guard: &LiteralGuard,
        window: &Window<'_>,
        position: usize,
    ) -> Result<bool, EngineError> {
        let total = window.total;
        let around_end = (position + 2).min(total);
        let mut all_stable = true;
        for index in position - 1..around_end {
            if !self.stable(boundary, window.char_at(index))? {
                all_stable = false;
                break;
            }
        }
        if all_stable {
            // Stable on both sides and beyond: the cut is its own
            // normalized cut, and the classes are the input classes.
            return Ok(true);
        }
        let floor_index = position.saturating_sub(ANCHOR_SCAN);
        let mut left = position - 1;
        let mut anchored = false;
        loop {
            if self.stable(boundary, window.char_at(left))? {
                anchored = true;
                break;
            }
            if left == floor_index {
                break;
            }
            left -= 1;
        }
        if !anchored {
            if floor_index > 0 {
                return Ok(false);
            }
            left = 0; // the start of the text anchors as well
        }
        let mut right = position + 1;
        let ceiling = total.min(position + 1 + ANCHOR_SCAN);
        while right < ceiling && !self.stable(boundary, window.char_at(right))? {
            right += 1;
        }
        if right == ceiling && ceiling < total {
            return Ok(false);
        }
        if guard.touches(window, left, right) {
            return Ok(false);
        }
        let before = self.normalize(window.slice(left, position))?;
        let after = self.normalize(window.slice(position, right))?;
        let (Some(last), Some(first)) = (before.chars().next_back(), after.chars().next()) else {
            return Ok(false);
        };
        let whole = self.normalize(window.slice(left, right))?;
        if whole.len() != before.len() + after.len()
            || !whole.starts_with(&before)
            || !whole.ends_with(&after)
        {
            return Ok(false);
        }
        Ok(boundary.accepts(last, first))
    }
}

// ------------------------------------------------------------- planner --

/// Per-engine admission, guard, normalizer rule and worker pool.
pub struct ChunkEncoder {
    boundary: BpeSyncBoundary,
    guard: LiteralGuard,
    mode: Option<NormalizerMode>,
    refusal: Option<String>,
    rule: Option<NfcRule>,
    free: Mutex<Vec<Gigatoken>>,
    forked: AtomicUsize,
}

impl std::fmt::Debug for ChunkEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChunkEncoder")
            .field("normalizer", &self.mode.map(NormalizerMode::as_str))
            .field("refusal", &self.refusal)
            .field("literals", &self.guard.count)
            .field("workers", &self.forked.load(Ordering::Relaxed))
            .finish()
    }
}

impl ChunkEncoder {
    /// Build the planner from the exact artifact document the engine was
    /// constructed over.
    pub fn new(
        tokenizer_json: &[u8],
        boundary: BpeSyncBoundary,
        reference: Arc<ReferenceEngine>,
    ) -> Result<Self, EngineError> {
        let document: Value = serde_json::from_slice(tokenizer_json)
            .map_err(|error| EngineError(format!("tokenizer document is not JSON: {error}")))?;
        let literals = face_literals(&document);
        let guard = LiteralGuard::new(&literals)?;
        let (mode, refusal) = match normalizer_mode(&document) {
            Ok(mode) => (Some(mode), None),
            Err(refusal) => (None, Some(refusal)),
        };
        let rule = (mode == Some(NormalizerMode::Nfc)).then(|| NfcRule::new(reference));
        Ok(ChunkEncoder {
            boundary,
            guard,
            mode,
            refusal,
            rule,
            free: Mutex::new(Vec::new()),
            forked: AtomicUsize::new(0),
        })
    }

    pub fn admitted(&self) -> bool {
        self.refusal.is_none()
    }

    pub fn refusal(&self) -> Option<&str> {
        self.refusal.as_deref()
    }

    pub fn normalizer(&self) -> Option<NormalizerMode> {
        self.mode
    }

    /// Unique non-empty added-token literals the guard knows.
    pub fn literal_count(&self) -> usize {
        self.guard.count
    }

    /// Workers forked so far (free or checked out).
    pub fn worker_count(&self) -> usize {
        self.forked.load(Ordering::Relaxed)
    }

    /// Whether a cut between characters `position - 1` and `position` is a
    /// certified boundary of the text as written.
    #[inline]
    fn boundary_ok(&self, window: &Window<'_>, position: usize) -> bool {
        self.boundary
            .accepts(window.char_at(position - 1), window.char_at(position))
    }

    /// The highest certified cut in `[low, high]`.
    fn scan(
        &self,
        window: &Window<'_>,
        low: usize,
        high: usize,
        rejects: &mut [usize; 2],
    ) -> Result<Option<usize>, EngineError> {
        if high < low {
            return Ok(None);
        }
        let forbidden = self.guard.forbidden(window, low, high);
        let mut position = high;
        loop {
            if self.boundary_ok(window, position) {
                if forbidden[position - low] {
                    rejects[0] += 1;
                } else if let Some(rule) = &self.rule {
                    if rule.accepts(&self.boundary, &self.guard, window, position)? {
                        return Ok(Some(position));
                    }
                    rejects[1] += 1;
                } else {
                    return Ok(Some(position));
                }
            }
            if position == low {
                return Ok(None);
            }
            position -= 1;
        }
    }

    /// The cut plan for `text`, as `toktier.chunked.plan` computes it
    /// (admission is the caller's concern; see [`ChunkEncoder::admitted`]).
    pub fn plan(
        &self,
        text: &str,
        k: usize,
        min_chunk: Option<usize>,
        scan_budget: usize,
    ) -> Result<ChunkPlan, EngineError> {
        if scan_budget == 0 {
            return Err(EngineError(
                "scan_budget must be a positive integer".to_owned(),
            ));
        }
        let total = count_chars(text);
        self.plan_counted(text, total, k, min_chunk, scan_budget)
    }

    fn plan_counted(
        &self,
        text: &str,
        total: usize,
        k: usize,
        min_chunk: Option<usize>,
        scan_budget: usize,
    ) -> Result<ChunkPlan, EngineError> {
        if k <= 1 || total < MIN_TEXT_CHARS {
            return Ok(ChunkPlan::single(total, text.len(), k));
        }
        let floor = min_chunk.unwrap_or_else(|| (total / (4 * k)).clamp(16, 1024));
        let quick = QUICK_SCAN.min(scan_budget);
        // The window around a scan range must hold everything the guard
        // and the normalization rule read: literal occurrences up to
        // `lmax` characters away and anchors up to `ANCHOR_SCAN` away.
        let margin = ANCHOR_SCAN + self.guard.lmax + 2;
        let mut cursor = CharCursor::new(text);
        let mut cuts: Vec<(usize, usize)> = Vec::new();
        let mut misses = 0usize;
        let mut rejects = [0usize; 2];
        let mut previous = 0usize;
        for index in 1..k {
            let target = total * index / k;
            if target <= previous {
                misses += 1;
                continue;
            }
            let mut low = (previous + 1).max(target.saturating_sub(quick));
            let mut window = Window::extract(
                &mut cursor,
                total,
                low.saturating_sub(margin),
                target + margin + 1,
            );
            let mut found = self.scan(&window, low, target, &mut rejects)?;
            if found.is_none() && scan_budget > quick && low > previous + 1 {
                let high = low - 1;
                low = (previous + 1).max(target.saturating_sub(scan_budget));
                window = Window::extract(
                    &mut cursor,
                    total,
                    low.saturating_sub(margin),
                    high + margin + 1,
                );
                found = self.scan(&window, low, high, &mut rejects)?;
            }
            // A cut that would leave either neighbouring chunk below the
            // floor -- the one before it, or the tail after it -- is not
            // made; the text stays with the preceding chunk.
            match found {
                Some(found) if found - previous >= floor && total - found >= floor => {
                    cuts.push((found, window.byte_of(found)));
                    previous = found;
                }
                _ => misses += 1,
            }
        }
        let mut bounds = Vec::with_capacity(cuts.len() + 2);
        bounds.push((0, 0));
        bounds.extend(cuts);
        bounds.push((total, text.len()));
        let spans = bounds
            .windows(2)
            .map(|pair| (pair[0].0, pair[1].0))
            .collect();
        let byte_spans = bounds
            .windows(2)
            .map(|pair| (pair[0].1, pair[1].1))
            .collect();
        Ok(ChunkPlan {
            spans,
            byte_spans,
            k,
            chars: total,
            cut_misses: misses,
            literal_rejects: rejects[0],
            norm_rejects: rejects[1],
        })
    }

    fn checkout(&self, engine: &FastCpuEngine) -> Result<(Gigatoken, bool), EngineError> {
        let popped = self
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop();
        match popped {
            Some(worker) => Ok((worker, false)),
            None => {
                let worker = engine.fork_worker()?;
                self.forked.fetch_add(1, Ordering::Relaxed);
                Ok((worker, true))
            }
        }
    }

    fn checkin(&self, worker: Gigatoken) {
        self.free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(worker);
    }

    /// Ensure at least `workers` forked workers exist, and run each new one
    /// over `text` so its caches are warm before the first request.
    /// Returns the number of workers in the pool.
    pub fn warm(
        &self,
        engine: &FastCpuEngine,
        workers: usize,
        text: Option<&str>,
    ) -> Result<usize, EngineError> {
        let existing = self.forked.load(Ordering::Relaxed);
        if workers > existing {
            let fresh = (existing..workers)
                .into_par_iter()
                .map(|_| {
                    let mut worker = engine.fork_worker()?;
                    if let Some(text) = text {
                        let mut out = Vec::new();
                        worker.encode_with_added_tokens_flat(text.as_bytes(), &mut out);
                    }
                    Ok(worker)
                })
                .collect::<Result<Vec<_>, EngineError>>()?;
            self.forked.fetch_add(fresh.len(), Ordering::Relaxed);
            self.free
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend(fresh);
        }
        Ok(self.forked.load(Ordering::Relaxed))
    }
}

impl FastCpuEngine {
    /// Encode `text` in certified chunks; the ids equal the serial
    /// core-stream encode of the same route.
    ///
    /// `k` is the requested chunk count; `min_chunk` the smallest chunk the
    /// plan may produce (`None` selects `min(1024, len / (4 k))`);
    /// `scan_budget` how far below each target the cut search looks. A
    /// tokenizer outside the admitted set and a document without a
    /// certified cut are encoded serially through the same guarded gate;
    /// the result says so in `serial_reason`.
    pub fn chunked_encode_ids(
        &self,
        text: &str,
        k: usize,
        min_chunk: Option<usize>,
        scan_budget: usize,
    ) -> Result<ChunkedIds, EngineError> {
        if scan_budget == 0 {
            return Err(EngineError(
                "scan_budget must be a positive integer".to_owned(),
            ));
        }
        let started = Instant::now();
        let encoder = self.chunk_encoder()?;
        let total = count_chars(text);
        let serial =
            |this: &Self, reason: Option<&'static str>, below: bool, plan: Option<&ChunkPlan>| {
                let plan_ns = started.elapsed().as_nanos() as u64;
                let encode_started = Instant::now();
                let (ids, source) = this.encode_ids_with_source(text)?;
                let encode_ns = encode_started.elapsed().as_nanos() as u64;
                Ok(ChunkedIds {
                    ids,
                    chunks: 1,
                    k_requested: k,
                    cuts: Vec::new(),
                    cut_misses: plan.map_or(0, |plan| plan.cut_misses),
                    literal_rejects: plan.map_or(0, |plan| plan.literal_rejects),
                    norm_rejects: plan.map_or(0, |plan| plan.norm_rejects),
                    serial_reason: reason,
                    below_threshold: below,
                    sources: vec![source],
                    row_bytes: vec![text.len()],
                    plan_ns,
                    encode_ns,
                    concat_ns: 0,
                    worker_ns_max: encode_ns,
                    worker_ns_min: encode_ns,
                    cold_forks: 0,
                })
            };
        if !encoder.admitted() {
            return serial(self, Some(R_CHUNK_FAMILY_UNSUPPORTED), false, None);
        }
        if k <= 1 || total < MIN_TEXT_CHARS {
            return serial(self, None, true, None);
        }
        let plan = encoder.plan_counted(text, total, k, min_chunk, scan_budget)?;
        if plan.spans.len() == 1 {
            return serial(self, Some(R_CHUNK_NO_CUT), false, Some(&plan));
        }
        let plan_ns = started.elapsed().as_nanos() as u64;

        let encode_started = Instant::now();
        let rows = plan
            .byte_spans
            .par_iter()
            .map(|&(low, high)| {
                let row_started = Instant::now();
                let (mut worker, cold) = encoder.checkout(self)?;
                let outcome = self.full_ids_with_source(&text[low..high], &mut worker);
                encoder.checkin(worker);
                let (ids, source) = outcome?;
                Ok((ids, source, row_started.elapsed().as_nanos() as u64, cold))
            })
            .collect::<Result<Vec<_>, EngineError>>()?;
        let encode_ns = encode_started.elapsed().as_nanos() as u64;

        let concat_started = Instant::now();
        let capacity = rows.iter().map(|row| row.0.len()).sum();
        let mut ids = Vec::with_capacity(capacity);
        let mut sources = Vec::with_capacity(rows.len());
        let mut worker_ns_max = 0u64;
        let mut worker_ns_min = u64::MAX;
        let mut cold_forks = 0usize;
        for (row, source, elapsed, cold) in &rows {
            ids.extend_from_slice(row);
            sources.push(*source);
            worker_ns_max = worker_ns_max.max(*elapsed);
            worker_ns_min = worker_ns_min.min(*elapsed);
            cold_forks += usize::from(*cold);
        }
        let concat_ns = concat_started.elapsed().as_nanos() as u64;
        Ok(ChunkedIds {
            ids,
            chunks: rows.len(),
            k_requested: k,
            cuts: plan.cuts(),
            cut_misses: plan.cut_misses,
            literal_rejects: plan.literal_rejects,
            norm_rejects: plan.norm_rejects,
            serial_reason: None,
            below_threshold: false,
            sources,
            row_bytes: plan
                .byte_spans
                .iter()
                .map(|&(low, high)| high - low)
                .collect(),
            plan_ns,
            encode_ns,
            concat_ns,
            worker_ns_max,
            worker_ns_min,
            cold_forks,
        })
    }

    /// The cut plan [`Self::chunked_encode_ids`] would use, without
    /// encoding. Errors for a tokenizer outside the admitted set.
    pub fn chunked_plan(
        &self,
        text: &str,
        k: usize,
        min_chunk: Option<usize>,
        scan_budget: usize,
    ) -> Result<ChunkPlan, EngineError> {
        let encoder = self.chunk_encoder()?;
        if let Some(refusal) = encoder.refusal() {
            return Err(EngineError(format!(
                "{} has no certified chunk-cut rule: {refusal}",
                self.spec().family
            )));
        }
        encoder.plan(text, k, min_chunk, scan_budget)
    }

    /// Fork (and optionally pre-warm) the chunk worker pool up to
    /// `workers` workers; returns the pool size.
    pub fn chunked_warm(&self, workers: usize, text: Option<&str>) -> Result<usize, EngineError> {
        self.chunk_encoder()?.warm(self, workers, text)
    }

    /// The planner's admission facts.
    pub fn chunk_encoder(&self) -> Result<&ChunkEncoder, EngineError> {
        self.chunked_cell()
            .get_or_init(|| {
                ChunkEncoder::new(
                    self.tokenizer_json_bytes(),
                    self.boundary().clone(),
                    self.reference_arc(),
                )
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// The once-cell type the engine holds for its chunk encoder.
pub(crate) type ChunkEncoderCell = OnceLock<Result<ChunkEncoder, EngineError>>;

/// The worker count the pool is sized to when the caller names none: the
/// Rayon thread count, which bounds the rows in flight at once.
pub fn default_chunk_workers() -> usize {
    rayon::current_num_threads().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use tokenizers::models::bpe::{Vocab, BPE};
    use tokenizers::normalizers::NFC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, Tokenizer};

    fn tokenizer_json(nfc: bool, added: &[&str]) -> Vec<u8> {
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
        if nfc {
            tokenizer.with_normalizer(Some(NFC));
        }
        if !added.is_empty() {
            tokenizer.add_special_tokens(
                &added
                    .iter()
                    .map(|item| AddedToken::from(*item, true).normalized(false))
                    .collect::<Vec<_>>(),
            );
        }
        tokenizer.to_string(false).unwrap().into_bytes()
    }

    fn pclass() -> Vec<u8> {
        // O=0, S=1, L=2, N=3, M=4; everything unnamed is O.
        let mut table = vec![0u8; N_CODEPOINTS];
        for value in b'a'..=b'z' {
            table[value as usize] = 2;
        }
        for value in b'A'..=b'Z' {
            table[value as usize] = 2;
        }
        for value in b'0'..=b'9' {
            table[value as usize] = 3;
        }
        for value in [b' ', b'\n', b'\r', b'\t'] {
            table[value as usize] = 1;
        }
        // A few non-ASCII letters and marks used by the tests.
        for value in [
            '\u{00e9}', '\u{0958}', '\u{0915}', '\u{4e2d}', '\u{1100}', '\u{1161}',
        ] {
            table[value as usize] = 2;
        }
        for value in ['\u{0301}', '\u{0335}', '\u{093c}'] {
            table[value as usize] = CLASS_M;
        }
        table
    }

    fn engine(nfc: bool, added: &[&str]) -> Arc<FastCpuEngine> {
        let json = tokenizer_json(nfc, added);
        let reference = Arc::new(ReferenceEngine::from_bytes(&json).unwrap());
        let digest = Sha256::digest(&json)
            .iter()
            .map(|value| format!("{value:02x}"))
            .collect::<String>();
        let spec = crate::FastRepairSpec::new("tiny_bytes".to_owned(), digest, 4, 2, nfc);
        Arc::new(FastCpuEngine::from_reference(&json, reference, spec, pclass()).unwrap())
    }

    fn prose(chars: usize, seed: u64) -> String {
        let words = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dogs",
            "1990s",
            "42",
            "don't",
            "caf\u{e9}",
            "\u{4e2d}\u{4e2d}",
            "x.y",
            "a,b",
        ];
        let mut state = seed;
        let mut out = String::new();
        while out.chars().count() < chars {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let word = words[(state >> 33) as usize % words.len()];
            out.push_str(word);
            out.push(if (state >> 20).is_multiple_of(7) {
                '\n'
            } else {
                ' '
            });
        }
        out
    }

    #[test]
    fn json_admission_mirrors_the_python_planner() {
        let none: Value = serde_json::json!({"normalizer": null});
        assert_eq!(normalizer_mode(&none), Ok(NormalizerMode::None));
        let missing: Value = serde_json::json!({});
        assert_eq!(normalizer_mode(&missing), Ok(NormalizerMode::None));
        let empty_sequence: Value =
            serde_json::json!({"normalizer": {"type": "Sequence", "normalizers": []}});
        assert_eq!(normalizer_mode(&empty_sequence), Ok(NormalizerMode::None));
        let sequence: Value = serde_json::json!({
            "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFC"}]}
        });
        assert_eq!(
            normalizer_mode(&sequence),
            Err("normalizer_not_supported:Sequence".to_owned())
        );
        let nfc: Value = serde_json::json!({
            "normalizer": {"type": "NFC"},
            "added_tokens": [{"content": "<a>", "normalized": false}]
        });
        assert_eq!(normalizer_mode(&nfc), Ok(NormalizerMode::Nfc));
        let nfc_normalized: Value = serde_json::json!({
            "normalizer": {"type": "NFC"},
            "added_tokens": [{"content": "<a>", "normalized": true}]
        });
        assert_eq!(
            normalizer_mode(&nfc_normalized),
            Err("nfc_with_normalized_added_tokens".to_owned())
        );
        let nfkc: Value = serde_json::json!({"normalizer": {"type": "NFKC"}});
        assert_eq!(
            normalizer_mode(&nfkc),
            Err("normalizer_not_supported:NFKC".to_owned())
        );
        let list: Value = serde_json::json!({"normalizer": [1]});
        assert_eq!(
            normalizer_mode(&list),
            Err("normalizer_not_supported".to_owned())
        );
        assert_eq!(face_literals(&nfc), vec!["<a>".to_owned()],);
    }

    #[test]
    fn char_cursor_seeks_both_ways_on_multibyte_text() {
        let text = "ab\u{e9}\u{4e2d}\u{1f642}cd";
        let mut cursor = CharCursor::new(text);
        // a b e-acute(2) CJK(3) emoji(4) c d: byte offsets 0 1 2 4 7 11 12.
        assert_eq!(cursor.seek(2), 2);
        assert_eq!(cursor.seek(4), 7);
        assert_eq!(cursor.seek(7), text.len());
        assert_eq!(cursor.seek(3), 4);
        assert_eq!(cursor.seek(5), 11);
        assert_eq!(cursor.seek(0), 0);
        let long = "\u{4e2d}".repeat(1000) + &"x".repeat(100);
        let mut cursor = CharCursor::new(&long);
        assert_eq!(cursor.seek(1000), 3000);
        assert_eq!(cursor.seek(1050), 3050);
        assert_eq!(cursor.seek(999), 2997);
        assert_eq!(count_chars(&long), 1100);
        assert_eq!(count_chars(text), 7);
    }

    #[test]
    fn literal_guard_is_conservative_over_overlaps() {
        let guard =
            LiteralGuard::new(&["abc".to_owned(), "bcd".to_owned(), "abcd".to_owned()]).unwrap();
        let text = "xxabcdxx";
        let total = count_chars(text);
        let mut cursor = CharCursor::new(text);
        let window = Window::extract(&mut cursor, total, 0, total);
        let forbidden = guard.forbidden(&window, 0, total);
        let positions = (0..=total).filter(|&p| forbidden[p]).collect::<Vec<_>>();
        // Inside "abcd" at 2..6: 3, 4, 5; "bcd" at 3..6 adds nothing new.
        assert_eq!(positions, vec![3, 4, 5]);
        assert!(guard.touches(&window, 5, 6));
        assert!(guard.touches(&window, 0, 3));
        assert!(!guard.touches(&window, 6, 8));
        assert!(!guard.touches(&window, 0, 2));
        let single = LiteralGuard::new(&["x".to_owned()]).unwrap();
        assert!(single.forbidden(&window, 0, total).iter().all(|flag| !flag));
        assert!(single.touches(&window, 0, 1));
        assert!(!single.touches(&window, 2, 6));
    }

    #[test]
    fn plan_covers_the_text_and_respects_the_floor_and_k() {
        let fast = engine(false, &["<|endoftext|>", "<|im_start|>"]);
        let encoder = fast.chunk_encoder().unwrap();
        assert!(encoder.admitted());
        let text = prose(30_000, 7);
        let total = count_chars(&text);
        for k in [2usize, 5, 16, 64] {
            let plan = encoder
                .plan(&text, k, None, CHUNKED_DEFAULT_SCAN_BUDGET)
                .unwrap();
            assert_eq!(plan.spans[0].0, 0);
            assert_eq!(plan.spans.last().unwrap().1, total);
            assert!(plan.spans.iter().all(|(a, b)| a < b));
            assert!(plan.spans.windows(2).all(|pair| pair[0].1 == pair[1].0));
            assert!(plan.spans.len() <= k);
            assert!(plan.spans.len() >= 2, "k={k}");
            let floor = (total / (4 * k)).clamp(16, 1024);
            assert!(plan.spans.iter().all(|(a, b)| b - a >= floor));
            assert_eq!(plan.byte_spans.len(), plan.spans.len());
            let mut cursor = CharCursor::new(&text);
            for (cut, byte) in plan.cuts().into_iter().zip(plan.byte_spans.iter().skip(1)) {
                assert_eq!(cursor.seek(cut), byte.0);
                let window = Window::extract(&mut cursor, total, cut - 1, cut + 1);
                assert!(encoder.boundary_ok(&window, cut));
            }
        }
        // A tight floor gives up most cuts.
        let tight = encoder
            .plan(&text, 8, Some(10_000), CHUNKED_DEFAULT_SCAN_BUDGET)
            .unwrap();
        assert!(tight.spans.len() <= 3 && tight.cut_misses >= 5);
    }

    #[test]
    fn min_chunk_and_scan_budget_bound_the_plan() {
        let fast = engine(false, &[]);
        let encoder = fast.chunk_encoder().unwrap();
        let text = format!("{}{}{}", "a".repeat(110), " ", "b".repeat(189));
        let one = encoder.plan(&text, 2, None, 1).unwrap();
        assert_eq!(one.spans.len(), 1);
        assert_eq!(one.cut_misses, 1);
        let cut = encoder.plan(&text, 2, None, 40).unwrap();
        assert_eq!(cut.cuts(), vec![110]);
        let text = "ab 12 ".repeat(50);
        let single = encoder
            .plan(&text, 3, Some(150), CHUNKED_DEFAULT_SCAN_BUDGET)
            .unwrap();
        assert_eq!(single.spans.len(), 1);
        assert_eq!(single.cut_misses, 2);
        let two = encoder
            .plan(&text, 3, Some(100), CHUNKED_DEFAULT_SCAN_BUDGET)
            .unwrap();
        assert_eq!(two.spans.len(), 2);
        assert!(two.spans.iter().all(|(a, b)| b - a >= 100));
        // Below the text floor nothing is cut.
        let short = encoder
            .plan(&"a b".repeat(80), 4, None, CHUNKED_DEFAULT_SCAN_BUDGET)
            .unwrap();
        assert_eq!(short.spans.len(), 1);
    }

    #[test]
    fn stable_split_points_follow_the_probes() {
        let fast = engine(true, &[]);
        let encoder = fast.chunk_encoder().unwrap();
        assert_eq!(encoder.normalizer(), Some(NormalizerMode::Nfc));
        let rule = encoder.rule.as_ref().unwrap();
        let stable = |value: char| rule.stable(&encoder.boundary, value).unwrap();
        assert!(stable('a'));
        assert!(stable('\u{e9}'));
        assert!(stable('\u{4e2d}'));
        assert!(!stable('\u{301}'), "a mark is never a split point");
        assert!(!stable('\u{958}'), "a decomposing letter changes under NFC");
        assert!(
            !stable('\u{1161}'),
            "a vowel jamo composes after a leading jamo"
        );
        assert!(
            !stable('\u{11a8}'),
            "a trailing jamo composes after a syllable"
        );
        // Memoized answers agree with fresh ones.
        assert!(!stable('\u{958}'));
        assert!(stable('\u{4e2d}'));
    }

    #[test]
    fn a_cut_the_normalizer_moves_is_not_made() {
        let text = format!("{}\u{958}/k{}", "a".repeat(127), " ".repeat(126));
        let nfc = engine(true, &[]);
        let plan = nfc
            .chunk_encoder()
            .unwrap()
            .plan(&text, 2, None, CHUNKED_DEFAULT_SCAN_BUDGET)
            .unwrap();
        assert_eq!(plan.spans.len(), 1);
        assert!(plan.norm_rejects >= 1);
        let encoded = nfc
            .chunked_encode_ids(&text, 2, None, CHUNKED_DEFAULT_SCAN_BUDGET)
            .unwrap();
        assert_eq!(encoded.serial_reason, Some(R_CHUNK_NO_CUT));
        assert_eq!(encoded.ids, nfc.encode_ids(&text).unwrap());
        let plain = engine(false, &[]);
        let plan = plain
            .chunk_encoder()
            .unwrap()
            .plan(&text, 2, None, CHUNKED_DEFAULT_SCAN_BUDGET)
            .unwrap();
        assert_eq!(plan.cuts(), vec![128]);
        let healthy = format!(
            "{}\u{958}{}/k{}",
            "a".repeat(120),
            "b".repeat(7),
            " ".repeat(126)
        );
        for fast in [&nfc, &plain] {
            let encoded = fast
                .chunked_encode_ids(&healthy, 2, None, CHUNKED_DEFAULT_SCAN_BUDGET)
                .unwrap();
            assert_eq!(encoded.cuts, vec![128]);
            assert_eq!(encoded.ids, fast.encode_ids(&healthy).unwrap());
        }
    }

    #[test]
    fn chunked_ids_equal_the_serial_encode_and_localize_guards() {
        let fast = engine(true, &["<|endoftext|>", "<|im_start|>"]);
        let mut text = prose(20_000, 3);
        text.insert_str(9_000, "<|im_start|>user\n");
        text.insert_str(7_000, " <|endoftext|> ");
        // A decomposed sequence the NFC invariance guard rejects.
        text.insert_str(3_000, " e\u{301} ");
        let expected = fast.encode_ids(&text).unwrap();
        assert_eq!(expected, fast.reference().encode_ids(&text, false).unwrap());
        for k in [2usize, 4, 9, 32] {
            let encoded = fast
                .chunked_encode_ids(&text, k, None, CHUNKED_DEFAULT_SCAN_BUDGET)
                .unwrap();
            assert_eq!(encoded.ids, expected, "k={k}");
            assert!(encoded.chunks >= 2);
            assert_eq!(encoded.serial_reason, None);
            assert_eq!(encoded.sources.len(), encoded.chunks);
            assert_eq!(encoded.row_bytes.iter().sum::<usize>(), text.len());
            // The literal-bearing chunks (both literals sit in the first
            // half) went to the reference on their own, the decomposed
            // sequence tripped the engine guard in its own chunk once the
            // chunks are small enough to separate it from a literal, and
            // the others stayed on corrected Gigatoken.
            for source in [
                FastEncodeSource::ReferenceAddedToken,
                FastEncodeSource::Gigatoken,
            ] {
                assert!(encoded.sources.contains(&source), "k={k} {source:?}");
            }
            if k >= 4 {
                assert!(
                    encoded
                        .sources
                        .contains(&FastEncodeSource::ReferenceEngineGuard),
                    "k={k}"
                );
            }
        }
        // Below the threshold and k = 1 are serial without a reason.
        let short = fast
            .chunked_encode_ids("tiny text", 4, None, CHUNKED_DEFAULT_SCAN_BUDGET)
            .unwrap();
        assert!(short.below_threshold && short.serial_reason.is_none() && short.chunks == 1);
        let one = fast
            .chunked_encode_ids(&text, 1, None, CHUNKED_DEFAULT_SCAN_BUDGET)
            .unwrap();
        assert!(one.below_threshold && one.ids == expected);
        // A refused family is serial under the family reason.
        let json = tokenizer_json(false, &[]);
        let mut document: Value = serde_json::from_slice(&json).unwrap();
        document["normalizer"] = serde_json::json!({"type": "NFKC"});
        let refused = ChunkEncoder::new(
            serde_json::to_vec(&document).unwrap().as_slice(),
            BpeSyncBoundary::new(pclass()).unwrap(),
            fast.reference_arc(),
        )
        .unwrap();
        assert!(!refused.admitted());
        assert_eq!(refused.refusal(), Some("normalizer_not_supported:NFKC"));
    }

    #[test]
    fn concurrent_calls_share_the_pool_without_interlock() {
        let fast = engine(false, &[]);
        let text = prose(40_000, 11);
        let expected = fast.encode_ids(&text).unwrap();
        assert_eq!(fast.chunked_warm(3, Some(&text[..2_000])).unwrap(), 3);
        std::thread::scope(|scope| {
            for k in [3usize, 5, 8, 13] {
                let fast = Arc::clone(&fast);
                let text = &text;
                let expected = &expected;
                scope.spawn(move || {
                    for _ in 0..4 {
                        let encoded = fast
                            .chunked_encode_ids(text, k, None, CHUNKED_DEFAULT_SCAN_BUDGET)
                            .unwrap();
                        assert_eq!(&encoded.ids, expected);
                    }
                });
            }
        });
        let encoder = fast.chunk_encoder().unwrap();
        assert!(encoder.worker_count() >= 3);
        assert_eq!(
            encoder.free.lock().unwrap().len(),
            encoder.worker_count(),
            "every worker is back on the free list"
        );
    }
}
