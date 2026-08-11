//! Native request routing and frozen reference-engine primitives.
//!
//! This is an internal supporting crate of TokTier, versioned with the
//! workspace and carrying no independent API stability promise; use the
//! `toktier` package for the supported Rust surface.
//!
//! The public policy and diagnostic objects remain in Python. This crate owns
//! the per-input hot decisions which otherwise allocate a complete UTF-8 copy:
//! byte-threshold selection, the necessary-condition added-token prefilter,
//! and the frozen BPE synchronizing-transition predicate used to seal session
//! prefixes.  It also owns the pinned Hugging Face `tokenizers` reference so
//! store and fallback work can run without a Python callback.

#![forbid(unsafe_code)]

use memchr::{memchr, memchr2, memchr3};
use std::fmt;

mod entry_store;
mod fast_cpu;
mod gpu;
mod reference;
mod runtime;

pub use entry_store::{
    EntryStoreOpenError, EntryStoreStats, NativeEntryStore, AUTO_MIN_BYTES,
    DEFAULT_CACHE_BUDGET_BYTES, MAX_AUTO_ENTRIES,
};
pub use fast_cpu::{
    FastCpuEngine, FastCpuEngineError, FastEncodeOutcome, FastEncodeSource, FastRepairSpec,
    FastRepairStats, FastSeedPayload,
};
pub use gpu::{NativePrebuiltGpu, NativePrebuiltGpuConfig};
pub use reference::{ReferenceEngine, ReferenceEngineError};
pub use runtime::{
    NativeGpuEngine, NativeGpuOpener, NativeGpuSource, NativeRouter, NativeRuntimeError,
    RayonSeedOverlap, RoutedIds, RuntimeEvent, RuntimeStats, BACKEND_FAST_CPU, BACKEND_GPU,
    BACKEND_REFERENCE, R_EXEC_FAULT, R_INPUT_ADDED_TOKEN, R_INPUT_BELOW_GPU_THRESHOLD,
    R_INPUT_GUARD_ROUTED, R_INPUT_POSTPROCESS_ROUTED,
};

/// Number of Unicode scalar-value slots in the frozen property table.
pub const N_CODEPOINTS: usize = 0x11_0000;

/// One input's immutable starting route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteDecision {
    /// UTF-8 byte count, or `None` when the Python string cannot be expressed
    /// as valid UTF-8 (the reference backend will surface the original error).
    pub input_bytes: Option<u64>,
    /// First eligible index in the frozen fallback chain.
    pub start_index: usize,
    /// Whether a GPU-headed plan skipped its head because of the crossover.
    pub below_gpu_threshold: bool,
    /// Whether an added-token literal *may* occur. False is a proven miss;
    /// true asks the exact frontend to decide.
    pub literal_candidate: bool,
}

/// Construction failure for immutable native routing data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteConfigError(String);

impl RouteConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for RouteConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RouteConfigError {}

/// A one- or two-byte prefix used by the exact frontend's cheap gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LiteralPrefix {
    pub first: u8,
    /// `None` represents a one-byte literal; its first byte alone is a hit.
    pub second: Option<u8>,
}

/// How much the native layer can prove before invoking the exact frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralMode {
    /// No added-token frontend exists on this route.
    Disabled,
    /// The frontend has no sound cheap rejection test; always ask it.
    AlwaysCandidate,
    /// Use the exact frontend's one-/two-byte necessary conditions.
    Prefixes,
}

#[derive(Debug, Clone)]
enum LiteralPrefilter {
    Disabled,
    AlwaysCandidate,
    Prefixes(Box<PrefixFilter>),
}

#[derive(Debug, Clone)]
struct PrefixFilter {
    first_bytes: Vec<u8>,
    first_mask: [bool; 256],
    single_mask: [bool; 256],
    pair_bits: Box<[u64; 1024]>,
}

impl PrefixFilter {
    fn new(prefixes: &[LiteralPrefix]) -> Self {
        let mut first_mask = [false; 256];
        let mut single_mask = [false; 256];
        let mut pair_bits = Box::new([0_u64; 1024]);
        for prefix in prefixes {
            first_mask[usize::from(prefix.first)] = true;
            if let Some(second) = prefix.second {
                let key = (usize::from(prefix.first) << 8) | usize::from(second);
                pair_bits[key >> 6] |= 1_u64 << (key & 63);
            } else {
                single_mask[usize::from(prefix.first)] = true;
            }
        }
        let first_bytes = first_mask
            .iter()
            .enumerate()
            .filter_map(|(value, &present)| present.then_some(value as u8))
            .collect();
        Self {
            first_bytes,
            first_mask,
            single_mask,
            pair_bits,
        }
    }

    #[inline]
    fn pair_present(&self, first: u8, second: u8) -> bool {
        let key = (usize::from(first) << 8) | usize::from(second);
        self.pair_bits[key >> 6] & (1_u64 << (key & 63)) != 0
    }

    fn next_candidate(&self, bytes: &[u8]) -> Option<usize> {
        match self.first_bytes.as_slice() {
            [] => None,
            [a] => memchr(*a, bytes),
            [a, b] => memchr2(*a, *b, bytes),
            [a, b, c] => memchr3(*a, *b, *c, bytes),
            _ => bytes
                .iter()
                .position(|value| self.first_mask[usize::from(*value)]),
        }
    }

    fn may_match(&self, bytes: &[u8]) -> bool {
        let mut base = 0_usize;
        while base < bytes.len() {
            let Some(relative) = self.next_candidate(&bytes[base..]) else {
                return false;
            };
            let index = base + relative;
            let first = bytes[index];
            if self.single_mask[usize::from(first)] {
                return true;
            }
            if let Some(&second) = bytes.get(index + 1) {
                if self.pair_present(first, second) {
                    return true;
                }
            }
            base = index + 1;
        }
        false
    }
}

/// Immutable per-input selector for one already-validated route plan.
#[derive(Debug, Clone)]
pub struct RouteSelector {
    thresholds: Box<[u64]>,
    reference_index: usize,
    gpu_head: bool,
    literals: LiteralPrefilter,
}

impl RouteSelector {
    pub fn new(
        thresholds: Vec<u64>,
        reference_index: usize,
        gpu_head: bool,
        literal_mode: LiteralMode,
        literal_prefixes: Vec<LiteralPrefix>,
    ) -> Result<Self, RouteConfigError> {
        if thresholds.is_empty() {
            return Err(RouteConfigError::new(
                "fallback thresholds must not be empty",
            ));
        }
        if reference_index != thresholds.len() - 1 {
            return Err(RouteConfigError::new(
                "reference_index must name the final fallback entry",
            ));
        }
        if thresholds[reference_index] != 0 {
            return Err(RouteConfigError::new(
                "the reference fallback threshold must be zero",
            ));
        }
        if literal_mode != LiteralMode::Prefixes && !literal_prefixes.is_empty() {
            return Err(RouteConfigError::new(
                "literal prefixes require prefix-filter mode",
            ));
        }
        let literals = match literal_mode {
            LiteralMode::Disabled => LiteralPrefilter::Disabled,
            LiteralMode::AlwaysCandidate => LiteralPrefilter::AlwaysCandidate,
            LiteralMode::Prefixes => {
                LiteralPrefilter::Prefixes(Box::new(PrefixFilter::new(&literal_prefixes)))
            }
        };
        Ok(Self {
            thresholds: thresholds.into_boxed_slice(),
            reference_index,
            gpu_head,
            literals,
        })
    }

    pub fn reference_index(&self) -> usize {
        self.reference_index
    }

    /// Choose a starting backend and cheaply reject impossible literal hits.
    /// `None` models a Python string that cannot be converted to UTF-8.
    pub fn decide(&self, input: Option<&[u8]>) -> RouteDecision {
        let Some(bytes) = input else {
            return RouteDecision {
                input_bytes: None,
                start_index: self.reference_index,
                below_gpu_threshold: false,
                literal_candidate: false,
            };
        };
        let size = bytes.len() as u64;
        let start_index = self
            .thresholds
            .iter()
            .position(|&threshold| size >= threshold)
            .unwrap_or(self.reference_index);
        let below_gpu_threshold = self.gpu_head && start_index > 0;
        let literal_candidate = if start_index == self.reference_index {
            false
        } else {
            match &self.literals {
                LiteralPrefilter::Disabled => false,
                LiteralPrefilter::AlwaysCandidate => true,
                LiteralPrefilter::Prefixes(filter) => filter.may_match(bytes),
            }
        };
        RouteDecision {
            input_bytes: Some(size),
            start_index,
            below_gpu_threshold,
            literal_candidate,
        }
    }
}

/// A certified token/character split returned by the BPE predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BpeBoundaryCut {
    pub cut_tokens: usize,
    pub cut_char: u64,
}

/// Frozen O/S/L/N/M class table and synchronizing-transition predicate.
#[derive(Debug, Clone)]
pub struct BpeSyncBoundary {
    classes: Box<[u8]>,
}

impl BpeSyncBoundary {
    pub fn new(classes: Vec<u8>) -> Result<Self, RouteConfigError> {
        if classes.len() != N_CODEPOINTS {
            return Err(RouteConfigError::new(format!(
                "BPE property table has {} rows, expected {N_CODEPOINTS}",
                classes.len()
            )));
        }
        if let Some((index, value)) = classes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value > 4)
        {
            return Err(RouteConfigError::new(format!(
                "BPE property table row {index} has unknown class {value}"
            )));
        }
        Ok(Self {
            classes: classes.into_boxed_slice(),
        })
    }

    #[inline]
    fn class(&self, value: char) -> u8 {
        self.classes[value as usize]
    }

    #[inline]
    fn accepts(&self, previous: char, current: char) -> bool {
        // O=0, S=1, L=2, N=3, M=4. These eight pairs and the two
        // carve-outs are the frozen global sync profile used by the eleven
        // certified corrected-Gigatoken artifacts.
        let pair = (self.class(previous), self.class(current));
        let in_set = matches!(
            pair,
            (2, 3) | (2, 0) | (2, 1) | (3, 2) | (3, 0) | (3, 1) | (0, 3) | (0, 1)
        );
        in_set
            && !(pair == (0, 1) && matches!(current, '\r' | '\n'))
            && !(pair == (2, 0) && current == '\'')
    }

    /// Return the last clean token boundary in `(floor_char, ceil_char]`.
    ///
    /// Span starts must be nondecreasing, as they are for the certified
    /// pre-postprocessor core streams. If a callback violates that premise,
    /// this method returns `None` rather than manufacturing a certificate.
    pub fn last_boundary(
        &self,
        text: &str,
        text_chars: u32,
        span_starts: &[u32],
        span_ends: &[u32],
        floor_char: u64,
        ceil_char: u64,
    ) -> Option<BpeBoundaryCut> {
        if span_starts.len() != span_ends.len() || span_starts.len() < 2 {
            return None;
        }
        if span_starts.windows(2).any(|pair| pair[0] > pair[1]) {
            return None;
        }
        let ceiling = ceil_char.min(u64::from(text_chars));
        let mut cursor = ReverseCharCursor::new(text, u64::from(text_chars));
        let mut right_index = span_starts.len() - 1;
        while right_index > 0 {
            let boundary_start = span_starts[right_index];
            let mut group_start = right_index;
            while group_start > 0 && span_starts[group_start - 1] == boundary_start {
                group_start -= 1;
            }
            let boundary = u64::from(boundary_start);
            if boundary <= floor_char {
                break;
            }
            if boundary <= ceiling
                && boundary > 0
                && boundary < u64::from(text_chars)
                && group_start > 0
                // A byte-fallback character can occupy several tokens sharing
                // one character span. `group_start` is the first token in the
                // group, so this check considers only the boundary before it.
                && u64::from(span_ends[group_start - 1]) <= boundary
            {
                let current = cursor.char_at(boundary)?;
                let previous = cursor.char_at(boundary - 1)?;
                if self.accepts(previous, current) {
                    return Some(BpeBoundaryCut {
                        cut_tokens: group_start,
                        cut_char: boundary,
                    });
                }
            }
            if group_start == 0 {
                break;
            }
            right_index = group_start - 1;
        }
        None
    }
}

/// Backward-materializing span view over fetched windows.
///
/// Windows are fetched newest-first in `window` -sized, stride-aligned
/// regions and retained, so the backward boundary scan touches only the
/// suffix it actually visits. Each fetched window is checked for the
/// nondecreasing-starts premise (within the window and across the seam
/// to the previously fetched right neighbour); a violation poisons the
/// view and the search returns no certificate.
struct WindowedSpans<F, E>
where
    F: FnMut(usize, usize) -> Result<(Vec<u32>, Vec<u32>), E>,
{
    fetch: F,
    window: usize,
    /// Fetched regions in decreasing `lo` order: `(lo, starts, ends)`.
    chunks: Vec<(usize, Vec<u32>, Vec<u32>)>,
    /// First covered token index (tokens `[covered_lo, n)` are fetched).
    covered_lo: usize,
    monotone: bool,
}

impl<F, E> WindowedSpans<F, E>
where
    F: FnMut(usize, usize) -> Result<(Vec<u32>, Vec<u32>), E>,
{
    fn new(fetch: F, window: usize, n_tokens: usize) -> Self {
        WindowedSpans {
            fetch,
            window: window.max(1),
            chunks: Vec::new(),
            covered_lo: n_tokens,
            monotone: true,
        }
    }

    /// Fetch stride-aligned windows until token `index` is covered.
    /// Returns false when a fetched window violated the premise.
    fn ensure(&mut self, index: usize) -> Result<bool, E> {
        while self.covered_lo > index {
            let hi = self.covered_lo;
            let lo = (hi - 1) - ((hi - 1) % self.window);
            let (starts, ends) = (self.fetch)(lo, hi)?;
            if starts.windows(2).any(|pair| pair[0] > pair[1]) {
                self.monotone = false;
            }
            if let (Some(&last), Some((_, right, _))) = (starts.last(), self.chunks.last()) {
                if right.first().is_some_and(|&first| last > first) {
                    self.monotone = false;
                }
            }
            self.chunks.push((lo, starts, ends));
            self.covered_lo = lo;
        }
        Ok(self.monotone)
    }

    /// `(start, end)` of a covered token index.
    fn bounds(&self, index: usize) -> (u32, u32) {
        for (lo, starts, ends) in self.chunks.iter().rev() {
            if index >= *lo && index - lo < starts.len() {
                return (starts[index - lo], ends[index - lo]);
            }
        }
        unreachable!("span window access outside the ensured coverage")
    }
}

impl BpeSyncBoundary {
    /// Windowed variant of [`Self::last_boundary`] for tails whose spans
    /// exist only as sparse checkpoints: span regions are fetched on
    /// demand (suffix first) instead of being read from whole-row
    /// arrays, so a search that certifies a cut near the tail end never
    /// materializes millions of spans.
    ///
    /// Premise: the fetched values come from the exact known-ID span
    /// converters, whose starts are nondecreasing by construction. The
    /// whole-array precheck of [`Self::last_boundary`] is therefore
    /// applied per fetched window (including the seam between windows);
    /// a violation inside the visited suffix returns `None` exactly like
    /// the whole-row search, while windows the backward scan never needs
    /// are not fetched at all.
    #[allow(clippy::too_many_arguments)]
    pub fn last_boundary_windowed<F, E>(
        &self,
        text: &str,
        text_chars: u32,
        n_tokens: usize,
        floor_char: u64,
        ceil_char: u64,
        window_tokens: usize,
        fetch: F,
    ) -> Result<Option<BpeBoundaryCut>, E>
    where
        F: FnMut(usize, usize) -> Result<(Vec<u32>, Vec<u32>), E>,
    {
        if n_tokens < 2 {
            return Ok(None);
        }
        let mut spans = WindowedSpans::new(fetch, window_tokens, n_tokens);
        let ceiling = ceil_char.min(u64::from(text_chars));
        let mut cursor = ReverseCharCursor::new(text, u64::from(text_chars));
        let mut right_index = n_tokens - 1;
        while right_index > 0 {
            if !spans.ensure(right_index)? {
                return Ok(None);
            }
            let boundary_start = spans.bounds(right_index).0;
            let mut group_start = right_index;
            while group_start > 0 {
                if !spans.ensure(group_start - 1)? {
                    return Ok(None);
                }
                if spans.bounds(group_start - 1).0 != boundary_start {
                    break;
                }
                group_start -= 1;
            }
            let boundary = u64::from(boundary_start);
            if boundary <= floor_char {
                break;
            }
            if boundary <= ceiling
                && boundary > 0
                && boundary < u64::from(text_chars)
                && group_start > 0
                // A byte-fallback character can occupy several tokens sharing
                // one character span. `group_start` is the first token in the
                // group, so this check considers only the boundary before it.
                && u64::from(spans.bounds(group_start - 1).1) <= boundary
            {
                let (Some(current), Some(previous)) =
                    (cursor.char_at(boundary), cursor.char_at(boundary - 1))
                else {
                    return Ok(None);
                };
                if self.accepts(previous, current) {
                    return Ok(Some(BpeBoundaryCut {
                        cut_tokens: group_start,
                        cut_char: boundary,
                    }));
                }
            }
            if group_start == 0 {
                break;
            }
            right_index = group_start - 1;
        }
        Ok(None)
    }
}

/// Monotone reverse character lookup without materializing `Vec<char>`.
struct ReverseCharCursor<'a> {
    iter: std::iter::Rev<std::str::Chars<'a>>,
    next_index: u64,
    cached: Option<(u64, char)>,
}

impl<'a> ReverseCharCursor<'a> {
    fn new(text: &'a str, text_chars: u64) -> Self {
        Self {
            iter: text.chars().rev(),
            next_index: text_chars,
            cached: None,
        }
    }

    fn char_at(&mut self, target: u64) -> Option<char> {
        if let Some((index, value)) = self.cached {
            if index == target {
                return Some(value);
            }
            if target > index {
                return None;
            }
        }
        while self.next_index > target {
            let value = self.iter.next()?;
            self.next_index -= 1;
            self.cached = Some((self.next_index, value));
        }
        self.cached
            .and_then(|(index, value)| (index == target).then_some(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selector(mode: LiteralMode, prefixes: Vec<LiteralPrefix>) -> RouteSelector {
        RouteSelector::new(vec![65_536, 0, 0], 2, true, mode, prefixes).unwrap()
    }

    #[test]
    fn threshold_is_byte_exact_and_invalid_utf8_uses_reference() {
        let route = selector(LiteralMode::Disabled, vec![]);
        assert_eq!(route.decide(Some(&vec![b'x'; 65_535])).start_index, 1);
        assert_eq!(route.decide(Some(&vec![b'x'; 65_536])).start_index, 0);
        let invalid = route.decide(None);
        assert_eq!(invalid.input_bytes, None);
        assert_eq!(invalid.start_index, 2);
        assert!(!invalid.below_gpu_threshold);
    }

    #[test]
    fn literal_prefilter_matches_one_and_two_byte_prefixes() {
        let route = selector(
            LiteralMode::Prefixes,
            vec![
                LiteralPrefix {
                    first: b'<',
                    second: Some(b'|'),
                },
                LiteralPrefix {
                    first: 0xff,
                    second: None,
                },
            ],
        );
        assert!(!route.decide(Some(b"plain text")).literal_candidate);
        assert!(!route.decide(Some(b"angle < bracket")).literal_candidate);
        assert!(route.decide(Some(b"has <| prefix")).literal_candidate);
        assert!(route.decide(Some(&[1, 0xff, 2])).literal_candidate);
    }

    fn classes() -> Vec<u8> {
        let mut table = vec![4_u8; N_CODEPOINTS];
        for value in b'a'..=b'z' {
            table[usize::from(value)] = 2;
        }
        for value in b'0'..=b'9' {
            table[usize::from(value)] = 3;
        }
        table[usize::from(b' ')] = 1;
        table[usize::from(b'\n')] = 1;
        table[usize::from(b'.')] = 0;
        table[usize::from(b'\'')] = 0;
        table
    }

    #[test]
    fn bpe_boundary_uses_the_last_clean_sync_transition() {
        let cert = BpeSyncBoundary::new(classes()).unwrap();
        let text = "alpha 123";
        let starts: Vec<u32> = (0..text.len() as u32).collect();
        let ends: Vec<u32> = (1..=text.len() as u32).collect();
        assert_eq!(
            cert.last_boundary(text, text.len() as u32, &starts, &ends, 0, u64::MAX),
            Some(BpeBoundaryCut {
                cut_tokens: 5,
                cut_char: 5,
            })
        );
    }

    #[test]
    fn bpe_boundary_keeps_crlf_apostrophe_and_byte_groups_uncut() {
        let cert = BpeSyncBoundary::new(classes()).unwrap();
        assert_eq!(cert.last_boundary(".\n", 2, &[0, 1], &[1, 2], 0, 2), None);
        assert_eq!(cert.last_boundary("a'", 2, &[0, 1], &[1, 2], 0, 2), None);

        // The candidate between the two tokens shares the same character;
        // span_end[0] > span_start[1], so it is not a clean character cut.
        assert_eq!(cert.last_boundary("a1", 2, &[0, 1], &[2, 2], 0, 2), None);
    }

    /// Windowed and whole-row searches must agree on converter-shaped
    /// (monotone) spans for every window size, floor, and ceiling.
    fn assert_windowed_matches(
        cert: &BpeSyncBoundary,
        text: &str,
        starts: &[u32],
        ends: &[u32],
        floor: u64,
        ceil: u64,
    ) {
        let text_chars = text.chars().count() as u32;
        let full = cert.last_boundary(text, text_chars, starts, ends, floor, ceil);
        for window in [1usize, 2, 3, 5, 7, 64] {
            let windowed = cert
                .last_boundary_windowed(
                    text,
                    text_chars,
                    starts.len(),
                    floor,
                    ceil,
                    window,
                    |lo, hi| {
                        Ok::<_, std::convert::Infallible>((
                            starts[lo..hi].to_vec(),
                            ends[lo..hi].to_vec(),
                        ))
                    },
                )
                .unwrap();
            assert_eq!(
                windowed, full,
                "window={window} floor={floor} ceil={ceil} over {text:?}"
            );
        }
    }

    #[test]
    fn windowed_boundary_matches_the_whole_row_search() {
        let cert = BpeSyncBoundary::new(classes()).unwrap();
        // Char-per-token rows, multi-char tokens, and byte-fallback
        // groups (several tokens sharing one span start) placed so that
        // groups straddle the small test window sizes.
        let text = "alpha 123 beta.think 42 gamma\ndelta 9";
        let n = text.chars().count() as u32;
        let starts: Vec<u32> = (0..n).collect();
        let ends: Vec<u32> = (1..=n).collect();
        for floor in [0u64, 3, 10, 30, 36, 64] {
            for ceil in [0u64, 1, 5, 17, 29, 36, 37, u64::MAX] {
                assert_windowed_matches(&cert, text, &starts, &ends, floor, ceil);
            }
        }
        // Grouped rows: tokens 4..=6 share span start 4 (a fallback
        // group), tokens 10..=11 share span start 10.
        let g_starts = [0u32, 1, 2, 3, 4, 4, 4, 7, 8, 9, 10, 10, 12, 13];
        let g_ends = [1u32, 2, 3, 4, 5, 5, 5, 8, 9, 10, 12, 12, 13, 14];
        let g_text = "word 12 ab cd12"; // 15 chars; spans cover 14
        for floor in [0u64, 2, 4, 9] {
            for ceil in [u64::MAX, 13, 11, 10, 5] {
                assert_windowed_matches(&cert, g_text, &g_starts, &g_ends, floor, ceil);
            }
        }
        // Deterministic randomized rows.
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        let pool: Vec<char> = "abc 123.\n'".chars().collect();
        for _ in 0..300 {
            let chars = 2 + (next() % 40) as usize;
            let text: String = (0..chars)
                .map(|_| pool[(next() % pool.len() as u64) as usize])
                .collect();
            let total = text.chars().count() as u32;
            // Random monotone token layout with occasional groups.
            let mut starts = Vec::new();
            let mut ends = Vec::new();
            let mut position = 0u32;
            while position < total {
                let width = 1 + (next() % 3) as u32;
                let end = (position + width).min(total);
                if next() % 4 == 0 {
                    // A fallback-style group: several tokens, one span.
                    for _ in 0..=(next() % 3) {
                        starts.push(position);
                        ends.push(end);
                    }
                } else {
                    starts.push(position);
                    ends.push(end);
                }
                position = end;
            }
            let floor = next() % (u64::from(total) + 1);
            let ceil = next() % (u64::from(total) + 2);
            assert_windowed_matches(&cert, &text, &starts, &ends, floor, ceil);
        }
    }

    #[test]
    fn windowed_boundary_rejects_a_visited_monotonicity_violation() {
        let cert = BpeSyncBoundary::new(classes()).unwrap();
        let text = "alpha 123";
        // Starts decrease at index 3: the premise is violated.
        let starts = [0u32, 1, 5, 2, 6, 7];
        let ends = [1u32, 5, 6, 6, 7, 9];
        assert_eq!(
            cert.last_boundary(text, 9, &starts, &ends, 0, u64::MAX),
            None
        );
        for window in [1usize, 2, 3, 8] {
            let windowed = cert
                .last_boundary_windowed(text, 9, starts.len(), 0, u64::MAX, window, |lo, hi| {
                    Ok::<_, std::convert::Infallible>((
                        starts[lo..hi].to_vec(),
                        ends[lo..hi].to_vec(),
                    ))
                })
                .unwrap();
            assert_eq!(windowed, None, "window={window}");
        }
    }

    #[test]
    fn windowed_boundary_propagates_fetch_errors() {
        let cert = BpeSyncBoundary::new(classes()).unwrap();
        let result = cert.last_boundary_windowed("alpha 123", 9, 9, 0, u64::MAX, 4, |_lo, _hi| {
            Err::<(Vec<u32>, Vec<u32>), &str>("window fetch failed")
        });
        assert_eq!(result, Err("window fetch failed"));
    }

    #[test]
    fn bpe_boundary_can_cut_before_a_multi_token_character_group() {
        let cert = BpeSyncBoundary::new(classes()).unwrap();
        assert_eq!(
            cert.last_boundary("a1", 2, &[0, 1, 1, 1], &[1, 2, 2, 2], 0, 2,),
            Some(BpeBoundaryCut {
                cut_tokens: 1,
                cut_char: 1,
            })
        );
    }
}
