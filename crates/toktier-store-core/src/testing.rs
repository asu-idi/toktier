//! Deterministic test encoders (feature `testing`).
//!
//! [`MockEncoder`] is a small context-free tokenizer built so that store
//! logic can be exercised hermetically: text splits into maximal
//! same-class character runs (whitespace / alphabetic / other), each run
//! chunks into pieces of at most `piece_chars` characters, and every
//! piece becomes one token whose id is a hash of its bytes. Because
//! chunking restarts at every run start, any run boundary is a sound
//! certified split point: encoding the two sides independently
//! reproduces the full encoding bit-exactly. That gives the tests a
//! real (if toy) certificate structure without any tokenizer dependency.
//!
//! This module is test support, not release API; the character
//! classification uses the Rust standard library and carries no claim
//! of parity with any production tokenizer.

use crate::engine::{
    AppendReport, BoundaryCut, Encoding, EngineError, SessionEncoder, WitnessCategory,
};
use crate::tail::TailState;

/// Deterministic toy encoder; see the module docs.
#[derive(Debug, Clone)]
pub struct MockEncoder {
    /// Maximum piece length in characters (>= 1).
    pub piece_chars: usize,
    /// When false, `last_certified_boundary` always returns `None` and
    /// the witness category is `None` (models an uncertified engine).
    pub certify: bool,
}

impl Default for MockEncoder {
    fn default() -> MockEncoder {
        MockEncoder {
            piece_chars: 3,
            certify: true,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Class {
    Space,
    Letter,
    Other,
}

fn class_of(c: char) -> Class {
    if c.is_whitespace() {
        Class::Space
    } else if c.is_alphabetic() {
        Class::Letter
    } else {
        Class::Other
    }
}

fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

impl MockEncoder {
    fn encode_chars(&self, chars: &[char]) -> Encoding {
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let mut ix = 0usize;
        while ix < chars.len() {
            let cls = class_of(chars[ix]);
            let mut end = ix + 1;
            while end < chars.len() && class_of(chars[end]) == cls {
                end += 1;
            }
            // Chunk the run [ix, end) into pieces.
            let mut p = ix;
            while p < end {
                let q = (p + self.piece_chars.max(1)).min(end);
                let piece: String = chars[p..q].iter().collect();
                ids.push(fnv1a(piece.as_bytes()));
                spans.push((p as u32, q as u32));
                p = q;
            }
            ix = end;
        }
        Encoding { ids, spans }
    }

    /// Whether tail-local char position `b` is a run boundary of `chars`.
    fn is_run_boundary(chars: &[char], b: usize) -> bool {
        b > 0 && b < chars.len() && class_of(chars[b - 1]) != class_of(chars[b])
    }
}

impl SessionEncoder for MockEncoder {
    fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
        let chars: Vec<char> = text.chars().collect();
        Ok(self.encode_chars(&chars))
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        let was_empty = tail.text().is_empty();
        let mut full = String::with_capacity(tail.text_bytes() + delta.len());
        full.push_str(tail.text());
        full.push_str(delta);
        let old_ids = tail.ids().to_vec();
        let old_spans = tail.spans();
        let enc = self.encode(&full)?;
        // Kept prefix: tokens identical in id and span.
        let mut kept = 0usize;
        while kept < old_ids.len()
            && kept < enc.ids.len()
            && old_ids[kept] == enc.ids[kept]
            && old_spans[kept] == enc.spans[kept]
        {
            kept += 1;
        }
        tail.fill(&full, enc)
            .map_err(|e| EngineError(format!("tail fill failed: {e}")))?;
        Ok(AppendReport {
            path: if was_empty {
                "cold_full".to_string()
            } else {
                "mock_full_reencode".to_string()
            },
            kept_tokens: kept,
        })
    }

    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        if !self.certify {
            return Ok(None);
        }
        let n = tail.n_tokens();
        if n < 2 {
            return Ok(None);
        }
        let chars: Vec<char> = tail.text().chars().collect();
        let ceil = ceil_char.min(chars.len() as u64);
        let starts = tail.span_starts();
        let ends = tail.span_ends();
        for j in (0..n - 1).rev() {
            let b = u64::from(starts[j + 1]);
            if b <= floor_char {
                break;
            }
            if b > ceil {
                continue;
            }
            // Only clean cuts: the left token must end at (or before) b.
            if u64::from(ends[j]) > b {
                continue;
            }
            if MockEncoder::is_run_boundary(&chars, b as usize) {
                return Ok(Some(BoundaryCut {
                    cut_tokens: j + 1,
                    cut_char: b,
                }));
            }
        }
        Ok(None)
    }

    fn witness_category(&self) -> WitnessCategory {
        if self.certify {
            WitnessCategory::BpeSyncTransition
        } else {
            WitnessCategory::NoneFullReencode
        }
    }
}

/// Deterministic 32-byte engine fingerprint for store tests: `tag` in the
/// first byte, `tag + 1` in the last, zeroes between. This is the one
/// definition shared by the store-core and store-sqlite test batteries,
/// so the same tag denotes the same fingerprint across both.
pub fn fp(tag: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = tag;
    out[31] = tag.wrapping_add(1);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_boundary_cuts_are_sound() {
        let enc = MockEncoder::default();
        let text = "hello world, this is a test 12345 with runs";
        let full = enc.encode(text).unwrap();
        let chars: Vec<char> = text.chars().collect();
        for b in 1..chars.len() {
            if !MockEncoder::is_run_boundary(&chars, b) {
                continue;
            }
            let head: String = chars[..b].iter().collect();
            let tail: String = chars[b..].iter().collect();
            let mut joined = enc.encode(&head).unwrap().ids;
            joined.extend(enc.encode(&tail).unwrap().ids);
            assert_eq!(joined, full.ids, "cut at {b} is not sound");
        }
    }

    #[test]
    fn append_reports_consistent_kept_prefix() {
        let enc = MockEncoder::default();
        let mut tail = TailState::new();
        let r1 = enc.append(&mut tail, "hello wor").unwrap();
        assert_eq!(r1.path, "cold_full");
        let before = tail.ids().to_vec();
        let r2 = enc.append(&mut tail, "ld and more").unwrap();
        assert!(r2.kept_tokens <= before.len());
        assert_eq!(
            tail.ids()[..r2.kept_tokens],
            before[..r2.kept_tokens],
            "kept prefix must be unchanged"
        );
        assert_eq!(tail.text(), "hello world and more");
        assert_eq!(tail.ids(), enc.encode("hello world and more").unwrap().ids);
    }
}
