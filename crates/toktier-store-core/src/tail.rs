//! Mutable tail state of a session: the text past the last certified
//! seal point together with its standalone encoding.
//!
//! Layout follows the pre-release prototype's session state: UTF-8 text plus
//! structure-of-arrays token storage (`ids`, `span_start`, `span_end`,
//! spans in character units). All mutation goes through validating
//! methods, so an ill-behaved encoder cannot leave the state with
//! mismatched lengths; illegal states are not representable through the
//! public API.

use crate::engine::Encoding;
use crate::error::StoreError;

/// Tail text and its encoding. Spans are character-unit `(start, end)`
/// pairs relative to the tail origin.
#[derive(Debug, Clone, Default)]
pub struct TailState {
    text: String,
    text_chars: u32,
    ids: Vec<u32>,
    span_start: Vec<u32>,
    span_end: Vec<u32>,
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
        self.ids = enc.ids;
        self.span_start.clear();
        self.span_end.clear();
        self.span_start.reserve(enc.spans.len());
        self.span_end.reserve(enc.spans.len());
        for (a, b) in enc.spans {
            self.span_start.push(a);
            self.span_end.push(b);
        }
        Ok(())
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
        self.ids.truncate(cut_idx);
        self.span_start.truncate(cut_idx);
        self.span_end.truncate(cut_idx);
        self.ids.extend_from_slice(&new.ids);
        self.span_start.reserve(new.spans.len());
        self.span_end.reserve(new.spans.len());
        for (a, b) in new.spans {
            self.span_start.push(a);
            self.span_end.push(b);
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
        &self.ids
    }

    pub fn n_tokens(&self) -> usize {
        self.ids.len()
    }

    pub fn span_starts(&self) -> &[u32] {
        &self.span_start
    }

    pub fn span_ends(&self) -> &[u32] {
        &self.span_end
    }

    /// Spans materialized as pairs (convenience for encoder adapters).
    pub fn spans(&self) -> Vec<(u32, u32)> {
        self.span_start
            .iter()
            .zip(self.span_end.iter())
            .map(|(&a, &b)| (a, b))
            .collect()
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
