//! The normalization condition of a session seal.
//!
//! A session store seals its tail at a token boundary that the frozen
//! synchronizing-transition predicate (`BpeSyncBoundary::accepts`) accepts,
//! and later encodes the text after that boundary on its own. The predicate
//! is a statement about the characters the pre-tokenizer reads. On a face
//! whose artifact normalizes its input those are the characters of the
//! *normalized* text, which are not the raw characters of the tail when the
//! tail is not stable under the normalizer: a raw letter that NFC turns into
//! a letter and a combining mark reads as a letter, while the pre-tokenizer
//! sees the mark.
//!
//! Every seal candidate on such a face therefore also passes the condition
//! the chunk planner applies to each of its cuts (`chunked.rs`, `NfcRule`):
//! the cut is anchored at the nearest stable split point on each side, the
//! anchored window must normalize to the concatenation of its two
//! normalized halves, and the predicate is decided on the two normalized
//! characters that meet at the cut. Where that condition cannot be decided
//! in its bounded neighbourhood (no stable split point within
//! `ANCHOR_SCAN` characters, or an added-token literal inside the anchored
//! window), the candidate is admitted only when the whole tail text is
//! stable under the normalizer, so that the characters read are the ones
//! the reference reads. A candidate that is not admitted is skipped and the
//! search continues below it.
//!
//! On a face without an effective normalizer, and on ASCII text under an
//! NFC normalizer, the condition holds for every candidate and costs
//! nothing. A face with an effective normalizer other than NFC seals
//! nothing (no packaged face has one).

use std::cell::Cell;
use std::sync::{Arc, OnceLock};

use toktier_store_core::EngineError;

use crate::chunked::{LiteralGuard, NfcRule, NfcVerdict, Window, ANCHOR_SCAN};
use crate::{BpeSyncBoundary, ReferenceEngine};

enum Mode {
    /// No effective normalizer: the raw characters are the ones read.
    Identity,
    /// NFC: the shared normalized-cut condition, built on first use.
    Nfc {
        reference: Arc<ReferenceEngine>,
        literals: Vec<String>,
        rule: OnceLock<Result<(NfcRule, LiteralGuard), String>>,
    },
    /// Any other normalizer, or a normalizer this lane cannot inspect: no
    /// candidate is admitted.
    Closed,
}

/// The seal-side admission rule of one tokenizer face.
pub struct SealCutRule {
    mode: Mode,
}

impl std::fmt::Debug for SealCutRule {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SealCutRule")
            .field("mode", &self.mode_name())
            .finish()
    }
}

impl SealCutRule {
    /// The rule for the face `reference` executes.
    pub fn for_reference(reference: Arc<ReferenceEngine>) -> Self {
        let mode = if !reference.has_effective_normalizer() {
            Mode::Identity
        } else if reference.normalizer_is_nfc() {
            let literals = reference
                .added_tokens()
                .into_iter()
                .map(|(_, content, _)| content)
                .collect();
            Mode::Nfc {
                reference,
                literals,
                rule: OnceLock::new(),
            }
        } else {
            Mode::Closed
        };
        SealCutRule { mode }
    }

    /// A rule that admits no candidate, for a lane that cannot see the
    /// face normalizer.
    pub fn closed() -> Self {
        SealCutRule { mode: Mode::Closed }
    }

    /// `identity`, `nfc` or `closed`.
    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            Mode::Identity => "identity",
            Mode::Nfc { .. } => "nfc",
            Mode::Closed => "closed",
        }
    }

    /// Whether no candidate can be admitted on any text.
    pub(crate) fn is_closed(&self) -> bool {
        matches!(self.mode, Mode::Closed)
    }

    /// Whether every candidate is admitted on `text` without inspection.
    pub(crate) fn admits_all(&self, text: &str, text_chars: u64) -> bool {
        match self.mode {
            Mode::Identity => true,
            Mode::Nfc { .. } => text.len() as u64 == text_chars,
            Mode::Closed => false,
        }
    }

    fn nfc(&self) -> Result<Option<&(NfcRule, LiteralGuard)>, EngineError> {
        let Mode::Nfc {
            reference,
            literals,
            rule,
        } = &self.mode
        else {
            return Ok(None);
        };
        rule.get_or_init(|| {
            LiteralGuard::new(literals)
                .map(|guard| (NfcRule::new(Arc::clone(reference)), guard))
                .map_err(|error| error.0)
        })
        .as_ref()
        .map(Some)
        .map_err(|message| EngineError(message.clone()))
    }

    /// Whether the candidate cut before character `cut_char` (which starts
    /// at byte `cut_byte` of `text`) is admitted. The caller has already
    /// applied the class predicate to the raw characters; `tail_stable`
    /// memoizes the whole-text stability answer across the candidates of
    /// one search.
    pub(crate) fn admits(
        &self,
        boundary: &BpeSyncBoundary,
        text: &str,
        text_chars: u64,
        cut_char: u64,
        cut_byte: usize,
        tail_stable: &Cell<Option<bool>>,
    ) -> Result<bool, EngineError> {
        if self.admits_all(text, text_chars) {
            return Ok(true);
        }
        let Some((rule, guard)) = self.nfc()? else {
            return Ok(false);
        };
        let total = text_chars as usize;
        let position = cut_char as usize;
        if position == 0 || position >= total {
            return Ok(false);
        }
        let reach = ANCHOR_SCAN + guard.lmax + 2;
        let window = Window::around(
            text,
            total,
            position,
            cut_byte,
            position.saturating_sub(reach),
            position + reach,
        );
        match rule.verdict(boundary, guard, &window, position)? {
            NfcVerdict::Accept => Ok(true),
            NfcVerdict::Reject => Ok(false),
            NfcVerdict::Undecided => {
                if let Some(known) = tail_stable.get() {
                    return Ok(known);
                }
                let Mode::Nfc { reference, .. } = &self.mode else {
                    return Ok(false);
                };
                let stable = reference
                    .normalize(text)
                    .map_err(|error| EngineError(error.to_string()))?
                    == text;
                tail_stable.set(Some(stable));
                Ok(stable)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BpeBoundaryCut, N_CODEPOINTS};
    use tokenizers::models::bpe::{Vocab, BPE};
    use tokenizers::normalizers::{Lowercase, NFC};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::Tokenizer;

    #[derive(Clone, Copy)]
    enum Face {
        Plain,
        Nfc,
        Lowercase,
    }

    fn reference(face: Face) -> Arc<ReferenceEngine> {
        let mut vocab = Vocab::default();
        for (index, value) in ByteLevel::alphabet().into_iter().enumerate() {
            vocab.insert(value.to_string(), index as u32);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        let byte_level = ByteLevel::new(false, true, true);
        tokenizer.with_pre_tokenizer(Some(byte_level));
        tokenizer.with_decoder(Some(byte_level));
        match face {
            Face::Plain => {}
            Face::Nfc => {
                tokenizer.with_normalizer(Some(NFC));
            }
            Face::Lowercase => {
                tokenizer.with_normalizer(Some(Lowercase));
            }
        }
        let json = tokenizer.to_string(false).unwrap();
        Arc::new(ReferenceEngine::from_bytes(json.as_bytes()).unwrap())
    }

    fn boundary() -> BpeSyncBoundary {
        // O=0, S=1, L=2, N=3, M=4; everything unnamed is O.
        let mut table = vec![0u8; N_CODEPOINTS];
        for value in ('a'..='z').chain('A'..='Z') {
            table[value as usize] = 2;
        }
        for value in '0'..='9' {
            table[value as usize] = 3;
        }
        for value in [' ', '\n', '\t'] {
            table[value as usize] = 1;
        }
        for value in [
            '\u{00e9}', '\u{0958}', '\u{0915}', '\u{4e2d}', '\u{1100}', '\u{1161}', '\u{ac00}',
        ] {
            table[value as usize] = 2;
        }
        for value in ['\u{0301}', '\u{0335}', '\u{093c}'] {
            table[value as usize] = 4;
        }
        BpeSyncBoundary::new(table).unwrap()
    }

    /// One token per character: every character start is a candidate.
    fn spans(text: &str) -> (u32, Vec<u32>, Vec<u32>) {
        let chars = text.chars().count() as u32;
        ((chars), (0..chars).collect(), (1..=chars).collect())
    }

    fn raw(text: &str, ceil: u64) -> Option<BpeBoundaryCut> {
        let (chars, starts, ends) = spans(text);
        boundary().last_boundary(text, chars, &starts, &ends, 0, ceil)
    }

    fn sealed(rule: &SealCutRule, text: &str, ceil: u64) -> Option<BpeBoundaryCut> {
        let (chars, starts, ends) = spans(text);
        let whole = boundary()
            .last_boundary_sealed(rule, text, chars, &starts, &ends, 0, ceil)
            .unwrap();
        for window in [1usize, 3, 64] {
            let windowed = boundary()
                .last_boundary_windowed_sealed(
                    rule,
                    text,
                    chars,
                    starts.len(),
                    0,
                    ceil,
                    window,
                    |lo, hi| Ok((starts[lo..hi].to_vec(), ends[lo..hi].to_vec())),
                )
                .unwrap();
            assert_eq!(windowed, whole, "windowed search differs ({window})");
        }
        whole
    }

    fn cut_at(found: Option<BpeBoundaryCut>) -> Option<u64> {
        found.map(|cut| cut.cut_char)
    }

    #[test]
    fn a_letter_that_nfc_decomposes_does_not_certify_the_cut_after_it() {
        // U+0958 normalizes to U+0915 U+093C: the pre-tokenizer reads a mark
        // before the full stop, where the raw text has a letter.
        let text = "ab \u{0958}.com";
        assert_eq!(cut_at(raw(text, u64::MAX)), Some(4));
        let rule = SealCutRule::for_reference(reference(Face::Nfc));
        assert_eq!(rule.mode_name(), "nfc");
        assert_eq!(cut_at(sealed(&rule, text, u64::MAX)), Some(2));
        // Already decomposed, the text is stable and reads a mark either way.
        let decomposed = "ab \u{0915}\u{093c}.com";
        assert_eq!(cut_at(raw(decomposed, u64::MAX)), Some(2));
        assert_eq!(cut_at(sealed(&rule, decomposed, u64::MAX)), Some(2));
    }

    #[test]
    fn a_face_without_a_normalizer_seals_where_the_raw_predicate_does() {
        let rule = SealCutRule::for_reference(reference(Face::Plain));
        assert_eq!(rule.mode_name(), "identity");
        for text in [
            "ab \u{0958}.com",
            "x\u{1161}.\u{0301}a 1.b",
            "plain ascii. text 12",
        ] {
            let total = text.chars().count() as u64;
            for ceil in 0..=total {
                assert_eq!(
                    sealed(&rule, text, ceil),
                    raw(text, ceil),
                    "{text:?} {ceil}"
                );
            }
        }
    }

    #[test]
    fn stable_text_seals_where_the_raw_predicate_does() {
        let rule = SealCutRule::for_reference(reference(Face::Nfc));
        let long_jamo = format!("x{}.ab", "\u{1161}".repeat(300));
        let texts = [
            "caf\u{00e9}. na\u{00ef}ve 12.x".to_owned(),
            "\u{4e2d}a.1 \u{4e2d}\u{4e2d}, b".to_owned(),
            "ab \u{0915}\u{093c}.com \u{0915}\u{093c}x 7".to_owned(),
            "\u{ac00}.\u{1161}a \u{1100}x.9".to_owned(),
            "q\u{0301}.a \u{00e9}\u{0335}.b".to_owned(),
            "plain ascii. text 12, and more".to_owned(),
            long_jamo,
        ];
        for text in &texts {
            let reference = reference(Face::Nfc);
            assert_eq!(
                reference.normalize(text).unwrap(),
                *text,
                "fixture must be stable"
            );
            let total = text.chars().count() as u64;
            for ceil in 0..=total {
                assert_eq!(
                    sealed(&rule, text, ceil),
                    raw(text, ceil),
                    "{text:?} {ceil}"
                );
            }
        }
    }

    #[test]
    fn an_undecided_cut_is_admitted_only_on_a_stable_tail() {
        let rule = SealCutRule::for_reference(reference(Face::Nfc));
        // More than ANCHOR_SCAN conjoining vowels: no stable split point on
        // the left of the cut before the full stop.
        let run = "\u{1161}".repeat(ANCHOR_SCAN + 20);
        let stable = format!("xyz{run}.ab");
        let cut = Some(3 + run.chars().count() as u64);
        assert_eq!(cut_at(raw(&stable, u64::MAX)), cut);
        assert_eq!(cut_at(sealed(&rule, &stable, u64::MAX)), cut);
        // The same tail behind a character NFC changes is not stable.
        let unstable = format!("\u{0958}yz{run}.ab");
        assert_eq!(cut_at(raw(&unstable, u64::MAX)), cut);
        assert_eq!(cut_at(sealed(&rule, &unstable, u64::MAX)), None);
    }

    #[test]
    fn other_normalizers_and_a_closed_rule_seal_nothing() {
        let lowercase = SealCutRule::for_reference(reference(Face::Lowercase));
        assert_eq!(lowercase.mode_name(), "closed");
        let closed = SealCutRule::closed();
        for text in ["plain ascii. text 12", "ab \u{0958}.com"] {
            assert!(raw(text, u64::MAX).is_some());
            assert_eq!(sealed(&lowercase, text, u64::MAX), None);
            assert_eq!(sealed(&closed, text, u64::MAX), None);
        }
    }
}
