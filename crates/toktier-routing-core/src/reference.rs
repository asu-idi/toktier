//! Pinned Hugging Face reference engine used by the native request path.

use std::collections::HashSet;
use std::fmt;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use sha2::{Digest, Sha256};
use tokenizers::{NormalizedString, Normalizer, Tokenizer};
use toktier_store_core::{
    AppendReport, BoundaryCut, Encoding, EngineError, SessionEncoder, SoaEncoding, TailState,
    WitnessCategory,
};

/// A load or execution failure from the frozen native reference engine.
#[derive(Debug, Clone)]
pub struct ReferenceEngineError(String);

impl ReferenceEngineError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ReferenceEngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ReferenceEngineError {}

/// The exact `tokenizers==0.22.2` implementation loaded from one verified
/// tokenizer artifact.
///
/// Artifact hashing and registry admission remain construction-time concerns
/// of the Python facade.  This type receives only the already-verified local
/// `tokenizer.json` path and owns all request-time execution thereafter.
#[derive(Clone)]
pub struct ReferenceEngine {
    tokenizer: Tokenizer,
    normalizer_is_identity: bool,
    artifact_sha256: [u8; 32],
    raw_byte_lengths: Arc<OnceLock<Result<Arc<[usize]>, ReferenceEngineError>>>,
}

impl fmt::Debug for ReferenceEngine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReferenceEngine")
            .field("oracle", &"tokenizers==0.22.2")
            .finish_non_exhaustive()
    }
}

impl ReferenceEngine {
    /// Load the artifact exactly as written.  No loader flags are accepted or
    /// synthesized here.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ReferenceEngineError> {
        let bytes = std::fs::read(path.as_ref()).map_err(|error| {
            ReferenceEngineError::new(format!(
                "failed to load tokenizer artifact {}: {error}",
                path.as_ref().display()
            ))
        })?;
        Self::from_bytes(&bytes).map_err(|error| {
            ReferenceEngineError::new(format!(
                "failed to load tokenizer artifact {}: {error}",
                path.as_ref().display()
            ))
        })
    }

    /// Load an already materialized tokenizer JSON document. This is used by
    /// the corrected CPU route when a configuration sidecar contributed
    /// added tokens to the live tokenizer before it was serialized.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReferenceEngineError> {
        let artifact_sha256 = Sha256::digest(bytes).into();
        let normalizer_is_identity = serde_json::from_slice::<serde_json::Value>(bytes)
            .ok()
            .and_then(|document| document.get("normalizer").cloned())
            .is_none_or(|normalizer| {
                normalizer.is_null()
                    || normalizer.as_object().is_some_and(|value| {
                        value.get("type").and_then(|kind| kind.as_str()) == Some("Sequence")
                            && value
                                .get("normalizers")
                                .and_then(|rows| rows.as_array())
                                .is_some_and(Vec::is_empty)
                    })
            });
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(|error| {
            ReferenceEngineError::new(format!("failed to load tokenizer JSON: {error}"))
        })?;
        Ok(Self {
            tokenizer,
            normalizer_is_identity,
            artifact_sha256,
            raw_byte_lengths: Arc::new(OnceLock::new()),
        })
    }

    /// Encode one input, optionally applying the artifact post-processor.
    pub fn encode_ids(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, ReferenceEngineError> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| ReferenceEngineError::new(error.to_string()))
    }

    /// Encode the core stream with character-unit spans for the native store.
    pub fn encode_core(&self, text: &str) -> Result<Encoding, ReferenceEngineError> {
        let encoded = self
            .tokenizer
            .encode_char_offsets(text, false)
            .map_err(|error| ReferenceEngineError::new(error.to_string()))?;
        let spans = encoded
            .get_offsets()
            .iter()
            .map(|&(start, end)| {
                let start = u32::try_from(start).map_err(|_| {
                    ReferenceEngineError::new("token start offset exceeds store format limits")
                })?;
                let end = u32::try_from(end).map_err(|_| {
                    ReferenceEngineError::new("token end offset exceeds store format limits")
                })?;
                Ok((start, end))
            })
            .collect::<Result<Vec<_>, ReferenceEngineError>>()?;
        Ok(Encoding {
            ids: encoded.get_ids().to_vec(),
            spans,
        })
    }

    /// Encode the core stream with character-unit spans directly in the
    /// store's structure-of-arrays layout. IDs and span values are exactly
    /// those of [`Self::encode_core`]; only the carrier differs, so the
    /// state path can adopt the arrays without pair formation.
    pub fn encode_state_soa(&self, text: &str) -> Result<SoaEncoding, ReferenceEngineError> {
        let encoded = self
            .tokenizer
            .encode_char_offsets(text, false)
            .map_err(|error| ReferenceEngineError::new(error.to_string()))?;
        let offsets = encoded.get_offsets();
        let mut span_starts = Vec::with_capacity(offsets.len());
        let mut span_ends = Vec::with_capacity(offsets.len());
        for &(start, end) in offsets {
            span_starts.push(u32::try_from(start).map_err(|_| {
                ReferenceEngineError::new("token start offset exceeds store format limits")
            })?);
            span_ends.push(u32::try_from(end).map_err(|_| {
                ReferenceEngineError::new("token end offset exceeds store format limits")
            })?);
        }
        Ok(SoaEncoding {
            ids: encoded.get_ids().to_vec(),
            span_starts,
            span_ends,
        })
    }

    /// Encode a batch through the crate's native parallel batch path.
    pub fn encode_batch_ids(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<u32>>, ReferenceEngineError> {
        self.tokenizer
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map(|rows| rows.into_iter().map(|row| row.get_ids().to_vec()).collect())
            .map_err(|error| ReferenceEngineError::new(error.to_string()))
    }

    /// Decode through the same frozen artifact implementation.
    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, ReferenceEngineError> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|error| ReferenceEngineError::new(error.to_string()))
    }

    /// Whether the artifact has an active normalizer.
    pub fn has_normalizer(&self) -> bool {
        self.tokenizer.get_normalizer().is_some()
    }

    pub fn artifact_sha256(&self) -> &[u8; 32] {
        &self.artifact_sha256
    }

    /// Whether the declared normalizer can change input text.  Hugging Face
    /// serializes an empty ``Sequence`` as a present normalizer even though
    /// it is the identity function; treating that shape as active would send
    /// every document to the added-token reference guard for no semantic
    /// reason (notably the DeepSeek artifacts).
    pub fn has_effective_normalizer(&self) -> bool {
        self.has_normalizer() && !self.normalizer_is_identity
    }

    /// Apply the artifact normalizer, or return the input unchanged.
    pub fn normalize(&self, text: &str) -> Result<String, ReferenceEngineError> {
        let Some(normalizer) = self.tokenizer.get_normalizer() else {
            return Ok(text.to_owned());
        };
        let mut normalized = NormalizedString::from(text);
        normalizer
            .normalize(&mut normalized)
            .map_err(|error| ReferenceEngineError::new(error.to_string()))?;
        Ok(normalized.get().to_owned())
    }

    /// Added-token definitions and IDs from the exact live artifact.
    pub fn added_tokens(&self) -> Vec<(u32, String, bool)> {
        let mut rows = self
            .tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, token)| (id, token.content, token.normalized))
            .collect::<Vec<_>>();
        rows.sort_by_key(|(id, _, _)| *id);
        rows
    }

    /// Raw-byte length of every dense vocabulary ID, reconstructed with the
    /// same GPT-2 byte-alphabet rule as the released Python repair adapter.
    pub fn raw_byte_lengths(&self) -> Result<Vec<usize>, ReferenceEngineError> {
        self.raw_byte_lengths_slice().map(<[usize]>::to_vec)
    }

    /// The frozen raw byte-length table as a shared immutable allocation.
    ///
    /// This is the exact table every span bridge in this module consumes;
    /// sharing the allocation lets a session tail rebuild span regions
    /// lazily from sparse checkpoints without copying the table or asking
    /// the engine back.
    pub fn raw_byte_lengths_arc(&self) -> Result<Arc<[usize]>, ReferenceEngineError> {
        self.raw_byte_lengths_result()
            .as_ref()
            .map(Arc::clone)
            .map_err(Clone::clone)
    }

    fn raw_byte_lengths_result(&self) -> &Result<Arc<[usize]>, ReferenceEngineError> {
        self.raw_byte_lengths
            .get_or_init(|| self.build_raw_byte_lengths().map(Vec::into))
    }

    fn raw_byte_lengths_slice(&self) -> Result<&[usize], ReferenceEngineError> {
        self.raw_byte_lengths_result()
            .as_ref()
            .map(|table| &table[..])
            .map_err(Clone::clone)
    }

    fn build_raw_byte_lengths(&self) -> Result<Vec<usize>, ReferenceEngineError> {
        let vocabulary = self.tokenizer.get_vocab(true);
        if vocabulary.is_empty() {
            return Err(ReferenceEngineError::new(
                "the live tokenizer has an empty vocabulary",
            ));
        }
        let max_id = vocabulary.values().copied().max().unwrap_or(0) as usize;
        let added = self
            .tokenizer
            .get_added_vocabulary()
            .get_vocab()
            .keys()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let alphabet = byte_alphabet();
        let mut lengths = vec![0usize; max_id + 1];
        let mut seen = vec![false; max_id + 1];
        for (token, raw_id) in vocabulary {
            let id = raw_id as usize;
            if id >= lengths.len() {
                return Err(ReferenceEngineError::new(format!(
                    "vocabulary id {id} is out of range"
                )));
            }
            lengths[id] = if !added.contains(token.as_str())
                && token.chars().all(|value| alphabet.contains(&value))
            {
                token.chars().count()
            } else {
                token.len()
            };
            seen[id] = true;
        }
        if let Some(missing) = seen.iter().position(|present| !present) {
            return Err(ReferenceEngineError::new(format!(
                "the live tokenizer has no vocabulary entry for id {missing}"
            )));
        }
        Ok(lengths)
    }

    /// Reconstruct character-unit spans for an exact core-stream ID row.
    ///
    /// This is the native GPU-to-store bridge. It uses the frozen HF
    /// artifact's own raw vocabulary table, so a GPU-routed state seed does
    /// not need to initialize the independent corrected-CPU engine merely to
    /// recover offsets.
    pub fn spans_for_ids(
        &self,
        text: &str,
        ids: &[u32],
    ) -> Result<Vec<(u32, u32)>, ReferenceEngineError> {
        spans_from_ids(ids, self.raw_byte_lengths_slice()?, text)
    }

    /// Reconstruct character-unit spans for an exact core-stream ID row
    /// directly in the store's structure-of-arrays layout, in one pass.
    ///
    /// This is the direct successor of [`Self::spans_for_ids`]: the ASCII
    /// path fills the final `u32` arrays while accumulating the closure
    /// cursor, and the non-ASCII path merges UTF-8 character boundaries
    /// with token byte endpoints in a single forward stream instead of
    /// materializing a byte-to-character map for every input byte. Both
    /// share the pair bridge's premises and failure behavior case for
    /// case: the input must be normalization-stable for the artifact
    /// (otherwise token bytes cannot close over the given text and the
    /// same fail-closed closure error is returned), byte-fallback tokens
    /// whose byte interval lies inside one code point share that
    /// character's span, and unknown-ID, zero-length, overflow, bounds,
    /// and closure errors carry the same messages. The pair bridge
    /// remains available as the retained fallback and comparison oracle.
    pub fn spans_soa_for_ids(
        &self,
        text: &str,
        ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<u32>), ReferenceEngineError> {
        spans_soa_from_ids(ids, self.raw_byte_lengths_slice()?, text)
    }

    /// Verify that an exact ID row's raw token bytes close over `text`
    /// without materializing any span: the same unknown-ID, zero-length,
    /// overflow, and closure rejections as the span bridges, in one
    /// allocation-free pass. ID-only callers use this to keep the span
    /// bridges' fail-closed acceptance behavior.
    pub fn verify_ids_close(&self, text: &str, ids: &[u32]) -> Result<(), ReferenceEngineError> {
        ids_close_over_text(ids, self.raw_byte_lengths_slice()?, text)
    }

    /// Build and freeze the raw byte-length table now instead of on the
    /// first span reconstruction, so a construction-time caller pays (and
    /// reports) table initialization instead of hiding it inside the first
    /// stateful request. A table failure is recorded by the underlying
    /// once-cell and replays identically at first use, preserving the
    /// existing fail-closed bridge behavior.
    pub fn prewarm_raw_byte_lengths(&self) -> Result<(), ReferenceEngineError> {
        self.raw_byte_lengths_slice().map(|_| ())
    }

    fn has_added_id(&self, ids: &[u32]) -> bool {
        let added_ids = self
            .tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder();
        ids.iter().any(|id| added_ids.contains_key(id))
    }

    /// Encode once and report whether an added-token ID occurred. The
    /// encoding is returned so a reference-routed caller does not repeat it.
    pub fn encode_core_with_added_flag(
        &self,
        text: &str,
    ) -> Result<(Encoding, bool), ReferenceEngineError> {
        let encoding = self.encode_core(text)?;
        let has_added = self.has_added_id(&encoding.ids);
        Ok((encoding, has_added))
    }

    /// Structure-of-arrays counterpart of
    /// [`Self::encode_core_with_added_flag`] for the state route.
    pub fn encode_state_soa_with_added_flag(
        &self,
        text: &str,
    ) -> Result<(SoaEncoding, bool), ReferenceEngineError> {
        let encoding = self.encode_state_soa(text)?;
        let has_added = self.has_added_id(&encoding.ids);
        Ok((encoding, has_added))
    }

    /// ID-only counterpart of [`Self::encode_core_with_added_flag`]: the
    /// same core stream and added-token decision without constructing
    /// character spans a stateless caller did not request.
    pub fn encode_ids_with_added_flag(
        &self,
        text: &str,
    ) -> Result<(Vec<u32>, bool), ReferenceEngineError> {
        let ids = self.encode_ids(text, false)?;
        let has_added = self.has_added_id(&ids);
        Ok((ids, has_added))
    }
}

fn byte_alphabet() -> HashSet<char> {
    let mut visible = (0x21u32..0x7f)
        .chain(0xa1..0xad)
        .chain(0xae..0x100)
        .collect::<Vec<_>>();
    let mut mapped = visible.clone();
    let mut extra = 0u32;
    for byte in 0u32..=255 {
        if !visible.contains(&byte) {
            visible.push(byte);
            mapped.push(256 + extra);
            extra += 1;
        }
    }
    mapped
        .into_iter()
        .filter_map(char::from_u32)
        .collect::<HashSet<_>>()
}

/// Raw byte length of one vocabulary ID, with the bridge's frozen
/// unknown-ID and zero-length rejections. Both span bridges and the
/// ID-only closure check share this single definition.
fn token_byte_length(byte_lengths: &[usize], id: u32) -> Result<usize, ReferenceEngineError> {
    let length = *byte_lengths
        .get(id as usize)
        .ok_or_else(|| ReferenceEngineError::new(format!("tokenizer returned unknown id {id}")))?;
    if length == 0 {
        return Err(ReferenceEngineError::new(format!(
            "tokenizer returned zero-byte vocabulary id {id}"
        )));
    }
    Ok(length)
}

fn no_ids_for_text() -> ReferenceEngineError {
    ReferenceEngineError::new("tokenizer returned no ids for non-empty text")
}

fn byte_length_overflow() -> ReferenceEngineError {
    ReferenceEngineError::new("token byte length overflow")
}

fn closure_mismatch(cursor: usize, text_len: usize) -> ReferenceEngineError {
    ReferenceEngineError::new(format!(
        "token bytes do not close: tokens={cursor}, text={text_len}"
    ))
}

/// Allocation-free closure check over an exact ID row: the same
/// per-token rejections and final closure error as the span bridges,
/// with no span or map materialization.
fn ids_close_over_text(
    ids: &[u32],
    byte_lengths: &[usize],
    text: &str,
) -> Result<(), ReferenceEngineError> {
    if ids.is_empty() {
        return if text.is_empty() {
            Ok(())
        } else {
            Err(no_ids_for_text())
        };
    }
    let mut cursor = 0usize;
    for &id in ids {
        cursor = cursor
            .checked_add(token_byte_length(byte_lengths, id)?)
            .ok_or_else(byte_length_overflow)?;
    }
    if cursor != text.len() {
        return Err(closure_mismatch(cursor, text.len()));
    }
    Ok(())
}

fn spans_from_ids(
    ids: &[u32],
    byte_lengths: &[usize],
    text: &str,
) -> Result<Vec<(u32, u32)>, ReferenceEngineError> {
    if ids.is_empty() {
        return if text.is_empty() {
            Ok(Vec::new())
        } else {
            Err(no_ids_for_text())
        };
    }
    let mut starts = Vec::with_capacity(ids.len());
    let mut ends = Vec::with_capacity(ids.len());
    let mut cursor = 0usize;
    for &id in ids {
        let length = token_byte_length(byte_lengths, id)?;
        starts.push(cursor);
        cursor = cursor
            .checked_add(length)
            .ok_or_else(byte_length_overflow)?;
        ends.push(cursor);
    }
    if cursor != text.len() {
        return Err(closure_mismatch(cursor, text.len()));
    }
    if text.is_ascii() {
        return starts
            .into_iter()
            .zip(ends)
            .map(|(start, end)| {
                Ok((
                    u32::try_from(start)
                        .map_err(|_| ReferenceEngineError::new("span exceeds u32"))?,
                    u32::try_from(end)
                        .map_err(|_| ReferenceEngineError::new("span exceeds u32"))?,
                ))
            })
            .collect();
    }
    let mut char_of_byte = Vec::with_capacity(text.len());
    for (index, value) in text.chars().enumerate() {
        char_of_byte.extend(std::iter::repeat_n(index, value.len_utf8()));
    }
    starts
        .into_iter()
        .zip(ends)
        .map(|(start, end)| {
            let start_char = *char_of_byte.get(start).ok_or_else(|| {
                ReferenceEngineError::new("token start is outside the UTF-8 input")
            })?;
            let end_char = char_of_byte
                .get(end - 1)
                .copied()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| ReferenceEngineError::new("token end is outside the UTF-8 input"))?;
            Ok((
                u32::try_from(start_char)
                    .map_err(|_| ReferenceEngineError::new("span exceeds u32"))?,
                u32::try_from(end_char)
                    .map_err(|_| ReferenceEngineError::new("span exceeds u32"))?,
            ))
        })
        .collect()
}

/// One-pass structure-of-arrays successor of [`spans_from_ids`].
///
/// Behavioral contract (checked element for element and error for error
/// against the pair bridge by the test battery below): identical spans on
/// success, identical error message for every rejected input. To keep the
/// pair bridge's error ordering -- where the whole row's per-ID checks and
/// the final closure check run before any span-conversion error can
/// surface -- conversion/bounds failures inside the single pass are
/// deferred: the scan continues accumulating byte lengths so a later
/// unknown/zero/overflow rejection or a closure mismatch still wins, and
/// the deferred error is returned only when the row otherwise closes.
fn spans_soa_from_ids(
    ids: &[u32],
    byte_lengths: &[usize],
    text: &str,
) -> Result<(Vec<u32>, Vec<u32>), ReferenceEngineError> {
    if ids.is_empty() {
        return if text.is_empty() {
            Ok((Vec::new(), Vec::new()))
        } else {
            Err(no_ids_for_text())
        };
    }
    if text.is_ascii() {
        ascii_spans_soa(ids, byte_lengths, text.len())
    } else {
        unicode_spans_soa(ids, byte_lengths, text)
    }
}

/// ASCII: byte positions are character positions, so the final `u32`
/// arrays fill directly while the closure cursor accumulates. No
/// intermediate allocation exists on this path.
fn ascii_spans_soa(
    ids: &[u32],
    byte_lengths: &[usize],
    text_len: usize,
) -> Result<(Vec<u32>, Vec<u32>), ReferenceEngineError> {
    let mut starts = Vec::with_capacity(ids.len());
    let mut ends = Vec::with_capacity(ids.len());
    let mut cursor = 0usize;
    let mut deferred: Option<ReferenceEngineError> = None;
    for &id in ids {
        let length = token_byte_length(byte_lengths, id)?;
        let start = cursor;
        cursor = cursor
            .checked_add(length)
            .ok_or_else(byte_length_overflow)?;
        if deferred.is_none() {
            match (u32::try_from(start), u32::try_from(cursor)) {
                (Ok(start), Ok(end)) => {
                    starts.push(start);
                    ends.push(end);
                }
                _ => deferred = Some(ReferenceEngineError::new("span exceeds u32")),
            }
        }
    }
    if cursor != text_len {
        return Err(closure_mismatch(cursor, text_len));
    }
    if let Some(error) = deferred {
        return Err(error);
    }
    Ok((starts, ends))
}

/// Streaming byte-to-character cursor over UTF-8 text. It advances
/// strictly forward and reports the index of the character containing a
/// given byte offset; queries must be monotonically non-decreasing, which
/// the token endpoint sequence guarantees.
struct CharCursor<'a> {
    rest: std::str::CharIndices<'a>,
    index: usize,
    start: usize,
    end: usize,
    exhausted: bool,
}

impl<'a> CharCursor<'a> {
    fn new(text: &'a str) -> CharCursor<'a> {
        let mut rest = text.char_indices();
        match rest.next() {
            Some((_, first)) => CharCursor {
                rest,
                index: 0,
                start: 0,
                end: first.len_utf8(),
                exhausted: false,
            },
            None => CharCursor {
                rest,
                index: 0,
                start: 0,
                end: 0,
                exhausted: true,
            },
        }
    }

    /// Index of the character containing byte offset `byte`, or `None`
    /// when the offset lies outside the streamed text.
    fn char_containing(&mut self, byte: usize) -> Option<usize> {
        if self.exhausted || byte < self.start {
            return None;
        }
        while self.end <= byte {
            let (offset, value) = self.rest.next()?;
            self.index += 1;
            self.start = offset;
            self.end = offset + value.len_utf8();
        }
        Some(self.index)
    }
}

/// Non-ASCII: one forward merge of UTF-8 character boundaries and token
/// byte endpoints. Byte-fallback tokens whose byte interval lies inside
/// one code point receive that character's span, exactly as the pair
/// bridge's byte-to-character map produces; no such map is built.
fn unicode_spans_soa(
    ids: &[u32],
    byte_lengths: &[usize],
    text: &str,
) -> Result<(Vec<u32>, Vec<u32>), ReferenceEngineError> {
    let mut starts = Vec::with_capacity(ids.len());
    let mut ends = Vec::with_capacity(ids.len());
    let mut cursor = CharCursor::new(text);
    let mut byte = 0usize;
    let mut deferred: Option<ReferenceEngineError> = None;
    for &id in ids {
        let length = token_byte_length(byte_lengths, id)?;
        let start_byte = byte;
        byte = byte.checked_add(length).ok_or_else(byte_length_overflow)?;
        if deferred.is_none() {
            deferred =
                push_unicode_span(&mut cursor, start_byte, byte, &mut starts, &mut ends).err();
        }
    }
    if byte != text.len() {
        return Err(closure_mismatch(byte, text.len()));
    }
    if let Some(error) = deferred {
        return Err(error);
    }
    Ok((starts, ends))
}

/// Map one token's byte interval `[start_byte, end_byte)` to a character
/// span, preserving the pair bridge's error kinds and per-token check
/// order (start bound, end bound, start width, end width).
fn push_unicode_span(
    cursor: &mut CharCursor<'_>,
    start_byte: usize,
    end_byte: usize,
    starts: &mut Vec<u32>,
    ends: &mut Vec<u32>,
) -> Result<(), ReferenceEngineError> {
    let start_char = cursor
        .char_containing(start_byte)
        .ok_or_else(|| ReferenceEngineError::new("token start is outside the UTF-8 input"))?;
    let end_char = cursor
        .char_containing(end_byte - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| ReferenceEngineError::new("token end is outside the UTF-8 input"))?;
    let start =
        u32::try_from(start_char).map_err(|_| ReferenceEngineError::new("span exceeds u32"))?;
    let end = u32::try_from(end_char).map_err(|_| ReferenceEngineError::new("span exceeds u32"))?;
    starts.push(start);
    ends.push(end);
    Ok(())
}

impl SessionEncoder for ReferenceEngine {
    fn encode(&self, text: &str) -> Result<Encoding, EngineError> {
        self.encode_core(text)
            .map_err(|error| EngineError(error.to_string()))
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        let was_empty = tail.text().is_empty();
        let mut full = String::with_capacity(tail.text_bytes() + delta.len());
        full.push_str(tail.text());
        full.push_str(delta);
        let encoding = self
            .encode_core(&full)
            .map_err(|error| EngineError(error.to_string()))?;
        tail.fill(&full, encoding)
            .map_err(|error| EngineError(error.to_string()))?;
        Ok(AppendReport {
            path: if was_empty {
                "cold_full_native_hf".to_string()
            } else {
                "native_hf_full_reencode".to_string()
            },
            kept_tokens: 0,
        })
    }

    fn last_certified_boundary(
        &self,
        _tail: &TailState,
        _floor_char: u64,
        _ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        Ok(None)
    }

    fn witness_category(&self) -> WitnessCategory {
        WitnessCategory::NoneFullReencode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::BPE;

    #[test]
    fn reference_engine_core_offsets_are_character_units() {
        let engine = ReferenceEngine {
            tokenizer: Tokenizer::new(BPE::default()),
            normalizer_is_identity: true,
            artifact_sha256: Sha256::digest(b"test-reference").into(),
            raw_byte_lengths: Arc::new(OnceLock::new()),
        };
        let encoded = engine.encode_core("é").unwrap();
        assert_eq!(encoded.ids.len(), encoded.spans.len());
        assert!(encoded.spans.iter().all(|&(_start, end)| end <= 1));
    }

    fn tiny_engine() -> ReferenceEngine {
        let model = BPE::builder()
            .vocab_and_merges([("a".to_owned(), 0)], Vec::new())
            .build()
            .unwrap();
        let tokenizer = Tokenizer::new(model);
        ReferenceEngine::from_bytes(tokenizer.to_string(false).unwrap().as_bytes()).unwrap()
    }

    #[test]
    fn state_soa_and_id_surfaces_match_the_pair_surface() {
        let engine = tiny_engine();
        let pairs = engine.encode_core("aaa").unwrap();
        let soa = engine.encode_state_soa("aaa").unwrap();
        assert_eq!(soa.ids, pairs.ids);
        assert_eq!(soa.clone().into_pairs().spans, pairs.spans);
        let (flag_soa, added_soa) = engine.encode_state_soa_with_added_flag("aaa").unwrap();
        let (flag_pairs, added_pairs) = engine.encode_core_with_added_flag("aaa").unwrap();
        assert_eq!(flag_soa.ids, flag_pairs.ids);
        assert_eq!(added_soa, added_pairs);
        let (ids, added_ids_flag) = engine.encode_ids_with_added_flag("aaa").unwrap();
        assert_eq!(ids, pairs.ids);
        assert_eq!(added_ids_flag, added_pairs);
    }

    #[test]
    fn engine_level_bridges_and_closure_check_agree() {
        let engine = tiny_engine();
        engine.prewarm_raw_byte_lengths().unwrap();
        let ids = vec![0u32, 0, 0];
        let pairs = engine.spans_for_ids("aaa", &ids).unwrap();
        let (starts, ends) = engine.spans_soa_for_ids("aaa", &ids).unwrap();
        assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(starts, vec![0, 1, 2]);
        assert_eq!(ends, vec![1, 2, 3]);
        engine.verify_ids_close("aaa", &ids).unwrap();
        let closure_err = engine.verify_ids_close("aa", &ids).unwrap_err().to_string();
        let pair_err = engine.spans_for_ids("aa", &ids).unwrap_err().to_string();
        let soa_err = engine
            .spans_soa_for_ids("aa", &ids)
            .unwrap_err()
            .to_string();
        assert_eq!(pair_err, "token bytes do not close: tokens=3, text=2");
        assert_eq!(closure_err, pair_err);
        assert_eq!(soa_err, pair_err);
    }

    /// The third span path: a lazily filled session tail rebuilding the
    /// requested region from sparse checkpoints. Success values must be
    /// element-identical to both converter bridges over any window; the
    /// acceptance decision must match the closure check.
    fn assert_lazy_tail_path(
        ids: &[u32],
        lengths: &[usize],
        text: &str,
        expected: Option<(&[u32], &[u32])>,
    ) {
        use toktier_store_core::{SharedIds, TailState};
        let table: std::sync::Arc<[usize]> = lengths.to_vec().into();
        let mut tail = TailState::new();
        let filled = tail.fill_lazy(text, SharedIds::from_vec(ids.to_vec()), table);
        let Some((want_starts, want_ends)) = expected else {
            assert!(
                filled.is_err(),
                "lazy fill accepted a rejected row {ids:?} over {text:?}"
            );
            return;
        };
        filled.unwrap_or_else(|error| {
            panic!("lazy fill rejected an accepted row {ids:?} over {text:?}: {error}")
        });
        let n = ids.len();
        let mut probes = vec![(0, n), (0, n / 2), (n / 2, n), (n / 3, 2 * n / 3)];
        if n > 0 {
            probes.push((n - 1, n));
        }
        for (lo, hi) in probes {
            let (starts, ends) = tail.span_window(lo, hi).unwrap();
            assert_eq!(
                starts,
                want_starts[lo..hi],
                "lazy starts [{lo}, {hi}) for {ids:?} over {text:?}"
            );
            assert_eq!(
                ends,
                want_ends[lo..hi],
                "lazy ends [{lo}, {hi}) for {ids:?} over {text:?}"
            );
        }
    }

    /// Both bridges must agree element for element on success and message
    /// for message on rejection; the ID-only closure check must agree with
    /// the bridges' pre-conversion acceptance decision; and the lazy
    /// session-tail rebuild must reproduce the same spans window by window
    /// (or reject the same rows).
    fn assert_dual_path(ids: &[u32], lengths: &[usize], text: &str) {
        let pair_result = spans_from_ids(ids, lengths, text);
        let soa_result = spans_soa_from_ids(ids, lengths, text);
        let closure = ids_close_over_text(ids, lengths, text);
        match (&pair_result, &soa_result) {
            (Ok(pairs), Ok((starts, ends))) => {
                assert_eq!(pairs.len(), starts.len(), "count for {ids:?} over {text:?}");
                assert_eq!(pairs.len(), ends.len(), "count for {ids:?} over {text:?}");
                for (index, &(start, end)) in pairs.iter().enumerate() {
                    assert_eq!(
                        (start, end),
                        (starts[index], ends[index]),
                        "span {index} for {ids:?} over {text:?}"
                    );
                }
                assert!(closure.is_ok(), "closure rejected an accepted row");
                assert_lazy_tail_path(ids, lengths, text, Some((starts, ends)));
            }
            (Err(pair_error), Err(soa_error)) => {
                assert_eq!(
                    pair_error.to_string(),
                    soa_error.to_string(),
                    "error mismatch for {ids:?} over {text:?}"
                );
                // None of the test inputs can reach the conversion-only
                // errors, so the closure check rejects with the same text.
                assert_eq!(
                    closure.unwrap_err().to_string(),
                    pair_error.to_string(),
                    "closure error mismatch for {ids:?} over {text:?}"
                );
                assert_lazy_tail_path(ids, lengths, text, None);
            }
            (pair_result, soa_result) => panic!(
                "dual-path disagreement for {ids:?} over {text:?}: \
                 pair={pair_result:?}, soa={soa_result:?}"
            ),
        }
    }

    #[test]
    fn dual_path_boundary_shapes() {
        // Lengths table: id i has raw byte length i for 1..=8.
        let lengths: Vec<usize> = (0..9).collect();
        let l = &lengths[..];

        // Empty row over empty text; missing row over non-empty text.
        assert_dual_path(&[], l, "");
        assert_dual_path(&[], l, "x");
        // Single token; token count 1 over the wrong text length.
        assert_dual_path(&[3], l, "abc");
        assert_dual_path(&[3], l, "abcd");
        // Maximum known ID and the first unknown ID.
        assert_dual_path(&[8], l, "12345678");
        assert_dual_path(&[9], l, "12345678");
        assert_dual_path(&[u32::MAX], l, "12345678");
        // Zero-byte vocabulary entry (id 0 has length 0).
        assert_dual_path(&[0], l, "");
        assert_dual_path(&[1, 0, 1], l, "ab");
        // Non-closing rows: undershoot and overshoot, ASCII and Unicode.
        assert_dual_path(&[1, 1], l, "abc");
        assert_dual_path(&[2, 2], l, "abc");
        assert_dual_path(&[2], l, "\u{4f60}");
        assert_dual_path(&[4], l, "\u{4f60}");
        // Every UTF-8 width with exact-width tokens.
        assert_dual_path(&[1], l, "a");
        assert_dual_path(&[2], l, "\u{e9}");
        assert_dual_path(&[3], l, "\u{4f60}");
        assert_dual_path(&[4], l, "\u{1f642}");
        // Byte-fallback groups: several tokens sharing one character span.
        assert_dual_path(&[1, 1], l, "\u{e9}");
        assert_dual_path(&[1, 1, 1], l, "\u{4f60}");
        assert_dual_path(&[1, 1, 1, 1], l, "\u{1f642}");
        assert_dual_path(&[2, 2], l, "\u{1f642}");
        assert_dual_path(&[1, 2], l, "\u{4f60}");
        assert_dual_path(&[2, 1], l, "\u{4f60}");
        // Tokens straddling character boundaries.
        assert_dual_path(&[2, 3], l, "a\u{4f60}b");
        assert_dual_path(&[4, 1], l, "a\u{4f60}b");
        assert_dual_path(&[1, 3, 1], l, "a\u{4f60}b");
        // Combining marks, ZWJ emoji, CJK, RTL, CRLF, and mixed ASCII.
        assert_dual_path(&[1, 2], l, "q\u{301}");
        assert_dual_path(&[1, 1, 1], l, "q\u{301}");
        assert_dual_path(&[4, 3, 4], l, "\u{1f469}\u{200d}\u{1f4bb}");
        assert_dual_path(&[5, 6], l, "\u{1f469}\u{200d}\u{1f4bb}");
        assert_dual_path(&[3, 3], l, "\u{4f60}\u{597d}");
        assert_dual_path(&[2, 2, 2], l, "\u{5d0}\u{5d1}\u{5d2}");
        assert_dual_path(&[1, 1], l, "\r\n");
        assert_dual_path(&[2, 1, 1], l, "\u{e9}\r\n");
        assert_dual_path(&[3, 1, 4, 2], l, " ab\u{4f60}\u{1f642}\u{e9}");
        // First and last token positions in a longer mixed row.
        assert_dual_path(&[1, 3, 1, 3, 1], l, "x\u{4f60}y\u{597d}z");
    }

    #[test]
    fn dual_path_overflow_and_error_ordering() {
        // Cursor overflow on the second token.
        let huge = [usize::MAX, 1];
        assert_dual_path(&[0, 0], &huge, "abc");
        // A huge closing failure: the closure error must carry the full
        // accumulated cursor even though span conversion cannot represent
        // it (the pair bridge never reaches conversion on closure failure).
        assert_dual_path(&[0], &huge, "abc");
        assert_dual_path(&[0], &huge, "\u{4f60}");
        // A later unknown ID must win over an earlier unrepresentable span,
        // matching the pair bridge's whole-row-first evaluation order.
        assert_dual_path(&[0, 99], &huge, "abc");
        assert_dual_path(&[0, 99], &huge, "\u{4f60}");
        // A later zero-length ID likewise.
        let huge_zero = [usize::MAX, 0];
        assert_dual_path(&[0, 1], &huge_zero, "abc");
    }

    #[test]
    fn lazy_tail_windows_match_the_converter_across_checkpoints() {
        use toktier_store_core::{SharedIds, TailState, SPAN_CHECKPOINT_STRIDE};
        // A long mixed row crossing several sparse checkpoints, with
        // byte-fallback splits of multi-byte characters placed on both
        // sides of the checkpoint stride.
        let lengths: Vec<usize> = (0..5).collect();
        let unit = "ab \u{4f60}\u{1f642}\u{e9}x"; // 7 chars, 12 bytes
        let unit_ids: Vec<u32> = vec![1, 1, 1, 2, 1, 4, 2, 1]; // closes over 12 bytes
        let repeats = 2 * SPAN_CHECKPOINT_STRIDE / unit_ids.len() + 7;
        let text: String = unit.repeat(repeats);
        let ids: Vec<u32> = unit_ids
            .iter()
            .copied()
            .cycle()
            .take(unit_ids.len() * repeats)
            .collect();
        let (want_starts, want_ends) = spans_soa_from_ids(&ids, &lengths, &text).unwrap();
        let table: std::sync::Arc<[usize]> = lengths.into();
        let mut tail = TailState::new();
        tail.fill_lazy(&text, SharedIds::from_vec(ids.clone()), table)
            .unwrap();
        let n = ids.len();
        let mut probes = vec![(0, n), (n - 3, n)];
        let mut mark = SPAN_CHECKPOINT_STRIDE;
        while mark < n {
            probes.push((mark - 2, (mark + 2).min(n)));
            probes.push((mark - 9, mark));
            probes.push((mark, (mark + 33).min(n)));
            mark += SPAN_CHECKPOINT_STRIDE;
        }
        for (lo, hi) in probes {
            let (starts, ends) = tail.span_window(lo, hi).unwrap();
            assert_eq!(starts, want_starts[lo..hi], "starts [{lo}, {hi})");
            assert_eq!(ends, want_ends[lo..hi], "ends [{lo}, {hi})");
        }
        assert!(!tail.spans_materialized());
        // The whole-row materialization agrees with the converter too.
        assert_eq!(tail.span_starts(), &want_starts[..]);
        assert_eq!(tail.span_ends(), &want_ends[..]);
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
    fn dual_path_property_fuzz() {
        // Character pool covering every UTF-8 width, combining marks, ZWJ,
        // CRLF, and RTL text (ASCII escapes only in source).
        let pool = [
            'a',
            'B',
            '7',
            ' ',
            '\u{e9}',
            '\u{430}',
            '\u{4f60}',
            '\u{597d}',
            '\u{1f642}',
            '\u{1f469}',
            '\u{200d}',
            '\u{301}',
            '\u{5d0}',
            '\r',
            '\n',
        ];
        // Lengths table: several IDs per byte length 1..=4, plus one
        // zero-byte entry at id 20 for malicious injections.
        let mut lengths = vec![0usize; 21];
        for (id, slot) in lengths.iter_mut().enumerate().take(20) {
            *slot = id % 4 + 1;
        }
        let ids_of_length = |target: usize| -> Vec<u32> {
            (0..20u32)
                .filter(|&id| lengths[id as usize] == target)
                .collect()
        };
        let by_length: Vec<Vec<u32>> = (1..=4).map(ids_of_length).collect();

        let mut rng = Lcg(0x5eed_1234_abcd_ef01);
        for round in 0..4000 {
            // Build a text: pure ASCII half the time, mixed otherwise.
            let ascii_only = round % 2 == 0;
            let char_count = 1 + rng.below(60);
            let mut text = String::new();
            for _ in 0..char_count {
                let value = if ascii_only {
                    pool[rng.below(4)]
                } else {
                    pool[rng.below(pool.len())]
                };
                text.push(value);
            }
            // Build a closing ID row over the text's bytes.
            let mut ids = Vec::new();
            let mut remaining = text.len();
            while remaining > 0 {
                let width = 1 + rng.below(4.min(remaining));
                let candidates = &by_length[width - 1];
                ids.push(candidates[rng.below(candidates.len())]);
                remaining -= width;
            }
            // Optionally corrupt the row or the text.
            match rng.below(8) {
                0 => {
                    ids.pop();
                }
                1 => ids.push(by_length[0][0]),
                2 => {
                    let at = rng.below(ids.len().max(1)).min(ids.len().saturating_sub(1));
                    if !ids.is_empty() {
                        ids[at] = 21 + rng.below(4) as u32; // unknown
                    }
                }
                3 => {
                    let at = rng.below(ids.len().max(1)).min(ids.len().saturating_sub(1));
                    if !ids.is_empty() {
                        ids[at] = 20; // zero-byte entry
                    }
                }
                4 => {
                    text.pop();
                }
                5 => text.push('x'),
                _ => {}
            }
            assert_dual_path(&ids, &lengths, &text);
        }
    }
}
