//! Pinned Hugging Face reference engine used by the native request path.

use std::collections::HashSet;
use std::fmt;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use aho_corasick::{AhoCorasick, MatchKind};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokenizers::{NormalizedString, Normalizer, OffsetReferential, OffsetType, Tokenizer};
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
    /// Facts of the face document this engine executes, read once at load
    /// for the added-token literal class.
    document: Arc<DocumentFacts>,
    /// The literal class, built (with its self-check) on first request by
    /// a router or corrected CPU engine constructor.
    literal_class: Arc<OnceLock<LiteralClass>>,
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
        let parsed = serde_json::from_slice::<Value>(bytes).ok();
        let normalizer_is_identity = parsed
            .as_ref()
            .and_then(|document| document.get("normalizer").cloned())
            .is_none_or(|normalizer| is_identity_normalizer(&normalizer));
        let document = Arc::new(DocumentFacts::read(parsed.as_ref()));
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(|error| {
            ReferenceEngineError::new(format!("failed to load tokenizer JSON: {error}"))
        })?;
        Ok(Self {
            tokenizer,
            normalizer_is_identity,
            artifact_sha256,
            raw_byte_lengths: Arc::new(OnceLock::new()),
            document,
            literal_class: Arc::new(OnceLock::new()),
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

    /// Require that `text` is unchanged by this face's normalizer.
    ///
    /// The span bridges place a row's tokens over `text` by their raw byte
    /// lengths. That placement is the row's own only when the row encodes
    /// `text` itself; a row of the normalized text can still close over
    /// `text` in bytes when the normalizer's length changes cancel, and its
    /// spans would then be shifted between the first and the last changed
    /// character. A caller that adopts a row it did not encode itself from
    /// verified-stable text runs this check beside the closure check. ASCII
    /// text is stable under an NFC normalizer and is not normalized here.
    pub fn require_normalization_stable(&self, text: &str) -> Result<(), ReferenceEngineError> {
        if !self.has_effective_normalizer()
            || (text.is_ascii() && matches!(self.document.shape, Ok(NormalizerShape::Nfc)))
        {
            return Ok(());
        }
        if self.normalize(text)? != text {
            return Err(ReferenceEngineError::new(
                "the text is not stable under the artifact normalizer, so token bytes \
                 do not locate its characters",
            ));
        }
        Ok(())
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

    /// The added-token literal class of this face, built on first use.
    ///
    /// Routers and corrected CPU engines ask for it in their constructors,
    /// so the class predicate and the segmenter self-check run at
    /// construction time rather than inside the first request.
    pub(crate) fn literal_class(&self) -> &LiteralClass {
        self.literal_class
            .get_or_init(|| LiteralClass::build(&self.tokenizer, &self.document, self))
    }

    /// Whether the literal class has been built (tests only).
    #[cfg(test)]
    pub(crate) fn literal_class_built(&self) -> bool {
        self.literal_class.get().is_some()
    }

    /// The plain-literal segmenter when this face is in the served class
    /// otherwise `None`.
    pub(crate) fn literal_segmenter(&self) -> Option<&LiteralSegmenter> {
        self.literal_class().segmenter.as_ref().ok()
    }

    /// Why this face is not served by literal segmentation, when it is not.
    pub fn literal_class_reason(&self) -> Option<&str> {
        self.literal_class()
            .segmenter
            .as_ref()
            .err()
            .map(String::as_str)
    }

    /// Whether the face is served by plain-literal segmentation.
    pub fn literal_segmentation_served(&self) -> bool {
        self.literal_segmenter().is_some()
    }

    /// Whether the corrected CPU engine's own one-pass added-token matcher
    /// agrees with the reference's two extraction passes on every input
    /// When it does not, no literal-bearing text may reach
    /// that matcher, so session repair is disabled for the face.
    pub fn engine_matcher_agrees(&self) -> bool {
        self.literal_class().engine_matcher_agrees
    }
}

/// Whether a serialized normalizer is the identity function: absent, `null`,
/// or an empty `Sequence`.
fn is_identity_normalizer(normalizer: &Value) -> bool {
    normalizer.is_null()
        || normalizer.as_object().is_some_and(|value| {
            value.get("type").and_then(Value::as_str) == Some("Sequence")
                && value
                    .get("normalizers")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty)
        })
}

/// One `added_tokens` row of the face document, as written.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DocumentRow {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

/// How the face normalizes text, within the served class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizerShape {
    Identity,
    Nfc,
}

/// Facts read once from the face document: its added-token rows and
/// whether the rest of the pipeline is in the shape under which the
/// reference handles every text split independently of its neighbours
/// (the post-processor included).
#[derive(Debug, Clone)]
pub(crate) struct DocumentFacts {
    rows: Result<Vec<DocumentRow>, String>,
    shape: Result<NormalizerShape, String>,
}

impl DocumentFacts {
    fn read(document: Option<&Value>) -> Self {
        match document {
            None => Self {
                rows: Err("the face document could not be read".to_owned()),
                shape: Err("the face document could not be read".to_owned()),
            },
            Some(document) => Self {
                rows: document_rows(document),
                shape: face_shape(document),
            },
        }
    }
}

fn document_rows(document: &Value) -> Result<Vec<DocumentRow>, String> {
    let Some(rows) = document.get("added_tokens") else {
        return Ok(Vec::new());
    };
    if rows.is_null() {
        return Ok(Vec::new());
    }
    let rows = rows
        .as_array()
        .ok_or_else(|| "added_tokens is not a list".to_owned())?;
    rows.iter()
        .map(|row| {
            let flag = |name: &str| {
                row.get(name)
                    .and_then(Value::as_bool)
                    .ok_or_else(|| format!("an added_tokens row has no boolean {name:?}"))
            };
            let id = row
                .get("id")
                .and_then(Value::as_u64)
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| "an added_tokens row has no u32 id".to_owned())?;
            let content = row
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| "an added_tokens row has no content".to_owned())?
                .to_owned();
            Ok(DocumentRow {
                id,
                content,
                single_word: flag("single_word")?,
                lstrip: flag("lstrip")?,
                rstrip: flag("rstrip")?,
                normalized: flag("normalized")?,
                special: flag("special")?,
            })
        })
        .collect()
}

fn type_of(value: &Value) -> Option<&str> {
    value.get("type").and_then(Value::as_str)
}

fn absent_or_null(document: &Value, key: &str) -> bool {
    document.get(key).is_none_or(Value::is_null)
}

fn face_shape(document: &Value) -> Result<NormalizerShape, String> {
    let shape = match document.get("normalizer") {
        None => NormalizerShape::Identity,
        Some(normalizer) if is_identity_normalizer(normalizer) => NormalizerShape::Identity,
        Some(normalizer) if type_of(normalizer) == Some("NFC") => NormalizerShape::Nfc,
        Some(normalizer) => {
            return Err(format!(
                "normalizer {:?} is outside the served class",
                type_of(normalizer).unwrap_or("unknown")
            ))
        }
    };
    let pre_tokenizer = document
        .get("pre_tokenizer")
        .filter(|value| !value.is_null())
        .ok_or_else(|| "the face has no pre-tokenizer".to_owned())?;
    if !pre_tokenizer_is_local(pre_tokenizer) {
        return Err("the pre-tokenizer is outside ByteLevel / Split / Sequence".to_owned());
    }
    let model = document
        .get("model")
        .ok_or_else(|| "the face has no model".to_owned())?;
    if type_of(model) != Some("BPE") {
        return Err("the model is not BPE".to_owned());
    }
    let dropout_off = match model.get("dropout") {
        None | Some(Value::Null) => true,
        Some(value) => value.as_f64() == Some(0.0),
    };
    if !dropout_off {
        return Err("BPE dropout is active".to_owned());
    }
    if !absent_or_null(document, "truncation") || !absent_or_null(document, "padding") {
        return Err("the face document sets truncation or padding".to_owned());
    }
    match document.get("post_processor") {
        None | Some(Value::Null) => {}
        Some(processor) if post_processor_keeps_core(processor) => {}
        Some(processor) => {
            return Err(format!(
                "post-processor {:?} is outside the identity / single-prefix kinds",
                type_of(processor).unwrap_or("unknown")
            ))
        }
    }
    Ok(shape)
}

fn pre_tokenizer_is_local(value: &Value) -> bool {
    match type_of(value) {
        Some("ByteLevel") | Some("Split") => true,
        Some("Sequence") => value
            .get("pretokenizers")
            .and_then(Value::as_array)
            .is_some_and(|members| members.iter().all(pre_tokenizer_is_local)),
        _ => false,
    }
}

/// A post-processor whose output with `add_special_tokens = false` is the
/// core stream itself: `ByteLevel`, a single-sequence template with at most
/// one leading special-token piece, or a `Sequence` whose members other
/// than at most one such template are `ByteLevel`.
fn post_processor_keeps_core(value: &Value) -> bool {
    match type_of(value) {
        Some("ByteLevel") => true,
        Some("TemplateProcessing") => template_keeps_core(value),
        Some("Sequence") => {
            let Some(members) = value.get("processors").and_then(Value::as_array) else {
                return false;
            };
            let mut templates = 0usize;
            members.iter().all(|member| match type_of(member) {
                Some("ByteLevel") => true,
                Some("TemplateProcessing") => {
                    templates += 1;
                    templates == 1 && template_keeps_core(member)
                }
                _ => false,
            })
        }
        _ => false,
    }
}

fn template_keeps_core(value: &Value) -> bool {
    let Some(single) = value.get("single").and_then(Value::as_array) else {
        return false;
    };
    let is_sequence_a = |piece: &Value| {
        piece
            .get("Sequence")
            .and_then(|sequence| sequence.get("id"))
            .and_then(Value::as_str)
            == Some("A")
    };
    match single.as_slice() {
        [only] => is_sequence_a(only),
        [first, second] => {
            let Some(key) = first
                .get("SpecialToken")
                .and_then(|piece| piece.get("id"))
                .and_then(Value::as_str)
            else {
                return false;
            };
            let prefix_is_one_id = value
                .get("special_tokens")
                .and_then(|table| table.get(key))
                .and_then(|entry| entry.get("ids"))
                .and_then(Value::as_array)
                .is_some_and(|ids| {
                    ids.len() == 1 && ids[0].as_u64().is_some_and(|id| id <= u64::from(u32::MAX))
                });
            prefix_is_one_id && is_sequence_a(second)
        }
        _ => false,
    }
}

/// One piece of a literal segmentation plan, in UTF-8 byte offsets of the
/// input: an added-token literal with the id the reference emits for it,
/// or a non-empty text segment holding no literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiteralPiece {
    Literal { id: u32, start: usize, end: usize },
    Text { start: usize, end: usize },
}

/// The native two-phase leftmost-longest segmenter.
///
/// Phase A runs over the raw text with the `normalized: false` literals;
/// phase B runs inside each phase-A gap with the `normalized: true`
/// literals and exists only when the face normalizer is the identity.
/// Both phases use the same matcher crate and match kind as the reference
/// library's own extraction, over the same two pattern sets.
#[derive(Debug)]
pub(crate) struct LiteralSegmenter {
    phase_a: Option<(AhoCorasick, Vec<u32>)>,
    phase_b: Option<(AhoCorasick, Vec<u32>)>,
}

fn leftmost_longest(patterns: &[&str]) -> Result<AhoCorasick, String> {
    AhoCorasick::builder()
        .match_kind(MatchKind::LeftmostLongest)
        .build(patterns)
        .map_err(|error| format!("failed to build the literal segmenter: {error}"))
}

impl LiteralSegmenter {
    fn new(rows: &[DocumentRow]) -> Result<Self, String> {
        let phase = |normalized: bool| -> Result<Option<(AhoCorasick, Vec<u32>)>, String> {
            let selected = rows
                .iter()
                .filter(|row| row.normalized == normalized)
                .collect::<Vec<_>>();
            if selected.is_empty() {
                return Ok(None);
            }
            let patterns = selected
                .iter()
                .map(|row| row.content.as_str())
                .collect::<Vec<_>>();
            Ok(Some((
                leftmost_longest(&patterns)?,
                selected.iter().map(|row| row.id).collect(),
            )))
        };
        Ok(Self {
            phase_a: phase(false)?,
            phase_b: phase(true)?,
        })
    }

    /// The segmentation plan of `text`. Text pieces are never empty; the
    /// concatenation of all pieces is `text`.
    pub(crate) fn plan(&self, text: &str) -> Vec<LiteralPiece> {
        let mut pieces = Vec::new();
        let mut cursor = 0usize;
        if let Some((matcher, ids)) = &self.phase_a {
            for found in matcher.find_iter(text) {
                self.phase_b_gap(text, cursor, found.start(), &mut pieces);
                pieces.push(LiteralPiece::Literal {
                    id: ids[found.pattern().as_usize()],
                    start: found.start(),
                    end: found.end(),
                });
                cursor = found.end();
            }
        }
        self.phase_b_gap(text, cursor, text.len(), &mut pieces);
        pieces
    }

    fn phase_b_gap(&self, text: &str, start: usize, end: usize, pieces: &mut Vec<LiteralPiece>) {
        let mut cursor = start;
        if let Some((matcher, ids)) = &self.phase_b {
            for found in matcher.find_iter(&text[start..end]) {
                let literal_start = start + found.start();
                if cursor < literal_start {
                    pieces.push(LiteralPiece::Text {
                        start: cursor,
                        end: literal_start,
                    });
                }
                pieces.push(LiteralPiece::Literal {
                    id: ids[found.pattern().as_usize()],
                    start: literal_start,
                    end: start + found.end(),
                });
                cursor = start + found.end();
            }
        }
        if cursor < end {
            pieces.push(LiteralPiece::Text { start: cursor, end });
        }
    }
}

/// The literal class of one face: the served segmenter or the reason the
/// face keeps the whole-document reference path, plus the agreement
/// predicate of the engine core's one-pass matcher.
#[derive(Debug)]
pub(crate) struct LiteralClass {
    segmenter: Result<LiteralSegmenter, String>,
    engine_matcher_agrees: bool,
}

impl LiteralClass {
    fn build(tokenizer: &Tokenizer, facts: &DocumentFacts, engine: &ReferenceEngine) -> Self {
        let rows = facts
            .rows
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|rows| rows_equal_live_table(tokenizer, rows).map(|()| rows));
        let engine_matcher_agrees = rows
            .as_ref()
            .is_ok_and(|rows| matcher_agreement(rows, engine.has_effective_normalizer()));
        // A self-check difference shows that the native reading of this
        // face's literals does not hold; the engine core's matcher must
        // not be trusted over literal-bearing repair windows either.
        let mut self_check_failed = false;
        let segmenter = rows.and_then(|rows| {
            if rows.is_empty() {
                return Err("the face has no added tokens".to_owned());
            }
            if tokenizer.get_encode_special_tokens() {
                return Err("the reference is set to split special tokens".to_owned());
            }
            if let Some(row) = rows
                .iter()
                .find(|row| row.single_word || row.lstrip || row.rstrip)
            {
                return Err(format!(
                    "added token {:?} sets single_word, lstrip or rstrip",
                    row.content
                ));
            }
            let shape = facts.shape.clone()?;
            if shape == NormalizerShape::Nfc && rows.iter().any(|row| row.normalized) {
                return Err("a normalized added token under an active normalizer".to_owned());
            }
            let segmenter = LiteralSegmenter::new(rows)?;
            segmenter_self_check(engine, tokenizer, &segmenter, rows)
                .inspect_err(|_| self_check_failed = true)?;
            Ok(segmenter)
        });
        Self {
            segmenter,
            engine_matcher_agrees: engine_matcher_agrees && !self_check_failed,
        }
    }
}

/// The document's rows must equal the live added-token
/// table in count, id, content, and flags, every content must be non-empty
/// and distinct, and the live id of each content must be its row id.
fn rows_equal_live_table(tokenizer: &Tokenizer, rows: &[DocumentRow]) -> Result<(), String> {
    let live = tokenizer.get_added_tokens_decoder();
    if live.len() != rows.len() {
        return Err(format!(
            "the document lists {} added tokens and the live table {}",
            rows.len(),
            live.len()
        ));
    }
    let mut contents = HashSet::with_capacity(rows.len());
    for row in rows {
        if row.content.is_empty() {
            return Err("an added token has empty content".to_owned());
        }
        if !contents.insert(row.content.as_str()) {
            return Err(format!("added token content {:?} is repeated", row.content));
        }
        let matches = live.get(&row.id).is_some_and(|token| {
            token.content == row.content
                && token.single_word == row.single_word
                && token.lstrip == row.lstrip
                && token.rstrip == row.rstrip
                && token.normalized == row.normalized
                && token.special == row.special
        });
        if !matches || tokenizer.token_to_id(&row.content) != Some(row.id) {
            return Err(format!(
                "added token {:?} differs between the document and the live table",
                row.content
            ));
        }
    }
    Ok(())
}

/// The engine core's one leftmost-longest pass over every
/// content (emitting document ids) equals the reference's two passes on
/// every input exactly when no flag is set, no `normalized: true` content
/// sits under an effective normalizer, and no pair (A `normalized: false`,
/// B `normalized: true`) has A inside B or a proper suffix of B equal to a
/// proper prefix of A. The caller has already required document ids to
/// equal live ids and contents to be distinct and non-empty.
fn matcher_agreement(rows: &[DocumentRow], effective_normalizer: bool) -> bool {
    if rows
        .iter()
        .any(|row| row.single_word || row.lstrip || row.rstrip)
    {
        return false;
    }
    if effective_normalizer && rows.iter().any(|row| row.normalized) {
        return false;
    }
    let plain = rows
        .iter()
        .filter(|row| !row.normalized)
        .collect::<Vec<_>>();
    rows.iter().filter(|row| row.normalized).all(|later| {
        plain.iter().all(|earlier| {
            !later.content.contains(earlier.content.as_str())
                && !suffix_meets_prefix(&later.content, &earlier.content)
        })
    })
}

/// Whether a proper suffix of `left` equals a proper prefix of `right`.
fn suffix_meets_prefix(left: &str, right: &str) -> bool {
    left.char_indices().skip(1).any(|(start, _)| {
        let suffix = &left[start..];
        suffix.len() < right.len() && right.starts_with(suffix)
    })
}

const SELF_CHECK_NEIGHBOURS: [&str; 11] = [
    "a", " ", "\n", "1", ".", "<", ">", "|", "\u{e9}", "\u{4e2d}", "\u{301}",
];
const SELF_CHECK_MERGED_BOUND: usize = 100_000;

/// Compare the native plan with the reference library's
/// own extraction on a probe set built from the face's table -- each
/// literal alone, doubled, truncated by one character, with one-character
/// neighbours, and each overlapping pair merged -- and the composed ids
/// with the reference's whole-input ids on a subset. Any difference
/// leaves the face unserved.
fn segmenter_self_check(
    engine: &ReferenceEngine,
    tokenizer: &Tokenizer,
    segmenter: &LiteralSegmenter,
    rows: &[DocumentRow],
) -> Result<(), String> {
    #[cfg(test)]
    if test_support::SELF_CHECK_DIFFERS.with(std::cell::Cell::get) {
        return Err("the segmenter self-check differs (injected by a test)".to_owned());
    }
    let mut split_probes = Vec::new();
    let mut id_probes = Vec::new();
    for row in rows {
        let literal = row.content.as_str();
        split_probes.push(literal.to_owned());
        split_probes.push(format!("{literal}{literal}"));
        for neighbour in SELF_CHECK_NEIGHBOURS {
            split_probes.push(format!("{neighbour}{literal}"));
            split_probes.push(format!("{literal}{neighbour}"));
            split_probes.push(format!("{neighbour}{literal}{neighbour}"));
        }
        if let Some((last, _)) = literal.char_indices().next_back() {
            if last > 0 {
                split_probes.push(format!("a{}b", &literal[..last]));
            }
        }
        if let Some((_, first)) = literal.char_indices().next() {
            if first.len_utf8() < literal.len() {
                split_probes.push(format!("a{}b", &literal[first.len_utf8()..]));
            }
        }
        id_probes.push(literal.to_owned());
        id_probes.push(format!("a{literal} b"));
        id_probes.push(format!(" {literal}\n"));
        id_probes.push(format!("\u{e9}{literal}\u{4e2d}"));
    }
    let mut prefixes = std::collections::HashMap::<&str, Vec<usize>>::new();
    for (index, row) in rows.iter().enumerate() {
        for (end, _) in row.content.char_indices().skip(1) {
            prefixes.entry(&row.content[..end]).or_default().push(index);
        }
    }
    let mut merged = 0usize;
    for row in rows {
        for (start, _) in row.content.char_indices().skip(1) {
            let suffix = &row.content[start..];
            for &other in prefixes.get(suffix).map(Vec::as_slice).unwrap_or(&[]) {
                merged += 1;
                if merged > SELF_CHECK_MERGED_BOUND {
                    return Err("the self-check probe set exceeds its bound".to_owned());
                }
                let probe = format!("{}{}", row.content, &rows[other].content[suffix.len()..]);
                split_probes.push(probe.clone());
                id_probes.push(probe);
            }
        }
    }
    for probe in &split_probes {
        let native = segmenter
            .plan(probe)
            .into_iter()
            .filter_map(|piece| match piece {
                LiteralPiece::Literal { id, start, end } => Some((id, start, end)),
                LiteralPiece::Text { .. } => None,
            })
            .collect::<Vec<_>>();
        let reference = reference_literal_splits(tokenizer, probe);
        if native != reference {
            return Err(format!(
                "the segmenter self-check differs from the reference extraction on {probe:?}"
            ));
        }
    }
    for probe in &id_probes {
        let expected = engine
            .encode_ids(probe, false)
            .map_err(|error| error.to_string())?;
        let mut composed = Vec::new();
        for piece in segmenter.plan(probe) {
            match piece {
                LiteralPiece::Literal { id, .. } => composed.push(id),
                LiteralPiece::Text { start, end } => composed.extend(
                    engine
                        .encode_ids(&probe[start..end], false)
                        .map_err(|error| error.to_string())?,
                ),
            }
        }
        if composed != expected {
            return Err(format!(
                "the composed segment ids differ from the reference on {probe:?}"
            ));
        }
    }
    Ok(())
}

/// The token-carrying splits of the reference library's own extraction:
/// (id, start byte, end byte) in the original text, in order.
fn reference_literal_splits(tokenizer: &Tokenizer, text: &str) -> Vec<(u32, usize, usize)> {
    tokenizer
        .get_added_vocabulary()
        .extract_and_normalize(tokenizer.get_normalizer(), text)
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .filter_map(|(_, (start, end), tokens)| {
            tokens
                .as_ref()
                .and_then(|tokens| tokens.first())
                .map(|token| (token.id, start, end))
        })
        .collect()
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
            document: Arc::new(DocumentFacts::read(None)),
            literal_class: Arc::new(OnceLock::new()),
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

/// Small byte-level faces for the routing-core unit tests of the literal
/// class: one token per byte, no merges, plus the given added tokens.
#[cfg(test)]
pub(crate) mod test_support {
    use serde_json::Value;
    use tokenizers::models::bpe::{Vocab, BPE};
    use tokenizers::normalizers::unicode::NFC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, Tokenizer};

    thread_local! {
        /// Makes the segmenter self-check report a difference on this
        /// thread, for the faces whose class is built while it is set.
        pub(crate) static SELF_CHECK_DIFFERS: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    /// Normalizer of a test face.
    #[derive(Clone, Copy)]
    pub(crate) enum TestNormalizer {
        None,
        EmptySequence,
        Nfc,
    }

    pub(crate) fn face_json(normalizer: TestNormalizer, tokens: &[AddedToken]) -> Vec<u8> {
        face_json_edited(normalizer, tokens, |_| {})
    }

    pub(crate) fn face_json_edited(
        normalizer: TestNormalizer,
        tokens: &[AddedToken],
        edit: impl FnOnce(&mut Value),
    ) -> Vec<u8> {
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
        match normalizer {
            TestNormalizer::None => {}
            TestNormalizer::EmptySequence => {
                tokenizer.with_normalizer(Some(tokenizers::normalizers::Sequence::new(vec![])));
            }
            TestNormalizer::Nfc => {
                tokenizer.with_normalizer(Some(NFC));
            }
        }
        if !tokens.is_empty() {
            tokenizer.add_special_tokens(tokens);
        }
        let mut document: Value =
            serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap();
        edit(&mut document);
        serde_json::to_vec(&document).unwrap()
    }

    /// A plain added token (all three flags false).
    pub(crate) fn plain(content: &str, normalized: bool) -> AddedToken {
        AddedToken::from(content.to_owned(), true).normalized(normalized)
    }

    /// Deterministic strings dense in the given literals, their fragments,
    /// and a few neighbour classes.
    pub(crate) fn literal_dense_strings(literals: &[&str], count: usize, seed: u64) -> Vec<String> {
        let mut fragments = vec![
            "a",
            " ",
            "\n",
            "1",
            ".",
            "<",
            ">",
            "|",
            "x",
            "y",
            "z",
            "q",
            "k",
            "\u{e9}",
            "e\u{301}",
            "\u{4e2d}",
            "\u{1f642}",
            "\u{200d}",
            "\u{1100}\u{1161}",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        for literal in literals {
            fragments.push((*literal).to_owned());
            let chars = literal.chars().collect::<Vec<_>>();
            for cut in 1..chars.len() {
                fragments.push(chars[..cut].iter().collect());
                fragments.push(chars[cut..].iter().collect());
            }
        }
        let mut state = seed;
        let mut next = move |bound: usize| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound
        };
        (0..count)
            .map(|_| {
                let pieces = 1 + next(12);
                (0..pieces)
                    .map(|_| fragments[next(fragments.len())].clone())
                    .collect::<String>()
            })
            .collect()
    }
}

#[cfg(test)]
mod literal_class_tests {
    use super::test_support::TestNormalizer;
    use super::test_support::{face_json, face_json_edited, literal_dense_strings, plain};
    use super::*;
    use tokenizers::AddedToken;

    type Edit = Box<dyn FnOnce(&mut Value)>;
    /// (name, normalizer, tokens, document edit, specimen text, reason).
    type Specimen = (
        &'static str,
        TestNormalizer,
        Vec<AddedToken>,
        Edit,
        &'static str,
        &'static str,
    );

    fn load(json: &[u8]) -> ReferenceEngine {
        ReferenceEngine::from_bytes(json).unwrap()
    }

    fn composed(engine: &ReferenceEngine, text: &str) -> Vec<u32> {
        let segmenter = engine.literal_segmenter().expect("served face");
        let mut ids = Vec::new();
        for piece in segmenter.plan(text) {
            match piece {
                LiteralPiece::Literal { id, .. } => ids.push(id),
                LiteralPiece::Text { start, end } => {
                    ids.extend(engine.encode_ids(&text[start..end], false).unwrap())
                }
            }
        }
        ids
    }

    fn plan_literals(engine: &ReferenceEngine, text: &str) -> Vec<(u32, usize, usize)> {
        engine
            .literal_segmenter()
            .unwrap()
            .plan(text)
            .into_iter()
            .filter_map(|piece| match piece {
                LiteralPiece::Literal { id, start, end } => Some((id, start, end)),
                LiteralPiece::Text { .. } => None,
            })
            .collect()
    }

    /// The plan covers the text exactly, text pieces are non-empty, and
    /// literals carry the reference's live id.
    fn assert_plan_shape(engine: &ReferenceEngine, text: &str) {
        let mut cursor = 0usize;
        for piece in engine.literal_segmenter().unwrap().plan(text) {
            match piece {
                LiteralPiece::Literal { id, start, end } => {
                    assert_eq!(start, cursor);
                    assert_eq!(engine.tokenizer.token_to_id(&text[start..end]), Some(id));
                    cursor = end;
                }
                LiteralPiece::Text { start, end } => {
                    assert_eq!(start, cursor);
                    assert!(end > start, "empty text piece in {text:?}");
                    cursor = end;
                }
            }
        }
        assert_eq!(cursor, text.len());
    }

    /// The native plan equals the reference library's own extraction and
    /// the composed ids equal the whole-input reference ids on
    /// literal-dense strings.
    fn assert_claim_c(engine: &ReferenceEngine, literals: &[&str], count: usize, seed: u64) {
        assert!(
            engine.literal_segmentation_served(),
            "{:?}",
            engine.literal_class_reason()
        );
        for text in literal_dense_strings(literals, count, seed) {
            assert_plan_shape(engine, &text);
            assert_eq!(
                plan_literals(engine, &text),
                reference_literal_splits(&engine.tokenizer, &text),
                "split of {text:?}"
            );
            assert_eq!(
                composed(engine, &text),
                engine.encode_ids(&text, false).unwrap(),
                "ids of {text:?}"
            );
        }
    }

    #[test]
    fn plain_literal_faces_are_served_and_match_the_reference_split() {
        let literals = [
            "<a>",
            "<ab>",
            "b<",
            "<|im_start|>",
            "<|im_end|>",
            "ab",
            "\u{e9}t\u{e9}",
            "\u{4e2d}\u{6587}",
        ];
        for normalizer in [
            TestNormalizer::None,
            TestNormalizer::EmptySequence,
            TestNormalizer::Nfc,
        ] {
            let tokens = literals
                .iter()
                .map(|literal| plain(literal, false))
                .collect::<Vec<_>>();
            let engine = load(&face_json(normalizer, &tokens));
            assert!(engine.engine_matcher_agrees());
            assert_claim_c(&engine, &literals, 3_000, 7);
        }
    }

    /// Cross-phase overlap faces (no shipped face has one). Two-phase
    /// segmentation stays exact on them, but the engine core's one-pass
    /// matcher would not, so the agreement predicate is false.
    #[test]
    fn cross_phase_overlap_faces_are_served_with_the_engine_matcher_disagreeing() {
        let cases: [(&[(&str, bool)], bool); 4] = [
            // A normalized:false content inside a normalized:true one.
            (&[("<zq>", false), ("x<zq>y", true)], false),
            // A proper suffix of the normalized:true content equals a
            // proper prefix of the normalized:false one.
            (&[("<zq>", false), ("kk<z", true)], false),
            // The harmless direction (the shipped deepseek_v4_flash shape).
            (
                &[
                    ("<\u{ff5c}/table>\u{ff5c}", false),
                    ("\u{ff5c}DSML\u{ff5c}", true),
                ],
                true,
            ),
            // Same-phase overlaps only.
            (&[("<zq>", true), ("zq>w", true), ("#", false)], true),
        ];
        for (rows, agrees) in cases {
            let tokens = rows
                .iter()
                .map(|(content, normalized)| plain(content, *normalized))
                .collect::<Vec<_>>();
            let literals = rows.iter().map(|(content, _)| *content).collect::<Vec<_>>();
            for normalizer in [TestNormalizer::None, TestNormalizer::EmptySequence] {
                let engine = load(&face_json(normalizer, &tokens));
                assert_eq!(engine.engine_matcher_agrees(), agrees, "{rows:?}");
                assert_claim_c(&engine, &literals, 3_000, 11);
            }
        }
    }

    #[test]
    fn an_overlap_across_phases_is_exact_under_two_phase_segmentation() {
        let engine = load(&face_json(
            TestNormalizer::EmptySequence,
            &[plain("<zq>", false), plain("x<zq>y", true)],
        ));
        let text = format!("{} note x<zq>y end kk<zq>", "lorem ipsum ".repeat(150));
        assert_eq!(
            composed(&engine, &text),
            engine.encode_ids(&text, false).unwrap()
        );
        // The reference extracts `<zq>` first, so `x<zq>y` never matches.
        let ids = plan_literals(&engine, &text);
        assert_eq!(ids.len(), 2);
        assert!(ids
            .iter()
            .all(|(id, _, _)| Some(*id) == engine.tokenizer.token_to_id("<zq>")));
    }

    fn template(single: Value) -> Value {
        serde_json::json!({
            "type": "TemplateProcessing",
            "single": single,
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [300], "tokens": ["<s>"]}},
        })
    }

    /// The refusal classes of the literal class and two synthetic
    /// post-processor shapes: each specimen shows that a plain
    /// plan would be wrong there, and the face stays unserved.
    #[test]
    fn refusal_specimens_keep_the_face_unserved() {
        let x = |token: AddedToken| vec![token];
        let flagged = |token: AddedToken| token.normalized(false);
        let no_edit = || -> Edit { Box::new(|_| {}) };
        let specimens: Vec<Specimen> = vec![
            (
                "lstrip",
                TestNormalizer::None,
                x(flagged(AddedToken::from("x", true).lstrip(true))),
                no_edit(),
                "\nx",
                "sets single_word, lstrip or rstrip",
            ),
            (
                "rstrip",
                TestNormalizer::None,
                x(flagged(AddedToken::from("x", true).rstrip(true))),
                no_edit(),
                "x\n",
                "sets single_word, lstrip or rstrip",
            ),
            (
                "single_word",
                TestNormalizer::None,
                x(flagged(AddedToken::from("xq", true).single_word(true))),
                no_edit(),
                "axq",
                "sets single_word, lstrip or rstrip",
            ),
            (
                "normalized under NFC",
                TestNormalizer::Nfc,
                x(plain("e", true)),
                no_edit(),
                "e\u{301}",
                "a normalized added token under an active normalizer",
            ),
            (
                "template repeats the sequence",
                TestNormalizer::None,
                x(plain("x", false)),
                Box::new(|document: &mut Value| {
                    document["post_processor"] = template(serde_json::json!([
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}}
                    ]));
                }),
                "hi x there",
                "post-processor",
            ),
            (
                "template drops the sequence",
                TestNormalizer::None,
                x(plain("x", false)),
                Box::new(|document: &mut Value| {
                    document["post_processor"] = template(serde_json::json!([]));
                }),
                "x",
                "post-processor",
            ),
            (
                "dropout",
                TestNormalizer::None,
                x(plain("x", false)),
                Box::new(|document: &mut Value| {
                    document["model"]["dropout"] = serde_json::json!(0.5);
                }),
                "",
                "dropout",
            ),
            (
                "lowercase normalizer",
                TestNormalizer::None,
                x(plain("x", false)),
                Box::new(|document: &mut Value| {
                    document["normalizer"] = serde_json::json!({"type": "Lowercase"});
                }),
                "",
                "normalizer",
            ),
        ];
        for (name, normalizer, tokens, edit, text, reason) in specimens {
            let json = face_json_edited(normalizer, &tokens, edit);
            let engine = load(&json);
            assert!(!engine.literal_segmentation_served(), "{name}");
            let observed = engine.literal_class_reason().unwrap();
            assert!(observed.contains(reason), "{name}: {observed}");
            if text.is_empty() {
                continue;
            }
            // What a plain two-phase plan would answer: the literal's live
            // id wherever its content occurs, each gap encoded alone.
            let literal = tokens[0].content.as_str();
            let literal_id = engine.tokenizer.token_to_id(literal).unwrap();
            let mut naive = Vec::new();
            for (index, gap) in text.split(literal).enumerate() {
                if index > 0 {
                    naive.push(literal_id);
                }
                if !gap.is_empty() {
                    naive.extend(engine.encode_ids(gap, false).unwrap());
                }
            }
            assert_ne!(naive, engine.encode_ids(text, false).unwrap(), "{name}");
        }
    }

    #[test]
    fn flagged_faces_disable_the_engine_matcher_and_shipped_shapes_do_not() {
        let flagged = load(&face_json(
            TestNormalizer::None,
            &[AddedToken::from("x", true)
                .single_word(true)
                .normalized(false)],
        ));
        assert!(!flagged.engine_matcher_agrees());
        let nfc_normalized = load(&face_json(TestNormalizer::Nfc, &[plain("e", true)]));
        assert!(!nfc_normalized.engine_matcher_agrees());
        let plain_face = load(&face_json(
            TestNormalizer::Nfc,
            &[plain("<|im_start|>", false)],
        ));
        assert!(plain_face.engine_matcher_agrees());
        assert!(plain_face.literal_segmentation_served());
    }

    /// A segmenter self-check difference leaves the face unserved, names
    /// itself as the reason, and also clears the engine matcher's
    /// agreement, so session repair stays off on that face.
    #[test]
    fn a_self_check_difference_also_turns_the_engine_matcher_off() {
        let json = face_json(TestNormalizer::Nfc, &[plain("<|im_start|>", false)]);
        assert!(load(&json).engine_matcher_agrees());
        test_support::SELF_CHECK_DIFFERS.with(|flag| flag.set(true));
        let engine = load(&json);
        let served = engine.literal_segmentation_served();
        let reason = engine.literal_class_reason().map(str::to_owned);
        let agrees = engine.engine_matcher_agrees();
        test_support::SELF_CHECK_DIFFERS.with(|flag| flag.set(false));
        assert!(!served);
        assert!(reason.unwrap().contains("self-check differs"));
        assert!(!agrees);
    }

    /// `dropout` 0.0 is no dropout (the shipped `laguna_s_2_1` shape),
    /// and the single-prefix template kinds keep the face served.
    #[test]
    fn zero_dropout_and_single_prefix_templates_are_served() {
        let edits: Vec<Edit> = vec![
            Box::new(|document: &mut Value| {
                document["model"]["dropout"] = serde_json::json!(0.0);
            }),
            Box::new(|document: &mut Value| {
                document["post_processor"] = template(serde_json::json!([
                    {"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}
                ]));
            }),
            Box::new(|document: &mut Value| {
                document["post_processor"] = serde_json::json!({
                    "type": "Sequence",
                    "processors": [
                        {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
                        template(serde_json::json!([
                            {"SpecialToken": {"id": "<s>", "type_id": 0}},
                            {"Sequence": {"id": "A", "type_id": 0}}
                        ])),
                    ],
                });
            }),
        ];
        for edit in edits {
            let json = face_json_edited(
                TestNormalizer::None,
                &[plain("<s>", false), plain("<x>", false)],
                edit,
            );
            let engine = load(&json);
            assert_claim_c(&engine, &["<s>", "<x>"], 500, 3);
        }
        // A template after the prefix member of a `Sequence` is refused by
        // the table itself, not only by the probes.
        let json = face_json_edited(
            TestNormalizer::None,
            &[plain("<s>", false)],
            |document: &mut Value| {
                document["post_processor"] = serde_json::json!({
                    "type": "Sequence",
                    "processors": [
                        template(serde_json::json!([
                            {"SpecialToken": {"id": "<s>", "type_id": 0}},
                            {"Sequence": {"id": "A", "type_id": 0}}
                        ])),
                        template(serde_json::json!([{"Sequence": {"id": "A", "type_id": 0}}])),
                    ],
                });
            },
        );
        assert!(!load(&json).literal_segmentation_served());
    }

    /// A document row whose id differs from the id the reference
    /// assigns makes the face unserved.
    #[test]
    fn document_rows_must_equal_the_live_table() {
        let json = face_json_edited(
            TestNormalizer::None,
            &[plain("<q>", false), plain("a", false)],
            |document: &mut Value| {
                let rows = document["added_tokens"].as_array_mut().unwrap();
                for row in rows {
                    if row["content"] == "a" {
                        row["id"] = serde_json::json!(4000);
                    }
                }
            },
        );
        if let Ok(engine) = ReferenceEngine::from_bytes(&json) {
            assert!(!engine.literal_segmentation_served());
            assert!(!engine.engine_matcher_agrees());
        }
    }
}
