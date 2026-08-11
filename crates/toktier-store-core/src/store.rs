//! The session store: append-only "text to token ids" sessions with
//! certified extension, block-hash prefix sharing, and strict record
//! import/export in store format v1.
//!
//! Behavior is a faithful port of the pre-release prototype store v1 (the
//! sealing, chaining, lookup, fork, eviction and cap semantics are
//! unchanged), refactored to the release contracts:
//!
//! * no Python types anywhere -- tokenization is behind the
//!   [`SessionEncoder`] trait, session state is plain Rust;
//! * append returns a structured [`AppendOutcome`] with `replace_from`,
//!   `replacement_ids` and `all_ids`, holding the frozen invariant
//!   `all_ids == old_ids[..replace_from] + replacement_ids`;
//! * every append carries an `expected_revision`; a mismatch is
//!   [`StoreError::RevisionConflict`], last-writer-wins is not offered;
//! * session revisions are chained per the frozen format: genesis is
//!   revision 0, every committed revision has a link hash binding the
//!   fingerprint, and exported records verify against that chain;
//! * records are stamped with the engine's witness predicate category
//!   and the store refuses to mix categories within a session lineage.
//!
//! Correctness red lines (not simplified):
//! * the 32-byte semantic fingerprint participates in every chain hash
//!   and every node key, so a wrong key can never structurally hit;
//! * every lookup verifies per-node checksums and linkage; any failure
//!   is counted and treated as a miss (prefer miss over wrong);
//! * the store saves the pre-postprocessor core token stream.

use std::collections::{BTreeMap, HashMap};

use sha2::{Digest, Sha256};

use crate::engine::{
    BoundaryCut, OverlapRunner, SessionEncoder, SharedIds, SoaEncoding, WitnessCategory,
};
use crate::error::StoreError;
use crate::format::{
    link_hash, payload_digest_parts, BlockHash, LinkInputs, PayloadHasher, SessionRecordV1,
    ZERO_HASH,
};
use crate::recovery_binding::{ContentDigest, ContentIndexEntry, RecoveryBindingV1};
use crate::sidecar::{node_key, NodeCacheRecord, SessionSidecar};
use crate::tail::TailState;

/// Public name of the frozen format ("schema" in reporting surfaces).
pub const FORMAT_NAME: &str = "toktier.store.v1";

/// Domain used by the private facade recovery binding. This is not part
/// of store format v1: it lets the native session incrementally maintain
/// the full-text digest that the facade already persisted in TKFR-v1.
const RECOVERY_TEXT_DOMAIN: &[u8] = b"toktier.facade.v1.recovery-text\0";

/// The 32-byte opaque semantic fingerprint of a key.
///
/// The store never interprets these bytes; the preimage is specified in
/// `docs/contracts/fingerprint.md` and produced by the caller. The red
/// line is structural: the fingerprint participates in every chain hash
/// and node key, so lookups under a different fingerprint cannot hit.
pub type SemanticFingerprint = [u8; 32];

/// Registered key id (index into the store's fingerprint table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KeyId(pub u32);

/// Process-local session handle. Not persistent: reloading a store
/// yields fresh handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionHandle(pub u64);

// -------------------------------------------------------------- config --

/// Store configuration. Defaults follow the pre-release prototype store v1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreConfig {
    /// Chain block size in characters (Unicode scalar values).
    pub block_chars: u64,
    /// Soft cap on tail bytes: exceeding it is counted, never unsafe.
    pub tail_soft_cap_bytes: usize,
    /// Hard cap on tail bytes: an append onto a tail already above it
    /// bypasses the repair path and fully re-encodes (documented
    /// degradation; correctness kept).
    pub tail_hard_cap_bytes: usize,
    /// A completing block whose unsealed text exceeds this is not
    /// written as a chain node; the session detaches from the chain.
    pub node_tail_cap_bytes: usize,
    /// Session-level LRU capacity.
    pub max_sessions: usize,
}

impl Default for StoreConfig {
    fn default() -> StoreConfig {
        StoreConfig {
            block_chars: 4096,
            tail_soft_cap_bytes: 65536,
            tail_hard_cap_bytes: 1_048_576,
            node_tail_cap_bytes: 65536,
            max_sessions: 1024,
        }
    }
}

impl StoreConfig {
    fn validate(&self) -> Result<(), StoreError> {
        if self.block_chars == 0 {
            return Err(StoreError::InvalidConfig("block_chars must be > 0".into()));
        }
        if self.tail_soft_cap_bytes > self.tail_hard_cap_bytes {
            return Err(StoreError::InvalidConfig(
                "tail_soft_cap_bytes must be <= tail_hard_cap_bytes".into(),
            ));
        }
        if self.max_sessions == 0 {
            return Err(StoreError::InvalidConfig("max_sessions must be > 0".into()));
        }
        Ok(())
    }
}

// ------------------------------------------------------------ outcomes --

/// Result of [`SessionStore::put`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PutOutcome {
    pub handle: SessionHandle,
    /// Initial session revision (genesis, always 0).
    pub revision: u64,
    pub token_count: u64,
}

/// Result of [`SessionStore::append`].
///
/// Invariant (frozen): `all_ids == old_ids[..replace_from] ++
/// replacement_ids`, where `old_ids` is the full pre-append token
/// stream. Indices are zero-based positions in the pre-postprocessor
/// core stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendOutcome {
    /// Path label (encoder-defined for repair paths, plus the
    /// store-level labels `noop` and `degraded_full_reencode`).
    pub path: String,
    /// Session revision after this append.
    pub revision: u64,
    /// First token index whose value may differ from the old stream.
    pub replace_from: u64,
    /// Tokens from `replace_from` to the end of the new stream.
    pub replacement_ids: Vec<u32>,
    /// The full new token stream.
    pub all_ids: Vec<u32>,
}

/// Patch-only result of [`SessionStore::append_patch`].
///
/// Unlike [`AppendOutcome`], this value never materializes the complete token
/// history.  It is the native serving surface for callers that already retain
/// the previous stream: truncate it to `replace_from`, then append
/// `replacement_ids`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendPatchOutcome {
    /// Encoder/store path used for this append.
    pub path: String,
    /// Session revision after this append.
    pub revision: u64,
    /// Number of tokens from the old stream that remain unchanged.
    pub replace_from: u64,
    /// New stream suffix beginning at `replace_from`.
    pub replacement_ids: Vec<u32>,
    /// Total number of tokens after applying the patch.
    pub token_count: u64,
}

/// Result of a successful [`SessionStore::lookup`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupHit {
    pub handle: SessionHandle,
    /// Characters of the query covered by the materialized session.
    pub matched_chars: u64,
    /// Initial revision of the materialized session (genesis, 0).
    pub revision: u64,
}

/// Introspection snapshot of one session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionInfo {
    pub key_id: KeyId,
    pub witness: WitnessCategory,
    pub revision: u64,
    pub total_chars: u64,
    pub safe_char: u64,
    /// Byte length of the certified stable text prefix.
    pub stable_prefix_bytes: u64,
    pub token_count: u64,
    pub sealed_tokens: u64,
    pub tail_chars: u64,
    pub tail_bytes: u64,
    pub buf_bytes: u64,
    pub blocks_end: u64,
    pub chain_ok: bool,
    pub last_replace_from: u64,
    /// Rough in-memory footprint of this session in bytes.
    pub approx_bytes: u64,
}

/// Integrity material for the facade's private recovery binding.
///
/// This contains no plaintext and is intentionally narrower than a store
/// record. `text_digest` is SHA-256 over
/// `RECOVERY_TEXT_DOMAIN || complete_utf8_text`; `record_hash` binds it to
/// the exact exported session revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryMaterial {
    pub record_hash: BlockHash,
    pub text_bytes: u64,
    pub text_digest: [u8; 32],
}

/// Counters snapshot. Field names follow the pre-release prototype store v1
/// battery, with `revision_conflicts` added by the release contract.
#[derive(Debug, Clone, PartialEq)]
pub struct StatsSnapshot {
    pub format: &'static str,
    pub block_chars: u64,
    pub tail_soft_cap_bytes: u64,
    pub tail_hard_cap_bytes: u64,
    pub max_sessions: u64,
    pub session_count: u64,
    pub node_count: u64,
    pub puts: u64,
    pub extends: u64,
    pub forks: u64,
    pub lookups: u64,
    pub lookup_hits: u64,
    pub lookup_misses: u64,
    pub hit_rate: Option<f64>,
    pub checksum_rejects: u64,
    pub k_cap_overflows: u64,
    pub hard_cap_degrades: u64,
    pub seals: u64,
    pub sealed_tokens: u64,
    pub sessions_evicted: u64,
    pub nodes_skipped_tail_cap: u64,
    pub chain_detaches: u64,
    pub import_rejects: u64,
    pub revision_conflicts: u64,
    pub path_counts: BTreeMap<String, u64>,
}

// ------------------------------------------------------------ internals --

struct KeyRec {
    fingerprint: SemanticFingerprint,
    /// Seal end guard in characters: no seal within this distance of the
    /// tail end. Sound bound = the longest added-token literal of the
    /// frozen tokenizer (a literal completed across a future append can
    /// rewrite tokenization retroactively up to that lookback; the
    /// certificates cover the pre-tokenizer, not the text-global
    /// added-token extraction pass).
    seal_guard_chars: u64,
}

/// In-memory sealed node: the cache record plus its captured checksum.
/// The checksum is captured at creation/import time and re-verified on
/// every lookup touch, so silent memory corruption (or the test
/// corruption hook) is caught and counted, never served.
struct NodeEntry {
    rec: NodeCacheRecord,
    checksum: [u8; 32],
}

/// Sealed-prefix ID storage: immutable shared blocks plus one growable
/// owned suffix.
///
/// The certified stable prefix is append-only, so it can be represented
/// as a sequence of immutable [`SharedIds`] ranges (typically the head
/// range of the engine's original seed allocation) followed by one
/// owned vector that absorbs the small heads of later seals. Pushing a
/// shared block freezes the current owned suffix into its own shared
/// block first, so block order always equals stream order and no block
/// is ever mutated after it becomes shared.
#[derive(Debug, Clone, Default)]
struct SealedIds {
    blocks: Vec<SharedIds>,
    owned: Vec<u32>,
    len: usize,
}

impl SealedIds {
    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Adopt an immutable shared range as the next sealed segment.
    fn push_shared(&mut self, ids: SharedIds) {
        if ids.is_empty() {
            return;
        }
        if !self.owned.is_empty() {
            let frozen = SharedIds::from_vec(std::mem::take(&mut self.owned));
            self.blocks.push(frozen);
        }
        self.len += ids.len();
        self.blocks.push(ids);
    }

    /// Copy a small sealed head into the growable owned suffix.
    fn extend_from_slice(&mut self, ids: &[u32]) {
        self.owned.extend_from_slice(ids);
        self.len += ids.len();
    }

    fn from_vec(ids: Vec<u32>) -> SealedIds {
        SealedIds {
            blocks: Vec::new(),
            len: ids.len(),
            owned: ids,
        }
    }

    /// The stream as contiguous slices, in order.
    fn parts(&self) -> impl Iterator<Item = &[u32]> {
        self.blocks
            .iter()
            .map(SharedIds::as_slice)
            .chain((!self.owned.is_empty()).then_some(&self.owned[..]))
    }

    /// Append the suffix starting at absolute index `from` to `out`.
    fn extend_suffix_into(&self, out: &mut Vec<u32>, from: usize) {
        let mut base = 0usize;
        for part in self.parts() {
            let end = base + part.len();
            if end > from {
                let lo = from.saturating_sub(base);
                out.extend_from_slice(&part[lo..]);
            }
            base = end;
        }
    }

    /// The suffix starting at absolute index `from` as an owned row.
    fn suffix_to_vec(&self, from: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.len.saturating_sub(from));
        self.extend_suffix_into(&mut out, from);
        out
    }

    /// The single shared range holding the entire sealed stream, when
    /// the representation is exactly one shared block.
    fn as_single_shared(&self) -> Option<&SharedIds> {
        match (self.blocks.as_slice(), self.owned.is_empty()) {
            ([single], true) => Some(single),
            _ => None,
        }
    }
}

/// Incremental full-text state for the facade recovery binding.
///
/// A bare import of a sealed record cannot reconstruct this state because
/// store format v1 deliberately omits its stable-prefix plaintext. Such a
/// session carries `None` until imported through the recovery-aware path.
#[derive(Clone)]
struct RecoveryDigest {
    hasher: Sha256,
    text_bytes: u64,
}

impl RecoveryDigest {
    fn empty() -> RecoveryDigest {
        let mut hasher = Sha256::new();
        hasher.update(RECOVERY_TEXT_DOMAIN);
        RecoveryDigest {
            hasher,
            text_bytes: 0,
        }
    }

    fn from_text(text: &str) -> Result<RecoveryDigest, StoreError> {
        let mut state = RecoveryDigest::empty();
        state.append(text)?;
        Ok(state)
    }

    fn append(&mut self, delta: &str) -> Result<(), StoreError> {
        let delta_bytes = u64::try_from(delta.len())
            .map_err(|_| StoreError::InvalidInput("text byte length exceeds u64".into()))?;
        let next = self
            .text_bytes
            .checked_add(delta_bytes)
            .ok_or_else(|| StoreError::InvalidInput("text byte length exceeds u64".into()))?;
        self.hasher.update(delta.as_bytes());
        self.text_bytes = next;
        Ok(())
    }

    fn material(&self, record_hash: BlockHash) -> RecoveryMaterial {
        let digest: [u8; 32] = self.hasher.clone().finalize().into();
        RecoveryMaterial {
            record_hash,
            text_bytes: self.text_bytes,
            text_digest: digest,
        }
    }
}

fn trailing_checksum(bytes: &[u8]) -> Result<[u8; 32], StoreError> {
    let n = bytes.len();
    if n < 32 {
        return Err(StoreError::Internal(
            "serialized cache record shorter than its checksum".into(),
        ));
    }
    Ok(bytes[n - 32..].try_into().expect("fixed slice"))
}

impl NodeEntry {
    fn create(rec: NodeCacheRecord) -> Result<NodeEntry, StoreError> {
        let bytes = rec.to_bytes()?;
        let checksum = trailing_checksum(&bytes)?;
        Ok(NodeEntry { rec, checksum })
    }

    fn verify(&self) -> bool {
        match self.rec.to_bytes() {
            Ok(bytes) => matches!(trailing_checksum(&bytes), Ok(cs) if cs == self.checksum),
            Err(_) => false,
        }
    }
}

/// One live session. Owns its sealed ids and its mutable tail; nothing
/// aliases another session (chain nodes are shared by content address in
/// the global table).
struct Session {
    key_id: KeyId,
    witness: WitnessCategory,
    /// Store revision, genesis 0, strictly increasing per append.
    revision: u64,
    /// Chain link hash of the predecessor revision's record (all-zero
    /// at genesis).
    prev_record_hash: BlockHash,
    /// Chain link hash of this revision's record.
    record_hash: BlockHash,
    last_replace_from: u64,
    total_chars: u64,
    /// Last certified split point; sealed_ids cover chars [0, safe_char).
    safe_char: u64,
    /// UTF-8 byte length of the stable prefix [0, safe_char).
    safe_byte: u64,
    /// Sealed prefix IDs: immutable shared blocks plus one owned suffix.
    sealed: SealedIds,
    /// Running payload-digest state over the sealed stream. Invariant:
    /// always equal to a fresh [`PayloadHasher`] fed the current sealed
    /// IDs, so a commit digests only the mutable tail instead of
    /// rehashing the append-only multi-million-token prefix. Every
    /// sealed-stream change site advances or rebuilds this state in the
    /// same statement group; [`Session::commit_hash`] cross-checks it
    /// against the full recomputation in debug builds.
    sealed_payload: PayloadHasher,
    /// Monotone key for [`Self::ids_snapshot`]: bumped before any
    /// mutation that can change the complete ID stream's content, so a
    /// cached snapshot can never be served for a newer stream.
    ids_generation: u64,
    /// Complete-row snapshot cache under [`Self::ids_generation`].
    /// Sealing repartitions the stream without changing its content, so
    /// it deliberately does not bump the generation.
    ids_snapshot: Option<(u64, SharedIds)>,
    /// Chars [0, blocks_end) are already hashed into the chain.
    blocks_end: u64,
    /// UTF-8 byte length of [0, blocks_end).
    blocks_end_byte: u64,
    /// Raw text [blocks_end, max(blocks_end, safe_char)), retained only
    /// for future block hashing; empty once the chain is detached.
    buf: String,
    chain_tail: Option<BlockHash>,
    chain_base_safe: u64,
    chain_base_idx: u64,
    chain_ok: bool,
    /// Ascending (safe_char, cumulative sealed ids) seal points; used to
    /// adopt already-existing chain nodes written by sibling sessions.
    seal_log: Vec<(u64, u64)>,
    /// Tail-local char position at or below which no certified boundary
    /// exists (scan memo; boundary positions are char-determined).
    scan_floor: u64,
    /// Text and encoding past the last certified seal point.
    tail: TailState,
    /// Full-text recovery digest when every historical byte is known.
    recovery: Option<RecoveryDigest>,
    /// Incremental personalized-BLAKE2b endpoint and geometric marks used
    /// by content lookup and TKFR-v1. Absent only when tracking was disabled
    /// or a sealed record was imported without its historical plaintext.
    content: Option<ContentDigest>,
    last_used: u64,
}

impl Session {
    fn fresh(key_id: KeyId, witness: WitnessCategory, clock: u64) -> Session {
        Session {
            key_id,
            witness,
            revision: 0,
            prev_record_hash: ZERO_HASH,
            record_hash: ZERO_HASH,
            last_replace_from: 0,
            total_chars: 0,
            safe_char: 0,
            safe_byte: 0,
            sealed: SealedIds::default(),
            sealed_payload: PayloadHasher::new(),
            ids_generation: 0,
            ids_snapshot: None,
            blocks_end: 0,
            blocks_end_byte: 0,
            buf: String::new(),
            chain_tail: None,
            chain_base_safe: 0,
            chain_base_idx: 0,
            chain_ok: true,
            seal_log: vec![(0, 0)],
            scan_floor: 0,
            tail: TailState::new(),
            recovery: None,
            content: None,
            last_used: clock,
        }
    }

    fn token_count(&self) -> u64 {
        self.sealed.len() as u64 + self.tail.n_tokens() as u64
    }

    /// Invalidate the complete-row snapshot before a content mutation.
    fn bump_ids_generation(&mut self) {
        self.ids_generation += 1;
        self.ids_snapshot = None;
    }

    /// `replace_token_offset` as recorded on disk: witness category 0
    /// carries the frozen cross-invariant (full re-encode semantics),
    /// so it always records 0.
    fn recorded_replace_offset(&self) -> u64 {
        if self.witness == WitnessCategory::NoneFullReencode {
            0
        } else {
            self.last_replace_from
        }
    }

    /// Chain link hash of the current committed state. The link hashes the
    /// Section 4.2 metadata (revision, lengths, fingerprint, previous hash,
    /// and witness) plus a payload digest over all core IDs followed by the
    /// stored mutable UTF-8 tail. That digest continues the sealed-ID hasher
    /// with only the tail IDs and text, without retaining sealed-prefix
    /// plaintext; debug builds compare it with a full
    /// [`payload_digest_parts`] recomputation on every commit.
    fn commit_hash(&self, fingerprint: &SemanticFingerprint) -> BlockHash {
        let digest = self
            .sealed_payload
            .digest_with_tail(self.tail.ids(), self.tail.text().as_bytes());
        debug_assert_eq!(
            digest,
            {
                let mut parts: Vec<&[u32]> = self.sealed.parts().collect();
                parts.push(self.tail.ids());
                payload_digest_parts(&parts, self.tail.text().as_bytes())
            },
            "sealed-prefix payload state must agree with the full recomputation"
        );
        let tail_len = self.tail.text_bytes() as u64;
        link_hash(&LinkInputs {
            prev_block_hash: &self.prev_record_hash,
            fingerprint,
            session_revision: self.revision,
            full_text_bytes: self.safe_byte + tail_len,
            stable_prefix_bytes: self.safe_byte,
            text_tail_bytes: tail_len,
            token_count: self.token_count(),
            replace_token_offset: self.recorded_replace_offset(),
            witness: self.witness,
            payload_digest: &digest,
        })
    }
}

#[derive(Default)]
struct Stats {
    puts: u64,
    extends: u64,
    forks: u64,
    lookups: u64,
    lookup_hits: u64,
    lookup_misses: u64,
    checksum_rejects: u64,
    k_cap_overflows: u64,
    hard_cap_degrades: u64,
    seals: u64,
    sealed_tokens: u64,
    sessions_evicted: u64,
    nodes_skipped_tail_cap: u64,
    chain_detaches: u64,
    import_rejects: u64,
    revision_conflicts: u64,
    path_counts: BTreeMap<String, u64>,
}

impl Stats {
    fn bump_path(&mut self, path: &str) {
        *self.path_counts.entry(path.to_string()).or_insert(0) += 1;
    }
}

fn internal(msg: impl std::fmt::Display) -> StoreError {
    StoreError::Internal(msg.to_string())
}

// -------------------------------------------------------- text helpers --

/// Byte offset of char index `chars` in `s` (`s.len()` when past the end).
fn char_to_byte(s: &str, chars: u64) -> usize {
    if chars == 0 {
        return 0;
    }
    for (count, (ix, _)) in s.char_indices().enumerate() {
        if count as u64 == chars {
            return ix;
        }
    }
    s.len()
}

/// Byte slices of the complete `block_chars`-sized char blocks of `text`.
fn block_slices(text: &str, block_chars: u64) -> Vec<&str> {
    let mut out = Vec::new();
    let mut prev = 0usize;
    let mut count = 0u64;
    for (ix, _) in text.char_indices() {
        if count != 0 && count.is_multiple_of(block_chars) {
            out.push(&text[prev..ix]);
            prev = ix;
        }
        count += 1;
    }
    if count != 0 && count.is_multiple_of(block_chars) {
        out.push(&text[prev..]);
    }
    out
}

/// Raw text chars [a, b) of a session, assembled from `buf` and the tail.
fn text_range(sess: &Session, a: u64, b: u64) -> Result<String, StoreError> {
    if a > b || b > sess.total_chars {
        return Err(internal(format!("text_range [{a}, {b}) out of bounds")));
    }
    let mut out = String::new();
    if a < sess.safe_char {
        if a < sess.blocks_end {
            return Err(internal(format!(
                "text_range start {a} precedes blocks_end {}",
                sess.blocks_end
            )));
        }
        let lo = char_to_byte(&sess.buf, a - sess.blocks_end);
        let hi = char_to_byte(&sess.buf, b.min(sess.safe_char) - sess.blocks_end);
        out.push_str(&sess.buf[lo..hi]);
    }
    if b > sess.safe_char {
        let lo_char = a.max(sess.safe_char) - sess.safe_char;
        let hi_char = b - sess.safe_char;
        let lo = sess.tail.byte_ix_of_char(
            u32::try_from(lo_char).map_err(|_| internal("char index exceeds u32"))?,
        );
        let hi = sess.tail.byte_ix_of_char(
            u32::try_from(hi_char).map_err(|_| internal("char index exceeds u32"))?,
        );
        out.push_str(&sess.tail.text()[lo..hi]);
    }
    Ok(out)
}

// ------------------------------------------------------------- sealing --

fn detach_chain(sess: &mut Session, stats: &mut Stats) {
    sess.chain_ok = false;
    sess.buf = String::new();
    stats.chain_detaches += 1;
}

/// Key material needed by the post-append bookkeeping.
struct KeyCtx {
    fingerprint: SemanticFingerprint,
    seal_guard_chars: u64,
}

/// Advance the block hash chain over newly completed blocks and write
/// (or adopt) the corresponding sealed nodes.
fn seal_blocks(
    cfg: &StoreConfig,
    sess: &mut Session,
    key: &KeyCtx,
    nodes: &mut HashMap<BlockHash, NodeEntry>,
    stats: &mut Stats,
) -> Result<(), StoreError> {
    loop {
        let end = match sess.blocks_end.checked_add(cfg.block_chars) {
            Some(end) if sess.chain_ok && end <= sess.total_chars => end,
            _ => break,
        };
        let index = sess.blocks_end / cfg.block_chars;
        let block = text_range(sess, sess.blocks_end, end)?;
        let key_hash = node_key(
            sess.chain_tail.as_ref(),
            &key.fingerprint,
            index,
            block.as_bytes(),
        );
        let end_byte = sess
            .blocks_end_byte
            .checked_add(block.len() as u64)
            .ok_or_else(|| internal("block byte accounting overflow"))?;
        if let Some(existing) = nodes.get(&key_hash) {
            // A sibling session already sealed this prefix. Adopt its
            // node as our chain base only if its seal point is one of
            // ours (same char position and cumulative token count);
            // otherwise our ids partition is incompatible - detach.
            let cum_end = existing.rec.ids_base + existing.rec.ids.len() as u64;
            let adopt = sess
                .seal_log
                .iter()
                .rev()
                .take_while(|&&(c, _)| c >= existing.rec.safe_char)
                .any(|&(c, i)| c == existing.rec.safe_char && i == cum_end);
            if !adopt {
                detach_chain(sess, stats);
                break;
            }
            sess.chain_tail = Some(key_hash);
            sess.chain_base_safe = existing.rec.safe_char;
            sess.chain_base_idx = cum_end;
        } else {
            if sess.safe_char > end {
                return Err(internal("safe_char beyond a completing block"));
            }
            let text_tail = text_range(sess, sess.safe_char, end)?;
            if text_tail.len() > cfg.node_tail_cap_bytes {
                stats.nodes_skipped_tail_cap += 1;
                detach_chain(sess, stats);
                break;
            }
            let base = usize::try_from(sess.chain_base_idx)
                .map_err(|_| internal("chain base index exceeds usize"))?;
            if base > sess.sealed.len() {
                return Err(internal("chain base index exceeds the sealed stream"));
            }
            let ids = sess.sealed.suffix_to_vec(base);
            let rec = NodeCacheRecord {
                fingerprint: key.fingerprint,
                witness: sess.witness,
                parent: sess.chain_tail,
                key: key_hash,
                block_index: index,
                end_char: end,
                safe_char: sess.safe_char,
                ids_base: sess.chain_base_idx,
                end_byte,
                ids,
                text_tail,
            };
            nodes.insert(key_hash, NodeEntry::create(rec)?);
            sess.chain_tail = Some(key_hash);
            sess.chain_base_safe = sess.safe_char;
            sess.chain_base_idx = sess.sealed.len() as u64;
        }
        sess.blocks_end = end;
        sess.blocks_end_byte = end_byte;
        let drop = char_to_byte(&sess.buf, cfg.block_chars);
        sess.buf.drain(..drop);
    }
    Ok(())
}

/// Seal the tail at a certified boundary: tokens [0, cut_tokens) covering
/// chars [0, cut_char) move into the sealed stream, and the tail state is
/// rebuilt over the remaining text with spans shifted to the new origin.
///
/// The sealed head is copy-minimal: when the tail still holds an adopted
/// shared allocation (the shape of an initial multi-million-token seed),
/// the head becomes an immutable shared range of that same allocation and
/// only the small residual tail is rebased; nothing head-sized is cloned.
/// An owned tail (the ordinary small post-seal shape) keeps the previous
/// copy behavior, which is proportional to the small head.
fn seal_tail(sess: &mut Session, cut: BoundaryCut, stats: &mut Stats) -> Result<(), StoreError> {
    let BoundaryCut {
        cut_tokens,
        cut_char,
    } = cut;
    let cut32 = u32::try_from(cut_char).map_err(|_| internal("seal cut_char exceeds u32"))?;
    let n_tokens = sess.tail.n_tokens();
    if cut_tokens == 0 || cut_tokens >= n_tokens {
        return Err(internal(format!(
            "seal cut_tokens {cut_tokens} out of range"
        )));
    }
    // The guards and the residual spans need exactly the window
    // [cut_tokens - 1, n): one region rebuild on a lazy tail, plain
    // slice copies on a materialized one.
    let (win_starts, win_ends) = sess.tail.span_window(cut_tokens - 1, n_tokens)?;
    if win_starts[1] != cut32 {
        return Err(internal("seal cut_char is not the next token start"));
    }
    if win_ends[0] > cut32 {
        // Multi-token single-char groups (byte fallback) share one
        // char span; cutting inside such a group would split a
        // character's token group. The boundary probe excludes
        // these; re-checked here as an invariant.
        return Err(internal("seal cut would split a token group mid-char"));
    }
    let bix = sess.tail.byte_ix_of_char(cut32);
    // Both range views are taken before any mutation; on a shared tail
    // they are two adjacent windows of one immutable allocation.
    let head_shared = match sess.tail.shared_ids() {
        Some(view) => Some(view.slice(0, cut_tokens)?),
        None => None,
    };
    let rest_shared = match sess.tail.shared_ids() {
        Some(view) => Some(view.slice(cut_tokens, n_tokens)?),
        None => None,
    };
    stats.seals += 1;
    stats.sealed_tokens += cut_tokens as u64;
    match head_shared {
        Some(view) => {
            sess.sealed_payload.update_ids(view.as_slice());
            sess.sealed.push_shared(view);
        }
        None => {
            let head_ids = &sess.tail.ids()[..cut_tokens];
            sess.sealed_payload.update_ids(head_ids);
            sess.sealed.extend_from_slice(head_ids);
        }
    }
    if sess.chain_ok {
        // buf covers [blocks_end, safe_char): when block hashing already
        // ran past the old safe point, that prefix of the sealed head is
        // hashed and must not enter buf.
        let head_text = &sess.tail.text()[..bix];
        let skip = char_to_byte(head_text, sess.blocks_end.saturating_sub(sess.safe_char));
        sess.buf.push_str(&head_text[skip..]);
    }
    sess.safe_char += cut_char;
    sess.safe_byte += bix as u64;
    sess.seal_log
        .push((sess.safe_char, sess.sealed.len() as u64));
    // Rebase the residual tail: small text copy plus origin-shifted spans.
    let rest_text = sess.tail.text()[bix..].to_string();
    let rest_starts: Vec<u32> = win_starts[1..].iter().map(|&start| start - cut32).collect();
    let rest_ends: Vec<u32> = win_ends[1..].iter().map(|&end| end - cut32).collect();
    let mut rebased = TailState::new();
    match rest_shared {
        Some(view) => rebased.fill_shared(&rest_text, view, rest_starts, rest_ends)?,
        None => rebased.fill_soa(
            &rest_text,
            SoaEncoding {
                ids: sess.tail.ids()[cut_tokens..].to_vec(),
                span_starts: rest_starts,
                span_ends: rest_ends,
            },
        )?,
    }
    sess.tail = rebased;
    Ok(())
}

/// Bookkeeping shared by put/append after the tail text grew: update the
/// totals, hash newly completed blocks, then try to advance the certified
/// seal point and account the caps.
fn post_text_ops(
    cfg: &StoreConfig,
    sess: &mut Session,
    key: &KeyCtx,
    nodes: &mut HashMap<BlockHash, NodeEntry>,
    stats: &mut Stats,
    engine: &dyn SessionEncoder,
) -> Result<(), StoreError> {
    let tail_chars = u64::from(sess.tail.text_chars());
    sess.total_chars = sess.safe_char + tail_chars;

    if sess.chain_ok {
        seal_blocks(cfg, sess, key, nodes, stats)?;
    }

    // Seal end guard: boundaries within seal_guard_chars of the tail end
    // are excluded (added-token literal completion lookback bound); the
    // scan memo tracks the verified-clear range accordingly.
    let tail_bytes = sess.tail.text_bytes();
    let ceil = tail_chars.saturating_sub(key.seal_guard_chars);
    if (tail_chars > cfg.block_chars || tail_bytes > cfg.tail_soft_cap_bytes)
        && ceil > sess.scan_floor
    {
        let found = engine.last_certified_boundary(&sess.tail, sess.scan_floor, ceil)?;
        if let Some(cut) = found {
            let cut_char = cut.cut_char;
            seal_tail(sess, cut, stats)?;
            sess.scan_floor = ceil - cut_char;
        } else {
            sess.scan_floor = ceil;
        }
    }

    if sess.tail.text_bytes() > cfg.tail_soft_cap_bytes {
        stats.k_cap_overflows += 1;
    }
    Ok(())
}

// --------------------------------------------------------------- store --

/// In-memory session store. Persistence is layered on top through the
/// record export/import surface (see the `toktier-store-sqlite` crate);
/// the core stays memory-only and dependency-free beyond the pinned hash
/// primitive.
pub struct SessionStore {
    cfg: StoreConfig,
    recovery_tracking: bool,
    content_tracking: bool,
    /// Optional bounded worker that overlaps the seed content-digest
    /// scan with the seed encode (PLAN/162 WP5/WP6). `None` keeps the
    /// fully serial path; the digest bytes are identical either way.
    seed_overlap: Option<std::sync::Arc<dyn OverlapRunner>>,
    /// Test-only fault: when set, the seed content-digest computation
    /// reports this failure instead of scanning. One hook serves the
    /// serial and overlap paths so their failure behavior can be
    /// compared directly.
    #[cfg(any(test, feature = "testing"))]
    injected_content_digest_fault: Option<String>,
    keys: Vec<KeyRec>,
    nodes: HashMap<BlockHash, NodeEntry>,
    sessions: HashMap<u64, Session>,
    next_handle: u64,
    clock: u64,
    stats: Stats,
    /// Structural-evidence counter: how many times a complete ID row was
    /// materialized into fresh memory (explicit `all_ids`/snapshot misses,
    /// append outcomes carrying the full row, and record export). Shared
    /// fast paths and cache hits do not count. Not part of the frozen
    /// stats snapshot; used to prove copy-behavior claims in tests.
    full_row_materializations: std::sync::atomic::AtomicU64,
}

impl SessionStore {
    /// New store with the given configuration.
    pub fn new(cfg: StoreConfig) -> Result<SessionStore, StoreError> {
        cfg.validate()?;
        Ok(SessionStore {
            cfg,
            recovery_tracking: false,
            content_tracking: false,
            seed_overlap: None,
            #[cfg(any(test, feature = "testing"))]
            injected_content_digest_fault: None,
            keys: Vec::new(),
            nodes: HashMap::new(),
            sessions: HashMap::new(),
            next_handle: 1,
            clock: 0,
            stats: Stats::default(),
            full_row_materializations: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// New store with default configuration.
    pub fn with_defaults() -> SessionStore {
        SessionStore::new(StoreConfig::default()).expect("default config is valid")
    }

    pub fn config(&self) -> &StoreConfig {
        &self.cfg
    }

    fn count_full_row_materialization(&self) {
        self.full_row_materializations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// How many complete-row materializations this store has performed.
    /// Structural evidence for copy-behavior tests; not a frozen stat.
    pub fn ids_materialization_count(&self) -> u64 {
        self.full_row_materializations
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether persistence recovery digests are maintained for new appends.
    pub fn recovery_tracking_enabled(&self) -> bool {
        self.recovery_tracking
    }

    /// Whether content-index checkpoints are maintained for new appends.
    pub fn content_tracking_enabled(&self) -> bool {
        self.content_tracking
    }

    /// Enable private full-text recovery tracking before creating or
    /// importing sessions. The default remains off, so ordinary in-memory
    /// stores pay no hashing cost for facade persistence they do not use.
    pub fn enable_recovery_tracking(&mut self) -> Result<(), StoreError> {
        if !self.sessions.is_empty() {
            return Err(StoreError::InvalidInput(
                "recovery tracking must be enabled before sessions exist".into(),
            ));
        }
        self.recovery_tracking = true;
        Ok(())
    }

    /// Enable the incremental content-index accumulator before sessions
    /// exist. Unlike recovery SHA tracking, the facade enables this for both
    /// memory and persistent stores because content lookup consumes it.
    pub fn enable_content_tracking(&mut self) -> Result<(), StoreError> {
        if !self.sessions.is_empty() {
            return Err(StoreError::InvalidInput(
                "content tracking must be enabled before sessions exist".into(),
            ));
        }
        self.content_tracking = true;
        Ok(())
    }

    /// Install or remove the bounded worker that overlaps the seed
    /// content-digest scan with the seed encode (PLAN/162 WP5/WP6).
    /// The default is `None`: the fully serial path. Overlap changes
    /// only wall-clock scheduling; the digest bytes, every validation,
    /// the failure ordering, and the atomic insertion are identical in
    /// both modes.
    pub fn set_seed_overlap(&mut self, runner: Option<std::sync::Arc<dyn OverlapRunner>>) {
        self.seed_overlap = runner;
    }

    /// Whether a seed-overlap worker is installed.
    pub fn seed_overlap_enabled(&self) -> bool {
        self.seed_overlap.is_some()
    }

    /// Test-only: make the seed content-digest computation fail with
    /// this message on both the serial and overlap paths.
    #[cfg(any(test, feature = "testing"))]
    pub fn inject_content_digest_fault(&mut self, message: Option<String>) {
        self.injected_content_digest_fault = message;
    }

    /// The injected content-digest fault, if any (release builds have
    /// none). Resolved once per seed so the serial and overlap paths
    /// observe the same decision.
    fn content_digest_fault(&self) -> Option<StoreError> {
        #[cfg(any(test, feature = "testing"))]
        if let Some(message) = &self.injected_content_digest_fault {
            return Some(StoreError::InvalidInput(message.clone()));
        }
        None
    }

    /// Content digest of a seed text: the personalized-BLAKE2b endpoint
    /// plus the geometric checkpoints. One definition serves the serial
    /// and overlap paths, so both compute byte-identical results.
    fn seed_content_digest(&self, text: &str) -> Result<ContentDigest, StoreError> {
        if let Some(error) = self.content_digest_fault() {
            return Err(error);
        }
        ContentDigest::from_bytes(text.as_bytes())
    }

    fn key(&self, key_id: KeyId) -> Result<KeyCtx, StoreError> {
        self.keys
            .get(key_id.0 as usize)
            .map(|k| KeyCtx {
                fingerprint: k.fingerprint,
                seal_guard_chars: k.seal_guard_chars,
            })
            .ok_or(StoreError::UnknownKey(key_id.0))
    }

    fn session(&self, handle: SessionHandle) -> Result<&Session, StoreError> {
        self.sessions
            .get(&handle.0)
            .ok_or(StoreError::UnknownSession(handle.0))
    }

    /// Evict least-recently-used sessions until one slot is free.
    /// Ties on `last_used` (possible after fork) break toward the
    /// smallest handle, keeping eviction deterministic.
    fn evict_for_capacity(&mut self) {
        while self.sessions.len() >= self.cfg.max_sessions {
            let Some((&h, _)) = self.sessions.iter().min_by_key(|(&h, s)| (s.last_used, h)) else {
                return;
            };
            self.sessions.remove(&h);
            self.stats.sessions_evicted += 1;
        }
    }

    fn insert_session(&mut self, sess: Session) -> SessionHandle {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.sessions.insert(handle, sess);
        SessionHandle(handle)
    }

    /// Intern a fingerprint; returns a stable key id (same fingerprint,
    /// same id). Re-registering with a different guard is an error.
    ///
    /// `seal_end_guard_chars` must be at least the longest added-token
    /// literal (in characters) of the frozen tokenizer behind the
    /// fingerprint (0 only when it has none): sealing is excluded within
    /// that distance of the tail end, because a literal completed by a
    /// future append rewrites tokenization retroactively across
    /// otherwise certified boundaries.
    pub fn register_fingerprint(
        &mut self,
        fingerprint: SemanticFingerprint,
        seal_end_guard_chars: u64,
    ) -> Result<KeyId, StoreError> {
        if let Some(ix) = self.keys.iter().position(|k| k.fingerprint == fingerprint) {
            if self.keys[ix].seal_guard_chars != seal_end_guard_chars {
                return Err(StoreError::GuardMismatch);
            }
            return Ok(KeyId(ix as u32));
        }
        let id = u32::try_from(self.keys.len())
            .map_err(|_| StoreError::InvalidInput("key table full".into()))?;
        self.keys.push(KeyRec {
            fingerprint,
            seal_guard_chars: seal_end_guard_chars,
        });
        Ok(KeyId(id))
    }

    /// Registered fingerprints as `(key_id, fingerprint, guard)` rows.
    pub fn export_fingerprints(&self) -> Vec<(u32, SemanticFingerprint, u64)> {
        self.keys
            .iter()
            .enumerate()
            .map(|(i, k)| (i as u32, k.fingerprint, k.seal_guard_chars))
            .collect()
    }

    /// Full encode of `text` into a new session at revision 0.
    ///
    /// With a seed-overlap runner installed (PLAN/162 WP5/WP6), the
    /// content-digest scan runs on the bounded worker while the routed
    /// encode runs on this thread. The join happens before any
    /// validation, sealing, or insertion; the joined digest result is
    /// then examined at exactly the position where the serial path
    /// computes it, so failure ordering and visible state transitions
    /// cannot differ between the two modes. Capacity eviction and the
    /// fresh session shell stay ahead of the window and are safe to
    /// discard: nothing becomes visible before the final insertion.
    pub fn put(
        &mut self,
        key_id: KeyId,
        text: &str,
        engine: &dyn SessionEncoder,
    ) -> Result<PutOutcome, StoreError> {
        let key = self.key(key_id)?;
        self.clock += 1;
        let clock = self.clock;
        self.evict_for_capacity();
        let mut sess = Session::fresh(key_id, engine.witness_category(), clock);
        self.stats.puts += 1;
        let mut overlap_digest: Option<Result<ContentDigest, StoreError>> = None;
        if !text.is_empty() {
            let report = match &self.seed_overlap {
                Some(runner) if self.content_tracking => {
                    let fault = self.content_digest_fault();
                    let mut encode_result = None;
                    {
                        let digest_slot = &mut overlap_digest;
                        let encode_slot = &mut encode_result;
                        let tail = &mut sess.tail;
                        runner.run_joined(
                            &mut || {
                                *digest_slot = Some(match fault.clone() {
                                    Some(error) => Err(error),
                                    None => ContentDigest::from_bytes(text.as_bytes()),
                                });
                            },
                            &mut || *encode_slot = Some(engine.append(tail, text)),
                        );
                    }
                    encode_result.ok_or_else(|| {
                        internal("the overlap runner skipped the foreground encode")
                    })??
                }
                _ => engine.append(&mut sess.tail, text)?,
            };
            verify_append_shape(&sess.tail, 0, text, &[], report.kept_tokens)?;
            self.stats.bump_path(&report.path);
            post_text_ops(
                &self.cfg,
                &mut sess,
                &key,
                &mut self.nodes,
                &mut self.stats,
                engine,
            )?;
        }
        if self.recovery_tracking {
            sess.recovery = Some(RecoveryDigest::from_text(text)?);
        }
        if self.content_tracking {
            sess.content = Some(match overlap_digest {
                Some(digest) => digest?,
                None => self.seed_content_digest(text)?,
            });
        }
        sess.record_hash = sess.commit_hash(&key.fingerprint);
        let token_count = sess.token_count();
        let handle = self.insert_session(sess);
        Ok(PutOutcome {
            handle,
            revision: 0,
            token_count,
        })
    }

    /// Certified append. `expected_revision` must equal the session's
    /// current revision (optimistic concurrency; conflicts are counted
    /// and returned, never resolved last-writer-wins).
    pub fn append(
        &mut self,
        handle: SessionHandle,
        delta: &str,
        expected_revision: u64,
        engine: &dyn SessionEncoder,
    ) -> Result<AppendOutcome, StoreError> {
        let patch = self.append_patch(handle, delta, expected_revision, engine)?;
        let all_ids = self.collect_all_ids(handle)?;
        Ok(AppendOutcome {
            path: patch.path,
            revision: patch.revision,
            replace_from: patch.replace_from,
            replacement_ids: patch.replacement_ids,
            all_ids,
        })
    }

    /// Certified append returning only the replacement suffix.
    ///
    /// This performs the same mutation, revision checks, sealing, recovery
    /// binding, and content-index updates as [`Self::append`], but it avoids
    /// allocating or copying the unchanged token prefix.
    pub fn append_patch(
        &mut self,
        handle: SessionHandle,
        delta: &str,
        expected_revision: u64,
        engine: &dyn SessionEncoder,
    ) -> Result<AppendPatchOutcome, StoreError> {
        let key_id = self.session(handle)?.key_id;
        let key = self.key(key_id)?;
        let hard_cap = self.cfg.tail_hard_cap_bytes;
        {
            let sess = self.session(handle)?;
            let (witness, revision) = (sess.witness, sess.revision);
            if witness != engine.witness_category() {
                return Err(StoreError::WitnessCategoryMismatch {
                    recorded: witness,
                    engine: engine.witness_category(),
                });
            }
            if revision != expected_revision {
                self.stats.revision_conflicts += 1;
                return Err(StoreError::RevisionConflict {
                    expected: expected_revision,
                    actual: revision,
                });
            }
        }
        self.clock += 1;
        let clock = self.clock;
        self.stats.extends += 1;
        let cfg = self.cfg.clone();
        let sess = self
            .sessions
            .get_mut(&handle.0)
            .ok_or(StoreError::UnknownSession(handle.0))?;
        sess.last_used = clock;

        let old_sealed = sess.sealed.len() as u64;
        let old_tail_ids = sess.tail.ids().to_vec();
        let old_tail_text_len = sess.tail.text_bytes();

        if !delta.is_empty() {
            // The stream content is about to change; a cached complete-row
            // snapshot must not survive into the new revision. A noop
            // append keeps the content, so the cache stays valid there.
            sess.bump_ids_generation();
        }
        let (path, kept_tokens) = if delta.is_empty() {
            ("noop".to_string(), old_tail_ids.len())
        } else if sess.tail.text_bytes() > hard_cap {
            // Hard-cap degradation: bypass the repair window path and
            // fully re-encode tail + delta through the reference encode.
            let mut full = String::with_capacity(sess.tail.text_bytes() + delta.len());
            full.push_str(sess.tail.text());
            full.push_str(delta);
            let enc = engine.encode(&full)?;
            sess.tail.fill(&full, enc)?;
            ("degraded_full_reencode".to_string(), 0)
        } else {
            let report = engine.append(&mut sess.tail, delta)?;
            (report.path, report.kept_tokens)
        };
        verify_append_shape(
            &sess.tail,
            old_tail_text_len,
            delta,
            &old_tail_ids,
            kept_tokens,
        )?;
        if !delta.is_empty() {
            post_text_ops(&cfg, sess, &key, &mut self.nodes, &mut self.stats, engine)?;
        }
        if path == "degraded_full_reencode" {
            self.stats.hard_cap_degrades += 1;
        }
        self.stats.bump_path(&path);

        if let Some(recovery) = sess.recovery.as_mut() {
            recovery.append(delta)?;
        }
        if let Some(content) = sess.content.as_mut() {
            content.append(delta.as_bytes())?;
        }

        let sess = self
            .sessions
            .get_mut(&handle.0)
            .ok_or(StoreError::UnknownSession(handle.0))?;
        sess.revision += 1;
        sess.prev_record_hash = sess.record_hash;
        let replace_from = old_sealed + kept_tokens as u64;
        sess.last_replace_from = replace_from;
        sess.record_hash = sess.commit_hash(&key.fingerprint);
        let replace =
            usize::try_from(replace_from).map_err(|_| internal("replace_from exceeds usize"))?;
        let sealed = sess.sealed.len();
        let token_count = sealed
            .checked_add(sess.tail.n_tokens())
            .ok_or_else(|| internal("token count overflow"))?;
        if replace > token_count {
            return Err(internal("replace_from exceeds the new token count"));
        }
        let mut replacement_ids = Vec::with_capacity(token_count - replace);
        if replace < sealed {
            sess.sealed
                .extend_suffix_into(&mut replacement_ids, replace);
            replacement_ids.extend_from_slice(sess.tail.ids());
        } else {
            replacement_ids.extend_from_slice(&sess.tail.ids()[replace - sealed..]);
        }
        Ok(AppendPatchOutcome {
            path,
            revision: sess.revision,
            replace_from,
            replacement_ids,
            token_count: u64::try_from(token_count)
                .map_err(|_| internal("token count exceeds u64"))?,
        })
    }

    /// Longest block-prefix hit for `text` under `key_id`. On a hit,
    /// materializes a fresh session from the deepest fully verified node
    /// and returns the handle with the covered character count; `None`
    /// on a miss. Every node on the walk is re-verified; the first
    /// failure counts as a checksum reject and truncates the walk
    /// (prefer miss over wrong).
    pub fn lookup(
        &mut self,
        key_id: KeyId,
        text: &str,
        engine: &dyn SessionEncoder,
    ) -> Result<Option<LookupHit>, StoreError> {
        let key = self.key(key_id)?;
        self.stats.lookups += 1;
        let witness = engine.witness_category();
        let mut parent: Option<BlockHash> = None;
        let mut cum_safe = 0u64;
        let mut cum_idx = 0u64;
        let mut cum_end_byte = 0u64;
        let mut path_keys: Vec<BlockHash> = Vec::new();
        for (i, block) in block_slices(text, self.cfg.block_chars).iter().enumerate() {
            let index = i as u64;
            let key_hash = node_key(parent.as_ref(), &key.fingerprint, index, block.as_bytes());
            let Some(entry) = self.nodes.get(&key_hash) else {
                break;
            };
            let Some(end) = index
                .checked_add(1)
                .and_then(|n| n.checked_mul(self.cfg.block_chars))
            else {
                break;
            };
            let rec = &entry.rec;
            let ok = entry.verify()
                && rec.fingerprint == key.fingerprint
                && rec.witness == witness
                && rec.parent == parent
                && rec.block_index == index
                && rec.end_char == end
                && rec.safe_char >= cum_safe
                && rec.safe_char <= end
                && rec.ids_base == cum_idx
                && rec.end_byte > cum_end_byte;
            if !ok {
                // Prefer miss over wrong: count and stop at the last
                // fully verified ancestor.
                self.stats.checksum_rejects += 1;
                break;
            }
            cum_safe = rec.safe_char;
            cum_idx += rec.ids.len() as u64;
            cum_end_byte = rec.end_byte;
            path_keys.push(key_hash);
            parent = Some(key_hash);
        }
        let Some(&deep_key) = path_keys.last() else {
            self.stats.lookup_misses += 1;
            return Ok(None);
        };

        let mut sealed: Vec<u32> = Vec::new();
        for k in &path_keys {
            sealed.extend_from_slice(&self.nodes[k].rec.ids);
        }
        let (end_char, end_byte, safe_char, tail_text) = {
            let deep = &self.nodes[&deep_key].rec;
            (
                deep.end_char,
                deep.end_byte,
                deep.safe_char,
                deep.text_tail.clone(),
            )
        };
        let enc = engine.encode(&tail_text)?;
        let mut tail = TailState::new();
        tail.fill(&tail_text, enc)?;

        self.clock += 1;
        let clock = self.clock;
        self.evict_for_capacity();
        let mut sess = Session::fresh(key_id, witness, clock);
        let tail_bytes = tail.text_bytes() as u64;
        sess.total_chars = end_char;
        sess.safe_char = safe_char;
        sess.safe_byte = end_byte
            .checked_sub(tail_bytes)
            .ok_or_else(|| internal("node byte accounting underflow"))?;
        sess.sealed_payload.update_ids(&sealed);
        sess.sealed = SealedIds::from_vec(sealed);
        sess.blocks_end = end_char;
        sess.blocks_end_byte = end_byte;
        sess.chain_tail = Some(deep_key);
        sess.chain_base_safe = safe_char;
        sess.chain_base_idx = cum_idx;
        if safe_char > 0 {
            sess.seal_log.push((safe_char, cum_idx));
        }
        sess.tail = tail;
        if self.recovery_tracking {
            let matched_end = byte_offset_at_char(text, end_char)?;
            sess.recovery = Some(RecoveryDigest::from_text(&text[..matched_end])?);
        }
        if self.content_tracking {
            let matched_end = byte_offset_at_char(text, end_char)?;
            sess.content = Some(ContentDigest::from_bytes(&text.as_bytes()[..matched_end])?);
        }
        sess.record_hash = sess.commit_hash(&key.fingerprint);
        let handle = self.insert_session(sess);
        self.stats.lookup_hits += 1;
        Ok(Some(LookupHit {
            handle,
            matched_chars: end_char,
            revision: 0,
        }))
    }

    /// Duplicate a session under a new handle. The copy starts a fresh
    /// revision lineage (genesis 0). Sealed chain nodes are shared by
    /// content addressing; the session-local state is copied.
    pub fn fork(&mut self, handle: SessionHandle) -> Result<SessionHandle, StoreError> {
        self.clock += 1;
        let clock = self.clock;
        let key = self.key(self.session(handle)?.key_id)?;
        let cloned = {
            let sess = self
                .sessions
                .get_mut(&handle.0)
                .ok_or(StoreError::UnknownSession(handle.0))?;
            sess.last_used = clock;
            let mut cloned = Session {
                key_id: sess.key_id,
                witness: sess.witness,
                revision: 0,
                prev_record_hash: ZERO_HASH,
                record_hash: ZERO_HASH,
                last_replace_from: 0,
                total_chars: sess.total_chars,
                safe_char: sess.safe_char,
                safe_byte: sess.safe_byte,
                sealed: sess.sealed.clone(),
                sealed_payload: sess.sealed_payload.clone(),
                // The fork's stream content equals the source's, so the
                // snapshot cache stays valid and is shared, not copied.
                ids_generation: sess.ids_generation,
                ids_snapshot: sess.ids_snapshot.clone(),
                blocks_end: sess.blocks_end,
                blocks_end_byte: sess.blocks_end_byte,
                buf: sess.buf.clone(),
                chain_tail: sess.chain_tail,
                chain_base_safe: sess.chain_base_safe,
                chain_base_idx: sess.chain_base_idx,
                chain_ok: sess.chain_ok,
                seal_log: sess.seal_log.clone(),
                scan_floor: sess.scan_floor,
                tail: sess.tail.clone(),
                recovery: sess.recovery.clone(),
                content: sess.content.clone(),
                last_used: clock,
            };
            cloned.record_hash = cloned.commit_hash(&key.fingerprint);
            cloned
        };
        self.stats.forks += 1;
        self.evict_for_capacity();
        Ok(self.insert_session(cloned))
    }

    /// Remove a session explicitly. Returns whether it existed.
    pub fn evict(&mut self, handle: SessionHandle) -> bool {
        self.sessions.remove(&handle.0).is_some()
    }

    /// Full token stream of a session (sealed prefix + tail).
    pub fn all_ids(&mut self, handle: SessionHandle) -> Result<Vec<u32>, StoreError> {
        self.clock += 1;
        let clock = self.clock;
        let sess = self
            .sessions
            .get_mut(&handle.0)
            .ok_or(StoreError::UnknownSession(handle.0))?;
        sess.last_used = clock;
        let out = materialize_all_ids(sess);
        self.count_full_row_materialization();
        Ok(out)
    }

    /// Complete token stream of a session as an immutable shared row.
    ///
    /// The row is cached under the session's ID generation, so repeated
    /// snapshots of an unchanged stream return the same allocation and a
    /// later append can never expose a stale row. When the sealed prefix
    /// and the tail are adjacent ranges of one engine allocation (the
    /// shape an initial seed-and-seal produces), the returned row shares
    /// that allocation without materializing anything.
    pub fn shared_all_ids(&mut self, handle: SessionHandle) -> Result<SharedIds, StoreError> {
        self.clock += 1;
        let clock = self.clock;
        let counter = &self.full_row_materializations;
        let sess = self
            .sessions
            .get_mut(&handle.0)
            .ok_or(StoreError::UnknownSession(handle.0))?;
        sess.last_used = clock;
        if let Some((generation, snapshot)) = &sess.ids_snapshot {
            if *generation == sess.ids_generation {
                return Ok(snapshot.clone());
            }
        }
        let snapshot = match (sess.sealed.as_single_shared(), sess.tail.shared_ids()) {
            (Some(sealed), Some(tail)) if sess.tail.n_tokens() == tail.len() => {
                sealed.join_adjacent(tail)
            }
            (None, Some(tail)) if sess.sealed.is_empty() && sess.tail.n_tokens() == tail.len() => {
                Some(tail.clone())
            }
            _ => None,
        };
        let snapshot = match snapshot {
            Some(shared) => shared,
            None => {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SharedIds::from_vec(materialize_all_ids(sess))
            }
        };
        sess.ids_snapshot = Some((sess.ids_generation, snapshot.clone()));
        Ok(snapshot)
    }

    fn collect_all_ids(&self, handle: SessionHandle) -> Result<Vec<u32>, StoreError> {
        self.count_full_row_materialization();
        Ok(materialize_all_ids(self.session(handle)?))
    }

    /// Current revision of a session.
    pub fn revision(&self, handle: SessionHandle) -> Result<u64, StoreError> {
        Ok(self.session(handle)?.revision)
    }

    /// Introspection snapshot of one session.
    pub fn session_info(&self, handle: SessionHandle) -> Result<SessionInfo, StoreError> {
        let sess = self.session(handle)?;
        Ok(SessionInfo {
            key_id: sess.key_id,
            witness: sess.witness,
            revision: sess.revision,
            total_chars: sess.total_chars,
            safe_char: sess.safe_char,
            stable_prefix_bytes: sess.safe_byte,
            token_count: sess.token_count(),
            sealed_tokens: sess.sealed.len() as u64,
            tail_chars: u64::from(sess.tail.text_chars()),
            tail_bytes: sess.tail.text_bytes() as u64,
            buf_bytes: sess.buf.len() as u64,
            blocks_end: sess.blocks_end,
            chain_ok: sess.chain_ok,
            last_replace_from: sess.last_replace_from,
            approx_bytes: (sess.sealed.len() * 4
                + sess.tail.text_bytes()
                + sess.tail.n_tokens() * 12
                + sess.buf.len()) as u64,
        })
    }

    /// All live handles, ascending.
    pub fn list_handles(&self) -> Vec<SessionHandle> {
        let mut out: Vec<u64> = self.sessions.keys().copied().collect();
        out.sort_unstable();
        out.into_iter().map(SessionHandle).collect()
    }

    /// Counters snapshot.
    pub fn stats(&self) -> StatsSnapshot {
        let s = &self.stats;
        StatsSnapshot {
            format: FORMAT_NAME,
            block_chars: self.cfg.block_chars,
            tail_soft_cap_bytes: self.cfg.tail_soft_cap_bytes as u64,
            tail_hard_cap_bytes: self.cfg.tail_hard_cap_bytes as u64,
            max_sessions: self.cfg.max_sessions as u64,
            session_count: self.sessions.len() as u64,
            node_count: self.nodes.len() as u64,
            puts: s.puts,
            extends: s.extends,
            forks: s.forks,
            lookups: s.lookups,
            lookup_hits: s.lookup_hits,
            lookup_misses: s.lookup_misses,
            hit_rate: if s.lookups > 0 {
                Some(s.lookup_hits as f64 / s.lookups as f64)
            } else {
                None
            },
            checksum_rejects: s.checksum_rejects,
            k_cap_overflows: s.k_cap_overflows,
            hard_cap_degrades: s.hard_cap_degrades,
            seals: s.seals,
            sealed_tokens: s.sealed_tokens,
            sessions_evicted: s.sessions_evicted,
            nodes_skipped_tail_cap: s.nodes_skipped_tail_cap,
            chain_detaches: s.chain_detaches,
            import_rejects: s.import_rejects,
            revision_conflicts: s.revision_conflicts,
            path_counts: s.path_counts.clone(),
        }
    }

    // ---------------------------------------------- record transport --

    /// Sealed node cache as `(node_key, cache_record_bytes)` pairs,
    /// sorted by key. Internal-cache transport: losing nodes only ever
    /// costs lookup hits, never correctness.
    pub fn export_node_items(&self) -> Result<Vec<(BlockHash, Vec<u8>)>, StoreError> {
        let mut keys: Vec<&BlockHash> = self.nodes.keys().collect();
        keys.sort_unstable();
        keys.into_iter()
            .map(|k| Ok((*k, self.nodes[k].rec.to_bytes()?)))
            .collect()
    }

    /// Import one sealed node. Returns `false` (and counts the reject)
    /// when the record fails strict decoding or does not match its key:
    /// prefer miss over wrong; node imports are never loud because a
    /// missing node only ever causes a miss.
    pub fn import_node_item(&mut self, node_key: &[u8], rec: &[u8]) -> bool {
        let Ok(key): Result<BlockHash, _> = node_key.try_into() else {
            self.stats.import_rejects += 1;
            return false;
        };
        match NodeCacheRecord::from_bytes(rec) {
            Ok(parsed) if parsed.key == key => match NodeEntry::create(parsed) {
                Ok(entry) => {
                    self.nodes.insert(key, entry);
                    true
                }
                Err(_) => {
                    self.stats.import_rejects += 1;
                    false
                }
            },
            _ => {
                self.stats.import_rejects += 1;
                false
            }
        }
    }

    /// Serialize one session as a store format v1 record (the portable
    /// full core-stream snapshot).
    pub fn export_session(&self, handle: SessionHandle) -> Result<Vec<u8>, StoreError> {
        let sess = self.session(handle)?;
        let key = self.key(sess.key_id)?;
        self.count_full_row_materialization();
        let ids = materialize_all_ids(sess);
        let rec = SessionRecordV1 {
            fingerprint: key.fingerprint,
            witness: sess.witness,
            revision: sess.revision,
            prev_block_hash: sess.prev_record_hash,
            curr_block_hash: sess.record_hash,
            stable_prefix_bytes: sess.safe_byte,
            replace_token_offset: sess.recorded_replace_offset(),
            ids,
            tail_text: sess.tail.text().to_string(),
        };
        debug_assert_eq!(rec.compute_curr(), sess.record_hash);
        rec.to_bytes()
    }

    /// Return the current private recovery-binding material, or `None`
    /// when the session was imported without its omitted stable-prefix
    /// plaintext. Finalizing clones the SHA state, so future appends keep
    /// updating the same accumulator.
    pub fn recovery_material(
        &self,
        handle: SessionHandle,
    ) -> Result<Option<RecoveryMaterial>, StoreError> {
        let sess = self.session(handle)?;
        let Some(recovery) = &sess.recovery else {
            return Ok(None);
        };
        let represented_bytes = sess
            .safe_byte
            .checked_add(sess.tail.text_bytes() as u64)
            .ok_or_else(|| internal("session text byte length overflow"))?;
        if recovery.text_bytes != represented_bytes {
            return Err(internal(
                "recovery byte length does not match the represented session text",
            ));
        }
        Ok(Some(recovery.material(sess.record_hash)))
    }

    /// Current native content-index row. `None` is fail-closed for a sealed
    /// import that did not present its historical plaintext.
    pub fn content_index_entry(
        &self,
        handle: SessionHandle,
    ) -> Result<Option<ContentIndexEntry>, StoreError> {
        Ok(self
            .session(handle)?
            .content
            .as_ref()
            .map(ContentDigest::entry))
    }

    /// Serialize the byte-identical TKFR-v1 sidecar from the two incremental
    /// digest states without scanning session history.
    pub fn export_recovery_binding(
        &self,
        handle: SessionHandle,
    ) -> Result<Option<Vec<u8>>, StoreError> {
        let sess = self.session(handle)?;
        let (Some(recovery), Some(content)) = (&sess.recovery, &sess.content) else {
            return Ok(None);
        };
        let material = recovery.material(sess.record_hash);
        if material.text_bytes != content.entry().byte_length {
            return Err(internal("recovery and content-index byte lengths disagree"));
        }
        RecoveryBindingV1::from_states(sess.record_hash, material.text_digest, content)
            .to_bytes()
            .map(Some)
    }

    /// Validate that `historical_text` and a TKFR-v1 binding describe the
    /// exact current session revision.
    ///
    /// This is useful to persistence layers that retain the omitted stable
    /// prefix plaintext separately from the portable store record.  It does
    /// not mutate the session and rejects before any caller-visible state is
    /// published.
    pub fn verify_recovery_binding(
        &self,
        handle: SessionHandle,
        historical_text: &str,
        binding_bytes: &[u8],
    ) -> Result<(), StoreError> {
        self.verified_tracking_state(handle, historical_text, binding_bytes)
            .map(|_| ())
    }

    /// Restore incremental recovery and content-index state on a session
    /// that was imported from its record and bookkeeping sidecar.
    ///
    /// The historical text is accepted only when the canonical TKFR-v1
    /// binding, byte length, record hash, digest, and bounded tail all agree.
    /// This lets a persistent owner continue delta-only appends after restart
    /// without weakening the existing fail-closed import gates.
    pub fn restore_tracking_with_binding(
        &mut self,
        handle: SessionHandle,
        historical_text: &str,
        binding_bytes: &[u8],
    ) -> Result<(), StoreError> {
        let (recovery, content) =
            self.verified_tracking_state(handle, historical_text, binding_bytes)?;
        let recovery_tracking = self.recovery_tracking;
        let content_tracking = self.content_tracking;
        let session = self
            .sessions
            .get_mut(&handle.0)
            .ok_or(StoreError::UnknownSession(handle.0))?;
        session.recovery = recovery_tracking.then_some(recovery);
        session.content = content_tracking.then_some(content);
        Ok(())
    }

    fn verified_tracking_state(
        &self,
        handle: SessionHandle,
        historical_text: &str,
        binding_bytes: &[u8],
    ) -> Result<(RecoveryDigest, ContentDigest), StoreError> {
        let session = self.session(handle)?;
        let binding = RecoveryBindingV1::from_bytes(binding_bytes)?;
        let represented_bytes = session
            .safe_byte
            .checked_add(session.tail.text_bytes() as u64)
            .ok_or_else(|| internal("session text byte length overflow"))?;
        let historical_bytes = u64::try_from(historical_text.len())
            .map_err(|_| StoreError::InvalidInput("historical text exceeds u64 bytes".into()))?;
        let recovery = RecoveryDigest::from_text(historical_text)?;
        let material = recovery.material(session.record_hash);
        let content = ContentDigest::from_bytes(historical_text.as_bytes())?;
        let valid = constant_time_eq(&binding.record_hash, &session.record_hash)
            && historical_bytes == represented_bytes
            && historical_bytes == binding.index_entry.byte_length
            && historical_text.ends_with(session.tail.text())
            && constant_time_eq(&binding.text_digest, &material.text_digest)
            && binding.index_entry == content.entry();
        if !valid {
            return Err(StoreError::MalformedRecord(
                "TKFR-v1 does not bind historical text to the current session".into(),
            ));
        }
        Ok((recovery, content))
    }

    /// Serialize the internal bookkeeping sidecar of one session (bound
    /// to the record exported by [`Self::export_session`] at the same
    /// revision). Importing record + sidecar restores exact pre-save
    /// behavior including block-chain continuation; the record alone
    /// restores a fully correct session with a detached chain.
    pub fn export_session_sidecar(&self, handle: SessionHandle) -> Result<Vec<u8>, StoreError> {
        let sess = self.session(handle)?;
        let sc = SessionSidecar {
            record_curr_hash: sess.record_hash,
            total_chars: sess.total_chars,
            safe_char: sess.safe_char,
            safe_byte: sess.safe_byte,
            blocks_end: sess.blocks_end,
            blocks_end_byte: sess.blocks_end_byte,
            scan_floor: sess.scan_floor,
            chain_base_safe: sess.chain_base_safe,
            chain_base_idx: sess.chain_base_idx,
            chain_tail: sess.chain_tail,
            chain_ok: sess.chain_ok,
            sealed_count: sess.sealed.len() as u64,
            last_replace_from: sess.last_replace_from,
            buf: sess.buf.clone(),
            seal_log: sess.seal_log.clone(),
        };
        sc.to_bytes()
    }

    /// Shared import core: strict decode, fingerprint and witness
    /// gates, tail re-encode verification, sealed/tail split recovery.
    fn import_record_gates(
        &mut self,
        key: &KeyCtx,
        rec_bytes: &[u8],
        engine: &dyn SessionEncoder,
    ) -> Result<(SessionRecordV1, usize, TailState), StoreError> {
        let parsed = match SessionRecordV1::from_bytes(rec_bytes) {
            Ok(p) => p,
            Err(e) => {
                self.stats.import_rejects += 1;
                return Err(e);
            }
        };
        if parsed.fingerprint != key.fingerprint {
            self.stats.import_rejects += 1;
            return Err(StoreError::FingerprintMismatch);
        }
        if parsed.witness != engine.witness_category() {
            self.stats.import_rejects += 1;
            return Err(StoreError::WitnessCategoryMismatch {
                recorded: parsed.witness,
                engine: engine.witness_category(),
            });
        }
        // Re-encode the tail and require bit-equality with the recorded
        // tail portion of the stream (prefer miss over wrong).
        let enc = engine.encode(&parsed.tail_text)?;
        let n = parsed.ids.len();
        let m = enc.ids.len();
        let sealed_count = match n.checked_sub(m) {
            Some(s) => s,
            None => {
                self.stats.import_rejects += 1;
                return Err(StoreError::ImportReencodeMismatch);
            }
        };
        if parsed.ids[sealed_count..] != enc.ids[..] {
            self.stats.import_rejects += 1;
            return Err(StoreError::ImportReencodeMismatch);
        }
        // Sealed ids exist exactly when a stable prefix exists.
        if (sealed_count == 0) != (parsed.stable_prefix_bytes == 0) {
            self.stats.import_rejects += 1;
            return Err(StoreError::MalformedRecord(
                "sealed token count and stable prefix length disagree".into(),
            ));
        }
        let mut tail = TailState::new();
        tail.fill(&parsed.tail_text, enc)?;
        Ok((parsed, sealed_count, tail))
    }

    /// Deserialize a session from a bare format v1 record.
    ///
    /// The session is fully correct (ids, revision chain, byte-unit
    /// stable prefix accounting) but conservative: the block chain is
    /// detached and character counters restart at the tail origin, so
    /// the imported session stops feeding the prefix-sharing cache.
    /// Use [`Self::import_session_with_sidecar`] to restore exact
    /// pre-save behavior. Rejections are counted and returned as errors
    /// - session imports are loud.
    pub fn import_session(
        &mut self,
        key_id: KeyId,
        rec_bytes: &[u8],
        engine: &dyn SessionEncoder,
    ) -> Result<SessionHandle, StoreError> {
        let key = self.key(key_id)?;
        let (parsed, sealed_count, tail) = self.import_record_gates(&key, rec_bytes, engine)?;
        let recovery = if self.recovery_tracking && parsed.stable_prefix_bytes == 0 {
            Some(RecoveryDigest::from_text(&parsed.tail_text)?)
        } else {
            None
        };
        let content = if self.content_tracking && parsed.stable_prefix_bytes == 0 {
            Some(ContentDigest::from_bytes(parsed.tail_text.as_bytes())?)
        } else {
            None
        };
        self.clock += 1;
        let clock = self.clock;
        self.evict_for_capacity();
        let sess = detached_import_session(
            key_id,
            &parsed,
            sealed_count,
            tail,
            clock,
            recovery,
            content,
        );
        Ok(self.insert_session(sess))
    }

    /// Deserialize a session and initialize its incremental recovery
    /// digest from caller-presented historical text.
    ///
    /// This is the only import path that may restore recovery state for a
    /// sealed record. The text must bind to the exact record revision,
    /// byte length, digest, and bounded tail before a handle is inserted.
    pub fn import_session_with_recovery(
        &mut self,
        key_id: KeyId,
        rec_bytes: &[u8],
        historical_text: &str,
        expected: &RecoveryMaterial,
        engine: &dyn SessionEncoder,
    ) -> Result<SessionHandle, StoreError> {
        let key = self.key(key_id)?;
        let (parsed, sealed_count, tail) = self.import_record_gates(&key, rec_bytes, engine)?;
        let historical_bytes = u64::try_from(historical_text.len())
            .map_err(|_| StoreError::InvalidInput("historical text exceeds u64 bytes".into()))?;
        let full_text_bytes = parsed
            .stable_prefix_bytes
            .checked_add(parsed.tail_text.len() as u64)
            .ok_or_else(|| internal("record full-text byte length overflow"))?;
        let recovery = RecoveryDigest::from_text(historical_text)?;
        let material = recovery.material(parsed.curr_block_hash);
        let valid = constant_time_eq(&expected.record_hash, &parsed.curr_block_hash)
            && historical_bytes == expected.text_bytes
            && historical_bytes == full_text_bytes
            && historical_text.ends_with(&parsed.tail_text)
            && constant_time_eq(&expected.text_digest, &material.text_digest);
        if !valid {
            self.stats.import_rejects += 1;
            return Err(StoreError::MalformedRecord(
                "recovery material does not bind historical text to the record".into(),
            ));
        }
        self.clock += 1;
        let clock = self.clock;
        self.evict_for_capacity();
        let sess = detached_import_session(
            key_id,
            &parsed,
            sealed_count,
            tail,
            clock,
            Some(recovery),
            self.content_tracking
                .then(|| ContentDigest::from_bytes(historical_text.as_bytes()))
                .transpose()?,
        );
        Ok(self.insert_session(sess))
    }

    /// Import a session and validate its canonical TKFR-v1 binding inside
    /// the same native operation. The caller presents historical text once;
    /// no Python checkpoint or digest pass participates.
    pub fn import_session_with_binding(
        &mut self,
        key_id: KeyId,
        rec_bytes: &[u8],
        historical_text: &str,
        binding_bytes: &[u8],
        engine: &dyn SessionEncoder,
    ) -> Result<SessionHandle, StoreError> {
        let key = self.key(key_id)?;
        let (parsed, sealed_count, tail) = self.import_record_gates(&key, rec_bytes, engine)?;
        let binding = match RecoveryBindingV1::from_bytes(binding_bytes) {
            Ok(binding) => binding,
            Err(error) => {
                self.stats.import_rejects += 1;
                return Err(error);
            }
        };
        let historical_bytes = u64::try_from(historical_text.len())
            .map_err(|_| StoreError::InvalidInput("historical text exceeds u64 bytes".into()))?;
        let full_text_bytes = parsed
            .stable_prefix_bytes
            .checked_add(parsed.tail_text.len() as u64)
            .ok_or_else(|| internal("record full-text byte length overflow"))?;
        let recovery = RecoveryDigest::from_text(historical_text)?;
        let material = recovery.material(parsed.curr_block_hash);
        let content = ContentDigest::from_bytes(historical_text.as_bytes())?;
        let valid = constant_time_eq(&binding.record_hash, &parsed.curr_block_hash)
            && historical_bytes == binding.index_entry.byte_length
            && historical_bytes == full_text_bytes
            && historical_text.ends_with(&parsed.tail_text)
            && constant_time_eq(&binding.text_digest, &material.text_digest)
            && binding.index_entry == content.entry();
        if !valid {
            self.stats.import_rejects += 1;
            return Err(StoreError::MalformedRecord(
                "TKFR-v1 does not bind historical text to the record".into(),
            ));
        }
        self.clock += 1;
        let clock = self.clock;
        self.evict_for_capacity();
        let sess = detached_import_session(
            key_id,
            &parsed,
            sealed_count,
            tail,
            clock,
            self.recovery_tracking.then_some(recovery),
            self.content_tracking.then_some(content),
        );
        Ok(self.insert_session(sess))
    }

    /// Recovery import from a caller's complete current transcript. Rust
    /// slices the historical prefix at the binding's byte length and rejects
    /// a non-UTF-8 boundary; Python need not allocate or scan a second copy.
    pub fn import_session_with_binding_candidate(
        &mut self,
        key_id: KeyId,
        rec_bytes: &[u8],
        candidate_text: &str,
        binding_bytes: &[u8],
        engine: &dyn SessionEncoder,
    ) -> Result<(SessionHandle, usize), StoreError> {
        let binding = RecoveryBindingV1::from_bytes(binding_bytes)?;
        let prefix_len = usize::try_from(binding.index_entry.byte_length)
            .map_err(|_| StoreError::InvalidInput("historical text exceeds usize".into()))?;
        let prefix_bytes = candidate_text
            .as_bytes()
            .get(..prefix_len)
            .ok_or_else(|| StoreError::InvalidInput("candidate is shorter than history".into()))?;
        let historical_text = std::str::from_utf8(prefix_bytes).map_err(|_| {
            StoreError::InvalidInput("historical byte length is not a UTF-8 boundary".into())
        })?;
        let historical_chars = historical_text.chars().count();
        let handle = self.import_session_with_binding(
            key_id,
            rec_bytes,
            historical_text,
            binding_bytes,
            engine,
        )?;
        Ok((handle, historical_chars))
    }

    /// Deserialize a session from a format v1 record plus its
    /// bookkeeping sidecar, restoring exact pre-save behavior. The
    /// sidecar must verify and must bind to this exact record (its
    /// stored `record_curr_hash` equals the record's chain link hash);
    /// any inconsistency is a loud, counted rejection.
    pub fn import_session_with_sidecar(
        &mut self,
        key_id: KeyId,
        rec_bytes: &[u8],
        sidecar_bytes: &[u8],
        engine: &dyn SessionEncoder,
    ) -> Result<SessionHandle, StoreError> {
        let key = self.key(key_id)?;
        let (parsed, sealed_count, tail) = self.import_record_gates(&key, rec_bytes, engine)?;
        let sc = match SessionSidecar::from_bytes(sidecar_bytes) {
            Ok(s) => s,
            Err(e) => {
                self.stats.import_rejects += 1;
                return Err(e);
            }
        };
        let tail_chars = u64::from(tail.text_chars());
        let consistent = sc.record_curr_hash == parsed.curr_block_hash
            && sc.sealed_count == sealed_count as u64
            && sc.safe_byte == parsed.stable_prefix_bytes
            && sc.total_chars == sc.safe_char + tail_chars
            && sc.chain_base_idx <= sealed_count as u64;
        if !consistent {
            self.stats.import_rejects += 1;
            return Err(StoreError::MalformedRecord(
                "session sidecar does not bind to the record".into(),
            ));
        }
        let recovery = if self.recovery_tracking && parsed.stable_prefix_bytes == 0 {
            Some(RecoveryDigest::from_text(&parsed.tail_text)?)
        } else {
            None
        };
        let content = if self.content_tracking && parsed.stable_prefix_bytes == 0 {
            Some(ContentDigest::from_bytes(parsed.tail_text.as_bytes())?)
        } else {
            None
        };
        self.clock += 1;
        let clock = self.clock;
        self.evict_for_capacity();
        let mut sess = Session::fresh(key_id, parsed.witness, clock);
        sess.revision = parsed.revision;
        sess.prev_record_hash = parsed.prev_block_hash;
        sess.record_hash = parsed.curr_block_hash;
        sess.last_replace_from = sc.last_replace_from;
        sess.total_chars = sc.total_chars;
        sess.safe_char = sc.safe_char;
        sess.safe_byte = sc.safe_byte;
        sess.sealed_payload.update_ids(&parsed.ids[..sealed_count]);
        sess.sealed = SealedIds::from_vec(parsed.ids[..sealed_count].to_vec());
        sess.blocks_end = sc.blocks_end;
        sess.blocks_end_byte = sc.blocks_end_byte;
        sess.buf = sc.buf;
        sess.chain_tail = sc.chain_tail;
        sess.chain_base_safe = sc.chain_base_safe;
        sess.chain_base_idx = sc.chain_base_idx;
        sess.chain_ok = sc.chain_ok;
        sess.seal_log = sc.seal_log;
        sess.scan_floor = sc.scan_floor;
        sess.tail = tail;
        sess.recovery = recovery;
        sess.content = content;
        Ok(self.insert_session(sess))
    }

    /// Test support only (corruption-injection batteries): flip one bit
    /// in a stored node's fields without updating its captured checksum,
    /// so the next lookup must reject it. Returns whether the node
    /// existed.
    #[cfg(any(test, feature = "testing"))]
    pub fn corrupt_node_for_tests(&mut self, node_key: &[u8]) -> Result<bool, StoreError> {
        let Ok(key): Result<BlockHash, _> = node_key.try_into() else {
            return Err(StoreError::InvalidInput("node_key must be 32 bytes".into()));
        };
        let Some(entry) = self.nodes.get_mut(&key) else {
            return Ok(false);
        };
        if let Some(v) = entry.rec.ids.first_mut() {
            *v ^= 1;
        } else {
            entry.rec.safe_char ^= 1;
        }
        Ok(true)
    }
}

/// Materialize the complete token stream (sealed parts then tail).
fn materialize_all_ids(sess: &Session) -> Vec<u32> {
    let mut out = Vec::with_capacity(sess.sealed.len() + sess.tail.n_tokens());
    for part in sess.sealed.parts() {
        out.extend_from_slice(part);
    }
    out.extend_from_slice(sess.tail.ids());
    out
}

fn detached_import_session(
    key_id: KeyId,
    parsed: &SessionRecordV1,
    sealed_count: usize,
    tail: TailState,
    clock: u64,
    recovery: Option<RecoveryDigest>,
    content: Option<ContentDigest>,
) -> Session {
    let mut sess = Session::fresh(key_id, parsed.witness, clock);
    sess.revision = parsed.revision;
    sess.prev_record_hash = parsed.prev_block_hash;
    sess.record_hash = parsed.curr_block_hash;
    sess.last_replace_from = parsed.replace_token_offset;
    sess.safe_byte = parsed.stable_prefix_bytes;
    sess.safe_char = 0; // character origin restarts at the tail
    sess.total_chars = u64::from(tail.text_chars());
    sess.sealed_payload.update_ids(&parsed.ids[..sealed_count]);
    sess.sealed = SealedIds::from_vec(parsed.ids[..sealed_count].to_vec());
    sess.chain_ok = false; // conservative: no chain continuation
    sess.buf = String::new();
    sess.seal_log = vec![(0, sealed_count as u64)];
    sess.tail = tail;
    sess.recovery = recovery;
    sess.content = content;
    sess
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}

/// UTF-8 byte offset at a Unicode-scalar boundary. The store's public
/// character counters use scalar values, while recovery integrity binds
/// the exact UTF-8 bytes.
fn byte_offset_at_char(text: &str, char_index: u64) -> Result<usize, StoreError> {
    let target =
        usize::try_from(char_index).map_err(|_| internal("character offset exceeds usize"))?;
    if target == 0 {
        return Ok(0);
    }
    if let Some((offset, _)) = text.char_indices().nth(target) {
        return Ok(offset);
    }
    if text.chars().count() == target {
        return Ok(text.len());
    }
    Err(internal("character offset exceeds text length"))
}

/// Structural verification of an engine's append claim (see the
/// [`SessionEncoder`] contract): text grew by exactly `delta`, and the
/// claimed kept-token prefix is bit-identical to the old encoding. The
/// certified content guarantee itself lives with the engine; these
/// checks make a misbehaving engine loud instead of silently corrupting
/// the store's `replace_from` invariant.
fn verify_append_shape(
    tail: &TailState,
    old_text_len: usize,
    delta: &str,
    old_ids: &[u32],
    kept_tokens: usize,
) -> Result<(), StoreError> {
    if delta.is_empty() {
        return Ok(());
    }
    let want_len = old_text_len
        .checked_add(delta.len())
        .ok_or_else(|| internal("text length overflow"))?;
    if tail.text_bytes() != want_len || !tail.text().ends_with(delta) {
        return Err(StoreError::Engine(
            "append contract violation: tail text is not old text + delta".into(),
        ));
    }
    if kept_tokens > old_ids.len()
        || kept_tokens > tail.n_tokens()
        || tail.ids()[..kept_tokens] != old_ids[..kept_tokens]
    {
        return Err(StoreError::Engine(
            "append contract violation: kept token prefix does not match".into(),
        ));
    }
    Ok(())
}
