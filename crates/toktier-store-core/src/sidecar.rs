//! Internal cache structures next to the frozen record format.
//!
//! Store format v1 (`format.rs`) freezes the *portable* session record:
//! a full core-stream snapshot per revision. Two store-internal
//! structures intentionally live outside that contract and are encoded
//! here instead:
//!
//! * [`NodeCacheRecord`] -- one sealed block of the prefix-sharing hash
//!   chain. Node entries hold a *sealed-ids delta* per text block (not a
//!   full-stream snapshot); they are an acceleration cache for
//!   `lookup`, derivable only at seal time and never required for
//!   correctness -- losing them costs hits, never wrong results.
//! * [`SessionSidecar`] -- the incremental bookkeeping a live session
//!   carries beyond the portable record (character-unit counters, block
//!   chain attachment, seal log, pending block buffer). A session
//!   imported from a bare format v1 record is fully correct but starts
//!   with a detached chain; importing record + sidecar restores exact
//!   pre-save behavior. The sidecar binds to its record through the
//!   record's `curr_block_hash`.
//!
//! Both encodings follow the same decode discipline as the frozen
//! format (checked arithmetic, strict bounds, trailing 32-byte SHA-256
//! checksum with a dedicated domain tag, exact consumption), but they
//! are internal: their layout may change with the store implementation,
//! they never travel between implementations, and corruption in them is
//! detected and rejected exactly like record corruption.

use sha2::{Digest, Sha256};

use crate::engine::WitnessCategory;
use crate::error::StoreError;
use crate::format::{BlockHash, ZERO_HASH};

const NODE_MAGIC: [u8; 4] = *b"TKNC";
const SIDECAR_MAGIC: [u8; 4] = *b"TKSS";
const CACHE_VERSION: u16 = 1;
const CHECKSUM_LEN: usize = 32;
const DOMAIN_NODE_KEY: &[u8] = b"toktier.store.v1.node\0";
const DOMAIN_NODE_CACHE: &[u8] = b"toktier.store.v1.nodecache\0";
const DOMAIN_SIDECAR: &[u8] = b"toktier.store.v1.sidecar\0";

fn malformed(msg: impl Into<String>) -> StoreError {
    StoreError::MalformedRecord(msg.into())
}

fn len_u32(n: usize, what: &str) -> Result<u32, StoreError> {
    u32::try_from(n).map_err(|_| StoreError::InvalidInput(format!("{what} length {n} exceeds u32")))
}

fn checksum(domain: &[u8], body: &[u8]) -> BlockHash {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(body);
    h.finalize().into()
}

/// Content address of a sealed chain node. The semantic fingerprint
/// participates, so a lookup under a different fingerprint can never
/// structurally hit (wrong key must miss).
pub fn node_key(
    parent: Option<&BlockHash>,
    fingerprint: &[u8; 32],
    block_index: u64,
    block_bytes: &[u8],
) -> BlockHash {
    let mut h = Sha256::new();
    h.update(DOMAIN_NODE_KEY);
    match parent {
        Some(p) => {
            h.update([1u8]);
            h.update(p);
        }
        None => {
            h.update([0u8]);
            h.update(ZERO_HASH);
        }
    }
    h.update(block_index.to_le_bytes());
    h.update(fingerprint);
    h.update(block_bytes);
    h.finalize().into()
}

// ------------------------------------------------------ shared reader --

struct Reader<'a> {
    buf: &'a [u8],
    ix: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, ix: 0 }
    }

    fn take(&mut self, n: usize, context: &'static str) -> Result<&'a [u8], StoreError> {
        let end = self
            .ix
            .checked_add(n)
            .ok_or(StoreError::Truncated { context })?;
        if end > self.buf.len() {
            return Err(StoreError::Truncated { context });
        }
        let out = &self.buf[self.ix..end];
        self.ix = end;
        Ok(out)
    }

    fn u8(&mut self, context: &'static str) -> Result<u8, StoreError> {
        Ok(self.take(1, context)?[0])
    }

    fn u16(&mut self, context: &'static str) -> Result<u16, StoreError> {
        Ok(u16::from_le_bytes(
            self.take(2, context)?.try_into().expect("fixed slice"),
        ))
    }

    fn u32(&mut self, context: &'static str) -> Result<u32, StoreError> {
        Ok(u32::from_le_bytes(
            self.take(4, context)?.try_into().expect("fixed slice"),
        ))
    }

    fn u64(&mut self, context: &'static str) -> Result<u64, StoreError> {
        Ok(u64::from_le_bytes(
            self.take(8, context)?.try_into().expect("fixed slice"),
        ))
    }

    fn hash(&mut self, context: &'static str) -> Result<BlockHash, StoreError> {
        Ok(self
            .take(CHECKSUM_LEN, context)?
            .try_into()
            .expect("fixed slice"))
    }

    fn ids(&mut self, n: u32, context: &'static str) -> Result<Vec<u32>, StoreError> {
        let total = (n as usize)
            .checked_mul(4)
            .ok_or(StoreError::Truncated { context })?;
        let raw = self.take(total, context)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("fixed slice")))
            .collect())
    }

    fn utf8(&mut self, n: u32, context: &'static str) -> Result<String, StoreError> {
        let raw = self.take(n as usize, context)?;
        String::from_utf8(raw.to_vec())
            .map_err(|e| malformed(format!("invalid UTF-8 in {context}: {e}")))
    }

    fn finish(&self) -> Result<(), StoreError> {
        if self.ix != self.buf.len() {
            return Err(malformed(format!(
                "{} trailing bytes",
                self.buf.len() - self.ix
            )));
        }
        Ok(())
    }
}

fn push_ids(w: &mut Vec<u8>, ids: &[u32]) {
    for &v in ids {
        w.extend_from_slice(&v.to_le_bytes());
    }
}

fn opt_hash_tagged(w: &mut Vec<u8>, h: Option<&BlockHash>) {
    match h {
        Some(p) => {
            w.push(1);
            w.extend_from_slice(p);
        }
        None => {
            w.push(0);
            w.extend_from_slice(&ZERO_HASH);
        }
    }
}

fn read_opt_hash(
    r: &mut Reader<'_>,
    context: &'static str,
) -> Result<Option<BlockHash>, StoreError> {
    let tag = r.u8(context)?;
    let raw = r.hash(context)?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(raw)),
        other => Err(malformed(format!("bad option tag {other} in {context}"))),
    }
}

// ---------------------------------------------------------- node cache --

/// One sealed chain node (prefix-sharing cache entry).
///
/// The node identifies the text prefix `[0, end_char)` of its chain
/// under its fingerprint. `ids` is the sealed delta covering characters
/// `[parent.safe_char, safe_char)`; `text_tail` is the raw text
/// `[safe_char, end_char)`; `ids_base` is the cumulative sealed token
/// count at the parent's safe point; `end_byte` is the UTF-8 byte
/// length of `[0, end_char)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCacheRecord {
    pub fingerprint: [u8; 32],
    pub witness: WitnessCategory,
    pub parent: Option<BlockHash>,
    pub key: BlockHash,
    pub block_index: u64,
    pub end_char: u64,
    pub safe_char: u64,
    pub ids_base: u64,
    pub end_byte: u64,
    pub ids: Vec<u32>,
    pub text_tail: String,
}

impl NodeCacheRecord {
    fn body(&self) -> Result<Vec<u8>, StoreError> {
        let n_ids = len_u32(self.ids.len(), "node ids")?;
        let tail_len = len_u32(self.text_tail.len(), "node text tail")?;
        let mut w = Vec::with_capacity(160 + self.ids.len() * 4 + self.text_tail.len());
        w.extend_from_slice(&NODE_MAGIC);
        w.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        w.extend_from_slice(&self.witness.as_u16().to_le_bytes());
        w.extend_from_slice(&self.fingerprint);
        opt_hash_tagged(&mut w, self.parent.as_ref());
        w.extend_from_slice(&self.key);
        w.extend_from_slice(&self.block_index.to_le_bytes());
        w.extend_from_slice(&self.end_char.to_le_bytes());
        w.extend_from_slice(&self.safe_char.to_le_bytes());
        w.extend_from_slice(&self.ids_base.to_le_bytes());
        w.extend_from_slice(&self.end_byte.to_le_bytes());
        w.extend_from_slice(&n_ids.to_le_bytes());
        w.extend_from_slice(&tail_len.to_le_bytes());
        push_ids(&mut w, &self.ids);
        w.extend_from_slice(self.text_tail.as_bytes());
        Ok(w)
    }

    /// Serialize with the trailing domain-tagged checksum.
    pub fn to_bytes(&self) -> Result<Vec<u8>, StoreError> {
        let mut w = self.body()?;
        let cs = checksum(DOMAIN_NODE_CACHE, &w);
        w.extend_from_slice(&cs);
        Ok(w)
    }

    /// Strict decode; every check failure is a rejection.
    pub fn from_bytes(bytes: &[u8]) -> Result<NodeCacheRecord, StoreError> {
        if bytes.len() < CHECKSUM_LEN {
            return Err(StoreError::Truncated {
                context: "node cache",
            });
        }
        let (body, cs) = bytes.split_at(bytes.len() - CHECKSUM_LEN);
        if checksum(DOMAIN_NODE_CACHE, body) != *cs {
            return Err(StoreError::ChecksumMismatch);
        }
        let mut r = Reader::new(body);
        if r.take(4, "node magic")? != NODE_MAGIC {
            return Err(StoreError::BadMagic);
        }
        let version = r.u16("node version")?;
        if version != CACHE_VERSION {
            return Err(StoreError::UnsupportedFormatVersion(version));
        }
        let witness = WitnessCategory::from_u16(r.u16("node witness")?)?;
        let fingerprint: [u8; 32] = r.take(32, "node fingerprint")?.try_into().expect("fixed");
        let parent = read_opt_hash(&mut r, "node parent")?;
        let key = r.hash("node key")?;
        let block_index = r.u64("node block_index")?;
        let end_char = r.u64("node end_char")?;
        let safe_char = r.u64("node safe_char")?;
        let ids_base = r.u64("node ids_base")?;
        let end_byte = r.u64("node end_byte")?;
        let n_ids = r.u32("node id count")?;
        let tail_len = r.u32("node tail length")?;
        let ids = r.ids(n_ids, "node ids")?;
        let text_tail = r.utf8(tail_len, "node text tail")?;
        r.finish()?;
        if safe_char > end_char {
            return Err(malformed("node safe_char exceeds end_char"));
        }
        if u64::from(tail_len) > end_byte {
            return Err(malformed("node text tail longer than end_byte"));
        }
        if ids_base.checked_add(u64::from(n_ids)).is_none() {
            return Err(malformed("node token count overflow"));
        }
        Ok(NodeCacheRecord {
            fingerprint,
            witness,
            parent,
            key,
            block_index,
            end_char,
            safe_char,
            ids_base,
            end_byte,
            ids,
            text_tail,
        })
    }
}

// ------------------------------------------------------ session sidecar --

/// Incremental bookkeeping of one session beyond the portable record.
///
/// Bound to a specific format v1 record through `record_curr_hash`; a
/// sidecar presented with a different record is rejected. All character
/// counters are Unicode scalar values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSidecar {
    /// `curr_block_hash` of the record this sidecar belongs to.
    pub record_curr_hash: BlockHash,
    pub total_chars: u64,
    pub safe_char: u64,
    pub safe_byte: u64,
    pub blocks_end: u64,
    pub blocks_end_byte: u64,
    pub scan_floor: u64,
    pub chain_base_safe: u64,
    pub chain_base_idx: u64,
    pub chain_tail: Option<BlockHash>,
    pub chain_ok: bool,
    /// Number of leading tokens of the record's stream that are sealed.
    pub sealed_count: u64,
    /// Raw `replace_from` of the last append (the record clamps this to
    /// 0 under witness category 0).
    pub last_replace_from: u64,
    /// Raw text `[blocks_end, max(blocks_end, safe_char))`.
    pub buf: String,
    /// Ascending `(safe_char, cumulative sealed ids)` seal points.
    pub seal_log: Vec<(u64, u64)>,
}

impl SessionSidecar {
    fn body(&self) -> Result<Vec<u8>, StoreError> {
        let buf_len = len_u32(self.buf.len(), "sidecar buf")?;
        let n_log = len_u32(self.seal_log.len(), "seal log")?;
        let mut w = Vec::with_capacity(200 + self.buf.len() + self.seal_log.len() * 16);
        w.extend_from_slice(&SIDECAR_MAGIC);
        w.extend_from_slice(&CACHE_VERSION.to_le_bytes());
        w.extend_from_slice(&self.record_curr_hash);
        w.extend_from_slice(&self.total_chars.to_le_bytes());
        w.extend_from_slice(&self.safe_char.to_le_bytes());
        w.extend_from_slice(&self.safe_byte.to_le_bytes());
        w.extend_from_slice(&self.blocks_end.to_le_bytes());
        w.extend_from_slice(&self.blocks_end_byte.to_le_bytes());
        w.extend_from_slice(&self.scan_floor.to_le_bytes());
        w.extend_from_slice(&self.chain_base_safe.to_le_bytes());
        w.extend_from_slice(&self.chain_base_idx.to_le_bytes());
        opt_hash_tagged(&mut w, self.chain_tail.as_ref());
        w.push(u8::from(self.chain_ok));
        w.extend_from_slice(&self.sealed_count.to_le_bytes());
        w.extend_from_slice(&self.last_replace_from.to_le_bytes());
        w.extend_from_slice(&buf_len.to_le_bytes());
        w.extend_from_slice(&n_log.to_le_bytes());
        w.extend_from_slice(self.buf.as_bytes());
        for &(c, i) in &self.seal_log {
            w.extend_from_slice(&c.to_le_bytes());
            w.extend_from_slice(&i.to_le_bytes());
        }
        Ok(w)
    }

    /// Serialize with the trailing domain-tagged checksum.
    pub fn to_bytes(&self) -> Result<Vec<u8>, StoreError> {
        let mut w = self.body()?;
        let cs = checksum(DOMAIN_SIDECAR, &w);
        w.extend_from_slice(&cs);
        Ok(w)
    }

    /// Strict decode; every check failure is a rejection.
    pub fn from_bytes(bytes: &[u8]) -> Result<SessionSidecar, StoreError> {
        if bytes.len() < CHECKSUM_LEN {
            return Err(StoreError::Truncated {
                context: "session sidecar",
            });
        }
        let (body, cs) = bytes.split_at(bytes.len() - CHECKSUM_LEN);
        if checksum(DOMAIN_SIDECAR, body) != *cs {
            return Err(StoreError::ChecksumMismatch);
        }
        let mut r = Reader::new(body);
        if r.take(4, "sidecar magic")? != SIDECAR_MAGIC {
            return Err(StoreError::BadMagic);
        }
        let version = r.u16("sidecar version")?;
        if version != CACHE_VERSION {
            return Err(StoreError::UnsupportedFormatVersion(version));
        }
        let record_curr_hash = r.hash("sidecar record binding")?;
        let total_chars = r.u64("sidecar total_chars")?;
        let safe_char = r.u64("sidecar safe_char")?;
        let safe_byte = r.u64("sidecar safe_byte")?;
        let blocks_end = r.u64("sidecar blocks_end")?;
        let blocks_end_byte = r.u64("sidecar blocks_end_byte")?;
        let scan_floor = r.u64("sidecar scan_floor")?;
        let chain_base_safe = r.u64("sidecar chain_base_safe")?;
        let chain_base_idx = r.u64("sidecar chain_base_idx")?;
        let chain_tail = read_opt_hash(&mut r, "sidecar chain_tail")?;
        let chain_ok = match r.u8("sidecar chain_ok")? {
            0 => false,
            1 => true,
            other => return Err(malformed(format!("bad chain_ok flag {other}"))),
        };
        let sealed_count = r.u64("sidecar sealed_count")?;
        let last_replace_from = r.u64("sidecar last_replace_from")?;
        let buf_len = r.u32("sidecar buf length")?;
        let n_log = r.u32("seal log count")?;
        let buf = r.utf8(buf_len, "sidecar buf")?;
        let mut seal_log = Vec::with_capacity(n_log as usize);
        for _ in 0..n_log {
            let c = r.u64("seal log entry")?;
            let i = r.u64("seal log entry")?;
            seal_log.push((c, i));
        }
        r.finish()?;
        if safe_char > total_chars {
            return Err(malformed("sidecar safe_char exceeds total_chars"));
        }
        if blocks_end > total_chars {
            return Err(malformed("sidecar blocks_end exceeds total_chars"));
        }
        Ok(SessionSidecar {
            record_curr_hash,
            total_chars,
            safe_char,
            safe_byte,
            blocks_end,
            blocks_end_byte,
            scan_floor,
            chain_base_safe,
            chain_base_idx,
            chain_tail,
            chain_ok,
            sealed_count,
            last_replace_from,
            buf,
            seal_log,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_node() -> NodeCacheRecord {
        NodeCacheRecord {
            fingerprint: [7u8; 32],
            witness: WitnessCategory::BpeSyncTransition,
            parent: Some([9u8; 32]),
            key: [3u8; 32],
            block_index: 3,
            end_char: 16384,
            safe_char: 16000,
            ids_base: 120,
            end_byte: 16500,
            ids: vec![1, 2, 3, u32::MAX],
            text_tail: "tail with unicode: \u{4f60}\u{597d} caf\u{e9}\r\n".to_string(),
        }
    }

    fn sample_sidecar() -> SessionSidecar {
        SessionSidecar {
            record_curr_hash: [4u8; 32],
            total_chars: 5000,
            safe_char: 4096,
            safe_byte: 4100,
            blocks_end: 4096,
            blocks_end_byte: 4100,
            scan_floor: 10,
            chain_base_safe: 4000,
            chain_base_idx: 900,
            chain_tail: Some([8u8; 32]),
            chain_ok: true,
            sealed_count: 903,
            last_replace_from: 17,
            buf: String::new(),
            seal_log: vec![(0, 0), (4096, 903)],
        }
    }

    #[test]
    fn node_roundtrip_and_corruption() {
        let rec = sample_node();
        let bytes = rec.to_bytes().unwrap();
        assert_eq!(NodeCacheRecord::from_bytes(&bytes).unwrap(), rec);
        for ix in 0..bytes.len() {
            let mut bad = bytes.clone();
            bad[ix] ^= 1;
            assert!(
                NodeCacheRecord::from_bytes(&bad).is_err(),
                "flip at {ix} accepted"
            );
        }
        assert!(NodeCacheRecord::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn sidecar_roundtrip_and_corruption() {
        let rec = sample_sidecar();
        let bytes = rec.to_bytes().unwrap();
        assert_eq!(SessionSidecar::from_bytes(&bytes).unwrap(), rec);
        for ix in (0..bytes.len()).step_by(3) {
            let mut bad = bytes.clone();
            bad[ix] ^= 1;
            assert!(
                SessionSidecar::from_bytes(&bad).is_err(),
                "flip at {ix} accepted"
            );
        }
        assert!(SessionSidecar::from_bytes(&bytes[..10]).is_err());
    }

    #[test]
    fn node_key_depends_on_all_inputs() {
        let fp1 = [1u8; 32];
        let fp2 = [2u8; 32];
        let root = node_key(None, &fp1, 0, b"block");
        assert_ne!(root, node_key(None, &fp2, 0, b"block"));
        assert_ne!(root, node_key(None, &fp1, 1, b"block"));
        assert_ne!(root, node_key(None, &fp1, 0, b"blocj"));
        assert_ne!(root, node_key(Some(&root), &fp1, 0, b"block"));
        assert_eq!(root, node_key(None, &fp1, 0, b"block"));
    }
}
