//! Store format v1 codec -- the frozen byte-level contract.
//!
//! Normative reference: `docs/contracts/store-format-v1.md`. This
//! module implements that document exactly: the 200-byte fixed header,
//! the optional TLV header extension, the payload layout (full core
//! token stream + text tail), the three SHA-256 constructions (payload
//! digest, chain link, record checksum), and the normative decode order
//! with its rejection discipline. All multi-byte integers are
//! little-endian; all length arithmetic is checked; bytes from disk are
//! untrusted until verified.
//!
//! Every record stores the **full pre-postprocessor core token stream**
//! at one session revision, plus the raw text tail past the certified
//! stable prefix. The stable prefix text itself is not stored (it is
//! reachable through the revision chain). Records link into a
//! per-session hash chain through `prev_block_hash`/`curr_block_hash`
//! with the semantic fingerprint bound into every link, so a wrong key
//! can never produce a verifying chain.

use sha2::{Digest, Sha256};

use crate::engine::WitnessCategory;
use crate::error::StoreError;

/// Record magic (8 bytes).
pub const MAGIC: [u8; 8] = *b"TOKTIERS";
/// The format version this module reads and writes.
pub const FORMAT_VERSION: u16 = 1;
/// Fixed header size.
pub const FIXED_HEADER_LEN: u16 = 200;
/// Upper bound on `header_length`.
pub const HEADER_LEN_MAX: u16 = 4096;
/// Little-endian marker byte at offset 16.
pub const ENDIANNESS_LE: u8 = 0x01;
/// Mandatory ("must understand") flag bits; v1 assigns none.
pub const MANDATORY_FLAGS_MASK: u32 = 0x0000_FFFF;
/// Frozen bound on `full_text_byte_length`.
pub const MAX_FULL_TEXT_BYTES: u64 = 1 << 40;
/// Frozen bound on `text_tail_byte_length` and `token_count`.
pub const MAX_TAIL_BYTES_OR_TOKENS: u64 = 1 << 31;

const DOMAIN_PAYLOAD: &[u8] = b"toktier.store.v1.payload\0";
const DOMAIN_LINK: &[u8] = b"toktier.store.v1.link\0";
const DOMAIN_RECORD: &[u8] = b"toktier.store.v1.record\0";

const OFF_CHECKSUM: usize = 168;
const HASH_LEN: usize = 32;

/// IDs per stack chunk when converting a `u32` row to little-endian
/// bytes (4096 * 4 = 16 KiB). Feeding the hash and the record writer in
/// blocks avoids one call per 4-byte value on multi-million-token rows;
/// the produced byte sequence is unchanged.
const CHUNK_IDS: usize = 4096;

/// Convert `ids` to their little-endian payload bytes through a stack
/// buffer and hand each block to `consume`. This is the one definition
/// of the payload's ID byte layout used by hashing and serialization.
fn ids_le_chunks(ids: &[u32], mut consume: impl FnMut(&[u8])) {
    let mut buffer = [0u8; CHUNK_IDS * 4];
    for chunk in ids.chunks(CHUNK_IDS) {
        let mut used = 0usize;
        for &value in chunk {
            buffer[used..used + 4].copy_from_slice(&value.to_le_bytes());
            used += 4;
        }
        consume(&buffer[..used]);
    }
}

/// A 32-byte SHA-256 output (chain hashes, digests, checksums).
pub type BlockHash = [u8; HASH_LEN];

/// All-zero hash: the genesis `prev_block_hash`.
pub const ZERO_HASH: BlockHash = [0u8; HASH_LEN];

fn malformed(msg: impl Into<String>) -> StoreError {
    StoreError::MalformedRecord(msg.into())
}

// ------------------------------------------------------------- hashing --

/// Payload digest (Section 4.1): one pass over the payload bytes.
/// The payload is the id array (u32 LE each) followed by the tail text;
/// this helper streams the same bytes from parts without materializing
/// the payload buffer.
pub fn payload_digest_parts(id_parts: &[&[u32]], tail_text: &[u8]) -> BlockHash {
    let mut h = Sha256::new();
    h.update(DOMAIN_PAYLOAD);
    for part in id_parts {
        ids_le_chunks(part, |block| h.update(block));
    }
    h.update(tail_text);
    h.finalize().into()
}

/// Incremental payload-digest state (Section 4.1).
///
/// The state holds the running SHA-256 over `DOMAIN_PAYLOAD` followed by
/// every ID fed so far (u32 LE each). A session keeps one such state over
/// its append-only sealed prefix, advances it exactly when the prefix
/// grows, and completes a commit digest by cloning the state and feeding
/// only the mutable tail. The finished digest is bit-identical to
/// [`payload_digest_parts`] over the same complete parts: SHA-256 is
/// invariant to update chunking, and the byte layout comes from the same
/// shared helper.
#[derive(Clone)]
pub struct PayloadHasher {
    hasher: Sha256,
}

impl PayloadHasher {
    /// State over the domain prefix and no IDs.
    pub fn new() -> PayloadHasher {
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_PAYLOAD);
        PayloadHasher { hasher }
    }

    /// Feed `ids` as little-endian payload bytes.
    pub fn update_ids(&mut self, ids: &[u32]) {
        ids_le_chunks(ids, |block| self.hasher.update(block));
    }

    /// Complete the digest with the trailing parts without disturbing the
    /// running prefix state.
    pub fn digest_with_tail(&self, tail_ids: &[u32], tail_text: &[u8]) -> BlockHash {
        let mut hasher = self.hasher.clone();
        ids_le_chunks(tail_ids, |block| hasher.update(block));
        hasher.update(tail_text);
        hasher.finalize().into()
    }
}

impl Default for PayloadHasher {
    fn default() -> PayloadHasher {
        PayloadHasher::new()
    }
}

/// Inputs of the chain link hash (Section 4.2), gathered so the link
/// construction has exactly one implementation.
pub struct LinkInputs<'a> {
    pub prev_block_hash: &'a BlockHash,
    pub fingerprint: &'a [u8; 32],
    pub session_revision: u64,
    pub full_text_bytes: u64,
    pub stable_prefix_bytes: u64,
    pub text_tail_bytes: u64,
    pub token_count: u64,
    pub replace_token_offset: u64,
    pub witness: WitnessCategory,
    pub payload_digest: &'a BlockHash,
}

/// Chain link hash (`curr_block_hash`, Section 4.2).
pub fn link_hash(inputs: &LinkInputs<'_>) -> BlockHash {
    let mut h = Sha256::new();
    h.update(DOMAIN_LINK);
    h.update(inputs.prev_block_hash);
    h.update(inputs.fingerprint);
    h.update(inputs.session_revision.to_le_bytes());
    h.update(inputs.full_text_bytes.to_le_bytes());
    h.update(inputs.stable_prefix_bytes.to_le_bytes());
    h.update(inputs.text_tail_bytes.to_le_bytes());
    h.update(inputs.token_count.to_le_bytes());
    h.update(inputs.replace_token_offset.to_le_bytes());
    h.update(inputs.witness.as_u16().to_le_bytes());
    h.update(inputs.payload_digest);
    h.finalize().into()
}

fn record_checksum(header_zeroed_checksum: &[u8], payload_digest: &BlockHash) -> BlockHash {
    let mut h = Sha256::new();
    h.update(DOMAIN_RECORD);
    h.update(header_zeroed_checksum);
    h.update(payload_digest);
    h.finalize().into()
}

// -------------------------------------------------------------- record --

/// A store format v1 record: one full core-stream snapshot of a session
/// at one revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRecordV1 {
    /// Opaque 32-byte semantic fingerprint (preimage specified in
    /// `docs/contracts/fingerprint.md`; never derived or inspected
    /// here).
    pub fingerprint: [u8; 32],
    /// Witness predicate category that certified the current safe cut.
    pub witness: WitnessCategory,
    /// Session revision; genesis is 0, strictly increasing.
    pub revision: u64,
    /// `curr_block_hash` of the predecessor record; all-zero at genesis.
    pub prev_block_hash: BlockHash,
    /// This record's chain link hash. Filled by [`Self::to_bytes`] /
    /// verified by [`Self::from_bytes`]; always recomputable via
    /// [`Self::compute_curr`].
    pub curr_block_hash: BlockHash,
    /// Byte length of the certified stable text prefix (its bytes are
    /// not stored in this record).
    pub stable_prefix_bytes: u64,
    /// `replace_from` of the append that produced this revision; a full
    /// re-encode records 0.
    pub replace_token_offset: u64,
    /// The full core token stream at this revision.
    pub ids: Vec<u32>,
    /// Raw text suffix starting at the stable prefix end (UTF-8).
    pub tail_text: String,
}

impl SessionRecordV1 {
    /// `full_text_byte_length` of this record.
    pub fn full_text_bytes(&self) -> u64 {
        self.stable_prefix_bytes + self.tail_text.len() as u64
    }

    fn check_writer_obligations(&self) -> Result<(), StoreError> {
        let tail_len = self.tail_text.len() as u64;
        let token_count = self.ids.len() as u64;
        let full = self
            .stable_prefix_bytes
            .checked_add(tail_len)
            .ok_or_else(|| malformed("full text byte length overflow"))?;
        if full > MAX_FULL_TEXT_BYTES {
            return Err(malformed("full_text_byte_length exceeds 2^40"));
        }
        if tail_len > MAX_TAIL_BYTES_OR_TOKENS {
            return Err(malformed("text_tail_byte_length exceeds 2^31"));
        }
        if token_count > MAX_TAIL_BYTES_OR_TOKENS {
            return Err(malformed("token_count exceeds 2^31"));
        }
        if self.replace_token_offset > token_count {
            return Err(malformed("replace_token_offset exceeds token_count"));
        }
        if self.witness == WitnessCategory::NoneFullReencode
            && (self.stable_prefix_bytes != 0 || self.replace_token_offset != 0)
        {
            return Err(malformed(
                "witness category 0 requires zero stable prefix and replace offset",
            ));
        }
        if (self.revision == 0) != (self.prev_block_hash == ZERO_HASH) {
            return Err(malformed(
                "prev_block_hash must be all-zero exactly at revision 0",
            ));
        }
        Ok(())
    }

    /// Recompute this record's chain link hash from its fields.
    pub fn compute_curr(&self) -> BlockHash {
        let digest = payload_digest_parts(&[&self.ids], self.tail_text.as_bytes());
        link_hash(&LinkInputs {
            prev_block_hash: &self.prev_block_hash,
            fingerprint: &self.fingerprint,
            session_revision: self.revision,
            full_text_bytes: self.full_text_bytes(),
            stable_prefix_bytes: self.stable_prefix_bytes,
            text_tail_bytes: self.tail_text.len() as u64,
            token_count: self.ids.len() as u64,
            replace_token_offset: self.replace_token_offset,
            witness: self.witness,
            payload_digest: &digest,
        })
    }

    /// Serialize per the frozen writer obligations: format version 1,
    /// zero flags, zero reserved fields, `header_length == 200`, no
    /// TLVs. The stored `curr_block_hash` field is ignored; the hash is
    /// recomputed so a serialized record is always self-consistent.
    pub fn to_bytes(&self) -> Result<Vec<u8>, StoreError> {
        self.check_writer_obligations()?;
        let tail = self.tail_text.as_bytes();
        let payload_digest = payload_digest_parts(&[&self.ids], tail);
        let curr = link_hash(&LinkInputs {
            prev_block_hash: &self.prev_block_hash,
            fingerprint: &self.fingerprint,
            session_revision: self.revision,
            full_text_bytes: self.full_text_bytes(),
            stable_prefix_bytes: self.stable_prefix_bytes,
            text_tail_bytes: tail.len() as u64,
            token_count: self.ids.len() as u64,
            replace_token_offset: self.replace_token_offset,
            witness: self.witness,
            payload_digest: &payload_digest,
        });
        let hl = FIXED_HEADER_LEN as usize;
        let mut w = Vec::with_capacity(hl + self.ids.len() * 4 + tail.len());
        w.extend_from_slice(&MAGIC);
        w.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        w.extend_from_slice(&FIXED_HEADER_LEN.to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes()); // flags
        w.push(ENDIANNESS_LE);
        w.push(0); // reserved0
        w.extend_from_slice(&self.witness.as_u16().to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes()); // reserved1
        w.extend_from_slice(&self.fingerprint);
        w.extend_from_slice(&self.revision.to_le_bytes());
        w.extend_from_slice(&self.prev_block_hash);
        w.extend_from_slice(&curr);
        w.extend_from_slice(&self.full_text_bytes().to_le_bytes());
        w.extend_from_slice(&self.stable_prefix_bytes.to_le_bytes());
        w.extend_from_slice(&(tail.len() as u64).to_le_bytes());
        w.extend_from_slice(&(self.ids.len() as u64).to_le_bytes());
        w.extend_from_slice(&self.replace_token_offset.to_le_bytes());
        w.extend_from_slice(&[0u8; HASH_LEN]); // checksum slot
        debug_assert_eq!(w.len(), hl);
        let checksum = record_checksum(&w, &payload_digest);
        w[OFF_CHECKSUM..OFF_CHECKSUM + HASH_LEN].copy_from_slice(&checksum);
        ids_le_chunks(&self.ids, |block| w.extend_from_slice(block));
        w.extend_from_slice(tail);
        Ok(w)
    }

    /// Strict decode in the normative order (Section 5, steps 1-5 minus
    /// the caller-side chain walk and fingerprint comparison, which
    /// need context this codec does not have).
    pub fn from_bytes(bytes: &[u8]) -> Result<SessionRecordV1, StoreError> {
        // Step 1: bounds gate.
        if bytes.len() < FIXED_HEADER_LEN as usize {
            return Err(StoreError::Truncated { context: "header" });
        }
        if bytes[0..8] != MAGIC {
            return Err(StoreError::BadMagic);
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if version != FORMAT_VERSION {
            return Err(StoreError::UnsupportedFormatVersion(version));
        }
        let header_length = u16::from_le_bytes([bytes[10], bytes[11]]);
        if !(FIXED_HEADER_LEN..=HEADER_LEN_MAX).contains(&header_length)
            || !header_length.is_multiple_of(8)
            || header_length as usize > bytes.len()
        {
            return Err(StoreError::BadHeaderLength {
                header_length,
                record_len: bytes.len(),
            });
        }
        if bytes[16] != ENDIANNESS_LE {
            return Err(StoreError::BadEndianMarker(bytes[16]));
        }
        if bytes[17] != 0 {
            return Err(malformed("reserved0 must be zero"));
        }
        let flags = u32::from_le_bytes(bytes[12..16].try_into().expect("fixed slice"));
        let unknown_mandatory = flags & MANDATORY_FLAGS_MASK;
        if unknown_mandatory != 0 {
            return Err(StoreError::UnknownMandatoryFlags(unknown_mandatory));
        }
        let witness = WitnessCategory::from_u16(u16::from_le_bytes([bytes[18], bytes[19]]))?;
        if bytes[20..24] != [0u8; 4] {
            return Err(malformed("reserved1 must be zero"));
        }

        // Step 2: field bounds.
        let u64_at = |off: usize| -> u64 {
            u64::from_le_bytes(bytes[off..off + 8].try_into().expect("fixed slice"))
        };
        let hash_at = |off: usize| -> BlockHash {
            bytes[off..off + HASH_LEN].try_into().expect("fixed slice")
        };
        let fingerprint: [u8; 32] = bytes[24..56].try_into().expect("fixed slice");
        let revision = u64_at(56);
        let prev_block_hash = hash_at(64);
        let stored_curr = hash_at(96);
        let full_text_bytes = u64_at(128);
        let stable_prefix_bytes = u64_at(136);
        let text_tail_bytes = u64_at(144);
        let token_count = u64_at(152);
        let replace_token_offset = u64_at(160);
        if full_text_bytes > MAX_FULL_TEXT_BYTES {
            return Err(malformed("full_text_byte_length exceeds 2^40"));
        }
        if stable_prefix_bytes > full_text_bytes {
            return Err(malformed(
                "stable_prefix_byte_length exceeds full_text_byte_length",
            ));
        }
        if text_tail_bytes > MAX_TAIL_BYTES_OR_TOKENS {
            return Err(malformed("text_tail_byte_length exceeds 2^31"));
        }
        match stable_prefix_bytes.checked_add(text_tail_bytes) {
            Some(sum) if sum == full_text_bytes => {}
            _ => {
                return Err(malformed(
                    "stable prefix + text tail does not equal full text length",
                ))
            }
        }
        if token_count > MAX_TAIL_BYTES_OR_TOKENS {
            return Err(malformed("token_count exceeds 2^31"));
        }
        if replace_token_offset > token_count {
            return Err(malformed("replace_token_offset exceeds token_count"));
        }
        // TLV extension parse (checked; v1 defines no non-padding
        // types, unknown types are skipped).
        let mut pos = FIXED_HEADER_LEN as usize;
        let hl = header_length as usize;
        while pos < hl {
            let end = pos
                .checked_add(4)
                .filter(|&e| e <= hl)
                .ok_or_else(|| malformed("TLV header extends past header_length"))?;
            let tlv_len = u16::from_le_bytes([bytes[pos + 2], bytes[pos + 3]]) as usize;
            pos = end
                .checked_add(tlv_len)
                .filter(|&e| e <= hl)
                .ok_or_else(|| malformed("TLV value extends past header_length"))?;
        }

        // Step 3: size closure.
        let ids_bytes = (token_count as usize)
            .checked_mul(4)
            .ok_or_else(|| malformed("token id array size overflow"))?;
        let want = hl
            .checked_add(ids_bytes)
            .and_then(|v| v.checked_add(text_tail_bytes as usize))
            .ok_or_else(|| malformed("record size overflow"))?;
        if bytes.len() != want {
            return Err(malformed(format!(
                "record size {} does not close over header + ids + tail {}",
                bytes.len(),
                want
            )));
        }

        // Step 4: integrity.
        let payload = &bytes[hl..];
        let mut ph = Sha256::new();
        ph.update(DOMAIN_PAYLOAD);
        ph.update(payload);
        let payload_digest: BlockHash = ph.finalize().into();
        let mut header_copy = bytes[..hl].to_vec();
        let mut stored_checksum = [0u8; HASH_LEN];
        stored_checksum.copy_from_slice(&header_copy[OFF_CHECKSUM..OFF_CHECKSUM + HASH_LEN]);
        header_copy[OFF_CHECKSUM..OFF_CHECKSUM + HASH_LEN].fill(0);
        if record_checksum(&header_copy, &payload_digest) != stored_checksum {
            return Err(StoreError::ChecksumMismatch);
        }
        let curr = link_hash(&LinkInputs {
            prev_block_hash: &prev_block_hash,
            fingerprint: &fingerprint,
            session_revision: revision,
            full_text_bytes,
            stable_prefix_bytes,
            text_tail_bytes,
            token_count,
            replace_token_offset,
            witness,
            payload_digest: &payload_digest,
        });
        if curr != stored_curr {
            return Err(StoreError::ChainLinkMismatch);
        }
        if (revision == 0) != (prev_block_hash == ZERO_HASH) {
            return Err(malformed(
                "prev_block_hash must be all-zero exactly at revision 0",
            ));
        }

        // Step 5: semantic checks.
        if witness == WitnessCategory::NoneFullReencode
            && (stable_prefix_bytes != 0 || replace_token_offset != 0)
        {
            return Err(malformed(
                "witness category 0 requires zero stable prefix and replace offset",
            ));
        }
        let ids: Vec<u32> = payload[..ids_bytes]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("fixed slice")))
            .collect();
        let tail_text = std::str::from_utf8(&payload[ids_bytes..])
            .map_err(|e| malformed(format!("text tail is not valid UTF-8: {e}")))?
            .to_string();
        Ok(SessionRecordV1 {
            fingerprint,
            witness,
            revision,
            prev_block_hash,
            curr_block_hash: stored_curr,
            stable_prefix_bytes,
            replace_token_offset,
            ids,
            tail_text,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SessionRecordV1 {
        SessionRecordV1 {
            fingerprint: [7u8; 32],
            witness: WitnessCategory::BpeSyncTransition,
            revision: 3,
            prev_block_hash: [9u8; 32],
            curr_block_hash: ZERO_HASH, // recomputed on encode
            stable_prefix_bytes: 4100,
            replace_token_offset: 2,
            ids: vec![10, 11, 12, 13, u32::MAX],
            tail_text: "tail text with unicode: \u{4f60}\u{597d} caf\u{e9}\r\n \u{1f642}"
                .to_string(),
        }
    }

    fn genesis() -> SessionRecordV1 {
        SessionRecordV1 {
            fingerprint: [5u8; 32],
            witness: WitnessCategory::NoneFullReencode,
            revision: 0,
            prev_block_hash: ZERO_HASH,
            curr_block_hash: ZERO_HASH,
            stable_prefix_bytes: 0,
            replace_token_offset: 0,
            ids: vec![1, 2, 3],
            tail_text: "whole text lives in the tail".to_string(),
        }
    }

    #[test]
    fn roundtrip_and_geometry() {
        for rec in [sample(), genesis()] {
            let bytes = rec.to_bytes().unwrap();
            // Frozen geometry spot checks.
            assert_eq!(&bytes[0..8], b"TOKTIERS");
            assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 1);
            assert_eq!(u16::from_le_bytes([bytes[10], bytes[11]]), 200);
            assert_eq!(bytes[16], 0x01);
            assert_eq!(bytes.len(), 200 + rec.ids.len() * 4 + rec.tail_text.len());
            let back = SessionRecordV1::from_bytes(&bytes).unwrap();
            assert_eq!(back.fingerprint, rec.fingerprint);
            assert_eq!(back.witness, rec.witness);
            assert_eq!(back.revision, rec.revision);
            assert_eq!(back.prev_block_hash, rec.prev_block_hash);
            assert_eq!(back.stable_prefix_bytes, rec.stable_prefix_bytes);
            assert_eq!(back.replace_token_offset, rec.replace_token_offset);
            assert_eq!(back.ids, rec.ids);
            assert_eq!(back.tail_text, rec.tail_text);
            assert_eq!(back.curr_block_hash, back.compute_curr());
        }
    }

    #[test]
    fn every_bit_flip_is_rejected() {
        let bytes = sample().to_bytes().unwrap();
        for ix in 0..bytes.len() {
            for bit in [0x01u8, 0x80u8] {
                let mut bad = bytes.clone();
                bad[ix] ^= bit;
                assert!(
                    SessionRecordV1::from_bytes(&bad).is_err(),
                    "bit flip at byte {ix} accepted"
                );
            }
        }
    }

    #[test]
    fn truncation_and_extension_are_rejected() {
        let bytes = sample().to_bytes().unwrap();
        for cut in [0, 7, 100, 199, 200, bytes.len() - 1] {
            assert!(SessionRecordV1::from_bytes(&bytes[..cut]).is_err());
        }
        let mut extended = bytes.clone();
        extended.push(0);
        assert!(SessionRecordV1::from_bytes(&extended).is_err());
    }

    #[test]
    fn header_rejections_carry_contract_codes() {
        let good = sample().to_bytes().unwrap();

        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert_eq!(
            SessionRecordV1::from_bytes(&bad_magic).unwrap_err().code(),
            "STORE_CORRUPT"
        );

        let mut future_version = good.clone();
        future_version[8] = 2;
        let err = SessionRecordV1::from_bytes(&future_version).unwrap_err();
        assert_eq!(err, StoreError::UnsupportedFormatVersion(2));
        assert_eq!(err.code(), "STORE_FORMAT_UNSUPPORTED");

        // header_length: below 200, above 4096, not multiple of 8.
        for hl in [192u16, 4104, 204] {
            let mut bad = good.clone();
            bad[10..12].copy_from_slice(&hl.to_le_bytes());
            assert!(matches!(
                SessionRecordV1::from_bytes(&bad).unwrap_err(),
                StoreError::BadHeaderLength { .. }
            ));
        }

        let mut bad_endian = good.clone();
        bad_endian[16] = 0x02;
        assert_eq!(
            SessionRecordV1::from_bytes(&bad_endian).unwrap_err(),
            StoreError::BadEndianMarker(0x02)
        );

        let mut mandatory_flag = good.clone();
        mandatory_flag[12] = 0x01;
        let err = SessionRecordV1::from_bytes(&mandatory_flag).unwrap_err();
        assert_eq!(err, StoreError::UnknownMandatoryFlags(1));
        assert_eq!(err.code(), "STORE_FORMAT_UNSUPPORTED");

        let mut bad_witness = good.clone();
        bad_witness[18..20].copy_from_slice(&0x00ffu16.to_le_bytes());
        let err = SessionRecordV1::from_bytes(&bad_witness).unwrap_err();
        assert_eq!(err, StoreError::UnknownWitnessCategory(0x00ff));
        assert_eq!(err.code(), "STORE_FORMAT_UNSUPPORTED");

        // Advisory flag bits (16..31) alone do not make the record
        // unsupported -- but any header change breaks the checksum.
        let mut advisory = good.clone();
        advisory[14] = 0x01;
        assert_eq!(
            SessionRecordV1::from_bytes(&advisory).unwrap_err(),
            StoreError::ChecksumMismatch
        );
    }

    #[test]
    fn witness_none_cross_invariant_enforced() {
        let mut rec = genesis();
        rec.stable_prefix_bytes = 4;
        assert!(rec.to_bytes().is_err());
        let mut rec2 = genesis();
        rec2.replace_token_offset = 1;
        assert!(rec2.to_bytes().is_err());
    }

    #[test]
    fn genesis_prev_hash_rule_enforced() {
        let mut rec = sample();
        rec.revision = 0;
        // revision 0 with nonzero prev must be rejected at write time.
        assert!(rec.to_bytes().is_err());
        let mut rec2 = sample();
        rec2.prev_block_hash = ZERO_HASH;
        // nonzero revision with zero prev likewise.
        assert!(rec2.to_bytes().is_err());
    }

    #[test]
    fn tlv_extension_is_skippable_and_checked() {
        // Simulate a future writer that appends one 8-byte padding TLV:
        // header_length 208, checksum and link recomputed accordingly.
        let rec = sample();
        let bytes = rec.to_bytes().unwrap();
        let hl_new: u16 = 208;
        let mut ext = Vec::with_capacity(bytes.len() + 8);
        ext.extend_from_slice(&bytes[..200]);
        // TLV: type 0 (padding), length 4, four value bytes.
        ext.extend_from_slice(&0u16.to_le_bytes());
        ext.extend_from_slice(&4u16.to_le_bytes());
        ext.extend_from_slice(&[0u8; 4]);
        ext.extend_from_slice(&bytes[200..]);
        ext[10..12].copy_from_slice(&hl_new.to_le_bytes());
        // Recompute checksum over the extended header.
        let payload_digest = payload_digest_parts(&[&rec.ids], rec.tail_text.as_bytes());
        ext[OFF_CHECKSUM..OFF_CHECKSUM + 32].fill(0);
        let cs = record_checksum(&ext[..hl_new as usize], &payload_digest);
        ext[OFF_CHECKSUM..OFF_CHECKSUM + 32].copy_from_slice(&cs);
        let back = SessionRecordV1::from_bytes(&ext).unwrap();
        assert_eq!(back.ids, rec.ids);
        assert_eq!(back.tail_text, rec.tail_text);

        // A TLV running past header_length is corrupt.
        let mut overrun = ext.clone();
        overrun[202..204].copy_from_slice(&100u16.to_le_bytes());
        assert!(SessionRecordV1::from_bytes(&overrun).is_err());
    }

    /// The pre-chunking implementation, retained verbatim as the digest
    /// oracle: one `update` per 4-byte little-endian ID.
    fn payload_digest_per_element(id_parts: &[&[u32]], tail_text: &[u8]) -> BlockHash {
        let mut h = Sha256::new();
        h.update(DOMAIN_PAYLOAD);
        for part in id_parts {
            for &v in *part {
                h.update(v.to_le_bytes());
            }
        }
        h.update(tail_text);
        h.finalize().into()
    }

    #[test]
    fn chunked_payload_digest_matches_the_per_element_oracle() {
        let row: Vec<u32> = (0..10_000u32)
            .map(|i| i.wrapping_mul(2_654_435_761).rotate_left(7))
            .collect();
        let tail = b"tail text with unicode: \xe4\xbd\xa0\xe5\xa5\xbd";
        for take in [0usize, 1, 3, 4095, 4096, 4097, 8192, 10_000] {
            let part = &row[..take];
            for split in [0usize, take / 3, take] {
                let parts: [&[u32]; 2] = [&part[..split], &part[split..]];
                assert_eq!(
                    payload_digest_parts(&parts, tail),
                    payload_digest_per_element(&parts, tail),
                    "take={take} split={split}"
                );
            }
        }
        assert_eq!(
            payload_digest_parts(&[], b""),
            payload_digest_per_element(&[], b"")
        );
    }

    #[test]
    fn prefix_hasher_digest_equals_full_recomputation() {
        let row: Vec<u32> = (0..9000u32)
            .map(|i| i.wrapping_mul(97).rotate_left(11))
            .collect();
        let tail_text = b"the mutable tail";
        for sealed_len in [0usize, 1, 4096, 4097, 9000] {
            let (sealed, tail_ids) = row.split_at(sealed_len);
            let mut prefix = PayloadHasher::new();
            prefix.update_ids(sealed);
            assert_eq!(
                prefix.digest_with_tail(tail_ids, tail_text),
                payload_digest_parts(&[sealed, tail_ids], tail_text),
                "sealed_len={sealed_len}"
            );
            // The state is reusable: completing a digest does not disturb it.
            assert_eq!(
                prefix.digest_with_tail(tail_ids, tail_text),
                payload_digest_parts(&[sealed, tail_ids], tail_text)
            );
        }
        // Incremental prefix growth in several steps equals one-shot growth.
        let mut stepped = PayloadHasher::new();
        for chunk in row.chunks(1000) {
            stepped.update_ids(chunk);
        }
        assert_eq!(
            stepped.digest_with_tail(&[], b"x"),
            payload_digest_parts(&[&row], b"x")
        );
    }

    #[test]
    fn payload_digest_golden_vector_is_stable() {
        // Frozen golden vector: any change to the domain prefix or the ID
        // byte layout changes this digest and must fail loudly.
        let digest = payload_digest_parts(&[&[0u32, 1, 2, u32::MAX]], b"tail");
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            hex,
            "7388ba641a9bda6c841035a3af0349e725f366a47fce8447046745c5ef6cb508"
        );
    }

    #[test]
    fn multi_chunk_record_roundtrips() {
        let mut rec = sample();
        rec.ids = (0..9000u32)
            .map(|i| i.wrapping_mul(31).rotate_left(3))
            .collect();
        let bytes = rec.to_bytes().unwrap();
        assert_eq!(bytes.len(), 200 + rec.ids.len() * 4 + rec.tail_text.len());
        let back = SessionRecordV1::from_bytes(&bytes).unwrap();
        assert_eq!(back.ids, rec.ids);
        assert_eq!(back.curr_block_hash, back.compute_curr());
    }

    #[test]
    fn link_hash_binds_fingerprint_and_revision() {
        let a = sample();
        let mut b = sample();
        b.fingerprint = [8u8; 32];
        assert_ne!(a.compute_curr(), b.compute_curr());
        let mut c = sample();
        c.revision = 4;
        assert_ne!(a.compute_curr(), c.compute_curr());
        let mut d = sample();
        d.prev_block_hash = [1u8; 32];
        assert_ne!(a.compute_curr(), d.compute_curr());
        assert_eq!(a.compute_curr(), sample().compute_curr());
    }
}
