//! Canonical TKFR-v1 recovery binding and incremental content checkpoints.

use blake2b_simd::{Params as Blake2Params, State as Blake2State};
use sha2::{Digest, Sha256};

use crate::{BlockHash, StoreError};

pub const MARK_FLOOR_BYTES: u64 = 4096;
pub const CONTENT_DIGEST_BYTES: usize = 16;
const MAGIC: &[u8; 4] = b"TKFR";
const VERSION: u16 = 1;
const MAX_MARKS: usize = 64;
const STATE_DOMAIN: &[u8] = b"toktier.facade.v1.recovery-state\0";
const INDEX_PERSON: &[u8] = b"toktier.fidx.v1";
const HEADER_BYTES: usize = 4 + 2 + 32 + 8 + 32 + 16 + 4;
const MARK_BYTES: usize = 8 + 16;
const CHECKSUM_BYTES: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentIndexEntry {
    pub byte_length: u64,
    pub end_digest: [u8; CONTENT_DIGEST_BYTES],
    pub marks: Vec<(u64, [u8; CONTENT_DIGEST_BYTES])>,
}

#[derive(Clone)]
pub struct ContentDigest {
    state: Blake2State,
    byte_length: u64,
    marks: Vec<(u64, [u8; CONTENT_DIGEST_BYTES])>,
    next_mark: u64,
}

impl ContentDigest {
    pub fn empty() -> Self {
        let state = Blake2Params::new()
            .hash_length(CONTENT_DIGEST_BYTES)
            .personal(INDEX_PERSON)
            .to_state();
        Self {
            state,
            byte_length: 0,
            marks: Vec::new(),
            next_mark: MARK_FLOOR_BYTES,
        }
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, StoreError> {
        let mut digest = Self::empty();
        digest.append(data)?;
        Ok(digest)
    }

    pub fn append(&mut self, delta: &[u8]) -> Result<(), StoreError> {
        let delta_len = u64::try_from(delta.len())
            .map_err(|_| StoreError::InvalidInput("text byte length exceeds u64".into()))?;
        let end = self
            .byte_length
            .checked_add(delta_len)
            .ok_or_else(|| StoreError::InvalidInput("text byte length exceeds u64".into()))?;
        let mut consumed = 0usize;
        while self.next_mark < end {
            let take_u64 = self
                .next_mark
                .checked_sub(self.byte_length)
                .ok_or_else(|| StoreError::Internal("content mark moved backwards".into()))?;
            let take = usize::try_from(take_u64)
                .map_err(|_| StoreError::InvalidInput("text byte length exceeds usize".into()))?;
            let next = consumed
                .checked_add(take)
                .ok_or_else(|| StoreError::Internal("content slice overflow".into()))?;
            self.state.update(&delta[consumed..next]);
            self.byte_length = self.next_mark;
            consumed = next;
            self.marks.push((self.next_mark, finalize16(&self.state)));
            self.next_mark = self
                .next_mark
                .checked_mul(2)
                .ok_or_else(|| StoreError::InvalidInput("content mark exceeds u64".into()))?;
        }
        self.state.update(&delta[consumed..]);
        self.byte_length = end;
        Ok(())
    }

    pub fn entry(&self) -> ContentIndexEntry {
        ContentIndexEntry {
            byte_length: self.byte_length,
            end_digest: finalize16(&self.state),
            marks: self.marks.clone(),
        }
    }
}

fn finalize16(state: &Blake2State) -> [u8; CONTENT_DIGEST_BYTES] {
    state
        .finalize()
        .as_bytes()
        .try_into()
        .expect("fixed digest")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryBindingV1 {
    pub record_hash: BlockHash,
    pub text_digest: [u8; 32],
    pub index_entry: ContentIndexEntry,
}

impl RecoveryBindingV1 {
    pub fn to_bytes(&self) -> Result<Vec<u8>, StoreError> {
        validate_entry(&self.index_entry)?;
        let mut body = Vec::with_capacity(
            HEADER_BYTES + self.index_entry.marks.len() * MARK_BYTES + CHECKSUM_BYTES,
        );
        body.extend_from_slice(MAGIC);
        body.extend_from_slice(&VERSION.to_le_bytes());
        body.extend_from_slice(&self.record_hash);
        body.extend_from_slice(&self.index_entry.byte_length.to_le_bytes());
        body.extend_from_slice(&self.text_digest);
        body.extend_from_slice(&self.index_entry.end_digest);
        body.extend_from_slice(&(self.index_entry.marks.len() as u32).to_le_bytes());
        for (position, digest) in &self.index_entry.marks {
            body.extend_from_slice(&position.to_le_bytes());
            body.extend_from_slice(digest);
        }
        let mut checksum = Sha256::new();
        checksum.update(STATE_DOMAIN);
        checksum.update(&body);
        body.extend_from_slice(&checksum.finalize());
        Ok(body)
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self, StoreError> {
        if raw.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(StoreError::InvalidInput(
                "recovery binding is truncated".into(),
            ));
        }
        let body_len = raw.len() - CHECKSUM_BYTES;
        let (body, observed_checksum) = raw.split_at(body_len);
        let mut checksum = Sha256::new();
        checksum.update(STATE_DOMAIN);
        checksum.update(body);
        if checksum.finalize().as_slice() != observed_checksum {
            return Err(StoreError::InvalidInput(
                "recovery binding checksum mismatch".into(),
            ));
        }
        let mut cursor = 0usize;
        let magic = take::<4>(body, &mut cursor)?;
        if &magic != MAGIC {
            return Err(StoreError::InvalidInput(
                "recovery binding has bad magic".into(),
            ));
        }
        let version = u16::from_le_bytes(take::<2>(body, &mut cursor)?);
        if version != VERSION {
            return Err(StoreError::InvalidInput(
                "recovery binding version is unsupported".into(),
            ));
        }
        let record_hash = take::<32>(body, &mut cursor)?;
        let byte_length = u64::from_le_bytes(take::<8>(body, &mut cursor)?);
        let text_digest = take::<32>(body, &mut cursor)?;
        let end_digest = take::<16>(body, &mut cursor)?;
        let mark_count = u32::from_le_bytes(take::<4>(body, &mut cursor)?) as usize;
        if mark_count > MAX_MARKS {
            return Err(StoreError::InvalidInput(
                "recovery binding has too many checkpoints".into(),
            ));
        }
        if body.len() != HEADER_BYTES + mark_count * MARK_BYTES {
            return Err(StoreError::InvalidInput(
                "recovery binding size does not close".into(),
            ));
        }
        let mut marks = Vec::with_capacity(mark_count);
        for _ in 0..mark_count {
            let position = u64::from_le_bytes(take::<8>(body, &mut cursor)?);
            let digest = take::<16>(body, &mut cursor)?;
            marks.push((position, digest));
        }
        let index_entry = ContentIndexEntry {
            byte_length,
            end_digest,
            marks,
        };
        validate_entry(&index_entry)?;
        Ok(Self {
            record_hash,
            text_digest,
            index_entry,
        })
    }

    pub(crate) fn from_states(
        record_hash: BlockHash,
        text_digest: [u8; 32],
        content: &ContentDigest,
    ) -> Self {
        Self {
            record_hash,
            text_digest,
            index_entry: content.entry(),
        }
    }
}

fn validate_entry(entry: &ContentIndexEntry) -> Result<(), StoreError> {
    if entry.marks.len() > MAX_MARKS {
        return Err(StoreError::InvalidInput(
            "too many recovery checkpoints".into(),
        ));
    }
    let mut expected = MARK_FLOOR_BYTES;
    for (position, _) in &entry.marks {
        if *position != expected || *position >= entry.byte_length {
            return Err(StoreError::InvalidInput(
                "checkpoint positions are not canonical".into(),
            ));
        }
        expected = expected
            .checked_mul(2)
            .ok_or_else(|| StoreError::InvalidInput("checkpoint position overflow".into()))?;
    }
    if expected < entry.byte_length {
        return Err(StoreError::InvalidInput(
            "checkpoint positions are incomplete".into(),
        ));
    }
    Ok(())
}

fn take<const N: usize>(raw: &[u8], cursor: &mut usize) -> Result<[u8; N], StoreError> {
    let end = cursor
        .checked_add(N)
        .ok_or_else(|| StoreError::InvalidInput("recovery binding offset overflow".into()))?;
    let bytes = raw
        .get(*cursor..end)
        .ok_or_else(|| StoreError::InvalidInput("recovery binding is truncated".into()))?;
    *cursor = end;
    Ok(bytes.try_into().expect("fixed slice"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT_DOMAIN: &[u8] = b"toktier.facade.v1.recovery-text\0";

    #[test]
    fn former_endpoint_becomes_a_mark_after_append() {
        let mut state = ContentDigest::from_bytes(&vec![b'a'; 4096]).unwrap();
        assert!(state.entry().marks.is_empty());
        state.append(b"x").unwrap();
        assert_eq!(state.entry().marks[0].0, 4096);
        assert_eq!(
            state.entry(),
            ContentDigest::from_bytes(&[vec![b'a'; 4096], b"x".to_vec()].concat())
                .unwrap()
                .entry()
        );
    }

    #[test]
    fn binding_roundtrip_is_canonical() {
        let data = vec![b'z'; 9000];
        let content = ContentDigest::from_bytes(&data).unwrap();
        let mut sha = Sha256::new();
        sha.update(TEXT_DOMAIN);
        sha.update(&data);
        let binding = RecoveryBindingV1::from_states([7; 32], sha.finalize().into(), &content);
        let raw = binding.to_bytes().unwrap();
        assert_eq!(RecoveryBindingV1::from_bytes(&raw).unwrap(), binding);
    }
}
