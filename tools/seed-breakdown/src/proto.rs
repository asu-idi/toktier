//! Measurement-only prototypes for the PLAN/163 W3 direct cells.
//!
//! Every function here deliberately reimplements a candidate design outside
//! the certified product source so its cost can be observed directly. A
//! prototype result is only accepted after an element-for-element (spans) or
//! bit-for-bit (digests) comparison against the corresponding product
//! implementation; a sample whose comparison fails is reported as an error
//! and never enters the timing record.

use sha2::{Digest, Sha256};

/// Mirror of the store-format payload-digest domain prefix. Agreement with
/// the product constant is enforced by the digest-equality assertion in every
/// hashing cell, so any drift fails the run instead of skewing it.
pub const DOMAIN_PAYLOAD: &[u8] = b"toktier.store.v1.payload\0";

/// IDs per stack chunk when feeding a hash in blocks (4096 * 4 = 16 KiB).
const CHUNK_IDS: usize = 4096;

pub type ProtoResult<T> = Result<T, String>;

fn token_len(lengths: &[usize], id: u32) -> ProtoResult<usize> {
    let length = *lengths
        .get(id as usize)
        .ok_or_else(|| format!("unknown vocabulary id {id}"))?;
    if length == 0 {
        return Err(format!("zero-byte vocabulary id {id}"));
    }
    Ok(length)
}

fn to_u32(value: usize) -> ProtoResult<u32> {
    u32::try_from(value).map_err(|_| "span exceeds u32".to_owned())
}

// ---------------------------------------------------------------- spans --

/// S2, ASCII path: one pass over the ID row filling the final `u32`
/// start/end arrays directly. No `Vec<usize>` temporaries, no pair vector,
/// no byte-to-character map.
pub fn soa_spans_ascii(
    ids: &[u32],
    lengths: &[usize],
    text_len: usize,
) -> ProtoResult<(Vec<u32>, Vec<u32>)> {
    let mut starts = Vec::with_capacity(ids.len());
    let mut ends = Vec::with_capacity(ids.len());
    let mut cursor = 0usize;
    for &id in ids {
        let length = token_len(lengths, id)?;
        let start = to_u32(cursor)?;
        cursor = cursor
            .checked_add(length)
            .ok_or("token byte length overflow")?;
        let end = to_u32(cursor)?;
        starts.push(start);
        ends.push(end);
    }
    if cursor != text_len {
        return Err(format!(
            "token bytes do not close: tokens={cursor}, text={text_len}"
        ));
    }
    Ok((starts, ends))
}

/// Streaming byte-to-character cursor over a UTF-8 window. It advances
/// strictly forward and reports the index of the character containing a
/// given relative byte offset.
struct CharCursor<'a> {
    rest: std::str::CharIndices<'a>,
    index: u32,
    start: usize,
    end: usize,
    exhausted: bool,
}

impl<'a> CharCursor<'a> {
    fn new(window: &'a str, base_char: u32) -> Self {
        let mut rest = window.char_indices();
        match rest.next() {
            Some((_, first)) => Self {
                rest,
                index: base_char,
                start: 0,
                end: first.len_utf8(),
                exhausted: false,
            },
            None => Self {
                rest,
                index: base_char,
                start: 0,
                end: 0,
                exhausted: true,
            },
        }
    }

    /// Character index (absolute, including the base) containing `byte`.
    fn char_containing(&mut self, byte: usize) -> ProtoResult<u32> {
        if self.exhausted || byte < self.start {
            return Err("byte offset is outside the streamed window".to_owned());
        }
        while self.end <= byte {
            match self.rest.next() {
                Some((offset, value)) => {
                    self.index = self
                        .index
                        .checked_add(1)
                        .ok_or("character index overflow")?;
                    self.start = offset;
                    self.end = offset + value.len_utf8();
                }
                None => return Err("byte offset is beyond the end of the window".to_owned()),
            }
        }
        Ok(self.index)
    }
}

/// S2, Unicode path: a single forward merge of UTF-8 character boundaries
/// and token byte endpoints. Byte-fallback tokens whose byte interval lies
/// inside one character share that character's span, matching the product
/// bridge. `expected_bytes` enables the full-line closure check; window
/// rebuilds pass `None` because they cover only part of the text.
pub fn soa_spans_unicode_window(
    ids: &[u32],
    lengths: &[usize],
    window: &str,
    base_char: u32,
    first_token_rel_byte: usize,
    expected_bytes: Option<usize>,
) -> ProtoResult<(Vec<u32>, Vec<u32>)> {
    let mut starts = Vec::with_capacity(ids.len());
    let mut ends = Vec::with_capacity(ids.len());
    let mut cursor = CharCursor::new(window, base_char);
    let mut byte = first_token_rel_byte;
    for &id in ids {
        let length = token_len(lengths, id)?;
        let start_char = cursor.char_containing(byte)?;
        byte = byte
            .checked_add(length)
            .ok_or("token byte length overflow")?;
        let end_char = cursor
            .char_containing(byte - 1)?
            .checked_add(1)
            .ok_or("character index overflow")?;
        starts.push(start_char);
        ends.push(end_char);
    }
    if let Some(expected) = expected_bytes {
        if byte != expected {
            return Err(format!(
                "token bytes do not close: tokens={byte}, text={expected}"
            ));
        }
    }
    Ok((starts, ends))
}

/// S3(a): allocation-free streaming closure check. One pass over the ID row
/// summing frozen byte lengths; returns the final byte cursor, which must
/// equal the input length.
pub fn closure_sum(ids: &[u32], lengths: &[usize], text_len: usize) -> ProtoResult<usize> {
    let mut cursor = 0usize;
    for &id in ids {
        cursor = cursor
            .checked_add(token_len(lengths, id)?)
            .ok_or("token byte length overflow")?;
    }
    if cursor != text_len {
        return Err(format!(
            "token bytes do not close: tokens={cursor}, text={text_len}"
        ));
    }
    Ok(cursor)
}

/// S3(b), ASCII path: construct only the tail-window spans by
/// back-projecting byte lengths from the end of the text. Needs the text
/// length and the tail IDs; no prefix scan and no full materialization.
pub fn tail_spans_ascii_from_suffix(
    tail_ids: &[u32],
    lengths: &[usize],
    text_len: usize,
) -> ProtoResult<(Vec<u32>, Vec<u32>)> {
    let mut ends_rev = Vec::with_capacity(tail_ids.len());
    let mut cursor = text_len;
    for &id in tail_ids.iter().rev() {
        let length = token_len(lengths, id)?;
        ends_rev.push(cursor);
        cursor = cursor
            .checked_sub(length)
            .ok_or("tail extends beyond the start of the text")?;
    }
    let mut starts = Vec::with_capacity(tail_ids.len());
    let mut ends = Vec::with_capacity(tail_ids.len());
    let mut start = cursor;
    for end in ends_rev.into_iter().rev() {
        starts.push(to_u32(start)?);
        ends.push(to_u32(end)?);
        start = end;
    }
    Ok((starts, ends))
}

// ---------------------------------------------------------- checkpoints --

/// A sparse cumulative anchor recorded every `interval` tokens. The
/// character fields describe the character containing the last byte of the
/// last covered token, so a rebuild can start its streaming merge at a
/// character boundary even when the token boundary splits a character.
#[derive(Clone, Copy, Debug)]
pub struct Checkpoint {
    pub tokens: usize,
    pub byte_end: usize,
    pub last_char_start: usize,
    pub last_char_index: u32,
}

/// S3(c) build, ASCII path: one pass over the ID row; characters equal
/// bytes, so no text scan is needed.
pub fn build_checkpoints_ascii(
    ids: &[u32],
    lengths: &[usize],
    text_len: usize,
    interval: usize,
) -> ProtoResult<Vec<Checkpoint>> {
    let mut checkpoints = Vec::with_capacity(ids.len() / interval + 1);
    let mut cursor = 0usize;
    for (index, &id) in ids.iter().enumerate() {
        cursor = cursor
            .checked_add(token_len(lengths, id)?)
            .ok_or("token byte length overflow")?;
        if (index + 1) % interval == 0 {
            checkpoints.push(Checkpoint {
                tokens: index + 1,
                byte_end: cursor,
                last_char_start: cursor - 1,
                last_char_index: to_u32(cursor - 1)?,
            });
        }
    }
    if cursor != text_len {
        return Err(format!(
            "token bytes do not close: tokens={cursor}, text={text_len}"
        ));
    }
    Ok(checkpoints)
}

/// S3(c) build, Unicode path: one streaming merge over the ID row and the
/// UTF-8 character boundaries, recording byte and character anchors every
/// `interval` tokens without materializing any span array.
pub fn build_checkpoints_unicode(
    ids: &[u32],
    lengths: &[usize],
    text: &str,
    interval: usize,
) -> ProtoResult<Vec<Checkpoint>> {
    let mut checkpoints = Vec::with_capacity(ids.len() / interval + 1);
    let mut cursor = CharCursor::new(text, 0);
    let mut byte = 0usize;
    for (index, &id) in ids.iter().enumerate() {
        byte = byte
            .checked_add(token_len(lengths, id)?)
            .ok_or("token byte length overflow")?;
        if (index + 1) % interval == 0 {
            let last_char_index = cursor.char_containing(byte - 1)?;
            checkpoints.push(Checkpoint {
                tokens: index + 1,
                byte_end: byte,
                last_char_start: cursor.start,
                last_char_index,
            });
        }
    }
    if byte != text.len() {
        return Err(format!(
            "token bytes do not close: tokens={byte}, text={}",
            text.len()
        ));
    }
    Ok(checkpoints)
}

/// S3(c) rebuild, ASCII path: fill one window's spans directly from the
/// anchoring checkpoint's byte offset.
pub fn rebuild_window_ascii(
    window_ids: &[u32],
    lengths: &[usize],
    start_byte: usize,
) -> ProtoResult<(Vec<u32>, Vec<u32>)> {
    let mut starts = Vec::with_capacity(window_ids.len());
    let mut ends = Vec::with_capacity(window_ids.len());
    let mut cursor = start_byte;
    for &id in window_ids {
        let length = token_len(lengths, id)?;
        starts.push(to_u32(cursor)?);
        cursor = cursor
            .checked_add(length)
            .ok_or("token byte length overflow")?;
        ends.push(to_u32(cursor)?);
    }
    Ok((starts, ends))
}

/// S3(c) rebuild, Unicode path: stream from the checkpoint's character
/// anchor and merge the window's token byte endpoints with the UTF-8
/// character boundaries.
pub fn rebuild_window_unicode(
    window_ids: &[u32],
    lengths: &[usize],
    text: &str,
    start_byte: usize,
    anchor: Checkpoint,
) -> ProtoResult<(Vec<u32>, Vec<u32>)> {
    if anchor.last_char_start > start_byte {
        return Err("checkpoint anchor is past the window start".to_owned());
    }
    let window = text
        .get(anchor.last_char_start..)
        .ok_or("checkpoint anchor is not a character boundary")?;
    soa_spans_unicode_window(
        window_ids,
        lengths,
        window,
        anchor.last_char_index,
        start_byte - anchor.last_char_start,
        None,
    )
}

// -------------------------------------------------------------- hashing --

fn update_ids_chunked(hasher: &mut Sha256, ids: &[u32]) {
    let mut buffer = [0u8; CHUNK_IDS * 4];
    for chunk in ids.chunks(CHUNK_IDS) {
        let mut used = 0usize;
        for &value in chunk {
            buffer[used..used + 4].copy_from_slice(&value.to_le_bytes());
            used += 4;
        }
        hasher.update(&buffer[..used]);
    }
}

/// H3: the payload digest fed through a 16 KiB stack buffer instead of one
/// 4-byte update per ID. The hashed byte sequence is identical; the caller
/// asserts the digest against the product function before accepting timing.
pub fn payload_digest_chunked(id_parts: &[&[u32]], tail_text: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_PAYLOAD);
    for part in id_parts {
        update_ids_chunked(&mut hasher, part);
    }
    hasher.update(tail_text);
    hasher.finalize().into()
}

/// H2 setup: the running prefix hasher a store session would keep at the
/// sealed boundary (domain prefix plus every sealed ID, little-endian).
pub fn sealed_prefix_hasher(sealed_ids: &[u32]) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_PAYLOAD);
    update_ids_chunked(&mut hasher, sealed_ids);
    hasher
}

/// H2 timed step: clone the saved prefix state and feed only the tail. The
/// result must be bit-identical to the product's full recomputation.
pub fn commit_digest_from_prefix(prefix: &Sha256, tail_ids: &[u32], tail_text: &[u8]) -> [u8; 32] {
    let mut hasher = prefix.clone();
    update_ids_chunked(&mut hasher, tail_ids);
    hasher.update(tail_text);
    hasher.finalize().into()
}
