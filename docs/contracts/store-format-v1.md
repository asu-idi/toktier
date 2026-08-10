# Session store record format v1 (frozen)

Status: frozen. This document is the normative byte-level contract for
store format version 1. The store implementation consumes the semantic
fingerprint as an opaque 32-byte value; the fingerprint preimage is
specified separately in `fingerprint.md` and is not re-derived here.

Red lines inherited by every rule below: a wrong key must never hit; a
hit must verify; on the read path any verification failure is a counted
miss, never a wrong result. We prefer a miss over a wrong result.

## 1. General conventions

- All multi-byte integers are **little-endian**. The header carries an
  explicit endianness marker as a tripwire; readers verify it.
- All length arithmetic during decode uses checked (overflow-detecting)
  operations. Bytes read from disk are untrusted until verified.
- Records store the **pre-postprocessor core token stream**. BOS/EOS and
  other postprocessor output are never persisted; whether and when a reader
  applies the postprocessor after reading is that reader's contract (the 0.x
  facade serves postprocessed requests outside the store; see `facade.md`
  Section 2).
- Hash domain-separation tags are ASCII strings including a trailing
  NUL, as written below.

## 2. Record layout

A record is: fixed header (200 bytes) + optional header extension (up to
`header_length`) + payload. The payload is the token id array followed by
the text tail.

### 2.1 Fixed header field table (offsets in bytes)

| Offset | Size | Field | Type | v1 constraint (violation => reject) |
|---|---|---|---|---|
| 0 | 8 | `magic` | bytes | must equal ASCII `TOKTIERS` |
| 8 | 2 | `format_version` | u16 | must equal `1`; other values => `STORE_FORMAT_UNSUPPORTED` |
| 10 | 2 | `header_length` | u16 | `200 <= header_length <= 4096`; multiple of 8; `<=` record size |
| 12 | 4 | `flags` | u32 | bits 0-15 are **mandatory** feature bits: any set bit unknown to the reader => `STORE_FORMAT_UNSUPPORTED`. Bits 16-31 are advisory: unknown set bits are ignored. v1 assigns no bits; writers write 0. |
| 16 | 1 | `endianness` | u8 | must equal `0x01` (little-endian marker) |
| 17 | 1 | `reserved0` | u8 | must equal 0 |
| 18 | 2 | `witness_category` | u16 | must be an assigned value (Section 3); unknown => `STORE_FORMAT_UNSUPPORTED` |
| 20 | 4 | `reserved1` | u32 | must equal 0 |
| 24 | 32 | `semantic_fingerprint` | bytes | opaque 32 bytes; compared for exact equality against the caller's fingerprint (see `fingerprint.md`); mismatch => miss (wrong key must never hit) |
| 56 | 8 | `session_revision` | u64 | genesis is 0; strictly increasing per session |
| 64 | 32 | `prev_block_hash` | bytes | all-zero for `session_revision == 0`; otherwise must equal the `curr_block_hash` of the predecessor record |
| 96 | 32 | `curr_block_hash` | bytes | must recompute per Section 4.2 |
| 128 | 8 | `full_text_byte_length` | u64 | `<= 2^40` |
| 136 | 8 | `stable_prefix_byte_length` | u64 | `<= full_text_byte_length` |
| 144 | 8 | `text_tail_byte_length` | u64 | `<= 2^31`; `stable_prefix_byte_length + text_tail_byte_length == full_text_byte_length` (checked arithmetic) |
| 152 | 8 | `token_count` | u64 | `<= 2^31` |
| 160 | 8 | `replace_token_offset` | u64 | `<= token_count` |
| 168 | 32 | `payload_checksum` | bytes | must recompute per Section 4.3 |

Total fixed portion: 200 bytes.

Field semantics:

- `stable_prefix_byte_length` -- byte length of the certified stable text
  prefix. Token ids covering the stable prefix are the sealed portion;
  the prefix bytes themselves are not stored in this record (they are
  reachable through the chain). The 0.x Python facade may keep a private,
  record-hash-bound digest sidecar so caller-presented historical bytes can be
  verified and reattached after restart; that sidecar is not part of this
  portable record format and cannot supply token IDs.
- `text_tail_byte_length` -- byte length of the text tail stored in this
  record's payload: the exact raw suffix of the full text starting at
  the stable prefix end. Must decode as valid UTF-8 (which also
  guarantees the cut falls on a code point boundary).
- `token_count` -- number of u32 token ids in the payload: the full core
  stream for the session at this revision.
- `replace_token_offset` -- the `replace_from` of the append that
  produced this revision (settled micro-decision): a **zero-based token
  index into the pre-postprocessor core stream**, with the invariant
  `all_ids == old_ids[:replace_from] + replacement_ids`. Recorded for
  verification and diagnostics; a full re-encode records 0.

### 2.2 Optional header extension (bytes 200 .. header_length)

- TLV sequence: `type` u16, `length` u16, `value` (`length` bytes),
  packed without gaps. `type == 0x0000` is padding (value ignored).
- v1 defines no non-padding types. Readers skip unknown TLV types.
  A future field that must not be skipped will be accompanied by a
  mandatory flag bit (Section 2.1 `flags`), which old readers reject -- this is
  the designed forward-compatibility mechanism (settled micro-decision:
  old readers skip trailing optional fields via `header_length`; unknown
  mandatory flags and unknown witness categories are rejected).
- TLV parsing uses checked arithmetic; a TLV extending past
  `header_length` => reject as corrupt.

### 2.3 Payload

| Order | Content | Size |
|---|---|---|
| 1 | token ids, u32 LE each | `token_count * 4` bytes |
| 2 | text tail, UTF-8 | `text_tail_byte_length` bytes |

- Record size must equal exactly
  `header_length + token_count * 4 + text_tail_byte_length`
  (checked arithmetic). Trailing bytes => reject.
- Ids come first so they stay 4-byte aligned (`header_length` is a
  multiple of 8).
- Token id values carry no format-level bound beyond u32; vocabulary
  range checks are an engine concern, not a decoder concern.

## 3. Witness category registry (u16, append-only)

The witness category records **which class of certification predicate
proved the current safe cut point** (the boundary between the stable
prefix and the text tail). Readers can therefore re-verify a cut with
the matching predicate class instead of trusting the writer.

| Value | Name | Meaning |
|---|---|---|
| `0x0000` | `WITNESS_NONE_FULL_REENCODE` | No certified incremental cut backs this record; the stream came from a from-scratch encode. Cross-field invariant: `stable_prefix_byte_length == 0` and `replace_token_offset == 0`. |
| `0x0001` | `WITNESS_BPE_SYNC_TRANSITION` | Byte-level BPE synchronizing-transition predicate certified the cut. Profile parameters (including family-specific sync profiles) are bound inside the semantic fingerprint's pipeline component; the category records the predicate class. |
| `0x0002` | `WITNESS_WORDPIECE_CONTINUATION` | WordPiece continuation witness-anchor predicate certified the cut. |
| `0x0003` | `WITNESS_METASPACE_WORD_START` | Metaspace/word-start-marker predicate certified the cut. |

- All other values are unassigned => `STORE_FORMAT_UNSUPPORTED`.
- Assignment is append-only through this document; values are never
  reused or re-meant.
- Engines without an applicable certificate never seal a prefix: they
  write category `0x0000` records with the whole text in the tail. This
  is correct by construction -- the tail simply grows and the caps count
  it.

## 4. Integrity: digests, checksum, chain

Algorithm for all three constructions: SHA-256.

### 4.1 Payload digest (internal value, not stored)

```
payload_digest = SHA-256( "toktier.store.v1.payload\0" || payload_bytes )
```

Computed in one pass over the payload; both Section 4.2 and Section 4.3 consume it, so
the payload is traversed once during verification.

### 4.2 Chain link (`curr_block_hash`)

```
curr_block_hash = SHA-256(
    "toktier.store.v1.link\0"
    || prev_block_hash                      (32 bytes)
    || semantic_fingerprint                 (32 bytes)
    || LE64(session_revision)
    || LE64(full_text_byte_length)
    || LE64(stable_prefix_byte_length)
    || LE64(text_tail_byte_length)
    || LE64(token_count)
    || LE64(replace_token_offset)
    || LE16(witness_category)
    || payload_digest                       (32 bytes)
)
```

- Because `semantic_fingerprint` enters every link, a wrong key can
  never produce a verifying chain: wrong key must miss.
- Genesis rule: `session_revision == 0` => `prev_block_hash` is 32 zero
  bytes.

### 4.3 Record checksum (`payload_checksum` field)

```
payload_checksum = SHA-256(
    "toktier.store.v1.record\0"
    || header_bytes[0 .. header_length)     (with the 32 checksum bytes at offset 168 set to zero)
    || payload_digest                       (32 bytes)
)
```

Coverage: the entire header directly (including the optional extension),
and the payload transitively through `payload_digest`. Every byte of the
record is covered by exactly one digest pass.

### 4.4 What integrity does and does not claim

The checksum and hash chain provide **corruption detection**. They do
not, by themselves, provide tamper resistance against an adversary who
can rewrite whole records consistently; that would require a keyed MAC
and is out of scope for format v1 (documented honestly, not implied).

## 5. Decode procedure (normative order)

1. Bounds gate: record size `>= 200`; verify `magic`, `endianness`,
   `reserved0`, `reserved1`, `format_version`, `header_length` bounds.
2. Field bounds: every constraint in the Section 2.1 table, all arithmetic
   checked; parse the TLV extension per Section 2.2.
3. Size closure: record size equals header + ids + tail exactly.
4. Integrity: compute `payload_digest`; verify `payload_checksum`;
   verify `curr_block_hash`; when walking a chain, verify
   `prev_block_hash` linkage and `session_revision` monotonicity.
5. Semantic checks: witness category assigned; cross-field invariants
   (Section 3); text tail is valid UTF-8; fingerprint equality against the
   caller's expected fingerprint.

Failure handling by path:

- **Read/lookup path**: any failure in steps 1-5 => treat the record as a
  miss, increment the corresponding rejection counter, continue. The
  read path never returns data from a record that failed any check, and
  never raises for integrity reasons.
- **Explicit verify path** (integrity-check API): structural and
  integrity failures raise `StoreCorrupt` (`STORE_CORRUPT`);
  well-formed-but-newer records raise `StoreFormatUnsupported`
  (`STORE_FORMAT_UNSUPPORTED`).

## 6. Writer obligations

- Writers write format_version 1, zero flags, zero reserved fields, and
  no TLVs (padding only as needed to reach a multiple of 8).
- Writes are atomic at the record level (temporary + rename, or the
  transactional guarantees of the containing database); a torn write
  must never be observable as a verifying record. Concurrency uses
  optimistic `expected_revision`; conflicts surface as
  `SESSION_REVISION_CONFLICT` (see `errors.md`). Last-writer-wins is not
  offered.
- Only streams produced by certified configurations may be written to a
  persistent store. Sessions running under `EXPERIMENTAL` routing policy
  are in-memory only (see `routing.md`); their streams carry no
  certification claim and must never become replayable into a certified
  configuration through the store.

## 7. Division of responsibility (settled micro-decision)

The store treats `semantic_fingerprint` as an opaque 32-byte input and
never inspects or re-derives it. Producing the fingerprint -- including
its preimage encoding and every binding decision -- is the contract of
`fingerprint.md`. This split lets the store be tested, fuzzed, and
verified with arbitrary 32-byte values.
