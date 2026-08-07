# Semantic fingerprint contract (v1, frozen)

Status: frozen. This document specifies the exact preimage bytes of the
semantic fingerprint, the 32-byte value that keys session store records
(`store-format-v1.md` consumes it as opaque input). Two configurations
receive the same fingerprint if and only if their preimages are
byte-identical under the rules below.

Purpose: the fingerprint binds every input that can change token output.
If any bound component differs, the fingerprint differs, and a store
lookup misses. A wrong key must never hit; we prefer a miss over a wrong
result.

## 1. Digest algorithm (frozen)

```
semantic_fingerprint = SHA-256( preimage )        # 32 bytes
```

SHA-256 is selected: it is already the project-wide content-hash
algorithm (artifact pinning, store integrity), it needs no additional
dependency, and fingerprint computation is a per-construction operation
where throughput is immaterial.

## 2. Preimage structure (frozen)

```
preimage = domain_tag || record_1 || record_2 || ... || record_n
domain_tag = ASCII "toktier.fingerprint.v1\0"     (23 bytes, includes NUL)
```

Each record encodes one field:

```
record = LE16(field_id) || LE32(byte_length_of(value)) || value
```

- Records appear in **strictly ascending `field_id` order**.
- **Every defined field id appears exactly once** (fixed schema -- no
  omission). Absence is encoded explicitly (Section 3), so absent and default
  can never collide and record ordering is never ambiguous.
- Any change to the field set or encoding rules requires a new domain
  tag (`toktier.fingerprint.v2\0`), which changes every fingerprint  --
  deliberate: readers of old stores keep missing rather than guessing.

## 3. Value encodings (frozen)

| Kind | Encoding |
|---|---|
| bytes | the raw bytes |
| string | `0x01` presence byte followed by the exact UTF-8 bytes of the string. **No Unicode normalization is applied** -- bytes are taken as-is from the source, because normalizing could merge artifacts that differ on disk. |
| absent | the single byte `0x00`. Distinct from every present value, including present-and-empty (`0x01` with zero following bytes) and present-default. |
| bool | `0x01 0x00` (false) or `0x01 0x01` (true) -- presence byte then value byte |
| u64 | `0x01` presence byte followed by 8 bytes LE |
| digest32 | `0x01` presence byte followed by 32 raw bytes |
| list | `0x01` presence byte, `LE32(count)`, then each element's encoding concatenated in order |

Rationale for the presence byte on every scalar: with a fixed schema, a
field explicitly set to its default value and a field left absent encode
differently (`0x01...` vs `0x00`), so "absent vs default" is distinguished
structurally, not by per-field convention.

## 4. Field table (frozen; ids are append-only)

| field_id | Name | Kind | Contents |
|---|---|---|---|
| `0x0001` | `artifact_sha256` | digest32 | SHA-256 of the exact tokenizer artifact bytes (the pinned `tokenizer.json`). Sibling artifacts (same pipeline, different added tables) therefore fingerprint differently, which is required for store correctness. |
| `0x0002` | `pipeline_fingerprint` | digest32 | Digest of the core pipeline, excluding added tokens (Section 5). |
| `0x0003` | `added_tokens` | list | The full added-token table encoded inline, **in artifact insertion order** (Section 6). |
| `0x0004` | `oracle_package` | string | Oracle package identifier, e.g. `tokenizers`. |
| `0x0005` | `oracle_semantic_id` | string | Oracle **semantic** version identifier assigned by the support registry (Section 7), not the raw package version string. |
| `0x0006` | `add_special_tokens_policy` | string | v1 value: `read_time` -- specials are not stored; the postprocessor is applied at read time. |
| `0x0007` | `normalization_policy` | string | v1 value: `artifact_default` -- the normalizer exactly as the artifact declares; overrides are not supported in v1. |
| `0x0008` | `special_token_extraction_policy` | string | v1 value: `artifact_default` -- added-token extraction per the artifact's declared flags. |
| `0x0009` | `truncation_policy` | string | v1 value: `none`. Sessions reject truncation and padding at construction (`UNSUPPORTED_CONFIG`), so no other value is reachable in v1. |
| `0x000A` | `postprocessor_policy` | string | v1 value: `read_time` -- stores hold the pre-postprocessor core stream. |
| `0x000B` | `session_api_version` | u64 | Semantic version of the session/append contract. v1 value: `1`. Bumped only when append/return semantics change the meaning of stored streams. |
| `0x000C` | `store_format_version` | u64 | v1 value: `1`. |

Policy fields are short ASCII identifiers from vocabularies fixed here;
any future new mode introduces a new identifier string, which changes
the fingerprint -- exactly the desired effect.

Considered and not bound: a family alias/name field. Family names are
registry-level aliases with no semantic content beyond the artifact they
resolve to; binding them would make renames invalidate stores without
any behavioral change.

The frozen v1 preimage above describes certified streams, whose producing
backend is id-equal to the reference. The 0.x facade also exposes an explicit
experimental repair adapter, so its concrete persistent-entry key uses the
separate domain `toktier.facade.fingerprint.v1\0` and additionally binds the
repair backend, repair configuration, engine version, native-module
digest, and repair-table digest. This stronger 0.x key prevents state created by
Fastokens, corrected Gigatoken, or HF-only execution from crossing engine
meanings. It does not alter the frozen field ids above; the facade contract is
the operative 0.x surface.

The redundancy among `0x0001`, `0x0002`, and `0x0003` (the artifact hash
already covers pipeline and added tokens) is intentional: the fingerprint
is a conjunction, redundancy in a hash preimage is harmless, and the
separated components let the fingerprint survive future artifact
container changes without weakening any binding.

## 5. `pipeline_fingerprint` (frozen)

```
pipeline_fingerprint = SHA-256(
    "toktier.pipeline.v1\0"
    || canonical_json( {
         "decoder":        <decoder section or null>,
         "model":          <model section or null>,
         "normalizer":     <normalizer section or null>,
         "pre_tokenizer":  <pre_tokenizer section or null>,
       } )
)
```

- The four sections are taken from the parsed artifact
  (`tokenizer.json`) verbatim as JSON values; a section missing from the
  artifact is encoded as JSON `null` (distinct from `{}`).
- `canonical_json` is RFC 8785 (JSON Canonicalization Scheme): UTF-8,
  object keys sorted by code point, no insignificant whitespace, JCS
  number formatting.
- Added tokens are deliberately excluded here; they are bound separately
  (Section 6), which is what makes pipeline capability identity usable for
  sibling-artifact reasoning in the registry.

## 6. `added_tokens` list encoding (frozen)

Element order: the order in which entries appear in the artifact's
added-token declarations (insertion order). Order is part of the
fingerprint because extraction behavior can depend on it.

Each element is the concatenation of, in this order:

| # | Sub-field | Kind |
|---|---|---|
| 1 | `content` | string (exact UTF-8 bytes, no normalization) |
| 2 | `id` | u64 |
| 3 | `special` | bool |
| 4 | `single_word` | bool |
| 5 | `lstrip` | bool |
| 6 | `rstrip` | bool |
| 7 | `normalized` | bool |

Sub-fields are encoded with the Section 3 rules (each carries its presence
byte); all seven are present after artifact resolution, but the encoding
keeps the uniform rule rather than a special case.

## 7. Oracle semantic id (frozen policy)

- `oracle_semantic_id` names a **behavior equivalence class** of oracle
  package versions, assigned in the support registry (see `registry.md`).
- Initial policy is conservative: each certified exact package version
  receives its own semantic id. Widening a semantic id to cover
  additional package versions requires certification evidence that the
  versions are behaviorally equivalent on the certified surface.
- Binding the semantic id (rather than the raw package version) means an
  oracle upgrade that is certified behavior-preserving keeps stored
  sessions valid, while any uncertified upgrade misses -- the safe
  direction in both cases.

## 8. Binding-set summary (mirrors the adoption baseline)

The fingerprint binds: exact artifact hash; pipeline fingerprint; the
complete added-token table with content/id/special/single_word/lstrip/
rstrip/normalized and insertion order; oracle semantic version; the
add_special_tokens / normalization / special-token / truncation /
postprocessor policies; the session API semantic version; and the store
format version. Modes the session subsystem does not support (padding
and similar) are rejected at construction, so they need no binding: no
stored stream can exist under them.
