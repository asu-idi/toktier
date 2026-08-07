# Store refactor v1: decisions and deltas against the pre-release prototype

Status: record of the release-form refactor of the session store
(ported from this package's pre-release prototype, store v1). The normative contracts are
`docs/contracts/store-format-v1.md`, `errors.md`, `fingerprint.md`,
`api.md`; this document records implementation decisions taken inside
the store tier and every place where the prototype implementation and
the frozen contracts pulled in different directions.

## 1. Shape

- Three crates, as targeted (no downgrade was needed):
  - `crates/toktier-store-core` -- pure Rust, `#![deny(unsafe_code)]`,
    no PyO3, no database; single pinned dependency (`sha2`).
  - `crates/toktier-store-sqlite` -- `rusqlite` (bundled); the database
    file is exclusively owned by the Rust layer
    (`locking_mode=EXCLUSIVE`), no other process touches it while open.
  - `crates/toktier-py` -- thin PyO3 facade (`[lib] name = "_native"`,
    module `toktier._native` per the repository-level packaging
    ruling); adapts Python callables onto the core encoder trait, maps
    errors onto the structured exception contract, exposes the SQLite
    tier.
- The three prototype-side couplings dissolved as planned:
  `RepairEngine` became the `SessionEncoder` trait (core defines, the
  facade implements via callbacks); `Py<RepairState>` became the plain
  `TailState` with validating mutators; `Py<PyDict>` returns became
  structs (`AppendOutcome` carries `replace_from` / `replacement_ids` /
  `all_ids` exactly as the frozen append contract requires).

## 2. Record semantics: full-stream snapshot vs sealed-delta chain

The frozen format (`store-format-v1.md` Section 2) stores **one full
core-stream snapshot per record**. Findings against the prototype v1,
recorded here:

- Prototype v1 **session** records already carried the full stream
  (sealed ids + tail ids as two arrays); serializing them as one id
  array plus the text tail is a lossless re-shaping. The spec's
  `token_count = full core stream` reading is correct for sessions --
  no spec amendment needed.
- Prototype v1 **node** records (the prefix-sharing block chain used by
  `lookup`) are genuinely *sealed-ids deltas per text block*, chained
  by content address. They are not session snapshots and were not
  force-fitted into the frozen record format. Decision: the node table
  is an **internal acceleration cache**, encoded by
  `toktier-store-core/src/sidecar.rs` (`NodeCacheRecord`, magic
  `TKNC`), outside the portable format. Rationale: losing a node can
  only ever cost a lookup hit, never correctness; the cache layout is
  implementation-internal and never travels between implementations.
  It follows the same decode discipline as the frozen format (checked
  arithmetic, strict bounds, domain-tagged SHA-256 checksums, exact
  consumption), and the semantic fingerprint participates in every
  node key (wrong key structurally cannot hit).
- Prototype v1 sessions also carried incremental bookkeeping that the
  frozen record intentionally does not (character-unit counters, block
  chain attachment, seal log, pending block buffer, scan memo).
  Decision: a **session sidecar** (`SessionSidecar`, magic `TKSS`),
  bound to its record through the record's `curr_block_hash`.
  - `import_session(record)` alone is fully correct (ids, revision
    chain, byte-unit stable prefix) but conservative: chain detached,
    character counters restart at the tail origin. This is the
    portable-import path.
  - `import_session_with_sidecar(record, sidecar)` restores exact
    pre-save behavior; the SQLite tier always stores and uses the
    pair. A sidecar that fails verification or does not bind to its
    record is a loud, counted rejection (session imports are loud;
    only the lookup path degrades silently to counted misses).

## 3. Contract mappings and small deltas

- **Revisions**: genesis is 0 per the spec (prototype v1 had no
  revisions). Every successful `append` (including `noop`) increments
  the revision and re-links the per-session chain hash
  (Section 4.2 `link` construction, computed at every commit point:
  put / append / fork / lookup-materialization). `fork` and
  lookup-materialized sessions start a fresh lineage at revision 0
  with a zero `prev_block_hash` (genesis rule is about chain origin,
  not empty content). Single-record decode enforces the genesis rule
  in both directions (`revision == 0` iff `prev` all-zero); full
  predecessor linkage is only checkable when walking a chain.
- **Witness categories** (u16 registry, four values): sessions are
  stamped at creation from the engine; the store refuses to mix
  categories within a session lineage (`SESSION_STATE_MISMATCH`) and
  rejects unknown values on decode (`STORE_FORMAT_UNSUPPORTED`).
  Prototype kinds map: `letter_space`(+sync, incl. family-specific sync
  profiles, which are bound in the fingerprint) to
  `WITNESS_BPE_SYNC_TRANSITION`; `wordpiece_continuation` to `0x0002`;
  `metaspace_word_start` to `0x0003`; uncertified/unknown to `0x0000`.
  The category-0 cross-invariant (`stable_prefix == 0`,
  `replace_token_offset == 0`) is enforced on encode and decode; for
  category-0 sessions the store records `replace_token_offset = 0`
  even though the in-memory append outcome may report a larger valid
  `replace_from` (the invariant `all_ids == old[:replace_from] +
  replacement_ids` holds for both values).
- **Error codes** follow `errors.md`: corruption shapes map to
  `STORE_CORRUPT`, well-formed-but-newer shapes to
  `STORE_FORMAT_UNSUPPORTED`, wrong-key/witness mismatches to
  `SESSION_STATE_MISMATCH`, conflicts to `SESSION_REVISION_CONFLICT`,
  configuration to `CONFIG_INVALID`. Plain argument misuse stays
  `ValueError`/`KeyError` at the Python surface (unknown handle is
  `KeyError` for prototype-battery parity). Structured exceptions carry
  `.code` and `.details`.
- **Node keys** are full 32-byte SHA-256 (the prototype used 20-byte
  truncation); all hash domains are new (`toktier.store.v1.*`), so
  prototype-era bytes are not readable by the release store --
  intentional, the release format starts at v1.
- **Eviction determinism**: the prototype LRU broke `last_used` ties (only
  possible after `fork`) by hash-map iteration order; the port breaks
  ties toward the smallest handle. Not observable in the equivalence
  battery (tie scenarios avoided there because the prototype side is
  itself nondeterministic in that corner).
- **Stats**: prototype counter names kept, `revision_conflicts` added,
  `schema` renamed to `format` (`toktier.store.v1`). `extends` counts
  only appends that pass the existence/witness/revision gates.
- **Engine trust model**: the store structurally verifies every append
  claim (text grew by exactly `delta`; the claimed kept-token prefix is
  bit-identical to the previous encoding) so a misbehaving encoder is
  loud, but the certified-content guarantee itself (ids equal a
  from-scratch reference encode) lives with the engine, exactly as in
  the prototype.
- **`corrupt_node_for_tests`** exists only under the core `testing`
  feature (the facade enables it; the prototype build carried it
  unconditionally).

## 4. Dependency pins

- `sha2 = "=0.11.0"` (same pin as the prototype, same major).
- `rusqlite = "=0.39.0"` bundled (first fetch 2026-08-05; 0.40.x
  requires a newer rustc than the frozen 1.93.1 toolchain --
  `libsqlite3-sys 0.38` uses `cfg_select!`).
- `pyo3 = "=0.23.5"` (`abi3-py39`, `extension-module`).

## 5. Known limitations (v1, documented not implied)

- Integrity machinery detects corruption; it is not a keyed MAC and
  does not resist an adversary who rewrites whole records consistently
  (spec Section 4.4).
- Per-commit chain hashing walks the full core stream (SHA-256 over
  ids + tail per append). Correct-first; measured optimization is
  future work and must not change the contract.
- The conservative (record-only) import restarts character counters;
  byte counters and ids are exact. Full fidelity requires the sidecar
  (always present in the SQLite tier).
