# Architecture

This document describes the shape of the system: five layers, the three-stage
routing path between them, and the contracts that each layer is expected to
keep across releases.

## 1. Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. Public API                                                        │
│    Tokenizer(family, config=...) · encode / encode_batch(ragged)     │
│    session() context manager · explain() · immutable Config          │
│    typed errors with .code · CLI (artifacts / doctor / inspect)      │
├──────────────────────────────────────────────────────────────────────┤
│ 2. Routing                                                           │
│    probe → plan → execute · native per-input selector                 │
│    RoutingPolicy · immutable fallback chain with reason codes        │
├──────────────────────────────────────────────────────────────────────┤
│ 3. Backends                                                          │
│    HF reference · corrected Gigatoken CPU repair · GPU prebuilt/JIT  │
│    added-token frontend · explicit experimental Fastokens adapter    │
├──────────────────────────────────────────────────────────────────────┤
│ 4. Artifacts and certification registry                              │
│    ArtifactSource (hub / local dir / mirror / air-gapped bundle)     │
│    sha256 verification · semantic fingerprint · registry + digest    │
├──────────────────────────────────────────────────────────────────────┤
│ 5. Session store                                                     │
│    store-core (no unsafe, no bindings) · store-sqlite · Python face  │
│    record format v1 · block hash chain · verification on hit         │
└──────────────────────────────────────────────────────────────────────┘
```

Layers are one-directional: the API layer does not know which backend will run,
backends do not read configuration, and the store does not know how the IDs it
holds were produced. Everything that crosses a layer boundary is either an
immutable value (`Config`, `RoutePlan`, `SessionUpdate`) or a typed error.

### 1.1 Public API

- `Tokenizer(family, config=...)` is the single entry point. `encode` returns
  token IDs; `encode_batch(texts, output="ragged")` returns values plus offsets,
  so that a batch never has to be materialised as a list of lists.
- `session()` is a context manager. `append` returns an update describing
  `replace_from`, `replacement_ids` and `all_ids`, because a correct append may
  rewrite the tail of previously returned IDs. Writes take an
  `expected_revision`; a concurrent write is rejected rather than resolved by
  last-writer-wins.
- Errors carry a machine-readable `.code`; the human-readable message is not
  part of the interface. The code set includes `ArtifactNotFound`,
  `ArtifactHashMismatch`, `UncertifiedTokenizer`, `OracleVersionUnsupported`,
  `BackendUnavailable`, `KernelIncompatible`, `CudaDriverTooOld`,
  `StoreCorrupt`, `SessionStateMismatch` and `SessionRevisionConflict`.
  Fallbacks are reported separately, with reason codes of their own.
- `Config` is immutable. Precedence is: method argument, constructor argument,
  `Config`, configuration file, environment variable, default. Environment
  variables are read once, when a `Config` is constructed.

### 1.2 Routing

See section 2 — the three stages are the substance of this layer. The public
plan and diagnostic records are Python value objects; the hot UTF-8 byte
crossover and added-token necessary-condition gate are projected once into
`toktier-routing-core` and evaluated through the private PyO3 module without a
temporary Python `bytes` allocation. A possible added-token hit still goes
through the exact reference frontend.

### 1.3 Backends

- The **reference backend** wraps Hugging Face `tokenizers` and defines
  correctness. It is always available and never disabled by policy.
- The **fast CPU backend** uses a corrected, data-version-pinned Gigatoken
  native module shipped under the private `toktier._vendor` namespace in the
  core wheel. The planner verifies its vendored delivery identity, build
  version, native-module digest, repair-table digest, oracle version, and
  tokenizer artifact before admitting it. The same paired engine/HF object
  serves stateless encoding and the native store's append callback. A failed
  premise returns the request to the reference path with a machine-readable
  reason.
- The **GPU backend** implements pre-tokenization and BPE encoding as CUDA
  kernels, with batched, fused and CUDA-Graph forms. This release ships a
  multi-architecture prebuilt fatbin and retains a JIT delivery; each has its
  own registry binding and architecture status.
- The **added-token frontend** decides whether a document contains added or
  special tokens, using byte-level lookup tables keyed by the content hash of
  the added-token table. Documents that do so are routed to the reference path.
- **Fastokens** is a separately installed full-session adapter for comparison.
  It can be requested only with `EXPERIMENTAL` policy, is never an automatic
  fallback, and reports that no TokTier exact-ID guarantee applies.

### 1.4 Artifacts and certification registry

- `ArtifactSource` has implementations for the upstream hub, a local directory,
  an internal mirror and an air-gapped bundle. A download is written to a
  temporary file, verified, fsynced, atomically renamed and marked as verified,
  under a per-artifact lock.
- A **semantic fingerprint** binds everything that can change token IDs:
  artifact SHA-256, pipeline fingerprint, added-token fingerprint (content, id,
  special, single-word, lstrip, rstrip, normalized and insertion order),
  reference semantic version, the `add_special_tokens` / normalization /
  special-token / truncation / post-processor policy, the session API semantic
  version, repair backend/version/configuration, and the store format version.
  Modes that a session cannot represent
  are rejected at construction time rather than at first append.
- The **registry** records, per artifact: the three identities used for routing
  eligibility (exact artifact SHA, pipeline capability, added-frontend
  capability), the reference package and version, the certification suite
  version, an evidence id, the recorded counts, and the state of each backend
  with its digest. It is generated by a tool, validated against a JSON schema,
  covered by a root digest, and checked in CI with `--check`; hand edits do not
  pass.
- **Reference version policy.** If the artifact is certified and the installed
  reference matches the certified version, accelerated backends may run. If the
  reference version does not match, accelerated backends are switched off and
  the installed reference runs, with the state reported as `reference-only`. If
  the artifact is not certified, the reference runs and the result is marked as
  uncertified. The package metadata does not pin the reference version.

### 1.5 Session store

- `store-core` holds the record format and the verification logic; it denies
  unsafe code and has no binding dependencies, so it can be tested and fuzzed on
  its own. `store-sqlite` owns the database file exclusively from Rust; the
  Python side never opens it directly. Network filesystems are refused or
  warned about, because file locking there is not dependable.
- **Record format v1 header**: magic, format version, header length, flags,
  explicit little-endian marker, semantic fingerprint, session revision,
  previous and current block hash, full text byte length, stable prefix byte
  length, text tail byte length, token count, replace token offset, payload
  checksum, and the class of boundary predicate that admitted this splice — so a
  reader can re-check the claim instead of trusting the writer.
- **Two invariants.** A wrong key must miss: the key includes the tokenizer
  content hash, the engine identity and the configuration name. A hit must be
  verified: the stored text tail and the hash chain are re-checked, and a failed
  check is counted as a miss. The hash chain gives corruption detection, not
  tamper resistance; authentication would need a keyed construction and is not
  claimed.
- The store keeps the core token stream produced before the post-processor;
  BOS/EOS and template effects are applied on read, so one stored prefix serves
  several request shapes.
- The Python facade supplies the store with the HF full-encode callback and,
  only for an admitted binding, the corrected-Gigatoken append callback. The
  callback returns IDs, spans, kept-token count, and an execution-path label;
  the native transaction commits only after its invariants pass.
- For the certified BPE repair roster, the digest-checked O/S/L/N/M table is
  installed once in the native encoder adapter. Stable-prefix seal decisions
  then execute in Rust with the frozen synchronizing-transition predicate,
  byte-fallback clean-cut rule, added-literal end guard, and a retained repair
  window. No per-seal Python token/span list is materialized.
- Files are created with restrictive permissions, logs do not contain source
  text, and capacity limits, TTL, per-session deletion and a full wipe are part
  of the interface.

## 2. Three-stage routing

Routing is split into three stages so that the decision can be inspected,
logged and tested without running a tokenizer.

```
        probe                      plan                        execute
  ┌───────────────────┐    ┌────────────────────┐    ┌──────────────────────┐
  │ artifact identity │    │ pure function      │    │ run the plan         │
  │ reference version │ →  │ (facts, policy)    │ →  │ per-request checks   │
  │ backend presence  │    │ → immutable        │    │ fallback chain with  │
  │ device / driver   │    │   RoutePlan        │    │ reason codes         │
  │ input properties  │    │ no side effects    │    │ counters + explain() │
  └───────────────────┘    └────────────────────┘    └──────────────────────┘
```

1. **Probe** collects facts: which artifact bytes are loaded, which reference
   version is installed, which backends are importable, what the device and
   driver report, and which properties of the input matter (for example whether
   the added-token frontend finds a candidate). Probing has no opinion.
2. **Plan** is a pure function from those facts plus the `RoutingPolicy` to an
   immutable `RoutePlan`: the backend to use, the fallback order, and the reason
   for each exclusion. Because it is pure, the same facts always yield the same
   plan, and a plan can be produced in tests without hardware.
3. **Execute** runs the plan, applies the per-request checks that the chosen
   path requires, and falls back along the recorded order when a check does not
   pass. Every fallback increments a counter with its reason code, and
   `explain()` reports the plan together with what actually happened.

`RoutingPolicy` values: `CERTIFIED` (default; accelerated routes only where a
certificate covers this exact artifact), `REFERENCE` (reference path only),
`REQUIRE_ACCELERATED` (raise rather than fall back — for benchmarking and for
deployments that would rather fail than change performance characteristics
silently), `EXPERIMENTAL` (permit uncertified combinations, for evaluation).

## 3. Version axes

Six versions are tracked independently, because they move at different speeds:
package version, public API version, registry schema version, store format
version, GPU kernel ABI version, and certification suite version. A change in
one is not, by itself, a reason to change another.

## 4. Repository layout

```
src/toktier/            Python package (maturin mixed layout, abi3)
  artifacts/            manifests, sources, fetch, verification, storage
crates/                 Rust workspace
  toktier-routing-core/  per-input threshold/literal routing + BPE seal predicate
  store-core/            record format, hash chain, verification (no bindings)
  store-sqlite/          exclusive owner of the database file
  toktier-py/            thin binding facade
tools/                  repository validation and development utilities
tests/                  unit, conformance, packaging, and tooling tests
docs/                   contracts, decisions, support matrix, integration notes
```

The native module is private (`toktier._native`), typed (`_native.pyi`,
`py.typed`), and never imported from user code. The C ABI, where it is exposed,
uses the `toktier_` prefix.
