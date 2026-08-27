# Architecture

This document describes the shape of the system: five layers, the three-stage
routing path between them, and the contracts that each layer is expected to
keep across releases.

## 1. Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. Public API                                                        │
│    load / from_pretrained / Tokenizer · encode / encode_batch        │
│    session= / lookup= keywords · explain() · immutable Config        │
│    typed errors with .code · CLI (artifacts / doctor / inspect)      │
├──────────────────────────────────────────────────────────────────────┤
│ 2. Routing                                                           │
│    probe → immutable plan → one-call native request execution         │
│    RoutingPolicy · immutable fallback chain with reason codes        │
├──────────────────────────────────────────────────────────────────────┤
│ 3. Backends                                                          │
│    native HF reference · corrected Gigatoken full/repair             │
│    Rust CUDA Driver prebuilt host · legacy Python/PyTorch JIT         │
│    added-token frontend · explicit experimental Fastokens adapter    │
├──────────────────────────────────────────────────────────────────────┤
│ 4. Artifacts and certification registry                              │
│    ArtifactSource (hub / local dir / mirror / air-gapped bundle)     │
│    sha256 verification · semantic fingerprint · registry + digest    │
├──────────────────────────────────────────────────────────────────────┤
│ 5. Session store                                                     │
│    store-core (no unsafe, no bindings) · store-sqlite · native face  │
│    record format v1 · block hash chain · verification on hit         │
└──────────────────────────────────────────────────────────────────────┘
```

Layers are one-directional: the API layer does not know which backend will run,
backends do not read configuration, and the store does not know how the IDs it
holds were produced. Everything that crosses a layer boundary is either an
immutable value (`Config`, `RoutePlan`, `SessionUpdate`) or a typed error.

### 1.1 Public API

- 0.x exports three constructors: `load(family)`, `from_pretrained(repo_id)`,
  and the `Tokenizer(family, config=...)` class. `encode` returns an immutable
  `Encoding` (carrying `.ids`); `encode_batch` returns `list[Encoding]`. The
  ragged batch output (`output="ragged"`) is the 1.0 target shape recorded in
  `docs/contracts/api.md`; the Rust API already returns ragged batches.
- Named session state is reached two ways: the `session=` keyword on
  `encode`, and the `session()` context manager of `api.md`, which ships
  in 0.2.4. A correct append may
  rewrite the tail of previously returned IDs, so an update describes
  `replace_from`, `replacement_ids` and `all_ids`. Writes take an
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

See section 2 — the three stages are the substance of this layer. For the
Python facade, artifact acquisition and construction-time probing live in its
Python construction layer, and the public plan and diagnostic records remain
Python value objects; the Rust `toktier` crate carries a parallel native lane
whose `ArtifactManager`, `Runtime`, and `RoutePlan` own acquisition,
verification, probing, and routing without Python. Once constructed, the
immutable fallback chain, thresholds, certified engine handles, added-token
gate, content index, and store identity are projected into
`toktier-routing-core`. Each eligible public encode, batch, named-session, or
content-lookup request then crosses PyO3 once, releases the GIL, and completes
routing, execution, fallback accounting, and persistence in Rust. A possible
added-token hit is handled by the native HF reference engine, so the exact
frontend does not require a Python callback.

### 1.3 Backends

- The **reference backend** loads the same tokenizer artifact with Hugging
  Face's `tokenizers` Rust crate and defines correctness. It is always available
  and never disabled by policy; both the Python compatibility adapter and the
  one-call runtime share that implementation.
- The **fast CPU backend** compiles the corrected, data-version-pinned
  Gigatoken core directly into `toktier._native`. The planner verifies its
  integrated delivery, domain-separated source identity, Cargo release flags,
  exact rustc, patch/configuration, oracle, and tokenizer artifact before
  admitting it. Full encode and bounded append repair execute directly under
  the native router. Its single core is verified at runtime construction so
  first-append latency is bounded; payload-sized batch workers remain lazy. A
  failed premise returns the request to the native HF path with a
  machine-readable reason.
- The **GPU backend** implements pre-tokenization and BPE encoding as CUDA
  kernels, with batched, fused and CUDA-Graph forms. The prebuilt path is hosted
  by Rust through the CUDA Driver API, including explicit fatbin-image
  selection, memory, streams, document offsets, and ordered per-row fallback.
  It does not require Python, PyTorch, or a callback during dispatch. The local
  JIT delivery retains its Python/PyTorch host and therefore uses the
  compatibility executor; each delivery has its own registry binding. A stored
  accelerated seed closure-verifies its ID row against the shared verified HF
  vocabulary and adopts shared IDs plus sparse span checkpoints; token spans
  are rebuilt per window on demand, while the materialized compatibility path
  still reconstructs full spans. Neither form constructs the CPU engine merely
  to recover offsets.
- The **added-token frontend** decides whether a document contains added or
  special tokens, using byte-level lookup tables keyed by the content hash of
  the added-token table. Documents that do so are routed to the reference path.
- **Fastokens** is a separately installed full-session adapter. It can be
  requested only with `EXPERIMENTAL` policy and is never an automatic
  fallback. It resolves the installed engine by its import package and reports
  `engine_assurance`: a guarded exact-ID guarantee applies only when the bytes
  are the pinned build this project publishes (`toktier-fastokens`) and the
  shipped registry lists them; every other build reports `false`.

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
  uncertified. Package metadata pins the released oracle exactly
  (`tokenizers==0.22.2`, with the Python loader at `transformers==4.57.6`);
  the registry additionally records the certified version set, and the runtime
  keeps the honest-labeling rule above for environments whose installed
  reference nevertheless differs.

### 1.5 Session store

- `store-core` holds the record format and the verification logic; it denies
  unsafe code and has no binding dependencies, so it can be tested and fuzzed on
  its own. `store-sqlite` owns the database file exclusively from Rust; the
  Python side never opens it directly. Network filesystems are unsupported and
  not recommended, because file locking there is not dependable; the current
  implementation performs no filesystem-type detection, so SQLite locking
  errors surface as-is rather than as a proactive refusal or warning.
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
  BOS/EOS and template effects are never persisted. In 0.x, stored rows are
  not postprocessed on read either: a request with `add_special_tokens=True`
  bypasses the store when `lookup` is omitted and is rejected with an explicit
  `session` or `lookup="auto"`. Holding the core stream keeps read-time
  postprocessing open as a compatible later extension.
- The one-call runtime gives the store direct Rust handles to the native router,
  HF engine, and (only for an admitted binding) corrected-Gigatoken engine. A
  cold seed follows the immutable full-encode route; a continuation uses native
  bounded repair or native HF fallback. No Python callback participates, and
  the native transaction commits only after its invariants pass. The older
  callback adapter remains only as a compatibility fallback for engine shapes
  that cannot enter the native runtime.
- For the certified BPE repair roster, the digest-checked O/S/L/N/M table is
  installed once in the native encoder adapter. Stable-prefix seal decisions
  then execute in Rust with the frozen synchronizing-transition predicate,
  byte-fallback clean-cut rule, added-literal end guard, and a retained repair
  window. No per-seal Python token/span list is materialized.
- The Python facade and Rust SQLite session path create store files with
  restrictive permissions (0700 directories, 0600 files). The Rust path
  tightens only directories and files it creates, leaving pre-existing user
  directories unchanged; its 0700 store directory remains the primary
  protection across SQLite-managed WAL/SHM sidecar lifecycles. Logs do not
  contain source text. Capacity limits and per-session
  deletion are exposed; TTL and a full-store wipe are not part of the current
  0.x public API (deleting the store directory remains the documented reset).

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
  toktier/               Rust-native serving API (artifacts, runtime, sessions)
  toktier-routing-core/  native route/execute/store orchestration
  toktier-gigatoken-core/ corrected Gigatoken full BPE and bounded repair
  toktier-cuda-driver/   prebuilt CUDA Driver API host
  toktier-store-core/    record format, hash chain, verification (no bindings)
  toktier-store-sqlite/  exclusive owner of the database file
  toktier-py/            thin binding facade
tools/                  repository validation and development utilities
tests/                  unit, conformance, packaging, and tooling tests
docs/                   contracts, decisions, support matrix, integration notes
```

The native module is private (`toktier._native`), typed (`_native.pyi`,
`py.typed`), and never imported from user code. The C ABI, where it is exposed,
uses the `toktier_` prefix.
