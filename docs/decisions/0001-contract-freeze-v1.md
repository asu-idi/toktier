# Decision log 0001 -- contract freeze v1 (contracts lane)

Date: 2026-08-05. Scope: micro-decisions made while drafting the S1
contract freeze package. Items settled upstream (the adoption baseline
A-1..A-11, the three settled store micro-decisions, the JIT honesty
clause, the wording discipline) are implemented, not re-decided, and are
not listed here. Each entry: decision / rationale / rejected
alternatives.

## D-01 `encode_batch` output modes and ragged element types

- **Decision**: `output="lists"` is the default; `output="ragged"`
  returns `values` as uint32 and `offsets` as **int64**, length
  `n_rows + 1`, `offsets[0] == 0`.
- **Rationale**: lists keep the simple path simple; the ragged shape is
  frozen now so adding the fast path later is non-breaking. int64
  offsets interoperate with the dominant array ecosystems, which use
  signed 64-bit indices.
- **Rejected**: ragged as default (surprising for casual users); uint64
  offsets (poor ecosystem fit); uint32 offsets (caps total batch tokens
  at 4G).

## D-02 Ragged buffers are protocol objects, not a named array type

- **Decision**: `values`/`offsets` expose the Python buffer protocol;
  no third-party array type appears in the contract.
- **Rationale**: zero-copy wrapping by numpy/torch without making any
  of them a dependency of the core contract.
- **Rejected**: returning numpy arrays (adds a hard dependency to the
  API contract); returning bytes (loses element typing).

## D-03 Session read-time surface

- **Decision**: `Session.append()` is the append method (ratifying the
  docs-lane spelling); `SessionUpdate` is immutable; the read-time
  materialization with postprocessor applied is `Session.final_ids()`;
  `session.ids` / `session.text` / `session.revision` are the state
  accessors.
- **Rationale**: `append` matches the contract's append-may-rewrite-tail
  semantics better than `extend` (which suggests pure growth); a
  distinct `final_ids()` keeps the core-stream vs read-time distinction
  visible at the call site.
- **Rejected**: `extend` (earlier draft name); storing post-processed
  ids (contradicts the frozen core-stream rule); a boolean flag on
  `ids` (hides the distinction).

## D-04 Constructor `policy=` keyword and family id spelling

- **Decision**: `Tokenizer(family, config=None, *, policy=None)`;
  `policy` occupies the constructor layer of the precedence chain.
  Canonical family ids are lowercase with underscores (`qwen3_8b`);
  aliases may carry other spellings. Ratifies the docs-lane spellings.
- **Rationale**: the policy is the single most common per-object
  override and deserves a first-class keyword; underscore ids survive
  unquoted in more contexts (identifiers, file names, CLI).
- **Rejected**: policy only via `Config` (verbose for the common case);
  dashed family ids as canonical (kept as aliases instead).

## D-05 `explain()` is a reserved name with an informational shape

- **Decision**: freeze the method name and the presence of plan +
  reason codes; do not freeze the mapping's full key set in v1.
- **Rationale**: diagnostics need room to grow; freezing keys now would
  either straitjacket them or force a v2 quickly.
- **Rejected**: fully frozen diagnostic schema (premature).

## D-06 Backend identifier vocabulary

- **Decision**: backend ids are lowercase strings; v1 assigns `hf`
  (reference), `gpu`, `fast_cpu`, and the explicit experimental session
  adapter `fastokens`; the table is append-only and ids are never renamed or
  reused.
- **Rationale**: short stable ids keep registry entries, plans, and
  reason details consistent; append-only mirrors the other namespaces.
- **Rejected**: dotted taxonomies like `cpu.reference.hf` (structure
  with no consumer yet; can be added as new ids later without breakage).

## D-07 Reason-code namespace structure

- **Decision**: one `R_*` namespace split into plan-time and run-time
  tables; append-only; consumers must tolerate unknown codes.
  `R_INPUT_GUARD_ROUTED` records a per-input premise failure in the guarded
  fast-CPU backend shipped in the first release.
- **Rationale**: the split matches the probe/plan/execute structure and
  keeps `REQUIRE_ACCELERATED` semantics crisp (plan-time only).
  Keeping the guard distinct from engine-load faults makes per-input
  reference routing auditable.
- **Rejected**: separate enums per phase (two namespaces to version);
  omitting the guard code (would force a later addition under time
  pressure).

## D-08 Store telemetry stays out of `R_*`

- **Decision**: store hit/miss/checksum-reject counters are store
  statistics, not routing reason codes; only `R_SESSION_NO_SAFE_CUT`
  (an execution fallback) crosses into `R_*`.
- **Rationale**: store misses are normal operation, not degradations of
  a route; mixing them would blur the "every fallback is counted"
  signal.
- **Rejected**: store events as reason codes (noise, and double
  counting).

## D-09 Error `.code` values are UPPER_SNAKE, plus a `details` mapping

- **Decision**: `.code` is an UPPER_SNAKE string distinct from the
  class name (`ArtifactNotFound` / `ARTIFACT_NOT_FOUND`); every
  exception carries a read-only `details` mapping for machine facts.
- **Rationale**: codes that do not look like Python identifiers travel
  better through logs, JSON, and non-Python consumers; `details` gives
  the machine payload a home so messages stay purely human.
- **Rejected**: class name as code (couples wire format to Python
  naming); error numbers (opaque).

## D-10 Four error codes added beyond the adopted list

- **Decision**: add `CONFIG_INVALID`, `UNSUPPORTED_CONFIG`,
  `STORE_FORMAT_UNSUPPORTED`, `REGISTRY_INVALID`.
- **Rationale**: each is required by another frozen clause -- strict
  config parsing, construction-time rejection of session-incompatible
  modes, the format-vs-corruption distinction, and registry generative
  discipline -- and adding codes later is cheap but adding them now
  keeps first-release consumers off generic exceptions.
- **Rejected**: overloading `StoreCorrupt` for future-format records
  (misdiagnoses healthy data as damage); ValueError for config
  (unstructured).

## D-11 Store read path degrades to miss; only explicit verify raises

- **Decision**: on lookup, any integrity or format failure is a counted
  miss; `STORE_CORRUPT` / `STORE_FORMAT_UNSUPPORTED` raise only from
  explicit verify-style APIs.
- **Rationale**: the red line is "prefer a miss over a wrong result"  --
  a raising read path would make store damage a availability incident
  instead of a cache miss, inverting the design.
- **Rejected**: raising on read-path corruption (turns self-healing
  behavior into an outage).

## D-12 Store header geometry

- **Decision**: 200-byte fixed header; magic `TOKTIERS`; explicit
  endianness marker byte `0x01`; `header_length` in [200, 4096], a
  multiple of 8; ids before text tail in the payload; record size must
  close exactly.
- **Rationale**: a fixed 200-byte prefix decodes with no lookahead; the
  8-multiple keeps the u32 id array aligned; exact size closure kills a
  whole class of trailing-bytes ambiguity.
- **Rejected**: variable-order fields (needless decoder complexity);
  tail before ids (unaligned id reads); unlimited header length
  (allocation hazard from untrusted bytes).

## D-13 Full 32-byte digests in the public format

- **Decision**: block hashes and checksums in format v1 are full
  SHA-256 outputs (32 bytes), unlike the prototype store's
  truncated node keys.
- **Rationale**: the public format is a compatibility surface for
  years; the space cost per record is trivial against text and ids, and
  full digests remove truncation-collision discussion entirely.
- **Rejected**: 20-byte truncated hashes (saves 24 bytes per record,
  buys an avoidable argument).

## D-14 One hash family, three domain tags, payload hashed once

- **Decision**: SHA-256 everywhere; domain tags
  `toktier.store.v1.payload\0` / `.link\0` / `.record\0`; the record
  checksum covers the header (checksum field zeroed) plus the payload
  digest, and the chain link consumes the same payload digest.
- **Rationale**: single primitive, no new dependency; domain separation
  prevents cross-context preimage reuse; the digest indirection means
  one pass over the payload verifies both checksum and chain.
- **Rejected**: BLAKE3 (faster, but a new dependency for a non-hot
  path); checksum over raw payload bytes directly (double traversal);
  CRC32 (detection strength not worth the special case).

## D-15 Integrity claims stated as corruption detection only

- **Decision**: the format documentation states plainly that checksums
  and hash chains detect corruption and do not resist a consistent
  rewrite by an adversary; keyed MACs are out of scope for v1.
- **Rationale**: honest labeling; implying tamper resistance we do not
  provide would be a correctness-adjacent overclaim.
- **Rejected**: silence on the topic (invites wrong assumptions).

## D-16 Witness category: u16, class-level granularity, four values

- **Decision**: `witness_category` is a u16 append-only registry with
  `0x0000` none/full-re-encode, `0x0001` BPE synchronizing transition,
  `0x0002` WordPiece continuation witness, `0x0003` metaspace word
  start. Family-specific predicate profiles are not separate categories
  (they are bound via the fingerprint's pipeline component). Cross
  invariant: category `0x0000` requires `stable_prefix_byte_length == 0`
  and `replace_token_offset == 0`.
- **Rationale**: the categories mirror the certified predicate classes
  of the boundary-repair machinery; class granularity keeps the
  registry stable while profiles evolve; the cross invariant makes
  "no certificate" structurally visible instead of trusted.
- **Rejected**: u8 (no room to grow with reserved semantics); free-form
  string (unbounded, unverifiable); per-profile categories (couples the
  format to family-level churn).

## D-17 Header extension is TLV with an explicit padding type

- **Decision**: optional header fields are `u16 type / u16 length /
  value` TLVs; type `0x0000` is padding; unknown types are skipped;
  anything that must not be skipped ships with a mandatory flag bit.
- **Rationale**: gives future versions a growth path that old readers
  handle correctly by construction, in both directions (skip vs
  reject).
- **Rejected**: implicit zero-padding only (no typed growth path);
  32-bit TLV headers (oversized for a 4096-byte cap).

## D-18 Fingerprint digest and preimage frame

- **Decision**: fingerprint = SHA-256 over a domain-tagged
  (`toktier.fingerprint.v1\0`) sequence of `LE16(field_id) /
  LE32(length) / value` records in strictly ascending field-id order,
  with **every defined field always present**.
- **Rationale**: fixed schema plus explicit ordering removes every
  encoding ambiguity a canonical-form argument would otherwise have to
  carry; version bumps change the domain tag, so cross-version
  collisions are impossible by construction.
- **Rejected**: canonical JSON preimage (pulls full JSON
  canonicalization into the hot correctness path where a byte encoder
  suffices); omitting absent fields (creates ordering/absence
  ambiguity classes).

## D-19 Absent vs default distinguished by a presence byte

- **Decision**: every scalar value carries a leading presence byte;
  absent encodes as the single byte `0x00`, present as `0x01` plus the
  value. A field set to its default and a field left absent therefore
  never collide.
- **Rationale**: the distinction is structural and uniform instead of
  per-field convention, which is where such schemes historically leak.
- **Rejected**: sentinel values (collide with real data); per-field
  documented defaults with omission (exactly the ambiguity the
  requirement exists to prevent).

## D-20 Strings are raw UTF-8, no Unicode normalization

- **Decision**: string values enter the preimage as their exact UTF-8
  bytes.
- **Rationale**: two artifacts that differ on disk must fingerprint
  differently; any normalization step could merge them.
- **Rejected**: NFC-normalizing content (could alias distinct added
  tokens).

## D-21 Pipeline fingerprint via RFC 8785 over four sections

- **Decision**: `pipeline_fingerprint` = SHA-256 with domain tag
  `toktier.pipeline.v1\0` over the RFC 8785 (JCS) canonical form of
  `{decoder, model, normalizer, pre_tokenizer}` taken verbatim from the
  parsed artifact; a missing section encodes as JSON `null` (distinct
  from `{}`); added tokens are excluded (bound separately).
- **Rationale**: the pipeline is inherently JSON-shaped, so a published
  canonicalization standard beats a bespoke one here (unlike D-18 where
  the data is flat); excluding added tokens is what makes pipeline
  capability identity usable for sibling artifacts.
- **Rejected**: hashing the raw artifact bytes minus added tokens
  (fragile to formatting churn); bespoke canonical JSON (reinvention).

## D-22 Added tokens encoded inline, in artifact insertion order

- **Decision**: the full added-token table is encoded inline in the
  preimage (content, id, special, single_word, lstrip, rstrip,
  normalized per element), ordered exactly as declared in the artifact.
- **Rationale**: the table is small, inline encoding keeps the
  fingerprint self-contained, and extraction behavior can depend on
  insertion order, so order is semantic.
- **Rejected**: nested digest of the table (hides structure for no
  gain at this size); sorted order (erases a semantic property).

## D-23 Oracle bound by semantic id, conservatively assigned

- **Decision**: the fingerprint binds a registry-assigned oracle
  **semantic id** (behavior equivalence class), not the raw package
  version; the initial mapping is one exact version per semantic id,
  and widening a class requires certification evidence.
- **Rationale**: certified behavior-preserving oracle upgrades keep
  stored sessions valid; uncertified upgrades miss. Both failure
  directions are the safe ones.
- **Rejected**: raw version string in the fingerprint (every upgrade
  invalidates all stores, even judged-equivalent ones); no oracle
  binding (wrong-key hits across behavior changes).

## D-24 Fingerprint does not bind family name or backend identity

- **Decision**: no family/alias field and no backend/engine field in
  the preimage. The uncertified-stream gap is closed structurally:
  sessions under `EXPERIMENTAL` policy never write to persistent
  stores.
- **Rationale**: family names are aliases (renames must not invalidate
  stores); certified backends are id-equal by definition, so backend
  identity adds nothing -- while the no-persist rule guarantees every
  stored stream is a certified stream.
- **Rejected**: binding backend id (spurious misses between certified
  backends); allowing experimental streams into stores with a marker
  field (a marker that must never be trusted is better made
  unrepresentable).

## D-25 Registry composition eligibility is default-closed

- **Decision**: pipeline capability + added-frontend capability matches
  grant accelerated eligibility only through an explicit,
  evidence-backed composition entry; absent that entry, capability
  matches alone grant nothing.
- **Rationale**: sibling-artifact coverage is real and wanted, but it
  must be an asserted, evidenced claim -- not an inference the router
  makes on its own.
- **Rejected**: automatic composition on double capability match
  (routes inputs no judgment covered); exact-only eligibility (throws
  away the sibling coverage the three-identity model exists for).

## D-26 Class-table digest joins the kernel certificate binding set

- **Decision**: generated lookup tables (character-class tables derived
  from a specific oracle and Unicode version) are first-class artifacts
  with sha256; `class_table_digest` is required in `certified_source`
  entries and verified at load; a mismatch closes the accelerated path.
  The GPU extra must declare its oracle-package dependency honestly
  until tables ship pre-generated and digest-pinned.
- **Rationale**: a porting review found that the tables are probed from the
  oracle wheel at runtime today, so oracle drift could silently change
  kernel split behavior under a green certificate unless the tables
  themselves are bound.
- **Rejected**: leaving tables outside the certificate (silent drift
  channel); pinning the oracle package version in metadata instead
  (the adoption baseline explicitly avoids resolver pins -- binding is
  by digest, not by resolver).

## D-27 Single loader, single flag set per process

- **Decision**: a `certified_source` certificate covers one kernel
  build configuration per process; multiple loaded builds with
  differing flags are a certificate-invalidation condition (accelerated
  path closes, reference runs).
- **Rationale**: a porting review found multiple JIT call sites with
  drifting flags in the prototype; the certificate binds
  flags, so a process mixing flag sets is outside the certified
  premises by definition.
- **Rejected**: per-call-site certification (multiplies the evidence
  surface for no user value).

## D-28 Registry is the single source of routing data

- **Decision**: runtime code must not carry a second copy of any
  mapping the registry expresses; derived lookups are generated from
  the registry.
- **Rationale**: a porting review found that two hand-maintained copies of
  the family-to-kernel mapping drifted historically; a drifted copy
  routes inputs the certificate never covered.
- **Rejected**: convenience constants in code with a consistency test
  (the test is the tell -- generate instead).

## D-29 Artifact manifests pin per-file sha256

- **Decision**: the contract states manifests carry per-file sha256 and
  verification is per file; a repository-revision pin alone does not
  satisfy the contract.
- **Rationale**: a porting review found that the prototype manifest froze the
  repo revision only; hub-side in-place edits are an observed
  phenomenon, so content hashes are the hard gate. The contract is
  written to the target state; backfilling digests is release work.
- **Rejected**: revision-only pinning (observed to be insufficient).

## D-30 No correctness-affecting switch in env or config-file form

- **Decision**: generalized the "deliberate absences" clause -- any
  switch that can change output correctness must not exist as an
  environment variable or config-file key; the only path to uncertified
  output is the explicit `policy=RoutingPolicy.EXPERIMENTAL`
  construction parameter.
- **Rationale**: a porting review found a non-exact-output env switch
  in the prototype; ambient switches are invisible at the
  call site and outlive their intent. Constructor parameters are
  visible, greppable, and diagnosable.
- **Rejected**: carrying such switches over with warnings (ambient
  correctness hazards do not belong in a released surface).

## D-31 Registry/evidence schema ids are URNs; root digest by removal

- **Decision**: JSON Schema `$id`s are URNs
  (`urn:toktier:schema:support-registry:1`), not URLs;
  `root_digest = sha256 over RFC 8785 canonical form with the
  root_digest member removed` (not blanked), with domain tags.
- **Rationale**: URNs avoid committing to a hosting domain before one
  exists; removal (vs blanking) makes the verifier's transform
  unambiguous.
- **Rejected**: https `$id`s on a placeholder domain (dead links in a
  frozen contract); digest over the file bytes (breaks under
  reformatting by the generator).

## D-32 Config file: TOML, under the resolved home, fail-closed

- **Decision**: `config.toml` under the resolved home; unknown keys
  raise; `home` itself cannot be set from the file; early releases may
  ship without runtime file support since the precedence slot and
  format are frozen now.
- **Rationale**: TOML has stdlib parsing (3.11+) and is the ecosystem
  default; fail-closed turns typos into errors instead of silently
  ignored intent; `home` from the file would be circular (the file's
  location depends on it).
- **Rejected**: YAML/JSON/INI (dependency or expressiveness problems);
  ignoring unknown keys (silent typo swallowing).

## D-33 Strict boolean parsing for environment values

- **Decision**: `1/true/yes/on` and `0/false/no/off`
  (case-insensitive); anything else raises `CONFIG_INVALID`.
- **Rationale**: `TOKTIER_OFFLINE=Talse` should be an error, not a
  guess, on a correctness-first surface.
- **Rejected**: truthiness of non-empty strings (makes `"0"` truthy in
  some conventions and is a classic footgun).

## D-34 A minimal `src/toktier/__init__.py` accompanies the stubs

- **Decision**: ship a placeholder `__init__.py` (re-exports plus
  `API_VERSION`) so the three contract stubs are importable as a
  package; the core-package lane owns its later extension.
- **Rationale**: the deliverable requires importable stubs; relative
  imports between `config.py` and `policy.py` need a package context.
- **Rejected**: absolute-import loose modules (breaks the eventual
  package layout); leaving import wiring to the core lane (stubs would
  not be verifiable now).
