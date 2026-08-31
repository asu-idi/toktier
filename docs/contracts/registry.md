# Certification registry contract (v1)

Status: frozen for the first public release: the three-identity model,
the status vocabulary, the oracle version policy, the root digest rule,
and the generative discipline. The JSON shapes are normatively defined
by the schemas in `schemas/` (`support_registry.schema.json`,
`evidence_manifest.schema.json`, `evidence_carryover.schema.json`, and
`sibling_aliases.schema.json`); this document explains their meaning.

Framing rule: the registry states exactly what has been certified, no
more. Anything not certified runs as reference and is labeled as such.
We prefer a miss over a wrong result, and an uncertified configuration
is a reference configuration.

## 1. Three identities (frozen model)

Every certification claim attaches to one of three identity kinds:

1. **Exact artifact identity** -- the SHA-256 of the tokenizer artifact
   bytes. The strongest identity: a certification record keyed here was
   produced by judging this exact artifact. Since 0.2.8 the certified
   *subject* extends past those bytes when the loader face does: a record
   whose ``tokenizer_config.json`` declares added tokens beyond the
   artifact file carries a ``config_added_tokens`` claim (the canonical
   digest and count of that subset), the artifact key and naming staying
   exactly as before. The loading paths recompute the subset from the
   files they are about to execute and fail closed on a mismatch
   (``ARTIFACT_HASH_MISMATCH`` with reason
   ``config_added_tokens_mismatch``); a record without the claim asserts
   the subset is empty. Readings taken before the 0.2.8 oracle
   redefinition remain valid under it through the corpus-equivalence
   carry-over annotation (``carryover``,
   ``docs/contracts/evidence-carryover.md`` Section 3).
2. **Pipeline capability identity** -- the pipeline fingerprint
   (`fingerprint.md` Section 5): the core pipeline excluding added tokens.
   Certifies that the accelerated engine implements this pipeline class.
   Since 0.2.9 the fingerprint is computed on the **loader-face document**
   -- the tokenizer JSON the pinned loader serializes after materializing
   the verified artifact directory (`facade.md` Section 5) -- rather than
   on the artifact file, so the capability id names the same subject the
   certification readings are taken against. The preimage encoding and the
   domain tag are unchanged; only the document they read moved.
3. **Added-frontend capability identity** -- a fingerprint of the
   added-token frontend surface (the added-token table encoding of
   `fingerprint.md` Section 6, hashed with domain tag
   `toktier.added_frontend.v1\0`). Certifies handling of an added-token
   table shape by the added-token frontend (prefilter + literal
   routing). Since 0.2.9 the table is read from the loader-face document,
   so an added token the configuration sidecar declares beyond the
   artifact file is part of the fingerprinted surface: the capability id
   and the certification subject describe the same added-token
   vocabulary. For an artifact whose loader face is the file-only
   construction -- a directory carrying no loader configuration file is
   materialized that way directly, never through a tokenizer class
   inferred from the directory path, and a configuration whose named
   class cannot be constructed degrades to the same face when its
   configuration-side subset is empty -- the same rule reads that
   degenerate face; there is no special case, and the face does not
   depend on where the verified bytes sit. The encoding and the domain
   tag are unchanged.

Both capability fingerprints are generation-time values: maintainer
tooling computes them under the locked loader when the registry is
written, and refuses to write when the installed loader pair is not the
locked one. The runtime consumes the recorded ids at the sha level only;
no runtime path, Python or Rust, recomputes a capability fingerprint.

### 1.1 Routing eligibility (frozen rule)

Under `CERTIFIED` policy, an accelerated backend is eligible for an
artifact when either:

- an **exact artifact record** exists with an eligible backend status
  (Section 3), and its constraints verify at load time; or
- the artifact's pipeline fingerprint matches a certified **pipeline
  capability** record, its added-token table matches a certified
  **added-frontend capability** record, **and** the registry contains an
  explicit composition entry allowing that pair. Compositions are
  evidence-backed and default-closed: absent an explicit composition
  entry, capability matches alone do not grant eligibility.

`EXPERIMENTAL` policy may route beyond these rules; its outputs are not
covered by certification claims (see `routing.md`).

For the Rust API runtime, this section is necessary but not sufficient:
certified acceleration additionally requires the executing build to match an
eligible entry in the registry's `runtime_builds` block (exact facade source
identity, features/profile, exact rustc, and the fast-CPU/native-host
digests). An unregistered build falls back to HF under `CERTIFIED`, and an
explicit CUDA request reports `UNCERTIFIED_RUNTIME` (see `docs/rust-api.md`).

### 1.2 The dependency graph the runtime build compiles (added in 0.2.6)

Added in 0.2.4 and narrowed in 0.2.6; the frozen rule above is unchanged.
A Rust API build is additionally required to compile the **certified
core** of the judged dependency closure: this project's own crates, the
packages they call directly from encode-path sources, and the
text-semantics libraries beneath them. Each package of the shipped
judged-closure record carries the tier and the rule that placed it
there. Packages outside the core are compared and reported, and do not
decide eligibility. A package of the core whose behaviour is defined by
an evolving external standard is compared by the version of the tables
it carries rather than by its package version; where that version cannot
be read, the package version is compared exactly.

### 1.3 One Unicode version across the compared engines (added in 0.2.6)

Every certification reading compares an accelerated engine's ids with
the ids the frozen oracle produces. Both sides cut text on Unicode
character classes, and that comparison only means what it says while
both sides answer alike about every code point. Accordingly: the Unicode
property data an accelerated engine reads must agree, over every scalar
value and every class the shipped pre-tokenizer patterns name, with the
classes the oracle's own regex engine answers. This is a stated premise
of every reading rather than a property of any one of them, and a
release that moves either side moves both and re-takes the readings.

## 2. Oracle version policy (frozen)

- Each record names the oracle package, the exact package versions
  certification was judged against, and the semantic id assigned to
  that behavior class (`fingerprint.md` Section 7).
- Runtime rule:
  - installed oracle version inside the record's certified set ->
    accelerated paths may open (subject to Section 1.1 and Section 3);
  - installed oracle version outside the certified set -> accelerated
    paths stay closed, the installed reference oracle still runs, and
    the state is reported as reference-only (`R_ORACLE_MISMATCH`);
  - artifact not in the registry at all -> reference, reported
    uncertified (`R_UNCERTIFIED_ARTIFACT`).
- Package metadata pins the released oracle exactly (`tokenizers==0.22.2`;
  the Python package also pins its loader, `transformers==4.57.6`). The
  registry still expresses the certified set independently, and the runtime
  enforces honest labeling for environments whose installed oracle
  nevertheless differs from the pin.
- Explicit-engine rule (same policy, below routing): the explicit GPU
  engine (`toktier.engine.gpu`) sits below the routing layer and
  constructs and runs regardless of the installed oracle version --
  it is an explicit entry point, and blocking it would be a form of
  install blocking. The honest-labeling half is not optional there
  either: the engine's binding set and its `explain()` report must
  record the installed oracle package and version together with the
  certified set of the records covering its artifacts, and when the
  installed version falls outside that set they must carry
  `uncertified_oracle: true` -- the certificate does not attach to
  that process, and per-family verdicts state
  `oracle_outside_certified_set` as the reason. A binding set that
  omitted the installed oracle could present a judged kernel identity
  for a process whose reference behavior the judgment never covered.

## 3. Backend status vocabulary (frozen)

Each record carries per-backend entries with one of:

| Status | Meaning |
|---|---|
| `certified` | The judged binary itself is bound: the record carries a binary digest, and the loader must match it before the backend opens under `CERTIFIED`. In this release, the prebuilt GPU delivery also binds the source/build identity of the Rust host paired with that binary; either mismatch closes the route. |
| `certified_source` | The judged implementation is bound by source identity plus its reproducibility inputs instead of by one platform-specific output binary. Every backend must match its schema-defined source digest, build flags, and toolchain exactly. GPU JIT additionally binds the generated class table, selected NVCC, torch runtime CUDA, exact PyTorch version, and judged device architecture. The integrated CPU engine binds its Rust/Python source set, Cargo release profile, exact rustc, repair configuration, patch, oracle, and artifact. This status is eligible under `CERTIFIED` only when every applicable axis verifies and is reported distinctly from `certified` everywhere. |
| `experimental` | Present but not certified; reachable only under `EXPERIMENTAL` policy. |
| `unsupported` | Known not to work; never planned. |

The reference backend needs no status: it is always available and is
the definition of correct output for certified configurations.

CPU engines additionally bind `engine`, exact `engine_version`,
`engine_delivery`, `engine_module`, `source_digest`, `build_flags`, exact
`toolchain`, `patch_sha256`, `config_id`, and `config_digest`. The corrected
Gigatoken implementation in this release is compiled into the single private
`toktier._native` extension (`engine_delivery=integrated`); no separately
installed Gigatoken distribution or second native extension participates. Its
build script embeds the source identity, Cargo release profile, and rustc
identity in the extension, and the planner compares those facts without
executing an untrusted sidecar. Any unavailable or mismatched axis closes the
route with `R_ENGINE_BINDING_MISMATCH`. Historical standalone-binary digests
remain provenance for the earlier differential campaign, not authority for the
engine that executes in this release.

For a prebuilt GPU delivery, `binary_digest` binds the fatbin and
`architecture_digests` bind every embedded image. The same delivery row also
binds `host_source_digest`, `host_build_flags`, and exact `host_toolchain` for
the Rust request host that selects the image, launches it, splits results, owns
store/routing state, and performs reference fallback. The extension embeds
those host facts at build time; a stable fatbin therefore cannot certify a
drifted host. Hardware parity readings must carry the same host identity before
the generator may mark the delivery `certified`.

Device-architecture rule (frozen): kernel records list the exact GPU
architectures judged. Loading selects the image or build for a listed
architecture explicitly; an unlisted architecture is not eligible under
`CERTIFIED` (`R_SM_UNCERTIFIED`) regardless of what a build system could
produce for it.

### 3.0 The `SUPPORTED` policy and the assurance vocabulary (added in 0.2.6)

The status vocabulary above is unchanged: it says what the registry
records about a backend, and every sentence of it still holds. What 0.2.6
adds is a second word for the running configuration, because "not
certified" was answering two different questions at once.

A campaign judges the devices and compilers it had. A device it never
ran on is not a device it found wrong, and until 0.2.6 the two were
refused with one word. From this release they are told apart:

| Assurance | What is true of the running route | Admitted under |
|---|---|---|
| `certified` | The configuration appears in the registry, its status is `certified` or `certified_source`, and every constraint that row binds verifies here. | every policy except `REFERENCE` |
| `supported_untested` | Every constraint the registry binds verifies, and the device architecture or the compiler toolchain is one no campaign judged. | `SUPPORTED` (the default) |
| `locally_verified` | The same, and a local check on this machine has compared the route with the reference engine and they agreed. | `SUPPORTED` |
| refused | Something bound did not verify: a digest, build flags, an engine binding, the certified core of the compiled closure, or an integrity condition such as a world-writable compiler component. | an explicit opt-in only |

The device-architecture rule of Section 3 keeps its wording exactly: an
unlisted architecture is not eligible **under `CERTIFIED`**, and that is
still true. `SUPPORTED` is a different policy, and it admits the route
and labels it rather than calling it certified. The `R_SM_UNCERTIFIED`
reason code is unchanged and is still the reason `CERTIFIED` refuses.

Two things this vocabulary does not do. It does not add a registry
status: nothing is written into a record, and the generator is
untouched. And `locally_verified` is not a certificate: it is a record
of a measurement somebody took on one machine, filed under the device,
delivery, image digest, compiler triple, driver, source identities and
family artifact it was taken under, and it stops applying as soon as any
of them moves. A local check that disagreed leaves the route labelled
exactly as it would have been had nobody run one.

Driver and CUDA runtime versions are environment facts rather than
certificate premises, and are reported as such on both faces. Where a
delivery row binds `driver_min`, that floor is a precondition for the
image loading at all and is still checked under every policy.

### 3.1 Generated lookup tables are bound artifacts (frozen)

Kernel behavior depends on generated lookup tables (character-class
tables derived from a specific oracle version and Unicode version).
Contract:

- These tables are **first-class artifacts**: shipped or materialized
  with a sha256 each, and their digest (`class_table_digest`) is part
  of the kernel certificate's binding set. A table that does not match
  the bound digest closes the accelerated path
  (`R_KERNEL_DIGEST_MISMATCH`), exactly as a kernel source mismatch
  would. Without this binding, an oracle version drift could silently
  change kernel split behavior while the certificate stays green -- the
  binding makes that impossible.
- Dependency honesty: if a release generates tables at install/load
  time by probing the oracle package, the GPU extra must declare that
  dependency explicitly; a claim of independence from the oracle
  package is only available once tables ship pre-generated and
  digest-pinned.

### 3.2 Single loader, single flag set (frozen)

A `certified_source` certificate covers exactly one build configuration per
process: one implementation identity, one toolchain, and one bound flag set.
If more than one build with differing facts is loaded in the same process, the
certificate's premises no longer hold and the status degrades to uncertified
(the accelerated path closes and reference runs). Multiple loaders or
divergent flag sets are a certificate-invalidation condition, not a tolerated
variation. For GPU JIT, the cache identity includes the resolved compiler path,
parsed release/build, torch runtime CUDA, and PyTorch distribution version, so
a local product cannot cross an unverified compiler boundary through cache
reuse. The integrated CPU extension exposes one immutable embedded fact set.

### 3.2 Engine distributions (added in 0.2.7)

The top-level `engine_distributions` node records an engine distribution
this project publishes for an explicitly admitted adapter; in 0.2.7 that is
`engine_distributions.fastokens`, the pinned `toktier-fastokens` build. It
is not a backend status and the vocabulary above does not apply to it: it
grants no route and moves nothing out of `experimental`. What it records is
which published bytes the adapter's evidence describes -- the published
wheels by file name, sha256 and engine digest, the sdist, the upstream
revision and patch series, the oracle and the families the evidence covers,
the Unicode guard set with its digest, and the gate readings with their
evidence id -- so that the adapter can recognise the bytes it is about to
run and report `engine_assurance` from the same digest-verified document the
planner reads. The node is generated from `tools/fastokens_binding.json` by
`tools/update_fastokens_registry.py`, checked by `generate_registry.py
--check` against the shipped readings, the evidence manifest, the packaging
kit and the adapter source, and refused by `--release-check` while any wheel
digest is a placeholder. An adapter that finds no node, no guard or no
matching wheel reports the weaker state; the node never widens anything by
its absence.

### 3.3 Single source of truth for routing data (frozen)

The registry is the **only** data source for family-to-kernel and
capability mappings. Runtime code must not carry a second copy of any
mapping the registry expresses (no parallel constant tables); derived
lookup structures must be generated from the registry at build or load
time. Two independently maintained mappings drift, and a drifted copy
routes inputs the certificate never covered.

## 4. Record contents (summary; schema is normative)

An artifact record carries: identity (artifact SHA-256, family name and
aliases), pipeline id and added-frontend id, oracle block (package,
certified versions, semantic id), certification suite version, evidence
id (pointing into an evidence manifest), readings (documents judged,
bytes judged, mismatch count), and per-backend entries (status plus the
digests/flags/toolchain/devices or CPU-engine binding the status requires).
Alongside the artifact records, the registry carries the top-level
`runtime_builds` block of Section 1.1: the evidence-bound native serving
builds allowed to admit accelerated routes for the Rust API.

Mismatch counts are recorded as read, whatever they are; the registry
never rounds a nonzero to zero.

Artifact manifests (the fetch-side companion of the registry) pin
**per-file sha256** for every file of an artifact, not merely a
repository revision. Verification is per file against these digests;
a revision pin without content digests is not sufficient under this
contract. Since 0.2.9 the pinned file set is the **loader-face input
closure**: every file the pinned loader reads when materializing the
certified loader face from the frozen artifact snapshot (the
configuration files alongside `tokenizer.json`), measured and proven
at generation time -- so a cache built from the manifest rebuilds the
input group the certification subject was materialized on. A locally
converted family pins exactly its conversion output, whose file-only
face is proven byte-identical to the frozen snapshot's face.

### 4.1 Verified sibling mappings

`src/toktier/artifacts/tables/sibling_aliases.v1.json` is an admission-side
companion, normatively shaped by `schemas/sibling_aliases.schema.json`. Each
row records a model repository, full audit revision, source file name/length/
sha256, comparison basis, canonical family/anchor sha256, and whether that
anchor is present in the wheel. Its public source projection contains the 214
sibling rows enumerated in `docs/support-matrix.md`, plus one canonical
self-row per family whose upstream source file is not `tokenizer.json` --
215 rows in this release. A self-row names the family's own repository, so
resolving that repository reports itself as the evidence repository rather
than a byte-identical sibling; the generator checks each self-row's
repository and revision against the shipped artifact manifest, and the audit
accounting in `docs/support-matrix.md` counts the 214 siblings only; its
dated audit equation counts the 210 that were in the snapshot, the other four
having been published after it.

Since 0.2.9 a fifth basis, `equivalent_loader_face`, admits a repository
whose audited tokenizer file group materializes, under the pinned loader, a
loader face byte-identical to the canonical family's (`facade.md` Section
5): the two artifacts are then two spellings of one certified object and
hold its capability ids. The audit step behind such a row materializes both
faces and compares the serializations byte for byte before the row is
written.

The repository and revision only choose bytes to inspect. Runtime admission
requires the sha256 of the bytes actually resolved to match a table row (or an
exact packaged anchor); familiar names with changed bytes do not match. A
canonicalisation, serialisation or loader-face match selects and executes the
recorded canonical anchor so existing backend certificates and binding checks
remain unchanged. Conflicting digest-to-family mappings, duplicate repositories,
count drift, malformed identities, or disagreement with the artifact manifest
raise `REGISTRY_INVALID`.

## 5. Evidence manifests (frozen relationship)

- Every `evidence_id` in the registry resolves to an entry described by
  `evidence_manifest.schema.json`: run id, source commit, oracle package
  and version, artifact hashes, corpus identifiers with document and
  byte counts, per-shard digests, mismatch count, environment summary,
  date, and a root digest.
- Evidence manifests ship with the first release so the published
  numbers are structurally traceable; the harness that produced them is
  planned for a later release. The manifest states what was run and what
  was observed; it does not claim more than that.

### 5.1 Explicit evidence carry-over

Full recertification remains the default after a covered source identity
changes. The two narrow exceptions are sentinel artifact equivalence and the
enumerated code-identity-v2 version axis. Their add-only record, applicability
rules, witness requirements, and chain limits are defined by
`evidence_carryover.v1` in `evidence-carryover.md`. A carry-over record points
to the original registry/readings records; it does not rewrite or relabel
those records.

## 6. Root digest (frozen construction)

The support registry, evidence manifests, and sibling mapping carry a
`root_digest` field:

```
root_digest = "sha256:" + hex( SHA-256(
    "toktier.registry.v1\0"          # or "toktier.evidence.v1\0"
                                      # or "toktier.sibling_aliases.v1\0"
    || canonical_json(document with the root_digest member removed)
) )
```

- `canonical_json` is RFC 8785, as in `fingerprint.md` Section 5.
- Verifiers recompute after deleting the `root_digest` member (not
  blanking it).

## 7. Generative discipline (frozen)

- Registry, sibling mappings, and evidence manifests are **generated by tooling only**,
  from judgment outputs; hand edits are prohibited.
- The generator supports `--check`: regenerate in memory, compare
  canonical forms, verify `root_digest`, and validate against the JSON
  Schema; any difference fails.
- CI runs `--check` on every change touching these files; a failing
  check blocks merge. Schema validation failures and root digest
  failures at load time raise `RegistryInvalid` (`REGISTRY_INVALID`).
- `tools/verify_carryover.py --check` validates every add-only carry-over
  record, resolves its original-evidence pointers, derives the chain, and
  enforces the chain and minor-version campaign rules as a release gate.
- The verifiers need the `jsonschema` package (it is in the `test`
  dependency group, not a runtime dependency of the library); without
  it, `--check` stops with an actionable install message.
- `generated_by` (tool name, tool version, source commit, timestamp) is
  required in both files, so every published table is traceable to the
  code that wrote it.
