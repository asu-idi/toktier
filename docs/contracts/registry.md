# Certification registry contract (v1)

Status: frozen for the first public release: the three-identity model,
the status vocabulary, the oracle version policy, the root digest rule,
and the generative discipline. The JSON shapes are normatively defined
by the schemas in `schemas/` (`support_registry.schema.json`,
`evidence_manifest.schema.json`, and `sibling_aliases.schema.json`); this
document explains their meaning.

Framing rule: the registry states exactly what has been certified, no
more. Anything not certified runs as reference and is labeled as such.
We prefer a miss over a wrong result, and an uncertified configuration
is a reference configuration.

## 1. Three identities (frozen model)

Every certification claim attaches to one of three identity kinds:

1. **Exact artifact identity** -- the SHA-256 of the tokenizer artifact
   bytes. The strongest identity: a certification record keyed here was
   produced by judging this exact artifact.
2. **Pipeline capability identity** -- the pipeline fingerprint
   (`fingerprint.md` Section 5): the core pipeline excluding added tokens.
   Certifies that the accelerated engine implements this pipeline class.
3. **Added-frontend capability identity** -- a fingerprint of the
   added-token frontend surface (the added-token table encoding of
   `fingerprint.md` Section 6, hashed with domain tag
   `toktier.added_frontend.v1\0`). Certifies handling of an added-token
   table shape by the added-token frontend (prefilter + literal
   routing).

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
contract.

### 4.1 Verified sibling mappings

`src/toktier/artifacts/tables/sibling_aliases.v1.json` is an admission-side
companion, normatively shaped by `schemas/sibling_aliases.schema.json`. Each
row records a model repository, full audit revision, source file name/length/
sha256, comparison basis, canonical family/anchor sha256, and whether that
anchor is present in the wheel. Its public source projection contains exactly
the 210 rows enumerated in `docs/support-matrix.md`.

The repository and revision only choose bytes to inspect. Runtime admission
requires the sha256 of the bytes actually resolved to match a table row (or an
exact packaged anchor); familiar names with changed bytes do not match. A
canonicalisation or serialisation match selects and executes the recorded
canonical anchor so existing backend certificates and binding checks remain
unchanged. Conflicting digest-to-family mappings, duplicate repositories,
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
- The verifiers need the `jsonschema` package (it is in the `test`
  dependency group, not a runtime dependency of the library); without
  it, `--check` stops with an actionable install message.
- `generated_by` (tool name, tool version, source commit, timestamp) is
  required in both files, so every published table is traceable to the
  code that wrote it.
