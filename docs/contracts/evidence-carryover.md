# Evidence carry-over contract (v1)

Status: explicit exception channel. Full recertification remains the default
whenever a certification source identity changes. Evidence is carried only
when one of the mechanisms below is machine-applicable and its complete
witness is retained. Sections 1 and 2 carry evidence across a *source
identity* change; Section 3 carries readings across a change in what the
*judge* means, on the strength of what the judged corpus provably does not
contain.

## 1. Artifact equivalence

Artifact equivalence states that two eligible source trees produced the same
sentinel-form certified payloads under one recorded recipe:

- the complete `toktier/_native.abi3.so` wheel member is byte-equal; and
- the complete `libtoktier.rlib` built for the `toktier` package is byte-equal.

The rlib is compared as one file. No archive member is removed, normalized, or
trusted separately. Both builds use `TOKTIER_IDENTITY_SENTINEL=1`, which pins
the three embedded source digests to the fixed equal-length sentinel. A
sentinel build is evidence only and cannot be released; the release-artifact
verifier rejects the sentinel in either its ASCII-hex or decoded-byte form.

`tools/verify_artifact_equivalence.py OLD_TREE NEW_TREE` applies these
preconditions in order:

1. The diff is confined to covered, non-byte-shipped Rust, manifest, or
   in-tree documentation/NOTICE files.
2. `Cargo.lock` is byte-equal.
3. `rust-toolchain.toml`, `pyproject.toml`, the covered Python facade files,
   and the covered repair tables are byte-equal.

The first miss returns `not_applicable` with a zero exit status. It is not a
tool error; it selects full recertification or, for the enumerated version
axis, the identity-v2 mechanism. Eligible trees are copied sequentially to the
same canonical source path. Each build receives a fresh but identically named
`CARGO_TARGET_DIR`, `--locked`, and one identical `RUSTFLAGS` value containing
the canonical tree remap and one `/cargo` remap per Cargo home root enumerated
from locked Cargo metadata. The witness records the exact argv, ambient and
effective environment, Cargo configuration state, tool versions, toolchain
file digest, and host fingerprint.

The current claim is same-host only. The host fingerprint deliberately omits
the host name, but binds the build environment facts relevant to that
restriction. Cross-host equality of the whole rlib has not been validated and
version 1 records declaring a cross-host scope are refused.

Artifact equivalence does not claim that:

- a source edit is correct in general or is equivalent on an unrecorded tool,
  host, feature set, profile, target, or environment;
- wheel ZIP containers are reproducible (generated packaging metadata is
  outside the compared payload set);
- a protected byte-shipped file, dependency resolution, toolchain input, or
  package-version change is harmless; or
- empirical byte equality is a proof over all possible builds or inputs.

An eligible build whose two payloads differ is `not_equivalent`. Both
`not_equivalent` and `not_applicable` return to the default certification
path.

## 2. Version-normalized code identity

`code_identity_v2` is separate from artifact equivalence. It applies only to
the package-version axis normalized by `tools/compute_identity_v2.py`: the root
workspace package version, the 11 enumerated internal path-dependency
constraints, and the seven corresponding workspace-member version rows in
`Cargo.lock`. The tool prints the complete normalization diff for review.
Any change outside that closed list changes identity v2 and makes this
mechanism unavailable.

Version strings may appear in build facts and package METADATA. They never
select, configure, seed, or otherwise enter tokenization. The machine gate for
this boundary is `tools/scan_version_constants.py`, which scans the covered
Rust and Python code and permits only its enumerated build-fact sites. A v2
match therefore carries code evidence across the package-version axis; it does
not claim byte equality of version-sensitive compiled artifacts.

## 3. Corpus-equivalence carry-over

Corpus-equivalence carry-over states that readings taken under an earlier
judge definition remain valid under the current one, because on the judged
corpus the two definitions are the same function. The 0.2.8 oracle
redefinition is its first application: the reference moved from the artifact
face (the ``tokenizer.json`` document alone) to the loader face (that
document plus the added tokens the configuration sidecar declares), and the
two faces differ only on inputs holding one of a small set of added-token
literals.

Three inputs make one carry-over, and all three must be retained:

1. **The in-archive readings** under the earlier judge definition, exactly
   as recorded. Carry-over restates which judge those readings hold under;
   it never changes a number, and the generation channel refuses a record
   whose expected readings differ from the registry row.
2. **The divergence set**: the complete, machine-derivable set of literals
   on which the two judge definitions can answer differently. For the
   artifact-face to loader-face move this is the configuration-side
   added-token subset (the loader face's added table minus the artifact
   file's); for a cross-artifact application it is the difference of the two
   added-token tables. The generation channel recomputes the set from the
   artifact pair rather than trusting the record.
3. **The absence certificate**: a scan reading stating that every literal of
   the divergence set occurs zero times in the judged corpus, carrying the
   corpus identity, the character total its own scan measured, per-literal
   zero counts, and positive controls that prove the scanner read the text.
   The certificate is recorded under ``readings/`` and is named by path and
   SHA-256. That is where the evidence set keeps it -- in the repository
   and in the source archive published beside the release; like every
   other reading it is not carried inside the wheel or the crate, so a
   reader following the path reads it there.

   **Corpus identity** is the corpus id at its pinned revision, the document
   count, and the number of units the scan covered. The character total is
   recorded beside that identity as a per-scan measurement rather than as
   part of it: two counting implementations reading the same documents can
   report totals that differ slightly, and each states what it counted. So a
   certificate whose character total differs from another reading's over the
   same documents and the same files is a second measurement of one corpus,
   not a claim about a different one. Where such a difference is known, the
   certificate's ``provenance`` says so and names both totals; where the
   document count or the file set differs, the corpora are different and the
   certificate does not apply.

When the three hold, the readings are annotated rather than re-taken: the
registry artifact record carries a ``carryover`` node
(``schemas/support_registry.schema.json``) naming the mechanism
(``corpus_equivalence``), the two subjects, the divergence set, and the
certificate with its digest and totals. The certificate corpus must cover at
least every document of the carried readings -- absence over a superset
implies absence over the readings' corpus -- and the release check re-reads
the shipped certificate byte for byte (``tools/generate_registry.py
--check``).

What this mechanism does not claim: it says nothing about inputs outside the
judged corpus. On an input that does hold a divergence-set literal, the two
judge definitions really do answer differently, and only the current
definition's answer is served; the certified readings simply never met such
an input, which is exactly what the certificate proves.

One degenerate boundary is stated here because it is the same machinery seen
from the other side. When the pinned loader cannot construct the loader
object at all -- the case this exists for is a configuration naming a loader
class it does not know -- the loader face is materialized file-only, and only
when the configuration-side subset is empty are the file-only face and the
loader face provably the same function: an empty divergence set needs no
certificate. That emptiness is the premise the loading paths verify before
taking the fallback, and they refuse it otherwise. They do not classify why
the construction failed, and do not need to: the equivalence follows from the
empty subset alone, whatever the cause.

The same limiting case settles when two artifacts materialize byte-identical
loader faces: the divergence set between the two subjects is empty, so they
hold the same capability ids (since 0.2.9 computed on that shared face,
`registry.md` Section 1) and their readings need no carry-over record at all
-- there is no difference for one to annotate. That is why an artifact-pair
difference needs a carry-over only when it is not empty; the cross-artifact
form below is for that case.

### 3.1 Cross-artifact application

A new artifact may hold the readings of a certified one, its *anchor*, when
the two differ only in added-token rows that the judged corpus never
contains. The registry records the new artifact as its own record. Its
`readings`, `evidence_id`, `suite_version`, `oracle_id` and `pipeline_id`
equal the anchor's, and its `carryover` node names the anchor by
`from_family` and `from_artifact_sha256`. The **divergence set** is the set
of added-token rows, keyed by id, that differ between the two **loader
faces** in content or in any matching flag; the literals of both sides are
members. The certificate's subject is the new artifact: it lists the set as
`{id, side, content}` rows (`side` is `from` for the anchor's row and `to`
for the new artifact's) and records `divergence_set_sha256`, the SHA-256 of
the canonical row form: the rows in ascending `(id, side)` order, each
carrying `id`, `side`, `content` and the five matching-semantics flags of
`fingerprint.md` Section 6 (`special`, `single_word`, `lstrip`, `rstrip`,
`normalized`), serialized with sorted keys, no whitespace and literal non-ASCII, the same
convention as the `config_added_tokens` digest.

The node is valid only when every condition below holds and the generation
channel has recomputed each one from the two artifact directories under the
pinned loader:

- **C1** -- the two loader faces are equal as JSON documents after the
  `added_tokens` member is removed. This covers the `normalizer`,
  `pre_tokenizer`, `model` (vocabulary, merges and every flag), `decoder`
  and `post_processor` sections and the remaining top-level members such as
  `truncation`, `padding` and `version`; it implies equal `pipeline_id`
  values, whose fingerprint hashes four of those sections verbatim.
- **C2** -- outside the divergence set, the two added-token tables are equal
  row for row.
- **C2'** -- no divergence literal is equal to a byte string of the shared
  `model.vocab`. An engine that extends its id-to-bytes vocabulary with the
  added rows would otherwise meet two candidate ids for one string, and the
  choice between them would be a property of that engine.
- **C3** -- every divergence row is `normalized: false`, or the shared
  normalizer is empty. A `normalized: true` row is matched against the
  normalized text; under this condition that text is the raw text, so the
  substring census of C4 is the census that matters.
- **C4** -- the certificate reads ABSENT with closed coverage, every positive
  control nonzero and no unit failing its controls, over a corpus whose id
  and revision are those of the anchor's readings (the evidence manifest of
  the record's `evidence_id` names them), and it covers at least their
  document count. Readings taken on a subset of the documents, or on their
  length-truncated texts, are covered as well: a literal absent from a
  document is absent from every part of it. This is why two character
  totals of one corpus are a measurement difference, not a corpus
  difference (Section 3, corpus identity).
- **C5** -- the anchor record holds no `carryover` node itself; one hop
  only, no chains.
- **C6** -- when the record carries a fast CPU entry, the repair parameters
  it carries (`backends.fast_cpu.repair`: margin, effective window length,
  normalizer presence) equal the anchor's.
- **C7** -- for each accelerated backend entry the record carries, a recorded
  engine-parity reading of the new artifact against the reference returns
  zero mismatches. For the fast CPU entry this is the reading named by the
  node's `engine_parity` member (the engine compared with the reference
  over the new artifact's own face, on a specimen set that includes
  documents holding the divergence literals); for a certified prebuilt GPU
  delivery it is the per-family row the hardware readings already require
  of every certified family. The fast CPU reading binds the engine it was
  taken on by the version-normalized source identity (`source_digest_v2`,
  Section 2) when it records one, and is then held to the entry's
  `source_digest_v2`: a release that moves only the package version leaves
  it in force, and a change to the engine's sources requires it to be
  taken again. A reading that records only the per-build digest is held to
  the entry's `source_digest`, and the check's refusal says to take it again
  recording the normalized identity.

The `config_added_tokens` member, when present, describes the new face as it
does for any record; the cross-artifact check does not compare its `count`
with the divergence set, which belongs to the configuration-side form of
Section 3.

Beside the node, the registry carries one `compositions` row for the new
pair (the anchor's `pipeline_id` with the new artifact's
`added_frontend_id`, under the anchor's `evidence_id`), and the
`engine_parity` member names the C7 reading by path, SHA-256, document count
and mismatch count. `tools/generate_registry.py --check` re-reads the
certificate, the parity reading and the evidence manifest for every such
node; a record and certificate that are wrong *consistently* are outside
what the shipped files can show, and the generation channel, which holds the
artifact pair, is the authority there.

C1-C4 establish that the reference returns the same ids for both artifacts
on every document of the carried readings: under C1-C3 a document holding no
divergence literal gives both added-token matchers the same matches, and the
remaining text passes through equal sections, so readings taken on the
anchor face over such documents are readings of the new face. For each
accelerated backend entry the record carries, the same conclusion rests on
the backend consuming the added-token table only through matching and id
lookup; the registry cannot state that property, and the recorded parity
reading of C7 is the check that it holds for this artifact. The claim covers
encoding only. Decoding is outside it, because special flags differ.

## 4. Add-only record

This section and Section 5 govern the source-identity mechanisms of
Sections 1 and 2. A corpus-equivalence carry-over (Section 3) is recorded
in the registry document itself, beside the readings it annotates, and is
verified by the registry checks named there.

Records live under `evidence/carryover/vMAJOR.MINOR/` and conform to
`schemas/evidence_carryover.schema.json`. The minor-version directory is part
of the chain scope; the JSON shape remains `evidence_carryover.v1`:

```json
{
  "record": "evidence_carryover.v1",
  "mechanism": "artifact_equivalence",
  "from_source_identity": {
    "fast_cpu": "64 lowercase hex characters",
    "native_host": "64 lowercase hex characters",
    "rust_api": "64 lowercase hex characters"
  },
  "to_source_identity": {
    "fast_cpu": "64 lowercase hex characters",
    "native_host": "64 lowercase hex characters",
    "rust_api": "64 lowercase hex characters"
  },
  "witness": {
    "sentinel_artifacts": [],
    "recipe": {},
    "applicability": {},
    "code_identity_v2": {}
  },
  "carried_evidence": [
    "readings/original_campaign.json#/source-bound-result"
  ]
}
```

The mechanism determines the required witness members. Artifact equivalence
requires equal hashes and byte sizes for both sentinel payloads plus the
complete replay recipe and successful applicability facts. Code identity v2
requires the equal v2 value, normalization diff, and exact computation command
line. `code_identity_v2` is absent from an artifact-equivalence record and is
required for a v2 record.

`carried_evidence` contains JSON pointers to the untouched registry/readings
records from the real campaign. It never points to another carry-over record.
The carried records retain their original observations and wording; the new
record supplies only the explicit identity bridge.

Records are add-only after review. A correction is a new record that makes the
superseding relationship explicit; an accepted record is not silently edited
or deleted.

## 5. Chain rules and release gate

`python3 tools/verify_carryover.py --check` validates the schema, witness
consistency, replay fields, evidence pointers, and the graph formed by the
from/to identity sets. A pointer whose selected original record binds all
three `from_source_identity` values marks a real campaign and resets the
counter. Otherwise the checker follows the incoming carry-over edge.

No chain may contain more than three consecutive carry-overs. Every minor
version represented by a carry-over record must contain at least one real
campaign anchor; a chain cannot cross into a new minor and substitute an older
campaign for that requirement. The verifier derives these facts from the
records and their pointers, not from filenames ordered by a maintainer. An
empty record set is valid and means that the default full-recertification path
has not used an exception record.
