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
   The certificate ships under ``readings/`` and is named by path and
   SHA-256.

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
from the other side. On an artifact whose configuration names no loader
class the pinned loader can resolve, the loader face is materialized
file-only, and only when the configuration-side subset is empty are the
file-only face and the loader face provably the same function -- an empty
divergence set needs no certificate. The loading paths verify that premise
before taking the fallback, and refuse it otherwise.

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
