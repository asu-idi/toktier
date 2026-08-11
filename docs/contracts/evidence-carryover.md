# Evidence carry-over contract (v1)

Status: explicit exception channel. Full recertification remains the default
whenever a certification source identity changes. Evidence is carried only
when one of the two mechanisms below is machine-applicable and its complete
witness is retained.

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

## 3. Add-only record

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

## 4. Chain rules and release gate

`python tools/verify_carryover.py --check` validates the schema, witness
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
