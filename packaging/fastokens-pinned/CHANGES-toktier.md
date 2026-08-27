# Changes carried by this pinned fastokens build

This distribution is upstream fastokens `v0.3.1`
(commit `fe854299553524f2156a22036a2cb4d1f2ef4d97`) plus the patch series in
`PATCHES/`. Patches 0001 to 0005 change code; applying them in order gives a
source tree with git tree hash `aa1924284ec4abaedcc8ed5823ee17e7959c55c5`.
Patch 0006 changes no code: it adds one comment line at the top of each of
the seven modified source files, as Apache License 2.0 section 4(b) asks, and
takes the tree to `aaa5fb94ea62b9379d03074640e267c8d837d649`.

The import package name is unchanged (`import fastokens`), so this build is a
drop-in for code written against upstream. Only the distribution metadata
differs, which means **only one of the two distributions should be installed
at a time**; see `README-dist.md`.

## Why the patches exist

Each code patch closes a case where the engine returned a token-id list that
differs from the reference tokenizer (`tokenizers`) for the same input,
without raising an error, or refused an input the reference accepts. These
were found by a differential campaign that compared engine output against
the reference token id by token id.

| Patch | Internal id | What it changes | How the boundary was established |
|---|---|---|---|
| 0001 | F040 | `Bpe` resolves out-of-vocabulary characters instead of failing the whole encode | the reference resolves them; the patch follows the same resolution |
| 0002 | F041 | Unicode classes in pre-tokenization patterns are compiled so the PCRE2 path denotes the same sets as the reference engine | full code-point sweep (0..0x10FFFF) against the reference: difference 0 |
| 0003 | F042 | chunk boundaries are placed where no pre-token of the family's pattern can cross them | every alternative of the 15 shipped patterns that can run past a newline was enumerated and argued individually |
| 0004 | F045 | the fused scanner accepts U+017F as `s` when matching contractions | exhaustive: iterating all scalar values, U+017F is the only code point whose simple case folding lands on the contraction letters `{s,t,r,e,v,m,l,d}` |
| 0005 | F046 | after the per-chunk match lists are merged, the tail past the last accepted match is rescanned | structural: after the last accepted match the merge logic is the serial scan, so the tail cannot differ from serial |
| 0006 | notices | one comment line at the top of each modified file, naming NOTICE and this file | no code change; the seven files are the complete set the series touches |

Each code patch carries its own regression test in the upstream test layout.
Every one of them aligns a mechanism with the pinned reference implementation
for an input class where an id divergence was observed at scale; reports
about those divergences have been submitted to the upstream project.

## Readings taken on the published wheel

The readings below were taken on one specific wheel, not on a source tree:

| Item | Value |
|---|---|
| wheel | `toktier_fastokens-0.3.1.1-cp39-abi3-manylinux_2_28_x86_64.whl` |
| wheel sha256 | `b99f2765fa1b900afe181844a85ed8eb784ba87972ac92e22cc924322d9c5468` |
| engine digest (the `fastokens/` payload, as toktier computes it) | `0bcf3ada9268e5aef1c9da515555f5e2ea6fc8d7a8accfbc444789853edfdfec` |
| source tree | `aa1924284ec4abaedcc8ed5823ee17e7959c55c5` (patches 0001 to 0005) |
| reference | `tokenizers==0.22.2`, per-token-id equality |

- Differential run: 998,857,881 documents per family across 15 tokenizer
  families, 14,982,868,215 document x family comparisons, eight visible CPUs
  per process. Engine errors: 0. Under a Unicode guard that routes a document
  containing any of a listed set of combining marks to the reference (505
  documents per family in this run), mismatches: 0. Without the guard the
  count is 28, all attributable to the reference's own Unicode data vintage
  rather than to the engine.
- Stateful replay (four arms, 0 drift events), topology robustness (150
  cells over 1/2/all visible CPUs and two throttling mechanisms, 0
  exceptions, 0 id mismatches against the full-core run) and a splice/edit
  battery (157,872 splices and 23,400 edits over 13 families, 0 failing) all
  came back clean on this wheel.
- The seam-related results carry a visible-CPU-count qualifier: the large run
  ran with 8 visible CPUs, and cross-topology evidence comes from the
  topology matrix.
- The guard the differential run used covered 108 code points. A
  re-derivation over the whole code space on this wheel found 154, a strict
  superset; toktier's adapter compiles the 154-code-point set from its
  shipped registry. A wider guard routes more documents to the reference and
  cannot turn an equal result into an unequal one, so the run's guarded
  reading stands under the wider set.

What toktier says about an installed build follows its content digest. A
wheel built from the same sources on another machine or toolchain is a
different artifact with a different digest, and the readings above are not
about it; the toktier documentation lists the states its adapter reports.

## Which tree the 0.3.1.1 artifacts come from

The 0.3.1.1 wheel above and its sdist were built from the five-patch tree
`aa1924284ec4abaedcc8ed5823ee17e7959c55c5`, before patch 0006 existed. The
notice patch is carried in the series so that source form built from it
states, in each modified file, that the file was changed. It cannot be
folded into the published wheel without changing the compiled bytes: source
line numbers reach the binary through diagnostics metadata, so a build from
the six-patch tree reports a different engine digest. `build_pinned.sh`
applies the full series by default and can be told to stop after 0005 to
reproduce the tree the published wheel came from.

## Upstream reports

Reports about the behaviour behind patches 0001, 0002, 0003, 0004 and 0005
have been submitted to the upstream project, each with a self-contained
reproducer. This file is updated when their status changes.

## Maintenance

When upstream publishes a new release, the patch series is rebased onto it
and the checks above are re-run before a new pinned build is published. If
upstream lands its own fix for one of these cases, the corresponding patch is
retired rather than carried forward.
