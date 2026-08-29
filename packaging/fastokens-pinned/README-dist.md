# toktier-fastokens

This distribution is a pinned build of fastokens 0.3.1 carrying the toktier
patch set 1. It keeps the upstream import name `fastokens`, so install either
this distribution or the upstream one, not both. The toktier project
publishes it for its own explicit experimental adapter; it is not affiliated
with or endorsed by the upstream project. Reports about the id divergences
the patch set addresses have been submitted to the upstream project.

It provides the same import name as upstream:

```python
import fastokens
```

so it is a drop-in for code written against fastokens 0.3.1.

## Install only one of the two

Both distributions install a top-level `fastokens` package. Installing this
one beside upstream `fastokens` in the same environment leaves whichever was
installed last in place, silently, with two sets of metadata pointing at the
same files; uninstalling either of them then removes the shared files. To
keep only the pinned build, reinstall rather than uninstall one of the two:

```bash
pip uninstall -y fastokens toktier-fastokens
pip install "toktier[fastokens]"
```

or, to keep upstream instead:

```bash
pip uninstall -y fastokens toktier-fastokens
pip install fastokens
```

If other code needs the upstream distribution, use a separate environment:
the two share one import name, so only one of them can own the bytes that
`import fastokens` runs, and uninstalling either removes the files of both.

## What the patches change

Five patches close cases where the engine returned a token-id list that
differs from the reference tokenizer for the same input, with no error
raised, or refused an input the reference accepts; a sixth adds a notice
line to each modified source file. `CHANGES-toktier.md` lists them one by
one, `NOTICE` states which files each one touches, and `PATCHES/` carries the
patches verbatim.

Applying patches 0001 to 0005 in order to upstream commit
`fe854299553524f2156a22036a2cb4d1f2ef4d97` (tag `v0.3.1`) reproduces the
source tree the 0.3.1.1 wheel comes from: git tree hash
`aa1924284ec4abaedcc8ed5823ee17e7959c55c5`.

## Rebuilding

`packaging/fastokens-pinned/build_pinned.sh` in the toktier repository
rebuilds this distribution from a clean upstream checkout with a pinned Rust
toolchain and a pinned maturin. It checks the patch digests and the resulting
tree hash before it builds.

## Licensing

Upstream fastokens is Apache-2.0; the upstream `LICENSE` text is included
verbatim, as is the upstream `NOTICES.txt` (also reproduced inside `NOTICE`,
which carries the statement of modification required by section 4(b)). The
license texts of the compiled dependency closure are collected in
`THIRD_PARTY_LICENSES-fastokens.txt`.

## What toktier reports about it

toktier recognises the wheels it publishes by the content digest of the
installed `fastokens/` package and reports what it knows as
`engine_assurance` next to the adapter's admission word. When the installed
bytes are a published wheel, that reads `certified_pinned` and the readings
toktier took on that wheel apply; the upstream distribution reads
`upstream_build`, and a wheel built elsewhere, from this sdist or otherwise,
reads `unrecognized_build`, because the readings are about specific bytes.
The adapter itself remains an explicit experimental route in toktier and is
never selected automatically. See the toktier documentation for the full list
of states.
