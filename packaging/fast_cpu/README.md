# Corrected Gigatoken provenance

TokTier's certified CPU path contains a downstream-corrected implementation of
[Gigatoken](https://github.com/marcelroed/gigatoken).  It is linked directly
into the core `toktier._native` extension; TokTier neither installs nor imports
a top-level `gigatoken` distribution, and the wheel contains no second native
extension.

The downstream patch fixes Unicode-data and normalization inputs to align with
the certified Hugging Face reference and adds checked loading behavior.
Gigatoken remains an independent MIT-licensed project; its authors do not
endorse this build or TokTier's certification results.

## Active build identity

The active certificate is source-bound because the corrected CPU engine now
shares one extension with the Rust router, store, recovery logic, native HF
fallback, and CUDA Driver host.  These two independent implementations
enumerate and hash the same domain-separated, path-bound source set:

```bash
python tools/fast_cpu_source_identity.py
maturin build --locked --release
python - <<'PY'
from toktier.backends.fast_cpu import fast_cpu_engine_facts
print(fast_cpu_engine_facts())
PY
```

`crates/toktier-py/build.rs` embeds the source digest, exact `rustc --version`,
and release/LTO flags in the executing extension.  At runtime the planner
compares those observed values, the repair-table digest, oracle version, and
tokenizer artifact with `tools/fast_cpu_binding.json`; any mismatch closes the
fast path and runs the HF reference.

## Historical campaign lineage

For reproducibility, the original standalone corrected build can still be
recreated with:

```bash
TOKTIER_GIGATOKEN_BUILD_ROOT="$PWD/.build/gigatoken" \
  packaging/fast_cpu/build_pinned.sh
```

That recipe fixes the upstream commit, downstream patch, Rust compiler, Cargo,
Maturin, ICU4X data generator, UCD input, and ICU export input.  Its historical
native digest is the one used by the archived 12.4-TB campaign.  It is retained
as lineage in `tools/fast_cpu_binding.json`, not as authority for the current
executing extension.  The integrated native front end has its own direct
differential reading in
`readings/fast_cpu_native_frontend_parity.json`.

The core wheel carries Gigatoken's MIT license, TokTier's modification notice,
and the current integrated `toktier._native` dependency graph's CycloneDX SBOM
and deterministic license bundle. The SBOM and bundle retain their historical
filenames in this directory for wheel-path compatibility, but are generated
from the repository's current locked `toktier-py` closure by
`tools/generate_native_legal.py`; they no longer describe the old standalone
Gigatoken wheel. The packaging and CI gates check them byte-for-byte.
