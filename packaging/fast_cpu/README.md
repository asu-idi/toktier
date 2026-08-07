# Corrected Gigatoken build

TokTier's certified CPU fast path uses a downstream build of
[Gigatoken](https://github.com/marcelroed/gigatoken), not an arbitrary
Gigatoken installation. The patch fixes the Unicode-data and normalization
inputs to the versions used by the certified Hugging Face reference and adds
checked loading behavior. Gigatoken remains a separate MIT-licensed project;
its authors do not endorse this build or TokTier's certification results.

From the repository root, build the wheel in a fresh directory:

```bash
TOKTIER_GIGATOKEN_BUILD_ROOT="$PWD/.build/gigatoken" \
  packaging/fast_cpu/build_pinned.sh
```

The recipe fixes the upstream commit, downstream patch, Rust compiler, Cargo,
Maturin, ICU4X data generator, UCD input and ICU export input. On the supported
Linux x86-64 build environment it reports:

```text
wheel_sha256=9fbfe0fda617763ec65dab98de15c28c94223f515ffd71a4a296716c60f220e7
native_sha256=9a701047dafa1cdebc168851d0548a0caaf08d0523d70911cc7a24112ccf92a3
```

The native digest is byte-identical to the native module used by the archived
12.33-trillion-character campaign. The release rebuild has a different wheel
digest because it adds public provenance and modification-notice metadata; it
does not change the certified native module. Both identities and that
equivalence statement are machine-readable in
[`tools/fast_cpu_binding.json`](../../tools/fast_cpu_binding.json).

TokTier redistributes only the resulting native module, byte-for-byte, under
the private import path `toktier._vendor.gigatoken_rs` in the core `toktier`
wheel. It does not publish or require a second Python package. The source wheel
is an independently reproducible build artifact, not an installation step.

The core wheel includes Gigatoken's MIT license, a TokTier modification notice,
the corrected build's CycloneDX SBOM, and a deterministic dependency-license
bundle generated from its locked Cargo graph. Repository copies are in this
directory. At runtime, TokTier opens the path only
when the module bytes, repair-table digest, oracle version and tokenizer
artifact all match the generated support registry.
