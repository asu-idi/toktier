# The GPU backend (kernel deliveries)

This note describes how the CUDA backend is delivered, what a
certificate for each delivery binds, and how the settings that used to
be environment variables are expressed now.

## 1. Two deliveries: prebuilt binary first, JIT fallback

The kernels reach a process in one of two forms, and the loader prefers
the first whenever it can serve:

**Prebuilt (the `gpu` extra, default).** The wheel ships the kernels as
a multi-architecture fatbin (`sm_75/80/86/89/90/100/120` plus a
`compute_75` PTX fallback) together with a build manifest that pins the
fatbin digest, per-architecture image digests and the full compiler
invocation. The loader verifies the fatbin against the manifest and
loads it through the CUDA driver API (bound via the standard library's
`ctypes` over the user's own `libcuda`; no new dependency). It needs an
r580-generation (CUDA 13) or newer driver, `torch` for tensors and
streams, and nothing else: no `nvcc`, no `ninja`, no first-load
compile. Because the judged binary *is* the shipped binary, the
registry records this delivery as `certified`, bound to the fatbin's
binary digest, on the architectures its verdict battery ran (sm_89 and
sm_120); the other embedded architectures are labeled `experimental`.
When the prebuilt delivery cannot serve — old driver, missing torch,
digest mismatch — the loader falls back to JIT if that delivery is
installed and eligible, and otherwise the request continues down the
routing chain, which is corrected Gigatoken → HF (Section 9). The
reason is recorded, never silent. An
explicit `delivery="prebuilt"` or `delivery="jit"` request that cannot
be served raises instead of substituting.

**JIT (the `gpu-jit` extra, fallback).** The wheel also ships the
kernel as CUDA source. The first time a process needs it, the loader
compiles it with `torch.utils.cpp_extension` into the cache directory
(`docs/contracts/config.md` Section 5). Compilation needs `torch` and
`ninja`, plus a CUDA toolchain (the build system resolves it through
`CUDA_HOME`, `CUDA_PATH`, the `nvcc` on `PATH`, or `/usr/local/cuda`,
in that order; `toktier doctor` reports the same search and the parsed
compiler release/build identity). A minimal construction and encode example
is in Section 9.

The same probe-only `doctor` command reports each detected device's index,
name, and architecture, the observed driver version, and the selected
delivery's certification status on every observed architecture. It never loads
a kernel. In its output, `cuda_available` means that TokTier's CUDA runtime
binding is installed; `cuda_hardware_present` is the separate device-presence
answer.

Under the JIT delivery it also reports the toolchain judgement itself, so the
refusal below is visible before a build is attempted:
`jit_toolchain_observed` is this machine's compiler/runtime triple,
`jit_toolchain_constraint` is the judged set it is compared against, and
`jit_toolchain_satisfied` is the shared judgement (`null` under the prebuilt
delivery, which has no compiler premise). `automatic_gpu_eligible` combines
that with candidacy, the observed architectures, and the delivery's own
materials, and `automatic_effective_backend` names what an at-or-above-
crossover automatic request would then use. `automatic_gpu_candidate` remains
the installation-level fact -- torch importable, GPU not disabled -- and is
not an eligibility answer.

Those fields describe the installation. `toktier doctor --family FAMILY`
adds a `family` section that describes one family on it: its certification
identity and evidence id, its `fast_cpu` and GPU statuses, the selected
delivery's status on each observed architecture, and two effective-backend
answers -- one at or above the GPU threshold and one below it. The second is
where families differ: a family whose CPU lane is the reference engine reads
`hf` below the threshold while the installation-level field, correctly, reads
`fast_cpu`. Without the option the section is `null`, and every other field is
unchanged.

Because a JIT build product is machine-local, it is not bit-identical
to the build the certification runs judged. The registry therefore
records the JIT delivery as `certified_source` rather than `certified`,
and binds:

| Bound value | Where it comes from |
|---|---|
| kernel source digest | `toktier.kernels.kernel_source_digest()` |
| build flags digest | `BuildFlags.digest()` |
| toolchain constraints | actual selected NVCC path/release/build, `torch.version.cuda`, and exact PyTorch distribution version recorded from the loading process |
| device architecture list | the architectures actually judged |
| class table digest | `ClassTableStore.binding_digest()` |

`certified_source` is reported distinctly from `certified` everywhere
they appear, on both reporting surfaces: the facade's `explain()`
reports which delivery the process runs (`kernel_delivery`), the
shipped availability of each delivery, and a per-architecture status
map for each under `kernel_deliveries`; the explicit engine's
`explain()` (Section 9) reports the same dimensions for the loaded
process, oracle state included. That distinction is contract, not
presentation.

## 2. One loader, one flag set

A `certified_source` certificate covers exactly one kernel build
configuration per process. `toktier.engine.gpu.loader.KernelLoader` is
the only place in the package that compiles the kernel, and the first
successful build fixes the flag set for the process.

Asking for a second, different flag set does not produce a second build:
it raises `KernelIncompatible` (`R_KERNEL_DIGEST_MISMATCH`) and marks the
process certificate void, so anything reported afterwards says
uncertified rather than claiming a certificate whose premise is gone.
Two host tests keep this honest: one asserts that exactly one module
mentions `cpp_extension`, the other exercises the runtime refusal.

Build products live under
`<cache>/kernels/<extension>-<flag digest>-tc-<toolchain digest>`; the latter
binds the selected compiler path/release/build plus the torch runtime and
distribution versions. A build made with NVCC 13.0 therefore cannot be reused
after that compiler selection drifts to NVCC 13.2.
Nothing sets `TORCH_EXTENSIONS_DIR`, and no path is hardcoded.

The same one-per-process rule applies to the delivery: the first load
fixes it, and a later request for the other delivery is refused rather
than served, so two kernel identities never appear under one report.

What a caller sees depends on who asked. The loader raises
`KernelIncompatible` (`R_KERNEL_DIGEST_MISMATCH`) to whoever called it.
A routed request reaches the loader through the GPU lane, which treats a
delivery that will not load the way it treats any other backend that
cannot serve: the refusal is recorded as a fallback -- `R_EXEC_FAULT` in
`fallback_counts`, with the loader's own sentence in `explain()`'s
`gpu_backend.load_error` -- and the request is answered on the exact CPU
lane. The ids are the ones the request would have had either way.

The certificate goes on describing the delivery that is loaded, because
that is the delivery that ran; the refused request contributed no
accelerated execution for it to describe. Whether such a refusal should
instead reach the caller as a raised `KernelIncompatible` and void the
process certificate outright is an open question (`ROADMAP.md`, "Kernel
distribution"); this section describes what the release does.

## 3. Generated lookup tables are artifacts

Kernel split behaviour depends on character-class tables derived from
the reference tokenizer package, and therefore from its Unicode version.
Generating them lazily inside the load path -- as the prototype did
-- would let a reference-package upgrade change split behaviour silently
while the certificate stayed green.

So generation is an explicit command, run from a source checkout:

```
python tools/generate_class_tables.py --table cl100k_v3 --out-dir DIR
python tools/generate_class_tables.py --out-dir DIR --manifest MANIFEST
python tools/generate_class_tables.py --out-dir DIR --check --manifest MANIFEST
```

Artifact inputs for the full run: most tables are probed from the
reference engine alone, but `deepseek_v1` reads split patterns from its
band's artifacts (`deepseek_v3`, `deepseek_v4_flash`, `hy3`) and
`kimi_v1` checks the frozen splitter fingerprint of `kimi_k3`. A
`--manifest` is a JSON object mapping family name to `{"local_dir":
...}`; families it does not define are looked up in the local toktier
artifact cache (`toktier artifacts fetch <family>`, honoring
`TOKTIER_HOME`), so the four families this check reads need no manifest
once fetched. To check a subset instead, name the tables:

```
python tools/generate_class_tables.py --out-dir src/toktier/kernels/tables \
    --check --table cl100k_v3 --table cl100k_m2l_v3 --table nfc_qc_v1
```

Each table is written with a sha256, the digests go into the routing
data, and the loader verifies the bytes before use. A table that does
not match closes the accelerated path with `R_KERNEL_DIGEST_MISMATCH`,
exactly as a kernel source mismatch would.

Delivery: the generated tables ship in the wheel, pre-generated and
digest-pinned in the packaged `toktier/kernels/tables/` directory, so
installing either GPU extra is sufficient to construct the engine --
*these tables* are not generated at install time or on first use, and
the shipped bytes are byte-identical to the tables the certification
campaigns judged against.

That statement is about the class tables and should not be read as "the
first use does no work". A first construction still derives the family's
BPE tables from the verified `tokenizer.json` and caches them: on
`qwen3_8b` that is about 17.8 MB of cache data and a cold
construction-plus-encode of roughly 2.2 s, after which the cache is
reused. What the prepackaged tables buy is that no *compiler* and no
class-table generation step is required -- the derived-cache work is
ordinary first-use preparation, not a build. The engine still looks for each table in order: an
explicit `table_dir` handed to `ClassTableStore`, the packaged
directory, then `class_tables/` under the resolved cache directory.
When a required table is absent or does not match its bound digest,
engine construction refuses with `KernelIncompatible`
(`R_KERNEL_DIGEST_MISMATCH`) naming the table id, its bound digest,
every searched path, and the generating command as a remedy. To supply
tables yourself (a source checkout without packaged tables, or a
staging directory), run the generator with `--out-dir` pointing at
`<cache>/class_tables` (without `--out-dir` it writes into the cache
directory the command's own environment resolves -- export
`TOKTIER_HOME` first when you keep a workspace-local cache), or copy
files generated elsewhere into that directory; the digest check binds
them either way. Because generation probes the reference package, the
tables depend on that package where they are *produced*; shipping them
pre-generated is exactly what makes the installed GPU path independent
of the local reference package's Unicode tables at run time.

## 4. Routing data has one source

`src/toktier/kernels/tables/kernel_families.v1.json` is generated by the
registry tooling and is the only place a family is named. It carries,
per family: the kernel band, the ruleset selector, the digits-per-piece
bound (or `null` when the class-table metadata carries it), the class
table id, whether the splitter has the contraction alternative, and any
shared-model relationship.

Runtime code carries no parallel copy: the encoder dispatch keys on
bands, and a host test parses every module to assert that no family name
appears as a string literal anywhere in the package. This is not
fastidiousness. When the mapping lived in two places, adding a family
meant editing both, and a release once did it in one place only.

## 5. Settings that used to be environment variables

The configuration contract keeps five long-term environment variables
and forbids any switch that can change output correctness from existing
in environment or configuration form. Everything the GPU path used to
read from the environment is now an explicit field of
`toktier.engine.gpu.options.GpuOptions`, an argument, or gone.

| Prototype environment variable | Released form |
|---|---|
| `*_CACHE_DIR` | `Config.cache_dir` (via `TOKTIER_HOME`) |
| `*_MANIFEST_EXTRA` | explicit overlay argument to `load_manifest` |
| `*_ADDED_FRONT` | `GpuOptions.added_token_frontend` |
| `*_MEMO` | `GpuOptions.piece_memoization` |
| `*_NVCC_EXTRA` | `BuildFlags` (bound by the certificate) |
| `*_O200K_CUDA` | `GpuOptions.o200k_cuda_starts` |
| `*_O200K_HOST_WIN` | `GpuOptions.o200k_host_windows` |
| `*_O200K_BATCH_CUDA_MIN` | `GpuOptions.o200k_batch_cuda_min` |
| `*_O200K_GRAPH_MAX` | `GpuOptions.graph_max_bytes` |
| `*_KIMI_CUDA` | `GpuOptions.kimi_cuda_starts` |
| `*_BPE_MONO_GUARD` | **removed**; the guard is unconditional |
| `*_PAR_MERGE_NONEXACT` | **removed from the kernel source** |
| `*_CONTENT_CHECK` | kernel build-time macro, default off |
| `*_L2_PIN` | removed (measured to give nothing) |
| `*_GPU_DIGEST`, `*_HOST_AMORTIZE` | removed (judgement harness only) |
| campaign and site variables | removed |

Two of those removals matter for correctness:

- The **non-monotone merge table guard** is what keeps batched merging
  exact for merge tables that are not rank-monotone. An option to switch
  it off is an option to produce ids that differ from the reference, so
  there is none. Families with a monotone table pass an empty bitmap and
  take the kernel's unchanged path.
- The **non-exact parallel plateau merge** was a prototype mode whose
  exactness would need an offline vocabulary certificate, and for which
  counterexamples exist where equal-rank merges interact. It is not
  switched off in the released kernel; it is not in it.

## 6. Bands and what each one supports

| Band | End to end | Delivery forms | Notes |
|---|---|---|---|
| cl100k | yes | eager, fused, graph | GPT-style splitter; a variant adds a stage-0 newline-run cut |
| deepseek | yes | eager, fused, graph | three-splitter ruleset, single-pass kernel |
| o200k | yes | eager, fused, graph | fused entry is CUDA-Graph capturable |
| kimi | yes | eager | o200k splitter plus a leading Han branch; the sparse cases are resolved in host-selected device windows, so there is no single capturable call |

Which delivery forms a band has is declared by its encoder entry point,
and a form a band does not provide is refused rather than replaced by
another: serving a different form silently would report a delivery the
caller never got. A caller that names no form gets the one the band
offers a single request.

A band with no end-to-end encoder at all is likewise declared as such in
the routing data, and the engine refuses end-to-end requests for it
rather than reporting it as GPU-certified end to end. No band currently
shipped is in that state.

## 7. Normalization

Artifacts declare no normalizer, an empty normalizer sequence, or NFC;
anything else is refused at construction. For NFC families the text is
normalized before encoding, and the normalization is performed by the
reference package's own normalizer -- never by another implementation.
The reason is concrete: the reference package's normalizer and its regex
engine were built against different Unicode versions, so a third-party
normalizer will apply compositions the reference engine does not know,
and the ids then differ.

A quick check runs on the device over the bytes that are being copied
there anyway. Passing it proves the reference normalizer is the identity
on this text, so nothing has to be done. Failing it goes to an exact
decision on the host, which cuts the text at safe starters and asks the
reference normalizer only about the segments that need it.

## 8. Running the tests

Host tests (no GPU, no torch) run in ordinary CI:

```
pytest tests/gpu
```

Two of them read the native host's compile-time identity out of the
compiled `toktier._native`. `pytest.ini` runs the suite against `src`,
so a source tree that has never been built has no extension for them to
read and they skip with that reason rather than failing. Building the
extension into `src/toktier` (`maturin develop`, or `maturin build
--locked` and placing it there) runs them.

The hardware suite:

```
pytest tests/gpu -m gpu \
    --artifact-manifest /path/to/tokenizer_manifest.json \
    --class-table-dir /path/to/generated/tables
```

It compiles the kernel once, compares every certified end-to-end family
against the reference tokenizer, checks that the eager, fused and graph
forms agree, checks the batched channel element by element against
per-document encoding, and checks the frozen invariants of the ragged
batch shape.

## 9. Automatic facade and explicit engine

The normal entry point is automatic. With `toktier[gpu]`, an eligible
cold/plain input of at least 64 KiB executes on the shipped prebuilt kernel;
smaller inputs and session appends execute on the corrected Gigatoken CPU
path. With `toktier[gpu-jit]`, the routing semantics are identical and only
kernel delivery changes to a local JIT build:

```python
import toktier

tok = toktier.load("qwen3_8b")
long_text = "TokTier routes this request automatically. " * 2048
ids = tok.encode(long_text).ids
print(tok.explain()["runtime_policy"])
```

`gpu_delivery="prebuilt"` or `gpu_delivery="jit"` overrides automatic profile
detection, and `gpu_min_bytes=` changes the UTF-8 byte crossover. The explicit
engine below remains available for benchmarking and low-level integration.

When the GPU engine opens is a separate question from which backend runs a
given input, and both deliveries answer it the same way. Under certified
prebuilt delivery -- the default path on a supported GPU -- requests take the
native single-call route, and that runtime opens the GPU engine on the first
request that actually routes to the GPU, that is at or above the crossover;
requests below it leave the device untouched.
`explain()["gpu_backend"]["loaded"]` therefore stays `false` while only short
requests have run, and a failed open is latched and reported as
`gpu_backend.load_error` while the frozen fallback chain keeps serving
exact ids. Under JIT delivery, and under the experimental
`repair_backend="fastokens"` adapter, requests stay on the Python host, whose
GPU backend opens the same way, at the first input that routes to the GPU.

A registry-judged triple of the actual selected NVCC release,
`torch.version.cuda`, and PyTorch distribution version can be compiled or
warmed explicitly:

```bash
toktier gpu compile qwen3_8b
```

Since 0.2.6 an unjudged triple is a coverage gap rather than a
verification failure, and the default `SUPPORTED` policy compiles it,
runs it, caches the product like any other, and labels the route
`supported_untested`. What still refuses is everything that did not
verify: a kernel source or host digest, build flags, the engine
bindings, or a compiler component anyone can write to.

Under `policy="certified"` the older behaviour is exactly what happens.
The triple gate is fail-closed there: that command and `device="cuda"`
fail with `BACKEND_UNAVAILABLE`, and the error names the selected
compiler path/release, torch runtime, certified constraint, and both
ways forward. The `device="auto"` path emits a `RuntimeWarning` and
continues through the corrected Gigatoken → HF chain. A matching torch
CUDA label is insufficient on its own: torch CUDA 13.0 with an actually
selected NVCC 13.2 is unjudged.

To measure an unjudged combination on your own text rather than take it
on trust:

```bash
toktier verify-local --family qwen3_8b --engine gpu --input my-text.txt
```

It encodes each document on the GPU route and on the reference engine
and compares every id; `--synthetic N` generates documents from rules
when there is nothing at hand. A check that agreed raises the label to
`locally_verified` until the driver, toolchain, kernel, source identity
or artifact moves; a check that disagreed is reported and changes
nothing, and `policy="certified"` is the way to hold the combination on
the reference route.

There is one deliberately loud escape hatch for experiments:

```bash
toktier gpu compile qwen3_8b --accept-uncertified-jit
```

The flag selects `EXPERIMENTAL` for that command only. It prints
`UNCERTIFIED JIT OPT-IN`, compiles and executes a probe, and returns the
machine-readable `experimental_waivers`. It neither edits the registry nor
persists permission for another process. Code that later executes the cached
kernel must opt in again:

```python
tok = toktier.load(
    "qwen3_8b", policy="experimental", gpu_delivery="jit"
)
print(tok.explain()["experimental_waivers"])
```

Those outputs are outside TokTier's certified exact-ID guarantee.
Prerequisites for either surface, all covered above:

1. the `gpu` extra installed with an r580-or-newer driver (or the
   `gpu-jit` extra with a CUDA toolchain the build system can find;
   Section 1);
2. the artifact fetched and verified (`toktier artifacts fetch <family>`,
   or a source that may fetch in the snippet below).

The generated class tables ship in the wheel (Section 3); construction
refuses with the searched paths and the remedy in the exceptional case
that a table is absent or fails its digest check.

```python
import toktier
from toktier.artifacts import ArtifactManifest, ArtifactStore
from toktier.artifacts.tables import ARTIFACT_MANIFEST
from toktier.config import Config
from toktier.engine.gpu import GpuEngine

config = Config.resolve()
manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
# source=None stays offline; HuggingFaceSource() would allow fetching.
store = ArtifactStore(manifest, config=config, source=None)

engine = GpuEngine.from_store(store, ["qwen3_8b"], config=config)
encoder = engine.encoder("qwen3_8b", kind="fused")

ids = encoder.encode("hello world")          # list[int]
reference = toktier.load("qwen3_8b")
assert ids == list(reference.encode("hello world").ids)
print(engine.binding_set())                  # the values a certificate binds
print(engine.explain())                      # delivery, certification, oracle
```

`engine.explain()` is the explicit-engine counterpart of the facade's
`explain()`: it reports the kernel delivery this process actually
loaded, the shipped prebuilt fact (the same answer `toktier doctor`
gives), the observed device architecture, the installed oracle version
against the certified set (`oracle`, with `uncertified_oracle: true`
whenever the installed version falls outside it -- the certificate does
not attach to such a process), and one certification verdict per
family (`state` plus machine-readable `reasons`) for the delivery,
architecture and oracle in effect. For JIT it also reports `toolchain_facts`
(selected/resolved NVCC path, release/build, torch CUDA and PyTorch versions)
and `jit_toolchain_satisfied`; an unverified triple cannot carry a per-family
certified verdict. `binding_set()` carries the same facts and oracle block, so
a logged binding set is never silent about the compiler or reference version
it ran against (registry.md Section 2).

Under the prebuilt delivery the engine is ready as soon as the fatbin
is verified and loaded; under the JIT delivery the first `encode`
triggers the one build of the process (Section 2) and later processes
reuse the cached build. The engine returns raw id
lists; the routing layer's guarantees (fallback accounting, plan
reasons) live above this entry point and are not part of it in this
release. The hardware test suite (Section 8) is the recorded form of
the reference comparison shown inline here.
