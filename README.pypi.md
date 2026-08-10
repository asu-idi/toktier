# toktier

**English** | [简体中文](https://github.com/asu-idi/toktier/blob/v0.2.0/README.zh-CN.md)

**Tokenize the conversation once — after that, only what's new.**

toktier is a stateful tokenization system for agentic LLM serving. It keeps
per-session token state, repairs appended text with a certified CPU path, and
offers a certified GPU path for fresh or large requests. Both fast paths return token IDs
**bit-identical to a full Hugging Face (HF) `tokenizers` encode from scratch**.

- **Exact, at scale.** The release campaign records **53.2 billion checks**
  across 14 tokenizer artifacts and 3.8 billion real documents
  (12.33 trillion characters), with zero observed divergence.
- **Fast on both paths.** On the recorded benchmark battery, the GPU path
  encodes a fresh 4-million-character request (~786K tokens) in
  **3.88 ms**. The bounded native CPU repair for a 256-character append to a
  4.19M-character session takes **1.68 ms**; that reading is the
  `toktier repair (HF tokenizers window)` lane, which is the repair window this
  cell measured. The corrected-Gigatoken window is a separate lane in the same
  figure (2.39 ms on the 65,536-character append). Both are bounded repairs
  under the routing table below; the figure data names the lane of each bar.
  This measures the repair operation
  itself; it excludes materializing the full historical token sequence as a
  Python tuple. A native Rust serving integration can avoid this full-sequence
  materialization by retaining session state and consuming only the repaired suffix.
- **Certified before acceleration.** Fast paths are admitted only for the exact
  tokenizer artifact, oracle version, kernel delivery, and architecture covered
  by recorded evidence. `explain()` reports the route and its reasons.

![Latency head-to-head: TokTier versus full re-encode across three workloads of a 4M-character session](https://raw.githubusercontent.com/asu-idi/toktier/v0.2.0/docs/figures/hero_session_vs_reencode.svg)

Every bar is a measured median. Exact values, workload sizes, and sample counts
are in
[`hero_session_vs_reencode.data.json`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/figures/hero_session_vs_reencode.data.json);
the complete sweeps are in [`docs/benchmarks.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/benchmarks.md).

## Quick start

Install with `pip install toktier`; GPU options are described in
[Install](#install).

```python
import toktier

tok = toktier.load("qwen3_8b")          # family id from the support matrix
enc = tok.encode("hello world")         # token IDs
print(enc.ids)
print(tok.decode(enc.ids))
print(tok.explain())                    # backend chosen, and why
```

`encode` then `decode` is not necessarily a text-identity round trip: a
tokenizer whose pipeline normalizes (NFC, for instance) returns the
normalized text, and that is the tokenizer's own behaviour rather than a
TokTier divergence. The guarantee TokTier makes is about IDs, and it is
unaffected: the IDs equal a from-scratch HF encode of the same input, and
both decoders return the same text.

If application code starts from a Hugging Face model repository instead of a
TokTier family id, resolve it by content:

```python
tok = toktier.from_pretrained("Qwen/Qwen3-0.6B")
```

`from_pretrained()` downloads the audited immutable revision for a recorded
sibling or canonical repository (an unknown repository resolves `main` unless
`revision=` is passed), hashes the exact file, and consults the root-digested
210-repository sibling registry. Byte-identical, canonicalisation-equivalent,
and serialisation-equivalent records use the already certified canonical
artifact through the same CPU/GPU router. A familiar repository whose bytes
changed—and any unregistered content—stays on HF under policies that permit
the reference fallback; `REQUIRE_ACCELERATED` raises instead. See
`explain()["model_resolution"]` for both the source identity and the
canonical identity actually executed. `load(family)` remains the direct
family API and the air-gap-friendly path.

Name a growing transcript with `session=` to persist its token state across
calls and processes:

```python
tok = toktier.load("qwen3_8b", store="./toktier-store")

transcript = "user: hello\nassistant: hello! how can I help?\n"
enc = tok.encode(transcript, session="chat-42")

transcript += "user: what changed since my last call?\n"
enc = tok.encode(transcript, session="chat-42")

# Store-backed and from-scratch paths return the same IDs.
assert enc.ids == tok.encode(transcript, lookup="off").ids
```

Without `session=`, the store can find a byte-verified stored prefix by content.
Use `lookup="off"` to skip that lookup. A failed byte check is a miss, never a
trusted hit; cache eviction changes latency, not output. Long sessions whose
stable prefix has been sealed remain reusable after restart: TokTier binds the
record to the caller-presented historical prefix before restoring it, and a
missing or corrupt binding becomes a cold encode.

Routing policy is selectable and inspectable:

```python
from toktier import RoutingPolicy

tok = toktier.load("qwen3_8b", policy=RoutingPolicy.CERTIFIED)
```

| Policy | What runs | If a fast-path premise fails |
|---|---|---|
| `CERTIFIED` (default) | Only routes covered for the exact artifact, HF version, engine/kernel bytes, delivery, and hardware | Falls back to HF and records the reason |
| `REFERENCE` | HF `tokenizers` only | No accelerated route is attempted |
| `REQUIRE_ACCELERATED` | The same certified routes | Construction raises if no fast path is eligible; per-input safety fallbacks remain enabled |
| `EXPERIMENTAL` | May admit an unjudged combination for evaluation | Labels every waived premise; never the default |

The install profile and input shape then determine the automatic route:

| Situation under the default `CERTIFIED` policy | Automatic route |
|---|---|
| `toktier`, one of 11 certified tokenizer artifacts (12 model families) | Corrected Gigatoken for full CPU encoding; HF if any binding check fails |
| `toktier[gpu]`, cold/plain input below the GPU crossover (64 KiB default) | Corrected Gigatoken CPU path (HF for a family without CPU-fast certification) |
| `toktier[gpu]`, cold/plain input at or above the GPU crossover (64 KiB default) | Shipped prebuilt GPU path; then corrected Gigatoken and HF in the frozen fallback chain |
| Existing session receives a strict append | Corrected Gigatoken CPU repair for the 12 covered model families, independent of total transcript size |
| Added-token or repair guard cannot prove its premise | HF reference path for that input |

`explain()` reports the fixed chain, the crossover decision
(`gpu_min_bytes`, 64 KiB by default), the backend that actually returned the
last result, and every fallback counter.

## Rust serving API

The workspace now includes a Python-free Rust serving facade for frontends
that retain token state directly. It exposes pinned artifact fetch/mirror/
air-gap operations, reference/corrected-CPU/prebuilt-or-direct-JIT GPU routing,
continuous token buffers, bounded executor-neutral batching, persistent named
sessions, and delta-native `TokenPatch` results:

```rust,no_run
use toktier::{Device, Runtime};

let runtime = Runtime::builder().device(Device::Auto).build()?;
let tokenizer = runtime.load("qwen3_8b")?;
let mut session = tokenizer.open_session("agent-42")?;
let seed = session.seed("user: hello\n")?;
let patch = session.append("assistant: hi\n")?;
# Ok::<(), toktier::Error>(())
```

`patch.keep_tokens()` says where a retained downstream ID buffer should be
truncated; `patch.replacement_ids()` is the exact repaired suffix. The append
does not allocate the complete historical ID sequence unless the caller asks
for `snapshot()`. The crate is published on crates.io from 0.2.0 onward and
tracks the package version, so `cargo add toktier` resolves it from the
registry. See [`docs/rust-api.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/rust-api.md) for the serving surface and
[`docs/rust-lifecycle.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/rust-lifecycle.md) for acquisition, JIT,
concurrency, and reproducible offline distribution.

Since 0.1.1, the UTF-8 crossover and no-hit added-token prefilter execute in
one allocation-free Rust selector call. On the recorded RTX 5090 host, its
4M-byte control-plane microprofile fell from 2.97 ms to 0.052 ms (57.5x); this
is a routing-only measurement, separate from tokenization and Python result
materialization. See [`docs/native-routing.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/native-routing.md).

## Install

```bash
pip install toktier                 # complete certified CPU product
pip install "toktier[gpu]"          # CPU product + automatic prebuilt GPU route
pip install "toktier[gpu-jit]"      # same routing, with local JIT delivery
cargo add toktier                   # Python-free Rust serving API
```

| Install | Delivery | Requirements |
|---|---|---|
| `toktier` | Corrected Gigatoken full CPU encode and session repair, HF fallback, persistent store, routing, and CLI | Linux x86_64 with glibc 2.34+, CPython 3.10+; installs `tokenizers==0.22.2` and `transformers==4.57.6` |
| `toktier[gpu]` | Strict superset of `toktier`; automatic 64 KiB crossover to the shipped multi-architecture CUDA fatbin | NVIDIA GPU, driver 580.65.06+, `torch`; no compiler or first-use build |
| `toktier[gpu-jit]` | Same CPU/GPU routing as `toktier[gpu]`; compiles the certified kernel source locally | judged NVCC / torch-runtime CUDA / PyTorch triple, `torch`, `ninja`; first-use compilation |

JIT is fail-closed at the toolchain boundary. Certification checks the actual
`nvcc` selected by PyTorch's extension builder, `torch.version.cuda`, and the
PyTorch distribution version as independent axes. If that exact triple is not
recorded in the registry, automatic routing emits a prominent warning and keeps
using the corrected Gigatoken → HF fallback chain; an explicit CUDA request
fails with the observed compiler/runtime triple, certified constraint, and a
copyable remedy. For example, torch CUDA 13.0 with NVCC 13.2 is not treated as
the judged NVCC 13.0 combination. A judged combination can be compiled ahead of
first use with:

```bash
toktier gpu compile qwen3_8b
```

For evaluation only, an unjudged pair can be compiled with an explicit risk
acceptance:

```bash
toktier gpu compile qwen3_8b --accept-uncertified-jit
```

**This does not certify the resulting kernel.** The command runs under
`EXPERIMENTAL`, prints an `UNCERTIFIED JIT OPT-IN` warning, and records every
waived premise. Application code must also opt in explicitly with
`policy="experimental", gpu_delivery="jit"`; the acceptance is deliberately
not persisted or inherited by later certified processes. Inspect
`explain()["experimental_waivers"]` before using those results.

The corrected, data-version-pinned Gigatoken implementation is linked directly
into the core `toktier._native` extension. TokTier does not install or trust a
top-level package named `gigatoken`, and the wheel carries no second CPU native
module. The base wheel also pins the HF loader and oracle versions needed to
open this certified route; there is no separate CPU-fast installation step.

For provenance, a source checkout can independently recompute the active source
identity and build the same release profile:

```bash
python tools/fast_cpu_source_identity.py
maturin build --locked --release
```

The [provenance and build record](https://github.com/asu-idi/toktier/blob/v0.2.0/packaging/fast_cpu/README.md) pins the
upstream commit, patch, Unicode inputs, compiler, and release flags. The
executing extension reports its domain-separated source digest, exact Rust
toolchain, and build flags; the registry verifies all of them together with the
repair configuration, oracle, and tokenizer artifact before opening the route.
The core wheel carries Gigatoken's MIT license, TokTier's modification notice,
the dependency SBOM, and the dependency-license bundle.

TokTier is currently published as an ABI3 Linux x86-64 wheel, not an sdist.
An arbitrary install-time rebuild would have a different toolchain/build
identity and therefore fail closed until separately certified. The tagged
repository contains the complete source and pinned build record; sdist
publication remains a separate release decision.

The prebuilt fatbin contains `sm_75/80/86/89/90/100/120` images and a
`compute_75` PTX fallback. Its binary-digest-bound certificate covers `sm_89`
and `sm_120`; the other embedded architectures are marked `experimental`. With
the default facade, `toktier[gpu]` selects this prebuilt delivery and
`toktier[gpu-jit]` selects JIT from the detected profile; an explicit
`gpu_delivery=` argument can override that detection. Under prebuilt delivery
the GPU engine opens when the native request path is constructed, on the first
request of any size, so `explain()["gpu_backend"]["loaded"]` can read `true`
after a short request; the crossover still decides per input which backend
executes. JIT delivery keeps the Python host, whose GPU backend opens lazily at
the first input that routes to the GPU. The
JIT delivery is `certified_source` on `sm_89` and `sm_120`, meaning its
certificate binds source, class tables, flags, and toolchain constraints rather
than a machine-local binary. See [`docs/gpu-jit.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/gpu-jit.md) for the
automatic facade, explicit engine API, and delivery diagnostics.

Tokenizer artifacts are fetched from pinned upstream revisions and verified by
SHA-256; they are not bundled in the wheel. The CLI supports connected,
mirrored, and air-gapped environments:

```bash
toktier artifacts fetch qwen3_8b
toktier artifacts export qwen3_8b --out qwen3_8b.tar
toktier artifacts import qwen3_8b.tar
toktier artifacts verify qwen3_8b
toktier inspect qwen3_8b
toktier doctor --json
```

This recipe transports tokenizer artifacts, and only those. A genuinely
disconnected host also needs the TokTier wheel and every dependency wheel
staged separately (a wheelhouse or a local index); the bundle format carries no
Python distributions.

The core package has no `torch` dependency and can be imported without CUDA,
network access, or hardware probing. Artifact caches, compiled-kernel caches,
and persistent session state use separate directories: the two caches follow
`XDG_CACHE_HOME` and the session store follows `XDG_STATE_HOME`, because state
is not a cache. Relocating everything at once is what `TOKTIER_HOME` is for
(`docs/contracts/config.md` Section 5).

Fastokens 0.3.1 is available only as an explicit experimental comparison:

```python
tok = toktier.load(
    "qwen3_8b", policy="experimental", repair_backend="fastokens"
)
```

This adapter re-encodes the full session and reports
`exact_id_guarantee: false`; it is never selected by the certified policy and
is not covered by the 12.4 TB corrected-Gigatoken claim.

## Correctness and evidence

The certified reference is Hugging Face `tokenizers` 0.22.2 with default
settings. Certification is bound to exact artifact bytes and that oracle
version. If the installed HF version is outside the certified set, accelerated
routing is disabled and the request remains on the installed reference path.

| Campaign | Scale | Recorded divergence |
|---|---:|---:|
| Full-corpus differential | 14 artifacts × 3,800,016,491 documents = **53,200,230,874 checks** | 0 |
| Corpus volume | 12,328,592,579,973 Unicode code points | — |
| Released-code parity | 15,960,166 documents | 0 |
| Corrected Gigatoken CPU repair | 11 unique artifacts × 3,800,016,491 documents = **41,800,181,401 checks** (12 model families by exact-artifact inheritance) | 0 |

The machine-readable records are
[`evidence/evidence_manifest.json`](https://github.com/asu-idi/toktier/blob/v0.2.0/evidence/evidence_manifest.json),
[`evidence/evidence_manifest_added_families.json`](https://github.com/asu-idi/toktier/blob/v0.2.0/evidence/evidence_manifest_added_families.json),
and [`tables/support_registry.json`](https://github.com/asu-idi/toktier/blob/v0.2.0/tables/support_registry.json). Shipped
per-artifact readings account for 49,920,199,013 checks; an archived earlier
phase accounts for the remaining 3,280,031,861. Together they produce the
headline total above. A focused end-to-end rerun through the historical public
session API is kept in
[`readings/fast_cpu_focused_parity.json`](https://github.com/asu-idi/toktier/blob/v0.2.0/readings/fast_cpu_focused_parity.json).
The executing one-call Rust front end is separately checked across all 11
CPU-fast artifacts in
[`readings/fast_cpu_native_frontend_parity.json`](https://github.com/asu-idi/toktier/blob/v0.2.0/readings/fast_cpu_native_frontend_parity.json).

Three status values keep evidence and runtime behavior distinct:

| Status | Meaning |
|---|---|
| `certified` | Evidence binds the exact artifact and accelerated binary. The prebuilt GPU delivery additionally binds the Rust host's source digest, exact rustc, and release build facts. |
| `certified_source` | Evidence binds source, build inputs, and toolchain; used by the integrated CPU engine and locally built GPU JIT. |
| `reference-only` | No accelerated route is admitted; HF `tokenizers` runs. |

These are empirical differential results, not a proof over all possible input.
Per-request checks and the reference fallback remain part of the contract.

Repository self-checks:

```bash
pip install pytest==9.1.1 jsonschema==4.26.0    # or: pip install --group test
python tools/generate_evidence.py --check
python tools/generate_native_legal.py --check    # needs cargo
python tools/validate_registry.py tables/support_registry.json
python tools/generate_registry.py --release-check
python tools/generate_sibling_aliases.py --check
python tools/dev.py test-packaging
```

Prerequisites, stated so that a failure means a real problem rather than
a missing tool. The schema checks need `jsonschema` (the `test`
dependency group in `pyproject.toml`); without it they refuse with
`error: the jsonschema package is required ...`. The last command runs
the packaging test suite and also needs `pytest`, which the same group
carries -- the first line above installs both pins directly and covers
every command in the block. The `pip install --group` alternative reads
that group from `pyproject.toml` and needs pip 25.1 or newer (PEP 735).

`generate_native_legal.py` reads the workspace's locked resolve graph
and needs `cargo` (the toolchain pinned by `rust-toolchain.toml`). On a
fresh checkout, populate the local Cargo cache once with
`cargo fetch --locked` (network required; a plain `cargo build` is not
enough, since the legal closure covers every target) -- the check
itself then runs offline.
`generate_registry.py` reads the native host's compile-time identity out
of the built extension: it uses this tree's `src/toktier/_native` when
one has been built, and otherwise the extension of an installed
`toktier` wheel, saying on stderr which it read; the identity must equal
the current source set either way. `pytest tests/gpu` follows the same
rule: the two tests that assert on that identity read whichever
extension is available and skip with a stated reason only when neither
is.

## Performance

The top figure compares the automatic GPU and repair routes with full
re-encoding on identical text. The released batched GPU path also has a
same-host throughput measurement:

| Path | Throughput | Setup |
|---|---:|---|
| GPU, end to end (text in, IDs out) | 0.6028 GB/s | one RTX PRO 6000 Blackwell |
| HF reference CPU path | 0.0047 GB/s | same host and input, one CPU core |

This pair uses 2.2 GB of RAM-resident real web text and reports UTF-8 bytes over
wall clock with host ID arrays materialized. Full protocols, all cells, and
provenance are in [`docs/benchmarks.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/benchmarks.md).

The primary study used an RTX PRO 6000 Blackwell, but a same-protocol consumer
RTX 5090 sweep was **11–17% faster** (4.24–5.50 GB/s across the reported
families). Consumer hardware is therefore a practical target, not a reduced
mode: the RTX 4090 also passed the `sm_89` correctness and prebuilt-delivery
battery. These observations do not promise the same speedup for every GPU;
architecture, workload, and host delivery still matter.

![Single-request latency](https://raw.githubusercontent.com/asu-idi/toktier/v0.2.0/docs/figures/f1_single_request_latency.svg)

![Session tail latency](https://raw.githubusercontent.com/asu-idi/toktier/v0.2.0/docs/figures/f2_session_tail_latency.svg)

![Session state memory](https://raw.githubusercontent.com/asu-idi/toktier/v0.2.0/docs/figures/f3_session_state_memory.svg)

![Repair-path equivalent throughput](https://raw.githubusercontent.com/asu-idi/toktier/v0.2.0/docs/figures/f4_repair_equivalent_throughput.svg)

The figures name Hugging Face (HF) `tokenizers` explicitly and link to their
machine-readable `docs/figures/*.data.json` files. The benchmark document also
shows the regimes where direct use of another engine is faster.

## Support matrix

| Track | Families | Coverage |
|---|---:|---|
| Certified CPU fast repair | 12 model families / 11 unique tokenizer artifacts | corrected Gigatoken, 12.33T characters, zero observed ID divergence |
| Byte-level BPE | 15 | CPU evidence; GPU status recorded per artifact |
| WordPiece | 3 | CPU evidence |
| Structural exclusions | 2 | reason recorded |

[`docs/support-matrix.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/support-matrix.md) lists every anchor artifact,
SHA-256, backend status, and **210 verified model repositories** that share an
identical or serialization-equivalent tokenizer. Coverage follows tokenizer
content, not repository naming. `toktier.from_pretrained(repo_id)` enforces
that rule at runtime: it hashes the resolved file, maps registered content to
the canonical artifact, and otherwise remains on HF.

Of the 210 sibling rows, 191 map to canonical artifacts present in this wheel.
The other 19 do not admit acceleration because their canonical artifacts are
not packaged: seven WordPiece rows run through HF, while the 12 source-level
`kimi_k3` rows currently need a conversion artifact and therefore produce an
actionable error rather than pretending that `tiktoken.model` is directly
loadable. `toktier inspect` is the authoritative packaged-family list.

## Relation to existing work

Incremental BPE work studies how the merge stage can be extended as bytes
arrive. toktier operates one layer above it: session state contains token IDs
and spans for the complete tokenizer pipeline—normalization, pre-tokenization,
merges, and added-token handling—and an append is accepted only after its
boundary check passes.

Serving projects such as `llm-tokenizer` and NVIDIA Dynamo's
`dynamo-tokenizers` also cache encodings. The main interface differences are:

| Property | toktier | In-process prefix caches |
|---|---|---|
| Lifetime | persistent and cross-process | tokenizer-process lifetime |
| Hit check | digest proposes; stored bytes verify | digest-keyed lookup |
| Reuse boundary | certified tokenizer boundary | typically special-token boundary |
| Surface | Python library for session-owning applications | serving-gateway component |

See [`docs/integration/dynamo.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/integration/dynamo.md) for using the two
layers together.

## Documentation

- [`docs/releases/v0.2.0.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/releases/v0.2.0.md) — release notes for this version.
- [`ARCHITECTURE.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/ARCHITECTURE.md) — layers, routing, and store format.
- [`ROADMAP.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/ROADMAP.md) — release scope and planned integration.
- [`docs/support-matrix.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/support-matrix.md) — artifacts and covered repositories.
- [`docs/gpu-jit.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/gpu-jit.md) — prebuilt and JIT GPU deliveries.
- [`docs/rust-api.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/rust-api.md) — Python-free Rust serving API.
- [`docs/rust-lifecycle.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/rust-lifecycle.md) — native artifacts, direct JIT, concurrency, and offline distribution.
- [`docs/integration/dynamo.md`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/integration/dynamo.md) — Dynamo integration.
- [`docs/paper/toktier-preprint.pdf`](https://github.com/asu-idi/toktier/blob/v0.2.0/docs/paper/toktier-preprint.pdf) — current preprint.

## Acknowledgements

TokTier's CPU Fast Pass and Fast Repair build on the excellent
[Gigatoken](https://github.com/marcelroed/gigatoken) and
[Fastokens](https://github.com/crusoecloud/fastokens) projects. We thank their
authors and contributors for making this work openly available.

Corrected Gigatoken is the default certified repair-window engine for 11
unique tokenizer artifacts, covering 12 model families because NVIDIA
Nemotron-Terminal ships the `qwen3_8b` tokenizer byte-for-byte. TokTier's
compatibility patch aligns Gigatoken's Unicode data and UTF-8 handling with the
frozen [Hugging Face tokenizers](https://github.com/huggingface/tokenizers)
reference; the resulting path recorded 41.8 billion checks over 12.33 trillion
characters with zero observed token-ID divergence.

Fastokens 0.3.1 remains an explicit experimental alternative. TokTier does not
claim exact-ID equivalence for that adapter and never chooses it automatically.
Fastokens is Apache-2.0 and Gigatoken is MIT; exact revisions, license copies,
the Gigatoken patch, and modification notices are in
[`THIRD_PARTY_NOTICES`](https://github.com/asu-idi/toktier/blob/v0.2.0/THIRD_PARTY_NOTICES) and [`packaging/`](https://github.com/asu-idi/toktier/tree/v0.2.0/packaging).

## License and citation

toktier is licensed under the [Apache License 2.0](https://github.com/asu-idi/toktier/blob/v0.2.0/LICENSE); see
[`NOTICE`](https://github.com/asu-idi/toktier/blob/v0.2.0/NOTICE) for attribution information.

**Paper:** [*TokTier: Exact Stateful CPU+GPU Tokenization for Agentic LLM
Serving*](https://arxiv.org/abs/2607.29678) ·
[PDF](https://arxiv.org/pdf/2607.29678)

If you use toktier in your research, please cite:

```bibtex
@misc{zhang2026toktier,
  title         = {TokTier: Exact Stateful CPU+GPU Tokenization for Agentic LLM Serving},
  author        = {Zhenyu Zhang and Zhichao Cao},
  year          = {2026},
  eprint        = {2607.29678},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2607.29678},
  url           = {https://arxiv.org/abs/2607.29678}
}
```

Machine-readable citation metadata is available in
[`CITATION.cff`](https://github.com/asu-idi/toktier/blob/v0.2.0/CITATION.cff).
