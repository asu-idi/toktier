# toktier

**English** | [简体中文](https://github.com/asu-idi/toktier/blob/v0.1.0/README.zh-CN.md)

**Tokenize the conversation once — after that, only what's new.**

toktier is a stateful tokenization system for agentic LLM serving. It keeps
per-session token state, repairs appended text with a certified CPU path, and
offers a certified GPU path for fresh or large requests. Both fast paths return token IDs
**bit-identical to a full Hugging Face (HF) `tokenizers` encode from scratch**.

- **Exact, at scale.** The release campaign records **53.2 billion checks**
  across 14 tokenizer artifacts and 3.8 billion real documents
  (12.33 trillion characters), with zero observed divergence.
- **Fast on both paths.** On the recorded benchmark battery, the explicit GPU
  engine encodes a fresh 4-million-character request (~786K tokens) in
  **3.88 ms**. CPU repair answers a 256-character append to a 4.19M-character
  session in **1.68 ms**.
- **Certified before acceleration.** Fast paths are admitted only for the exact
  tokenizer artifact, oracle version, kernel delivery, and architecture covered
  by recorded evidence. `explain()` reports the route and its reasons.

![Latency head-to-head: TokTier versus full re-encode across three workloads of a 4M-character session](https://raw.githubusercontent.com/asu-idi/toktier/v0.1.0/docs/figures/hero_session_vs_reencode.svg)

Every bar is a measured median. Exact values, workload sizes, and sample counts
are in
[`hero_session_vs_reencode.data.json`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/figures/hero_session_vs_reencode.data.json);
the complete sweeps are in [`docs/benchmarks.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/benchmarks.md).

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
trusted hit; cache eviction changes latency, not output.

Routing is explicit:

```python
from toktier import RoutingPolicy

tok = toktier.load("qwen3_8b", policy=RoutingPolicy.CERTIFIED)
```

`CERTIFIED` is the default, `REFERENCE` pins HF `tokenizers`,
`REQUIRE_ACCELERATED` raises when no admitted fast path exists, and
`EXPERIMENTAL` permits uncertified combinations for evaluation. With the exact
corrected Gigatoken binding installed, the facade selects `fast_cpu` for the
11 certified tokenizer artifacts; without that binding it stays on HF and
reports why. The GPU engine has a separate explicit API.

## Install

```bash
pip install toktier                 # reference backend, store, routing, CLI
pip install "toktier[fast]"          # HF/Transformers versions for certified CPU repair
pip install "toktier[gpu]"          # prebuilt GPU delivery
pip install "toktier[gpu-jit]"      # locally compiled GPU delivery
```

| Install | Delivery | Requirements |
|---|---|---|
| `toktier` | CPU reference path and persistent store | Linux x86_64, CPython 3.10+ |
| `toktier[fast]` + corrected Gigatoken wheel | Certified stateless CPU and incremental repair | exact version, native-module and repair-table binding from the recipe below |
| `toktier[gpu]` | Shipped multi-architecture CUDA fatbin | NVIDIA GPU, driver 580.65.06+, `torch`; no compiler or first-use build |
| `toktier[gpu-jit]` | Same kernel source compiled locally | matching CUDA toolkit, `nvcc`, `torch`; first-use compilation |

The certified CPU engine is a corrected, data-version-pinned Gigatoken build,
not an arbitrary package with the same import name. From a source checkout:

```bash
pip install ".[fast]"
TOKTIER_GIGATOKEN_BUILD_ROOT="$PWD/.build/gigatoken" \
  packaging/fast_cpu/build_pinned.sh
pip install .build/gigatoken/wheels/gigatoken-*.whl
```

The [reproducible build recipe](https://github.com/asu-idi/toktier/blob/v0.1.0/packaging/fast_cpu/README.md) pins the upstream
commit, patch, Unicode inputs, compiler, and build backend; the registry then
verifies the installed version, native-module
digest, repair configuration, and tokenizer artifact before opening the route.
Its wheel carries Gigatoken's MIT license and the TokTier modification notice.

The prebuilt fatbin contains `sm_75/80/86/89/90/100/120` images and a
`compute_75` PTX fallback. Its binary-digest-bound certificate covers `sm_89`
and `sm_120`; the other embedded architectures are marked `experimental`. The
JIT delivery is `certified_source` on `sm_89` and `sm_120`, meaning its
certificate binds source, class tables, flags, and toolchain constraints rather
than a machine-local binary. See [`docs/gpu-jit.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/gpu-jit.md) for the
explicit engine API and delivery diagnostics.

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

The core package has no `torch` dependency and can be imported without CUDA,
network access, or hardware probing. Artifact caches, compiled-kernel caches,
and persistent session state use separate directories.

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
[`evidence/evidence_manifest.json`](https://github.com/asu-idi/toktier/blob/v0.1.0/evidence/evidence_manifest.json),
[`evidence/evidence_manifest_added_families.json`](https://github.com/asu-idi/toktier/blob/v0.1.0/evidence/evidence_manifest_added_families.json),
and [`tables/support_registry.json`](https://github.com/asu-idi/toktier/blob/v0.1.0/tables/support_registry.json). Shipped
per-artifact readings account for 49,920,199,013 checks; an archived earlier
phase accounts for the remaining 3,280,031,861. Together they produce the
headline total above. A focused end-to-end rerun through the released public
session API covers all 11 CPU-fast artifacts in
[`readings/fast_cpu_focused_parity.json`](https://github.com/asu-idi/toktier/blob/v0.1.0/readings/fast_cpu_focused_parity.json).

Three status values keep evidence and runtime behavior distinct:

| Status | Meaning |
|---|---|
| `certified` | Evidence binds the exact artifact and accelerated binary. |
| `certified_source` | Evidence binds source, build inputs, and constraints for a local build. |
| `reference-only` | No accelerated route is admitted; HF `tokenizers` runs. |

These are empirical differential results, not a proof over all possible input.
Per-request checks and the reference fallback remain part of the contract.

Repository self-checks:

```bash
python tools/generate_evidence.py --check
python tools/validate_registry.py
python tools/generate_registry.py --release-check
python tools/dev.py test-packaging
```

## Performance

The top figure compares the explicit GPU and repair engines with full
re-encoding on identical text. The released batched GPU path also has a
same-host throughput measurement:

| Path | Throughput | Setup |
|---|---:|---|
| GPU, end to end (text in, IDs out) | 0.6028 GB/s | one RTX PRO 6000 Blackwell |
| HF reference CPU path | 0.0047 GB/s | same host and input, one CPU core |

This pair uses 2.2 GB of RAM-resident real web text and reports UTF-8 bytes over
wall clock with host ID arrays materialized. Full protocols, all cells, and
provenance are in [`docs/benchmarks.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/benchmarks.md).

![Single-request latency](https://raw.githubusercontent.com/asu-idi/toktier/v0.1.0/docs/figures/f1_single_request_latency.svg)

![Session tail latency](https://raw.githubusercontent.com/asu-idi/toktier/v0.1.0/docs/figures/f2_session_tail_latency.svg)

![Session state memory](https://raw.githubusercontent.com/asu-idi/toktier/v0.1.0/docs/figures/f3_session_state_memory.svg)

![Repair-path equivalent throughput](https://raw.githubusercontent.com/asu-idi/toktier/v0.1.0/docs/figures/f4_repair_equivalent_throughput.svg)

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

[`docs/support-matrix.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/support-matrix.md) lists every anchor artifact,
SHA-256, backend status, and **210 verified model repositories** that share an
identical or serialization-equivalent tokenizer. Coverage follows tokenizer
content, not repository naming.

The wheel currently resolves the 14 artifacts in its generated manifest.
`kimi_k3` and the three WordPiece families have evidence but are not yet wired
into that manifest; `toktier inspect` is the authoritative packaged list.

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

See [`docs/integration/dynamo.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/integration/dynamo.md) for using the two
layers together.

## Documentation

- [`ARCHITECTURE.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/ARCHITECTURE.md) — layers, routing, and store format.
- [`ROADMAP.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/ROADMAP.md) — release scope and planned integration.
- [`docs/support-matrix.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/support-matrix.md) — artifacts and covered repositories.
- [`docs/gpu-jit.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/gpu-jit.md) — prebuilt and JIT GPU deliveries.
- [`docs/integration/dynamo.md`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/integration/dynamo.md) — Dynamo integration.
- [`docs/paper/toktier-preprint.pdf`](https://github.com/asu-idi/toktier/blob/v0.1.0/docs/paper/toktier-preprint.pdf) — current preprint.

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
[`THIRD_PARTY_NOTICES`](https://github.com/asu-idi/toktier/blob/v0.1.0/THIRD_PARTY_NOTICES) and [`packaging/`](https://github.com/asu-idi/toktier/tree/v0.1.0/packaging).

## License and citation

toktier is licensed under the [Apache License 2.0](https://github.com/asu-idi/toktier/blob/v0.1.0/LICENSE); see
[`NOTICE`](https://github.com/asu-idi/toktier/blob/v0.1.0/NOTICE) for attribution information.

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
[`CITATION.cff`](https://github.com/asu-idi/toktier/blob/v0.1.0/CITATION.cff).
