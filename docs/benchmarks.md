# Benchmarks

Every number on this page comes from a recorded, manifest-reproducible
run; the readings, sample manifests, and harness scripts are archived
with the evidence set that backs this page, and releasing the harness
is on the roadmap. Figures show the regions the library is designed
for; the full sweeps, including configurations where other engines are
faster, are in the tables here.

## Hardware and versions

| | |
|---|---|
| Machine A (GPU + single-request) | 2x AMD EPYC 9115 (32 cores, SMT off), 1 TB RAM, NVIDIA RTX PRO 6000 Blackwell Server Edition (single card used) |
| Machine B (CPU batteries) | AMD Ryzen 9 9950X (16 cores), 123 GB RAM |
| Tokenizer family | `llama_3_1_8b`, pinned artifact sha256 `76e48799b099...` |
| Reference engine | Hugging Face (HF) `tokenizers` 0.22.2 (the certified oracle version) |
| Other engines | fastokens 0.2.0; gigatoken 0.10.0 (project-pinned build) |
| Python | 3.12.3 |

Measurement discipline: one measurement per sample per process (fresh
samples only — the reference engine keeps a word-level cache, so
re-timing the same sample in one process reads too fast); warmup
samples never enter statistics; sample order seed-shuffled and
replayed identically for every engine; CPU lanes pinned to one
physical core with single-thread environment settings; machines idle
(preflight recorded). Every timed lane's output is verified
token-identical against the reference engine; a mismatch aborts the
lane.

Note on fastokens lanes: fastokens 0.2.0 raises an internal error
when the process affinity mask has exactly one CPU and the input
exceeds 16,384 UTF-8 bytes (observed on both machines, deterministic;
independent of thread-count environment variables). Its lanes
therefore run with a two-core affinity mask; the thread-count
environment keeps its compute single-threaded (verified: CPU time /
wall time = 1.05 on an 8 MB input).

## F1 — single-request encode latency (Machine A)

One document per call, batch of one. GPU lane = the released GPU
engine (fused kernel), including transfer and launch overhead and
host materialization of ids. Cold = the first encode of a fresh
process (engine construction excluded); warm = long-lived engine
object after fixed warmups.

| Engine | Doc tier | n | p50 | p90 | p99 |
|---|---|---|---|---|---|
| toktier GPU (fused) | 1-4 KB | 1000 | 0.162 ms | 0.512 ms | 1.494 ms |
| toktier GPU (fused) | 16-64 KB | 1000 | 0.676 ms | 2.385 ms | 3.799 ms |
| toktier GPU (fused) | 128-512 KB | 482 | 0.413 ms | 3.008 ms | 4.393 ms |
| toktier GPU (fused) | ~1 MB | 239 | 0.820 ms | 1.187 ms | 3.206 ms |
| HF tokenizers 0.22.2 | 1-4 KB | 1000 | 0.527 ms | 1.567 ms | 2.395 ms |
| HF tokenizers 0.22.2 | 16-64 KB | 1000 | 19.014 ms | 35.374 ms | 39.446 ms |
| HF tokenizers 0.22.2 | 128-512 KB | 500 | 76.659 ms | 129.733 ms | 167.763 ms |
| HF tokenizers 0.22.2 | ~1 MB | 247 | 340.978 ms | 411.397 ms | 557.463 ms |
| fastokens 0.2.0 | 1-4 KB | 1000 | 0.071 ms | 0.272 ms | 0.409 ms |
| fastokens 0.2.0 | 16-64 KB | 1000 | 3.068 ms | 5.763 ms | 8.287 ms |
| fastokens 0.2.0 | 128-512 KB | 500 | 6.538 ms | 11.999 ms | 15.883 ms |
| fastokens 0.2.0 | ~1 MB | 247 | 22.767 ms | 30.523 ms | 47.371 ms |
| gigatoken 0.10 (warm) | 1-4 KB | 1000 | 0.015 ms | 0.306 ms | 0.503 ms |
| gigatoken 0.10 (warm) | 16-64 KB | 1000 | 2.715 ms | 5.350 ms | 9.164 ms |
| gigatoken 0.10 (warm) | 128-512 KB | 500 | 1.105 ms | 9.431 ms | 13.443 ms |
| gigatoken 0.10 (warm) | ~1 MB | 247 | 3.243 ms | 5.905 ms | 23.134 ms |
| gigatoken 0.10 (cold) | 1-4 KB | 250 | 1.066 ms | 26.208 ms | 26.614 ms |
| gigatoken 0.10 (cold) | 16-64 KB | 250 | 29.926 ms | 32.984 ms | 34.168 ms |
| gigatoken 0.10 (cold) | 128-512 KB | 250 | 27.687 ms | 38.659 ms | 41.751 ms |
| gigatoken 0.10 (cold) | ~1 MB | 247 | 30.145 ms | 33.107 ms | 47.224 ms |
| HF tokenizers 0.22.2 (cold) | 16-64 KB | 250 | 18.059 ms | 34.662 ms | 37.798 ms |
| fastokens 0.2.0 (cold) | 16-64 KB | 250 | 6.024 ms | 10.234 ms | 11.980 ms |
| toktier GPU (fused) | 2 MB concat | 500 | 1.958 ms | 2.249 ms | 2.968 ms |
| toktier GPU (fused) | 3 MB concat | 500 | 2.922 ms | 3.191 ms | 3.987 ms |
| toktier GPU (fused) | 4 MB concat | 500 | 3.882 ms | 4.247 ms | 5.149 ms |
| toktier GPU (fused) | 5 MB concat | 500 | 5.017 ms | 5.604 ms | 6.699 ms |
| HF tokenizers 0.22.2 | 2 MB concat | 500 | 598.033 ms | 617.587 ms | 676.790 ms |
| HF tokenizers 0.22.2 | 3 MB concat | 500 | 895.970 ms | 915.356 ms | 959.361 ms |
| HF tokenizers 0.22.2 | 4 MB concat | 500 | 1250.985 ms | 1276.876 ms | 1360.762 ms |
| HF tokenizers 0.22.2 | 5 MB concat | 500 | 1544.182 ms | 1573.572 ms | 1608.980 ms |
| fastokens 0.2.0 | 2 MB concat | 500 | 34.630 ms | 37.136 ms | 40.451 ms |
| fastokens 0.2.0 | 3 MB concat | 500 | 51.244 ms | 54.383 ms | 60.521 ms |
| fastokens 0.2.0 | 4 MB concat | 500 | 70.333 ms | 73.156 ms | 78.013 ms |
| fastokens 0.2.0 | 5 MB concat | 500 | 88.221 ms | 91.418 ms | 98.626 ms |
| gigatoken 0.10 (warm) | 2 MB concat | 500 | 5.271 ms | 6.127 ms | 11.686 ms |
| gigatoken 0.10 (warm) | 3 MB concat | 500 | 7.017 ms | 7.667 ms | 9.361 ms |
| gigatoken 0.10 (warm) | 4 MB concat | 500 | 8.817 ms | 9.727 ms | 11.382 ms |
| gigatoken 0.10 (warm) | 5 MB concat | 500 | 10.666 ms | 11.339 ms | 13.168 ms |
| gigatoken 0.10 (cold) | 2 MB concat | 250 | 35.281 ms | 35.937 ms | 37.288 ms |
| gigatoken 0.10 (cold) | 3 MB concat | 250 | 39.657 ms | 40.485 ms | 42.084 ms |
| gigatoken 0.10 (cold) | 4 MB concat | 250 | 43.216 ms | 44.277 ms | 45.530 ms |
| gigatoken 0.10 (cold) | 5 MB concat | 250 | 46.620 ms | 47.578 ms | 49.398 ms |

The 2-5 MB tiers extend the battery beyond the natural-document supply (which ends near 1 MB in this corpus): each sample is a chain of distinct real web documents concatenated to the exact target length, no document reused across samples (protocol amendment A6). The 1 KB - 1 MB tiers above are whole natural documents; the two calibers are never mixed within a tier.

Notes: at small document sizes CPU engines are competitive or faster
than the GPU path (per-call launch and transfer overhead dominates);
the GPU path pulls ahead as documents grow. gigatoken's cold first
encode pays one-time per-process initialization; its warm path is the
strongest CPU single-request number here.

## F2 — per-turn latency in a growing session (Machine B)

200 sessions of real transcript text; each session = one initial
context (log-uniform 2K-256K chars) + 30 appends (log-uniform
16-4,096 chars). The session-store lane runs the released store core
with certified boundary repair (the repair window engine is the
reference HF tokenizers crate in every variant; the variant names the
engine used for full encodes: session creation and degraded paths).
The baseline lane re-encodes the accumulated text with the bare
engine on every turn. Every session's final id stream is verified
against a fresh reference encode.

Configuration caliber: the repair lanes here run the **certified
repair configuration** — the released store core driven by the
certified repair engine through the store's public encoder interface.
The facade shipped in the current package (`toktier.load(...)`, `session=`)
uses corrected Gigatoken repair when the exact certified binding is installed;
otherwise it performs a verified HF full re-encode and records the reason.
Both routes retain durable session state, id-identical certified output, and
content lookup without manual bookkeeping.

| Lane | p50 | p90 | p99 | paths (healed / window-all) |
|---|---|---|---|---|
| tier-hf | 0.284 ms | 0.816 ms | 1.819 ms | 4001 / 1999 |
| bare-hf | 8.769 ms | 42.713 ms | 66.160 ms | - |
| tier-fastokens | 0.283 ms | 0.811 ms | 1.844 ms | 4001 / 1999 |
| bare-fastokens | 0.229 ms | 0.813 ms | 1.336 ms | - |
| tier-gt | 0.295 ms | 0.872 ms | 1.960 ms | 4001 / 1999 |
| bare-gt | 0.062 ms | 0.264 ms | 0.684 ms | - |

Raw single-engine latency can beat the tier at small sessions (bare gigatoken and bare fastokens above); the tier buys exact-id sessions, durable state, and tail stability against the certified reference engine.

## F3 — per-session state to resume tokenization

What the store durably keeps per session (sealed block records +
session record, exported bytes) vs holding the same information in
plain Python structures. Memory is machine-independent; token counts
from the pinned llama_3_1_8b artifact.

| Context | Tokens | toktier durable | store approx (mem) | naive Python | utf-8+uint32 | dynamo L1 RSS/session |
|---|---|---|---|---|---|---|
| 4,096 chars | 1,937 | 21.5 KB | 30.1 KB | 83.1 KB | 14.7 KB | 5.0 KB |
| 32,768 chars | 14,203 | 222.7 KB | 57.0 KB | 619.0 KB | 105.3 KB | 263.0 KB |
| 262,144 chars | 79,670 | 860.1 KB | 318.8 KB | 3.5 MB | 581.7 KB | 19.8 MB |
| 1,048,576 chars | 301,380 | 1.7 MB | 1.2 MB | 14.0 MB | 2.3 MB | 301.8 MB |
| 4,194,304 chars | 1,197,771 | 5.3 MB | 4.8 MB | 56.6 MB | 9.0 MB | 4.9 GB |

Context on other systems: in-memory tokenizer prefix caches exist in
the serving ecosystem (llm-tokenizer, 2026-01; dynamo-tokenizers L1,
default-on since 2026-06-30). They are throughput-optimization
components: process-local, non-persistent (the cache lifetime is
bound to a single tokenizer instance), keyed by hash with the prefix
text discarded, and cut only at special-token boundaries. The store
measured here is a durable, revision-checked session record usable
across processes and restarts, with certified cut points at arbitrary
append positions. These are different design points; the table
measures ours.

## F4 — appending a small delta to a large context (Machine B)

Caliber: **single request**. Each cell is the median wall clock of
one `append(delta)` repair call (n per cell below); equivalent
throughput = (context + delta) **chars** (not bytes) / that call's
wall clock — i.e. "one update call does the work of tokenizing this
much text from scratch". No batching, no concurrency; single CPU
core. The repair lanes splice at a certified boundary and re-encode
only a bounded window; the full lanes re-encode everything. Every
repair output is verified token-identical to a full reference
re-encode of context + delta. The repair lanes measure the certified
repair configuration (see the F2 caliber note). The shipped facade now wires
the corrected-Gigatoken configuration into its default certified session path.

The figure (`f4_repair_equivalent_throughput`) draws the frozen
measurement grid (context up to 4M chars). The 8M and 16M rows below
are off-grid extension readings taken beyond the frozen grid at low
sample counts (8M: n=5, 16M: n=1) — indicative only, and the 8M/16M
medians are not monotone with n this small.

| Context | delta | repair (HF tokenizers win.) | repair (gigatoken win.) | gigatoken full | fastokens full | HF tokenizers full |
|---|---|---|---|---|---|---|
| 4K | 16 | 10 M | 16 M | 53 M | 8 M | 3 M |
| 4K | 256 | 10 M | 17 M | 270 M | 45 M | 5 M |
| 4K | 4096 | 5 M | 27 M | 160 M | 29 M | 5 M |
| 4K | 65536 | 3 M | 77 M | 153 M | 33 M | 4 M |
| 32K | 16 | 87 M | 127 M | 111 M | 22 M | 4 M |
| 32K | 256 | 76 M | 127 M | 413 M | 87 M | 4 M |
| 32K | 4096 | 25 M | 121 M | 375 M | 80 M | 5 M |
| 32K | 65536 | 5 M | 119 M | 199 M | 42 M | 4 M |
| 262K | 16 | 559 M | 1.00 G | 277 M | 49 M | 4 M |
| 262K | 256 | 497 M | 992 M | 560 M | 155 M | 4 M |
| 262K | 4096 | 151 M | 900 M | 545 M | 152 M | 4 M |
| 262K | 65536 | 16 M | 485 M | 473 M | 105 M | 3 M |
| 1M | 16 | 1.48 G | 1.73 G | 316 M | 56 M | 3 M |
| 1M | 256 | 1.57 G | 2.40 G | 518 M | 169 M | 3 M |
| 1M | 4096 | 570 M | 2.31 G | 548 M | 184 M | 3 M |
| 1M | 65536 | 47 M | 1.21 G | 501 M | 146 M | 3 M |
| 4M | 16 | 3.22 G | 398 M | 382 M | 64 M | 3 M |
| 4M | 256 | 2.50 G | 391 M | 564 M | 171 M | 3 M |
| 4M | 4096 | 1.29 G | 282 M | 552 M | 188 M | 3 M |
| 4M | 65536 | 171 M | 1.78 G | 556 M | 178 M | 3 M |
| 8M | 16 | 2.34 G | 850 M | 348 M | 58 M | 3 M |
| 8M | 256 | 2.08 G | 637 M | 478 M | 150 M | 3 M |
| 8M | 4096 | 1.70 G | 309 M | 511 M | 152 M | 3 M |
| 8M | 65536 | 273 M | 367 M | 481 M | 154 M | 3 M |
| 16M | 16 | 1.03 G | 1.96 G | 443 M | 56 M | 3 M |
| 16M | 256 | 4.07 G | 2.19 G | 653 M | 137 M | 3 M |

Units: equivalent chars/s, median of the cell (n per cell in the aggregate JSON; grid cells n=30, 4M cells n=19; the 8M cells n=5 and the 16M cells n=1 are off-grid, low-sample extension readings — see the caliber note above). Methodology note: two harness artifacts were identified and removed during measurement — (a) first-append-on-fresh-state one-time costs, and (b) an allocator page-fault storm induced by the audit's own full re-encode adjacent to microsecond-scale timed cells; the final caliber times all cells of an instance back-to-back and audits afterwards (every output still verified). Raw first-pass files are archived alongside.

The repair path's advantage grows linearly with context size at fixed
delta: the work is bounded by the window and the delta, not the
context. At small contexts (~4K chars) a fast full re-encode
(gigatoken) is the better tool; the crossover is in the tens of
kilobytes of context.

## Bulk throughput (README headline pair)

Released GPU batched path vs the released reference CPU path, same
machine, same input (>= 2 GB of RAM-resident real web text), UTF-8
bytes / wall clock:

| Input | Path | GB/s (UTF-8) | Mtok/s |
|---|---|---|---|
| 2.2 GB web documents (avg 4.4 KB) | toktier GPU batched (1 card) | 0.603 | 119.3 |
| 2.2 GB web documents (avg 4.4 KB) | reference backend, 1 core | 0.005 | 0.9 |
| 1.07 GB larger documents (avg 180 KB) | toktier GPU batched (1 card) | 0.490 | 143.7 |
| 1.07 GB larger documents (avg 180 KB) | reference backend, 1 core | 0.003 | 0.9 |

## Session seed and append latency (0.2.0 recertified tree)

The F1-F4 batteries above are the retained release batteries with their
own recorded provenance. The 0.2.0 seed-path rework was measured
separately, on an earlier certified tree during the 0.2.0 release
preparation (since superseded by the release recertification, and differing
from the v0.2.0 release tree only in version metadata and documentation),
with 31 fresh processes per cell and a post-timing exact-ID check against
the frozen HF reference in every process. Hosts:
"RTX 5090 host" is an Intel Core Ultra 9 285K with a GeForce RTX 5090
(driver 595.84); "EPYC host" is a 2x AMD EPYC 9115. Workload: a 4 MiB
(4,194,304-character) `qwen3_8b` transcript, 2,348,809 tokens.

| Cell (p50 unless noted) | Before (pre-0.2.0 tree) | 0.2.0 |
|---|---:|---:|
| GPU public in-memory `Session::seed`, RTX 5090 host | 59.739 ms | **11.145 ms** (p95 12.010 ms) |
| GPU ID-only `encode`, RTX 5090 host | 5.181 ms | 4.113 ms |
| Explicit offsets contrast (`encode(offsets) - encode`), RTX 5090 host | 39.047 ms (span bridge) | 15.820 ms |
| GPU SQLite-backed seed, RTX 5090 host | 98.309 ms | 45.743 ms |
| CPU public in-memory seed, EPYC host | 67.395 ms | 26.465 ms |
| CPU SQLite-backed seed, EPYC host | 115.325 ms | 70.142 ms |
| Corrected-CPU engine control cell, EPYC host | 47.454 ms | 47.209 ms |
| Warm 256-char append (Python facade, 21 samples, RTX 5090 host) | 11.445 ms | 8.697 ms |
| Warm 1,500-char append (Python facade, 21 samples, RTX 5090 host) | 12.826 ms | 9.417 ms |

The machine-readable aggregate is
[`readings/rust_zero_copy_seed_w5.json`](../readings/rust_zero_copy_seed_w5.json);
the retained pre-0.2.0 breakdown study is
[`rust-session-seed-breakdown.md`](rust-session-seed-breakdown.md). The
optional `seed_digest_overlap` runtime switch and its measured effect are
described in the [0.2.0 release notes](releases/v0.2.0.md).

## Correctness gates behind these numbers

- Every timed lane verifies its token output against the pinned
  reference (HF `tokenizers` 0.22.2) — per sample in F1/F4, per session
  in F2, warmup batch + spot checks in the throughput pair.
- Historical Fastokens 0.2.0 observation (not a runtime certificate and not an
  exact-ID guarantee for the optional Fastokens 0.3.1 adapter): 1,000,000 real documents x
  {hy3, kimi_k3, ling_3_0_flash, laguna_s_2_1}, full token-id list
  equality against the reference: kimi_k3: 1,000,000 docs, 0 mismatches; ling_3_0_flash: 1,000,000 docs, 0 mismatches; laguna_s_2_1: 1,000,000 docs, 0 mismatches; hy3: 1,000,000 docs, 0 mismatches, 3 documents rejected with a loud error (rare characters; wrong ids are never produced).
