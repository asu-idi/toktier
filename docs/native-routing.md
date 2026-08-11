# Native request path

TokTier 0.1.1 first moved the UTF-8 crossover and added-token prefilter into
Rust. The current source tree completes the larger request-path migration: an
eligible encode, batch, named-session request, or content lookup enters one
private PyO3 method, releases the GIL, and returns only after native routing,
backend execution, fallback accounting, store work, and persistence finish.
The public Python API and frozen routing contract are unchanged.

## Construction boundary

Python still owns installation, configuration parsing, artifact acquisition,
construction-time probing, the public immutable `RoutePlan`, object lifetime,
and conversion into `Encoding` and `explain()` shapes. Construction validates
the registry and projects only the already-admitted route into an immutable
Rust runtime. Registry parsing is not repeated on each request.

The runtime contains:

- the ordered backend chain and the configured GPU crossover (64 KiB by
  default, via `gpu_min_bytes`);
- a native Hugging Face `tokenizers` reference loaded from the verified
  artifact;
- the corrected Gigatoken full-encode and bounded-repair implementation,
  compiled into the same `toktier._native` extension when its source/build
  certificate is admitted;
- a Rust CUDA Driver API host for the digest-checked prebuilt fatbin, when that
  delivery and device are admitted. Its domain-separated source digest, exact
  rustc, and release build facts are checked independently of the fatbin;
- the exact added-token guard, execution ledger, native content index, session
  names, store-format-v1 state, and optional TKFR-v1 recovery state.

Construction reuses the already parsed native HF reference. For the common
case where the verified `tokenizer.json` carries every added token, the CPU
wrapper binds those same bytes and that same reference instead of importing
`transformers` or parsing a second HF object. A configuration-only added token
keeps the slower live-materialization path. The single corrected-Gigatoken
core is initialized and verified at runtime construction so an agent's first
append does not inherit setup latency. Batch worker caches remain lazy and grow
only to the parallelism justified by the observed payload.

If construction cannot reproduce that complete native route, the facade keeps
the compatibility executor. In particular, local GPU JIT still uses the
Python/PyTorch host. This guard changes implementation placement, never
eligibility or fallback order.

## One-call execution

For an admitted runtime, each public operation has one Python-to-Rust crossing:

1. PyO3 borrows the Python string's cached UTF-8 view and releases the GIL.
2. Rust applies the immutable threshold and added-token guards.
3. It runs the selected native engine. A failure moves only downward along the
   frozen chain and records the same reason code.
4. Named sessions and automatic content reuse perform lookup, seed, repair,
   revision checking, content-index maintenance, and persistence under the
   same native operation.
5. Rust returns final IDs and compact diagnostics; Python materializes the
   documented public result once.

The native HF route includes normalization, added tokens, pre-tokenization,
model encoding, post-processing, decoding, and offsets. The CPU route owns the
patched Gigatoken BPE and O/S/L/N/M repair table directly; there is no second
extension and no Python callback. The prebuilt GPU route owns CUDA context,
stream, module, buffers, explicit image selection, document offsets, launch,
and result splitting through the Driver API. A public GPU batch is one native
batch launch sequence, not a Python loop over documents.

When an accelerated result seeds stored state, its ID row is closure-verified
against the shared verified HF reference vocabulary and adopted together with
sparse span checkpoints; token spans are rebuilt per window on demand, while
the materialized compatibility route still reconstructs full spans. This
payload reports `accelerated_with_lazy_span_checkpoints` in the facade's
``state_encode`` diagnostics. The materialized route retains the published
`accelerated_with_reconstructed_spans` value. This bridge does not initialize
Gigatoken; a failed premise routes the seed through native HF before anything
is committed.

## Recovery and content lookup

The native entry store preserves the existing identities and bytes:

- personalized BLAKE2b-128 content endpoints and geometric 4 KiB marks;
- incremental domain-separated SHA-256 and exact UTF-8 length for recovery;
- store format v1 records and TKFR-v1 bindings;
- record-hash, checksum, tail, length, checkpoint, and fingerprint gates;
- optimistic revision conflicts and downward-only cold fallback.

A resident append updates both digest accumulators from the delta only. It does
not rescan the transcript or recompute a checkpoint row. A restart may scan the
caller-presented historical prefix once to prove recovery; subsequent resident
appends are delta-only. In-memory runtimes do not allocate recovery state.

## Evidence and diagnostics

The source-certified CPU rerun is recorded in
[`readings/fast_cpu_native_frontend_parity.json`](../readings/fast_cpu_native_frontend_parity.json).
It checks all 11 unique CPU-fast artifacts through the executing native front
end, including exact HF IDs, bounded repair, one call per batch, and GIL
release. GPU hardware readings bind the rebuilt fatbin, judged architecture,
and executing Rust-host identity in the support registry. Unit and conformance
suites additionally exercise
added tokens, Unicode, unusual inputs, ordered faults, persistent reopen,
corruption, and diagnostic parity.

`explain()` exposes `request_path`, Python-to-native call counts, execution and
fallback counters, append paths, native store counters, and any construction
guard. These are compatibility diagnostics: public policy, status vocabulary,
reason codes, and error codes did not change.

## Performance accounting

The Phase-1 routing microprofile remains useful but deliberately narrow. On an
Intel Core Ultra 9 285K / Python 3.12.3 host, the isolated UTF-8 threshold and
necessary-condition gate for a 4,000,000-byte input changed from 2,969.548 us
to 51.634 us (57.51x). That is control-plane time, not tokenization throughput
and not the cost of materializing a million-token Python tuple.

Phase-2 profiles report native engine/store time separately from Python result
materialization. This separation matters: the internal route can be entirely
Rust while the ergonomic Python API still pays once to allocate its documented
`tuple[int, ...]`. A Rust frontend can skip that allocation entirely: the
`toktier` crate shipped in 0.2.0 consumes the same native buffers directly,
without changing TokTier's tokenization semantics
([`rust-api.md`](rust-api.md)).

## Remaining native work

The prebuilt path is native; the Python facade's local GPU JIT compilation and
hosting are not -- that delivery keeps the Python/PyTorch host. The Rust crate's
own `jit` feature compiles through the selected `nvcc` directly
([`rust-lifecycle.md`](rust-lifecycle.md)), so the two surfaces differ here.

Shipped since 0.2.0, and therefore no longer open work: the Rust serving
facade with `TokenBuffer`/`RaggedEncoding` borrowing and zero-copy pool row
views, and native artifact acquisition, mirroring, and air-gap bundles. What
remains open is a Python-side zero-copy result view (the facade still
materializes a `tuple[int, ...]`), a non-Rust C ABI (deliberately deferred, see
[`rust-api.md`](rust-api.md) and the lifecycle document's boundary decision),
and moving the facade's JIT delivery onto the native compiler. These are
performance or integration work, not prerequisites for the one-call Python
request path.
