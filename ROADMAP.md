# Roadmap

What is planned, and what is deliberately outside the first release. Items are
grouped by theme rather than by date; nothing here is a commitment to a
schedule.

## Native request path: Rust routing behind a thin Python binding

Version 0.1.1 begins this migration: the UTF-8 size crossover and the
added-token necessary-condition gate now run together in the pure-Rust
`toktier-routing-core` crate, and the session store evaluates the certified BPE
seal predicate there without materializing token/span lists in Python. The
measured scope and remaining boundary are documented in
[`docs/native-routing.md`](docs/native-routing.md).

Artifact and registry views, probe/plan construction, fallback invocation and
accounting, facade orchestration, and part of the content-prefix index still
live in Python. The persistent token-state store and corrected Gigatoken engine
are Rust, while GPU BPE runs in CUDA, so a request can still cross the
Python/Rust boundary several times before its ids are returned.

The direction is one native request pipeline behind the existing Python API:

- each encode, batch, session lookup, or strict append enters through one PyO3
  call; the binding releases the GIL and does not call back into Python on an
  eligible fast path;
- registry projection, immutable route planning, the 64 KiB crossover,
  added-token/repair guards, fallback execution, counters, session naming,
  content-prefix lookup, and persistence orchestration move into Rust;
- corrected Gigatoken full BPE and repair run directly under that router. The
  reference fallback uses the pinned Hugging Face `tokenizers` Rust
  implementation rather than a Python callback, so correctness fallback does
  not leave the native pipeline;
- the prebuilt GPU route is dispatched by the Rust host layer through the CUDA
  Driver API. CUDA remains the device implementation, but Python and PyTorch
  are absent from request dispatch, buffer management, and fallback;
- Python remains the thin product surface for installation, configuration,
  artifact acquisition, object lifetime, and conversion of native diagnostics
  into the public `Encoding` / `explain()` shapes.

This is a latency project, not a semantic rewrite. The frozen policy enum,
reason codes, certificate checks, downward-only fallback chain, store format,
and exact-ID contract stay unchanged. Acceptance requires:

1. one Python-to-native crossing per public request and zero Python callbacks
   on store hits, certified CPU full encodes/repairs, and prebuilt GPU dispatch;
2. no GIL held while the native request pipeline performs lookup,
   tokenization, repair, or GPU work;
3. byte-for-byte diagnostic parity and fresh-HF id equality across the same
   routing, persistence, fault-injection, and 11-artifact differential suites;
4. separately reported p50/p95 control-plane and end-to-end latency showing a
   lower short-append/store-hit critical path without regressing large-request
   throughput.

## Kernel distribution: a native Driver API host layer

The first release already ships a verified multi-architecture fatbin through
the `gpu` extra and retains local compilation through `gpu-jit`. The prebuilt
loader currently uses Python `ctypes` around the CUDA Driver API and PyTorch for
tensors and streams; JIT uses `torch.utils.cpp_extension`.

The remaining direction is a Rust-owned Driver API host layer behind a stable
C ABI, with image selection, memory, streams and graph capture managed by the
library rather than Python or PyTorch. Two consequences shape the design:

- the loader selects an image **explicitly**, from a manifest, and matches its
  digest against the certificate; leaving the choice to the driver would break
  the binding between evidence and the code that ran. Architectures without
  certification evidence remain available under `EXPERIMENTAL` and otherwise
  route to the reference path;
- the list of shipped architectures is therefore short by construction: it
  contains the devices on which verdicts were actually run.

Related, and dependent on the above: an optional adapter for PyTorch tensors
across a stable ABI. Free-threaded CPython is out of scope while the package
ships abi3 wheels.

## Session store: a server mode

The store in the first release is a local file owned exclusively by one
process's Rust layer. Sharing a SQLite file across machines is not supported
and will not be: the direction for multi-machine deployments is a store server
that speaks the same record format and keeps the same two invariants — a wrong
key misses, and a hit is verified — so that a client cannot be given a prefix
it did not ask for.

## Platforms

The first release targets Linux x86_64, which is the platform the certification
evidence was produced on. macOS and Windows support is planned for the layers
that do not depend on CUDA (reference backend, routing, store); the accelerated
path stays tied to platforms where evidence exists.

## Additional CPU engines

The first release ships the corrected Gigatoken engine as its certified CPU
full-encode and session-repair path for 11 unique tokenizer artifacts covering
12 model families. Fastokens remains an explicit experimental comparison and
does not carry TokTier's exact-ID guarantee. Additional third-party CPU engines
may be admitted only behind per-input guards, exact artifact/engine bindings,
published differential evidence, and any coordinated disclosure their fixes
require; uncertainty continues to route to the HF reference implementation.

## Certification suite (second wave)

The harness that produces the evidence — corpus fleets, differential judges,
boundary-certificate checkers, adversarial batteries, replay campaigns — is a
substantial body of code and is planned as a separate, later release. The first
release ships the *evidence* in machine-readable form
(evidence/evidence_manifest.json) so that the claims can be
inspected and re-run against the released code before the harness itself is
public.

## Families under evaluation

Families that have been through a first structural reconnaissance and are
candidates for a certification campaign. Reconnaissance is not coverage; none of
these are in the support matrix yet.

| Family | Upstream repository | Note |
|---|---|---|
| Step 3.7 Flash | `stepfun-ai/Step-3.7-Flash` | pre-tokenizer, normalizer and decoder segments match an already certified splitter group; new vocabulary, merges and added-token table |
| LongCat 2.0 | `meituan-longcat/LongCat-2.0` | script-partitioned pre-splits; each stage is a plain character-class run, so an existing construction applies, but it is a new kernel group |

Requests for additional families are welcome as issues; what determines the
effort is the pre-tokenizer structure, not the vocabulary size.

## Interoperability

- Feeding pre-computed token IDs to inference servers that accept them (see
  [`docs/integration/dynamo.md`](docs/integration/dynamo.md)) works today and
  needs nothing from either side.
- A persistent layer that implements the tokenizer traits used by the NVIDIA
  Dynamo frontend, so that it can sit next to the existing in-process cache, is
  a candidate for collaboration rather than a unilateral plan; the same document
  describes the shape it would take.
