# Roadmap

What is planned, and what is deliberately outside the first release. Items are
grouped by theme rather than by date; nothing here is a commitment to a
schedule.

## Native request path: completed core, thinner integrations next

The one-call request pipeline is now implemented behind the existing Python
API. Construction-time Python validates artifacts, registry entries, policy,
and device facts, then projects the immutable admitted route into Rust. Encode,
batch, named-session, and content-reuse operations cross PyO3 once with the GIL
released. Native routing owns the guards, fallback ledger, Hugging Face
reference, corrected Gigatoken full/repair engine, content index, persistence,
and prebuilt CUDA Driver API dispatch. The exact boundary and evidence are in
[`docs/native-routing.md`](docs/native-routing.md).

Two of the planned integrations shipped in 0.2.0:

- **Done (0.2.0)** — a stable Rust/server-facing adapter: the public `toktier`
  crate consumes native token buffers directly, with no Python
  `tuple[int, ...]` allocation on the request path
  ([`docs/rust-api.md`](docs/rust-api.md));
- **Done (0.2.0)** — artifact acquisition and verified-cache management behind
  a native service boundary, including air-gap export and import, for
  deployments that need a Python-free lifecycle
  ([`docs/rust-lifecycle.md`](docs/rust-lifecycle.md)).

Still open:

- offer an optional borrowed-buffer or array result without weakening the
  current immutable Python `Encoding` contract;
- keep diagnostics generated from one schema so future native consumers cannot
  drift from the Python `explain()` shape.

The frozen policy enum, reason codes, certificate checks, downward-only
fallback chain, store/TKFR formats, and exact-ID contract remain unchanged.

## Kernel distribution: finish the JIT host

The prebuilt multi-architecture fatbin is now hosted in Rust through the CUDA
Driver API: image selection, digest verification, memory, streams, launches,
document offsets, and ordered fallback do not use Python or PyTorch on the
request path. Since 0.2.0 the Rust crate also performs JIT delivery natively:
the compiler toolchain is selected explicitly, the compiler and runtime tuple
is recorded in the runtime-builds registry, and uncertified builds are refused
unless explicitly opted in. The Python `[gpu-jit]` extra still uses
`torch.utils.cpp_extension` and its legacy host; moving it onto the same
native delivery must preserve two rules:

- the loader selects an image **explicitly**, from a manifest, and matches its
  digest against the certificate; leaving the choice to the driver would break
  the binding between evidence and the code that ran. Architectures without
  certification evidence remain available under `EXPERIMENTAL` and otherwise
  route to the reference path;
- the list of shipped architectures is therefore short by construction: it
  contains the devices on which verdicts were actually run.

One question the legacy host leaves open. A process that has loaded one
kernel delivery and is then asked for the other is refused, and the GPU lane
records that refusal as a fallback and answers exactly on the CPU lane, with
the certificate still describing the delivery that is loaded and that ran
(`docs/gpu-jit.md` Section 2). The alternative -- letting the refusal reach
the caller as a raised `KernelIncompatible` and voiding the process
certificate outright -- is louder, and would tell a caller that asked for a
specific delivery that it did not get one. Which of the two a serving process
wants is not obvious, and the answer belongs with the move onto the native
delivery rather than ahead of it.

Related: an optional adapter for PyTorch tensors across a stable ABI.
Free-threaded CPython remains out of scope while the package ships abi3 wheels.

## Session store: a server mode

The store in the first release is a local file owned exclusively by one
process's Rust layer. Sharing a SQLite file across machines is not supported
and will not be: the direction for multi-machine deployments is a store server
that speaks the same record format and keeps the same two invariants — a wrong
key misses, and a hit is verified — so that a client cannot be given a prefix
it did not ask for.

A related item on the same path: the Python `Session.revision` is monotone
within one process only, and a session re-read in a later process begins its
count again at zero (`docs/contracts/facade.md` Section 7). The conversation
itself is durable; that counter is not.

Earlier releases said carrying it across processes would be a stored-format
change. That was wrong, and the correction is worth stating plainly: the
record has carried `session_revision` at offset 56 since format v1, the value
is written and read on the Python face as well, and the Rust
`Session::revision()` already reports it after a restart. What is missing is
a path from that field to the Python property, which on the default request
path does not consult the store at all. No format version moves, so no stored
session is invalidated. The work is still deliberately not part of 0.2.5.

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

## CPU parallel BPE for long requests

Long requests on the certified CPU path are encoded serially today. The
planned direction is chunk-parallel BPE: split a long input into chunks,
encode the chunks on multiple cores, and repair the chunk boundaries so that
the merged result is byte-for-byte the serial encoding. The exact-ID contract
shapes the design: boundary repair has to be verified, an input that fails a
safety check falls back to the serial path, and the feature ships only behind
the same differential evidence as the engine it accelerates.

## Routing large session deltas to the GPU

Session appends run on the CPU repair lane regardless of the delta's size;
the size threshold in the admitted route applies to whole encode requests
only. For a large appended delta, most of the work is ordinary encoding and
only the seam against the sealed prefix needs repair. The planned split:
encode the body of the delta on the accelerated backend when the route admits
one, repair the boundary on the CPU lane, and keep the result byte-for-byte
identical to the serial path. The admission rules apply unchanged — without
certification evidence for the accelerated backend, the delta stays on the
CPU lane.

## Dependency-graph judgement: narrowed to the compiled closure (done in 0.2.4)

Since 0.2.4 an accelerated route also asks that the dependency graph of the
build be the graph the certification campaign judged (`docs/rust-api.md`).
Taking the whole lockfile closure as the judged side would have been the
easier reading and the wrong one: a lockfile's dependency lists are the union
over features and targets, so a consumer who resolved a graph would be refused
over transitive versions that never entered the compiled artifact -- a
WebAssembly binding on a Linux build, a Windows shim, an optional dependency
no feature enabled. Such refusals are safe and exact and ask the reader to
align packages that have no bearing on what ran.

0.2.4 judges the set Cargo compiled for the certified build, taken from
Cargo's own account of that build and shipped with the crate. On this
workspace that is 165 packages where the reachable set is 227, and a fresh
consumer resolution diverges in six packages rather than fourteen -- each of
which really is compiled, so each command that aligns it changes the bytes
that run.

Two things this did not do, both written up where the answers are read
(`docs/rust-api.md`). It does not make a fresh resolution certified: six
packages still drift, because exact versions age. And it gives up one
incidental check -- a consumer whose own feature unification pulls an extra
package into a judged dependency is no longer reported -- which is structural
rather than pending, since a consumer's build script cannot compute its own
feature resolution. An earlier draft of this section offered "a build-script
helper for another target" as an example of what falls outside; measurement
says otherwise, since `cc` declares `find-msvc-tools` unconditionally and
Cargo compiles it on Linux too. It stays inside the judgement.

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
