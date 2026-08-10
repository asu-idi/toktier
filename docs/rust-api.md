# Rust serving API

TokTier's `toktier` crate is a synchronous, Python-free serving surface over
the same Rust reference, corrected-CPU, prebuilt-CUDA, routing, and state-store
implementation used by the product. It is designed for a Rust frontend that
retains token buffers between agent turns.

The crate is published on crates.io from 0.2.0 onward and tracks the package
version, so `cargo add toktier` resolves it from the registry. The published
package carries the same data as the wheel, the multi-architecture fatbin
included, and its default features need no CUDA toolchain at build time.

## Construction and artifact trust

```rust,no_run
use toktier::{Device, GpuDelivery, Policy, Runtime};

let runtime = Runtime::builder()
    .home("/var/lib/toktier")
    .artifact_cache("/var/cache/toktier/artifacts")
    .policy(Policy::Certified)
    .device(Device::Auto)
    .gpu_delivery(GpuDelivery::Prebuilt)
    .build()?;
let tokenizer = runtime.load("qwen3_8b")?;
# Ok::<(), toktier::Error>(())
```

`build()` validates configuration and the embedded registries. `load()` opens
the manifest-pinned `tokenizer.json` as a regular, non-symlink file and checks
its size and SHA-256 before retaining the verified bytes. A sibling repository
can be selected with `load_repository(repo_id, revision)` only when the exact
immutable revision is in the shipped identity/equivalence table; execution is
then bound to the verified canonical artifact.

The default cache entry is
`<cache>/<family>-<revision-prefix>/tokenizer.json`. `ArtifactManager` can
acquire an immutable 40-hex revision over TLS, use an HTTPS mirror, consume a
local mirror, or enforce a network-silent offline mode. Every byte is streamed
through the recorded digest and size before an atomic cache publication. An
explicit verified directory remains available through `artifact_directory()`
or `load_local()`. See [Rust lifecycle and distribution](rust-lifecycle.md).

`Runtime::doctor()` returns typed build and CUDA-probe facts. Device probing
does not load a kernel. `Tokenizer::plan()` is the immutable admitted route;
every result carries `ExecutionFacts` naming the backend that actually ran.
Accelerated admission also requires the exact Rust facade source, rustc,
features, target, and release-profile facts to appear in the shipped
`runtime_builds` registry. A development or otherwise unregistered build falls
to HF under `Policy::Certified`; an explicit CUDA request returns
`UNCERTIFIED_RUNTIME`. Only `Policy::Experimental` can opt an unregistered
build into an accelerated candidate, and every result remains labelled
experimental.

## Buffers, batches, and decode

```rust,no_run
# use toktier::Runtime;
# let tok = Runtime::builder().build()?.load("qwen3_8b")?;
let encoded = tok.encode("hello")?;
let token_count = encoded.ids().len(); // borrowing is allocation-free
let shared = encoded.into_token_buffer();
let decoded = tok.decode(&shared, Default::default())?;

let batch = tok.encode_batch(&["first", "second"])?;
let all_values: &[u32] = batch.values();
let row_boundaries: &[u64] = batch.offsets();
let second: &[u32] = batch.row(1)?;
# let _ = (token_count, decoded, all_values, row_boundaries, second);
# Ok::<(), toktier::Error>(())
```

`TokenBuffer` is immutable shared ownership of one continuous `u32`
allocation. Borrowing it is allocation-free; `into_vec()` is an explicit full
copy. Constructing a buffer from a `Vec<u32>` adopts that allocation without
copying its elements. `RaggedEncoding` owns one values allocation and one
offsets allocation, not a public `Vec<Vec<u32>>`. Offsets for one-shot
encoding are computed only when `EncodeOptions::offsets` is requested.

## Delta-native sessions

BPE repair may replace old tail tokens, so a session append returns a
`TokenPatch`, not merely “new IDs”:

```rust,no_run
# use toktier::Runtime;
# let tok = Runtime::builder().build()?.load("qwen3_8b")?;
let mut session = tok.open_session("agent-42")?;
let seed = session.seed("user: hello\n")?;
let mut serving_ids = seed.ids().to_vec();

let patch = session.append("assistant: hello!\n")?;
serving_ids.truncate(patch.keep_tokens() as usize);
serving_ids.extend_from_slice(patch.replacement_ids());
assert_eq!(serving_ids.len() as u64, patch.token_count());
# Ok::<(), toktier::Error>(())
```

An append returns only a suffix patch and never materializes the sealed
prefix; `snapshot()` does so explicitly, and repeated snapshots of an
unchanged session return one shared immutable row (cached under the session's
mutation generation, so an append can never expose a stale row). The
`Encoding` returned by `seed()` shares the session store's own adopted ID
allocation -- on an accelerated seed that is the engine's result buffer
itself -- rather than a copy, and it remains valid and unchanged after later
appends, `close()`, or dropping the runtime. Exact copy boundaries that
remain by design: `seed(&str)` copies the borrowed input text once into
session state; append validation copies the current mutable-tail ID row, and
the hard-cap and reference fallback paths re-encode the whole mutable tail
plus the delta (before any prefix has been sealed, the mutable tail is the
complete history); durable (SQLite) mode additionally retains a recovery
transcript and serializes the complete row as part of the store record. `encode_transcript()` is the migration
operation for callers that still send the complete transcript. It proves the
stored prefix with the native content digest before repairing only the
suffix.

`RuntimeBuilder::seed_digest_overlap(true)` lets a seed run its content-digest
scan (the recovery-index endpoint plus geometric checkpoints) on the
process-wide bounded worker pool while the seed encode runs on the calling
thread; both results join before validation and the atomic session insertion.
The digest bytes, validation ordering, error codes, and failure atomicity are
identical to the default serial path -- the option changes only wall-clock
scheduling, hiding the scan under the encode when a pool worker is free. The
pool is the same bounded Rayon pool the batch encode path shares (sized by
`RAYON_NUM_THREADS` or the CPU count); no thread is spawned per request.
Recovery hashing and durable serialization stay serial in the durable tier.
The default is off.

A `Session` is non-cloneable and mutation requires `&mut`, which enforces one
writer per handle. The runtime also leases stable names and checks revisions.
`fork()` shares content-addressed sealed nodes while starting revision zero;
`close()` releases the writer lease and `delete()` removes the state.

With `home()`, named sessions are SQLite-backed. Store-v1 deliberately omits
stable-prefix plaintext, so the durable Rust owner stores the transcript and
canonical TKFR-v1 binding in the same transaction as the record and sidecar.
On restart it restores delta state only after record hash, byte length, text
digest, bounded tail, and content checkpoints all match. Persistent stores
therefore contain transcript plaintext and must be protected as application
data.

## Bounded concurrent serving

The optional executor-neutral serving adapter implements both `Future` and a
blocking wait without requiring Tokio:

```rust,no_run
# use toktier::{Runtime, ServingPool};
# let tok = Runtime::builder().build()?.load("qwen3_8b")?;
let pool = ServingPool::builder(tok).build()?;
let pending = pool.submit("independent request")?;
let response = pending.wait()?;
println!("queue={:?} engine={:?}", response.timings.queue, response.timings.engine);
pool.shutdown();
# Ok::<(), toktier::Error>(())
```

The queue bounds request count, UTF-8 bytes, batch rows, batch bytes, worker
threads, and per-session in-flight work. Compatible stateless rows share one
ragged allocation; each returned row is a zero-copy view. Stateful sessions
remain pinned to one observable device index. A per-session FIFO ticket
preserves submission order even when several pool workers dequeue that session
concurrently; rejected tickets are advanced so later appends cannot stall.
Cancellation before execution skips work; cancellation after execution begins
does not claim that a CUDA launch or committed SQLite transaction was rolled
back. Timings report queue, engine, durable-store, and result-view
materialization separately.

## Threading, errors, and features

`Runtime`, `Tokenizer`, `TokenBuffer`, and `Encoding` are `Send + Sync`.
Independent sessions can move between worker threads. A single `Session` can
move but has one mutable owner; duplicate processes still meet the store's
revision and integrity gates.

Errors expose a stable `ErrorCode`; display messages are diagnostic and are
not a compatibility interface. `ArtifactHashMismatch` is reserved for actual
content-hash verification failures; a hash-verified artifact that later fails
to parse reports the failing stage instead (`KernelIncompatible` from GPU
table construction, `Internal` from a reference-engine load). The `serde`
feature serializes typed plans, execution facts, doctor facts, and store
statistics.

The current feature surface is:

| Feature | Default | Effect |
|---|---:|---|
| `sqlite` | yes | durable named sessions and restart recovery |
| `prebuilt-gpu` | yes | embedded, digest-bound CUDA fatbin delivery |
| `network` | yes | TLS artifact acquisition with pinned revisions |
| `serving` | yes | bounded executor-neutral queue, batching, and device policy |
| `jit` | no | direct shell-free NVCC compilation into the Rust CUDA host |
| `serde` | no | serialization for public diagnostic records |

The normal dependency graph contains no PyO3, Python, PyTorch, or mandatory
async executor. The prebuilt GPU host dynamically uses the CUDA Driver API and
does not link a system CUDA runtime.

## Runnable verification

Run the examples with `--release`. A development (dev-profile) build is
not present in the shipped runtime-build registry, so certified
accelerated routes are not admitted in it: a debug run of the CPU
example silently routes `[HuggingFace]` only, and the GPU examples
refuse with `UncertifiedRuntime`.

- `cargo run --release -p toktier --example cpu`
- `cargo run --release -p toktier --example batch`
- `cargo run --release -p toktier --example session`
- `cargo run --release -p toktier --example persistent_session`
- `cargo run --release -p toktier --example content_lookup`
- `cargo run --release -p toktier --example prebuilt_gpu`
- `cargo run --release -p toktier --all-features --example native_jit`
- `cargo run --release -p toktier --example correctness_campaign`
- `cargo run --release -p toktier --example rust_api_bench`

The `toktier-rust` binary exposes `doctor`, artifact lifecycle operations, and
direct JIT compilation without importing Python. Run `toktier-rust` without
arguments for its compact command reference.

The correctness campaign constructs separate certified and reference runtimes,
checks one-shot/decode/adversarial inputs, reconstructs complete session output
from patches, and requires exact integer equality. The benchmark labels full
encode and patch-only timings separately and states that no snapshot was
requested. The 31-process decomposition of cold 4 MiB session creation is in
[Rust `Session::seed` latency breakdown](rust-session-seed-breakdown.md); it
keeps ID-only GPU time, HF span reconstruction, state construction, and
durability costs separate.
