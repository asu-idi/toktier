# Rust-native lifecycle, JIT, serving, and distribution

TokTier's Rust surface can acquire and verify tokenizer artifacts, move them
through an air gap, compile a CUDA image, and serve concurrent work without a
Python runtime. All routes consume the same embedded manifests, registry,
class tables, corrected CPU engine, state-store format, and CUDA Driver host as
the Python facade.

## Artifact acquisition and cache policy

Remote construction always names an immutable lowercase 40-hex revision:

```rust,no_run
use toktier::{ArtifactManager, Device, Revision, Runtime};

let manager = ArtifactManager::builder()
    .cache("/var/cache/toktier/artifacts")
    .build()?;
let runtime = Runtime::builder()
    .artifacts(manager)
    .device(Device::Auto)
    .build()?;
let tokenizer = runtime.from_pretrained(
    "Qwen/Qwen3-8B",
    Revision::commit("b968826d9c46dd6066d109eabc6255188de91218")?,
)?;
# let _ = tokenizer;
# Ok::<(), toktier::Error>(())
```

The manager streams each response into a mode-0600 temporary file, enforces
the manifest size while reading, hashes the bytes, fsyncs the file, and renames
it while holding a family/revision inter-process lock. Cache directories are
mode 0700. Existing corrupt members are quarantined; a verified handle is not
published from a partial directory.

### Families this crate cannot download

One family's certified artifact is not published anywhere: `kimi_k3` is a
conversion of the `tiktoken.model` its upstream repository does carry
(`docs/support-matrix.md`), produced on the installing machine. This crate
runs no conversion, so asking it to acquire such a family is refused with
`ARTIFACT_NOT_FOUND` naming the conversion, its pinned upstream inputs, and
where the bytes do come from -- before any request is made, rather than as a
404 from the hub. The shipped conversion table is the single place a derived
family is named, and both faces read the same bytes of it.

Everything after acquisition is unaffected: produce the artifact once with
the Python package (`toktier artifacts fetch kimi_k3`) or receive it as an
air-gap bundle, and this crate verifies and runs those bytes like any other
-- an artifact cache holding them is used as it stands, `import_bundle`
installs one, and `Runtime::load_local` opens an explicit directory.

### Acquisition is behind an opt-in feature

Network acquisition lives behind the `network` feature, which is **not** in
the default set from 0.2.5 on. Add it when a build has to fetch:

```toml
toktier = { version = "0.2.5", features = ["network"] }
```

Everything else in this document works without it: a verified artifact
cache, `ArtifactSource::LocalDirectory`, `verify`, `inspect`, air-gap
`import_bundle` and export over cached bytes, and `Runtime::load_local`.
A build without the feature that reaches an uncached remote object answers
`NETWORK_DISABLED` and names the feature; `Runtime::doctor()` reports
`network_compiled` so a control plane can see the answer in advance. The
feature pulls sixteen packages, including the TLS stack, into the compiled
closure, so builds that never fetch do not carry them.

The network implementation itself is unchanged, and uses this explicit
policy:

| Axis | Contract |
|---|---|
| Revision | immutable 40-hex commit; branches and tags are rejected |
| TLS | HTTPS only; plaintext is available solely for explicit loopback tests |
| Timeout | 300 seconds for the complete request by default |
| Retry | two transport retries after the first request, with bounded exponential backoff; digest, size, registry, and auth failures are not retried |
| Redirect | at most five without credentials; zero when a secret provider is configured, so bearer material is never forwarded |
| Proxy | disabled in the native client; configure an explicit HTTPS mirror instead |
| Resume/range | not used in format v1; every retry starts and revalidates the complete object |
| ETag | not trusted as identity; the shipped SHA-256 and size are authoritative |
| Stale temporary file | reclaimed after 24 hours by default while holding the family lock; configurable or disableable |

Authentication enters through `SecretProvider`; `EnvironmentToken` reads only
the named variable when constructing a request. Token values are redacted from
`Debug`, errors, paths, registries, marker files, and bundle manifests.

`offline(true)` gates acquisition before URL construction, DNS, or socket use.
Use `ArtifactSource::None` as an additional explicit statement that only the
verified cache or an imported bundle is allowed.

### Where the crate puts its directories

Since 0.2.4 the crate reads `TOKTIER_HOME` and the XDG variables with the
same precedence as the Python product (`docs/contracts/config.md`
Sections 4-5), so one environment places both layers:

| Directory | Explicit setting | Most specific variable | Roots, in order | Final fallback |
|---|---|---|---|---|
| Artifact cache | `RuntimeBuilder::artifact_cache()` / `ArtifactManager::builder().cache()` | `TOKTIER_ARTIFACT_CACHE` | `$TOKTIER_HOME/cache`, `$XDG_CACHE_HOME/toktier`, `$HOME/.cache/toktier` | `./.toktier/artifacts` |
| Direct-JIT cache (`jit` feature) | the JIT builder's cache setting | `TOKTIER_JIT_CACHE` | the same three roots | `./.toktier/jit-rust` |
| Session state | `RuntimeBuilder::home()` | none | `$TOKTIER_HOME/state`, `$XDG_STATE_HOME/toktier`, `$HOME/.local/state/toktier` | none: persistent sessions with no resolvable home are refused |

Notes that matter when running the shipped examples:

- An empty variable counts as unset, which is what the XDG specification
  says and what the Python product does.
- The crate's own tests are not configured by this table. They locate a
  host's existing artifacts through `TOKTIER_TEST_ARTIFACTS`, falling
  back to `$HOME/.cache/toktier/artifacts`, because they are looking for
  where the artifacts actually are rather than where this crate would
  place them. `TOKTIER_HOME` therefore does not move that fixture path,
  which is deliberate; `TOKTIER_TEST_ARTIFACTS` does.
- The leaf names stay the crate's own: artifacts land in `artifacts` (the
  same leaf the Python product uses) and JIT products in `jit-rust`, which
  is deliberately distinct from the Python `kernels` directory because the
  two hold different products.
- Session state is not a cache. When no home can be resolved from any of
  the three roots, persistent sessions are a configuration error rather
  than a directory placed by guesswork; in-memory sessions are unaffected.
- `RuntimeBuilder::home()` governs session state, not the artifact cache.
  The two are still set independently.

## Mirrors and air-gap bundles

`ArtifactManager::mirror` writes the familiar
`<repo>/resolve/<revision>/<file>` tree. `ArtifactSource::LocalDirectory`
accepts that tree and the native content-cache layout. Export carries the
artifact plus the exact registries, schemas, tables, CUDA source/fatbin, and
provenance embedded in the process.

The archive is deterministic and compatible with the Python v1 importer. Its
domain-separated root digest binds every declared path, byte count, and SHA-256.
Import rejects absolute/traversing paths, duplicates, links, special files,
undeclared files, oversized archives, and digest changes. It verifies into a
private staging tree, fsyncs every directory from leaves to root, and publishes
the alias with one rename. Re-import is idempotent only when the visible tree
still authenticates exactly.

The Rust-only CLI mirrors these calls:

```text
toktier-rust artifacts fetch qwen3_8b
toktier-rust artifacts verify qwen3_8b --offline
toktier-rust artifacts mirror qwen3_8b --out /srv/toktier-mirror
toktier-rust artifacts export qwen3_8b --out qwen3_8b.tar
toktier-rust artifacts import qwen3_8b.tar --offline
```

## Direct native JIT

Enable `jit` to invoke the selected absolute `nvcc` executable directly with
an argument vector. No shell, Python, PyTorch, Ninja, or callback participates.
The compiler runs with a cleared environment, fixed locale/PATH, null stdin,
private source and temporary files, bounded output/product size, and a wall
timeout. The cache key binds:

- both CUDA source files and normalized compiler arguments;
- the compiler's canonical path, complete file SHA-256, release, and build;
- target architecture and CUDA driver API version;
- family, authenticated tokenizer SHA-256, frozen oracle, class-table identity,
  and both artifact-campaign and direct-JIT evidence IDs;
- Rust CUDA-host source, rustc, release flags, and certification result.

The product and manifest are reopened and authenticated before their directory
is atomically published. Every binding-input field is recomputed into the
binding digest on reuse, and the compiled fatbin bytes are separately
re-authenticated against their recorded SHA-256; changing metadata or fatbin
bytes quarantines the entry.

An exact compiler-binary/architecture tuple must appear in the registry for
automatic **certified** compilation. Since 0.2.6 a tuple that is merely
unjudged is a different case from one that fails to verify. Under the
default `Policy::Supported` the compile proceeds, the product is cached
and reused like any other, and its manifest carries
`assurance: supported_untested`; under `Policy::Certified` the older
behaviour is what happens, and the error prints the explicit opt-in. A
source, host or flags mismatch, and a world-writable compiler component,
refuse under every policy unless the caller accepts them.

Experimental use is deliberately local to one builder and remains labelled
after successful comparison:

```text
toktier-rust gpu compile qwen3_8b --accept-uncertified-jit
```

Application code must also select `Policy::Experimental`; the waiver is stored
in the binding manifest but never promoted into certification. A
`supported_untested` product needs no such selection, and
`toktier-rust verify-local --engine gpu --family <family>` is how its route
is compared with this binary's reference engine here.

## Serving policy

`ServingPool` is an optional executor-neutral layer over the synchronous core.
It bounds count/bytes/rows/window/worker/session pressure, batches only rows
with the same tokenizer, device, and options, and returns zero-copy row views.
Multiple verified tokenizer handles for the same canonical artifact can be
registered as explicit device slots. Stateless requests use observable
round-robin selection; a stateful session remains pinned to one slot.

`DeviceFailurePolicy::FallbackOnly` retains the tokenizer's own ordered
GPU-to-corrected-CPU-to-HF chain. `RetryEligiblePeer` may retry another handle
for the identical artifact after that chain returns an execution error.
Each stateful session also has a bounded FIFO ticket sequence, so worker
scheduling cannot reorder its seed/appends; rejected and cancelled tickets are
advanced without touching session state. Cancellation and deadline errors
distinguish skipped work from already-started work that may still have
committed. Shutdown stops acceptance, drains accepted requests, and joins
workers. `worker_threads` is a hard caller-selected bound; adding device
handles never raises it implicitly.

## Distribution contract

- MSRV: Rust 1.93.1, verified against the complete feature matrix. The bound
  preserves the frozen corrected-CPU implementation, including its stable
  AVX-512 intrinsics, rather than rewriting certified SIMD code solely to
  claim an older compiler.
- Supported release target: Linux x86-64, glibc 2.34 or newer. CPU-only feature
  builds do not require a CUDA installation; GPU features open `libcuda` at
  runtime.
- Default features: `sqlite`, `prebuilt-gpu`, and `serving`. Through 0.2.4
  `network` was in this list; from 0.2.5 it is opt-in.
- Optional features: `network`, `jit`, and `serde`; all supported feature
  combinations are compiled in the release matrix.
- The crate ships one binary, `toktier-rust`, the Python-free lifecycle CLI
  used throughout this document. It is built with the crate's default
  features, so `cargo install --locked toktier` installs a CLI that cannot
  fetch: `artifacts fetch` on an uncached family prints `NETWORK_DISABLED`
  and exits 2, while `verify`, `inspect`, `import`, `export`, `doctor`, and
  encoding over a verified cache all work. Use `cargo install --locked
  --features network toktier` for a CLI that acquires artifacts. Either way
  the install gives a working CLI on the reference engine: an installed package
  is built in a temporary directory with no lockfile above it, so the
  dependency-closure check answers `unlocated` and the build is not certified
  for an accelerated route. Certification for a Rust consumer is earned in a
  workspace whose own lockfile can be located and whose certified core
  resolves the judged packages; packages outside that core are reported as an
  advisory instead of withholding the route. `doctor` prints the commands that
  take a workspace there, and `docs/rust-api.md` describes the check in full.
- The crate is published on crates.io from 0.2.0 onward and carries the
  package version (axis 1 of `docs/contracts/versioning.md`); the earlier
  `0.0.1` number was a source/workspace preview. Patch versions preserve the
  public API and pre-1.0 minor versions may make documented breaking changes.
  Publication is a release step of its own, separate from building or testing
  artifacts.

`tools/build_rust_source_archive.py` creates a deterministic, fully vendored
offline source archive with normalized tar/gzip metadata and a per-file/root
digest manifest. A clean `CARGO_HOME` can compile it with Cargo network access
disabled. Run Cargo from the extracted archive root so it discovers the
shipped `.cargo/config.toml`; using only `--manifest-path` from another working
directory does not activate that vendoring configuration.
`tools/generate_rust_distribution_metadata.py` independently emits
a CycloneDX 1.5 SBOM and a dependency-license bundle for the non-development
Rust closure; these are embedded in the Rust package data and air-gap export.

The public crate and Python wheel copy the data they carry from one checked
source set through `tools/sync_rust_package_data.py`; drift fails the release
check. Both surfaces carry the same runtime registry, the same routing,
kernel, repair and artifact tables, the same kernel sources and fatbin --
twenty files, byte-identical on each side -- and the same four legal
documents, which the wheel keeps under `.dist-info/licenses/`. Two kinds of
file ride in the crate alone: the four JSON schemas, which are inputs to the
maintainer tools rather than to either runtime, and the Gigatoken patch,
which records how the engine was built rather than anything either runtime
reads.

Crates.io publication is not performed by these tools. The workspace currently
uses exact-version path dependencies for TokTier's internal crates; the
vendored source archive is the independently replayable distribution until a
separate release authorizes publishing that dependency set.

## Non-Rust boundary decision

A C ABI is intentionally deferred until the Rust API has a released stability
baseline. Freezing opaque-handle, buffer-ownership, error-code, and unwind
rules while the lifecycle types are still preview APIs would create a second
compatibility contract without helping Rust-native serving. A future ABI, if
approved, will use opaque handles and explicit ownership/free functions and
will never expose Rust layout or permit unwinding across the boundary.
