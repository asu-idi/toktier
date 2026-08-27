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

Without `artifact_cache()`, the cache root comes from `TOKTIER_ARTIFACT_CACHE`,
then `$TOKTIER_HOME/cache`, `$XDG_CACHE_HOME/toktier`, or
`$HOME/.cache/toktier` -- the same precedence the Python product uses, so one
environment places both layers. Persistent sessions with no `home()` resolve
`$TOKTIER_HOME/state`, `$XDG_STATE_HOME/toktier`, or
`$HOME/.local/state/toktier`, and are refused outright when none of the three
is set: session state is not rebuildable, so it is not placed by guesswork.
`home()` still governs session state only, not the artifact cache. The
complete table, including the `jit` feature's `TOKTIER_JIT_CACHE`, is in
[Rust lifecycle and distribution](rust-lifecycle.md).

`Runtime::doctor()` returns typed build and CUDA-probe facts. Device probing
does not load a kernel. `toktier-rust doctor` prints those facts, and
`toktier-rust doctor --json` prints them as JSON under the same field names,
for a control plane that would otherwise parse the debug rendering; the
command refuses an option it does not implement rather than ignoring it.
`Tokenizer::plan()` is the immutable admitted route; every result carries
`ExecutionFacts` naming the backend that actually ran.

Since 0.2.5 those facts also carry `reason: Option<ReasonCode>`: the
routing decision that moved this input off the first admitted backend,
as the code the router recorded for it rather than a second reading of
`path`. It is `None` when the admitted route ran and there was nothing
to record. Why the admitted route is what it is belongs to
`plan().reasons`, so the two do not restate each other, and a fresh
build with no accelerated route admitted reports `None` per execution
while the plan explains itself once. The codes are the frozen `R_*`
namespace of `docs/contracts/routing.md` Section 5; a code this release
has no variant for arrives as `ReasonCode::Other` carrying the code
itself, which is what that contract asks consumers to expect. What
`Other` carries is a code from that namespace and nothing else: through
0.2.6 a session that re-encoded an inconsistent stored tail reported
`Other("invalid_prior_state")`, a string in neither face's vocabulary,
and 0.2.7 registers that outcome as `R_INVALID_PRIOR_STATE` with the
named variant `ReasonCode::InvalidPriorState`. Because a
reason describes a departure from the admitted route, a session append
that re-encodes its whole window reports a reason on a certified build,
where the admitted engine is the repaired CPU one, and none on an
uncertified build, where the admitted engine is the reference and
re-encoding the window is what that engine does. The ids are the same on
both.
Accelerated admission also requires the exact Rust facade source, rustc,
features, target, and profile facts to appear in the shipped
`runtime_builds` registry, **and** the certified core of the compiled
closure to be the judged one. The whole closure is still compared;
`runtime_build.dependency_closure` (`toktier::DEPENDENCY_CLOSURE`) reports
that reading and `runtime_build.dependency_advisory` names packages outside
the core that differ, with the command that aligns them. Neither of the two
decides `certified`; `runtime_build.core_closure` does. Both answers take
the same three shapes: `verified`, a line starting `unlocated:` when no
governing lockfile could be named, or one starting `mismatched:` naming
every package that differs. When the flags are what stands in the way,
`runtime_build.build_flag_divergence` names which key differs and what to
do about it.

### What the certificate speaks for

The **certified core** is TokTier's own crates, the packages they call
directly, and the text-semantics libraries beneath them. Three mechanical
rules draw it, and the record that travels with the crate carries both the
rules and the answer, as `tier_rule` and a `tier` on every package in
`data/build/judged_compiled_closure.json`:

- **R0** -- the six crates this repository publishes.
- **R1** -- their non-development direct dependencies that an encode-path
  source file names: `tokenizers`, `icu_properties`, `spm_precompiled`,
  `unicode-normalization-alignments`, `memchr`, `aho-corasick`, `rayon`,
  `sonic-rs`, `winnow`, `serde`, `sha2` and the rest of that list. Every
  one of them is pinned exactly by at least one of our own edges, which a
  generation gate requires, so **an R1 package cannot drift by version**:
  a consumer resolving today gets the judged version or a refusal about a
  copy that is not the judged package. `fs2` and `tar` are direct
  dependencies that no encode-path file names -- they serve the artifact
  cache and bundle import -- so they are outside the core.
- **R2** -- the text-semantics libraries an engine crate reaches through
  normal edges: `regex`, `regex-automata`, `regex-syntax`, `onig`,
  `onig_sys`, `unicode-segmentation`, `unicode_categories` and
  `icu_properties_data`. What these do is defined by an evolving external
  standard -- a Unicode version, a regex dialect -- so a version of one of
  them really can change ids by design rather than by fault. FINDING 041
  and 043 in the research record are two measured cases of exactly that.

R2 is compared by **behaviour version** rather than by package version:
the version of the tables, not of the crate that ships them.
`Runtime::doctor()` reports each unit in `runtime_build.behavior_versions`,
beside the value the evidence was taken on, and says where it is read:
the newest `\p{Age=...}` the regex engine accepts, `onig::version()`, and
`unicode_segmentation::UNICODE_VERSION`. A unit whose version cannot be
read in a binary falls back to comparing its package versions exactly,
which is the strict direction. So a build whose `regex` moved a patch
version while its Unicode tables stayed put is still certified, and the
move is named in `dependency_advisory`; **a build whose tables moved to
another Unicode version is not certified** and routes to the reference
engine.

Which packages are text-semantics libraries is data, and its completeness
is a gate: any package in the closure whose name matches
`regex|onig|pcre|unicode[-_]|icu_|ucd|spm_|*normaliz*|*segment*|*graphem*|tokenizers`
must be classified by hand, on one side or the other, with a reason, or the
record cannot be generated. `icu_collections`, `icu_locale_core`,
`icu_provider`, `unicode-ident` and `unicode-width` are classified outside
the core: container, plumbing, compile-time and display-width code that
carries no Unicode knowledge of its own.

Everything else the build compiles is **periphery**: error handling,
logging, containers, build-script helpers, JSON internals, the SQLite C
sources. When one of them differs, `dependency_advisory` names it and
carries the `cargo update --precise` command that aligns it, and the
accelerated route is not withheld. That is a statement about what the
certificate covers, not a claim that a package outside the core cannot
influence behaviour.

**One premise the comparison used to leave unwritten.** Certification
compares this crate's ids with the ids the reference engine produces, and
that engine cuts text on Unicode character classes it reads from
Oniguruma, while the fast CPU pre-tokenizer reads the same classes from
ICU property data. The two have to answer alike about every code point for
that comparison to mean what it says. Since 0.2.6 the property data is
pinned to the Unicode version Oniguruma carries and a gate compares the
two exhaustively, over every scalar value and every class the shipped
patterns name (`tools/check_class_parity.py`). A release that moves either
side moves both.

### What the dependency judgement is, and is not

The judged side is **the set of packages Cargo compiled for the certified
build**, taken from Cargo's own account of that build and shipped with the
crate as `data/build/judged_compiled_closure.json`.

Judging every package reachable in the judged `Cargo.lock` instead would read
more thorough and would not be. A lockfile's dependency lists are the union
over every feature and every target, so a Linux consumer would be refused
over a WebAssembly binding, a Windows shim, or an optional dependency no
feature enabled -- none of which enters the artifact. Such refusals are safe
and exact, and they are also about the wrong thing.

**What is judged.** Every package the certified build compiled must appear in
this build's resolved graph at the same version, with the same content hash
for a registry package, and with no semver-compatible sibling that Cargo would
have unified onto instead. That covers the linked crates, the proc macros
whose expansion becomes source, and the build dependencies whose output is
linked -- `cc` and what it pulls in. The line is *does Cargo compile it*, not
*is the compiled code ever called*: the first is one command anyone can
re-run, the second would need a cross-language call graph and could not be
re-checked in a gate. `find-msvc-tools` illustrates the cost of drawing it
there. `cc` declares it unconditionally, so Cargo compiles it on Linux even
though only a Windows build would call it, and it stays inside the judgement.

Every package it names is compared. What the answer is then used for
depends on the tier the record gives it: the core decides `certified`, the
periphery is reported.

**What is deliberately not judged.** Packages this build compiles that the
certified build did not. Concretely: a consumer's own feature unification can
activate an optional dependency of a judged package and so add a package to
the graph, and this comparison does not report that. Asking the question in
the other direction -- every package this build resolves must have been
judged -- would catch it, but only incidentally: feature activation that adds
no package would still be invisible, and features are not recorded in a
lockfile at all. The gap is named here rather than left to be discovered.
It is a structural boundary, not a to-do: a consumer's build
script cannot compute its own feature resolution (it would have to run Cargo
inside a Cargo build, against a workspace an unpacked registry copy cannot
see), so no later release is promised to close it. This crate's *own*
features are judged, in the `features` key of the build flags; it is the
features of transitive packages that are not observable.

Also outside: development dependencies, dependencies of a target this build is
not for, and cross-compilation, where the build dependencies compiled for the
host are not the set recorded for the judged target.

**What each answer is worth.** `verified` says the third-party code linked
into this build is the code the certification campaigns ran. `mismatched`
says at least one package that really does enter the artifact is not the
judged one, and carries what aligns it. `unlocated` says the comparison could
not be made, so nothing is claimed. The same three answers are reported in
two places -- `dependency_closure` for the whole compiled closure and
`core_closure` for the certified core -- and only the second one gates. The mechanism addresses accidental drift,
not a hostile build host: anyone able to edit the sources, the lockfile, or
the shipped records on their own machine can make any self-report say
anything. A crate's checksum is verified when the package is downloaded and
unpacked, not on the builds that follow, so it does not stand behind the
unpacked copies of either record.

The governing lockfile is the first `Cargo.lock` above `OUT_DIR` -- the
consumer's own workspace root in the usual layout -- and above the manifest
directory when this crate is built from its own workspace. Unusual layouts
can name it with `TOKTIER_CARGO_LOCK`; the named file still has to match.
A shared `CARGO_TARGET_DIR` outside the project tree is the common such
layout, and a benign one: `OUT_DIR` then has no lockfile above it, so the
answer is `unlocated` even though the workspace resolved the judged
graph. Point `TOKTIER_CARGO_LOCK` at that workspace's own `Cargo.lock`
and the comparison runs as usual.
Content hashes and origins come from the judged `Cargo.lock`, which travels
with the crate as well, so the two shipped records cannot hold different
opinions about one package; a compiled closure naming something that lockfile
does not hold is refused rather than answered.

### What the build flags claim

The `build_flags` key of a `runtime_builds` entry records what the build
script can observe: `profile`, `opt-level`, `target`, `debug`,
`target-features`, `rustflags`, and this crate's own `features`. Through
0.2.1 it also carried `lto`, `codegen-units`, and `panic`. Cargo does not
expose any of the three to a build script, and those entries were inferred
from the profile name -- so a build with `lto = false` reported `lto=fat` and
matched the judged key on it. Reading `CARGO_CFG_PANIC` is not a fix either:
it reports the panic strategy of the build script, which Cargo always
compiles for the host with unwinding, and it still reads `unwind` under a
`panic = "abort"` profile. Since 0.2.4 the three are not claimed at all,
which is the honest form of not knowing them, and `rustflags`
(`CARGO_ENCODED_RUSTFLAGS`) is claimed instead, because it can be observed and
is where a `-C` codegen switch appears when a caller sets one. A build that
sets RUSTFLAGS is therefore not the judged build; `Policy::Experimental` is
the way to take the accelerated route knowingly anyway.

### What a fresh resolution answers, and how to align it

A consumer who adds the crate today and lets Cargo resolve the graph should
expect `dependency_closure: mismatched`, not `verified`. TokTier's own edges
are pinned exactly, but the transitive versions underneath third-party
dependencies are not, and they move as those crates publish. A resolution
taken on a later day than the certification campaign is therefore the normal
case.

What that costs depends on where the packages that moved sit. When only the
periphery moved, `core_closure` reads `verified`, `certified` is `true`, the
accelerated route is admitted, and `dependency_advisory` names the packages
and prints the commands. When an R2 unit moved to another Unicode version,
`core_closure` reads `mismatched` and this build routes to the reference
engine until its lockfile is aligned or the next release is judged against
the newer tables.

What still earns no certificate at all:

- an `unlocated` build -- no governing lockfile could be named (see below);
- a build whose flags are not the judged ones: `RUSTFLAGS`, a development
  profile, or a feature list no judged recipe carries, `network` among them;
- a build whose R0 or R1 packages resolve from somewhere other than the
  judged package -- a `[patch]`, a fork, a mirror, a vendored copy;
- a build whose R2 tables read another Unicode version, as above;
- a dependency on this crate taken as a `git` dependency, even at the exact
  release commit and with byte-identical sources: Cargo records the origin
  as `git+...` rather than the registry, and this crate's own siblings then
  resolve as copies of the judged packages rather than as them. Depend on
  the published crate to be certified.

A build refused on its core is not a broken one. `Policy::Certified` routes it to the
frozen Hugging Face reference engine, which is the same implementation the
certification campaigns compare against, so the IDs are exactly the ones the
accelerated route would have produced -- what is lost is the acceleration, not
the answer. An explicit `Device::Cuda` request says so with
`UNCERTIFIED_RUNTIME` rather than running uncertified. `Policy::Experimental`
can opt such a build into an accelerated candidate, and every result stays
labelled experimental.

Two paths lead back to `verified`, on either reading, and `doctor()` prints
the first one:

- **Align the consumer's own lockfile.** A `mismatched` line carries a
  `cargo update --precise <judged version> <package>` command for every
  package it names, ordered so the commands can be run one after another in
  the workspace that owns the governing lockfile. Rebuild afterwards and the
  same field reads `verified`. Packages that differ by origin rather than by
  version -- a `[patch]`, a vendored copy, a mirror -- are named with that
  reason instead, since no version change moves them. Since 0.2.4 the list
  holds only packages that really are compiled, so it is both shorter and
  worth acting on: every command on it changes the bytes that run.
- **Build inside a workspace that already holds the judged graph.** The
  shipped wheel is always produced inside the source workspace with
  `--locked`, so it carries the judged graph by construction and is never
  affected by any of this. The judged `Cargo.lock` also travels inside the
  crate archive as `data/build/judged_dependencies.lock` and can be read
  there. Note that pointing `TOKTIER_CARGO_LOCK` at that packaged copy -- or
  at a copy of its bytes kept elsewhere -- is refused rather than compared,
  since the file is this crate's record of the judged build and not an
  account of the build now running, so it is not a way to earn a
  certificate. The answer is an `unlocated` line naming both routes above.

`cargo install --locked toktier` is a third case worth naming, because
`--locked` does make Cargo resolve the judged versions. It still answers
`unlocated`: `cargo install` unpacks the package under `CARGO_HOME` and builds
it in a temporary directory, so no lockfile stands above `OUT_DIR`, and a
registry build deliberately does not read the copy sitting beside its own
manifest -- comparing a packaged file against itself would answer nothing.
The installed `toktier-rust` binary is fully usable; it runs the reference
engine. (`cargo install --locked --path <unpacked crate>` does report
`verified`, because Cargo then builds under the package's own directory and
the lockfile it finds there is the one `--locked` actually used.) This is the
`unlocated` rule rather than the tier rule: no comparison was made at all, so
neither reading can answer, and the advisory has nothing to report.

### Device and toolchain assurance

The sections above are about what code was compiled. This one is about
where it runs, which is a separate question with a separate answer.

A certification campaign measures a kernel on the devices and with the
compilers it actually had. Nothing in that measurement says a device it
never saw is wrong -- only that nobody looked. Until 0.2.6 the two were
answered with the same word, so an unlisted architecture and a build
whose identity did not verify were both refused as "uncertified". They
are now separate:

| Assurance | What it says | Reached under |
|---|---|---|
| `certified` | The exact configuration appears in the shipped registry and every constraint it binds verifies here. | every policy except `Reference` |
| `supported_untested` | The engines and digests are the judged ones; the device architecture or the compiler triple is one no campaign ran on, and the kernel loads and runs here. | `Policy::Supported` (the default) |
| `locally_verified` | The same, plus a local check on this machine compared the route with this binary's reference engine and they agreed. | `Policy::Supported` |
| refused | Something that was bound did not verify: a source digest, a host digest, build flags, the certified core of the closure, or a compiler component anyone can write to. | nothing but an explicit opt-in |

`Policy::Supported` is the default from 0.2.6. `Policy::Certified` keeps
its frozen meaning exactly and is the strict switch: under it an
unlisted device or an unjudged compiler triple routes to the reference
engine as it always has. The last row does not move under either: a
coverage gap is admitted, a verification failure is not.

The JIT path splits the same way. A source, host or flags mismatch still
refuses, and so does a world-writable compiler component unless the
caller accepts it explicitly; only an unjudged (triple, device) pair is
admitted, and the product it builds is cached and reused like any other.

Driver and CUDA runtime versions are reported as environment facts and
are not certificate premises: `driver_api_version: 13020 (environment
fact; not a certificate premise)`. Where a registry row does bind a
driver floor, that floor is a loadability precondition and is still
checked.

To measure a route yourself:

```text
toktier-rust verify-local --engine both --family qwen3_8b --input my-text.txt
```

The Python package carries the same command under the same name and the
same options for the same job (`toktier verify-local --engine both --family
qwen3_8b --input my-text.txt`); the Rust face adds `--delivery` and
`--device`, and the two faces default `--synthetic` and `--seed`
differently, so a machine with both installed has one command to remember
and two small differences to know about. Each face keeps its own records:
they describe different engines.

It encodes your documents on the accelerated route and on this binary's
reference engine and compares every id. `--synthetic N` builds documents
from rules instead, so a check needs no corpus and no network.
`--engine cpu` asks the same question of the integrated CPU engine. That
is what the `behavior_versions` advisory points at when an R2 package
moved while the tables it carries did not: the certificate holds there
and the CPU route is admitted, so the command has something to compare.
It is not what a `mismatched` core points at -- that state holds the
accelerated engines on the reference route, so there is no accelerated
CPU route to compare, and the line names the alignment command instead.

A record is written only when the route served every document. A run it
served in part is reported and not recorded: those documents were
compared and agreed, but the run does not cover the route. A run it
served not at all measured nothing, because both sides were then the
reference engine, and the note says which of two things happened. A route
the plan did not admit: the note carries the plan's own reason codes, the
same ones `Tokenizer::plan().reasons` holds in typed form, and
`toktier-rust doctor` reports the build facts they rest on. Or a route the
plan admitted that every document left for a per-input reason: the note
lists the codes the ledger recorded, which is what each encode's
`ExecutionFacts::reason` carried. This crate has no `explain()`, so no
note sends a reader to one; the Python package's `verify-local` names its
own `toktier doctor --family` and `explain()` for the same two states.

What it writes is a record, not a certificate. The record is filed under
the engine it measured, the device architecture, the delivery, the image
digest, the compiler triple, the driver, the two source identities, the
family artifact and the version of the tool that wrote it, so it stops
applying the moment any of those moves; `--forget` removes it. The
architecture rather than the device index: two devices of one
architecture share a record, which is what the campaigns measure as
well. For a JIT product the compiler triple is the NVCC release, build
and binary digest the compiler stage recorded; before 0.2.7 the Rust
face filed JIT records without it, so a record taken on such a build is
filed under a key this release does not look up. It is ignored rather
than re-taken -- nothing re-measures a route on its own -- and running
`verify-local` again files one under the new key. A prebuilt record
carries no compiler: its image digest names the shipped bytes outright.
A check that disagreed is kept and reported, and the route keeps the
label it would have had if nobody had run one -- running the tool never
makes a configuration more restricted than not running it, and nothing
is changed on the caller's behalf.

### `RoutePlan` on the Rust surface

`Tokenizer::plan()` returns the immutable route the tokenizer was
constructed with. `docs/contracts/routing.md` Section 3 describes the
Python facade's `RoutePlan`; this crate has a type of the same name whose
fields differ, and these are they:

| Field | Type | What it holds |
|---|---|---|
| `family` | `String` | The family this plan is for. |
| `artifact_sha256` | `String` | The verified artifact the plan is bound to; a different artifact is a different plan. |
| `backends` | `Vec<Backend>` | The admitted backends in order. The reference engine is always reachable; a plan that admitted nothing accelerated holds `[Backend::HuggingFace]`. |
| `gpu_min_bytes` | `u64` | The input size at or above which the GPU backend is selected, when one is admitted. Below it execution starts at the next admitted backend, which is the ordinary latency policy rather than a fault. |
| `certification` | `Certification` | The label of the admitted route: `Certified`, `CertifiedSource`, `Supported`, `LocallyVerified`, `Experimental` or `Reference`. |
| `reasons` | `Vec<ReasonCode>` | One entry per accelerated option that was considered and not admitted. This is where a build whose closure or device the registry does not judge says so, and what `verify-local` names when it has no accelerated route to compare. |

The type is `#[non_exhaustive]`, so it is read rather than constructed by
a consumer, and both plans are immutable for the lifetime of the
tokenizer: a device appearing or an environment variable changing does
not alter an existing plan. Construct a new tokenizer to re-plan.

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
not a compatibility interface. `docs/contracts/errors.md` is the Python
facade's frozen contract; the table below is the Rust surface's own, and the
two use the same stable code strings. Where a Rust row is wider than the
Python row of the same name, the row says so. `ArtifactHashMismatch` is
reserved for actual content-hash verification failures; a hash-verified
artifact that later fails to parse reports the failing stage instead
(`KernelIncompatible` from GPU table construction, `Internal` from a
reference-engine load). The `serde` feature serializes typed plans, execution
facts, doctor facts, and store statistics.

### `ErrorCode` on the Rust surface

Codes are append-only: a variant is never renamed, reused, or re-meant. The
enum is `#[non_exhaustive]`, so a `match` on it needs a catch-all arm.

| Variant / string | Raised when |
|---|---|
| `InvalidArgument` / `INVALID_ARGUMENT` | Caller misuse the type system cannot refuse: a batch row out of range, an offset request on the ID-only batch API, a revision that is not an immutable 40-hex commit, a value that does not fit the platform. |
| `ConfigInvalid` / `CONFIG_INVALID` | A configuration value is impossible or self-contradictory: a CPU device with GPU delivery, an empty or zero bound, a URL carrying credentials or control characters, and (since 0.2.4) a **path this crate refuses by policy** -- a parent-directory component, a symbolic-link component, a non-directory component, or a final path that is not a regular directory. Wider than the Python row, which covers configuration values only. |
| `ArtifactNotFound` / `ARTIFACT_NOT_FOUND` | An artifact cannot be resolved: an unknown family (the message carries the closest valid ids), a missing or non-regular `tokenizer.json`, a missing bundle or bundle member, an empty cache under offline mode. |
| `ArtifactHashMismatch` / `ARTIFACT_HASH_MISMATCH` | Verified bytes do not match the manifest digest. Content-hash failures only. |
| `ArtifactSizeMismatch` / `ARTIFACT_SIZE_MISMATCH` | A file's byte count differs from the manifest, including a stream that stops early or overruns. Reported before the digest so the cheaper fact is not hidden behind the expensive one. |
| `BundleInvalid` / `BUNDLE_INVALID` | An air-gap bundle violates the frozen archive format: unsafe or duplicate members, link members, a member set that disagrees with its manifest, resource-limit violations. |
| `CacheBusy` / `CACHE_BUSY` | A bounded wait for a cache lock expired, or a staging name could not be allocated. Retryable by construction. |
| `Network` / `NETWORK_ERROR` | An HTTPS request failed or exhausted its retry budget. Only with the `network` feature. |
| `NetworkDisabled` / `NETWORK_DISABLED` | Acquisition was requested with the `network` feature off, which from 0.2.5 is the default build. The message names the feature and the offline ways to supply the bytes. Explicit offline mode is the row above: nothing was attempted, so the answer is that the cache does not hold it. |
| `RegistryInvalid` / `REGISTRY_INVALID` | Shipped registry, manifest, or table bytes fail their embedded digest, schema, or cross-reference checks. This is a package-integrity failure, not a caller error. |
| `UncertifiedRuntime` / `UNCERTIFIED_RUNTIME` | An accelerated route was demanded from a build whose identity is not in the shipped `runtime_builds` register. |
| `UncertifiedTokenizer` / `UNCERTIFIED_TOKENIZER` | The artifact itself has no certification row for the demanded route, or a repository/revision is outside the shipped equivalence table. |
| `KernelIncompatible` / `KERNEL_INCOMPATIBLE` | The kernel cannot be admitted: uncertified SM, fatbin or class-table digest mismatch, a family the kernel does not cover, a device-side constraint. |
| `JitCompileFailed` / `JIT_COMPILE_FAILED` | The `jit` feature ran NVCC and the compile, its inputs, or its products failed. |
| `UncertifiedJit` / `UNCERTIFIED_JIT` | A JIT product outside the judged toolchain set was requested under a policy that does not admit it. Since 0.2.6 the default `Policy::Supported` does admit an unjudged (toolchain, device) pair and labels the product `supported_untested`, so this code is what `Policy::Certified` returns for that case; a product built on an explicit waiver still needs `Policy::Experimental`. |
| `QueueFull` / `QUEUE_FULL` | A bounded serving queue, byte budget, or per-session in-flight bound is at its limit. Backpressure, not failure. |
| `RequestCancelled` / `REQUEST_CANCELLED` | A serving request was cancelled. Work already started may still have committed; the code does not claim a rollback. |
| `DeadlineExceeded` / `DEADLINE_EXCEEDED` | A serving request passed its deadline. Same non-rollback caveat. |
| `RuntimeShutdown` / `RUNTIME_SHUTDOWN` | A request arrived after the serving pool began shutting down. |
| `BackendExecutionFault` / `BACKEND_EXECUTION_FAULT` | A backend failed on an input in a way the router may recover from, and the store's `ENGINE_ERROR` when the session encoder reports one. |
| `SessionRevisionConflict` / `SESSION_REVISION_CONFLICT` | An optimistic write met a different stored revision, a session name was seeded twice, or a second writer in this process asked for the same session. Wider than the Python row, which covers the stored-revision case. |
| `SessionStateMismatch` / `SESSION_STATE_MISMATCH` | Stored state does not belong to this tokenizer or does not extend as claimed: fingerprint or witness-category mismatch, a transcript that is shorter than or diverges from the stored session, a missing recovery transcript. Wider than the Python row. |
| `StoreCorrupt` / `STORE_CORRUPT` | An explicit integrity operation found a checksum, linkage, or structural failure. On the ordinary read path an integrity failure becomes a counted miss instead: a miss is preferred over a wrong result. |
| `StoreFormatUnsupported` / `STORE_FORMAT_UNSUPPORTED` | A record is well-formed but not decodable by this reader (future version, unknown mandatory flag, unknown witness category). Reached through the store conversion, distinct from corruption by design. |
| `Io` / `IO_ERROR` | A filesystem or process operation failed as attempted -- the underlying `std::io::Error` text is carried. Policy refusals about a path are `ConfigInvalid`, not this. |
| `Internal` / `INTERNAL` | An invariant of this crate was broken. Never a caller error; report it. |

Before 0.2.4 the four path-policy refusals in the third row reported
`IO_ERROR`. That answer was hard to act on -- a caller retrying I/O would
retry forever -- so they were moved to `CONFIG_INVALID` and named here.
Code matching on `Io` for a refused cache, bundle, or JIT root needs the
one-line update; nothing else changed, and the messages are unchanged.

The current feature surface is:

| Feature | Default | Effect |
|---|---:|---|
| `sqlite` | yes | durable named sessions and restart recovery |
| `prebuilt-gpu` | yes | embedded, digest-bound CUDA fatbin delivery |
| `serving` | yes | bounded executor-neutral queue, batching, and device policy |
| `network` | no | TLS artifact acquisition with pinned revisions |
| `jit` | no | direct shell-free NVCC compilation into the Rust CUDA host |
| `serde` | no | serialization for public diagnostic records |

`network` was a default feature through 0.2.4. It is opt-in from 0.2.5,
because it is the only feature that pulls a TLS stack into builds that may
never fetch. It is sixteen packages: the default build of this release
compiles 148, and enabling `network` makes that 164.
A build without it keeps every offline lifecycle surface and answers
`NETWORK_DISABLED` when acquisition is actually required;
`Runtime::doctor()` reports it as `network_compiled`. Because the judged
build recipe records this crate's feature list, the certified default build
for 0.2.6 is the one without `network`, and a build that enables it
diverges from that recipe on the `features` key. The feature list is a
build flag rather than a package, so the tier rules above do not reach it:
a build that turns `network` on is not certified even though the two
engines it compiles are byte for byte the judged ones.

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
