# Using toktier with SGLang

This page describes two ways toktier can be used together with
[SGLang](https://github.com/sgl-project/sglang). The first needs no
changes on either side and runs against the shipped Python facade. The
second is the adapter that ships in `toktier.integrations.sglang` since
0.3.0, installed either by a small tokenizer-backend patch, shaped like
the `fastokens` backend SGLang already ships, or without patching
anything. The patch is something we would rather discuss with the SGLang
maintainers than carry on our own. Both were exercised on a Qwen3-8B
server; the numbers quoted below come from that setup and are described
in the last section. Which SGLang versions the adapter has been checked
against, and how to install the two packages side by side, are in
[Supported SGLang versions](#supported-sglang-versions) and
[Installing beside SGLang](#installing-beside-sglang).

## What SGLang already does

SGLang tokenizes each request in its `TokenizerManager` (the HTTP server
process, or `--tokenizer-worker-num N` worker processes), by calling the
Hugging Face tokenizer it loaded through `get_tokenizer(...)`. Two things in
0.5.19 matter here:

- a request that carries `input_ids` skips tokenization entirely
  (`_tokenize_one_request`), and the prefix cache keys on token ids, so a
  text request and an ids request that encode the same transcript hit the
  same radix entries;
- since 0.5.17 a request may carry a `session_id`, and with
  `--enable-session-radix-cache` the KV cache keeps per-session references
  (`/close_session` releases them). The `session_id` is also the natural key
  for a tokenizer-side session.

`--tokenizer-backend fastokens` shows that SGLang is open to swapping the
encode implementation behind the same object: it patches `transformers` and
loads `AutoTokenizer` as usual. The `toktier` backend below is the same
shape, without patching `transformers`.

## Surface A — feed token IDs, no changes to SGLang

`/generate` accepts `input_ids`; an application that keeps its own session
state can therefore compute ids with toktier and hand them over, exactly as
with Dynamo (see [`dynamo.md`](dynamo.md)):

```python
import requests, toktier

tok = toktier.load("qwen3_8b", store="./toktier-store")

# The argument is the complete rendered transcript, not the delta; toktier
# verifies the stored prefix byte for byte and encodes only what is new.
transcript = rendered_history + new_turn
ids = list(tok.encode(transcript, session=conversation_id).ids)

requests.post("http://sglang:30000/generate", json={
    "input_ids": ids,
    "session_id": conversation_id,              # optional, radix-native session
    "sampling_params": {"max_new_tokens": 256},
})
```

We checked the key-space identity on 0.5.19 (24K-character prompts,
per-request `meta_info.cached_tokens`): text→text, ids→ids, text→ids and
ids→text second calls all hit 0.9998 of the prompt, and greedy outputs are
identical across the two channels. The ids equal a from-scratch encode of
the same text (`tok.encode(transcript, lookup="off").ids`), which is the
facade contract. The usual note applies: the client now owns template
rendering and special-token handling; `add_special_tokens` stays at its
default `False` on store-backed paths, and Qwen3's template adds nothing
around a single sequence, so this is the exact stream the server would have
produced.

## Surface P — a `toktier` tokenizer backend

The backend patch (about fifty added lines) adds
`--tokenizer-backend toktier`. It comes in two files, one per SGLang
release line: `patches/sglang-0.5.19-toktier-backend.patch` for 0.5.19,
and `patches/sglang-0.5.21-toktier-backend.patch` for 0.5.20 and 0.5.21
(0.5.20 moved the `--tokenizer-backend` option to
`arg_groups/fields/serving.py`); the published Rust source archive also
carries both patches, under the same paths. Each patch touches two files:

- `server_args.py` (0.5.19) or `arg_groups/fields/serving.py` (0.5.20
  and 0.5.21): the choice and its help text;
- `managers/tokenizer_manager.py`: `init_tokenizer_and_processor` wraps
  the tokenizer it has loaded with
  `toktier.integrations.sglang.wrap_tokenizer`, before the dynamic-batch
  tokenizer is built, so that one holds the wrapper as well;
  `_tokenize_one_request` binds the request's `session_id` (else
  `cache_salt`) in a context variable around the tokenize call, so the
  backend can serve the transcript as a verified prefix append; and
  `_tokenize_texts` first offers the request to the adapter's worker pool
  (`toktier.integrations.sglang.tokenize_texts_with_pool`) and otherwise
  runs as before.

`get_tokenizer` is not patched. SGLang's scheduler, TP worker and
detokenizer load their own tokenizer through it, and they keep the plain
Hugging Face object, so the wrapper, and the worker pool when it is on,
exist only in the `TokenizerManager` process (with
`--tokenizer-worker-num N`, once in each of the N worker processes).

The wrapper (`src/toktier/integrations/sglang.py`) is a duck-typed
tokenizer: `__call__` and `encode` run on toktier's certified CPU route,
every other attribute (`eos_token_id`, `apply_chat_template`, `decode`,
chat template, vocabulary) is delegated to the wrapped Hugging Face object,
and calls with arguments the wrapper does not model (text pairs,
truncation, `return_token_type_ids`) are delegated whole. Session state can
be keyed by the request's `session_id` (`TOKTIER_SGLANG_SESSION_KEY=
contextvar`), or left to content lookup (`auto`, no protocol change at all),
or turned off (`off`, stateless encode). A shadow mode
(`TOKTIER_SGLANG_SHADOW=sync|async`) re-encodes each request with the
wrapped tokenizer and compares id by id; `sync` corrects on the spot and
counts, `async` counts after the fact, judged against an independent
copy of the wrapped object (never a fresh load by name, which could
differ from it), or synchronously against the object itself when no copy
can be made (`stats()["shadow"]` says which). A JSONL timing log records
each encode next to SGLang's own tokenize window.

The same wrapper can be attached without patching anything, for evaluation:
a `sitecustomize` on `PYTHONPATH` calls
`attach_to_tokenizer_manager(TokenizerManager)` once the module is
imported. That is how the numbers below were taken. The patch produces the
same wrapper through the supported option, at the same place: the
attach form wraps the `TokenizerManager`'s tokenizer once its
constructor returns, the patch does so inside the constructor right
after loading it, and neither touches the tokenizers of SGLang's other
processes. We checked this on 2026-10-13 with each patch applied to the
published wheels of 0.5.19, 0.5.20 and 0.5.21, running SGLang's own
tokenizer-loading code of each process on the CPU side, without a model
server: the `TokenizerManager` holds the wrapper and the one worker
pool, its dynamic-batch tokenizer holds the same wrapper, and the loads
of the scheduler, the TP worker and the detokenizer return the plain
Hugging Face tokenizer.

Two things the wrapper deliberately does not do. It does not touch
`--tokenizer-worker-num` semantics: each worker process gets its own
wrapper and, when a persistent `store=` is configured, its own store
directory (the store is single-process). And under
`--enable-dynamic-batch-tokenizer` the encode runs in an executor thread
that does not carry context variables, so the `contextvar` key degrades to
content lookup there; the timing log records which path served each call.

## Long stateless requests: chunk-parallel encode (0.3.0)

A request that carries no session key and will not be extended -- a
single-shot prompt of a few hundred thousand characters, say a document
pasted into one turn -- is a stateless full encode, and a single encode
runs on one core. Since 0.3.0 the wrapper can hand such requests to
`toktier.chunked.encode`, the certified chunk-parallel encoder of the
CPU route (`docs/serving.md` Section 3): the text is cut at certified
stable boundaries, the chunks run as one batch on the engine's worker
pool, and the ids are concatenated -- the same ids as the serial encode,
by construction.

It is off by default and switched on with a threshold:

```
TOKTIER_SGLANG_CHUNKED_MIN_CHARS=256000   # characters; 0 (default) = off
TOKTIER_SGLANG_CHUNKED_K=0                # chunk count; 0 = the shipped table
```

Which requests it serves: every request in `off` mode, and a request
*without* a key in `contextvar` mode (one that carries neither
`session_id` nor `cache_salt`), when it is at least
`TOKTIER_SGLANG_CHUNKED_MIN_CHARS` characters long. A keyed request stays on
the session path whatever its length; `auto` mode is content lookup by
design and is not touched. The threshold worth setting is the table's
own floor, 256 K characters (below it the batch path is not faster than
a single encode); `TOKTIER_SGLANG_CHUNKED_K` exists to force the chunk count
for a measurement and is otherwise best left at `0`. An explicit
`AdapterConfig.long_request_encoder` takes precedence over the built-in
encoder for the same requests.

The parallelism comes from the engine's Rayon worker pool, sized by
`RAYON_NUM_THREADS` at first use in the tokenizer process; with that
variable at `1` the chunks are encoded one after another (still exact).
The workers are forked lazily on the first batch that needs them, about
20 ms each, once per process.

Each served request is recorded in the timing log with `how: "chunked"`
and the fields `chunked_k`, `chunk_count` and `chunked_serial_reason`
(`null` when the chunks ran, and a reason code such as `R_CHUNK_NO_CUT`
when the document was encoded serially after all). The stats file
carries the tokenizer's `chunked` diagnostics block. For an admitted
artifact the encoder takes the native path wherever it can run and the
Python path otherwise, with the native path's evidence level shown as
`native_evidence_label` in that block (`docs/serving.md` Section 3), and
an artifact the table does not admit is encoded serially and counted
under `R_CHUNK_FAMILY_UNSUPPORTED`.

Correctness posture is unchanged: the chunked result equals the
from-scratch reference encode (for an NFC family such as qwen3_8b, cuts
are made only where they are also cuts of the normalized text;
`docs/serving.md` Section 3), and the shadow check compares it like any
other.

## Scheduled encodes: a process pool behind the wrapper (0.3.0)

Everything above runs the encode on the thread of SGLang's event loop,
one request after another, so the tokenizer is a serial stage of the
request path: while one long transcript is being encoded every other
request waits, and a long request occupies one core however many the
host has (`--tokenizer-worker-num` overlaps requests across HTTP
processes but cannot shorten one, and each worker keeps its own session
store). The certified CPU engine adds a constraint of its own: its main
engine is serialized inside one process, so parallel encodes need one
process each.

`toktier.integrations.sglang_sched` gives the wrapper a pool of worker
processes, each holding one loaded tokenizer (about 320 MB each), and a
policy that decides what each idle worker takes next:

- keyed requests go to a *home* worker chosen from the key and take the
  session path there (`encode(text, session=key)`: only the new turn is
  encoded), because the store is one directory with one writer;
- stateless requests go to a shared queue ordered by an estimate of
  their service time (shortest first, aged so nothing waits without
  bound; `fifo` is available);
- a stateless request of at least `chunk_min_chars` is cut at certified
  stable boundaries (`toktier.chunked.plan`, the same cuts as the
  chunk-parallel encoder) into `min(workers, kmax, ceil(N /
  chunk_target))` chunk tasks that any worker may take, so a long
  request neither monopolizes one worker nor holds the short ones behind
  it; the ids are concatenated, which is the serial encoding by
  construction.

`attach_to_tokenizer_manager` also patches `_tokenize_texts`. With the
pool on, a single string, or a list of strings, is then awaited through
`SGLangToktierTokenizer.encode_async`, so the event loop stays free while
the pool works and several requests are in flight at once. Without a
pool, and for cross-encoder inputs and other shapes, the original method
runs. Other entry points (`__call__`, the dynamic-batch executor) use the
pool synchronously. The backend patch adds the same hook: both forms go
through `toktier.integrations.sglang.tokenize_texts_with_pool`.

```
TOKTIER_SGLANG_SCHED_WORKERS=4        # 0 (default) = no pool
TOKTIER_SGLANG_SCHED_CORES=0-3        # optional: pin worker i to cores[i]
TOKTIER_SGLANG_SCHED_POLICY=srpt      # or fifo
TOKTIER_SGLANG_SCHED_CHUNK_MIN_CHARS=131072
TOKTIER_SGLANG_SCHED_CHUNK_TARGET=65536
TOKTIER_SGLANG_SCHED_KMAX=16
```

Correctness posture is unchanged: the pool decides where and in what
order the same encodes run, never what they return. Whatever a worker
answers passes through the wrapper's template, shadow check and timing
log like a serial encode (`how: "sched_session" | "sched_auto" |
"sched_free" | "sched_chunked"`, with the chunk count, the wait and the
service time per request), and a request the pool cannot answer -- a
worker that failed to start, exited, raised, or did not answer within
`worker_timeout_s` -- is encoded in-process on the route it would have
taken without the pool and counted (`stats()["sched_fallback"]`). A
keyless request in `contextvar` mode keeps its content lookup (on a home
worker chosen from the text's start) unless it is long enough to be
chunked; `auto` mode is content lookup by design and is never chunked.

What the pool costs: the text and the ids cross a process boundary on
every request, which on the host of `docs/serving.md` is about 0.6 ms at
100 K characters and 6 ms at 1 M on the session path (0.4 and 4.8 ms
serial), so the pool pays for itself on long or concurrent requests, not
on short sequential ones. The registered patch-shaped native return
(`docs/releases/v0.3.0.md`) is what removes the per-request stream copy.

## Settings

The adapter is configured through `AdapterConfig` and, for the pool,
`SchedulerConfig`. `AdapterConfig.from_env()` reads the variables below
once, when the adapter is set up, and hands the scheduler its settings
as a mapping, so nothing reads the environment later. An unset or empty
variable keeps the default. A value the adapter does not accept is
refused rather than ignored: the attach form then leaves SGLang
unchanged and passes the error to its `log` callback, and with the
backend patch SGLang stops at startup. When a numeric value does not
parse, or an on/off value is outside its set, the error message starts
with the variable's name. These names belong to the
adapter: they sit outside the frozen variable set of
`docs/contracts/config.md`, and like the adapter itself they are
provisional in 0.3.0.

| Variable | Default | What it sets |
|---|---|---|
| `TOKTIER_SGLANG_FAMILY` | unset | the toktier family id to load; unset resolves the wrapped tokenizer's `name_or_path` through `toktier.from_pretrained` |
| `TOKTIER_SGLANG_REPO` | unset | the repository id to resolve when no family is named, in place of `name_or_path` |
| `TOKTIER_SGLANG_SESSION_KEY` | `auto` | `auto` (content lookup), `contextvar` (the request's key), `off` (stateless) or `passthrough` (the wrapped tokenizer answers, the wrapper only times) |
| `TOKTIER_SGLANG_RID_SEPARATOR` | unset | when set, the part of the request id before it is a key of last resort (attach form only; the backend patch keys on `session_id`, else `cache_salt`) |
| `TOKTIER_SGLANG_STORE` | unset | a directory for persistent session state |
| `TOKTIER_SGLANG_PER_PROCESS_STORE` | `1` | `1`, `true` or `yes` (any letter case) appends the process id to the store directory, so tokenizer worker processes never share one; `0`, `false` or `no` turns that off; any other value is refused |
| `TOKTIER_SGLANG_IDENTITY` | `require` | the identity check is always on and `require` is the only accepted value; any other value is refused (the attach form does not attach and logs why; with the backend patch SGLang stops at startup) |
| `TOKTIER_SGLANG_SHADOW` | `off` | `off`, `sync` (re-encode with the wrapped tokenizer, compare and correct) or `async` (compare after the fact), in any letter case; `0` and `false` also mean `off`, and `1` and `true` mean `sync` |
| `TOKTIER_SGLANG_SHADOW_RATE` | `1.0` | the share of requests the shadow check takes |
| `TOKTIER_SGLANG_SHADOW_DUMP_DIR` | unset | a directory that receives one JSON file per shadow mismatch |
| `TOKTIER_SGLANG_TIMING_LOG` | unset | a JSONL file with one record per encode |
| `TOKTIER_SGLANG_STATS_FILE` | unset | a JSON file that receives the counters at exit |
| `TOKTIER_SGLANG_EXPLAIN_EVERY` | `0` | every this many calls, add the last execution path to the timing record; `0` is never |
| `TOKTIER_SGLANG_CHUNKED_MIN_CHARS` | `0` | hand stateless requests of at least this many characters to `toktier.chunked.encode`; `0` is off |
| `TOKTIER_SGLANG_CHUNKED_K` | `0` | force the chunk count of that encoder; `0` reads the shipped table |
| `TOKTIER_SGLANG_SCHED_WORKERS` | `0` | the size of the worker pool, a non-negative integer (a negative value is refused); `0` is no pool, and the settings below are read only when it is positive |

| Variable | Default | What it sets |
|---|---|---|
| `TOKTIER_SGLANG_SCHED_CORES` | unset | pin worker `i` to the `i`-th listed core (`0-3,8` form) |
| `TOKTIER_SGLANG_SCHED_POLICY` | `srpt` | `srpt` (estimated shortest first, with aging) or `fifo` |
| `TOKTIER_SGLANG_SCHED_AGING` | `1.0` | milliseconds of priority a task gains per millisecond it waits |
| `TOKTIER_SGLANG_SCHED_CHUNK_MIN_CHARS` | `131072` | stateless requests at least this long are cut into chunk tasks |
| `TOKTIER_SGLANG_SCHED_CHUNK_TARGET` | `65536` | the chunk size the cut aims for, in characters |
| `TOKTIER_SGLANG_SCHED_KMAX` | `16` | the most chunks one request is cut into; never more than the workers |
| `TOKTIER_SGLANG_SCHED_K_RULE` | `fixed` | `idle` also caps the chunk count by the idle workers at arrival |
| `TOKTIER_SGLANG_SCHED_AFFINITY` | `1` | `1`, `true` or `yes` (any letter case) keeps keyed requests on their home worker; `0`, `false` or `no` turns that off, and they are then encoded stateless on any worker; any other value is refused |
| `TOKTIER_SGLANG_SCHED_MS_FIXED` | `0.2` | the fixed part of the service-time estimate, in milliseconds |
| `TOKTIER_SGLANG_SCHED_MS_PER_CHAR` | `3e-5` | the per-character estimate for stateless work and chunks |
| `TOKTIER_SGLANG_SCHED_MS_PER_CHAR_SESSION` | `1e-5` | the per-character estimate for session work |
| `TOKTIER_SGLANG_SCHED_NON_ASCII_RATIO` | `5.0` | how much more a non-ASCII character is estimated to cost |
| `TOKTIER_SGLANG_SCHED_CALIBRATE` | `1` | `1`, `true` or `yes` (any letter case) lets measured service times move the per-character estimates; `0`, `false` or `no` keeps the configured estimates; any other value is refused |
| `TOKTIER_SGLANG_SCHED_WORKER_TIMEOUT_S` | `60` | how long one task may take before its request is served in-process; `0` is no bound |
| `TOKTIER_SGLANG_SCHED_START_TIMEOUT_S` | `300` | how long the pool may take to start |
| `TOKTIER_SGLANG_SCHED_RAYON_THREADS` | `1` | the Rayon thread count each worker starts with |

The estimate only orders the queues, so it can be rough; the defaults
are the values our measurements used, and most deployments need to set
only `WORKERS` and `CORES`. The pool's worker loader is a construction
argument of `SchedulerConfig` used by tests and has no variable.

## Correctness posture, briefly

Every toktier path returns the ids a from-scratch reference encode would
return; a store layer can decline to serve, never answer differently. On
the 0.5.19 server we ran the toktier arm with the shadow check on every
request: 0 mismatches, prompt token counts equal to the local Hugging Face
reference for every request, and 50 of 50 greedy completions identical to
the unpatched server's outputs. Special tokens are applied at read time
from a prefix/suffix template derived from the wrapped tokenizer and
re-verified; a tokenizer whose template is not a fixed prefix/suffix makes
the wrapper delegate those calls rather than guess. The wrapper loads the
family by id (`TOKTIER_SGLANG_FAMILY`) or by resolving the tokenizer's
`name_or_path` through `toktier.from_pretrained`, which admits acceleration
only for tokenizer bytes it has certified.

## The wrapped tokenizer must be the one toktier loaded

A family or repository name alone cannot establish that the Hugging Face
object SGLang hands the wrapper is the tokenizer toktier loaded. The
supplied object may carry another revision, other loader options, or
live vocabulary changes (`add_tokens` after loading is the concrete
case, and a token added that way is one id on the supplied object and
several on toktier's). So before the wrapper accelerates anything it
serializes the supplied object's fast backend (`backend_tokenizer.
to_str()`, which includes tokens added after loading) and the loader
face toktier executes against, and compares the sections that decide
the core id stream (`check_core_identity`). Four details of that
comparison matter in practice.

- The sections compared are the model, the normalizer, the
  pre-tokenizer, the post-processor, the added tokens, and the
  truncation and padding state.
- The supplied object's `split_special_tokens` setting must be off.
  The serialization does not carry it, and when it is on, added tokens
  are split like ordinary text, which toktier's subject never does.
- The `trim_offsets` flag of a byte-level pre-tokenizer or
  post-processor is left out on both sides. It decides character
  offsets, not ids, and `transformers` 5.12.1 sets it on the live
  objects of the two Qwen families (qwen3_8b and qwen3_5_08b) where
  their artifact files do not, so comparing it would send every request
  for those families to the supplied tokenizer under SGLang's
  `transformers` with no difference in the ids.
- Admission is checked again after construction. Before every
  accelerated encode the wrapper compares a cheap fingerprint of the
  supplied object's mutable surface (added tokens, the split setting,
  normalizer, pre-tokenizer, post-processor, truncation and padding)
  with the one taken at admission, and a mutating call made through the
  wrapper marks it stale at once. On any difference the full comparison
  runs again. An object that is no longer equivalent switches the
  wrapper to delegation for good (the pool, if any, is closed, and an
  `identity_drift` record is written); one that still is keeps
  acceleration and has its shadow reference rebuilt.

When the two are not equivalent, or one side cannot be serialized, the
wrapper delegates every call to the supplied tokenizer for its whole
life. The outcome is in `stats()["identity"]` (`status`, `reason`, the
differing `sections`) and in the init record of the timing log, and
`mode` then reads `passthrough`. The check cannot be turned off.
`TOKTIER_SGLANG_IDENTITY` accepts only `require`; another value is
refused rather than ignored, so a tokenizer that cannot be serialized is
served by SGLang's own tokenizer.

Attribute writes follow attribute reads: an assignment SGLang makes on
the tokenizer -- `tokenizer.chat_template = ...` in its template manager
when a chat template is installed -- lands on the wrapped object, where
the delegated `apply_chat_template` reads it.

## Supported SGLang versions

The adapter has been checked against SGLang 0.5.19, 0.5.20 and 0.5.21
on 2026-10-13: its unit tests in each release's environment, the attach
to the real `TokenizerManager` class, the backend patch applied to each
release's published wheel, and token ids on a real Qwen3-8B tokenizer
through both forms, including the worker pool. These checks ran on the
CPU side, without a model server. In all three releases the two methods
the adapter wraps have the same signatures. The backend patch comes in
two files: `patches/sglang-0.5.19-toktier-backend.patch` for 0.5.19,
and `patches/sglang-0.5.21-toktier-backend.patch` for 0.5.20 and 0.5.21
(0.5.20 moved the `--tokenizer-backend` option to
`arg_groups/fields/serving.py`).

The attach checks the shape of SGLang's `TokenizerManager` before it
changes anything: the two methods it wraps, `_tokenize_one_request` and
`_tokenize_texts`, must exist and take the arguments it expects. On a
version where they do not, `attach_to_tokenizer_manager` leaves the
class as it is and returns `False`, the adapter records
`sglang_shape_unverified` together with the SGLang version string, and
SGLang tokenizes as it would without toktier. The ids are then SGLang's
own, and nothing is accelerated. `attach_to_tokenizer_manager` then
passes one line to its `log` callback and, when a timing log is
configured, appends an `{"kind": "attach", "status":
"sglang_shape_unverified", ...}` record with the reason, the parameter
names it found and the SGLang version.
`toktier.integrations.sglang.check_manager_shape(TokenizerManager)`
returns the same report without changing anything. When the shape
matches, the wrapper's `stats()` carries `sglang_shape: "verified"`, and
every wrapper's `stats()` carries `sglang_version`. The backend patch
has no such check, since it is applied to one release line's source;
use the patch file written for that line.

## Installing beside SGLang

toktier admits `tokenizers>=0.22.2,<0.24` and `transformers>=4.57.6,<6`,
so it installs beside SGLang with its dependencies resolved the ordinary
way:

```
pip install "sglang==<version>"
pip install toktier==0.3.0
pip check
```

SGLang 0.5.19, 0.5.20 and 0.5.21 pin `transformers==5.12.1`,
`tokenizers==0.22.2` and `torch==2.13.0`; SGLang's main branch has moved
to `transformers==5.19.0` and `tokenizers==0.23.2`. Both pairs are inside
toktier's ranges, so `pip check` has nothing to report about toktier in
either environment.
With toktier 0.3.0, `pip check` reported "No broken requirements found."
on 2026-10-14 in a new environment holding SGLang 0.5.21's two pins
(`tokenizers==0.22.2`, `transformers==5.12.1`) and in one holding SGLang
main's two pins (`tokenizers==0.23.2`, `transformers==5.19.0`), and on
2026-10-15 in the Python 3.10 environment the release is rehearsed in. In
an existing SGLang 0.5.21 environment, `pip install --dry-run
toktier==0.3.0` lists toktier and no other package to install, since
SGLang's own pins already satisfy toktier's requirements.
Older toktier releases pinned `transformers==4.57.6` and had to be
installed with `--no-deps` beside SGLang, together with `platformdirs`,
the one dependency SGLang's own dependencies do not bring; that is no
longer needed, and `pip install toktier` brings `platformdirs` itself.
If you want toktier's pinned Fastokens adapter, install
`toktier-fastokens==0.3.1.2` as well, and keep in mind that it uses the
import name `fastokens`, which is also the name of the package SGLang's
own `fastokens` backend installs. Install one of the two, not both
(`packaging/fastokens-pinned/README.md`).

What this means for toktier. Routing admission is bound to the artifact
bytes, the engine identity and the installed `tokenizers` being one of the
certified reference versions the shipped registry lists; the
`transformers` version is not part of it. The reference engine compiled
into toktier is the `tokenizers` crate at exactly 0.22.2, the release
SGLang 0.5.19 to 0.5.21 pin, so beside those releases the certified
routes are the same as in a toktier-only environment. A later
`tokenizers` release is a certified reference version only once the
parity readings taken under it ship with the registry
(`docs/contracts/registry.md` Section 2); an installed release outside
that set runs on the reference route and is reported as such
(`R_ORACLE_MISMATCH` in `explain()`, `uncertified_reference` in
`toktier doctor`).
This release ships three certified reference versions: `tokenizers`
0.22.2, 0.23.2 and 0.23.3. They cover both of SGLang's pins, 0.22.2 for
SGLang 0.5.19 to 0.5.21 and 0.23.2 for SGLang's main branch. The two 0.23
releases were certified on readings taken with each of them installed
beside `transformers` 5.19.0: the fast CPU engine on 1,437,469 documents
and the prebuilt GPU engine on sm_120 (16,540,125 documents) and on sm_89
(1,500,000 documents), every id equal to the installed release's, and the
chunk-parallel path's adversarial battery (460,000 strings over eleven
artifacts), with no difference in any comparison. The parity readings
ship in `readings/oracle_compat/`.
For every packaged artifact
except `qwen3_5_08b`, toktier executes the verified artifact file itself;
`qwen3_5_08b` declares added tokens in its configuration file, so its
loader face is materialized by the installed `transformers`. Under
`transformers` 5.12.1 that face is not the one the readings were taken
with: the 5.12.1 tokenizer class rebuilds the pre-tokenizer's split
pattern without `\p{M}`, so a letter followed by a combining mark is
split differently, and so is a mark after a symbol, such as the
variation selector of an emoji. toktier then returns the ids of the
installed Hugging Face object, the same object SGLang uses. Those ids
are outside what the certification covers, and since this release
toktier says so itself: each
certification record carries the digest of the loader face its readings
were taken on (`loader_face_sha256`, `docs/contracts/registry.md`
Section 1), and at load the facade compares it with the digest of the
face it materialized. Beside `transformers` 5.12.1, `explain()` for
`qwen3_5_08b` reports `certification.state` `reference_only` and
`effective_verdict` `unverified` (the identity is still `exact`: the
bytes are the certified ones), `planned_backend` `hf`, and under
`plan_reasons` an `R_LOADER_FACE_MISMATCH` entry for `fast_cpu` whose
detail names the recorded face (`a57b5c62...`) and the observed one
(`ffb7a28b...`); `probe.loader_face_sha256` carries the observed digest.
The ids are those of the installed Hugging Face object, as they were, and
no certified claim attaches to them. The chunk-parallel path does not
chunk it either. In the admission table as it stands its row holds
adversarial evidence only, so `explain()["chunked"]["refusal"]` reads
`chunked_evidence_incomplete:adversarial_only` under both versions; once
the row is certified, the face recorded in it refuses a different one
(`loader_face_differs_from_chunked_evidence:<digest>`). The other ten
families materialize the same face under both versions, and nothing
about their `explain()` changes.
Beside `transformers` 5.19.0, the loader SGLang's main branch pins,
`qwen3_5_08b` materializes a third face (`4d765ac5...`), which the load
check refuses in the same way, and the other ten families materialize
their recorded faces.

What it means for `toktier doctor`. Its `certified_cpu_profile_ready`
field says whether the installed engine reported its facts and the
installed `tokenizers` is one of the certified reference versions;
`certified_reference_versions` lists that set as the shipped registry
records it, and `reference_certification_reason` reads `null` inside the
set and `uncertified_reference` outside it. The `transformers` release is
not a premise of the field, so beside SGLang `doctor` reads as it does in
a toktier-only environment whenever the installed `tokenizers` is
certified, and the fields built on it (`automatic_effective_backend` and,
under `--family`, `automatic_effective_backend_below_gpu_threshold`)
follow. Beside a `tokenizers` release outside the set they answer `hf`
where they would otherwise answer `fast_cpu`, and the reason says why.
That is a statement about the installation profile, not about routing.
What a loaded tokenizer actually runs is in its `explain()`, and
`doctor --family FAMILY` lists the planner's own reasons under
`family.plan_reasons`. `doctor` does not materialize the loader face,
so the face comparison is made only when a tokenizer is loaded and
shows only in its `explain()`: beside `transformers` 5.12.1,
`family.plan_reasons` for `qwen3_5_08b` carries no
`R_LOADER_FACE_MISMATCH`, and its reasons can differ from those in the
loaded tokenizer's `explain()`.
<!-- TODO(fill): add the release-candidate reading of automatic_effective_backend and of explain() backend beside SGLang; the 2026-10-14 check ran on a build whose engine binding was not yet recertified, where both environments read hf for that reason. -->

The combinations we have checked:

- SGLang 0.5.19 with `transformers` 5.12.1 and `tokenizers` 0.22.2,
  checked on 2026-09-17 on the code that became this release: the
  adapter's identity check reads `equivalent` for `qwen3_8b`, the worker
  pool starts, and the session-path and chunked pool ids equal the
  Hugging Face ids.
- `transformers` 5.12.1 with `tokenizers` 0.22.2 against `transformers`
  4.57.6, checked on 2026-10-14 on a pre-release build of this release:
  the eleven roster families, 100,000 web documents each, the same
  documents in both environments. In each environment toktier's ids
  equal the reference built from that environment's loader face, and
  under 5.12.1 also the ids of the installed Hugging Face object, for
  every document and family. Ten families have the same loader face
  under both versions, and their ids are the same in both environments. `qwen3_5_08b` does not: its 5.12.1
  face differs as described above, its `explain()` there reads
  `reference_only` and `unverified` with `R_LOADER_FACE_MISMATCH`, and
  280 of the 100,000 documents get different ids under 5.12.1 than under
  4.57.6, 269 of them with a combining mark after a letter and 11 with
  the variation selector of an emoji. These counts are readings of that
  check; they are not among the readings that ship with the package.
- SGLang 0.5.19, 0.5.20 and 0.5.21 (each with `transformers` 5.12.1 and
  `tokenizers` 0.22.2), checked on 2026-10-13 on a pre-release build of
  this release: identity `equivalent` for `qwen3_8b`; a 390,000-character
  document encoded statelessly gives the Hugging Face ids; a two-worker
  pool starts and stops, and its chunked, session and session-append
  ids equal the Hugging Face ids, through the attach form and through
  the backend patch.
<!-- TODO(fill): add the real-server smoke test on the release candidate. -->

The risks, stated plainly. Under `transformers` 5.12.1 the live
tokenizer objects of the two Qwen families report `trim_offsets`
differently from 4.57.6. The ids are the same, and the identity check
leaves that flag out on both sides. A loader face that `transformers`
materializes itself, as for `qwen3_5_08b`, follows the installed
version, and the identity check compares two objects built by that same
version, so it does not see a change against the certified face. The
record's loader-face digest does: toktier compares the face it
materialized with the one the readings were taken on and reports the
tokenizer as reference-only (`R_LOADER_FACE_MISMATCH`) rather than
certified, with the ids unchanged; that is the `qwen3_5_08b` case
above. For an artifact whose face toktier
reads from the verified file, a later `transformers` that materialized
the supplied tokenizer differently in a way that changes ids would make
the identity check find the two sides different, and the wrapper would
hand that tokenizer to the Hugging Face object for its whole life, with
the reason in `stats()["identity"]`. That outcome is safe but slow, and
it is visible.

## What we are not proposing

The radix cache, the session references and the KV layout key on token ids;
our output is their input. We do not propose changing how SGLang keys or
evicts KV, nor the `session_params` continuation path (server-side
concatenation), which sends a delta rather than a transcript and is served
by content lookup, not by a keyed session.

## Measurements

Setup: Qwen3-8B (bf16) on one RTX PRO 6000 Blackwell (96 GB), SGLang 0.5.19
with `--context-length 131072` (YaRN), `--enable-session-radix-cache`, the
default `flashinfer` attention backend; HTTP process pinned to one NUMA node
(16 cores); three independent server processes per arm; warm turns only;
TTFT is the client's first streaming chunk; paired bootstrap 95% CI on the
median difference against the unpatched server (`hf`).

Sequential paired TTFT (one session, the whole transcript re-sent with
+400 characters per turn, 30 warm turns per context):

| context | hf | toktier (session_id) | toktier (content lookup) | Δ median |
|---|---|---|---|---|
| 8K tokens (31K chars) | 29.3 ms | 29.2 ms | 29.2 ms | −0.4% [−0.9, +0.0] |
| 32K tokens (125K chars) | 55.4 ms | 37.6 ms | 37.5 ms | −32.0% [−34.6, −31.4] |
| 100K tokens (390K chars) | 135.1 ms | 53.6 ms | 53.6 ms | −60.3% [−60.6, −60.1] |

The removed cost is the per-request full re-encode in the HTTP process
(about 18 ms at 125K characters and 82 ms at 390K with the Hugging Face
tokenizer); the toktier component per request is 0.25 / 0.65 / 1.9 ms on
the session path and 8.7 ms at 390K characters when stateless. SGLang's
built-in `fastokens` backend reaches the same latency on this shape, with
the caveat in the next paragraph. `--tokenizer-worker-num 4` on the
unpatched server changes nothing here (a single request's encode is not
parallel).

Concurrent sessions (N sessions replaying transcripts that grow from 8K to
48K tokens over 30 turns in ~1.4K-token increments, each session pausing a
jittered 1–3 s between turns): the median TTFT falls by 13.0% [−17.7, −8.9]
at N=4 and 11.5% [−15.6, −8.6] at N=8, with the p99 down 20% and 6%. Without
the pause (a saturating closed loop) the median moves the other way while
completed turns per second, end-to-end time and the p99 improve — requests
that no longer wait for tokenization reach the scheduler together and share
prefill steps — so we report the paced numbers. An open-loop burst (12
sessions with 70K-character prefixes, Poisson 15 events/s, +400 characters
per event) gives −36% at the median and −23% at the p99 measured from the
scheduled arrival.

Correctness during these runs: every toktier request's `prompt_tokens`
equalled the local Hugging Face count, and the shadow check (on for the
smoke runs) reported 0 differences in 177 compared requests.

The `fastokens` backend gave a different answer on the same transcripts.
Replayed in serving order on 2026-09-17, with SGLang 0.5.19 and the
`fastokens` release that `sglang[fastokens]` installs, 0.1.2 (SGLang
declares `fastokens>=0.1.1,<0.2.0`), the backend returned a different id
stream from the Hugging Face reference for 78 of 279 transcripts, one or
two tokens more in each. In every case a run of `\n`, spaces and `\n`
that ends the previous turn is split where the reference keeps it whole,
and the split carries into every later turn of that transcript. The same
text encoded once in a fresh process gets the reference ids, so a test
that encodes each text once does not see the difference. We mention it
because a latency comparison between the two backends only means
something with that difference on the table. SGLang 0.5.20 and 0.5.21
declare the same `fastokens` range, so `sglang[fastokens]` still
installs 0.1.2 (checked on 2026-10-13). We have not re-run this replay
on those versions.
