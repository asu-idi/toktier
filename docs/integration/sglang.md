# Using toktier with SGLang

This page describes two ways toktier can be used together with
[SGLang](https://github.com/sgl-project/sglang) (written against v0.5.19).
The first needs no changes on either side and runs against the shipped
Python facade. The second is a small tokenizer-backend patch, shaped like
the `fastokens` backend SGLang already ships, that we would rather discuss
with the SGLang maintainers than carry on our own. Both were exercised on a
Qwen3-8B server; the numbers quoted below come from that setup and are
described in the last section.

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

`patches/sglang-0.5.19-toktier-backend.patch` (about fifty added lines)
adds `--tokenizer-backend toktier`:

- `server_args.py`: the choice and its help text;
- `utils/hf_transformers/tokenizer.py`: after the post-load fixes,
  `get_tokenizer` wraps the loaded tokenizer with
  `toktier.integrations.sglang.wrap_tokenizer`;
- `managers/tokenizer_manager.py`: `_tokenize_one_request` binds the
  request's `session_id` (else `cache_salt`) in a context variable around
  the tokenize call, so the backend can serve the transcript as a verified
  prefix append.

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
imported. That is how the numbers below were taken; the patch produces the
same object through the supported option.

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

Each served request is recorded in the timing log with `how: "chunked"` and
the fields `chunked_k`, `chunked_chunks` and `chunked_serial_reason` (`null` when the
chunks ran; a reason code such as `R_CHUNK_NO_CUT` when the document was
encoded serially after all), and the stats file carries the tokenizer's
`chunked` diagnostics block. Correctness posture is unchanged: the chunked
result equals the from-scratch reference encode (for an NFC family such
as qwen3_8b, cuts are made only where they are also cuts of the
normalized text; `docs/serving.md` Section 3), and the shadow check
compares it like any other.

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

Form Z also patches `_tokenize_texts`: a single string (or a list of
strings) is awaited through `SGLangToktierTokenizer.encode_async`, so the
event loop stays free while the pool works and several requests are in
flight at once; cross-encoder inputs and other shapes take the original
path, and other entry points (`__call__`, the dynamic-batch executor)
use the pool synchronously.

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
object SGLang hands the wrapper is the tokenizer toktier loaded: the
supplied object may carry another revision, other loader options, or
live vocabulary changes (`add_tokens` after loading is the concrete
case, and a token added that way is one id on the supplied object and
several on toktier's). So before the wrapper accelerates anything it
serializes the supplied object's fast backend (`backend_tokenizer.
to_str()`, which includes tokens added after loading) and the loader
face toktier executes against, and compares the sections that decide
the core id stream: model, normalizer, pre-tokenizer, added tokens, and
the truncation and padding state (`check_core_identity`). When the two
are not equivalent, or one side cannot be serialized, the wrapper
delegates every call to the supplied tokenizer for its whole life; the
outcome is in `stats()["identity"]` (`status`, `reason`, the differing
`sections`) and in the init record of the timing log, and `mode` then
reads `passthrough`. `TOKTIER_SGLANG_IDENTITY=skip` turns the check off
for a tokenizer that cannot be serialized, at the operator's risk (the
shadow check is then the only guard).

Attribute writes follow attribute reads: an assignment SGLang makes on
the tokenizer -- `tokenizer.chat_template = ...` in its template manager
when a chat template is installed -- lands on the wrapped object, where
the delegated `apply_chat_template` reads it.

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
smoke runs) reported 0 differences in 177 compared requests. On the same
transcripts, replayed in serving order, the `fastokens` backend (0.1.2, the
version SGLang pins) produced a different id stream for 78 of 279
transcripts (+1 or +2 tokens): a `\n<spaces>\n` run at the previous turn's
end is split, and the split is carried into later turns; the same text
encoded once in a fresh process is correct, so single-shot tests do not
see it. We mention it because a latency comparison between the two
backends is only meaningful with that difference on the table.
