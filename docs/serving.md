# Serving guide: which encode call for which request

This page is for a process that serves many requests through one loaded
`Tokenizer` -- an inference server's tokenizer worker, a gateway, a
batch job. It says which call to make for each request shape, what each
call costs beyond the encode itself, and how to encode a very long
request on several cores without leaving the exact-ID contract. The
contract reference is [`contracts/facade.md`](contracts/facade.md); the
numbers quoted here were taken on one 16-core NUMA node of an EPYC 9115
host with the certified CPU engine, on English prose, and are meant as
ratios rather than as a benchmark (the protocol for those is
[`benchmarks.md`](benchmarks.md)).

## 1. Four request shapes, four calls

| The request | The call | What the store does |
|---|---|---|
| A turn of a conversation the caller can name | `encode(text, session=key)`, or a `Session` object | Verifies the stored transcript is a prefix of `text` and appends only the remainder |
| A turn of a conversation the caller cannot name | `encode(text)` (`lookup="auto"`) | Asks the content index whether a stored text is a prefix of `text` |
| A single-shot request that will not be extended | `encode(text, lookup="off")` | Nothing |
| A single-shot request of hundreds of thousands of characters or more | `toktier.chunked.encode(tokenizer, text)` | Nothing (Section 3) |

The third row is the one worth stating plainly. An omitted `lookup`
means `"auto"` for a core-stream call, and `"auto"` is the right default
for a library that does not know what comes next: it costs nothing when
the store is empty and pays for itself as soon as one request extends
another. But on a request that is never extended it is pure overhead,
and not a small one: the content index has to digest the input so that
a later request can find it, and on a 1 M-character input that state
construction is 55-79 % of the encode itself (24 ms of encode against
43 ms with the index). A serving path that knows a request is
single-shot should say so with `lookup="off"`. The shipped SGLang
adapter does exactly that in its stateless mode
([`integration/sglang.md`](integration/sglang.md)).

The default is not changing. `lookup="auto"` stays the default of
`encode` because the facade cannot tell a single-shot request from the
first turn of a long conversation; the serving layer can.

## 2. Appending to a session: two answers

`Session.append(text)` returns a `SessionUpdate` carrying the whole
stream (`all_ids`) together with the splice (`replace_from`,
`replacement_ids`). That is the frozen 0.x shape, and it stays. A
serving process usually holds the stream itself -- it is what it sends
to the model -- and only needs the splice, so since 0.3.0
`Session.append_patch(text)` returns a `SessionPatch` with the splice
alone and no copy of the stream:

```python
with tokenizer.session(session_id=conversation_id, text=history) as session:
    stream = list(session.ids)
    ...
    patch = session.append_patch(new_turn)
    stream[patch.replace_from :] = patch.replacement_ids
```

The two calls report the same `replace_from` and the same replacement,
and `Session.ids` still answers with the full stream when asked. What
`append_patch` saves today is the update object and its copies; the
engine's own answer on the native request path is still the whole
stream as one Python list, so the call is not yet proportional to the
appended turn (measured: 7.6 ms for `append` against a stream of
220 K tokens, of which 7.2 ms is that list). A patch-shaped native
return is registered for a later release; the Python surface is the
one that will not change when it lands.

## 3. Long requests: certified chunk-parallel encode

A single encode runs on one core. For a request of several hundred
thousand characters that is tens of milliseconds on the certified CPU
engine, and it is the one place in this library where more cores could
help a single request. `toktier.chunked.encode` does that without
touching the exact-ID contract:

```python
from toktier import chunked

encoded = chunked.encode(tokenizer, text)          # k from the table
encoded = chunked.encode(tokenizer, text, k=16)    # explicit chunk count
assert encoded.ids == tokenizer.encode(text, lookup="off").ids
```

How it keeps the contract. The document is cut only at *certified
stable boundaries*: positions at which the pre-tokenizer of the
certified byte-level BPE families cannot carry a pre-token across the
cut. That predicate is the synchronizing-transition rule the
session-repair witness already uses (eight class pairs over the frozen
O/S/L/N/M character table, minus two carve-outs: a line break after
punctuation, and an apostrophe after letters), plus a guard that keeps
every cut outside any added-token literal of the loaded artifact. The
chunks run through the tokenizer's own batch path -- the same plan, the
same per-input guards, the same fallback ledger -- and their id rows are
concatenated. Nothing is matched or retried at the seams, because no
seam exists at which the encoding could differ. A position where no
certified cut can be found is simply not cut: fewer chunks, never a
different id.

What decides whether it is worth it:

- **Admission.** Only a tokenizer whose family and artifact digest have
  an entry in the certified repair roster may be chunked (those are the
  eleven byte-level BPE artifacts the sync profile was certified on),
  and only when its loader face carries no normalizer, the empty
  normalizer sequence, or NFC with every added token matched on the
  input text (`normalized: false`) -- the normalizer shapes the cut rule
  reasons about. Any other tokenizer is encoded serially, and the result
  says so in `serial_reason` (`R_CHUNK_FAMILY_UNSUPPORTED`; the `chunked`
  block of `explain()` names the refusal).
- **Normalization.** The cut rule speaks about the text the
  pre-tokenizer sees, which for an NFC family is the normalized text,
  not the input. A letter such as U+0958, which NFC writes as a letter
  followed by a combining mark, shows why that matters: as written it
  forms a letter-to-punctuation transition with a following slash;
  normalized, the mark and the slash share one pre-token. For an NFC
  family a candidate cut is therefore accepted only when it is also a
  certified cut of the normalized text: the input on each side is
  anchored at the nearest stable split point (a character that is not a
  mark, is unchanged by NFC on its own, does not reorder against a
  combining mark on either side and does not compose after a Hangul
  jamo or syllable -- decided with the artifact's own normalizer, the
  reference engine's object, so there is no Unicode-version skew; ASCII
  qualifies without a test), no added-token literal touches the
  anchored window (the reference normalizes each literal-free segment
  on its own), the normalizer maps the window to the concatenation of
  its two halves, and the rule holds on the two normalized characters
  that meet at the cut. Three stable characters around a cut satisfy
  this without any window, the common case in prose, code and CJK text.
  A candidate that fails is counted (`norm_rejects`) and the search
  moves on; the normalizer is used for equality only, and every
  character class is read from the frozen property table.
- **The chunk table.** With `k=None` the chunk count follows a
  staircase measured on three families and four corpus types: one chunk
  below 256 K characters, 4 up to 1 M, 16 up to 4 M, 32 above, capped
  by one chunk per 64 KiB of payload (the native batch path opens at
  most one worker per 64 KiB, so more chunks than that only add
  concatenation work). Below 256 K characters a single encode is
  already as fast as the batch path, which is why the table says one
  chunk there.
- **Cores.** The parallelism is the native engine's batch worker pool,
  which is sized by the Rayon thread count at first use
  (`RAYON_NUM_THREADS`; the default is the core count of the process's
  affinity). A process started with `RAYON_NUM_THREADS=1` gets the
  chunks encoded one after another -- still exact, not faster. The
  workers are forked lazily on the first batch that needs them, at
  roughly 20 ms per worker, once per process; a serving process should
  send one long batch at start-up rather than pay that inside a
  request.
- **What it saves.** On 1 M-character English prose with 16 workers,
  8.6 ms against 26.5 ms serial for the slower of the certified engines
  (3.1x), and 6.9 ms against 9.8 ms for the faster ones (1.4x): the
  remaining cost is the Python boundary, where each chunk's ids are
  handed over and concatenated. A per-input guard that would have
  routed a whole document to the reference engine now routes one chunk,
  which on a document with one such passage turns a reference-speed
  encode into one chunk at reference speed (162 ms to 15 ms in the
  measurement behind the table).

Diagnostics. Once `chunked.encode` has served a tokenizer, its
`explain()` report carries a `chunked` block: the admission decision and
the normalizer shape, calls, how many were chunked, serial fallbacks by
reason code (`R_CHUNK_NO_CUT`, `R_CHUNK_FAMILY_UNSUPPORTED`), positions
given up, literal and normalization rejections, and the last call's
facts. These counters live in that block and not in
the routing ledger's top-level `fallback_counts`, because a serial
fallback here runs the same certified backend the chunks would have
used: nothing fell back between backends
([`contracts/routing.md`](contracts/routing.md) Section 5.2).

Two things the chunked call does not do. It does not consult the store
(it is a stateless call by construction; use a session for a transcript
that will grow), and it returns the core stream only -- special tokens
are a read-time template around it, as for sessions.

Native chunk encoder. The same plan --
position for position, including the literal guard and the
normalization-stability condition -- is also implemented inside the
native runtime (`crates/toktier-routing-core/src/chunked.rs`), where the
chunks are encoded on a persistent pool of corrected-Gigatoken workers
(checked out per chunk, no prefix sharing between concurrent callers,
pool size independent of the chunk count) and concatenated before the
ids cross into Python. `chunked.encode(..., native=True)` and
`Tokenizer.encode(text, parallel="auto" | k)` select it; the default
of both remains the Python path above. `chunked.warm(tokenizer, workers=,
text=)` forks and pre-warms the pool ahead of the first request;
`chunked.admission(tokenizer)` reads the native planner's admission
facts, and `ChunkedEncoding.native_facts` carries the per-call row
sources and phase timings. This build is not a certified release: its
fast_cpu source identity has moved, so the corrected CPU route is
admitted only under the experimental policy until the identity is
registered.

## 4. Process shape

- A `Session` object has one owner; the tokenizer itself may be shared
  by threads, and its native request path releases the GIL for the
  duration of an encode.
- The session store is one directory, one writing process. A pool of
  tokenizer workers with a persistent store should give each worker its
  own directory (the SGLang adapter appends the pid); sessions are
  therefore worker-local, and a request must be routed to the worker
  that holds its session.
- Batch encodes share the worker pool: two threads calling
  `encode_batch` at once take turns on the same workers rather than
  doubling them.
