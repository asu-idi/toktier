# Using toktier with NVIDIA Dynamo

This page describes two ways toktier can be used together with
[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo). The first needs no
changes on either side and runs against the shipped 0.2.x Python facade. The
second is a sketch we would like to discuss with the Dynamo maintainers rather
than a plan we would carry out on our own.

## What Dynamo already does

The Dynamo frontend tokenizes each request in its Rust preprocessor, and since
2026-06-30 it enables an in-process prefix cache by default
([`ai-dynamo/dynamo` PR #11078](https://github.com/ai-dynamo/dynamo/pull/11078),
implementation in
[`ai-dynamo/frontend-crates`](https://github.com/ai-dynamo/frontend-crates)
`tokenizers/src/cache/`). It keys entries by a BLAKE3 digest of a text prefix,
stores the accumulated token ID vector, and cuts prefixes at special-token
boundaries — a place where `encode(prefix) + encode(suffix)` equals
`encode(prefix + suffix)` by construction. The source notes the reasoning
plainly: no fallback to whitespace or punctuation, better to miss than to
corrupt. The cache lives as long as the tokenizer instance that owns it, and the
crate ships a dedicated cache-correctness test.

We read that design as sound, and the two properties toktier adds are
complementary rather than competing: **persistence across processes and
restarts**, and **re-verification when a stored prefix is used**, which in turn
allows splice points that are not limited to special-token boundaries. Nothing
below asks for the existing cache to change.

## Surface A — feed token IDs, no changes to Dynamo

Dynamo already accepts pre-computed token IDs: `/v1/completions` takes an
integer array as `prompt`, and the preprocessor honours `nvext.token_data` by
skipping tokenization. An application that keeps its own session state can
therefore compute IDs with toktier and hand them over.

```python
from openai import OpenAI
import toktier

client = OpenAI(base_url="http://dynamo-frontend:8000/v1", api_key="unused")

# Same artifact the server serves. `store=` makes the session state
# persistent, so it survives this process.
tok = toktier.load("deepseek_v3", store="./toktier-store")

# The argument is the complete rendered transcript, not the delta. toktier
# verifies the stored prefix byte for byte and encodes only what is new;
# `enc.ids` is the full token id sequence for `transcript`.
transcript = rendered_history + new_turn
enc = tok.encode(transcript, session=conversation_id)
ids = list(enc.ids)

client.completions.create(model="deepseek-v3", prompt=ids, max_tokens=256)
```

The ids are the whole sequence on every turn, which is exactly what
`prompt=` wants, and they equal a from-scratch encode of the same text:

```python
assert enc.ids == tok.encode(transcript, lookup="off").ids
```

`tok.explain(summary=True)` reports which backend actually served the call;
a served append shows the bounded repair rather than a full re-encode.
A client that would rather keep the previous turn's ids in its own buffer
and splice only the repaired suffix can use the Rust crate, whose
`Session::append` returns a `TokenPatch` instead of the full sequence
(see [`../rust-api.md`](../rust-api.md)).

The chat-completions path carries the same data through the vendor extension
field:

```jsonc
{
  "model": "deepseek-v3",
  "messages": [ /* ... */ ],
  "nvext": { "token_data": [128000, 3923, 374 /* ... */] }
}
```

Two practical notes. The tokenizer artifact on both sides has to be the same
bytes — toktier pins and verifies its artifact by SHA-256, and
`toktier inspect deepseek_v3` prints the pinned digest, so the check is a
one-liner in a deployment script. And
because the client now owns template rendering, whatever the server would have
applied has to be applied before encoding; the session stores the token stream
produced before the post-processor precisely so that BOS/EOS handling stays a
read-time decision. That is also why the call above leaves
`add_special_tokens` at its default `False`: store-backed paths hold the core
stream, and asking for postprocessing on one raises `UNSUPPORTED_CONFIG`
rather than serving a stream it cannot prove.

This path is where we would start a demo: it exercises the interesting part
(session state and verified reuse) without asking anyone to merge anything.

### A 1.0 sketch, for contrast

The frozen 1.0 API (`docs/contracts/api.md` Section 5) also describes a
context-manager session that returns a delta object. It is **not** part of
the 0.x facade (`docs/contracts/facade.md` Section 7 records the deviation),
so the following does not run against 0.2.x and is included only to show
where the surface is heading:

```python
# Target sketch for 1.0 — not available in 0.2.x.
with tok.session(store="./toktier-store", session_id=conversation_id) as s:
    update = s.append(new_turn)      # SessionUpdate: replace_from, replacement_ids
    ids = list(update.all_ids)
```

Until then, the complete-transcript call above is the supported Python
surface, and the Rust `TokenPatch` is the supported delta surface.

## Surface B — a persistent layer next to the in-process cache

`CachedTokenizer` is a decorator over `Arc<dyn Tokenizer>`, and the trait
surface it needs is small: `Encoder` (`encode`, `encode_batch`,
`encode_segments`), `Decoder` (`decode`), and `Tokenizer`, which adds
`validate_prefix_cache` — whose default implementation refuses, so a backend has
to opt in explicitly. A persistent layer with the same shape can sit either
inside or outside the existing cache:

```rust
// sketch: a persistent, verified layer wrapped by the existing in-process cache
let inner: Arc<dyn Tokenizer> = tokenizer_from_hf(&model_path)?;
let persistent = ToktierStore::open(&state_dir)?.layer(inner);  // verified on hit
let tokenizer = CachedTokenizer::new(Arc::new(persistent), specials, budget)?;
```

In that arrangement the in-process cache keeps serving the hot path at memory
speed, and the persistent layer catches what it cannot: a restarted or
rescheduled frontend replica, a conversation whose turns land on different
replicas, and appends inside a long span that contains no special-token
boundary. A miss in the persistent layer costs one lookup and a verification;
by construction it returns nothing it cannot re-check.

Questions we would want to settle with the maintainers before writing anything
substantial, rather than after:

- whether a persistent layer belongs in the frontend at all, or whether the
  cleaner place is client-side (Surface A) with the frontend left alone;
- what `validate_prefix_cache` should mean for a layer that verifies on hit, and
  how it interacts with the `add_special_tokens` guard added in
  `frontend-crates` #114/#115;
- whether the loss of offsets and attention mask under the current cached path
  is a constraint we should preserve, or an opportunity to carry byte spans
  through;
- how state directories, capacity limits and per-session deletion should be
  expressed in a Kubernetes deployment.

## What we are not proposing

The KV router, the KV block manager and the hashing libraries key on token IDs.
Our output is their input, not their peer; wiring a text-keyed store into those
paths would confuse two different notions of identity. We would leave them
untouched.

## Correctness posture, briefly

Two invariants make it reasonable to put a persistent cache in a serving path at
all. A wrong key must miss: the key includes the tokenizer content hash, the
engine identity and the configuration, so a tokenizer swap cannot produce a
stale hit. And a hit must be verified: the record keeps an explicit text tail and
a block hash chain, both re-checked before any stored IDs are used, with a failed
check counted as a miss. The hash chain detects corruption; it is not an
authentication mechanism, and we do not describe it as one. Evidence for the
underlying token-level equality claims ships with the release in machine-readable
form (evidence/evidence_manifest.json).

## Getting in touch

If any of this is interesting, an issue on our repository or a note on a Dynamo
discussion thread works; we are happy to bring benchmarks for whichever
workload shape is most useful, and equally happy to conclude that Surface A is
where this belongs.
