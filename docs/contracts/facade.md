# Facade contract (0.x)

Status: **0.x stable**. The names and semantics in this document are
kept for the whole 0.x line; internals behind them move freely. Where
this document and `api.md` describe the same call differently, this
document is the operative surface for 0.x and `api.md` records the 1.0
target shape; the differences are listed in Section 7 so none of them is
an accident.

Guiding rule, inherited from `api.md`: correctness first. Every facade
path returns token ids equal to a from-scratch encode by the pinned
reference oracle. Store layers accelerate; they never answer
differently. Whenever a stored entry cannot be located, verified, or
extended, the call degrades to a full reference encode -- we prefer a
miss over a wrong result.

## 1. Surface

```python
tok = toktier.load(family, *, store=None, device="cpu")
enc = tok.encode(text)                       # Encoding; enc.ids
enc = tok.encode(text, session="chat-42")    # named session entry
enc = tok.encode(text, lookup="auto")        # content lookup (the default)
ids = tok.encode_batch(texts)                # list[Encoding]
txt = tok.decode(enc.ids)
tok.explain()                                # plan, reasons, counters
```

- `load(family, *, store=None, device="cpu", config=None, policy=None,
  manifest=None, cache_budget_bytes=None) -> Tokenizer`. The extra
  keywords inject a `Config`, a routing policy, an artifact manifest, or
  the in-process cache budget; none of them reads the process
  environment (`config.md` owns environment capture).
- `Tokenizer(family, config=None, *, policy=None, ...)` keeps the frozen
  constructor shape of `api.md` Section 2; the facade keywords are
  additive and keyword-only.
- `device="cpu"` is the only accepted value in 0.x. The GPU engine
  remains an explicit, separate entry point; asking the facade for it
  raises `UNSUPPORTED_CONFIG` rather than being silently ignored.
- Artifacts resolve through the shipped manifest and the verified cache
  (per-file sha256, `registry.md` Section 4). Routing follows the
  standing policy semantics: the plan is fixed at construction, runs the
  reference backend today, and adopts a certified accelerated CPU
  backend automatically when one exists, with the same fallback
  accounting.
- `Encoding` is immutable and carries at least `.ids`
  (`tuple[int, ...]`). Additional diagnostic attributes may appear
  within 0.x; `.ids` is stable.

## 2. `encode` semantics

```python
def encode(self, text, *, session=None, lookup=None,
           add_special_tokens=False) -> Encoding
```

- Default output is the **pre-postprocessor core stream**
  (`add_special_tokens=False`), matching what stores hold
  (`store-format-v1.md` Section 1).
- `session=<id>`: the store holds one `(text, ids)` entry per session
  id. If the stored text is a **prefix** of the input, only the
  remainder is appended through the store's certified append machinery
  and the entry advances; if it is not a prefix, the input is encoded
  whole and the entry is overwritten. Any ambiguity or store failure
  runs the full reference encode. The returned ids are bit-identical to
  a from-scratch reference encode in every case.
- `lookup="auto"` (the default when no session is given): content
  lookup. The store's checkpoint index proposes the longest stored text
  whose endpoint digest matches a prefix of the input; the proposal is
  then **byte-verified against the stored text** (the anti-collision
  hard gate) before anything is served from it. A verified full-prefix
  hit appends the remainder; anything else is a miss and a full encode.
  Inputs below a small size floor bypass the store. Auto entries are
  capacity-capped; named session entries are never displaced by auto
  traffic.
- `lookup="off"`: skip the store entirely. `session` and `lookup` do
  not combine.
- `add_special_tokens=True` is served by the plain routed path.
  Combining it with an explicit `session` or `lookup="auto"` raises
  `UNSUPPORTED_CONFIG` (the stored stream is the core stream; we reject
  rather than silently re-encode).
- `encode_batch` runs the plain routed path row by row; content lookup
  is a single-document affair in 0.x.

## 3. Store, index, and cache layers

Three layers with distinct loss semantics, named honestly:

1. **Store entries** (`store=<directory>`): persistent state, one store
   format v1 record per entry (`store-format-v1.md`, unchanged and
   frozen), written atomically. Deleting the directory loses sessions.
   Loading a record re-verifies it byte-level and re-encodes its tail
   through the current engine; anything that fails is a miss, never a
   wrong answer. A store written under a different semantic fingerprint
   (artifact, oracle, or engine semantics changed) is refused with
   `SESSION_STATE_MISMATCH`.
2. **Checkpoint index** (sidecar file): a derived cache mapping content
   digests (endpoint plus geometric byte positions) to entries. Missing
   or corrupt means rebuild from the records or serve misses; it is
   never authoritative -- digests propose, bytes decide.
3. **In-process cache** (budget via `cache_budget_bytes`, default
   128 MiB): resident texts and native handles for speed. Eviction only
   costs a reload (persistent stores) or a re-encode (in-memory
   stores) -- never correctness, never persisted state.

Without `store=`, sessions and lookup state live in this process only
and follow the same rules minus durability.

Concurrency: one writing process per store directory. Records are
individually self-verifying, so a reader never observes a torn or
half-trusted entry; cross-process write coordination is not offered in
0.x.

## 4. Errors

No new codes. The facade raises the existing `errors.md` codes
(`ARTIFACT_NOT_FOUND`, `ARTIFACT_HASH_MISMATCH`, `UNSUPPORTED_CONFIG`,
`SESSION_STATE_MISMATCH`, `STORE_FORMAT_UNSUPPORTED`, ...). Plain
argument misuse (an unknown `lookup` value, conflicting arguments) stays
a plain `ValueError`. Store-side failures on the read path never raise:
they degrade to a counted miss and a full encode.

## 5. Diagnostics

`explain()` returns the routing layer's explanation of the active plan
(the requested routing policy under the key ``routing_policy``, backend,
fallback chain, plan reasons, experimental waivers, a ``certification``
block, and a probe summary), runtime fallback counts, and -- once the
store has been touched -- store counters (hits, appends, overwrites,
misses, collision rejects, degradations, rebuilds, evictions). The
routing policy and the certification state are deliberately separate
keys: their vocabularies share the word ``certified`` while answering
different questions. Because the 0.x facade plans against an empty
registry view (no accelerated backend is adopted; see Section 1), the
``certification`` block reads ``uncertified`` -- no certification
identity was consulted for the request -- and the per-family evidence
records remain in the support matrix and the shipped support registry.
The facade likewise supplies no device probe, and the report says so:
the probe summary carries ``devices_probed: false`` and the GPU option
is recorded as ``R_BACKEND_UNAVAILABLE`` (modules absent) or
``R_ACCELERATOR_NOT_ADOPTED`` (modules present, no probe supplied) --
never as a claim about the machine's hardware. The installed reference
oracle is visible as ``probe.oracle_package`` / ``probe.oracle_version``;
the certified version set lives in the shipped support registry.

"Not adopted" and "not available" are separate statements, and the
report keeps them separate. The ``kernel_deliveries`` block carries,
per kernel delivery (``prebuilt`` / ``jit``): the read-only shipped
facts (whether the prebuilt fatbin and the JIT sources are installed
-- the same answer ``toktier doctor`` gives, through the same helper),
whether this process loaded that delivery, the shipped identity digest,
and the per-architecture certification status map of this artifact's
record in the shipped support registry (with ``driver_min`` for the
prebuilt delivery). The registry lookup is read-only and
reporting-only: it feeds this block and grants nothing -- planning
still runs against the empty registry view, and the ``certification``
block still answers for the active request only. An absent record
reports ``status: null`` and an empty architecture map: the absence of
a claim, not a claim of absence. The loaded-process counterpart of
this block lives on the explicit GPU engine
(``GpuEngine.explain()`` / ``binding_set()``, `docs/gpu-jit.md`
Section 9).

One store-counter definition worth spelling out: ``*_hits`` count exact
whole-text reuse (the stored text equals the input). A served prefix
extension -- the store's main job -- counts in ``*_appends`` instead,
so an append-mostly workload legitimately shows zero hits next to
growing appends; ``*_appends`` are successes, not misses. Keys are
informational and may grow within 0.x; the method and the plan/reason
presence are stable.

## 6. Record reader

`toktier.records.decode_record(record: bytes) -> RecordView` is the
public read-side counterpart of `store-format-v1.md`: it decodes and
verifies one record in pure Python, raising `STORE_CORRUPT` for
structural or integrity failures and `STORE_FORMAT_UNSUPPORTED` for
well-formed-but-newer records, in the contract's explicit-verify split.

## 7. Deltas against `api.md` (deliberate, 0.x)

- `encode` returns an `Encoding` (with `.ids`), not a bare `list[int]`.
- `add_special_tokens` defaults to `False` (core stream first;
  `api.md` Section 3 records `True` for the 1.0 shape). One of the two
  defaults will be reconciled before the API version axis moves.
- `decode` exists here; `api.md` Section 7 lists it as out of scope for
  the frozen v1 set.
- The `session()` context-manager surface of `api.md` Section 5 is not
  part of the facade; the facade's `session=` keyword covers the
  store-backed use. `SessionUpdate` (Section 5.1) ships as a value
  object with its splice invariant enforced at construction.
