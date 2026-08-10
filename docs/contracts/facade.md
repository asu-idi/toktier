# Facade contract (0.x)

Status: **0.x stable**. The names and semantics in this document are
kept for the whole 0.x line; internals behind them move freely. Where
this document and `api.md` describe the same call differently, this
document is the operative surface for 0.x and `api.md` records the 1.0
target shape; the differences are listed in Section 7 so none of them is
an accident.

Guiding rule, inherited from `api.md`: correctness first. Under `CERTIFIED` and
`REFERENCE`, every facade path returns token ids equal to a from-scratch encode
by the pinned reference oracle. Store layers accelerate; they never answer
differently on those policies. An explicitly selected Fastokens adapter under
`EXPERIMENTAL` is outside that guarantee and labels itself accordingly.
Whenever a stored entry cannot be located, verified, or extended, the call
degrades to the active certified full-encode route -- we prefer a miss over a
wrong result. That route may be corrected Gigatoken or GPU, but its returned
ids must still equal the reference stream.

## 1. Surface

```python
tok = toktier.load(
    family,
    *,
    store=None,
    device="auto",
    repair_backend="auto",
    gpu_delivery="auto",
    gpu_min_bytes=65536,
)
tok = toktier.from_pretrained(model_repo, revision=None, **the_same_keywords)
enc = tok.encode(text)                       # Encoding; enc.ids
enc = tok.encode(text, session="chat-42")    # named session entry
enc = tok.encode(text, lookup="auto")        # content lookup (the default)
ids = tok.encode_batch(texts)                # list[Encoding]
txt = tok.decode(enc.ids)
tok.explain()                                # plan, reasons, counters
```

- `load(family, *, store=None, device="auto", config=None, policy=None,
  manifest=None, cache_budget_bytes=None, repair_backend="auto",
  gpu_delivery="auto", gpu_min_bytes=65536) -> Tokenizer`.
  The extra
  keywords inject a `Config`, a routing policy, an artifact manifest, or
  the in-process cache budget; none of them reads the process
  environment (`config.md` owns environment capture).
- `from_pretrained(repo_id, *, revision=None, store=None, device="auto",
  config=None, policy=None, manifest=None, cache_budget_bytes=None,
  repair_backend="auto", gpu_delivery="auto", gpu_min_bytes=65536) ->
  Tokenizer` is the model-repository entry point. The repository id is a
  lookup hint, never a certification identity. For one of the 210 audited
  sibling repositories, omitted `revision` selects the full audit revision;
  for a canonical repository it selects the artifact-manifest revision; for
  an unknown repository it selects `main`. An explicit revision overrides the
  lookup choice, but the bytes resolved at that revision still decide routing.
- `from_pretrained` hashes the exact resolved `tokenizer.json` (or the recorded
  Kimi source file) before construction. An exact packaged-anchor digest or an
  exact entry in the root-digested sibling table may select a packaged
  canonical family. Canonicalisation/serialisation siblings execute that
  canonical artifact; their alternative serialisation is not passed directly
  to an accelerated engine. Unknown or drifted tokenizer content is imported
  into the verified cache under a content-derived id and is eligible only for
  HF. Repository-name-only admission is impossible. Source-only Kimi rows fail
  with `ARTIFACT_NOT_FOUND` until their conversion artifact is packaged.
- `load(family)` retains its frozen meaning: `family` is a registry family id,
  not a model repository. It remains the direct and air-gapped entry point.
- `Tokenizer(family, config=None, *, policy=None, ...)` keeps the frozen
  constructor shape of `api.md` Section 2; the facade keywords are
  additive and keyword-only.
- `device="auto"` is the default: outside the `REFERENCE` policy, the facade
  probes a GPU runtime when one is installed, adopts a certified GPU route for inputs at least
  `gpu_min_bytes` UTF-8 bytes long, and uses the next eligible backend below
  that crossover. `device="cpu"` disables GPU adoption and device probing;
  `device="cuda"` requires an eligible GPU route and raises
  `BACKEND_UNAVAILABLE` if the planner cannot open one.
- `gpu_delivery="auto"` maps the `gpu` installation profile to the shipped
  prebuilt delivery and the `gpu-jit` profile to local JIT. The explicit
  values `"prebuilt"` and `"jit"` take precedence over profile detection.
- Artifacts resolve through the shipped manifest and the verified cache
  (per-file sha256, `registry.md` Section 4). Routing follows the
  standing policy semantics: the plan is fixed at construction and adopts the
  GPU and corrected Gigatoken CPU backends only when their complete binding
  sets match certified records. For the CPU backend that includes the exact
  artifact, oracle, engine version, integrated module, source digest, release
  flags, exact rustc, patch, and repair configuration. An ineligible route is skipped with its reason recorded;
  the immutable fallback chain always ends at the reference backend.
- `repair_backend="reference"` disables session repair acceleration.
  `repair_backend="fastokens"` requires `policy="experimental"`, re-encodes
  the full session, and carries no exact-ID guarantee. It is never an automatic
  fallback.
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
- `session=<id>`: the store holds one `(text, ids)` entry per session id. A
  cold entry, or an overwrite whose old text is not a prefix, is encoded by
  the active full-encode route: below the crossover this is corrected
  Gigatoken for the certified CPU roster; at or above it an eligible GPU is
  tried first. An accelerated seed's ID row is closure-verified against the
  verified native HF vocabulary before the state is stored, and its spans are
  kept as sparse checkpoints rebuilt on demand; this bridge does not
  initialize the CPU engine. If the stored text is a **prefix** of the input, only the
  remainder is processed through corrected Gigatoken's certified CPU repair
  machinery for the 12 covered model families, independent of total
  transcript size. A failed span/repair premise falls back to HF. The returned
  ids are bit-identical to a from-scratch reference encode in every case.
- `lookup="auto"` (the default when no session is given): content
  lookup. The store's checkpoint index proposes the longest stored text
  whose endpoint digest matches a prefix of the input; the proposal is
  then **byte-verified against the stored text** (the anti-collision
  hard gate) before anything is served from it. A verified full-prefix
  hit appends the remainder; anything else is a miss and a full encode.
  Inputs below a small size floor bypass the store. Auto entries are
  capacity-capped; named session entries are never displaced by auto traffic.
  Cold and overwrite encodes use the same automatic full-encode route as
  named sessions, while strict prefix extensions use CPU repair.
- `lookup="off"`: skip the store entirely. `session` and `lookup` do
  not combine.
- `add_special_tokens=True` is served by the plain routed path.
  Combining it with an explicit `session` or `lookup="auto"` raises
  `UNSUPPORTED_CONFIG` (the stored stream is the core stream; we reject
  rather than silently re-encode).
- `encode_batch` runs the plain routed path row by row; content lookup
  is a single-document affair in 0.x.

## 3. Store, index, and cache layers

Four layers with distinct loss semantics, named honestly:

1. **Store entries** (`store=<directory>`): persistent state, one store
   format v1 record per entry (`store-format-v1.md`, unchanged and
   frozen), written atomically. Deleting the directory loses sessions.
   Loading a record re-verifies it byte-level and re-encodes its tail
   through the current engine; anything that fails is a miss, never a
   wrong answer. A store written under a different semantic fingerprint
   (artifact, oracle, or engine semantics changed) is refused with
   `SESSION_STATE_MISMATCH`. The fingerprint includes the selected repair
   backend, its delivery/version/configuration, and corrected-Gigatoken
   binary/config digests, so state does not cross engine meanings.
2. **Sealed-entry recovery binding** (`entries/<name>.binding`, private
   facade sidecar): store-format v1 intentionally omits stable-prefix
   plaintext. This sidecar stores no plaintext or token IDs; it binds the full
   historical UTF-8 length/digest and canonical checkpoint row to the exact
   record hash. After restart, the caller's candidate prefix is accepted only
   after the length, UTF-8 boundary, stored tail, full digest, checkpoint row,
   and record hash all match. Missing, corrupt, or stale means a cold encode.
   It neither changes nor extends the frozen portable record format.
3. **Checkpoint index** (`index.json`): a derived cache mapping content
   digests (endpoint plus geometric byte positions) to entries. Missing
   or corrupt means rebuild from the record/recovery bindings or serve misses; it is
   never authoritative -- digests propose, bytes decide.
4. **In-process cache** (budget via `cache_budget_bytes`, default
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
block, and a probe summary), runtime route/fallback counts, and -- once the
store has been touched -- store counters (hits, appends, overwrites,
misses, collision rejects, degradations, rebuilds, evictions).

The headline ``backend`` answers "what ran", not "what was planned".
Once this tokenizer has returned a result it is the backend of that
result -- the same value as
``runtime_policy.last_execution.executed_backend`` -- and before any
request it is the planned backend. ``backend_basis`` states which of the
two the value is (``"last_execution"`` / ``"plan"``) and
``planned_backend`` reports the plan in both cases. Routing is a
per-input decision, so a plan is a prediction: the GPU crossover, a
bounded session repair below that crossover, and an added-token literal
routed to HF all produce a headline that differs from the plan, and each
is a correct report rather than a fallback of the plan.

The GPU backend ships two deliveries whose certification kinds differ --
the judged prebuilt binary is ``certified``, a locally compiled JIT
build is ``certified_source`` -- so ``certification.backend_status.gpu``
and the ``certification.state`` headline report the status of the
delivery this process loaded (or, before any load, selected).
``certification.gpu_delivery`` names that delivery, and is ``null``
when none is loaded or selected or when the record carries no
per-delivery rows (then the status is the backend-level one). The
per-delivery, per-architecture detail under ``certification.deliveries``
and ``kernel_deliveries`` is unchanged by this: the headline reports one
of those rows rather than a row of its own. ``probe.kernel_delivery`` is
likewise refreshed from the loader when a delivery is loaded, since the
construction-time snapshot can predate the first kernel load.

The routing policy and the certification state are deliberately separate
keys: their vocabularies share the word ``certified`` while answering
different questions. The 0.x facade plans against the digest-verified shipped
registry. Its ``certification`` block names the artifact identity consulted for
the request, while every runtime eligibility premise is still checked by the
planner.
Outside the ``REFERENCE`` policy, ``device="auto"`` and ``device="cuda"``
supply a real device probe; ``probe.devices_probed`` is true even when it finds
no usable GPU, so ``R_NO_GPU_DETECTED`` is an observed machine fact. Under
``device="cpu"`` no enumeration is performed and
``R_ACCELERATOR_NOT_ADOPTED`` describes that caller choice, not the machine.
Missing runtime modules remain ``R_BACKEND_UNAVAILABLE``. The installed
reference oracle is visible as ``probe.oracle_package`` /
``probe.oracle_version``; the certified version set lives in the shipped
support registry.

For the JIT delivery, the selected NVCC executable/release, torch runtime CUDA,
and PyTorch distribution version are independent binding axes. A
`toolchain_unverified` refusal under the default
automatic device request also emits a `RuntimeWarning` containing the observed
toolchain, certified constraint, selected fallback, and
`toktier gpu compile <family> --accept-uncertified-jit` remedy. An explicit
`device="cuda"` request raises `BackendUnavailable` and carries that remedy in
`details.remedy`. The CLI flag is a one-process `EXPERIMENTAL` opt-in: it does
not modify the registry, persist policy, or extend the exact-ID certificate.
The corresponding waived reason remains visible in
``experimental_waivers``.

"Not adopted" and "not available" are separate statements, and the
report keeps them separate. The ``kernel_deliveries`` block carries,
per kernel delivery (``prebuilt`` / ``jit``): the read-only shipped
facts (whether the prebuilt fatbin and the JIT sources are installed
-- the same answer ``toktier doctor`` gives, through the same helper),
whether this process loaded that delivery, the shipped identity digest and
the prebuilt Rust host's source/build identity,
and the per-architecture certification status map of this artifact's
record in the shipped support registry (with ``driver_min`` for the
prebuilt delivery). The same read-only registry view feeds planning and
reporting; merely finding a record grants nothing until all runtime binding
checks pass. An absent record
reports ``status: null`` and an empty architecture map: the absence of
a claim, not a claim of absence. The facade adds ``runtime_policy``
(requested device/delivery, selected delivery, crossover, execution counts,
and last route), ``gpu_backend`` (planned/loaded/device/load error), and
``state_encode`` (how stored state was seeded). The loaded-process counterpart
also remains available on the explicit GPU engine
(``GpuEngine.explain()`` / ``binding_set()``; `docs/gpu-jit.md` Section 9).
If state seeding calls HF directly (for an added-token literal or a failed span
guard), ``runtime_policy.last_execution`` reports `executed_backend="hf"` plus
`source="state_encode"` and the matching state path. The final-result execution
count and canonical runtime fallback counter are updated exactly once; the
static plan remains unchanged.

The ``session_repair`` block reports the active repair engine and request
path -- Rust-native corrected repair (``request_path: rust_native``), the
compatibility callback, the explicitly selected experimental Fastokens
adapter, or HF full re-encode fallback -- together with its
backend/engine/configuration identity, path counters, and the last executed
path. Corrected-Gigatoken fallback paths start with ``hf_full_`` and name the
guard reason. Fastokens reports ``certification: experimental``,
``mode: full_reencode``, and ``exact_id_guarantee: false``.

Objects created by `from_pretrained` also report `model_resolution`: requested
repository/revision, resolved file/digest/size, sibling-registry root, match
basis and evidence repository, canonical family/anchor, admission boolean,
machine-readable refusal reason, and the artifact digest actually executed.
For an admitted sibling, `certification.identity="exact"` describes the
canonical artifact that runs; `model_resolution.basis` separately explains why
the source repository was allowed to select it. A plain `load(family)` reports
`model_resolution: null`.

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
