# Routing contract (v1)

Status: frozen for the first public release. The policy enum, the
three-phase structure, the immutability of `RoutePlan`, and the reason
code namespace are contract; new reason codes may be appended (see Section 5.4).

## 1. `RoutingPolicy` (frozen enum)

| Value | Meaning |
|---|---|
| `CERTIFIED` (default) | Accelerated paths are used only where the support registry certifies them for this exact configuration; everything else runs on the reference backend. |
| `REFERENCE` | Always run the reference backend (pinned HF tokenizers path). No accelerated code is planned or executed. |
| `REQUIRE_ACCELERATED` | Like `CERTIFIED`, but if no certified accelerated path is eligible at plan time, construction raises the specific cause error instead of quietly planning reference. |
| `EXPERIMENTAL` | Permits uncertified accelerated paths (unlisted GPU architectures, PTX-JIT products, uncertified families). Outputs under this policy are not covered by the certification claims. Never the default. |

- `tier="auto"`, where accepted, is a convenience alias for `CERTIFIED`
  and introduces no additional behavior.
- Under every policy, correctness-motivated fallbacks remain enabled:
  we prefer a miss over a wrong result, and an uncertified path is only
  reachable by explicit `EXPERIMENTAL` opt-in.
- `EXPERIMENTAL` is opt-in **only** through an explicit construction
  parameter (`policy=RoutingPolicy.EXPERIMENTAL`) or the equally explicit
  `toktier gpu compile <family> --accept-uncertified-jit` command; no
  environment variable or config-file value can select it. The CLI acceptance
  applies only to that process and does not modify or extend the registry.
  Persistent experimental state
  uses a backend/version/configuration-bound fingerprint and therefore cannot
  be replayed into a certified or differently versioned configuration.
- Registry status `certified` (binary digest bound) and
  `certified_source` (source digest + build flags + exact toolchain bound;
  used by the integrated CPU engine and GPU JIT) are both eligible under `CERTIFIED`,
  provided every bound constraint verifies at load time. The two statuses
  are reported distinctly in `explain()` and in the registry -- see
  `registry.md` for the honest-labeling rules.

### 1.1 The `SUPPORTED` policy (added in 0.2.6)

The four rows above keep their meanings exactly. This release appends a
fifth value and makes it the default, so the `(default)` marker in the v1
table names what v1 shipped rather than what a fresh `Tokenizer` selects
today.

| Value | Meaning |
|---|---|
| `SUPPORTED` (default) | Everything `CERTIFIED` admits, and in addition a device architecture or compiler toolchain no certification campaign has judged, provided the shipped kernel loads and runs there and every constraint the registry does bind still verifies. Such a route is reported as `supported_untested`, or `locally_verified` once a local check on that machine has compared it with the reference engine. |

What this supersedes, said plainly: the bullet above that reads "an
uncertified path is only reachable by explicit `EXPERIMENTAL` opt-in"
described two different things with one word. A path whose engines,
digests or build identity do not verify is still reachable only through
`EXPERIMENTAL`, and that half stands unchanged. A path whose every bound
constraint verifies and whose device or compiler no campaign has run on
is a coverage gap rather than a verification failure; from 0.2.6 the
default policy admits it and labels it, and `CERTIFIED` refuses it as
before. `registry.md` Section 1.1 is unaffected: an unlisted architecture
remains ineligible under `CERTIFIED`.

- `REQUIRE_ACCELERATED` follows the default rather than `CERTIFIED`: it
  asks that some accelerated path be eligible, and on a device outside
  the judged list the honest answer to that question is now yes.
- `tier="auto"` still names `CERTIFIED`, so it selects the stricter of
  the two rather than the default.
- The new labels appear in `explain()` under `certification.state`
  (`supported_untested`, `locally_verified`), under
  `certification.effective_verdict` (`supported`), and as a
  `supported_untested` list of the coverage reasons the policy admitted,
  which is the same reason vocabulary and adds no code.
- Nothing about this is automatic on the request path: the local check
  is a command a person runs, and its record only ever adds a label.

## 2. Three phases: probe, plan, execute (frozen structure)

1. **Probe** -- collect facts, change nothing. The probe gathers: importable
   backends, device inventory and driver version, kernel cache state,
   registry entries for the requested family, installed oracle version.
   Probing never builds kernels, never downloads artifacts, and never
   mutates state. Device facts are supplied by the integrating caller
   (enumerating them needs the accelerator runtime); when no device
   probe is supplied, the snapshot records that enumeration was not
   performed, and the plan reports that case as its own reason
   (`R_ACCELERATOR_NOT_ADOPTED`) rather than as a hardware observation.
2. **Plan** -- a pure function. `plan(probe_snapshot, policy, registry,
   config) -> RoutePlan`. Same inputs, same plan; no I/O, no clock, no
   randomness. The plan records the selected backend, the ordered
   fallback chain, and one reason entry for every accelerated option that
   was considered and not selected.
3. **Execute** -- follow the plan. Execution may only move along the
   plan's fallback chain, never sideways or upward: an execution that
   started under a reference plan does not opportunistically upgrade to
   an accelerated path mid-run. Every runtime fallback is recorded with a
   reason code.

## 3. `RoutePlan` immutability (frozen)

- `RoutePlan` is an immutable value object. Fields: `policy`,
  `backend` (selected backend identifier), `fallback_chain`
  (ordered tuple of backend identifiers ending in the reference backend),
  `reasons` (tuple of plan-time reason entries).
- A `Tokenizer` holds exactly one plan for its lifetime. Environment
  changes after construction (a GPU appearing, an env var changing) do
  not alter an existing plan; construct a new `Tokenizer` to re-plan.

## 4. Backend identifiers (frozen namespace)

Backend ids are lowercase strings. v1 assigns:

| Id | Description |
|---|---|
| `hf` | Reference backend: pinned HF tokenizers path. Always present; always last in every fallback chain. |
| `gpu` | CUDA kernel backend, with prebuilt and JIT deliveries. |
| `fast_cpu` | Corrected Gigatoken backend compiled into the private `toktier._native` extension in the core wheel; eligible only under its exact integrated module/source/build/toolchain/configuration and artifact binding. |
| `fastokens` | Explicit experimental session adapter; never an automatic certified route. |

New backends append to this table; ids are never reused or renamed.

## 5. Fallback reason codes (`R_*`)

Reason codes are stable machine identifiers. The natural-language text
that may accompany them is not a machine interface.

### 5.1 Plan-time reasons (why an accelerated option was not selected)

| Code | Meaning |
|---|---|
| `R_POLICY_REFERENCE` | Policy is `REFERENCE`; accelerated options were not considered. |
| `R_UNCERTIFIED_ARTIFACT` | The artifact has no eligible registry identity (exact or capability composition) for this backend. |
| `R_ORACLE_MISMATCH` | Installed oracle version is outside the certified set for this record; acceleration is off, the installed reference still runs (reference-only state). |
| `R_BACKEND_UNAVAILABLE` | Required backend package/extra is not importable (for example the GPU extra without torch). |
| `R_GPU_DISABLED` | GPU use disabled by configuration (`TOKTIER_DISABLE_GPU` or config field). |
| `R_NO_GPU_DETECTED` | A performed device probe found no usable CUDA device. |
| `R_ACCELERATOR_NOT_ADOPTED` | Device enumeration was not performed: the integrating caller supplied no device probe, i.e. it adopts no accelerator runtime on this path (`device="cpu"` is such a path). Says nothing about the machine's hardware. |
| `R_DRIVER_TOO_OLD` | CUDA driver below the certified minimum for the kernel record. |
| `R_SM_UNCERTIFIED` | Device architecture has no certified kernel entry; only `EXPERIMENTAL` may proceed. |
| `R_KERNEL_DIGEST_MISMATCH` | A registry-bound digest does not verify: kernel source digest (`certified_source`) or binary digest (`certified`), or the bound class-table digest of the generated lookup tables the kernel consumes. |
| `R_KERNEL_BUILD_FAILED` | JIT build attempted at load and failed. |
| `R_ENGINE_BINDING_MISMATCH` | The CPU engine's delivery identity, private module, build version, source digest, build flags, exact toolchain, patch, or repair configuration does not match the registry binding. |

### 5.2 Run-time reasons (per call or per input, recorded during execute)

| Code | Meaning |
|---|---|
| `R_INPUT_ADDED_TOKEN` | Input contains an added-token literal; this input is routed to the reference frontend path. Part of the certified pipeline design, not a correctness incident. |
| `R_INPUT_BELOW_GPU_THRESHOLD` | Input is smaller than the configured GPU crossover, so execution starts at the next eligible backend in the immutable fallback chain. This is a normal latency policy, not a fault. |
| `R_INPUT_GUARD_ROUTED` | A per-input guard premise on an accelerated path could not be proved -- a guarded fast-CPU input, or a state-seed closure/span premise -- so this input was routed to the reference backend. Every event detail identifies the failing `stage`: `engine_guard` for the fast-CPU engine guard, `span_bridge` for the native accelerated state-seed bridge, or `state_encode` for a facade-owned state-encoding guard. |
| `R_SESSION_NO_SAFE_CUT` | A session append found no certified safe cut point; the accumulated text was fully re-encoded. Correctness preserved by construction. |
| `R_EXEC_FAULT` | An accelerated engine failed to open or raised an internal error while executing; execution continued at the next eligible backend in the fallback chain, and the reference runs when the chain reaches it. Returned certified IDs remain reference-equal. |
| `R_INPUT_POSTPROCESS_ROUTED` | A core-stream-only accelerated backend was bypassed before execution because the request asked for postprocessing that can change the ID stream. The reference backend produced the postprocessed result. This is a capability route, not an execution fault. |

### 5.3 Interaction with `REQUIRE_ACCELERATED`

`REQUIRE_ACCELERATED` constrains plan time only: construction raises if
no certified accelerated backend can be planned. Run-time input-level
routing (`R_INPUT_ADDED_TOKEN`, `R_INPUT_BELOW_GPU_THRESHOLD`,
`R_INPUT_POSTPROCESS_ROUTED`) and correctness fallbacks
(`R_INPUT_GUARD_ROUTED`, `R_EXEC_FAULT`, `R_SESSION_NO_SAFE_CUT`) remain
active -- they are part of the certified configuration, and disabling
them is not offered.

### 5.4 Extension policy (frozen)

- The `R_*` namespace is append-only: codes are never renamed, reused,
  or re-meant. New codes may be added in any 0.x release.
- Consumers must tolerate unknown reason codes (treat as opaque
  diagnostics). Machine logic should switch on the codes it knows and
  pass the rest through.

### 5.5 The same codes on the Rust face

Since 0.2.5 a Rust `ExecutionFacts` carries `reason`, the run-time code
behind that one execution, as `toktier::ReasonCode`. It is the code the
router recorded for the input, not a second reading of the path string,
so the two cannot come to disagree. `R_INPUT_ADDED_TOKEN`,
`R_INPUT_BELOW_GPU_THRESHOLD`, `R_INPUT_GUARD_ROUTED`, `R_EXEC_FAULT`,
`R_INPUT_POSTPROCESS_ROUTED`, and `R_SESSION_NO_SAFE_CUT` have named
variants; a code without one arrives as `ReasonCode::Other` carrying the
code itself, which is how Section 5.4 asks consumers to treat it.
`reason` is `None` when the admitted route ran the input and there was
nothing to record; plan-time reasons stay on `RoutePlan::reasons`.

## 6. Telemetry boundary

Reason codes cover routing and execution fallback. Session store
telemetry (hit/miss/checksum-reject counters) is a separate surface
reported by store statistics, not through `R_*` codes; the single shared
principle is that every degradation is counted, none is silent.
