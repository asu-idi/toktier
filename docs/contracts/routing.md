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
  `certified_source` (source digest + build flags + toolchain constraint
  bound; the JIT delivery mode) are both eligible under `CERTIFIED`,
  provided every bound constraint verifies at load time. The two statuses
  are reported distinctly in `explain()` and in the registry -- see
  `registry.md` for the honest-labeling rules.

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
| `fast_cpu` | Corrected Gigatoken backend vendored privately in the core wheel; eligible only under its exact delivery/module/binary/configuration and artifact binding. |
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
| `R_ENGINE_BINDING_MISMATCH` | The CPU engine's delivery identity, private module, build version, native-module digest, or repair configuration does not match the registry binding. |

### 5.2 Run-time reasons (per call or per input, recorded during execute)

| Code | Meaning |
|---|---|
| `R_INPUT_ADDED_TOKEN` | Input contains an added-token literal; this input is routed to the reference frontend path. Part of the certified pipeline design, not a correctness incident. |
| `R_INPUT_BELOW_GPU_THRESHOLD` | Input is smaller than the configured GPU crossover, so execution starts at the next eligible backend in the immutable fallback chain. This is a normal latency policy, not a fault. |
| `R_INPUT_GUARD_ROUTED` | A guarded fast-CPU input could not satisfy a per-input premise and was routed to the reference backend. |
| `R_SESSION_NO_SAFE_CUT` | A session append found no certified safe cut point; the accumulated text was fully re-encoded. Correctness preserved by construction. |
| `R_EXEC_FAULT` | An accelerated path raised an internal error; the affected input was re-run on the next backend in the fallback chain. The reference result is returned. |

### 5.3 Interaction with `REQUIRE_ACCELERATED`

`REQUIRE_ACCELERATED` constrains plan time only: construction raises if
no certified accelerated backend can be planned. Run-time input-level
routing (`R_INPUT_ADDED_TOKEN`, `R_INPUT_BELOW_GPU_THRESHOLD`) and correctness fallbacks
(`R_EXEC_FAULT`, `R_SESSION_NO_SAFE_CUT`) remain active -- they are part
of the certified configuration, and disabling them is not offered.

### 5.4 Extension policy (frozen)

- The `R_*` namespace is append-only: codes are never renamed, reused,
  or re-meant. New codes may be added in minor releases.
- Consumers must tolerate unknown reason codes (treat as opaque
  diagnostics). Machine logic should switch on the codes it knows and
  pass the rest through.

## 6. Telemetry boundary

Reason codes cover routing and execution fallback. Session store
telemetry (hit/miss/checksum-reject counters) is a separate surface
reported by store statistics, not through `R_*` codes; the single shared
principle is that every degradation is counted, none is silent.
