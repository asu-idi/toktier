# Configuration contract (v1)

Status: frozen for the first public release: the immutability rule, the
precedence chain, the read-once environment rule, and the long-term
environment variable set.

## 1. Immutable `Config` (frozen)

- `Config` is an immutable value object. All fields are resolved at
  construction time; a `Config` never changes after it exists.
- Deriving a modified configuration produces a new object
  (`dataclasses.replace`-style); nothing observes in-place mutation.
- A `Tokenizer` captures its `Config` at construction; later environment
  or file changes never affect an existing object.

## 2. Precedence chain (frozen)

From highest to lowest priority; the first layer that provides a value
wins:

1. Per-call method argument (for example `add_special_tokens=` on
   `encode`).
2. `Tokenizer` constructor argument.
3. Explicit `Config` object field set by the caller.
4. Configuration file.
5. Environment variable.
6. Built-in default.

## 3. Environment: read once, at `Config` construction (frozen)

- Environment variables are consulted exactly once, when a `Config` is
  constructed. No code path re-reads the environment afterwards.
- Boolean env values parse strictly: `1/true/yes/on` are true,
  `0/false/no/off` are false (case-insensitive); anything else raises
  `ConfigInvalid`. No silent guessing.
- Scope note: this rule governs toktier's configuration surface.
  Diagnostics (`toktier doctor`) additionally *observe and report*
  foreign toolchain variables (`CUDA_HOME`, `CUDA_PATH`) to describe
  the CUDA discovery the kernel build system performs; those values
  configure nothing in toktier and are never stored or acted on.

## 4. Long-term environment variables (frozen set)

| Variable | Type | Meaning |
|---|---|---|
| `TOKTIER_HOME` | path | Root directory for toktier's on-disk footprint. When set, cache, state, and the config file all live under it. |
| `TOKTIER_OFFLINE` | bool | Never touch the network; artifacts must already be present and verified locally. Missing artifacts raise `ARTIFACT_NOT_FOUND`. |
| `TOKTIER_LOG_LEVEL` | string | Standard logging level name for the library logger. |
| `TOKTIER_DISABLE_GPU` | bool | Exclude GPU backends at plan time (`R_GPU_DISABLED`). |
| `TOKTIER_DIAGNOSTICS` | bool | Enable extended diagnostics collection (feeds `explain()` and reason-code detail). |

Deliberate absences (frozen as absences): there is no environment
variable to skip hash verification, bypass certification checks, or
disable correctness fallbacks. These are not configuration; they are the
product. The rule generalizes: **no switch that can change output
correctness may exist in environment-variable or config-file form.**
The only path to uncertified output is the explicit
`policy=RoutingPolicy.EXPERIMENTAL` construction parameter
(see `routing.md`), which is visible at the call site and labeled in
diagnostics.

Variables outside this set that may exist during development are not
contract and can disappear without notice.

## 5. Directory layout: cache vs state (frozen distinction)

- **Cache** (artifacts, built kernels): fully rebuildable. Deleting it
  costs re-download or re-build time, never data.
- **State** (session store): deleting it loses sessions. Tools and docs
  must never describe state as "just a cache".
- Resolution:
  - `TOKTIER_HOME` set: `$TOKTIER_HOME/cache`, `$TOKTIER_HOME/state`,
    `$TOKTIER_HOME/config.toml`.
  - Otherwise: platform-conventional user cache and state directories
    for application name `toktier` (XDG on Linux:
    `~/.cache/toktier`, `~/.local/state/toktier`; the platformdirs
    conventions on other platforms).
- Store files default to owner-only permissions (0700 directories,
  0600 files).

## 6. Configuration file (frozen slot; minimal v1 scope)

- Location: `config.toml` under the resolved home (see Section 5). Format: TOML.
- The file may set exactly the same knobs as the long-term environment
  variables plus the default routing policy; unknown keys raise
  `ConfigInvalid` (fail closed rather than silently ignoring typos).
- The file is read once at `Config` construction, after explicit fields
  and before environment fallback, per Section 2.
- Early releases may ship with file support minimal or absent at
  runtime; the precedence slot and format are frozen now so adding it is
  not a breaking change.

## 7. `Config` fields (frozen names)

| Field | Type | Default |
|---|---|---|
| `home` | path | platform convention (Section 5) |
| `cache_dir` | path | derived from `home` rules |
| `state_dir` | path | derived from `home` rules |
| `offline` | bool | `False` |
| `log_level` | str | `"WARNING"` |
| `disable_gpu` | bool | `False` |
| `diagnostics` | bool | `False` |
| `routing_policy` | `RoutingPolicy` | `CERTIFIED` |

New fields may be added with defaults in minor releases; existing fields
keep their names and meanings.
