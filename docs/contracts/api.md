# Public Python API contract (v1)

Status: frozen for the first public release. Changes to anything marked
**frozen** require a major bump of the public API version axis (see
`versioning.md`).

Guiding rule, stated once and inherited by every section below: correctness
comes first. Certified configurations produce token ids equal to the pinned
reference oracle; whenever a certified result cannot be guaranteed, the
library falls back to the reference path (or, inside a session, to a full
re-encode). We prefer a miss over a wrong result, and an uncertified
configuration runs as reference.

## 1. Module surface (frozen)

```python
from toktier import Tokenizer, Config, RoutingPolicy
```

- Package, import name, CLI name, env prefix, and cache directory all use
  the `toktier` name (`TOKTIER_*` for env, `toktier_` for the C ABI).
- `toktier.errors` exposes the structured exception hierarchy
  (see `errors.md`).
- `toktier.API_VERSION: int` reports the public API version axis.

## 2. `Tokenizer` (frozen constructor shape)

```python
Tokenizer(
    family: str,
    config: Config | None = None,
    *,
    policy: RoutingPolicy | None = None,
)
```

- `family` -- a registered family identifier. Canonical family ids are
  lowercase with underscores (for example `"qwen3_8b"`); registry
  aliases may carry other spellings. Resolution goes through the
  support registry and the artifact manifest; artifacts are fetched (or
  found locally) and verified against the manifest's **per-file
  sha256** before use. A hash mismatch is an error, never a silent
  acceptance.
- `policy` -- routing policy for this object; occupies the constructor
  layer of the precedence chain (above `Config.routing_policy`, below
  per-call arguments). `None` defers to the configuration.
- `config` -- an immutable `Config` (see `config.md`). When omitted, a
  default `Config` is resolved once at construction.
- Construction performs probe and plan (see `routing.md`). The resulting
  `RoutePlan` is immutable for the lifetime of the object.
- Options that the session subsystem cannot support losslessly (padding,
  truncation, and similar output-rewriting modes) are rejected at
  construction -- see `errors.md` (code `UNSUPPORTED_CONFIG`). They are
  not silently ignored.

## 3. `encode` (frozen)

```python
def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]
```

- Returns token ids as a Python `list[int]`.
- `add_special_tokens=True` applies the artifact's postprocessor
  (BOS/EOS and template specials) exactly as the reference oracle would.
- Per-call keyword arguments take the highest precedence in the
  configuration chain (see `config.md`).
- Layering note (2026-08-06): this section describes the *backend
  protocol*, whose `True` default mirrors the reference oracle. The
  public facade (`docs/contracts/facade.md`) defaults to `False` --
  the certified core-stream caliber -- and always passes the flag
  explicitly when calling a backend, so no call site depends on the
  protocol default.

## 4. `encode_batch` (frozen shape, including the ragged output)

```python
def encode_batch(
    self,
    texts: Sequence[str],
    *,
    output: Literal["lists", "ragged"] = "lists",
    add_special_tokens: bool = True,
) -> list[list[int]] | RaggedBatch
```

- `output="lists"` (default) returns `list[list[int]]`, row i for input i.
- `output="ragged"` returns a `RaggedBatch` object with two attributes:

  ```python
  class RaggedBatch:
      values: Buffer   # uint32, contiguous, all rows concatenated in order
      offsets: Buffer  # int64, length == n_rows + 1
  ```

  Contract for the ragged shape (frozen now; early releases may build it
  from the simple path internally):
  - `offsets[0] == 0`; `offsets` is monotonically non-decreasing;
    `offsets[-1] == len(values)`.
  - Row `i` is `values[offsets[i] : offsets[i + 1]]`.
  - `values` element type is unsigned 32-bit; `offsets` element type is
    signed 64-bit (chosen for ecosystem interoperability with common
    array libraries).
  - Both attributes expose the Python buffer protocol so they can be
    wrapped zero-copy by array libraries; no third-party array type is
    part of the contract.
- Batch results are row-for-row equal to calling `encode` on each element
  with the same settings.

## 5. `session()` (frozen)

```python
def session(
    self,
    store: str | os.PathLike | None = None,
    *,
    session_id: str | None = None,
) -> ContextManager[Session]
```

- Returns a context manager yielding a `Session`. With `store=None` the
  session is in-memory only; with a store path, session state persists
  under the state directory rules of `config.md` (state is distinct from
  cache: deleting state loses sessions; deleting cache only costs
  recomputation).
- Session ids stored on disk follow the store format contract
  (`store-format-v1.md`).

### 5.1 `Session.append` and the update object (frozen)

```python
update = session.append(new_text)

class SessionUpdate:            # immutable
    replace_from: int           # zero-based token index, core stream
    replacement_ids: Sequence[int]
    all_ids: Sequence[int]
```

Semantics (frozen):

- All three fields refer to the **pre-postprocessor core token stream**:
  the stream without BOS/EOS or other postprocessor output. Stores hold
  the core stream; special tokens are applied at read time.
- `replace_from` is a zero-based token index into the core stream held
  before this append. Appending is allowed to rewrite a tail of the
  previously returned tokens; consumers must treat tokens at indices
  `>= replace_from` as replaced.
- Invariant (frozen): `all_ids == old_ids[:replace_from] + replacement_ids`,
  where `old_ids` is the core stream before the append.
- Correctness invariant (frozen): `all_ids` is bit-identical to a
  from-scratch reference encode of the accumulated session text (core
  stream). When no certified safe cut point exists, the implementation
  re-encodes the whole text -- the result is then reported with
  `replace_from == 0`. We prefer a full re-encode over an uncertified
  splice.

### 5.2 Other `Session` members (frozen names)

```python
session.ids            -> Sequence[int]   # current core stream
session.text           -> str             # accumulated text
session.revision       -> int             # store revision, monotone
session.final_ids(add_special_tokens: bool = True) -> list[int]
```

- `final_ids()` materializes the read-time view: postprocessor applied on
  top of the stored core stream.
- Persistent writes use optimistic concurrency: each write carries the
  expected revision; a mismatch raises `SessionRevisionConflict`
  (code `SESSION_REVISION_CONFLICT`). Last-writer-wins behavior is not
  offered.

## 6. Diagnostics (reserved name, shape not frozen)

```python
tokenizer.explain() -> Mapping[str, object]
```

- `explain()` is a reserved public name returning the active `RoutePlan`,
  the probe snapshot summary, and accumulated fallback reason codes.
  The mapping's exact keys are informational in v1 and may grow; the
  method name and the presence of the plan and reason codes are stable.

## 7. Out of scope for v1 (recorded so absence is deliberate)

- `decode` and offset mappings -- planned, not part of the frozen v1 set.
- Padding/truncation convenience -- rejected at construction for sessions;
  batch padding helpers may arrive later as pure post-steps that do not
  change the id contract.
- Async APIs, streaming iterators -- roadmap items.
