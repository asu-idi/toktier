# Structured errors contract (v1)

Status: frozen for the first public release. The code strings and the
`.code` attribute contract are machine interface; exception class names
mirror the codes. Natural-language messages are for humans and carry no
machine contract -- tools must switch on `.code`, never parse messages.

## 1. Shape of every toktier exception (frozen)

```python
class ToktierError(Exception):
    code: str                     # stable UPPER_SNAKE identifier
    details: Mapping[str, object] # machine-readable payload, may be empty
```

- All exceptions raised by the library's public surface that represent
  library-level conditions derive from `ToktierError`.
- `.code` is a stable identifier from the table below; codes are
  append-only, never renamed, reused, or re-meant.
- `.details` carries structured facts (expected/observed hashes, paths,
  a suggested remediation command). Keys used by v1 are documented per
  code below; unknown keys must be tolerated.
- Standard Python exceptions (`TypeError`, `ValueError` for plain
  argument misuse) remain standard; only library-domain conditions get
  structured codes.

## 2. Error code table (frozen, append-only)

| Class | `.code` | Raised when | Typical `details` keys |
|---|---|---|---|
| `ArtifactNotFound` | `ARTIFACT_NOT_FOUND` | The tokenizer artifact cannot be resolved: unknown family, missing local file, or offline mode with an empty cache. | `family`, `searched`, `offline` |
| `ArtifactHashMismatch` | `ARTIFACT_HASH_MISMATCH` | A fetched or cached artifact fails content-hash verification. Online: after quarantine and one re-fetch both failed. Offline: immediately. | `expected_sha256`, `observed_sha256`, `path`, `remedy` |
| `UncertifiedTokenizer` | `UNCERTIFIED_TOKENIZER` | `REQUIRE_ACCELERATED` policy and the artifact has no eligible certification identity. (Under `CERTIFIED` this is not an error; the plan falls back to reference with `R_UNCERTIFIED_ARTIFACT`.) | `artifact_sha256`, `family` |
| `OracleVersionUnsupported` | `ORACLE_VERSION_UNSUPPORTED` | The installed oracle package version is outside every certified set and reference execution itself is impossible (for example the package is absent or incompatible at import level). | `package`, `installed`, `certified` |
| `BackendUnavailable` | `BACKEND_UNAVAILABLE` | A required backend cannot be used under a policy or explicit device request that demands it (`REQUIRE_ACCELERATED` or `device="cuda"`). `missing` says which precondition failed: the backend module name when the package/extra is not importable, `"cuda_device"` when a performed device probe found no usable device, or `"device_probe"` when device enumeration was deliberately omitted (`R_ACCELERATOR_NOT_ADOPTED`, for example `device="cpu"`). An unjudged explicit JIT request also carries the one-process experimental compile command in `remedy`; invoking it does not expand certification. | `backend`, `missing`, `reason_code`, `reason`, `remedy` |
| `BackendExecutionFault` | `BACKEND_EXECUTION_FAULT` | A backend failed on an input in a way the routing layer may recover from: an accelerated backend wraps expected device and runtime failures in this type, and the executor re-runs the affected input on the next backend in the chain (`R_EXEC_FAULT`). Any other exception type propagates unchanged. | `backend`, `stage` |
| `KernelIncompatible` | `KERNEL_INCOMPATIBLE` | Kernel constraints fail verification under a policy that demands the kernel: uncertified SM, kernel or class-table digest mismatch, or build failure. | `backend`, `reason_code`, `sm`, `expected_digest`, `observed_digest`, `class_table_digest` |
| `CudaDriverTooOld` | `CUDA_DRIVER_TOO_OLD` | Driver below certified minimum, under a policy that demands the GPU backend. | `installed`, `required` |
| `StoreCorrupt` | `STORE_CORRUPT` | An explicit integrity operation (verify/fsck-style API) found a checksum, linkage, or structural failure. On the ordinary read path, integrity failures degrade to a counted miss instead of raising -- we prefer a miss over a wrong result. | `path`, `record`, `failure` |
| `StoreFormatUnsupported` | `STORE_FORMAT_UNSUPPORTED` | A store record is well-formed but not decodable by this reader: future format version, unknown mandatory flag bit, or unknown witness category. Distinct from corruption by design. | `format_version`, `flags`, `witness_category` |
| `SessionStateMismatch` | `SESSION_STATE_MISMATCH` | A session is opened against state whose semantic fingerprint does not match the current tokenizer/configuration. | `expected_fingerprint`, `stored_fingerprint` |
| `SessionRevisionConflict` | `SESSION_REVISION_CONFLICT` | An optimistic-concurrency write carried an `expected_revision` that no longer matches the stored revision. Last-writer-wins is not offered. | `expected_revision`, `stored_revision` |
| `ConfigInvalid` | `CONFIG_INVALID` | A configuration value cannot be parsed or is out of range (including strict boolean env parsing failures and config-file syntax errors). | `field`, `value`, `source` |
| `UnsupportedConfig` | `UNSUPPORTED_CONFIG` | A syntactically valid option combination is outside the supported envelope and is rejected at construction rather than silently ignored (for example padding/truncation modes with sessions). | `option`, `value`, `reason` |
| `RegistryInvalid` | `REGISTRY_INVALID` | The support registry or an evidence manifest fails schema validation or root digest verification. | `path`, `failure` |
| `BundleInvalid` | `BUNDLE_INVALID` | An air-gap bundle violates the frozen bundle archive format (decision 0004). Covers the tar container and the embedded `bundle_manifest.json`: unreadable or truncated archives, unsafe member paths, link members, duplicate members, member-set mismatches against the manifest, resource-limit violations, and a bundle manifest that fails parsing, schema, or root digest verification. Content-hash failures of the artifact files inside the bundle raise `ARTIFACT_HASH_MISMATCH`; a missing bundle file or requested member raises `ARTIFACT_NOT_FOUND`. | `path`, `failure`, `member`, `cause` |

## 3. Relationship to fallback reason codes

Errors and `R_*` reason codes (see `routing.md`) are different layers:

- A reason code records a **permitted** degradation (fallback happened,
  correctness preserved, execution continued).
- An error records a condition under which the requested operation
  **cannot proceed** as asked.

The same underlying fact can surface as either depending on policy: an
uncertified artifact yields `R_UNCERTIFIED_ARTIFACT` under `CERTIFIED`
and `UncertifiedTokenizer` under `REQUIRE_ACCELERATED`.

## 4. Command-line surface

A failed command exits `2` and writes its report to standard error. A
command invoked with `--json` writes that report as one JSON object so
that `--json` describes the whole command rather than only its success
path:

```json
{"error": {"code": "BACKEND_UNAVAILABLE",
           "message": "...",
           "details": {"backend": "gpu", "remedy": "..."}}}
```

`code` is the stable switch key of Section 2, `message` is the human
text, and `details` is the same mapping the exception carries in
process. `details` is open, so a value of a type JSON cannot express is
rendered as its `repr` rather than dropped or allowed to break the
envelope. Without `--json` the single line `error <CODE>: <message>` is
written instead, unchanged.

## 5. Extension policy (frozen)

- New codes may be added in minor releases; existing codes never change
  meaning. Consumers must tolerate unknown codes.
- Exception class hierarchy may gain intermediate bases; catching
  `ToktierError` and switching on `.code` is the forward-compatible
  pattern and is the documented recommendation.
