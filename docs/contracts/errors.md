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
| `ArtifactNotFound` | `ARTIFACT_NOT_FOUND` | The tokenizer artifact cannot be resolved: unknown family, missing local file, offline mode with an empty cache, or a configured source that could not deliver the bytes. Since 0.2.4 the last case includes a failure raised inside a download client this package does not own (a gated or private repository, an expired token, a transport failure): it is classified at the fetch boundary, with `cause` naming the client's exception type, rather than escaping as that type. | `family`, `searched`, `offline`, `suggestions`; from a source that failed, also `cause`, `cause_message`, `remedy` |
| `AliasConflict` | `ALIAS_CONFLICT` | An artifact cache already holds the alias an air-gap bundle is being imported under, and the installed tree does not authenticate as that bundle: an undeclared or missing file, something that is not a regular file, or a differing byte count or digest. The cache's own `.toktier-verified.json` marker is not an undeclared file: it is toktier's sidecar rather than bundle content, and it is neither read nor rewritten by an import. Since 0.2.9 a leftover of that marker's own write, left at the top of the alias by a process that stopped mid-write, is removed by the import rather than counted against the tree; a file of the marker's name deeper in the tree, and every other undeclared file, are refused as before. Added in 0.2.8, together with the conditional idempotency the Rust face already documented: an installed tree that does authenticate ends the import successfully instead. Before 0.2.8 the Python facade refused every alias its cache held, whatever it held, and reported `ARTIFACT_NOT_FOUND` for it. `path` names one file, chosen in two phases: an entry the tree holds that the bundle does not declare, at any depth, if there is one, and otherwise the first declared file that is missing or does not match; within either phase the relative path that sorts first is the one named. | `family`, `searched`, `path`, `failure`, `remedy`; for a byte difference also `expected_sha256`, `observed_sha256` or `expected_size`, `observed_size` |
| `ArtifactHashMismatch` | `ARTIFACT_HASH_MISMATCH` | A fetched or cached artifact fails content-hash verification. Online: after quarantine and one re-fetch both failed. Offline: immediately. Since 0.2.4 also raised when the conversion gate (`artifacts check-conversion`) finds that the locally derived artifact is not the one the shipped manifest pins; `failures` names which of its claims did not hold. | `expected_sha256`, `observed_sha256`, `path`, `remedy`; from the conversion gate also `family`, `failures` |
| `UncertifiedTokenizer` | `UNCERTIFIED_TOKENIZER` | `REQUIRE_ACCELERATED` policy and the artifact has no eligible certification identity. (Under `CERTIFIED` this is not an error; the plan falls back to reference with `R_UNCERTIFIED_ARTIFACT`.) | `artifact_sha256`, `family` |
| `OracleVersionUnsupported` | `ORACLE_VERSION_UNSUPPORTED` | The installed oracle package version is outside every certified set and reference execution itself is impossible (for example the package is absent or incompatible at import level). | `package`, `installed`, `certified` |
| `BackendUnavailable` | `BACKEND_UNAVAILABLE` | A required backend cannot be used under a policy or explicit device request that demands it (`REQUIRE_ACCELERATED` or `device="cuda"`). `missing` says which precondition failed: the backend module name when the package/extra is not importable, `"cuda_device"` when a performed device probe found no usable device, or `"device_probe"` when device enumeration was deliberately omitted (`R_ACCELERATOR_NOT_ADOPTED`, for example `device="cpu"`). An unjudged explicit JIT request also carries the one-process experimental compile command in `remedy`; invoking it does not expand certification. | `backend`, `missing`, `reason_code`, `reason`, `remedy` |
| `BackendExecutionFault` | `BACKEND_EXECUTION_FAULT` | A backend failed on an input in a way the routing layer may recover from: an accelerated backend wraps expected device and runtime failures in this type, and the executor re-runs the affected input on the next backend in the chain (`R_EXEC_FAULT`). Core-stream-only adapters also use `stage="add_special_tokens"` as an internal route signal; the executor records that planned pre-execution bypass as `R_INPUT_POSTPROCESS_ROUTED`, not as a fault. Any other exception type propagates unchanged. | `backend`, `stage` |
| `KernelIncompatible` | `KERNEL_INCOMPATIBLE` | Kernel constraints fail verification under a policy that demands the kernel: uncertified SM, kernel or class-table digest mismatch, or build failure. | `backend`, `reason_code`, `sm`, `expected_digest`, `observed_digest`, `class_table_digest` |
| `CudaDriverTooOld` | `CUDA_DRIVER_TOO_OLD` | Driver below certified minimum, under a policy that demands the GPU backend. | `installed`, `required` |
| `StoreCorrupt` | `STORE_CORRUPT` | An explicit integrity operation (verify/fsck-style API) found a checksum, linkage, or structural failure. On the ordinary read path, integrity failures degrade to a counted miss instead of raising -- we prefer a miss over a wrong result. | `path`, `record`, `failure` |
| `StoreFormatUnsupported` | `STORE_FORMAT_UNSUPPORTED` | A store record is well-formed but not decodable by this reader: future format version, unknown mandatory flag bit, or unknown witness category. Distinct from corruption by design. | `format_version`, `flags`, `witness_category` |
| `SessionStateMismatch` | `SESSION_STATE_MISMATCH` | A session is opened against state whose semantic fingerprint does not match the current tokenizer/configuration. | `expected_fingerprint`, `stored_fingerprint` |
| `SessionRevisionConflict` | `SESSION_REVISION_CONFLICT` | An optimistic-concurrency write carried an `expected_revision` that no longer matches the stored revision. Last-writer-wins is not offered. | `expected_revision`, `stored_revision` |
| `ConfigInvalid` | `CONFIG_INVALID` | A configuration value cannot be parsed or is out of range (including strict boolean env parsing failures and config-file syntax errors). Since 0.2.4 it also answers for a resolved root that cannot hold private state: a `TOKTIER_HOME` that is a regular file, a root this user cannot create, and an environment with no home directory to hang the platform conventions on. Those conditions used to escape as the operating system's own exception (`NotADirectoryError`, `PermissionError`); `cause` and `cause_message` now carry it inside the envelope. The Rust crate reports a filesystem failure met on such a path as `IO_ERROR`, a code the Python contract does not have; both faces agree on `CONFIG_INVALID` for the configured location itself. | `field`, `value`, `source`; from an unusable root also `cause`, `cause_message`, `remedy` |
| `UnsupportedConfig` | `UNSUPPORTED_CONFIG` | A syntactically valid option combination is outside the supported envelope and is rejected at construction rather than silently ignored (for example padding/truncation modes with sessions). Since 0.2.4 also raised when a command is asked for something a valid, known family does not have -- `artifacts check-conversion` on a family that is downloaded whole rather than converted. | `option`, `value`, `reason`; where another command answers the question also `remedy` |
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

A failed command exits `2` and writes its report to standard error.
Without `--json` that report is the single line `error <CODE>: <message>`.

Since 0.2.4 `--json` is accepted by every command, and equally before or
after it: `toktier --json artifacts verify FAMILY` and
`toktier artifacts verify --json FAMILY` are the same request. (Through
0.2.1 it was a per-command option carried by `doctor`, `inspect`,
`artifacts check-conversion` and `gpu compile` only; the others rejected
it with exit `64`.)

It describes the whole command rather than only its success path: a
failure writes the report as one JSON object on standard error and still
exits `2`. On success each command prints one object; the prose line each
one printed before is unchanged when the flag is absent.

```json
{"error": {"code": "BACKEND_UNAVAILABLE",
           "message": "...",
           "details": {"backend": "gpu", "remedy": "..."}}}
```

`code` is the stable switch key of Section 2, `message` is the human
text, and `details` is the same mapping the exception carries in
process. `details` is open, so a value of a type JSON cannot express is
rendered as its `repr` rather than dropped or allowed to break the
envelope.

The envelope shape is frozen. The success payloads are per command and
their keys may grow; unknown keys must be tolerated. Exit status remains
the coarse signal a script can use without reading anything: `0` for
success, `2` for a library-domain condition, `64` for a usage error. The
same `.code` and `.details` are available in process through the
`toktier.artifacts` API.

## 5. Extension policy (frozen)

- New codes may be added in minor releases; existing codes never change
  meaning. Consumers must tolerate unknown codes.
- Exception class hierarchy may gain intermediate bases; catching
  `ToktierError` and switching on `.code` is the forward-compatible
  pattern and is the documented recommendation.
