# Versioning contract (v1)

Status: frozen. Six version axes are accounted independently. No axis is
ever inferred from another; compatibility statements always name the
axis they speak about.

## 1. The six axes

| # | Axis | Recorded where | Bump rule |
|---|---|---|---|
| 1 | **Package version** | PyPI / `toktier.__version__` (PEP 440) | Release cadence; carries no implication about any other axis. |
| 2 | **Public API version** | `toktier.API_VERSION` (integer) | Bumped only on a breaking change to the frozen API surface (`api.md`). Additive changes do not bump it. |
| 3 | **Registry schema version** | `schema_version` in the registry and evidence files; `$id` of the JSON Schemas | Bumped when the JSON shape changes incompatibly. Readers reject a schema version they do not know. |
| 4 | **Store format version** | `format_version` u16 in every record header (`store-format-v1.md`) | Bumped on any change to record bytes that the flag/TLV mechanism cannot express. Old readers reject newer versions (`STORE_FORMAT_UNSUPPORTED`); they never guess. |
| 5 | **Kernel ABI version** | Registry backend entries; C symbol prefix `toktier_` | In the JIT delivery mode this axis versions the source-level kernel interface (bound via source digest + build flags + toolchain); when prebuilt kernels ship, it versions the stable C ABI between core and kernel package. |
| 6 | **Certification suite version** | `suite_version` in registry records and evidence manifests | Identifies the judgment battery that produced the evidence. New suite versions never silently re-label old evidence. |

## 2. Sub-axes bound inside the fingerprint

The semantic fingerprint (`fingerprint.md`) additionally binds a
**session API semantic version** (field `0x000B`). It is subordinate to
axis 2: it bumps only when append/return semantics change the meaning of
stored token streams, which both bumps the public API version and
invalidates stored sessions by construction (fingerprint change => miss).
The safe failure direction is built in: version drift produces misses
and re-encodes, never wrong ids.

## 3. Cross-axis discipline

- A package release may move any subset of axes; the changelog lists
  which axes moved and why.
- Certification claims name their axes explicitly: an entry certifies
  (artifact, oracle semantic id, suite version, backend constraints)  --
  never "version X of toktier" as a whole.
- Downgrade behavior is uniform: any component meeting data or metadata
  from a newer axis value it does not understand refuses or misses
  (per that surface's contract) rather than guessing.
