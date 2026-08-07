# 0004 - Error surface rulings (2026-08-05)

1. **Bundle format failures get their own code** - Air-gap bundle
   validation failures raise the new `BUNDLE_INVALID` code (class
   `BundleInvalid`), not `REGISTRY_INVALID`. The frozen meaning of
   `REGISTRY_INVALID` covers the support registry and evidence
   manifests, extended by ruling 0003 item 6 to malformed artifact
   manifests; the surrounding archive format of an air-gap bundle is a
   different condition and re-using the code would re-mean it. The code
   table in `docs/contracts/errors.md` is append-only, so adding
   `BUNDLE_INVALID` is a permitted minor-release extension.
2. **Scope of `BUNDLE_INVALID`** - Everything about the bundle
   container and its embedded `bundle_manifest.json`: tar read and I/O
   failures, path traversal or absolute member paths, symbolic and hard
   link members, duplicate members, member-count and uncompressed-size
   limit violations, member-set mismatches against the manifest, and a
   bundle manifest that fails parsing, schema, or root digest
   verification. Export-side validation of a bundle being produced uses
   the same code. Unchanged: content-hash failures of artifact files
   inside a bundle stay `ARTIFACT_HASH_MISMATCH`; a missing bundle
   file, a missing requested member, and cache-install failures stay
   `ARTIFACT_NOT_FOUND`.
3. **Structured details for `BUNDLE_INVALID`** - `path` (the bundle
   file) and `failure` are always present; `member` names the archive
   member when one is implicated; `cause` carries the underlying I/O or
   parse error text when one exists.
4. **Native exceptions are the public classes** - The `toktier._native`
   extension does not define a public exception hierarchy of its own.
   At the binding boundary it instantiates the classes from
   `toktier.errors`, so a native failure is caught by
   `toktier.ToktierError`, carries the frozen `.code`, and exposes
   `.details` as a read-only mapping exactly like a Python-raised
   error. The names re-exported on `toktier._native` are the same
   class objects. Only when the extension is loaded standalone without
   the `toktier` package on the path does a private
   contract-equivalent shim stand in; the shim
   observes the same `.code`/read-only-`.details` shape and is not a
   public surface.
