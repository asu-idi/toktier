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
5. **The alias an import cannot install gets its own code** (added
   2026-08-28, released in 0.2.8) - Re-importing an air-gap bundle into a
   cache that already holds its alias is idempotent when the installed
   tree still authenticates as that bundle, which is the rule
   `docs/rust-lifecycle.md` has always stated for the Rust face and the
   rule the Python `import_bundle` now applies too. The remaining
   refusal has one subject -- the tree the cache holds is not this
   bundle -- and raises the new `ALIAS_CONFLICT` code (class
   `AliasConflict`), on both faces, with `path` naming the first file
   that does not match. This supersedes item 2's sentence that
   cache-install failures stay `ARTIFACT_NOT_FOUND` for this one
   condition: an install that fails for any other reason still reports
   `ARTIFACT_NOT_FOUND` with `cause: install_failed`. On the Rust face
   the same condition previously reported `BUNDLE_INVALID`,
   `ARTIFACT_NOT_FOUND` or `ARTIFACT_HASH_MISMATCH` depending on which
   part of the tree disagreed; code matching on those for a re-import
   needs the one-line update, as the 0.2.4 path-policy move did.
   Adding a code is a permitted minor-release extension by the same
   reasoning as item 1.
6. **The path an `ALIAS_CONFLICT` names is chosen in two phases**
   (added 2026-08-31, released in 0.2.9) - Item 5 describes `path` as
   naming "the first file that does not match". That is the outcome
   only for a tree whose sole disagreement is a declared file. The rule
   both faces implement, which `bundle.rs::verify_installed` states in
   its own rustdoc, has a phase in front of that one: anything the tree
   holds that the bundle does not declare -- an undeclared file, a
   symbolic link, a special file, at any depth -- is named first, and
   only a tree with no such entry is searched for the first declared
   path that is missing or does not match. Within either phase the
   relative path that sorts first is the one named, so the same tree
   names the same file on every run and on both faces. This corrects
   item 5's description rather than changing anything: no code moved
   for it, and the `ALIAS_CONFLICT` code, the `AliasConflict` class
   name and the `details` key list are untouched.
   `docs/contracts/errors.md`, `docs/contracts/facade.md` and
   `docs/rust-lifecycle.md` state the rule in this form.
