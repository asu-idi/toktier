//! Single definition of the certification source-identity path sets.
//!
//! Both build scripts (`crates/toktier/build.rs` and
//! `crates/toktier-py/build.rs`) include this module with `#[path]`, so
//! the hashed path lists exist exactly once on the Rust side. The
//! tools-side transcription lives in `tools/source_identity_common.py`;
//! `tools/generate_registry.py --check` cross-checks the two against
//! each other and watches routing-core for unenrolled source files.
//!
//! A source identity is a bare SHA-256 over a domain tag plus
//! length-framed path/content pairs, in `PathBuf` component order.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub const RUST_API_DOMAIN: &[u8] = b"toktier.rust_api.integrated_source.v1\0";
pub const FAST_CPU_DOMAIN: &[u8] = b"toktier.fast_cpu.integrated_source.v1\0";
pub const NATIVE_HOST_DOMAIN: &[u8] = b"toktier.prebuilt.native_host_source.v1\0";
pub const IDENTITY_SENTINEL_ENV: &str = "TOKTIER_IDENTITY_SENTINEL";
pub const IDENTITY_SENTINEL_HEX: &str = concat!(
    "73656e74696e656c",
    "73656e74696e656c",
    "73656e74696e656c",
    "73656e74696e656c",
);

pub fn identity_sentinel_enabled() -> bool {
    env::var_os(IDENTITY_SENTINEL_ENV).as_deref() == Some(OsStr::new("1"))
}

// ---------------------------------------------------------------------
// Resolved dependency closure.
// ---------------------------------------------------------------------
//
// In workspace mode the lockfile is one of the hashed files above, so
// the resolved graph is already part of every identity. A build from an
// unpacked registry copy cannot hash the sibling crates, replays the
// recorded digests instead, and would otherwise say nothing at all
// about how Cargo resolved this build's transitive versions. The
// helpers below let the build script compare the graph that governs
// this build against the graph that was judged, and report the answer;
// `Registry::rust_api_build_certified` requires `verified`.

/// Explicit pointer to the lockfile that governs this build, for
/// layouts where it cannot be found by walking up from `OUT_DIR`.
pub const CARGO_LOCK_ENV: &str = "TOKTIER_CARGO_LOCK";

/// The judged lockfile as shipped inside the crate archive.
pub const JUDGED_LOCK_RELATIVE: &str = "data/build/judged_dependencies.lock";

/// The package whose transitive closure is compared.
pub const CLOSURE_ROOT: &str = "toktier";

/// One `[[package]]` block of a Cargo lockfile.
pub struct LockPackage {
    pub name: String,
    pub version: String,
    pub source: Option<String>,
    pub checksum: Option<String>,
    pub dependencies: Vec<String>,
}

fn quoted_value(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim_start();
    let rest = rest.strip_prefix('=')?.trim();
    let rest = rest.strip_prefix('"')?;
    rest.strip_suffix('"').map(str::to_owned)
}

/// Parse the `[[package]]` blocks of a Cargo lockfile.
///
/// Cargo writes this file, so the accepted shape is deliberately narrow:
/// only blocks introduced by `[[package]]` are read, which leaves
/// `[[patch.unused]]` and `[metadata]` out by construction.
pub fn parse_lock(text: &str) -> Vec<LockPackage> {
    let mut packages = Vec::new();
    let mut current: Option<LockPackage> = None;
    let mut in_dependencies = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_dependencies = false;
            if let Some(package) = current.take() {
                packages.push(package);
            }
            if trimmed == "[[package]]" {
                current = Some(LockPackage {
                    name: String::new(),
                    version: String::new(),
                    source: None,
                    checksum: None,
                    dependencies: Vec::new(),
                });
            }
            continue;
        }
        let Some(package) = current.as_mut() else {
            continue;
        };
        if in_dependencies {
            if trimmed.starts_with(']') {
                in_dependencies = false;
            } else if let Some(entry) = trimmed.strip_prefix('"') {
                if let Some(entry) = entry.split('"').next() {
                    package.dependencies.push(entry.to_owned());
                }
            }
            continue;
        }
        if trimmed.starts_with("dependencies") && trimmed.ends_with('[') {
            in_dependencies = true;
        } else if let Some(value) = quoted_value(trimmed, "name") {
            package.name = value;
        } else if let Some(value) = quoted_value(trimmed, "version") {
            package.version = value;
        } else if let Some(value) = quoted_value(trimmed, "source") {
            package.source = Some(value);
        } else if let Some(value) = quoted_value(trimmed, "checksum") {
            package.checksum = Some(value);
        }
    }
    if let Some(package) = current.take() {
        packages.push(package);
    }
    packages
}

fn resolve<'a>(packages: &'a [LockPackage], entry: &str) -> Vec<&'a LockPackage> {
    let mut fields = entry.split_whitespace();
    let name = fields.next().unwrap_or_default();
    let version = fields.next();
    // An entry without a version is unambiguous by Cargo's own rules;
    // taking every same-named package if one ever is not keeps the
    // comparison on the conservative side.
    packages
        .iter()
        .filter(|package| {
            package.name == name && version.is_none_or(|value| package.version == value)
        })
        .collect()
}

/// Every package reachable from `root`, itself included.
///
/// A lockfile's dependency lists are the union over features and
/// targets, so this is a superset of what any one build links: judging
/// too much rather than too little.
pub fn closure<'a>(packages: &'a [LockPackage], root: &str) -> Vec<&'a LockPackage> {
    let mut seen: Vec<&LockPackage> = Vec::new();
    let mut queue: Vec<&LockPackage> = packages
        .iter()
        .filter(|package| package.name == root)
        .collect();
    while let Some(package) = queue.pop() {
        if seen.iter().any(|found| std::ptr::eq(*found, package)) {
            continue;
        }
        seen.push(package);
        for entry in &package.dependencies {
            queue.extend(resolve(packages, entry));
        }
    }
    seen.sort_by(|left, right| {
        (&left.name, &left.version)
            .cmp(&(&right.name, &right.version))
            .then_with(|| left.checksum.cmp(&right.checksum))
    });
    seen
}

/// Whether every package this build resolves is the judged one.
///
/// The direction is deliberate: each package in `found` must appear in
/// `judged`. Packages the judged graph carries but this build does not
/// resolve (the root's dev-dependencies, which a consumer never builds)
/// are not required to appear.
pub fn compare_closures(judged: &[LockPackage], found: &[LockPackage]) -> Result<(), String> {
    let judged_closure = closure(judged, CLOSURE_ROOT);
    let found_closure = closure(found, CLOSURE_ROOT);
    if found_closure.is_empty() {
        return Err(format!(
            "no {CLOSURE_ROOT} package in the governing lockfile"
        ));
    }
    for package in found_closure {
        let candidates = judged_closure
            .iter()
            .filter(|other| other.name == package.name && other.version == package.version)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(format!(
                "{} {} is not judged",
                package.name, package.version
            ));
        }
        let accepted = candidates.iter().any(|other| match &other.checksum {
            // A registry package is judged by its content hash.
            Some(checksum) => package.checksum.as_deref() == Some(checksum.as_str()),
            // The judged graph holds this crate's own siblings as
            // workspace members, which carry no checksum. Their exact
            // versions are pinned by this crate's manifest; requiring a
            // registry origin keeps a `[patch]` to some other source
            // from passing as the judged crate.
            None => package
                .source
                .as_deref()
                .is_none_or(|source| source.starts_with("registry+")),
        });
        if !accepted {
            return Err(format!(
                "{} {} does not match the judged package",
                package.name, package.version
            ));
        }
    }
    Ok(())
}

/// The lockfile that governs this build, if it can be named.
///
/// `OUT_DIR` sits inside the target directory of the build that is
/// consuming this crate, so its ancestors reach that build's workspace
/// root. The unpacked crate's own directory is searched only in
/// workspace mode: a registry copy ships the lockfile recorded at
/// publication time, and comparing that against itself would answer
/// nothing.
pub fn locate_governing_lock(out_dir: &Path, manifest_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(explicit) = env::var_os(CARGO_LOCK_ENV) {
        let path = PathBuf::from(explicit);
        return path.is_file().then_some(path);
    }
    let roots = [Some(out_dir), manifest_dir];
    for root in roots.into_iter().flatten() {
        for ancestor in root.ancestors() {
            let candidate = ancestor.join("Cargo.lock");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// One line naming how this build's resolved graph compares with the
/// judged one: `verified`, `unlocated`, or `mismatched: <reason>`.
pub fn dependency_closure_status(judged_lock: &Path, governing: Option<&Path>) -> String {
    let Some(governing) = governing else {
        return "unlocated".to_owned();
    };
    let judged = match fs::read_to_string(judged_lock) {
        Ok(text) => parse_lock(&text),
        Err(error) => return format!("mismatched: judged lockfile is unreadable: {error}"),
    };
    let found = match fs::read_to_string(governing) {
        Ok(text) => parse_lock(&text),
        Err(error) => return format!("mismatched: governing lockfile is unreadable: {error}"),
    };
    match compare_closures(&judged, &found) {
        Ok(()) => "verified".to_owned(),
        Err(reason) => format!("mismatched: {reason}"),
    }
}

fn collect_tree(root: &Path, path: &Path, output: &mut Vec<PathBuf>) {
    let mut entries = fs::read_dir(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
        .map(|entry| entry.expect("source directory entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            collect_tree(root, &entry, output);
        } else if entry.is_file() {
            output.push(entry.strip_prefix(root).expect("workspace path").to_owned());
        }
    }
}

fn paths_from(root: &Path, files: &[&str], trees: &[&str]) -> Vec<PathBuf> {
    let mut paths = files.iter().map(PathBuf::from).collect::<Vec<_>>();
    for tree in trees {
        collect_tree(root, &root.join(tree), &mut paths);
    }
    paths.sort();
    paths.dedup();
    paths
}

pub fn fast_cpu_source_paths(root: &Path) -> Vec<PathBuf> {
    paths_from(
        root,
        &[
            "Cargo.lock",
            "Cargo.toml",
            "pyproject.toml",
            "rust-toolchain.toml",
            "crates/toktier/build_support/source_identity.rs",
            "crates/toktier-routing-core/Cargo.toml",
            "crates/toktier-routing-core/src/fast_cpu.rs",
            "crates/toktier-routing-core/src/lib.rs",
            "crates/toktier-routing-core/src/reference.rs",
            "crates/toktier-routing-core/src/runtime.rs",
            "crates/toktier-store-core/Cargo.toml",
            "crates/toktier-py/Cargo.toml",
            "crates/toktier-py/build.rs",
            "crates/toktier-py/src/lib.rs",
            "src/toktier/backends/fast_cpu.py",
            "src/toktier/repair/tables/fast_repair_families.v1.json",
            "src/toktier/repair/tables/repair_pclass.v1.zlib",
        ],
        &[
            "crates/toktier-gigatoken-core",
            "crates/toktier-store-core/src",
        ],
    )
}

pub fn native_host_source_paths(root: &Path) -> Vec<PathBuf> {
    paths_from(
        root,
        &[
            "Cargo.lock",
            "Cargo.toml",
            "pyproject.toml",
            "rust-toolchain.toml",
            "crates/toktier/build_support/source_identity.rs",
            "crates/toktier-py/Cargo.toml",
            "crates/toktier-py/build.rs",
            "crates/toktier-py/src/lib.rs",
            "src/toktier/engine/gpu/native.py",
            "src/toktier/kernels/bpe_tables.py",
        ],
        &[
            "crates/toktier-cuda-driver",
            "crates/toktier-routing-core",
            "crates/toktier-store-core",
            "crates/toktier-store-sqlite",
        ],
    )
}

pub fn rust_api_source_paths(root: &Path) -> Vec<PathBuf> {
    paths_from(
        root,
        &[
            "Cargo.lock",
            "Cargo.toml",
            "rust-toolchain.toml",
            "crates/toktier/Cargo.toml",
            "crates/toktier/build.rs",
            "crates/toktier/build_support/source_identity.rs",
            "src/toktier/repair/tables/fast_repair_families.v1.json",
            "src/toktier/repair/tables/repair_pclass.v1.zlib",
        ],
        &[
            "crates/toktier/src",
            "crates/toktier-cuda-driver",
            "crates/toktier-gigatoken-core",
            "crates/toktier-routing-core",
            "crates/toktier-store-core",
            "crates/toktier-store-sqlite",
        ],
    )
}

pub fn source_digest(root: &Path, domain: &[u8], paths: &[PathBuf]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    for relative in paths {
        let rendered = relative
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        let path = rendered.as_bytes();
        let content = fs::read(root.join(relative))
            .unwrap_or_else(|error| panic!("cannot read {rendered}: {error}"));
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path);
        digest.update((content.len() as u64).to_le_bytes());
        digest.update(content);
    }
    hex(digest.finalize().as_slice())
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::{closure, compare_closures, parse_lock};

    const JUDGED: &str = r#"
version = 4

[[package]]
name = "toktier"
version = "0.2.3"
dependencies = [
 "serde",
 "tempfile",
 "toktier-routing-core",
]

[[package]]
name = "serde"
version = "1.0.228"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "aaaa"
dependencies = [
 "memchr",
]

[[package]]
name = "memchr"
version = "2.8.3"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "bbbb"

[[package]]
name = "tempfile"
version = "3.23.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "cccc"

[[package]]
name = "toktier-routing-core"
version = "0.2.3"

[[package]]
name = "unrelated"
version = "9.9.9"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "dddd"

[[patch.unused]]
name = "serde"
version = "1.0.999"
"#;

    /// A consumer's lockfile: no dev-dependency of the root, and the
    /// sibling crate arrives from the registry rather than as a
    /// workspace member.
    const CONSUMER: &str = r#"
version = 4

[[package]]
name = "toktier"
version = "0.2.3"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "eeee"
dependencies = [
 "serde",
 "toktier-routing-core",
]

[[package]]
name = "serde"
version = "1.0.228"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "aaaa"
dependencies = [
 "memchr 2.8.3",
]

[[package]]
name = "memchr"
version = "2.8.3"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "bbbb"

[[package]]
name = "toktier-routing-core"
version = "0.2.3"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "ffff"
"#;

    #[test]
    fn parsing_reads_packages_and_skips_other_blocks() {
        let packages = parse_lock(JUDGED);
        assert_eq!(packages.len(), 6);
        let serde = packages
            .iter()
            .find(|package| package.name == "serde")
            .unwrap();
        assert_eq!(serde.version, "1.0.228");
        assert_eq!(serde.checksum.as_deref(), Some("aaaa"));
        assert_eq!(serde.dependencies, vec!["memchr".to_owned()]);
        // The unused patch entry is not a package.
        assert!(packages.iter().all(|package| package.version != "1.0.999"));
    }

    #[test]
    fn the_closure_is_what_the_root_reaches() {
        let packages = parse_lock(JUDGED);
        let names = closure(&packages, "toktier")
            .into_iter()
            .map(|package| package.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "memchr",
                "serde",
                "tempfile",
                "toktier",
                "toktier-routing-core"
            ]
        );
    }

    #[test]
    fn a_matching_consumer_graph_is_accepted() {
        assert_eq!(
            compare_closures(&parse_lock(JUDGED), &parse_lock(CONSUMER)),
            Ok(())
        );
    }

    #[test]
    fn a_drifted_transitive_version_is_refused() {
        let drifted = CONSUMER
            .replace("memchr 2.8.3", "memchr 2.8.4")
            .replace("version = \"2.8.3\"", "version = \"2.8.4\"");
        assert_eq!(
            compare_closures(&parse_lock(JUDGED), &parse_lock(&drifted)),
            Err("memchr 2.8.4 is not judged".to_owned())
        );
    }

    #[test]
    fn a_repackaged_registry_crate_is_refused() {
        let repackaged = CONSUMER.replace("checksum = \"bbbb\"", "checksum = \"b0b0\"");
        assert_eq!(
            compare_closures(&parse_lock(JUDGED), &parse_lock(&repackaged)),
            Err("memchr 2.8.3 does not match the judged package".to_owned())
        );
    }

    #[test]
    fn a_patched_sibling_source_is_refused() {
        let patched = CONSUMER.replace(
            "source = \"registry+https://github.com/rust-lang/crates.io-index\"\nchecksum = \"ffff\"",
            "source = \"git+https://example.invalid/fork\"",
        );
        assert_eq!(
            compare_closures(&parse_lock(JUDGED), &parse_lock(&patched)),
            Err("toktier-routing-core 0.2.3 does not match the judged package".to_owned())
        );
    }

    #[test]
    fn a_lockfile_without_the_root_is_refused() {
        let without = CONSUMER.replace("name = \"toktier\"", "name = \"somebody-else\"");
        assert_eq!(
            compare_closures(&parse_lock(JUDGED), &parse_lock(&without)),
            Err("no toktier package in the governing lockfile".to_owned())
        );
    }

    #[test]
    fn the_shipped_judged_lockfile_matches_the_workspace_lockfile() {
        // The packaged copy is what a registry build compares against;
        // a stale copy would judge the wrong graph.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .unwrap()
            .to_path_buf();
        let workspace = std::fs::read(root.join("Cargo.lock")).unwrap();
        let packaged = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(super::JUDGED_LOCK_RELATIVE),
        )
        .unwrap();
        assert_eq!(workspace, packaged);
    }
}
