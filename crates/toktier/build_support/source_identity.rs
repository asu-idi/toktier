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
