use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

// Single definition of the certification path sets, shared with
// crates/toktier-py/build.rs and cross-checked against the tools-side
// table by tools/generate_registry.py --check.
#[path = "build_support/source_identity.rs"]
mod source_identity;

use source_identity::{
    dependency_closure_status, fast_cpu_source_paths, hex, identity_sentinel_enabled,
    locate_governing_lock, native_host_source_paths, rust_api_source_paths, source_digest,
    Consumption, GoverningLock, JudgedPackage, CARGO_LOCK_ENV, FAST_CPU_DOMAIN,
    IDENTITY_SENTINEL_ENV, IDENTITY_SENTINEL_HEX, JUDGED_CLOSURE_RELATIVE, JUDGED_CLOSURE_SCHEMA,
    JUDGED_LOCK_RELATIVE, NATIVE_HOST_DOMAIN, RUST_API_DOMAIN,
};

const PACKAGE_IDENTITY_SCHEMA: &str = "toktier.rust_package_source_identity.v1";

/// The packages the judged build compiled, read from the record that
/// travels with the crate.
///
/// Every failure comes back as a reason string rather than a panic. A
/// missing, truncated, or malformed record means this build cannot be
/// compared with the judged one, which is a reason to withhold
/// certification -- not a reason to stop a build that is otherwise fine.
fn read_judged_closure(path: &Path) -> Result<Vec<JudgedPackage>, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("the judged compiled closure is unreadable: {error}"))?;
    let document: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("the judged compiled closure is unreadable: {error}"))?;
    if document.get("schema").and_then(serde_json::Value::as_str) != Some(JUDGED_CLOSURE_SCHEMA) {
        return Err("the judged compiled closure has an unexpected schema".to_owned());
    }
    let packages = document
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "the judged compiled closure names no packages".to_owned())?;
    packages
        .iter()
        .map(|entry| {
            let field = |name: &str| {
                entry
                    .get(name)
                    .and_then(serde_json::Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
            };
            match (field("name"), field("version")) {
                (Some(name), Some(version)) => Ok(JudgedPackage { name, version }),
                _ => Err("the judged compiled closure has an unreadable entry".to_owned()),
            }
        })
        .collect()
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let workspace_mode = root.join("crates/toktier-routing-core").is_dir();
    let computed_sources = if workspace_mode {
        let rust_api_paths = rust_api_source_paths(root);
        let fast_cpu_paths = fast_cpu_source_paths(root);
        let native_host_paths = native_host_source_paths(root);
        for path in rust_api_paths
            .iter()
            .chain(fast_cpu_paths.iter())
            .chain(native_host_paths.iter())
        {
            println!("cargo:rerun-if-changed={}", root.join(path).display());
        }
        (
            source_digest(root, RUST_API_DOMAIN, &rust_api_paths),
            source_digest(root, FAST_CPU_DOMAIN, &fast_cpu_paths),
            source_digest(root, NATIVE_HOST_DOMAIN, &native_host_paths),
        )
    } else {
        let identity_path = manifest.join("data/build/source_identity.json");
        println!("cargo:rerun-if-changed={}", identity_path.display());
        let identity: serde_json::Value = serde_json::from_slice(
            &fs::read(&identity_path).expect("read packaged source identity"),
        )
        .expect("parse packaged source identity");
        assert_eq!(
            identity.get("schema").and_then(serde_json::Value::as_str),
            Some(PACKAGE_IDENTITY_SCHEMA),
            "unexpected packaged source-identity schema"
        );
        let field = |name: &str| {
            let value = identity
                .get(name)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("packaged source identity has no {name}"));
            assert!(
                value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
                "invalid packaged source identity {name}"
            );
            value.to_owned()
        };
        (
            field("rust_api_source_sha256"),
            field("fast_cpu_source_sha256"),
            field("native_host_source_sha256"),
        )
    };
    // How this build's resolved dependency graph compares with the one
    // the certification evidence was taken on. In workspace mode the
    // lockfile is already hashed into the identities above; the check
    // runs there too, because a checkout used as a path dependency of a
    // larger workspace is governed by that workspace's lockfile rather
    // than by the one sitting next to these sources.
    let judged_lock = if workspace_mode {
        root.join("Cargo.lock")
    } else {
        manifest.join(JUDGED_LOCK_RELATIVE)
    };
    println!("cargo:rerun-if-changed={}", judged_lock.display());
    println!("cargo:rerun-if-env-changed={CARGO_LOCK_ENV}");
    let judged_closure_path = manifest.join(JUDGED_CLOSURE_RELATIVE);
    println!("cargo:rerun-if-changed={}", judged_closure_path.display());
    let judged_closure = read_judged_closure(&judged_closure_path);
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("out dir"));
    let consumption = if workspace_mode {
        Consumption::Workspace(manifest.as_path())
    } else {
        Consumption::Packaged(manifest.as_path())
    };
    let governing = locate_governing_lock(&out_dir, consumption);
    if let GoverningLock::Found(path) | GoverningLock::JudgedRecordNamed(path) = &governing {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let closure_status = dependency_closure_status(&judged_lock, &judged_closure, &governing);
    println!("cargo:rustc-env=TOKTIER_RUST_API_DEPENDENCY_CLOSURE={closure_status}");

    println!("cargo:rerun-if-env-changed={IDENTITY_SENTINEL_ENV}");
    let (rust_api_source, fast_cpu_source, native_host_source) = if identity_sentinel_enabled() {
        let sentinel = IDENTITY_SENTINEL_HEX.to_owned();
        (sentinel.clone(), sentinel.clone(), sentinel)
    } else {
        computed_sources
    };

    let support_registry = if workspace_mode {
        root.join("src/toktier/routing/tables/support_registry.v1.json")
    } else {
        manifest.join("data/src/toktier/routing/tables/support_registry.v1.json")
    };
    println!("cargo:rerun-if-changed={}", support_registry.display());
    let support_sha256 = hex(&Sha256::digest(
        fs::read(&support_registry).expect("read shipped support registry"),
    ));

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let rustc_version = Command::new(rustc)
        .arg("--version")
        .output()
        .expect("run rustc --version");
    assert!(rustc_version.status.success(), "rustc --version failed");
    let rustc_version = String::from_utf8(rustc_version.stdout)
        .expect("rustc version is UTF-8")
        .trim()
        .to_owned();

    let profile = env::var("PROFILE").expect("Cargo profile");
    let target = env::var("TARGET").expect("Cargo target");
    let opt_level = env::var("OPT_LEVEL").expect("Cargo opt level");
    let debug = env::var("DEBUG").unwrap_or_else(|_| "unknown".to_owned());
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    // Every entry below is something this build script can observe. The
    // three that are not -- `lto`, `codegen-units`, `panic` -- used to be
    // here as values inferred from the profile name, which is to say
    // stated rather than measured: a build with `lto = false` reported
    // `lto=fat` and matched the judged key on it. `CARGO_CFG_PANIC` is
    // not the way out, either. It reports the strategy of the build
    // script itself, which Cargo always compiles for the host with
    // unwinding, not the strategy of the library being built; under a
    // `panic = "abort"` profile it still reads `unwind`. Claiming
    // nothing about the three is the honest form of not knowing.
    // RUSTFLAGS, by contrast, is observable, and is what carries `-C`
    // codegen switches when someone sets them.
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .replace('\x1f', " ");
    let mut features = env::vars()
        .filter_map(|(key, _)| key.strip_prefix("CARGO_FEATURE_").map(str::to_owned))
        .map(|value| value.to_ascii_lowercase().replace('_', "-"))
        .collect::<Vec<_>>();
    features.sort();
    let flags = [
        format!("profile={profile}"),
        format!("opt-level={opt_level}"),
        format!("target={target}"),
        format!("debug={debug}"),
        format!("target-features={target_features}"),
        format!("rustflags={rustflags}"),
        format!("features={}", features.join(",")),
    ]
    .join("\x1f");

    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_SOURCE_SHA256={}",
        rust_api_source
    );
    println!("cargo:rustc-env=TOKTIER_RUST_API_TOOLCHAIN={rustc_version}");
    println!("cargo:rustc-env=TOKTIER_RUST_API_BUILD_FLAGS={flags}");
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_FAST_CPU_SOURCE_SHA256={}",
        fast_cpu_source
    );
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256={}",
        native_host_source
    );
    println!("cargo:rustc-env=TOKTIER_SUPPORT_REGISTRY_SHA256={support_sha256}");
}
