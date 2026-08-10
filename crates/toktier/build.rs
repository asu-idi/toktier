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
    fast_cpu_source_paths, hex, native_host_source_paths, rust_api_source_paths, source_digest,
    FAST_CPU_DOMAIN, NATIVE_HOST_DOMAIN, RUST_API_DOMAIN,
};

const PACKAGE_IDENTITY_SCHEMA: &str = "toktier.rust_package_source_identity.v1";

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let workspace_mode = root.join("crates/toktier-routing-core").is_dir();
    let (rust_api_source, fast_cpu_source, native_host_source) = if workspace_mode {
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
    let mut features = env::vars()
        .filter_map(|(key, _)| key.strip_prefix("CARGO_FEATURE_").map(str::to_owned))
        .map(|value| value.to_ascii_lowercase().replace('_', "-"))
        .collect::<Vec<_>>();
    features.sort();
    let (lto, codegen_units) = if profile == "release" {
        ("fat", "1")
    } else {
        ("off", "default")
    };
    let flags = [
        format!("profile={profile}"),
        format!("opt-level={opt_level}"),
        format!("lto={lto}"),
        format!("codegen-units={codegen_units}"),
        "panic=unwind".to_owned(),
        format!("target={target}"),
        format!("debug={debug}"),
        format!("target-features={target_features}"),
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
