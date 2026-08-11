use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

// Single definition of the certification path sets, shared with
// crates/toktier/build.rs and cross-checked against the tools-side
// table by tools/generate_registry.py --check. The rust_api helpers in
// the module serve the sibling build script only.
#[path = "../toktier/build_support/source_identity.rs"]
#[allow(dead_code)]
mod source_identity;

use source_identity::{
    fast_cpu_source_paths, identity_sentinel_enabled, native_host_source_paths, source_digest,
    FAST_CPU_DOMAIN, IDENTITY_SENTINEL_ENV, IDENTITY_SENTINEL_HEX, NATIVE_HOST_DOMAIN,
};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let fast_cpu_paths = fast_cpu_source_paths(root);
    let native_host_paths = native_host_source_paths(root);
    for path in fast_cpu_paths.iter().chain(native_host_paths.iter()) {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    let computed_digests = (
        source_digest(root, FAST_CPU_DOMAIN, &fast_cpu_paths),
        source_digest(root, NATIVE_HOST_DOMAIN, &native_host_paths),
    );
    println!("cargo:rerun-if-env-changed={IDENTITY_SENTINEL_ENV}");
    let (fast_cpu_digest, native_host_digest) = if identity_sentinel_enabled() {
        let sentinel = IDENTITY_SENTINEL_HEX.to_owned();
        (sentinel.clone(), sentinel)
    } else {
        computed_digests
    };

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
        "pyo3=abi3-py310+extension-module".to_owned(),
    ]
    .join("\x1f");

    println!("cargo:rustc-env=TOKTIER_FAST_CPU_SOURCE_SHA256={fast_cpu_digest}");
    println!("cargo:rustc-env=TOKTIER_FAST_CPU_TOOLCHAIN={rustc_version}");
    println!("cargo:rustc-env=TOKTIER_FAST_CPU_BUILD_FLAGS={flags}");
    println!("cargo:rustc-env=TOKTIER_NATIVE_HOST_SOURCE_SHA256={native_host_digest}");
    println!("cargo:rustc-env=TOKTIER_NATIVE_HOST_TOOLCHAIN={rustc_version}");
    println!("cargo:rustc-env=TOKTIER_NATIVE_HOST_BUILD_FLAGS={flags}");
}
