use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

// Single definition of the certification path sets, shared with
// crates/toktier-py/build.rs and cross-checked against the tools-side
// table by tools/generate_registry.py --check. Three includers use
// different parts of it -- this script asks for every answer at once,
// the sibling script asks only for two digests, and the crate's test
// build asks for the comparison one subset at a time -- so each include
// carries the allowance rather than the module dropping helpers that
// another includer needs.
#[path = "build_support/source_identity.rs"]
#[allow(dead_code)]
mod source_identity;

// Which lockfile this build reads: the refusal of a `TOKTIER_CARGO_LOCK`
// value that cannot name a consumer's lockfile, and the word that says how
// the lockfile was located. The library compiles the same file into its
// test build, where its rules are tested.
#[path = "src/lock_reading.rs"]
mod lock_reading;

use source_identity::{
    check_lock_format, closure_statuses, fast_cpu_source_paths, hex, identity_sentinel_enabled,
    locate_governing_lock, native_host_source_paths, parse_lock, rust_api_source_paths,
    source_digest, Consumption, GoverningLock, JudgedPackage, Tier, CARGO_LOCK_ENV,
    FAST_CPU_DOMAIN, IDENTITY_SENTINEL_ENV, IDENTITY_SENTINEL_HEX, JUDGED_CLOSURE_RELATIVE,
    JUDGED_CLOSURE_SCHEMA, JUDGED_LOCK_RELATIVE, NATIVE_HOST_DOMAIN, RUST_API_DOMAIN,
};

const PACKAGE_IDENTITY_SCHEMA: &str = "toktier.rust_package_source_identity.v1";

/// The two crates whose copies decide the property-data standing
/// (`src/property_guard.rs`).
const PROPERTY_CRATES: [&str; 2] = ["icu_properties", "icu_properties_data"];
/// The variable with which the ICU4X data crates compile the baked data of
/// a directory in place of their own.
const DATA_DIR_ENV: &str = "ICU4X_DATA_DIR";
/// Between the records of the property-data reading, and between the
/// fields of one record: the separators `src/property_guard.rs` reads.
const PROPERTY_RECORD_SEPARATOR: char = '\u{1f}';
const PROPERTY_FIELD_SEPARATOR: char = '\u{1e}';

/// One record per copy of the two property crates in the governing
/// lockfile: name, version, source, checksum (empty when absent). A
/// lockfile that cannot be read gives a one-field record naming why, which
/// the library refuses as unreadable (untabled), never as judged.
fn property_data_record(governing: &GoverningLock) -> String {
    let GoverningLock::Found(path) = governing else {
        return String::new();
    };
    let clean = |value: &str| {
        value
            .chars()
            .map(|character| {
                if character.is_control() {
                    ' '
                } else {
                    character
                }
            })
            .collect::<String>()
    };
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => return clean(&format!("the governing lockfile is unreadable: {error}")),
    };
    if let Err(reason) = check_lock_format("the governing lockfile", &text) {
        return clean(&reason);
    }
    parse_lock(&text)
        .into_iter()
        .filter(|package| PROPERTY_CRATES.contains(&package.name.as_str()))
        .map(|package| {
            [
                clean(&package.name),
                clean(&package.version),
                clean(package.source.as_deref().unwrap_or_default()),
                clean(package.checksum.as_deref().unwrap_or_default()),
            ]
            .join(&PROPERTY_FIELD_SEPARATOR.to_string())
        })
        .collect::<Vec<_>>()
        .join(&PROPERTY_RECORD_SEPARATOR.to_string())
}

/// The packages the judged build compiled, read from the record that
/// travels with the crate.
///
/// Every failure comes back as a reason string rather than a panic. A
/// missing, truncated, or malformed record means this build cannot be
/// compared with the judged one, which is a reason to withhold
/// certification -- not a reason to stop a build that is otherwise fine.
///
/// An entry that names no tier, or one this reader does not know, is
/// read as part of the certified core: a package this reader cannot
/// place decides the certificate rather than being reported next to it.
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
            // The certified reference versions, when the entry carries
            // them. An entry that names the key with anything other than
            // a list of non-empty strings is unreadable, which withholds
            // certification rather than widening it.
            let certified_versions = match entry.get("certified_versions") {
                None => Ok(Vec::new()),
                Some(serde_json::Value::Array(values)) => values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .filter(|version| !version.is_empty())
                            .map(str::to_owned)
                            .ok_or_else(|| {
                                "the judged compiled closure has an unreadable certified version"
                                    .to_owned()
                            })
                    })
                    .collect::<Result<Vec<_>, _>>(),
                Some(_) => Err(
                    "the judged compiled closure has an unreadable certified version".to_owned(),
                ),
            }?;
            let optional = match entry.get("optional") {
                None => false,
                Some(serde_json::Value::Bool(value)) => *value,
                Some(_) => {
                    return Err(
                        "the judged compiled closure has an unreadable optional mark".to_owned(),
                    )
                }
            };
            match (field("name"), field("version")) {
                (Some(name), Some(version)) => Ok(JudgedPackage {
                    name,
                    version,
                    tier: Tier::from_record(field("tier").as_deref()),
                    behavior_unit: field("behavior_unit"),
                    behavior_version: field("behavior_version"),
                    behavior_source: field("behavior_source"),
                    certified_versions,
                    optional,
                }),
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
    // A value of the variable that cannot name the lockfile of a build that
    // consumes this crate (empty, relative, or inside this crate's own
    // directory) is refused, and every reading below is then taken as
    // missing, the closure readings too, so that no line of `doctor()`
    // says "certified" beside an unread property-data line.
    let named = env::var_os(CARGO_LOCK_ENV);
    let refusal = named
        .as_deref()
        .and_then(|value| lock_reading::refuse_named(value, &manifest));
    let governing = if refusal.is_some() {
        GoverningLock::Missing
    } else {
        locate_governing_lock(&out_dir, consumption)
    };
    // A lockfile that holds no package of this crate's name is the lockfile
    // of another build (an unrelated project above an outside target
    // directory), and is not read either. It stays watched, so a change to
    // it reruns this script.
    let package_name = env::var("CARGO_PKG_NAME").expect("Cargo package name");
    let mut foreign = None;
    let governing = match governing {
        GoverningLock::Found(path) => match fs::read_to_string(&path) {
            Ok(text) if !lock_reading::holds_package(&parse_lock(&text), &package_name) => {
                println!("cargo:rerun-if-changed={}", path.display());
                foreign = Some(path);
                GoverningLock::Missing
            }
            _ => GoverningLock::Found(path),
        },
        other => other,
    };
    if let GoverningLock::Found(path) | GoverningLock::JudgedRecordNamed(path) = &governing {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    // The property-data reading of the same lockfile, for the guard of
    // the fast CPU route (`src/property_guard.rs`).
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_PROPERTY_DATA={}",
        property_data_record(&governing)
    );
    // `ICU4X_DATA_DIR` makes the ICU4X data crates compile the baked data of
    // that directory in place of their own, while the lockfile still names
    // the copy; the library then reads the standing as untabled. Taken as
    // set whenever it is present, which is a little wider than the data
    // crates' own test and so on the safe side.
    println!("cargo:rerun-if-env-changed={DATA_DIR_ENV}");
    let data_dir_set = env::var_os(DATA_DIR_ENV).is_some();
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_PROPERTY_DATA_DIR={}",
        if data_dir_set { "set" } else { "" }
    );
    if data_dir_set {
        println!(
            "cargo:warning=toktier: {DATA_DIR_ENV} is set, so the property data compiled into \
             this build is that directory's rather than the data of the copies the lockfile \
             names, and the fast CPU route is not built (ReasonCode::PropertyDataUntabled); \
             unset {DATA_DIR_ENV} to build the data the lockfile names"
        );
    }
    let located = lock_reading::located_by(&governing, named.is_some(), &out_dir);
    println!("cargo:rustc-env=TOKTIER_RUST_API_LOCK_LOCATED_BY={located}");
    if matches!(located, "manifest" | "none") {
        // Cargo shows this for a path dependency and for this workspace's
        // own builds, not for a `git` or registry dependency; there the
        // plan's reasons and `doctor()` carry the same reading.
        let why = lock_reading::unread_reason(
            refusal.as_ref(),
            named.as_deref().map(Path::new),
            foreign.as_deref(),
            &governing,
        );
        println!(
            "cargo:warning=toktier: {why}, so the fast CPU route is not built \
             (ReasonCode::PropertyDataUnread); set {CARGO_LOCK_ENV} to the absolute path of the \
             Cargo.lock of the build that consumes this crate"
        );
    }
    // One reading of the two lockfiles, reported five ways: the whole
    // closure as before; the certified core, which is what decides
    // certification; the core packages whose behaviour version the
    // running binary reads for itself, which the build script can only
    // report the package move of; everything outside the core and the
    // implementation libraries, which travel as an advisory; and the
    // reference libraries judged by crate version, each with the version
    // this build resolved beside its certified reference versions. The
    // judged behaviour versions travel with them so the runtime has both
    // sides of that comparison.
    let statuses = closure_statuses(&judged_lock, &judged_closure, &governing);
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_DEPENDENCY_CLOSURE={}",
        statuses.strict
    );
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_CORE_CLOSURE_STATIC={}",
        statuses.core_static
    );
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_R2_DRIFT={}",
        statuses.r2_drift
    );
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_DEPENDENCY_ADVISORY={}",
        statuses.advisory
    );
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_JUDGED_BEHAVIOR={}",
        statuses.judged_behavior
    );
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_REFERENCE_VERSIONS={}",
        statuses.reference_versions
    );
    println!(
        "cargo:rustc-env=TOKTIER_RUST_API_REFERENCE_UNCERTIFIED={}",
        statuses.reference_uncertified
    );

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
    // The chunked admission table is regenerated whenever the native
    // engine's identity moves, so its digest is read here rather than
    // written by hand beside the payload, the way the support registry's
    // is.
    let chunked_admission = if workspace_mode {
        root.join("src/toktier/routing/tables/chunked_admission.v1.json")
    } else {
        manifest.join("data/src/toktier/routing/tables/chunked_admission.v1.json")
    };
    println!("cargo:rerun-if-changed={}", chunked_admission.display());
    let chunked_admission_sha256 = hex(&Sha256::digest(
        fs::read(&chunked_admission).expect("read shipped chunked admission table"),
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
    println!("cargo:rustc-env=TOKTIER_CHUNKED_ADMISSION_SHA256={chunked_admission_sha256}");
}
