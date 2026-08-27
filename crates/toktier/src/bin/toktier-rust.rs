//! Minimal Python-free lifecycle CLI over the public Rust API.

use std::error::Error as StdError;

use serde_json::{json, Value};
use toktier::{ArtifactManager, ArtifactSource, DoctorFacts, Runtime};

fn usage() -> &'static str {
    "toktier-rust doctor [--json]\n\
     toktier-rust artifacts fetch|verify|inspect FAMILY [--cache PATH] [--offline]\n\
     toktier-rust artifacts mirror FAMILY --out DIRECTORY [--cache PATH]\n\
     toktier-rust artifacts export FAMILY --out BUNDLE [--cache PATH]\n\
     toktier-rust artifacts import BUNDLE [--cache PATH] [--offline]\n\
     toktier-rust gpu compile FAMILY [--device N] [--jit-cache PATH] [--nvcc PATH] [--accept-uncertified-jit]\n\
     toktier-rust verify-local --family FAMILY [--engine cpu|gpu|both] [--input PATH|-] [--synthetic N] [--max-bytes N] [--seed N] [--delivery prebuilt|jit|auto] [--device N] [--json] [--forget]"
}

fn flag_value(arguments: &[String], flag: &str) -> Result<Option<String>, Box<dyn StdError>> {
    let Some(position) = arguments.iter().position(|argument| argument == flag) else {
        return Ok(None);
    };
    arguments
        .get(position + 1)
        .cloned()
        .map(Some)
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn required_flag(arguments: &[String], flag: &str) -> Result<String, Box<dyn StdError>> {
    flag_value(arguments, flag)?.ok_or_else(|| format!("{flag} is required").into())
}

fn artifact_manager(arguments: &[String]) -> Result<ArtifactManager, Box<dyn StdError>> {
    let mut builder = ArtifactManager::builder();
    if let Some(cache) = flag_value(arguments, "--cache")? {
        builder = builder.cache(cache);
    }
    if arguments.iter().any(|argument| argument == "--offline") {
        builder = builder.offline(true).source(ArtifactSource::None);
    }
    if let Some(mirror) = flag_value(arguments, "--mirror-url")? {
        builder = builder.source(ArtifactSource::Mirror { base_url: mirror });
    }
    Ok(builder.build()?)
}

fn artifact_command(arguments: &[String]) -> Result<(), Box<dyn StdError>> {
    let operation = arguments.first().ok_or_else(|| usage().to_owned())?;
    let target = arguments.get(1).ok_or_else(|| usage().to_owned())?;
    let manager = artifact_manager(arguments)?;
    match operation.as_str() {
        "fetch" => println!("{}", serde_json::to_string_pretty(&manager.fetch(target)?)?),
        "verify" | "inspect" => {
            println!(
                "{}",
                serde_json::to_string_pretty(&manager.inspect(target)?)?
            )
        }
        "mirror" => {
            let output = required_flag(arguments, "--out")?;
            println!("{}", manager.mirror(target, output)?.display());
        }
        "export" => {
            let output = required_flag(arguments, "--out")?;
            println!(
                "{}",
                serde_json::to_string_pretty(&manager.export(target, output)?)?
            );
        }
        "import" => println!(
            "{}",
            serde_json::to_string_pretty(&manager.import(target)?)?
        ),
        _ => return Err(usage().into()),
    }
    Ok(())
}

#[cfg(feature = "jit")]
fn gpu_command(arguments: &[String]) -> Result<(), Box<dyn StdError>> {
    use toktier::JitCompiler;

    if arguments.first().map(String::as_str) != Some("compile") {
        return Err(usage().into());
    }
    let family = arguments.get(1).ok_or_else(|| usage().to_owned())?;
    let mut builder = JitCompiler::builder();
    if let Some(cache) = flag_value(arguments, "--jit-cache")? {
        builder = builder.cache(cache);
    }
    if let Some(nvcc) = flag_value(arguments, "--nvcc")? {
        builder = builder.nvcc(nvcc);
    }
    let accept = arguments
        .iter()
        .any(|argument| argument == "--accept-uncertified-jit");
    builder = builder.accept_uncertified_jit(accept);
    let ordinal = flag_value(arguments, "--device")?
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(0);
    println!(
        "{}",
        serde_json::to_string_pretty(&builder.build()?.compile(family, ordinal)?)?
    );
    Ok(())
}

#[cfg(not(feature = "jit"))]
fn gpu_command(_arguments: &[String]) -> Result<(), Box<dyn StdError>> {
    Err("the `jit` Cargo feature is required for `gpu compile`".into())
}

/// The typed doctor facts as JSON, for a control plane that would
/// otherwise parse the Rust debug rendering.
///
/// Written by hand rather than derived: the `serde` feature is not in
/// the default set, and a diagnostic command is not a reason to move a
/// published crate's default features. The field names are the ones on
/// the typed structs, so the two readings stay one answer.
fn doctor_json(facts: &DoctorFacts) -> Value {
    let build = &facts.runtime_build;
    json!({
        "crate_version": facts.crate_version,
        "oracle": facts.oracle,
        "registry_verified": facts.registry_verified,
        "python_required": facts.python_required,
        "sqlite_compiled": facts.sqlite_compiled,
        "prebuilt_gpu_compiled": facts.prebuilt_gpu_compiled,
        "jit_compiled": facts.jit_compiled,
        "network_compiled": facts.network_compiled,
        "runtime_build": {
            "source_digest": build.source_digest,
            "fast_cpu_source_digest": build.fast_cpu_source_digest,
            "native_host_source_digest": build.native_host_source_digest,
            "toolchain": build.toolchain,
            "build_flags": build.build_flags,
            "dependency_closure": build.dependency_closure,
            "build_flag_divergence": build.build_flag_divergence,
            "certified": build.certified,
            "core_closure": build.core_closure,
            "dependency_advisory": build.dependency_advisory,
            "behavior_versions": build.behavior_versions.iter().map(|unit| json!({
                "unit": unit.unit,
                "observed": unit.observed,
                "judged": unit.judged,
                "source": unit.source,
            })).collect::<Vec<_>>(),
        },
        "cuda": facts.cuda.as_ref().map(|cuda| json!({
            "device_ordinal": cuda.device_ordinal,
            "available": cuda.available,
            "architecture": cuda.architecture,
            "driver_api_version": cuda.driver_api_version,
            "error": cuda.error,
        })),
    })
}

/// Whether `doctor` was asked for JSON, or which option it does not know.
///
/// An option this command does not implement is refused rather than
/// ignored: accepting an unknown flag silently and printing something
/// else is the one answer a caller cannot tell apart from success.
fn doctor_wants_json(arguments: &[String]) -> Result<bool, String> {
    let mut json_output = false;
    for argument in arguments {
        match argument.as_str() {
            "--json" => json_output = true,
            other => return Err(format!("doctor: unknown option {other}")),
        }
    }
    Ok(json_output)
}

fn doctor_command(arguments: &[String]) -> Result<(), Box<dyn StdError>> {
    let json_output = doctor_wants_json(arguments)?;
    let facts = Runtime::builder().build()?.doctor();
    if json_output {
        println!("{}", serde_json::to_string_pretty(&doctor_json(&facts))?);
    } else {
        println!("{facts:#?}");
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn StdError>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
        Some("doctor") => doctor_command(&arguments[1..]),
        Some("artifacts") => artifact_command(&arguments[1..]),
        Some("gpu") => gpu_command(&arguments[1..]),
        Some("verify-local") => {
            println!("{}", toktier::verify_local_command(&arguments[1..])?);
            Ok(())
        }
        _ => Err(usage().into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("toktier-rust: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::{doctor_wants_json, usage};

    /// The command implements `--json` and refuses anything else, and
    /// the usage line says so. It used to accept every unknown flag and
    /// print the Rust debug rendering regardless.
    #[test]
    fn doctor_implements_json_and_refuses_what_it_does_not_know() {
        assert!(!doctor_wants_json(&[]).expect("no options"));
        assert!(doctor_wants_json(&["--json".to_owned()]).expect("json asked for"));

        let refusal = doctor_wants_json(&["--jsn".to_owned()]).expect_err("unknown");
        assert!(refusal.contains("unknown option --jsn"), "{refusal}");
        assert!(
            usage().contains("toktier-rust doctor [--json]"),
            "{}",
            usage()
        );
    }
}
