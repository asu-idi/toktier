//! Minimal Python-free lifecycle CLI over the public Rust API.

use std::error::Error as StdError;

use toktier::{ArtifactManager, ArtifactSource, Runtime};

fn usage() -> &'static str {
    "toktier-rust doctor\n\
     toktier-rust artifacts fetch|verify|inspect FAMILY [--cache PATH] [--offline]\n\
     toktier-rust artifacts mirror FAMILY --out DIRECTORY [--cache PATH]\n\
     toktier-rust artifacts export FAMILY --out BUNDLE [--cache PATH]\n\
     toktier-rust artifacts import BUNDLE [--cache PATH] [--offline]\n\
     toktier-rust gpu compile FAMILY [--device N] [--jit-cache PATH] [--nvcc PATH] [--accept-uncertified-jit]"
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

fn run() -> Result<(), Box<dyn StdError>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
        Some("doctor") => {
            let runtime = Runtime::builder().build()?;
            println!("{:#?}", runtime.doctor());
            Ok(())
        }
        Some("artifacts") => artifact_command(&arguments[1..]),
        Some("gpu") => gpu_command(&arguments[1..]),
        _ => Err(usage().into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("toktier-rust: {error}");
        std::process::exit(2);
    }
}
