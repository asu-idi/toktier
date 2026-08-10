#[cfg(feature = "jit")]
fn main() -> toktier::Result<()> {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;

    use serde::Serialize;
    use serde_json::Value;
    use toktier::{Backend, Device, GpuDelivery, Policy, Runtime};

    #[derive(Serialize)]
    struct Row {
        family: String,
        artifact_sha256: String,
        oracle_id: String,
        oracle_version: String,
        artifact_evidence_id: String,
        jit_evidence_id: Option<String>,
        documents: u64,
        token_checks: u64,
        mismatches: u64,
        backend: String,
        delivery: String,
        binding_digest: String,
        product_sha256: String,
        product_domain_digest: String,
        certified: bool,
        first_compile_cache_hit: bool,
        second_compile_cache_hit: bool,
        cold_compile_us: u64,
        authenticated_cache_hit_us: u64,
        first_gpu_request_us: u64,
    }

    #[derive(Serialize)]
    struct ResultDocument {
        schema: &'static str,
        architecture: String,
        driver_api_version: i32,
        runtime_source_digest: String,
        native_host_source_digest: String,
        rust_toolchain: String,
        rust_build_flags: Vec<String>,
        runtime_build_certified: bool,
        compiler_resolved_path: String,
        compiler_release: String,
        compiler_build: String,
        compiler_sha256: String,
        compiler_world_writable_component: Option<String>,
        source_digest: String,
        direct_build_flags: [&'static str; 5],
        experimental_opt_in: bool,
        families: usize,
        documents: u64,
        token_checks: u64,
        mismatches: u64,
        rows: Vec<Row>,
    }

    let manifest_path = std::env::var_os("TOKTIER_CAMPAIGN_MANIFEST")
        .map(PathBuf::from)
        .ok_or_else(|| std::io::Error::other("set TOKTIER_CAMPAIGN_MANIFEST"))?;
    let manifest: BTreeMap<String, Value> = serde_json::from_slice(&fs::read(manifest_path)?)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let device = std::env::var("TOKTIER_CAMPAIGN_DEVICE")
        .unwrap_or_else(|_| "0".to_owned())
        .parse::<u32>()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    let experimental = std::env::var_os("TOKTIER_ACCEPT_UNCERTIFIED_JIT").is_some();
    let cache = std::env::var_os("TOKTIER_JIT_CACHE")
        .map(PathBuf::from)
        .ok_or_else(|| std::io::Error::other("set TOKTIER_JIT_CACHE"))?;

    let compiler = toktier::JitCompiler::builder()
        .cache(&cache)
        .accept_uncertified_jit(experimental)
        .build()?;
    let toolchain = compiler.toolchain()?;
    let mut candidate = Runtime::builder()
        .device(Device::Cuda(device))
        .gpu_delivery(GpuDelivery::Jit)
        .gpu_min_bytes(0)
        .policy(if experimental {
            Policy::Experimental
        } else {
            Policy::Certified
        })
        .jit_compiler(compiler.clone());
    let mut reference = Runtime::builder()
        .device(Device::Cpu)
        .gpu_delivery(GpuDelivery::Disabled)
        .policy(Policy::Reference);
    for (family, value) in &manifest {
        let directory = value
            .get("local_dir")
            .and_then(Value::as_str)
            .ok_or_else(|| std::io::Error::other(format!("{family}: no local_dir")))?;
        candidate = candidate.artifact_directory(family, directory);
        reference = reference.artifact_directory(family, directory);
    }
    let candidate = candidate.build()?;
    let reference = reference.build()?;
    let doctor = candidate.doctor();
    let cuda = doctor
        .cuda
        .as_ref()
        .filter(|facts| facts.available)
        .ok_or_else(|| std::io::Error::other("CUDA device is unavailable"))?;
    let architecture = cuda
        .architecture
        .clone()
        .ok_or_else(|| std::io::Error::other("CUDA architecture is absent"))?;
    let driver_api_version = cuda
        .driver_api_version
        .ok_or_else(|| std::io::Error::other("CUDA driver version is absent"))?;

    let cases = [
        "TokTier direct JIT exactness probe with plain text 12345. ".repeat(2_048),
        String::new(),
        "hello world".to_owned(),
        " leading and trailing ".to_owned(),
        "中文🙂 café e\u{301} \r\n\t".to_owned(),
        "agent request: exact token IDs matter. ".repeat(256),
        "0123456789 JSON {\"key\":\"value\"}\n".repeat(512),
    ];
    let mut rows = Vec::new();
    let mut documents = 0;
    let mut token_checks = 0;
    let mut mismatches = 0;
    for family in manifest.keys() {
        let compile_started = std::time::Instant::now();
        let first = compiler.compile(family, device)?;
        let cold_compile_us = compile_started.elapsed().as_micros() as u64;
        let lookup_started = std::time::Instant::now();
        let second = compiler.compile(family, device)?;
        let authenticated_cache_hit_us = lookup_started.elapsed().as_micros() as u64;
        if first.product_sha256 != second.product_sha256
            || first.binding_digest != second.binding_digest
            || first.cache_hit
            || !second.cache_hit
        {
            return Err(std::io::Error::other(format!(
                "{family}: direct JIT cache did not transition miss -> authenticated hit"
            ))
            .into());
        }
        let accelerated = candidate.load(family)?;
        let hf = reference.load(family)?;
        let facts = accelerated
            .jit_facts()
            .ok_or_else(|| std::io::Error::other(format!("{family}: no JIT facts")))?;
        let mut row_checks = 0;
        let mut row_mismatches = 0;
        let mut first_gpu_request_us = 0;
        for (case_index, text) in cases.iter().enumerate() {
            let request_started = std::time::Instant::now();
            let actual = accelerated.encode(text)?;
            let request_us = request_started.elapsed().as_micros() as u64;
            let expected = hf.encode(text)?;
            let large_probe = case_index == 0;
            if large_probe {
                first_gpu_request_us = request_us;
            }
            if actual.ids() != expected.ids()
                || large_probe
                    && (actual.execution().backend != Backend::Gpu
                        || actual.execution().source.as_deref() != Some("jit"))
            {
                row_mismatches += 1;
            }
            row_checks += expected.ids().len() as u64;
        }
        // Take the identity from the handle which authenticated the artifact,
        // not from the campaign's path-only inventory.  This keeps evidence
        // independent of optional metadata in that inventory.
        let artifact_sha256 = hf.artifact().identity().tokenizer_sha256.clone();
        documents += cases.len() as u64;
        token_checks += row_checks;
        mismatches += row_mismatches;
        rows.push(Row {
            family: family.clone(),
            artifact_sha256,
            oracle_id: facts.oracle_id.clone(),
            oracle_version: facts.oracle_version.clone(),
            artifact_evidence_id: facts.artifact_evidence_id.clone(),
            jit_evidence_id: facts.jit_evidence_id.clone(),
            documents: cases.len() as u64,
            token_checks: row_checks,
            mismatches: row_mismatches,
            backend: "Gpu".to_owned(),
            delivery: "jit".to_owned(),
            binding_digest: facts.binding_digest.clone(),
            product_sha256: facts.product_sha256.clone(),
            product_domain_digest: facts.domain_digest.clone(),
            certified: facts.certified,
            first_compile_cache_hit: first.cache_hit,
            second_compile_cache_hit: second.cache_hit,
            cold_compile_us,
            authenticated_cache_hit_us,
            first_gpu_request_us,
        });
    }
    if rows.len() != 14 || mismatches != 0 {
        return Err(std::io::Error::other(format!(
            "direct JIT matrix has {} families and {mismatches} divergences",
            rows.len()
        ))
        .into());
    }
    // The compiler API intentionally does not expose mutable source bytes.
    // Recompute the public domain digest from the embedded source files using
    // the same release helper included in package data.
    let source_digest = direct_source_digest();
    let result = ResultDocument {
        schema: "toktier.rust_direct_jit.matrix.v1",
        architecture,
        driver_api_version,
        runtime_source_digest: doctor.runtime_build.source_digest,
        native_host_source_digest: doctor.runtime_build.native_host_source_digest,
        rust_toolchain: doctor.runtime_build.toolchain,
        rust_build_flags: doctor.runtime_build.build_flags,
        runtime_build_certified: doctor.runtime_build.certified,
        compiler_resolved_path: toolchain.resolved_path.display().to_string(),
        compiler_release: toolchain.release,
        compiler_build: toolchain.build,
        compiler_sha256: toolchain.compiler_sha256,
        compiler_world_writable_component: toolchain
            .world_writable_component
            .map(|path| path.display().to_string()),
        source_digest,
        direct_build_flags: [
            "-fatbin",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-DTOKTIER_DEVICE_ONLY",
        ],
        experimental_opt_in: experimental,
        families: rows.len(),
        documents,
        token_checks,
        mismatches,
        rows,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&result)
            .map_err(|error| std::io::Error::other(error.to_string()))?
    );
    Ok(())
}

#[cfg(feature = "jit")]
fn direct_source_digest() -> String {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    digest.update(b"toktier.rust_jit_source.v1\0");
    for (name, bytes) in [
        (
            "prebuilt_unit.cu",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/data/src/toktier/kernels/prebuilt_unit.cu"
            ))
            .as_slice(),
        ),
        (
            "pretok_kernel.cu",
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/data/src/toktier/kernels/pretok_kernel.cu"
            ))
            .as_slice(),
        ),
    ] {
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(not(feature = "jit"))]
fn main() {
    eprintln!("build with --features jit");
}
