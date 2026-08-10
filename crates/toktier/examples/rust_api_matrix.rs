use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::Serialize;
use serde_json::Value;
use toktier::{Backend, Device, GpuDelivery, Policy, Runtime};

#[derive(Debug, Serialize)]
struct FamilyResult {
    family: String,
    artifact_sha256: String,
    backend: String,
    documents: u64,
    token_checks: u64,
    mismatches: u64,
}

#[derive(Debug, Serialize)]
struct CampaignResult {
    schema: &'static str,
    requested_device: String,
    runtime_source_digest: String,
    fast_cpu_source_digest: String,
    native_host_source_digest: String,
    toolchain: String,
    build_flags: Vec<String>,
    runtime_build_certified: bool,
    families: usize,
    documents: u64,
    token_checks: u64,
    mismatches: u64,
    rows: Vec<FamilyResult>,
}

fn main() -> toktier::Result<()> {
    let manifest_path = std::env::var_os("TOKTIER_CAMPAIGN_MANIFEST")
        .map(PathBuf::from)
        .ok_or_else(|| std::io::Error::other("set TOKTIER_CAMPAIGN_MANIFEST"))?;
    let manifest: BTreeMap<String, Value> = serde_json::from_slice(&fs::read(manifest_path)?)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let requested = std::env::var("TOKTIER_CAMPAIGN_DEVICE").unwrap_or_else(|_| "cpu".into());
    let (device, delivery, gpu_requested) = if requested == "cpu" {
        (Device::Cpu, GpuDelivery::Disabled, false)
    } else if let Some(raw) = requested.strip_prefix("cuda:") {
        let ordinal = raw
            .parse::<u32>()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
        (Device::Cuda(ordinal), GpuDelivery::Prebuilt, true)
    } else {
        return Err(std::io::Error::other("device must be cpu or cuda:<ordinal>").into());
    };
    let policy = match std::env::var("TOKTIER_CAMPAIGN_POLICY")
        .unwrap_or_else(|_| "experimental".into())
        .as_str()
    {
        "certified" => Policy::Certified,
        "experimental" => Policy::Experimental,
        _ => {
            return Err(std::io::Error::other(
                "TOKTIER_CAMPAIGN_POLICY must be certified or experimental",
            )
            .into());
        }
    };

    let mut accelerated_builder = Runtime::builder()
        .device(device)
        .gpu_delivery(delivery)
        // A candidate which is not yet entered into the registry must opt in
        // explicitly and remains labelled experimental. The final replay uses
        // certified policy and proves that the shipped row admits this build.
        .policy(policy)
        .gpu_min_bytes(1);
    let mut reference_builder = Runtime::builder()
        .device(Device::Cpu)
        .gpu_delivery(GpuDelivery::Disabled)
        .policy(Policy::Reference);
    for (family, row) in &manifest {
        let local_dir = row
            .get("local_dir")
            .and_then(Value::as_str)
            .ok_or_else(|| std::io::Error::other(format!("{family}: no local_dir")))?;
        accelerated_builder = accelerated_builder.artifact_directory(family, local_dir);
        reference_builder = reference_builder.artifact_directory(family, local_dir);
    }
    let accelerated = accelerated_builder.build()?;
    let reference = reference_builder.build()?;
    let doctor = accelerated.doctor();

    let base_cases = [
        String::new(),
        "hello world".to_owned(),
        " leading and trailing ".to_owned(),
        "中文🙂 café e\u{301} \r\n\t".to_owned(),
        "agent request: exact token IDs matter. ".repeat(256),
        "0123456789 JSON {\"key\":\"value\"}\n".repeat(512),
    ];
    let gpu_probe = "TokTier GPU exactness probe with plain text and numbers 12345. ".repeat(2_048);
    let mut rows = Vec::new();
    let mut documents = 0u64;
    let mut token_checks = 0u64;
    let mut mismatches = 0u64;
    for family in manifest.keys() {
        let fast = accelerated.load(family)?;
        let hf = reference.load(family)?;
        let expected_backend = if gpu_requested {
            Backend::Gpu
        } else if fast.plan().backends.contains(&Backend::FastCpu) {
            Backend::FastCpu
        } else {
            Backend::HuggingFace
        };
        let mut family_checks = 0u64;
        let mut family_mismatches = 0u64;
        let mut observed = None;
        for text in base_cases.iter().chain(std::iter::once(&gpu_probe)) {
            let actual = fast.encode(text)?;
            let expected = hf.encode(text)?;
            if actual.ids() != expected.ids() {
                family_mismatches += 1;
            }
            family_checks += expected.ids().len() as u64;
            if text.len() == gpu_probe.len() {
                observed = Some(actual.execution().backend);
            }
        }
        let observed = observed.expect("GPU probe case is present");
        if observed != expected_backend {
            return Err(std::io::Error::other(format!(
                "{family}: requested {expected_backend:?}, observed {observed:?}"
            ))
            .into());
        }
        // Record the identity authenticated by the runtime rather than
        // trusting optional digest fields in the path inventory.
        let artifact_sha256 = hf.artifact().identity().tokenizer_sha256.clone();
        documents += 7;
        token_checks += family_checks;
        mismatches += family_mismatches;
        rows.push(FamilyResult {
            family: family.clone(),
            artifact_sha256,
            backend: format!("{observed:?}"),
            documents: 7,
            token_checks: family_checks,
            mismatches: family_mismatches,
        });
    }
    if mismatches != 0 {
        return Err(std::io::Error::other(format!(
            "Rust API matrix found {mismatches} exact-ID divergences"
        ))
        .into());
    }
    let result = CampaignResult {
        schema: "toktier.rust_api.matrix.v1",
        requested_device: requested,
        runtime_source_digest: doctor.runtime_build.source_digest,
        fast_cpu_source_digest: doctor.runtime_build.fast_cpu_source_digest,
        native_host_source_digest: doctor.runtime_build.native_host_source_digest,
        toolchain: doctor.runtime_build.toolchain,
        build_flags: doctor.runtime_build.build_flags,
        runtime_build_certified: doctor.runtime_build.certified,
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
