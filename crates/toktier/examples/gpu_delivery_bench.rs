#[cfg(all(feature = "jit", feature = "prebuilt-gpu"))]
fn main() -> toktier::Result<()> {
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use serde::Serialize;
    use toktier::{Backend, Device, GpuDelivery, Policy, Runtime};

    #[derive(Serialize)]
    struct Summary {
        schema: &'static str,
        delivery: String,
        input_bytes: usize,
        iterations: usize,
        load_us: u64,
        encode_p50_us: u64,
        encode_p95_us: u64,
        encode_p99_us: u64,
        gib_per_second_p50: f64,
        exact_iterations: usize,
        backend: String,
        source: Option<String>,
        runtime_build_certified: bool,
        jit_cache_hit: Option<bool>,
        jit_product_sha256: Option<String>,
        jit_certified: Option<bool>,
    }

    fn micros(value: Duration) -> u64 {
        value.as_micros().min(u128::from(u64::MAX)) as u64
    }

    fn percentile(values: &mut [u64], percentile: usize) -> u64 {
        values.sort_unstable();
        values[(values.len().saturating_sub(1) * percentile) / 100]
    }

    let artifact_cache = std::env::var_os("TOKTIER_ARTIFACT_CACHE")
        .map(PathBuf::from)
        .ok_or_else(|| std::io::Error::other("set TOKTIER_ARTIFACT_CACHE"))?;
    let delivery_name =
        std::env::var("TOKTIER_GPU_DELIVERY").unwrap_or_else(|_| "prebuilt".to_owned());
    let delivery = match delivery_name.as_str() {
        "prebuilt" => GpuDelivery::Prebuilt,
        "jit" => GpuDelivery::Jit,
        _ => return Err(std::io::Error::other("delivery must be prebuilt or jit").into()),
    };
    let input_bytes = std::env::var("TOKTIER_GPU_BENCH_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4 * 1024 * 1024);
    let iterations = std::env::var("TOKTIER_GPU_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    if input_bytes == 0 || iterations == 0 {
        return Err(std::io::Error::other("input and iteration counts must be non-zero").into());
    }
    let unit = "TokTier GPU delivery exact throughput 0123456789 agent-context. ";
    let mut input = unit.repeat(input_bytes.div_ceil(unit.len()));
    input.truncate(input_bytes);

    let reference = Runtime::builder()
        .artifact_cache(&artifact_cache)
        .device(Device::Cpu)
        .gpu_delivery(GpuDelivery::Disabled)
        .policy(Policy::Reference)
        .build()?
        .load("qwen3_8b")?;
    let expected = reference.encode(&input)?;
    let runtime = Runtime::builder()
        .artifact_cache(&artifact_cache)
        .device(Device::Cuda(0))
        .gpu_delivery(delivery)
        .gpu_min_bytes(1)
        .policy(Policy::Certified)
        .build()?;
    let runtime_build_certified = runtime.doctor().runtime_build.certified;
    let started = Instant::now();
    let tokenizer = runtime.load("qwen3_8b")?;
    let load_us = micros(started.elapsed());
    for _ in 0..3 {
        let warmup = tokenizer.encode(&input)?;
        if warmup.ids() != expected.ids() || warmup.execution().backend != Backend::Gpu {
            return Err(std::io::Error::other("GPU delivery warmup diverged").into());
        }
    }
    let mut samples = Vec::with_capacity(iterations);
    let mut exact_iterations = 0;
    let mut last_execution = None;
    for _ in 0..iterations {
        let started = Instant::now();
        let actual = tokenizer.encode(&input)?;
        samples.push(micros(started.elapsed()));
        if actual.ids() == expected.ids() && actual.execution().backend == Backend::Gpu {
            exact_iterations += 1;
        }
        last_execution = Some(actual.execution().clone());
    }
    if exact_iterations != iterations {
        return Err(std::io::Error::other("GPU delivery benchmark diverged").into());
    }
    let execution = last_execution.expect("at least one iteration");
    let jit = tokenizer.jit_facts();
    let p50 = percentile(&mut samples, 50);
    let summary = Summary {
        schema: "toktier.rust_gpu_delivery.performance.v1",
        delivery: delivery_name,
        input_bytes,
        iterations,
        load_us,
        encode_p50_us: p50,
        encode_p95_us: percentile(&mut samples, 95),
        encode_p99_us: percentile(&mut samples, 99),
        gib_per_second_p50: input_bytes as f64
            / (1024.0 * 1024.0 * 1024.0)
            / (p50 as f64 / 1_000_000.0),
        exact_iterations,
        backend: format!("{:?}", execution.backend),
        source: execution.source,
        runtime_build_certified,
        jit_cache_hit: jit.map(|facts| facts.cache_hit),
        jit_product_sha256: jit.map(|facts| facts.product_sha256.clone()),
        jit_certified: jit.map(|facts| facts.certified),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&summary)
            .map_err(|error| std::io::Error::other(error.to_string()))?
    );
    Ok(())
}

#[cfg(not(all(feature = "jit", feature = "prebuilt-gpu")))]
fn main() {
    eprintln!("build with the jit feature");
}
