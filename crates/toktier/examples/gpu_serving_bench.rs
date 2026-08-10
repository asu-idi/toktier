#[cfg(all(feature = "serving", feature = "prebuilt-gpu"))]
fn main() -> toktier::Result<()> {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use serde::Serialize;
    use toktier::{Device, GpuDelivery, Policy, Runtime, ServingLimits, ServingPool};

    #[derive(Serialize)]
    struct Summary {
        schema: &'static str,
        documents: usize,
        bytes_per_document: usize,
        utf8_bytes: usize,
        devices: Vec<u32>,
        worker_threads: usize,
        queued_total_us: u64,
        docs_per_second: f64,
        gib_per_second: f64,
        response_total_p50_us: u64,
        response_total_p95_us: u64,
        response_total_p99_us: u64,
        queue_p50_us: u64,
        queue_p95_us: u64,
        queue_p99_us: u64,
        engine_p50_us: u64,
        engine_p95_us: u64,
        engine_p99_us: u64,
        maximum_observed_batch_rows: usize,
        device_rows: BTreeMap<usize, usize>,
        exact_matches: usize,
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
    let devices = std::env::var("TOKTIER_SERVING_CUDA_DEVICES")
        .unwrap_or_else(|_| "0".to_owned())
        .split(',')
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    if devices.is_empty() {
        return Err(std::io::Error::other("at least one CUDA device is required").into());
    }
    let documents = std::env::var("TOKTIER_SERVING_DOCUMENTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256);
    let bytes_per_document = std::env::var("TOKTIER_SERVING_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64 * 1024);
    if documents == 0 || bytes_per_document == 0 {
        return Err(std::io::Error::other("document and byte counts must be non-zero").into());
    }

    let inputs = (0..documents)
        .map(|index| {
            let unit = format!("gpu-serving-row-{index:04}-exact-ids-0123456789 ");
            let mut text = unit.repeat(bytes_per_document.div_ceil(unit.len()));
            text.truncate(bytes_per_document);
            text
        })
        .collect::<Vec<_>>();
    let reference = Runtime::builder()
        .artifact_cache(&artifact_cache)
        .device(Device::Cpu)
        .gpu_delivery(GpuDelivery::Disabled)
        .policy(Policy::Reference)
        .build()?
        .load("qwen3_8b")?;
    let expected = inputs
        .iter()
        .map(|text| {
            reference
                .encode(text)
                .map(|encoding| encoding.ids().to_vec())
        })
        .collect::<toktier::Result<Vec<_>>>()?;

    let mut tokenizers = devices
        .iter()
        .map(|device| {
            Runtime::builder()
                .artifact_cache(&artifact_cache)
                .device(Device::Cuda(*device))
                .gpu_delivery(GpuDelivery::Prebuilt)
                .gpu_min_bytes(1)
                .policy(Policy::Certified)
                .build()?
                .load("qwen3_8b")
        })
        .collect::<toktier::Result<Vec<_>>>()?;
    let first = tokenizers.remove(0);
    let mut builder = ServingPool::builder(first);
    for tokenizer in tokenizers {
        builder = builder.device(tokenizer);
    }
    let utf8_bytes = documents.saturating_mul(bytes_per_document);
    let worker_threads = devices.len();
    let pool = builder
        .limits(ServingLimits {
            max_queued_requests: documents,
            max_queued_bytes: utf8_bytes,
            max_batch_rows: 64,
            max_batch_bytes: 4 * 1024 * 1024,
            max_session_requests: 16,
            batch_window: Duration::from_micros(200),
            worker_threads,
        })
        .build()?;

    let started = Instant::now();
    let pending = inputs
        .into_iter()
        .map(|text| pool.submit(text))
        .collect::<toktier::Result<Vec<_>>>()?;
    let mut responses = Vec::with_capacity(documents);
    for request in pending {
        responses.push(request.wait()?);
    }
    let elapsed = started.elapsed();
    pool.shutdown();
    let exact_matches = responses
        .iter()
        .zip(&expected)
        .filter(|(actual, expected)| actual.value.ids() == expected.as_slice())
        .count();
    if exact_matches != documents {
        return Err(std::io::Error::other("GPU serving output diverged").into());
    }

    let mut total = responses
        .iter()
        .map(|row| micros(row.timings.total))
        .collect::<Vec<_>>();
    let mut queue = responses
        .iter()
        .map(|row| micros(row.timings.queue))
        .collect::<Vec<_>>();
    let mut engine = responses
        .iter()
        .map(|row| micros(row.timings.engine))
        .collect::<Vec<_>>();
    let mut device_rows = BTreeMap::new();
    for row in &responses {
        *device_rows.entry(row.timings.device_index).or_insert(0) += 1;
    }
    let seconds = elapsed.as_secs_f64();
    let summary = Summary {
        schema: "toktier.rust_gpu_serving.performance.v1",
        documents,
        bytes_per_document,
        utf8_bytes,
        devices,
        worker_threads,
        queued_total_us: micros(elapsed),
        docs_per_second: documents as f64 / seconds,
        gib_per_second: utf8_bytes as f64 / (1024.0 * 1024.0 * 1024.0) / seconds,
        response_total_p50_us: percentile(&mut total, 50),
        response_total_p95_us: percentile(&mut total, 95),
        response_total_p99_us: percentile(&mut total, 99),
        queue_p50_us: percentile(&mut queue, 50),
        queue_p95_us: percentile(&mut queue, 95),
        queue_p99_us: percentile(&mut queue, 99),
        engine_p50_us: percentile(&mut engine, 50),
        engine_p95_us: percentile(&mut engine, 95),
        engine_p99_us: percentile(&mut engine, 99),
        maximum_observed_batch_rows: responses
            .iter()
            .map(|row| row.timings.batch_rows)
            .max()
            .unwrap_or(0),
        device_rows,
        exact_matches,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&summary)
            .map_err(|error| std::io::Error::other(error.to_string()))?
    );
    Ok(())
}

#[cfg(not(all(feature = "serving", feature = "prebuilt-gpu")))]
fn main() {
    eprintln!("build with the serving and prebuilt-gpu features");
}
