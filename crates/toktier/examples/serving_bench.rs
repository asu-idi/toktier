#[cfg(feature = "serving")]
fn main() -> toktier::Result<()> {
    use std::time::{Duration, Instant};

    use serde::Serialize;
    use toktier::{Device, Runtime, ServingLimits, ServingPool};

    #[derive(Serialize)]
    struct Summary {
        schema: &'static str,
        documents: usize,
        utf8_bytes: usize,
        worker_threads: usize,
        sync_total_us: u64,
        queued_total_us: u64,
        sync_docs_per_second: f64,
        queued_docs_per_second: f64,
        response_total_p50_us: u64,
        response_total_p95_us: u64,
        response_total_p99_us: u64,
        queue_p50_us: u64,
        queue_p95_us: u64,
        queue_p99_us: u64,
        engine_p50_us: u64,
        engine_p95_us: u64,
        engine_p99_us: u64,
        materialization_p50_us: u64,
        maximum_observed_batch_rows: usize,
        exact_matches: usize,
    }

    fn micros(value: Duration) -> u64 {
        value.as_micros().min(u128::from(u64::MAX)) as u64
    }

    fn percentile(values: &mut [u64], percentile: usize) -> u64 {
        values.sort_unstable();
        let index = (values.len().saturating_sub(1) * percentile) / 100;
        values[index]
    }

    let root = std::env::var_os("TOKTIER_ARTIFACT_CACHE")
        .ok_or_else(|| std::io::Error::other("set TOKTIER_ARTIFACT_CACHE"))?;
    let documents = std::env::var("TOKTIER_SERVING_DOCUMENTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(512);
    let workers = std::env::var("TOKTIER_SERVING_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4);
    let tokenizer = Runtime::builder()
        .artifact_cache(root)
        .device(Device::Cpu)
        .build()?
        .load("qwen3_8b")?;
    let inputs = (0..documents)
        .map(|index| {
            format!("request {index}: agent-serving exact IDs 中🙂. ").repeat(64 + index % 8)
        })
        .collect::<Vec<_>>();
    let utf8_bytes: usize = inputs.iter().map(String::len).sum();

    let sync_started = Instant::now();
    let expected = inputs
        .iter()
        .map(|text| tokenizer.encode(text).map(|value| value.ids().to_vec()))
        .collect::<toktier::Result<Vec<_>>>()?;
    let sync_total = sync_started.elapsed();

    let pool = ServingPool::builder(tokenizer)
        .limits(ServingLimits {
            max_queued_requests: documents.max(1),
            max_queued_bytes: utf8_bytes.max(1),
            max_batch_rows: 64,
            max_batch_bytes: utf8_bytes.clamp(1, 4 * 1024 * 1024),
            max_session_requests: 16,
            batch_window: Duration::from_micros(200),
            worker_threads: workers,
        })
        .build()?;
    let queued_started = Instant::now();
    let pending = inputs
        .into_iter()
        .map(|text| pool.submit(text))
        .collect::<toktier::Result<Vec<_>>>()?;
    let mut responses = Vec::with_capacity(pending.len());
    for request in pending {
        responses.push(request.wait()?);
    }
    let queued_total = queued_started.elapsed();
    pool.shutdown();

    let exact_matches = responses
        .iter()
        .zip(&expected)
        .filter(|(actual, expected)| actual.value.ids() == expected.as_slice())
        .count();
    if exact_matches != documents {
        return Err(std::io::Error::other("queued serving output diverged").into());
    }
    let mut totals = responses
        .iter()
        .map(|row| micros(row.timings.total))
        .collect::<Vec<_>>();
    let mut queues = responses
        .iter()
        .map(|row| micros(row.timings.queue))
        .collect::<Vec<_>>();
    let mut engines = responses
        .iter()
        .map(|row| micros(row.timings.engine))
        .collect::<Vec<_>>();
    let mut materialization = responses
        .iter()
        .map(|row| micros(row.timings.materialization))
        .collect::<Vec<_>>();
    let maximum_observed_batch_rows = responses
        .iter()
        .map(|row| row.timings.batch_rows)
        .max()
        .unwrap_or(0);
    let result = Summary {
        schema: "toktier.rust_serving.performance.v1",
        documents,
        utf8_bytes,
        worker_threads: workers,
        sync_total_us: micros(sync_total),
        queued_total_us: micros(queued_total),
        sync_docs_per_second: documents as f64 / sync_total.as_secs_f64(),
        queued_docs_per_second: documents as f64 / queued_total.as_secs_f64(),
        response_total_p50_us: percentile(&mut totals, 50),
        response_total_p95_us: percentile(&mut totals, 95),
        response_total_p99_us: percentile(&mut totals, 99),
        queue_p50_us: percentile(&mut queues, 50),
        queue_p95_us: percentile(&mut queues, 95),
        queue_p99_us: percentile(&mut queues, 99),
        engine_p50_us: percentile(&mut engines, 50),
        engine_p95_us: percentile(&mut engines, 95),
        engine_p99_us: percentile(&mut engines, 99),
        materialization_p50_us: percentile(&mut materialization, 50),
        maximum_observed_batch_rows,
        exact_matches,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&result)
            .map_err(|error| std::io::Error::other(error.to_string()))?
    );
    Ok(())
}

#[cfg(not(feature = "serving"))]
fn main() {
    eprintln!("build with --features serving");
}
