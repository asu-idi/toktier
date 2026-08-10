use std::hint::black_box;
use std::time::{Duration, Instant};

use toktier::{Device, Policy, Runtime};

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() -> toktier::Result<()> {
    let iterations = std::env::var("TOKTIER_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(7)
        .max(1);
    let family = std::env::var("TOKTIER_FAMILY").unwrap_or_else(|_| "qwen3_8b".to_owned());
    let text = "The quick brown fox 中🙂. ".repeat(160_000);
    let runtime = Runtime::builder().device(Device::Cpu).build()?;
    let tokenizer = runtime.load(&family)?;
    let reference = Runtime::builder()
        .device(Device::Cpu)
        .policy(Policy::Reference)
        .build()?
        .load(&family)?;

    let mut native = Vec::with_capacity(iterations);
    let mut hf = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(tokenizer.encode(black_box(&text))?);
        native.push(start.elapsed());
        let start = Instant::now();
        black_box(reference.encode(black_box(&text))?);
        hf.push(start.elapsed());
    }

    let mut session = tokenizer.open_session("bench")?;
    session.seed(&text)?;
    let mut patch = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(session.append(black_box(" new agent turn"))?);
        patch.push(start.elapsed());
    }
    println!(
        "family={family} bytes={} iterations={iterations} certified_p50_us={} reference_p50_us={} patch_p50_us={} snapshot_requested=false",
        text.len(),
        median(native).as_micros(),
        median(hf).as_micros(),
        median(patch).as_micros(),
    );
    Ok(())
}
