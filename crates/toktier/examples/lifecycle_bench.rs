use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use toktier::{ArtifactManager, ArtifactSource, Device, Policy, Runtime};

#[derive(Serialize)]
struct Summary {
    schema: &'static str,
    family: &'static str,
    artifact_bytes: u64,
    bundle_bytes: u64,
    cold_local_acquire_us: u64,
    verified_cache_hit_us: u64,
    mirror_us: u64,
    export_us: u64,
    import_us: u64,
    offline_load_us: u64,
    offline_encode_us: u64,
    session_seed_chars: usize,
    session_seed_us: u64,
    append_chars: usize,
    session_append_us: u64,
    snapshot_us: u64,
    exact_patch: bool,
    backend: String,
}

fn micros(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
}

fn main() -> toktier::Result<()> {
    const FAMILY: &str = "qwen3_8b";
    let source = std::env::var_os("TOKTIER_LIFECYCLE_SOURCE")
        .map(PathBuf::from)
        .ok_or_else(|| std::io::Error::other("set TOKTIER_LIFECYCLE_SOURCE"))?;
    let session_chars = std::env::var("TOKTIER_LIFECYCLE_SESSION_CHARS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4 * 1024 * 1024);
    let temporary = tempfile::tempdir()?;
    let cache = temporary.path().join("cache");
    let manager = ArtifactManager::builder()
        .cache(&cache)
        .source(ArtifactSource::LocalDirectory { root: source })
        .build()?;

    let started = Instant::now();
    let artifact = manager.fetch(FAMILY)?;
    let cold_local_acquire_us = micros(started);
    let started = Instant::now();
    manager.fetch(FAMILY)?;
    let verified_cache_hit_us = micros(started);

    let started = Instant::now();
    manager.mirror(FAMILY, temporary.path().join("mirror"))?;
    let mirror_us = micros(started);
    let bundle = temporary.path().join("qwen3_8b.toktier.tar.gz");
    let started = Instant::now();
    manager.export(FAMILY, &bundle)?;
    let export_us = micros(started);

    let imported_cache = temporary.path().join("imported");
    let offline = ArtifactManager::builder()
        .cache(&imported_cache)
        .source(ArtifactSource::None)
        .offline(true)
        .build()?;
    let started = Instant::now();
    offline.import(&bundle)?;
    let import_us = micros(started);

    let started = Instant::now();
    let tokenizer = Runtime::builder()
        .artifacts(offline)
        .device(Device::Cpu)
        .policy(Policy::Certified)
        .build()?
        .load(FAMILY)?;
    let offline_load_us = micros(started);
    let probe = "offline Rust lifecycle exactness probe 中🙂".repeat(128);
    let started = Instant::now();
    let probe_ids = tokenizer.encode(&probe)?;
    let offline_encode_us = micros(started);

    let unit = "agent-history-0123456789\n";
    let mut transcript = unit.repeat(session_chars.div_ceil(unit.len()));
    transcript.truncate(session_chars);
    let suffix = " appended-turn-0123456789".repeat(11);
    let mut session = tokenizer.open_session("lifecycle-bench")?;
    let started = Instant::now();
    let seed = session.seed(&transcript)?;
    let session_seed_us = micros(started);
    let mut patched = seed.ids().to_vec();
    let started = Instant::now();
    let patch = session.append(&suffix)?;
    let session_append_us = micros(started);
    patched.truncate(patch.keep_tokens() as usize);
    patched.extend_from_slice(patch.replacement_ids());
    transcript.push_str(&suffix);
    let exact_patch = patched == tokenizer.encode(&transcript)?.ids();
    if !exact_patch {
        return Err(std::io::Error::other("session patch diverged").into());
    }
    let started = Instant::now();
    let snapshot = session.snapshot()?;
    let snapshot_us = micros(started);
    if snapshot.ids() != patched {
        return Err(std::io::Error::other("session snapshot diverged").into());
    }

    let summary = Summary {
        schema: "toktier.rust_lifecycle.performance.v1",
        family: FAMILY,
        artifact_bytes: artifact.tokenizer_size,
        bundle_bytes: std::fs::metadata(bundle)?.len(),
        cold_local_acquire_us,
        verified_cache_hit_us,
        mirror_us,
        export_us,
        import_us,
        offline_load_us,
        offline_encode_us,
        session_seed_chars: session_chars,
        session_seed_us,
        append_chars: suffix.len(),
        session_append_us,
        snapshot_us,
        exact_patch,
        backend: format!("{:?}", probe_ids.execution().backend),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&summary)
            .map_err(|error| std::io::Error::other(error.to_string()))?
    );
    Ok(())
}
