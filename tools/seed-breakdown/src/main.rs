mod proto;

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use aho_corasick::{AhoCorasick, MatchKind};
use flate2::read::ZlibDecoder;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use toktier::{Device, EncodeOptions, GpuDelivery, Policy, Runtime};
use toktier_routing_core::{
    FastCpuEngine, FastRepairSpec, NativeRouter, RayonSeedOverlap, ReferenceEngine,
};
use toktier_store_core::{
    payload_digest_parts, AppendReport, BoundaryCut, ContentDigest, Encoding as CoreEncoding,
    EngineError, KeyId, PayloadHasher, SessionEncoder, SessionStore, StoreConfig, TailState,
    WitnessCategory,
};
use toktier_store_sqlite::{NamedSessionRef, SingleEngine, StoreDb};

const FAMILY: &str = "qwen3_8b";
const INPUT_BYTES: usize = 4 * 1024 * 1024;
const RECOVERY_TEXT_DOMAIN: &[u8] = b"toktier.facade.v1.recovery-text\0";

type AnyResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug)]
struct Args {
    phase: String,
    artifact: PathBuf,
    device: String,
    home: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct RepairTable {
    families: Vec<RepairRow>,
}

#[derive(Debug, Deserialize)]
struct RepairRow {
    family: String,
    artifact_sha256: String,
    margin: usize,
    effective_l_max: usize,
    has_normalizer: bool,
}

#[derive(Debug, Serialize)]
struct Observation {
    schema: &'static str,
    phase: String,
    elapsed_ns: u64,
    input_bytes: usize,
    input_chars: usize,
    token_count: usize,
    device: String,
    exact: bool,
    actual_backend: Option<String>,
    actual_path: Option<String>,
    product_commit: String,
    host: String,
    pid: u32,
    rust_api_source_sha256: String,
    fast_cpu_source_sha256: String,
    native_host_source_sha256: String,
    runtime_build_certified: bool,
    details: BTreeMap<String, Value>,
}

struct Engines {
    reference: Arc<ReferenceEngine>,
    fast_cpu: Arc<FastCpuEngine>,
    router: Arc<NativeRouter>,
    fingerprint: [u8; 32],
    seal_guard: u64,
}

#[derive(Clone)]
struct PrecomputedEncoder {
    encoding: CoreEncoding,
    boundary: Arc<NativeRouter>,
    certify_boundaries: bool,
}

impl SessionEncoder for PrecomputedEncoder {
    fn encode(&self, _text: &str) -> Result<CoreEncoding, EngineError> {
        Ok(self.encoding.clone())
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        if !tail.text().is_empty() {
            return Err(EngineError(
                "the seed profiler's precomputed encoder only accepts an empty tail".to_owned(),
            ));
        }
        tail.fill(delta, self.encoding.clone())
            .map_err(|error| EngineError(error.to_string()))?;
        Ok(AppendReport {
            path: "precomputed_exact_seed".to_owned(),
            kept_tokens: 0,
        })
    }

    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        if self.certify_boundaries {
            self.boundary
                .last_certified_boundary(tail, floor_char, ceil_char)
        } else {
            Ok(None)
        }
    }

    fn witness_category(&self) -> WitnessCategory {
        self.boundary.witness_category()
    }
}

/// The retained W4a seed shape: corrected-Gigatoken structure-of-arrays
/// encode adopted through `fill_soa`, with the engine's own certified
/// boundary probe. Used only as the profiler's same-tree control.
struct SoaSeedEncoder {
    fast: Arc<FastCpuEngine>,
}

impl SessionEncoder for SoaSeedEncoder {
    fn encode(&self, text: &str) -> Result<CoreEncoding, EngineError> {
        self.fast.encode(text)
    }

    fn append(&self, tail: &mut TailState, delta: &str) -> Result<AppendReport, EngineError> {
        if !tail.text().is_empty() {
            return self.fast.append(tail, delta);
        }
        let (encoding, _source) = self.fast.encode_state_with_source(delta)?;
        tail.fill_soa(delta, encoding)
            .map_err(|error| EngineError(error.to_string()))?;
        Ok(AppendReport {
            path: "cold_full_soa_control".to_owned(),
            kept_tokens: 0,
        })
    }

    fn last_certified_boundary(
        &self,
        tail: &TailState,
        floor_char: u64,
        ceil_char: u64,
    ) -> Result<Option<BoundaryCut>, EngineError> {
        self.fast
            .last_certified_boundary(tail, floor_char, ceil_char)
    }

    fn witness_category(&self) -> WitnessCategory {
        self.fast.witness_category()
    }
}

#[derive(Clone, Copy)]
enum PostMode {
    Full,
    None,
    BlocksOnly,
    BoundaryOnly,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("seed-breakdown: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args = parse_args()?;
    let text = if args.phase.ends_with("_unicode") {
        unicode_payload()
    } else {
        historical_payload()
    };
    let bytes = fs::read(&args.artifact)?;
    let verification = ReferenceEngine::from_bytes(&bytes)?;
    let doctor = Runtime::builder().device(Device::Cpu).build()?.doctor();
    // The PLAN/161 phases measure certified public/runtime routes, so they
    // require an admitted build. The PLAN/163 W3 direct cells measure
    // library functions beneath the routing registry; on an intermediate
    // tree whose certificates are regenerated in a later batch they still
    // run, and every observation records `runtime_build_certified` honestly.
    if !doctor.runtime_build.certified && !is_w3_phase(&args.phase) {
        return Err(
            "the product build is not admitted by the shipped Rust runtime registry".into(),
        );
    }

    let mut details = BTreeMap::new();
    let mut backend = None;
    let mut path = None;
    let (elapsed_ns, ids) = if args.phase.starts_with("public_") {
        run_public(&args, &text, &mut details, &mut backend, &mut path)?
    } else {
        let engines = build_engines(&args.artifact, bytes)?;
        run_internal(&args, &text, &engines, &mut details, &mut path)?
    };
    // Correctness is deliberately checked after the timed operation. Running
    // the 4 MiB HF oracle first would turn a cold production seed into a
    // cache-warmed workload even though the comparison itself is out of band.
    let expected = verification.encode_ids(&text, false)?;
    let exact = ids == expected;
    if !exact {
        return Err(format!(
            "{} returned IDs different from the frozen HF reference ({} != {} tokens)",
            args.phase,
            ids.len(),
            expected.len()
        )
        .into());
    }
    black_box(&ids);

    let observation = Observation {
        schema: "toktier.seed_breakdown.observation.v1",
        phase: args.phase,
        elapsed_ns,
        input_bytes: text.len(),
        input_chars: text.chars().count(),
        token_count: ids.len(),
        device: args.device,
        exact,
        actual_backend: backend,
        actual_path: path,
        product_commit: std::env::var("TOKTIER_PROFILE_PRODUCT_COMMIT")
            .unwrap_or_else(|_| "unknown".to_owned()),
        host: std::env::var("HOSTNAME").unwrap_or_else(|_| {
            fs::read_to_string("/etc/hostname")
                .map(|value| value.trim().to_owned())
                .unwrap_or_else(|_| "unknown".to_owned())
        }),
        pid: std::process::id(),
        rust_api_source_sha256: doctor.runtime_build.source_digest,
        fast_cpu_source_sha256: doctor.runtime_build.fast_cpu_source_digest,
        native_host_source_sha256: doctor.runtime_build.native_host_source_digest,
        runtime_build_certified: doctor.runtime_build.certified,
        details,
    };
    println!("{}", serde_json::to_string(&observation)?);
    Ok(())
}

fn parse_args() -> AnyResult<Args> {
    let mut phase = None;
    let mut artifact = None;
    let mut device = "cpu".to_owned();
    let mut home = None;
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--phase" => phase = values.next(),
            "--artifact" => artifact = values.next().map(PathBuf::from),
            "--device" => device = values.next().ok_or("--device requires a value")?,
            "--home" => home = values.next().map(PathBuf::from),
            "--help" | "-h" => {
                println!(
                    "usage: toktier-seed-breakdown --phase NAME --artifact PATH [--device cpu|gpu] [--home PATH]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let phase = phase.ok_or("--phase is required")?;
    let artifact = artifact.ok_or("--artifact is required")?;
    if !artifact.is_file() {
        return Err(format!("artifact does not exist: {}", artifact.display()).into());
    }
    if device != "cpu" && device != "gpu" {
        return Err("--device must be cpu or gpu".into());
    }
    Ok(Args {
        phase,
        artifact,
        device,
        home,
    })
}

fn historical_payload() -> String {
    let unit = "agent-history-0123456789\n";
    let mut text = unit.repeat(INPUT_BYTES.div_ceil(unit.len()));
    text.truncate(INPUT_BYTES);
    assert_eq!(text.len(), INPUT_BYTES);
    assert_eq!(text.chars().count(), INPUT_BYTES);
    text
}

/// One-line description of the Unicode payload recorded with every
/// Unicode-cell observation, so the row is self-describing.
const UNICODE_GENERATOR_NOTE: &str = "repeat mixed-script unit (ASCII, two-byte \
Latin/Cyrillic, three-byte CJK/Hangul, normalization-stable combining marks, \
four-byte emoji with ZWJ joins), truncate at a character boundary, pad with \
ASCII dots to exactly 4194304 bytes";

/// PLAN/163 W3 Unicode payload. The emoji/ZWJ and combining clusters give
/// byte-level BPE tokens that split inside one character, which exercises
/// the byte-fallback-group span semantics; the CJK/Hangul/Cyrillic segments
/// exercise multi-byte UTF-8 without splits. Every combining mark is chosen
/// with no precomposed form, so the artifact's normalizer maps the payload
/// to itself; the known-ID span bridge presumes that stability, and the
/// Unicode cells assert it before timing. Only ASCII escapes appear in this
/// source file; the generated bytes are bound by the SHA-256 recorded in
/// each observation.
fn unicode_payload() -> String {
    let unit = concat!(
        "agent-history-0123456789 ",
        "caf\u{e9} r\u{e9}sum\u{e9} na\u{ef}ve ",
        "\u{4e16}\u{754c}\u{6a21}\u{578b}\u{5206}\u{8bcd} ",
        "\u{d55c}\u{ad6d}\u{c5b4} ",
        "\u{421}\u{43b}\u{43e}\u{432}\u{43e} ",
        "q\u{301}x\u{20d7}a\u{305}n\u{30a} ",
        "\u{1f30d}\u{1f680}\u{1f9ec} ",
        "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}\n"
    );
    let mut text = unit.repeat(INPUT_BYTES.div_ceil(unit.len()));
    let mut cut = INPUT_BYTES;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    text.truncate(cut);
    while text.len() < INPUT_BYTES {
        text.push('.');
    }
    assert_eq!(text.len(), INPUT_BYTES);
    assert!(!text.is_ascii());
    text
}

fn is_w3_phase(phase: &str) -> bool {
    phase.starts_with("spans_")
        || phase.starts_with("payload_digest_")
        || phase.starts_with("store_seed_")
        || phase == "added_gate_scan"
}

fn nanos(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

/// Resident set size in KiB from /proc/self/status, when available.
fn resident_kb() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Record the overlap on/off axis (CHANGE-162 C5): worker count and pool
/// shape go into the observation details so overlap readings never mix
/// with serial readings.
fn record_overlap_details(
    store: &mut SessionStore,
    overlap: bool,
    details: &mut BTreeMap<String, Value>,
) {
    details.insert("overlap".to_owned(), json!(overlap));
    if overlap {
        store.set_seed_overlap(Some(Arc::new(RayonSeedOverlap)));
        details.insert(
            "overlap_workers".to_owned(),
            json!(toktier_store_core::OverlapRunner::worker_count(
                &RayonSeedOverlap
            )),
        );
        details.insert(
            "overlap_pool".to_owned(),
            json!("rayon_global_in_place_scope"),
        );
    }
}

fn run_public(
    args: &Args,
    text: &str,
    details: &mut BTreeMap<String, Value>,
    backend: &mut Option<String>,
    path: &mut Option<String>,
) -> AnyResult<(u64, Vec<u32>)> {
    let directory = args
        .artifact
        .parent()
        .ok_or("artifact path has no parent directory")?;
    let runtime = build_public_runtime(args, directory)?;
    let tokenizer = runtime.load(FAMILY)?;
    details.insert(
        "plan_backends".to_owned(),
        json!(tokenizer
            .plan()
            .backends
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>()),
    );
    details.insert(
        "plan_certification".to_owned(),
        json!(format!("{:?}", tokenizer.plan().certification)),
    );
    details.insert(
        "artifact_sha256".to_owned(),
        json!(tokenizer.artifact().identity().tokenizer_sha256),
    );

    match args.phase.as_str() {
        "public_encode" => {
            let started = Instant::now();
            let encoding = tokenizer.encode(text)?;
            let elapsed = nanos(started);
            *backend = Some(format!("{:?}", encoding.execution().backend));
            *path = Some(encoding.execution().path.clone());
            Ok((elapsed, encoding.ids().to_vec()))
        }
        "public_encode_offsets" => {
            let started = Instant::now();
            let encoding = tokenizer.encode_with(
                text,
                EncodeOptions {
                    add_special_tokens: false,
                    offsets: true,
                },
            )?;
            let elapsed = nanos(started);
            *backend = Some(format!("{:?}", encoding.execution().backend));
            *path = Some(encoding.execution().path.clone());
            details.insert(
                "offset_count".to_owned(),
                json!(encoding.offsets().map_or(0, <[(u32, u32)]>::len)),
            );
            Ok((elapsed, encoding.ids().to_vec()))
        }
        "public_seed_memory" | "public_seed_sqlite" => {
            let name = format!("seed-{}", std::process::id());
            let mut session = tokenizer.open_session(name.clone())?;
            let started = Instant::now();
            let encoding = session.seed(text)?;
            let elapsed = nanos(started);
            let ids = encoding.ids().to_vec();
            let stats = tokenizer.store_stats();
            let selected = stats
                .path_counts
                .iter()
                .find(|(_, count)| **count > 0)
                .map(|(name, _)| name.clone());
            *path = selected.clone();
            *backend = selected.as_deref().map(|name| {
                if name.contains("gpu") {
                    "Gpu".to_owned()
                } else if name.contains("gigatoken") {
                    "FastCpu".to_owned()
                } else {
                    "HuggingFace".to_owned()
                }
            });
            details.insert("store_path_counts".to_owned(), json!(stats.path_counts));
            details.insert("store_node_count".to_owned(), json!(stats.node_count));
            details.insert("store_session_count".to_owned(), json!(stats.session_count));
            if args.phase.ends_with("_sqlite") {
                let snapshot = session.snapshot()?;
                if snapshot.ids() != ids {
                    return Err("SQLite seed snapshot changed before reopen".into());
                }
                session.close()?;
                drop(tokenizer);
                drop(runtime);
                let reopened_runtime = build_public_runtime(args, directory)?;
                let reopened_tokenizer = reopened_runtime.load(FAMILY)?;
                let reopened_session = reopened_tokenizer.open_session(name)?;
                let reopened = reopened_session.snapshot()?;
                if reopened.ids() != ids {
                    return Err("SQLite seed changed after a fresh Runtime reopen".into());
                }
                details.insert("fresh_runtime_reopen_exact".to_owned(), json!(true));
            }
            Ok((elapsed, ids))
        }
        other => Err(format!("unknown public phase {other:?}").into()),
    }
}

fn build_public_runtime(args: &Args, directory: &Path) -> AnyResult<Runtime> {
    let mut builder = Runtime::builder()
        .artifact_directory(FAMILY, directory)
        .policy(Policy::Certified)
        .diagnostics(true);
    builder = if args.device == "gpu" {
        builder
            .device(Device::Cuda(0))
            .gpu_delivery(GpuDelivery::Prebuilt)
            .gpu_min_bytes(1)
    } else {
        builder
            .device(Device::Cpu)
            .gpu_delivery(GpuDelivery::Disabled)
    };
    if args.phase.ends_with("_sqlite") {
        let home = args.home.as_ref().ok_or("SQLite phases require --home")?;
        builder = builder.home(home);
    }
    Ok(builder.build()?)
}

fn build_engines(artifact: &Path, bytes: Vec<u8>) -> AnyResult<Engines> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("cannot resolve product root")?;
    let repairs: RepairTable = serde_json::from_slice(&fs::read(
        root.join("src/toktier/repair/tables/fast_repair_families.v1.json"),
    )?)?;
    let row = repairs
        .families
        .into_iter()
        .find(|row| row.family == FAMILY)
        .ok_or("qwen3_8b has no corrected-CPU repair row")?;
    let observed = hex(&Sha256::digest(&bytes));
    if observed != row.artifact_sha256 {
        return Err(format!(
            "artifact {} has digest {observed}, expected {}",
            artifact.display(),
            row.artifact_sha256
        )
        .into());
    }
    let mut decoder = ZlibDecoder::new(fs::File::open(
        root.join("src/toktier/repair/tables/repair_pclass.v1.zlib"),
    )?);
    let mut pclass = Vec::new();
    decoder.read_to_end(&mut pclass)?;
    let reference = Arc::new(ReferenceEngine::from_bytes(&bytes)?);
    let spec = FastRepairSpec::new(
        FAMILY.to_owned(),
        row.artifact_sha256,
        row.margin,
        row.effective_l_max,
        row.has_normalizer,
    );
    let fast_cpu = Arc::new(FastCpuEngine::from_reference(
        &bytes,
        Arc::clone(&reference),
        spec,
        pclass,
    )?);
    let postprocessor_adds_tokens = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|document| document.get("post_processor").cloned())
        .is_some_and(|value| !value.is_null());
    let router = Arc::new(NativeRouter::new(
        vec!["fast_cpu".to_owned(), "hf".to_owned()],
        vec![0, 0],
        Arc::clone(&reference),
        Some(Arc::clone(&fast_cpu)),
        true,
        None,
        postprocessor_adds_tokens,
        true,
    )?);
    let literal_guard = reference
        .added_tokens()
        .iter()
        .map(|(_, content, _)| content.chars().count())
        .max()
        .unwrap_or(0);
    let seal_guard = u64::try_from(literal_guard.max(fast_cpu.minimum_seal_tail_chars()))?;
    Ok(Engines {
        reference,
        fast_cpu,
        router,
        fingerprint: semantic_fingerprint(observed.as_bytes(), true),
        seal_guard,
    })
}

fn run_internal(
    args: &Args,
    text: &str,
    engines: &Engines,
    details: &mut BTreeMap<String, Value>,
    path: &mut Option<String>,
) -> AnyResult<(u64, Vec<u32>)> {
    details.insert("seal_guard_chars".to_owned(), json!(engines.seal_guard));
    // The PLAN/163 W3 direct cells measure post-encode stages, so their exact
    // ID rows are deliberately precomputed before the timer starts.
    if is_w3_phase(&args.phase) {
        return run_w3(args, text, engines, details);
    }
    // These production-path cells must run before any 4 MiB precomputation on
    // the same engines. That preserves the historical first-large-request
    // cache state while setup/parsing remains outside the timer.
    match args.phase.as_str() {
        "reference_core" => {
            let started = Instant::now();
            let encoding = engines.reference.encode_core(text)?;
            return Ok((nanos(started), encoding.ids));
        }
        "fast_cpu_encode" => {
            let started = Instant::now();
            let outcome = engines.fast_cpu.encode_with_source(text)?;
            let elapsed = nanos(started);
            details.insert(
                "fast_source".to_owned(),
                json!(format!("{:?}", outcome.source)),
            );
            return Ok((elapsed, outcome.encoding.ids));
        }
        "router_encode" => {
            let started = Instant::now();
            let routed = engines.router.encode_ids(text, false)?;
            let elapsed = nanos(started);
            *path = routed.path;
            details.insert("routed_backend".to_owned(), json!(routed.backend));
            return Ok((elapsed, routed.ids));
        }
        "router_append" => {
            let mut tail = TailState::new();
            let started = Instant::now();
            let report = engines.router.append(&mut tail, text)?;
            let elapsed = nanos(started);
            *path = Some(report.path);
            return Ok((elapsed, tail.ids().to_vec()));
        }
        "store_put_real_memory" => {
            return run_store_put(
                text,
                engines,
                engines.router.as_ref(),
                PostMode::Full,
                true,
                false,
                details,
                path,
            );
        }
        "store_put_real_tracked" => {
            return run_store_put(
                text,
                engines,
                engines.router.as_ref(),
                PostMode::Full,
                true,
                true,
                details,
                path,
            );
        }
        "all_ids" => {
            let (mut store, key) = new_store(engines, PostMode::Full, true, false)?;
            let put = store.put(key, text, engines.router.as_ref())?;
            let started = Instant::now();
            let ids = store.all_ids(put.handle)?;
            return Ok((nanos(started), ids));
        }
        "sqlite_save" => {
            return run_sqlite_save(args, text, engines, details);
        }
        "sqlite_load" => {
            return run_sqlite_load(args, text, engines, details);
        }
        _ => {}
    }

    let reference_encoding = engines.reference.encode_core(text)?;
    let precomputed = PrecomputedEncoder {
        encoding: reference_encoding.clone(),
        boundary: Arc::clone(&engines.router),
        certify_boundaries: true,
    };
    let precomputed_without_boundary = PrecomputedEncoder {
        encoding: reference_encoding.clone(),
        boundary: Arc::clone(&engines.router),
        certify_boundaries: false,
    };

    match args.phase.as_str() {
        "encoding_clone" => {
            let started = Instant::now();
            let encoding = black_box(reference_encoding.clone());
            let elapsed = nanos(started);
            Ok((elapsed, encoding.ids))
        }
        "tail_fill" => {
            let encoding = reference_encoding.clone();
            let mut tail = TailState::new();
            let started = Instant::now();
            tail.fill(text, encoding)?;
            let elapsed = nanos(started);
            Ok((elapsed, tail.ids().to_vec()))
        }
        "precomputed_append" => {
            let mut tail = TailState::new();
            let started = Instant::now();
            let report = precomputed.append(&mut tail, text)?;
            let elapsed = nanos(started);
            *path = Some(report.path);
            Ok((elapsed, tail.ids().to_vec()))
        }
        "boundary_search" => {
            let mut tail = TailState::new();
            tail.fill(text, reference_encoding)?;
            let ceil = u64::from(tail.text_chars()).saturating_sub(engines.seal_guard);
            let started = Instant::now();
            let cut = engines.router.last_certified_boundary(&tail, 0, ceil)?;
            let elapsed = nanos(started);
            details.insert(
                "cut".to_owned(),
                json!(cut.map(|value| json!({
                    "cut_tokens": value.cut_tokens,
                    "cut_char": value.cut_char
                }))),
            );
            Ok((elapsed, tail.ids().to_vec()))
        }
        "store_put_precomputed_full_memory" => run_store_put(
            text,
            engines,
            &precomputed,
            PostMode::Full,
            true,
            false,
            details,
            path,
        ),
        "store_put_precomputed_full_tracked" => run_store_put(
            text,
            engines,
            &precomputed,
            PostMode::Full,
            true,
            true,
            details,
            path,
        ),
        "store_put_precomputed_no_post" => run_store_put(
            text,
            engines,
            &precomputed,
            PostMode::None,
            true,
            false,
            details,
            path,
        ),
        "store_put_precomputed_blocks_only" => run_store_put(
            text,
            engines,
            &precomputed,
            PostMode::BlocksOnly,
            true,
            false,
            details,
            path,
        ),
        "store_put_precomputed_blocks_no_boundary" => run_store_put(
            text,
            engines,
            &precomputed_without_boundary,
            PostMode::BlocksOnly,
            true,
            false,
            details,
            path,
        ),
        "store_put_precomputed_boundary_only" => run_store_put(
            text,
            engines,
            &precomputed,
            PostMode::BoundaryOnly,
            true,
            false,
            details,
            path,
        ),
        "store_put_precomputed_no_tracking" => run_store_put(
            text,
            engines,
            &precomputed,
            PostMode::Full,
            false,
            false,
            details,
            path,
        ),
        "content_digest" => {
            let started = Instant::now();
            let digest = ContentDigest::from_bytes(text.as_bytes())?;
            let elapsed = nanos(started);
            details.insert(
                "checkpoint_count".to_owned(),
                json!(digest.entry().marks.len()),
            );
            Ok((elapsed, reference_encoding.ids))
        }
        "recovery_sha256" => {
            let started = Instant::now();
            let mut digest = Sha256::new();
            digest.update(RECOVERY_TEXT_DOMAIN);
            digest.update(text.as_bytes());
            let digest = black_box(digest.finalize());
            let elapsed = nanos(started);
            details.insert(
                "digest_prefix".to_owned(),
                json!(hex(&digest)[..16].to_owned()),
            );
            Ok((elapsed, reference_encoding.ids))
        }
        other => Err(format!("unknown internal phase {other:?}").into()),
    }
}

/// PLAN/163 W3: direct cells for the span bridge, the lazy-span candidates,
/// and payload hashing. Setup (reference encode, raw byte-length table,
/// boundary search) stays outside every timer; each prototype result is
/// compared against the product implementation after timing, and a sample
/// whose comparison fails becomes a process error instead of a data row.
fn run_w3(
    args: &Args,
    text: &str,
    engines: &Engines,
    details: &mut BTreeMap<String, Value>,
) -> AnyResult<(u64, Vec<u32>)> {
    const CHECKPOINT_INTERVAL: usize = 4096;
    let unicode = args.phase.ends_with("_unicode");
    let base = args.phase.trim_end_matches("_unicode");
    // The raw byte-length table is initialized before any timer, matching
    // the warm-table state of the measured PLAN/161 GPU seed contrast.
    let lengths = engines.reference.raw_byte_lengths()?;
    details.insert("raw_table_prewarmed".to_owned(), json!(true));
    details.insert("raw_table_entries".to_owned(), json!(lengths.len()));
    if unicode {
        details.insert(
            "unicode_payload_sha256".to_owned(),
            json!(hex(&Sha256::digest(text.as_bytes()))),
        );
        details.insert(
            "unicode_payload_generator".to_owned(),
            json!(UNICODE_GENERATOR_NOTE),
        );
        // The known-ID span bridge reconstructs byte offsets of the exact
        // input, so it presumes the artifact's normalizer maps this text to
        // itself. Assert that stability instead of measuring a payload the
        // production bridge would reject.
        if engines.reference.normalize(text)? != *text {
            return Err("the Unicode payload is not normalization-stable for this artifact".into());
        }
        details.insert("normalization_stable".to_owned(), json!(true));
    } else if !text.is_ascii() {
        return Err("the ASCII cell payload is unexpectedly non-ASCII".into());
    }

    match base {
        // W4b sanity pair: the complete state-seed shape (text in, session
        // committed, complete row returned) through the retained W4a
        // structure-of-arrays adoption versus the W4b lazy shared-buffer
        // adoption, both over the real corrected-Gigatoken engine. Store
        // construction and fingerprint registration stay outside the timer;
        // runtime/artifact setup was already outside it.
        "store_seed_soa_shape" => {
            let encoder = SoaSeedEncoder {
                fast: Arc::clone(&engines.fast_cpu),
            };
            let mut store = SessionStore::new(StoreConfig::default())?;
            store.enable_content_tracking()?;
            let key = store.register_fingerprint(engines.fingerprint, engines.seal_guard)?;
            let started = Instant::now();
            let put = store.put(key, text, &encoder)?;
            let ids = store.all_ids(put.handle)?;
            let elapsed = nanos(started);
            black_box(&ids);
            let info = store.session_info(put.handle)?;
            details.insert("sealed_tokens".to_owned(), json!(info.sealed_tokens));
            details.insert("seals".to_owned(), json!(store.stats().seals));
            details.insert(
                "full_row_materializations".to_owned(),
                json!(store.ids_materialization_count()),
            );
            Ok((elapsed, ids))
        }
        "store_seed_lazy_shape" | "store_seed_lazy_shape_overlap" => {
            let overlap = base.ends_with("_overlap");
            let mut store = SessionStore::new(StoreConfig::default())?;
            store.enable_content_tracking()?;
            record_overlap_details(&mut store, overlap, details);
            let key = store.register_fingerprint(engines.fingerprint, engines.seal_guard)?;
            let started = Instant::now();
            let put = store.put(key, text, engines.router.as_ref())?;
            let shared = store.shared_all_ids(put.handle)?;
            let elapsed = nanos(started);
            black_box(shared.as_slice());
            let info = store.session_info(put.handle)?;
            details.insert("sealed_tokens".to_owned(), json!(info.sealed_tokens));
            details.insert("seals".to_owned(), json!(store.stats().seals));
            details.insert(
                "full_row_materializations".to_owned(),
                json!(store.ids_materialization_count()),
            );
            if overlap {
                // The overlap path must be byte-identical to the serial
                // digest; assert it inside the measured process, outside
                // the timer.
                let entry = store
                    .content_index_entry(put.handle)?
                    .ok_or("overlap seed has no content-index entry")?;
                if entry != ContentDigest::from_bytes(text.as_bytes())?.entry() {
                    return Err("overlap content digest diverges from the direct scan".into());
                }
                details.insert("digest_equals_serial_scan".to_owned(), json!(true));
            }
            Ok((elapsed, shared.as_slice().to_vec()))
        }
        // W4c concurrency contrast: four independent stores and engine
        // sets seed the same 4 MiB payload from four caller threads.
        // Engine construction, store setup, and thread creation stay
        // outside the timer; the wall clock covers barrier release to
        // the last committed seed (a throughput observation).
        "store_seed_concurrent4" | "store_seed_concurrent4_overlap" => {
            let overlap = base.ends_with("_overlap");
            const LANES: usize = 4;
            let bytes = fs::read(&args.artifact)?;
            let mut lanes = Vec::with_capacity(LANES);
            for _ in 0..LANES {
                lanes.push(build_engines(&args.artifact, bytes.clone())?);
            }
            details.insert("lanes".to_owned(), json!(LANES));
            details.insert("overlap".to_owned(), json!(overlap));
            details.insert(
                "overlap_workers".to_owned(),
                json!(toktier_store_core::OverlapRunner::worker_count(
                    &RayonSeedOverlap
                )),
            );
            let barrier = std::sync::Barrier::new(LANES + 1);
            let mut rows: Vec<Option<Vec<u32>>> = vec![None; LANES];
            let mut lane_nanos = [0u64; LANES];
            let elapsed = std::thread::scope(|scope| -> AnyResult<u64> {
                let mut handles = Vec::with_capacity(LANES);
                for ((engines, row), lane_elapsed) in
                    lanes.iter().zip(rows.iter_mut()).zip(lane_nanos.iter_mut())
                {
                    let barrier = &barrier;
                    handles.push(scope.spawn(move || -> Result<(), String> {
                        let mut store =
                            SessionStore::new(StoreConfig::default()).map_err(|e| e.to_string())?;
                        store.enable_content_tracking().map_err(|e| e.to_string())?;
                        if overlap {
                            store.set_seed_overlap(Some(Arc::new(RayonSeedOverlap)));
                        }
                        let key = store
                            .register_fingerprint(engines.fingerprint, engines.seal_guard)
                            .map_err(|e| e.to_string())?;
                        barrier.wait();
                        let lane_started = Instant::now();
                        let put = store
                            .put(key, text, engines.router.as_ref())
                            .map_err(|e| e.to_string())?;
                        let shared = store
                            .shared_all_ids(put.handle)
                            .map_err(|e| e.to_string())?;
                        *lane_elapsed = nanos(lane_started);
                        *row = Some(shared.as_slice().to_vec());
                        Ok(())
                    }));
                }
                barrier.wait();
                let started = Instant::now();
                for handle in handles {
                    handle
                        .join()
                        .map_err(|_| "a concurrent seed lane panicked")??;
                }
                Ok(nanos(started))
            })?;
            let mut rows = rows.into_iter().map(Option::unwrap);
            let first = rows.next().ok_or("no concurrent seed rows")?;
            for row in rows {
                if row != first {
                    return Err("concurrent seed lanes disagree on the ID row".into());
                }
            }
            details.insert(
                "lane_elapsed_ms".to_owned(),
                json!(lane_nanos
                    .iter()
                    .map(|nanos| *nanos as f64 / 1e6)
                    .collect::<Vec<_>>()),
            );
            Ok((elapsed, first))
        }
        // W4c long-run stability: repeated overlap seeds in one process,
        // with resident memory recorded before and after. Every round's
        // row must match the first; the final row goes through the
        // standard out-of-band HF oracle.
        "store_seed_overlap_longrun" => {
            const ROUNDS: usize = 40;
            details.insert("rounds".to_owned(), json!(ROUNDS));
            details.insert(
                "overlap_workers".to_owned(),
                json!(toktier_store_core::OverlapRunner::worker_count(
                    &RayonSeedOverlap
                )),
            );
            let mut first: Option<Vec<u32>> = None;
            let started = Instant::now();
            for round in 0..ROUNDS {
                let mut store = SessionStore::new(StoreConfig::default())?;
                store.enable_content_tracking()?;
                store.set_seed_overlap(Some(Arc::new(RayonSeedOverlap)));
                let key = store.register_fingerprint(engines.fingerprint, engines.seal_guard)?;
                let put = store.put(key, text, engines.router.as_ref())?;
                let shared = store.shared_all_ids(put.handle)?;
                match &first {
                    None => {
                        first = Some(shared.as_slice().to_vec());
                        if let Some(kb) = resident_kb() {
                            details.insert("rss_kb_after_first_round".to_owned(), json!(kb));
                        }
                    }
                    Some(expected) => {
                        if shared.as_slice() != &expected[..] {
                            return Err(format!("long-run round {round} diverged").into());
                        }
                    }
                }
            }
            let elapsed = nanos(started);
            if let Some(kb) = resident_kb() {
                details.insert("rss_kb_after_last_round".to_owned(), json!(kb));
            }
            Ok((elapsed, first.ok_or("no long-run rows")?))
        }
        // S1: the current production bridge, timed directly.
        "spans_direct" => {
            let encoding = engines.reference.encode_core(text)?;
            let started = Instant::now();
            let spans = engines.reference.spans_for_ids(text, &encoding.ids)?;
            let elapsed = nanos(started);
            black_box(&spans);
            if spans != encoding.spans {
                return Err("spans_for_ids disagrees with the frozen HF character spans".into());
            }
            details.insert("span_count".to_owned(), json!(spans.len()));
            details.insert("bridge_equals_hf_spans".to_owned(), json!(true));
            if unicode {
                record_unicode_span_facts(text, &encoding.ids, &lengths, &spans, details)?;
            }
            Ok((elapsed, encoding.ids))
        }
        // S1b: the production one-pass SoA bridge (the W4a WP1 landing),
        // timed directly and compared element for element against the
        // retained pair bridge inside the same process.
        "spans_soa_direct" => {
            let ids = engines.reference.encode_ids(text, false)?;
            let started = Instant::now();
            let (starts, ends) = engines.reference.spans_soa_for_ids(text, &ids)?;
            let elapsed = nanos(started);
            black_box((&starts, &ends));
            let product = engines.reference.spans_for_ids(text, &ids)?;
            assert_soa_equals_product(&product, &starts, &ends)?;
            details.insert("soa_equals_pair_bridge".to_owned(), json!(true));
            details.insert("span_count".to_owned(), json!(starts.len()));
            Ok((elapsed, ids))
        }
        // H2b: the production incremental payload hasher (the W4a A1
        // landing): clone the sealed-prefix state and feed only the tail.
        "payload_digest_incremental_direct" => {
            let (encoding, cut_tokens, cut_char) = append_shape(engines, text, details)?;
            let sealed = &encoding.ids[..cut_tokens];
            let tail_ids = &encoding.ids[cut_tokens..];
            let tail_text = &text[cut_char..];
            let mut prefix = PayloadHasher::new();
            prefix.update_ids(sealed);
            let started = Instant::now();
            let digest = prefix.digest_with_tail(tail_ids, tail_text.as_bytes());
            let elapsed = nanos(started);
            black_box(digest);
            let product = payload_digest_parts(&[sealed, tail_ids], tail_text.as_bytes());
            if digest != product {
                return Err("product incremental digest diverges from the full digest".into());
            }
            details.insert("digest_equals_product".to_owned(), json!(true));
            details.insert(
                "digest_prefix".to_owned(),
                json!(hex(&digest)[..16].to_owned()),
            );
            Ok((elapsed, encoding.ids))
        }
        // S2: one-pass SoA prototype (WP1 candidate).
        "spans_soa_proto" => {
            let ids = engines.reference.encode_ids(text, false)?;
            let (elapsed, starts, ends) = if unicode {
                let started = Instant::now();
                let (starts, ends) =
                    proto::soa_spans_unicode_window(&ids, &lengths, text, 0, 0, Some(text.len()))
                        .map_err(box_error)?;
                (nanos(started), starts, ends)
            } else {
                let started = Instant::now();
                let (starts, ends) =
                    proto::soa_spans_ascii(&ids, &lengths, text.len()).map_err(box_error)?;
                (nanos(started), starts, ends)
            };
            black_box((&starts, &ends));
            let product = engines.reference.spans_for_ids(text, &ids)?;
            assert_soa_equals_product(&product, &starts, &ends)?;
            details.insert("prototype_equals_product".to_owned(), json!(true));
            details.insert("span_count".to_owned(), json!(starts.len()));
            Ok((elapsed, ids))
        }
        // S3(a): allocation-free streaming closure check.
        "spans_lazy_closure" => {
            let ids = engines.reference.encode_ids(text, false)?;
            let started = Instant::now();
            let total = proto::closure_sum(&ids, &lengths, text.len()).map_err(box_error)?;
            let elapsed = nanos(started);
            black_box(total);
            details.insert("closure_bytes".to_owned(), json!(total));
            Ok((elapsed, ids))
        }
        // S3(b): construct only the mutable-tail window's spans.
        "spans_lazy_tail_window" => {
            let encoding = engines.reference.encode_core(text)?;
            let (cut_tokens, cut_char, boundary_available) =
                window_start(engines, text, &encoding)?;
            let tail_ids = &encoding.ids[cut_tokens..];
            let (elapsed, starts, ends) = if unicode {
                let cut_byte = byte_of_char(text, cut_char)?;
                let window = &text[cut_byte..];
                let base_char = u32::try_from(cut_char).map_err(|_| "cut exceeds u32")?;
                let started = Instant::now();
                let (starts, ends) = proto::soa_spans_unicode_window(
                    tail_ids,
                    &lengths,
                    window,
                    base_char,
                    0,
                    Some(window.len()),
                )
                .map_err(box_error)?;
                (nanos(started), starts, ends)
            } else {
                let started = Instant::now();
                let (starts, ends) =
                    proto::tail_spans_ascii_from_suffix(tail_ids, &lengths, text.len())
                        .map_err(box_error)?;
                (nanos(started), starts, ends)
            };
            black_box((&starts, &ends));
            let product = engines.reference.spans_for_ids(text, &encoding.ids)?;
            assert_soa_equals_product(&product[cut_tokens..], &starts, &ends)?;
            details.insert("prototype_equals_product".to_owned(), json!(true));
            details.insert("boundary_available".to_owned(), json!(boundary_available));
            details.insert("window_start_token".to_owned(), json!(cut_tokens));
            details.insert("window_tokens".to_owned(), json!(tail_ids.len()));
            Ok((elapsed, encoding.ids))
        }
        // S3(c) build: sparse cumulative checkpoints, no span materialization.
        "spans_checkpoint_build" => {
            let ids = engines.reference.encode_ids(text, false)?;
            let (elapsed, checkpoints) = if unicode {
                let started = Instant::now();
                let checkpoints =
                    proto::build_checkpoints_unicode(&ids, &lengths, text, CHECKPOINT_INTERVAL)
                        .map_err(box_error)?;
                (nanos(started), checkpoints)
            } else {
                let started = Instant::now();
                let checkpoints =
                    proto::build_checkpoints_ascii(&ids, &lengths, text.len(), CHECKPOINT_INTERVAL)
                        .map_err(box_error)?;
                (nanos(started), checkpoints)
            };
            black_box(&checkpoints);
            details.insert("checkpoint_interval".to_owned(), json!(CHECKPOINT_INTERVAL));
            details.insert("checkpoint_count".to_owned(), json!(checkpoints.len()));
            details.insert(
                "checkpoint_struct_bytes".to_owned(),
                json!(checkpoints.len() * std::mem::size_of::<proto::Checkpoint>()),
            );
            Ok((elapsed, ids))
        }
        // S3(c) rebuild: one arbitrary window from its nearest checkpoint.
        "spans_checkpoint_window" => {
            let ids = engines.reference.encode_ids(text, false)?;
            let checkpoints = if unicode {
                proto::build_checkpoints_unicode(&ids, &lengths, text, CHECKPOINT_INTERVAL)
                    .map_err(box_error)?
            } else {
                proto::build_checkpoints_ascii(&ids, &lengths, text.len(), CHECKPOINT_INTERVAL)
                    .map_err(box_error)?
            };
            let window_index = (ids.len() / 2) / CHECKPOINT_INTERVAL;
            if window_index == 0 {
                return Err("the input is too small for a checkpoint window".into());
            }
            let anchor = checkpoints[window_index - 1];
            if anchor.tokens != window_index * CHECKPOINT_INTERVAL {
                return Err("checkpoint anchor does not align with the window start".into());
            }
            let start = anchor.tokens;
            let end = (start + CHECKPOINT_INTERVAL).min(ids.len());
            let window_ids = &ids[start..end];
            let (elapsed, starts, ends) = if unicode {
                let started = Instant::now();
                let (starts, ends) = proto::rebuild_window_unicode(
                    window_ids,
                    &lengths,
                    text,
                    anchor.byte_end,
                    anchor,
                )
                .map_err(box_error)?;
                (nanos(started), starts, ends)
            } else {
                let started = Instant::now();
                let (starts, ends) =
                    proto::rebuild_window_ascii(window_ids, &lengths, anchor.byte_end)
                        .map_err(box_error)?;
                (nanos(started), starts, ends)
            };
            black_box((&starts, &ends));
            let product = engines.reference.spans_for_ids(text, &ids)?;
            assert_soa_equals_product(&product[start..end], &starts, &ends)?;
            details.insert("prototype_equals_product".to_owned(), json!(true));
            details.insert("checkpoint_interval".to_owned(), json!(CHECKPOINT_INTERVAL));
            details.insert("checkpoint_count".to_owned(), json!(checkpoints.len()));
            details.insert("window_start_token".to_owned(), json!(start));
            details.insert("window_tokens".to_owned(), json!(end - start));
            Ok((elapsed, ids))
        }
        // H1 (seed shape): the payload digest over the whole row at once,
        // as a pre-seal commit hashes it.
        "payload_digest_seed_shape" => {
            let ids = engines.reference.encode_ids(text, false)?;
            let started = Instant::now();
            let digest = payload_digest_parts(&[&ids], text.as_bytes());
            let elapsed = nanos(started);
            black_box(digest);
            details.insert(
                "digest_prefix".to_owned(),
                json!(hex(&digest)[..16].to_owned()),
            );
            Ok((elapsed, ids))
        }
        // H1 (append shape): the same product function over the sealed
        // prefix plus tail, which every post-seal commit recomputes today.
        "payload_digest_append_shape" => {
            let (encoding, cut_tokens, cut_char) = append_shape(engines, text, details)?;
            let sealed = &encoding.ids[..cut_tokens];
            let tail_ids = &encoding.ids[cut_tokens..];
            let tail_text = &text[cut_char..];
            let started = Instant::now();
            let digest = payload_digest_parts(&[sealed, tail_ids], tail_text.as_bytes());
            let elapsed = nanos(started);
            black_box(digest);
            details.insert(
                "digest_prefix".to_owned(),
                json!(hex(&digest)[..16].to_owned()),
            );
            Ok((elapsed, encoding.ids))
        }
        // H2: clone a saved sealed-prefix hasher state and feed only the
        // tail; the digest must be bit-identical to the product function.
        "payload_digest_incremental_proto" => {
            let (encoding, cut_tokens, cut_char) = append_shape(engines, text, details)?;
            let sealed = &encoding.ids[..cut_tokens];
            let tail_ids = &encoding.ids[cut_tokens..];
            let tail_text = &text[cut_char..];
            let prefix = proto::sealed_prefix_hasher(sealed);
            let started = Instant::now();
            let digest = proto::commit_digest_from_prefix(&prefix, tail_ids, tail_text.as_bytes());
            let elapsed = nanos(started);
            black_box(digest);
            let product = payload_digest_parts(&[sealed, tail_ids], tail_text.as_bytes());
            if digest != product {
                return Err("incremental prefix digest diverges from the product digest".into());
            }
            details.insert("digest_equals_product".to_owned(), json!(true));
            details.insert(
                "digest_prefix".to_owned(),
                json!(hex(&digest)[..16].to_owned()),
            );
            Ok((elapsed, encoding.ids))
        }
        // H3: whole-row digest fed through a stack chunk buffer instead of
        // one 4-byte update per ID.
        "payload_digest_chunked" => {
            let ids = engines.reference.encode_ids(text, false)?;
            let started = Instant::now();
            let digest = proto::payload_digest_chunked(&[&ids], text.as_bytes());
            let elapsed = nanos(started);
            black_box(digest);
            let product = payload_digest_parts(&[&ids], text.as_bytes());
            if digest != product {
                return Err("chunked payload digest diverges from the product digest".into());
            }
            details.insert("digest_equals_product".to_owned(), json!(true));
            details.insert(
                "digest_prefix".to_owned(),
                json!(hex(&digest)[..16].to_owned()),
            );
            Ok((elapsed, ids))
        }
        // G1: one Aho-Corasick pass over the input with the same pattern
        // set the router's added-token gate uses.
        "added_gate_scan" => {
            let ids = engines.reference.encode_ids(text, false)?;
            let has_normalizer = engines.reference.has_effective_normalizer();
            let patterns = engines
                .reference
                .added_tokens()
                .into_iter()
                .filter_map(|(_, content, normalized)| {
                    (!content.is_empty() && (!normalized || !has_normalizer)).then_some(content)
                })
                .collect::<Vec<_>>();
            if patterns.is_empty() {
                return Err("the artifact has no added-token patterns to scan for".into());
            }
            let matcher = AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&patterns)
                .map_err(|error| format!("failed to build the added-token matcher: {error}"))?;
            let started = Instant::now();
            let matched = matcher.is_match(text);
            let elapsed = nanos(started);
            black_box(matched);
            details.insert("pattern_count".to_owned(), json!(patterns.len()));
            details.insert("matched".to_owned(), json!(matched));
            Ok((elapsed, ids))
        }
        other => Err(format!("unknown W3 phase {other:?}").into()),
    }
}

fn box_error(message: String) -> Box<dyn Error> {
    message.into()
}

/// Element-for-element comparison between the product pair spans and a
/// prototype SoA result.
fn assert_soa_equals_product(
    product: &[(u32, u32)],
    starts: &[u32],
    ends: &[u32],
) -> AnyResult<()> {
    if product.len() != starts.len() || product.len() != ends.len() {
        return Err(format!(
            "prototype span count diverges: product={}, starts={}, ends={}",
            product.len(),
            starts.len(),
            ends.len()
        )
        .into());
    }
    for (index, &(start, end)) in product.iter().enumerate() {
        if start != starts[index] || end != ends[index] {
            return Err(format!(
                "prototype span {index} diverges: product=({start},{end}), \
                 prototype=({},{})",
                starts[index], ends[index]
            )
            .into());
        }
    }
    Ok(())
}

/// Locate the mutable-tail window start: the certified boundary when the
/// oracle finds one, otherwise the nearest earlier clean-character token
/// start (recorded honestly via `boundary_available`).
fn window_start(
    engines: &Engines,
    text: &str,
    encoding: &CoreEncoding,
) -> AnyResult<(usize, usize, bool)> {
    let mut tail = TailState::new();
    tail.fill(text, encoding.clone())?;
    let ceil = u64::from(tail.text_chars()).saturating_sub(engines.seal_guard);
    let cut = engines.router.last_certified_boundary(&tail, 0, ceil)?;
    if let Some(cut) = cut {
        let cut_char = usize::try_from(cut.cut_char).map_err(|_| "cut_char exceeds usize")?;
        return Ok((cut.cut_tokens, cut_char, true));
    }
    let spans = &encoding.spans;
    let mut token = spans.len().saturating_sub(292).max(1);
    while token > 1 && spans[token].0 < spans[token - 1].1 {
        token -= 1;
    }
    Ok((token, spans[token].0 as usize, false))
}

/// Byte offset of a character index; `char_index == chars` maps to the end
/// of the text. Runs outside every timer.
fn byte_of_char(text: &str, char_index: usize) -> AnyResult<usize> {
    if char_index == 0 {
        return Ok(0);
    }
    let mut seen = 0usize;
    for (byte, _) in text.char_indices() {
        if seen == char_index {
            return Ok(byte);
        }
        seen += 1;
    }
    // `seen` is now the total character count.
    if seen == char_index {
        return Ok(text.len());
    }
    Err("character index is outside the text".into())
}

/// The post-seal commit shape: the exact encoding plus the certified
/// boundary's token and character cut. ASCII-only cells use this helper,
/// so the character cut equals the byte cut.
fn append_shape(
    engines: &Engines,
    text: &str,
    details: &mut BTreeMap<String, Value>,
) -> AnyResult<(CoreEncoding, usize, usize)> {
    if !text.is_ascii() {
        return Err("the append-shape cells expect the ASCII payload".into());
    }
    let encoding = engines.reference.encode_core(text)?;
    let (cut_tokens, cut_char, boundary_available) = window_start(engines, text, &encoding)?;
    if !boundary_available {
        return Err("no certified boundary was found for the append shape".into());
    }
    details.insert("sealed_tokens".to_owned(), json!(cut_tokens));
    details.insert(
        "tail_tokens".to_owned(),
        json!(encoding.ids.len() - cut_tokens),
    );
    details.insert("tail_text_bytes".to_owned(), json!(text.len() - cut_char));
    Ok((encoding, cut_tokens, cut_char))
}

/// Facts that prove the Unicode payload exercises the intended span cases.
fn record_unicode_span_facts(
    text: &str,
    ids: &[u32],
    lengths: &[usize],
    spans: &[(u32, u32)],
    details: &mut BTreeMap<String, Value>,
) -> AnyResult<()> {
    let shared_pairs = spans.windows(2).filter(|pair| pair[0] == pair[1]).count();
    let mut split_starts = 0usize;
    let mut cursor = 0usize;
    for &id in ids {
        if !text.is_char_boundary(cursor) {
            split_starts += 1;
        }
        let length = *lengths
            .get(id as usize)
            .ok_or_else(|| format!("unknown vocabulary id {id}"))?;
        cursor += length;
    }
    details.insert("adjacent_shared_span_pairs".to_owned(), json!(shared_pairs));
    details.insert(
        "token_starts_inside_character".to_owned(),
        json!(split_starts),
    );
    Ok(())
}

fn run_sqlite_save(
    args: &Args,
    text: &str,
    engines: &Engines,
    details: &mut BTreeMap<String, Value>,
) -> AnyResult<(u64, Vec<u32>)> {
    let home = args.home.as_ref().ok_or("sqlite_save requires --home")?;
    fs::create_dir_all(home)?;
    let database_path = home.join("store.sqlite3");
    let (mut store, key) = new_store(engines, PostMode::Full, true, true)?;
    let put = store.put(key, text, engines.router.as_ref())?;
    let mut database = StoreDb::open(&database_path)?;
    let session = [NamedSessionRef {
        name: "profiled-seed",
        handle: put.handle,
        transcript: text,
    }];
    let started = Instant::now();
    database.save_named_recoverable(&store, &session)?;
    let elapsed = nanos(started);
    details.insert(
        "database_bytes".to_owned(),
        json!(fs::metadata(database_path)?.len()),
    );
    Ok((elapsed, store.all_ids(put.handle)?))
}

fn run_sqlite_load(
    args: &Args,
    text: &str,
    engines: &Engines,
    details: &mut BTreeMap<String, Value>,
) -> AnyResult<(u64, Vec<u32>)> {
    let home = args.home.as_ref().ok_or("sqlite_load requires --home")?;
    fs::create_dir_all(home)?;
    let database_path = home.join("store.sqlite3");
    let (mut store, key) = new_store(engines, PostMode::Full, true, true)?;
    let put = store.put(key, text, engines.router.as_ref())?;
    {
        let mut database = StoreDb::open(&database_path)?;
        database.save_named_recoverable(
            &store,
            &[NamedSessionRef {
                name: "profiled-seed",
                handle: put.handle,
                transcript: text,
            }],
        )?;
    }
    let started = Instant::now();
    let database = StoreDb::open(&database_path)?;
    let (mut loaded, recovered) =
        database.load_named_recoverable(&SingleEngine(engines.router.as_ref()))?;
    let elapsed = nanos(started);
    let handle = recovered
        .first()
        .ok_or("SQLite load recovered no named session")?
        .handle;
    details.insert("recovered_sessions".to_owned(), json!(recovered.len()));
    Ok((elapsed, loaded.all_ids(handle)?))
}

#[allow(clippy::too_many_arguments)]
fn run_store_put(
    text: &str,
    engines: &Engines,
    encoder: &dyn SessionEncoder,
    post_mode: PostMode,
    content_tracking: bool,
    recovery_tracking: bool,
    details: &mut BTreeMap<String, Value>,
    path: &mut Option<String>,
) -> AnyResult<(u64, Vec<u32>)> {
    let (mut store, key) = new_store(engines, post_mode, content_tracking, recovery_tracking)?;
    let started = Instant::now();
    let put = store.put(key, text, encoder)?;
    let elapsed = nanos(started);
    let stats = store.stats();
    *path = stats
        .path_counts
        .iter()
        .find(|(_, count)| **count > 0)
        .map(|(name, _)| name.clone());
    let info = store.session_info(put.handle)?;
    details.insert("node_count".to_owned(), json!(stats.node_count));
    details.insert("seals".to_owned(), json!(stats.seals));
    details.insert("sealed_tokens".to_owned(), json!(stats.sealed_tokens));
    details.insert("chain_detaches".to_owned(), json!(stats.chain_detaches));
    details.insert("safe_char".to_owned(), json!(info.safe_char));
    details.insert("tail_bytes".to_owned(), json!(info.tail_bytes));
    details.insert("approx_bytes".to_owned(), json!(info.approx_bytes));
    Ok((elapsed, store.all_ids(put.handle)?))
}

fn new_store(
    engines: &Engines,
    mode: PostMode,
    content_tracking: bool,
    recovery_tracking: bool,
) -> AnyResult<(SessionStore, KeyId)> {
    let mut store = SessionStore::new(store_config(mode))?;
    if recovery_tracking {
        store.enable_recovery_tracking()?;
    }
    if content_tracking {
        store.enable_content_tracking()?;
    }
    let key = store.register_fingerprint(engines.fingerprint, engines.seal_guard)?;
    Ok((store, key))
}

fn store_config(mode: PostMode) -> StoreConfig {
    let beyond = INPUT_BYTES + 1;
    match mode {
        PostMode::Full => StoreConfig::default(),
        PostMode::None => StoreConfig {
            block_chars: beyond as u64,
            tail_soft_cap_bytes: beyond,
            tail_hard_cap_bytes: beyond,
            node_tail_cap_bytes: beyond,
            max_sessions: 1024,
        },
        PostMode::BlocksOnly => StoreConfig {
            block_chars: 4096,
            tail_soft_cap_bytes: beyond,
            tail_hard_cap_bytes: beyond,
            node_tail_cap_bytes: 65536,
            max_sessions: 1024,
        },
        PostMode::BoundaryOnly => StoreConfig {
            block_chars: beyond as u64,
            tail_soft_cap_bytes: 65536,
            tail_hard_cap_bytes: beyond,
            node_tail_cap_bytes: beyond,
            max_sessions: 1024,
        },
    }
}

fn semantic_fingerprint(artifact_sha256: &[u8], fast_cpu: bool) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"toktier.rust.fingerprint.v1\0");
    for component in [
        artifact_sha256,
        b"tokenizers" as &[u8],
        b"0.22.2",
        b"toktier-rust-serving-v1",
        if fast_cpu { b"fast_cpu" } else { b"hf" },
        if fast_cpu {
            b"toktier-fast-repair-v1" as &[u8]
        } else {
            b""
        },
    ] {
        digest.update((component.len() as u32).to_le_bytes());
        digest.update(component);
    }
    digest.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
