use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use toktier_routing_core::{FastCpuEngine, FastRepairSpec, NativeRouter, ReferenceEngine};

use crate::artifact::{ArtifactManager, Revision};
use crate::diagnostics::{
    Backend, Certification, CudaFacts, DoctorFacts, ExecutionFacts, ReasonCode, RoutePlan,
    RuntimeBuildFacts, StoreStats,
};
use crate::manifest::{LocalArtifact, Registry};
use crate::session::{SessionStoreState, TokenizerInner};
#[cfg(feature = "jit")]
use crate::JitCompiler;
use crate::{
    Device, Encoding, Error, ErrorCode, GpuDelivery, Policy, RaggedEncoding, Result, Session,
    TokenBuffer,
};

/// Options for a one-shot encode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncodeOptions {
    pub add_special_tokens: bool,
    pub offsets: bool,
}

/// Options for decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    pub skip_special_tokens: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
        }
    }
}

#[derive(Debug, Clone)]
struct RuntimeConfig {
    home: Option<PathBuf>,
    artifact_cache: PathBuf,
    artifact_manager: Option<ArtifactManager>,
    explicit_artifacts: HashMap<String, PathBuf>,
    policy: Policy,
    device: Device,
    gpu_delivery: GpuDelivery,
    gpu_min_bytes: u64,
    diagnostics: bool,
    persistent_sessions: bool,
    seed_digest_overlap: bool,
    #[cfg(feature = "jit")]
    jit_compiler: Option<JitCompiler>,
}

/// Builder for the immutable Rust runtime.
#[derive(Debug, Clone)]
pub struct RuntimeBuilder {
    config: RuntimeConfig,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        let artifact_cache =
            crate::fsutil::default_cache_directory("TOKTIER_ARTIFACT_CACHE", "artifacts");
        Self {
            config: RuntimeConfig {
                home: None,
                artifact_cache,
                artifact_manager: None,
                explicit_artifacts: HashMap::new(),
                policy: Policy::Certified,
                device: Device::Auto,
                gpu_delivery: GpuDelivery::Auto,
                gpu_min_bytes: 64 * 1024,
                diagnostics: false,
                persistent_sessions: false,
                seed_digest_overlap: false,
                #[cfg(feature = "jit")]
                jit_compiler: None,
            },
        }
    }
}

impl RuntimeBuilder {
    pub fn home(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.home = Some(path.into());
        self.config.persistent_sessions = true;
        self
    }

    pub fn artifact_cache(mut self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        self.config.artifact_cache = path.clone();
        if let Some(manager) = &self.config.artifact_manager {
            self.config.artifact_manager = Some(manager.with_cache(path));
        }
        self
    }

    /// Install a shareable Rust-native artifact acquisition policy.
    pub fn artifacts(mut self, manager: ArtifactManager) -> Self {
        self.config.artifact_cache = manager.cache_root().to_path_buf();
        self.config.artifact_manager = Some(manager);
        self
    }

    /// Bind one family to an explicitly supplied local artifact directory.
    pub fn artifact_directory(
        mut self,
        family: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        self.config
            .explicit_artifacts
            .insert(family.into(), path.into());
        self
    }

    pub fn policy(mut self, policy: Policy) -> Self {
        self.config.policy = policy;
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.config.device = device;
        self
    }

    pub fn gpu_delivery(mut self, delivery: GpuDelivery) -> Self {
        self.config.gpu_delivery = delivery;
        self
    }

    pub fn gpu_min_bytes(mut self, bytes: u64) -> Self {
        self.config.gpu_min_bytes = bytes;
        self
    }

    pub fn diagnostics(mut self, enabled: bool) -> Self {
        self.config.diagnostics = enabled;
        self
    }

    pub fn persistent_sessions(mut self, enabled: bool) -> Self {
        self.config.persistent_sessions = enabled;
        self
    }

    /// Overlap the session-seed content-digest scan with the seed encode
    /// on the process-wide bounded worker pool (PLAN/162 WP5/WP6). The
    /// digest bytes, validation ordering, and failure behavior are
    /// identical in both modes; overlap changes only wall-clock
    /// scheduling, hiding the scan under the encode when a worker is
    /// free. Defaults to off, which keeps the fully serial seed path.
    pub fn seed_digest_overlap(mut self, enabled: bool) -> Self {
        self.config.seed_digest_overlap = enabled;
        self
    }

    /// Select a direct-NVCC compiler/cache policy for `GpuDelivery::Jit`.
    #[cfg(feature = "jit")]
    pub fn jit_compiler(mut self, compiler: JitCompiler) -> Self {
        self.config.jit_compiler = Some(compiler);
        self
    }

    /// Conspicuous one-runtime opt-in for an unregistered direct JIT tuple.
    /// It does not persist permission and does not make the result certified.
    #[cfg(feature = "jit")]
    pub fn accept_uncertified_jit(mut self, accept: bool) -> Result<Self> {
        let compiler = JitCompiler::builder()
            .accept_uncertified_jit(accept)
            .build()?;
        self.config.jit_compiler = Some(compiler);
        Ok(self)
    }

    pub fn build(self) -> Result<Runtime> {
        if self.config.artifact_cache.as_os_str().is_empty() {
            return Err(Error::new(
                ErrorCode::ConfigInvalid,
                "artifact_cache must not be empty",
            ));
        }
        if self.config.persistent_sessions && self.config.home.is_none() {
            return Err(Error::new(
                ErrorCode::ConfigInvalid,
                "persistent sessions require RuntimeBuilder::home",
            ));
        }
        if matches!(self.config.device, Device::Cpu)
            && !matches!(
                self.config.gpu_delivery,
                GpuDelivery::Auto | GpuDelivery::Disabled
            )
        {
            return Err(Error::new(
                ErrorCode::ConfigInvalid,
                "a CPU device cannot request GPU kernel delivery",
            ));
        }
        let registry = Registry::load()?;
        let artifacts = match &self.config.artifact_manager {
            Some(manager) => manager.with_cache(self.config.artifact_cache.clone()),
            None => ArtifactManager::builder()
                .cache(self.config.artifact_cache.clone())
                .build()?,
        };
        Ok(Runtime {
            inner: Arc::new(RuntimeInner {
                registry,
                artifacts,
                config: self.config,
            }),
        })
    }
}

#[derive(Debug)]
struct RuntimeInner {
    registry: Arc<Registry>,
    artifacts: ArtifactManager,
    config: RuntimeConfig,
}

/// Immutable root object. Clone is cheap and thread-safe.
#[derive(Debug, Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// Deterministically release this root handle. Tokenizers cloned from it
    /// retain their own resources until they are dropped or shut down.
    pub fn shutdown(self) {}

    /// Return machine-readable build and non-invasive device facts. This does
    /// not load a tokenizer or CUDA kernel and never widens routing policy.
    pub fn doctor(&self) -> DoctorFacts {
        let runtime_build = RuntimeBuildFacts {
            source_digest: env!("TOKTIER_RUST_API_SOURCE_SHA256").to_owned(),
            fast_cpu_source_digest: env!("TOKTIER_RUST_API_FAST_CPU_SOURCE_SHA256").to_owned(),
            native_host_source_digest: env!("TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256")
                .to_owned(),
            toolchain: env!("TOKTIER_RUST_API_TOOLCHAIN").to_owned(),
            build_flags: env!("TOKTIER_RUST_API_BUILD_FLAGS")
                .split('\x1f')
                .map(str::to_owned)
                .collect(),
            certified: self.inner.registry.rust_api_build_certified(),
        };
        let ordinal = match self.inner.config.device {
            Device::Cuda(ordinal) => ordinal,
            Device::Auto => 0,
            Device::Cpu => {
                return DoctorFacts {
                    crate_version: env!("CARGO_PKG_VERSION").to_owned(),
                    oracle: crate::ORACLE.to_owned(),
                    registry_verified: true,
                    python_required: false,
                    sqlite_compiled: cfg!(feature = "sqlite"),
                    prebuilt_gpu_compiled: cfg!(feature = "prebuilt-gpu"),
                    jit_compiled: cfg!(feature = "jit"),
                    runtime_build,
                    cuda: None,
                };
            }
        };
        let cuda = cuda_facts(ordinal);
        DoctorFacts {
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            oracle: crate::ORACLE.to_owned(),
            registry_verified: true,
            python_required: false,
            sqlite_compiled: cfg!(feature = "sqlite"),
            prebuilt_gpu_compiled: cfg!(feature = "prebuilt-gpu"),
            jit_compiled: cfg!(feature = "jit"),
            runtime_build,
            cuda: Some(cuda),
        }
    }

    /// Load a canonical family from the verified local cache.
    pub fn load(&self, family: &str) -> Result<Tokenizer> {
        let explicit = self.inner.config.explicit_artifacts.get(family);
        if explicit.is_none() {
            self.inner.artifacts.ensure(family, &self.inner.registry)?;
        }
        let artifact = self.inner.registry.verify_local(
            family,
            &self.inner.config.artifact_cache,
            explicit.map(PathBuf::as_path),
        )?;
        self.build_tokenizer(artifact)
    }

    /// Verify and load a canonical family from one explicit local directory.
    pub fn load_local(&self, family: &str, directory: impl AsRef<Path>) -> Result<Tokenizer> {
        let artifact = self.inner.registry.verify_local(
            family,
            &self.inner.config.artifact_cache,
            Some(directory.as_ref()),
        )?;
        self.build_tokenizer(artifact)
    }

    /// Resolve a frozen repository/revision through the shipped verified
    /// sibling table, then execute its canonical digest-identical or proven
    /// equivalent tokenizer artifact.
    pub fn load_repository(&self, repo_id: &str, revision: &str) -> Result<Tokenizer> {
        let family = self.family_for_repo(repo_id, revision)?;
        self.load(&family)
    }

    /// Resolve an immutable repository commit, acquire its canonical artifact,
    /// and return a verified tokenizer handle.
    pub fn from_pretrained(&self, repo_id: &str, revision: Revision) -> Result<Tokenizer> {
        self.load_repository(repo_id, revision.as_str())
    }

    /// Clone the runtime's artifact lifecycle manager.
    pub fn artifacts(&self) -> ArtifactManager {
        self.inner.artifacts.clone()
    }

    pub(crate) fn family_for_repo(&self, repo_id: &str, revision: &str) -> Result<String> {
        self.inner.registry.resolve_repo(repo_id, revision)
    }

    fn build_tokenizer(&self, artifact: LocalArtifact) -> Result<Tokenizer> {
        let bytes = artifact.bytes();
        // The bytes already passed content-hash verification against the
        // shipped manifest, so a reference-engine load failure is not a
        // hash mismatch: it means the shipped tables admit an artifact the
        // frozen oracle cannot execute, which is an internal inconsistency.
        let reference = Arc::new(ReferenceEngine::from_bytes(&bytes).map_err(|error| {
            Error::new(ErrorCode::Internal, error.to_string())
                .with_path(artifact.tokenizer_json())
                .with_family(&artifact.identity().family)
        })?);
        let family = artifact.identity().family.clone();
        let mut reasons = Vec::new();
        let runtime_build_certified = self.inner.registry.rust_api_build_certified();
        let accelerated_build_allowed =
            runtime_build_certified || self.inner.config.policy == Policy::Experimental;
        if !accelerated_build_allowed && self.inner.config.policy != Policy::Reference {
            reasons.push(ReasonCode::RuntimeBuildUncertified);
        }
        let fast_cpu = if self.inner.config.policy == Policy::Reference {
            reasons.push(ReasonCode::ReferenceRequested);
            None
        } else if !accelerated_build_allowed {
            None
        } else {
            self.build_fast_cpu(&artifact, Arc::clone(&reference), &mut reasons)?
        };

        #[allow(unused_mut)]
        let mut gpu = None;
        #[allow(unused_mut)]
        let mut gpu_detail = None;
        #[cfg(feature = "jit")]
        let mut jit_detail = None;
        let wants_gpu = !matches!(self.inner.config.device, Device::Cpu)
            && !matches!(self.inner.config.gpu_delivery, GpuDelivery::Disabled)
            && self.inner.config.policy != Policy::Reference;
        if wants_gpu && !accelerated_build_allowed {
            if matches!(self.inner.config.device, Device::Cuda(_)) {
                return Err(Error::new(
                    ErrorCode::UncertifiedRuntime,
                    "this Rust build is not present in the shipped runtime-build registry; use a registered release build or explicitly select Policy::Experimental",
                ));
            }
            reasons.push(ReasonCode::GpuUncertified);
        } else if wants_gpu {
            #[cfg(feature = "prebuilt-gpu")]
            {
                let ordinal = match self.inner.config.device {
                    Device::Cuda(value) => i32::try_from(value).map_err(|_| {
                        Error::new(ErrorCode::InvalidArgument, "CUDA ordinal exceeds i32")
                    })?,
                    Device::Auto => 0,
                    Device::Cpu => unreachable!(),
                };
                let built = if self.inner.config.gpu_delivery == GpuDelivery::Jit {
                    #[cfg(feature = "jit")]
                    {
                        let compiler = match &self.inner.config.jit_compiler {
                            Some(compiler) => compiler.clone(),
                            None => JitCompiler::builder().build()?,
                        };
                        compiler
                            .compile_product(&family, &self.inner.registry, ordinal)
                            .and_then(|product| {
                                let facts = product.facts.clone();
                                crate::gpu_data::build_jit(
                                    &self.inner.registry,
                                    &artifact,
                                    Arc::clone(&reference),
                                    ordinal,
                                    self.inner.config.policy,
                                    &product,
                                )
                                .map(|built| (built, Some(facts)))
                            })
                    }
                    #[cfg(not(feature = "jit"))]
                    {
                        Err(Error::new(
                            ErrorCode::KernelIncompatible,
                            "GpuDelivery::Jit requires the crate's `jit` feature",
                        ))
                    }
                } else {
                    let prebuilt = crate::gpu_data::build_prebuilt(
                        &self.inner.registry,
                        &artifact,
                        Arc::clone(&reference),
                        ordinal,
                        self.inner.config.policy,
                    );
                    #[cfg(feature = "jit")]
                    {
                        prebuilt.map(|built| (built, None))
                    }
                    #[cfg(not(feature = "jit"))]
                    {
                        prebuilt.map(|built| (built, ()))
                    }
                };
                match built {
                    Ok((built, _maybe_facts)) => {
                        gpu_detail = Some((built.architecture, built.driver_api_version));
                        gpu = Some(built.engine);
                        #[cfg(feature = "jit")]
                        {
                            jit_detail = _maybe_facts;
                        }
                    }
                    Err(error) if matches!(self.inner.config.device, Device::Auto) => {
                        reasons.push(if error.code() == ErrorCode::UncertifiedTokenizer {
                            ReasonCode::GpuUncertified
                        } else {
                            ReasonCode::GpuUnavailable
                        });
                    }
                    Err(error) => return Err(error),
                }
            }
            #[cfg(not(feature = "prebuilt-gpu"))]
            {
                if matches!(self.inner.config.device, Device::Cuda(_)) {
                    return Err(Error::new(
                        ErrorCode::KernelIncompatible,
                        "the crate was built without the prebuilt-gpu feature",
                    ));
                }
                reasons.push(ReasonCode::GpuUnavailable);
            }
        }

        let mut chain = Vec::new();
        let mut thresholds = Vec::new();
        let mut backends = Vec::new();
        if gpu.is_some() {
            chain.push("gpu".to_owned());
            thresholds.push(self.inner.config.gpu_min_bytes);
            backends.push(Backend::Gpu);
        }
        if fast_cpu.is_some() {
            chain.push("fast_cpu".to_owned());
            thresholds.push(0);
            backends.push(Backend::FastCpu);
        }
        chain.push("hf".to_owned());
        thresholds.push(0);
        backends.push(Backend::HuggingFace);
        if self.inner.config.policy == Policy::RequireAccelerated
            && fast_cpu.is_none()
            && gpu.is_none()
        {
            return Err(Error::new(
                ErrorCode::UncertifiedTokenizer,
                format!("family {family} has no certified accelerated route on this runtime"),
            ));
        }
        let postprocessor_adds_tokens = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|document| document.get("post_processor").cloned())
            .is_some_and(|value| !value.is_null());
        let repair_fast_cpu = fast_cpu.is_some();
        let router = Arc::new(
            NativeRouter::new(
                chain,
                thresholds,
                Arc::clone(&reference),
                fast_cpu.clone(),
                repair_fast_cpu,
                gpu,
                postprocessor_adds_tokens,
                self.inner.config.diagnostics,
            )
            .map_err(|error| Error::new(ErrorCode::ConfigInvalid, error.to_string()))?,
        );
        let fingerprint = semantic_fingerprint(
            artifact.identity().tokenizer_sha256.as_bytes(),
            repair_fast_cpu,
        );
        let literal_guard = reference
            .added_tokens()
            .iter()
            .map(|(_, content, _)| content.chars().count())
            .max()
            .unwrap_or(0);
        let fast_guard = fast_cpu
            .as_ref()
            .map(|engine| engine.minimum_seal_tail_chars())
            .unwrap_or(0);
        let seal_guard = u64::try_from(literal_guard.max(fast_guard))
            .map_err(|_| Error::new(ErrorCode::ConfigInvalid, "session seal guard exceeds u64"))?;
        let store_path = if self.inner.config.persistent_sessions {
            let home = self
                .inner
                .config
                .home
                .as_ref()
                .expect("builder validated persistent home");
            Some(home.join("sessions").join(format!("{family}.sqlite3")))
        } else {
            None
        };
        let store = SessionStoreState::new(
            Arc::clone(&router),
            fingerprint,
            seal_guard,
            store_path.as_deref(),
            self.inner.config.seed_digest_overlap,
        )?;
        let plan = RoutePlan {
            family: family.clone(),
            artifact_sha256: artifact.identity().tokenizer_sha256.clone(),
            backends,
            gpu_min_bytes: self.inner.config.gpu_min_bytes,
            certification: if self.inner.config.policy == Policy::Experimental {
                Certification::Experimental
            } else if self.inner.config.policy == Policy::Reference {
                Certification::Reference
            } else if gpu_detail.is_some() {
                Certification::Certified
            } else if fast_cpu.is_some() {
                Certification::CertifiedSource
            } else {
                Certification::Reference
            },
            reasons,
        };
        Ok(Tokenizer {
            inner: Arc::new(TokenizerInner {
                artifact,
                reference,
                router,
                plan,
                store: std::sync::Mutex::new(store),
                gpu_detail,
                #[cfg(feature = "jit")]
                jit_detail,
            }),
        })
    }

    fn build_fast_cpu(
        &self,
        artifact: &LocalArtifact,
        reference: Arc<ReferenceEngine>,
        reasons: &mut Vec<ReasonCode>,
    ) -> Result<Option<Arc<FastCpuEngine>>> {
        let family = &artifact.identity().family;
        let Some(repair) = self.inner.registry.repair(family) else {
            reasons.push(ReasonCode::CpuUncertified);
            return Ok(None);
        };
        let support = self.inner.registry.support(family).ok_or_else(|| {
            Error::new(
                ErrorCode::RegistryInvalid,
                format!("repair family {family} has no support-registry row"),
            )
        })?;
        let fast_support = support.backends.get("fast_cpu");
        let fast_status = fast_support
            .and_then(|value| value.get("status"))
            .and_then(serde_json::Value::as_str);
        let fast_source = fast_support
            .and_then(|value| value.get("source_digest"))
            .and_then(serde_json::Value::as_str);
        let fast_toolchain = fast_support
            .and_then(|value| value.get("toolchain"))
            .and_then(serde_json::Value::as_str);
        if !matches!(fast_status, Some("certified") | Some("certified_source"))
            || support.artifact_sha256 != artifact.identity().tokenizer_sha256
            || repair.artifact_sha256 != artifact.identity().tokenizer_sha256
            || fast_source != Some(env!("TOKTIER_RUST_API_FAST_CPU_SOURCE_SHA256"))
            || fast_toolchain != Some(env!("TOKTIER_RUST_API_TOOLCHAIN"))
        {
            reasons.push(ReasonCode::CpuUncertified);
            return Ok(None);
        }
        let spec = FastRepairSpec::new(
            family.clone(),
            repair.artifact_sha256.clone(),
            repair.margin,
            repair.effective_l_max,
            repair.has_normalizer,
        );
        let engine = FastCpuEngine::from_reference(
            &artifact.bytes(),
            reference,
            spec,
            self.inner.registry.pclass(),
        )
        .map_err(|error| {
            Error::new(
                ErrorCode::BackendExecutionFault,
                format!("certified corrected-CPU construction failed: {error}"),
            )
        })?;
        Ok(Some(Arc::new(engine)))
    }
}

#[cfg(feature = "prebuilt-gpu")]
fn cuda_facts(ordinal: u32) -> CudaFacts {
    match i32::try_from(ordinal) {
        Ok(ordinal_i32) => match toktier_cuda_driver::CudaContext::new(ordinal_i32) {
            Ok(context) => {
                let (major, minor) = context.architecture();
                CudaFacts {
                    device_ordinal: ordinal,
                    available: true,
                    architecture: Some(format!("sm_{major}{minor}")),
                    driver_api_version: Some(context.driver_version()),
                    error: None,
                }
            }
            Err(error) => CudaFacts {
                device_ordinal: ordinal,
                available: false,
                architecture: None,
                driver_api_version: None,
                error: Some(error.to_string()),
            },
        },
        Err(_) => CudaFacts {
            device_ordinal: ordinal,
            available: false,
            architecture: None,
            driver_api_version: None,
            error: Some("CUDA ordinal exceeds i32".to_owned()),
        },
    }
}

#[cfg(not(feature = "prebuilt-gpu"))]
fn cuda_facts(ordinal: u32) -> CudaFacts {
    CudaFacts {
        device_ordinal: ordinal,
        available: false,
        architecture: None,
        driver_api_version: None,
        error: Some("CUDA support was not compiled into this feature set".to_owned()),
    }
}

/// A verified tokenizer and immutable route. Clone is cheap; sessions retain
/// their own single-writer handles.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub(crate) inner: Arc<TokenizerInner>,
}

impl Tokenizer {
    pub fn family(&self) -> &str {
        &self.inner.artifact.identity().family
    }

    pub fn artifact(&self) -> &LocalArtifact {
        &self.inner.artifact
    }

    pub fn plan(&self) -> &RoutePlan {
        &self.inner.plan
    }

    /// Snapshot native store counters without exposing implementation-crate
    /// types or locks.
    pub fn store_stats(&self) -> StoreStats {
        self.inner
            .store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .store
            .stats()
            .into()
    }

    pub fn gpu_facts(&self) -> Option<(&str, i32)> {
        self.inner
            .gpu_detail
            .as_ref()
            .map(|(architecture, version)| (architecture.as_str(), *version))
    }

    /// Exact direct-JIT compiler/cache binding, when that delivery was loaded.
    #[cfg(feature = "jit")]
    pub fn jit_facts(&self) -> Option<&crate::JitArtifact> {
        self.inner.jit_detail.as_ref()
    }

    pub fn encode(&self, text: &str) -> Result<Encoding> {
        self.encode_with(text, EncodeOptions::default())
    }

    pub fn encode_with(&self, text: &str, options: EncodeOptions) -> Result<Encoding> {
        if options.offsets && options.add_special_tokens {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "offsets with post-processor special tokens are not exposed by Rust API v1",
            ));
        }
        let routed = self
            .inner
            .router
            .encode_ids(text, options.add_special_tokens)
            .map_err(|error| Error::new(ErrorCode::BackendExecutionFault, error.to_string()))?;
        // Explicitly requested offsets are reconstructed through the
        // one-pass structure-of-arrays bridge and materialized as the
        // public pair layout; values and failure behavior match the
        // retained pair bridge exactly.
        let offsets = options
            .offsets
            .then(|| {
                self.inner
                    .reference
                    .spans_soa_for_ids(text, &routed.ids)
                    .map(|(starts, ends)| starts.into_iter().zip(ends).collect::<Vec<_>>())
            })
            .transpose()
            .map_err(|error| Error::new(ErrorCode::BackendExecutionFault, error.to_string()))?;
        let facts = facts_from_routed(
            &routed.backend,
            routed.path.as_deref(),
            routed.source,
            text.len(),
            self.inner.plan.certification,
        );
        Ok(Encoding::new(routed.ids, offsets, facts))
    }

    pub fn encode_batch<S: AsRef<str>>(&self, documents: &[S]) -> Result<RaggedEncoding> {
        self.encode_batch_with(documents, EncodeOptions::default())
    }

    pub fn encode_batch_with<S: AsRef<str>>(
        &self,
        documents: &[S],
        options: EncodeOptions,
    ) -> Result<RaggedEncoding> {
        if options.offsets {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "batch offsets are not part of the ragged ID-only API",
            ));
        }
        let texts = documents.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        let routed = self
            .inner
            .router
            .encode_batch_ids(&texts, options.add_special_tokens)
            .map_err(|error| Error::new(ErrorCode::BackendExecutionFault, error.to_string()))?;
        let rows = routed
            .into_iter()
            .zip(texts)
            .map(|(row, text)| {
                let facts = facts_from_routed(
                    &row.backend,
                    row.path.as_deref(),
                    row.source,
                    text.len(),
                    self.inner.plan.certification,
                );
                (row.ids, facts)
            })
            .collect();
        RaggedEncoding::from_rows(rows)
    }

    pub fn decode(&self, ids: impl AsRef<[u32]>, options: DecodeOptions) -> Result<String> {
        self.inner
            .reference
            .decode(ids.as_ref(), options.skip_special_tokens)
            .map_err(|error| Error::new(ErrorCode::BackendExecutionFault, error.to_string()))
    }

    pub fn open_session(&self, name: impl Into<String>) -> Result<Session> {
        Session::open(Arc::clone(&self.inner), name.into())
    }

    /// Content-prefix lookup without Python. A hit returns its exact current
    /// token stream; a miss does not tokenize or mutate the store.
    pub fn lookup(&self, text: &str) -> Result<Option<Encoding>> {
        let mut state = self
            .inner
            .store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.ensure_healthy()?;
        let key = state.key;
        let Some(hit) = state.store.lookup(key, text, self.inner.router.as_ref())? else {
            return Ok(None);
        };
        let matched_chars = usize::try_from(hit.matched_chars).map_err(|_| {
            Error::new(
                ErrorCode::Internal,
                "content-prefix length does not fit this platform",
            )
        })?;
        let total_chars = text.chars().count();
        if matched_chars > total_chars {
            return Err(Error::new(
                ErrorCode::StoreCorrupt,
                "content-prefix lookup returned a length beyond the query",
            ));
        }
        let suffix_byte = if matched_chars == total_chars {
            text.len()
        } else {
            text.char_indices()
                .nth(matched_chars)
                .map(|(offset, _)| offset)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::StoreCorrupt,
                        "content-prefix lookup returned a non-existent character boundary",
                    )
                })?
        };
        if suffix_byte < text.len() {
            state.store.append_patch(
                hit.handle,
                &text[suffix_byte..],
                hit.revision,
                self.inner.router.as_ref(),
            )?;
        }
        let ids = state.store.shared_all_ids(hit.handle)?;
        // Lookup materializes a temporary session only to complete the
        // verified prefix. The returned Encoding shares the immutable row
        // (which outlives the evicted session), so retaining the handle
        // would only consume the bounded named-session capacity.
        state.store.evict(hit.handle);
        let facts = ExecutionFacts {
            backend: Backend::FastCpu,
            path: "content_lookup".to_owned(),
            source: Some("native_store".to_owned()),
            input_bytes: text.len() as u64,
            certification: self.inner.plan.certification,
        };
        Ok(Some(Encoding::from_buffer(
            TokenBuffer::from_shared(ids),
            facts,
        )))
    }

    /// Deterministically release this handle. Shared clones and outstanding
    /// sessions continue to own the runtime until they are dropped.
    pub fn shutdown(self) {}
}

pub(crate) fn facts_from_routed(
    backend: &str,
    path: Option<&str>,
    source: Option<String>,
    input_bytes: usize,
    certification: Certification,
) -> ExecutionFacts {
    ExecutionFacts {
        backend: Backend::from_internal(backend),
        path: path.unwrap_or("unknown").to_owned(),
        source,
        input_bytes: input_bytes as u64,
        certification,
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

#[allow(dead_code)]
fn _assert_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Runtime>();
    assert_send_sync::<Tokenizer>();
    assert_send_sync::<TokenBuffer>();
}
