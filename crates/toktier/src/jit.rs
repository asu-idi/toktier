//! Direct, shell-free NVCC compilation into the native CUDA Driver host.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use toktier_cuda_driver::CudaContext;

use crate::fsutil::{hex, monotonic_nonce, open_private_lock, set_private_file, sync_directory};
use crate::manifest::{domain_sha256_hex, sha256_hex, Registry};
use crate::{Error, ErrorCode, Result};

const UNIT_NAME: &str = "prebuilt_unit.cu";
const KERNEL_NAME: &str = "pretok_kernel.cu";
const UNIT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/prebuilt_unit.cu"
));
const KERNEL_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/pretok_kernel.cu"
));
const SOURCE_DOMAIN: &[u8] = b"toktier.rust_jit_source.v1\0";
const BINDING_DOMAIN: &[u8] = b"toktier.rust_jit_binding.v2\0";
const FATBIN_DOMAIN: &[u8] = b"toktier.kernel_fatbin.v1\0";
const MANIFEST_SCHEMA: &str = "toktier.rust_jit_binding.v2";
const FROZEN_ORACLE_VERSION: &str = "0.22.2";
const NORMALIZED_FLAGS: &[&str] = &[
    "-fatbin",
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-DTOKTIER_DEVICE_ONLY",
];

/// Non-invasive identity of the selected direct CUDA compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JitToolchainFacts {
    pub selected_path: PathBuf,
    pub resolved_path: PathBuf,
    pub release: String,
    pub build: String,
    pub compiler_sha256: String,
    pub world_writable_component: Option<PathBuf>,
}

/// A verified cached compiler product, without exposing mutable cache bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JitArtifact {
    pub family: String,
    pub artifact_sha256: String,
    pub oracle_id: String,
    pub oracle_version: String,
    pub artifact_evidence_id: String,
    pub jit_evidence_id: Option<String>,
    pub architecture: String,
    pub product_path: PathBuf,
    pub product_sha256: String,
    pub domain_digest: String,
    pub binding_digest: String,
    pub certified: bool,
    pub cache_hit: bool,
    pub compiler: JitToolchainFacts,
    pub warning: Option<String>,
}

/// Configuration for the direct compiler and its private content cache.
#[derive(Debug, Clone)]
pub struct JitCompilerBuilder {
    cache: PathBuf,
    nvcc: Option<PathBuf>,
    timeout: Duration,
    max_output_bytes: usize,
    max_product_bytes: u64,
    lock_timeout: Duration,
    accept_uncertified: bool,
}

impl Default for JitCompilerBuilder {
    fn default() -> Self {
        let cache = crate::fsutil::default_cache_directory("TOKTIER_JIT_CACHE", "jit-rust");
        Self {
            cache,
            nvcc: None,
            timeout: Duration::from_secs(10 * 60),
            max_output_bytes: 1024 * 1024,
            max_product_bytes: 128 * 1024 * 1024,
            lock_timeout: Duration::from_secs(60),
            accept_uncertified: false,
        }
    }
}

impl JitCompilerBuilder {
    pub fn cache(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache = path.into();
        self
    }

    pub fn nvcc(mut self, path: impl Into<PathBuf>) -> Self {
        self.nvcc = Some(path.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn max_output_bytes(mut self, bytes: usize) -> Self {
        self.max_output_bytes = bytes;
        self
    }

    pub fn max_product_bytes(mut self, bytes: u64) -> Self {
        self.max_product_bytes = bytes;
        self
    }

    pub fn lock_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    /// Deliberately loud, per-object escape hatch. This permission is never
    /// serialized into a global config and every product remains labelled.
    pub fn accept_uncertified_jit(mut self, accept: bool) -> Self {
        self.accept_uncertified = accept;
        self
    }

    pub fn build(self) -> Result<JitCompiler> {
        if self.cache.as_os_str().is_empty()
            || self.timeout.is_zero()
            || self.max_output_bytes == 0
            || self.max_product_bytes == 0
            || self.lock_timeout.is_zero()
        {
            return Err(Error::new(
                ErrorCode::ConfigInvalid,
                "JIT cache and resource bounds must be non-empty/non-zero",
            ));
        }
        Ok(JitCompiler {
            inner: Arc::new(JitCompilerConfig {
                cache: self.cache,
                nvcc: self.nvcc,
                timeout: self.timeout,
                max_output_bytes: self.max_output_bytes,
                max_product_bytes: self.max_product_bytes,
                lock_timeout: self.lock_timeout,
                accept_uncertified: self.accept_uncertified,
            }),
        })
    }
}

#[derive(Debug)]
struct JitCompilerConfig {
    cache: PathBuf,
    nvcc: Option<PathBuf>,
    timeout: Duration,
    max_output_bytes: usize,
    max_product_bytes: u64,
    lock_timeout: Duration,
    accept_uncertified: bool,
}

/// Shareable direct-NVCC compiler. No method invokes Python, PyTorch, a shell,
/// `ninja`, or `torch.utils.cpp_extension`.
#[derive(Debug, Clone)]
pub struct JitCompiler {
    inner: Arc<JitCompilerConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BindingManifest {
    schema: String,
    binding_digest: String,
    source_digest: String,
    compiler_path: String,
    compiler_sha256: String,
    compiler_release: String,
    compiler_build: String,
    normalized_arguments: Vec<String>,
    architecture: String,
    driver_api_version: i32,
    rust_host_source_digest: String,
    rust_toolchain: String,
    rust_build_flags: Vec<String>,
    family: String,
    artifact_sha256: String,
    oracle_id: String,
    oracle_version: String,
    artifact_evidence_id: String,
    jit_evidence_id: Option<String>,
    class_table_digest: String,
    product_file: String,
    product_sha256: String,
    product_domain_digest: String,
    product_size: u64,
    certified: bool,
    experimental_waiver: Option<String>,
}

pub(crate) struct JitProduct {
    pub(crate) image: Arc<[u8]>,
    pub(crate) architecture: String,
    pub(crate) compute_major: i32,
    pub(crate) compute_minor: i32,
    pub(crate) driver_api_version: i32,
    pub(crate) domain_digest: String,
    pub(crate) certified: bool,
    pub(crate) facts: JitArtifact,
}

impl JitCompiler {
    pub fn builder() -> JitCompilerBuilder {
        JitCompilerBuilder::default()
    }

    pub fn cache_root(&self) -> &Path {
        &self.inner.cache
    }

    pub fn accepts_uncertified_jit(&self) -> bool {
        self.inner.accept_uncertified
    }

    /// Resolve and hash NVCC without compiling or creating a cache path.
    pub fn toolchain(&self) -> Result<JitToolchainFacts> {
        probe_toolchain(
            self.inner.nvcc.as_deref(),
            self.inner.timeout.min(Duration::from_secs(5)),
        )
    }

    /// Compile or verify one family/device binding and return public facts.
    pub fn compile(&self, family: &str, device_ordinal: u32) -> Result<JitArtifact> {
        let registry = Registry::load()?;
        let ordinal = i32::try_from(device_ordinal)
            .map_err(|_| Error::new(ErrorCode::InvalidArgument, "CUDA ordinal exceeds i32"))?;
        Ok(self.compile_product(family, &registry, ordinal)?.facts)
    }

    pub(crate) fn compile_product(
        &self,
        family: &str,
        registry: &Registry,
        device_ordinal: i32,
    ) -> Result<JitProduct> {
        let context = CudaContext::new(device_ordinal)
            .map_err(|error| Error::new(ErrorCode::KernelIncompatible, error.to_string()))?;
        let (major, minor) = context.architecture();
        let architecture = format!("sm_{major}{minor}");
        let driver_api_version = context.driver_version();
        drop(context);

        let compiler = self.toolchain()?;
        let source_digest = source_digest();
        let support = registry.support(family).ok_or_else(|| {
            Error::new(
                ErrorCode::UncertifiedTokenizer,
                format!("family {family:?} has no certification row"),
            )
        })?;
        let jit = support
            .backends
            .get("gpu")
            .and_then(|gpu| gpu.get("deliveries"))
            .and_then(|deliveries| deliveries.get("jit"))
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::KernelIncompatible,
                    format!("family {family:?} has no JIT delivery"),
                )
            })?;
        let class_table_digest = jit
            .get("class_table_digest")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();
        let jit_evidence_id = direct_evidence_id(
            jit,
            &source_digest,
            &compiler,
            &architecture,
            env!("TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256"),
        )
        .map(str::to_owned);
        let unsafe_component = compiler.world_writable_component.is_some();
        let certified = jit_evidence_id.is_some() && !unsafe_component;
        let warning = (!certified).then(|| {
            format!(
                "UNCERTIFIED JIT OPT-IN: direct NVCC {} {} for {architecture} is not an admitted TokTier binding; outputs remain outside the certified exact-ID guarantee",
                compiler.release, compiler.build
            )
        });
        if !certified && !self.inner.accept_uncertified {
            return Err(Error::new(
                ErrorCode::UncertifiedJit,
                format!(
                    "direct JIT is fail-closed for NVCC {} {} on {architecture}; no cache was created. Explicitly construct JitCompiler::builder().accept_uncertified_jit(true), or run `toktier gpu compile {family} --accept-uncertified-jit` for an experimental one-shot build",
                    compiler.release, compiler.build
                ),
            )
            .with_family(family));
        }
        if let Some(warning) = &warning {
            eprintln!("WARNING: {warning}");
        }

        let normalized_arguments = normalized_arguments(&architecture);
        let compiler_path = compiler.resolved_path.to_string_lossy().into_owned();
        let binding = BindingInput {
            source_digest: &source_digest,
            compiler_path: &compiler_path,
            compiler_sha256: &compiler.compiler_sha256,
            compiler_release: &compiler.release,
            compiler_build: &compiler.build,
            normalized_arguments: &normalized_arguments,
            architecture: &architecture,
            driver_api_version,
            rust_host_source_digest: env!("TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256"),
            rust_toolchain: env!("TOKTIER_RUST_API_TOOLCHAIN"),
            rust_build_flags: env!("TOKTIER_RUST_API_BUILD_FLAGS"),
            family,
            artifact_sha256: &support.artifact_sha256,
            oracle_id: &support.oracle_id,
            oracle_version: FROZEN_ORACLE_VERSION,
            artifact_evidence_id: &support.evidence_id,
            jit_evidence_id: jit_evidence_id.as_deref(),
            class_table_digest: &class_table_digest,
            certified,
        };
        let binding_digest = binding_digest(&binding)?;

        ensure_private_directory(&self.inner.cache)?;
        let locks = self.inner.cache.join(".locks");
        ensure_private_directory(&locks)?;
        let lock_path = locks.join(format!("{}.lock", &binding_digest[..32]));
        let lock = open_private_lock(&lock_path)?;
        lock_exclusive_bounded(&lock, self.inner.lock_timeout)?;
        let directory = self.inner.cache.join(&binding_digest[..32]);
        if let Ok(product) = self.read_cached(
            family,
            &directory,
            &binding_digest,
            &architecture,
            major,
            minor,
            driver_api_version,
            &compiler,
            warning.clone(),
        ) {
            return Ok(product);
        }
        if directory.exists() {
            let quarantine = self.inner.cache.join(".quarantine");
            ensure_private_directory(&quarantine)?;
            let target = quarantine.join(format!(
                "{}-{}-{}",
                &binding_digest[..16],
                std::process::id(),
                monotonic_nonce()
            ));
            fs::rename(&directory, target)
                .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
        }

        let staging = self.inner.cache.join(format!(
            ".build-{}-{}-{}",
            &binding_digest[..16],
            std::process::id(),
            monotonic_nonce()
        ));
        fs::create_dir(&staging)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&staging))?;
        ensure_private_directory(&staging)?;
        let result = self.build_uncached(
            family,
            &staging,
            &directory,
            &binding_digest,
            &source_digest,
            &architecture,
            major,
            minor,
            driver_api_version,
            &compiler,
            normalized_arguments,
            support.artifact_sha256.clone(),
            support.oracle_id.clone(),
            support.evidence_id.clone(),
            jit_evidence_id,
            class_table_digest,
            certified,
            warning,
        );
        let _ = fs::remove_dir_all(&staging);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn build_uncached(
        &self,
        family: &str,
        staging: &Path,
        directory: &Path,
        binding_digest: &str,
        source_digest: &str,
        architecture: &str,
        major: i32,
        minor: i32,
        driver_api_version: i32,
        compiler: &JitToolchainFacts,
        normalized_arguments: Vec<String>,
        artifact_sha256: String,
        oracle_id: String,
        artifact_evidence_id: String,
        jit_evidence_id: Option<String>,
        class_table_digest: String,
        certified: bool,
        warning: Option<String>,
    ) -> Result<JitProduct> {
        let unit = staging.join(UNIT_NAME);
        let kernel = staging.join(KERNEL_NAME);
        write_private(&unit, UNIT_BYTES)?;
        write_private(&kernel, KERNEL_BYTES)?;
        let output = staging.join("kernel.fatbin");
        let architecture_number = architecture.strip_prefix("sm_").ok_or_else(|| {
            Error::new(
                ErrorCode::JitCompileFailed,
                "invalid CUDA architecture label",
            )
        })?;
        let mut command = Command::new(&compiler.resolved_path);
        // NORMALIZED_FLAGS is the single source of the frozen flag set:
        // the same constant feeds the recorded binding arguments and their
        // cache-hit comparison, so the command cannot drift from what the
        // binding manifest states.
        command
            .args(NORMALIZED_FLAGS)
            .arg(format!(
                "-gencode=arch=compute_{architecture_number},code=sm_{architecture_number}"
            ))
            .arg(UNIT_NAME)
            .arg("-o")
            .arg("kernel.fatbin")
            .current_dir(staging)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LANG", "C")
            .env("LC_ALL", "C")
            .env("TMPDIR", staging)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let process = run_bounded(command, self.inner.timeout, self.inner.max_output_bytes)?;
        if !process.status.success() {
            return Err(Error::new(
                ErrorCode::JitCompileFailed,
                format!(
                    "NVCC exited with {} (output{}): {}",
                    process.status,
                    if process.truncated { " truncated" } else { "" },
                    String::from_utf8_lossy(&process.output)
                ),
            ));
        }
        let metadata = fs::symlink_metadata(&output).map_err(|error| {
            Error::new(ErrorCode::JitCompileFailed, error.to_string()).with_path(&output)
        })?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.len() == 0
            || metadata.len() > self.inner.max_product_bytes
        {
            return Err(Error::new(
                ErrorCode::JitCompileFailed,
                "NVCC product is absent, special, empty, or exceeds the configured bound",
            )
            .with_path(output));
        }
        let image = read_bounded(&output, self.inner.max_product_bytes)?;
        let product_sha256 = sha256_hex(&image);
        let product_domain_digest = domain_sha256_hex(FATBIN_DOMAIN, &image);
        let manifest = BindingManifest {
            schema: MANIFEST_SCHEMA.to_owned(),
            binding_digest: binding_digest.to_owned(),
            source_digest: source_digest.to_owned(),
            compiler_path: compiler.resolved_path.display().to_string(),
            compiler_sha256: compiler.compiler_sha256.clone(),
            compiler_release: compiler.release.clone(),
            compiler_build: compiler.build.clone(),
            normalized_arguments,
            architecture: architecture.to_owned(),
            driver_api_version,
            rust_host_source_digest: env!("TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256").to_owned(),
            rust_toolchain: env!("TOKTIER_RUST_API_TOOLCHAIN").to_owned(),
            rust_build_flags: env!("TOKTIER_RUST_API_BUILD_FLAGS")
                .split('\x1f')
                .map(str::to_owned)
                .collect(),
            family: family.to_owned(),
            artifact_sha256,
            oracle_id,
            oracle_version: FROZEN_ORACLE_VERSION.to_owned(),
            artifact_evidence_id,
            jit_evidence_id,
            class_table_digest,
            product_file: "kernel.fatbin".to_owned(),
            product_sha256: product_sha256.clone(),
            product_domain_digest: product_domain_digest.clone(),
            product_size: image.len() as u64,
            certified,
            experimental_waiver: warning.clone(),
        };
        let manifest_path = staging.join("binding.json");
        let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?;
        manifest_bytes.push(b'\n');
        write_private(&manifest_path, &manifest_bytes)?;
        // Re-open both products before the directory itself becomes visible.
        verify_manifest(&manifest, &manifest_path, &output, binding_digest)?;
        sync_directory(staging)?;
        fs::rename(staging, directory)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(directory))?;
        sync_directory(&self.inner.cache)?;
        let mut product = self.read_cached(
            family,
            directory,
            binding_digest,
            architecture,
            major,
            minor,
            driver_api_version,
            compiler,
            warning,
        )?;
        product.facts.cache_hit = false;
        Ok(product)
    }

    #[allow(clippy::too_many_arguments)]
    fn read_cached(
        &self,
        family: &str,
        directory: &Path,
        binding_digest: &str,
        architecture: &str,
        major: i32,
        minor: i32,
        driver_api_version: i32,
        compiler: &JitToolchainFacts,
        warning: Option<String>,
    ) -> Result<JitProduct> {
        let manifest_path = directory.join("binding.json");
        let product_path = directory.join("kernel.fatbin");
        let manifest_bytes = read_bounded(&manifest_path, 4 * 1024 * 1024)?;
        let manifest: BindingManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|error| {
                Error::new(
                    ErrorCode::JitCompileFailed,
                    format!("cannot parse JIT binding manifest: {error}"),
                )
            })?;
        verify_manifest(&manifest, &manifest_path, &product_path, binding_digest)?;
        if manifest.family != family
            || manifest.architecture != architecture
            || manifest.driver_api_version != driver_api_version
            || manifest.compiler_sha256 != compiler.compiler_sha256
            || manifest.compiler_path != compiler.resolved_path.to_string_lossy()
        {
            return Err(Error::new(
                ErrorCode::JitCompileFailed,
                "cached JIT binding facts do not match the current request",
            ));
        }
        let image = read_bounded(&product_path, self.inner.max_product_bytes)?;
        Ok(JitProduct {
            image: image.into(),
            architecture: architecture.to_owned(),
            compute_major: major,
            compute_minor: minor,
            driver_api_version,
            domain_digest: manifest.product_domain_digest.clone(),
            certified: manifest.certified,
            facts: JitArtifact {
                family: family.to_owned(),
                artifact_sha256: manifest.artifact_sha256,
                oracle_id: manifest.oracle_id,
                oracle_version: manifest.oracle_version,
                artifact_evidence_id: manifest.artifact_evidence_id,
                jit_evidence_id: manifest.jit_evidence_id,
                architecture: architecture.to_owned(),
                product_path,
                product_sha256: manifest.product_sha256,
                domain_digest: manifest.product_domain_digest,
                binding_digest: binding_digest.to_owned(),
                certified: manifest.certified,
                cache_hit: true,
                compiler: compiler.clone(),
                warning,
            },
        })
    }
}

#[derive(Serialize)]
struct BindingInput<'a> {
    source_digest: &'a str,
    compiler_path: &'a str,
    compiler_sha256: &'a str,
    compiler_release: &'a str,
    compiler_build: &'a str,
    normalized_arguments: &'a [String],
    architecture: &'a str,
    driver_api_version: i32,
    rust_host_source_digest: &'a str,
    rust_toolchain: &'a str,
    rust_build_flags: &'a str,
    family: &'a str,
    artifact_sha256: &'a str,
    oracle_id: &'a str,
    oracle_version: &'a str,
    artifact_evidence_id: &'a str,
    jit_evidence_id: Option<&'a str>,
    class_table_digest: &'a str,
    certified: bool,
}

fn binding_digest(binding: &BindingInput<'_>) -> Result<String> {
    let bytes = serde_json::to_vec(binding)
        .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?;
    Ok(domain_sha256_hex(BINDING_DOMAIN, &bytes))
}

fn direct_evidence_id<'a>(
    jit: &'a serde_json::Map<String, serde_json::Value>,
    source_digest: &str,
    compiler: &JitToolchainFacts,
    architecture: &str,
    host_source_digest: &str,
) -> Option<&'a str> {
    let source_matches = jit
        .get("direct_source_digest")
        .and_then(serde_json::Value::as_str)
        == Some(source_digest);
    let host_matches = jit
        .get("direct_host_source_digest")
        .and_then(serde_json::Value::as_str)
        == Some(host_source_digest);
    let flags_match = jit
        .get("direct_build_flags")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|values| {
            values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .eq(NORMALIZED_FLAGS.iter().copied())
        });
    let device_matches = jit
        .get("direct_devices")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|values| {
            values
                .iter()
                .any(|value| value.as_str() == Some(architecture))
        });
    let evidence_id = jit
        .get("direct_toolchains")
        .and_then(serde_json::Value::as_array)
        .and_then(|values| {
            values.iter().find_map(|value| {
                let matches = value.get("release").and_then(serde_json::Value::as_str)
                    == Some(compiler.release.as_str())
                    && value.get("build").and_then(serde_json::Value::as_str)
                        == Some(compiler.build.as_str())
                    && value
                        .get("compiler_sha256")
                        .and_then(serde_json::Value::as_str)
                        == Some(compiler.compiler_sha256.as_str())
                    && value
                        .get("architecture")
                        .and_then(serde_json::Value::as_str)
                        == Some(architecture);
                matches
                    .then(|| value.get("evidence_id").and_then(serde_json::Value::as_str))
                    .flatten()
                    .filter(|value| !value.is_empty())
            })
        });
    (source_matches && host_matches && flags_match && device_matches)
        .then_some(evidence_id)
        .flatten()
}

fn normalized_arguments(architecture: &str) -> Vec<String> {
    let number = architecture.trim_start_matches("sm_");
    let mut arguments = NORMALIZED_FLAGS
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    arguments.extend([
        format!("-gencode=arch=compute_{number},code=sm_{number}"),
        format!("<source>/{UNIT_NAME}"),
        "-o".to_owned(),
        "<output>/kernel.fatbin".to_owned(),
    ]);
    arguments
}

fn source_digest() -> String {
    let mut hasher = Sha256::new();
    hasher.update(SOURCE_DOMAIN);
    for (name, bytes) in [(UNIT_NAME, UNIT_BYTES), (KERNEL_NAME, KERNEL_BYTES)] {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    hex(&hasher.finalize())
}

fn probe_toolchain(explicit: Option<&Path>, timeout: Duration) -> Result<JitToolchainFacts> {
    let selected = match explicit {
        Some(path) => path.to_path_buf(),
        None => locate_nvcc()?,
    };
    let selected_metadata = fs::symlink_metadata(&selected).map_err(|error| {
        Error::new(ErrorCode::JitCompileFailed, error.to_string()).with_path(&selected)
    })?;
    if !selected_metadata.file_type().is_file() && !selected_metadata.file_type().is_symlink() {
        return Err(Error::new(
            ErrorCode::JitCompileFailed,
            "selected NVCC path is not a regular file or symlink",
        )
        .with_path(selected));
    }
    let resolved = fs::canonicalize(&selected).map_err(|error| {
        Error::new(ErrorCode::JitCompileFailed, error.to_string()).with_path(&selected)
    })?;
    let metadata = fs::metadata(&resolved).map_err(|error| {
        Error::new(ErrorCode::JitCompileFailed, error.to_string()).with_path(&resolved)
    })?;
    if !metadata.is_file() {
        return Err(
            Error::new(ErrorCode::JitCompileFailed, "resolved NVCC is not a file")
                .with_path(resolved),
        );
    }
    let world_writable_component = first_world_writable(&resolved)?;
    let compiler_bytes = read_bounded(&resolved, 1024 * 1024 * 1024)?;
    let mut command = Command::new(&resolved);
    command
        .arg("--version")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = run_bounded(command, timeout, 64 * 1024)?;
    if !output.status.success() {
        return Err(Error::new(
            ErrorCode::JitCompileFailed,
            format!("nvcc --version failed with {}", output.status),
        ));
    }
    let text = String::from_utf8_lossy(&output.output);
    let (release, build) = parse_nvcc_version(&text)?;
    Ok(JitToolchainFacts {
        selected_path: selected,
        resolved_path: resolved,
        release,
        build,
        compiler_sha256: sha256_hex(&compiler_bytes),
        world_writable_component,
    })
}

fn locate_nvcc() -> Result<PathBuf> {
    for variable in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            let candidate = PathBuf::from(root).join("bin/nvcc");
            if candidate.is_file() {
                return Ok(candidate);
            }
            return Err(Error::new(
                ErrorCode::JitCompileFailed,
                format!("{variable} is set but {candidate:?} is unavailable"),
            ));
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for root in std::env::split_paths(&paths) {
            let candidate = root.join("nvcc");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    let conventional = PathBuf::from("/usr/local/cuda/bin/nvcc");
    if conventional.is_file() {
        Ok(conventional)
    } else {
        Err(Error::new(
            ErrorCode::JitCompileFailed,
            "nvcc was not found through CUDA_HOME, CUDA_PATH, PATH, or /usr/local/cuda/bin/nvcc",
        ))
    }
}

fn parse_nvcc_version(output: &str) -> Result<(String, String)> {
    let release_marker = "release ";
    let start = output
        .find(release_marker)
        .map(|index| index + release_marker.len())
        .ok_or_else(|| {
            Error::new(
                ErrorCode::JitCompileFailed,
                "unrecognized nvcc version output",
            )
        })?;
    let rest = &output[start..];
    let (release, rest) = rest
        .split_once(',')
        .ok_or_else(|| Error::new(ErrorCode::JitCompileFailed, "unrecognized nvcc release"))?;
    let build = rest
        .split_whitespace()
        .find(|word| word.starts_with('V'))
        .ok_or_else(|| Error::new(ErrorCode::JitCompileFailed, "unrecognized nvcc build"))?;
    if !release
        .chars()
        .all(|character| character.is_ascii_digit() || character == '.')
        || !build
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '.')
    {
        return Err(Error::new(
            ErrorCode::JitCompileFailed,
            "nvcc version contains unexpected characters",
        ));
    }
    Ok((release.to_owned(), build.to_owned()))
}

#[derive(Debug)]
struct BoundedOutput {
    status: std::process::ExitStatus,
    output: Vec<u8>,
    truncated: bool,
}

fn run_bounded(mut command: Command, timeout: Duration, limit: usize) -> Result<BoundedOutput> {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        // Isolate NVCC and every compiler helper it may spawn. A wall-time
        // bound must also close descendant-held output pipes, otherwise the
        // reader threads could outlive the process we timed.
        command.process_group(0);
    }
    let mut child = command
        .spawn()
        .map_err(|error| Error::new(ErrorCode::JitCompileFailed, error.to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::new(ErrorCode::Internal, "compiler stdout was not piped"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| Error::new(ErrorCode::Internal, "compiler stderr was not piped"))?;
    let out_thread = std::thread::spawn(move || read_capped(stdout, limit));
    let err_thread = std::thread::spawn(move || read_capped(stderr, limit));
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child
            .try_wait()
            .map_err(|error| Error::new(ErrorCode::JitCompileFailed, error.to_string()))?
        {
            Some(status) => break status,
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            None => {
                #[cfg(unix)]
                kill_process_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
                let _ = out_thread.join();
                let _ = err_thread.join();
                return Err(Error::new(
                    ErrorCode::JitCompileFailed,
                    format!(
                        "NVCC exceeded the {} second wall-time limit",
                        timeout.as_secs_f64()
                    ),
                ));
            }
        }
    };
    let (mut stdout, stdout_truncated) = out_thread
        .join()
        .map_err(|_| Error::new(ErrorCode::Internal, "compiler stdout reader panicked"))?;
    let (stderr, stderr_truncated) = err_thread
        .join()
        .map_err(|_| Error::new(ErrorCode::Internal, "compiler stderr reader panicked"))?;
    if !stdout.is_empty() && !stderr.is_empty() {
        stdout.push(b'\n');
    }
    let remaining = limit.saturating_sub(stdout.len());
    stdout.extend_from_slice(&stderr[..stderr.len().min(remaining)]);
    Ok(BoundedOutput {
        status,
        output: stdout,
        truncated: stdout_truncated || stderr_truncated || stderr.len() > remaining,
    })
}

#[cfg(unix)]
fn kill_process_group(pid: u32) {
    let group = format!("-{pid}");
    for executable in ["/usr/bin/kill", "/bin/kill"] {
        if !Path::new(executable).is_file() {
            continue;
        }
        let status = Command::new(executable)
            .args(["-KILL", "--", &group])
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if status.is_ok_and(|status| status.success()) {
            return;
        }
    }
}

fn read_capped(mut reader: impl Read, limit: usize) -> (Vec<u8>, bool) {
    let mut kept = Vec::with_capacity(limit.min(64 * 1024));
    let mut buffer = [0u8; 8192];
    let mut truncated = false;
    loop {
        match reader.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(count) => {
                let room = limit.saturating_sub(kept.len());
                kept.extend_from_slice(&buffer[..count.min(room)]);
                truncated |= count > room;
            }
        }
    }
    (kept, truncated)
}

fn verify_manifest(
    manifest: &BindingManifest,
    manifest_path: &Path,
    product_path: &Path,
    expected_binding_digest: &str,
) -> Result<()> {
    let manifest_metadata = fs::symlink_metadata(manifest_path).map_err(|error| {
        Error::new(ErrorCode::JitCompileFailed, error.to_string()).with_path(manifest_path)
    })?;
    let product_metadata = fs::symlink_metadata(product_path).map_err(|error| {
        Error::new(ErrorCode::JitCompileFailed, error.to_string()).with_path(product_path)
    })?;
    let joined_build_flags = manifest.rust_build_flags.join("\x1f");
    let recomputed_binding = binding_digest(&BindingInput {
        source_digest: &manifest.source_digest,
        compiler_path: &manifest.compiler_path,
        compiler_sha256: &manifest.compiler_sha256,
        compiler_release: &manifest.compiler_release,
        compiler_build: &manifest.compiler_build,
        normalized_arguments: &manifest.normalized_arguments,
        architecture: &manifest.architecture,
        driver_api_version: manifest.driver_api_version,
        rust_host_source_digest: &manifest.rust_host_source_digest,
        rust_toolchain: &manifest.rust_toolchain,
        rust_build_flags: &joined_build_flags,
        family: &manifest.family,
        artifact_sha256: &manifest.artifact_sha256,
        oracle_id: &manifest.oracle_id,
        oracle_version: &manifest.oracle_version,
        artifact_evidence_id: &manifest.artifact_evidence_id,
        jit_evidence_id: manifest.jit_evidence_id.as_deref(),
        class_table_digest: &manifest.class_table_digest,
        certified: manifest.certified,
    })?;
    if manifest_metadata.file_type().is_symlink()
        || product_metadata.file_type().is_symlink()
        || !manifest_metadata.is_file()
        || !product_metadata.is_file()
        || manifest.schema != MANIFEST_SCHEMA
        || manifest.binding_digest != expected_binding_digest
        || recomputed_binding != expected_binding_digest
        || manifest.product_file != "kernel.fatbin"
        || product_metadata.len() != manifest.product_size
    {
        return Err(Error::new(
            ErrorCode::JitCompileFailed,
            "JIT cache manifest or product shape is invalid",
        ));
    }
    let product = read_bounded(product_path, manifest.product_size.max(1))?;
    if sha256_hex(&product) != manifest.product_sha256
        || domain_sha256_hex(FATBIN_DOMAIN, &product) != manifest.product_domain_digest
    {
        return Err(Error::new(
            ErrorCode::JitCompileFailed,
            "JIT cache product digest mismatch",
        )
        .with_path(product_path));
    }
    Ok(())
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    set_private_file(&file)?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(Error::new(
            ErrorCode::JitCompileFailed,
            "file is special or exceeds the configured bound",
        )
        .with_path(path));
    }
    fs::read(path).map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))
}

fn ensure_private_directory(path: &Path) -> Result<()> {
    crate::fsutil::ensure_private_directory(
        path,
        "JIT cache",
        "JIT cache path is not a regular directory",
    )
}

fn lock_exclusive_bounded(file: &File, timeout: Duration) -> Result<()> {
    crate::fsutil::lock_exclusive_bounded(file, timeout, "JIT cache")
}

fn first_world_writable(path: &Path) -> Result<Option<PathBuf>> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for candidate in path.ancestors() {
            let metadata = fs::metadata(candidate).map_err(|error| {
                Error::new(ErrorCode::Io, error.to_string()).with_path(candidate)
            })?;
            if metadata.permissions().mode() & 0o002 != 0 {
                return Ok(Some(candidate.to_path_buf()));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parser_accepts_nvcc_shape() {
        let (release, build) =
            parse_nvcc_version("Cuda compilation tools, release 13.2, V13.2.78\nBuild cuda_13.2")
                .unwrap();
        assert_eq!(release, "13.2");
        assert_eq!(build, "V13.2.78");
    }

    #[test]
    fn source_identity_is_stable_and_nonempty() {
        assert_eq!(source_digest().len(), 64);
        assert_eq!(source_digest(), source_digest());
    }

    #[cfg(unix)]
    #[test]
    fn bounded_runner_kills_the_compiler_process_group() {
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "sleep 30 & wait"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let started = Instant::now();
        let error = run_bounded(command, Duration::from_millis(50), 1024).unwrap_err();
        assert_eq!(error.code(), ErrorCode::JitCompileFailed);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn cache_manifest_binds_every_identity_field_and_product_byte() {
        let temporary = tempfile::tempdir().unwrap();
        let product_path = temporary.path().join("kernel.fatbin");
        let manifest_path = temporary.path().join("binding.json");
        let product = b"fatbin bytes";
        fs::write(&product_path, product).unwrap();
        let arguments = normalized_arguments("sm_120");
        let flags = vec!["profile=release".to_owned()];
        let joined_flags = flags.join("\x1f");
        let source = source_digest();
        let compiler_sha = "1".repeat(64);
        let host_source = "2".repeat(64);
        let class_table = "3".repeat(64);
        let binding = BindingInput {
            source_digest: &source,
            compiler_path: "/opt/cuda/bin/nvcc",
            compiler_sha256: &compiler_sha,
            compiler_release: "13.2",
            compiler_build: "V13.2.78",
            normalized_arguments: &arguments,
            architecture: "sm_120",
            driver_api_version: 13020,
            rust_host_source_digest: &host_source,
            rust_toolchain: "rustc test",
            rust_build_flags: &joined_flags,
            family: "qwen3_8b",
            artifact_sha256: "4444444444444444444444444444444444444444444444444444444444444444",
            oracle_id: "tokenizers",
            oracle_version: FROZEN_ORACLE_VERSION,
            artifact_evidence_id: "ev-artifact-test",
            jit_evidence_id: Some("ev-direct-jit-test"),
            class_table_digest: &class_table,
            certified: true,
        };
        let digest = binding_digest(&binding).unwrap();
        let mut manifest = BindingManifest {
            schema: MANIFEST_SCHEMA.to_owned(),
            binding_digest: digest.clone(),
            source_digest: binding.source_digest.to_owned(),
            compiler_path: binding.compiler_path.to_owned(),
            compiler_sha256: binding.compiler_sha256.to_owned(),
            compiler_release: binding.compiler_release.to_owned(),
            compiler_build: binding.compiler_build.to_owned(),
            normalized_arguments: arguments.clone(),
            architecture: binding.architecture.to_owned(),
            driver_api_version: binding.driver_api_version,
            rust_host_source_digest: binding.rust_host_source_digest.to_owned(),
            rust_toolchain: binding.rust_toolchain.to_owned(),
            rust_build_flags: flags,
            family: binding.family.to_owned(),
            artifact_sha256: binding.artifact_sha256.to_owned(),
            oracle_id: binding.oracle_id.to_owned(),
            oracle_version: binding.oracle_version.to_owned(),
            artifact_evidence_id: binding.artifact_evidence_id.to_owned(),
            jit_evidence_id: binding.jit_evidence_id.map(str::to_owned),
            class_table_digest: binding.class_table_digest.to_owned(),
            product_file: "kernel.fatbin".to_owned(),
            product_sha256: sha256_hex(product),
            product_domain_digest: domain_sha256_hex(FATBIN_DOMAIN, product),
            product_size: product.len() as u64,
            certified: true,
            experimental_waiver: None,
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        verify_manifest(&manifest, &manifest_path, &product_path, &digest).unwrap();

        manifest.family = "tampered_family".to_owned();
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert_eq!(
            verify_manifest(&manifest, &manifest_path, &product_path, &digest)
                .unwrap_err()
                .code(),
            ErrorCode::JitCompileFailed
        );
    }

    #[test]
    fn registry_admission_binds_the_exact_compiler_binary() {
        let compiler = JitToolchainFacts {
            selected_path: "/opt/cuda/bin/nvcc".into(),
            resolved_path: "/opt/cuda/bin/nvcc".into(),
            release: "13.2".to_owned(),
            build: "V13.2.78".to_owned(),
            compiler_sha256: "a".repeat(64),
            world_writable_component: None,
        };
        let source = source_digest();
        let host = "b".repeat(64);
        let value = serde_json::json!({
            "direct_source_digest": source,
            "direct_host_source_digest": host,
            "direct_build_flags": NORMALIZED_FLAGS,
            "direct_devices": ["sm_120"],
            "direct_toolchains": [{
                "release": "13.2",
                "build": "V13.2.78",
                "compiler_sha256": "a".repeat(64),
                "architecture": "sm_120",
                "evidence_id": "ev-direct-jit-test"
            }]
        });
        let row = value.as_object().unwrap();
        assert_eq!(
            direct_evidence_id(row, &source, &compiler, "sm_120", &host),
            Some("ev-direct-jit-test")
        );
        let mut changed = compiler.clone();
        changed.compiler_sha256 = "c".repeat(64);
        assert_eq!(
            direct_evidence_id(row, &source, &changed, "sm_120", &host),
            None
        );
    }
}
