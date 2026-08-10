/// Runtime certification policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Policy {
    /// Admit only evidence-bound accelerated routes; fall back exactly.
    #[default]
    Certified,
    /// Force the frozen Hugging Face reference implementation.
    Reference,
    /// Require at least one certified accelerated backend at construction.
    /// Per-input safety guards may still fall back to the reference.
    RequireAccelerated,
    /// Permit explicitly requested experimental delivery while labelling it.
    Experimental,
}

/// Requested execution device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Device {
    #[default]
    Auto,
    Cpu,
    Cuda(u32),
}

/// CUDA kernel delivery preference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum GpuDelivery {
    #[default]
    Auto,
    Prebuilt,
    Jit,
    Disabled,
}

/// Backend that actually produced token IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Backend {
    HuggingFace,
    FastCpu,
    Gpu,
}

impl Backend {
    pub(crate) fn from_internal(value: &str) -> Self {
        match value {
            "gpu" => Self::Gpu,
            "fast_cpu" => Self::FastCpu,
            _ => Self::HuggingFace,
        }
    }
}

/// Stable route/fallback reason identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ReasonCode {
    InputBelowGpuThreshold,
    InputAddedToken,
    InputGuardRouted,
    ExecutionFault,
    GpuUnavailable,
    GpuUncertified,
    CpuUncertified,
    RuntimeBuildUncertified,
    ReferenceRequested,
    Other(String),
}

/// Certification status attached to one admitted route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Certification {
    Certified,
    CertifiedSource,
    Experimental,
    Reference,
}

/// Immutable route selected at tokenizer construction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RoutePlan {
    pub family: String,
    pub artifact_sha256: String,
    pub backends: Vec<Backend>,
    pub gpu_min_bytes: u64,
    pub certification: Certification,
    pub reasons: Vec<ReasonCode>,
}

/// Facts for one completed encode or append operation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExecutionFacts {
    pub backend: Backend,
    pub path: String,
    pub source: Option<String>,
    pub input_bytes: u64,
    pub certification: Certification,
}

/// Build/runtime facts that can be emitted by a control plane without
/// parsing human-readable output.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DoctorFacts {
    pub crate_version: String,
    pub oracle: String,
    pub registry_verified: bool,
    pub python_required: bool,
    pub sqlite_compiled: bool,
    pub prebuilt_gpu_compiled: bool,
    pub jit_compiled: bool,
    pub runtime_build: RuntimeBuildFacts,
    pub cuda: Option<CudaFacts>,
}

/// Exact build identity embedded by the Python-free Rust facade.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuntimeBuildFacts {
    pub source_digest: String,
    pub fast_cpu_source_digest: String,
    pub native_host_source_digest: String,
    pub toolchain: String,
    pub build_flags: Vec<String>,
    pub certified: bool,
}

/// Non-invasive CUDA probe result. Probe failure is data, not a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CudaFacts {
    pub device_ordinal: u32,
    pub available: bool,
    pub architecture: Option<String>,
    pub driver_api_version: Option<i32>,
    pub error: Option<String>,
}

/// Public snapshot of the native state-store counters.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StoreStats {
    pub format: String,
    pub session_count: u64,
    pub node_count: u64,
    pub puts: u64,
    pub appends: u64,
    pub forks: u64,
    pub lookups: u64,
    pub lookup_hits: u64,
    pub lookup_misses: u64,
    pub checksum_rejects: u64,
    pub sessions_evicted: u64,
    pub import_rejects: u64,
    pub revision_conflicts: u64,
    pub path_counts: BTreeMap<String, u64>,
}

impl From<toktier_store_core::StatsSnapshot> for StoreStats {
    fn from(value: toktier_store_core::StatsSnapshot) -> Self {
        Self {
            format: value.format.to_owned(),
            session_count: value.session_count,
            node_count: value.node_count,
            puts: value.puts,
            appends: value.extends,
            forks: value.forks,
            lookups: value.lookups,
            lookup_hits: value.lookup_hits,
            lookup_misses: value.lookup_misses,
            checksum_rejects: value.checksum_rejects,
            sessions_evicted: value.sessions_evicted,
            import_rejects: value.import_rejects,
            revision_conflicts: value.revision_conflicts,
            path_counts: value.path_counts,
        }
    }
}
use std::collections::BTreeMap;
