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
///
/// The run-time variants are this crate's spelling of the frozen `R_*`
/// namespace in `docs/contracts/routing.md` Section 5: the router records
/// those codes in its own ledger, and the ones reachable from a Rust
/// execution are named here rather than in a second vocabulary. The
/// plan-time variants describe admission decisions this crate makes for
/// itself and are not claimed to be the same codes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ReasonCode {
    /// `R_INPUT_BELOW_GPU_THRESHOLD`
    InputBelowGpuThreshold,
    /// `R_INPUT_ADDED_TOKEN`
    InputAddedToken,
    /// `R_INPUT_GUARD_ROUTED`
    InputGuardRouted,
    /// `R_EXEC_FAULT`
    ExecutionFault,
    /// `R_INPUT_POSTPROCESS_ROUTED`
    InputPostprocessRouted,
    /// `R_SESSION_NO_SAFE_CUT`
    SessionNoSafeCut,
    GpuUnavailable,
    GpuUncertified,
    CpuUncertified,
    RuntimeBuildUncertified,
    ReferenceRequested,
    /// A reason the router named that this release has no frozen code
    /// for. The string is the router's own token, passed through rather
    /// than mapped onto a code that does not mean it.
    Other(String),
}

impl ReasonCode {
    /// The reason behind one routed execution, from the code the router
    /// recorded for it. Read rather than inferred, so this answer and the
    /// ledger cannot come to disagree.
    pub(crate) fn from_ledger_code(code: &str) -> Self {
        match code {
            toktier_routing_core::R_INPUT_BELOW_GPU_THRESHOLD => Self::InputBelowGpuThreshold,
            toktier_routing_core::R_INPUT_ADDED_TOKEN => Self::InputAddedToken,
            toktier_routing_core::R_INPUT_GUARD_ROUTED => Self::InputGuardRouted,
            toktier_routing_core::R_EXEC_FAULT => Self::ExecutionFault,
            toktier_routing_core::R_INPUT_POSTPROCESS_ROUTED => Self::InputPostprocessRouted,
            other => Self::Other(other.to_owned()),
        }
    }
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
    /// Why this execution ran where it did, when a routing decision
    /// moved it off the first admitted backend. `None` means the
    /// admitted route ran and there was nothing to record; why the
    /// admitted route is what it is belongs to
    /// [`RoutePlan::reasons`](crate::RoutePlan), not to one execution.
    pub reason: Option<ReasonCode>,
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
    /// Whether this build can acquire an artifact over the network. The
    /// `network` feature is not in the default set, so a build that has
    /// to fetch and does not carry it answers `NETWORK_DISABLED`.
    pub network_compiled: bool,
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
    /// How the resolved dependency graph of this build compares with the
    /// judged one (see [`crate::DEPENDENCY_CLOSURE`]). Anything other
    /// than `"verified"` holds `certified` below at `false`, and says
    /// both which packages differ and what aligns them.
    pub dependency_closure: String,
    /// Why the build flags of this build are not the judged ones, when a
    /// judged entry agrees with it on every other axis. `None` when the
    /// flags match, or when something else is the disagreement.
    pub build_flag_divergence: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every run-time code the router can record maps onto this crate's
    /// name for it, and an unrecognised one is passed through rather
    /// than turned into a code that does not mean it.
    #[test]
    fn every_recorded_reason_maps_onto_the_frozen_namespace() {
        for (code, expected) in [
            (
                toktier_routing_core::R_INPUT_BELOW_GPU_THRESHOLD,
                ReasonCode::InputBelowGpuThreshold,
            ),
            (
                toktier_routing_core::R_INPUT_ADDED_TOKEN,
                ReasonCode::InputAddedToken,
            ),
            (
                toktier_routing_core::R_INPUT_GUARD_ROUTED,
                ReasonCode::InputGuardRouted,
            ),
            (
                toktier_routing_core::R_EXEC_FAULT,
                ReasonCode::ExecutionFault,
            ),
            (
                toktier_routing_core::R_INPUT_POSTPROCESS_ROUTED,
                ReasonCode::InputPostprocessRouted,
            ),
        ] {
            assert_eq!(ReasonCode::from_ledger_code(code), expected, "for {code}");
        }
        assert_eq!(
            ReasonCode::from_ledger_code("R_SOMETHING_LATER"),
            ReasonCode::Other("R_SOMETHING_LATER".to_owned())
        );
    }
}
