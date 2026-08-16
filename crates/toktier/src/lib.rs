//! Rust-native serving API for [TokTier](https://github.com/asu-idi/toktier).
//!
//! The crate owns artifact verification, exact one-shot and batch encoding,
//! and stateful append repair. Request execution does not require Python,
//! PyO3, or PyTorch. Accelerated routes are admitted only from shipped,
//! digest-bound certification data and otherwise fall back to the frozen
//! Hugging Face `tokenizers` reference implementation.

#![forbid(unsafe_code)]

mod artifact;
mod behavior_version;
mod buffer;
mod bundle;
mod diagnostics;
mod error;
mod fsutil;
#[cfg(feature = "prebuilt-gpu")]
mod gpu_data;
#[cfg(feature = "jit")]
mod jit;
mod manifest;
mod package_data;
mod runtime;
#[cfg(feature = "serving")]
mod serving;
mod session;
mod suggest;
mod verify_local;

// The build script's dependency-closure comparison is compiled into the
// test build so its unit tests run with the rest of the suite; the path
// lists it also carries are unused here.
#[cfg(test)]
#[allow(dead_code)]
#[path = "../build_support/source_identity.rs"]
mod build_support;

pub use artifact::{
    ArtifactInspection, ArtifactManager, ArtifactManagerBuilder, ArtifactSource, BearerToken,
    EnvironmentToken, Revision, SecretProvider,
};
pub use buffer::{Encoding, RaggedEncoding, TokenBuffer, TokenPatch};
pub use bundle::{
    export_bundle, import_bundle, inspect_bundle, BundleFileInspection, BundleInspection,
};
pub use diagnostics::{
    Backend, BehaviorVersion, Certification, CudaFacts, Device, DoctorFacts, ExecutionFacts,
    GpuDelivery, Policy, ReasonCode, RoutePlan, RuntimeBuildFacts, StoreStats,
};
pub use error::{Error, ErrorCode, Result};
#[cfg(feature = "jit")]
pub use jit::{JitArtifact, JitCompiler, JitCompilerBuilder, JitToolchainFacts};
pub use manifest::{ArtifactIdentity, LocalArtifact};
pub use runtime::{DecodeOptions, EncodeOptions, Runtime, RuntimeBuilder, Tokenizer};
#[cfg(feature = "serving")]
pub use serving::{
    DeviceFailurePolicy, ServingLimits, ServingPool, ServingPoolBuilder, ServingRequest,
    ServingResponse, ServingSession, ServingTimings,
};
pub use session::{Session, SessionStats};
pub use verify_local::verify_local_command;

/// Version of the frozen native Hugging Face oracle.
pub const ORACLE: &str = "tokenizers==0.22.2";

/// How the whole compiled closure of this build compares with the graph
/// the certification evidence was taken on: `"verified"`, or a line
/// beginning `"unlocated: "` when no governing lockfile could be named,
/// or one beginning `"mismatched: "` naming every package that is not
/// the judged one.
///
/// Anything other than `"verified"` also says what closes the gap: a
/// `mismatched` line carries the `cargo update --precise` command for
/// each package it names, and an `unlocated` line says where the judged
/// graph travels and how to name the governing lockfile.
///
/// Since 0.2.6 this is a reading, not a gate. An accelerated route
/// requires the certified core of that closure to be the judged one,
/// which [`RuntimeBuildFacts::core_closure`] reports; packages outside
/// the core are compared as well and reported in
/// [`RuntimeBuildFacts::dependency_advisory`]. Readers who want the
/// 0.2.4 and 0.2.5 rule can require this constant to read `"verified"`
/// themselves.
pub const DEPENDENCY_CLOSURE: &str = env!("TOKTIER_RUST_API_DEPENDENCY_CLOSURE");
