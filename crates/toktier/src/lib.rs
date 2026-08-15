//! Rust-native serving API for [TokTier](https://github.com/asu-idi/toktier).
//!
//! The crate owns artifact verification, exact one-shot and batch encoding,
//! and stateful append repair. Request execution does not require Python,
//! PyO3, or PyTorch. Accelerated routes are admitted only from shipped,
//! digest-bound certification data and otherwise fall back to the frozen
//! Hugging Face `tokenizers` reference implementation.

#![forbid(unsafe_code)]

mod artifact;
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
    Backend, Certification, CudaFacts, Device, DoctorFacts, ExecutionFacts, GpuDelivery, Policy,
    ReasonCode, RoutePlan, RuntimeBuildFacts, StoreStats,
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

/// Version of the frozen native Hugging Face oracle.
pub const ORACLE: &str = "tokenizers==0.22.2";

/// How this build's resolved dependency graph compares with the graph
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
/// Reported by [`Runtime::doctor`] and required for an accelerated
/// route: this crate's own sources are judged by digest, and the
/// versions Cargo resolved around them are judged here.
pub const DEPENDENCY_CLOSURE: &str = env!("TOKTIER_RUST_API_DEPENDENCY_CLOSURE");

pub(crate) fn dependency_closure_verified() -> bool {
    DEPENDENCY_CLOSURE == "verified"
}
