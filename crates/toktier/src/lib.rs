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

pub use artifact::{
    ArtifactInspection, ArtifactManager, ArtifactManagerBuilder, ArtifactSource, BearerToken,
    EnvironmentToken, Revision, SecretProvider,
};
pub use buffer::{Encoding, RaggedEncoding, TokenBuffer, TokenPatch};
pub use bundle::{export_bundle, import_bundle, inspect_bundle, BundleInspection};
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
