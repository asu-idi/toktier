//! toktier session store, core crate.
//!
//! This is an internal supporting crate of TokTier, versioned with the
//! workspace and carrying no independent API stability promise; use the
//! `toktier` package for the supported Rust surface.
//!
//! An append-only "text to token ids" session store with certified
//! extension: every session's delivered ids stay bit-identical to a
//! from-scratch reference encode of the accumulated text. The store owns
//! bookkeeping only; all tokenization semantics are delegated to a
//! caller-supplied [`SessionEncoder`]. Prefix sharing across sessions
//! runs over a block hash chain keyed by the opaque 32-byte semantic
//! fingerprint, so a wrong key can never hit; every hit is verified
//! before it is served (prefer miss over wrong).
//!
//! Persistence uses store format v1 (see [`format`] and the FORMAT.md
//! specification next to this crate): a strict, checksummed,
//! little-endian record layout with explicit versioning, mandatory-flag
//! rejection and a witness predicate category registry. SQLite-backed
//! storage lives in the separate `toktier-store-sqlite` crate; a thin
//! Python facade lives in `toktier-py`. This crate is pure Rust with no
//! binding or database dependencies.

#![deny(unsafe_code)]

pub mod engine;
pub mod error;
pub mod format;
pub mod recovery_binding;
pub mod sidecar;
pub mod store;
pub mod tail;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use engine::{
    AppendReport, BoundaryCut, Encoding, EngineError, OverlapRunner, SessionEncoder, SharedIds,
    SoaEncoding, WitnessCategory,
};
pub use error::StoreError;
pub use format::{
    link_hash, payload_digest_parts, BlockHash, LinkInputs, PayloadHasher, SessionRecordV1,
    ENDIANNESS_LE, FIXED_HEADER_LEN, FORMAT_VERSION, HEADER_LEN_MAX, MAGIC, MANDATORY_FLAGS_MASK,
    ZERO_HASH,
};
pub use recovery_binding::{
    ContentDigest, ContentIndexEntry, RecoveryBindingV1, CONTENT_DIGEST_BYTES, MARK_FLOOR_BYTES,
};
pub use sidecar::{node_key, NodeCacheRecord, SessionSidecar};
pub use store::{
    AppendOutcome, AppendPatchOutcome, KeyId, LookupHit, PutOutcome, RecoveryMaterial,
    SemanticFingerprint, SessionHandle, SessionInfo, SessionStore, StatsSnapshot, StoreConfig,
    FORMAT_NAME,
};
pub use tail::{TailState, SPAN_CHECKPOINT_STRIDE};
