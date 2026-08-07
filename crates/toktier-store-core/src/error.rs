//! Store error type carrying the frozen contract error codes.
//!
//! Code strings follow `docs/contracts/errors.md`: machine interfaces
//! switch on [`StoreError::code`], never on messages. The variants keep
//! finer diagnostic structure than the code table; several variants
//! share one contract code (for example, every corruption shape maps to
//! `STORE_CORRUPT`, while well-formed-but-newer records map to
//! `STORE_FORMAT_UNSUPPORTED` -- distinct from corruption by design).

use std::fmt;

use crate::engine::WitnessCategory;

/// Errors returned by the session store and the format v1 codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// The record ends before a field that the format requires.
    Truncated {
        /// Which structure was being read when the input ran out.
        context: &'static str,
    },
    /// The record does not start with the format magic.
    BadMagic,
    /// The record declares a format version this reader does not
    /// support (well-formed but newer; not corruption).
    UnsupportedFormatVersion(u16),
    /// The explicit endianness marker byte is wrong.
    BadEndianMarker(u8),
    /// The declared header length violates the frozen bounds
    /// (`200 <= header_length <= 4096`, multiple of 8, within record).
    BadHeaderLength {
        header_length: u16,
        record_len: usize,
    },
    /// The witness category value is not in the frozen registry.
    UnknownWitnessCategory(u16),
    /// One or more mandatory flag bits are set that this reader does
    /// not understand.
    UnknownMandatoryFlags(u32),
    /// The record checksum does not match its content.
    ChecksumMismatch,
    /// The chain link hash (`curr_block_hash`) does not recompute.
    ChainLinkMismatch,
    /// The record parsed structurally but violates a frozen field
    /// constraint or cross-field invariant.
    MalformedRecord(String),
    /// A write carried an `expected_revision` that does not match the
    /// session's current revision. Last-writer-wins is not offered.
    RevisionConflict { expected: u64, actual: u64 },
    /// The key id is not registered in this store.
    UnknownKey(u32),
    /// The session handle does not exist (never created, or evicted).
    UnknownSession(u64),
    /// A record's semantic fingerprint does not match the key it was
    /// imported under. Wrong key must miss; explicit imports are loud.
    FingerprintMismatch,
    /// The engine's witness category does not match the category
    /// recorded for the session state. Certified stable prefixes are
    /// only sound under the predicate class that produced them.
    WitnessCategoryMismatch {
        recorded: WitnessCategory,
        engine: WitnessCategory,
    },
    /// A fingerprint was re-registered with a different seal guard.
    GuardMismatch,
    /// Invalid store configuration.
    InvalidConfig(String),
    /// Invalid argument to a store operation (plain misuse, not a
    /// library-domain condition).
    InvalidInput(String),
    /// The tail re-encode during session import does not reproduce the
    /// recorded ids. Prefer miss over wrong: the record is rejected.
    ImportReencodeMismatch,
    /// The session encoder reported a failure.
    Engine(String),
    /// An internal invariant of the store was broken. This indicates a
    /// bug in the store or a misbehaving encoder, never a caller error.
    Internal(String),
}

impl StoreError {
    /// Stable machine-readable error code (`docs/contracts/errors.md`).
    /// Argument-misuse and internal variants use non-contract codes and
    /// surface as plain exceptions at binding level.
    pub fn code(&self) -> &'static str {
        match self {
            StoreError::Truncated { .. }
            | StoreError::BadMagic
            | StoreError::BadEndianMarker(_)
            | StoreError::BadHeaderLength { .. }
            | StoreError::ChecksumMismatch
            | StoreError::ChainLinkMismatch
            | StoreError::MalformedRecord(_)
            | StoreError::ImportReencodeMismatch => "STORE_CORRUPT",
            StoreError::UnsupportedFormatVersion(_)
            | StoreError::UnknownWitnessCategory(_)
            | StoreError::UnknownMandatoryFlags(_) => "STORE_FORMAT_UNSUPPORTED",
            StoreError::RevisionConflict { .. } => "SESSION_REVISION_CONFLICT",
            StoreError::FingerprintMismatch | StoreError::WitnessCategoryMismatch { .. } => {
                "SESSION_STATE_MISMATCH"
            }
            StoreError::InvalidConfig(_) => "CONFIG_INVALID",
            StoreError::UnknownKey(_)
            | StoreError::UnknownSession(_)
            | StoreError::GuardMismatch
            | StoreError::InvalidInput(_) => "INVALID_ARGUMENT",
            StoreError::Engine(_) => "ENGINE_ERROR",
            StoreError::Internal(_) => "INTERNAL",
        }
    }

    /// Whether this error indicates a corrupt or untrustworthy record
    /// (`STORE_CORRUPT` family plus unsupported-format shapes). These
    /// are the failures the store converts into counted misses on the
    /// silent read path.
    pub fn is_rejection(&self) -> bool {
        matches!(self.code(), "STORE_CORRUPT" | "STORE_FORMAT_UNSUPPORTED")
    }
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] ", self.code())?;
        match self {
            StoreError::Truncated { context } => {
                write!(f, "record truncated while reading {context}")
            }
            StoreError::BadMagic => write!(f, "bad record magic"),
            StoreError::UnsupportedFormatVersion(v) => {
                write!(f, "unsupported format version {v}")
            }
            StoreError::BadEndianMarker(m) => {
                write!(f, "bad endianness marker {m:#04x} (expected 0x01)")
            }
            StoreError::BadHeaderLength {
                header_length,
                record_len,
            } => write!(
                f,
                "bad header length {header_length} for record of {record_len} bytes"
            ),
            StoreError::UnknownWitnessCategory(w) => {
                write!(f, "unknown witness category {w:#06x}")
            }
            StoreError::UnknownMandatoryFlags(bits) => {
                write!(f, "unknown mandatory flag bits {bits:#010x}")
            }
            StoreError::ChecksumMismatch => write!(f, "record checksum mismatch"),
            StoreError::ChainLinkMismatch => {
                write!(f, "chain link hash does not recompute")
            }
            StoreError::MalformedRecord(msg) => write!(f, "malformed record: {msg}"),
            StoreError::RevisionConflict { expected, actual } => write!(
                f,
                "session revision conflict: expected {expected}, stored {actual}"
            ),
            StoreError::UnknownKey(id) => write!(f, "unknown key id {id}"),
            StoreError::UnknownSession(h) => write!(f, "unknown session handle {h}"),
            StoreError::FingerprintMismatch => {
                write!(f, "record fingerprint does not match the key")
            }
            StoreError::WitnessCategoryMismatch { recorded, engine } => write!(
                f,
                "witness category mismatch: state has {recorded:?}, engine has {engine:?}"
            ),
            StoreError::GuardMismatch => write!(
                f,
                "fingerprint already registered with a different seal guard"
            ),
            StoreError::InvalidConfig(msg) => write!(f, "invalid configuration: {msg}"),
            StoreError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            StoreError::ImportReencodeMismatch => write!(
                f,
                "session import rejected: tail re-encode differs from recorded ids"
            ),
            StoreError::Engine(msg) => write!(f, "encoder error: {msg}"),
            StoreError::Internal(msg) => {
                write!(f, "internal store invariant broken: {msg}")
            }
        }
    }
}

impl std::error::Error for StoreError {}
