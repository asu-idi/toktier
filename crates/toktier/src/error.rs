use std::fmt;
use std::path::PathBuf;

/// Stable machine-readable public error code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ErrorCode {
    InvalidArgument,
    ConfigInvalid,
    ArtifactNotFound,
    ArtifactHashMismatch,
    ArtifactSizeMismatch,
    BundleInvalid,
    CacheBusy,
    Network,
    NetworkDisabled,
    RegistryInvalid,
    UncertifiedRuntime,
    UncertifiedTokenizer,
    KernelIncompatible,
    JitCompileFailed,
    UncertifiedJit,
    QueueFull,
    RequestCancelled,
    DeadlineExceeded,
    RuntimeShutdown,
    BackendExecutionFault,
    SessionRevisionConflict,
    SessionStateMismatch,
    StoreCorrupt,
    StoreFormatUnsupported,
    Io,
    Internal,
}

impl ErrorCode {
    /// Frozen string representation used by logs and non-Rust adapters.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::ConfigInvalid => "CONFIG_INVALID",
            Self::ArtifactNotFound => "ARTIFACT_NOT_FOUND",
            Self::ArtifactHashMismatch => "ARTIFACT_HASH_MISMATCH",
            Self::ArtifactSizeMismatch => "ARTIFACT_SIZE_MISMATCH",
            Self::BundleInvalid => "BUNDLE_INVALID",
            Self::CacheBusy => "CACHE_BUSY",
            Self::Network => "NETWORK_ERROR",
            Self::NetworkDisabled => "NETWORK_DISABLED",
            Self::RegistryInvalid => "REGISTRY_INVALID",
            Self::UncertifiedRuntime => "UNCERTIFIED_RUNTIME",
            Self::UncertifiedTokenizer => "UNCERTIFIED_TOKENIZER",
            Self::KernelIncompatible => "KERNEL_INCOMPATIBLE",
            Self::JitCompileFailed => "JIT_COMPILE_FAILED",
            Self::UncertifiedJit => "UNCERTIFIED_JIT",
            Self::QueueFull => "QUEUE_FULL",
            Self::RequestCancelled => "REQUEST_CANCELLED",
            Self::DeadlineExceeded => "DEADLINE_EXCEEDED",
            Self::RuntimeShutdown => "RUNTIME_SHUTDOWN",
            Self::BackendExecutionFault => "BACKEND_EXECUTION_FAULT",
            Self::SessionRevisionConflict => "SESSION_REVISION_CONFLICT",
            Self::SessionStateMismatch => "SESSION_STATE_MISMATCH",
            Self::StoreCorrupt => "STORE_CORRUPT",
            Self::StoreFormatUnsupported => "STORE_FORMAT_UNSUPPORTED",
            Self::Io => "IO_ERROR",
            Self::Internal => "INTERNAL",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Structured public error. Callers should branch on [`Error::code`], not on
/// the human-readable message.
#[derive(Debug)]
pub struct Error {
    code: ErrorCode,
    message: String,
    path: Option<PathBuf>,
    family: Option<String>,
}

impl Error {
    pub(crate) fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            path: None,
            family: None,
        }
    }

    pub(crate) fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub(crate) fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = Some(family.into());
        self
    }

    pub const fn code(&self) -> ErrorCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
    }

    pub fn family(&self) -> Option<&str> {
        self.family.as_deref()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::new(ErrorCode::Io, error.to_string())
    }
}

impl From<toktier_store_core::StoreError> for Error {
    fn from(error: toktier_store_core::StoreError) -> Self {
        let code = match error.code() {
            "CONFIG_INVALID" => ErrorCode::ConfigInvalid,
            "SESSION_REVISION_CONFLICT" => ErrorCode::SessionRevisionConflict,
            "SESSION_STATE_MISMATCH" => ErrorCode::SessionStateMismatch,
            "STORE_CORRUPT" => ErrorCode::StoreCorrupt,
            "STORE_FORMAT_UNSUPPORTED" => ErrorCode::StoreFormatUnsupported,
            "INVALID_ARGUMENT" => ErrorCode::InvalidArgument,
            "ENGINE_ERROR" => ErrorCode::BackendExecutionFault,
            _ => ErrorCode::Internal,
        };
        Self::new(code, error.to_string())
    }
}

#[cfg(feature = "sqlite")]
impl From<toktier_store_sqlite::DbError> for Error {
    fn from(error: toktier_store_sqlite::DbError) -> Self {
        match error {
            toktier_store_sqlite::DbError::Store(error) => error.into(),
            toktier_store_sqlite::DbError::Sqlite(error) => {
                Self::new(ErrorCode::Io, error.to_string())
            }
            toktier_store_sqlite::DbError::Schema(message) => {
                Self::new(ErrorCode::StoreCorrupt, message)
            }
            toktier_store_sqlite::DbError::MissingEngine(_) => Self::new(
                ErrorCode::SessionStateMismatch,
                "persistent state belongs to a different tokenizer runtime",
            ),
        }
    }
}

/// Public result alias.
pub type Result<T> = std::result::Result<T, Error>;
