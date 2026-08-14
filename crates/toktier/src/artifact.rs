//! Rust-native artifact acquisition, verification, cache, and mirror surface.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::fsutil::{hex, monotonic_nonce, open_private_lock, set_private_file, sync_directory};
use crate::manifest::{ArtifactFileRow, ArtifactRow, Registry};
use crate::{export_bundle, import_bundle, BundleInspection};
use crate::{Error, ErrorCode, Result};

const MARKER_NAME: &str = ".toktier-verified.json";
const MARKER_SCHEMA: &str = "toktier.rust.artifact_marker.v1";
const READ_CHUNK: usize = 1024 * 1024;

/// Immutable repository revision accepted by network acquisition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Revision(String);

impl Revision {
    /// Construct a pinned 40-hex repository commit.
    pub fn commit(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.len() != 40
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "a remote revision must be an immutable 40-character lowercase hex commit",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Secret bearer material. `Debug` and `Display` deliberately redact it.
#[derive(Clone)]
pub struct BearerToken(#[allow(dead_code)] String);

impl BearerToken {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty() || value.contains(['\r', '\n', '\0']) {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "bearer token must be non-empty and contain no control separators",
            ));
        }
        Ok(Self(value))
    }

    #[cfg(feature = "network")]
    fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for BearerToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BearerToken([REDACTED])")
    }
}

/// Supplies authentication only at request construction. Implementations must
/// not persist credentials in cache paths or diagnostics.
pub trait SecretProvider: Send + Sync {
    fn bearer_token(&self, repo_id: &str, revision: &Revision) -> Result<Option<BearerToken>>;
}

/// Reads one explicitly named environment variable on demand.
#[derive(Debug, Clone)]
pub struct EnvironmentToken {
    variable: String,
}

impl EnvironmentToken {
    pub fn new(variable: impl Into<String>) -> Result<Self> {
        let variable = variable.into();
        if variable.is_empty()
            || !variable
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "secret environment-variable name must be non-empty ASCII alphanumeric/underscore",
            ));
        }
        Ok(Self { variable })
    }
}

impl SecretProvider for EnvironmentToken {
    fn bearer_token(&self, _repo_id: &str, _revision: &Revision) -> Result<Option<BearerToken>> {
        std::env::var(&self.variable)
            .ok()
            .map(BearerToken::new)
            .transpose()
    }
}

/// Source used when a verified cache object is absent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArtifactSource {
    #[default]
    HuggingFace,
    Mirror {
        base_url: String,
    },
    LocalDirectory {
        root: PathBuf,
    },
    None,
}

/// Builder for the shareable artifact lifecycle manager.
#[derive(Clone)]
pub struct ArtifactManagerBuilder {
    cache: PathBuf,
    source: ArtifactSource,
    offline: bool,
    timeout: Duration,
    redirects: u32,
    retries: u32,
    retry_backoff: Duration,
    stale_temporary_after: Duration,
    lock_timeout: Duration,
    allow_insecure_loopback: bool,
    secret_provider: Option<Arc<dyn SecretProvider>>,
}

impl fmt::Debug for ArtifactManagerBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactManagerBuilder")
            .field("cache", &self.cache)
            .field("source", &self.source)
            .field("offline", &self.offline)
            .field("timeout", &self.timeout)
            .field("redirects", &self.redirects)
            .field("retries", &self.retries)
            .field("retry_backoff", &self.retry_backoff)
            .field("stale_temporary_after", &self.stale_temporary_after)
            .field("lock_timeout", &self.lock_timeout)
            .field("allow_insecure_loopback", &self.allow_insecure_loopback)
            .field(
                "secret_provider",
                &self.secret_provider.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

impl Default for ArtifactManagerBuilder {
    fn default() -> Self {
        let cache = crate::fsutil::default_cache_directory("TOKTIER_ARTIFACT_CACHE", "artifacts");
        Self {
            cache,
            source: ArtifactSource::default(),
            offline: false,
            timeout: Duration::from_secs(300),
            redirects: 5,
            retries: 2,
            retry_backoff: Duration::from_millis(100),
            stale_temporary_after: Duration::from_secs(24 * 60 * 60),
            lock_timeout: Duration::from_secs(60),
            allow_insecure_loopback: false,
            secret_provider: None,
        }
    }
}

impl ArtifactManagerBuilder {
    pub fn cache(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache = path.into();
        self
    }

    pub fn source(mut self, source: ArtifactSource) -> Self {
        self.source = source;
        self
    }

    pub fn offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn max_redirects(mut self, redirects: u32) -> Self {
        self.redirects = redirects.min(10);
        self
    }

    /// Number of retries after the first request for transport-level errors.
    /// Hash, size, registry, and authentication failures are never retried.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.retries = retries.min(8);
        self
    }

    /// Initial exponential delay between transport retries.
    pub fn retry_backoff(mut self, backoff: Duration) -> Self {
        self.retry_backoff = backoff;
        self
    }

    /// Remove abandoned private `.part` files older than this duration while
    /// holding the family cache lock. A zero duration disables reclamation.
    pub fn stale_temporary_after(mut self, age: Duration) -> Self {
        self.stale_temporary_after = age;
        self
    }

    pub fn lock_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    /// Permit plaintext HTTP only when the configured URL host is loopback.
    /// This exists for hermetic tests and never permits a remote HTTP host.
    pub fn allow_insecure_loopback(mut self, allow: bool) -> Self {
        self.allow_insecure_loopback = allow;
        self
    }

    pub fn secret_provider(mut self, provider: Arc<dyn SecretProvider>) -> Self {
        self.secret_provider = Some(provider);
        self
    }

    pub fn build(self) -> Result<ArtifactManager> {
        if self.cache.as_os_str().is_empty() {
            return Err(Error::new(
                ErrorCode::ConfigInvalid,
                "artifact cache path must not be empty",
            ));
        }
        if self.timeout.is_zero() || self.lock_timeout.is_zero() {
            return Err(Error::new(
                ErrorCode::ConfigInvalid,
                "artifact timeout must be greater than zero",
            ));
        }
        if self.retries > 0 && self.retry_backoff.is_zero() {
            return Err(Error::new(
                ErrorCode::ConfigInvalid,
                "artifact retry backoff must be greater than zero when retries are enabled",
            ));
        }
        if let ArtifactSource::Mirror { base_url } = &self.source {
            validate_base_url(base_url, self.allow_insecure_loopback)?;
        }
        Ok(ArtifactManager {
            inner: Arc::new(ArtifactManagerInner {
                cache: self.cache,
                source: self.source,
                offline: self.offline,
                timeout: self.timeout,
                redirects: self.redirects,
                retries: self.retries,
                retry_backoff: self.retry_backoff,
                stale_temporary_after: self.stale_temporary_after,
                lock_timeout: self.lock_timeout,
                allow_insecure_loopback: self.allow_insecure_loopback,
                secret_provider: self.secret_provider,
            }),
        })
    }
}

struct ArtifactManagerInner {
    cache: PathBuf,
    source: ArtifactSource,
    offline: bool,
    timeout: Duration,
    redirects: u32,
    retries: u32,
    retry_backoff: Duration,
    stale_temporary_after: Duration,
    lock_timeout: Duration,
    allow_insecure_loopback: bool,
    secret_provider: Option<Arc<dyn SecretProvider>>,
}

/// Rust-native artifact lifecycle manager.
#[derive(Clone)]
pub struct ArtifactManager {
    inner: Arc<ArtifactManagerInner>,
}

impl fmt::Debug for ArtifactManager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactManager")
            .field("cache", &self.inner.cache)
            .field("source", &self.inner.source)
            .field("offline", &self.inner.offline)
            .field("timeout", &self.inner.timeout)
            .field("redirects", &self.inner.redirects)
            .field("retries", &self.inner.retries)
            .field(
                "secret_provider",
                &self.inner.secret_provider.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

/// Verified cache facts for diagnostics and control planes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct ArtifactInspection {
    pub family: String,
    pub repo_id: String,
    pub revision: String,
    pub directory: PathBuf,
    pub tokenizer_sha256: String,
    pub tokenizer_size: u64,
    pub verified: bool,
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Marker {
    schema: String,
    family: String,
    revision: String,
    tokenizer_sha256: String,
    tokenizer_size: u64,
}

impl ArtifactManager {
    pub fn builder() -> ArtifactManagerBuilder {
        ArtifactManagerBuilder::default()
    }

    pub fn cache_root(&self) -> &Path {
        &self.inner.cache
    }

    pub fn offline(&self) -> bool {
        self.inner.offline
    }

    pub(crate) fn with_cache(&self, cache: PathBuf) -> Self {
        Self {
            inner: Arc::new(ArtifactManagerInner {
                cache,
                source: self.inner.source.clone(),
                offline: self.inner.offline,
                timeout: self.inner.timeout,
                redirects: self.inner.redirects,
                retries: self.inner.retries,
                retry_backoff: self.inner.retry_backoff,
                stale_temporary_after: self.inner.stale_temporary_after,
                lock_timeout: self.inner.lock_timeout,
                allow_insecure_loopback: self.inner.allow_insecure_loopback,
                secret_provider: self.inner.secret_provider.clone(),
            }),
        }
    }

    pub fn fetch(&self, family: &str) -> Result<ArtifactInspection> {
        let registry = Registry::load()?;
        self.ensure(family, &registry)?;
        self.inspect_with_registry(family, &registry)
    }

    pub fn verify(&self, family: &str) -> Result<ArtifactInspection> {
        let registry = Registry::load()?;
        self.inspect_with_registry(family, &registry)
    }

    pub fn inspect(&self, family: &str) -> Result<ArtifactInspection> {
        self.verify(family)
    }

    pub fn mirror(&self, family: &str, root: impl AsRef<Path>) -> Result<PathBuf> {
        let registry = Registry::load()?;
        let directory = self.ensure(family, &registry)?;
        let row = registry.artifact(family)?;
        let destination = root
            .as_ref()
            .join(&row.repo_id)
            .join("resolve")
            .join(&row.revision);
        ensure_private_directory(&destination)?;
        for name in row.files.keys() {
            let source = checked_file(&directory, name)?;
            let target = checked_join(&destination, name)?;
            copy_atomic(&source, &target)?;
        }
        Ok(destination)
    }

    /// Export a canonical Python-v1-compatible air-gap archive. Besides the
    /// model files, it carries the exact embedded registries, schemas, class
    /// tables, source, prebuilt binary, and provenance used by this runtime.
    pub fn export(&self, family: &str, bundle: impl AsRef<Path>) -> Result<BundleInspection> {
        let registry = Registry::load()?;
        let directory = self.ensure(family, &registry)?;
        let row = registry.artifact(family)?;
        ensure_private_directory(&self.inner.cache)?;
        let staging = allocate_staging_directory(&self.inner.cache, ".bundle-metadata")?;
        let result = (|| {
            let mut files = std::collections::BTreeMap::new();
            for name in row.files.keys() {
                files.insert(name.clone(), checked_file(&directory, name)?);
            }
            for (name, bytes) in crate::package_data::FILES {
                let relative = format!("toktier-runtime/{name}");
                let target = checked_join(&staging, &relative)?;
                if let Some(parent) = target.parent() {
                    ensure_private_directory(parent)?;
                }
                let mut output = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&target)
                    .map_err(|error| {
                        Error::new(ErrorCode::Io, error.to_string()).with_path(&target)
                    })?;
                set_private_file(&output)?;
                output
                    .write_all(bytes)
                    .and_then(|()| output.sync_all())
                    .map_err(|error| {
                        Error::new(ErrorCode::Io, error.to_string()).with_path(&target)
                    })?;
                files.insert(relative, target);
            }
            export_bundle(bundle, &directory_name(family, &row.revision), &files)
        })();
        let _ = fs::remove_dir_all(&staging);
        result
    }

    /// Import and authenticate a v1 bundle into this manager's cache. The
    /// installed alias must be one of the exact shipped family revisions.
    pub fn import(&self, bundle: impl AsRef<Path>) -> Result<ArtifactInspection> {
        let registry = Registry::load()?;
        let bundle_facts = crate::inspect_bundle(&bundle)?;
        let (family, row) = registry
            .artifacts()
            .find(|(family, row)| directory_name(family, &row.revision) == bundle_facts.alias)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::UncertifiedTokenizer,
                    format!(
                        "bundle alias {:?} is not a shipped artifact identity",
                        bundle_facts.alias
                    ),
                )
            })?;
        import_bundle(bundle, &self.inner.cache)?;
        let directory = self.inner.cache.join(&bundle_facts.alias);
        verify_directory(&directory, row)?;
        write_marker(&directory, family, row)?;
        self.inspect_with_registry(family, &registry)
    }

    pub(crate) fn ensure(&self, family: &str, registry: &Registry) -> Result<PathBuf> {
        let row = registry.artifact(family)?;
        validate_entry(family, row)?;
        ensure_private_directory(&self.inner.cache)?;
        let directory = self.inner.cache.join(directory_name(family, &row.revision));
        if verify_directory(&directory, row).is_ok() {
            return Ok(directory);
        }
        prune_stale_temporary_files(&directory, self.inner.stale_temporary_after)?;

        let locks = self.inner.cache.join(".locks");
        ensure_private_directory(&locks)?;
        let lock_path = locks.join(format!("{family}-{}.lock", &row.revision[..12]));
        let lock = open_private_lock(&lock_path)?;
        lock_exclusive_bounded(&lock, self.inner.lock_timeout)?;
        if verify_directory(&directory, row).is_ok() {
            return Ok(directory);
        }
        // A family whose artifact is derived locally has nothing to
        // download, so this answer does not depend on whether anything
        // could be downloaded: the pinned upstream revision does not
        // publish the file the manifest names, and asking for it returns
        // a 404 and a `NETWORK_ERROR` that tells the reader nothing they
        // can act on. It is asked before the offline gate because that
        // gate is the general "these bytes did not arrive" answer, and
        // an air-gapped reader of a derived family is exactly the one
        // who most needs to be told which two routes carry the bytes.
        if let Some(conversion) = registry.conversion(family) {
            let inputs = conversion
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::new(
                ErrorCode::ArtifactNotFound,
                format!(
                    "artifact {family:?} is derived locally by the {:?} conversion, not published: \
                     upstream {} at the pinned revision carries {inputs}, not the file this \
                     manifest names, so there is nothing for this crate to download. This crate \
                     runs the conversion for no family. Produce the artifact once with the Python \
                     package (`toktier artifacts fetch {family}`, which converts and verifies it \
                     against the same pinned digest) or receive it as an air-gap bundle, then \
                     point this crate at those bytes: an artifact cache holding them is used as \
                     it stands, and `Runtime::load_local` opens one explicit directory",
                    conversion.converter, row.repo_id,
                ),
            )
            .with_family(family));
        }

        if self.inner.offline || matches!(self.inner.source, ArtifactSource::None) {
            return Err(Error::new(
                ErrorCode::ArtifactNotFound,
                format!(
                    "artifact {family:?} is not verified in {} and artifact acquisition is offline",
                    self.inner.cache.display()
                ),
            )
            .with_family(family));
        }

        ensure_private_directory(&directory)?;
        for (name, file) in &row.files {
            self.install_file(family, row, name, file, &directory)?;
        }
        verify_directory(&directory, row)?;
        write_marker(&directory, family, row)?;
        sync_directory(&directory)?;
        Ok(directory)
    }

    fn inspect_with_registry(
        &self,
        family: &str,
        registry: &Registry,
    ) -> Result<ArtifactInspection> {
        let row = registry.artifact(family)?;
        let directory = self.inner.cache.join(directory_name(family, &row.revision));
        verify_directory(&directory, row)?;
        let tokenizer = row.files.get("tokenizer.json").ok_or_else(|| {
            Error::new(ErrorCode::RegistryInvalid, "artifact has no tokenizer.json")
        })?;
        Ok(ArtifactInspection {
            family: family.to_owned(),
            repo_id: row.repo_id.clone(),
            revision: row.revision.clone(),
            directory,
            tokenizer_sha256: tokenizer.sha256.clone(),
            tokenizer_size: tokenizer.size,
            verified: true,
            source: source_name(&self.inner.source).to_owned(),
        })
    }

    fn install_file(
        &self,
        family: &str,
        row: &ArtifactRow,
        name: &str,
        expected: &ArtifactFileRow,
        directory: &Path,
    ) -> Result<()> {
        let target = checked_join(directory, name)?;
        if target.exists() {
            if verify_file(&target, expected).is_ok() {
                return Ok(());
            }
            quarantine(&self.inner.cache, family, &target)?;
        }
        if fs2::available_space(directory).unwrap_or(u64::MAX) < expected.size {
            return Err(Error::new(
                ErrorCode::Io,
                format!("insufficient free space for {family}/{name}"),
            ));
        }
        let temporary = unique_temporary(directory, name)?;
        let result = (|| {
            match &self.inner.source {
                ArtifactSource::LocalDirectory { root } => {
                    let cache_layout = root.join(directory_name(family, &row.revision)).join(name);
                    let mirror_layout = root
                        .join(&row.repo_id)
                        .join("resolve")
                        .join(&row.revision)
                        .join(name);
                    let source = if cache_layout.exists() {
                        cache_layout
                    } else {
                        mirror_layout
                    };
                    copy_stream_checked(&source, &temporary, expected)?;
                }
                ArtifactSource::HuggingFace => {
                    let url = format!(
                        "https://huggingface.co/{}/resolve/{}/{}",
                        row.repo_id, row.revision, name
                    );
                    self.download_checked(&url, &row.repo_id, &row.revision, &temporary, expected)?;
                }
                ArtifactSource::Mirror { base_url } => {
                    let url = format!(
                        "{}/{}/resolve/{}/{}",
                        base_url.trim_end_matches('/'),
                        row.repo_id,
                        row.revision,
                        name
                    );
                    self.download_checked(&url, &row.repo_id, &row.revision, &temporary, expected)?;
                }
                ArtifactSource::None => unreachable!("offline gate handled no source"),
            }
            fs::rename(&temporary, &target)
                .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&target))?;
            sync_directory(directory)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    #[cfg(feature = "network")]
    fn download_checked(
        &self,
        url: &str,
        repo_id: &str,
        raw_revision: &str,
        destination: &Path,
        expected: &ArtifactFileRow,
    ) -> Result<()> {
        let mut last_error = None;
        for attempt in 0..=self.inner.retries {
            let _ = fs::remove_file(destination);
            match self.download_once(url, repo_id, raw_revision, destination, expected) {
                Ok(()) => return Ok(()),
                Err(error)
                    if error.code() == ErrorCode::Network && attempt < self.inner.retries =>
                {
                    last_error = Some(error);
                    let multiplier = 1u32 << attempt.min(8);
                    std::thread::sleep(
                        self.inner
                            .retry_backoff
                            .checked_mul(multiplier)
                            .unwrap_or(Duration::from_secs(30)),
                    );
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            Error::new(
                ErrorCode::Network,
                "artifact request exhausted its retry budget",
            )
        }))
    }

    #[cfg(feature = "network")]
    fn download_once(
        &self,
        url: &str,
        repo_id: &str,
        raw_revision: &str,
        destination: &Path,
        expected: &ArtifactFileRow,
    ) -> Result<()> {
        validate_base_url(url, self.inner.allow_insecure_loopback)?;
        let revision = Revision::commit(raw_revision.to_owned())?;
        // A bearer credential is never forwarded by this client. Authenticated
        // repositories must return the object directly; callers can configure
        // the final HTTPS mirror URL if their service normally redirects.
        let redirects =
            if self.inner.secret_provider.is_some() || self.inner.allow_insecure_loopback {
                0
            } else {
                self.inner.redirects
            };
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(self.inner.timeout))
            .https_only(!self.inner.allow_insecure_loopback)
            .max_redirects(redirects)
            .proxy(None)
            .build()
            .into();
        let mut request = agent.get(url).header(
            "User-Agent",
            concat!("toktier-rust/", env!("CARGO_PKG_VERSION")),
        );
        if let Some(provider) = &self.inner.secret_provider {
            if let Some(token) = provider.bearer_token(repo_id, &revision)? {
                request = request.header("Authorization", format!("Bearer {}", token.expose()));
            }
        }
        let mut response = request.call().map_err(|error| {
            Error::new(
                ErrorCode::Network,
                format!("artifact request failed for {repo_id}@{raw_revision}: {error}"),
            )
        })?;
        if let Some(length) = response
            .headers()
            .get("content-length")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
        {
            if length != expected.size {
                return Err(Error::new(
                    ErrorCode::ArtifactSizeMismatch,
                    format!(
                        "remote object declares {length} bytes; manifest requires {}",
                        expected.size
                    ),
                ));
            }
        }
        write_reader_checked(response.body_mut().as_reader(), destination, expected)
    }

    #[cfg(not(feature = "network"))]
    fn download_checked(
        &self,
        _url: &str,
        repo_id: &str,
        raw_revision: &str,
        _destination: &Path,
        _expected: &ArtifactFileRow,
    ) -> Result<()> {
        Err(Error::new(
            ErrorCode::NetworkDisabled,
            format!(
                "network acquisition for {repo_id}@{raw_revision} requires the `network` \
                 feature, which is not in this crate's default set; rebuild with \
                 `--features network`, or supply the artifact from a verified cache, \
                 a local directory, or an air-gap bundle"
            ),
        ))
    }
}

fn prune_stale_temporary_files(directory: &Path, maximum_age: Duration) -> Result<()> {
    if maximum_age.is_zero() || !directory.exists() {
        return Ok(());
    }
    let now = std::time::SystemTime::now();
    for entry in fs::read_dir(directory)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(directory))?
    {
        let entry = entry.map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
        let path = entry.path();
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with('.') || !name.ends_with(".part") {
            continue;
        }
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&path))?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            continue;
        }
        let stale = metadata
            .modified()
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age >= maximum_age);
        if stale {
            fs::remove_file(&path)
                .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
        }
    }
    Ok(())
}

fn source_name(source: &ArtifactSource) -> &'static str {
    match source {
        ArtifactSource::HuggingFace => "huggingface",
        ArtifactSource::Mirror { .. } => "mirror",
        ArtifactSource::LocalDirectory { .. } => "local_directory",
        ArtifactSource::None => "none",
    }
}

fn directory_name(family: &str, revision: &str) -> String {
    format!("{family}-{}", &revision[..12])
}

fn validate_entry(family: &str, row: &ArtifactRow) -> Result<()> {
    if family.is_empty()
        || !family
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        || row.repo_id.is_empty()
        || row.repo_id.starts_with('/')
        || row.repo_id.contains("..")
        || row.repo_id.contains(['\\', '\0', '\r', '\n'])
        || row.repo_id.split('/').any(|segment| {
            segment.is_empty()
                || !segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        })
    {
        return Err(Error::new(
            ErrorCode::RegistryInvalid,
            "artifact manifest contains an unsafe family or repository path",
        ));
    }
    Revision::commit(row.revision.clone())?;
    if row.files.is_empty() {
        return Err(Error::new(
            ErrorCode::RegistryInvalid,
            "artifact manifest entry has no files",
        ));
    }
    for name in row.files.keys() {
        checked_relative(name)?;
    }
    Ok(())
}

fn validate_base_url(url: &str, allow_loopback: bool) -> Result<()> {
    if url.contains(['\0', '\r', '\n', '\\', '@', '#', '?']) {
        return Err(Error::new(
            ErrorCode::ConfigInvalid,
            "artifact URL contains credentials, controls, or a non-base URL component",
        ));
    }
    if url.starts_with("https://") {
        return Ok(());
    }
    if allow_loopback
        && (url.starts_with("http://127.0.0.1:")
            || url.starts_with("http://[::1]:")
            || url.starts_with("http://localhost:"))
    {
        return Ok(());
    }
    Err(Error::new(
        ErrorCode::ConfigInvalid,
        "artifact URLs must use HTTPS; plaintext HTTP is limited to explicit loopback tests",
    ))
}

fn checked_relative(name: &str) -> Result<&Path> {
    let path = Path::new(name);
    if name.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
        || name.contains(['\\', '\0'])
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b'/'))
    {
        return Err(Error::new(
            ErrorCode::RegistryInvalid,
            format!("unsafe artifact-relative path {name:?}"),
        ));
    }
    Ok(path)
}

fn checked_join(root: &Path, name: &str) -> Result<PathBuf> {
    Ok(root.join(checked_relative(name)?))
}

fn checked_file(root: &Path, name: &str) -> Result<PathBuf> {
    let path = checked_join(root, name)?;
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        Error::new(ErrorCode::ArtifactNotFound, error.to_string()).with_path(&path)
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(Error::new(
            ErrorCode::ArtifactNotFound,
            "artifact member must be a regular non-symlink file",
        )
        .with_path(path));
    }
    Ok(path)
}

fn verify_directory(directory: &Path, row: &ArtifactRow) -> Result<()> {
    let metadata = fs::symlink_metadata(directory).map_err(|error| {
        Error::new(ErrorCode::ArtifactNotFound, error.to_string()).with_path(directory)
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Error::new(
            ErrorCode::ArtifactNotFound,
            "artifact directory must be a regular non-symlink directory",
        )
        .with_path(directory));
    }
    for (name, expected) in &row.files {
        verify_file(&checked_file(directory, name)?, expected)?;
    }
    Ok(())
}

fn verify_file(path: &Path, expected: &ArtifactFileRow) -> Result<()> {
    let (digest, size) = sha256_file(path)?;
    if size != expected.size {
        return Err(Error::new(
            ErrorCode::ArtifactSizeMismatch,
            format!(
                "{} has {size} bytes; expected {}",
                path.display(),
                expected.size
            ),
        )
        .with_path(path));
    }
    if digest != expected.sha256 {
        return Err(Error::new(
            ErrorCode::ArtifactHashMismatch,
            format!(
                "{} has sha256 {digest}; expected {}",
                path.display(),
                expected.sha256
            ),
        )
        .with_path(path));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let file = File::open(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    hash_reader(file)
}

fn hash_reader(mut reader: impl Read) -> Result<(String, u64)> {
    let mut digest = Sha256::new();
    let mut length = 0u64;
    let mut buffer = vec![0u8; READ_CHUNK];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(count as u64)
            .ok_or_else(|| Error::new(ErrorCode::ArtifactSizeMismatch, "artifact size overflow"))?;
        digest.update(&buffer[..count]);
    }
    Ok((hex(&digest.finalize()), length))
}

fn copy_stream_checked(
    source: &Path,
    destination: &Path,
    expected: &ArtifactFileRow,
) -> Result<()> {
    let metadata = fs::symlink_metadata(source).map_err(|error| {
        Error::new(ErrorCode::ArtifactNotFound, error.to_string()).with_path(source)
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(Error::new(
            ErrorCode::ArtifactNotFound,
            "local source must be a regular non-symlink file",
        )
        .with_path(source));
    }
    let reader = File::open(source)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(source))?;
    write_reader_checked(reader, destination, expected)
}

fn write_reader_checked(
    mut reader: impl Read,
    destination: &Path,
    expected: &ArtifactFileRow,
) -> Result<()> {
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(destination))?;
    set_private_file(&output)?;
    let mut digest = Sha256::new();
    let mut length = 0u64;
    let mut buffer = vec![0u8; READ_CHUNK];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| Error::new(ErrorCode::Network, error.to_string()))?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(count as u64)
            .ok_or_else(|| Error::new(ErrorCode::ArtifactSizeMismatch, "artifact size overflow"))?;
        if length > expected.size {
            return Err(Error::new(
                ErrorCode::ArtifactSizeMismatch,
                "download exceeded the manifest-declared size",
            ));
        }
        digest.update(&buffer[..count]);
        output
            .write_all(&buffer[..count])
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
    }
    output
        .sync_all()
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
    let observed = hex(&digest.finalize());
    if length != expected.size {
        return Err(Error::new(
            ErrorCode::ArtifactSizeMismatch,
            format!(
                "received {length} bytes; manifest requires {}",
                expected.size
            ),
        ));
    }
    if observed != expected.sha256 {
        return Err(Error::new(
            ErrorCode::ArtifactHashMismatch,
            format!(
                "received sha256 {observed}; manifest requires {}",
                expected.sha256
            ),
        ));
    }
    Ok(())
}

fn unique_temporary(directory: &Path, name: &str) -> Result<PathBuf> {
    crate::fsutil::unique_temporary(
        directory,
        name,
        "part",
        "cannot allocate a private artifact staging path",
    )
}

fn quarantine(root: &Path, family: &str, target: &Path) -> Result<()> {
    let quarantine = root.join(".quarantine");
    ensure_private_directory(&quarantine)?;
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    let destination = quarantine.join(format!(
        "{family}-{file_name}-{}-{}",
        std::process::id(),
        monotonic_nonce()
    ));
    fs::rename(target, &destination)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(target))?;
    sync_directory(&quarantine)
}

fn copy_atomic(source: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = target.parent() {
        ensure_private_directory(parent)?;
    }
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("copy");
    let temporary = unique_temporary(parent, name)?;
    let expected = ArtifactFileRow {
        sha256: sha256_file(source)?.0,
        size: fs::metadata(source)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?
            .len(),
    };
    copy_stream_checked(source, &temporary, &expected)?;
    fs::rename(&temporary, target)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(target))?;
    sync_directory(parent)
}

fn write_marker(directory: &Path, family: &str, row: &ArtifactRow) -> Result<()> {
    let tokenizer = row
        .files
        .get("tokenizer.json")
        .ok_or_else(|| Error::new(ErrorCode::RegistryInvalid, "artifact has no tokenizer.json"))?;
    let payload = serde_json::to_vec_pretty(&Marker {
        schema: MARKER_SCHEMA.to_owned(),
        family: family.to_owned(),
        revision: row.revision.clone(),
        tokenizer_sha256: tokenizer.sha256.clone(),
        tokenizer_size: tokenizer.size,
    })
    .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?;
    let target = directory.join(MARKER_NAME);
    let temporary = unique_temporary(directory, MARKER_NAME)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
    set_private_file(&file)?;
    file.write_all(&payload)
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.sync_all())
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
    fs::rename(&temporary, &target)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
    Ok(())
}

fn lock_exclusive_bounded(file: &File, timeout: Duration) -> Result<()> {
    crate::fsutil::lock_exclusive_bounded(file, timeout, "artifact cache")
}

fn allocate_staging_directory(root: &Path, prefix: &str) -> Result<PathBuf> {
    for nonce in 0..1024u32 {
        let path = root.join(format!("{prefix}.{}.{}", std::process::id(), nonce));
        match fs::create_dir(&path) {
            Ok(()) => {
                ensure_private_directory(&path)?;
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(Error::new(ErrorCode::Io, error.to_string()).with_path(path));
            }
        }
    }
    Err(Error::new(
        ErrorCode::CacheBusy,
        "cannot allocate artifact metadata staging directory",
    ))
}

fn ensure_private_directory(path: &Path) -> Result<()> {
    crate::fsutil::ensure_private_directory(
        path,
        "cache",
        "cache path must be a non-symlink directory",
    )
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "network")]
    use std::io::{Read as _, Write as _};
    use std::net::TcpListener;
    #[cfg(feature = "network")]
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(feature = "network")]
    use std::sync::Arc;

    use super::*;

    #[cfg(feature = "network")]
    fn serve(listener: TcpListener, payload: &'static [u8], requests: Arc<AtomicUsize>) {
        for response_index in 0..2 {
            let (mut stream, _) = listener.accept().expect("accept loopback request");
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request).expect("read request");
            requests.fetch_add(1, Ordering::SeqCst);
            if response_index == 0 {
                stream
                    .write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .expect("write retry response");
            } else {
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    payload.len()
                )
                .expect("write headers");
                stream.write_all(payload).expect("write body");
            }
        }
    }

    /// A family whose artifact is produced locally answers with what it
    /// is, before opening a socket. It used to ask the hub for a file
    /// the pinned revision does not publish and report the 404 as
    /// `NETWORK_ERROR`, which a Python-free consumer could neither act
    /// on nor route around.
    #[test]
    fn a_derived_family_says_so_instead_of_asking_the_hub() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let address = listener.local_addr().expect("listener address");
        // Nothing ever accepts on this listener: reaching it would hang
        // the test rather than pass it.
        let temporary = tempfile::tempdir().expect("temporary directory");
        let manager = ArtifactManager::builder()
            .cache(temporary.path().join("cache"))
            .source(ArtifactSource::Mirror {
                base_url: format!("http://{address}"),
            })
            .allow_insecure_loopback(true)
            .build()
            .expect("manager");
        let registry = crate::manifest::Registry::load().expect("registry");
        let derived = registry
            .conversion("kimi_k3")
            .expect("the shipped table names this family");

        let error = manager
            .ensure("kimi_k3", &registry)
            .expect_err("a derived artifact cannot be downloaded");

        assert_eq!(error.code(), ErrorCode::ArtifactNotFound);
        let message = error.message();
        assert!(message.contains(&derived.converter), "{message}");
        assert!(message.contains("derived locally"), "{message}");
        assert!(message.contains("toktier artifacts fetch"), "{message}");
        assert!(message.contains("load_local"), "{message}");
        for input in &derived.inputs {
            assert!(message.contains(&input.name), "{message}");
        }
        drop(listener);
    }

    #[cfg(feature = "network")]
    #[test]
    fn network_retries_transport_failures_and_auth_is_redacted() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let address = listener.local_addr().expect("listener address");
        let requests = Arc::new(AtomicUsize::new(0));
        let server_requests = Arc::clone(&requests);
        let server = std::thread::spawn(move || serve(listener, b"exact bytes", server_requests));
        let temporary = tempfile::tempdir().expect("temporary directory");
        let destination = temporary.path().join("object.part");
        let mut digest = Sha256::new();
        digest.update(b"exact bytes");
        let expected = ArtifactFileRow {
            sha256: hex(&digest.finalize()),
            size: 11,
        };
        let manager = ArtifactManager::builder()
            .cache(temporary.path().join("cache"))
            .source(ArtifactSource::Mirror {
                base_url: format!("http://{address}"),
            })
            .allow_insecure_loopback(true)
            .max_retries(1)
            .retry_backoff(Duration::from_millis(1))
            .build()
            .expect("manager");
        manager
            .download_checked(
                &format!("http://{address}/object"),
                "repo/id",
                "0123456789abcdef0123456789abcdef01234567",
                &destination,
                &expected,
            )
            .expect("retry succeeds");
        server.join().expect("server joins");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(fs::read(destination).expect("read object"), b"exact bytes");
    }

    /// A derived family has nothing to download, so being offline is not
    /// what stands in the way -- and an air-gapped reader is exactly the
    /// one who needs to be told which two routes carry the bytes. The
    /// offline gate used to answer first and leave them with the general
    /// "acquisition is offline" line.
    #[test]
    fn an_offline_derived_family_still_gets_the_conversion_answer() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let manager = ArtifactManager::builder()
            .cache(temporary.path().join("cache"))
            .source(ArtifactSource::None)
            .offline(true)
            .build()
            .expect("manager");

        let refusal = manager.fetch("kimi_k3").expect_err("derived family");

        assert_eq!(refusal.code(), ErrorCode::ArtifactNotFound);
        let message = refusal.to_string();
        assert!(message.contains("derived locally by the"), "{message}");
        assert!(
            message.contains("nothing for this crate to download"),
            "{message}"
        );
        // Both routes that put the bytes on an air-gapped machine.
        assert!(message.contains("air-gap bundle"), "{message}");
        assert!(message.contains("load_local"), "{message}");
        assert!(!message.contains("acquisition is offline"), "{message}");
    }

    #[test]
    fn offline_gate_opens_no_socket_and_stale_parts_are_reclaimed() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        listener.set_nonblocking(true).expect("nonblocking");
        let address = listener.local_addr().expect("listener address");
        let temporary = tempfile::tempdir().expect("temporary directory");
        let cache = temporary.path().join("cache");
        let manager = ArtifactManager::builder()
            .cache(&cache)
            .source(ArtifactSource::Mirror {
                base_url: format!("http://{address}"),
            })
            .allow_insecure_loopback(true)
            .offline(true)
            .build()
            .expect("manager");
        let refusal = manager.fetch("qwen3_8b").expect_err("offline miss");
        assert_eq!(refusal.code(), ErrorCode::ArtifactNotFound);
        // A family published whole: offline is the whole of the answer.
        assert!(
            refusal.to_string().contains("acquisition is offline"),
            "{refusal}"
        );
        assert_eq!(
            listener.accept().expect_err("no request").kind(),
            std::io::ErrorKind::WouldBlock
        );

        let object = cache.join("parts");
        ensure_private_directory(&object).expect("part directory");
        let stale = object.join(".tokenizer.json.1.0.part");
        fs::write(&stale, b"partial").expect("part file");
        std::thread::sleep(Duration::from_millis(2));
        prune_stale_temporary_files(&object, Duration::from_millis(1)).expect("prune");
        assert!(!stale.exists());
    }
}
