//! Shared filesystem and cache-safety helpers.
//!
//! The crate keeps three private cache surfaces (verified artifacts,
//! air-gap bundles, and JIT compiler products). Their security-sensitive
//! primitives are identical by design; this module is their single
//! implementation so the three surfaces cannot drift apart. Message
//! wording stays caller-owned: each surface passes its established
//! vocabulary (for example `"cache"`, `"bundle"`, `"JIT cache"`) so the
//! reported errors keep their existing text.

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Duration;

use fs2::FileExt;

use crate::{Error, ErrorCode, Result};

/// Refuse parent-directory, symlink, and non-directory components on the
/// way to `path`. `label` names the caller's surface in the messages.
pub(crate) fn reject_symlink_components(path: &Path, label: &str) -> Result<()> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?
            .join(path)
    };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(Error::new(
                ErrorCode::Io,
                format!("{label} paths may not contain parent-directory components"),
            )
            .with_path(path));
        }
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(Error::new(
                    ErrorCode::Io,
                    format!("{label} path contains a symbolic-link component"),
                )
                .with_path(current));
            }
            Ok(metadata) if current != absolute && !metadata.is_dir() => {
                return Err(Error::new(
                    ErrorCode::Io,
                    format!("{label} path contains a non-directory component"),
                )
                .with_path(current));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(Error::new(ErrorCode::Io, error.to_string()).with_path(current));
            }
        }
    }
    Ok(())
}

/// Create `path` as a private (0o700) directory, re-checking the component
/// chain after creation (TOCTOU mitigation). `not_directory` is the exact
/// message reported when the final path is not a regular directory.
pub(crate) fn ensure_private_directory(
    path: &Path,
    label: &str,
    not_directory: &str,
) -> Result<()> {
    reject_symlink_components(path, label)?;
    fs::create_dir_all(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    reject_symlink_components(path, label)?;
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(Error::new(ErrorCode::Io, not_directory).with_path(path));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    }
    Ok(())
}

pub(crate) fn set_private_file(file: &File) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
    }
    Ok(())
}

pub(crate) fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))
}

/// Open (creating if needed) a private lock file without truncating it.
pub(crate) fn open_private_lock(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    set_private_file(&file)?;
    Ok(file)
}

/// Acquire an exclusive advisory lock, polling until `timeout`. `lock_name`
/// names the lock in the timeout message (for example `"artifact cache"`).
pub(crate) fn lock_exclusive_bounded(
    file: &File,
    timeout: Duration,
    lock_name: &str,
) -> Result<()> {
    let started = std::time::Instant::now();
    loop {
        match FileExt::try_lock_exclusive(file) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if started.elapsed() >= timeout {
                    return Err(Error::new(
                        ErrorCode::CacheBusy,
                        format!("{lock_name} lock exceeded {timeout:?}"),
                    ));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(Error::new(ErrorCode::CacheBusy, error.to_string())),
        }
    }
}

/// Pick an unused hidden staging path `.{name}.{pid}.{nonce}.{suffix}`.
pub(crate) fn unique_temporary(
    directory: &Path,
    name: &str,
    suffix: &str,
    busy_message: &'static str,
) -> Result<PathBuf> {
    for nonce in 0..1024u32 {
        let path = directory.join(format!(".{name}.{}.{}.{suffix}", std::process::id(), nonce));
        if !path.exists() {
            return Ok(path);
        }
    }
    Err(Error::new(ErrorCode::CacheBusy, busy_message))
}

pub(crate) fn monotonic_nonce() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Resolve a cache directory: explicit environment override, then the
/// user cache home, then a relative in-tree fallback.
pub(crate) fn default_cache_directory(env_var: &str, subdirectory: &str) -> PathBuf {
    std::env::var_os(env_var)
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".cache/toktier").join(subdirectory))
        })
        .unwrap_or_else(|| PathBuf::from(".toktier").join(subdirectory))
}
