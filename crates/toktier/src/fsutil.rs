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
///
/// The three refusals are policy decisions about a configured location,
/// not failures of an attempted operation, so they report
/// [`ErrorCode::ConfigInvalid`]; a real filesystem failure met while
/// walking the chain still reports [`ErrorCode::Io`].
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
                ErrorCode::ConfigInvalid,
                format!("{label} paths may not contain parent-directory components"),
            )
            .with_path(path));
        }
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(Error::new(
                    ErrorCode::ConfigInvalid,
                    format!("{label} path contains a symbolic-link component"),
                )
                .with_path(current));
            }
            Ok(metadata) if current != absolute && !metadata.is_dir() => {
                return Err(Error::new(
                    ErrorCode::ConfigInvalid,
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
        return Err(Error::new(ErrorCode::ConfigInvalid, not_directory).with_path(path));
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

/// The application directory name shared with the Python product.
const APPLICATION: &str = "toktier";

/// Read an environment variable, treating an empty value as unset --
/// the rule the XDG specification states and the Python product
/// follows.
fn present(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The user cache root for this application, by the same precedence the
/// Python product uses: `TOKTIER_HOME/cache`, then `XDG_CACHE_HOME`,
/// then `$HOME/.cache` (the layout platformdirs resolves to on Linux).
fn cache_root_from(lookup: impl Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
    lookup("TOKTIER_HOME")
        .map(|home| home.join("cache"))
        .or_else(|| lookup("XDG_CACHE_HOME").map(|base| base.join(APPLICATION)))
        .or_else(|| lookup("HOME").map(|home| home.join(".cache").join(APPLICATION)))
}

/// The user state root for this application: `TOKTIER_HOME/state`, then
/// `XDG_STATE_HOME`, then `$HOME/.local/state`.
fn state_root_from(lookup: impl Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
    lookup("TOKTIER_HOME")
        .map(|home| home.join("state"))
        .or_else(|| lookup("XDG_STATE_HOME").map(|base| base.join(APPLICATION)))
        .or_else(|| lookup("HOME").map(|home| home.join(".local").join("state").join(APPLICATION)))
}

/// Where session state goes when the caller did not name a home.
///
/// State is not a cache -- deleting it loses sessions -- so there is no
/// in-tree fallback: a host with none of these variables set gets no
/// implicit state directory, and persistent sessions there stay a
/// configuration error rather than a surprise directory.
pub(crate) fn default_state_directory() -> Option<PathBuf> {
    state_root_from(present)
}

/// Resolve a cache directory: the surface's own environment override
/// first (it names one directory exactly), then the user cache root,
/// then a relative in-tree fallback.
pub(crate) fn default_cache_directory(env_var: &str, subdirectory: &str) -> PathBuf {
    std::env::var_os(env_var)
        .map(PathBuf::from)
        .or_else(|| cache_root_from(present).map(|root| root.join(subdirectory)))
        .unwrap_or_else(|| PathBuf::from(".toktier").join(subdirectory))
}

#[cfg(test)]
mod tests {
    use super::{cache_root_from, state_root_from};
    use std::path::PathBuf;

    /// The resolution rules are tested through their lookup, not by
    /// setting variables in this process: other tests read the same
    /// environment while these run.
    fn environment(pairs: &[(&'static str, &'static str)]) -> impl Fn(&str) -> Option<PathBuf> {
        let pairs = pairs.to_vec();
        move |name: &str| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| *value)
                // An empty value is unset, as the XDG specification says
                // and the Python product treats it.
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        }
    }

    /// One case: the variables in force, the cache root they resolve,
    /// and the state root they resolve.
    type Case = (
        &'static [(&'static str, &'static str)],
        Option<&'static str>,
        Option<&'static str>,
    );

    #[test]
    fn the_roots_follow_the_python_precedence() {
        let cases: [Case; 5] = [
            (
                &[
                    ("TOKTIER_HOME", "/tmp/home"),
                    ("XDG_CACHE_HOME", "/tmp/xdg"),
                    ("XDG_STATE_HOME", "/tmp/xdg-state"),
                    ("HOME", "/tmp/user"),
                ],
                Some("/tmp/home/cache"),
                Some("/tmp/home/state"),
            ),
            (
                &[
                    ("XDG_CACHE_HOME", "/tmp/xdg"),
                    ("XDG_STATE_HOME", "/tmp/xdg-state"),
                    ("HOME", "/tmp/user"),
                ],
                Some("/tmp/xdg/toktier"),
                Some("/tmp/xdg-state/toktier"),
            ),
            (
                &[
                    ("TOKTIER_HOME", ""),
                    ("XDG_CACHE_HOME", ""),
                    ("XDG_STATE_HOME", ""),
                    ("HOME", "/tmp/user"),
                ],
                Some("/tmp/user/.cache/toktier"),
                Some("/tmp/user/.local/state/toktier"),
            ),
            (
                &[("HOME", "/tmp/user")],
                Some("/tmp/user/.cache/toktier"),
                Some("/tmp/user/.local/state/toktier"),
            ),
            (&[], None, None),
        ];
        for (pairs, cache, state) in cases {
            let lookup = environment(pairs);
            assert_eq!(cache_root_from(&lookup), cache.map(PathBuf::from));
            assert_eq!(state_root_from(&lookup), state.map(PathBuf::from));
        }
    }
}
