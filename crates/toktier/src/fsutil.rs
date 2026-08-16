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

/// Create `path` and every missing directory on the way to it, each with
/// owner-only permissions.
///
/// `create_dir_all` leaves the intermediate directories at the process
/// umask and only the caller's final `set_permissions` makes the last one
/// private, so a fresh cache root used to arrive as `0775/0775/0700`.
/// `config.md` section 5 offers 0700 for every directory this layer
/// creates, so the components are created one at a time. Directories that
/// were already there are left exactly as the operator has them.
pub(crate) fn create_private_dir_all(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

        let mut current = PathBuf::new();
        for component in path.components() {
            current.push(component.as_os_str());
            let mut builder = fs::DirBuilder::new();
            builder.mode(0o700);
            match builder.create(&current) {
                // The mode is set again explicitly: `DirBuilder::mode`
                // still goes through the umask.
                Ok(()) => fs::set_permissions(&current, fs::Permissions::from_mode(0o700))
                    .map_err(|error| {
                        Error::new(ErrorCode::Io, error.to_string()).with_path(&current)
                    })?,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(Error::new(ErrorCode::Io, error.to_string()).with_path(&current));
                }
            }
        }
        // Keep `create_dir_all`'s judgement of a pre-existing final path
        // without touching any directory this call did not create.
        fs::create_dir_all(path)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
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
    // Judge an existing final path before trying to create it. Creation
    // over a regular file fails with "already exists", which is a true
    // statement about the syscall and a misleading one about the
    // configuration: the location cannot be a private directory, and
    // that is the answer to give.
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Error::new(ErrorCode::ConfigInvalid, not_directory).with_path(path));
        }
    }
    create_private_dir_all(path)?;
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

/// Write `bytes` to `path` as an owner-only file, replacing whatever was
/// there.
///
/// Unlike the JIT cache's writer this one replaces: a local verification
/// record is the latest answer for one combination, not an immutable
/// product, and the combination it is about is already in its name.
pub(crate) fn write_private_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension("tmp");
    let _ = fs::remove_file(&temporary);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&temporary))?;
    set_private_file(&file)?;
    use std::io::Write;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&temporary))?;
    fs::rename(&temporary, path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))
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
///
/// The override is read through [`present`] like every other variable
/// here, so an empty value counts as unset for all of them rather than
/// naming a relative directory under the working directory.
pub(crate) fn default_cache_directory(env_var: &str, subdirectory: &str) -> PathBuf {
    cache_directory_from(present, env_var, subdirectory)
}

fn cache_directory_from(
    lookup: impl Fn(&str) -> Option<PathBuf> + Copy,
    env_var: &str,
    subdirectory: &str,
) -> PathBuf {
    lookup(env_var)
        .or_else(|| cache_root_from(lookup).map(|root| root.join(subdirectory)))
        .unwrap_or_else(|| PathBuf::from(".toktier").join(subdirectory))
}

#[cfg(test)]
mod tests {
    use super::{cache_directory_from, cache_root_from, ensure_private_directory, state_root_from};
    use std::path::PathBuf;

    /// Every directory this layer creates is owner-only, the ones on the
    /// way included: `create_dir_all` leaves those at the process umask,
    /// which `config.md` section 5 does not offer.
    #[cfg(unix)]
    #[test]
    fn intermediate_directories_are_created_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "toktier-private-dirs-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let leaf = root.join("cache").join("artifacts").join(".locks");
        ensure_private_directory(&leaf, "cache", "cache path is not a directory")
            .expect("private directory");

        let mut current = leaf.as_path();
        loop {
            let mode = std::fs::metadata(current)
                .expect("created directory")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o700, "{}", current.display());
            if current == root {
                break;
            }
            current = current.parent().expect("parent inside the root");
        }
        std::fs::remove_dir_all(&root).expect("clean up");
    }

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
    fn the_surface_variable_counts_as_unset_when_empty() {
        // Three states of one variable: unset and empty must resolve
        // the same directory, and a named one must still win. The
        // interesting pair is the one that looks alike from outside.
        let unset = environment(&[("HOME", "/tmp/user")]);
        let empty = environment(&[("TOKTIER_ARTIFACT_CACHE", ""), ("HOME", "/tmp/user")]);
        let named = environment(&[
            ("TOKTIER_ARTIFACT_CACHE", "/tmp/explicit"),
            ("HOME", "/tmp/user"),
        ]);
        let resolve = |lookup: &dyn Fn(&str) -> Option<PathBuf>| {
            cache_directory_from(lookup, "TOKTIER_ARTIFACT_CACHE", "artifacts")
        };
        assert_eq!(
            resolve(&unset),
            PathBuf::from("/tmp/user/.cache/toktier/artifacts")
        );
        assert_eq!(resolve(&empty), resolve(&unset));
        assert_eq!(resolve(&named), PathBuf::from("/tmp/explicit"));

        // The same rule for the JIT cache, and the in-tree fallback when
        // nothing at all resolves.
        let nothing = environment(&[("TOKTIER_JIT_CACHE", "")]);
        assert_eq!(
            cache_directory_from(&nothing, "TOKTIER_JIT_CACHE", "jit-rust"),
            PathBuf::from(".toktier/jit-rust")
        );
    }

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
