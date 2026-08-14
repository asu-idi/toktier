//! Canonical, Python-v1-compatible air-gap bundle export and import.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::fsutil::{hex, set_private_file, sync_directory};
use crate::{Error, ErrorCode, Result};

const MANIFEST_NAME: &str = "bundle_manifest.json";
const ROOT_DOMAIN: &[u8] = b"toktier.bundle.v1\0";
const MAX_MEMBERS: usize = 4096;
const MAX_UNCOMPRESSED: u64 = 8 * 1024 * 1024 * 1024;
const MAX_MANIFEST: u64 = 16 * 1024 * 1024;
const COPY_BUFFER: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleFile {
    path: String,
    sha256: String,
    size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleManifest {
    alias: String,
    files: Vec<BundleFile>,
    root_digest: String,
}

/// Facts authenticated by a bundle's domain-separated root digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct BundleInspection {
    pub alias: String,
    pub files: Vec<BundleFileInspection>,
    pub root_digest: String,
    pub total_size: u64,
}

/// One regular file declared by an air-gap bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct BundleFileInspection {
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

/// Write a deterministic v1 bundle from verified regular files.
///
/// The map keys are portable, bundle-relative POSIX paths. The archive is
/// compatible with `toktier artifacts import` in the Python distribution.
pub fn export_bundle(
    bundle: impl AsRef<Path>,
    alias: &str,
    files: &BTreeMap<String, PathBuf>,
) -> Result<BundleInspection> {
    let bundle = bundle.as_ref();
    validate_relative(alias, "bundle alias")?;
    if files.is_empty() {
        return Err(bundle_error(
            "bundle must contain at least one file",
            bundle,
        ));
    }
    if files.len() + 1 > MAX_MEMBERS {
        return Err(bundle_error("bundle member limit exceeded", bundle));
    }

    let mut manifest_files = Vec::with_capacity(files.len());
    let mut total = 0u64;
    for (name, source) in files {
        validate_relative(name, "bundle member")?;
        if name == MANIFEST_NAME {
            return Err(bundle_error("bundle manifest name is reserved", bundle));
        }
        let metadata = fs::symlink_metadata(source).map_err(|error| {
            Error::new(ErrorCode::ArtifactNotFound, error.to_string()).with_path(source)
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(Error::new(
                ErrorCode::ArtifactNotFound,
                "bundle sources must be regular non-symlink files",
            )
            .with_path(source));
        }
        let (sha256, size) = hash_file(source)?;
        total = total
            .checked_add(size)
            .ok_or_else(|| bundle_error("bundle size overflow", bundle))?;
        if total > MAX_UNCOMPRESSED {
            return Err(bundle_error("bundle exceeds the 8 GiB limit", bundle));
        }
        manifest_files.push(BundleFile {
            path: name.clone(),
            sha256,
            size,
        });
    }
    manifest_files.sort_by(|left, right| left.path.cmp(&right.path));
    let root_digest = root_digest(alias, &manifest_files)?;
    let manifest = BundleManifest {
        alias: alias.to_owned(),
        files: manifest_files,
        root_digest,
    };
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| bundle_error(format!("cannot encode manifest: {error}"), bundle))?;
    manifest_bytes.push(b'\n');

    let parent = bundle.parent().unwrap_or_else(|| Path::new("."));
    ensure_private_directory(parent)?;
    let temporary = unique_temporary(
        parent,
        bundle
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("bundle"),
    )?;
    let result: Result<()> = (|| {
        let output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&temporary))?;
        set_private_file(&output)?;
        let mut archive = tar::Builder::new(output);
        archive.mode(tar::HeaderMode::Deterministic);
        append_bytes(&mut archive, MANIFEST_NAME, &manifest_bytes)?;
        for row in &manifest.files {
            let source = files.get(&row.path).expect("manifest came from map");
            append_file(&mut archive, row, source)?;
        }
        let output = archive
            .into_inner()
            .map_err(|error| bundle_error(format!("cannot finish archive: {error}"), bundle))?;
        output
            .sync_all()
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&temporary))?;
        // Re-open and authenticate the exact bytes before publication.
        inspect_bundle(&temporary)?;
        fs::rename(&temporary, bundle)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(bundle))?;
        sync_directory(parent)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result?;
    inspection(&manifest)
}

/// Verify an archive and all declared file bytes without installing it.
pub fn inspect_bundle(bundle: impl AsRef<Path>) -> Result<BundleInspection> {
    let bundle = bundle.as_ref();
    let manifest = read_and_validate_manifest(bundle)?;
    verify_archive_files(bundle, &manifest, None)?;
    inspection(&manifest)
}

/// Verify a v1 archive into private staging and atomically install its alias.
pub fn import_bundle(bundle: impl AsRef<Path>, cache_root: impl AsRef<Path>) -> Result<PathBuf> {
    let bundle = bundle.as_ref();
    let root = cache_root.as_ref();
    let manifest = read_and_validate_manifest(bundle)?;
    ensure_private_directory(root)?;
    let staging = unique_directory(root, ".toktier-bundle-import")?;
    let target = safe_join(root, &manifest.alias)?;
    let result = (|| {
        verify_archive_files(bundle, &manifest, Some(&staging))?;
        sync_tree(&staging)?;
        if target.exists() {
            // Duplicate imports are idempotent only if the visible tree still
            // authenticates as the exact bundle contents.
            verify_installed(&target, &manifest)?;
            fs::remove_dir_all(&staging).map_err(|error| {
                Error::new(ErrorCode::Io, error.to_string()).with_path(&staging)
            })?;
            return Ok(target.clone());
        }
        if let Some(parent) = target.parent() {
            ensure_private_directory(parent)?;
        }
        fs::rename(&staging, &target)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&target))?;
        sync_directory(target.parent().unwrap_or(root))?;
        Ok(target.clone())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn read_and_validate_manifest(bundle: &Path) -> Result<BundleManifest> {
    let metadata = fs::symlink_metadata(bundle).map_err(|error| {
        Error::new(ErrorCode::ArtifactNotFound, error.to_string()).with_path(bundle)
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(Error::new(
            ErrorCode::ArtifactNotFound,
            "bundle must be a regular non-symlink file",
        )
        .with_path(bundle));
    }
    let input = File::open(bundle)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(bundle))?;
    let mut archive = tar::Archive::new(input);
    let mut manifest_bytes = None;
    let mut names = HashSet::new();
    let mut count = 0usize;
    let mut total = 0u64;
    for entry in archive
        .entries()
        .map_err(|error| bundle_error(format!("cannot read tar: {error}"), bundle))?
    {
        let mut entry = entry
            .map_err(|error| bundle_error(format!("cannot read tar member: {error}"), bundle))?;
        count += 1;
        if count > MAX_MEMBERS {
            return Err(bundle_error("bundle member limit exceeded", bundle));
        }
        let raw = entry.path_bytes();
        let name = std::str::from_utf8(&raw)
            .map_err(|_| bundle_error("bundle member path is not UTF-8", bundle))?
            .trim_end_matches('/')
            .to_owned();
        validate_relative(&name, "bundle member")?;
        if !names.insert(name.clone()) {
            return Err(bundle_error(
                format!("duplicate bundle member {name:?}"),
                bundle,
            ));
        }
        if entry.header().entry_type().is_symlink() || entry.header().entry_type().is_hard_link() {
            return Err(bundle_error(
                format!("link member {name:?} is forbidden"),
                bundle,
            ));
        }
        if !(entry.header().entry_type().is_file() || entry.header().entry_type().is_dir()) {
            return Err(bundle_error(
                format!("special member {name:?} is forbidden"),
                bundle,
            ));
        }
        let size = entry.size();
        total = total
            .checked_add(size)
            .ok_or_else(|| bundle_error("bundle size overflow", bundle))?;
        if total > MAX_UNCOMPRESSED {
            return Err(bundle_error("bundle exceeds the 8 GiB limit", bundle));
        }
        if name == MANIFEST_NAME {
            if !entry.header().entry_type().is_file() || size > MAX_MANIFEST {
                return Err(bundle_error("invalid bundle manifest member", bundle));
            }
            let mut bytes = Vec::with_capacity(size as usize);
            entry
                .read_to_end(&mut bytes)
                .map_err(|error| bundle_error(format!("cannot read manifest: {error}"), bundle))?;
            manifest_bytes = Some(bytes);
        }
    }
    let bytes =
        manifest_bytes.ok_or_else(|| bundle_error("bundle_manifest.json is missing", bundle))?;
    let manifest: BundleManifest = serde_json::from_slice(&bytes)
        .map_err(|error| bundle_error(format!("cannot parse bundle manifest: {error}"), bundle))?;
    validate_relative(&manifest.alias, "bundle alias")?;
    if manifest.files.is_empty() {
        return Err(bundle_error("bundle manifest has no files", bundle));
    }
    let mut declared = HashSet::new();
    for row in &manifest.files {
        validate_relative(&row.path, "manifest file")?;
        if row.path == MANIFEST_NAME || !declared.insert(row.path.clone()) {
            return Err(bundle_error("reserved or duplicate manifest path", bundle));
        }
        if row.sha256.len() != 64
            || !row
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(bundle_error("invalid manifest sha256", bundle));
        }
    }
    let expected = root_digest(&manifest.alias, &manifest.files)?;
    if manifest.root_digest != expected {
        return Err(bundle_error(
            format!(
                "bundle root digest mismatch: expected {expected}, observed {}",
                manifest.root_digest
            ),
            bundle,
        ));
    }
    Ok(manifest)
}

fn verify_archive_files(
    bundle: &Path,
    manifest: &BundleManifest,
    destination: Option<&Path>,
) -> Result<()> {
    if let Some(destination) = destination {
        ensure_private_directory(destination)?;
    }
    let expected = manifest
        .files
        .iter()
        .map(|row| (row.path.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let input = File::open(bundle)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(bundle))?;
    let mut archive = tar::Archive::new(input);
    let mut seen = HashSet::new();
    for entry in archive
        .entries()
        .map_err(|error| bundle_error(format!("cannot read tar: {error}"), bundle))?
    {
        let mut entry = entry
            .map_err(|error| bundle_error(format!("cannot read tar member: {error}"), bundle))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let raw = entry.path_bytes();
        let name = std::str::from_utf8(&raw)
            .map_err(|_| bundle_error("bundle member path is not UTF-8", bundle))?
            .to_owned();
        if name == MANIFEST_NAME {
            continue;
        }
        let row = expected
            .get(name.as_str())
            .ok_or_else(|| bundle_error(format!("undeclared archive file {name:?}"), bundle))?;
        if !seen.insert(name.clone()) {
            return Err(bundle_error(
                format!("duplicate archive file {name:?}"),
                bundle,
            ));
        }
        let output = match destination {
            Some(root) => {
                let target = safe_join(root, &name)?;
                if let Some(parent) = target.parent() {
                    ensure_private_directory(parent)?;
                }
                Some(
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&target)
                        .map_err(|error| {
                            Error::new(ErrorCode::Io, error.to_string()).with_path(&target)
                        })?,
                )
            }
            None => None,
        };
        copy_and_hash(&mut entry, output, row, bundle)?;
    }
    if seen.len() != expected.len() {
        let missing = expected
            .keys()
            .filter(|name| !seen.contains(**name))
            .copied()
            .collect::<Vec<_>>();
        return Err(bundle_error(
            format!("missing archive files: {missing:?}"),
            bundle,
        ));
    }
    Ok(())
}

fn copy_and_hash(
    reader: &mut impl Read,
    mut output: Option<File>,
    row: &BundleFile,
    bundle: &Path,
) -> Result<()> {
    if let Some(file) = output.as_ref() {
        set_private_file(file)?;
    }
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = vec![0u8; COPY_BUFFER];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| bundle_error(format!("cannot read member: {error}"), bundle))?;
        if count == 0 {
            break;
        }
        size = size
            .checked_add(count as u64)
            .ok_or_else(|| bundle_error("member size overflow", bundle))?;
        if size > row.size {
            return Err(Error::new(
                ErrorCode::ArtifactSizeMismatch,
                format!("bundle member {:?} exceeds declared size", row.path),
            ));
        }
        hasher.update(&buffer[..count]);
        if let Some(file) = output.as_mut() {
            file.write_all(&buffer[..count])
                .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
        }
    }
    if let Some(file) = output {
        file.sync_all()
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
    }
    let digest = hex(&hasher.finalize());
    if size != row.size {
        return Err(Error::new(
            ErrorCode::ArtifactSizeMismatch,
            format!(
                "bundle member {:?} has {size} bytes; expected {}",
                row.path, row.size
            ),
        ));
    }
    if digest != row.sha256 {
        return Err(Error::new(
            ErrorCode::ArtifactHashMismatch,
            format!(
                "bundle member {:?} has sha256 {digest}; expected {}",
                row.path, row.sha256
            ),
        ));
    }
    Ok(())
}

fn verify_installed(root: &Path, manifest: &BundleManifest) -> Result<()> {
    let expected = manifest
        .files
        .iter()
        .map(|row| row.path.as_str())
        .collect::<HashSet<_>>();
    let mut observed = HashSet::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&directory))?
        {
            let path = entry
                .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?
                .path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&path))?;
            if metadata.file_type().is_symlink() {
                return Err(bundle_error("installed bundle contains a symlink", &path));
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|_| bundle_error("installed bundle path escaped its root", &path))?;
                let name = relative
                    .to_str()
                    .ok_or_else(|| bundle_error("installed bundle path is not UTF-8", &path))?;
                if !expected.contains(name) {
                    return Err(bundle_error(
                        format!("installed bundle contains undeclared file {name:?}"),
                        &path,
                    ));
                }
                observed.insert(name.to_owned());
            } else {
                return Err(bundle_error(
                    "installed bundle contains a special file",
                    &path,
                ));
            }
        }
    }
    if observed.len() != expected.len() {
        return Err(bundle_error(
            "installed bundle is missing one or more declared files",
            root,
        ));
    }
    for row in &manifest.files {
        let path = safe_join(root, &row.path)?;
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            Error::new(ErrorCode::ArtifactNotFound, error.to_string()).with_path(&path)
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(Error::new(
                ErrorCode::ArtifactNotFound,
                "installed bundle member is not a regular file",
            )
            .with_path(path));
        }
        let (digest, size) = hash_file(&path)?;
        if size != row.size || digest != row.sha256 {
            return Err(Error::new(
                ErrorCode::ArtifactHashMismatch,
                format!("existing bundle target is not identical for {:?}", row.path),
            )
            .with_path(path));
        }
    }
    Ok(())
}

fn root_digest(alias: &str, files: &[BundleFile]) -> Result<String> {
    // This exact shape contains only strings and non-negative integers, so
    // serde_json's compact escaping and lexicographically ordered field names
    // are the RFC-8785 representation used by the Python v1 implementation.
    let mut canonical = String::from("{\"alias\":");
    canonical.push_str(
        &serde_json::to_string(alias)
            .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?,
    );
    canonical.push_str(",\"files\":[");
    for (index, row) in files.iter().enumerate() {
        if index != 0 {
            canonical.push(',');
        }
        canonical.push_str("{\"path\":");
        canonical.push_str(
            &serde_json::to_string(&row.path)
                .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?,
        );
        canonical.push_str(",\"sha256\":");
        canonical.push_str(
            &serde_json::to_string(&row.sha256)
                .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?,
        );
        canonical.push_str(",\"size\":");
        canonical.push_str(&row.size.to_string());
        canonical.push('}');
    }
    canonical.push_str("]}");
    let mut hasher = Sha256::new();
    hasher.update(ROOT_DOMAIN);
    hasher.update(canonical.as_bytes());
    Ok(format!("sha256:{}", hex(&hasher.finalize())))
}

fn inspection(manifest: &BundleManifest) -> Result<BundleInspection> {
    let total_size = manifest.files.iter().try_fold(0u64, |total, row| {
        total
            .checked_add(row.size)
            .ok_or_else(|| Error::new(ErrorCode::BundleInvalid, "bundle size overflow"))
    })?;
    Ok(BundleInspection {
        alias: manifest.alias.clone(),
        files: manifest
            .files
            .iter()
            .map(|row| BundleFileInspection {
                path: row.path.clone(),
                sha256: row.sha256.clone(),
                size: row.size,
            })
            .collect(),
        root_digest: manifest.root_digest.clone(),
        total_size,
    })
}

fn append_bytes(archive: &mut tar::Builder<File>, name: &str, bytes: &[u8]) -> Result<()> {
    let header = regular_header(name, bytes.len() as u64)?;
    archive
        .append(&header, bytes)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?;
    Ok(())
}

fn append_file(archive: &mut tar::Builder<File>, row: &BundleFile, source: &Path) -> Result<()> {
    let header = regular_header(&row.path, row.size)?;
    let mut file = File::open(source)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(source))?;
    archive
        .append(&header, &mut file)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(source))
}

fn regular_header(name: &str, size: u64) -> Result<tar::Header> {
    let mut header = tar::Header::new_gnu();
    header
        .set_path(name)
        .map_err(|error| Error::new(ErrorCode::BundleInvalid, error.to_string()))?;
    header.set_size(size);
    header.set_mode(0o600);
    header.set_mtime(0);
    header.set_uid(0);
    header.set_gid(0);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_cksum();
    Ok(header)
}

fn validate_relative(name: &str, what: &str) -> Result<()> {
    let path = Path::new(name);
    if name.is_empty()
        || name.contains('\\')
        || name.contains('\0')
        || name.contains('\r')
        || name.contains('\n')
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::new(
            ErrorCode::BundleInvalid,
            format!("unsafe {what} {name:?}"),
        ));
    }
    Ok(())
}

fn safe_join(root: &Path, name: &str) -> Result<PathBuf> {
    validate_relative(name, "bundle path")?;
    Ok(root.join(name))
}

fn hash_file(path: &Path) -> Result<(String, u64)> {
    let mut file = File::open(path)
        .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut buffer = vec![0u8; COPY_BUFFER];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(path))?;
        if count == 0 {
            break;
        }
        size = size
            .checked_add(count as u64)
            .ok_or_else(|| Error::new(ErrorCode::ArtifactSizeMismatch, "file size overflow"))?;
        hasher.update(&buffer[..count]);
    }
    Ok((hex(&hasher.finalize()), size))
}

fn ensure_private_directory(path: &Path) -> Result<()> {
    crate::fsutil::ensure_private_directory(
        path,
        "bundle",
        "directory path is not a regular directory",
    )
}

fn unique_temporary(parent: &Path, name: &str) -> Result<PathBuf> {
    crate::fsutil::unique_temporary(parent, name, "tmp", "cannot allocate bundle staging file")
}

fn unique_directory(parent: &Path, prefix: &str) -> Result<PathBuf> {
    for nonce in 0..1024u32 {
        let candidate = parent.join(format!("{prefix}.{}.{}", std::process::id(), nonce));
        match fs::create_dir(&candidate) {
            Ok(()) => {
                ensure_private_directory(&candidate)?;
                return Ok(candidate);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(Error::new(ErrorCode::Io, error.to_string()).with_path(candidate))
            }
        }
    }
    Err(Error::new(
        ErrorCode::CacheBusy,
        "cannot allocate bundle staging directory",
    ))
}

fn sync_tree(root: &Path) -> Result<()> {
    let mut directories = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        directories.push(directory.clone());
        for entry in fs::read_dir(&directory)
            .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&directory))?
        {
            let path = entry
                .map_err(|error| Error::new(ErrorCode::Io, error.to_string()))?
                .path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| Error::new(ErrorCode::Io, error.to_string()).with_path(&path))?;
            if metadata.file_type().is_symlink() {
                return Err(bundle_error("staging tree contains a symlink", &path));
            }
            if metadata.is_dir() {
                pending.push(path);
            }
        }
    }
    directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for directory in directories {
        sync_directory(&directory)?;
    }
    Ok(())
}

fn bundle_error(message: impl Into<String>, path: &Path) -> Error {
    Error::new(ErrorCode::BundleInvalid, message).with_path(path)
}
