use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use flate2::read::ZlibDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{Error, ErrorCode, Result};

pub(crate) const ARTIFACT_MANIFEST_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/artifacts/tables/artifact_manifest.v1.json"
));
pub(crate) const SUPPORT_REGISTRY_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/routing/tables/support_registry.v1.json"
));
pub(crate) const SIBLING_ALIASES_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/artifacts/tables/sibling_aliases.v1.json"
));
pub(crate) const REPAIR_MANIFEST_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/repair/tables/fast_repair_families.v1.json"
));
pub(crate) const REPAIR_PCLASS_COMPRESSED: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/repair/tables/repair_pclass.v1.zlib"
));
pub(crate) const KERNEL_FAMILIES_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/tables/kernel_families.v1.json"
));
pub(crate) const PREBUILT_MANIFEST_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/prebuilt/build_manifest.json"
));
#[cfg(feature = "prebuilt-gpu")]
pub(crate) const PREBUILT_FATBIN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/prebuilt/pretok_kernel.fatbin"
));

const ARTIFACT_MANIFEST_SHA256: &str =
    "aae335b469142ea935b0573ed8ce0f2770a512daa6f70e65580875609c6fa417";
const SUPPORT_REGISTRY_SHA256: &str = env!("TOKTIER_SUPPORT_REGISTRY_SHA256");
const SIBLING_ALIASES_SHA256: &str =
    "db979dca87435879c78fdd31fb2eb914e262416d6c2a6cfce0399263de813aac";
const REPAIR_MANIFEST_SHA256: &str =
    "8801781427f98456a7dfce6d9cc8f8ddd8dd4ec31c1ff37b71913d83182f3853";
const KERNEL_FAMILIES_SHA256: &str =
    "87ed2b84499ac37430bb1dc2c60728c8e90304987c9c17db46f286ce78c391f7";
const PREBUILT_MANIFEST_SHA256: &str =
    "9f193e6b42408a09ee0796cbff525e283836389c40a91d67aa4ec1271db8acbe";

/// Content and provenance identity of a verified tokenizer artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ArtifactIdentity {
    pub family: String,
    pub repo_id: String,
    pub revision: String,
    pub tokenizer_sha256: String,
    pub tokenizer_size: u64,
}

/// A local tokenizer whose required bytes were checked against the shipped
/// manifest during this process.
#[derive(Debug, Clone)]
pub struct LocalArtifact {
    identity: ArtifactIdentity,
    directory: PathBuf,
    tokenizer_json: PathBuf,
    bytes: Arc<[u8]>,
}

impl LocalArtifact {
    pub fn identity(&self) -> &ArtifactIdentity {
        &self.identity
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn tokenizer_json(&self) -> &Path {
        &self.tokenizer_json
    }

    pub(crate) fn bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ArtifactFileRow {
    pub(crate) sha256: String,
    pub(crate) size: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ArtifactRow {
    pub(crate) repo_id: String,
    pub(crate) revision: String,
    pub(crate) files: BTreeMap<String, ArtifactFileRow>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RepairRow {
    pub(crate) family: String,
    pub(crate) artifact_sha256: String,
    pub(crate) margin: usize,
    pub(crate) effective_l_max: usize,
    pub(crate) has_normalizer: bool,
}

#[derive(Debug, Deserialize)]
struct RepairManifest {
    schema: String,
    pclass: PclassRow,
    families: Vec<RepairRow>,
}

#[derive(Debug, Deserialize)]
struct PclassRow {
    raw_sha256: String,
    compressed_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SupportArtifact {
    pub(crate) family: String,
    pub(crate) artifact_sha256: String,
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    pub(crate) oracle_id: String,
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    pub(crate) evidence_id: String,
    pub(crate) backends: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct SupportRegistry {
    schema_version: u64,
    #[serde(default)]
    runtime_builds: Vec<RuntimeBuildRow>,
    artifacts: Vec<SupportArtifact>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RuntimeBuildRow {
    runtime: String,
    source_digest: String,
    build_flags: Vec<String>,
    toolchain: String,
    fast_cpu_source_digest: String,
    native_host_source_digest: String,
    evidence_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(feature = "prebuilt-gpu"), allow(dead_code))]
pub(crate) struct KernelFamily {
    pub(crate) band: String,
    pub(crate) ruleset: String,
    pub(crate) digits_max: Option<i32>,
    pub(crate) class_table: String,
    #[serde(default)]
    pub(crate) contractions: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(feature = "prebuilt-gpu"), allow(dead_code))]
pub(crate) struct ClassTableRow {
    pub(crate) file: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: String,
    pub(crate) sha256: String,
    pub(crate) meta_file: Option<String>,
    pub(crate) meta_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct KernelManifest {
    schema: String,
    class_tables: BTreeMap<String, ClassTableRow>,
    bands: BTreeMap<String, KernelBand>,
    families: BTreeMap<String, KernelFamily>,
}

#[derive(Debug, Deserialize)]
struct KernelBand {
    #[serde(default)]
    e2e: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(feature = "prebuilt-gpu"), allow(dead_code))]
pub(crate) struct PrebuiltManifest {
    pub(crate) toolchain: String,
    pub(crate) architectures: BTreeMap<String, serde_json::Value>,
    pub(crate) fatbin: PrebuiltFatbin,
    pub(crate) kernels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(not(feature = "prebuilt-gpu"), allow(dead_code))]
pub(crate) struct PrebuiltFatbin {
    pub(crate) size: u64,
    pub(crate) digest: String,
}

#[derive(Debug, Deserialize)]
struct AliasManifest {
    schema_version: u64,
    aliases: Vec<AliasRow>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AliasRow {
    pub(crate) repo_id: String,
    pub(crate) revision: String,
    pub(crate) source_sha256: String,
    pub(crate) source_size: u64,
    pub(crate) canonical_family: String,
    pub(crate) canonical_anchor_sha256: String,
    pub(crate) basis: String,
    pub(crate) canonical_packaged: bool,
}

#[derive(Debug)]
#[cfg_attr(not(feature = "prebuilt-gpu"), allow(dead_code))]
pub(crate) struct Registry {
    artifacts: BTreeMap<String, ArtifactRow>,
    repairs: HashMap<String, RepairRow>,
    support: HashMap<String, SupportArtifact>,
    runtime_builds: Vec<RuntimeBuildRow>,
    pub(crate) kernel_families: BTreeMap<String, KernelFamily>,
    pub(crate) class_tables: BTreeMap<String, ClassTableRow>,
    pub(crate) prebuilt: PrebuiltManifest,
    pub(crate) aliases: Vec<AliasRow>,
    pclass: Vec<u8>,
}

impl Registry {
    /// Verify and parse the embedded tables once per process, then share
    /// the result. The embedded bytes are compile-time constants, so the
    /// digest checks, JSON parses, and pclass decompression in
    /// [`Registry::load_embedded`] cannot change between calls; a failure
    /// is replayed with its recorded code and message, matching the
    /// per-call verification behavior this replaces.
    pub(crate) fn load() -> Result<Arc<Self>> {
        static REGISTRY: OnceLock<std::result::Result<Arc<Registry>, (ErrorCode, String)>> =
            OnceLock::new();
        REGISTRY
            .get_or_init(|| {
                Self::load_embedded()
                    .map(Arc::new)
                    .map_err(|error| (error.code(), error.message().to_owned()))
            })
            .clone()
            .map_err(|(code, message)| Error::new(code, message))
    }

    fn load_embedded() -> Result<Self> {
        verify_embedded(
            "artifact manifest",
            ARTIFACT_MANIFEST_BYTES,
            ARTIFACT_MANIFEST_SHA256,
        )?;
        verify_embedded(
            "support registry",
            SUPPORT_REGISTRY_BYTES,
            SUPPORT_REGISTRY_SHA256,
        )?;
        verify_embedded(
            "sibling aliases",
            SIBLING_ALIASES_BYTES,
            SIBLING_ALIASES_SHA256,
        )?;
        verify_embedded(
            "repair manifest",
            REPAIR_MANIFEST_BYTES,
            REPAIR_MANIFEST_SHA256,
        )?;
        verify_embedded(
            "kernel families",
            KERNEL_FAMILIES_BYTES,
            KERNEL_FAMILIES_SHA256,
        )?;
        verify_embedded(
            "prebuilt manifest",
            PREBUILT_MANIFEST_BYTES,
            PREBUILT_MANIFEST_SHA256,
        )?;

        let artifacts: BTreeMap<String, ArtifactRow> =
            parse_json("artifact manifest", ARTIFACT_MANIFEST_BYTES)?;
        let repair: RepairManifest = parse_json("repair manifest", REPAIR_MANIFEST_BYTES)?;
        if repair.schema != "toktier.fast_repair_families.v1" {
            return Err(registry_error("unexpected repair manifest schema"));
        }
        verify_embedded(
            "compressed repair property table",
            REPAIR_PCLASS_COMPRESSED,
            &repair.pclass.compressed_sha256,
        )?;
        let mut decoder = ZlibDecoder::new(REPAIR_PCLASS_COMPRESSED);
        let mut pclass = Vec::with_capacity(0x11_0000);
        decoder.read_to_end(&mut pclass).map_err(|error| {
            Error::new(
                ErrorCode::RegistryInvalid,
                format!("cannot decompress repair property table: {error}"),
            )
        })?;
        if pclass.len() != 0x11_0000 || sha256_hex(&pclass) != repair.pclass.raw_sha256 {
            return Err(registry_error(
                "decompressed repair property table has the wrong identity",
            ));
        }
        let support: SupportRegistry = parse_json("support registry", SUPPORT_REGISTRY_BYTES)?;
        if support.schema_version != 1 {
            return Err(registry_error("unexpected support registry schema"));
        }
        let kernel: KernelManifest = parse_json("kernel families", KERNEL_FAMILIES_BYTES)?;
        if kernel.schema != "toktier.kernel_families.v1" {
            return Err(registry_error("unexpected kernel-family schema"));
        }
        for (family, row) in &kernel.families {
            let band = kernel.bands.get(&row.band).ok_or_else(|| {
                registry_error(format!("kernel family {family} names an unknown band"))
            })?;
            if !kernel.class_tables.contains_key(&row.class_table) {
                return Err(registry_error(format!(
                    "kernel family {family} names an unknown class table"
                )));
            }
            if !band.e2e {
                return Err(registry_error(format!(
                    "unexpected non-e2e kernel family {family}"
                )));
            }
        }
        let aliases: AliasManifest = parse_json("sibling aliases", SIBLING_ALIASES_BYTES)?;
        if aliases.schema_version != 1 {
            return Err(registry_error("unexpected sibling-alias schema"));
        }
        let prebuilt: PrebuiltManifest = parse_json("prebuilt manifest", PREBUILT_MANIFEST_BYTES)?;

        let repairs = repair
            .families
            .into_iter()
            .map(|row| (row.family.clone(), row))
            .collect();
        let runtime_builds = support.runtime_builds;
        let support = support
            .artifacts
            .into_iter()
            .map(|row| (row.family.clone(), row))
            .collect();
        Ok(Self {
            artifacts,
            repairs,
            support,
            runtime_builds,
            kernel_families: kernel.families,
            class_tables: kernel.class_tables,
            prebuilt,
            aliases: aliases.aliases,
            pclass,
        })
    }

    pub(crate) fn artifact(&self, family: &str) -> Result<&ArtifactRow> {
        self.artifacts.get(family).ok_or_else(|| {
            Error::new(
                ErrorCode::ArtifactNotFound,
                format!("unknown tokenizer family {family:?}"),
            )
            .with_family(family)
        })
    }

    pub(crate) fn artifacts(&self) -> impl Iterator<Item = (&str, &ArtifactRow)> {
        self.artifacts
            .iter()
            .map(|(family, row)| (family.as_str(), row))
    }

    pub(crate) fn repair(&self, family: &str) -> Option<&RepairRow> {
        self.repairs.get(family)
    }

    pub(crate) fn support(&self, family: &str) -> Option<&SupportArtifact> {
        self.support.get(family)
    }

    pub(crate) fn rust_api_build_certified(&self) -> bool {
        let observed_flags = env!("TOKTIER_RUST_API_BUILD_FLAGS")
            .split('\x1f')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        self.runtime_builds.iter().any(|row| {
            row.runtime == "rust_api"
                && row.source_digest == env!("TOKTIER_RUST_API_SOURCE_SHA256")
                && row.build_flags == observed_flags
                && row.toolchain == env!("TOKTIER_RUST_API_TOOLCHAIN")
                && row.fast_cpu_source_digest == env!("TOKTIER_RUST_API_FAST_CPU_SOURCE_SHA256")
                && row.native_host_source_digest
                    == env!("TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256")
                && !row.evidence_id.is_empty()
        })
    }

    pub(crate) fn pclass(&self) -> Vec<u8> {
        self.pclass.clone()
    }

    pub(crate) fn resolve_repo(&self, repo_id: &str, revision: &str) -> Result<String> {
        if let Some((family, _)) = self
            .artifacts
            .iter()
            .find(|(_, row)| row.repo_id == repo_id && row.revision == revision)
        {
            return Ok(family.clone());
        }
        let alias = self
            .aliases
            .iter()
            .find(|row| row.repo_id == repo_id && row.revision == revision)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::UncertifiedTokenizer,
                    format!("{repo_id}@{revision} is not in the shipped verified-repository table"),
                )
            })?;
        if !alias.canonical_packaged
            || alias.source_sha256.len() != 64
            || alias.source_size == 0
            || !matches!(
                alias.basis.as_str(),
                "identical"
                    | "identical_source"
                    | "equivalent_canonicalisation"
                    | "equivalent_serialisation"
            )
        {
            return Err(Error::new(
                ErrorCode::UncertifiedTokenizer,
                format!("{repo_id}@{revision} has no admitted canonical execution artifact"),
            ));
        }
        let canonical = self.artifact(&alias.canonical_family)?;
        let file = canonical
            .files
            .get("tokenizer.json")
            .ok_or_else(|| registry_error("canonical artifact does not bind tokenizer.json"))?;
        if file.sha256 != alias.canonical_anchor_sha256 {
            return Err(registry_error("sibling alias canonical anchor drifted"));
        }
        Ok(alias.canonical_family.clone())
    }

    pub(crate) fn verify_local(
        &self,
        family: &str,
        cache_root: &Path,
        explicit: Option<&Path>,
    ) -> Result<LocalArtifact> {
        let row = self.artifact(family)?;
        let file = row.files.get("tokenizer.json").ok_or_else(|| {
            registry_error(format!("artifact {family} does not bind tokenizer.json"))
        })?;
        let directory = explicit.map_or_else(
            || cache_root.join(format!("{}-{}", family, &row.revision[..12])),
            Path::to_path_buf,
        );
        let tokenizer_json = if directory.is_file() {
            directory.clone()
        } else {
            directory.join("tokenizer.json")
        };
        let metadata = fs::symlink_metadata(&tokenizer_json).map_err(|error| {
            Error::new(
                ErrorCode::ArtifactNotFound,
                format!("verified tokenizer artifact is unavailable: {error}"),
            )
            .with_path(&tokenizer_json)
            .with_family(family)
        })?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(Error::new(
                ErrorCode::ArtifactNotFound,
                "tokenizer artifact must be a regular, non-symlink file",
            )
            .with_path(&tokenizer_json)
            .with_family(family));
        }
        if metadata.len() != file.size {
            return Err(Error::new(
                ErrorCode::ArtifactSizeMismatch,
                format!(
                    "tokenizer.json has {} bytes; manifest requires {}",
                    metadata.len(),
                    file.size
                ),
            )
            .with_path(&tokenizer_json)
            .with_family(family));
        }
        let bytes = fs::read(&tokenizer_json).map_err(|error| {
            Error::new(ErrorCode::Io, error.to_string())
                .with_path(&tokenizer_json)
                .with_family(family)
        })?;
        let observed = sha256_hex(&bytes);
        if observed != file.sha256 {
            return Err(Error::new(
                ErrorCode::ArtifactHashMismatch,
                format!(
                    "tokenizer.json digest mismatch: expected {}, observed {observed}",
                    file.sha256
                ),
            )
            .with_path(&tokenizer_json)
            .with_family(family));
        }
        let artifact_directory = tokenizer_json
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Ok(LocalArtifact {
            identity: ArtifactIdentity {
                family: family.to_owned(),
                repo_id: row.repo_id.clone(),
                revision: row.revision.clone(),
                tokenizer_sha256: file.sha256.clone(),
                tokenizer_size: file.size,
            },
            directory: artifact_directory,
            tokenizer_json,
            bytes: bytes.into(),
        })
    }
}

fn parse_json<T: for<'de> Deserialize<'de>>(name: &str, bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|error| {
        Error::new(
            ErrorCode::RegistryInvalid,
            format!("cannot parse shipped {name}: {error}"),
        )
    })
}

fn verify_embedded(name: &str, bytes: &[u8], expected: &str) -> Result<()> {
    let observed = sha256_hex(bytes);
    if observed == expected.trim_start_matches("sha256:") {
        Ok(())
    } else {
        Err(registry_error(format!(
            "shipped {name} digest mismatch: expected {expected}, observed {observed}"
        )))
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    crate::fsutil::hex(&Sha256::digest(bytes))
}

#[cfg_attr(not(feature = "prebuilt-gpu"), allow(dead_code))]
pub(crate) fn domain_sha256_hex(domain: &[u8], bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(bytes);
    crate::fsutil::hex(&digest.finalize())
}

fn registry_error(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::RegistryInvalid, message)
}
