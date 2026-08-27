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
pub(crate) const ARTIFACT_CONVERSIONS_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/artifacts/tables/artifact_conversions.v1.json"
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
    "3850fb1c9ae3f6ff634115f68540de10c08f3f3f4939d5439750676fb264942c";
const ARTIFACT_CONVERSIONS_SHA256: &str =
    "8fa33251a20e12ab8e35894732e49ea93fc2599f4e3425d1730d81bb1e9f2719";
const SUPPORT_REGISTRY_SHA256: &str = env!("TOKTIER_SUPPORT_REGISTRY_SHA256");
const SIBLING_ALIASES_SHA256: &str =
    "50ed8634057c96295abaad1bb90fd7a3125fdb5e6764471b97655413bee3995a";
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

/// One family whose certified artifact is produced from pinned upstream
/// inputs rather than downloaded whole.
///
/// The shipped conversion table is the only place a converted family is
/// named; both faces read this same file, so neither can hold an opinion
/// the other does not.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ConversionRow {
    pub(crate) converter: String,
    pub(crate) inputs: Vec<ConversionInputRow>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ConversionInputRow {
    pub(crate) name: String,
}

#[derive(Debug, Deserialize)]
struct ConversionTable {
    conversions: BTreeMap<String, ConversionRow>,
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
    conversions: BTreeMap<String, ConversionRow>,
    repairs: HashMap<String, RepairRow>,
    support: HashMap<String, SupportArtifact>,
    runtime_builds: Vec<RuntimeBuildRow>,
    pub(crate) kernel_families: BTreeMap<String, KernelFamily>,
    pub(crate) class_tables: BTreeMap<String, ClassTableRow>,
    pub(crate) prebuilt: PrebuiltManifest,
    pub(crate) aliases: Vec<AliasRow>,
    pclass: Vec<u8>,
}

/// The build flags of this build set beside the judged ones, as a
/// sentence a reader can act on.
///
/// Per key rather than as two lists: the reader needs the one entry to
/// change, not a diff of nine.
/// The judged row nearest a build, by symmetric difference of build flags.
///
/// Counting only the flags a row shares with the build would reward a row
/// for being long; the distance that matters is how many flags one side
/// has and the other does not. Where two rows are equally far -- which is
/// what happens to a build whose feature list is neither row's, since each
/// row then differs from it in exactly one flag on each side -- the row
/// that adds no optional feature wins, so the reader is shown the plain
/// recipe rather than an optional one.
fn nearest_judged_row<'a>(
    observed: &[&str],
    candidates: &[&'a RuntimeBuildRow],
) -> Option<&'a RuntimeBuildRow> {
    candidates.iter().copied().min_by_key(|row| {
        let missing = row
            .build_flags
            .iter()
            .filter(|flag| !observed.contains(&flag.as_str()))
            .count();
        let extra = observed
            .iter()
            .filter(|flag| !row.build_flags.iter().any(|other| other == *flag))
            .count();
        (missing + extra, usize::from(names_optional_feature(row)))
    })
}

/// Whether a judged row's feature list names an optional build feature.
///
/// The register carries one row per judged recipe, and the recipes differ
/// from the default one by naming an extra feature. When two rows are
/// equally far from the build asking, the one that adds nothing is the
/// one to hold up as the recipe.
fn names_optional_feature(row: &RuntimeBuildRow) -> bool {
    const OPTIONAL: [&str; 2] = ["jit", "network"];
    row.build_flags
        .iter()
        .filter_map(|flag| flag.strip_prefix("features="))
        .any(|features| {
            features
                .split(',')
                .any(|feature| OPTIONAL.contains(&feature.trim()))
        })
}

fn describe_flag_divergence(observed: &[&str], judged: &[String]) -> Option<String> {
    let key = |flag: &str| flag.split_once('=').map(|(key, _)| key.to_owned());
    let value = |flag: &str| flag.split_once('=').map(|(_, value)| value.to_owned());
    let mut differences = Vec::new();
    for flag in observed {
        let Some(name) = key(flag) else { continue };
        match judged
            .iter()
            .find(|other| key(other).as_deref() == Some(name.as_str()))
        {
            Some(other) if other == flag => {}
            Some(other) => differences.push(format!(
                "{name} is {:?} here and {:?} in the judged build",
                value(flag).unwrap_or_default(),
                value(other).unwrap_or_default(),
            )),
            None => differences.push(format!("{name} is not a judged build flag")),
        }
    }
    for other in judged {
        let Some(name) = key(other) else { continue };
        if !observed
            .iter()
            .any(|flag| key(flag).as_deref() == Some(name.as_str()))
        {
            differences.push(format!(
                "{name} is judged but this build does not report it"
            ));
        }
    }
    if differences.is_empty() {
        return None;
    }
    Some(format!(
        "the build flags of this build are not the judged ones: {}. Building with the judged \
         flags brings it back -- `rustflags` is the one a caller usually sets, through RUSTFLAGS \
         or a Cargo configuration file, and the judged builds set none. A build that means to \
         keep them can still take the accelerated route by selecting Policy::Experimental, which \
         labels the result experimental rather than certified",
        differences.join("; ")
    ))
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

        verify_embedded(
            "artifact conversion table",
            ARTIFACT_CONVERSIONS_BYTES,
            ARTIFACT_CONVERSIONS_SHA256,
        )?;

        let artifacts: BTreeMap<String, ArtifactRow> =
            parse_json("artifact manifest", ARTIFACT_MANIFEST_BYTES)?;
        let conversions: ConversionTable =
            parse_json("artifact conversion table", ARTIFACT_CONVERSIONS_BYTES)?;
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
            conversions: conversions.conversions,
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
            let mut message = format!("unknown tokenizer family {family:?}");
            // The same closest-id hint the Python facade gives, from the
            // same ranking, so a typo is answered identically on either
            // surface. Nothing is suggested when nothing is close.
            let suggestions =
                crate::suggest::close_matches(family, self.artifacts.keys().map(String::as_str));
            if !suggestions.is_empty() {
                let rendered = suggestions
                    .iter()
                    .map(|candidate| format!("{candidate:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                message.push_str(&format!("; closest valid family IDs: {rendered}"));
            }
            Error::new(ErrorCode::ArtifactNotFound, message).with_family(family)
        })
    }

    /// The conversion recipe of `family`, when its artifact is derived
    /// locally rather than published.
    pub(crate) fn conversion(&self, family: &str) -> Option<&ConversionRow> {
        self.conversions.get(family)
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

    /// Why the build flags of this build are not the judged ones, when
    /// that is what stands between it and certification.
    ///
    /// `certified = false` should never be a verdict without a reason a
    /// reader can act on, and the flags are the axis most likely to be
    /// the reader's own doing. Reported only when a judged row agrees
    /// with this build on every other axis: when the sources or the
    /// toolchain differ, the flags are not the thing to change.
    pub(crate) fn rust_api_build_flag_divergence(&self) -> Option<String> {
        let observed = env!("TOKTIER_RUST_API_BUILD_FLAGS")
            .split('\x1f')
            .collect::<Vec<_>>();
        let candidates = self
            .runtime_builds
            .iter()
            .filter(|row| {
                row.runtime == "rust_api"
                    && row.source_digest == env!("TOKTIER_RUST_API_SOURCE_SHA256")
                    && row.fast_cpu_source_digest == env!("TOKTIER_RUST_API_FAST_CPU_SOURCE_SHA256")
                    && row.native_host_source_digest
                        == env!("TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256")
                    && row.toolchain == env!("TOKTIER_RUST_API_TOOLCHAIN")
            })
            .collect::<Vec<_>>();
        if candidates.is_empty()
            || candidates
                .iter()
                .any(|row| row.build_flags.iter().eq(observed.iter().copied()))
        {
            return None;
        }
        // The row nearest this build, so that naming every difference
        // against every row does not bury the one the reader has to
        // change. Nearest is the smallest symmetric difference: counting
        // only the flags a row shares with this build rewards a row for
        // being long, and the rows this register carries differ by a
        // single feature token. Where that still ties -- it does for a
        // build whose feature list is neither row's -- the plain row wins
        // over an optional-feature one, because the reader building
        // without that option should be told what the plain recipe is.
        let closest = nearest_judged_row(&observed, &candidates).expect("a candidate row");
        describe_flag_divergence(&observed, &closest.build_flags)
    }

    pub(crate) fn rust_api_build_certified(&self) -> bool {
        // The register binds compile-time facts about this crate's own
        // sources. Those sources are compiled together with whatever
        // versions Cargo resolved for the transitive graph, so a build
        // whose certified core is not the judged one is not the build
        // the evidence was taken on, whatever its source digests say.
        // The core is this crate's own crates, the packages they call
        // directly, and the text-semantics libraries beneath them;
        // everything else the build compiles is compared and reported
        // rather than gating (`crate::DEPENDENCY_CLOSURE`).
        if !crate::behavior_version::core_closure_verified() {
            return false;
        }
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

#[cfg(test)]
mod tests {
    use super::{
        describe_flag_divergence, nearest_judged_row, parse_json, registry_error, verify_embedded,
        ArtifactRow, RuntimeBuildRow, SUPPORT_REGISTRY_BYTES, SUPPORT_REGISTRY_SHA256,
    };
    use crate::ErrorCode;

    /// A judged row carrying one feature list, with every other axis
    /// equal to the build under test: the divergence reporter only ever
    /// sees rows that already agree on sources and toolchain.
    fn judged_row(features: &str) -> RuntimeBuildRow {
        RuntimeBuildRow {
            runtime: "rust_api".to_owned(),
            source_digest: "s".to_owned(),
            build_flags: vec![
                "profile=release".to_owned(),
                "opt-level=3".to_owned(),
                "target=x86_64-unknown-linux-gnu".to_owned(),
                "debug=false".to_owned(),
                "target-features=fxsr,sse,sse2".to_owned(),
                "rustflags=".to_owned(),
                format!("features={features}"),
            ],
            toolchain: "t".to_owned(),
            fast_cpu_source_digest: "f".to_owned(),
            native_host_source_digest: "n".to_owned(),
            evidence_id: "e".to_owned(),
        }
    }

    /// The build flags a build with this feature list reports about itself,
    /// in the shape the register records them.
    fn observed_flags(features: &str) -> Vec<String> {
        judged_row(features).build_flags
    }

    /// From 0.2.5 the register carries two rows whose feature lists differ
    /// by one token, and a build with `network` matches neither. It is the
    /// same distance from both, so the tie is what decides which recipe
    /// the reader is shown -- and the recipe to show is the default one,
    /// not the one that also turns on an unrelated optional feature. The
    /// reader is being told which flag to change, and `jit` is not it.
    #[test]
    fn the_divergence_reference_is_the_plain_recipe_not_an_optional_one() {
        let default_row = judged_row("default,prebuilt-gpu,serving,sqlite");
        let jit_row = judged_row("default,jit,prebuilt-gpu,serving,sqlite");
        let flags = observed_flags("default,network,prebuilt-gpu,serving,sqlite");
        let observed = flags.iter().map(String::as_str).collect::<Vec<_>>();

        for candidates in [vec![&default_row, &jit_row], vec![&jit_row, &default_row]] {
            let chosen = nearest_judged_row(&observed, &candidates).expect("a candidate row");
            assert_eq!(
                chosen.build_flags.last().map(String::as_str),
                Some("features=default,prebuilt-gpu,serving,sqlite"),
                "the plain recipe is the reference whatever order the rows arrive in",
            );
            let message = describe_flag_divergence(&observed, &chosen.build_flags)
                .expect("a build with network diverges from the default recipe");
            assert!(
                message.contains("network") && !message.contains("jit"),
                "the reported difference names the feature the reader turned on: {message}",
            );
        }
    }

    /// Distance, not overlap: a row that shares more flags simply because
    /// it carries more of them is not nearer. The exact row still wins
    /// outright, which is the case the caller short-circuits before ever
    /// asking.
    #[test]
    fn the_nearest_row_is_the_one_with_the_smallest_difference() {
        let exact = judged_row("default,prebuilt-gpu,serving,sqlite");
        let far = judged_row("default,jit,network,serde,serving,sqlite");
        let flags = observed_flags("default,prebuilt-gpu,serving,sqlite");
        let observed = flags.iter().map(String::as_str).collect::<Vec<_>>();
        let candidates = vec![&far, &exact];

        let chosen = nearest_judged_row(&observed, &candidates).expect("a candidate row");
        assert_eq!(
            chosen.build_flags.last().map(String::as_str),
            Some("features=default,prebuilt-gpu,serving,sqlite"),
        );
        assert_eq!(
            describe_flag_divergence(&observed, &chosen.build_flags),
            None
        );
    }

    /// Every shipped table is admitted by its digest and then by its
    /// shape, and both refusals are `REGISTRY_INVALID`: a package whose
    /// own records do not hold together is not a caller error. The
    /// product bytes stay untouched here -- these are the two gates
    /// themselves, asked about bytes that are not the shipped ones.
    #[test]
    fn a_shipped_record_that_does_not_hold_together_is_refused() {
        // The shipped bytes pass their own digest.
        verify_embedded(
            "support registry",
            SUPPORT_REGISTRY_BYTES,
            SUPPORT_REGISTRY_SHA256,
        )
        .expect("the shipped registry verifies");

        let error = verify_embedded(
            "support registry",
            b"not the shipped bytes",
            SUPPORT_REGISTRY_SHA256,
        )
        .expect_err("a digest mismatch");
        assert_eq!(error.code(), ErrorCode::RegistryInvalid);
        assert_eq!(error.code().as_str(), "REGISTRY_INVALID");
        assert!(
            error.message().contains("digest mismatch")
                && error.message().contains("support registry"),
            "{}",
            error.message()
        );

        // A digest that verifies over the wrong shape still stops here.
        let error = parse_json::<ArtifactRow>("artifact manifest", b"{\"repo_id\": 7}")
            .expect_err("a shape mismatch");
        assert_eq!(error.code(), ErrorCode::RegistryInvalid);
        assert!(
            error
                .message()
                .starts_with("cannot parse shipped artifact manifest"),
            "{}",
            error.message()
        );

        // And the cross-reference refusals share the code.
        let error = registry_error("kernel family x names an unknown band");
        assert_eq!(error.code(), ErrorCode::RegistryInvalid);
    }

    fn judged() -> Vec<String> {
        [
            "profile=release",
            "opt-level=3",
            "target=x86_64-unknown-linux-gnu",
            "debug=false",
            "target-features=fxsr,sse,sse2",
            "rustflags=",
            "features=default,network,prebuilt-gpu,serving,sqlite",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn matching_flags_have_nothing_to_report() {
        let judged = judged();
        let observed = judged.iter().map(String::as_str).collect::<Vec<_>>();
        assert_eq!(describe_flag_divergence(&observed, &judged), None);
    }

    /// The one a caller sets themselves, and the reason this key is
    /// admitted at all: it is the codegen switch a build script can
    /// actually observe.
    #[test]
    fn a_caller_s_rustflags_are_named_with_both_ways_out() {
        let judged = judged();
        let mut observed = judged.clone();
        observed[5] = "rustflags=-C target-cpu=native".to_owned();
        let observed = observed.iter().map(String::as_str).collect::<Vec<_>>();

        let reported = describe_flag_divergence(&observed, &judged).expect("a divergence");

        assert!(
            reported.contains(
                "rustflags is \"-C target-cpu=native\" here and \"\" in the judged build"
            ),
            "{reported}"
        );
        assert!(reported.contains("RUSTFLAGS"), "{reported}");
        assert!(reported.contains("Policy::Experimental"), "{reported}");
        assert!(!reported.contains('\n'), "{reported}");
    }

    #[test]
    fn only_the_keys_that_differ_are_named() {
        let judged = judged();
        let mut observed = judged.clone();
        observed[1] = "opt-level=2".to_owned();
        let observed = observed.iter().map(String::as_str).collect::<Vec<_>>();

        let reported = describe_flag_divergence(&observed, &judged).expect("a divergence");

        assert!(reported.contains("opt-level is \"2\""), "{reported}");
        assert!(!reported.contains("profile is"), "{reported}");
        assert!(!reported.contains("rustflags is"), "{reported}");
    }

    #[test]
    fn a_key_on_only_one_side_is_named_as_such() {
        let judged = judged();
        let observed = ["profile=release", "lto=fat"];
        let reported = describe_flag_divergence(&observed, &judged).expect("a divergence");
        assert!(
            reported.contains("lto is not a judged build flag"),
            "{reported}"
        );
        assert!(
            reported.contains("rustflags is judged but this build does not report it"),
            "{reported}"
        );
    }
}
