use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use serde_json::Value;
use toktier_cuda_driver::CudaContext;
use toktier_routing_core::{
    NativeGpuEngine, NativePrebuiltGpu, NativePrebuiltGpuConfig, ReferenceEngine,
};

#[cfg(feature = "jit")]
use crate::jit::JitProduct;
use crate::manifest::{domain_sha256_hex, sha256_hex, LocalArtifact, Registry, PREBUILT_FATBIN};
use crate::{Error, ErrorCode, Policy, Result};

const EMPTY: u64 = u64::MAX;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;

const CLASS_CL100K: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/tables/pretok_classes_cl100k.v3.npy"
));
const CLASS_CL100K_M2L: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/tables/pretok_classes_cl100k_marks_as_letters.v3.npy"
));
const CLASS_DEEPSEEK: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/tables/pretok_classes_deepseek.v1.npy"
));
const CLASS_O200K: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/tables/pretok_classes_o200k.v4.npy"
));
const DEEPSEEK_META: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/src/toktier/kernels/tables/pretok_classes_deepseek.v1.meta.json"
));

pub(crate) struct BuiltGpu {
    pub(crate) engine: Arc<dyn NativeGpuEngine>,
    pub(crate) architecture: String,
    pub(crate) driver_api_version: i32,
}

pub(crate) fn build_prebuilt(
    registry: &Registry,
    artifact: &LocalArtifact,
    reference: Arc<ReferenceEngine>,
    device_ordinal: i32,
    policy: Policy,
) -> Result<BuiltGpu> {
    let family = &artifact.identity().family;
    let support = registry.support(family).ok_or_else(|| {
        Error::new(
            ErrorCode::UncertifiedTokenizer,
            format!("family {family} has no shipped certification row"),
        )
    })?;
    if support.artifact_sha256 != artifact.identity().tokenizer_sha256 {
        return Err(Error::new(
            ErrorCode::UncertifiedTokenizer,
            "support registry and verified artifact digest disagree",
        ));
    }
    let prebuilt_support = support
        .backends
        .get("gpu")
        .and_then(|gpu| gpu.get("deliveries"))
        .and_then(|deliveries| deliveries.get("prebuilt"))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::KernelIncompatible,
                format!("family {family} has no prebuilt GPU delivery"),
            )
        })?;
    let expected_host_source = prebuilt_support
        .get("host_source_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::RegistryInvalid,
                "prebuilt delivery has no native-host source binding",
            )
        })?;
    if expected_host_source != env!("TOKTIER_RUST_API_NATIVE_HOST_SOURCE_SHA256") {
        return Err(Error::new(
            ErrorCode::UncertifiedRuntime,
            "the Rust CUDA host source does not match the prebuilt delivery certificate",
        ));
    }
    let context = CudaContext::new(device_ordinal).map_err(|error| {
        Error::new(ErrorCode::KernelIncompatible, error.to_string()).with_family(family)
    })?;
    let (major, minor) = context.architecture();
    let architecture = format!("sm_{major}{minor}");
    let driver_api_version = context.driver_version();
    let certified = json_string_array(prebuilt_support.get("devices"))
        .iter()
        .any(|device| device == &architecture);
    let experimental = json_string_array(prebuilt_support.get("devices_experimental"))
        .iter()
        .any(|device| device == &architecture);
    if !(certified || policy == Policy::Experimental && experimental) {
        return Err(Error::new(
            ErrorCode::KernelIncompatible,
            format!(
                "prebuilt GPU architecture {} is not admitted by policy {policy:?}",
                architecture
            ),
        ));
    }
    if registry.prebuilt.toolchain.is_empty()
        || !registry.prebuilt.architectures.contains_key(&architecture)
    {
        return Err(Error::new(
            ErrorCode::RegistryInvalid,
            "prebuilt manifest does not describe the selected architecture/toolchain",
        ));
    }

    let expected_binary = prebuilt_support
        .get("binary_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new(ErrorCode::RegistryInvalid, "prebuilt row has no digest"))?;
    if expected_binary.trim_start_matches("sha256:")
        != registry
            .prebuilt
            .fatbin
            .digest
            .trim_start_matches("sha256:")
    {
        return Err(Error::new(
            ErrorCode::RegistryInvalid,
            "support registry and build manifest bind different fatbins",
        ));
    }
    if PREBUILT_FATBIN.len() as u64 != registry.prebuilt.fatbin.size {
        return Err(Error::new(
            ErrorCode::KernelIncompatible,
            "shipped fatbin size differs from its build manifest",
        ));
    }
    let observed_binary = domain_sha256_hex(b"toktier.kernel_fatbin.v1\0", PREBUILT_FATBIN);
    if observed_binary != expected_binary.trim_start_matches("sha256:") {
        return Err(Error::new(
            ErrorCode::KernelIncompatible,
            "shipped fatbin digest differs from its build manifest",
        ));
    }

    build_image(
        registry,
        artifact,
        reference,
        device_ordinal,
        architecture,
        driver_api_version,
        PREBUILT_FATBIN,
        &registry.prebuilt.fatbin.digest,
        "prebuilt",
    )
}

#[cfg(feature = "jit")]
pub(crate) fn build_jit(
    registry: &Registry,
    artifact: &LocalArtifact,
    reference: Arc<ReferenceEngine>,
    device_ordinal: i32,
    policy: Policy,
    product: &JitProduct,
) -> Result<BuiltGpu> {
    if product.architecture != format!("sm_{}{}", product.compute_major, product.compute_minor) {
        return Err(Error::new(
            ErrorCode::JitCompileFailed,
            "JIT product architecture facts are internally inconsistent",
        ));
    }
    if !product.certified && policy != Policy::Experimental {
        return Err(Error::new(
            ErrorCode::UncertifiedJit,
            "an unregistered JIT product is usable only under Policy::Experimental",
        ));
    }
    build_image(
        registry,
        artifact,
        reference,
        device_ordinal,
        product.architecture.clone(),
        product.driver_api_version,
        &product.image,
        &product.domain_digest,
        "jit",
    )
}

#[allow(clippy::too_many_arguments)]
fn build_image(
    registry: &Registry,
    artifact: &LocalArtifact,
    reference: Arc<ReferenceEngine>,
    device_ordinal: i32,
    architecture: String,
    driver_api_version: i32,
    image: &[u8],
    image_digest: &str,
    delivery: &str,
) -> Result<BuiltGpu> {
    let family = &artifact.identity().family;

    let kernel = registry.kernel_families.get(family).ok_or_else(|| {
        Error::new(
            ErrorCode::KernelIncompatible,
            format!("family {family} has no end-to-end kernel mapping"),
        )
    })?;
    let class_spec = registry
        .class_tables
        .get(&kernel.class_table)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::RegistryInvalid,
                "kernel class-table mapping is incomplete",
            )
        })?;
    if class_spec.dtype != "uint8" || class_spec.shape.as_slice() != [0x11_0000] {
        return Err(Error::new(
            ErrorCode::RegistryInvalid,
            "GPU class table has an unsupported shape or dtype",
        ));
    }
    let class_file = class_file(&class_spec.file)?;
    if sha256_hex(class_file) != class_spec.sha256.trim_start_matches("sha256:") {
        return Err(Error::new(
            ErrorCode::KernelIncompatible,
            "GPU class-table file does not match its registry digest",
        ));
    }
    let class_table = npy_u8_payload(class_file, 0x11_0000)?;
    let digits_max = match kernel.digits_max {
        Some(value) => value,
        None => {
            if class_spec.meta_file.as_deref() != Some("pretok_classes_deepseek.v1.meta.json") {
                return Err(Error::new(
                    ErrorCode::RegistryInvalid,
                    "unbound digits_max metadata",
                ));
            }
            if let Some(expected) = class_spec.meta_sha256.as_deref() {
                if sha256_hex(DEEPSEEK_META) != expected.trim_start_matches("sha256:") {
                    return Err(Error::new(
                        ErrorCode::KernelIncompatible,
                        "class-table metadata digest mismatch",
                    ));
                }
            }
            serde_json::from_slice::<Value>(DEEPSEEK_META)
                .ok()
                .and_then(|value| value.get("digits_max").and_then(Value::as_i64))
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| {
                    Error::new(ErrorCode::RegistryInvalid, "metadata has no digits_max")
                })?
        }
    };
    let tables = BpeTables::from_tokenizer_json(&artifact.bytes())?;
    // Hash verification already passed; an unparseable document is a
    // kernel-construction failure, not a content-hash mismatch.
    let document: Value = serde_json::from_slice(&artifact.bytes()).map_err(|error| {
        Error::new(
            ErrorCode::KernelIncompatible,
            format!("verified tokenizer JSON cannot be parsed: {error}"),
        )
    })?;
    let needs_nfc = document
        .get("normalizer")
        .and_then(Value::as_object)
        .and_then(|normalizer| normalizer.get("type"))
        .and_then(Value::as_str)
        == Some("NFC");
    let config = NativePrebuiltGpuConfig {
        family: family.clone(),
        artifact_sha256: artifact.identity().tokenizer_sha256.clone(),
        expected_fatbin_sha256: image_digest.to_owned(),
        expected_architecture: architecture.clone(),
        device_ordinal,
        ruleset: kernel.ruleset.clone(),
        digits_max,
        contractions: kernel.contractions,
        needs_nfc,
        ignore_merges: i32::from(tables.ignore_merges),
        pair_count: tables.pair_count,
        vocab_count: tables.vocab_count,
        delivery: delivery.to_owned(),
    };
    let engine = NativePrebuiltGpu::new(
        config,
        (*reference).clone(),
        image,
        registry.prebuilt.kernels.clone(),
        class_table,
        &tables.pair_keys,
        &tables.pair_vals,
        &tables.byte_id,
        &tables.vocab_keys,
        &tables.vocab_vals,
        &tables.vocab_blob,
        &tables.unsafe_bits,
    )
    .map_err(|error| Error::new(ErrorCode::KernelIncompatible, error.to_string()))?;
    Ok(BuiltGpu {
        engine: Arc::new(engine),
        architecture,
        driver_api_version,
    })
}

fn class_file(name: &str) -> Result<&'static [u8]> {
    match name {
        "pretok_classes_cl100k.v3.npy" => Ok(CLASS_CL100K),
        "pretok_classes_cl100k_marks_as_letters.v3.npy" => Ok(CLASS_CL100K_M2L),
        "pretok_classes_deepseek.v1.npy" => Ok(CLASS_DEEPSEEK),
        "pretok_classes_o200k.v4.npy" => Ok(CLASS_O200K),
        other => Err(Error::new(
            ErrorCode::KernelIncompatible,
            format!("class table {other:?} is not embedded in the Rust crate"),
        )),
    }
}

fn npy_u8_payload(bytes: &[u8], expected: usize) -> Result<&[u8]> {
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err(Error::new(ErrorCode::RegistryInvalid, "invalid NPY magic"));
    }
    let major = *bytes
        .get(6)
        .ok_or_else(|| Error::new(ErrorCode::RegistryInvalid, "truncated NPY header"))?;
    let header_len = match major {
        1 => u16::from_le_bytes(
            bytes
                .get(8..10)
                .and_then(|raw| raw.try_into().ok())
                .ok_or_else(|| Error::new(ErrorCode::RegistryInvalid, "truncated NPY header"))?,
        ) as usize,
        2 | 3 => u32::from_le_bytes(
            bytes
                .get(8..12)
                .and_then(|raw| raw.try_into().ok())
                .ok_or_else(|| Error::new(ErrorCode::RegistryInvalid, "truncated NPY header"))?,
        ) as usize,
        _ => {
            return Err(Error::new(
                ErrorCode::RegistryInvalid,
                "unsupported NPY version",
            ))
        }
    };
    let base = if major == 1 { 10 } else { 12 };
    let offset = base + header_len;
    let payload = bytes
        .get(offset..)
        .ok_or_else(|| Error::new(ErrorCode::RegistryInvalid, "truncated NPY payload"))?;
    if payload.len() != expected {
        return Err(Error::new(
            ErrorCode::RegistryInvalid,
            format!(
                "NPY payload has {} bytes; expected {expected}",
                payload.len()
            ),
        ));
    }
    Ok(payload)
}

fn json_string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

struct BpeTables {
    pair_keys: Vec<u8>,
    pair_vals: Vec<u8>,
    byte_id: Vec<u8>,
    vocab_keys: Vec<u8>,
    vocab_vals: Vec<u8>,
    vocab_blob: Vec<u8>,
    unsafe_bits: Vec<u8>,
    ignore_merges: bool,
    pair_count: usize,
    vocab_count: usize,
}

impl BpeTables {
    fn from_tokenizer_json(bytes: &[u8]) -> Result<Self> {
        // Hash verification already passed; an unparseable document is a
        // kernel-construction failure, not a content-hash mismatch.
        let document: Value = serde_json::from_slice(bytes).map_err(|error| {
            Error::new(
                ErrorCode::KernelIncompatible,
                format!("verified tokenizer JSON cannot be parsed: {error}"),
            )
        })?;
        let model = document
            .get("model")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::new(ErrorCode::KernelIncompatible, "tokenizer has no model"))?;
        let vocab_object = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::new(ErrorCode::KernelIncompatible, "BPE model has no vocab"))?;
        let mut vocab = HashMap::with_capacity(vocab_object.len());
        for (token, id) in vocab_object {
            let id = id
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    Error::new(ErrorCode::KernelIncompatible, "vocabulary id is not u32")
                })?;
            vocab.insert(token.clone(), id);
        }
        let merges = model
            .get("merges")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new(ErrorCode::KernelIncompatible, "BPE model has no merges"))?;
        if merges.len() >= (1 << 27) {
            return Err(Error::new(
                ErrorCode::KernelIncompatible,
                "merge count exceeds kernel packing limits",
            ));
        }
        let mut pair_keys = Vec::with_capacity(merges.len());
        let mut pair_values = Vec::with_capacity(merges.len());
        let mut rules = Vec::with_capacity(merges.len());
        let mut referenced = HashSet::new();
        for (rank, merge) in merges.iter().enumerate() {
            let (left, right) = parse_merge(merge)?;
            let merged = format!("{left}{right}");
            referenced.insert(left.clone());
            referenced.insert(right.clone());
            referenced.insert(merged.clone());
            let left_id = *vocab.get(&left).ok_or_else(|| {
                Error::new(ErrorCode::KernelIncompatible, "merge left token is absent")
            })?;
            let right_id = *vocab.get(&right).ok_or_else(|| {
                Error::new(ErrorCode::KernelIncompatible, "merge right token is absent")
            })?;
            let merged_id = *vocab.get(&merged).ok_or_else(|| {
                Error::new(ErrorCode::KernelIncompatible, "merged token is absent")
            })?;
            pair_keys.push((u64::from(left_id) << 32) | u64::from(right_id));
            pair_values.push(((rank as u64) << 32) | u64::from(merged_id));
            rules.push((left, right, merged));
        }
        let (pair_keys, pair_vals) = build_hash(&pair_keys, &pair_values)?;
        let alphabet = byte_alphabet();
        let reverse = alphabet
            .iter()
            .map(|(byte, character)| (*character, *byte))
            .collect::<HashMap<_, _>>();
        let mut byte_ids = [-1_i32; 256];
        let mut missing = Vec::new();
        for (byte, character) in alphabet {
            match vocab.get(&character.to_string()) {
                Some(id) => {
                    byte_ids[usize::from(byte)] = i32::try_from(*id).map_err(|_| {
                        Error::new(ErrorCode::KernelIncompatible, "token id exceeds i32")
                    })?
                }
                None => missing.push(byte),
            }
        }
        let ignore_merges = model
            .get("ignore_merges")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if ignore_merges && !missing.is_empty() {
            return Err(Error::new(
                ErrorCode::KernelIncompatible,
                "ignore_merges requires complete byte-alphabet coverage",
            ));
        }
        let mut blob = Vec::new();
        let mut vocab_hashes = Vec::new();
        let mut vocab_values = Vec::new();
        let mut seen = HashMap::<u64, Vec<u8>>::new();
        for (token, &id) in &vocab {
            if id >= (1 << 20) {
                return Err(Error::new(
                    ErrorCode::KernelIncompatible,
                    "vocabulary id exceeds kernel packing limits",
                ));
            }
            let Some(raw) = token_raw(token, &reverse) else {
                if referenced.contains(token) {
                    return Err(Error::new(
                        ErrorCode::KernelIncompatible,
                        "merge references a non-byte-level token",
                    ));
                }
                continue;
            };
            if raw.len() >= 1024 || blob.len() >= (1usize << 34) {
                return Err(Error::new(
                    ErrorCode::KernelIncompatible,
                    "vocabulary entry exceeds kernel packing limits",
                ));
            }
            let key = fnv1a64(&raw);
            if let Some(previous) = seen.get(&key) {
                if previous != &raw {
                    return Err(Error::new(
                        ErrorCode::KernelIncompatible,
                        "whole-vocabulary FNV hash collision",
                    ));
                }
                continue;
            }
            seen.insert(key, raw.clone());
            vocab_hashes.push(key);
            vocab_values
                .push(((blob.len() as u64) << 30) | ((raw.len() as u64) << 20) | u64::from(id));
            blob.extend_from_slice(&raw);
        }
        let (vocab_keys, vocab_vals) = build_hash(&vocab_hashes, &vocab_values)?;
        let mut first_use = HashMap::<String, usize>::new();
        for (rank, (left, right, _)) in rules.iter().enumerate() {
            first_use.entry(left.clone()).or_insert(rank);
            first_use.entry(right.clone()).or_insert(rank);
        }
        let mut unsafe_words = vec![0u32; merges.len().div_ceil(32)];
        for (rank, (_, _, result)) in rules.iter().enumerate() {
            if first_use.get(result).is_some_and(|used_at| *used_at < rank) {
                unsafe_words[rank >> 5] |= 1u32 << (rank & 31);
            }
        }
        if unsafe_words.iter().all(|word| *word == 0) {
            unsafe_words.clear();
        }
        Ok(Self {
            pair_count: pair_keys.len(),
            vocab_count: vocab_keys.len(),
            pair_keys: u64_bytes(&pair_keys),
            pair_vals: u64_bytes(&pair_vals),
            byte_id: i32_bytes(&byte_ids),
            vocab_keys: u64_bytes(&vocab_keys),
            vocab_vals: u64_bytes(&vocab_vals),
            vocab_blob: blob,
            unsafe_bits: u32_bytes(&unsafe_words),
            ignore_merges,
        })
    }
}

fn parse_merge(value: &Value) -> Result<(String, String)> {
    if let Some(parts) = value.as_array() {
        if parts.len() == 2 {
            if let (Some(left), Some(right)) = (parts[0].as_str(), parts[1].as_str()) {
                return Ok((left.to_owned(), right.to_owned()));
            }
        }
    } else if let Some(row) = value.as_str() {
        if let Some((left, right)) = row.split_once(' ') {
            return Ok((left.to_owned(), right.to_owned()));
        }
    }
    Err(Error::new(
        ErrorCode::KernelIncompatible,
        "unsupported BPE merge serialization",
    ))
}

fn build_hash(keys: &[u64], values: &[u64]) -> Result<(Vec<u64>, Vec<u64>)> {
    let mut size = 1usize;
    while size < keys.len().saturating_mul(3) {
        size = size.checked_mul(2).ok_or_else(|| {
            Error::new(
                ErrorCode::KernelIncompatible,
                "GPU hash table size overflow",
            )
        })?;
    }
    let mut out_keys = vec![EMPTY; size];
    let mut out_values = vec![0; size];
    let mask = size - 1;
    for (&key, &value) in keys.iter().zip(values) {
        let mut slot = (splitmix64(key) as usize) & mask;
        while out_keys[slot] != EMPTY {
            if out_keys[slot] == key {
                return Err(Error::new(
                    ErrorCode::KernelIncompatible,
                    "duplicate GPU lookup-table key",
                ));
            }
            slot = (slot + 1) & mask;
        }
        out_keys[slot] = key;
        out_values[slot] = value;
    }
    Ok((out_keys, out_values))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |value, byte| {
        (value ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn byte_alphabet() -> BTreeMap<u8, char> {
    let mut visible = (33u16..127)
        .chain(161..173)
        .chain(174..256)
        .collect::<Vec<_>>();
    let mut mapped = visible.clone();
    let mut extra = 0u16;
    for byte in 0u16..=255 {
        if !visible.contains(&byte) {
            visible.push(byte);
            mapped.push(256 + extra);
            extra += 1;
        }
    }
    visible
        .into_iter()
        .zip(mapped)
        .map(|(byte, codepoint)| {
            (
                byte as u8,
                char::from_u32(u32::from(codepoint)).expect("byte alphabet is valid Unicode"),
            )
        })
        .collect()
}

fn token_raw(token: &str, reverse: &HashMap<char, u8>) -> Option<Vec<u8>> {
    token
        .chars()
        .map(|character| reverse.get(&character).copied())
        .collect()
}

fn u64_bytes(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unparseable_tokenizer_json_is_a_kernel_failure_not_a_hash_mismatch() {
        // Content-hash verification happens before table construction, so
        // a parse failure here must not reuse the hash-mismatch contract
        // code; it reports as a kernel-construction failure.
        let error = match BpeTables::from_tokenizer_json(b"not json") {
            Ok(_) => panic!("non-JSON bytes must not build kernel tables"),
            Err(error) => error,
        };
        assert_eq!(error.code(), ErrorCode::KernelIncompatible);
    }
}
