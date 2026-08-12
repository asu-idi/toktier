//! Rust-owned CUDA Driver host for the shipped prebuilt tokenizer image.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Mutex;

use sha2::{Digest, Sha256};
use toktier_cuda_driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DeviceAllocation, DevicePtr, KernelArg,
};

use crate::{NativeGpuEngine, NativeRuntimeError, ReferenceEngine};

const TPB: u32 = 256;
const SCAN_ITEMS: usize = 2 * TPB as usize;
const MED_MAX: usize = 128;
const HI: u32 = 0x7f7f_7f7f;
const FULL_PLANE: usize = 0x110000;
const MAX_BATCH_DOCS: usize = 512;
const MAX_BATCH_BYTES: usize = 4_000_000;

#[derive(Debug, Clone)]
pub struct NativePrebuiltGpuConfig {
    pub family: String,
    pub artifact_sha256: String,
    pub expected_fatbin_sha256: String,
    pub expected_architecture: String,
    pub device_ordinal: i32,
    pub ruleset: String,
    pub digits_max: i32,
    pub contractions: bool,
    pub needs_nfc: bool,
    pub ignore_merges: i32,
    pub pair_count: usize,
    pub vocab_count: usize,
    /// Audited delivery label (`prebuilt` or `jit`) reported by routing.
    pub delivery: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ruleset {
    Cl100k,
    Deepseek,
    Laguna,
    O200k,
    Kimi,
}

impl Ruleset {
    fn parse(value: &str) -> Result<Self, NativeRuntimeError> {
        match value {
            "cl100k" => Ok(Self::Cl100k),
            "deepseek" => Ok(Self::Deepseek),
            "laguna" => Ok(Self::Laguna),
            "o200k" => Ok(Self::O200k),
            "kimi" => Ok(Self::Kimi),
            other => Err(NativeRuntimeError::new(format!(
                "unsupported native GPU ruleset {other:?}"
            ))),
        }
    }

    fn numeric(self) -> Option<i32> {
        match self {
            Self::Cl100k => Some(0),
            Self::Deepseek => Some(1),
            Self::Laguna => Some(2),
            Self::O200k | Self::Kimi => None,
        }
    }

    /// Whether this ruleset runs the windowed splitter group: the same
    /// five kernels, with the Han branch selected by a runtime flag.
    fn windowed(self) -> bool {
        matches!(self, Self::O200k | Self::Kimi)
    }

    /// Whether the Han branch of the windowed group is active.
    fn han_branch(self) -> bool {
        matches!(self, Self::Kimi)
    }
}

struct Tables {
    class: DeviceAllocation,
    pair_keys: DeviceAllocation,
    pair_vals: DeviceAllocation,
    byte_id: DeviceAllocation,
    vocab_keys: DeviceAllocation,
    vocab_vals: DeviceAllocation,
    vocab_blob: DeviceAllocation,
    unsafe_bits: DeviceAllocation,
}

struct ScanLevel {
    sums: DeviceAllocation,
    scanned: DeviceAllocation,
}

struct Workspace {
    capacity: usize,
    data: DeviceAllocation,
    nb: DeviceAllocation,
    flags_i32: DeviceAllocation,
    cpos: DeviceAllocation,
    cp: DeviceAllocation,
    bo: DeviceAllocation,
    seed: DeviceAllocation,
    carrier_rid: DeviceAllocation,
    carrier_pos: DeviceAllocation,
    bmask: DeviceAllocation,
    dso: DeviceAllocation,
    ars: DeviceAllocation,
    cls: DeviceAllocation,
    heads: DeviceAllocation,
    rids: DeviceAllocation,
    run_start: DeviceAllocation,
    chan_hi: DeviceAllocation,
    chan_neg: DeviceAllocation,
    starts: DeviceAllocation,
    doc_starts: DeviceAllocation,
    trig2: DeviceAllocation,
    sparse_flags: DeviceAllocation,
    pb: DeviceAllocation,
    counts: DeviceAllocation,
    dispatch_flags: DeviceAllocation,
    scan_out: DeviceAllocation,
    lists: DeviceAllocation,
    scratch: DeviceAllocation,
    long_buffer: DeviceAllocation,
    token_counts: DeviceAllocation,
    offsets: DeviceAllocation,
    output: DeviceAllocation,
    meta: DeviceAllocation,
    scan_levels: Vec<ScanLevel>,
}

impl Workspace {
    fn new(context: &CudaContext, capacity: usize) -> Result<Self, NativeRuntimeError> {
        let alloc = |bytes| {
            context
                .alloc(bytes)
                .map_err(|error| NativeRuntimeError::new(error.to_string()))
        };
        let i32s = |count: usize| count.saturating_mul(4);
        let mut scan_levels = Vec::new();
        let mut length = capacity;
        loop {
            let blocks = length.div_ceil(SCAN_ITEMS);
            scan_levels.push(ScanLevel {
                sums: alloc(i32s(blocks))?,
                scanned: alloc(i32s(blocks))?,
            });
            if blocks <= 1 {
                break;
            }
            length = blocks;
        }
        Ok(Self {
            capacity,
            data: alloc(capacity)?,
            nb: alloc(4)?,
            flags_i32: alloc(i32s(capacity))?,
            cpos: alloc(i32s(capacity))?,
            cp: alloc(i32s(capacity))?,
            bo: alloc(i32s(capacity))?,
            seed: alloc(i32s(capacity))?,
            carrier_rid: alloc(i32s(capacity))?,
            carrier_pos: alloc(i32s(capacity))?,
            bmask: alloc(capacity)?,
            dso: alloc(i32s(capacity))?,
            ars: alloc(i32s(capacity))?,
            cls: alloc(capacity)?,
            heads: alloc(4usize.saturating_mul(capacity))?,
            rids: alloc(i32s(4usize.saturating_mul(capacity)))?,
            run_start: alloc(i32s(capacity))?,
            chan_hi: alloc(i32s(7usize.saturating_mul(capacity)))?,
            chan_neg: alloc(i32s(4usize.saturating_mul(capacity)))?,
            starts: alloc(capacity)?,
            doc_starts: alloc(capacity)?,
            // Three trigger planes: the punctuation-or-mark plane, the
            // contraction-chain plane, and the Han plane the Han branch
            // writes. Each is one byte per input byte.
            trig2: alloc(3usize.saturating_mul(capacity))?,
            sparse_flags: alloc(8)?,
            pb: alloc(i32s(capacity.saturating_add(1)))?,
            counts: alloc(16)?,
            dispatch_flags: alloc(3usize.saturating_mul(capacity))?,
            scan_out: alloc(i32s(capacity))?,
            lists: alloc(i32s(3usize.saturating_mul(capacity)))?,
            scratch: alloc(i32s(capacity))?,
            long_buffer: alloc(i32s(capacity))?,
            token_counts: alloc(i32s(capacity))?,
            offsets: alloc(i32s(capacity.saturating_add(1)))?,
            output: alloc(i32s(capacity))?,
            meta: alloc(12)?,
            scan_levels,
        })
    }
}

/// What one pipeline pass reports after the merge stage.
///
/// `chain`, `ambiguous` and `han` are the sparse surfaces of the
/// windowed splitter group. Each one marks input the group's parallel
/// rules do not decide by themselves; the exact answer needs a local
/// sequential re-resolution, which this host does not run. Reporting the
/// surface as a fault instead routes the input to the next backend in
/// the chain, so the ids stay exact and the fallback stays counted.
#[derive(Debug, Clone, Copy, Default)]
struct Tail {
    tokens: i32,
    chain: i32,
    ambiguous: i32,
    han: i32,
}

impl Tail {
    fn sparse_guard(self) -> Option<String> {
        if self.chain == 0 && self.ambiguous == 0 && self.han == 0 {
            return None;
        }
        Some(format!(
            "windowed splitter sparse guard requested exact fallback \
             (chain={}, ambiguous={}, han={})",
            self.chain, self.ambiguous, self.han
        ))
    }
}

struct Executor {
    context: CudaContext,
    _module: CudaModule,
    stream: CudaStream,
    functions: BTreeMap<String, CudaFunction>,
    tables: Tables,
    workspace: Option<Workspace>,
    ruleset: Ruleset,
    digits_max: i32,
    contractions: bool,
    ignore_merges: i32,
    pair_count: usize,
    vocab_count: usize,
}

impl Executor {
    #[allow(clippy::too_many_arguments)]
    fn new(
        config: &NativePrebuiltGpuConfig,
        fatbin: &[u8],
        symbols: &BTreeMap<String, String>,
        class_table: &[u8],
        pair_keys: &[u8],
        pair_vals: &[u8],
        byte_id: &[u8],
        vocab_keys: &[u8],
        vocab_vals: &[u8],
        vocab_blob: &[u8],
        unsafe_bits: &[u8],
    ) -> Result<Self, NativeRuntimeError> {
        validate_tables(
            config,
            class_table,
            pair_keys,
            pair_vals,
            byte_id,
            vocab_keys,
            vocab_vals,
            unsafe_bits,
        )?;
        let observed = sha256_hex(fatbin);
        let expected = config
            .expected_fatbin_sha256
            .strip_prefix("sha256:")
            .unwrap_or(&config.expected_fatbin_sha256);
        if observed != expected {
            return Err(NativeRuntimeError::new(format!(
                "prebuilt fatbin digest mismatch: expected {expected}, observed {observed}"
            )));
        }
        let context = CudaContext::new(config.device_ordinal)
            .map_err(|error| NativeRuntimeError::new(error.to_string()))?;
        let architecture = context.architecture();
        let observed_arch = format!("sm_{}{}", architecture.0, architecture.1);
        if observed_arch != config.expected_architecture {
            return Err(NativeRuntimeError::new(format!(
                "native GPU device is {observed_arch}, planner selected {}",
                config.expected_architecture
            )));
        }
        let module = context
            .load_module(fatbin)
            .map_err(|error| NativeRuntimeError::new(error.to_string()))?;
        let required = required_functions(Ruleset::parse(&config.ruleset)?);
        let mut functions = BTreeMap::new();
        for logical in required {
            let symbol = symbols.get(logical).map(String::as_str).unwrap_or(logical);
            functions.insert(
                logical.to_owned(),
                module
                    .function(symbol)
                    .map_err(|error| NativeRuntimeError::new(error.to_string()))?,
            );
        }
        let stream = context
            .stream()
            .map_err(|error| NativeRuntimeError::new(error.to_string()))?;
        let upload = |raw: &[u8]| -> Result<DeviceAllocation, NativeRuntimeError> {
            let allocation = context
                .alloc(raw.len())
                .map_err(|error| NativeRuntimeError::new(error.to_string()))?;
            stream
                .copy_to_device(allocation.ptr(), raw)
                .map_err(|error| NativeRuntimeError::new(error.to_string()))?;
            Ok(allocation)
        };
        let tables = Tables {
            class: upload(class_table)?,
            pair_keys: upload(pair_keys)?,
            pair_vals: upload(pair_vals)?,
            byte_id: upload(byte_id)?,
            vocab_keys: upload(vocab_keys)?,
            vocab_vals: upload(vocab_vals)?,
            vocab_blob: upload(vocab_blob)?,
            unsafe_bits: upload(unsafe_bits)?,
        };
        stream
            .synchronize()
            .map_err(|error| NativeRuntimeError::new(error.to_string()))?;
        Ok(Self {
            context,
            _module: module,
            stream,
            functions,
            tables,
            workspace: None,
            ruleset: Ruleset::parse(&config.ruleset)?,
            digits_max: config.digits_max,
            contractions: config.contractions,
            ignore_merges: config.ignore_merges,
            pair_count: config.pair_count,
            vocab_count: config.vocab_count,
        })
    }

    fn function(&self, name: &str) -> Result<&CudaFunction, NativeRuntimeError> {
        self.functions.get(name).ok_or_else(|| {
            NativeRuntimeError::new(format!("native GPU kernel {name:?} was not loaded"))
        })
    }

    fn launch(
        &self,
        name: &str,
        grid: u32,
        block: u32,
        args: &[KernelArg],
    ) -> Result<(), NativeRuntimeError> {
        self.stream
            .launch(
                self.function(name)?,
                (grid.max(1), 1, 1),
                (block, 1, 1),
                0,
                args,
            )
            .map_err(|error| NativeRuntimeError::new(error.to_string()))
    }

    fn ensure_workspace(&mut self, required: usize) -> Result<(), NativeRuntimeError> {
        if self
            .workspace
            .as_ref()
            .is_some_and(|workspace| workspace.capacity >= required)
        {
            return Ok(());
        }
        let capacity = required.max(4096).next_power_of_two();
        self.workspace = Some(Workspace::new(&self.context, capacity)?);
        Ok(())
    }

    fn scan(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        length: usize,
        input_u8: bool,
        workspace: &Workspace,
    ) -> Result<(), NativeRuntimeError> {
        self.scan_level(input, output, length, input_u8, workspace, 0)
    }

    fn scan_level(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        length: usize,
        input_u8: bool,
        workspace: &Workspace,
        level: usize,
    ) -> Result<(), NativeRuntimeError> {
        let blocks = length.div_ceil(SCAN_ITEMS);
        let scan_level = &workspace.scan_levels[level];
        self.launch(
            if input_u8 {
                "tk_scan_u8_blocks"
            } else {
                "tk_scan_i32_blocks"
            },
            u32_of(blocks)?,
            TPB,
            &[
                KernelArg::Ptr(input),
                KernelArg::Ptr(output),
                KernelArg::Ptr(scan_level.sums.ptr()),
                KernelArg::I32(i32_of(length)?),
            ],
        )?;
        if blocks > 1 {
            self.scan_level(
                scan_level.sums.ptr(),
                scan_level.scanned.ptr(),
                blocks,
                false,
                workspace,
                level + 1,
            )?;
            self.launch(
                "tk_scan_add",
                grid(length)?,
                TPB,
                &[
                    KernelArg::Ptr(output),
                    KernelArg::Ptr(scan_level.scanned.ptr()),
                    KernelArg::I32(i32_of(length)?),
                ],
            )?;
        }
        Ok(())
    }

    fn carrier(
        &self,
        flags: DevicePtr,
        output: DevicePtr,
        length: usize,
        workspace: &Workspace,
    ) -> Result<(), NativeRuntimeError> {
        self.scan(flags, workspace.carrier_rid.ptr(), length, false, workspace)?;
        self.launch(
            "tk_carrier_scatter",
            grid(length)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.carrier_rid.ptr()),
                KernelArg::Ptr(flags),
                KernelArg::Ptr(workspace.carrier_pos.ptr()),
                KernelArg::I32(i32_of(length)?),
            ],
        )?;
        self.launch(
            "tk_carrier_gather",
            grid(length)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.carrier_rid.ptr()),
                KernelArg::Ptr(workspace.carrier_pos.ptr()),
                KernelArg::Ptr(output),
                KernelArg::I32(i32_of(length)?),
            ],
        )
    }

    fn select(
        &self,
        values: DevicePtr,
        flags: DevicePtr,
        output: DevicePtr,
        count: DevicePtr,
        length: usize,
        workspace: &Workspace,
    ) -> Result<(), NativeRuntimeError> {
        self.scan(flags, workspace.scan_out.ptr(), length, true, workspace)?;
        self.launch(
            "tk_select_scatter",
            grid(length)?,
            TPB,
            &[
                KernelArg::Ptr(values),
                KernelArg::Ptr(flags),
                KernelArg::Ptr(workspace.scan_out.ptr()),
                KernelArg::Ptr(output),
                KernelArg::Ptr(count),
                KernelArg::I32(i32_of(length)?),
            ],
        )
    }

    fn encode(&mut self, text: &str) -> Result<Vec<u32>, NativeRuntimeError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let bytes = text.as_bytes();
        if bytes.len() >= i32::MAX as usize {
            return Err(NativeRuntimeError::new(
                "native GPU input exceeds the 2^31-byte kernel limit",
            ));
        }
        self.ensure_workspace(bytes.len())?;
        let workspace = self.workspace.take().expect("workspace ensured");
        let result = self.encode_in_workspace(bytes, &workspace);
        self.workspace = Some(workspace);
        result
    }

    /// Issue the tail reads of one pipeline pass and wait for them.
    ///
    /// The windowed splitter group reports its token count and its two
    /// sparse flags through one device-side pack; the Han branch adds a
    /// third surface, a per-character plane whose set positions are
    /// counted by an inclusive scan. Every other ruleset reads the token
    /// count alone. The caller may issue further copies before this
    /// call; the stream is ordered, so one wait covers them all.
    fn read_tail(&self, cap: usize, workspace: &Workspace) -> Result<Tail, NativeRuntimeError> {
        let mut metadata = [0i32; 3];
        let mut han = [0i32; 1];
        if self.ruleset.windowed() {
            self.launch(
                "k_o2k_meta3",
                1,
                1,
                &[
                    KernelArg::Ptr(add(workspace.offsets.ptr(), 4 * cap)),
                    KernelArg::Ptr(workspace.sparse_flags.ptr()),
                    KernelArg::Ptr(workspace.meta.ptr()),
                ],
            )?;
            self.stream
                .copy_from_device(&mut metadata, workspace.meta.ptr())
                .map_err(driver_error)?;
            if self.ruleset.han_branch() {
                // The scan scratch is free once the piece dispatch has
                // consumed it, which happens before the merge stage.
                self.scan(
                    han_plane(cap, workspace),
                    workspace.scan_out.ptr(),
                    cap,
                    true,
                    workspace,
                )?;
                self.stream
                    .copy_from_device(&mut han, add(workspace.scan_out.ptr(), 4 * (cap - 1)))
                    .map_err(driver_error)?;
            }
        } else {
            self.stream
                .copy_from_device(&mut metadata[..1], add(workspace.offsets.ptr(), 4 * cap))
                .map_err(driver_error)?;
        }
        self.stream.synchronize().map_err(driver_error)?;
        Ok(Tail {
            tokens: metadata[0],
            chain: metadata[1],
            ambiguous: metadata[2],
            han: han[0],
        })
    }

    fn encode_in_workspace(
        &self,
        bytes: &[u8],
        workspace: &Workspace,
    ) -> Result<Vec<u32>, NativeRuntimeError> {
        self.launch_pipeline(bytes, 0, workspace)?;
        let cap = bytes.len();
        let tail = self.read_tail(cap, workspace)?;
        let count = usize::try_from(tail.tokens)
            .map_err(|_| NativeRuntimeError::new("native GPU returned a negative token count"))?;
        if count > cap {
            return Err(NativeRuntimeError::new(format!(
                "native GPU returned {count} tokens for capacity {cap}"
            )));
        }
        if let Some(reason) = tail.sparse_guard() {
            return Err(NativeRuntimeError::new(reason));
        }
        let mut signed = vec![0i32; count];
        self.stream
            .copy_from_device(&mut signed, workspace.output.ptr())
            .map_err(driver_error)?;
        self.stream.synchronize().map_err(driver_error)?;
        signed
            .into_iter()
            .map(|value| {
                u32::try_from(value)
                    .map_err(|_| NativeRuntimeError::new("native GPU returned a negative token id"))
            })
            .collect()
    }

    /// Encode several already-normalized documents in one device pass.
    /// Document starts are forced pretok boundaries, and the compacted token
    /// stream is split with the device's own piece/token offset arrays.
    fn encode_batch(&mut self, docs: &[&str]) -> Result<Vec<Vec<u32>>, NativeRuntimeError> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        if docs.len() == 1 {
            return self.encode(docs[0]).map(|ids| vec![ids]);
        }

        let total_bytes = docs.iter().try_fold(0usize, |total, text| {
            total
                .checked_add(text.len())
                .ok_or_else(|| NativeRuntimeError::new("native GPU batch byte length overflow"))
        })?;
        if total_bytes == 0 {
            return Ok(vec![Vec::new(); docs.len()]);
        }
        if total_bytes >= i32::MAX as usize {
            return Err(NativeRuntimeError::new(
                "native GPU batch exceeds the 2^31-byte kernel limit",
            ));
        }

        let mut joined = Vec::with_capacity(total_bytes);
        let mut byte_offsets = Vec::with_capacity(docs.len() + 1);
        let mut char_offsets = Vec::with_capacity(docs.len());
        let mut chars = 0usize;
        for text in docs {
            byte_offsets.push(joined.len());
            char_offsets.push(chars);
            joined.extend_from_slice(text.as_bytes());
            chars = chars.checked_add(text.chars().count()).ok_or_else(|| {
                NativeRuntimeError::new("native GPU batch character length overflow")
            })?;
        }
        byte_offsets.push(joined.len());

        self.ensure_workspace(joined.len())?;
        let workspace = self.workspace.take().expect("workspace ensured");
        let result = (|| {
            let mut marks = vec![0u8; joined.len()];
            for (text, offset) in docs.iter().zip(char_offsets) {
                if !text.is_empty() {
                    marks[offset] = 1;
                }
            }
            self.stream
                .copy_to_device(workspace.doc_starts.ptr(), &marks)
                .map_err(driver_error)?;
            self.launch_pipeline(&joined, workspace.doc_starts.ptr(), &workspace)?;

            let cap = joined.len();
            let mut piece_count = [0i32; 1];
            self.stream
                .copy_from_device(&mut piece_count, workspace.counts.ptr())
                .map_err(driver_error)?;
            let tail = self.read_tail(cap, &workspace)?;

            if let Some(reason) = tail.sparse_guard() {
                return Err(NativeRuntimeError::new(format!(
                    "batched {reason}; the rows are redone one at a time"
                )));
            }
            let token_count = usize::try_from(tail.tokens).map_err(|_| {
                NativeRuntimeError::new("native GPU returned a negative batch token count")
            })?;
            let pieces = usize::try_from(piece_count[0]).map_err(|_| {
                NativeRuntimeError::new("native GPU returned a negative batch piece count")
            })?;
            if token_count > cap || pieces > cap {
                return Err(NativeRuntimeError::new(format!(
                    "native GPU returned invalid batch geometry: {token_count} tokens, {pieces} pieces, capacity {cap}"
                )));
            }

            let mut signed = vec![0i32; token_count];
            let mut piece_bytes = vec![0i32; pieces + 1];
            let mut token_offsets = vec![0i32; pieces + 1];
            self.stream
                .copy_from_device(&mut signed, workspace.output.ptr())
                .map_err(driver_error)?;
            self.stream
                .copy_from_device(&mut piece_bytes, workspace.pb.ptr())
                .map_err(driver_error)?;
            self.stream
                .copy_from_device(&mut token_offsets, workspace.offsets.ptr())
                .map_err(driver_error)?;
            self.stream.synchronize().map_err(driver_error)?;

            let ids = signed
                .into_iter()
                .map(|value| {
                    u32::try_from(value).map_err(|_| {
                        NativeRuntimeError::new("native GPU returned a negative token id")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let piece_bytes = piece_bytes
                .into_iter()
                .map(|value| {
                    usize::try_from(value).map_err(|_| {
                        NativeRuntimeError::new("native GPU returned a negative piece boundary")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let token_offsets = token_offsets
                .into_iter()
                .map(|value| {
                    usize::try_from(value).map_err(|_| {
                        NativeRuntimeError::new("native GPU returned a negative token offset")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if piece_bytes.last().copied() != Some(total_bytes)
                || token_offsets.last().copied() != Some(token_count)
            {
                return Err(NativeRuntimeError::new(
                    "native GPU batch sentinel does not match the compacted output",
                ));
            }

            byte_offsets
                .windows(2)
                .map(|bounds| {
                    let low_piece = piece_bytes.binary_search(&bounds[0]).map_err(|_| {
                        NativeRuntimeError::new(format!(
                            "native GPU omitted document-start boundary {}",
                            bounds[0]
                        ))
                    })?;
                    let high_piece = piece_bytes.binary_search(&bounds[1]).map_err(|_| {
                        NativeRuntimeError::new(format!(
                            "native GPU omitted document-end boundary {}",
                            bounds[1]
                        ))
                    })?;
                    let low = token_offsets[low_piece];
                    let high = token_offsets[high_piece];
                    ids.get(low..high).map(|row| row.to_vec()).ok_or_else(|| {
                        NativeRuntimeError::new("native GPU batch token slice is out of range")
                    })
                })
                .collect()
        })();
        self.workspace = Some(workspace);
        result
    }

    fn launch_pipeline(
        &self,
        bytes: &[u8],
        doc_starts: DevicePtr,
        workspace: &Workspace,
    ) -> Result<(), NativeRuntimeError> {
        let cap = bytes.len();
        self.stream
            .copy_to_device(workspace.data.ptr(), bytes)
            .map_err(driver_error)?;
        self.stream
            .copy_to_device(workspace.nb.ptr(), &[i32_of(cap)?])
            .map_err(driver_error)?;
        self.launch(
            "tk_utf8_lead_i32",
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.data.ptr()),
                KernelArg::Ptr(workspace.flags_i32.ptr()),
                KernelArg::I32(i32_of(cap)?),
            ],
        )?;
        self.scan(
            workspace.flags_i32.ptr(),
            workspace.cpos.ptr(),
            cap,
            false,
            workspace,
        )?;
        self.launch(
            "k_utf8_decode",
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.data.ptr()),
                KernelArg::Ptr(workspace.cpos.ptr()),
                KernelArg::Ptr(workspace.cp.ptr()),
                KernelArg::Ptr(workspace.bo.ptr()),
                KernelArg::I32(i32_of(cap)?),
                KernelArg::Ptr(0),
            ],
        )?;
        let char_count = add(workspace.cpos.ptr(), 4 * (cap - 1));
        match self.ruleset {
            ruleset if ruleset.windowed() => {
                self.windowed_middle(cap, char_count, doc_starts, workspace)?
            }
            ruleset => self.generic_middle(ruleset, cap, char_count, doc_starts, workspace)?,
        }
        self.bpe_tail(cap, workspace)
    }

    fn generic_middle(
        &self,
        ruleset: Ruleset,
        cap: usize,
        char_count: DevicePtr,
        doc_starts: DevicePtr,
        workspace: &Workspace,
    ) -> Result<(), NativeRuntimeError> {
        let numeric = ruleset.numeric().expect("generic ruleset");
        let (bmask, dso, ars) = match ruleset {
            Ruleset::Deepseek => {
                self.launch(
                    "k_ds_seed_n",
                    grid(cap)?,
                    TPB,
                    &[
                        KernelArg::Ptr(workspace.cp.ptr()),
                        KernelArg::Ptr(self.tables.class.ptr()),
                        KernelArg::Ptr(doc_starts),
                        KernelArg::Ptr(workspace.seed.ptr()),
                        KernelArg::I32(i32_of(cap)?),
                        KernelArg::Ptr(char_count),
                    ],
                )?;
                self.launch(
                    "tk_eq_index_i32",
                    grid(cap)?,
                    TPB,
                    &[
                        KernelArg::Ptr(workspace.seed.ptr()),
                        KernelArg::Ptr(workspace.flags_i32.ptr()),
                        KernelArg::I32(i32_of(cap)?),
                    ],
                )?;
                self.carrier(
                    workspace.flags_i32.ptr(),
                    workspace.dso.ptr(),
                    cap,
                    workspace,
                )?;
                self.launch(
                    "k_ds_bmask",
                    grid(cap)?,
                    TPB,
                    &[
                        KernelArg::Ptr(workspace.cp.ptr()),
                        KernelArg::Ptr(self.tables.class.ptr()),
                        KernelArg::Ptr(doc_starts),
                        KernelArg::Ptr(workspace.dso.ptr()),
                        KernelArg::I32(self.digits_max),
                        KernelArg::Ptr(workspace.bmask.ptr()),
                        KernelArg::Ptr(workspace.seed.ptr()),
                        KernelArg::I32(i32_of(cap)?),
                        KernelArg::Ptr(char_count),
                    ],
                )?;
                self.launch(
                    "tk_eq_index_i32",
                    grid(cap)?,
                    TPB,
                    &[
                        KernelArg::Ptr(workspace.seed.ptr()),
                        KernelArg::Ptr(workspace.flags_i32.ptr()),
                        KernelArg::I32(i32_of(cap)?),
                    ],
                )?;
                self.carrier(
                    workspace.flags_i32.ptr(),
                    workspace.ars.ptr(),
                    cap,
                    workspace,
                )?;
                self.launch(
                    "tk_u8_flags_i32",
                    grid(cap)?,
                    TPB,
                    &[
                        KernelArg::Ptr(workspace.bmask.ptr()),
                        KernelArg::Ptr(workspace.flags_i32.ptr()),
                        KernelArg::I32(i32_of(cap)?),
                        KernelArg::Bool(true),
                    ],
                )?;
                self.carrier(
                    workspace.flags_i32.ptr(),
                    workspace.dso.ptr(),
                    cap,
                    workspace,
                )?;
                (
                    workspace.bmask.ptr(),
                    workspace.dso.ptr(),
                    workspace.ars.ptr(),
                )
            }
            Ruleset::Laguna => {
                self.launch(
                    "k_lag_bmask",
                    grid(cap)?,
                    TPB,
                    &[
                        KernelArg::Ptr(workspace.cp.ptr()),
                        KernelArg::Ptr(doc_starts),
                        KernelArg::Ptr(workspace.bmask.ptr()),
                        KernelArg::I32(i32_of(cap)?),
                        KernelArg::Ptr(char_count),
                    ],
                )?;
                self.launch(
                    "tk_u8_flags_i32",
                    grid(cap)?,
                    TPB,
                    &[
                        KernelArg::Ptr(workspace.bmask.ptr()),
                        KernelArg::Ptr(workspace.flags_i32.ptr()),
                        KernelArg::I32(i32_of(cap)?),
                        KernelArg::Bool(true),
                    ],
                )?;
                self.carrier(
                    workspace.flags_i32.ptr(),
                    workspace.dso.ptr(),
                    cap,
                    workspace,
                )?;
                (workspace.bmask.ptr(), workspace.dso.ptr(), 0)
            }
            Ruleset::Cl100k if doc_starts != 0 => {
                self.launch(
                    "tk_u8_flags_i32",
                    grid(cap)?,
                    TPB,
                    &[
                        KernelArg::Ptr(doc_starts),
                        KernelArg::Ptr(workspace.flags_i32.ptr()),
                        KernelArg::I32(i32_of(cap)?),
                        KernelArg::Bool(true),
                    ],
                )?;
                self.carrier(
                    workspace.flags_i32.ptr(),
                    workspace.dso.ptr(),
                    cap,
                    workspace,
                )?;
                (doc_starts, workspace.dso.ptr(), 0)
            }
            Ruleset::Cl100k => (0, 0, 0),
            Ruleset::O200k | Ruleset::Kimi => unreachable!(),
        };
        self.stream
            .memset_u8(workspace.heads.ptr(), 0, cap)
            .map_err(driver_error)?;
        self.launch(
            &format!("k_classify_rs{numeric}"),
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.cp.ptr()),
                KernelArg::Ptr(self.tables.class.ptr()),
                KernelArg::Ptr(bmask),
                KernelArg::Ptr(workspace.cls.ptr()),
                KernelArg::Ptr(workspace.heads.ptr()),
                KernelArg::I32(0),
                KernelArg::Ptr(char_count),
            ],
        )?;
        self.scan(
            workspace.heads.ptr(),
            workspace.rids.ptr(),
            cap,
            true,
            workspace,
        )?;
        let run_count = add(workspace.rids.ptr(), 4 * (cap - 1));
        self.stream
            .memset_u32(workspace.chan_hi.ptr(), HI, cap)
            .map_err(driver_error)?;
        self.stream
            .memset_u32(workspace.chan_neg.ptr(), u32::MAX, cap)
            .map_err(driver_error)?;
        self.launch(
            &format!("k_runinfo_rs{numeric}"),
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.cp.ptr()),
                KernelArg::Ptr(workspace.cls.ptr()),
                KernelArg::Ptr(workspace.heads.ptr()),
                KernelArg::Ptr(workspace.rids.ptr()),
                KernelArg::Ptr(workspace.run_start.ptr()),
                KernelArg::Ptr(workspace.chan_hi.ptr()),
                KernelArg::Ptr(workspace.chan_neg.ptr()),
                KernelArg::I32(0),
                KernelArg::Ptr(char_count),
            ],
        )?;
        self.stream
            .memset_u8(workspace.starts.ptr(), 0, cap)
            .map_err(driver_error)?;
        self.launch(
            &format!("k_rules_rs{numeric}"),
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.cp.ptr()),
                KernelArg::Ptr(workspace.cls.ptr()),
                KernelArg::Ptr(workspace.heads.ptr()),
                KernelArg::Ptr(workspace.rids.ptr()),
                KernelArg::Ptr(workspace.run_start.ptr()),
                KernelArg::Ptr(workspace.chan_hi.ptr()),
                KernelArg::Ptr(workspace.chan_neg.ptr()),
                KernelArg::Ptr(dso),
                KernelArg::Ptr(bmask),
                KernelArg::Ptr(ars),
                KernelArg::I32(0),
                KernelArg::I32(self.digits_max),
                KernelArg::Ptr(workspace.starts.ptr()),
                KernelArg::I32(0),
                KernelArg::Ptr(char_count),
                KernelArg::Ptr(run_count),
            ],
        )?;
        self.piece_dispatch(cap, workspace)
    }

    /// The middle section of the windowed splitter group.
    ///
    /// One set of five kernels serves both members: the Han branch is a
    /// runtime flag, and when it is on the rules kernel also fills the
    /// Han trigger plane. Everything else -- the four run channels, the
    /// anchor and back-reference channels, the rule evaluation -- is the
    /// same code for both.
    fn windowed_middle(
        &self,
        cap: usize,
        char_count: DevicePtr,
        doc_starts: DevicePtr,
        workspace: &Workspace,
    ) -> Result<(), NativeRuntimeError> {
        let han = self.ruleset.han_branch();
        self.stream
            .memset_u8(workspace.heads.ptr(), 0, 4 * cap)
            .map_err(driver_error)?;
        self.launch(
            "k_o2k_heads",
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.cp.ptr()),
                KernelArg::Ptr(self.tables.class.ptr()),
                KernelArg::I32(0),
                KernelArg::Ptr(char_count),
                KernelArg::Ptr(doc_starts),
                KernelArg::Ptr(workspace.cls.ptr()),
                KernelArg::Ptr(workspace.heads.ptr()),
                KernelArg::Ptr(add(workspace.heads.ptr(), cap)),
                KernelArg::Ptr(add(workspace.heads.ptr(), 2 * cap)),
                KernelArg::Ptr(add(workspace.heads.ptr(), 3 * cap)),
                KernelArg::Bool(han),
            ],
        )?;
        for channel in 0..4 {
            self.scan(
                add(workspace.heads.ptr(), channel * cap),
                add(workspace.rids.ptr(), channel * 4 * cap),
                cap,
                true,
                workspace,
            )?;
        }
        let rid_m = workspace.rids.ptr();
        let rid_cs = add(workspace.rids.ptr(), 4 * cap);
        let rid_lw = add(workspace.rids.ptr(), 8 * cap);
        let rid_pm = add(workspace.rids.ptr(), 12 * cap);
        self.stream
            .memset_u32(workspace.chan_hi.ptr(), HI, 7 * cap)
            .map_err(driver_error)?;
        self.stream
            .memset_u32(workspace.chan_neg.ptr(), u32::MAX, 4 * cap)
            .map_err(driver_error)?;
        let first_anchor = workspace.chan_hi.ptr();
        let s_fl = add(workspace.chan_hi.ptr(), 4 * cap);
        let p_fl = add(workspace.chan_hi.ptr(), 8 * cap);
        let f_l0 = add(workspace.chan_hi.ptr(), 12 * cap);
        let f_l1 = add(workspace.chan_hi.ptr(), 16 * cap);
        let f_l2 = add(workspace.chan_hi.ptr(), 20 * cap);
        let pm_fp = add(workspace.chan_hi.ptr(), 24 * cap);
        let last_m_pm = workspace.chan_neg.ptr();
        let s_lc = add(workspace.chan_neg.ptr(), 4 * cap);
        let last_l = add(workspace.chan_neg.ptr(), 8 * cap);
        let last_c = add(workspace.chan_neg.ptr(), 12 * cap);
        self.launch(
            "k_o2k_runinfo1",
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.cp.ptr()),
                KernelArg::Ptr(workspace.cls.ptr()),
                KernelArg::Ptr(workspace.heads.ptr()),
                KernelArg::Ptr(rid_m),
                KernelArg::Ptr(rid_cs),
                KernelArg::Ptr(rid_pm),
                KernelArg::I32(0),
                KernelArg::Ptr(char_count),
                KernelArg::Ptr(doc_starts),
                KernelArg::Ptr(workspace.run_start.ptr()),
                KernelArg::Ptr(first_anchor),
                KernelArg::Ptr(last_m_pm),
                KernelArg::Bool(han),
            ],
        )?;
        self.launch(
            "k_o2k_runinfo2",
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.cp.ptr()),
                KernelArg::Ptr(workspace.cls.ptr()),
                KernelArg::Ptr(rid_m),
                KernelArg::Ptr(rid_cs),
                KernelArg::Ptr(rid_lw),
                KernelArg::Ptr(rid_pm),
                KernelArg::Ptr(workspace.run_start.ptr()),
                KernelArg::Ptr(first_anchor),
                KernelArg::I32(0),
                KernelArg::Ptr(char_count),
                KernelArg::Ptr(s_fl),
                KernelArg::Ptr(s_lc),
                KernelArg::Ptr(p_fl),
                KernelArg::Ptr(last_l),
                KernelArg::Ptr(last_c),
                KernelArg::Ptr(f_l0),
                KernelArg::Ptr(f_l1),
                KernelArg::Ptr(f_l2),
                KernelArg::Ptr(pm_fp),
                KernelArg::Bool(han),
            ],
        )?;
        self.stream
            .memset_u8(workspace.starts.ptr(), 0, cap)
            .map_err(driver_error)?;
        self.stream
            .memset_u8(workspace.trig2.ptr(), 0, 3 * cap)
            .map_err(driver_error)?;
        self.stream
            .memset_u32(workspace.sparse_flags.ptr(), 0, 2)
            .map_err(driver_error)?;
        self.launch(
            "k_o2k_rules",
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.cp.ptr()),
                KernelArg::Ptr(workspace.cls.ptr()),
                KernelArg::Ptr(rid_m),
                KernelArg::Ptr(rid_cs),
                KernelArg::Ptr(rid_lw),
                KernelArg::Ptr(rid_pm),
                KernelArg::Ptr(workspace.run_start.ptr()),
                KernelArg::Ptr(first_anchor),
                KernelArg::Ptr(s_fl),
                KernelArg::Ptr(s_lc),
                KernelArg::Ptr(p_fl),
                KernelArg::Ptr(last_l),
                KernelArg::Ptr(last_c),
                KernelArg::Ptr(f_l0),
                KernelArg::Ptr(f_l1),
                KernelArg::Ptr(f_l2),
                KernelArg::Ptr(pm_fp),
                KernelArg::Ptr(last_m_pm),
                KernelArg::I32(0),
                KernelArg::I32(self.digits_max),
                KernelArg::Bool(self.contractions),
                KernelArg::Ptr(workspace.starts.ptr()),
                KernelArg::Ptr(workspace.trig2.ptr()),
                KernelArg::Ptr(add(workspace.trig2.ptr(), cap)),
                KernelArg::Ptr(workspace.sparse_flags.ptr()),
                KernelArg::I32(0),
                KernelArg::Ptr(char_count),
                KernelArg::Ptr(add(workspace.sparse_flags.ptr(), 4)),
                KernelArg::Ptr(doc_starts),
                KernelArg::Bool(han),
                KernelArg::Ptr(if han { han_plane(cap, workspace) } else { 0 }),
            ],
        )?;
        self.piece_dispatch(cap, workspace)
    }

    fn piece_dispatch(&self, cap: usize, workspace: &Workspace) -> Result<(), NativeRuntimeError> {
        self.select(
            workspace.bo.ptr(),
            workspace.starts.ptr(),
            workspace.pb.ptr(),
            workspace.counts.ptr(),
            cap,
            workspace,
        )?;
        self.launch(
            "k_pb_sentinel",
            1,
            1,
            &[
                KernelArg::Ptr(workspace.pb.ptr()),
                KernelArg::Ptr(workspace.counts.ptr()),
                KernelArg::Ptr(workspace.nb.ptr()),
            ],
        )?;
        self.stream
            .memset_u32(workspace.token_counts.ptr(), 0, cap)
            .map_err(driver_error)?;
        self.launch(
            "k_dispatch_flags",
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.pb.ptr()),
                KernelArg::Ptr(workspace.counts.ptr()),
                KernelArg::Ptr(workspace.dispatch_flags.ptr()),
                KernelArg::Ptr(add(workspace.dispatch_flags.ptr(), cap)),
                KernelArg::Ptr(add(workspace.dispatch_flags.ptr(), 2 * cap)),
                KernelArg::Ptr(workspace.token_counts.ptr()),
                KernelArg::I32(i32_of(cap)?),
            ],
        )?;
        for channel in 0..3 {
            self.select(
                0,
                add(workspace.dispatch_flags.ptr(), channel * cap),
                add(workspace.lists.ptr(), channel * 4 * cap),
                add(workspace.counts.ptr(), 4 * (channel + 1)),
                cap,
                workspace,
            )?;
        }
        Ok(())
    }

    fn bpe_tail(&self, cap: usize, workspace: &Workspace) -> Result<(), NativeRuntimeError> {
        let pair_mask = u32_of(self.pair_count - 1)?;
        let vocab_mask = u32_of(self.vocab_count - 1)?;
        let unsafe_ptr = if self.tables.unsafe_bits.bytes() == 0 {
            0
        } else {
            self.tables.unsafe_bits.ptr()
        };
        self.launch(
            "k_bpe_thread_cap32",
            grid(cap)?,
            TPB,
            &[
                KernelArg::Ptr(workspace.data.ptr()),
                KernelArg::Ptr(workspace.pb.ptr()),
                KernelArg::Ptr(workspace.lists.ptr()),
                KernelArg::I32(0),
                KernelArg::Ptr(self.tables.pair_keys.ptr()),
                KernelArg::Ptr(self.tables.pair_vals.ptr()),
                KernelArg::U32(pair_mask),
                KernelArg::Ptr(self.tables.byte_id.ptr()),
                KernelArg::Ptr(self.tables.vocab_keys.ptr()),
                KernelArg::Ptr(self.tables.vocab_vals.ptr()),
                KernelArg::U32(vocab_mask),
                KernelArg::Ptr(self.tables.vocab_blob.ptr()),
                KernelArg::I32(self.ignore_merges),
                KernelArg::Ptr(workspace.scratch.ptr()),
                KernelArg::Ptr(workspace.token_counts.ptr()),
                KernelArg::Ptr(add(workspace.counts.ptr(), 4)),
                KernelArg::Ptr(0),
                KernelArg::Ptr(0),
                KernelArg::Ptr(0),
                KernelArg::Ptr(0),
                KernelArg::U32(0),
            ],
        )?;
        self.launch(
            "k_bpe_warp",
            grid(cap)?.min(8192),
            TPB,
            &[
                KernelArg::Ptr(workspace.data.ptr()),
                KernelArg::Ptr(workspace.pb.ptr()),
                KernelArg::Ptr(add(workspace.lists.ptr(), 4 * cap)),
                KernelArg::I32(0),
                KernelArg::Ptr(add(workspace.counts.ptr(), 8)),
                KernelArg::Ptr(self.tables.pair_keys.ptr()),
                KernelArg::Ptr(self.tables.pair_vals.ptr()),
                KernelArg::U32(pair_mask),
                KernelArg::Ptr(self.tables.byte_id.ptr()),
                KernelArg::Ptr(self.tables.vocab_keys.ptr()),
                KernelArg::Ptr(self.tables.vocab_vals.ptr()),
                KernelArg::U32(vocab_mask),
                KernelArg::Ptr(self.tables.vocab_blob.ptr()),
                KernelArg::I32(self.ignore_merges),
                KernelArg::Ptr(workspace.scratch.ptr()),
                KernelArg::Ptr(workspace.token_counts.ptr()),
                KernelArg::Ptr(unsafe_ptr),
            ],
        )?;
        self.launch(
            "k_bpe_long",
            u32_of((cap / (MED_MAX + 1) + 1).min(8192))?,
            TPB,
            &[
                KernelArg::Ptr(workspace.data.ptr()),
                KernelArg::Ptr(workspace.pb.ptr()),
                KernelArg::Ptr(add(workspace.lists.ptr(), 8 * cap)),
                KernelArg::Ptr(self.tables.pair_keys.ptr()),
                KernelArg::Ptr(self.tables.pair_vals.ptr()),
                KernelArg::U32(pair_mask),
                KernelArg::Ptr(self.tables.byte_id.ptr()),
                KernelArg::Ptr(self.tables.vocab_keys.ptr()),
                KernelArg::Ptr(self.tables.vocab_vals.ptr()),
                KernelArg::U32(vocab_mask),
                KernelArg::Ptr(self.tables.vocab_blob.ptr()),
                KernelArg::I32(self.ignore_merges),
                KernelArg::Ptr(workspace.scratch.ptr()),
                KernelArg::Ptr(workspace.long_buffer.ptr()),
                KernelArg::Ptr(workspace.token_counts.ptr()),
                KernelArg::I32(0),
                KernelArg::Ptr(add(workspace.counts.ptr(), 12)),
                KernelArg::Ptr(unsafe_ptr),
            ],
        )?;
        self.stream
            .memset_u32(workspace.offsets.ptr(), 0, 1)
            .map_err(driver_error)?;
        self.scan(
            workspace.token_counts.ptr(),
            add(workspace.offsets.ptr(), 4),
            cap,
            false,
            workspace,
        )?;
        self.launch(
            "k_bpe_compact",
            u32_of(cap.min(65535))?,
            64,
            &[
                KernelArg::Ptr(workspace.pb.ptr()),
                KernelArg::Ptr(workspace.offsets.ptr()),
                KernelArg::Ptr(workspace.scratch.ptr()),
                KernelArg::Ptr(workspace.output.ptr()),
                KernelArg::I32(0),
                KernelArg::Ptr(workspace.counts.ptr()),
            ],
        )
    }
}

pub struct NativePrebuiltGpu {
    family: String,
    artifact_sha256: String,
    expected_architecture: String,
    needs_nfc: bool,
    delivery: String,
    reference: ReferenceEngine,
    executor: Mutex<Executor>,
}

impl std::fmt::Debug for NativePrebuiltGpu {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativePrebuiltGpu")
            .field("family", &self.family)
            .field("artifact_sha256", &self.artifact_sha256)
            .field("architecture", &self.expected_architecture)
            .finish_non_exhaustive()
    }
}

impl NativePrebuiltGpu {
    /// Construction-time validation that needs no device: the ruleset name
    /// and the table shapes. A deferred engine runs this when its inputs
    /// are projected, so a malformed projection still fails at
    /// construction; the digest re-check and every device check run in
    /// [`NativePrebuiltGpu::new`] when the engine actually opens.
    #[allow(clippy::too_many_arguments)]
    pub fn preflight(
        config: &NativePrebuiltGpuConfig,
        class_table: &[u8],
        pair_keys: &[u8],
        pair_vals: &[u8],
        byte_id: &[u8],
        vocab_keys: &[u8],
        vocab_vals: &[u8],
        unsafe_bits: &[u8],
    ) -> Result<(), NativeRuntimeError> {
        Ruleset::parse(&config.ruleset)?;
        validate_tables(
            config,
            class_table,
            pair_keys,
            pair_vals,
            byte_id,
            vocab_keys,
            vocab_vals,
            unsafe_bits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: NativePrebuiltGpuConfig,
        reference: ReferenceEngine,
        fatbin: &[u8],
        symbols: BTreeMap<String, String>,
        class_table: &[u8],
        pair_keys: &[u8],
        pair_vals: &[u8],
        byte_id: &[u8],
        vocab_keys: &[u8],
        vocab_vals: &[u8],
        vocab_blob: &[u8],
        unsafe_bits: &[u8],
    ) -> Result<Self, NativeRuntimeError> {
        let executor = Executor::new(
            &config,
            fatbin,
            &symbols,
            class_table,
            pair_keys,
            pair_vals,
            byte_id,
            vocab_keys,
            vocab_vals,
            vocab_blob,
            unsafe_bits,
        )?;
        Ok(Self {
            family: config.family,
            artifact_sha256: config.artifact_sha256,
            expected_architecture: config.expected_architecture,
            needs_nfc: config.needs_nfc,
            delivery: config.delivery,
            reference,
            executor: Mutex::new(executor),
        })
    }
}

impl NativeGpuEngine for NativePrebuiltGpu {
    fn encode_ids(&self, text: &str) -> Result<Vec<u32>, NativeRuntimeError> {
        let normalized;
        let input = if self.needs_nfc && !text.is_ascii() {
            normalized = self
                .reference
                .normalize(text)
                .map_err(|error| NativeRuntimeError::new(error.to_string()))?;
            normalized.as_str()
        } else {
            text
        };
        self.executor
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .encode(input)
    }

    fn encode_batch_ids(&self, texts: &[&str]) -> Vec<Result<Vec<u32>, NativeRuntimeError>> {
        let normalized = texts
            .iter()
            .map(|text| {
                if self.needs_nfc && !text.is_ascii() {
                    self.reference
                        .normalize(text)
                        .map(Cow::Owned)
                        .map_err(|error| NativeRuntimeError::new(error.to_string()))
                } else {
                    Ok(Cow::Borrowed(*text))
                }
            })
            .collect::<Vec<_>>();
        let mut output = (0..texts.len()).map(|_| None).collect::<Vec<_>>();
        let valid = normalized
            .iter()
            .enumerate()
            .filter_map(|(index, value)| match value {
                Ok(_) => Some(index),
                Err(error) => {
                    output[index] = Some(Err(error.clone()));
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut executor = self
            .executor
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut cursor = 0usize;
        while cursor < valid.len() {
            let start = cursor;
            let mut bytes = 0usize;
            while cursor < valid.len() && cursor - start < MAX_BATCH_DOCS {
                let index = valid[cursor];
                let text = normalized[index].as_ref().expect("valid index").as_ref();
                if cursor > start && bytes.saturating_add(text.len()) > MAX_BATCH_BYTES {
                    break;
                }
                bytes = bytes.saturating_add(text.len());
                cursor += 1;
                if bytes >= MAX_BATCH_BYTES {
                    break;
                }
            }
            let indices = &valid[start..cursor];
            let batch = indices
                .iter()
                .map(|index| normalized[*index].as_ref().expect("valid index").as_ref())
                .collect::<Vec<_>>();
            match executor.encode_batch(&batch) {
                Ok(rows) if rows.len() == indices.len() => {
                    for (index, row) in indices.iter().copied().zip(rows) {
                        output[index] = Some(Ok(row));
                    }
                }
                Ok(_) => {
                    let error =
                        NativeRuntimeError::new("native GPU batch returned the wrong row count");
                    for index in indices {
                        output[*index] = Some(Err(error.clone()));
                    }
                }
                Err(_batch_error) => {
                    // Sparse o200k guards and other document-local failures
                    // are isolated so one row cannot force unrelated rows off
                    // the accelerated path.
                    for (index, text) in indices.iter().copied().zip(batch) {
                        output[index] = Some(executor.encode(text));
                    }
                }
            }
        }
        output
            .into_iter()
            .map(|value| {
                value.unwrap_or_else(|| {
                    Err(NativeRuntimeError::new(
                        "native GPU batch left a row unresolved",
                    ))
                })
            })
            .collect()
    }

    fn delivery(&self) -> &str {
        &self.delivery
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_tables(
    config: &NativePrebuiltGpuConfig,
    class_table: &[u8],
    pair_keys: &[u8],
    pair_vals: &[u8],
    byte_id: &[u8],
    vocab_keys: &[u8],
    vocab_vals: &[u8],
    unsafe_bits: &[u8],
) -> Result<(), NativeRuntimeError> {
    if class_table.len() < FULL_PLANE {
        return Err(NativeRuntimeError::new(
            "native GPU class table does not cover U+10FFFF",
        ));
    }
    if !config.pair_count.is_power_of_two()
        || pair_keys.len() != 8 * config.pair_count
        || pair_vals.len() != pair_keys.len()
    {
        return Err(NativeRuntimeError::new(
            "native GPU pair tables have an invalid shape",
        ));
    }
    if !config.vocab_count.is_power_of_two()
        || vocab_keys.len() != 8 * config.vocab_count
        || vocab_vals.len() != vocab_keys.len()
    {
        return Err(NativeRuntimeError::new(
            "native GPU vocabulary tables have an invalid shape",
        ));
    }
    if byte_id.len() != 256 * 4 || !unsafe_bits.len().is_multiple_of(4) {
        return Err(NativeRuntimeError::new(
            "native GPU byte-id or unsafe-rank table has an invalid shape",
        ));
    }
    Ok(())
}

/// Third trigger plane of the workspace, one byte per input byte: where
/// the Han branch met a Han character that is also a number or a
/// punctuation mark, which the parallel rules do not decide alone.
fn han_plane(cap: usize, workspace: &Workspace) -> DevicePtr {
    add(workspace.trig2.ptr(), 2 * cap)
}

fn required_functions(ruleset: Ruleset) -> Vec<&'static str> {
    let mut names = vec![
        "tk_scan_u8_blocks",
        "tk_scan_i32_blocks",
        "tk_scan_add",
        "tk_utf8_lead_i32",
        "tk_u8_flags_i32",
        "tk_eq_index_i32",
        "tk_carrier_scatter",
        "tk_carrier_gather",
        "tk_select_scatter",
        "k_utf8_decode",
        "k_pb_sentinel",
        "k_dispatch_flags",
        "k_bpe_thread_cap32",
        "k_bpe_warp",
        "k_bpe_long",
        "k_bpe_compact",
    ];
    match ruleset {
        Ruleset::Cl100k | Ruleset::Deepseek | Ruleset::Laguna => {
            let numeric = ruleset.numeric().expect("generic");
            names.push(match numeric {
                0 => "k_classify_rs0",
                1 => "k_classify_rs1",
                _ => "k_classify_rs2",
            });
            names.push(match numeric {
                0 => "k_runinfo_rs0",
                1 => "k_runinfo_rs1",
                _ => "k_runinfo_rs2",
            });
            names.push(match numeric {
                0 => "k_rules_rs0",
                1 => "k_rules_rs1",
                _ => "k_rules_rs2",
            });
            if ruleset == Ruleset::Deepseek {
                names.extend(["k_ds_seed_n", "k_ds_bmask"]);
            } else if ruleset == Ruleset::Laguna {
                names.push("k_lag_bmask");
            }
        }
        Ruleset::O200k | Ruleset::Kimi => names.extend([
            "k_o2k_heads",
            "k_o2k_runinfo1",
            "k_o2k_runinfo2",
            "k_o2k_rules",
            "k_o2k_meta3",
        ]),
    }
    names
}

fn sha256_hex(raw: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"toktier.kernel_fatbin.v1\0");
    hasher.update(raw);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn add(pointer: DevicePtr, bytes: usize) -> DevicePtr {
    pointer + bytes as u64
}

fn i32_of(value: usize) -> Result<i32, NativeRuntimeError> {
    i32::try_from(value).map_err(|_| NativeRuntimeError::new("GPU dimension exceeds i32"))
}

fn u32_of(value: usize) -> Result<u32, NativeRuntimeError> {
    u32::try_from(value).map_err(|_| NativeRuntimeError::new("GPU dimension exceeds u32"))
}

fn grid(length: usize) -> Result<u32, NativeRuntimeError> {
    u32_of(length.div_ceil(TPB as usize))
}

fn driver_error(error: toktier_cuda_driver::CudaError) -> NativeRuntimeError {
    NativeRuntimeError::new(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rulesets_and_grid_are_frozen() {
        assert_eq!(Ruleset::parse("cl100k").unwrap().numeric(), Some(0));
        assert_eq!(Ruleset::parse("deepseek").unwrap().numeric(), Some(1));
        assert_eq!(Ruleset::parse("laguna").unwrap().numeric(), Some(2));
        assert_eq!(Ruleset::parse("o200k").unwrap().numeric(), None);
        assert_eq!(Ruleset::parse("kimi").unwrap().numeric(), None);
        assert!(Ruleset::parse("nonesuch").is_err());
        assert_eq!(grid(1).unwrap(), 1);
        assert_eq!(grid(257).unwrap(), 2);
    }

    #[test]
    fn the_windowed_group_shares_one_kernel_set() {
        let o200k = required_functions(Ruleset::O200k);
        assert_eq!(required_functions(Ruleset::Kimi), o200k);
        assert!(Ruleset::O200k.windowed() && Ruleset::Kimi.windowed());
        assert!(!Ruleset::Cl100k.windowed());
        // Only the Han member turns the branch on.
        assert!(Ruleset::Kimi.han_branch());
        assert!(!Ruleset::O200k.han_branch());
    }

    #[test]
    fn the_sparse_guard_reports_every_surface() {
        assert!(Tail::default().sparse_guard().is_none());
        for tail in [
            Tail {
                chain: 1,
                ..Tail::default()
            },
            Tail {
                ambiguous: 1,
                ..Tail::default()
            },
            Tail {
                han: 1,
                ..Tail::default()
            },
        ] {
            assert!(tail.sparse_guard().is_some());
        }
    }
}
