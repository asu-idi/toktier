// Prebuilt-delivery compilation unit (fatbin build input).
//
// This translation unit includes `pretok_kernel.cu` verbatim -- the JIT
// source stays byte-identical, so every existing `certified_source`
// record keeps binding the digest it was judged against -- and adds the
// small set of launcher-support kernels the Python-side launcher needs
// to replace the host-side CUB calls (DeviceScan / DeviceSelect) that
// a driver-API module cannot reach. The torch/pybind host glue in the
// included file is parsed but contributes no device code; only the
// `__global__` kernels land in the fatbin.
//
// Built by `tools/build_fatbin.py` with a single `nvcc -fatbin`
// invocation; the build manifest records the toolchain, the full
// argument list, the per-architecture cubin digests and the source
// digest lineage of both this file and the included kernel source.

// ---- sub-sm_80 compatibility shim -----------------------------------
// `__reduce_min_sync` is a compute-capability >= 8.0 builtin; on the
// sm_75 (and compute_75 PTX) passes the identifier does not exist, so
// the include below would not compile. The shim substitutes a shuffle
// reduction with the same result for a full-warp mask, which is the
// only mask the kernel uses (`bpe_warp_one` passes 0xFFFFFFFF). Device
// passes for sm_80 and newer keep the hardware builtin, and the host
// pass (no __CUDA_ARCH__) never sees the macro. Architectures below
// sm_80 are experimental-tier deliveries; the substitution is recorded
// in the build manifest.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ __forceinline__ unsigned toktier_warp_min_u32(unsigned mask,
                                                         unsigned v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const unsigned other = __shfl_xor_sync(mask, v, offset);
    v = other < v ? other : v;
  }
  return v;
}
#define __reduce_min_sync(m, v) toktier_warp_min_u32((m), (v))
#endif

#include "pretok_kernel.cu"

// Direct Rust JIT defines TOKTIER_DEVICE_ONLY so the legacy Torch host glue
// is not parsed. Those host launch sites used to instantiate these templates
// as a side effect; keep the exact device symbols explicit for the shared
// CUDA Driver host instead.
#ifdef TOKTIER_DEVICE_ONLY
template __global__ void k_classify<RS_CL100K>(
    const int32_t*, const uint8_t*, const uint8_t*, uint8_t*, uint8_t*, int,
    const int32_t*);
template __global__ void k_classify<RS_DEEPSEEK>(
    const int32_t*, const uint8_t*, const uint8_t*, uint8_t*, uint8_t*, int,
    const int32_t*);
template __global__ void k_classify<RS_LAGUNA>(
    const int32_t*, const uint8_t*, const uint8_t*, uint8_t*, uint8_t*, int,
    const int32_t*);

template __global__ void k_runinfo<RS_CL100K>(
    const int32_t*, const uint8_t*, const uint8_t*, const int32_t*, int32_t*,
    int32_t*, int32_t*, int, const int32_t*);
template __global__ void k_runinfo<RS_DEEPSEEK>(
    const int32_t*, const uint8_t*, const uint8_t*, const int32_t*, int32_t*,
    int32_t*, int32_t*, int, const int32_t*);
template __global__ void k_runinfo<RS_LAGUNA>(
    const int32_t*, const uint8_t*, const uint8_t*, const int32_t*, int32_t*,
    int32_t*, int32_t*, int, const int32_t*);

template __global__ void k_rules<RS_CL100K>(
    const int32_t*, const uint8_t*, const uint8_t*, const int32_t*,
    const int32_t*, const int32_t*, const int32_t*, const int32_t*,
    const uint8_t*, const int32_t*, int, int, bool*, int, const int32_t*,
    const int32_t*);
template __global__ void k_rules<RS_DEEPSEEK>(
    const int32_t*, const uint8_t*, const uint8_t*, const int32_t*,
    const int32_t*, const int32_t*, const int32_t*, const int32_t*,
    const uint8_t*, const int32_t*, int, int, bool*, int, const int32_t*,
    const int32_t*);
template __global__ void k_rules<RS_LAGUNA>(
    const int32_t*, const uint8_t*, const uint8_t*, const int32_t*,
    const int32_t*, const int32_t*, const int32_t*, const int32_t*,
    const uint8_t*, const int32_t*, int, int, bool*, int, const int32_t*,
    const int32_t*);

template __global__ void k_bpe_thread<TOKTIER_SHORT_MAX>(
    const uint8_t*, const int32_t*, const int32_t*, int, const uint64_t*,
    const uint64_t*, unsigned, const int32_t*, const uint64_t*,
    const uint64_t*, unsigned, const uint8_t*, int, int32_t*, int32_t*,
    const int32_t*, const uint64_t*, const int32_t*, const uint8_t*,
    const int32_t*, unsigned);
#endif

// ---- launcher-support kernels ---------------------------------------
// The Python launcher replaces the two host-side CUB algorithms with
// exact integer primitives:
//
// * `cub::DeviceScan::InclusiveSum` -> `torch.cumsum` (integer, exact).
// * `cub::DeviceScan::InclusiveScan(maximum)` over the seed arrays of
//   this file -> the carrier reformulation below. Every max-scan site
//   in the kernel source uses seeds of the shape `seed[i] = i` on
//   "carrier" positions and `seed[i] = 0` elsewhere, with the carrier
//   set containing position 0 (seed[0] is 0 = its own index in every
//   entry path). For such seeds the inclusive max equals "the index of
//   the most recent carrier at or before i": carriers contribute their
//   own index, non-carriers contribute 0, and indices grow, so the
//   maximum over a prefix is the latest carrier index in it. That value
//   is recovered exactly with a carrier-run id (an inclusive sum over
//   the carrier flags), a scatter of each carrier's index, and a
//   gather -- three integer passes, deterministic on any architecture.
// * `cub::DeviceSelect::Flagged` -> an inclusive sum over the flags
//   plus `tk_select_scatter` (stable compaction with the selected count
//   stored on the device, as the fused graph path requires).
//
// All three kernels are `extern "C"` so the driver-API loader can look
// them up without a mangled-name map.

// Inclusive integer scan used by the native CUDA Driver host.  One block
// consumes consecutive pairs, so the BlockScan prefix is in array order.
// The host recursively scans `block_sums`, then `tk_scan_add` applies the
// preceding-block prefix.  This is the same associative integer sum as CUB
// and torch.cumsum; no floating-point or architecture-dependent operation is
// involved.
template <typename Input>
__device__ __forceinline__ void tk_scan_block_body(
    const Input* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ block_sums,
    int n) {
  using Scan = cub::BlockScan<int32_t, TPB>;
  __shared__ typename Scan::TempStorage temporary;
  const int pair = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t first = pair < n ? static_cast<int32_t>(input[pair]) : 0;
  const int32_t second = pair + 1 < n
      ? static_cast<int32_t>(input[pair + 1]) : 0;
  const int32_t local = first + second;
  int32_t inclusive = 0;
  Scan(temporary).InclusiveSum(local, inclusive);
  const int32_t exclusive = inclusive - local;
  if (pair < n) output[pair] = exclusive + first;
  if (pair + 1 < n) output[pair + 1] = inclusive;
  if (threadIdx.x == blockDim.x - 1 && block_sums)
    block_sums[blockIdx.x] = inclusive;
}

extern "C" __global__ void tk_scan_u8_blocks(
    const uint8_t* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ block_sums,
    int n) {
  tk_scan_block_body(input, output, block_sums, n);
}

extern "C" __global__ void tk_scan_i32_blocks(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ block_sums,
    int n) {
  tk_scan_block_body(input, output, block_sums, n);
}

extern "C" __global__ void tk_scan_add(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ scanned_block_sums,
    int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int owner = i / (2 * TPB);
  if (owner > 0) output[i] += scanned_block_sums[owner - 1];
}

extern "C" __global__ void tk_utf8_lead_i32(
    const uint8_t* __restrict__ input,
    int32_t* __restrict__ output,
    int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = (input[i] & 0xC0) != 0x80;
}

extern "C" __global__ void tk_u8_flags_i32(
    const uint8_t* __restrict__ input,
    int32_t* __restrict__ output,
    int n,
    bool force_first) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = (force_first && i == 0) || input[i] != 0;
}

extern "C" __global__ void tk_eq_index_i32(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[i] == i;
}

extern "C" __global__ void tk_carrier_scatter(
    const int32_t* __restrict__ rid,   // inclusive sum of carrier flags
    const int32_t* __restrict__ flag,  // nonzero = carrier; flag[0] != 0
    int32_t* __restrict__ pos,         // [num carriers] carrier indices
    int cap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  if (flag[i]) pos[rid[i] - 1] = i;
}

extern "C" __global__ void tk_carrier_gather(
    const int32_t* __restrict__ rid,
    const int32_t* __restrict__ pos,
    int32_t* __restrict__ out,         // out[i] = latest carrier <= i
    int cap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  out[i] = pos[rid[i] - 1];
}

extern "C" __global__ void tk_select_scatter(
    const int32_t* __restrict__ vals,   // nullable: null = identity (iota)
    const uint8_t* __restrict__ flags,  // nonzero selects (bool or uint8)
    const int32_t* __restrict__ psum,   // inclusive sum of flags
    int32_t* __restrict__ out,
    int32_t* __restrict__ count_out,    // nullable: device-side count
    int cap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  if (flags[i]) out[psum[i] - 1] = vals ? vals[i] : i;
  if (count_out && i == cap - 1) *count_out = psum[i];
}

// Self-description of the DeepSeek inline constants, read from the
// device code itself (same role as the host-side `ds_constants()` JSON:
// a build carrying planted `-DTOKTIER_DS_*` mutations shows up here,
// because the values below come from the same macros and predicates the
// judged kernels compile against). Layout: out[0..5] = the three CJK
// ranges (lo, hi pairs), out[6..12] = the class enum values
// (O, L, M, N, PS, WS, CRLF), out[13 + c] = bit 0 apunct(c), bit 1
// alpha(c) for c in 0..127. Total 141 int32 values.
extern "C" __global__ void tk_ds_constants_dump(int32_t* __restrict__ out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  out[0] = TOKTIER_DS_CJK0_LO;
  out[1] = TOKTIER_DS_CJK0_HI;
  out[2] = TOKTIER_DS_CJK1_LO;
  out[3] = TOKTIER_DS_CJK1_HI;
  out[4] = TOKTIER_DS_CJK2_LO;
  out[5] = TOKTIER_DS_CJK2_HI;
  out[6] = DS_O;
  out[7] = DS_L;
  out[8] = DS_M;
  out[9] = DS_N;
  out[10] = DS_PS;
  out[11] = DS_WS;
  out[12] = DS_CRLF;
  for (int c = 0; c < 128; ++c)
    out[13 + c] = (ds_apunct(c) ? 1 : 0) | (ds_alpha(c) ? 2 : 0);
}
