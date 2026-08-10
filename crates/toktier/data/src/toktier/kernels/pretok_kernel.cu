// Fused CUDA implementation of GPT-style pre-tokenization plus
// byte-level BPE encoding. The splitter semantics equal those of the
// sequential reference tokenizer and are pinned against the reference
// engine. It runs as 4 kernels plus 1 CUB scan and decides piece starts
// per character; the target memory traffic is ~35B per character (an
// eager torch formulation costs ~300B per character).
//
// Per-character reformulation of the rules (equivalent to the run-level
// scatter form of the torch reference; each branch was argued
// separately):
//   i==0                -> always a piece start
//   class N             -> (i - run_start) % digits_max == 0
//   class L, run head   -> not merged into the preceding piece: the
//                          previous character is neither "non-CRLF
//                          whitespace" (trailing-space merge) nor a
//                          "len-1 P run with arrival" (A2/A1 merge)
//   class L, non-head   -> A1 contraction remainder: after 'X
//                          (look-back 2) or 'XY (look-back 3), provided
//                          that apostrophe is a len-1 P run, has
//                          arrival, and the case-folded letters match
//   class P, run head   -> previous character is not an ASCII space
//                          (otherwise A4's ' '? consumes it)
//   class S             -> decided from run-level quantities
//                          (rs/re/first_noncrlf/last_crlf): the A5
//                          start, the trailing-segment start and the
//                          standalone final space
//
// Ruleset variants share this machinery: RS_CL100K is the GPT-style
// path, RS_DEEPSEEK adds the DeepSeek 3-splitter rules and RS_LAGUNA
// prepends a newline-run stage to the GPT-style body; self-contained
// kernels further down cover the o200k and kimi splitters. The
// byte-level BPE stage and the fully fused, CUDA-Graph capturable
// bytes -> token ids path live in the second half of this file.

#ifndef TOKTIER_DEVICE_ONLY
#include <torch/extension.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cuda/functional>

#ifndef TOKTIER_TPB
#define TOKTIER_TPB 256
#endif
constexpr int TPB = TOKTIER_TPB;               // -DTOKTIER_TPB per-arch tune
// TPB must be a power of two: the tree reduction in k_bpe_long halves
// blockDim.x, and the WPB=TPB/32 warp split plus the lane bit arithmetic
// all rely on it. It must also be >= 32: with TPB<32 we get WPB=TPB/32=0,
// so the per-warp shared arrays of k_bpe_warp are zero-length and the
// warp predicates are empty (power of two + >=32 => multiple of 32).
static_assert(TPB >= 32 && (TPB & (TPB - 1)) == 0,
              "TOKTIER_TPB must be a power of two >= 32");
constexpr uint8_t P_ = 0, L_ = 1, N_ = 2, S_ = 3;

// ==================== DeepSeek 3-splitter ruleset ====================
// Compile-time ruleset branch: RS_CL100K=0 is the existing GPT-style
// path (no code change; covered bit-identically by the family
// zero-regression gate); RS_DEEPSEEK=1 consumes the 7-class table
// pretok_deepseek_classes_v1_hfengine.npy (build_deepseek_class_table.py).
// Semantics = the single-pass DeepSeek formulation (0/5,206 differential
// mismatches against the torch reference): S0/S1 cut points are OR-ed as
// a B mask into the dstart channel, the S2 rules run in one pass, and
// run segmentation / look-back / end-of-piece assertions all take B as a
// reset boundary.
constexpr int RS_CL100K = 0, RS_DEEPSEEK = 1;
// RS_LAGUNA=2 = the Laguna-S-2.1 (poolside) two-stage Split group. Its
// stage-1 pattern is byte-identical to the qwen3_8b pattern, so the rule
// body and class table are reused from the RS_CL100K path in full
// (k_classify/k_runinfo/k_rules take the existing else branch for
// RS_LAGUNA; the RS_CL100K/RS_DEEPSEEK instantiations generate unchanged
// code). Stage-0 `(?:\r?\n)+` (behavior=MergedWithNext, i.e. cut only
// before a match) is the newline-run start predicate, OR-ed into the
// dstart channel (a B mask: a proper subset of the DeepSeek case, with
// segment-start/boundary meaning only, no ars channel, no extra rules).
// The negative look-ahead (?!\r?\n) never fires under the greedy +
// (measured 0/60,058), so the predicate need not express it.
constexpr int RS_LAGUNA = 2;
// DeepSeek class enum (value-for-value aligned with the single source
// build_deepseek_class_table.py; the Python side asserts it against
// ds_constants() instead of keeping a second copy)
constexpr uint8_t DS_O = 0, DS_L = 1, DS_M = 2, DS_N = 3, DS_PS = 4,
                  DS_WS = 5, DS_CRLF = 6;
// Inline constants (the three CJK ranges, ASCII punctuation and ASCII
// letters are deliberately kept out of the class table; the values are
// mechanically extracted from the class-table audit report, field
// extracted_constants). TOKTIER_DS_* may be overridden with -D as a
// planted-mutation channel for testing; production builds must not.
#ifndef TOKTIER_DS_CJK0_LO
#define TOKTIER_DS_CJK0_LO 0x4E00
#endif
#ifndef TOKTIER_DS_CJK0_HI
#define TOKTIER_DS_CJK0_HI 0x9FA5      // upper bound is 9FA5, not 9FFF
#endif
#ifndef TOKTIER_DS_CJK1_LO
#define TOKTIER_DS_CJK1_LO 0x3040      // Hiragana; includes 3099/309A Mn
                                       // and the unassigned 3040
#endif
#ifndef TOKTIER_DS_CJK1_HI
#define TOKTIER_DS_CJK1_HI 0x309F
#endif
#ifndef TOKTIER_DS_CJK2_LO
#define TOKTIER_DS_CJK2_LO 0x30A0      // Katakana; 30A0 is a Pd
#endif
#ifndef TOKTIER_DS_CJK2_HI
#define TOKTIER_DS_CJK2_HI 0x30FF
#endif
__host__ __device__ __forceinline__ bool ds_cjk(int c) {
  return (c >= TOKTIER_DS_CJK0_LO && c <= TOKTIER_DS_CJK0_HI) ||
         (c >= TOKTIER_DS_CJK1_LO && c <= TOKTIER_DS_CJK1_HI) ||
         (c >= TOKTIER_DS_CJK2_LO && c <= TOKTIER_DS_CJK2_HI);
}
__host__ __device__ __forceinline__ bool ds_alpha(int c) {   // A1 [A-Za-z]
  return (c >= 0x41 && c <= 0x5A) || (c >= 0x61 && c <= 0x7A);
}
__host__ __device__ __forceinline__ bool ds_apunct(int c) {  // A1 head, 32 ch
  return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) ||
         (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);
}
// Merged run class: L/M in one run (M joins the letter body), WS/CRLF in
// one run (WSF)
__device__ __forceinline__ uint8_t ds_rc(uint8_t c) {
  if (c == DS_M) return DS_L;
  if (c == DS_CRLF) return DS_WS;
  return c;
}
// R6 (A1) trigger test: h = start of the A1 letter segment = the re of
// an APUNCT PS-run of effective length 1. That "effective length 1" is
// not read off the pattern; it follows from the whole-run lemma for
// PS-runs. All look-back is <= 2 and bounded by B (the dstart channel).
// rs=h-1 may itself be a B position: punctuation at a segment start
// still consumes the following letter (probe: batch ["!a","bc"]).
__device__ __forceinline__ bool ds_a1_trig(int h, const int32_t* cp,
                                           const uint8_t* cls,
                                           const uint8_t* B) {
  if (h < 1 || B[h] || !ds_alpha(cp[h])) return false;
  if (cls[h - 1] != DS_PS || !ds_apunct(cp[h - 1])) return false;
  bool len1 = (h - 1 == 0) || B[h - 1] || cls[h - 2] != DS_PS;
  if (!len1) return false;                     // effective run len >=2 => A3
  bool smerge = (h - 1 > 0) && !B[h - 1] && cp[h - 2] == 32;  // R4 preempts
  return !smerge;
}

__device__ __forceinline__ int fold(int c) {
  // simple case folding: domain = A-Z plus 0x17F -> s (reference probe)
  if (c >= 65 && c <= 90) return c + 32;
  if (c == 0x17F) return 0x73;
  return c;
}

// dstart (nullable): document-start marks for a multi-document batch;
// runs never cross a document. n_dev (nullable): device-side source of
// the data-dependent bound (encode_fused / CUDA Graph path: geometry is
// launched by capacity and out-of-range threads exit on the device
// count; every kernel taking n_dev follows this convention).
template <int RS>
__global__ void k_classify(const int32_t* __restrict__ cp,
                           const uint8_t* __restrict__ tab,
                           const uint8_t* __restrict__ dstart,
                           uint8_t* __restrict__ cls,
                           uint8_t* __restrict__ head, int n,
                           const int32_t* __restrict__ n_dev) {
  if (n_dev) n = *n_dev;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint8_t c = tab[cp[i]];
  cls[i] = c;
  if (RS == RS_DEEPSEEK) {
    // runs segmented by merged class (LM/WSF); B/dstart force-breaks runs
    head[i] = (i == 0 || (dstart && dstart[i]))
                  ? 1 : (ds_rc(tab[cp[i - 1]]) != ds_rc(c));
  } else {
    head[i] = (i == 0 || (dstart && dstart[i])) ? 1 : (tab[cp[i - 1]] != c);
  }
}

template <int RS>
__global__ void k_runinfo(const int32_t* __restrict__ cp,
                          const uint8_t* __restrict__ cls,
                          const uint8_t* __restrict__ head,
                          const int32_t* __restrict__ rid,  // 1-based
                          int32_t* __restrict__ run_start,
                          int32_t* __restrict__ fnc,   // init 0x7f7f7f7f
                          int32_t* __restrict__ lc,    // init -1
                          int n, const int32_t* __restrict__ n_dev) {
  if (n_dev) n = *n_dev;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int r = rid[i] - 1;
  if (head[i]) run_start[r] = i;
  const bool wsf = RS == RS_DEEPSEEK
                       ? (cls[i] == DS_WS || cls[i] == DS_CRLF)
                       : (cls[i] == S_);
  if (wsf) {
    bool crlf = (cp[i] == 10) | (cp[i] == 13);
    if (crlf) atomicMax(&lc[r], i);
    else      atomicMin(&fnc[r], i);
  }
}

// dso (nullable): start index of the document each character belongs to;
// dstart (nullable): document-start marks. For a single document both are
// nullptr, d is always 0 and the "document end" is n (original semantics).
template <int RS>
__global__ void k_rules(const int32_t* __restrict__ cp,
                        const uint8_t* __restrict__ cls,
                        const uint8_t* __restrict__ head,
                        const int32_t* __restrict__ rid,
                        const int32_t* __restrict__ run_start,
                        const int32_t* __restrict__ fnc,
                        const int32_t* __restrict__ lc,
                        const int32_t* __restrict__ dso,
                        const uint8_t* __restrict__ dstart,
                        const int32_t* __restrict__ ars,  // DS: A1 run head
                        int R, int dmax, bool* __restrict__ starts, int n,
                        const int32_t* __restrict__ n_dev,
                        const int32_t* __restrict__ R_dev) {
  if (n_dev) { n = *n_dev; R = *R_dev; }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (i == 0 || (dstart && dstart[i])) { starts[i] = true; return; }
  const int d = dso ? dso[i] : 0;          // look-back base in doc/B seg
  uint8_t c = cls[i];
  bool st = false;
  if (RS == RS_DEEPSEEK) {
    // ---- DeepSeek S2 single pass (B = the dstart channel has already
    // forced a piece start above; the R1-R11 rule table is argued branch
    // by branch; dstart/dso/ars are always non-null for DS) ----
    if (c == DS_L || c == DS_M) {                // R8/R9 LM run + R6
      if (head[i]) {
        uint8_t pc = cls[i - 1];                 // !B[i] => i-1 >= d: ok
        // R8 merge-into-previous prefix classes {WS non-CRLF, O, N}
        // (N is a dead branch, kept faithfully; CRLF is not a prefix
        // class => piece start)
        bool merged = pc == DS_WS || pc == DS_O || pc == DS_N;
        st = !merged && !ds_a1_trig(i, cp, cls, dstart);  // R6 head suppress
      } else if (!ds_alpha(cp[i]) && ds_alpha(cp[i - 1])) {
        // R6 cut at the ASCII/non-ASCII boundary inside an L-run: j=ae
        // (e.g. "!ab" U+00E9 -> "!ab" | U+00E9)
        st = ds_a1_trig(ars[i - 1], cp, cls, dstart);
      }
    } else if (c == DS_PS) {
      st = head[i] && cp[i - 1] != 32;           // R4 ' ?' (32 = a3_space)
    } else if (c == DS_O) {                      // R10 unmatched span
      // The head always starts a piece; if the run's last character is
      // followed by LM (same segment) it defects to an A2 prefix, so
      // i+1 does not start a piece but this position does
      // ("\x00\x00a" -> \x00 | \x00a)
      st = head[i] || (i + 1 < n && !(dstart && dstart[i + 1]) &&
                       (cls[i + 1] == DS_L || cls[i + 1] == DS_M));
    } else if (c == DS_WS || c == DS_CRLF) {     // R3 WSF triple + R5 absorb
      int r = rid[i] - 1;
      int rs = run_start[r];
      int re = (r + 1 < R) ? run_start[r + 1] : n;
      // R5 anchor: ~B[rs] (<=> rs-d>0) and prev in PS => the whole
      // prefix CRLF streak is absorbed. (Anchor uniqueness: an A1 piece
      // always ends in a letter, so prev in PS implies the previous
      // piece went through A3.)
      bool prevP = rs - d > 0 && cls[rs - 1] == DS_PS;
      int f = fnc[r]; if (f > re) f = re;
      int rs_eff = prevP ? f : rs;               // < rs_eff: never a start
      int l = lc[r];
      bool has_nl = l >= rs_eff;
      int t0 = has_nl ? l + 1 : rs_eff;
      int m = re - t0;
      bool seg_end = re >= n || (dstart && dstart[re]);  // EOS = B seg end
      st = (has_nl && i == rs_eff) || (m > 0 && i == t0) ||
           (m >= 2 && !seg_end && i == re - 1);
    }
    // c == DS_N: all S0 group heads are in B (already a piece start
    // above), non-head positions inside a run never start (digit pieces
    // stay as unmatched spans; the S2 layer has no logic here - R1)
    starts[i] = st;
    return;
  }
  if (c == N_) {
    st = (i - run_start[rid[i] - 1]) % dmax == 0;
  } else if (c == L_) {
    if (head[i]) {
      uint8_t pc = cls[i - 1];
      bool crlf1 = cp[i - 1] == 10 || cp[i - 1] == 13;
      bool merged = (pc == S_ && !crlf1);
      if (pc == P_) {
        bool plen1 = (i - d < 2) || cls[i - 2] != P_;
        bool arr   = (i - d < 2) || cp[i - 2] != 32;
        merged |= (plen1 && arr);          // A2 merge / A1 contraction
      }
      st = !merged;
    } else {
      if (i - d >= 2 && cp[i - 2] == 39) { // 'X remainder (X in s,t,m,d)
        bool plen1 = (i - d < 3) || cls[i - 3] != P_;
        bool arr   = (i - d < 3) || cp[i - 3] != 32;
        int f1 = fold(cp[i - 1]);
        st = plen1 && arr &&
             (f1 == 0x73 || f1 == 0x74 || f1 == 0x6D || f1 == 0x64);
      }
      if (!st && i - d >= 3 && cp[i - 3] == 39 && cls[i - 2] == L_) {
        bool plen1 = (i - d < 4) || cls[i - 4] != P_;  // 'XY remainder
        bool arr   = (i - d < 4) || cp[i - 4] != 32;
        int f1 = fold(cp[i - 2]), f2 = fold(cp[i - 1]);
        st = plen1 && arr && ((f1 == 0x72 && f2 == 0x65) ||
                              (f1 == 0x76 && f2 == 0x65) ||
                              (f1 == 0x6C && f2 == 0x6C));
      }
    }
  } else if (c == P_) {
    st = head[i] && cp[i - 1] != 32;
  } else {                                  // class S
    int r = rid[i] - 1;
    int rs = run_start[r];
    int re = (r + 1 < R) ? run_start[r + 1] : n;
    bool prevP = rs - d > 0 && cls[rs - 1] == P_;
    int f = fnc[r]; if (f > re) f = re;
    int rs_eff = prevP ? f : rs;            // A4 [\r\n]* absorbs prefix
    int l = lc[r];
    bool has_nl = l >= rs_eff;
    int t0 = has_nl ? l + 1 : rs_eff;
    int m = re - t0;
    bool doc_end = re >= n || (dstart && dstart[re]);   // EOS = doc end
    st = (has_nl && i == rs_eff) || (m > 0 && i == t0) ||
         (m >= 2 && !doc_end && i == re - 1);
  }
  starts[i] = st;
}

struct CastU8 {
  __host__ __device__ int32_t operator()(uint8_t v) const { return v; }
};

// ------ GPU UTF-8 decode (serving path: bytes direct, no CPU pass) ---
struct IsLead {
  __host__ __device__ int32_t operator()(uint8_t b) const {
    return (b & 0xC0) != 0x80;
  }
};

// Kernel-level part of the input hardening: full boundary validation
// (remaining length / continuation-byte shape / F8-FF leads rejected /
// decoded values > U+10FFFF rejected). Anything illegal is sanitized to
// U+FFFD and sets the (nullable) err flag via atomicOr. After that,
// cp <= 0x10FFFF always holds, so tab[cp[i]] in k_classify (tab covers
// 0x110000) is safe for free and needs no separate guard. The earlier
// version validated nothing: a truncated lead read out of bounds (a real
// OOB for a tightly sized standalone tensor; inside the fused bucket
// buffer it read leftovers of the previous request = cross-request
// information flow), and an F8-FF lead decoded as a 4-byte lead gives
// <= 0x1FFFFF used directly as a tab index = out-of-bounds read plus
// silent garbage decoding. Two further classes were added on the same
// U+FFFD + err path, matching Python str and the UTF-8 specification:
// (1) non-shortest (overlong) forms -- need==1 with v<0x80 / need==2
// with v<0x800 / need==3 with v<0x10000 (e.g. C0 80 smuggling a NUL);
// (2) the UTF-16 surrogate range U+D800-DFFF (not Unicode scalar
// values, not representable in str).
__global__ void k_utf8_decode(const uint8_t* __restrict__ bytes,
                              const int32_t* __restrict__ cpos,  // incl. sum
                              int32_t* __restrict__ cp,
                              int32_t* __restrict__ bo,   // opt: char->byte
                              int nb,
                              int32_t* __restrict__ err) {  // opt: err flag
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  uint8_t b0 = bytes[i];
  if ((b0 & 0xC0) == 0x80) return;              // continuation byte
  const int need = b0 < 0x80 ? 0 : (b0 < 0xE0 ? 1 : (b0 < 0xF0 ? 2 : 3));
  int v = -1;
  if (b0 < 0xF8 && i + need < nb) {             // reject F8-FF; length check
    bool cont_ok = true;
    for (int k = 1; k <= need; ++k)
      cont_ok &= (bytes[i + k] & 0xC0) == 0x80;
    if (cont_ok) {
      if (need == 0)      v = b0;
      else if (need == 1) v = ((b0 & 0x1F) << 6)  | (bytes[i + 1] & 0x3F);
      else if (need == 2) v = ((b0 & 0x0F) << 12)
                              | ((bytes[i + 1] & 0x3F) << 6)
                              | (bytes[i + 2] & 0x3F);
      else                v = ((b0 & 0x07) << 18)
                              | ((bytes[i + 1] & 0x3F) << 12)
                              | ((bytes[i + 2] & 0x3F) << 6)
                              | (bytes[i + 3] & 0x3F);
      if (v > 0x10FFFF) v = -1;                 // F5-F7 leads can overflow
      // Non-shortest (overlong) forms: each lead length has a unique
      // legal lower bound; below it the encoding is a redundant longer
      // form of the same codepoint (C0 80 / E0 80 80 / F0 80 80 80)
      else if ((need == 1 && v < 0x80) || (need == 2 && v < 0x800) ||
               (need == 3 && v < 0x10000)) v = -1;
      // UTF-16 surrogate range (ED A0 80 - ED BF BF => U+D800-DFFF) is
      // not a Unicode scalar value; U+D7FF (ED 9F BF) and U+E000
      // (EE 80 80) on either side stay legal
      else if (v >= 0xD800 && v <= 0xDFFF) v = -1;
    }
  }
  if (v < 0) {
    v = 0xFFFD;
    if (err) atomicOr(err, 1);
  }
  cp[cpos[i] - 1] = v;                          // inclusive sum -> 0-based
  if (bo) bo[cpos[i] - 1] = i;
}

#ifndef TOKTIER_DEVICE_ONLY
static std::vector<torch::Tensor> utf8_decode_impl(torch::Tensor bytes,
                                                   bool want_bo) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(bytes));
  TORCH_CHECK(bytes.is_cuda() && bytes.dtype() == torch::kUInt8 &&
              bytes.is_contiguous());
  const int64_t nb64 = bytes.numel();
  TORCH_CHECK(nb64 < INT32_MAX);
  const int nb = (int)nb64;
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32)
                      .device(bytes.device());
  if (nb == 0)
    return {torch::empty({0}, opts_i32), torch::empty({0}, opts_i32)};
  auto stream = at::cuda::getCurrentCUDAStream();
  auto cpos = torch::empty({nb64}, opts_i32);
  auto it = thrust::make_transform_iterator(
      (const uint8_t*)bytes.data_ptr<uint8_t>(), IsLead{});
  size_t tmp_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tmp_bytes, it,
                                cpos.data_ptr<int32_t>(), nb, stream);
  auto tmp = torch::empty({(int64_t)tmp_bytes},
                          torch::TensorOptions().dtype(torch::kUInt8)
                              .device(bytes.device()));
  cub::DeviceScan::InclusiveSum(tmp.data_ptr<uint8_t>(), tmp_bytes, it,
                                cpos.data_ptr<int32_t>(), nb, stream);
  int C = cpos[nb64 - 1].item<int32_t>();
  auto cp = torch::empty({(int64_t)C}, opts_i32);
  auto bo = torch::empty({want_bo ? (int64_t)C : 0}, opts_i32);
  auto err = torch::zeros({1}, opts_i32);
  const int nblk = (nb + TPB - 1) / TPB;
  k_utf8_decode<<<nblk, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), cpos.data_ptr<int32_t>(),
      cp.data_ptr<int32_t>(),
      want_bo ? bo.data_ptr<int32_t>() : nullptr, nb,
      err.data_ptr<int32_t>());
  // Eager layer: this path already has a D2H sync, so reading the flag
  // rides along at nearly zero cost; illegal input is rejected
  // deterministically (the graph path does no host read, see
  // encode_fused).
  if (err.item<int32_t>() != 0)
    throw pybind11::value_error(
        "invalid UTF-8 in input bytes (truncated sequence / illegal lead "
        "F8-FF / bad continuation / overlong encoding / surrogate "
        "U+D800-DFFF / codepoint > U+10FFFF)");
  return {cp, bo};
}

torch::Tensor utf8_to_cp(torch::Tensor bytes) {
  return utf8_decode_impl(bytes, false)[0];
}

std::vector<torch::Tensor> utf8_to_cp_bo(torch::Tensor bytes) {
  return utf8_decode_impl(bytes, true);
}

template <int RS>
static torch::Tensor pretok_impl_t(torch::Tensor cp, torch::Tensor tab,
                                   int64_t dmax,
                                   const uint8_t* dstart_ptr,
                                   const int32_t* dso_ptr,
                                   const int32_t* ars_ptr) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(cp.is_cuda() && cp.dtype() == torch::kInt32 &&
              cp.is_contiguous());
  TORCH_CHECK(tab.is_cuda() && tab.dtype() == torch::kUInt8);
  const int64_t n64 = cp.numel();
  TORCH_CHECK(n64 < INT32_MAX, "single-buffer limit 2^31 chars");
  const int n = (int)n64;
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8)
                     .device(cp.device());
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32)
                      .device(cp.device());
  auto starts = torch::empty({n64}, torch::TensorOptions()
                                        .dtype(torch::kBool)
                                        .device(cp.device()));
  if (n == 0) return starts;
  auto cls = torch::empty({n64}, opts_u8);
  auto head = torch::empty({n64}, opts_u8);
  auto rid = torch::empty({n64}, opts_i32);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nb = (n + TPB - 1) / TPB;

  k_classify<RS><<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(), dstart_ptr,
      cls.data_ptr<uint8_t>(), head.data_ptr<uint8_t>(), n, nullptr);

  auto it = thrust::make_transform_iterator(
      (const uint8_t*)head.data_ptr<uint8_t>(), CastU8{});
  size_t tmp_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tmp_bytes, it,
                                rid.data_ptr<int32_t>(), n, stream);
  auto tmp = torch::empty({(int64_t)tmp_bytes}, opts_u8);
  cub::DeviceScan::InclusiveSum(tmp.data_ptr<uint8_t>(), tmp_bytes, it,
                                rid.data_ptr<int32_t>(), n, stream);

  int R = rid[n64 - 1].item<int32_t>();   // single 4B D2H sync
  auto run_start = torch::empty({R}, opts_i32);
  auto fnc = torch::empty({R}, opts_i32);
  auto lc = torch::empty({R}, opts_i32);
  cudaMemsetAsync(fnc.data_ptr<int32_t>(), 0x7f, (size_t)R * 4, stream);
  cudaMemsetAsync(lc.data_ptr<int32_t>(), 0xff, (size_t)R * 4, stream);

  k_runinfo<RS><<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      head.data_ptr<uint8_t>(), rid.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), fnc.data_ptr<int32_t>(),
      lc.data_ptr<int32_t>(), n, nullptr);

  k_rules<RS><<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      head.data_ptr<uint8_t>(), rid.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), fnc.data_ptr<int32_t>(),
      lc.data_ptr<int32_t>(), dso_ptr, dstart_ptr, ars_ptr,
      R, (int)dmax, starts.data_ptr<bool>(), n, nullptr, nullptr);

  return starts;
}

static torch::Tensor pretok_impl(torch::Tensor cp, torch::Tensor tab,
                                 int64_t dmax,
                                 const uint8_t* dstart_ptr,
                                 const int32_t* dso_ptr) {
  return pretok_impl_t<RS_CL100K>(cp, tab, dmax, dstart_ptr, dso_ptr,
                                  nullptr);
}

torch::Tensor pretok_starts(torch::Tensor cp, torch::Tensor tab,
                            int64_t dmax) {
  return pretok_impl(cp, tab, dmax, nullptr, nullptr);
}

#endif

__global__ void k_dso_seed(const uint8_t* __restrict__ dstart,
                           int32_t* __restrict__ seed, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  seed[i] = dstart[i] ? i : 0;
}

// Multi-document batch: dstart[i]=1 marks a document start (dstart[0]
// must be 1). dso (the document start of each character) is computed
// with a CUB max-scan here, because the CUDA kernel behind torch.cummax
// is pathologically slow (~55ms for 41M characters) and was the source
// of an apparent batch-throughput bottleneck. Runs are force-broken at
// document boundaries and look-back is bounded by the document start.
#ifndef TOKTIER_DEVICE_ONLY
torch::Tensor pretok_starts_batched(torch::Tensor cp, torch::Tensor dstart,
                                    torch::Tensor tab, int64_t dmax) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(dstart.is_cuda() && dstart.dtype() == torch::kUInt8 &&
              dstart.is_contiguous() && dstart.numel() == cp.numel());
  const int64_t n64 = cp.numel();
  TORCH_CHECK(n64 < INT32_MAX);
  const int n = (int)n64;
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32)
                      .device(cp.device());
  auto seed = torch::empty({n64}, opts_i32);
  auto dso = torch::empty({n64}, opts_i32);
  if (n > 0) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const int nb = (n + TPB - 1) / TPB;
    k_dso_seed<<<nb, TPB, 0, stream>>>(dstart.data_ptr<uint8_t>(),
                                       seed.data_ptr<int32_t>(), n);
    size_t tmp_bytes = 0;
    cub::DeviceScan::InclusiveScan(nullptr, tmp_bytes,
                                   seed.data_ptr<int32_t>(),
                                   dso.data_ptr<int32_t>(),
                                   cuda::maximum<int32_t>{}, n, stream);
    auto tmp = torch::empty({(int64_t)tmp_bytes},
                            torch::TensorOptions().dtype(torch::kUInt8)
                                .device(cp.device()));
    cub::DeviceScan::InclusiveScan(tmp.data_ptr<uint8_t>(), tmp_bytes,
                                   seed.data_ptr<int32_t>(),
                                   dso.data_ptr<int32_t>(),
                                   cuda::maximum<int32_t>{}, n, stream);
  }
  return pretok_impl(cp, tab, dmax, dstart.data_ptr<uint8_t>(),
                     dso.data_ptr<int32_t>());
}

// ==================== DeepSeek B mask prepass ====================
// B = document starts, union the S0 cut points (N-run boundaries plus
// one cut every dmax characters from the run start), union the S1 cut
// points (both ends of a run in one of the three CJK ranges). Digit
// grouping only depends on the document cut (S0 and S1 are disjoint at
// table level: the audit reports s0_s1_disjoint=0, i.e. CJK cut points
// never fall inside an N-run). nrs/ars propagate the run head with a
// CUB max-scan (same trick as k_dso_seed: non-member positions seed
// their own index and member non-head positions seed 0, so after the
// scan a member position holds the index of its run head).

#endif

__global__ void k_ds_seed_n(const int32_t* __restrict__ cp,
                            const uint8_t* __restrict__ tab,
                            const uint8_t* __restrict__ doc,
                            int32_t* __restrict__ seed, int cap,
                            const int32_t* __restrict__ n_dev) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  const int n = n_dev ? *n_dev : cap;
  if (i >= n || tab[cp[i]] != DS_N) { seed[i] = i; return; }
  bool hd = i == 0 || (doc && doc[i]) || tab[cp[i - 1]] != DS_N;
  seed[i] = hd ? i : 0;
}

__global__ void k_ds_bmask(const int32_t* __restrict__ cp,
                           const uint8_t* __restrict__ tab,
                           const uint8_t* __restrict__ doc,
                           const int32_t* __restrict__ nrs, int dmax,
                           uint8_t* __restrict__ B,
                           int32_t* __restrict__ aseed, int cap,
                           const int32_t* __restrict__ n_dev) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  const int n = n_dev ? *n_dev : cap;
  if (i >= n) { B[i] = 0; aseed[i] = i; return; }   // deterministic clear
  const int c = cp[i], p = i > 0 ? cp[i - 1] : -1;
  const bool nm = tab[c] == DS_N, pn = i > 0 && tab[p] == DS_N;
  const bool ck = ds_cjk(c), pk = i > 0 && ds_cjk(p);
  bool b = i == 0 || (doc && doc[i]);
  b |= nm && (i - nrs[i]) % dmax == 0;              // S0 group head
  b |= !nm && pn;                                   // S0 run end
  b |= ck && !pk;                                   // S1 left (both ends)
  b |= !ck && pk;                                   // S1 right end
  B[i] = b;
  // A1 letter-run head seed (B-cut; depends only on B at this position)
  const bool al = ds_alpha(c);
  aseed[i] = !al ? i : ((i == 0 || b || !ds_alpha(p)) ? i : 0);
}

// (B, dso, ars): dso = B segment start (dso semantics of dstart reused)
#ifndef TOKTIER_DEVICE_ONLY
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ds_prepass(
    torch::Tensor cp, torch::Tensor tab, int64_t dmax,
    const uint8_t* doc_ptr, int cap, const int32_t* n_dev) {
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32)
                      .device(cp.device());
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8)
                     .device(cp.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto seed = torch::empty({(int64_t)cap}, opts_i32);
  auto nrs = torch::empty({(int64_t)cap}, opts_i32);
  auto B = torch::empty({(int64_t)cap}, opts_u8);
  auto ars = torch::empty({(int64_t)cap}, opts_i32);
  auto dso = torch::empty({(int64_t)cap}, opts_i32);
  const int nb = (cap + TPB - 1) / TPB;
  size_t tb = 0;
  cub::DeviceScan::InclusiveScan(nullptr, tb, seed.data_ptr<int32_t>(),
                                 nrs.data_ptr<int32_t>(),
                                 cuda::maximum<int32_t>{}, cap, stream);
  auto tmp = torch::empty({(int64_t)tb}, opts_u8);
  k_ds_seed_n<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(), doc_ptr,
      seed.data_ptr<int32_t>(), cap, n_dev);
  cub::DeviceScan::InclusiveScan(tmp.data_ptr<uint8_t>(), tb,
                                 seed.data_ptr<int32_t>(),
                                 nrs.data_ptr<int32_t>(),
                                 cuda::maximum<int32_t>{}, cap, stream);
  k_ds_bmask<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(), doc_ptr,
      nrs.data_ptr<int32_t>(), (int)dmax, B.data_ptr<uint8_t>(),
      seed.data_ptr<int32_t>(), cap, n_dev);       // seed reused as aseed
  cub::DeviceScan::InclusiveScan(tmp.data_ptr<uint8_t>(), tb,
                                 seed.data_ptr<int32_t>(),
                                 ars.data_ptr<int32_t>(),
                                 cuda::maximum<int32_t>{}, cap, stream);
  k_dso_seed<<<nb, TPB, 0, stream>>>(B.data_ptr<uint8_t>(),
                                     seed.data_ptr<int32_t>(), cap);
  cub::DeviceScan::InclusiveScan(tmp.data_ptr<uint8_t>(), tb,
                                 seed.data_ptr<int32_t>(),
                                 dso.data_ptr<int32_t>(),
                                 cuda::maximum<int32_t>{}, cap, stream);
  return {B, dso, ars};
}

// DeepSeek single-doc entry (i==0 = implicit doc head, B[0] always 1)
torch::Tensor pretok_starts_ds(torch::Tensor cp, torch::Tensor tab,
                               int64_t dmax) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(cp.is_cuda() && cp.dtype() == torch::kInt32 &&
              cp.is_contiguous());
  const int64_t n64 = cp.numel();
  TORCH_CHECK(n64 < INT32_MAX);
  if (n64 == 0)
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool)
                                 .device(cp.device()));
  auto pre = ds_prepass(cp, tab, dmax, nullptr, (int)n64, nullptr);
  return pretok_impl_t<RS_DEEPSEEK>(
      cp, tab, dmax, std::get<0>(pre).data_ptr<uint8_t>(),
      std::get<1>(pre).data_ptr<int32_t>(),
      std::get<2>(pre).data_ptr<int32_t>());
}

// DeepSeek batched: doc boundaries and stage cuts share the B channel
torch::Tensor pretok_starts_batched_ds(torch::Tensor cp,
                                       torch::Tensor dstart,
                                       torch::Tensor tab, int64_t dmax) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(cp.is_cuda() && cp.dtype() == torch::kInt32 &&
              cp.is_contiguous());
  TORCH_CHECK(dstart.is_cuda() && dstart.dtype() == torch::kUInt8 &&
              dstart.is_contiguous() && dstart.numel() == cp.numel());
  const int64_t n64 = cp.numel();
  TORCH_CHECK(n64 < INT32_MAX);
  if (n64 == 0)
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool)
                                 .device(cp.device()));
  auto pre = ds_prepass(cp, tab, dmax, dstart.data_ptr<uint8_t>(),
                        (int)n64, nullptr);
  return pretok_impl_t<RS_DEEPSEEK>(
      cp, tab, dmax, std::get<0>(pre).data_ptr<uint8_t>(),
      std::get<1>(pre).data_ptr<int32_t>(),
      std::get<2>(pre).data_ptr<int32_t>());
}

// ==================== Laguna stage-0 B mask ====================
// Predicate (2-character window, purely local, no backtracking, no
// look-ahead): B[i]=1 <=> i is the start of a maximal newline run (a
// match of `(?:\r?\n)+`), unioned with the document starts.
//   unit(i)    = cp[i]=='\n' && !(prev same seg && cp[i-1]=='\r')  unit \n
//              || cp[i]=='\r' && next same seg && cp[i+1]=='\n'  unit \r\n
//   chained(i) = prev same seg && cp[i-1]=='\n' (a unit ends exactly
//                at i-1, so the greedy + has already folded the unit
//                at i into the previous match)
//   B[i] = i==0 || doc[i] || (unit(i) && !chained(i))
// Argument: every '\n' belongs to some unit ('\n' can form a unit by
// itself), so cp[i-1]=='\n' <=> a unit ends at i-1. A '\n' preceded in
// the same segment by '\r' is covered by the \r\n unit headed by that
// '\r' and is therefore never a start; an isolated '\r' (not followed
// by '\n') takes part in no match.
// Consumer side = the existing RS_CL100K dstart/dso machinery: B forces
// a piece start, resets run segmentation, provides the look-back base d
// and the segment-end test (doc_end) for `\s+(?!\S)`. That corresponds
// bit for bit to "stage-1 applied independently to each stage-0
// segment": assertions of a two-level composition never cross levels,
// which is exactly the HuggingFace Sequence semantics and the reason the
// measured Laguna behavior (37,327/60,058) differs from the
// single-stage qwen3 case.
#endif

__global__ void k_lag_bmask(const int32_t* __restrict__ cp,
                            const uint8_t* __restrict__ doc,
                            uint8_t* __restrict__ B, int cap,
                            const int32_t* __restrict__ n_dev) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap) return;
  const int n = n_dev ? *n_dev : cap;
  if (i >= n) { B[i] = 0; return; }             // deterministic clear
  if (i == 0 || (doc && doc[i])) { B[i] = 1; return; }
  const int c = cp[i], p = cp[i - 1];           // !doc[i] => i-1 same seg
  bool b = false;
  if (c == 10) b = (p != 13) && (p != 10);
  else if (c == 13)
    b = (i + 1 < n) && !(doc && doc[i + 1]) && cp[i + 1] == 10 && p != 10;
  B[i] = b;
}

// (B, dso): dso = start of the B segment (reusing the dso semantics of
// the dstart channel; a proper subset of ds_prepass, with no nrs/ars
// channel). The geometry only depends on cap and all data-dependent
// quantities go through n_dev, so this is CUDA-Graph capturable (the
// encode_fused_laguna path).
#ifndef TOKTIER_DEVICE_ONLY
static std::tuple<torch::Tensor, torch::Tensor> lag_prepass(
    torch::Tensor cp, const uint8_t* doc_ptr, int cap,
    const int32_t* n_dev) {
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32)
                      .device(cp.device());
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8)
                     .device(cp.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto B = torch::empty({(int64_t)cap}, opts_u8);
  auto seed = torch::empty({(int64_t)cap}, opts_i32);
  auto dso = torch::empty({(int64_t)cap}, opts_i32);
  const int nb = (cap + TPB - 1) / TPB;
  k_lag_bmask<<<nb, TPB, 0, stream>>>(cp.data_ptr<int32_t>(), doc_ptr,
                                      B.data_ptr<uint8_t>(), cap, n_dev);
  k_dso_seed<<<nb, TPB, 0, stream>>>(B.data_ptr<uint8_t>(),
                                     seed.data_ptr<int32_t>(), cap);
  size_t tb = 0;
  cub::DeviceScan::InclusiveScan(nullptr, tb, seed.data_ptr<int32_t>(),
                                 dso.data_ptr<int32_t>(),
                                 cuda::maximum<int32_t>{}, cap, stream);
  auto tmp = torch::empty({(int64_t)tb}, opts_u8);
  cub::DeviceScan::InclusiveScan(tmp.data_ptr<uint8_t>(), tb,
                                 seed.data_ptr<int32_t>(),
                                 dso.data_ptr<int32_t>(),
                                 cuda::maximum<int32_t>{}, cap, stream);
  return {B, dso};
}

// Laguna single-doc entry (i==0 = implicit segment head, B[0] always 1)
torch::Tensor pretok_starts_laguna(torch::Tensor cp, torch::Tensor tab,
                                   int64_t dmax) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(cp.is_cuda() && cp.dtype() == torch::kInt32 &&
              cp.is_contiguous());
  const int64_t n64 = cp.numel();
  TORCH_CHECK(n64 < INT32_MAX);
  if (n64 == 0)
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool)
                                 .device(cp.device()));
  auto pre = lag_prepass(cp, nullptr, (int)n64, nullptr);
  return pretok_impl_t<RS_LAGUNA>(
      cp, tab, dmax, std::get<0>(pre).data_ptr<uint8_t>(),
      std::get<1>(pre).data_ptr<int32_t>(), nullptr);
}

// Laguna batched: doc boundaries and stage-0 cuts share the B channel
torch::Tensor pretok_starts_batched_laguna(torch::Tensor cp,
                                           torch::Tensor dstart,
                                           torch::Tensor tab, int64_t dmax) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(cp.is_cuda() && cp.dtype() == torch::kInt32 &&
              cp.is_contiguous());
  TORCH_CHECK(dstart.is_cuda() && dstart.dtype() == torch::kUInt8 &&
              dstart.is_contiguous() && dstart.numel() == cp.numel());
  const int64_t n64 = cp.numel();
  TORCH_CHECK(n64 < INT32_MAX);
  if (n64 == 0)
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kBool)
                                 .device(cp.device()));
  auto pre = lag_prepass(cp, dstart.data_ptr<uint8_t>(), (int)n64, nullptr);
  return pretok_impl_t<RS_LAGUNA>(
      cp, tab, dmax, std::get<0>(pre).data_ptr<uint8_t>(),
      std::get<1>(pre).data_ptr<int32_t>(), nullptr);
}

// Self-description of the inline constants (JSON): the Python side
// asserts it field by field against the mechanically extracted metadata,
// which guards against transcription errors. A build carrying planted
// mutations (-DTOKTIER_DS_*) also shows up here.
static std::string ds_constants() {
  std::string ap = "[", al = "[";
  for (int c = 0; c < 128; ++c) {
    if (ds_apunct(c)) {
      if (ap.size() > 1) ap += ",";
      ap += std::to_string(c);
    }
    if (ds_alpha(c)) {
      if (al.size() > 1) al += ",";
      al += std::to_string(c);
    }
  }
  ap += "]"; al += "]";
  char buf[256];
  snprintf(buf, sizeof buf,
           "{\"cjk_ranges\":[[%d,%d],[%d,%d],[%d,%d]],"
           "\"a3_space\":32,\"crlf_cps\":[10,13],"
           "\"class_enum\":{\"O\":%d,\"L\":%d,\"M\":%d,\"N\":%d,"
           "\"PS\":%d,\"WS\":%d,\"CRLF\":%d},",
           (int)TOKTIER_DS_CJK0_LO, (int)TOKTIER_DS_CJK0_HI,
           (int)TOKTIER_DS_CJK1_LO, (int)TOKTIER_DS_CJK1_HI,
           (int)TOKTIER_DS_CJK2_LO, (int)TOKTIER_DS_CJK2_HI,
           (int)DS_O, (int)DS_L, (int)DS_M, (int)DS_N, (int)DS_PS,
           (int)DS_WS, (int)DS_CRLF);
  return std::string(buf) + "\"apunct\":" + ap + ",\"alpha\":" + al + "}";
}

#endif

// ============ In-piece BPE merge (byte-level BPE families) ============
//
// Semantics = the HuggingFace tokenizers BPE: the initial symbols are
// the ByteLevel single-byte ids; repeatedly merge the adjacent pair with
// the smallest rank (equal rank <=> equal pair, leftmost first).
// Equivalent rewrite: in each round, batch-merge all leftmost
// non-overlapping occurrences of "the pair with the current smallest
// rank" at once (bit-identical to merging one at a time; for a uniform
// long run it reduces the number of rounds from O(m) to O(log m)).
// ignore_merges=true (the llama3 family): look the whole piece up in the
// vocabulary first, a hit becomes a single token.
// Two-level dispatch: m <= SHORT_MAX uses thread-per-piece, otherwise
// block-per-piece (global ping-pong buffers, the piece stays resident
// in L2).

#ifndef TOKTIER_SHORT_MAX
#define TOKTIER_SHORT_MAX 32
#endif
constexpr int SHORT_MAX = TOKTIER_SHORT_MAX;   // thread/warp cutoff, -D
constexpr int MED_MAX = 128;
// The dispatch predicates len<=SHORT_MAX / SHORT_MAX<len<=MED_MAX /
// len>MED_MAX require 0<SHORT_MAX<=MED_MAX; otherwise the warp bucket
// predicate is the empty set and the register array in
// k_bpe_thread<SHORT_MAX> has zero or negative length.
static_assert(0 < SHORT_MAX && SHORT_MAX <= MED_MAX,
              "TOKTIER_SHORT_MAX must be in (0, MED_MAX]");

__device__ __forceinline__ uint64_t splitmix64_d(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Merge-pair lookup: returns rank<<32|id_m, or -1 when not found
__device__ __forceinline__ long long pair_lookup(
    const uint64_t* __restrict__ K, const uint64_t* __restrict__ V,
    unsigned mask, int idl, int idr) {
  uint64_t key = ((uint64_t)(uint32_t)idl << 32) | (uint32_t)idr;
  unsigned i = (unsigned)splitmix64_d(key) & mask;
  while (true) {
    uint64_t k = K[i];
    if (k == 0xFFFFFFFFFFFFFFFFULL) return -1;
    if (k == key) return (long long)V[i];
    i = (i + 1) & mask;
  }
}

// Whole-piece vocab hit (ignore_merges): FNV1a64 + byte-exact recheck
__device__ __forceinline__ int vocab_lookup(
    const uint8_t* __restrict__ bytes, int s, int m,
    const uint64_t* __restrict__ VK, const uint64_t* __restrict__ VV,
    unsigned vmask, const uint8_t* __restrict__ blob) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (int j = 0; j < m; ++j) h = (h ^ bytes[s + j]) * 0x100000001B3ULL;
  unsigned i = (unsigned)splitmix64_d(h) & vmask;
  while (true) {
    uint64_t k = VK[i];
    if (k == 0xFFFFFFFFFFFFFFFFULL) return -1;
    if (k == h) {
      uint64_t v = VV[i];
      int len = (int)((v >> 20) & 0x3FF);
      long long off = (long long)(v >> 30);
      if (len == m) {
        bool eq = true;
        for (int j = 0; j < m && eq; ++j) eq = blob[off + j] == bytes[s + j];
        if (eq) return (int)(v & 0xFFFFF);
      }
    }
    i = (i + 1) & vmask;
  }
}

// thread-per-piece: CAP=32 stays in registers (the bulk of short
// pieces), CAP=128 uses local memory (medium pieces: the typical CJK
// sentence piece is 100-400B, and whole-piece parallelism is the key to
// CJK throughput)
constexpr int MEMO_LEN = 16, MEMO_IDS = 8, MEMO_PROBE = 4;

__device__ __forceinline__ uint64_t memo_hash(const uint8_t* b, int len) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < len; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
  return h | 1ULL;                               // 0 = empty-slot sentinel
}

template <int CAP>
__global__ void k_bpe_thread(const uint8_t* __restrict__ bytes,
                             const int32_t* __restrict__ pb,  // [P+1] bounds
                             const int32_t* __restrict__ plist, int n_piece,
                             const uint64_t* __restrict__ PK,
                             const uint64_t* __restrict__ PV, unsigned pmask,
                             const int32_t* __restrict__ byte_id,
                             const uint64_t* __restrict__ VK,
                             const uint64_t* __restrict__ VV, unsigned vmask,
                             const uint8_t* __restrict__ blob, int ign,
                             int32_t* __restrict__ scratch,
                             int32_t* __restrict__ cnt,
                             const int32_t* __restrict__ np_dev,
                             const uint64_t* __restrict__ mkeys,
                             const int32_t* __restrict__ mmeta,
                             const uint8_t* __restrict__ mbytes,
                             const int32_t* __restrict__ mvals,
                             unsigned mmask) {
  if (np_dev) n_piece = *np_dev;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_piece) return;
  int p = plist[t];
  int s = pb[p], e = pb[p + 1], m = e - s;
  // Inline memoization fast path: only entries inserted by a previous
  // call are read (stream ordering isolates them; entries inserted by
  // this call are not visible in this kernel, so there is no torn read;
  // a miss falls back to the merge loop)
  if (mkeys && m <= MEMO_LEN) {
    uint64_t h = memo_hash(bytes + s, m);
    for (int k = 0; k < MEMO_PROBE; ++k) {
      unsigned slot = (unsigned)(h + k) & mmask;
      uint64_t key = mkeys[slot];
      if (key == 0) break;
      if (key != h) continue;
      int meta = mmeta[slot];
      if ((meta >> 8) != m) continue;
      bool eq = true;
      for (int i = 0; i < m; ++i)
        if (mbytes[(size_t)slot * MEMO_LEN + i] != bytes[s + i]) {
          eq = false;
          break;
        }
      if (!eq) continue;
      int n = meta & 0xff;
      for (int i = 0; i < n; ++i)
        scratch[s + i] = mvals[(size_t)slot * MEMO_IDS + i];
      cnt[p] = n;
      return;
    }
  }
  if (ign) {
    int hit = vocab_lookup(bytes, s, m, VK, VV, vmask, blob);
    if (hit >= 0) { scratch[s] = hit; cnt[p] = 1; return; }
  }
  int ids[CAP];
  // byte_id<0 (a byte with no alphabet slot in an unk-free BPE, e.g. CR
  // in some families) reproduces the reference semantics "silently
  // dropped before merging": after compaction the now-adjacent symbols
  // merge as usual (measured on the reference engine: "\r\r\n\n" yields
  // the single token for "\n\n"). Families with full byte coverage
  // always have v>=0, so the path is bit-identical there.
  int m2 = 0;
  for (int j = 0; j < m; ++j) {
    int v = byte_id[bytes[s + j]];
    if (v >= 0) ids[m2++] = v;
  }
  m = m2;
  while (m > 1) {
    long long best = -1; int bj = -1;
    for (int j = 0; j < m - 1; ++j) {
      long long v = pair_lookup(PK, PV, pmask, ids[j], ids[j + 1]);
      if (v >= 0 && (best < 0 || (v >> 32) < (best >> 32))) {
        best = v; bj = j;                       // strict < => leftmost wins
      }
    }
    if (bj < 0) break;
    ids[bj] = (int)(best & 0xFFFFFFFF);
    for (int j = bj + 1; j < m - 1; ++j) ids[j] = ids[j + 1];
    --m;
  }
  for (int j = 0; j < m; ++j) scratch[s + j] = ids[j];
  cnt[p] = m;
}

// ---- Warp-cooperative BPE, one path for 2-128 bytes (it replaces the
// two-level SHORT/MED thread dispatch). The piece lives in a per-warp
// shared-memory slice. Each round: 32 lanes look up pairs in parallel
// (at most 4 per lane) -> shuffle-reduce the smallest rank (equal rank
// <=> equal pair, so positions need not be compared) -> each lane marks
// its hits using the pair value it already looked up this round (no
// second probe) -> lane 0 serially batch-merges the leftmost
// non-overlapping occurrences (in-place forward compaction, where
// wpos <= j always holds). Only __syncwarp() is used, never a block
// barrier.
constexpr int WPB = TPB / 32;                    // warps per block

// 128-bit bitvector (len <= MED_MAX=128 => pair positions fit 2x u64)
struct B128 { uint64_t lo, hi; };
__device__ __forceinline__ B128 shl128(B128 x, int k) {
  if (k == 0) return x;
  if (k >= 64) return {0ULL, x.lo << (k - 64)};
  return {x.lo << k, (x.hi << k) | (x.lo >> (64 - k))};
}
__device__ __forceinline__ B128 shr128(B128 x, int k) {
  if (k == 0) return x;
  if (k >= 64) return {x.hi >> (k - 64), 0ULL};
  return {(x.lo >> k) | (x.hi << (64 - k)), x.hi >> k};
}
__device__ __forceinline__ B128 and128(B128 a, B128 b) {
  return {a.lo & b.lo, a.hi & b.hi};
}
__device__ __forceinline__ B128 or128(B128 a, B128 b) {
  return {a.lo | b.lo, a.hi | b.hi};
}
__device__ __forceinline__ bool bit128(B128 x, int j) {
  return j < 64 ? (x.lo >> j) & 1ULL : (x.hi >> (j - 64)) & 1ULL;
}
// Popcount of bits below j (excluding j) = target index of compaction
__device__ __forceinline__ int rank128(B128 x, int j) {
  if (j < 64) return __popcll(x.lo & ((j == 0) ? 0ULL : (~0ULL >> (64 - j))));
  int r = __popcll(x.lo);
  int jh = j - 64;
  return r + __popcll(x.hi & ((jh == 0) ? 0ULL : (~0ULL >> (64 - jh))));
}

// Incremental pair-value cache: a pair value is a pure function of
// (id_l, id_r), so only pairs immediately adjacent to this round's
// merges can change. The first round probes everything; every later
// round re-probes only O(2 x merges), taking the total probe count from
// O(m x rounds) to O(m + 2 x merges) (the L1->L2 round trip is the
// measured top bottleneck). In-place compaction: values go to registers
// first and are written back to the same shared array after
// __syncwarp.
__device__ __forceinline__ void bpe_warp_one(
    int p, int lane, int32_t* __restrict__ ids, long long* __restrict__ pvv,
    uint8_t* __restrict__ msel,
    const uint8_t* __restrict__ bytes, const int32_t* __restrict__ pb,
    const uint64_t* __restrict__ PK, const uint64_t* __restrict__ PV,
    unsigned pmask, const int32_t* __restrict__ byte_id,
    const uint64_t* __restrict__ VK, const uint64_t* __restrict__ VV,
    unsigned vmask, const uint8_t* __restrict__ blob, int ign,
    int32_t* __restrict__ scratch, int32_t* __restrict__ cnt,
    const uint32_t* __restrict__ ub) {
  const int s = pb[p];
  int m = pb[p + 1] - s;
  if (ign) {
    int hit = -1;
    if (lane == 0) hit = vocab_lookup(bytes, s, m, VK, VV, vmask, blob);
    hit = __shfl_sync(0xFFFFFFFFu, hit, 0);
    if (hit >= 0) {
      if (lane == 0) { scratch[s] = hit; cnt[p] = 1; }
      return;
    }
  }
  bool my_drop = false;
  for (int j = lane; j < m; j += 32) {
    const uint8_t b = bytes[s + j];
    ids[j] = byte_id[b];
    my_drop |= byte_id[b] < 0;
  }
  if (__any_sync(0xFFFFFFFFu, my_drop)) {
    // Rare path (unk-free family missing an alphabet slot for a byte,
    // e.g. CR): lane 0 compacts serially, reproducing the reference
    // "dropped before merging"; full-coverage families never enter here
    __syncwarp();
    int m2 = 0;
    if (lane == 0) {
      for (int j = 0; j < m; ++j) {
        const int v = byte_id[bytes[s + j]];
        if (v >= 0) ids[m2++] = v;
      }
    }
    __syncwarp();
    m = __shfl_sync(0xFFFFFFFFu, m2, 0);
    if (m == 0) { if (lane == 0) cnt[p] = 0; return; }
  }
  __syncwarp();
  int len = m;
  bool first = true;
  while (len > 1) {
    long long my = -1, vv[4];
    int val[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int j = lane + 32 * k;
      vv[k] = -1;
      val[k] = (j < len) ? ids[j] : 0;
      if (j < len - 1) {
        vv[k] = first ? pair_lookup(PK, PV, pmask, ids[j], ids[j + 1])
                      : pvv[j];
        if (vv[k] >= 0 && (my < 0 || (vv[k] >> 32) < (my >> 32))) my = vv[k];
      }
    }
    B128 hit;
    int mid = 0;
    {
      // Single-instruction warp min reduction: rank (< 2^27; real
      // vocabularies are < 2^20) << 5 | lane packed into a u32
      const unsigned enc = my >= 0
          ? (((unsigned)(my >> 32)) << 5) | (unsigned)lane : 0xFFFFFFFFu;
      const unsigned win = __reduce_min_sync(0xFFFFFFFFu, enc);
      if (win == 0xFFFFFFFFu) break;
      const long long best = __shfl_sync(0xFFFFFFFFu, my, (int)(win & 31));
      const int rank = (int)(best >> 32);
      mid = (int)(best & 0xFFFFFFFF);
      // Hit bitmap: ballot straight into registers (same value in every
      // lane, no shared-memory round trip)
      uint32_t hw[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int j = lane + 32 * k;
        hw[k] = __ballot_sync(0xFFFFFFFFu,
                              j < len - 1 && vv[k] >= 0 &&
                              (int)(vv[k] >> 32) == rank);
      }
      hit = {(uint64_t)hw[0] | ((uint64_t)hw[1] << 32),
             (uint64_t)hw[2] | ((uint64_t)hw[3] << 32)};
      // Non-monotone merge-table guard: the batch-merge equivalence
      // theorem requires that every rule using z as a component have a
      // rank greater than the rank of the rule producing z
      // (monotonicity). For a flagged (non-monotone) rank, merging one
      // occurrence can already create a new pair of smaller rank, so
      // such a round merges only the leftmost occurrence (= the
      // one-at-a-time semantics, exact unconditionally). ub==nullptr
      // (families with a monotone table) never enters here, so the
      // existing path is bit-identical.
      if (ub && ((ub[(unsigned)rank >> 5] >> ((unsigned)rank & 31u)) & 1u)) {
        if (hit.lo) hit = {hit.lo & (0ULL - hit.lo), 0ULL};
        else        hit = {0ULL, hit.hi & (0ULL - hit.hi)};
      }
    }
    // Leftmost non-overlapping selection sel[j] = hit[j] & ~sel[j-1]:
    // starting at a run head, take every other position. Closed form by
    // doubling (computed redundantly in every lane): J_2=hit,
    // J_{2k}=J_k & (J_k<<k), and sel advances from the run head by the
    // reachable jump distances.
    B128 sel = and128(hit, {~(hit.lo << 1), ~((hit.hi << 1) | (hit.lo >> 63))});
    // No adjacent hits (the vast majority of rounds) => every hit is a
    // run head, so the doubling loop is skipped
    if (((hit.lo & (hit.lo << 1)) |
         (hit.hi & ((hit.hi << 1) | (hit.lo >> 63)))) != 0ULL) {
      B128 J = hit;
#pragma unroll
      for (int k = 2; k < 128; k <<= 1) {
        sel = or128(sel, and128(shl128(sel, k), J));
        J = and128(J, shl128(J, k));
      }
    }
    // kept = positions not swallowed; new length (same in every lane)
    B128 kept = {~shl128(sel, 1).lo, ~shl128(sel, 1).hi};
    B128 lenmask = {len >= 64 ? ~0ULL : (~0ULL >> (64 - len)),
                    len <= 64 ? 0ULL
                              : (len >= 128 ? ~0ULL : (~0ULL >> (128 - len)))};
    const int newlen = __popcll(and128(kept, lenmask).lo)
                     + __popcll(and128(kept, lenmask).hi);
    // In-place parallel scatter (values already in registers): ids,
    // pair values and merge marks all move together
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int j = lane + 32 * k;
      if (j < len && bit128(kept, j)) {
        const int r = rank128(kept, j);
        const bool ms = bit128(sel, j);
        ids[r] = ms ? mid : val[k];
        pvv[r] = vv[k];               // old (j,j+1); merged re-probed below
        msel[r] = ms;
      }
    }
    __syncwarp();
    // Repair: re-probe only the pairs adjacent to a new token (this is
    // the entire cost of the incremental maintenance)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = lane + 32 * k;
      if (r < newlen - 1 && (msel[r] || msel[r + 1]))
        pvv[r] = pair_lookup(PK, PV, pmask, ids[r], ids[r + 1]);
    }
    len = newlen;
    first = false;
    __syncwarp();
  }
  for (int j = lane; j < len; j += 32) scratch[s + j] = ids[j];
  if (lane == 0) cnt[p] = len;
}

__global__ void k_bpe_warp(const uint8_t* __restrict__ bytes,
                           const int32_t* __restrict__ pb,
                           const int32_t* __restrict__ plist, int n_piece,
                           const int32_t* __restrict__ np_dev,
                           const uint64_t* __restrict__ PK,
                           const uint64_t* __restrict__ PV, unsigned pmask,
                           const int32_t* __restrict__ byte_id,
                           const uint64_t* __restrict__ VK,
                           const uint64_t* __restrict__ VV, unsigned vmask,
                           const uint8_t* __restrict__ blob, int ign,
                           int32_t* __restrict__ scratch,
                           int32_t* __restrict__ cnt,
                           const uint32_t* __restrict__ ub) {
  if (np_dev) n_piece = *np_dev;
  const int lane = threadIdx.x & 31;
  const int wib = threadIdx.x >> 5;              // warp-in-block
  __shared__ int32_t sh_ids[WPB][MED_MAX];
  __shared__ long long sh_vv[WPB][MED_MAX];      // pair-value cache (incr.)
  __shared__ uint8_t sh_m[WPB][MED_MAX];         // merge marks this round
  const int w0 = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int wstride = (gridDim.x * blockDim.x) >> 5;
  for (int w = w0; w < n_piece; w += wstride) {
    bpe_warp_one(plist[w], lane, sh_ids[wib], sh_vv[wib], sh_m[wib],
                 bytes, pb, PK, PV, pmask, byte_id, VK, VV, vmask, blob,
                 ign, scratch, cnt, ub);
    __syncwarp();
  }
}

__global__ void k_bpe_long(const uint8_t* __restrict__ bytes,
                           const int32_t* __restrict__ pb,
                           const int32_t* __restrict__ plist,
                           const uint64_t* __restrict__ PK,
                           const uint64_t* __restrict__ PV, unsigned pmask,
                           const int32_t* __restrict__ byte_id,
                           const uint64_t* __restrict__ VK,
                           const uint64_t* __restrict__ VV, unsigned vmask,
                           const uint8_t* __restrict__ blob, int ign,
                           int32_t* __restrict__ scratch,  // output + bufA
                           int32_t* __restrict__ bufB,
                           int32_t* __restrict__ cnt,
                           int n_long, const int32_t* __restrict__ nl_dev,
                           const uint32_t* __restrict__ ub) {
  if (nl_dev) n_long = *nl_dev;
  __shared__ int sh_hit;
  __shared__ int sh_drop;      // missing-byte flag, kept separate from
                               // sh_hit: on the ign-miss path, lagging
                               // threads are still reading sh_hit, so
                               // reusing it as this flag would be a data
                               // race (observed intermittently)
  __shared__ long long sh_rank[TPB];
  __shared__ int sh_j[TPB];
  __shared__ int sh_len;
  // Candidate bitmap (<= 64K symbols): the thread-0 merge scan no
  // longer performs serial hash lookups
  __shared__ uint32_t sh_cand[2048];
  // Block-level grid-stride loop over pieces (the same striding pattern
  // as the warp-level loop in k_bpe_warp). The graph path of
  // encode_fused caps the grid at 8192, while n_long for a 4 MiB bucket
  // can reach ~32000; with the older block-per-piece launch the cnt of
  // every piece beyond the grid kept its torch::empty garbage, which
  // made InclusiveSum produce garbage offsets and k_bpe_compact write
  // out[o+j] at a garbage o = an out-of-bounds device write.
  for (int pi = blockIdx.x; pi < n_long; pi += gridDim.x) {
  int p = plist[pi];
  int s = pb[p], e = pb[p + 1], m = e - s;
  if (ign) {
    if (threadIdx.x == 0)
      sh_hit = vocab_lookup(bytes, s, m, VK, VV, vmask, blob);
    __syncthreads();
    if (sh_hit >= 0) {
      if (threadIdx.x == 0) { scratch[s] = sh_hit; cnt[p] = 1; }
      // continue skips the barrier at the end of the loop, so one is
      // needed here: otherwise thread 0 could enter the next iteration
      // and overwrite sh_hit while lagging threads of this iteration
      // are still reading it (two lines above)
      __syncthreads();
      continue;                       // was a return before the loop
    }
  }
  int32_t* cur = scratch + s;
  int32_t* alt = bufB + s;
  if (threadIdx.x == 0) sh_drop = 0;
  __syncthreads();
  bool my_drop = false;
  for (int j = threadIdx.x; j < m; j += blockDim.x) {
    const int v = byte_id[bytes[s + j]];
    cur[j] = v;
    my_drop |= v < 0;
  }
  if (my_drop) atomicOr(&sh_drop, 1);
  __syncthreads();
  int len = m;
  if (sh_drop) {
    // Rare path (unk-free family missing a byte): thread 0 compacts
    // forward in place (m2 <= j always holds), reproducing the
    // reference "dropped before merging"; full-coverage families never
    // enter here
    if (threadIdx.x == 0) {
      int m2 = 0;
      for (int j = 0; j < m; ++j) {
        const int v = cur[j];
        if (v >= 0) cur[m2++] = v;
      }
      sh_len = m2;
    }
    __syncthreads();
    len = sh_len;
  }
  while (len > 1) {
    // 1) find the smallest rank in parallel (equal rank <=> equal pair,
    //    so positions need not be compared)
    long long my = -1;
    for (int j = threadIdx.x; j < len - 1; j += blockDim.x) {
      long long v = pair_lookup(PK, PV, pmask, cur[j], cur[j + 1]);
      if (v >= 0 && (my < 0 || (v >> 32) < (my >> 32))) my = v;
    }
    sh_rank[threadIdx.x] = my;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
      if (threadIdx.x < w) {
        long long a = sh_rank[threadIdx.x], b = sh_rank[threadIdx.x + w];
        if (a < 0 || (b >= 0 && (b >> 32) < (a >> 32)))
          sh_rank[threadIdx.x] = b;
      }
      __syncthreads();
    }
    long long best = sh_rank[0];
    if (best < 0) break;
    int merged_id = (int)(best & 0xFFFFFFFF);
    int rank = (int)(best >> 32);
    // Non-monotone merge-table guard (same reasoning as in
    // bpe_warp_one): a round with a flagged rank merges only the
    // leftmost occurrence. For monotone-table families ub==nullptr, so
    // this is always false.
    const bool ub1 =
        ub && ((ub[(unsigned)rank >> 5] >> ((unsigned)rank & 31u)) & 1u);
    // 2) mark the occurrences of that pair in parallel (bitmap),
    //    replacing the serial probe by thread 0
    bool use_bitmap = len <= 65536;
    if (use_bitmap) {
      for (int w = threadIdx.x; w < (len + 31) / 32; w += blockDim.x)
        sh_cand[w] = 0;
      __syncthreads();
      for (int j = threadIdx.x; j < len - 1; j += blockDim.x) {
        long long v = pair_lookup(PK, PV, pmask, cur[j], cur[j + 1]);
        if (v >= 0 && (int)(v >> 32) == rank)
          atomicOr(&sh_cand[j >> 5], 1u << (j & 31));
      }
      __syncthreads();
    }
    // 3) single-threaded batch merge of all leftmost non-overlapping
    //    occurrences (compacting into alt; on an overlapping chain the
    //    even positions win). For a ub1 round, merging stops (halt)
    //    after the leftmost occurrence and the rest is copied through.
    if (threadIdx.x == 0) {
      int w = 0;
      bool halt = false;
      for (int j = 0; j < len; ) {
        bool hit;
        if (use_bitmap) {
          hit = j + 1 < len && (sh_cand[j >> 5] >> (j & 31)) & 1u;
        } else {
          long long v = j + 1 < len
              ? pair_lookup(PK, PV, pmask, cur[j], cur[j + 1]) : -1;
          hit = v >= 0 && (int)(v >> 32) == rank;
        }
        if (hit && !halt) { alt[w++] = merged_id; j += 2; halt = ub1; }
        else              { alt[w++] = cur[j]; ++j; }
      }
      sh_len = w;
    }
    __syncthreads();
    len = sh_len;
    int32_t* t2 = cur; cur = alt; alt = t2;
    __syncthreads();
  }
  // 3) land the result in scratch uniformly
  if (cur != scratch + s)
    for (int j = threadIdx.x; j < len; j += blockDim.x)
      (scratch + s)[j] = cur[j];
  if (threadIdx.x == 0) cnt[p] = len;
  // Barrier at the end of the loop. sh_hit/sh_rank/sh_len/sh_cand are
  // reused across iterations: while lagging threads of iteration i are
  // still reading (e.g. sh_rank[0] before breaking out of the while),
  // the writers of the next iteration must not run ahead. Without it
  // this is a data race that small-scale tests still pass.
  __syncthreads();
  }
}

__global__ void k_bpe_compact(const int32_t* __restrict__ pb,
                              const int32_t* __restrict__ cnt_off,  // [P+1]
                              const int32_t* __restrict__ scratch,
                              int32_t* __restrict__ out, int P,
                              const int32_t* __restrict__ P_dev) {
  if (P_dev) P = *P_dev;
  for (int p = blockIdx.x; p < P; p += gridDim.x) {
    int s = pb[p], o = cnt_off[p], c = cnt_off[p + 1] - o;
    for (int j = threadIdx.x; j < c; j += blockDim.x)
      out[o + j] = scratch[s + j];
  }
}

// bytes + piece byte bounds pb[P+1] -> (token_ids, offsets off[P+1])

// ---- Piece memoization: an exact hash -> ids cache for pieces of
// len <= 16B. Correctness: key = FNV64(bytes) plus a full byte
// comparison; insert and query keep stream order (the lookup of a call
// precedes its insert, and calls are isolated by stream ordering), so
// there are no torn reads. Collisions and over-length pieces only cost
// speed, never correctness. The table is SoA (keys/meta/bytes/vals in
// separate arrays, following the measurement that ruled out AoS).


__global__ void k_memo_insert(const uint8_t* bytes, const int32_t* pb, int P,
                              uint64_t* mkeys, int32_t* mmeta,
                              uint8_t* mbytes, int32_t* mvals,
                              unsigned mmask, const int32_t* scratch,
                              const int32_t* cnt) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  int s = pb[p], len = pb[p + 1] - s;
  if (len <= 0 || len > MEMO_LEN) return;
  int n = cnt[p];
  if (n <= 0 || n > MEMO_IDS) return;
  uint64_t h = memo_hash(bytes + s, len);
  for (int k = 0; k < MEMO_PROBE; ++k) {
    unsigned slot = (unsigned)(h + k) & mmask;
    uint64_t old = atomicCAS((unsigned long long*)&mkeys[slot], 0ULL,
                             (unsigned long long)h);
    if (old == 0) {
      for (int i = 0; i < len; ++i)
        mbytes[(size_t)slot * MEMO_LEN + i] = bytes[s + i];
      for (int i = 0; i < n; ++i)
        mvals[(size_t)slot * MEMO_IDS + i] = scratch[s + i];
    __threadfence();
      mmeta[slot] = (len << 8) | n;              // commit meta last
      return;
    }
    if (old == h) return;                        // dup: 1st writer wins
  }
}

// Metadata-level contract check (zero cost, shared by the two entries
// bpe_encode_impl and encode_fused): pmask/vmask use numel-1 as the
// wrap-around probe mask, so a table length must be a power of two and
// keys/vals must have equal length; byte_id must cover the whole byte
// domain; and every table must be contiguous and on the same device as
// bytes. A violation used to mean a silent out-of-bounds probe or a
// wrong-table read inside the kernel.
#ifndef TOKTIER_DEVICE_ONLY
static void check_bpe_tables_meta(const torch::Tensor& bytes,
                                  const torch::Tensor& pair_keys,
                                  const torch::Tensor& pair_vals,
                                  const torch::Tensor& byte_id,
                                  const torch::Tensor& vocab_keys,
                                  const torch::Tensor& vocab_vals,
                                  const torch::Tensor& vocab_blob) {
  auto pow2 = [](int64_t n) { return n > 0 && (n & (n - 1)) == 0; };
  TORCH_CHECK(pow2(pair_keys.numel()) &&
              pair_vals.numel() == pair_keys.numel(),
              "pair_keys/pair_vals must be equal-length power-of-two tables");
  TORCH_CHECK(pow2(vocab_keys.numel()) &&
              vocab_vals.numel() == vocab_keys.numel(),
              "vocab_keys/vocab_vals must be equal-length power-of-two tables");
  // The dtypes must match the types the kernel reads. All four
  // pair/vocab tables are reinterpreted as uint64 inside the kernel
  // (data_ptr() cast to const uint64_t*), and the canonical Python path
  // loads them as an int64 view (np.uint64.view(np.int64)); any dtype
  // other than an 8-byte integer (an int32 or float64 table, say) is a
  // silent wrong-table probe.
  for (const torch::Tensor* t : {&pair_keys, &pair_vals,
                                 &vocab_keys, &vocab_vals})
    TORCH_CHECK(t->dtype() == torch::kInt64 || t->dtype() == torch::kUInt64,
                "pair/vocab keys/vals must be 64-bit integer tensors "
                "(kernels reinterpret them as uint64)");
  TORCH_CHECK(byte_id.dtype() == torch::kInt32 && byte_id.numel() == 256,
              "byte_id must be int32[256]");
  TORCH_CHECK(vocab_blob.dtype() == torch::kUInt8,
              "vocab_blob must be uint8");
  for (const torch::Tensor* t : {&pair_keys, &pair_vals, &byte_id,
                                 &vocab_keys, &vocab_vals, &vocab_blob})
    TORCH_CHECK(t->is_cuda() && t->device() == bytes.device() &&
                t->is_contiguous(),
                "BPE tables must be contiguous CUDA tensors on bytes' device");
}

// Non-monotone merge-table guard bitmap (optional input). bit[rank]=1
// <=> the target token of that rule is used as a component by a rule of
// smaller rank (exported by bpe_tables.unsafe_bits, from the same source
// and with the same length as merges). An empty tensor means a monotone
// table (all previously certified families), the kernel then takes the
// nullptr branch and behaves bit-identically. The size contract (number
// of bits >= n_merges) is guaranteed by the Python-side build, like the
// other tables.
static const uint32_t* unsafe_bits_ptr(const torch::Tensor& t,
                                       const torch::Tensor& bytes) {
  if (t.numel() == 0) return nullptr;
  TORCH_CHECK(t.is_cuda() && t.device() == bytes.device() &&
              t.dtype() == torch::kInt32 && t.is_contiguous(),
              "unsafe_bits must be a contiguous int32 CUDA tensor on "
              "bytes' device (uint32 bitmap viewed as int32)");
  return (const uint32_t*)t.data_ptr<int32_t>();
}

// Optional argument-level contract check on the piece-boundary array.
// It is O(P) plus a host synchronisation, so it is available only on the
// eager entry points: encode_fused is the graph-capture path, and reading
// a device value from the host during capture is illegal.
//
// This is a build-time option, never an environment or configuration
// switch: it only adds assertions and can never change output. The
// certified build sets it to 0, which is also the default.
#ifndef TOKTIER_PB_CONTENT_CHECK
#define TOKTIER_PB_CONTENT_CHECK 0
#endif

std::vector<torch::Tensor> bpe_encode_impl(
    torch::Tensor bytes, torch::Tensor pb, torch::Tensor mkeys,
    torch::Tensor mmeta, torch::Tensor mbytes, torch::Tensor mvals,
    torch::Tensor pair_keys,
                                      torch::Tensor pair_vals,
                                      torch::Tensor byte_id,
                                      torch::Tensor vocab_keys,
                                      torch::Tensor vocab_vals,
                                      torch::Tensor vocab_blob,
                                      int64_t ignore_merges,
                                      torch::Tensor unsafe_bits) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(bytes));
  TORCH_CHECK(bytes.is_cuda() && bytes.dtype() == torch::kUInt8);
  TORCH_CHECK(pb.dtype() == torch::kInt32 && pb.is_contiguous());
  // Metadata-level checks (bytes contiguity, pb device, table metadata)
  TORCH_CHECK(bytes.is_contiguous(), "bytes must be contiguous");
  TORCH_CHECK(pb.is_cuda() && pb.device() == bytes.device() &&
              pb.numel() >= 1,
              "pb must be a non-empty CUDA tensor on bytes' device");
  check_bpe_tables_meta(bytes, pair_keys, pair_vals, byte_id,
                        vocab_keys, vocab_vals, vocab_blob);
  const int64_t P = pb.numel() - 1;
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32)
                      .device(bytes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (P <= 0)
    return {torch::empty({0}, opts_i32), torch::zeros({1}, opts_i32)};
  // Argument-level contract check (this impl is reached only from the
  // eager entries bpe_encode / bpe_encode_memo; encode_fused never gets
  // here): pb non-decreasing and within bounds. Off in certified builds.
#if TOKTIER_PB_CONTENT_CHECK
  {
    TORCH_CHECK((pb.diff() >= 0).all().item<bool>(),
                "pb must be non-decreasing");
    TORCH_CHECK(pb[0].item<int32_t>() >= 0 &&
                pb[P].item<int32_t>() <= bytes.numel(),
                "pb out of range [0, bytes.numel()]");
  }
#endif

  const bool memo = mkeys.numel() > 0;
  unsigned mmask = memo ? (unsigned)mkeys.numel() - 1 : 0;
  auto scratch = torch::empty({bytes.numel()}, opts_i32);
  auto cnt = torch::empty({P}, opts_i32);
  auto lens = pb.slice(0, 1, P + 1) - pb.slice(0, 0, P);
  auto short_list = torch::nonzero(lens <= SHORT_MAX)
                        .flatten().to(torch::kInt32);
  auto warp_list = torch::nonzero((lens > SHORT_MAX) & (lens <= MED_MAX))
                       .flatten().to(torch::kInt32);
  auto long_list = torch::nonzero(lens > MED_MAX)
                       .flatten().to(torch::kInt32);
  const int n_short = (int)short_list.numel();
  const int n_warp = (int)warp_list.numel();
  const int n_long = (int)long_list.numel();
  unsigned pmask = (unsigned)pair_keys.numel() - 1;
  unsigned vmask = (unsigned)vocab_keys.numel() - 1;
  const uint32_t* ubp = unsafe_bits_ptr(unsafe_bits, bytes);  // guard bits

  // The three piece classes are disjoint (scratch/cnt are written per
  // piece), so the warp and long paths go to side streams and overlap
  // with the thread path (profiling showed the long path filling 162
  // blocks across 188 SMs, i.e. nearly idle, and running it serially
  // cost ~13% end to end). The cnt scan resumes on the main stream only
  // after the events join, and everything is joined before returning, so
  // the stream-ordering semantics of the caching allocator stay intact.
  at::cuda::CUDAEvent ev_ready, ev_warp, ev_long;
  auto s_warp = at::cuda::getStreamFromPool(false, bytes.device().index());
  auto s_long = at::cuda::getStreamFromPool(false, bytes.device().index());
  ev_ready.record(stream);
  if (n_short > 0)
    k_bpe_thread<SHORT_MAX><<<(n_short + TPB - 1) / TPB, TPB, 0, stream>>>(
        bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(),
        short_list.data_ptr<int32_t>(), n_short,
        (const uint64_t*)pair_keys.data_ptr(),
        (const uint64_t*)pair_vals.data_ptr(), pmask,
        byte_id.data_ptr<int32_t>(),
        (const uint64_t*)vocab_keys.data_ptr(),
        (const uint64_t*)vocab_vals.data_ptr(), vmask,
        vocab_blob.data_ptr<uint8_t>(), (int)ignore_merges,
        scratch.data_ptr<int32_t>(), cnt.data_ptr<int32_t>(), nullptr,
        memo ? (const uint64_t*)mkeys.data_ptr() : nullptr,
        memo ? mmeta.data_ptr<int32_t>() : nullptr,
        memo ? mbytes.data_ptr<uint8_t>() : nullptr,
        memo ? mvals.data_ptr<int32_t>() : nullptr, mmask);
  if (n_warp > 0) {
    ev_ready.block(s_warp);
    k_bpe_warp<<<(int)(((int64_t)n_warp * 32 + TPB - 1) / TPB), TPB, 0,
                 s_warp>>>(
        bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(),
        warp_list.data_ptr<int32_t>(), n_warp, nullptr,
        (const uint64_t*)pair_keys.data_ptr(),
        (const uint64_t*)pair_vals.data_ptr(), pmask,
        byte_id.data_ptr<int32_t>(),
        (const uint64_t*)vocab_keys.data_ptr(),
        (const uint64_t*)vocab_vals.data_ptr(), vmask,
        vocab_blob.data_ptr<uint8_t>(), (int)ignore_merges,
        scratch.data_ptr<int32_t>(), cnt.data_ptr<int32_t>(),
        ubp);
    ev_warp.record(s_warp);
    ev_warp.block(stream);
  }
  torch::Tensor bufB;
  if (n_long > 0) {
    bufB = torch::empty({bytes.numel()}, opts_i32);
    ev_ready.block(s_long);
    k_bpe_long<<<n_long, TPB, 0, s_long>>>(
        bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(),
        long_list.data_ptr<int32_t>(),
        (const uint64_t*)pair_keys.data_ptr(),
        (const uint64_t*)pair_vals.data_ptr(), pmask,
        byte_id.data_ptr<int32_t>(),
        (const uint64_t*)vocab_keys.data_ptr(),
        (const uint64_t*)vocab_vals.data_ptr(), vmask,
        vocab_blob.data_ptr<uint8_t>(), (int)ignore_merges,
        scratch.data_ptr<int32_t>(), bufB.data_ptr<int32_t>(),
        cnt.data_ptr<int32_t>(), n_long, nullptr, ubp);
    ev_long.record(s_long);
    ev_long.block(stream);
  }

  if (memo)
    k_memo_insert<<<(int)((P + TPB - 1) / TPB), TPB, 0, stream>>>(
        bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(), (int)P,
        (uint64_t*)mkeys.data_ptr(), mmeta.data_ptr<int32_t>(),
        mbytes.data_ptr<uint8_t>(), mvals.data_ptr<int32_t>(), mmask,
        scratch.data_ptr<int32_t>(), cnt.data_ptr<int32_t>());

  auto off = torch::zeros({P + 1}, opts_i32);
  size_t tmp_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tmp_bytes,
                                cnt.data_ptr<int32_t>(),
                                off.data_ptr<int32_t>() + 1, (int)P, stream);
  auto tmp = torch::empty({(int64_t)tmp_bytes},
                          torch::TensorOptions().dtype(torch::kUInt8)
                              .device(bytes.device()));
  cub::DeviceScan::InclusiveSum(tmp.data_ptr<uint8_t>(), tmp_bytes,
                                cnt.data_ptr<int32_t>(),
                                off.data_ptr<int32_t>() + 1, (int)P, stream);
  int T = off[P].item<int32_t>();
  auto out = torch::empty({(int64_t)T}, opts_i32);
  k_bpe_compact<<<(int)P, 64, 0, stream>>>(
      pb.data_ptr<int32_t>(), off.data_ptr<int32_t>(),
      scratch.data_ptr<int32_t>(), out.data_ptr<int32_t>(), (int)P, nullptr);
  return {out, off};
}

std::vector<torch::Tensor> bpe_encode(torch::Tensor bytes, torch::Tensor pb,
                                      torch::Tensor pair_keys,
                                      torch::Tensor pair_vals,
                                      torch::Tensor byte_id,
                                      torch::Tensor vocab_keys,
                                      torch::Tensor vocab_vals,
                                      torch::Tensor vocab_blob,
                                      int64_t ignore_merges) {
  auto dev = bytes.device();
  auto e64 = torch::empty({0}, torch::TensorOptions()
                                   .dtype(torch::kInt64).device(dev));
  auto e32 = torch::empty({0}, torch::TensorOptions()
                                   .dtype(torch::kInt32).device(dev));
  auto e8 = torch::empty({0}, torch::TensorOptions()
                                  .dtype(torch::kUInt8).device(dev));
  return bpe_encode_impl(bytes, pb, e64, e32, e8, e32, pair_keys, pair_vals,
                         byte_id, vocab_keys, vocab_vals, vocab_blob,
                         ignore_merges, e32);
}

// Overload carrying the non-monotone guard bitmap (called from Python
// under the same name with a 10th argument; the old 9-argument call
// resolves to the original signature above and is bit-identical)
std::vector<torch::Tensor> bpe_encode_v2(
    torch::Tensor bytes, torch::Tensor pb, torch::Tensor pair_keys,
    torch::Tensor pair_vals, torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob,
    int64_t ignore_merges, torch::Tensor unsafe_bits) {
  auto dev = bytes.device();
  auto e64 = torch::empty({0}, torch::TensorOptions()
                                   .dtype(torch::kInt64).device(dev));
  auto e32 = torch::empty({0}, torch::TensorOptions()
                                   .dtype(torch::kInt32).device(dev));
  auto e8 = torch::empty({0}, torch::TensorOptions()
                                  .dtype(torch::kUInt8).device(dev));
  return bpe_encode_impl(bytes, pb, e64, e32, e8, e32, pair_keys, pair_vals,
                         byte_id, vocab_keys, vocab_vals, vocab_blob,
                         ignore_merges, unsafe_bits);
}

std::vector<torch::Tensor> bpe_encode_memo(
    torch::Tensor bytes, torch::Tensor pb, torch::Tensor pair_keys,
    torch::Tensor pair_vals, torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob, int64_t ignore_merges,
    torch::Tensor mkeys, torch::Tensor mmeta, torch::Tensor mbytes,
    torch::Tensor mvals) {
  auto e32 = torch::empty({0}, torch::TensorOptions()
                                   .dtype(torch::kInt32)
                                   .device(bytes.device()));
  return bpe_encode_impl(bytes, pb, mkeys, mmeta, mbytes, mvals, pair_keys,
                         pair_vals, byte_id, vocab_keys, vocab_vals,
                         vocab_blob, ignore_merges, e32);
}

// Guard-bitmap overload of the memo entry (14th argument; the old
// 13-argument call is unchanged)
std::vector<torch::Tensor> bpe_encode_memo_v2(
    torch::Tensor bytes, torch::Tensor pb, torch::Tensor pair_keys,
    torch::Tensor pair_vals, torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob, int64_t ignore_merges,
    torch::Tensor mkeys, torch::Tensor mmeta, torch::Tensor mbytes,
    torch::Tensor mvals, torch::Tensor unsafe_bits) {
  return bpe_encode_impl(bytes, pb, mkeys, mmeta, mbytes, mvals, pair_keys,
                         pair_vals, byte_id, vocab_keys, vocab_vals,
                         vocab_blob, ignore_merges, unsafe_bits);
}


// ========= Fully fused single-request path (CUDA Graph capturable) ====
//
// bytes[cap] (the tail may be 0x80 continuation padding, which the
// decoder ignores naturally) -> token ids. Every data-dependent count
// (C/R/P/T and the three dispatch lists) stays on the device and the
// kernel geometry depends only on cap, so there is no host sync and the
// whole chain can be captured and replayed. All nonzero calls became CUB
// DeviceSelect; the caller reads the returned 1-element tensor
// off[cap]=T to learn the valid length.

#endif

__global__ void k_pb_sentinel(int32_t* __restrict__ pb,
                              const int32_t* __restrict__ nP,
                              const int32_t* __restrict__ nb) {
  pb[*nP] = *nb;                     // last piece ends at real nb, not cap
}

__global__ void k_dispatch_flags(const int32_t* __restrict__ pb,
                                 const int32_t* __restrict__ nP,
                                 uint8_t* __restrict__ fS,
                                 uint8_t* __restrict__ fW,
                                 uint8_t* __restrict__ fL,
                                 int32_t* __restrict__ cnt, int cap) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= cap) return;
  const int P = *nP;
  if (p >= P) { fS[p] = fW[p] = fL[p] = 0; cnt[p] = 0; return; }
  const int len = pb[p + 1] - pb[p];
  fS[p] = len <= SHORT_MAX;
  fW[p] = (len > SHORT_MAX) & (len <= MED_MAX);
  fL[p] = len > MED_MAX;
}

#ifndef TOKTIER_DEVICE_ONLY
template <int RS>
static std::vector<torch::Tensor> encode_fused_t(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, torch::Tensor pair_keys, torch::Tensor pair_vals,
    torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob,
    int64_t ignore_merges, torch::Tensor unsafe_bits) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(bytes));
  TORCH_CHECK(bytes.is_cuda() && bytes.dtype() == torch::kUInt8 &&
              bytes.is_contiguous());
  TORCH_CHECK(nb_dev.is_cuda() && nb_dev.dtype() == torch::kInt32 &&
              nb_dev.numel() == 1);
  // Metadata-level checks only; content-level checks are never added
  // here because a host read is forbidden during graph capture. tab
  // covering 0x110000 is the premise of "tab[cp] needs no guard after
  // sanitizing".
  TORCH_CHECK(nb_dev.device() == bytes.device(),
              "nb_dev must live on bytes' device");
  TORCH_CHECK(tab.is_cuda() && tab.device() == bytes.device() &&
              tab.dtype() == torch::kUInt8 && tab.is_contiguous() &&
              tab.numel() >= 0x110000,
              "tab must be a contiguous uint8 CUDA table covering U+10FFFF");
  check_bpe_tables_meta(bytes, pair_keys, pair_vals, byte_id,
                        vocab_keys, vocab_vals, vocab_blob);
  const int64_t cap64 = bytes.numel();
  TORCH_CHECK(cap64 > 0 && cap64 < INT32_MAX);
  const int cap = (int)cap64;
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32)
                      .device(bytes.device());
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8)
                     .device(bytes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int gsB = (cap + TPB - 1) / TPB;

  // CUB workspace: query the temporary size of every algorithm once,
  // take the maximum, and share one buffer for the whole chain
  auto lead_it = thrust::make_transform_iterator(
      (const uint8_t*)bytes.data_ptr<uint8_t>(), IsLead{});
  auto head_it = thrust::make_transform_iterator(
      (const uint8_t*)nullptr, CastU8{});
  thrust::counting_iterator<int32_t> cnt_it(0);
  size_t tmax = 0, tq = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tq, lead_it, (int32_t*)nullptr,
                                cap, stream);
  tmax = std::max(tmax, tq); tq = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tq, head_it, (int32_t*)nullptr,
                                cap, stream);
  tmax = std::max(tmax, tq); tq = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tq, (const int32_t*)nullptr,
                                (int32_t*)nullptr, cap, stream);
  tmax = std::max(tmax, tq); tq = 0;
  cub::DeviceSelect::Flagged(nullptr, tq, (const int32_t*)nullptr,
                             (const bool*)nullptr, (int32_t*)nullptr,
                             (int32_t*)nullptr, cap, stream);
  tmax = std::max(tmax, tq); tq = 0;
  cub::DeviceSelect::Flagged(nullptr, tq, cnt_it, (const uint8_t*)nullptr,
                             (int32_t*)nullptr, (int32_t*)nullptr,
                             cap, stream);
  tmax = std::max(tmax, tq);
  auto tmp = torch::empty({(int64_t)tmax}, opts_u8);
  void* tp = tmp.data_ptr();

  // ---- UTF-8 decode (padding = continuation bytes, no chars) ----
  auto cpos = torch::empty({cap64}, opts_i32);
  cub::DeviceScan::InclusiveSum(tp, tmax, lead_it,
                                cpos.data_ptr<int32_t>(), cap, stream);
  auto cp = torch::empty({cap64}, opts_i32);
  auto bo = torch::empty({cap64}, opts_i32);
  // Graph layer: err is nullptr, because reading a device value from the
  // host during capture is illegal. Validity comes from the upstream str
  // contract (serving input is guaranteed by the Python str type), and
  // the U+FFFD clamp inside the kernel is only defense in depth (it
  // never fires on valid input).
  k_utf8_decode<<<gsB, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), cpos.data_ptr<int32_t>(),
      cp.data_ptr<int32_t>(), bo.data_ptr<int32_t>(), cap, nullptr);
  const int32_t* d_C = cpos.data_ptr<int32_t>() + (cap - 1);

  // ---- DeepSeek: B mask prepass (S0/S1 cut points OR-ed into the
  // dstart channel; the kernel geometry only depends on cap and the
  // data-dependent quantities go through d_C, so this section stays
  // CUDA-Graph capturable) ----
  torch::Tensor dsB, dsDso, dsArs;
  const uint8_t* dsB_p = nullptr;
  const int32_t* dsDso_p = nullptr;
  const int32_t* dsArs_p = nullptr;
  if (RS == RS_DEEPSEEK) {
    auto pre = ds_prepass(cp, tab, dmax, nullptr, cap, d_C);
    dsB = std::get<0>(pre);
    dsDso = std::get<1>(pre);
    dsArs = std::get<2>(pre);
    dsB_p = dsB.data_ptr<uint8_t>();
    dsDso_p = dsDso.data_ptr<int32_t>();
    dsArs_p = dsArs.data_ptr<int32_t>();
  } else if (RS == RS_LAGUNA) {
    // stage-0 newline-run cut points are OR-ed into the B/dso channel
    // (ars does not apply); the geometry still depends only on cap, so
    // CUDA-Graph capturability is unchanged
    auto pre = lag_prepass(cp, nullptr, cap, d_C);
    dsB = std::get<0>(pre);
    dsDso = std::get<1>(pre);
    dsB_p = dsB.data_ptr<uint8_t>();
    dsDso_p = dsDso.data_ptr<int32_t>();
  }

  // ---- classify / run segmentation (the head tail must be cleared so
  // that the scan keeps R constant past the end) ----
  auto cls = torch::empty({cap64}, opts_u8);
  auto head = torch::empty({cap64}, opts_u8);
  cudaMemsetAsync(head.data_ptr<uint8_t>(), 0, cap, stream);
  k_classify<RS><<<gsB, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(), dsB_p,
      cls.data_ptr<uint8_t>(), head.data_ptr<uint8_t>(), 0, d_C);
  auto rid = torch::empty({cap64}, opts_i32);
  auto head_it2 = thrust::make_transform_iterator(
      (const uint8_t*)head.data_ptr<uint8_t>(), CastU8{});
  cub::DeviceScan::InclusiveSum(tp, tmax, head_it2,
                                rid.data_ptr<int32_t>(), cap, stream);
  const int32_t* d_R = rid.data_ptr<int32_t>() + (cap - 1);
  auto run_start = torch::empty({cap64}, opts_i32);
  auto fnc = torch::empty({cap64}, opts_i32);
  auto lc = torch::empty({cap64}, opts_i32);
  cudaMemsetAsync(fnc.data_ptr<int32_t>(), 0x7f, (size_t)cap * 4, stream);
  cudaMemsetAsync(lc.data_ptr<int32_t>(), 0xff, (size_t)cap * 4, stream);
  k_runinfo<RS><<<gsB, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      head.data_ptr<uint8_t>(), rid.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), fnc.data_ptr<int32_t>(),
      lc.data_ptr<int32_t>(), 0, d_C);
  auto starts = torch::empty({cap64}, torch::TensorOptions()
                                          .dtype(torch::kBool)
                                          .device(bytes.device()));
  cudaMemsetAsync(starts.data_ptr<bool>(), 0, cap, stream);
  k_rules<RS><<<gsB, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      head.data_ptr<uint8_t>(), rid.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), fnc.data_ptr<int32_t>(),
      lc.data_ptr<int32_t>(), dsDso_p, dsB_p, dsArs_p,
      0, (int)dmax, starts.data_ptr<bool>(), 0, d_C, d_R);

  // ---- piece bounds and dispatch (nonzero -> DeviceSelect, counts
  // stay on the device) ----
  auto pb = torch::empty({cap64 + 1}, opts_i32);
  auto d_cnts = torch::empty({4}, opts_i32);   // [P, nS, nM, nL]
  int32_t* d_P = d_cnts.data_ptr<int32_t>();
  cub::DeviceSelect::Flagged(tp, tmax, bo.data_ptr<int32_t>(),
                             starts.data_ptr<bool>(), pb.data_ptr<int32_t>(),
                             d_P, cap, stream);
  k_pb_sentinel<<<1, 1, 0, stream>>>(pb.data_ptr<int32_t>(), d_P,
                                     nb_dev.data_ptr<int32_t>());
  auto flags = torch::empty({3 * cap64}, opts_u8);
  uint8_t* fS = flags.data_ptr<uint8_t>();
  auto cnt = torch::empty({cap64}, opts_i32);
  // Defense in depth: cnt is pre-cleared in full. k_dispatch_flags only
  // clears the p>=P tail, so if the cnt of a valid piece were never
  // written by any BPE kernel (a recurrence of a bound or cap bug) it
  // would hold torch::empty garbage -> garbage offsets -> an
  // out-of-bounds compact write; clearing degrades the failure mode to
  // "missing tokens". The memset is issued on the capture stream and is
  // captured as a graph node (the starts memset above is the precedent);
  // a one-off host-side clear would not survive graph replay.
  cudaMemsetAsync(cnt.data_ptr<int32_t>(), 0, (size_t)cap * 4, stream);
  k_dispatch_flags<<<gsB, TPB, 0, stream>>>(
      pb.data_ptr<int32_t>(), d_P, fS, fS + cap, fS + 2 * cap,
      cnt.data_ptr<int32_t>(), cap);
  auto lists = torch::empty({3 * cap64}, opts_i32);
  int32_t* lS = lists.data_ptr<int32_t>();
  cub::DeviceSelect::Flagged(tp, tmax, cnt_it, fS, lS,
                             d_P + 1, cap, stream);
  cub::DeviceSelect::Flagged(tp, tmax, cnt_it, fS + cap, lS + cap,
                             d_P + 2, cap, stream);
  cub::DeviceSelect::Flagged(tp, tmax, cnt_it, fS + 2 * cap, lS + 2 * cap,
                             d_P + 3, cap, stream);

  // ---- BPE (geometry launched at the capacity bound; out-of-range
  // threads exit or stride according to the device counts) ----
  auto scratch = torch::empty({cap64}, opts_i32);
  auto bufB = torch::empty({cap64}, opts_i32);
  unsigned pmask = (unsigned)pair_keys.numel() - 1;
  unsigned vmask = (unsigned)vocab_keys.numel() - 1;
  const uint32_t* ubp = unsafe_bits_ptr(unsafe_bits, bytes);  // guard bits
  k_bpe_thread<SHORT_MAX><<<gsB, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(), lS, 0,
      (const uint64_t*)pair_keys.data_ptr(),
      (const uint64_t*)pair_vals.data_ptr(), pmask,
      byte_id.data_ptr<int32_t>(),
      (const uint64_t*)vocab_keys.data_ptr(),
      (const uint64_t*)vocab_vals.data_ptr(), vmask,
      vocab_blob.data_ptr<uint8_t>(), (int)ignore_merges,
      scratch.data_ptr<int32_t>(), cnt.data_ptr<int32_t>(), d_P + 1,
      nullptr, nullptr, nullptr, nullptr, 0u);
  // warp-path pieces are >32B, so there are <= cap/33 of them and the
  // thread demand is ~cap, which gsB already covers
  const int gw = std::min(gsB, 8192);
  k_bpe_warp<<<gw, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(), lS + cap, 0,
      d_P + 2,
      (const uint64_t*)pair_keys.data_ptr(),
      (const uint64_t*)pair_vals.data_ptr(), pmask,
      byte_id.data_ptr<int32_t>(),
      (const uint64_t*)vocab_keys.data_ptr(),
      (const uint64_t*)vocab_vals.data_ptr(), vmask,
      vocab_blob.data_ptr<uint8_t>(), (int)ignore_merges,
      scratch.data_ptr<int32_t>(), cnt.data_ptr<int32_t>(),
      ubp);
  const int gl = std::min(cap / (MED_MAX + 1) + 1, 8192);
  k_bpe_long<<<gl, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(), lS + 2 * cap,
      (const uint64_t*)pair_keys.data_ptr(),
      (const uint64_t*)pair_vals.data_ptr(), pmask,
      byte_id.data_ptr<int32_t>(),
      (const uint64_t*)vocab_keys.data_ptr(),
      (const uint64_t*)vocab_vals.data_ptr(), vmask,
      vocab_blob.data_ptr<uint8_t>(), (int)ignore_merges,
      scratch.data_ptr<int32_t>(), bufB.data_ptr<int32_t>(),
      cnt.data_ptr<int32_t>(), 0, d_P + 3, ubp);

  // ---- prefix sum + compaction (the cnt tail is cleared, so off[cap]
  // saturates to T) ----
  auto off = torch::zeros({cap64 + 1}, opts_i32);
  cub::DeviceScan::InclusiveSum(tp, tmax, cnt.data_ptr<int32_t>(),
                                off.data_ptr<int32_t>() + 1, cap, stream);
  auto out = torch::empty({cap64}, opts_i32);
  k_bpe_compact<<<std::min(cap, 65535), 64, 0, stream>>>(
      pb.data_ptr<int32_t>(), off.data_ptr<int32_t>(),
      scratch.data_ptr<int32_t>(), out.data_ptr<int32_t>(), 0, d_P);
  return {out, off.slice(0, cap64, cap64 + 1)};
}

// Wrapper for the old signature: passing an empty unsafe_bits tensor
// selects the kernel's nullptr branch and is bit-identical; the *_v2
// variants carrying the bitmap are registered as pybind overloads under
// the same name.
static torch::Tensor empty_ub(const torch::Tensor& bytes) {
  return torch::empty({0}, torch::TensorOptions()
                               .dtype(torch::kInt32)
                               .device(bytes.device()));
}

std::vector<torch::Tensor> encode_fused(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, torch::Tensor pair_keys, torch::Tensor pair_vals,
    torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob,
    int64_t ignore_merges) {
  return encode_fused_t<RS_CL100K>(bytes, nb_dev, tab, dmax, pair_keys,
                                   pair_vals, byte_id, vocab_keys,
                                   vocab_vals, vocab_blob, ignore_merges,
                                   empty_ub(bytes));
}

std::vector<torch::Tensor> encode_fused_v2(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, torch::Tensor pair_keys, torch::Tensor pair_vals,
    torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob,
    int64_t ignore_merges, torch::Tensor unsafe_bits) {
  return encode_fused_t<RS_CL100K>(bytes, nb_dev, tab, dmax, pair_keys,
                                   pair_vals, byte_id, vocab_keys,
                                   vocab_vals, vocab_blob, ignore_merges,
                                   unsafe_bits);
}

std::vector<torch::Tensor> encode_fused_ds(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, torch::Tensor pair_keys, torch::Tensor pair_vals,
    torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob,
    int64_t ignore_merges) {
  return encode_fused_t<RS_DEEPSEEK>(bytes, nb_dev, tab, dmax, pair_keys,
                                     pair_vals, byte_id, vocab_keys,
                                     vocab_vals, vocab_blob, ignore_merges,
                                     empty_ub(bytes));
}

std::vector<torch::Tensor> encode_fused_ds_v2(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, torch::Tensor pair_keys, torch::Tensor pair_vals,
    torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob,
    int64_t ignore_merges, torch::Tensor unsafe_bits) {
  return encode_fused_t<RS_DEEPSEEK>(bytes, nb_dev, tab, dmax, pair_keys,
                                     pair_vals, byte_id, vocab_keys,
                                     vocab_vals, vocab_blob, ignore_merges,
                                     unsafe_bits);
}

std::vector<torch::Tensor> encode_fused_laguna(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, torch::Tensor pair_keys, torch::Tensor pair_vals,
    torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob,
    int64_t ignore_merges) {
  return encode_fused_t<RS_LAGUNA>(bytes, nb_dev, tab, dmax, pair_keys,
                                   pair_vals, byte_id, vocab_keys,
                                   vocab_vals, vocab_blob, ignore_merges,
                                   empty_ub(bytes));
}

std::vector<torch::Tensor> encode_fused_laguna_v2(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, torch::Tensor pair_keys, torch::Tensor pair_vals,
    torch::Tensor byte_id, torch::Tensor vocab_keys,
    torch::Tensor vocab_vals, torch::Tensor vocab_blob,
    int64_t ignore_merges, torch::Tensor unsafe_bits) {
  return encode_fused_t<RS_LAGUNA>(bytes, nb_dev, tab, dmax, pair_keys,
                                   pair_vals, byte_id, vocab_keys,
                                   vocab_vals, vocab_blob, ignore_merges,
                                   unsafe_bits);
}

// ---------------------------------------------------------------------
// NFC quick-check single-pass scan (moves the CPU quick check of the
// qwen3 cold path onto the GPU). Table semantics = build_nfc_qc_table
// v1_hfengine (0 = safe starter, 1..K = the ccc order index of a safe
// non-starter, 255 = unsafe); the predicate is bit-identical to
// qc_pass_ref: the flag is set <=> there is a 255, or an adjacent ccc
// order violation (prev>cur with cur!=0), or an out-of-range smuggled
// codepoint. Each thread handles one byte: on a lead byte it decodes
// this character in place and looks back one character (if the previous
// character is multi-byte its lead sits exactly at i-k with encoded
// length == k; a 1-byte previous character is ASCII with ccc=0 and needs
// no check). Padding with 0x80 = continuation is always silent. Purely
// device side, no host sync, geometry depends only on the buffer
// capacity, so it is CUDA-Graph capturable. This is additive only: no
// existing kernel line was touched, which keeps the exposure of the
// bit-identical zero-regression gate minimal.
#endif

__device__ __forceinline__ int nfcqc_len(uint8_t c) {
  if (c < 0x80) return 1;
  if ((c >> 5) == 0x6) return 2;
  if ((c >> 4) == 0xE) return 3;
  if ((c >> 3) == 0x1E) return 4;
  return 0;                                   // continuation / bad lead
}

__device__ __forceinline__ int32_t nfcqc_cp(const uint8_t* b, int i,
                                            int len, int n) {
  uint8_t c = b[i];
  if (len == 1) return c;
  int32_t v = c & (0x7F >> len);              // len 2/3/4 -> 1F/0F/07
  for (int k = 1; k < len; ++k)
    v = (v << 6) | (i + k < n ? b[i + k] & 0x3F : 0);
  return v;
}

__global__ void k_nfc_qc(const uint8_t* __restrict__ b,
                         const uint8_t* __restrict__ tab,
                         int n, int32_t* __restrict__ flag) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int len = nfcqc_len(b[i]);
  if (len == 0) return;                       // continuation, not a lead
  int32_t cp = nfcqc_cp(b, i, len, n);
  if (cp > 0x10FFFF) { atomicOr(flag, 1); return; }  // F5+: fail safe
  uint8_t v = tab[cp];
  if (v == 255) { atomicOr(flag, 1); return; }
  if (v != 0) {                               // non-starter: check prev ccc
    for (int k = 2; k <= 4 && k <= i; ++k) {  // 1B prev is ccc0, skipped
      int pl = nfcqc_len(b[i - k]);
      if (pl == k) {
        int32_t pcp = nfcqc_cp(b, i - k, k, n);
        if (pcp <= 0x10FFFF) {
          uint8_t pv = tab[pcp];
          if (pv != 255 && pv > v) atomicOr(flag, 1);
        }                                     // pv==255/oob flagged by lead
        break;
      }
    }
  }
}

#ifndef TOKTIER_DEVICE_ONLY
torch::Tensor nfc_qc_scan(torch::Tensor bytes, torch::Tensor tab) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(bytes));
  TORCH_CHECK(bytes.is_cuda() && bytes.dtype() == torch::kUInt8 &&
              bytes.is_contiguous());
  TORCH_CHECK(tab.is_cuda() && tab.dtype() == torch::kUInt8 &&
              tab.is_contiguous() && tab.numel() == 0x110000,
              "QC table must be a full-plane uint8[0x110000]");
  const int64_t nb64 = bytes.numel();
  TORCH_CHECK(nb64 < INT32_MAX);
  auto flag = torch::zeros({1}, torch::TensorOptions()
                                    .dtype(torch::kInt32)
                                    .device(bytes.device()));
  if (nb64 == 0) return flag;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = (int)nb64;
  const int nblk = (n + TPB - 1) / TPB;
  k_nfc_qc<<<nblk, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), tab.data_ptr<uint8_t>(), n,
      flag.data_ptr<int32_t>());
  return flag;
}

// ==================== o200k splitter group ====================
// The authoritative semantics are the torch reference in
// pretok_gpu_o200k.py (pinned at the split level with 0/5,076 and
// 0/5,056 mismatches). This is a self-contained section: dedicated
// kernels and dedicated entry points that never touch the existing
// RS_CL100K / RS_DEEPSEEK template paths (zero-regression discipline).
// Structure: 4 rid streams (main class with letters merged / cs =
// crlf|slash / lowish / P|M) plus per-run atomic channels plus
// per-position rules (bounded look-back <= 4). Two sparse cases raise a
// flag back to the host: true chains (a fully swallowing link pointing
// at another candidate) and {P union M} ambiguous spans (a P earlier
// than an M).

#endif

namespace o2k {
constexpr uint8_t P = 0, U = 1, L = 2, C = 3, N = 4, S = 5, M = 6;
__device__ __forceinline__ bool letterish(uint8_t c) {
  return c == U || c == L || c == C || c == M;
}
__device__ __forceinline__ bool clike(uint8_t c) { return c == C || c == M; }
__device__ __forceinline__ bool lowish(uint8_t c) {
  return c == L || c == C || c == M;
}
__device__ __forceinline__ bool is_cs(int32_t v) {
  return v == 10 || v == 13 || v == 47;
}
// ---- kimi_k3 (selected by a runtime flag rather than a template) ----
// Class table = pretok_kimi_classes_v1_hfengine (the first 7 classes are
// in the same order as o200k, with HL=7/HN=8/HP=9 appended); the
// absorbed trailer has no slash (cs = crlf); b0 takes a whole Han run.
constexpr uint8_t HL = 7, HN = 8, HP = 9;
__device__ __forceinline__ bool k2_han(uint8_t c) { return c >= HL; }
__device__ __forceinline__ bool k2_cs(int32_t v) {
  return v == 10 || v == 13;
}
__device__ __forceinline__ bool k2_num(uint8_t c) {
  return c == N || c == HN;
}
__device__ __forceinline__ bool k2_punct4(uint8_t c) {
  return c == P || c == M || c == HP;
}
__device__ __forceinline__ bool k2_prefix_ok(uint8_t c, int32_t v) {
  // [^\r\n\p{L}\p{N}]: \p{L} = U|L|C|HL, \p{N} = N|HN; M/HP/S(non-CRLF)/P ok
  return !(v == 10 || v == 13) && c != U && c != L && c != C && c != HL &&
         c != N && c != HN;
}
// cs is family-dependent: o200k = crlf|slash (A4's cross-run absorb
// includes /), kimi = crlf
__device__ __forceinline__ bool cs_of(int32_t v, bool kimi) {
  return kimi ? k2_cs(v) : is_cs(v);
}
// Contraction candidate (bounded look-back 1 plus look-ahead 2; aligned
// clause by clause with l1ok/l2ok of the torch reference, l1 first).
// Returns 0 for "not a candidate", or 1/2 for the number of letters
// consumed. dstart (nullable) = document-start marks of a batch: neither
// the preceding letter nor the suffix may cross a document boundary.
__device__ __forceinline__ int cand_k(int q, int n, const int32_t* cp,
                                      const uint8_t* cls, bool contr,
                                      const uint8_t* dstart) {
  if (!contr || q < 1 || q >= n || cp[q] != 39) return 0;
  if (dstart && dstart[q]) return 0;
  if (!letterish(cls[q - 1])) return 0;
  if (dstart && q + 1 < n && dstart[q + 1]) return 0;
  bool lt1 = q + 1 < n && letterish(cls[q + 1]);
  int f1 = q + 1 < n ? fold(cp[q + 1]) : -1;
  if (lt1 && (f1 == 0x73 || f1 == 0x74 || f1 == 0x6D || f1 == 0x64))
    return 1;
  bool cross2 = dstart && q + 2 < n && dstart[q + 2];
  bool lt2 = !cross2 && q + 2 < n && letterish(cls[q + 2]);
  int f2 = q + 2 < n ? fold(cp[q + 2]) : -1;
  if (lt2 && ((f1 == 0x72 && f2 == 0x65) || (f1 == 0x76 && f2 == 0x65) ||
              (f1 == 0x6C && f2 == 0x6C)))
    return 2;
  return 0;
}
}  // namespace o2k

// Four kinds of head: the main class (the four letter subclasses merged,
// P/N/S as they are), cs, lowish and P|M.
// n_dev (nullable): device-side bound for the graph path (same
// convention as k_classify); on that path the caller memsets the head
// tail to zero so the scan keeps R constant past the end.
__global__ void k_o2k_heads(const int32_t* __restrict__ cp,
                            const uint8_t* __restrict__ tab, int n,
                            const int32_t* __restrict__ n_dev,
                            const uint8_t* __restrict__ dstart,
                            uint8_t* __restrict__ cls,
                            uint8_t* __restrict__ headM,
                            uint8_t* __restrict__ headCS,
                            uint8_t* __restrict__ headLW,
                            uint8_t* __restrict__ headPM, bool kimi) {
  if (n_dev) n = *n_dev;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int32_t v = cp[i];
  uint8_t c = tab[v];
  cls[i] = c;
  // Main-class merge: letter -> 1; the three kimi Han subclasses -> HL
  // (b0 treats a whole Han span as one run)
  uint8_t mc = o2k::letterish(c) ? 1
               : (kimi && o2k::k2_han(c)) ? o2k::HL : c;
  bool cs = o2k::cs_of(v, kimi);
  bool lw = o2k::lowish(c);
  bool pm = (c == o2k::P) || (c == o2k::M);
  if (i == 0 || (dstart && dstart[i])) {       // doc start breaks all 4
    headM[i] = 1;
    headCS[i] = cs;
    headLW[i] = lw;
    headPM[i] = pm;
    return;
  }
  int32_t pv = cp[i - 1];
  uint8_t pc = tab[pv];
  uint8_t pmc = o2k::letterish(pc) ? 1
                : (kimi && o2k::k2_han(pc)) ? o2k::HL : pc;
  headM[i] = (pmc != mc);
  headCS[i] = cs && !o2k::cs_of(pv, kimi);
  headLW[i] = lw && !o2k::lowish(pc);
  headPM[i] = pm && !((pc == o2k::P) || (pc == o2k::M));
}

// Channel pass 1: main-class run_start, the cs anchor (firstAnchor),
// and lastM of the pm stream
__global__ void k_o2k_runinfo1(const int32_t* __restrict__ cp,
                               const uint8_t* __restrict__ cls,
                               const uint8_t* __restrict__ headM,
                               const int32_t* __restrict__ ridM,
                               const int32_t* __restrict__ ridCS,
                               const int32_t* __restrict__ ridPM, int n,
                               const int32_t* __restrict__ n_dev,
                               const uint8_t* __restrict__ dstart,
                               int32_t* __restrict__ run_start,
                               int32_t* __restrict__ firstAnchor,
                               int32_t* __restrict__ lastM_pm, bool kimi) {
  if (n_dev) n = *n_dev;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (headM[i]) run_start[ridM[i] - 1] = i;
  int32_t v = cp[i];
  if (o2k::cs_of(v, kimi)) {
    bool anchor = (v == 10 || v == 13) && i > 0 &&
                  !(dstart && dstart[i]) && cls[i - 1] == o2k::P;
    if (anchor) atomicMin(&firstAnchor[ridCS[i] - 1], i);
  }
  if (cls[i] == o2k::M) atomicMax(&lastM_pm[ridPM[i] - 1], i);
}

// Channel pass 2 (consumes the absorbed predicate): fl/lc for S, p_fl
// for P, lastL/lastC for letters, the three firstL levels for lowish,
// and firstPlive for pm
__global__ void k_o2k_runinfo2(const int32_t* __restrict__ cp,
                               const uint8_t* __restrict__ cls,
                               const int32_t* __restrict__ ridM,
                               const int32_t* __restrict__ ridCS,
                               const int32_t* __restrict__ ridLW,
                               const int32_t* __restrict__ ridPM,
                               const int32_t* __restrict__ run_start,
                               const int32_t* __restrict__ firstAnchor,
                               int n, const int32_t* __restrict__ n_dev,
                               int32_t* __restrict__ s_fl,
                               int32_t* __restrict__ s_lc,
                               int32_t* __restrict__ p_fl,
                               int32_t* __restrict__ lastL,
                               int32_t* __restrict__ lastC,
                               int32_t* __restrict__ fL0,
                               int32_t* __restrict__ fL1,
                               int32_t* __restrict__ fL2,
                               int32_t* __restrict__ pm_firstPlive,
                               bool kimi) {
  if (n_dev) n = *n_dev;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int32_t v = cp[i];
  uint8_t c = cls[i];
  bool absorbed = o2k::cs_of(v, kimi) && firstAnchor[ridCS[i] - 1] <= i;
  int r = ridM[i] - 1;
  // Atomics are issued only from "extremum candidate positions" (the
  // head or tail of a streak), which cuts the atomic count from 1-3 per
  // character to 1-2 per streak (~20-30x on English corpora). The
  // semantics are unchanged: the extremum of a min/max necessarily lies
  // in the candidate set (argued per channel in the branch comments).
  auto prev_same_run = [&](int j) { return j >= 0 && ridM[j] == ridM[i]; };
  if (c == o2k::S) {
    if (!absorbed) {
      // s_fl = first non-absorbed S in the run; candidates are those
      // whose predecessor is not a non-absorbed S of the same run
      bool pna = prev_same_run(i - 1) && cls[i - 1] == o2k::S &&
                 !(o2k::cs_of(cp[i - 1], kimi) &&
                   firstAnchor[ridCS[i - 1] - 1] <= i - 1);
      if (!pna) atomicMin(&s_fl[r], i);
      // s_lc = last non-absorbed CRLF in the run; candidates are those
      // whose successor is not a non-absorbed CRLF of the same run
      if (v == 10 || v == 13) {
        bool nna = i + 1 < n && ridM[i + 1] == ridM[i] &&
                   (cp[i + 1] == 10 || cp[i + 1] == 13) &&
                   !(firstAnchor[ridCS[i + 1] - 1] <= i + 1);
        if (!nna) atomicMax(&s_lc[r], i);
      }
    }
  } else if (c == o2k::P) {
    if (!absorbed) {
      bool pna = prev_same_run(i - 1) && cls[i - 1] == o2k::P &&
                 !(o2k::cs_of(cp[i - 1], kimi) &&
                   firstAnchor[ridCS[i - 1] - 1] <= i - 1);
      if (!pna) atomicMin(&p_fl[r], i);
      // first live P of the pm run; candidates are those whose
      // predecessor is not a live P of the same pm run
      bool ppm = i > 0 && ridPM[i - 1] == ridPM[i] &&
                 cls[i - 1] == o2k::P &&
                 !(o2k::cs_of(cp[i - 1], kimi) &&
                   firstAnchor[ridCS[i - 1] - 1] <= i - 1);
      if (!ppm) atomicMin(&pm_firstPlive[ridPM[i] - 1], i);
    }
  }
  if (o2k::letterish(c)) {
    if (c == o2k::L) {
      // lastL = last L in the run; candidates are those whose
      // successor is not an L of the same run
      if (!(i + 1 < n && ridM[i + 1] == ridM[i] && cls[i + 1] == o2k::L))
        atomicMax(&lastL[r], i);
      int d = i - run_start[r];
      int lw = ridLW[i] - 1;
      // fLk = first L with d>=k inside a lowish run; candidates are the
      // head of an L-streak (for any d, the first position in a streak
      // satisfying d>=k is either the streak head or the d==k position)
      // and the d==k position itself
      bool lhead = !(i > 0 && ridLW[i - 1] == lw + 1 &&
                     cls[i - 1] == o2k::L);
      if (lhead) atomicMin(&fL0[lw], i);
      if ((lhead && d >= 1) || d == 1) atomicMin(&fL1[lw], i);
      if ((lhead && d >= 2) || d == 2) atomicMin(&fL2[lw], i);
    }
    if (o2k::clike(c)) {
      if (!(i + 1 < n && ridM[i + 1] == ridM[i] &&
            o2k::clike(cls[i + 1])))
        atomicMax(&lastC[r], i);
    }
  }
}

__global__ void k_o2k_rules(const int32_t* __restrict__ cp,
                            const uint8_t* __restrict__ cls,
                            const int32_t* __restrict__ ridM,
                            const int32_t* __restrict__ ridCS,
                            const int32_t* __restrict__ ridLW,
                            const int32_t* __restrict__ ridPM,
                            const int32_t* __restrict__ run_start,
                            const int32_t* __restrict__ firstAnchor,
                            const int32_t* __restrict__ s_fl,
                            const int32_t* __restrict__ s_lc,
                            const int32_t* __restrict__ p_fl,
                            const int32_t* __restrict__ lastL,
                            const int32_t* __restrict__ lastC,
                            const int32_t* __restrict__ fL0,
                            const int32_t* __restrict__ fL1,
                            const int32_t* __restrict__ fL2,
                            const int32_t* __restrict__ pm_firstPlive,
                            const int32_t* __restrict__ lastM_pm, int R,
                            int dmax, bool contractions,
                            bool* __restrict__ starts,
                            uint8_t* __restrict__ pm_trig,
                            uint8_t* __restrict__ chain_trig,
                            int32_t* __restrict__ chain_flag, int n,
                            const int32_t* __restrict__ n_dev,
                            int32_t* __restrict__ pm_flag,
                            const uint8_t* __restrict__ dstart, bool kimi,
                            uint8_t* __restrict__ hanx_trig) {
  // Graph path: both n and R are taken from the device (the head tail is
  // cleared, so the ridM tail stays at R and ridM[n-1] is the last run
  // number); pm_flag is a scalar flag read back alongside the tcnt sync.
  if (n_dev) {
    n = *n_dev;
    R = ridM[n - 1];
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int32_t v = cp[i];
  uint8_t c = cls[i];
  bool absorbed = o2k::cs_of(v, kimi) && firstAnchor[ridCS[i] - 1] <= i;
  int r = ridM[i] - 1;
  int rs = run_start[r];
  int re = (r + 1 < R) ? run_start[r + 1] : n;
  bool st = false;
  if (kimi && o2k::k2_han(c)) {
    // b0: the head of a Han run always starts a piece (both ends of a
    // pure HL run are closed by the existing rules); HN/HP positions
    // raise a sparse flag, because deciding how the digit phase and the
    // punctuation arm interleave with b0 is inherently sequential and is
    // covered in full by window re-resolution (same approach as the
    // torch reference).
    st = (i == 0) || (dstart && dstart[i]) ||
         !o2k::k2_han(cls[i - 1]);
    if ((c == o2k::HN || c == o2k::HP) && hanx_trig) hanx_trig[i] = 1;
  } else if (c == o2k::N) {
    st = (i - rs) % dmax == 0;
  } else if (c == o2k::S) {
    int fl = s_fl[r];
    if (fl > re) fl = re;                     // rs_eff (all absorbed => re)
    int lc = s_lc[r];
    bool has_nl = lc >= fl;
    int t0 = has_nl ? lc + 1 : fl;
    int m = re - t0;
    // A6's drop-the-last-space needs a character of this doc after the
    // run (a doc boundary counts as EOS)
    st = (has_nl && i == fl) || (m > 0 && i == t0) ||
         (m >= 2 && re < n && !(dstart && dstart[re]) && i == re - 1);
  } else if (c == o2k::P) {
    int pe = p_fl[r];
    if (i == pe && pe < re)
      st = (i == 0) || (dstart && dstart[i]) || cp[i - 1] != 32;
    // A fired candidate (singleton assumption): the apostrophe merges
    // into the preceding piece and does not start one
    if (o2k::cand_k(i, n, cp, cls, contractions, dstart)) st = false;
  } else if (o2k::letterish(c)) {
    int k = (rs > 0 && !(dstart && dstart[rs]))
                ? o2k::cand_k(rs - 1, n, cp, cls, contractions, dstart)
                : 0;
    int eff = rs + k;
    if (i < eff) {
      st = false;                             // letters eaten by suffix
    } else if (i == eff) {
      if (eff < re) {
        if (k > 0) {
          st = true;                          // remainder starts a piece
        } else {
          bool merged = false;
          if (i > 0 && !(dstart && dstart[i])) {
            uint8_t pc = cls[i - 1];
            bool prev_crlf = cp[i - 1] == 10 || cp[i - 1] == 13;
            if (pc == o2k::S && !prev_crlf) {
              merged = true;                  // non-CRLF space merges
            } else if (pc == o2k::P) {
              // merge-forward: the effective head of the previous P run
              // == i-1 (hence effective length 1) and arrival (the
              // character before that is not an ASCII space; a doc start
              // counts as arrival)
              int rp = ridM[i - 1] - 1;
              bool plen1 = p_fl[rp] == i - 1;
              bool arr = (i - 1 == 0) || (dstart && dstart[i - 1]) ||
                         cp[i - 2] != 32;
              merged = plen1 && arr;
            }
          }
          st = !merged;
        }
      }
    } else if (c == o2k::U) {
      // Interior run boundary: U-only, with the previous character in
      // L-only or in the lower role (C/M)
      uint8_t pc = cls[i - 1];
      bool pl = pc == o2k::L;
      if (!pl && o2k::clike(pc)) {
        int j = i - 1;
        int lw = ridLW[j] - 1;
        int flv = (k == 0) ? fL0[lw] : (k == 1 ? fL1[lw] : fL2[lw]);
        bool hasL_le = flv <= j;              // an L prefix >= eff exists
        int lC = lastC[r] >= eff ? lastC[r] : -1;
        int lL = lastL[r] >= eff ? lastL[r] : -1;
        pl = hasL_le || (j == lC && lL < j);
      }
      st = pl;
    }
  }
  if (i == 0 || (dstart && dstart[i])) st = true;   // doc start always
  if (absorbed) st = false;                   // absorbed: never a start
  starts[i] = st;
  // ---- sparse-case flags ----
  // Start of a {P union M} ambiguous span: the pm run contains a live
  // P earlier than some M
  if (c == o2k::P && !absorbed) {
    int pmr = ridPM[i] - 1;
    if (i == pm_firstPlive[pmr] && lastM_pm[pmr] > i) {
      pm_trig[i] = 1;
      if (pm_flag) atomicOr(pm_flag, 1);
    }
  }
  // True chain: candidate i has a fully swallowing link (i+1+k == the
  // end of the following run) that points at another candidate
  int kk = o2k::cand_k(i, n, cp, cls, contractions, dstart);
  if (kk && i + 1 < n) {
    int r2 = ridM[i + 1] - 1;
    int re2 = (r2 + 1 < R) ? run_start[r2 + 1] : n;
    int link = i + 1 + kk;
    if (link == re2 && link < n &&
        o2k::cand_k(link, n, cp, cls, contractions, dstart)) {
      chain_trig[i] = 1;                     // mark link source and target
      chain_trig[link] = 1;                  // redundant 1 writes are safe
      atomicOr(chain_flag, 1);
    }
  }
}

#ifndef TOKTIER_DEVICE_ONLY
static std::vector<torch::Tensor> pretok_starts_o200k(torch::Tensor cp,
                                                      torch::Tensor tab,
                                                      int64_t dmax,
                                                      bool contractions) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(cp.is_cuda() && cp.dtype() == torch::kInt32 &&
              cp.is_contiguous());
  TORCH_CHECK(tab.is_cuda() && tab.dtype() == torch::kUInt8);
  const int64_t n64 = cp.numel();
  TORCH_CHECK(n64 < INT32_MAX);
  const int n = (int)n64;
  auto dev = cp.device();
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto starts = torch::zeros({n64},
                             torch::TensorOptions().dtype(torch::kBool)
                                 .device(dev));
  auto pm_trig = torch::zeros({std::max<int64_t>(n64, 1)}, u8);
  auto chain_trig = torch::zeros({std::max<int64_t>(n64, 1)}, u8);
  auto chain = torch::zeros({1}, i32);
  if (n == 0) return {starts, pm_trig, chain_trig, chain};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nb = (n + TPB - 1) / TPB;
  auto cls = torch::empty({n64}, u8);
  auto headM = torch::empty({n64}, u8);
  auto headCS = torch::empty({n64}, u8);
  auto headLW = torch::empty({n64}, u8);
  auto headPM = torch::empty({n64}, u8);
  k_o2k_heads<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(), n, nullptr, nullptr,
      cls.data_ptr<uint8_t>(), headM.data_ptr<uint8_t>(),
      headCS.data_ptr<uint8_t>(), headLW.data_ptr<uint8_t>(),
      headPM.data_ptr<uint8_t>(), false);
  auto scan = [&](torch::Tensor& h, torch::Tensor& out) {
    auto it = thrust::make_transform_iterator(
        (const uint8_t*)h.data_ptr<uint8_t>(), CastU8{});
    size_t tb = 0;
    cub::DeviceScan::InclusiveSum(nullptr, tb, it,
                                  out.data_ptr<int32_t>(), n, stream);
    auto tmp = torch::empty({(int64_t)tb}, u8);
    cub::DeviceScan::InclusiveSum(tmp.data_ptr<uint8_t>(), tb, it,
                                  out.data_ptr<int32_t>(), n, stream);
  };
  auto ridM = torch::empty({n64}, i32);
  auto ridCS = torch::empty({n64}, i32);
  auto ridLW = torch::empty({n64}, i32);
  auto ridPM = torch::empty({n64}, i32);
  scan(headM, ridM);
  scan(headCS, ridCS);
  scan(headLW, ridLW);
  scan(headPM, ridPM);
  // pack the four run counts into a single D2H
  auto lasts = torch::stack({ridM[n64 - 1], ridCS[n64 - 1],
                             ridLW[n64 - 1], ridPM[n64 - 1]})
                   .cpu();
  const int R = lasts[0].item<int32_t>();
  const int Rcs = std::max<int>(lasts[1].item<int32_t>(), 1);
  const int Rlw = std::max<int>(lasts[2].item<int32_t>(), 1);
  const int Rpm = std::max<int>(lasts[3].item<int32_t>(), 1);
  auto run_start = torch::empty({R}, i32);
  auto firstAnchor = torch::empty({Rcs}, i32);
  auto lastM_pm = torch::empty({Rpm}, i32);
  auto s_fl = torch::empty({R}, i32);
  auto s_lc = torch::empty({R}, i32);
  auto p_fl = torch::empty({R}, i32);
  auto lastL = torch::empty({R}, i32);
  auto lastC = torch::empty({R}, i32);
  auto fL0 = torch::empty({Rlw}, i32);
  auto fL1 = torch::empty({Rlw}, i32);
  auto fL2 = torch::empty({Rlw}, i32);
  auto pm_fp = torch::empty({Rpm}, i32);
  auto fill_hi = [&](torch::Tensor& t) {
    cudaMemsetAsync(t.data_ptr<int32_t>(), 0x7f,
                    (size_t)t.numel() * 4, stream);
  };
  auto fill_neg = [&](torch::Tensor& t) {
    cudaMemsetAsync(t.data_ptr<int32_t>(), 0xff,
                    (size_t)t.numel() * 4, stream);
  };
  fill_hi(firstAnchor);
  fill_neg(lastM_pm);
  fill_hi(s_fl);
  fill_neg(s_lc);
  fill_hi(p_fl);
  fill_neg(lastL);
  fill_neg(lastC);
  fill_hi(fL0);
  fill_hi(fL1);
  fill_hi(fL2);
  fill_hi(pm_fp);
  k_o2k_runinfo1<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      headM.data_ptr<uint8_t>(), ridM.data_ptr<int32_t>(),
      ridCS.data_ptr<int32_t>(), ridPM.data_ptr<int32_t>(), n, nullptr,
      nullptr, run_start.data_ptr<int32_t>(),
      firstAnchor.data_ptr<int32_t>(), lastM_pm.data_ptr<int32_t>(), false);
  k_o2k_runinfo2<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      ridM.data_ptr<int32_t>(), ridCS.data_ptr<int32_t>(),
      ridLW.data_ptr<int32_t>(), ridPM.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), firstAnchor.data_ptr<int32_t>(), n,
      nullptr, s_fl.data_ptr<int32_t>(), s_lc.data_ptr<int32_t>(),
      p_fl.data_ptr<int32_t>(), lastL.data_ptr<int32_t>(),
      lastC.data_ptr<int32_t>(), fL0.data_ptr<int32_t>(),
      fL1.data_ptr<int32_t>(), fL2.data_ptr<int32_t>(),
      pm_fp.data_ptr<int32_t>(), false);
  k_o2k_rules<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      ridM.data_ptr<int32_t>(), ridCS.data_ptr<int32_t>(),
      ridLW.data_ptr<int32_t>(), ridPM.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), firstAnchor.data_ptr<int32_t>(),
      s_fl.data_ptr<int32_t>(), s_lc.data_ptr<int32_t>(),
      p_fl.data_ptr<int32_t>(), lastL.data_ptr<int32_t>(),
      lastC.data_ptr<int32_t>(), fL0.data_ptr<int32_t>(),
      fL1.data_ptr<int32_t>(), fL2.data_ptr<int32_t>(),
      pm_fp.data_ptr<int32_t>(), lastM_pm.data_ptr<int32_t>(), R,
      (int)dmax, contractions, starts.data_ptr<bool>(),
      pm_trig.data_ptr<uint8_t>(), chain_trig.data_ptr<uint8_t>(),
      chain.data_ptr<int32_t>(), n, nullptr, nullptr, nullptr, false,
      nullptr);
  return {starts, pm_trig, chain_trig, chain};
}

// ---------------------------------------------------------------------
// GPU version of the sparse-case window fallback (thread-per-window
// sequential matching). The authoritative semantics are
// pretok_o200k.SeqPretokO200k (_match/_letters/_suffix transcribed line
// by line; pure rules, independent of the vocabulary). Two-phase
// protocol: phase 1 computes the extension interval [lo,hi) for each
// window (in chain mode including the backward search for a safe point)
// and the host selects the set to apply by resolved_until (the applied
// intervals are disjoint, argued as in the host version); phase 2
// re-resolves the selected windows and writes starts directly. The
// device reads global cp directly, so the host version's D2H of window
// contents and its re-fetch when widening a window disappear entirely.
// ---------------------------------------------------------------------
#endif

namespace o2k {
__device__ __forceinline__ uint8_t sq_cls(const int32_t* cp,
                                          const uint8_t* tab, int j) {
  return tab[cp[j]];
}
__device__ __forceinline__ bool sq_upperish(uint8_t c) {  // [Lu Lt Lm Lo M]
  return c == U || c == C || c == M;
}
__device__ __forceinline__ bool sq_lowerish(uint8_t c) {  // [Ll Lm Lo M]
  return c == L || c == C || c == M;
}
__device__ __forceinline__ bool sq_punct4(uint8_t c) {    // [^\s\p{L}\p{N}]
  return c == P || c == M;
}
__device__ __forceinline__ bool sq_prefix_ok(uint8_t c, int32_t v) {
  // [^\r\n\p{L}\p{N}]: \p{L}=U|L|C (M is not \p{L}, so it may prefix)
  return !(v == 10 || v == 13) && c != U && c != L && c != C && c != N;
}
__device__ int sq_suffix(const int32_t* cp, int e, int n, bool contr) {
  if (!contr) return e;
  if (e < n && cp[e] == 39 && e + 1 < n) {
    int f1 = fold(cp[e + 1]);
    if (f1 == 0x73 || f1 == 0x74 || f1 == 0x6D || f1 == 0x64) return e + 2;
    if (e + 2 < n) {
      int f2 = fold(cp[e + 2]);
      if ((f1 == 0x72 && f2 == 0x65) || (f1 == 0x76 && f2 == 0x65) ||
          (f1 == 0x6C && f2 == 0x6C))
        return e + 3;
    }
  }
  return e;
}
__device__ int sq_letters(const int32_t* cp, const uint8_t* tab, int i,
                          int n, bool need_lower, bool contr) {
  for (int pre = 1; pre >= 0; --pre) {
    if (pre && !(sq_prefix_ok(sq_cls(cp, tab, i), cp[i]) && i + 1 <= n - 1))
      continue;
    int j0 = i + pre;
    if (j0 >= n) continue;
    int u = j0;
    while (u < n && sq_upperish(sq_cls(cp, tab, u))) u++;
    int e;
    if (need_lower) {
      int k = -1;
      for (int t = min(u, n - 1); t >= j0; --t)
        if (sq_lowerish(sq_cls(cp, tab, t))) { k = t; break; }
      if (k < 0) continue;
      e = k;
      while (e < n && sq_lowerish(sq_cls(cp, tab, e))) e++;
    } else {
      if (u == j0) continue;
      e = u;
      while (e < n && sq_lowerish(sq_cls(cp, tab, e))) e++;
    }
    return sq_suffix(cp, e, n, contr);
  }
  return -1;
}
__device__ int sq_match(const int32_t* cp, const uint8_t* tab, int i,
                        int n, int dmax, bool contr) {
  int e = sq_letters(cp, tab, i, n, true, contr);   // A1
  if (e >= 0) return e;
  e = sq_letters(cp, tab, i, n, false, contr);      // A2
  if (e >= 0) return e;
  uint8_t c = sq_cls(cp, tab, i);
  if (c == N) {                                     // A3 \p{N}{1,dmax}
    int j = i + 1;
    while (j < n && j - i < dmax && sq_cls(cp, tab, j) == N) j++;
    return j;
  }
  int p = i;                                        // A4 ' '? punct+ crlfsl*
  if (cp[i] == 32 && i + 1 < n && sq_punct4(sq_cls(cp, tab, i + 1)))
    p = i + 1;
  if (p < n && sq_punct4(sq_cls(cp, tab, p))) {
    int j = p + 1;
    while (j < n && sq_punct4(sq_cls(cp, tab, j))) j++;
    while (j < n && (cp[j] == 13 || cp[j] == 10 || cp[j] == 47)) j++;
    return j;
  }
  if (c == S) {                                     // A5/A6/A7
    int j = i + 1;
    while (j < n && sq_cls(cp, tab, j) == S) j++;
    int run_end = j;
    for (int t = run_end - 1; t >= i; --t)
      if (cp[t] == 13 || cp[t] == 10) return t + 1; // A5
    if (run_end == n) return run_end;               // A6 EOS
    if (run_end - i >= 2) return run_end - 1;       // A6 drop last space
    return run_end;                                 // A7
  }
  return i + 1;                                     // defensive, unreachable
}
// ---- kimi sequential matcher (SeqPretokKimi transcribed line by line;
// dmax=3, contractions always on, the absorbed trailer has no slash, and
// b0 takes the leftmost whole Han span) ----
__device__ int sq_letters_k2(const int32_t* cp, const uint8_t* tab, int i,
                             int n, bool need_lower) {
  for (int pre = 1; pre >= 0; --pre) {
    if (pre && !(k2_prefix_ok(sq_cls(cp, tab, i), cp[i]) &&
                 i + 1 <= n - 1))
      continue;
    int j0 = i + pre;
    if (j0 >= n) continue;
    int u = j0;
    while (u < n && sq_upperish(sq_cls(cp, tab, u))) u++;
    int e;
    if (need_lower) {
      int k = -1;
      for (int t = min(u, n - 1); t >= j0; --t)
        if (sq_lowerish(sq_cls(cp, tab, t))) { k = t; break; }
      if (k < 0) continue;
      e = k;
      while (e < n && sq_lowerish(sq_cls(cp, tab, e))) e++;
    } else {
      if (u == j0) continue;
      e = u;
      while (e < n && sq_lowerish(sq_cls(cp, tab, e))) e++;
    }
    return sq_suffix(cp, e, n, true);
  }
  return -1;
}
__device__ int sq_match_k2(const int32_t* cp, const uint8_t* tab, int i,
                           int n) {
  uint8_t c = sq_cls(cp, tab, i);
  if (k2_han(c)) {                                  // b0 [\p{Han}]+ leftmost
    int j = i + 1;
    while (j < n && k2_han(sq_cls(cp, tab, j))) j++;
    return j;
  }
  int e = sq_letters_k2(cp, tab, i, n, true);       // b1
  if (e >= 0) return e;
  e = sq_letters_k2(cp, tab, i, n, false);          // b2
  if (e >= 0) return e;
  if (k2_num(c)) {                                  // b3 (may absorb HN)
    int j = i + 1;
    while (j < n && j - i < 3 && k2_num(sq_cls(cp, tab, j))) j++;
    return j;
  }
  int p = i;                                        // b4 (may absorb HP)
  if (cp[i] == 32 && i + 1 < n && k2_punct4(sq_cls(cp, tab, i + 1)))
    p = i + 1;
  if (p < n && k2_punct4(sq_cls(cp, tab, p))) {
    int j = p + 1;
    while (j < n && k2_punct4(sq_cls(cp, tab, j))) j++;
    while (j < n && (cp[j] == 13 || cp[j] == 10)) j++;   // no slash
    return j;
  }
  if (c == S) {                                     // b5/b6/b7
    int j = i + 1;
    while (j < n && sq_cls(cp, tab, j) == S) j++;
    int run_end = j;
    for (int t = run_end - 1; t >= i; --t)
      if (cp[t] == 13 || cp[t] == 10) return t + 1;
    if (run_end == n) return run_end;
    if (run_end - i >= 2) return run_end - 1;
    return run_end;
  }
  return i + 1;
}
// kimi handoff test (transcribed from _fallback_cpu.handoff): a clean S
// head, an N head (prev not in {N,HN}), or an HL head (prev not Han)
__device__ bool k2_handoff(const int32_t* cp, const uint8_t* tab, int e) {
  uint8_t c = sq_cls(cp, tab, e);
  uint8_t p = sq_cls(cp, tab, e - 1);
  if (c == S) return p != S;
  if (c == N) return p != N && p != HN;
  if (c == HL) return p < HL;
  return false;
}
// Window start, by mode:
//   0 = o200k pm (move one position left if the character before sp is
//       an ASCII space)
//   1 = o200k chain (search backwards from sp-2, at most 4096, for a
//       safe S/N restart point)
//   2 = kimi pm (same space rule as mode 0)
//   3 = kimi anchor search (search backwards from sp, at most 4096, for
//       "doc start or clean S-run head", where clean means
//       not(CRLF and previous character in punct4{P,M,HP}) - the same
//       test as last_clean in the torch reference)
// Returns -1 when nothing is found, in which case the host takes over
// the whole string or batch. ds = start of the containing document
// (0 for a single string).
__device__ int sq_win_lo(const int32_t* cp, const uint8_t* tab, int sp,
                         int mode, int ds) {
  if (mode == 0 || mode == 2)
    return (sp > ds && cp[sp - 1] == 32) ? sp - 1 : sp;
  int wlo = max(sp - 4096, ds);
  if (mode == 3) {
    for (int j = sp; j >= wlo; --j) {
      if (j == ds) return ds;                       // doc start is clean
      if (sq_cls(cp, tab, j) == S) {
        uint8_t pj = sq_cls(cp, tab, j - 1);
        bool head = pj != S;
        bool bad = (cp[j] == 10 || cp[j] == 13) && k2_punct4(pj);
        if (head && !bad) return j;
      }
    }
    return -1;
  }
  for (int j = sp - 2; j >= wlo; --j) {             // mode 1
    uint8_t c = sq_cls(cp, tab, j);
    if (c == S || c == N) {
      int pj = (j > ds) ? (int)sq_cls(cp, tab, j - 1) : -1;
      if (pj != (int)c) {
        if (c == N) return j;
        int32_t v = cp[j];
        if (!((v == 10 || v == 13) && pj == (int)P)) return j;
      }
    }
    if (j == ds) return ds;
  }
  return -1;
}
// Body of the sequential window resolution. Phase 1 (starts=nullptr)
// only computes hi; phase 2 first clears [lo,hi) and then writes the
// piece starts. Handoff: in o200k mode it is an S/N run head (in chain
// mode it must also pass qL+3); in kimi mode it is k2_handoff and must
// pass the trigger position qL (the end_abs > sp test of _fallback_cpu,
// shared by the pm and anchor cases).
__device__ int sq_resolve(const int32_t* cp, const uint8_t* tab, int lo,
                          int qL, int n, int dmax, bool contr,
                          int mode, bool* starts) {
  int i = lo;
  while (true) {
    if (i >= n) return n;
    if (starts) starts[i] = true;
    int j = (mode >= 2) ? sq_match_k2(cp, tab, i, n)
                        : sq_match(cp, tab, i, n, dmax, contr);
    i = j;
    if (i >= n) return n;
    if (mode >= 2) {
      if (i > qL && k2_handoff(cp, tab, i)) return i;
    } else if (mode == 0 || i > qL + 3) {
      uint8_t ce = sq_cls(cp, tab, i);
      if ((ce == S || ce == N) && sq_cls(cp, tab, i - 1) != ce) return i;
    }
  }
}
}  // namespace o2k

__global__ void k_o2k_win_extents(const int32_t* __restrict__ cp,
                                  const uint8_t* __restrict__ tab,
                                  const int32_t* __restrict__ sp,
                                  const int32_t* __restrict__ qL,
                                  const int32_t* __restrict__ ds,
                                  const int32_t* __restrict__ de,
                                  int nwin, int dmax, bool contr,
                                  int mode,
                                  int32_t* __restrict__ lo_out,
                                  int32_t* __restrict__ hi_out,
                                  int32_t* __restrict__ nosafe) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwin) return;
  int lo = o2k::sq_win_lo(cp, tab, sp[w], mode, ds[w]);
  if (lo < 0) {
    atomicOr(nosafe, 1);
    lo_out[w] = -1;
    hi_out[w] = -1;
    return;
  }
  lo_out[w] = lo;
  hi_out[w] = o2k::sq_resolve(cp, tab, lo, qL[w], de[w], dmax, contr,
                              mode, nullptr);
}

// Two application passes, in the same order as the host version: clear
// all windows to false first, then write true. Window intervals may
// overlap: on an overlap the two resolutions necessarily agree, because
// every window start is itself a certain piece start and the matcher is
// memoryless, so the same span yields the same pieces. The order of the
// two passes is guaranteed by the stream, and parallel writes of the
// same value within a pass are harmless.
__global__ void k_o2k_win_clear(const int32_t* __restrict__ lo,
                                const int32_t* __restrict__ hi, int nwin,
                                bool* __restrict__ starts) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwin) return;
  for (int j = lo[w]; j < hi[w]; ++j) starts[j] = false;
}

__global__ void k_o2k_win_mark(const int32_t* __restrict__ cp,
                               const uint8_t* __restrict__ tab,
                               const int32_t* __restrict__ lo,
                               const int32_t* __restrict__ qL,
                               const int32_t* __restrict__ de, int nwin,
                               int dmax, bool contr,
                               int mode, bool* __restrict__ starts) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nwin) return;
  o2k::sq_resolve(cp, tab, lo[w], qL[w], de[w], dmax, contr, mode,
                  starts);
}

#ifndef TOKTIER_DEVICE_ONLY
static std::vector<torch::Tensor> o200k_win_extents(
    torch::Tensor cp, torch::Tensor tab, torch::Tensor sp, torch::Tensor qL,
    torch::Tensor ds, torch::Tensor de, int64_t dmax, bool contractions,
    int64_t mode) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(cp.is_cuda() && cp.dtype() == torch::kInt32 &&
              cp.is_contiguous());
  TORCH_CHECK(sp.is_cuda() && sp.dtype() == torch::kInt32 &&
              qL.dtype() == torch::kInt32 && sp.numel() == qL.numel() &&
              ds.numel() == sp.numel() && de.numel() == sp.numel());
  const int nwin = (int)sp.numel();
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(cp.device());
  auto lo = torch::empty({(int64_t)nwin}, i32);
  auto hi = torch::empty({(int64_t)nwin}, i32);
  auto nosafe = torch::zeros({1}, i32);
  if (nwin == 0) return {lo, hi, nosafe};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nb = (nwin + 63) / 64;
  k_o2k_win_extents<<<nb, 64, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(),
      sp.data_ptr<int32_t>(), qL.data_ptr<int32_t>(),
      ds.data_ptr<int32_t>(), de.data_ptr<int32_t>(), nwin, (int)dmax,
      contractions, (int)mode, lo.data_ptr<int32_t>(),
      hi.data_ptr<int32_t>(), nosafe.data_ptr<int32_t>());
  return {lo, hi, nosafe};
}

static void o200k_win_apply(torch::Tensor cp, torch::Tensor tab,
                            torch::Tensor starts, torch::Tensor lo,
                            torch::Tensor hi, torch::Tensor qL,
                            torch::Tensor de, int64_t dmax,
                            bool contractions, int64_t mode) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(starts.is_cuda() && starts.dtype() == torch::kBool);
  TORCH_CHECK(lo.is_cuda() && lo.dtype() == torch::kInt32 &&
              hi.numel() == lo.numel() && qL.numel() == lo.numel() &&
              de.numel() == lo.numel());
  const int nwin = (int)lo.numel();
  if (nwin == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nb = (nwin + 63) / 64;
  k_o2k_win_clear<<<nb, 64, 0, stream>>>(
      lo.data_ptr<int32_t>(), hi.data_ptr<int32_t>(), nwin,
      starts.data_ptr<bool>());
  k_o2k_win_mark<<<nb, 64, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(),
      lo.data_ptr<int32_t>(), qL.data_ptr<int32_t>(),
      de.data_ptr<int32_t>(), nwin, (int)dmax, contractions, (int)mode,
      starts.data_ptr<bool>());
}

// o200k batched entry (runs never cross a document). Structurally the
// same as pretok_starts_o200k with one extra dstart channel (it breaks
// all head streams and guards the document boundary in the anchor,
// contraction and look-back rules; see the comments inside each kernel).
// The sparse-case flag semantics are unchanged, and window resolution is
// issued by the Python layer per document boundary (the ds/de arguments
// of o200k_win_*).
static std::vector<torch::Tensor> pretok_starts_batched_o200k(
    torch::Tensor cp, torch::Tensor dstart, torch::Tensor tab,
    int64_t dmax, bool contractions) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(cp.is_cuda() && cp.dtype() == torch::kInt32 &&
              cp.is_contiguous());
  TORCH_CHECK(dstart.is_cuda() && dstart.dtype() == torch::kUInt8 &&
              dstart.numel() == cp.numel() && dstart.is_contiguous());
  TORCH_CHECK(tab.is_cuda() && tab.dtype() == torch::kUInt8);
  const int64_t n64 = cp.numel();
  TORCH_CHECK(n64 < INT32_MAX);
  const int n = (int)n64;
  auto dev = cp.device();
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto starts = torch::zeros({n64},
                             torch::TensorOptions().dtype(torch::kBool)
                                 .device(dev));
  auto pm_trig = torch::zeros({std::max<int64_t>(n64, 1)}, u8);
  auto chain_trig = torch::zeros({std::max<int64_t>(n64, 1)}, u8);
  auto chain = torch::zeros({1}, i32);
  if (n == 0) return {starts, pm_trig, chain_trig, chain};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nb = (n + TPB - 1) / TPB;
  const uint8_t* dsp = dstart.data_ptr<uint8_t>();
  auto cls = torch::empty({n64}, u8);
  auto headM = torch::empty({n64}, u8);
  auto headCS = torch::empty({n64}, u8);
  auto headLW = torch::empty({n64}, u8);
  auto headPM = torch::empty({n64}, u8);
  k_o2k_heads<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(), n, nullptr, dsp,
      cls.data_ptr<uint8_t>(), headM.data_ptr<uint8_t>(),
      headCS.data_ptr<uint8_t>(), headLW.data_ptr<uint8_t>(),
      headPM.data_ptr<uint8_t>(), false);
  auto scan = [&](torch::Tensor& h, torch::Tensor& out) {
    auto it = thrust::make_transform_iterator(
        (const uint8_t*)h.data_ptr<uint8_t>(), CastU8{});
    size_t tb = 0;
    cub::DeviceScan::InclusiveSum(nullptr, tb, it,
                                  out.data_ptr<int32_t>(), n, stream);
    auto tmp = torch::empty({(int64_t)tb}, u8);
    cub::DeviceScan::InclusiveSum(tmp.data_ptr<uint8_t>(), tb, it,
                                  out.data_ptr<int32_t>(), n, stream);
  };
  auto ridM = torch::empty({n64}, i32);
  auto ridCS = torch::empty({n64}, i32);
  auto ridLW = torch::empty({n64}, i32);
  auto ridPM = torch::empty({n64}, i32);
  scan(headM, ridM);
  scan(headCS, ridCS);
  scan(headLW, ridLW);
  scan(headPM, ridPM);
  auto lasts = torch::stack({ridM[n64 - 1], ridCS[n64 - 1],
                             ridLW[n64 - 1], ridPM[n64 - 1]})
                   .cpu();
  const int R = lasts[0].item<int32_t>();
  const int Rcs = std::max<int>(lasts[1].item<int32_t>(), 1);
  const int Rlw = std::max<int>(lasts[2].item<int32_t>(), 1);
  const int Rpm = std::max<int>(lasts[3].item<int32_t>(), 1);
  auto run_start = torch::empty({R}, i32);
  auto firstAnchor = torch::empty({Rcs}, i32);
  auto lastM_pm = torch::empty({Rpm}, i32);
  auto s_fl = torch::empty({R}, i32);
  auto s_lc = torch::empty({R}, i32);
  auto p_fl = torch::empty({R}, i32);
  auto lastL = torch::empty({R}, i32);
  auto lastC = torch::empty({R}, i32);
  auto fL0 = torch::empty({Rlw}, i32);
  auto fL1 = torch::empty({Rlw}, i32);
  auto fL2 = torch::empty({Rlw}, i32);
  auto pm_fp = torch::empty({Rpm}, i32);
  auto fill_hi = [&](torch::Tensor& t) {
    cudaMemsetAsync(t.data_ptr<int32_t>(), 0x7f,
                    (size_t)t.numel() * 4, stream);
  };
  auto fill_neg = [&](torch::Tensor& t) {
    cudaMemsetAsync(t.data_ptr<int32_t>(), 0xff,
                    (size_t)t.numel() * 4, stream);
  };
  fill_hi(firstAnchor);
  fill_neg(lastM_pm);
  fill_hi(s_fl);
  fill_neg(s_lc);
  fill_hi(p_fl);
  fill_neg(lastL);
  fill_neg(lastC);
  fill_hi(fL0);
  fill_hi(fL1);
  fill_hi(fL2);
  fill_hi(pm_fp);
  k_o2k_runinfo1<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      headM.data_ptr<uint8_t>(), ridM.data_ptr<int32_t>(),
      ridCS.data_ptr<int32_t>(), ridPM.data_ptr<int32_t>(), n, nullptr,
      dsp, run_start.data_ptr<int32_t>(), firstAnchor.data_ptr<int32_t>(),
      lastM_pm.data_ptr<int32_t>(), false);
  k_o2k_runinfo2<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      ridM.data_ptr<int32_t>(), ridCS.data_ptr<int32_t>(),
      ridLW.data_ptr<int32_t>(), ridPM.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), firstAnchor.data_ptr<int32_t>(), n,
      nullptr, s_fl.data_ptr<int32_t>(), s_lc.data_ptr<int32_t>(),
      p_fl.data_ptr<int32_t>(), lastL.data_ptr<int32_t>(),
      lastC.data_ptr<int32_t>(), fL0.data_ptr<int32_t>(),
      fL1.data_ptr<int32_t>(), fL2.data_ptr<int32_t>(),
      pm_fp.data_ptr<int32_t>(), false);
  k_o2k_rules<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      ridM.data_ptr<int32_t>(), ridCS.data_ptr<int32_t>(),
      ridLW.data_ptr<int32_t>(), ridPM.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), firstAnchor.data_ptr<int32_t>(),
      s_fl.data_ptr<int32_t>(), s_lc.data_ptr<int32_t>(),
      p_fl.data_ptr<int32_t>(), lastL.data_ptr<int32_t>(),
      lastC.data_ptr<int32_t>(), fL0.data_ptr<int32_t>(),
      fL1.data_ptr<int32_t>(), fL2.data_ptr<int32_t>(),
      pm_fp.data_ptr<int32_t>(), lastM_pm.data_ptr<int32_t>(), R,
      (int)dmax, contractions, starts.data_ptr<bool>(),
      pm_trig.data_ptr<uint8_t>(), chain_trig.data_ptr<uint8_t>(),
      chain.data_ptr<int32_t>(), n, nullptr, nullptr, dsp, false,
      nullptr);
  return {starts, pm_trig, chain_trig, chain};
}

// kimi_k3 starts for a single string (selected by a runtime flag: the
// kimi=true branch of the three kernels plus the hanx sparse flag).
// dmax=3, contractions always on, cs=crlf; the sparse cases
// (pm/hanx/chain) are re-resolved by the Python layer through
// o200k_win_* in mode 2/3.
static std::vector<torch::Tensor> pretok_starts_kimi(torch::Tensor cp,
                                                     torch::Tensor tab) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(cp));
  TORCH_CHECK(cp.is_cuda() && cp.dtype() == torch::kInt32 &&
              cp.is_contiguous());
  TORCH_CHECK(tab.is_cuda() && tab.dtype() == torch::kUInt8);
  const int64_t n64 = cp.numel();
  TORCH_CHECK(n64 < INT32_MAX);
  const int n = (int)n64;
  auto dev = cp.device();
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto starts = torch::zeros({n64},
                             torch::TensorOptions().dtype(torch::kBool)
                                 .device(dev));
  auto pm_trig = torch::zeros({std::max<int64_t>(n64, 1)}, u8);
  auto chain_trig = torch::zeros({std::max<int64_t>(n64, 1)}, u8);
  auto hanx_trig = torch::zeros({std::max<int64_t>(n64, 1)}, u8);
  auto chain = torch::zeros({1}, i32);
  if (n == 0) return {starts, pm_trig, chain_trig, hanx_trig, chain};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int nb = (n + TPB - 1) / TPB;
  auto cls = torch::empty({n64}, u8);
  auto headM = torch::empty({n64}, u8);
  auto headCS = torch::empty({n64}, u8);
  auto headLW = torch::empty({n64}, u8);
  auto headPM = torch::empty({n64}, u8);
  k_o2k_heads<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(), n, nullptr, nullptr,
      cls.data_ptr<uint8_t>(), headM.data_ptr<uint8_t>(),
      headCS.data_ptr<uint8_t>(), headLW.data_ptr<uint8_t>(),
      headPM.data_ptr<uint8_t>(), true);
  auto scan = [&](torch::Tensor& h, torch::Tensor& out) {
    auto it = thrust::make_transform_iterator(
        (const uint8_t*)h.data_ptr<uint8_t>(), CastU8{});
    size_t tb = 0;
    cub::DeviceScan::InclusiveSum(nullptr, tb, it,
                                  out.data_ptr<int32_t>(), n, stream);
    auto tmp = torch::empty({(int64_t)tb}, u8);
    cub::DeviceScan::InclusiveSum(tmp.data_ptr<uint8_t>(), tb, it,
                                  out.data_ptr<int32_t>(), n, stream);
  };
  auto ridM = torch::empty({n64}, i32);
  auto ridCS = torch::empty({n64}, i32);
  auto ridLW = torch::empty({n64}, i32);
  auto ridPM = torch::empty({n64}, i32);
  scan(headM, ridM);
  scan(headCS, ridCS);
  scan(headLW, ridLW);
  scan(headPM, ridPM);
  auto lasts = torch::stack({ridM[n64 - 1], ridCS[n64 - 1],
                             ridLW[n64 - 1], ridPM[n64 - 1]})
                   .cpu();
  const int R = lasts[0].item<int32_t>();
  const int Rcs = std::max<int>(lasts[1].item<int32_t>(), 1);
  const int Rlw = std::max<int>(lasts[2].item<int32_t>(), 1);
  const int Rpm = std::max<int>(lasts[3].item<int32_t>(), 1);
  auto run_start = torch::empty({R}, i32);
  auto firstAnchor = torch::empty({Rcs}, i32);
  auto lastM_pm = torch::empty({Rpm}, i32);
  auto s_fl = torch::empty({R}, i32);
  auto s_lc = torch::empty({R}, i32);
  auto p_fl = torch::empty({R}, i32);
  auto lastL = torch::empty({R}, i32);
  auto lastC = torch::empty({R}, i32);
  auto fL0 = torch::empty({Rlw}, i32);
  auto fL1 = torch::empty({Rlw}, i32);
  auto fL2 = torch::empty({Rlw}, i32);
  auto pm_fp = torch::empty({Rpm}, i32);
  auto fill_hi = [&](torch::Tensor& t) {
    cudaMemsetAsync(t.data_ptr<int32_t>(), 0x7f,
                    (size_t)t.numel() * 4, stream);
  };
  auto fill_neg = [&](torch::Tensor& t) {
    cudaMemsetAsync(t.data_ptr<int32_t>(), 0xff,
                    (size_t)t.numel() * 4, stream);
  };
  fill_hi(firstAnchor);
  fill_neg(lastM_pm);
  fill_hi(s_fl);
  fill_neg(s_lc);
  fill_hi(p_fl);
  fill_neg(lastL);
  fill_neg(lastC);
  fill_hi(fL0);
  fill_hi(fL1);
  fill_hi(fL2);
  fill_hi(pm_fp);
  k_o2k_runinfo1<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      headM.data_ptr<uint8_t>(), ridM.data_ptr<int32_t>(),
      ridCS.data_ptr<int32_t>(), ridPM.data_ptr<int32_t>(), n, nullptr,
      nullptr, run_start.data_ptr<int32_t>(),
      firstAnchor.data_ptr<int32_t>(), lastM_pm.data_ptr<int32_t>(), true);
  k_o2k_runinfo2<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      ridM.data_ptr<int32_t>(), ridCS.data_ptr<int32_t>(),
      ridLW.data_ptr<int32_t>(), ridPM.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), firstAnchor.data_ptr<int32_t>(), n,
      nullptr, s_fl.data_ptr<int32_t>(), s_lc.data_ptr<int32_t>(),
      p_fl.data_ptr<int32_t>(), lastL.data_ptr<int32_t>(),
      lastC.data_ptr<int32_t>(), fL0.data_ptr<int32_t>(),
      fL1.data_ptr<int32_t>(), fL2.data_ptr<int32_t>(),
      pm_fp.data_ptr<int32_t>(), true);
  k_o2k_rules<<<nb, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(),
      ridM.data_ptr<int32_t>(), ridCS.data_ptr<int32_t>(),
      ridLW.data_ptr<int32_t>(), ridPM.data_ptr<int32_t>(),
      run_start.data_ptr<int32_t>(), firstAnchor.data_ptr<int32_t>(),
      s_fl.data_ptr<int32_t>(), s_lc.data_ptr<int32_t>(),
      p_fl.data_ptr<int32_t>(), lastL.data_ptr<int32_t>(),
      lastC.data_ptr<int32_t>(), fL0.data_ptr<int32_t>(),
      fL1.data_ptr<int32_t>(), fL2.data_ptr<int32_t>(),
      pm_fp.data_ptr<int32_t>(), lastM_pm.data_ptr<int32_t>(), R,
      3, true, starts.data_ptr<bool>(),
      pm_trig.data_ptr<uint8_t>(), chain_trig.data_ptr<uint8_t>(),
      chain.data_ptr<int32_t>(), n, nullptr, nullptr, nullptr, true,
      hanx_trig.data_ptr<uint8_t>());
  return {starts, pm_trig, chain_trig, hanx_trig, chain};
}

// Pack three metadata values {token count, chain flag, pm flag} so that
// a single 12B D2H reads back tcnt together with the sparse-case flags.
#endif

__global__ void k_o2k_meta3(const int32_t* __restrict__ tcnt,
                            const int32_t* __restrict__ flags2,
                            int32_t* __restrict__ meta) {
  meta[0] = *tcnt;
  meta[1] = flags2[0];
  meta[2] = flags2[1];
}

// Fully fused single-request o200k path (CUDA-Graph capturable).
// Structure = the decode/dispatch/BPE skeleton of encode_fused_t with
// the o2k four-stream middle section. The three obstacles to graph
// capture are handled as follows: R is never read back (the channels are
// preallocated at the cap bound and the kernel uses R=ridM[n-1]); the
// chain and pm flags are folded into meta and read back once; and when a
// flag fires, the Python layer redoes the work on the eager path, so the
// sparse cases never enter the graph.
#ifndef TOKTIER_DEVICE_ONLY
static std::vector<torch::Tensor> encode_fused_o200k_impl(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, bool contractions, torch::Tensor pair_keys,
    torch::Tensor pair_vals, torch::Tensor byte_id,
    torch::Tensor vocab_keys, torch::Tensor vocab_vals,
    torch::Tensor vocab_blob, int64_t ignore_merges,
    torch::Tensor unsafe_bits) {
  const c10::cuda::OptionalCUDAGuard guard(at::device_of(bytes));
  TORCH_CHECK(bytes.is_cuda() && bytes.dtype() == torch::kUInt8 &&
              bytes.is_contiguous());
  TORCH_CHECK(nb_dev.is_cuda() && nb_dev.dtype() == torch::kInt32 &&
              nb_dev.numel() == 1);
  TORCH_CHECK(nb_dev.device() == bytes.device(),
              "nb_dev must live on bytes' device");
  TORCH_CHECK(tab.is_cuda() && tab.device() == bytes.device() &&
              tab.dtype() == torch::kUInt8 && tab.is_contiguous() &&
              tab.numel() >= 0x110000,
              "tab must be a contiguous uint8 CUDA table covering U+10FFFF");
  check_bpe_tables_meta(bytes, pair_keys, pair_vals, byte_id,
                        vocab_keys, vocab_vals, vocab_blob);
  const int64_t cap64 = bytes.numel();
  TORCH_CHECK(cap64 > 0 && cap64 < INT32_MAX);
  const int cap = (int)cap64;
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32)
                      .device(bytes.device());
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8)
                     .device(bytes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int gsB = (cap + TPB - 1) / TPB;

  // CUB workspace (as in encode_fused_t: take the maximum temporary
  // size over all algorithms and share one buffer)
  auto lead_it = thrust::make_transform_iterator(
      (const uint8_t*)bytes.data_ptr<uint8_t>(), IsLead{});
  auto head_it = thrust::make_transform_iterator(
      (const uint8_t*)nullptr, CastU8{});
  thrust::counting_iterator<int32_t> cnt_it(0);
  size_t tmax = 0, tq = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tq, lead_it, (int32_t*)nullptr,
                                cap, stream);
  tmax = std::max(tmax, tq); tq = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tq, head_it, (int32_t*)nullptr,
                                cap, stream);
  tmax = std::max(tmax, tq); tq = 0;
  cub::DeviceScan::InclusiveSum(nullptr, tq, (const int32_t*)nullptr,
                                (int32_t*)nullptr, cap, stream);
  tmax = std::max(tmax, tq); tq = 0;
  cub::DeviceSelect::Flagged(nullptr, tq, (const int32_t*)nullptr,
                             (const bool*)nullptr, (int32_t*)nullptr,
                             (int32_t*)nullptr, cap, stream);
  tmax = std::max(tmax, tq); tq = 0;
  cub::DeviceSelect::Flagged(nullptr, tq, cnt_it, (const uint8_t*)nullptr,
                             (int32_t*)nullptr, (int32_t*)nullptr,
                             cap, stream);
  tmax = std::max(tmax, tq);
  auto tmp = torch::empty({(int64_t)tmax}, opts_u8);
  void* tp = tmp.data_ptr();

  // ---- UTF-8 decode (padding = continuation bytes, no chars) ----
  auto cpos = torch::empty({cap64}, opts_i32);
  cub::DeviceScan::InclusiveSum(tp, tmax, lead_it,
                                cpos.data_ptr<int32_t>(), cap, stream);
  auto cp = torch::empty({cap64}, opts_i32);
  auto bo = torch::empty({cap64}, opts_i32);
  k_utf8_decode<<<gsB, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), cpos.data_ptr<int32_t>(),
      cp.data_ptr<int32_t>(), bo.data_ptr<int32_t>(), cap, nullptr);
  const int32_t* d_C = cpos.data_ptr<int32_t>() + (cap - 1);

  // ---- o2k four-stream segmentation (the head tails are cleared, so
  // the four rid scans stay constant past the end) ----
  auto cls = torch::empty({cap64}, opts_u8);
  auto heads4 = torch::empty({4 * cap64}, opts_u8);
  uint8_t* hM = heads4.data_ptr<uint8_t>();
  cudaMemsetAsync(hM, 0, (size_t)4 * cap, stream);
  k_o2k_heads<<<gsB, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), tab.data_ptr<uint8_t>(), 0, d_C, nullptr,
      cls.data_ptr<uint8_t>(), hM, hM + cap, hM + 2 * cap, hM + 3 * cap,
      false);
  auto rids4 = torch::empty({4 * cap64}, opts_i32);
  int32_t* rM = rids4.data_ptr<int32_t>();
  for (int k = 0; k < 4; ++k) {
    auto it = thrust::make_transform_iterator(
        (const uint8_t*)(hM + (size_t)k * cap), CastU8{});
    cub::DeviceScan::InclusiveSum(tp, tmax, it, rM + (size_t)k * cap,
                                  cap, stream);
  }
  int32_t* ridM = rM;
  int32_t* ridCS = rM + cap;
  int32_t* ridLW = rM + 2 * (size_t)cap;
  int32_t* ridPM = rM + 3 * (size_t)cap;

  // ---- per-run channels (preallocated at the cap bound; two blocks
  // with aggregated init: hi=0x7f / neg=0xff)
  auto run_start = torch::empty({cap64}, opts_i32);
  auto chan_hi = torch::empty({7 * cap64}, opts_i32);
  auto chan_neg = torch::empty({4 * cap64}, opts_i32);
  int32_t* pHi = chan_hi.data_ptr<int32_t>();
  int32_t* pNeg = chan_neg.data_ptr<int32_t>();
  cudaMemsetAsync(pHi, 0x7f, (size_t)7 * cap * 4, stream);
  cudaMemsetAsync(pNeg, 0xff, (size_t)4 * cap * 4, stream);
  int32_t* firstAnchor = pHi;
  int32_t* s_fl = pHi + cap;
  int32_t* p_fl = pHi + 2 * (size_t)cap;
  int32_t* fL0 = pHi + 3 * (size_t)cap;
  int32_t* fL1 = pHi + 4 * (size_t)cap;
  int32_t* fL2 = pHi + 5 * (size_t)cap;
  int32_t* pm_fp = pHi + 6 * (size_t)cap;
  int32_t* lastM_pm = pNeg;
  int32_t* s_lc = pNeg + cap;
  int32_t* lastL = pNeg + 2 * (size_t)cap;
  int32_t* lastC = pNeg + 3 * (size_t)cap;
  k_o2k_runinfo1<<<gsB, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(), hM, ridM, ridCS,
      ridPM, 0, d_C, nullptr, run_start.data_ptr<int32_t>(), firstAnchor,
      lastM_pm, false);
  k_o2k_runinfo2<<<gsB, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(), ridM, ridCS, ridLW,
      ridPM, run_start.data_ptr<int32_t>(), firstAnchor, 0, d_C, s_fl,
      s_lc, p_fl, lastL, lastC, fL0, fL1, fL2, pm_fp, false);

  // ---- rules (starts plus sparse-case flags; the trig arrays are only
  // write targets, the readback goes through meta) ----
  auto starts = torch::empty({cap64}, torch::TensorOptions()
                                          .dtype(torch::kBool)
                                          .device(bytes.device()));
  cudaMemsetAsync(starts.data_ptr<bool>(), 0, cap, stream);
  auto trig2 = torch::empty({2 * cap64}, opts_u8);
  uint8_t* pm_trig = trig2.data_ptr<uint8_t>();
  cudaMemsetAsync(pm_trig, 0, (size_t)2 * cap, stream);
  auto flags2 = torch::empty({2}, opts_i32);
  cudaMemsetAsync(flags2.data_ptr<int32_t>(), 0, 8, stream);
  k_o2k_rules<<<gsB, TPB, 0, stream>>>(
      cp.data_ptr<int32_t>(), cls.data_ptr<uint8_t>(), ridM, ridCS, ridLW,
      ridPM, run_start.data_ptr<int32_t>(), firstAnchor, s_fl, s_lc, p_fl,
      lastL, lastC, fL0, fL1, fL2, pm_fp, lastM_pm, 0, (int)dmax,
      contractions, starts.data_ptr<bool>(), pm_trig, pm_trig + cap,
      flags2.data_ptr<int32_t>(), 0, d_C,
      flags2.data_ptr<int32_t>() + 1, nullptr, false, nullptr);

  // ---- piece bounds and dispatch (as in encode_fused_t) ----
  auto pb = torch::empty({cap64 + 1}, opts_i32);
  auto d_cnts = torch::empty({4}, opts_i32);   // [P, nS, nM, nL]
  int32_t* d_P = d_cnts.data_ptr<int32_t>();
  cub::DeviceSelect::Flagged(tp, tmax, bo.data_ptr<int32_t>(),
                             starts.data_ptr<bool>(), pb.data_ptr<int32_t>(),
                             d_P, cap, stream);
  k_pb_sentinel<<<1, 1, 0, stream>>>(pb.data_ptr<int32_t>(), d_P,
                                     nb_dev.data_ptr<int32_t>());
  auto flags = torch::empty({3 * cap64}, opts_u8);
  uint8_t* fS = flags.data_ptr<uint8_t>();
  auto cnt = torch::empty({cap64}, opts_i32);
  cudaMemsetAsync(cnt.data_ptr<int32_t>(), 0, (size_t)cap * 4, stream);
  k_dispatch_flags<<<gsB, TPB, 0, stream>>>(
      pb.data_ptr<int32_t>(), d_P, fS, fS + cap, fS + 2 * cap,
      cnt.data_ptr<int32_t>(), cap);
  auto lists = torch::empty({3 * cap64}, opts_i32);
  int32_t* lS = lists.data_ptr<int32_t>();
  cub::DeviceSelect::Flagged(tp, tmax, cnt_it, fS, lS,
                             d_P + 1, cap, stream);
  cub::DeviceSelect::Flagged(tp, tmax, cnt_it, fS + cap, lS + cap,
                             d_P + 2, cap, stream);
  cub::DeviceSelect::Flagged(tp, tmax, cnt_it, fS + 2 * cap, lS + 2 * cap,
                             d_P + 3, cap, stream);

  // ---- BPE (geometry launched at the capacity bound; out-of-range
  // threads exit or stride according to the device counts) ----
  auto scratch = torch::empty({cap64}, opts_i32);
  auto bufB = torch::empty({cap64}, opts_i32);
  unsigned pmask = (unsigned)pair_keys.numel() - 1;
  unsigned vmask = (unsigned)vocab_keys.numel() - 1;
  const uint32_t* ubp = unsafe_bits_ptr(unsafe_bits, bytes);  // guard bits
  k_bpe_thread<SHORT_MAX><<<gsB, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(), lS, 0,
      (const uint64_t*)pair_keys.data_ptr(),
      (const uint64_t*)pair_vals.data_ptr(), pmask,
      byte_id.data_ptr<int32_t>(),
      (const uint64_t*)vocab_keys.data_ptr(),
      (const uint64_t*)vocab_vals.data_ptr(), vmask,
      vocab_blob.data_ptr<uint8_t>(), (int)ignore_merges,
      scratch.data_ptr<int32_t>(), cnt.data_ptr<int32_t>(), d_P + 1,
      nullptr, nullptr, nullptr, nullptr, 0u);
  const int gw = std::min(gsB, 8192);
  k_bpe_warp<<<gw, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(), lS + cap, 0,
      d_P + 2,
      (const uint64_t*)pair_keys.data_ptr(),
      (const uint64_t*)pair_vals.data_ptr(), pmask,
      byte_id.data_ptr<int32_t>(),
      (const uint64_t*)vocab_keys.data_ptr(),
      (const uint64_t*)vocab_vals.data_ptr(), vmask,
      vocab_blob.data_ptr<uint8_t>(), (int)ignore_merges,
      scratch.data_ptr<int32_t>(), cnt.data_ptr<int32_t>(),
      ubp);
  const int gl = std::min(cap / (MED_MAX + 1) + 1, 8192);
  k_bpe_long<<<gl, TPB, 0, stream>>>(
      bytes.data_ptr<uint8_t>(), pb.data_ptr<int32_t>(), lS + 2 * cap,
      (const uint64_t*)pair_keys.data_ptr(),
      (const uint64_t*)pair_vals.data_ptr(), pmask,
      byte_id.data_ptr<int32_t>(),
      (const uint64_t*)vocab_keys.data_ptr(),
      (const uint64_t*)vocab_vals.data_ptr(), vmask,
      vocab_blob.data_ptr<uint8_t>(), (int)ignore_merges,
      scratch.data_ptr<int32_t>(), bufB.data_ptr<int32_t>(),
      cnt.data_ptr<int32_t>(), 0, d_P + 3, ubp);

  // ---- prefix sum + compaction + meta packing (single 12B readback) ----
  auto off = torch::zeros({cap64 + 1}, opts_i32);
  cub::DeviceScan::InclusiveSum(tp, tmax, cnt.data_ptr<int32_t>(),
                                off.data_ptr<int32_t>() + 1, cap, stream);
  auto out = torch::empty({cap64}, opts_i32);
  k_bpe_compact<<<std::min(cap, 65535), 64, 0, stream>>>(
      pb.data_ptr<int32_t>(), off.data_ptr<int32_t>(),
      scratch.data_ptr<int32_t>(), out.data_ptr<int32_t>(), 0, d_P);
  auto meta = torch::empty({3}, opts_i32);
  k_o2k_meta3<<<1, 1, 0, stream>>>(off.data_ptr<int32_t>() + cap,
                                   flags2.data_ptr<int32_t>(),
                                   meta.data_ptr<int32_t>());
  return {out, meta};
}

// Old and new signature wrappers for the fused o200k entry (the old
// 12-argument form passes an empty bitmap and is bit-identical; the
// 13-argument form carries the guard bitmap)
static std::vector<torch::Tensor> encode_fused_o200k(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, bool contractions, torch::Tensor pair_keys,
    torch::Tensor pair_vals, torch::Tensor byte_id,
    torch::Tensor vocab_keys, torch::Tensor vocab_vals,
    torch::Tensor vocab_blob, int64_t ignore_merges) {
  return encode_fused_o200k_impl(bytes, nb_dev, tab, dmax, contractions,
                                 pair_keys, pair_vals, byte_id, vocab_keys,
                                 vocab_vals, vocab_blob, ignore_merges,
                                 empty_ub(bytes));
}

static std::vector<torch::Tensor> encode_fused_o200k_v2(
    torch::Tensor bytes, torch::Tensor nb_dev, torch::Tensor tab,
    int64_t dmax, bool contractions, torch::Tensor pair_keys,
    torch::Tensor pair_vals, torch::Tensor byte_id,
    torch::Tensor vocab_keys, torch::Tensor vocab_vals,
    torch::Tensor vocab_blob, int64_t ignore_merges,
    torch::Tensor unsafe_bits) {
  return encode_fused_o200k_impl(bytes, nb_dev, tab, dmax, contractions,
                                 pair_keys, pair_vals, byte_id, vocab_keys,
                                 vocab_vals, vocab_blob, ignore_merges,
                                 unsafe_bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pretok_starts", &pretok_starts,
        "fused GPT-style pre-tokenization piece-start kernel");
  m.def("pretok_starts_batched", &pretok_starts_batched,
        "multi-document batched variant (runs cut at doc boundaries)");
  m.def("pretok_starts_kimi", &pretok_starts_kimi,
        "kimi_k3 starts (kimi flag branches; sparse faces via o200k_win_* "
        "mode 2/3)");
  m.def("pretok_starts_batched_o200k", &pretok_starts_batched_o200k,
        "o200k batched starts (dstart channel, runs never cross docs)");
  m.def("o200k_win_extents", &o200k_win_extents,
        "phase 1: per-window [lo,hi) extents (+nosafe flag, chain mode)");
  m.def("o200k_win_apply", &o200k_win_apply,
        "phase 2: clear+mark selected windows into starts");
  m.def("encode_fused_o200k", &encode_fused_o200k,
        "o200k fused bytes->ids (CUDA-Graph capturable); returns "
        "{ids, meta[tcnt, chain_flag, pm_flag]}");
  m.def("pretok_starts_o200k", &pretok_starts_o200k,
        "o200k splitter piece-start kernel (returns starts, "
        "pm-ambiguity span flags, contraction-chain flag)");
  m.def("pretok_starts_laguna", &pretok_starts_laguna,
        "Laguna-S-2.1 starts (stage-0 newline-run cut points "
        "OR-ed into the B/dso channel + qwen3 rule body via RS_CL100K "
        "else-branches)");
  m.def("pretok_starts_batched_laguna", &pretok_starts_batched_laguna,
        "Laguna batched variant (doc boundaries OR-ed into stage-0 mask B)");
  m.def("encode_fused_laguna", &encode_fused_laguna,
        "encode_fused with Laguna ruleset (CUDA-Graph capturable)");
  m.def("pretok_starts_ds", &pretok_starts_ds,
        "DeepSeek 3-splitter (single-pass) piece-start kernel");
  m.def("pretok_starts_batched_ds", &pretok_starts_batched_ds,
        "DeepSeek batched variant (doc boundaries OR-ed into stage mask B)");
  m.def("encode_fused_ds", &encode_fused_ds,
        "encode_fused with DeepSeek ruleset (CUDA-Graph capturable)");
  m.def("ds_constants", &ds_constants,
        "DeepSeek inline constants self-description (JSON, for meta cross-"
        "check against mechanically extracted values)");
  m.def("utf8_to_cp", &utf8_to_cp,
        "GPU UTF-8 decode: byte tensor -> codepoint tensor");
  m.def("utf8_to_cp_bo", &utf8_to_cp_bo,
        "GPU UTF-8 decode returning (codepoints, byte offset of each char)");
  m.def("bpe_encode_memo", &bpe_encode_memo,
        "bpe_encode with the per-piece memoization table");
  m.def("bpe_encode", &bpe_encode,
        "per-piece byte-level BPE merge -> token ids (+ per-piece offsets)");
  m.def("encode_fused", &encode_fused,
        "fully-fused single-request bytes -> token ids; zero host sync, "
        "CUDA-Graph capturable (returns ids buffer + 1-elem token count)");
  m.def("nfc_qc_scan", &nfc_qc_scan,
        "NFC quick-check single-pass scan (uint8 bytes + full-plane "
        "QC table -> 1-elem int32 flag; zero host sync, graph capturable)");
  // Overloads carrying the non-monotone merge-table guard bitmap
  // (registering the same name twice gives a pybind overload; calls with
  // the old argument count resolve to the original registration above
  // and are bit-identical)
  m.def("bpe_encode", &bpe_encode_v2,
        "bpe_encode overload with non-monotone merge-table guard bitmap "
        "(trailing unsafe_bits int32 tensor; empty = identical behavior)");
  m.def("bpe_encode_memo", &bpe_encode_memo_v2,
        "bpe_encode_memo overload with guard bitmap (trailing unsafe_bits)");
  m.def("encode_fused", &encode_fused_v2,
        "encode_fused overload with guard bitmap (trailing unsafe_bits)");
  m.def("encode_fused_ds", &encode_fused_ds_v2,
        "encode_fused_ds overload with guard bitmap (trailing unsafe_bits)");
  m.def("encode_fused_laguna", &encode_fused_laguna_v2,
        "encode_fused_laguna overload with guard bitmap (trailing "
        "unsafe_bits)");
  m.def("encode_fused_o200k", &encode_fused_o200k_v2,
        "encode_fused_o200k overload with guard bitmap (trailing "
        "unsafe_bits)");
}
#endif
