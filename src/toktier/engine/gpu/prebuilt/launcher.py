"""Python-side launchers for the prebuilt kernel delivery.

:class:`PrebuiltExtension` exposes the same callable surface as the
pybind module the JIT delivery builds (``toktier_pretok_cuda``): same
function names, same argument order, same tensor contracts, same
return shapes -- so the engine, the encoders and the pretokenizers run
unchanged on either delivery. Every method here is a line-by-line port
of the corresponding host function in ``pretok_kernel.cu``; comments
reference the C++ names rather than repeating their arguments.

Two host-side CUB algorithms are replaced with exact integer
equivalents (see ``prebuilt_unit.cu`` for the argument):

- ``cub::DeviceScan::InclusiveSum`` -> ``torch.cumsum`` with an explicit
  ``int32`` accumulator (integer, exact, capture-safe).
- ``cub::DeviceScan::InclusiveScan(maximum)`` over index seeds ->
  the carrier reformulation (cumsum + ``tk_carrier_scatter`` +
  ``tk_carrier_gather``).
- ``cub::DeviceSelect::Flagged`` -> cumsum + ``tk_select_scatter``.

Stream discipline mirrors the C++: everything launches on the torch
current stream of the tensor's device, the eager BPE path uses two side
streams joined back through events before returning, and the fused
paths issue no host synchronisation, so they stay CUDA-Graph
capturable (driver launches on a capturing stream are captured as graph
nodes, exactly like torch's own kernels).
"""

from __future__ import annotations

import ctypes
import json
import threading
from typing import Any

import torch

from .driver import CudaDriver

__all__ = ["PrebuiltExtension"]

# Compile-time constants of the shipped build (the build script passes
# no -DTOKTIER_* overrides, which the manifest's nvcc argv pins). The
# block size must match the compiled TPB: two kernels size their shared
# memory arrays with it.
TPB = 256
SHORT_MAX = 32
MED_MAX = 128
_INT32_MAX = 2**31 - 1
_HI = 0x7F7F7F7F  # the memset-0x7f init pattern, as an int32 value

_I32 = torch.int32
_U8 = torch.uint8


def _p(t: torch.Tensor | None) -> ctypes.c_void_p:
    return ctypes.c_void_p(0 if t is None else t.data_ptr())


def _pa(address: int) -> ctypes.c_void_p:
    return ctypes.c_void_p(address)


def _i(v: int) -> ctypes.c_int:
    return ctypes.c_int(int(v))


def _u(v: int) -> ctypes.c_uint:
    return ctypes.c_uint(int(v) & 0xFFFFFFFF)


def _b(v: bool) -> ctypes.c_bool:
    return ctypes.c_bool(bool(v))


def _grid(n: int, block: int = TPB) -> int:
    return (n + block - 1) // block


def _check(condition: bool, message: str) -> None:
    """Mirror of TORCH_CHECK: contract violations raise RuntimeError."""
    if not condition:
        raise RuntimeError(message)


def _cumsum_i32(values: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(values, 0, dtype=_I32)


class PrebuiltExtension:
    """Driver-API launcher surface over the shipped fatbin.

    One instance covers every CUDA device in the process: the module is
    loaded lazily per device (into the same primary context torch
    uses), and every method dispatches on its tensors' device.
    """

    #: Marker the engine and reports can read to tell deliveries apart.
    delivery = "prebuilt"

    def __init__(self, fatbin: bytes, kernel_symbols: dict[str, str]):
        self._driver = CudaDriver.get()
        self._fatbin = fatbin
        self._symbols = dict(kernel_symbols)
        self._modules: dict[int, int] = {}
        self._functions: dict[tuple[int, str], int] = {}
        self._lock = threading.Lock()

    # -- module / function resolution ---------------------------------

    def load_for_device(self, index: int) -> None:
        """Load the module into device ``index``'s primary context."""
        self._module(index)

    def _module(self, index: int) -> int:
        with self._lock:
            handle = self._modules.get(index)
            if handle is None:
                with torch.cuda.device(index):
                    # A torch allocation makes the device's primary
                    # context current on this thread; the module then
                    # loads into the context torch computes in.
                    torch.empty(1, device=f"cuda:{index}")
                    handle = self._driver.load_module(self._fatbin)
                self._modules[index] = handle
            return handle

    def _fn(self, index: int, logical: str) -> int:
        key = (index, logical)
        function = self._functions.get(key)
        if function is None:
            module = self._module(index)
            symbol = self._symbols.get(logical)
            if symbol is None:
                raise RuntimeError(
                    f"kernel {logical!r} is not in the build manifest's "
                    "symbol map"
                )
            function = self._driver.get_function(module, symbol)
            self._functions[key] = function
        return function

    def _launch(
        self,
        index: int,
        logical: str,
        grid: int,
        args: list[Any],
        block: int = TPB,
    ) -> None:
        stream = torch.cuda.current_stream(index).cuda_stream
        self._driver.launch(
            self._fn(index, logical),
            (int(grid), 1, 1),
            (int(block), 1, 1),
            stream,
            args,
        )

    @staticmethod
    def _dev(t: torch.Tensor) -> int:
        index = t.device.index
        if index is None:
            return int(torch.cuda.current_device())
        return int(index)

    # -- shared building blocks ---------------------------------------

    def _carrier_propagate(
        self, dev: int, flag_i32: torch.Tensor, cap: int
    ) -> torch.Tensor:
        """latest-carrier-index propagation (max-scan replacement).

        ``flag_i32[0]`` must be nonzero (the callers force it), so every
        ``rid`` value is >= 1 and the gather never reads below index 0.
        """
        rid = _cumsum_i32(flag_i32)
        device = flag_i32.device
        pos = torch.empty(cap, dtype=_I32, device=device)
        self._launch(
            dev, "tk_carrier_scatter", _grid(cap),
            [_p(rid), _p(flag_i32), _p(pos), _i(cap)],
        )
        out = torch.empty(cap, dtype=_I32, device=device)
        self._launch(
            dev, "tk_carrier_gather", _grid(cap),
            [_p(rid), _p(pos), _p(out), _i(cap)],
        )
        return out

    def _select(
        self,
        dev: int,
        values: torch.Tensor | None,
        flags: torch.Tensor,
        out_ptr: ctypes.c_void_p,
        count_ptr: ctypes.c_void_p,
        cap: int,
    ) -> None:
        """DeviceSelect::Flagged replacement (stable, device-side count).

        ``values=None`` selects the counting-iterator form (the value at
        a selected position is its index).
        """
        psum = _cumsum_i32(flags.to(_I32))
        self._launch(
            dev, "tk_select_scatter", _grid(cap),
            [_p(values), _p(flags), _p(psum), out_ptr, count_ptr, _i(cap)],
        )

    @staticmethod
    def _check_bpe_tables_meta(
        data: torch.Tensor,
        pair_keys: torch.Tensor,
        pair_vals: torch.Tensor,
        byte_id: torch.Tensor,
        vocab_keys: torch.Tensor,
        vocab_vals: torch.Tensor,
        vocab_blob: torch.Tensor,
    ) -> None:
        """Port of check_bpe_tables_meta."""

        def pow2(n: int) -> bool:
            return n > 0 and (n & (n - 1)) == 0

        _check(
            pow2(pair_keys.numel())
            and pair_vals.numel() == pair_keys.numel(),
            "pair_keys/pair_vals must be equal-length power-of-two tables",
        )
        _check(
            pow2(vocab_keys.numel())
            and vocab_vals.numel() == vocab_keys.numel(),
            "vocab_keys/vocab_vals must be equal-length power-of-two tables",
        )
        for table in (pair_keys, pair_vals, vocab_keys, vocab_vals):
            _check(
                table.dtype in (torch.int64, torch.uint64),
                "pair/vocab keys/vals must be 64-bit integer tensors "
                "(kernels reinterpret them as uint64)",
            )
        _check(
            byte_id.dtype == _I32 and byte_id.numel() == 256,
            "byte_id must be int32[256]",
        )
        _check(vocab_blob.dtype == _U8, "vocab_blob must be uint8")
        for table in (
            pair_keys, pair_vals, byte_id, vocab_keys, vocab_vals, vocab_blob
        ):
            _check(
                table.is_cuda
                and table.device == data.device
                and table.is_contiguous(),
                "BPE tables must be contiguous CUDA tensors on bytes' device",
            )

    @staticmethod
    def _unsafe_bits(
        unsafe_bits: torch.Tensor | None, data: torch.Tensor
    ) -> torch.Tensor | None:
        """Port of unsafe_bits_ptr; None selects the kernel nullptr branch."""
        if unsafe_bits is None or unsafe_bits.numel() == 0:
            return None
        _check(
            unsafe_bits.is_cuda
            and unsafe_bits.device == data.device
            and unsafe_bits.dtype == _I32
            and unsafe_bits.is_contiguous(),
            "unsafe_bits must be a contiguous int32 CUDA tensor on "
            "bytes' device (uint32 bitmap viewed as int32)",
        )
        return unsafe_bits

    # ======================= UTF-8 decode ============================

    def _utf8_decode_impl(
        self, data: torch.Tensor, want_bo: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of utf8_decode_impl (eager: reads the error flag back)."""
        dev = self._dev(data)
        with torch.cuda.device(dev):
            _check(
                data.is_cuda
                and data.dtype == _U8
                and data.is_contiguous(),
                "bytes must be a contiguous uint8 CUDA tensor",
            )
            nb = data.numel()
            _check(nb < _INT32_MAX, "single-buffer limit 2^31 bytes")
            opts = {"dtype": _I32, "device": data.device}
            if nb == 0:
                return (
                    torch.empty(0, **opts),
                    torch.empty(0, **opts),
                )
            lead = (data & 0xC0) != 0x80
            cpos = _cumsum_i32(lead.to(_I32))
            n_chars = int(cpos[-1].item())
            cp = torch.empty(n_chars, **opts)
            bo = torch.empty(n_chars if want_bo else 0, **opts)
            err = torch.zeros(1, **opts)
            self._launch(
                dev, "k_utf8_decode", _grid(nb),
                [
                    _p(data), _p(cpos), _p(cp),
                    _p(bo) if want_bo else _p(None),
                    _i(nb), _p(err),
                ],
            )
            if int(err.item()) != 0:
                raise ValueError(
                    "invalid UTF-8 in input bytes (truncated sequence / "
                    "illegal lead F8-FF / bad continuation / overlong "
                    "encoding / surrogate U+D800-DFFF / codepoint > "
                    "U+10FFFF)"
                )
            return cp, bo

    def utf8_to_cp(self, data: torch.Tensor) -> torch.Tensor:
        return self._utf8_decode_impl(data, False)[0]

    def utf8_to_cp_bo(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp, bo = self._utf8_decode_impl(data, True)
        return cp, bo

    # ================== GPT-style / DS / Laguna pretok ===============

    def _pretok_impl_t(
        self,
        rs: int,
        cp: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        dstart: torch.Tensor | None,
        dso: torch.Tensor | None,
        ars: torch.Tensor | None,
    ) -> torch.Tensor:
        """Port of pretok_impl_t<RS> (eager: one 4-byte D2H for R)."""
        dev = self._dev(cp)
        with torch.cuda.device(dev):
            _check(
                cp.is_cuda and cp.dtype == _I32 and cp.is_contiguous(),
                "cp must be a contiguous int32 CUDA tensor",
            )
            _check(tab.is_cuda and tab.dtype == _U8, "tab must be uint8 CUDA")
            n = cp.numel()
            _check(n < _INT32_MAX, "single-buffer limit 2^31 chars")
            device = cp.device
            starts = torch.empty(n, dtype=torch.bool, device=device)
            if n == 0:
                return starts
            cls = torch.empty(n, dtype=_U8, device=device)
            head = torch.empty(n, dtype=_U8, device=device)
            self._launch(
                dev, f"k_classify_rs{rs}", _grid(n),
                [_p(cp), _p(tab), _p(dstart), _p(cls), _p(head), _i(n),
                 _p(None)],
            )
            rid = _cumsum_i32(head.to(_I32))
            n_runs = int(rid[-1].item())  # single 4B D2H sync
            run_start = torch.empty(n_runs, dtype=_I32, device=device)
            fnc = torch.full((n_runs,), _HI, dtype=_I32, device=device)
            lc = torch.full((n_runs,), -1, dtype=_I32, device=device)
            self._launch(
                dev, f"k_runinfo_rs{rs}", _grid(n),
                [_p(cp), _p(cls), _p(head), _p(rid), _p(run_start),
                 _p(fnc), _p(lc), _i(n), _p(None)],
            )
            self._launch(
                dev, f"k_rules_rs{rs}", _grid(n),
                [_p(cp), _p(cls), _p(head), _p(rid), _p(run_start),
                 _p(fnc), _p(lc), _p(dso), _p(dstart), _p(ars),
                 _i(n_runs), _i(dmax), _p(starts), _i(n), _p(None),
                 _p(None)],
            )
            return starts

    def _forced_carrier_flags(self, mask: torch.Tensor) -> torch.Tensor:
        """uint8 mask -> int32 carrier flags with position 0 forced on."""
        flags = mask.to(_I32)
        flags[0:1].fill_(1)
        return flags

    def pretok_starts(
        self, cp: torch.Tensor, tab: torch.Tensor, dmax: int
    ) -> torch.Tensor:
        return self._pretok_impl_t(0, cp, tab, dmax, None, None, None)

    def pretok_starts_batched(
        self,
        cp: torch.Tensor,
        dstart: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
    ) -> torch.Tensor:
        """Port of pretok_starts_batched (dso via carrier propagation)."""
        dev = self._dev(cp)
        with torch.cuda.device(dev):
            _check(
                dstart.is_cuda
                and dstart.dtype == _U8
                and dstart.is_contiguous()
                and dstart.numel() == cp.numel(),
                "dstart must be a contiguous uint8 CUDA tensor matching cp",
            )
            n = cp.numel()
            _check(n < _INT32_MAX, "single-buffer limit 2^31 chars")
            dso = None
            if n > 0:
                flags = self._forced_carrier_flags(dstart)
                dso = self._carrier_propagate(dev, flags, n)
            return self._pretok_impl_t(0, cp, tab, dmax, dstart, dso, None)

    # -- DeepSeek prepass ---------------------------------------------

    def _ds_prepass(
        self,
        cp: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        doc: torch.Tensor | None,
        cap: int,
        n_dev: ctypes.c_void_p,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Port of ds_prepass: (B, dso, ars), capture-safe."""
        dev = self._dev(cp)
        device = cp.device
        seed = torch.empty(cap, dtype=_I32, device=device)
        self._launch(
            dev, "k_ds_seed_n", _grid(cap),
            [_p(cp), _p(tab), _p(doc), _p(seed), _i(cap), n_dev],
        )
        iota = torch.arange(cap, dtype=_I32, device=device)
        # nrs: carriers are exactly the positions seeding their own
        # index (non-members and N-run heads; position 0 always).
        nrs = self._carrier_propagate(
            dev, torch.eq(seed, iota).to(_I32), cap
        )
        bmask = torch.empty(cap, dtype=_U8, device=device)
        # seed is reused as the aseed output, as in the C++ (the eq
        # above has already been consumed by the cumsum chain on the
        # same stream, so the overwrite is ordered after it).
        self._launch(
            dev, "k_ds_bmask", _grid(cap),
            [_p(cp), _p(tab), _p(doc), _p(nrs), _i(dmax), _p(bmask),
             _p(seed), _i(cap), n_dev],
        )
        ars = self._carrier_propagate(
            dev, torch.eq(seed, iota).to(_I32), cap
        )
        dso = self._carrier_propagate(
            dev, self._forced_carrier_flags(bmask), cap
        )
        return bmask, dso, ars

    def pretok_starts_ds(
        self, cp: torch.Tensor, tab: torch.Tensor, dmax: int
    ) -> torch.Tensor:
        dev = self._dev(cp)
        with torch.cuda.device(dev):
            _check(
                cp.is_cuda and cp.dtype == _I32 and cp.is_contiguous(),
                "cp must be a contiguous int32 CUDA tensor",
            )
            n = cp.numel()
            _check(n < _INT32_MAX, "single-buffer limit 2^31 chars")
            if n == 0:
                return torch.empty(0, dtype=torch.bool, device=cp.device)
            bmask, dso, ars = self._ds_prepass(
                cp, tab, dmax, None, n, _p(None)
            )
            return self._pretok_impl_t(1, cp, tab, dmax, bmask, dso, ars)

    def pretok_starts_batched_ds(
        self,
        cp: torch.Tensor,
        dstart: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
    ) -> torch.Tensor:
        dev = self._dev(cp)
        with torch.cuda.device(dev):
            _check(
                cp.is_cuda and cp.dtype == _I32 and cp.is_contiguous(),
                "cp must be a contiguous int32 CUDA tensor",
            )
            _check(
                dstart.is_cuda
                and dstart.dtype == _U8
                and dstart.is_contiguous()
                and dstart.numel() == cp.numel(),
                "dstart must be a contiguous uint8 CUDA tensor matching cp",
            )
            n = cp.numel()
            _check(n < _INT32_MAX, "single-buffer limit 2^31 chars")
            if n == 0:
                return torch.empty(0, dtype=torch.bool, device=cp.device)
            bmask, dso, ars = self._ds_prepass(
                cp, tab, dmax, dstart, n, _p(None)
            )
            return self._pretok_impl_t(1, cp, tab, dmax, bmask, dso, ars)

    # -- Laguna prepass -----------------------------------------------

    def _lag_prepass(
        self,
        cp: torch.Tensor,
        doc: torch.Tensor | None,
        cap: int,
        n_dev: ctypes.c_void_p,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of lag_prepass: (B, dso), capture-safe."""
        dev = self._dev(cp)
        bmask = torch.empty(cap, dtype=_U8, device=cp.device)
        self._launch(
            dev, "k_lag_bmask", _grid(cap),
            [_p(cp), _p(doc), _p(bmask), _i(cap), n_dev],
        )
        dso = self._carrier_propagate(
            dev, self._forced_carrier_flags(bmask), cap
        )
        return bmask, dso

    def pretok_starts_laguna(
        self, cp: torch.Tensor, tab: torch.Tensor, dmax: int
    ) -> torch.Tensor:
        dev = self._dev(cp)
        with torch.cuda.device(dev):
            _check(
                cp.is_cuda and cp.dtype == _I32 and cp.is_contiguous(),
                "cp must be a contiguous int32 CUDA tensor",
            )
            n = cp.numel()
            _check(n < _INT32_MAX, "single-buffer limit 2^31 chars")
            if n == 0:
                return torch.empty(0, dtype=torch.bool, device=cp.device)
            bmask, dso = self._lag_prepass(cp, None, n, _p(None))
            return self._pretok_impl_t(2, cp, tab, dmax, bmask, dso, None)

    def pretok_starts_batched_laguna(
        self,
        cp: torch.Tensor,
        dstart: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
    ) -> torch.Tensor:
        dev = self._dev(cp)
        with torch.cuda.device(dev):
            _check(
                cp.is_cuda and cp.dtype == _I32 and cp.is_contiguous(),
                "cp must be a contiguous int32 CUDA tensor",
            )
            _check(
                dstart.is_cuda
                and dstart.dtype == _U8
                and dstart.is_contiguous()
                and dstart.numel() == cp.numel(),
                "dstart must be a contiguous uint8 CUDA tensor matching cp",
            )
            n = cp.numel()
            _check(n < _INT32_MAX, "single-buffer limit 2^31 chars")
            if n == 0:
                return torch.empty(0, dtype=torch.bool, device=cp.device)
            bmask, dso = self._lag_prepass(cp, dstart, n, _p(None))
            return self._pretok_impl_t(2, cp, tab, dmax, bmask, dso, None)

    # -- DeepSeek constants self-description --------------------------

    def ds_constants(self) -> str:
        """Port of ds_constants, read from the device code itself."""
        dev = torch.cuda.current_device()
        with torch.cuda.device(dev):
            out = torch.zeros(141, dtype=_I32, device=f"cuda:{dev}")
            self._launch(dev, "tk_ds_constants_dump", 1, [_p(out)], block=1)
            values = out.cpu().tolist()
        cjk = values[0:6]
        enum = values[6:13]
        bits = values[13:141]
        payload = {
            "cjk_ranges": [
                [cjk[0], cjk[1]], [cjk[2], cjk[3]], [cjk[4], cjk[5]]
            ],
            "a3_space": 32,
            "crlf_cps": [10, 13],
            "class_enum": {
                "O": enum[0], "L": enum[1], "M": enum[2], "N": enum[3],
                "PS": enum[4], "WS": enum[5], "CRLF": enum[6],
            },
            "apunct": [c for c in range(128) if bits[c] & 1],
            "alpha": [c for c in range(128) if bits[c] & 2],
        }
        return json.dumps(payload, separators=(",", ":"))

    # ======================= per-piece BPE ===========================

    def _bpe_encode_impl(
        self,
        data: torch.Tensor,
        pb: torch.Tensor,
        mkeys: torch.Tensor | None,
        mmeta: torch.Tensor | None,
        mbytes: torch.Tensor | None,
        mvals: torch.Tensor | None,
        pair_keys: torch.Tensor,
        pair_vals: torch.Tensor,
        byte_id: torch.Tensor,
        vocab_keys: torch.Tensor,
        vocab_vals: torch.Tensor,
        vocab_blob: torch.Tensor,
        ignore_merges: int,
        unsafe_bits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of bpe_encode_impl (eager; two side streams + events)."""
        dev = self._dev(data)
        with torch.cuda.device(dev):
            _check(
                data.is_cuda and data.dtype == _U8 and data.is_contiguous(),
                "bytes must be a contiguous uint8 CUDA tensor",
            )
            _check(
                pb.dtype == _I32 and pb.is_contiguous(),
                "pb must be a contiguous int32 tensor",
            )
            _check(
                pb.is_cuda and pb.device == data.device and pb.numel() >= 1,
                "pb must be a non-empty CUDA tensor on bytes' device",
            )
            self._check_bpe_tables_meta(
                data, pair_keys, pair_vals, byte_id,
                vocab_keys, vocab_vals, vocab_blob,
            )
            device = data.device
            n_pieces = pb.numel() - 1
            if n_pieces <= 0:
                return (
                    torch.empty(0, dtype=_I32, device=device),
                    torch.zeros(1, dtype=_I32, device=device),
                )
            memo = mkeys is not None and mkeys.numel() > 0
            mmask = (mkeys.numel() - 1) if mkeys is not None and memo else 0
            scratch = torch.empty(data.numel(), dtype=_I32, device=device)
            cnt = torch.empty(n_pieces, dtype=_I32, device=device)
            lens = pb[1:] - pb[:-1]
            short_list = (
                torch.nonzero(lens <= SHORT_MAX).flatten().to(_I32)
            )
            warp_list = (
                torch.nonzero((lens > SHORT_MAX) & (lens <= MED_MAX))
                .flatten()
                .to(_I32)
            )
            long_list = torch.nonzero(lens > MED_MAX).flatten().to(_I32)
            n_short = short_list.numel()
            n_warp = warp_list.numel()
            n_long = long_list.numel()
            pmask = pair_keys.numel() - 1
            vmask = vocab_keys.numel() - 1
            ub = self._unsafe_bits(unsafe_bits, data)
            current = torch.cuda.current_stream(dev)
            ev_ready = torch.cuda.Event()
            ev_ready.record(current)
            if n_short > 0:
                self._launch(
                    dev, "k_bpe_thread_cap32", _grid(n_short),
                    [
                        _p(data), _p(pb), _p(short_list), _i(n_short),
                        _p(pair_keys), _p(pair_vals), _u(pmask),
                        _p(byte_id), _p(vocab_keys), _p(vocab_vals),
                        _u(vmask), _p(vocab_blob), _i(ignore_merges),
                        _p(scratch), _p(cnt), _p(None),
                        _p(mkeys if memo else None),
                        _p(mmeta if memo else None),
                        _p(mbytes if memo else None),
                        _p(mvals if memo else None),
                        _u(mmask),
                    ],
                )
            if n_warp > 0:
                s_warp = torch.cuda.Stream(device)
                s_warp.wait_event(ev_ready)
                with torch.cuda.stream(s_warp):
                    self._launch(
                        dev, "k_bpe_warp", _grid(n_warp * 32),
                        [
                            _p(data), _p(pb), _p(warp_list), _i(n_warp),
                            _p(None),
                            _p(pair_keys), _p(pair_vals), _u(pmask),
                            _p(byte_id), _p(vocab_keys), _p(vocab_vals),
                            _u(vmask), _p(vocab_blob), _i(ignore_merges),
                            _p(scratch), _p(cnt), _p(ub),
                        ],
                    )
                ev_warp = torch.cuda.Event()
                ev_warp.record(s_warp)
                current.wait_event(ev_warp)
            if n_long > 0:
                buf_b = torch.empty(data.numel(), dtype=_I32, device=device)
                s_long = torch.cuda.Stream(device)
                s_long.wait_event(ev_ready)
                with torch.cuda.stream(s_long):
                    self._launch(
                        dev, "k_bpe_long", n_long,
                        [
                            _p(data), _p(pb), _p(long_list),
                            _p(pair_keys), _p(pair_vals), _u(pmask),
                            _p(byte_id), _p(vocab_keys), _p(vocab_vals),
                            _u(vmask), _p(vocab_blob), _i(ignore_merges),
                            _p(scratch), _p(buf_b), _p(cnt),
                            _i(n_long), _p(None), _p(ub),
                        ],
                    )
                ev_long = torch.cuda.Event()
                ev_long.record(s_long)
                current.wait_event(ev_long)
            if memo:
                self._launch(
                    dev, "k_memo_insert", _grid(n_pieces),
                    [
                        _p(data), _p(pb), _i(n_pieces),
                        _p(mkeys), _p(mmeta), _p(mbytes), _p(mvals),
                        _u(mmask), _p(scratch), _p(cnt),
                    ],
                )
            off = torch.zeros(n_pieces + 1, dtype=_I32, device=device)
            torch.cumsum(cnt, 0, out=off[1:])
            total = int(off[n_pieces].item())
            out = torch.empty(total, dtype=_I32, device=device)
            self._launch(
                dev, "k_bpe_compact", n_pieces,
                [_p(pb), _p(off), _p(scratch), _p(out), _i(n_pieces),
                 _p(None)],
                block=64,
            )
            return out, off

    def bpe_encode(
        self,
        data: torch.Tensor,
        pb: torch.Tensor,
        pair_keys: torch.Tensor,
        pair_vals: torch.Tensor,
        byte_id: torch.Tensor,
        vocab_keys: torch.Tensor,
        vocab_vals: torch.Tensor,
        vocab_blob: torch.Tensor,
        ignore_merges: int,
        unsafe_bits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bpe_encode_impl(
            data, pb, None, None, None, None,
            pair_keys, pair_vals, byte_id,
            vocab_keys, vocab_vals, vocab_blob,
            ignore_merges, unsafe_bits,
        )

    def bpe_encode_memo(
        self,
        data: torch.Tensor,
        pb: torch.Tensor,
        pair_keys: torch.Tensor,
        pair_vals: torch.Tensor,
        byte_id: torch.Tensor,
        vocab_keys: torch.Tensor,
        vocab_vals: torch.Tensor,
        vocab_blob: torch.Tensor,
        ignore_merges: int,
        mkeys: torch.Tensor | None = None,
        mmeta: torch.Tensor | None = None,
        mbytes: torch.Tensor | None = None,
        mvals: torch.Tensor | None = None,
        unsafe_bits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bpe_encode_impl(
            data, pb, mkeys, mmeta, mbytes, mvals,
            pair_keys, pair_vals, byte_id,
            vocab_keys, vocab_vals, vocab_blob,
            ignore_merges, unsafe_bits,
        )

    # ================ fused single-request path ======================

    def _encode_fused_t(
        self,
        rs: int,
        data: torch.Tensor,
        nb_dev: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        pair_keys: torch.Tensor,
        pair_vals: torch.Tensor,
        byte_id: torch.Tensor,
        vocab_keys: torch.Tensor,
        vocab_vals: torch.Tensor,
        vocab_blob: torch.Tensor,
        ignore_merges: int,
        unsafe_bits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of encode_fused_t<RS>: no host sync, capture-safe."""
        dev = self._dev(data)
        with torch.cuda.device(dev):
            _check(
                data.is_cuda and data.dtype == _U8 and data.is_contiguous(),
                "bytes must be a contiguous uint8 CUDA tensor",
            )
            _check(
                nb_dev.is_cuda
                and nb_dev.dtype == _I32
                and nb_dev.numel() == 1,
                "nb_dev must be a one-element int32 CUDA tensor",
            )
            _check(
                nb_dev.device == data.device,
                "nb_dev must live on bytes' device",
            )
            _check(
                tab.is_cuda
                and tab.device == data.device
                and tab.dtype == _U8
                and tab.is_contiguous()
                and tab.numel() >= 0x110000,
                "tab must be a contiguous uint8 CUDA table covering U+10FFFF",
            )
            self._check_bpe_tables_meta(
                data, pair_keys, pair_vals, byte_id,
                vocab_keys, vocab_vals, vocab_blob,
            )
            cap = data.numel()
            _check(0 < cap < _INT32_MAX, "capacity must be in (0, 2^31)")
            device = data.device
            gs_b = _grid(cap)

            # ---- UTF-8 decode ----
            lead = (data & 0xC0) != 0x80
            cpos = _cumsum_i32(lead.to(_I32))
            cp = torch.empty(cap, dtype=_I32, device=device)
            bo = torch.empty(cap, dtype=_I32, device=device)
            self._launch(
                dev, "k_utf8_decode", gs_b,
                [_p(data), _p(cpos), _p(cp), _p(bo), _i(cap), _p(None)],
            )
            d_c = _pa(cpos.data_ptr() + 4 * (cap - 1))

            # ---- ruleset prepass ----
            bmask = dso = ars = None
            if rs == 1:
                bmask, dso, ars = self._ds_prepass(
                    cp, tab, dmax, None, cap, d_c
                )
            elif rs == 2:
                bmask, dso = self._lag_prepass(cp, None, cap, d_c)

            # ---- classify / run segmentation ----
            cls = torch.empty(cap, dtype=_U8, device=device)
            head = torch.zeros(cap, dtype=_U8, device=device)
            self._launch(
                dev, f"k_classify_rs{rs}", gs_b,
                [_p(cp), _p(tab), _p(bmask), _p(cls), _p(head), _i(0), d_c],
            )
            rid = _cumsum_i32(head.to(_I32))
            d_r = _pa(rid.data_ptr() + 4 * (cap - 1))
            run_start = torch.empty(cap, dtype=_I32, device=device)
            fnc = torch.full((cap,), _HI, dtype=_I32, device=device)
            lc = torch.full((cap,), -1, dtype=_I32, device=device)
            self._launch(
                dev, f"k_runinfo_rs{rs}", gs_b,
                [_p(cp), _p(cls), _p(head), _p(rid), _p(run_start),
                 _p(fnc), _p(lc), _i(0), d_c],
            )
            starts = torch.zeros(cap, dtype=torch.bool, device=device)
            self._launch(
                dev, f"k_rules_rs{rs}", gs_b,
                [_p(cp), _p(cls), _p(head), _p(rid), _p(run_start),
                 _p(fnc), _p(lc), _p(dso), _p(bmask), _p(ars),
                 _i(0), _i(dmax), _p(starts), _i(0), d_c, d_r],
            )

            # ---- piece bounds and dispatch ----
            pb = torch.empty(cap + 1, dtype=_I32, device=device)
            d_cnts = torch.empty(4, dtype=_I32, device=device)
            d_p = _p(d_cnts)
            self._select(dev, bo, starts, _p(pb), d_p, cap)
            self._launch(
                dev, "k_pb_sentinel", 1,
                [_p(pb), d_p, _p(nb_dev)], block=1,
            )
            flags3 = torch.empty(3 * cap, dtype=_U8, device=device)
            cnt = torch.zeros(cap, dtype=_I32, device=device)
            f_base = flags3.data_ptr()
            self._launch(
                dev, "k_dispatch_flags", gs_b,
                [_p(pb), d_p, _pa(f_base), _pa(f_base + cap),
                 _pa(f_base + 2 * cap), _p(cnt), _i(cap)],
            )
            lists = torch.empty(3 * cap, dtype=_I32, device=device)
            l_base = lists.data_ptr()
            for k in range(3):
                self._select(
                    dev, None, flags3[k * cap:(k + 1) * cap],
                    _pa(l_base + 4 * k * cap),
                    _pa(d_cnts.data_ptr() + 4 * (k + 1)),
                    cap,
                )

            # ---- BPE at capacity-bound geometry ----
            scratch = torch.empty(cap, dtype=_I32, device=device)
            buf_b = torch.empty(cap, dtype=_I32, device=device)
            pmask = pair_keys.numel() - 1
            vmask = vocab_keys.numel() - 1
            ub = self._unsafe_bits(unsafe_bits, data)
            self._launch(
                dev, "k_bpe_thread_cap32", gs_b,
                [
                    _p(data), _p(pb), _pa(l_base), _i(0),
                    _p(pair_keys), _p(pair_vals), _u(pmask),
                    _p(byte_id), _p(vocab_keys), _p(vocab_vals),
                    _u(vmask), _p(vocab_blob), _i(ignore_merges),
                    _p(scratch), _p(cnt),
                    _pa(d_cnts.data_ptr() + 4),
                    _p(None), _p(None), _p(None), _p(None), _u(0),
                ],
            )
            gw = min(gs_b, 8192)
            self._launch(
                dev, "k_bpe_warp", gw,
                [
                    _p(data), _p(pb), _pa(l_base + 4 * cap), _i(0),
                    _pa(d_cnts.data_ptr() + 8),
                    _p(pair_keys), _p(pair_vals), _u(pmask),
                    _p(byte_id), _p(vocab_keys), _p(vocab_vals),
                    _u(vmask), _p(vocab_blob), _i(ignore_merges),
                    _p(scratch), _p(cnt), _p(ub),
                ],
            )
            gl = min(cap // (MED_MAX + 1) + 1, 8192)
            self._launch(
                dev, "k_bpe_long", gl,
                [
                    _p(data), _p(pb), _pa(l_base + 8 * cap),
                    _p(pair_keys), _p(pair_vals), _u(pmask),
                    _p(byte_id), _p(vocab_keys), _p(vocab_vals),
                    _u(vmask), _p(vocab_blob), _i(ignore_merges),
                    _p(scratch), _p(buf_b), _p(cnt),
                    _i(0), _pa(d_cnts.data_ptr() + 12), _p(ub),
                ],
            )

            # ---- prefix sum + compaction ----
            off = torch.zeros(cap + 1, dtype=_I32, device=device)
            torch.cumsum(cnt, 0, out=off[1:])
            out = torch.empty(cap, dtype=_I32, device=device)
            self._launch(
                dev, "k_bpe_compact", min(cap, 65535),
                [_p(pb), _p(off), _p(scratch), _p(out), _i(0), d_p],
                block=64,
            )
            return out, off[cap:cap + 1]

    def encode_fused(
        self,
        data: torch.Tensor,
        nb_dev: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        pair_keys: torch.Tensor,
        pair_vals: torch.Tensor,
        byte_id: torch.Tensor,
        vocab_keys: torch.Tensor,
        vocab_vals: torch.Tensor,
        vocab_blob: torch.Tensor,
        ignore_merges: int,
        unsafe_bits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_fused_t(
            0, data, nb_dev, tab, dmax, pair_keys, pair_vals, byte_id,
            vocab_keys, vocab_vals, vocab_blob, ignore_merges, unsafe_bits,
        )

    def encode_fused_ds(
        self,
        data: torch.Tensor,
        nb_dev: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        pair_keys: torch.Tensor,
        pair_vals: torch.Tensor,
        byte_id: torch.Tensor,
        vocab_keys: torch.Tensor,
        vocab_vals: torch.Tensor,
        vocab_blob: torch.Tensor,
        ignore_merges: int,
        unsafe_bits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_fused_t(
            1, data, nb_dev, tab, dmax, pair_keys, pair_vals, byte_id,
            vocab_keys, vocab_vals, vocab_blob, ignore_merges, unsafe_bits,
        )

    def encode_fused_laguna(
        self,
        data: torch.Tensor,
        nb_dev: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        pair_keys: torch.Tensor,
        pair_vals: torch.Tensor,
        byte_id: torch.Tensor,
        vocab_keys: torch.Tensor,
        vocab_vals: torch.Tensor,
        vocab_blob: torch.Tensor,
        ignore_merges: int,
        unsafe_bits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_fused_t(
            2, data, nb_dev, tab, dmax, pair_keys, pair_vals, byte_id,
            vocab_keys, vocab_vals, vocab_blob, ignore_merges, unsafe_bits,
        )

    # ===================== NFC quick check ===========================

    def nfc_qc_scan(
        self, data: torch.Tensor, tab: torch.Tensor
    ) -> torch.Tensor:
        dev = self._dev(data)
        with torch.cuda.device(dev):
            _check(
                data.is_cuda and data.dtype == _U8 and data.is_contiguous(),
                "bytes must be a contiguous uint8 CUDA tensor",
            )
            _check(
                tab.is_cuda
                and tab.dtype == _U8
                and tab.is_contiguous()
                and tab.numel() == 0x110000,
                "QC table must be a full-plane uint8[0x110000]",
            )
            nb = data.numel()
            _check(nb < _INT32_MAX, "single-buffer limit 2^31 bytes")
            flag = torch.zeros(1, dtype=_I32, device=data.device)
            if nb == 0:
                return flag
            self._launch(
                dev, "k_nfc_qc", _grid(nb),
                [_p(data), _p(tab), _i(nb), _p(flag)],
            )
            return flag

    # ==================== o200k splitter group =======================

    def _o2k_channels(
        self,
        dev: int,
        cp: torch.Tensor,
        tab: torch.Tensor,
        dstart: torch.Tensor | None,
        kimi: bool,
        n: int,
    ) -> dict[str, Any]:
        """Shared eager middle section of the three o200k entries."""
        device = cp.device
        gs = _grid(n)
        cls = torch.empty(n, dtype=_U8, device=device)
        head_m = torch.empty(n, dtype=_U8, device=device)
        head_cs = torch.empty(n, dtype=_U8, device=device)
        head_lw = torch.empty(n, dtype=_U8, device=device)
        head_pm = torch.empty(n, dtype=_U8, device=device)
        self._launch(
            dev, "k_o2k_heads", gs,
            [_p(cp), _p(tab), _i(n), _p(None), _p(dstart), _p(cls),
             _p(head_m), _p(head_cs), _p(head_lw), _p(head_pm), _b(kimi)],
        )
        rid_m = _cumsum_i32(head_m.to(_I32))
        rid_cs = _cumsum_i32(head_cs.to(_I32))
        rid_lw = _cumsum_i32(head_lw.to(_I32))
        rid_pm = _cumsum_i32(head_pm.to(_I32))
        lasts = torch.stack(
            [rid_m[-1], rid_cs[-1], rid_lw[-1], rid_pm[-1]]
        ).cpu()  # pack the four run counts into a single D2H
        n_runs = int(lasts[0].item())
        r_cs = max(int(lasts[1].item()), 1)
        r_lw = max(int(lasts[2].item()), 1)
        r_pm = max(int(lasts[3].item()), 1)

        def hi(count: int) -> torch.Tensor:
            return torch.full((count,), _HI, dtype=_I32, device=device)

        def neg(count: int) -> torch.Tensor:
            return torch.full((count,), -1, dtype=_I32, device=device)

        channels = {
            "cls": cls, "head_m": head_m,
            "rid_m": rid_m, "rid_cs": rid_cs,
            "rid_lw": rid_lw, "rid_pm": rid_pm,
            "n_runs": n_runs,
            "run_start": torch.empty(n_runs, dtype=_I32, device=device),
            "first_anchor": hi(r_cs),
            "last_m_pm": neg(r_pm),
            "s_fl": hi(n_runs), "s_lc": neg(n_runs), "p_fl": hi(n_runs),
            "last_l": neg(n_runs), "last_c": neg(n_runs),
            "f_l0": hi(r_lw), "f_l1": hi(r_lw), "f_l2": hi(r_lw),
            "pm_fp": hi(r_pm),
        }
        ch = channels
        self._launch(
            dev, "k_o2k_runinfo1", gs,
            [_p(cp), _p(cls), _p(head_m), _p(rid_m), _p(rid_cs),
             _p(rid_pm), _i(n), _p(None), _p(dstart), _p(ch["run_start"]),
             _p(ch["first_anchor"]), _p(ch["last_m_pm"]), _b(kimi)],
        )
        self._launch(
            dev, "k_o2k_runinfo2", gs,
            [_p(cp), _p(cls), _p(rid_m), _p(rid_cs), _p(rid_lw),
             _p(rid_pm), _p(ch["run_start"]), _p(ch["first_anchor"]),
             _i(n), _p(None), _p(ch["s_fl"]), _p(ch["s_lc"]),
             _p(ch["p_fl"]), _p(ch["last_l"]), _p(ch["last_c"]),
             _p(ch["f_l0"]), _p(ch["f_l1"]), _p(ch["f_l2"]),
             _p(ch["pm_fp"]), _b(kimi)],
        )
        return channels

    def _o2k_rules(
        self,
        dev: int,
        cp: torch.Tensor,
        ch: dict[str, Any],
        dmax: int,
        contractions: bool,
        starts: torch.Tensor,
        pm_trig: torch.Tensor,
        chain_trig: torch.Tensor,
        chain: torch.Tensor,
        n: int,
        dstart: torch.Tensor | None,
        kimi: bool,
        hanx_trig: torch.Tensor | None,
    ) -> None:
        self._launch(
            dev, "k_o2k_rules", _grid(n),
            [_p(cp), _p(ch["cls"]), _p(ch["rid_m"]), _p(ch["rid_cs"]),
             _p(ch["rid_lw"]), _p(ch["rid_pm"]), _p(ch["run_start"]),
             _p(ch["first_anchor"]), _p(ch["s_fl"]), _p(ch["s_lc"]),
             _p(ch["p_fl"]), _p(ch["last_l"]), _p(ch["last_c"]),
             _p(ch["f_l0"]), _p(ch["f_l1"]), _p(ch["f_l2"]),
             _p(ch["pm_fp"]), _p(ch["last_m_pm"]), _i(ch["n_runs"]),
             _i(dmax), _b(contractions), _p(starts), _p(pm_trig),
             _p(chain_trig), _p(chain), _i(n), _p(None), _p(None),
             _p(dstart), _b(kimi), _p(hanx_trig)],
        )

    def _o2k_starts_common(
        self,
        cp: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        contractions: bool,
        dstart: torch.Tensor | None,
        kimi: bool,
    ) -> tuple[torch.Tensor, ...]:
        dev = self._dev(cp)
        with torch.cuda.device(dev):
            _check(
                cp.is_cuda and cp.dtype == _I32 and cp.is_contiguous(),
                "cp must be a contiguous int32 CUDA tensor",
            )
            if dstart is not None:
                _check(
                    dstart.is_cuda
                    and dstart.dtype == _U8
                    and dstart.numel() == cp.numel()
                    and dstart.is_contiguous(),
                    "dstart must be a contiguous uint8 CUDA tensor "
                    "matching cp",
                )
            _check(tab.is_cuda and tab.dtype == _U8, "tab must be uint8 CUDA")
            n = cp.numel()
            _check(n < _INT32_MAX, "single-buffer limit 2^31 chars")
            device = cp.device
            starts = torch.zeros(n, dtype=torch.bool, device=device)
            pm_trig = torch.zeros(max(n, 1), dtype=_U8, device=device)
            chain_trig = torch.zeros(max(n, 1), dtype=_U8, device=device)
            hanx_trig = (
                torch.zeros(max(n, 1), dtype=_U8, device=device)
                if kimi
                else None
            )
            chain = torch.zeros(1, dtype=_I32, device=device)
            if n == 0:
                if kimi:
                    assert hanx_trig is not None
                    return starts, pm_trig, chain_trig, hanx_trig, chain
                return starts, pm_trig, chain_trig, chain
            ch = self._o2k_channels(dev, cp, tab, dstart, kimi, n)
            self._o2k_rules(
                dev, cp, ch, dmax, contractions, starts, pm_trig,
                chain_trig, chain, n, dstart, kimi, hanx_trig,
            )
            if kimi:
                assert hanx_trig is not None
                return starts, pm_trig, chain_trig, hanx_trig, chain
            return starts, pm_trig, chain_trig, chain

    def pretok_starts_o200k(
        self,
        cp: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        contractions: bool,
    ) -> tuple[torch.Tensor, ...]:
        return self._o2k_starts_common(
            cp, tab, dmax, contractions, None, False
        )

    def pretok_starts_batched_o200k(
        self,
        cp: torch.Tensor,
        dstart: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        contractions: bool,
    ) -> tuple[torch.Tensor, ...]:
        return self._o2k_starts_common(
            cp, tab, dmax, contractions, dstart, False
        )

    def pretok_starts_kimi(
        self, cp: torch.Tensor, tab: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        # kimi: dmax=3, contractions always on (see pretok_starts_kimi).
        return self._o2k_starts_common(cp, tab, 3, True, None, True)

    def o200k_win_extents(
        self,
        cp: torch.Tensor,
        tab: torch.Tensor,
        sp: torch.Tensor,
        q_l: torch.Tensor,
        ds: torch.Tensor,
        de: torch.Tensor,
        dmax: int,
        contractions: bool,
        mode: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self._dev(cp)
        with torch.cuda.device(dev):
            _check(
                cp.is_cuda and cp.dtype == _I32 and cp.is_contiguous(),
                "cp must be a contiguous int32 CUDA tensor",
            )
            _check(
                sp.is_cuda
                and sp.dtype == _I32
                and q_l.dtype == _I32
                and sp.numel() == q_l.numel()
                and ds.numel() == sp.numel()
                and de.numel() == sp.numel(),
                "sp/qL/ds/de must be equal-length int32 CUDA tensors",
            )
            nwin = sp.numel()
            device = cp.device
            lo = torch.empty(nwin, dtype=_I32, device=device)
            hi = torch.empty(nwin, dtype=_I32, device=device)
            nosafe = torch.zeros(1, dtype=_I32, device=device)
            if nwin == 0:
                return lo, hi, nosafe
            self._launch(
                dev, "k_o2k_win_extents", _grid(nwin, 64),
                [_p(cp), _p(tab), _p(sp), _p(q_l), _p(ds), _p(de),
                 _i(nwin), _i(dmax), _b(contractions), _i(mode),
                 _p(lo), _p(hi), _p(nosafe)],
                block=64,
            )
            return lo, hi, nosafe

    def o200k_win_apply(
        self,
        cp: torch.Tensor,
        tab: torch.Tensor,
        starts: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        q_l: torch.Tensor,
        de: torch.Tensor,
        dmax: int,
        contractions: bool,
        mode: int,
    ) -> None:
        dev = self._dev(cp)
        with torch.cuda.device(dev):
            _check(
                starts.is_cuda and starts.dtype == torch.bool,
                "starts must be a bool CUDA tensor",
            )
            _check(
                lo.is_cuda
                and lo.dtype == _I32
                and hi.numel() == lo.numel()
                and q_l.numel() == lo.numel()
                and de.numel() == lo.numel(),
                "lo/hi/qL/de must be equal-length int32 CUDA tensors",
            )
            nwin = lo.numel()
            if nwin == 0:
                return
            self._launch(
                dev, "k_o2k_win_clear", _grid(nwin, 64),
                [_p(lo), _p(hi), _i(nwin), _p(starts)],
                block=64,
            )
            self._launch(
                dev, "k_o2k_win_mark", _grid(nwin, 64),
                [_p(cp), _p(tab), _p(lo), _p(q_l), _p(de), _i(nwin),
                 _i(dmax), _b(contractions), _i(mode), _p(starts)],
                block=64,
            )

    def encode_fused_o200k(
        self,
        data: torch.Tensor,
        nb_dev: torch.Tensor,
        tab: torch.Tensor,
        dmax: int,
        contractions: bool,
        pair_keys: torch.Tensor,
        pair_vals: torch.Tensor,
        byte_id: torch.Tensor,
        vocab_keys: torch.Tensor,
        vocab_vals: torch.Tensor,
        vocab_blob: torch.Tensor,
        ignore_merges: int,
        unsafe_bits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of encode_fused_o200k_impl: capture-safe, meta readback."""
        dev = self._dev(data)
        with torch.cuda.device(dev):
            _check(
                data.is_cuda and data.dtype == _U8 and data.is_contiguous(),
                "bytes must be a contiguous uint8 CUDA tensor",
            )
            _check(
                nb_dev.is_cuda
                and nb_dev.dtype == _I32
                and nb_dev.numel() == 1,
                "nb_dev must be a one-element int32 CUDA tensor",
            )
            _check(
                nb_dev.device == data.device,
                "nb_dev must live on bytes' device",
            )
            _check(
                tab.is_cuda
                and tab.device == data.device
                and tab.dtype == _U8
                and tab.is_contiguous()
                and tab.numel() >= 0x110000,
                "tab must be a contiguous uint8 CUDA table covering U+10FFFF",
            )
            self._check_bpe_tables_meta(
                data, pair_keys, pair_vals, byte_id,
                vocab_keys, vocab_vals, vocab_blob,
            )
            cap = data.numel()
            _check(0 < cap < _INT32_MAX, "capacity must be in (0, 2^31)")
            device = data.device
            gs_b = _grid(cap)

            # ---- UTF-8 decode ----
            lead = (data & 0xC0) != 0x80
            cpos = _cumsum_i32(lead.to(_I32))
            cp = torch.empty(cap, dtype=_I32, device=device)
            bo = torch.empty(cap, dtype=_I32, device=device)
            self._launch(
                dev, "k_utf8_decode", gs_b,
                [_p(data), _p(cpos), _p(cp), _p(bo), _i(cap), _p(None)],
            )
            d_c = _pa(cpos.data_ptr() + 4 * (cap - 1))

            # ---- o2k four-stream segmentation (head tails cleared) ----
            cls = torch.empty(cap, dtype=_U8, device=device)
            heads4 = torch.zeros(4 * cap, dtype=_U8, device=device)
            h_base = heads4.data_ptr()
            self._launch(
                dev, "k_o2k_heads", gs_b,
                [_p(cp), _p(tab), _i(0), d_c, _p(None), _p(cls),
                 _pa(h_base), _pa(h_base + cap), _pa(h_base + 2 * cap),
                 _pa(h_base + 3 * cap), _b(False)],
            )
            rids4 = torch.empty(4 * cap, dtype=_I32, device=device)
            r_base = rids4.data_ptr()
            for k in range(4):
                torch.cumsum(
                    heads4[k * cap:(k + 1) * cap].to(_I32), 0,
                    out=rids4[k * cap:(k + 1) * cap],
                )
            rid_m = _pa(r_base)
            rid_cs = _pa(r_base + 4 * cap)
            rid_lw = _pa(r_base + 8 * cap)
            rid_pm = _pa(r_base + 12 * cap)

            # ---- per-run channels at the cap bound ----
            run_start = torch.empty(cap, dtype=_I32, device=device)
            chan_hi = torch.full((7 * cap,), _HI, dtype=_I32, device=device)
            chan_neg = torch.full((4 * cap,), -1, dtype=_I32, device=device)
            p_hi = chan_hi.data_ptr()
            p_neg = chan_neg.data_ptr()
            first_anchor = _pa(p_hi)
            s_fl = _pa(p_hi + 4 * cap)
            p_fl = _pa(p_hi + 8 * cap)
            f_l0 = _pa(p_hi + 12 * cap)
            f_l1 = _pa(p_hi + 16 * cap)
            f_l2 = _pa(p_hi + 20 * cap)
            pm_fp = _pa(p_hi + 24 * cap)
            last_m_pm = _pa(p_neg)
            s_lc = _pa(p_neg + 4 * cap)
            last_l = _pa(p_neg + 8 * cap)
            last_c = _pa(p_neg + 12 * cap)
            self._launch(
                dev, "k_o2k_runinfo1", gs_b,
                [_p(cp), _p(cls), _pa(h_base), rid_m, rid_cs, rid_pm,
                 _i(0), d_c, _p(None), _p(run_start), first_anchor,
                 last_m_pm, _b(False)],
            )
            self._launch(
                dev, "k_o2k_runinfo2", gs_b,
                [_p(cp), _p(cls), rid_m, rid_cs, rid_lw, rid_pm,
                 _p(run_start), first_anchor, _i(0), d_c, s_fl, s_lc,
                 p_fl, last_l, last_c, f_l0, f_l1, f_l2, pm_fp,
                 _b(False)],
            )

            # ---- rules + sparse-case flags ----
            starts = torch.zeros(cap, dtype=torch.bool, device=device)
            trig2 = torch.zeros(2 * cap, dtype=_U8, device=device)
            t_base = trig2.data_ptr()
            flags2 = torch.zeros(2, dtype=_I32, device=device)
            self._launch(
                dev, "k_o2k_rules", gs_b,
                [_p(cp), _p(cls), rid_m, rid_cs, rid_lw, rid_pm,
                 _p(run_start), first_anchor, s_fl, s_lc, p_fl, last_l,
                 last_c, f_l0, f_l1, f_l2, pm_fp, last_m_pm, _i(0),
                 _i(dmax), _b(contractions), _p(starts), _pa(t_base),
                 _pa(t_base + cap), _p(flags2), _i(0), d_c,
                 _pa(flags2.data_ptr() + 4), _p(None), _b(False),
                 _p(None)],
            )

            # ---- piece bounds and dispatch ----
            pb = torch.empty(cap + 1, dtype=_I32, device=device)
            d_cnts = torch.empty(4, dtype=_I32, device=device)
            d_p = _p(d_cnts)
            self._select(dev, bo, starts, _p(pb), d_p, cap)
            self._launch(
                dev, "k_pb_sentinel", 1,
                [_p(pb), d_p, _p(nb_dev)], block=1,
            )
            flags3 = torch.empty(3 * cap, dtype=_U8, device=device)
            cnt = torch.zeros(cap, dtype=_I32, device=device)
            f_base = flags3.data_ptr()
            self._launch(
                dev, "k_dispatch_flags", gs_b,
                [_p(pb), d_p, _pa(f_base), _pa(f_base + cap),
                 _pa(f_base + 2 * cap), _p(cnt), _i(cap)],
            )
            lists = torch.empty(3 * cap, dtype=_I32, device=device)
            l_base = lists.data_ptr()
            for k in range(3):
                self._select(
                    dev, None, flags3[k * cap:(k + 1) * cap],
                    _pa(l_base + 4 * k * cap),
                    _pa(d_cnts.data_ptr() + 4 * (k + 1)),
                    cap,
                )

            # ---- BPE at capacity-bound geometry ----
            scratch = torch.empty(cap, dtype=_I32, device=device)
            buf_b = torch.empty(cap, dtype=_I32, device=device)
            pmask = pair_keys.numel() - 1
            vmask = vocab_keys.numel() - 1
            ub = self._unsafe_bits(unsafe_bits, data)
            self._launch(
                dev, "k_bpe_thread_cap32", gs_b,
                [
                    _p(data), _p(pb), _pa(l_base), _i(0),
                    _p(pair_keys), _p(pair_vals), _u(pmask),
                    _p(byte_id), _p(vocab_keys), _p(vocab_vals),
                    _u(vmask), _p(vocab_blob), _i(ignore_merges),
                    _p(scratch), _p(cnt),
                    _pa(d_cnts.data_ptr() + 4),
                    _p(None), _p(None), _p(None), _p(None), _u(0),
                ],
            )
            gw = min(gs_b, 8192)
            self._launch(
                dev, "k_bpe_warp", gw,
                [
                    _p(data), _p(pb), _pa(l_base + 4 * cap), _i(0),
                    _pa(d_cnts.data_ptr() + 8),
                    _p(pair_keys), _p(pair_vals), _u(pmask),
                    _p(byte_id), _p(vocab_keys), _p(vocab_vals),
                    _u(vmask), _p(vocab_blob), _i(ignore_merges),
                    _p(scratch), _p(cnt), _p(ub),
                ],
            )
            gl = min(cap // (MED_MAX + 1) + 1, 8192)
            self._launch(
                dev, "k_bpe_long", gl,
                [
                    _p(data), _p(pb), _pa(l_base + 8 * cap),
                    _p(pair_keys), _p(pair_vals), _u(pmask),
                    _p(byte_id), _p(vocab_keys), _p(vocab_vals),
                    _u(vmask), _p(vocab_blob), _i(ignore_merges),
                    _p(scratch), _p(buf_b), _p(cnt),
                    _i(0), _pa(d_cnts.data_ptr() + 12), _p(ub),
                ],
            )

            # ---- prefix sum + compaction + meta packing ----
            off = torch.zeros(cap + 1, dtype=_I32, device=device)
            torch.cumsum(cnt, 0, out=off[1:])
            out = torch.empty(cap, dtype=_I32, device=device)
            self._launch(
                dev, "k_bpe_compact", min(cap, 65535),
                [_p(pb), _p(off), _p(scratch), _p(out), _i(0), d_p],
                block=64,
            )
            meta = torch.empty(3, dtype=_I32, device=device)
            self._launch(
                dev, "k_o2k_meta3", 1,
                [_pa(off.data_ptr() + 4 * cap), _p(flags2), _p(meta)],
                block=1,
            )
            return out, meta
