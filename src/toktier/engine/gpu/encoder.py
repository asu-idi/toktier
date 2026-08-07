"""End-to-end GPU encoders for the cl100k and three-splitter bands.

Chain: UTF-8 bytes -> device decode (codepoints plus the byte offset of
each character) -> piece starts -> piece byte boundaries -> byte-level
BPE merge inside each piece -> token ids.

Two forms live here:

:class:`GpuTokenizer`
    The eager form. Each stage is a separate extension call, which makes
    it the easiest form to reason about and the one the batched channel
    builds on.
:class:`FusedGpuTokenizer`
    The whole chain in one extension call. Every count stays on the
    device, so there is no host synchronisation inside the call and the
    geometry depends only on the buffer capacity, which makes the call
    capturable into a CUDA Graph. Static buffers are bucketed by size and
    padded with a UTF-8 continuation byte, which decodes to nothing.

Both forms produce the same ids; the fused form is a delivery choice.

Normalization
-------------
Artifacts either declare no normalizer, an empty normalizer sequence
(equivalent to none), or NFC. Anything else is refused at construction.

For NFC families the text has to be normalized before the bytes are
encoded, and it has to be normalized the way the *reference tokenizer*
does it. Its normalizer and its regex engine were built against different
Unicode versions, so a third-party normalizer -- including the standard
library's -- will happily apply compositions the reference engine does
not know, and the ids then differ. The rule is therefore: a quick check
may prove that no work is needed, but any actual normalization goes
through the reference package's own normalizer.

The quick check itself runs on the device: a table lookup plus an
adjacent combining-class order check over the bytes that are being copied
to the device anyway. Passing the check proves the reference normalizer
is the identity on this text, so the bytes can be encoded as they are.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ...errors import UnsupportedConfig
from ...frontend.added import AddedTokenFrontendProtocol as AddedTokenFrontend
from .class_tables import ClassTableStore, LoadedClassTable
from .options import DEFAULT_GPU_OPTIONS, GpuOptions
from .pretok import CudaPretok

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...kernels.bpe_tables import BpeTables
    from .families import KernelFamily

__all__ = [
    "NFC_QUICK_CHECK_ROLE",
    "AddedTokenFrontend",
    "FusedGpuTokenizer",
    "GpuTokenizer",
]

#: Role name of the NFC quick-check table in the routing data.
NFC_QUICK_CHECK_ROLE = "nfc_quick_check"


def _post_processor_adds_tokens(section: Any) -> bool:
    """Whether an artifact's post-processor can insert tokens.

    ``None`` and plain ``ByteLevel`` insert nothing; a ``Sequence``
    inserts exactly when one of its members does. Anything else is
    treated as inserting, which only routes ``add_special_tokens=True``
    calls to the reference backend -- never a wrong id.
    """
    if section is None:
        return False
    if not isinstance(section, dict):
        return True
    kind = section.get("type")
    if kind == "ByteLevel":
        return False
    if kind == "Sequence":
        return any(
            _post_processor_adds_tokens(member)
            for member in section.get("processors") or []
        )
    return True


def _read_artifact(artifact_dir: Path) -> tuple[bool, list[str], bool]:
    """``(needs_nfc, added_token_literals, adds_special_tokens)``."""
    document = json.loads(
        (artifact_dir / "tokenizer.json").read_text(encoding="utf-8")
    )
    normalizer = document.get("normalizer")
    empty_sequence = (
        isinstance(normalizer, dict)
        and normalizer.get("type") == "Sequence"
        and not normalizer.get("normalizers")
    )
    known = (
        normalizer is None
        or empty_sequence
        or normalizer.get("type") == "NFC"
    )
    if not known:
        raise UnsupportedConfig(
            f"unsupported normalizer {normalizer!r}",
            details={
                "option": "normalizer",
                "value": normalizer,
                "reason": "only none, an empty sequence, or NFC are certified",
            },
        )
    needs_nfc = normalizer is not None and normalizer.get("type") == "NFC"
    literals = [
        entry["content"] for entry in document.get("added_tokens", []) or []
    ]
    adds_special = _post_processor_adds_tokens(document.get("post_processor"))
    return needs_nfc, literals, adds_special


class GpuTokenizer:
    """Eager end-to-end encoder for one family.

    Instances are not thread-safe: an encoder reuses device buffers,
    graph captures, pinned host staging and memo arrays across calls,
    so concurrent calls on one instance can overwrite each other's
    state. Use one encoder instance per thread; instances for the same
    family share the loaded tables through the engine's stores, so a
    per-thread instance costs little beyond its own buffers.
    """

    def __init__(
        self,
        *,
        ext: Any,
        family: KernelFamily,
        artifact_dir: Path,
        bpe: BpeTables,
        class_tables: ClassTableStore,
        options: GpuOptions = DEFAULT_GPU_OPTIONS,
        frontend: AddedTokenFrontend | None = None,
    ) -> None:
        self.name = family.name
        self.family = family
        self.ext = ext
        self.options = options
        self.dev = torch.device(options.device)

        table = class_tables.load(family.class_table)
        self.pre = self._build_pretok(ext, table, family, options)

        # torch has no unsigned 64-bit dtype: the arrays are moved as
        # int64 with the bits preserved, and the extension reads them
        # back as uint64.
        def as_u64(array: np.ndarray[Any, Any]) -> torch.Tensor:
            return torch.from_numpy(array.view(np.int64)).to(self.dev)

        self.pair_keys = as_u64(bpe.pair_keys)
        self.pair_vals = as_u64(bpe.pair_vals)
        self.vocab_keys = as_u64(bpe.vocab_keys)
        self.vocab_vals = as_u64(bpe.vocab_vals)
        self.byte_id = torch.from_numpy(bpe.byte_id).to(self.dev)
        self.blob = torch.from_numpy(bpe.vocab_blob).to(self.dev)
        self.ignore_merges = int(bpe.ignore_merges)

        # Guard bitmap for merge tables that are not rank-monotone. It is
        # unconditional: there is no switch to run without it, because
        # running without it can produce ids that differ from the
        # reference. Families with a monotone table pass an empty tensor,
        # which selects the kernel's null-pointer branch, so their
        # behaviour is unchanged.
        self.unsafe_bits = (
            torch.from_numpy(bpe.unsafe_bits.view(np.int32)).to(self.dev)
            if bpe.unsafe_bits is not None
            else torch.empty(0, dtype=torch.int32, device=self.dev)
        )

        self.nfc, self.added, self.adds_special_tokens = _read_artifact(
            artifact_dir
        )
        self._qc_table: torch.Tensor | None = None
        self._qc_table_np: np.ndarray[Any, Any] | None = None
        self._reference_nfc: Any | None = None
        if self.nfc:
            qc = class_tables.load_role(NFC_QUICK_CHECK_ROLE)
            self._qc_table_np = qc.array
            self._qc_table = torch.from_numpy(qc.array).to(self.dev)
            from tokenizers import normalizers

            self._reference_nfc = normalizers.NFC()

        self._frontend = frontend if options.added_token_frontend else None

        self.memo: tuple[torch.Tensor, ...] | None = None
        if options.piece_memoization:
            slots = 1 << 20
            self.memo = (
                torch.zeros(slots, dtype=torch.int64, device=self.dev),
                torch.zeros(slots, dtype=torch.int32, device=self.dev),
                torch.zeros(slots * 16, dtype=torch.uint8, device=self.dev),
                torch.zeros(slots * 8, dtype=torch.int32, device=self.dev),
            )

    # -- band hook ------------------------------------------------------

    @staticmethod
    def resolve_digits_max(
        family: KernelFamily, table: LoadedClassTable
    ) -> int:
        """The family's digits-max, deferring to the class table metadata.

        A family whose splitter pattern carries the value gets it from
        the table's metadata sidecar, which extracted it mechanically
        from the artifact's own pattern; keeping a second copy in the
        routing data would be a second source of truth.
        """
        if family.digits_max is not None:
            return int(family.digits_max)
        meta = table.meta or {}
        if "digits_max" not in meta:
            raise UnsupportedConfig(
                f"family {family.name!r} defers digits_max to the class "
                "table, but the table has no metadata sidecar",
                details={
                    "option": "digits_max",
                    "value": None,
                    "reason": "missing class table metadata",
                },
            )
        return int(meta["digits_max"])

    def _build_pretok(
        self,
        ext: Any,
        table: LoadedClassTable,
        family: KernelFamily,
        options: GpuOptions,
    ) -> Any:
        """Construct the band's piece-start object. Overridden per band."""
        return CudaPretok(
            ext,
            table.array,
            digits_max=self.resolve_digits_max(family, table),
            ruleset=family.ruleset,
            device=options.device,
            table_meta=table.meta,
        )

    # -- core encode path ----------------------------------------------

    def encode_dev(self, data: torch.Tensor) -> torch.Tensor:
        """Device byte tensor to a device token-id tensor."""
        if data.numel() == 0:
            return torch.empty(0, dtype=torch.int32, device=self.dev)
        codepoints, byte_offsets = self.ext.utf8_to_cp_bo(data)
        mask = self.pre.starts(codepoints)
        char_starts = torch.nonzero(mask).flatten()
        piece_bounds = torch.cat(
            [
                byte_offsets[char_starts],
                torch.tensor(
                    [data.numel()], dtype=torch.int32, device=self.dev
                ),
            ]
        ).contiguous()
        if self.memo is not None:
            ids, _offsets = self.ext.bpe_encode_memo(
                data,
                piece_bounds,
                self.pair_keys,
                self.pair_vals,
                self.byte_id,
                self.vocab_keys,
                self.vocab_vals,
                self.blob,
                self.ignore_merges,
                *self.memo,
                self.unsafe_bits,
            )
        else:
            ids, _offsets = self.ext.bpe_encode(
                data,
                piece_bounds,
                self.pair_keys,
                self.pair_vals,
                self.byte_id,
                self.vocab_keys,
                self.vocab_vals,
                self.blob,
                self.ignore_merges,
                self.unsafe_bits,
            )
        result: torch.Tensor = ids
        return result

    # -- public surface -------------------------------------------------

    def _frontend_plan(self, text: str) -> Any | None:
        """The frontend's split plan, or ``None`` when it is not in use."""
        return self._frontend.scan(text) if self._frontend is not None else None

    def encode(self, text: str) -> list[int]:
        """Token ids for one document."""
        if not text:
            return []
        plan = self._frontend_plan(text)
        if plan is not None and self._frontend is not None:
            return self._frontend.assemble(plan, self._encode_plain)
        return self._encode_plain(text)

    def _encode_plain(self, text: str) -> list[int]:
        text = self.normalize(text)
        buffer = torch.frombuffer(
            bytearray(text.encode()), dtype=torch.uint8
        ).to(self.dev)
        result: list[int] = self.encode_dev(buffer).cpu().tolist()
        return result

    def encode_np(self, text: str) -> np.ndarray[Any, Any]:
        """Token ids as a numpy array: one device-to-host copy, no list."""
        if not text:
            return np.empty(0, dtype=np.int32)
        plan = self._frontend_plan(text)
        if plan is not None and self._frontend is not None:
            return np.asarray(
                self._frontend.assemble(plan, self._encode_plain), dtype=np.int32
            )
        text = self.normalize(text)
        buffer = torch.frombuffer(
            bytearray(text.encode()), dtype=torch.uint8
        ).to(self.dev)
        out: np.ndarray[Any, Any] = self.encode_dev(buffer).cpu().numpy()
        return out

    # -- normalization --------------------------------------------------

    def normalize(self, text: str) -> str:
        """Apply the artifact's normalizer, doing nothing when possible.

        ASCII text is normalized by definition, so it short-circuits.
        Otherwise a standard-library normalization check is used only as
        a *proof that no work is needed*: its Unicode version is newer
        than the reference normalizer's, so anything it considers already
        normalized is a fixed point of the reference normalizer too. When
        it says work is needed, the work is done by the reference
        package's normalizer, never by the standard library's.
        """
        if not self.nfc or text.isascii():
            return text
        import unicodedata

        if unicodedata.is_normalized("NFC", text):
            return text
        assert self._reference_nfc is not None
        normalized: str = self._reference_nfc.normalize_str(text)
        return normalized

    def _quick_check_flag(self, buffer: torch.Tensor) -> torch.Tensor:
        """Device UTF-8 buffer to a one-element flag: non-zero means fail.

        One pass on the device: decode the lead byte in place, gather the
        whole-plane quick-check table, and compare adjacent combining
        classes for an ordering violation. No host synchronisation, and
        the geometry depends only on the buffer capacity, so it can be
        captured into a graph together with the encode. Padding with a
        continuation byte is safe: a continuation byte is never a lead
        byte, so padding is silent.
        """
        flag: torch.Tensor = self.ext.nfc_qc_scan(buffer, self._qc_table)
        return flag

    def _exact_normalization(self, text: str) -> str | None:
        """Decide precisely after the quick check fired.

        ``None`` means the text is already normalized, so ids computed
        from the original bytes are valid and nothing has to be redone.

        The quick check is deliberately conservative: it fires on the
        whole maybe-class, which includes combining marks that are in
        fact blocked and therefore do not compose. This refines that
        decision by cutting the text at safe starters (combining class
        zero, never a composition target). No composition and no
        reordering can cross such a cut, so normalization distributes
        over the segments: segments with no flagged codepoint are
        normalized by the table's own predicate, and only the flagged
        segments, usually tiny, go to the reference normalizer.
        """
        reference = self._reference_nfc
        assert reference is not None

        def whole(value: str) -> str | None:
            result: str = reference.normalize_str(value)
            return None if result == value else result

        table = self._qc_table_np
        if table is None:  # pragma: no cover - only reachable if misconfigured
            return whole(text)
        codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        classes = table[codepoints]
        flagged = classes == 255
        if classes.size >= 2:
            flagged[1:] |= (classes[:-1] > classes[1:]) & (classes[1:] != 0)
        flagged_index = np.flatnonzero(flagged)
        if flagged_index.size == 0:
            # Defensive: the device check fired and the host disagrees.
            return whole(text)
        safe_index = np.flatnonzero(classes == 0)
        if safe_index.size == 0:
            # No safe cut point at all, for example an all-mark string.
            return whole(text)
        left = np.searchsorted(safe_index, flagged_index, side="right") - 1
        starts = np.where(
            left >= 0, safe_index[np.clip(left, 0, None)], 0
        )
        right = np.searchsorted(safe_index, flagged_index, side="right")
        ends = np.where(
            right < safe_index.size,
            safe_index[np.clip(right, None, safe_index.size - 1)],
            len(text),
        )
        segments: list[tuple[int, int]] = []
        for lo, hi in zip(starts.tolist(), ends.tolist(), strict=True):
            if segments and lo <= segments[-1][1]:
                segments[-1] = (segments[-1][0], max(segments[-1][1], hi))
            else:
                segments.append((lo, hi))
        for lo, hi in segments:
            segment = text[lo:hi]
            if reference.normalize_str(segment) != segment:
                return whole(text)
        return None


class FusedGpuTokenizer(GpuTokenizer):
    """Fused single-request encoder, optionally CUDA-Graph captured.

    The fused entry takes the whole bytes-to-ids chain in one call: the
    per-stage counts stay on the device, so there is no host
    synchronisation inside the call, and the launch geometry depends only
    on the buffer capacity. That makes the call capturable.

    Buffers are static and bucketed from 4 KiB to 4 MiB; the tail is
    padded with a UTF-8 continuation byte, which decodes to nothing. Each
    bucket is captured once, lazily. After that a request costs two small
    host-to-device copies, a replay, a four-byte count read and one
    result copy back.
    """

    BUCKETS = (1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_graph = self.options.use_cuda_graph
        self._graphs: dict[int, tuple[Any, ...]] = {}
        self._pinned = torch.empty(
            self.BUCKETS[-1], dtype=torch.uint8
        ).pin_memory()
        self._pinned_np = self._pinned.numpy()

    def _call(
        self, buffer: torch.Tensor, n_bytes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ruleset = self.pre.ruleset
        if ruleset == "deepseek":
            entry = self.ext.encode_fused_ds
        elif ruleset == "laguna":
            entry = self.ext.encode_fused_laguna
        else:
            entry = self.ext.encode_fused
        out: tuple[torch.Tensor, torch.Tensor] = entry(
            buffer,
            n_bytes,
            self.pre.table,
            self.pre.dmax,
            self.pair_keys,
            self.pair_vals,
            self.byte_id,
            self.vocab_keys,
            self.vocab_vals,
            self.blob,
            self.ignore_merges,
            self.unsafe_bits,
        )
        return out

    def _graph_for(self, capacity: int) -> tuple[Any, ...]:
        if capacity not in self._graphs:
            buffer = torch.full(
                (capacity,), 0x61, dtype=torch.uint8, device=self.dev
            )  # 'a': placeholder content during capture
            n_bytes = torch.tensor(
                [capacity], dtype=torch.int32, device=self.dev
            )
            side = torch.cuda.Stream(self.dev)
            side.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(side):  # warm up before capturing
                for _ in range(3):
                    self._call(buffer, n_bytes)
                    if self._qc_table is not None:
                        self._quick_check_flag(buffer)
            torch.cuda.current_stream(self.dev).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # The quick check is captured with the encode, so it
                # costs no extra synchronisation.
                out, count = self._call(buffer, n_bytes)
                quick_check = (
                    self._quick_check_flag(buffer)
                    if self._qc_table is not None
                    else None
                )
            self._graphs[capacity] = (
                graph,
                buffer,
                n_bytes,
                out,
                count,
                quick_check,
            )
        return self._graphs[capacity]

    def _encode_raw(
        self, text: str, normalized: bool = False
    ) -> tuple[torch.Tensor, int]:
        """Shared path: returns the device output tensor and token count.

        For NFC families the quick check runs on the device and its flag
        is read back together with the token count, so it costs no extra
        synchronisation. A pass proves the reference normalizer is the
        identity here, so the original bytes were the right thing to
        encode. A failure goes to the exact decision on the host and, if
        that really changes the text, re-encodes the normalized form
        (NFC is idempotent, so the check is not repeated).
        """
        check_enabled = self._qc_table is not None and not normalized
        encoded = text.encode()
        n_bytes = len(encoded)
        if not self.use_graph or n_bytes > self.BUCKETS[-1]:
            buffer = torch.frombuffer(
                bytearray(encoded), dtype=torch.uint8
            ).to(self.dev)
            count_dev = torch.tensor(
                [n_bytes], dtype=torch.int32, device=self.dev
            )
            flag = self._quick_check_flag(buffer) if check_enabled else None
            out, count = self._call(buffer, count_dev)
            total = int(count.item())
            if flag is not None and bool(flag.item()):
                exact = self._exact_normalization(text)
                if exact is None:
                    return out, total
                return self._encode_raw(exact, normalized=True)
            return out, total

        capacity = next(size for size in self.BUCKETS if size >= n_bytes)
        graph, buffer, count_dev, out, count, quick_check = self._graph_for(
            capacity
        )
        self._pinned_np[:n_bytes] = np.frombuffer(encoded, dtype=np.uint8)
        buffer[:n_bytes].copy_(self._pinned[:n_bytes], non_blocking=True)
        if n_bytes < capacity:
            buffer[n_bytes:].fill_(0x80)
        count_dev.fill_(n_bytes)
        graph.replay()
        total = int(count.item())
        if check_enabled and bool(quick_check.item()):
            exact = self._exact_normalization(text)
            if exact is None:
                return out, total
            return self._encode_raw(exact, normalized=True)
        return out, total

    def _encode_plain(self, text: str) -> list[int]:
        out, count = self._encode_raw(text)
        result: list[int] = out[:count].cpu().tolist()
        return result

    def encode_np(self, text: str) -> np.ndarray[Any, Any]:
        """Token ids as a numpy array: one device-to-host copy."""
        if not text:
            return np.empty(0, dtype=np.int32)
        plan = self._frontend_plan(text)
        if plan is not None and self._frontend is not None:
            return np.asarray(
                self._frontend.assemble(plan, self._encode_plain), dtype=np.int32
            )
        out, count = self._encode_raw(text)
        result: np.ndarray[Any, Any] = out[:count].cpu().numpy()
        return result

    def encode_view(self, text: str) -> torch.Tensor:
        """Device-side delivery: a view into the graph's output buffer.

        The view aliases the captured buffer, so the next encode of the
        same bucket overwrites it. Consumers must finish with it first or
        clone it. A document routed through the frontend returns a freshly
        allocated tensor with no aliasing constraint.
        """
        if not text:
            return torch.empty(0, dtype=torch.int32, device=self.dev)
        plan = self._frontend_plan(text)
        if plan is not None and self._frontend is not None:
            ids = self._frontend.assemble(plan, self._encode_plain)
            return torch.tensor(ids, dtype=torch.int32, device=self.dev)
        out, count = self._encode_raw(text)
        return out[:count]
