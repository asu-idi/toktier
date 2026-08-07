"""Fused CUDA pre-tokenization for the cl100k and three-splitter bands.

This is the split layer: text (as codepoints, or as UTF-8 bytes decoded
on the device) in, piece starts out. The BPE layer on top of it lives in
``encoder.py``.

The semantics are pinned to the reference tokenizer's own splitter, not
to a re-derivation of it: the character-class tables come from probing
the reference engine, and the kernel implements a per-character
reformulation of the splitter's alternatives that was checked against the
reference engine piece for piece.

Three rulesets share this class, selected by the family routing data:

``cl100k``
    The GPT-style seven-alternative splitter.
``laguna``
    The same rule body plus a stage-0 pass that cuts before newline runs;
    the extra cut points are OR-ed into the kernel's document-boundary
    channel.
``deepseek``
    A three-splitter ruleset with its own seven-class table. Its inline
    kernel constants are cross-checked against the table's metadata at
    construction time, so a transcription error fails immediately instead
    of mis-splitting a rare codepoint in silence.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ...errors import UnsupportedConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .class_tables import LoadedClassTable
    from .families import KernelFamily
    from .options import GpuOptions

__all__ = ["CudaPretok"]

#: Rulesets that share the cl100k rule body and class table.
_CL100K_RULESETS = frozenset({"cl100k", "laguna"})


class CudaPretok:
    """Piece-start computation on the device.

    Args:
        ext: the loaded kernel extension module (see ``loader.py``).
            It is injected rather than loaded here, because a process has
            exactly one kernel build and the loader owns it.
        class_table: the verified character-class table for this family.
        digits_max: maximum digits per piece.
        ruleset: ``cl100k``, ``laguna`` or ``deepseek``.
        device: CUDA device string.
        table_meta: the class table's metadata sidecar. Required for the
            three-splitter ruleset, whose constants are cross-checked.
    """

    def __init__(
        self,
        ext: Any,
        class_table: np.ndarray[Any, Any],
        *,
        digits_max: int,
        ruleset: str = "cl100k",
        device: str = "cuda:0",
        table_meta: dict[str, Any] | None = None,
    ) -> None:
        if ruleset not in _CL100K_RULESETS and ruleset != "deepseek":
            raise UnsupportedConfig(
                f"unknown pre-tokenization ruleset {ruleset!r}",
                details={
                    "option": "ruleset",
                    "value": ruleset,
                    "reason": "not implemented by this kernel",
                },
            )
        self.ext = ext
        self.ruleset = ruleset
        self.dmax = int(digits_max)
        self.dev = torch.device(device)
        self.table_np = class_table
        self.table = torch.from_numpy(class_table).to(self.dev)
        self.table_meta = table_meta
        if ruleset == "deepseek":
            if table_meta is None:
                raise UnsupportedConfig(
                    "the three-splitter ruleset needs its class-table metadata",
                    details={
                        "option": "table_meta",
                        "value": None,
                        "reason": "constants cross-check is mandatory",
                    },
                )
            self._check_inline_constants(table_meta)

    @classmethod
    def from_family(
        cls,
        ext: Any,
        table: LoadedClassTable,
        *,
        family: KernelFamily,
        digits_max: int,
        options: GpuOptions,
    ) -> CudaPretok:
        """Uniform construction hook for the engine's data-driven dispatch.

        Every pretokenizer entry point exposes this signature, so the
        engine can build any band's split layer without knowing which
        arguments the concrete class needs.
        """
        return cls(
            ext,
            table.array,
            digits_max=digits_max,
            ruleset=family.ruleset,
            device=options.device,
            table_meta=table.meta,
        )

    # -- constants cross-check -----------------------------------------

    def _check_inline_constants(self, meta: dict[str, Any]) -> None:
        """Compare the kernel's inline constants with the table metadata.

        The three-splitter kernel carries the codepoint ranges, the
        punctuation set, the alphabet ranges, the leading-space codepoint
        and the class enumeration as compile-time constants, while the
        table generator extracts the same values mechanically from the
        artifact's own patterns. Hand-copied interval constants are
        exactly the kind of thing that goes wrong on a rare codepoint and
        stays invisible; comparing them here turns that into an immediate
        failure.
        """
        reported = json.loads(self.ext.ds_constants())
        alphabet = [
            code
            for low, high in meta["alpha_ranges"]
            for code in range(low, high + 1)
        ]
        mismatches: list[str] = []
        if reported["cjk_ranges"] != [list(item) for item in meta["cjk_ranges"]]:
            mismatches.append("cjk_ranges")
        if reported["apunct"] != list(meta["apunct"]):
            mismatches.append("apunct")
        if reported["alpha"] != alphabet:
            mismatches.append("alpha")
        if reported["a3_space"] != meta["a3_space"]:
            mismatches.append("a3_space")
        if reported["crlf_cps"] != list(meta["crlf_cps"]):
            mismatches.append("crlf_cps")
        if reported["class_enum"] != meta["enum"]:
            mismatches.append("class_enum")
        if mismatches:
            raise UnsupportedConfig(
                "kernel inline constants disagree with the class table: "
                + ", ".join(mismatches),
                details={
                    "option": "class_table",
                    "value": meta.get("table"),
                    "reason": "constant drift between kernel and table",
                },
            )

    # -- device helpers -------------------------------------------------

    def encode_str(self, text: str) -> torch.Tensor:
        """String to an int32 codepoint tensor on the device."""
        codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        return torch.from_numpy(codepoints.astype(np.int32)).to(self.dev)

    def utf8_to_cp(self, data: torch.Tensor) -> torch.Tensor:
        """Decode UTF-8 bytes to codepoints on the device."""
        result: torch.Tensor = self.ext.utf8_to_cp(data)
        return result

    # -- piece starts ---------------------------------------------------

    def starts(self, codepoints: torch.Tensor) -> torch.Tensor:
        """Codepoints to a boolean piece-start mask."""
        if self.ruleset == "deepseek":
            out: torch.Tensor = self.ext.pretok_starts_ds(
                codepoints, self.table, self.dmax
            )
            return out
        if self.ruleset == "laguna":
            out = self.ext.pretok_starts_laguna(codepoints, self.table, self.dmax)
            return out
        out = self.ext.pretok_starts(codepoints, self.table, self.dmax)
        return out

    def starts_batched(
        self, codepoints: torch.Tensor, doc_offsets: torch.Tensor
    ) -> torch.Tensor:
        """Piece starts for several documents in one launch.

        ``doc_offsets`` holds each document's first codepoint index in the
        concatenated buffer and must contain 0. The kernel guarantees a
        piece start at every document offset, so no piece can straddle a
        document boundary.
        """
        marks = torch.zeros(codepoints.numel(), dtype=torch.uint8, device=self.dev)
        marks[doc_offsets.long()] = 1
        if self.ruleset == "deepseek":
            out: torch.Tensor = self.ext.pretok_starts_batched_ds(
                codepoints, marks, self.table, self.dmax
            )
            return out
        if self.ruleset == "laguna":
            out = self.ext.pretok_starts_batched_laguna(
                codepoints, marks, self.table, self.dmax
            )
            return out
        out = self.ext.pretok_starts_batched(
            codepoints, marks, self.table, self.dmax
        )
        return out

    # -- convenience split forms ----------------------------------------

    def split_utf8(self, text: str) -> list[tuple[int, int]]:
        """Piece boundaries in codepoint coordinates, decoding on device."""
        if not text:
            return []
        buffer = torch.frombuffer(
            bytearray(text.encode()), dtype=torch.uint8
        ).to(self.dev)
        codepoints = self.utf8_to_cp(buffer)
        mask = self.starts(codepoints)
        index = torch.nonzero(mask).flatten().cpu().tolist()
        return list(zip(index, [*index[1:], codepoints.numel()], strict=True))

    def split_docs(self, docs: list[str]) -> list[list[tuple[int, int]]]:
        """Per-document piece boundaries, in per-document coordinates."""
        lengths = [len(doc) for doc in docs]
        joined = "".join(docs)
        if not joined:
            return [[] for _ in docs]
        codepoints = self.encode_str(joined)
        offsets: list[int] = []
        running = 0
        for length in lengths:
            offsets.append(running)
            running += length
        doc_offsets = torch.tensor(
            sorted(
                {
                off
                for off, length in zip(offsets, lengths, strict=True)
                if length > 0
            }
            | {0}
            ),
            dtype=torch.int32,
            device=self.dev,
        )
        mask = self.starts_batched(codepoints, doc_offsets)
        index = torch.nonzero(mask).flatten().cpu().tolist()
        bounds = list(zip(index, [*index[1:], len(joined)], strict=True))
        # Single-pass assignment, linear in pieces plus documents: the
        # boundaries are ascending, each document is contiguous in the
        # joined buffer, and the kernel guarantees a start at every
        # document offset, so no piece straddles a boundary. Scanning all
        # boundaries per document instead would be quadratic and became
        # the host-side bottleneck at batch throughput.
        out: list[list[tuple[int, int]]] = [[] for _ in docs]
        ends = [
            off + length for off, length in zip(offsets, lengths, strict=True)
        ]
        cursor = 0
        for lo, hi in bounds:
            while cursor < len(docs) and lo >= ends[cursor]:
                cursor += 1  # step over finished and zero-length documents
            if cursor < len(docs) and lo >= offsets[cursor] and hi <= ends[cursor]:
                out[cursor].append((lo - offsets[cursor], hi - offsets[cursor]))
        return out
