"""Batched end-to-end encoding: many documents in one kernel pass.

Contract reference: ``docs/contracts/api.md`` Section 4 --
``encode_batch(texts, output="ragged")`` is frozen public API, and this
is the channel that implements it on the GPU backend.

Why a batched channel exists
----------------------------
Encoding one document at a time costs a launch and a synchronisation per
document. On a corpus whose mean document is a few kilobytes, that
per-request overhead dominates completely: the measured cost per small
document is a fraction of a millisecond, which over a corpus of billions
of documents is two orders of magnitude away from the kernel's streaming
rate. Concatenating a batch into one byte buffer and making one pass
brings it back to the streaming rate.

Why the result is still exactly per-document
--------------------------------------------
Three properties, all of which have to hold:

1. The batched piece-start entry marks a piece start at every document
   offset, so no piece straddles a document boundary.
2. BPE merging is closed inside a piece: the encoder solves each piece
   independently from the piece-boundary array, so the ids of a piece do
   not depend on its neighbours in the batch.
3. Therefore a document's ids are a contiguous slice of the batch's id
   array, delimited by the piece offsets of its first and last piece.

The argument does not replace measurement: the batched channel is checked
against per-document encoding element by element in the test suite.

Normalization discipline
------------------------
Each document is normalized *before* concatenation, matching the order
the reference tokenizer uses (truncate, then normalize), and through the
encoder's own normalizer path so that the reference package's normalizer
does the work. Normalizing the concatenation instead would be a different
function: a combining mark at the head of one document could compose with
the last character of the one before it.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .options import DEFAULT_GPU_OPTIONS, GpuOptions

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

    from .encoder import GpuTokenizer

__all__ = ["BatchedE2E", "digest_ids"]


def digest_ids(ids: np.ndarray[Any, Any] | list[int]) -> bytes:
    """The per-document id digest used by the evidence manifests.

    Little-endian unsigned 32-bit ids, concatenated, SHA-256, full 32
    bytes. Published document digests are computed exactly this way, so
    anything that reproduces a published number has to use this function.
    """
    return hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).digest()


class BatchedE2E:
    """Encode many documents in one pass over the device.

    Args:
        encoder: an end-to-end encoder for one family.
        options: batch size limits. A single document longer than the
            character limit becomes a batch of one.
    """

    def __init__(
        self,
        encoder: GpuTokenizer,
        *,
        options: GpuOptions = DEFAULT_GPU_OPTIONS,
        windowed_starts: bool = False,
    ) -> None:
        self.encoder = encoder
        self.ext = encoder.ext
        self.pre = encoder.pre
        self.dev = encoder.dev
        self.max_docs = options.max_batch_docs
        self.max_chars = options.max_batch_chars
        # The o200k band's batched entry needs the joined text as well,
        # because its sparse cases are resolved on the host.
        self._windowed_starts = windowed_starts

    def _starts(
        self, codepoints: torch.Tensor, doc_offsets: torch.Tensor, joined: str
    ) -> torch.Tensor:
        if self._windowed_starts:
            out: torch.Tensor = self.pre.starts_batched(
                codepoints, doc_offsets, text=joined
            )
            return out
        out = self.pre.starts_batched(codepoints, doc_offsets)
        return out

    # -- the pass -------------------------------------------------------

    def _encode_joined(
        self, docs: list[str]
    ) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ]:
        """One device pass, returning the pieces of a per-document split.

        Returns ``(ids, piece_offsets, low_piece, high_piece)`` where
        document ``i`` occupies
        ``ids[piece_offsets[low_piece[i]] : piece_offsets[high_piece[i]]]``.
        """
        normalized = [self.encoder.normalize(doc) for doc in docs]
        joined = "".join(normalized)
        lengths = [len(doc) for doc in normalized]
        encoded = joined.encode()
        n_bytes = len(encoded)
        buffer = torch.frombuffer(bytearray(encoded), dtype=torch.uint8).to(
            self.dev
        )
        codepoints, byte_offsets = self.ext.utf8_to_cp_bo(buffer)

        offsets: list[int] = []
        running = 0
        for length in lengths:
            offsets.append(running)
            running += length
        # Zero-length documents share an offset with their successor, and
        # a marked offset must appear once; index 0 is always marked.
        doc_offsets = torch.tensor(
            sorted(
                {
                    off
                    for off, length in zip(offsets, lengths, strict=True)
                    if length > 0
                }
                | {0}
            ),
            dtype=torch.long,
            device=self.dev,
        )
        mask = self._starts(codepoints, doc_offsets, joined)
        char_starts = torch.nonzero(mask).flatten()
        piece_bounds = torch.cat(
            [
                byte_offsets[char_starts],
                torch.tensor([n_bytes], dtype=torch.int32, device=self.dev),
            ]
        ).contiguous()
        ids, piece_offsets = self.ext.bpe_encode(
            buffer,
            piece_bounds,
            self.encoder.pair_keys,
            self.encoder.pair_vals,
            self.encoder.byte_id,
            self.encoder.vocab_keys,
            self.encoder.vocab_vals,
            self.encoder.blob,
            self.encoder.ignore_merges,
            self.encoder.unsafe_bits,  # non-monotone merge table guard
        )
        ids_np = ids.cpu().numpy()
        piece_offsets_np = piece_offsets.cpu().numpy().astype(np.int64)
        starts_np = char_starts.cpu().numpy().astype(np.int64)
        begin = np.asarray(offsets, dtype=np.int64)
        end = begin + np.asarray(lengths, dtype=np.int64)
        # side="left" is the semantics that makes an empty document map
        # to an empty slice: its begin and end coincide, so both land on
        # the same piece index.
        low_piece = np.searchsorted(starts_np, begin, side="left")
        high_piece = np.searchsorted(starts_np, end, side="left")
        return ids_np, piece_offsets_np, low_piece, high_piece

    def encode_batch(self, docs: list[str]) -> list[np.ndarray[Any, Any]]:
        """Per-document id arrays for one batch of non-empty documents."""
        if not docs:
            return []
        ids, piece_offsets, low_piece, high_piece = self._encode_joined(docs)
        out: list[np.ndarray[Any, Any]] = []
        for index in range(len(docs)):
            lo = int(piece_offsets[low_piece[index]])
            hi = int(piece_offsets[high_piece[index]])
            out.append(ids[lo:hi])
        return out

    def encode_batch_ragged(
        self, docs: list[str]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """The ragged shape of the public API: ``(values, offsets)``.

        ``values`` is uint32 with every row concatenated in order;
        ``offsets`` is int64 of length ``len(docs) + 1``, starts at zero,
        is non-decreasing, and ends at ``len(values)``. Row ``i`` is
        ``values[offsets[i]:offsets[i + 1]]``.
        """
        rows = self.encode_batch(docs)
        if not rows:
            return (
                np.empty(0, dtype=np.uint32),
                np.zeros(1, dtype=np.int64),
            )
        lengths = np.fromiter(
            (row.size for row in rows), dtype=np.int64, count=len(rows)
        )
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        values = np.concatenate(
            [row.astype(np.uint32, copy=False) for row in rows]
        )
        return values, offsets

    def digest_batch(self, docs: list[str]) -> list[bytes]:
        """Per-document id digests, in the published digest convention."""
        return [digest_ids(row) for row in self.encode_batch(docs)]

    # -- packing --------------------------------------------------------

    def pack(self, docs: list[str]) -> Iterator[list[str]]:
        """Split a document stream into batches under both limits."""
        current: list[str] = []
        chars = 0
        for doc in docs:
            if current and (
                len(current) >= self.max_docs
                or chars + len(doc) > self.max_chars
            ):
                yield current
                current, chars = [], 0
            current.append(doc)
            chars += len(doc)
        if current:
            yield current
