"""Export a frozen tokenizer artifact into the kernel's lookup tables.

This module reads a frozen ``tokenizer.json`` and writes the byte-level
BPE structures the CUDA kernel consumes. It depends on ``json``,
``numpy`` and the standard library only -- **never on torch** -- so a
machine with no GPU can still build and verify these tables. That is why
it lives under ``toktier.kernels`` rather than inside the GPU extra.

Exported arrays
---------------
``pair_keys`` / ``pair_vals`` (uint64)
    Open-addressed hash of the merge rules.
    ``key = id_left << 32 | id_right`` (empty slot ``~0``);
    ``val = rank << 32 | id_merged``.
``byte_id`` (int32[256])
    Raw byte to initial token id, through the byte-level alphabet. A
    ``-1`` entry means the artifact has no id for that byte.
``vocab_keys`` / ``vocab_vals`` (uint64)
    Whole-piece vocabulary hash, used only by families whose model sets
    ``ignore_merges``. ``key = fnv1a64(raw bytes)`` (empty slot ``~0``);
    ``val = blob_offset << 30 | length << 20 | id``. A hit must be
    re-checked byte by byte against the blob, so a hash collision cannot
    produce a wrong id.
``vocab_blob`` (uint8)
    Every vocabulary entry's raw bytes, concatenated.
``meta`` (int64[3])
    ``ignore_merges``, number of merges, vocabulary size.

Notes carried over from the artifacts this was measured against:

- One family has more merges than vocabulary entries, so several pairs
  merge to the same token. The merge table is keyed by pair; duplicate
  merged ids are expected and allowed.
- Two ``merges`` encodings occur in the wild: a list ``[a, b]`` and a
  single string ``"a b"``. Both are accepted.
- Added tokens are not exported. Recognising added-token literals in
  text is the frontend's job, not the kernel's.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..backends.protocol import TOKENIZER_FILE
from ..errors import ArtifactHashMismatch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..backends.protocol import ArtifactHandle

__all__ = [
    "BpeTableStore",
    "BpeTables",
    "bytes_to_unicode",
    "fnv1a64",
    "token_to_raw",
]

FNV_OFFSET = np.uint64(0xCBF29CE484222325)
FNV_PRIME = np.uint64(0x100000001B3)
EMPTY = np.uint64(0xFFFFFFFFFFFFFFFF)

#: Version of the exported archive layout. Part of the cache identity:
#: bumping it makes older caches unreadable rather than misread.
TABLE_FORMAT = 2

def bytes_to_unicode() -> dict[int, str]:
    """The byte-level alphabet, matching the reference implementation."""
    printable = (
        list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    )
    mapped = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return dict(zip(printable, [chr(code) for code in mapped], strict=True))


_BYTE_TO_UNICODE = bytes_to_unicode()
_UNICODE_TO_BYTE = {char: byte for byte, char in _BYTE_TO_UNICODE.items()}


def token_to_raw(token: str) -> bytes:
    """Byte-level character string back to the raw bytes it encodes."""
    return bytes(_UNICODE_TO_BYTE[char] for char in token)


def fnv1a64(data: bytes) -> int:
    """FNV-1a over 64 bits, matching the kernel's host-side hash."""
    value = int(FNV_OFFSET)
    for byte in data:
        value = ((value ^ byte) * int(FNV_PRIME)) & 0xFFFFFFFFFFFFFFFF
    return value


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _build_hash(
    keys: list[int], values: list[int]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Open-addressed table with a load factor of at most one third."""
    size = 1
    while size < len(keys) * 3:
        size *= 2
    key_array = np.full(size, EMPTY, dtype=np.uint64)
    value_array = np.zeros(size, dtype=np.uint64)
    mask = size - 1
    for key, value in zip(keys, values, strict=True):
        slot = _splitmix64(key) & mask
        while key_array[slot] != EMPTY:
            if int(key_array[slot]) == key:
                raise ValueError(f"duplicate merge key {key}")
            slot = (slot + 1) & mask
        key_array[slot] = np.uint64(key)
        value_array[slot] = np.uint64(value)
    return key_array, value_array


@dataclass(frozen=True)
class BpeTables:
    """The loaded arrays for one family."""

    name: str
    path: Path
    pair_keys: np.ndarray[Any, Any]
    pair_vals: np.ndarray[Any, Any]
    byte_id: np.ndarray[Any, Any]
    vocab_keys: np.ndarray[Any, Any]
    vocab_vals: np.ndarray[Any, Any]
    vocab_blob: np.ndarray[Any, Any]
    ignore_merges: bool
    n_merges: int
    vocab_size: int
    #: Guard bitmap for merge tables that are not rank-monotone; see
    #: :meth:`BpeTableStore.unsafe_bits`. ``None`` for monotone tables.
    unsafe_bits: np.ndarray[Any, Any] | None


class BpeTableStore:
    """Exports, caches and loads the per-family kernel tables.

    Args:
        artifacts: family name to a **verified artifact handle**
            (``toktier.backends.protocol.ArtifactHandle``), as the
            artifact layer resolves them. This store never reads a
            manifest of its own and never accepts a bare directory: by
            the time a handle exists, every file was checked against the
            manifest's per-file sha256, and the one file this store
            opens is re-checked against the handle's digest map on every
            read.
        cache_dir: where exported tables are written. Exported tables are
            cache: deleting them costs export time, never data.
        shared_model: families whose model section (vocabulary plus
            merges) is identical to another family's, so one exported
            table serves both. Supplied by the caller from the routing
            data rather than kept here, because which families share a
            model section is routing data. The claim is never taken on
            trust: both model sections are hashed and compared before
            every use, and a difference refuses the shared table.
    """

    def __init__(
        self,
        artifacts: Mapping[str, ArtifactHandle],
        cache_dir: Path,
        *,
        shared_model: Mapping[str, str] | None = None,
    ) -> None:
        self._artifacts = artifacts
        self._cache_dir = Path(cache_dir)
        self._shared_model: Mapping[str, str] = dict(shared_model or {})
        self._loaded: MutableMapping[str, BpeTables] = {}

    # -- artifact access ------------------------------------------------

    def _tokenizer_json(self, name: str) -> dict[str, Any]:
        """Read one family's ``tokenizer.json``, re-checking its digest.

        The handle was verified when it was produced; the cheap re-check
        here closes the window in which bytes on disk change after
        verification, and costs one hash of a file that is being read
        anyway.
        """
        handle = self._artifacts.get(name)
        if handle is None:
            raise KeyError(
                f"family {name!r} has no verified artifact handle"
            )
        path = handle.path(TOKENIZER_FILE)
        raw = path.read_bytes()
        expected = handle.files.get(TOKENIZER_FILE)
        observed = hashlib.sha256(raw).hexdigest()
        if expected != observed:
            raise ArtifactHashMismatch(
                f"content hash mismatch for {name}/{TOKENIZER_FILE}",
                details={
                    "family": name,
                    "file": TOKENIZER_FILE,
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "path": str(path),
                    "remedy": (
                        "re-run artifact verification; a handle whose "
                        "bytes moved after verification is not usable"
                    ),
                },
            )
        document: dict[str, Any] = json.loads(raw.decode("utf-8"))
        return document

    def _model_sha(self, name: str) -> str:
        model = self._tokenizer_json(name)["model"]
        payload = json.dumps(model, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _resolve_shared(self, name: str) -> str:
        """Redirect a shared-model family, verifying the sharing claim."""
        target = self._shared_model.get(name)
        if target is None:
            return name
        mine, theirs = self._model_sha(name), self._model_sha(target)
        if mine != theirs:
            raise ValueError(
                f"{name} and {target} no longer share a model section "
                f"({mine[:16]} != {theirs[:16]}); refusing the shared table"
            )
        return target

    def _source_sha256(self, name: str) -> str:
        """The artifact digest that identifies a family's cache entries.

        Cache file names carry a prefix of this digest, so changing the
        tokenizer data under the same family name yields new cache files
        instead of silently reusing stale tables.
        """
        handle = self._artifacts.get(name)
        if handle is None:
            raise KeyError(f"family {name!r} has no verified artifact handle")
        digest = handle.files.get(TOKENIZER_FILE)
        if digest is None:
            raise KeyError(
                f"family {name!r} has no recorded digest for {TOKENIZER_FILE}"
            )
        return digest

    @staticmethod
    def _write_atomic(out: Path, save: Callable[[Path], None]) -> None:
        """Write through a process-private temporary file, then rename.

        The content of every cache file is a deterministic function of
        the artifact digest in its name, so concurrent writers produce
        the same bytes and last-writer-wins is safe; the rename only has
        to be atomic so a reader never sees a partial file. The
        temporary name keeps the real suffix because numpy appends one
        otherwise, and carries the process id so concurrent writers
        never share a temporary path.
        """
        tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp{out.suffix}")
        try:
            save(tmp)
            tmp.replace(out)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    # -- export ---------------------------------------------------------

    def export(self, name: str) -> Path:
        """Write the family's tables to the cache, returning the path."""
        name = self._resolve_shared(name)
        source_sha = self._source_sha256(name)
        out = self._cache_dir / (
            f"bpe_tables_{name}.{source_sha[:16]}.v{TABLE_FORMAT}.npz"
        )
        if out.exists():
            return out

        model = self._tokenizer_json(name)["model"]
        vocab: dict[str, int] = model["vocab"]
        merges_raw: list[Any] = model["merges"]
        ignore_merges = bool(model.get("ignore_merges", False))

        pair_key_list: list[int] = []
        pair_val_list: list[int] = []
        merge_referenced: set[str] = set()
        for rank, merge in enumerate(merges_raw):
            left, right = (
                merge if isinstance(merge, list) else merge.split(" ", 1)
            )
            merge_referenced |= {left, right, left + right}
            pair_key_list.append((vocab[left] << 32) | vocab[right])
            pair_val_list.append((rank << 32) | vocab[left + right])
        pair_keys, pair_vals = _build_hash(pair_key_list, pair_val_list)

        # Byte alphabet coverage may be incomplete: one artifact has no
        # id for fourteen bytes and no unknown token. The reference
        # implementation silently drops an initial symbol it has no id
        # for, before merging, and neighbours then merge across the gap.
        # A missing byte is recorded as -1 and the kernel reproduces that
        # by compacting first. Whole-piece vocabulary lookup depends on
        # full coverage, so an incomplete alphabet is refused there.
        byte_id = np.full(256, -1, dtype=np.int32)
        missing_bytes: list[int] = []
        for byte in range(256):
            char = _BYTE_TO_UNICODE[byte]
            if char in vocab:
                byte_id[byte] = vocab[char]
            else:
                missing_bytes.append(byte)
        if missing_bytes and ignore_merges:
            raise ValueError(
                "a family that ignores merges needs full byte-alphabet "
                f"coverage; missing {missing_bytes}"
            )

        # Vocabulary entries that are not byte-level decodable (some
        # artifacts repeat their added-token literals inside the model
        # vocabulary) are skipped: recognising literals is the frontend's
        # job. It is asserted that no merge rule references them.
        blob = bytearray()
        vocab_key_list: list[int] = []
        vocab_val_list: list[int] = []
        seen: dict[int, bytes] = {}
        skipped_non_byte: list[str] = []
        for token, token_id in vocab.items():
            try:
                raw = token_to_raw(token)
            except KeyError:
                if token in merge_referenced:
                    raise ValueError(
                        f"a merge rule references the undecodable token {token!r}"
                    ) from None
                skipped_non_byte.append(token)
                continue
            key = fnv1a64(raw)
            if len(raw) >= 1024 or len(blob) >= (1 << 34):
                raise ValueError("vocabulary entry exceeds the packing limits")
            if key in seen:
                # A whole-vocabulary hash collision would make a lookup
                # ambiguous. It has never been observed; if it happens the
                # export refuses rather than picking a winner.
                if seen[key] != raw:
                    raise ValueError(
                        f"hash collision between {raw!r} and {seen[key]!r}"
                    )
                continue
            seen[key] = raw
            vocab_key_list.append(key)
            vocab_val_list.append(
                (len(blob) << 30) | (len(raw) << 20) | token_id
            )
            blob.extend(raw)
        vocab_keys, vocab_vals = _build_hash(vocab_key_list, vocab_val_list)

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "pair_keys": pair_keys,
            "pair_vals": pair_vals,
            "byte_id": byte_id,
            "vocab_keys": vocab_keys,
            "vocab_vals": vocab_vals,
            "vocab_blob": np.frombuffer(bytes(blob), dtype=np.uint8),
            "meta": np.array(
                [int(ignore_merges), len(merges_raw), len(vocab)],
                dtype=np.int64,
            ),
            # Identity header, re-checked on load: a cache file that was
            # renamed or copied under the wrong name is refused instead
            # of read.
            "format_version": np.array([TABLE_FORMAT], dtype=np.int64),
            "source_sha256": np.array(source_sha),
        }
        self._write_atomic(out, lambda path: np.savez(path, **arrays))
        return out

    # -- load -----------------------------------------------------------

    def load(self, name: str) -> BpeTables:
        """Export if needed, then load and verify the family's tables."""
        cached = self._loaded.get(name)
        if cached is not None:
            return cached
        path = self.export(name)
        expected_sha = self._source_sha256(self._resolve_shared(name))
        with np.load(path) as archive:
            observed_version = int(archive["format_version"][0])
            observed_sha = str(archive["source_sha256"])
            if observed_version != TABLE_FORMAT or observed_sha != expected_sha:
                raise ArtifactHashMismatch(
                    f"cached BPE tables for {name} do not match their "
                    "recorded identity",
                    details={
                        "family": name,
                        "path": str(path),
                        "expected_sha256": expected_sha,
                        "observed_sha256": observed_sha,
                        "expected_format": TABLE_FORMAT,
                        "observed_format": observed_version,
                        "remedy": "delete the cache file and re-export",
                    },
                )
            tables = BpeTables(
                name=name,
                path=path,
                pair_keys=archive["pair_keys"],
                pair_vals=archive["pair_vals"],
                byte_id=archive["byte_id"],
                vocab_keys=archive["vocab_keys"],
                vocab_vals=archive["vocab_vals"],
                vocab_blob=archive["vocab_blob"],
                ignore_merges=bool(archive["meta"][0]),
                n_merges=int(archive["meta"][1]),
                vocab_size=int(archive["meta"][2]),
                unsafe_bits=self.unsafe_bits(name),
            )
        self._loaded[name] = tables
        return tables

    # -- non-monotone merge table guard ---------------------------------
    #
    # A merge rule ``(a, b) -> z`` at rank ``r`` is batch-safe exactly
    # when every rule that uses ``z`` as a left or right component has a
    # rank greater than ``r``. When the whole table is safe (monotone),
    # merging every leftmost non-overlapping occurrence of the current
    # lowest-rank pair in one round is bit-identical to merging one
    # occurrence at a time.
    #
    # Not every published merge table is monotone: at least one extends
    # its table with a late rule whose result an earlier rule consumes.
    # For such a rank a batched round could consume a token that a
    # lower-rank pair, created by the same round, would have needed. The
    # kernel therefore degrades a flagged rank's round to a single
    # leftmost merge, which is the one-at-a-time semantics and is exact
    # unconditionally.
    #
    # The bitmap is built from the same merges list as the tables, so the
    # two cannot drift. A table with no flagged rank yields ``None``, the
    # loader passes an empty tensor, and the kernel takes its null
    # pointer branch: behaviour for monotone families is unchanged.

    def unsafe_ranks(self, name: str) -> list[int]:
        """Ranks of the rules that are not batch-safe."""
        name = self._resolve_shared(name)
        merges = self._tokenizer_json(name)["model"]["merges"]
        rules: list[tuple[str, str, str]] = []
        for merge in merges:
            left, right = (
                merge if isinstance(merge, list) else merge.split(" ", 1)
            )
            rules.append((left, right, left + right))
        first_use: dict[str, int] = {}
        for rank, (left, right, _result) in enumerate(rules):
            first_use.setdefault(left, rank)
            first_use.setdefault(right, rank)
        flagged: list[int] = []
        for rank, (_left, _right, result) in enumerate(rules):
            used_at = first_use.get(result)
            if used_at is not None and used_at < rank:
                flagged.append(rank)
        return flagged

    def unsafe_bits(self, name: str) -> np.ndarray[Any, Any] | None:
        """Bitmap of flagged ranks, or ``None`` for a monotone table.

        Cached beside the exported tables rather than inside them, so
        adding the guard never rewrites an archive another process may be
        reading. The file name carries the artifact digest, so the
        bitmap and the tables it guards share one identity.
        """
        name = self._resolve_shared(name)
        source_sha = self._source_sha256(name)
        out = self._cache_dir / f"bpe_unsafe_{name}.{source_sha[:16]}.v1.npy"
        if out.exists():
            bits = np.load(out)
        else:
            flagged = self.unsafe_ranks(name)
            n_merges = len(self._tokenizer_json(name)["model"]["merges"])
            if n_merges >= (1 << 27):
                raise ValueError("merge count exceeds the kernel's rank packing")
            bits = np.zeros((n_merges + 31) // 32, dtype=np.uint32)
            for rank in flagged:
                bits[rank >> 5] |= np.uint32(1 << (rank & 31))
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._write_atomic(out, lambda path: np.save(path, bits))
        return bits if bool(bits.any()) else None
