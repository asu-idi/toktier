"""Certified append repair using the corrected Gigatoken window encoder.

The algorithm is the product form of the archived repair campaign:
re-tokenize a suffix window with the corrected Gigatoken build, find a run
whose token id, global span, and same-span ordinal agree with the previous
reference state, require the frozen BPE synchronizing-transition witness, and
splice only after that certificate.  If any premise is absent, the complete
text is encoded by the Hugging Face reference callback.

Gigatoken does not expose offsets.  For these eleven positively identified
ByteLevel BPE artifacts, token byte lengths are fixed by the vocabulary.  The
adapter reconstructs character spans from those lengths and refuses a window
unless the byte stream closes exactly and any configured normalizer leaves the
window unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from .registry import CONFIG_ID, RepairFamily, pclass_table

ReferenceEncode = Callable[[str], tuple[list[int], list[tuple[int, int]]]]

_PCLASS_LABELS = "OSLNM"
_SYNC = frozenset(
    {
        ("L", "N"),
        ("L", "O"),
        ("L", "S"),
        ("N", "L"),
        ("N", "O"),
        ("N", "S"),
        ("O", "N"),
        ("O", "S"),
    }
)


class _Gigatoken(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...

    @property
    def vocab(self) -> Mapping[int, bytes]: ...

    @property
    def vocab_size(self) -> int: ...


class _FastTokenizer(Protocol):
    def get_vocab(self) -> Mapping[str, int]: ...


class _Normalizer(Protocol):
    def normalize_str(self, text: str) -> str: ...


class WindowUnsupported(RuntimeError):
    """The offset-reconstruction or certificate premises do not hold."""


# These are expected failures of the optional window engine or of the
# certificate guards around its output.  The Hugging Face reference callback
# deliberately stays outside this boundary: an oracle failure is a real caller
# error and must propagate instead of being disguised as a fast-path fallback.
_WINDOW_GUARD_ERRORS = (
    OSError,
    RuntimeError,
    UnicodeError,
    ValueError,
    WindowUnsupported,
)


@dataclass(frozen=True)
class _Record:
    original_index: int
    global_start: int
    global_end: int
    token_id: int
    same_span_ordinal: int


@dataclass(frozen=True)
class _Match:
    left_start: int
    right_start: int
    length: int
    covered_chars: int


def _bytes_to_unicode() -> dict[int, str]:
    visible = (
        list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    )
    mapped = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return dict(zip(visible, (chr(value) for value in mapped), strict=True))


_BYTE_ALPHABET = frozenset(_bytes_to_unicode().values())


def _byte_lengths_from_hf(tokenizer: _FastTokenizer) -> list[int]:
    """Derive one raw-byte length per id from the paired HF object."""
    vocabulary = tokenizer.get_vocab()
    if not vocabulary:
        raise WindowUnsupported("the live HF tokenizer has an empty vocabulary")
    added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(added_vocab):
        added = set(added_vocab())
    else:
        added_decoder = getattr(tokenizer, "get_added_tokens_decoder", None)
        decoded = added_decoder() if callable(added_decoder) else {}
        added = {str(getattr(token, "content", token)) for token in decoded.values()}
    lengths = [0] * (max(vocabulary.values()) + 1)
    seen = [False] * len(lengths)
    for token, raw_id in vocabulary.items():
        token_id = int(raw_id)
        if token_id < 0 or token_id >= len(lengths):
            raise WindowUnsupported(f"vocabulary id {token_id} is out of range")
        if token not in added and all(char in _BYTE_ALPHABET for char in token):
            lengths[token_id] = len(token)
        else:
            lengths[token_id] = len(token.encode("utf-8"))
        seen[token_id] = True
    if not all(seen):
        missing = seen.index(False)
        raise WindowUnsupported(f"the HF vocabulary has no entry for id {missing}")
    return lengths


def _verified_byte_lengths(
    tokenizer: _FastTokenizer, engine: _Gigatoken
) -> tuple[int, ...]:
    """Cross-check HF-derived lengths against Gigatoken's raw vocabulary."""
    expected = _byte_lengths_from_hf(tokenizer)
    size = int(engine.vocab_size)
    if len(expected) != size:
        raise WindowUnsupported(
            f"vocabulary sizes differ: HF={len(expected)}, Gigatoken={size}"
        )
    observed = [0] * size
    seen = [False] * size
    for raw_id, token_bytes in engine.vocab.items():
        token_id = int(raw_id)
        if token_id < 0 or token_id >= size:
            raise WindowUnsupported(f"Gigatoken vocabulary id {token_id} is invalid")
        observed[token_id] = len(token_bytes)
        seen[token_id] = True
    if not all(seen):
        missing = seen.index(False)
        raise WindowUnsupported(f"Gigatoken has no raw vocabulary entry for {missing}")
    if observed != expected:
        first = next(
            index
            for index, (left, right) in enumerate(zip(expected, observed, strict=True))
            if left != right
        )
        raise WindowUnsupported(
            "HF/Gigatoken byte-length tables differ at id "
            f"{first}: {expected[first]} != {observed[first]}"
        )
    return tuple(observed)


def _normalizer(tokenizer: object) -> _Normalizer | None:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return cast("_Normalizer | None", getattr(backend, "normalizer", None))


def _spans_from_ids(
    ids: Sequence[int], byte_lengths: Sequence[int], text: str
) -> list[tuple[int, int]]:
    if not ids:
        if text:
            raise WindowUnsupported("Gigatoken returned no ids for non-empty text")
        return []
    byte_starts: list[int] = []
    byte_ends: list[int] = []
    cursor = 0
    for raw_id in ids:
        token_id = int(raw_id)
        if token_id < 0 or token_id >= len(byte_lengths):
            raise WindowUnsupported(f"Gigatoken returned unknown id {token_id}")
        length = byte_lengths[token_id]
        byte_starts.append(cursor)
        cursor += length
        byte_ends.append(cursor)
    encoded = text.encode("utf-8")
    if cursor != len(encoded):
        raise WindowUnsupported(
            f"window bytes do not close: tokens={cursor}, text={len(encoded)}"
        )
    if text.isascii():
        return list(zip(byte_starts, byte_ends, strict=True))

    char_of_byte: list[int] = []
    for char_index, char in enumerate(text):
        char_of_byte.extend([char_index] * len(char.encode("utf-8")))
    return [
        (char_of_byte[start], char_of_byte[end - 1] + 1)
        for start, end in zip(byte_starts, byte_ends, strict=True)
    ]


def _build_records(
    ids: Sequence[int],
    spans: Sequence[tuple[int, int]],
    *,
    base: int,
    overlap_start: int,
    overlap_end: int,
) -> list[_Record]:
    if len(ids) != len(spans):
        raise WindowUnsupported(
            f"ids/spans length mismatch: {len(ids)} != {len(spans)}"
        )
    records: list[_Record] = []
    ordinals: dict[tuple[int, int], int] = {}
    for index, ((local_start, local_end), raw_id) in enumerate(
        zip(spans, ids, strict=True)
    ):
        start, end = base + local_start, base + local_end
        if start == end or start < overlap_start or end > overlap_end:
            continue
        key = (start, end)
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        records.append(_Record(index, start, end, int(raw_id), ordinal))
    return records


def _record_key(record: _Record) -> tuple[int, int, int, int]:
    return (
        record.global_start,
        record.global_end,
        record.same_span_ordinal,
        record.token_id,
    )


def _find_match(left: list[_Record], right: list[_Record]) -> _Match | None:
    right_index: dict[tuple[int, int, int, int], list[int]] = {}
    for index, record in enumerate(right):
        right_index.setdefault(_record_key(record), []).append(index)
    best: _Match | None = None
    covered: set[tuple[int, int]] = set()
    for left_index, left_record in enumerate(left):
        for right_index_value in right_index.get(_record_key(left_record), ()):
            if (left_index, right_index_value) in covered:
                continue
            right_record = right[right_index_value]
            length = 1
            while (
                left_index + length < len(left)
                and right_index_value + length < len(right)
                and left[left_index + length].original_index
                == left_record.original_index + length
                and right[right_index_value + length].original_index
                == right_record.original_index + length
                and _record_key(left[left_index + length])
                == _record_key(right[right_index_value + length])
            ):
                covered.add((left_index + length, right_index_value + length))
                length += 1
            candidate = _Match(
                left_start=left_record.original_index,
                right_start=right_record.original_index,
                length=length,
                covered_chars=(
                    left[left_index + length - 1].global_end - left_record.global_start
                ),
            )
            if best is None or (
                candidate.length,
                candidate.covered_chars,
            ) > (best.length, best.covered_chars):
                best = candidate
    return best


class GigatokenRepair:
    """Callable append adapter for :class:`toktier._native.CallbackEncoder`."""

    def __init__(
        self,
        *,
        spec: RepairFamily,
        engine: _Gigatoken,
        hf_tokenizer: _FastTokenizer,
        reference_encode: ReferenceEncode,
    ) -> None:
        self.spec = spec
        self._engine = engine
        self._reference_encode = reference_encode
        self._normalizer = _normalizer(hf_tokenizer)
        if spec.has_normalizer and self._normalizer is None:
            raise WindowUnsupported(
                f"{spec.family} requires a normalizer but the live HF object has none"
            )
        if not spec.has_normalizer and self._normalizer is not None:
            raise WindowUnsupported(
                f"{spec.family} is certified without a normalizer but one is active"
            )
        self._byte_lengths = _verified_byte_lengths(hf_tokenizer, engine)
        self._pclass = pclass_table()
        self._path_counts: dict[str, int] = {}
        self._window_calls = 0
        self._window_chars = 0
        self._last: dict[str, object] | None = None

    @property
    def config_id(self) -> str:
        return CONFIG_ID

    @property
    def bpe_sync_pclass(self) -> bytes:
        """Frozen O/S/L/N/M table consumed by the native seal predicate."""
        return self._pclass

    @property
    def minimum_seal_tail_chars(self) -> int:
        """Smallest retained tail that can enter a Gigatoken window.

        Sealing closer to the end would remain correct, but the next append
        would immediately take the HF ``window_covers_all`` path. Keep the
        first power-of-two repair window whose usable overlap can exceed the
        family's certified ``effective_l_max``.
        """
        window = self.spec.window_chars
        while window - self.spec.margin <= self.spec.effective_l_max:
            window *= 2
        return window + 1

    def _count(self, path: str) -> None:
        self._path_counts[path] = self._path_counts.get(path, 0) + 1

    def _require_byte_identity(self, text: str) -> None:
        """Prove that token byte lengths describe the original text."""
        text.encode("utf-8")
        normalizer = self._normalizer
        if normalizer is not None and not text.isascii():
            normalized = normalizer.normalize_str(text)
            if normalized != text:
                raise WindowUnsupported(
                    "the repair window is not invariant under the certified normalizer"
                )

    def spans_for_ids(self, text: str, ids: Sequence[int]) -> list[tuple[int, int]]:
        """Reconstruct character spans for an exact accelerated id stream.

        This is the bridge from a cold GPU/full-CPU encode into persistent
        session state. It accepts no looser premise than repair itself:
        UTF-8 must be valid, the certified normalizer must leave the text
        unchanged, every id must be known, and token bytes must close
        exactly over the original text. Otherwise ``WindowUnsupported``
        asks the facade to seed the state with the HF reference instead.
        """
        self._require_byte_identity(text)
        return _spans_from_ids(ids, self._byte_lengths, text)

    def _window_encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        self._require_byte_identity(text)
        raw_ids = self._engine.encode(text)
        ids = [int(value) for value in raw_ids]
        spans = _spans_from_ids(ids, self._byte_lengths, text)
        self._window_calls += 1
        self._window_chars += len(text)
        return ids, spans

    def _class(self, char: str) -> str:
        return _PCLASS_LABELS[self._pclass[ord(char)]]

    def _has_sync_witness(
        self,
        match: _Match,
        spans: Sequence[tuple[int, int]],
        text: str,
    ) -> bool:
        for index in range(match.left_start, match.left_start + match.length - 1):
            boundary = spans[index + 1][0]
            if not 0 < boundary < len(text):
                continue
            current = text[boundary]
            pair = (self._class(text[boundary - 1]), self._class(current))
            if (
                pair in _SYNC
                and not (pair == ("O", "S") and current in "\r\n")
                and not (pair == ("L", "O") and current == "'")
            ):
                return True
        return False

    def _accepted(
        self,
        match: _Match | None,
        spans: Sequence[tuple[int, int]],
        text: str,
    ) -> bool:
        if match is None or match.length < self.spec.min_match_tokens:
            return False
        covered = match.covered_chars
        if self._normalizer is not None:
            start = spans[match.left_start][0]
            end = spans[match.left_start + match.length - 1][1]
            covered = len(self._normalizer.normalize_str(text[start:end]))
        return covered > self.spec.effective_l_max and self._has_sync_witness(
            match, spans, text
        )

    def _reference_full(
        self, text: str, *, reason: str, detail: object | None = None
    ) -> tuple[list[int], list[tuple[int, int]], int, str]:
        ids, spans = self._reference_encode(text)
        path = f"hf_full_{reason}"
        self._count(path)
        self._last = {
            "path": path,
            "reason": reason,
            "detail": detail,
            "input_chars": len(text),
        }
        return ids, spans, 0, path

    def __call__(
        self,
        tail_text: str,
        tail_ids: list[int],
        tail_spans: Sequence[tuple[int, int]],
        delta: str,
    ) -> tuple[list[int], list[tuple[int, int]], int, str]:
        """Repair one strict append, or safely re-run the HF reference."""
        new_text = tail_text + delta
        if not delta:
            path = "gigatoken_repair_noop"
            self._count(path)
            self._last = {"path": path, "kept_tokens": len(tail_ids)}
            return list(tail_ids), list(tail_spans), len(tail_ids), path
        if len(tail_ids) != len(tail_spans):
            return self._reference_full(new_text, reason="invalid_prior_state")

        previous_chars = len(tail_text)
        window = self.spec.window_chars
        retries = 0
        while window < previous_chars:
            try:
                window_start = previous_chars - window
                window_text = new_text[window_start:]
                window_ids, window_spans = self._window_encode(window_text)
                overlap_start = window_start + self.spec.margin
                left = _build_records(
                    tail_ids,
                    tail_spans,
                    base=0,
                    overlap_start=overlap_start,
                    overlap_end=previous_chars,
                )
                right = _build_records(
                    window_ids,
                    window_spans,
                    base=window_start,
                    overlap_start=overlap_start,
                    overlap_end=previous_chars,
                )
                match = _find_match(left, right)
                if self._accepted(match, tail_spans, new_text):
                    assert match is not None
                    left_end = match.left_start + match.length
                    right_end = match.right_start + match.length
                    shifted = [
                        (start + window_start, end + window_start)
                        for start, end in window_spans[right_end:]
                    ]
                    ids = [*tail_ids[:left_end], *window_ids[right_end:]]
                    spans = [*tail_spans[:left_end], *shifted]
                    path = "gigatoken_repair"
                    self._count(path)
                    self._last = {
                        "path": path,
                        "window_chars": window,
                        "retries": retries,
                        "match_tokens": match.length,
                        "match_chars": match.covered_chars,
                        "kept_tokens": left_end,
                        "delta_chars": len(delta),
                    }
                    return ids, spans, left_end, path
            except _WINDOW_GUARD_ERRORS as exc:
                # Run the oracle outside the exception handler so a genuine
                # reference error is never swallowed by the optional backend.
                return self._reference_full(
                    new_text,
                    reason="engine_guard",
                    detail={"error": type(exc).__name__, "message": str(exc)},
                )
            if retries >= self.spec.max_retries:
                return self._reference_full(
                    new_text,
                    reason="no_safe_cut",
                    detail={"window_chars": window, "retries": retries},
                )
            retries += 1
            window *= 2
        return self._reference_full(
            new_text,
            reason="window_covers_all",
            detail={"window_chars": window, "retries": retries},
        )

    def stats(self) -> dict[str, object]:
        return {
            "backend": "fast_cpu",
            "engine": "gigatoken",
            "config_id": CONFIG_ID,
            "family": self.spec.family,
            "artifact_sha256": self.spec.artifact_sha256,
            "path_counts": dict(sorted(self._path_counts.items())),
            "window_calls": self._window_calls,
            "window_chars": self._window_chars,
            "last": dict(self._last) if self._last is not None else None,
        }
