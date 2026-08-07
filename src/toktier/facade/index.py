"""Content checkpoint index for the facade's auto lookup.

The index is a derived cache over store entries: for each entry text
``S`` it records a prefix digest at the endpoint ``|S|`` (the primary
hit surface for append workloads) and at geometric byte positions
(4 KiB doubling up to ``|S|``, so ~log2 marks per entry). A query
streams the input once, snapshots the running hash at every indexed
length, and proposes the longest endpoint match first.

Digests locate candidates only; they decide nothing. Every proposed hit
is re-verified by the caller with a byte comparison against the stored
text (the anti-collision hard gate), so a forged or colliding digest can
cost a wasted comparison, never a wrong result. Losing or corrupting the
index costs recomputation: it is rebuilt from the stored records, or
queries simply miss into a full encode.

This module is dependency-free (standard library only).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

__all__ = [
    "INDEX_FORMAT",
    "MARK_FLOOR_BYTES",
    "CheckpointIndex",
    "IndexEntry",
    "mark_positions",
    "prefix_digests",
]

#: Serialized index format; a payload with any other value is discarded
#: and rebuilt rather than guessed at.
INDEX_FORMAT = 1

#: First geometric mark. Marks double from here up to the entry length.
MARK_FLOOR_BYTES = 4096

#: Keyed 128-bit digest domain for checkpoint digests. The digest
#: primitive is not load-bearing for correctness (the byte re-check is),
#: so it can change together with :data:`INDEX_FORMAT`.
_PERSON = b"toktier.fidx.v1"
_DIGEST_SIZE = 16


def _hasher() -> hashlib.blake2b:
    return hashlib.blake2b(digest_size=_DIGEST_SIZE, person=_PERSON)


def mark_positions(byte_length: int) -> tuple[int, ...]:
    """Geometric mark positions strictly inside an entry of this size."""
    marks: list[int] = []
    position = MARK_FLOOR_BYTES
    while position < byte_length:
        marks.append(position)
        position *= 2
    return tuple(marks)


def prefix_digests(data: bytes, lengths: list[int]) -> dict[int, str]:
    """Digest of every requested byte-prefix of ``data``, in one pass.

    ``lengths`` must be sorted ascending and bounded by ``len(data)``.
    The running hash is fed once and snapshotted (``copy``) at each
    requested length, so the cost is one traversal plus a constant
    finalize per point.
    """
    digests: dict[int, str] = {}
    running = _hasher()
    consumed = 0
    for length in lengths:
        running.update(data[consumed:length])
        consumed = length
        digests[length] = running.copy().hexdigest()
    return digests


@dataclass(frozen=True)
class IndexEntry:
    """Checkpoint digests of one store entry."""

    byte_length: int
    end_digest: str
    #: ``(position, digest)`` pairs for the geometric marks.
    marks: tuple[tuple[int, str], ...]


def entry_for(data: bytes) -> IndexEntry:
    """Index row for an entry text given as UTF-8 bytes."""
    positions = [*mark_positions(len(data)), len(data)]
    digests = prefix_digests(data, positions)
    return IndexEntry(
        byte_length=len(data),
        end_digest=digests[len(data)],
        marks=tuple((p, digests[p]) for p in positions[:-1]),
    )


class CheckpointIndex:
    """Mutable digest index over named store entries."""

    def __init__(self) -> None:
        self._entries: dict[str, IndexEntry] = {}

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def get(self, name: str) -> IndexEntry | None:
        return self._entries.get(name)

    def put(self, name: str, entry: IndexEntry) -> None:
        self._entries[name] = entry

    def discard(self, name: str) -> None:
        self._entries.pop(name, None)

    def query_lengths(self, limit: int) -> list[int]:
        """Indexed lengths within ``limit``, ascending, without repeats."""
        lengths: set[int] = set()
        for entry in self._entries.values():
            if 0 < entry.byte_length <= limit:
                lengths.add(entry.byte_length)
            for position, _ in entry.marks:
                if 0 < position <= limit:
                    lengths.add(position)
        return sorted(lengths)

    def endpoint_candidates(
        self, digests: dict[int, str]
    ) -> list[tuple[int, str]]:
        """Entries whose endpoint digest matches a query digest.

        Returns ``(byte_length, name)`` pairs, longest first. Candidates
        are proposals: the caller must byte-verify each one against the
        stored text before serving anything from it.
        """
        matches = [
            (entry.byte_length, name)
            for name, entry in self._entries.items()
            if entry.byte_length > 0
            and digests.get(entry.byte_length) == entry.end_digest
        ]
        matches.sort(key=lambda item: (-item[0], item[1]))
        return matches

    # -- serialization -------------------------------------------------

    def to_payload(self) -> dict[str, object]:
        """JSON-shaped payload for the sidecar file."""
        return {
            "format": INDEX_FORMAT,
            "entries": {
                name: {
                    "bytes": entry.byte_length,
                    "end": entry.end_digest,
                    "marks": [[p, d] for p, d in entry.marks],
                }
                for name, entry in self._entries.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: object) -> CheckpointIndex | None:
        """Rebuild from a sidecar payload; ``None`` on any malformation.

        The index is a derived cache, so a payload that fails any shape
        check is discarded wholesale rather than partially trusted.
        """
        if not isinstance(payload, dict) or payload.get("format") != INDEX_FORMAT:
            return None
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            return None
        index = cls()
        for name, raw in raw_entries.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                return None
            byte_length = raw.get("bytes")
            end_digest = raw.get("end")
            raw_marks = raw.get("marks")
            if (
                not isinstance(byte_length, int)
                or isinstance(byte_length, bool)
                or byte_length < 0
                or not isinstance(end_digest, str)
                or not isinstance(raw_marks, list)
            ):
                return None
            marks: list[tuple[int, str]] = []
            for item in raw_marks:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not isinstance(item[0], int)
                    or isinstance(item[0], bool)
                    or not isinstance(item[1], str)
                ):
                    return None
                marks.append((item[0], item[1]))
            index.put(
                name,
                IndexEntry(
                    byte_length=byte_length,
                    end_digest=end_digest,
                    marks=tuple(marks),
                ),
            )
        return index
