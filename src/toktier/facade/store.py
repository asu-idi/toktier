"""Entry store behind the facade's session and auto-lookup paths.

One entry is ``(text, ids)`` under a name: named entries carry a
user-supplied session id, auto entries are created by content lookup.
The token streams live in the native session store
(``toktier._native``); this module adds naming, residency, persistence
and the checkpoint index, and it decides nothing about token ids -- every
stream is produced by the full-encode/repair callbacks the store was
built with. Certified callbacks return the reference id stream; the
explicitly selected experimental Fastokens adapter is keyed under its own
fingerprint and carries no exact-ID certificate.

Correctness posture, inherited from the store contract: the read path
never raises for store reasons. Any failure to locate, verify or append
-- a missing record, a corrupt file, an evicted native handle, a digest
collision -- degrades to "the store cannot serve this call" and the caller
runs the active plain routed encode instead. On certified configurations
a wrong id is never an outcome; the store can only be slow, never wrong.

Persistence uses store format v1 records verbatim, one file per entry
(``entries/<name>.rec``, atomic replace). Loading a record re-verifies
it byte-level (:mod:`toktier.records`) and re-encodes its tail through
the current engine inside the native import, so a stale or foreign
record fails closed into a miss. A private ``.binding`` sidecar lets a sealed
record recover its omitted historical prefix only from caller-presented bytes
that match its record hash and full-text digest; it contains no plaintext or
token IDs. The checkpoint index and the in-process text cache are derived
layers: rebuildable, evictable, never authoritative.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from array import array
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import records
from ..errors import SessionStateMismatch, StoreFormatUnsupported, ToktierError
from ..paths import FILE_MODE, ensure_private_dir
from .index import CheckpointIndex, IndexEntry, entry_for, prefix_digests
from .recovery import RecoveryBinding

if TYPE_CHECKING:
    from .. import _native

__all__ = [
    "AUTO_MIN_BYTES",
    "DEFAULT_CACHE_BUDGET_BYTES",
    "MAX_AUTO_ENTRIES",
    "AppendEncode",
    "EntryStore",
    "ReferenceEncode",
]

#: Full encode of one text into reference-equal core-stream ids and token spans.
#: The historical name remains part of the typed facade surface.
ReferenceEncode = Callable[[str], "tuple[list[int], list[tuple[int, int]]]"]

#: Optional certified append callback understood by the native store.
AppendEncode = Callable[
    [str, list[int], "Sequence[tuple[int, int]]", str],
    "tuple[list[int], list[tuple[int, int]], int, str]",
]

#: Content lookup ignores texts below this size: for them the entry
#: bookkeeping costs more than the encode it would save. Named sessions
#: have no threshold -- an explicit session id is explicit intent.
AUTO_MIN_BYTES = 4096

#: Default budget of the in-process text cache. Deliberately not tiny:
#: the cache trades memory for speed, and evictions only cost reloads
#: (persistent stores) or re-encodes (in-memory stores), never
#: correctness.
DEFAULT_CACHE_BUDGET_BYTES = 128 * 1024 * 1024

#: Cap on auto-created entries; the oldest is dropped beyond it.
MAX_AUTO_ENTRIES = 1024

#: Capacity of the native store; above the in-process cache so facade
#: entries are evicted by budget before native capacity forces them out.
_NATIVE_MAX_SESSIONS = 4096

_META_NAME = "meta.json"
_INDEX_NAME = "index.json"
_ENTRIES_DIR = "entries"
_META_FORMAT = 1
_RECOVERY_SUFFIX = ".binding"

#: Failures the read path converts into a miss. Broad on purpose: the
#: caller re-runs the reference encode, which re-raises any genuine
#: encoder problem with its proper type.
_MISS_ERRORS = (ToktierError, ValueError, KeyError, RuntimeError, OSError)


class _CandidateMismatch(Exception):
    """The supplied text cannot be the historical text of an entry."""


def _outcome_ids(outcome: dict[str, object]) -> list[int]:
    """Full id stream from a native append outcome."""
    raw = outcome["all_ids"]
    if not isinstance(raw, bytes):  # pragma: no cover - binding invariant
        raise ValueError("append outcome carried no id bytes")
    return _ids_from_bytes(raw)


def _outcome_revision(outcome: dict[str, object]) -> int:
    revision = outcome["revision"]
    if not isinstance(revision, int):  # pragma: no cover - binding invariant
        raise ValueError("append outcome carried no revision")
    return revision


def _ids_from_bytes(raw: bytes) -> list[int]:
    values = array("I")
    if values.itemsize != 4:  # pragma: no cover - platform dependent
        return [
            int.from_bytes(raw[i : i + 4], "little")
            for i in range(0, len(raw), 4)
        ]
    values.frombytes(raw)
    if sys.byteorder == "big":  # pragma: no cover - platform dependent
        values.byteswap()
    return values.tolist()


def _atomic_write(path: Path, data: bytes) -> None:
    """Write, fsync, rename, then fsync the directory.

    The same persistence discipline as the artifact store's installer:
    the payload is durable before the rename makes it visible, and the
    directory entry is flushed afterwards, so a crash never publishes a
    torn record. The directory fsync is best effort because not every
    platform exposes directory descriptors; the shipped wheels are
    Linux-only, where both steps are real.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    handle = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)
    os.chmod(temporary, FILE_MODE)
    os.replace(temporary, path)
    try:
        directory_handle = os.open(path.parent, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory fds
        return
    try:
        os.fsync(directory_handle)
    except OSError:  # pragma: no cover - filesystems without directory fsync
        pass
    finally:
        os.close(directory_handle)


def _entry_name(kind: str, token: str) -> str:
    if kind == "session":
        encoded = base64.urlsafe_b64encode(token.encode("utf-8")).decode("ascii")
        return "s-" + encoded.rstrip("=")
    return "a-" + token


def _native_index_entry(
    row: tuple[int, str, list[tuple[int, str]]] | None,
) -> IndexEntry | None:
    """Translate the native checkpoint row without rescanning text."""
    if row is None:
        return None
    byte_length, end_digest, marks = row
    return IndexEntry(
        byte_length=byte_length,
        end_digest=end_digest,
        marks=tuple(marks),
    )


@dataclass
class _Entry:
    name: str
    kind: str  # "session" | "auto"
    byte_length: int
    #: Resident text, or ``None`` when evicted to disk.
    text: str | None = None
    #: Native session handle, present only while resident.
    handle: int | None = None
    revision: int = 0
    index_entry: IndexEntry | None = None


@dataclass
class _Stats:
    """Counters of the store's decisions.

    Caliber note: ``session_hits``/``auto_hits`` count exact whole-text
    reuse (the stored text equals the input). A served prefix extension
    counts in ``session_appends``/``auto_appends`` instead -- it both
    reuses the stored stream and encodes the remainder -- so an
    append-mostly workload legitimately shows zero hits next to growing
    appends. Appends are successes, not misses.
    """

    session_hits: int = 0
    session_appends: int = 0
    session_overwrites: int = 0
    session_misses: int = 0
    auto_hits: int = 0
    auto_appends: int = 0
    auto_misses: int = 0
    collision_rejects: int = 0
    degraded: int = 0
    index_rebuilds: int = 0
    entries_evicted: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, int]:
        counters = {
            name: value
            for name, value in self.__dict__.items()
            if isinstance(value, int)
        }
        counters.update(self.extra)
        return counters


class EntryStore:
    """Named (text, ids) entries over the native session store."""

    def __init__(
        self,
        *,
        fingerprint: bytes,
        encode: ReferenceEncode,
        append: AppendEncode | None = None,
        append_stats: Callable[[], dict[str, object]] | None = None,
        certified_bpe_witness: bool = False,
        bpe_sync_pclass: bytes | None = None,
        seal_end_guard_chars: int = 0,
        native_encoder: object | None = None,
        directory: Path | None = None,
        cache_budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES,
    ) -> None:
        self._fingerprint = fingerprint
        self._encode = encode
        self._append = append
        self._append_stats = append_stats
        self._certified_bpe_witness = certified_bpe_witness
        self._bpe_sync_pclass = bpe_sync_pclass
        if type(seal_end_guard_chars) is not int or seal_end_guard_chars < 0:
            raise ValueError("seal_end_guard_chars must be a non-negative integer")
        self._seal_end_guard_chars = seal_end_guard_chars
        self._native_encoder = native_encoder
        self._directory = directory
        self._budget = max(0, int(cache_budget_bytes))
        self._entries: dict[str, _Entry] = {}
        self._index = CheckpointIndex()
        self._stats = _Stats()
        self._native: Any = None
        self._store: _native.SessionStore | None = None
        self._encoder: Any = None
        self._key_id = 0
        if directory is not None:
            self._open_directory(directory)

    # -- native store --------------------------------------------------

    def _backend(self) -> tuple[_native.SessionStore, Any]:
        if self._store is None or self._encoder is None:
            from .. import _native as native

            self._native = native
            store = native.SessionStore(
                max_sessions=_NATIVE_MAX_SESSIONS,
                track_recovery=self._directory is not None,
                track_content_index=True,
            )
            witness = (
                native.WITNESS_BPE_SYNC_TRANSITION
                if self._certified_bpe_witness
                else native.WITNESS_NONE_FULL_REENCODE
            )
            encoder = self._native_encoder
            if encoder is None:
                encoder = native.CallbackEncoder(
                    witness,
                    self._encode,
                    self._append,
                    None,
                    self._bpe_sync_pclass,
                )
            self._key_id = store.register_fingerprint(
                self._fingerprint, self._seal_end_guard_chars
            )
            self._store = store
            self._encoder = encoder
        return self._store, self._encoder

    # -- persistent layout ---------------------------------------------

    def _open_directory(self, directory: Path) -> None:
        ensure_private_dir(directory)
        meta_path = directory / _META_NAME
        fresh = not meta_path.is_file()
        if fresh:
            _atomic_write(
                meta_path,
                json.dumps(
                    {
                        "format": _META_FORMAT,
                        "fingerprint": self._fingerprint.hex(),
                    },
                    sort_keys=True,
                ).encode("utf-8"),
            )
        else:
            self._check_meta(meta_path)
        ensure_private_dir(directory / _ENTRIES_DIR)
        self._load_or_rebuild_index(directory, fresh=fresh)

    def _check_meta(self, meta_path: Path) -> None:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionStateMismatch(
                "the store's metadata file cannot be read",
                details={"path": str(meta_path)},
            ) from exc
        if not isinstance(meta, dict) or not isinstance(
            meta.get("fingerprint"), str
        ):
            raise SessionStateMismatch(
                "the store's metadata file has an unexpected shape",
                details={"path": str(meta_path)},
            )
        if meta.get("format") != _META_FORMAT:
            raise StoreFormatUnsupported(
                "the store metadata names a format this reader does not know",
                details={"format_version": meta.get("format")},
            )
        if meta["fingerprint"] != self._fingerprint.hex():
            raise SessionStateMismatch(
                "the store was written under a different semantic fingerprint",
                details={
                    "expected_fingerprint": self._fingerprint.hex(),
                    "stored_fingerprint": meta["fingerprint"],
                },
            )

    def _record_path(self, name: str) -> Path:
        assert self._directory is not None
        return self._directory / _ENTRIES_DIR / f"{name}.rec"

    def _recovery_path(self, name: str) -> Path:
        assert self._directory is not None
        return self._directory / _ENTRIES_DIR / f"{name}{_RECOVERY_SUFFIX}"

    def _load_or_rebuild_index(self, directory: Path, *, fresh: bool) -> None:
        index_path = directory / _INDEX_NAME
        index: CheckpointIndex | None = None
        if index_path.is_file():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            index = CheckpointIndex.from_payload(payload)
        if index is None:
            index = self._rebuild_index(directory)
            self._index = index
            if not fresh:
                self._stats.index_rebuilds += 1
            self._persist_index()
        self._index = index
        for name in index.names():
            row = index.get(name)
            assert row is not None
            kind = "session" if name.startswith("s-") else "auto"
            self._entries[name] = _Entry(
                name=name,
                kind=kind,
                byte_length=row.byte_length,
                index_entry=row,
            )

    def _rebuild_index(self, directory: Path) -> CheckpointIndex:
        """Recompute the index from the record files themselves."""
        index = CheckpointIndex()
        for path in sorted((directory / _ENTRIES_DIR).glob("*.rec")):
            try:
                view = records.decode_record(path.read_bytes())
            except _MISS_ERRORS:
                continue
            if view.stable_prefix_byte_length == 0:
                index.put(path.stem, entry_for(view.text_tail.encode("utf-8")))
                continue
            try:
                binding_path = (
                    directory / _ENTRIES_DIR / f"{path.stem}{_RECOVERY_SUFFIX}"
                )
                binding = RecoveryBinding.from_bytes(
                    binding_path.read_bytes()
                )
            except (OSError, ValueError):
                # Legacy sealed records have no recovery binding. They
                # remain valid portable records, but cannot be proposed
                # for facade reuse after the derived index is lost.
                continue
            if (
                binding.record_hash != view.curr_block_hash
                or binding.index_entry.byte_length != view.full_text_byte_length
            ):
                continue
            index.put(path.stem, binding.index_entry)
        return index

    def _persist_index(self) -> None:
        if self._directory is None:
            return
        payload = json.dumps(self._index.to_payload(), sort_keys=True)
        try:
            _atomic_write(self._directory / _INDEX_NAME, payload.encode("utf-8"))
        except OSError:
            # The index is derived; failing to write it costs a rebuild.
            self._count_extra("persist_failures")

    def _persist_entry(self, entry: _Entry) -> None:
        if self._directory is None or entry.handle is None or entry.text is None:
            return
        store, _ = self._backend()
        try:
            raw = bytes(store.export_session(entry.handle))
            binding = store.export_recovery_binding(entry.handle)
            if binding is None:
                raise ValueError("native session has no recovery binding")
            _atomic_write(self._record_path(entry.name), raw)
            _atomic_write(self._recovery_path(entry.name), bytes(binding))
        except _MISS_ERRORS:
            # Durability degrades for this entry; served ids are already
            # correct, and the next process simply misses it.
            self._count_extra("persist_failures")

    def _count_extra(self, name: str) -> None:
        self._stats.extra[name] = self._stats.extra.get(name, 0) + 1

    # -- residency and the cache budget --------------------------------

    def _resident(
        self, entry: _Entry, *, candidate_text: str | None = None
    ) -> _Entry | None:
        """Bring an entry's text and native handle back, or ``None``."""
        if entry.text is not None and entry.handle is not None:
            return entry
        if self._directory is None:
            return None
        store, encoder = self._backend()
        try:
            raw = self._record_path(entry.name).read_bytes()
            view = records.decode_record(raw)
            if view.stable_prefix_byte_length == 0:
                historical_text = view.text_tail
                entry.handle = store.import_session(self._key_id, raw, encoder)
            else:
                if candidate_text is None:
                    return None
                binding = self._recovery_path(entry.name).read_bytes()
                entry.handle, historical_chars = store.import_session_with_binding(
                    self._key_id,
                    raw,
                    candidate_text,
                    binding,
                    encoder,
                )
                historical_text = candidate_text[:historical_chars]
                entry.index_entry = _native_index_entry(
                    store.content_index_entry(entry.handle)
                )
            entry.text = historical_text
            entry.byte_length = view.full_text_byte_length
            entry.revision = view.session_revision
        except _MISS_ERRORS:
            self._drop(entry, remove_file=False)
            return None
        self._evict_over_budget(pin=entry.name)
        return entry

    def _resident_bytes(self) -> int:
        return sum(
            item.byte_length for item in self._entries.values() if item.text is not None
        )

    def _touch(self, entry: _Entry) -> None:
        self._entries.pop(entry.name, None)
        self._entries[entry.name] = entry

    def _evict_over_budget(self, *, pin: str) -> None:
        if self._resident_bytes() <= self._budget:
            return
        for name in list(self._entries):
            if name == pin:
                continue
            entry = self._entries[name]
            if entry.text is None:
                continue
            self._evict_entry(entry)
            if self._resident_bytes() <= self._budget:
                return

    def _evict_entry(self, entry: _Entry) -> None:
        """Drop the in-memory layer of one entry.

        With a persistent store the record file stays and the entry can
        be reloaded; without one the entry is gone and the next call for
        its content is a full encode. Either way only speed is lost.
        """
        self._stats.entries_evicted += 1
        if entry.handle is not None:
            store, _ = self._backend()
            store.evict(entry.handle)
        if self._directory is None:
            self._entries.pop(entry.name, None)
            self._index.discard(entry.name)
        else:
            entry.text = None
            entry.handle = None

    def _drop(self, entry: _Entry, *, remove_file: bool) -> None:
        if entry.handle is not None:
            store, _ = self._backend()
            store.evict(entry.handle)
        self._entries.pop(entry.name, None)
        self._index.discard(entry.name)
        if remove_file and self._directory is not None:
            with suppress(OSError):
                self._record_path(entry.name).unlink(missing_ok=True)
            with suppress(OSError):
                self._recovery_path(entry.name).unlink(missing_ok=True)

    def _cap_auto_entries(self) -> None:
        auto_names = [
            name for name, item in self._entries.items() if item.kind == "auto"
        ]
        while len(auto_names) > MAX_AUTO_ENTRIES:
            oldest = auto_names.pop(0)
            self._drop(self._entries[oldest], remove_file=True)

    # -- writing -------------------------------------------------------

    def _write_entry(
        self,
        name: str,
        kind: str,
        text: str,
        row: IndexEntry | None = None,
    ) -> list[int] | None:
        """Full encode of ``text`` into a fresh entry; ids or ``None``."""
        store, encoder = self._backend()
        previous = self._entries.get(name)
        try:
            if previous is not None and previous.handle is not None:
                store.evict(previous.handle)
            handle, revision, _count = store.put(self._key_id, text, encoder)
            ids = _ids_from_bytes(store.ids_bytes(handle))
        except _MISS_ERRORS:
            if previous is not None:
                self._drop(previous, remove_file=False)
            self._stats.degraded += 1
            return None
        row = _native_index_entry(store.content_index_entry(handle))
        if row is None:  # pragma: no cover - native tracking invariant
            store.evict(handle)
            self._stats.degraded += 1
            return None
        entry = _Entry(
            name=name,
            kind=kind,
            byte_length=row.byte_length,
            text=text,
            handle=handle,
            revision=revision,
            index_entry=row,
        )
        self._entries.pop(name, None)
        self._entries[name] = entry
        assert entry.index_entry is not None
        self._index.put(name, entry.index_entry)
        self._cap_auto_entries()
        self._persist_entry(entry)
        self._persist_index()
        self._evict_over_budget(pin=name)
        return ids

    def _append_entry(self, entry: _Entry, delta: str) -> list[int] | None:
        """Append ``delta`` to a resident entry; new full ids or ``None``."""
        store, encoder = self._backend()
        assert entry.handle is not None and entry.text is not None
        try:
            outcome = store.append(entry.handle, delta, entry.revision, encoder)
        except _MISS_ERRORS:
            self._drop(entry, remove_file=False)
            self._stats.degraded += 1
            return None
        entry.text = entry.text + delta
        entry.revision = _outcome_revision(outcome)
        entry.index_entry = _native_index_entry(
            store.content_index_entry(entry.handle)
        )
        if entry.index_entry is None:  # pragma: no cover - native invariant
            self._drop(entry, remove_file=False)
            self._stats.degraded += 1
            return None
        entry.byte_length = entry.index_entry.byte_length
        self._index.put(entry.name, entry.index_entry)
        self._persist_entry(entry)
        self._persist_index()
        self._evict_over_budget(pin=entry.name)
        return _outcome_ids(outcome)

    # -- public paths --------------------------------------------------

    def encode_session(self, session_id: str, text: str) -> list[int] | None:
        """Serve one named-session encode; ``None`` asks the caller to
        run the plain routed path instead (never a wrong answer)."""
        name = _entry_name("session", session_id)
        entry = self._entries.get(name)
        if entry is not None:
            try:
                entry = self._resident(entry, candidate_text=text)
            except _CandidateMismatch:
                self._stats.session_overwrites += 1
                return self._write_entry(name, "session", text)
        if entry is None:
            self._stats.session_misses += 1
            return self._write_entry(name, "session", text)
        self._touch(entry)
        assert entry.text is not None
        if entry.text == text:
            self._stats.session_hits += 1
            return self._entry_ids(entry)
        if text.startswith(entry.text):
            ids = self._append_entry(entry, text[len(entry.text) :])
            if ids is not None:
                self._stats.session_appends += 1
            return ids
        self._stats.session_overwrites += 1
        return self._write_entry(name, "session", text)

    def encode_auto(self, text: str) -> list[int] | None:
        """Serve one content-lookup encode; ``None`` asks the caller to
        run the plain routed path (below threshold, miss that is not
        worth storing, or a store-side failure)."""
        data = text.encode("utf-8")
        served = self._serve_from_index(text, data)
        if served is not None:
            return served
        if len(data) < AUTO_MIN_BYTES:
            return None
        self._stats.auto_misses += 1
        row = entry_for(data)
        return self._write_entry(_entry_name("auto", row.end_digest), "auto", text, row)

    def _serve_from_index(self, text: str, data: bytes) -> list[int] | None:
        lengths = self._index.query_lengths(len(data))
        if not lengths:
            return None
        digests = prefix_digests(data, lengths)
        for _length, name in self._index.endpoint_candidates(digests):
            entry = self._entries.get(name)
            if entry is None:
                self._index.discard(name)
                continue
            try:
                entry = self._resident(entry, candidate_text=text)
            except _CandidateMismatch:
                self._stats.collision_rejects += 1
                continue
            if entry is None:
                continue
            assert entry.text is not None
            # Anti-collision hard gate: a digest proposes, bytes decide.
            if not text.startswith(entry.text):
                self._stats.collision_rejects += 1
                continue
            self._touch(entry)
            if entry.text == text:
                self._stats.auto_hits += 1
                return self._entry_ids(entry)
            return self._serve_extension(entry, text, data)
        return None

    def _serve_extension(
        self, entry: _Entry, text: str, data: bytes
    ) -> list[int] | None:
        """The verified entry is a proper prefix: append the remainder."""
        assert entry.text is not None
        delta = text[len(entry.text) :]
        if entry.kind == "auto":
            ids = self._append_entry(entry, delta)
            if ids is not None:
                self._stats.auto_appends += 1
            return ids
        # Named sessions are user state: extend a fork, never the entry.
        store, encoder = self._backend()
        assert entry.handle is not None
        try:
            fork = store.fork(entry.handle)
        except _MISS_ERRORS:
            self._stats.degraded += 1
            return None
        try:
            outcome = store.append(fork, delta, 0, encoder)
            ids = _outcome_ids(outcome)
        except _MISS_ERRORS:
            store.evict(fork)
            self._stats.degraded += 1
            return None
        self._stats.auto_appends += 1
        if len(data) >= AUTO_MIN_BYTES:
            self._adopt_fork(fork, text, data, _outcome_revision(outcome))
        else:
            store.evict(fork)
        return ids

    def _adopt_fork(
        self, fork: int, text: str, data: bytes, revision: int
    ) -> None:
        """Keep an extended fork as a fresh auto entry."""
        store, _ = self._backend()
        row = _native_index_entry(store.content_index_entry(fork))
        if row is None:  # pragma: no cover - native tracking invariant
            store.evict(fork)
            self._stats.degraded += 1
            return
        name = _entry_name("auto", row.end_digest)
        previous = self._entries.get(name)
        if previous is not None:
            self._drop(previous, remove_file=False)
        entry = _Entry(
            name=name,
            kind="auto",
            byte_length=row.byte_length,
            text=text,
            handle=fork,
            revision=revision,
            index_entry=row,
        )
        self._entries[name] = entry
        self._index.put(name, row)
        self._cap_auto_entries()
        self._persist_entry(entry)
        self._persist_index()
        self._evict_over_budget(pin=name)

    def _entry_ids(self, entry: _Entry) -> list[int] | None:
        store, _ = self._backend()
        assert entry.handle is not None
        try:
            return _ids_from_bytes(store.ids_bytes(entry.handle))
        except _MISS_ERRORS:
            self._drop(entry, remove_file=False)
            self._stats.degraded += 1
            return None

    # -- introspection -------------------------------------------------

    def stats(self) -> dict[str, object]:
        """Counters of the store's decisions so far.

        ``*_hits`` count exact whole-text reuse only; a served prefix
        extension counts in ``*_appends`` (a success, not a miss), so
        zero hits alongside growing appends is the expected shape of an
        append-mostly workload.
        """
        counters: dict[str, object] = dict(self._stats.as_mapping())
        counters["entries"] = len(self._entries)
        counters["resident_bytes"] = self._resident_bytes()
        if self._store is not None:
            native_stats = self._store.stats()
            paths = native_stats.get("path_counts")
            if isinstance(paths, dict):
                counters["append_paths"] = dict(paths)
        if self._append_stats is not None:
            counters["repair"] = self._append_stats()
        return counters
