"""Typed surface of the private native module (crates/toktier-py).

The implementation is the pyo3 extension; this stub records the calling
shapes the Python side may rely on. Exception classes raised at the
boundary are the ``toktier.errors`` objects themselves and are therefore
not re-declared here.
"""

from collections.abc import Callable, Sequence

FORMAT_NAME: str
WITNESS_NONE_FULL_REENCODE: int
WITNESS_BPE_SYNC_TRANSITION: int
WITNESS_WORDPIECE_CONTINUATION: int
WITNESS_METASPACE_WORD_START: int

_Spans = Sequence[tuple[int, int]]
_EncodeCb = Callable[[str], tuple[Sequence[int], _Spans]]
_AppendCb = Callable[
    [str, list[int], _Spans, str], tuple[Sequence[int], _Spans, int, str]
]
_BoundaryCb = Callable[
    [str, list[int], _Spans, int, int], tuple[int, int] | None
]

class RouteSelector:
    def __init__(
        self,
        thresholds: Sequence[int],
        reference_index: int,
        gpu_head: bool,
        literal_mode: int = 0,
        literal_prefixes: Sequence[tuple[int, int]] = (),
    ) -> None: ...
    def route(self, text: str) -> tuple[int | None, int, bool, bool]: ...
    @property
    def reference_index(self) -> int: ...

class CallbackEncoder:
    def __init__(
        self,
        witness_category: int,
        encode_cb: _EncodeCb,
        append_cb: _AppendCb | None = None,
        boundary_cb: _BoundaryCb | None = None,
        bpe_sync_pclass: bytes | None = None,
    ) -> None: ...
    @property
    def witness_category(self) -> int: ...

class SessionStore:
    def __init__(
        self,
        block_chars: int = 4096,
        tail_soft_cap_bytes: int = 65536,
        tail_hard_cap_bytes: int = 1048576,
        node_tail_cap_bytes: int = 65536,
        max_sessions: int = 1024,
    ) -> None: ...
    def register_fingerprint(
        self, fingerprint: bytes, seal_end_guard_chars: int
    ) -> int: ...
    def put(
        self, key_id: int, text: str, engine: CallbackEncoder
    ) -> tuple[int, int, int]: ...
    def append(
        self,
        handle: int,
        delta: str,
        expected_revision: int,
        engine: CallbackEncoder,
    ) -> dict[str, object]: ...
    def lookup(
        self, key_id: int, text: str, engine: CallbackEncoder
    ) -> tuple[int, int, int] | None: ...
    def fork(self, handle: int) -> int: ...
    def evict(self, handle: int) -> bool: ...
    def ids_bytes(self, handle: int) -> bytes: ...
    def revision(self, handle: int) -> int: ...
    def session_info(self, handle: int) -> dict[str, object]: ...
    def list_handles(self) -> list[int]: ...
    def stats(self) -> dict[str, object]: ...
    def export_fingerprints(self) -> list[tuple[int, bytes, int]]: ...
    def export_session(self, handle: int) -> bytes: ...
    def export_session_sidecar(self, handle: int) -> bytes: ...
    def import_session(
        self, key_id: int, rec: bytes, engine: CallbackEncoder
    ) -> int: ...
    def import_session_with_sidecar(
        self, key_id: int, rec: bytes, sidecar: bytes, engine: CallbackEncoder
    ) -> int: ...
    def export_node_items(self) -> list[tuple[bytes, bytes]]: ...
    def import_node_item(self, node_key: bytes, rec: bytes) -> bool: ...
    def corrupt_node_for_tests(self, node_key: bytes) -> bool: ...
    def save_sqlite(self, path: str) -> None: ...
    @staticmethod
    def load_sqlite(
        path: str, engine: CallbackEncoder
    ) -> tuple[SessionStore, dict[int, int]]: ...
