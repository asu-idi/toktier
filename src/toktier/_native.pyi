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

def fast_cpu_build_facts() -> dict[str, object]: ...
def native_host_build_facts() -> dict[str, object]: ...

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

class ReferenceEngine:
    def __init__(self, path: str) -> None: ...
    @staticmethod
    def from_bytes(data: bytes) -> ReferenceEngine: ...
    def encode(
        self, text: str, add_special_tokens: bool = True
    ) -> list[int]: ...
    def encode_with_offsets(
        self, text: str
    ) -> tuple[list[int], list[tuple[int, int]]]: ...
    def encode_batch(
        self, texts: Sequence[str], add_special_tokens: bool = True
    ) -> list[list[int]]: ...
    def decode(
        self, ids: Sequence[int], skip_special_tokens: bool = True
    ) -> str: ...
    @property
    def oracle_version(self) -> str: ...

class CallbackEncoder:
    def __init__(
        self,
        witness_category: int,
        encode_cb: _EncodeCb,
        append_cb: _AppendCb | None = None,
        boundary_cb: _BoundaryCb | None = None,
        bpe_sync_pclass: bytes | None = None,
    ) -> None: ...
    @staticmethod
    def native_fast_cpu(
        tokenizer_json: bytes,
        family: str,
        artifact_sha256: str,
        margin: int,
        effective_l_max: int,
        has_normalizer: bool,
        bpe_sync_pclass: bytes,
        reference: ReferenceEngine | None = None,
    ) -> CallbackEncoder: ...
    @property
    def witness_category(self) -> int: ...
    @property
    def native_request_path(self) -> bool: ...
    @property
    def engine_initialized(self) -> bool: ...
    @property
    def batch_worker_count(self) -> int: ...
    @property
    def minimum_seal_tail_chars(self) -> int: ...
    def stats(self) -> dict[str, object]: ...
    def encode(self, text: str) -> list[int]: ...
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def vocab(self) -> dict[int, bytes]: ...

class NativePrebuiltGpu:
    def __init__(
        self,
        family: str,
        artifact_sha256: str,
        fatbin: bytes,
        expected_fatbin_sha256: str,
        expected_architecture: str,
        device_ordinal: int,
        ruleset: str,
        digits_max: int,
        contractions: bool,
        needs_nfc: bool,
        ignore_merges: int,
        symbols: dict[str, str],
        class_table: bytes,
        pair_keys: bytes,
        pair_vals: bytes,
        byte_id: bytes,
        vocab_keys: bytes,
        vocab_vals: bytes,
        vocab_blob: bytes,
        unsafe_bits: bytes,
        pair_count: int,
        vocab_count: int,
        reference: ReferenceEngine,
    ) -> None: ...

class NativeRuntime:
    def __init__(
        self,
        fallback_chain: Sequence[str],
        minimum_input_bytes: Sequence[int],
        reference: ReferenceEngine,
        fast_encoder: CallbackEncoder | None,
        gpu_encoder: NativePrebuiltGpu | None,
        repair_fast_cpu: bool,
        fingerprint: bytes,
        seal_end_guard_chars: int,
        postprocessor_adds_tokens: bool,
        diagnostics: bool = False,
        store_directory: str | None = None,
        cache_budget_bytes: int = 134217728,
    ) -> None: ...
    def encode(
        self,
        text: str,
        session: str | None = None,
        lookup_auto: bool = True,
        add_special_tokens: bool = False,
    ) -> list[int]: ...
    def encode_batch(
        self,
        texts: Sequence[str],
        add_special_tokens: bool = False,
    ) -> list[list[int]]: ...
    def session_revision(self, session: str) -> int | None: ...
    def runtime_stats(self) -> dict[str, object]: ...
    def store_stats(self) -> dict[str, object]: ...
    @property
    def fallback_chain(self) -> list[str]: ...

class SessionStore:
    def __init__(
        self,
        block_chars: int = 4096,
        tail_soft_cap_bytes: int = 65536,
        tail_hard_cap_bytes: int = 1048576,
        node_tail_cap_bytes: int = 65536,
        max_sessions: int = 1024,
        track_recovery: bool = False,
        track_content_index: bool = False,
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
    def recovery_material(
        self, handle: int
    ) -> tuple[bytes, int, bytes] | None: ...
    def content_index_entry(
        self, handle: int
    ) -> tuple[int, str, list[tuple[int, str]]] | None: ...
    def export_recovery_binding(self, handle: int) -> bytes | None: ...
    def export_session_sidecar(self, handle: int) -> bytes: ...
    def import_session(
        self, key_id: int, rec: bytes, engine: CallbackEncoder
    ) -> int: ...
    def import_session_with_recovery(
        self,
        key_id: int,
        rec: bytes,
        historical_text: str,
        expected_material: tuple[bytes, int, bytes],
        engine: CallbackEncoder,
    ) -> int: ...
    def import_session_with_binding(
        self,
        key_id: int,
        rec: bytes,
        candidate_text: str,
        binding: bytes,
        engine: CallbackEncoder,
    ) -> tuple[int, int]: ...
    def import_session_with_sidecar(
        self, key_id: int, rec: bytes, sidecar: bytes, engine: CallbackEncoder
    ) -> int: ...
    def export_node_items(self) -> list[tuple[bytes, bytes]]: ...
    def import_node_item(self, node_key: bytes, rec: bytes) -> bool: ...
    # corrupt_node_for_tests is deliberately not declared here: the
    # binding compiles it only under its `testing` cargo feature, which
    # no shipped or development build enables, so the stub would
    # describe a method the extension does not have.
    def save_sqlite(self, path: str) -> None: ...
    @staticmethod
    def load_sqlite(
        path: str, engine: CallbackEncoder
    ) -> tuple[SessionStore, dict[int, int]]: ...
