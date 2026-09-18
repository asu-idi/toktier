"""toktier as an SGLang tokenizer backend (CPU path).

SGLang's ``TokenizerManager`` holds one tokenizer object and calls
``tokenizer(texts)`` / ``tokenizer.encode(text)`` on it; everything else it
needs (``eos_token_id``, ``apply_chat_template``, ``decode`` ...) it reads
from the same object. This module wraps the Hugging Face tokenizer SGLang
already loaded in a duck-typed object that answers the encode calls from
toktier's certified CPU route and delegates every other attribute to the
wrapped tokenizer. Nothing here imports ``sglang``; the two ways to attach
the wrapper are:

* form P -- a small patch to ``sglang.srt.utils.hf_transformers.tokenizer.
  get_tokenizer`` that adds ``tokenizer_backend == "toktier"`` and calls
  :func:`wrap_tokenizer` (``patches/sglang-0.5.19-toktier-backend.patch``);
* form Z -- no change to SGLang: an injector (``sitecustomize`` on
  ``PYTHONPATH``) calls :func:`attach_to_tokenizer_manager` on the
  ``TokenizerManager`` class, which wraps the tokenizer after construction
  and binds the request's session key around ``_tokenize_one_request``.

Session state can be keyed two ways, selected by ``AdapterConfig.session_key``:

* ``"contextvar"`` -- the request's ``session_id`` (or ``cache_salt``, or a
  prefix of ``rid``) is placed in :data:`SESSION_KEY` by the caller and the
  encode runs as ``Tokenizer.encode(text, session=key)``; a request without a
  key falls back to content lookup;
* ``"auto"`` -- no key at all; ``Tokenizer.encode(text)`` asks the store
  whether a stored text is a prefix of this input (``lookup="auto"``);
* ``"off"`` -- stateless certified CPU encode (``lookup="off"``);
* ``"passthrough"`` -- the wrapped tokenizer answers; the wrapper only times
  the calls (a measurement aid for the baseline arm, not a toktier path).

Correctness posture: every path above returns the ids a from-scratch
reference encode would return (the facade contract). That contract is about
the tokenizer toktier loaded, so before the wrapper accelerates anything it
establishes that the supplied Hugging Face object *is* that tokenizer: the
supplied object's serialized fast backend (which includes tokens added
after loading) and the loader face toktier executes against are compared
section by section -- model, normalizer, pre-tokenizer, post-processor,
added tokens, truncation and padding state -- and the supplied object's
``split_special_tokens`` setting, which is not in the serialization and
changes how added tokens are matched, must be off (:func:`check_core_
identity`). A family or repository name alone cannot establish this: the
supplied object may carry another revision, other loader options or live
vocabulary changes. When the two are not equivalent, or the comparison
cannot be made, the wrapper delegates every call to the supplied tokenizer
for its whole life and records the reason (``stats()["identity"]``;
``TOKTIER_SGLANG_IDENTITY=skip`` turns the check off for tokenizers that
cannot be serialized, at the operator's risk). Admission is not a one-time
fact: the supplied object can change after construction -- ``add_tokens``
through the wrapper, an assignment such as ``split_special_tokens = True``,
or a change made on the retained object directly -- so before every
accelerated encode the wrapper compares a cheap fingerprint of the
object's mutable surface (added tokens, the split flag, normalizer,
pre-tokenizer, post-processor, truncation and padding) with the one taken
at admission, and a forwarded mutation marks it stale outright; on any
difference the full comparison runs again, and a supplied object that is
no longer equivalent latches passthrough for good (the pool, if any, is
closed), while one that still is has its shadow reference rebuilt from the
current state. The optional shadow check re-encodes each request with
the supplied tokenizer and compares id by id; in ``"sync"`` mode a
difference is corrected on the spot (the reference ids are returned) and
counted; in ``"async"`` mode it is counted after the fact, judged against an
independent copy of the supplied object (never a fresh load by name, which
could differ from it), or synchronously against the object itself when no
copy can be made. Attribute writes follow attribute reads: an assignment
SGLang makes on the tokenizer (``tokenizer.chat_template = ...`` in its
template manager) lands on the wrapped object, where the delegated methods
read it. Special tokens: the store holds the core stream, so
``add_special_tokens=True`` is served by applying the wrapped tokenizer's
single-sequence template (prefix/suffix ids derived once from probes and
re-verified) around the core ids; a tokenizer whose template cannot be
expressed that way makes the wrapper delegate those calls to the wrapped
tokenizer rather than guess.

Long stateless requests: with ``AdapterConfig.cpe_min_chars`` set (env
``TOKTIER_SGLANG_CPE_MIN_CHARS``), a request that would otherwise be a
stateless full encode -- every request in ``"off"`` mode, and a request
without a key in ``"contextvar"`` mode -- and that is at least that many
characters long is encoded through ``toktier.cpe.chunked_encode``
(certified chunk-parallel encode on the CPU route, ``docs/integration/
sglang.md``). The default leaves it off; ``TOKTIER_SGLANG_CPE_K`` forces
the chunk count (``0`` reads the shipped table).

Threading: SGLang serializes tokenizer calls (event loop, or the single
thread of ``AsyncDynamicbatchTokenizer``); the wrapper adds a lock anyway
because a toktier session has one owner. ``loop.run_in_executor`` does not
carry context variables, so with ``--enable-dynamic-batch-tokenizer`` the
``"contextvar"`` mode degrades to content lookup; that is recorded per call
(``how`` field of the timing log) rather than hidden.

Scheduled encodes (``TOKTIER_SGLANG_SCHED_WORKERS`` > 0): the wrapper
starts the process pool of :mod:`toktier.integrations.sglang_sched` and
sends every encode to it -- keyed requests to their home worker's session
path, stateless ones to the shared queue, long stateless ones as certified
chunk tasks. Form Z also patches ``_tokenize_texts`` so a single-string
(or list-of-strings) request is awaited (:meth:`SGLangToktierTokenizer.
encode_async`) instead of encoded on the event loop's thread; other entry
points use the pool synchronously. Whatever the pool answers passes through
the same template, shadow and timing path as a serial encode, and a request
the pool cannot answer is encoded in-process on the route it would have
taken without the pool.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import contextvars
import json
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .sglang_sched import SchedResult, Scheduler, SchedulerConfig

__all__ = [
    "SESSION_KEY",
    "AdapterConfig",
    "SGLangToktierTokenizer",
    "attach_to_tokenizer_manager",
    "bind_session_key",
    "check_core_identity",
    "session_key_from_request",
    "wrap_tokenizer",
]

#: "Read the context variable" marker for the explicit-key parameter.
_KEY_FROM_CONTEXT = object()

#: The session key the current request carries, bound by the caller around
#: the tokenize call (see :func:`bind_session_key`). ``None`` means "no key".
SESSION_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "toktier_sglang_session_key", default=None
)

_KEY_MODES = ("contextvar", "auto", "off", "passthrough")
_SHADOW_MODES = ("off", "sync", "async")
_IDENTITY_MODES = ("require", "skip")
_ENV_PREFIX = "TOKTIER_SGLANG_"

#: Loader-face sections that decide the core id stream of a fast tokenizer:
#: the model (vocabulary, merges, byte fallback ...), what it normalizes and
#: pre-tokenizes with, the added-token table, and the truncation and padding
#: state (absent on the toktier side, so it must be absent on the supplied
#: object as well).
_CORE_SECTIONS = (
    "model",
    "normalizer",
    "pre_tokenizer",
    "post_processor",
    "added_tokens",
    "truncation",
    "padding",
)

#: Forwarded methods of the supplied tokenizer that change what an encode
#: returns; a call through the wrapper marks admission stale.
_MUTATORS = frozenset(
    {
        "add_tokens",
        "add_special_tokens",
        "sanitize_special_tokens",
        "set_truncation_and_padding",
        "enable_truncation",
        "enable_padding",
        "no_truncation",
        "no_padding",
        "train",
        "train_from_iterator",
    }
)


@dataclass(frozen=True)
class AdapterConfig:
    """Everything the wrapper needs beyond the tokenizer it wraps.

    ``family`` names a toktier family id (``toktier.load``); when it is
    ``None`` the wrapper resolves the wrapped tokenizer's ``name_or_path``
    through ``toktier.from_pretrained`` (content-addressed admission).
    ``store`` is a directory for persistent session state; with
    ``per_process_store`` each process appends its pid so several
    ``--tokenizer-worker-num`` workers never share one store. ``timing_log``
    is a JSONL path receiving one record per encode; ``stats_file`` receives
    the counters at exit. ``long_request_encoder`` is a hook: a callable
    ``(text) -> Sequence[int]`` used for stateless encodes of inputs at
    least ``long_request_min_chars`` long; ``None`` keeps the certified
    single-threaded route. ``cpe_min_chars`` (``0`` = off) enables the
    shipped chunk-parallel encoder for stateless inputs at least that long,
    and ``cpe_k`` forces its chunk count (``0`` reads the shipped table);
    an explicit ``long_request_encoder`` takes precedence over it. Stateless
    means: every request in ``"off"`` mode, and a request without a key in
    ``"contextvar"`` mode (which otherwise takes content lookup).
    ``identity`` is ``"require"`` (the wrapper accelerates only when
    :func:`check_core_identity` finds the supplied tokenizer equivalent to
    the toktier subject, and delegates otherwise) or ``"skip"``.
    ``scheduler`` (a :class:`~toktier.integrations.sglang_sched.
    SchedulerConfig` with ``workers`` > 0) starts the process-pool
    scheduler; ``None`` keeps every encode in this process.
    """

    family: str | None = None
    repo_id: str | None = None
    session_key: str = "auto"
    store: str | None = None
    per_process_store: bool = True
    shadow: str = "off"
    shadow_rate: float = 1.0
    shadow_dump_dir: str | None = None
    shadow_queue_size: int = 256
    timing_log: str | None = None
    stats_file: str | None = None
    explain_every: int = 0
    rid_separator: str | None = None
    long_request_encoder: Callable[[str], Sequence[int]] | None = None
    long_request_min_chars: int = 1 << 30
    cpe_min_chars: int = 0
    cpe_k: int = 0
    device: str = "cpu"
    identity: str = "require"
    scheduler: SchedulerConfig | None = None

    def __post_init__(self) -> None:
        if self.session_key not in _KEY_MODES:
            raise ValueError(
                f"session_key must be one of {_KEY_MODES!r}, not {self.session_key!r}"
            )
        if self.shadow not in _SHADOW_MODES:
            raise ValueError(
                f"shadow must be one of {_SHADOW_MODES!r}, not {self.shadow!r}"
            )
        if not 0.0 <= self.shadow_rate <= 1.0:
            raise ValueError("shadow_rate must lie in [0, 1]")
        if self.cpe_min_chars < 0 or self.cpe_k < 0:
            raise ValueError("cpe_min_chars and cpe_k must be non-negative")
        if self.identity not in _IDENTITY_MODES:
            raise ValueError(
                f"identity must be one of {_IDENTITY_MODES!r}, not {self.identity!r}"
            )
        if self.scheduler is not None and self.scheduler.workers < 1:
            raise ValueError("scheduler, when given, needs at least one worker")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AdapterConfig:
        """Read ``TOKTIER_SGLANG_*`` variables; unset ones keep the defaults.

        ``TOKTIER_SGLANG_SHADOW`` accepts ``0``/``off``, ``1``/``sync`` and
        ``async``. Other names map one to one onto the fields
        (``FAMILY``, ``REPO``, ``SESSION_KEY``, ``STORE``, ``SHADOW_RATE``,
        ``SHADOW_DUMP_DIR``, ``TIMING_LOG``, ``STATS_FILE``,
        ``EXPLAIN_EVERY``, ``RID_SEPARATOR``, ``PER_PROCESS_STORE``,
        ``CPE_MIN_CHARS``, ``CPE_K``, ``IDENTITY``). The scheduler is read
        from ``TOKTIER_SGLANG_SCHED_*`` when ``TOKTIER_SGLANG_SCHED_WORKERS``
        is set to a positive number (:meth:`SchedulerConfig.from_env`).
        """
        e = os.environ if env is None else env

        def get(name: str, default: Any = None) -> Any:
            value = e.get(_ENV_PREFIX + name)
            return default if value in (None, "") else value

        shadow_raw = str(get("SHADOW", "off")).lower()
        shadow = {"0": "off", "1": "sync", "true": "sync", "false": "off"}.get(
            shadow_raw, shadow_raw
        )
        scheduler: SchedulerConfig | None = None
        if int(get("SCHED_WORKERS", 0)) > 0:
            scheduler = SchedulerConfig.from_env(e)
        return cls(
            family=get("FAMILY"),
            repo_id=get("REPO"),
            session_key=str(get("SESSION_KEY", "auto")),
            store=get("STORE"),
            per_process_store=str(get("PER_PROCESS_STORE", "1")).lower()
            not in ("0", "false"),
            shadow=shadow,
            shadow_rate=float(get("SHADOW_RATE", 1.0)),
            shadow_dump_dir=get("SHADOW_DUMP_DIR"),
            timing_log=get("TIMING_LOG"),
            stats_file=get("STATS_FILE"),
            explain_every=int(get("EXPLAIN_EVERY", 0)),
            rid_separator=get("RID_SEPARATOR"),
            cpe_min_chars=int(get("CPE_MIN_CHARS", 0)),
            cpe_k=int(get("CPE_K", 0)),
            identity=str(get("IDENTITY", "require")),
            scheduler=scheduler,
        )


# -- identity of the wrapped tokenizer ------------------------------------


def _without_offset_facts(component: Any) -> Any:
    """``component`` with the facts that decide character offsets, not ids,
    removed: ``trim_offsets`` of a ``ByteLevel`` pre-tokenizer or
    post-processor (transformers 5 sets it to ``true`` on the live object
    where the artifact file says ``false``; the id stream is the same).
    ``Sequence`` components are handled member by member."""
    if not isinstance(component, Mapping):
        return component
    kind = component.get("type")
    if kind == "Sequence":
        out = dict(component)
        for key in ("pretokenizers", "processors", "normalizers"):
            members = component.get(key)
            if isinstance(members, list):
                out[key] = [_without_offset_facts(member) for member in members]
        return out
    if kind == "ByteLevel":
        return {key: value for key, value in component.items() if key != "trim_offsets"}
    return component


def _core_sections(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: (
            _without_offset_facts(document.get(name))
            if name in ("pre_tokenizer", "post_processor")
            else document.get(name)
        )
        for name in _CORE_SECTIONS
    }


def check_core_identity(hf_tokenizer: Any, toktier_tokenizer: Any) -> dict[str, Any]:
    """Compare the supplied tokenizer's core configuration with toktier's.

    Serializes the supplied fast tokenizer (``backend_tokenizer.to_str()``,
    which includes tokens added after loading) and the loader face toktier
    executes against, and compares the sections that decide the core id
    stream (:data:`_CORE_SECTIONS`). The artifact file may predate the
    live crate's serializer defaults, so on a first difference the face is
    canonicalized through the live crate once before deciding. A
    ``ByteLevel`` component's ``trim_offsets`` decides character offsets,
    not ids, and is left out of the comparison on both sides (transformers
    5 sets it on the live object where the artifact file does not). The
    supplied object's ``split_special_tokens`` setting (and the backend's
    ``encode_special_tokens`` flag it drives) is not in the serialization;
    when it is on, added tokens are split like ordinary text, which the
    toktier subject never does, so it is a mismatch on its own.

    Returns ``{"status", "reason", "sections"}``: ``status`` is
    ``"equivalent"``, ``"mismatch"`` (``sections`` names the differing
    ones) or ``"unavailable"`` (one side cannot be serialized). The wrapper
    accelerates only on ``"equivalent"``.
    """
    backend = getattr(hf_tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        return {
            "status": "unavailable",
            "reason": "the supplied tokenizer has no serializable fast backend",
            "sections": [],
        }
    if _splits_special_tokens(hf_tokenizer):
        return {
            "status": "mismatch",
            "reason": "split_special_tokens is set on the supplied tokenizer",
            "sections": ["split_special_tokens"],
        }
    materialize = getattr(
        getattr(toktier_tokenizer, "_backend", None),
        "materialized_tokenizer_json",
        None,
    )
    if not callable(materialize):
        return {
            "status": "unavailable",
            "reason": "the toktier subject exposes no loader face",
            "sections": [],
        }
    try:
        live = _core_sections(json.loads(to_str()))
        face_text = materialize()
        face = _core_sections(json.loads(face_text))
    except Exception as ex:
        return {
            "status": "unavailable",
            "reason": f"serialization failed: {type(ex).__name__}",
            "sections": [],
        }
    if live != face:
        try:
            import tokenizers

            face = _core_sections(
                json.loads(tokenizers.Tokenizer.from_str(face_text).to_str())
            )
        except Exception:  # the raw comparison then decides
            pass
    differing = [name for name in _CORE_SECTIONS if live[name] != face[name]]
    if not differing:
        return {"status": "equivalent", "reason": None, "sections": []}
    return {
        "status": "mismatch",
        "reason": "core configuration differs in " + ", ".join(differing),
        "sections": differing,
    }


def _splits_special_tokens(hf_tokenizer: Any) -> bool:
    """The effective ``split_special_tokens`` setting of the supplied object."""
    if bool(getattr(hf_tokenizer, "split_special_tokens", False)):
        return True
    backend = getattr(hf_tokenizer, "backend_tokenizer", None)
    return bool(getattr(backend, "encode_special_tokens", False))


def _live_fingerprint(hf_tokenizer: Any) -> tuple[Any, ...] | None:
    """A cheap digest of the supplied object's mutable, encode-affecting
    surface: the added-token table, the split flag, and the serialized
    normalizer, pre-tokenizer, post-processor, truncation and padding
    state. Tens of microseconds on a real tokenizer (the model itself is
    not read; it is replaced only by an assignment the wrapper marks
    stale). ``None`` when the object does not expose that surface."""
    backend = getattr(hf_tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    try:
        added = tuple(
            sorted(
                (
                    int(index),
                    str(token.content),
                    bool(token.special),
                    bool(token.normalized),
                    bool(token.lstrip),
                    bool(token.rstrip),
                    bool(token.single_word),
                )
                for index, token in backend.get_added_tokens_decoder().items()
            )
        )

        def state(component: Any) -> Any:
            return None if component is None else component.__getstate__()

        return (
            added,
            _splits_special_tokens(hf_tokenizer),
            state(backend.normalizer),
            state(backend.pre_tokenizer),
            state(backend.post_processor),
            repr(backend.truncation),
            repr(backend.padding),
        )
    except Exception:
        return None


# -- session key plumbing -------------------------------------------------


def session_key_from_request(obj: Any, rid_separator: str | None = None) -> str | None:
    """The session key a request carries, or ``None``.

    Order: ``session_id`` (the radix-native session of SGLang >= 0.5.17),
    then ``cache_salt``, then the part of ``rid`` before ``rid_separator``
    when one is configured. A request that uses server-side continuation
    (``session_params``) sends only the new turn's text, which is not a
    transcript a toktier session could extend, so it deliberately yields
    ``None`` and takes the content-lookup path.
    """
    if getattr(obj, "session_params", None):
        return None
    sid = getattr(obj, "session_id", None)
    if isinstance(sid, str) and sid:
        return sid
    salt = getattr(obj, "cache_salt", None)
    if isinstance(salt, str) and salt:
        return salt
    rid = getattr(obj, "rid", None)
    if rid_separator and isinstance(rid, str) and rid_separator in rid:
        return rid.split(rid_separator, 1)[0]
    return None


@contextmanager
def bind_session_key(key: str | None) -> Iterator[None]:
    """Bind :data:`SESSION_KEY` for the duration of the block."""
    token = SESSION_KEY.set(key)
    try:
        yield
    finally:
        SESSION_KEY.reset(token)


# -- special-token template ------------------------------------------------


def _derive_special_template(hf: Any) -> tuple[list[int], list[int], bool]:
    """Prefix/suffix ids the wrapped tokenizer adds around a single sequence.

    Derived from two probes and cross-checked; ``ok=False`` when the
    template is not a fixed prefix/suffix (the wrapper then delegates
    ``add_special_tokens=True`` calls instead of guessing).
    """
    try:
        probes = ("a", "hello world")
        derived: list[tuple[list[int], list[int]]] = []
        for text in probes:
            with_special = list(hf(text, add_special_tokens=True)["input_ids"])
            core = list(hf(text, add_special_tokens=False)["input_ids"])
            if not core:
                return [], [], False
            start = -1
            for i in range(len(with_special) - len(core) + 1):
                if with_special[i : i + len(core)] == core:
                    start = i
                    break
            if start < 0:
                return [], [], False
            derived.append((with_special[:start], with_special[start + len(core) :]))
        if derived[0] != derived[1]:
            return [], [], False
        return derived[0][0], derived[0][1], True
    except Exception:  # a tokenizer that refuses the probes gets delegation
        return [], [], False


# -- the wrapper ------------------------------------------------------------


class SGLangToktierTokenizer:
    """Duck-typed tokenizer: encodes through toktier, delegates the rest."""

    def __init__(
        self,
        hf_tokenizer: Any,
        config: AdapterConfig | None = None,
        *,
        shadow_reference: Callable[[], Any] | None = None,
    ) -> None:
        self._hf = hf_tokenizer
        self._cfg = config if config is not None else AdapterConfig()
        self._mode = self._cfg.session_key
        self._lock = threading.Lock()
        self._calls = 0
        self._stats: dict[str, Any] = {
            "calls": 0,
            "chars": 0,
            "ids": 0,
            "ms_total": 0.0,
            "how": {},
            "shadow_checked": 0,
            "shadow_mismatch": 0,
            "shadow_dropped": 0,
            "delegated": 0,
            "pid": os.getpid(),
            "mode": self._mode,
        }
        self._prefix, self._suffix, self._template_ok = _derive_special_template(
            hf_tokenizer
        )
        self._tok: Any = None
        self._store_dir: str | None = None
        self._identity: dict[str, Any] = {
            "status": "skipped",
            "reason": None,
            "sections": [],
        }
        self._fingerprint: tuple[Any, ...] | None = None
        self._stale = False
        if self._mode != "passthrough":
            self._tok = self._load_toktier()
            if self._cfg.identity == "require":
                self._identity = check_core_identity(hf_tokenizer, self._tok)
                if self._identity["status"] != "equivalent":
                    # Not the tokenizer toktier loaded (or not provably so):
                    # the supplied object answers everything, for good.
                    self._tok.close()
                    self._tok = None
                    self._mode = "passthrough"
                else:
                    self._fingerprint = _live_fingerprint(hf_tokenizer)
        self._stats["mode"] = self._mode
        self._stats["identity"] = dict(self._identity)
        self._stats["sched_fallback"] = 0
        self._sched: Scheduler | None = None
        self._sched_note: str | None = None
        if self._cfg.scheduler is not None and self._tok is not None:
            self._sched = self._start_scheduler(self._cfg.scheduler)
        self._log_fh = None
        if self._cfg.timing_log:
            Path(self._cfg.timing_log).parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(  # noqa: SIM115 - lives as long as the wrapper
                self._cfg.timing_log, "a", buffering=1
            )
        self._shadow_ref: Any = None
        self._shadow_q: queue.Queue[Any] | None = None
        self._shadow_thread: threading.Thread | None = None
        self._shadow_mode = self._cfg.shadow
        shadow_note: str | None = None
        if self._shadow_mode == "sync":
            self._shadow_ref = hf_tokenizer
        elif self._shadow_mode == "async":
            factory = shadow_reference or self._default_shadow_factory
            try:
                self._shadow_ref = factory()
            except Exception as ex:
                # No independent copy of the supplied object: check
                # synchronously against the object itself rather than
                # asynchronously against something else.
                self._shadow_ref = hf_tokenizer
                self._shadow_mode = "sync"
                shadow_note = f"async reference unavailable ({type(ex).__name__})"
            else:
                self._shadow_q = queue.Queue(maxsize=self._cfg.shadow_queue_size)
                self._shadow_thread = threading.Thread(
                    target=self._shadow_worker,
                    name="toktier-sglang-shadow",
                    daemon=True,
                )
                self._shadow_thread.start()
        self._stats["shadow"] = self._shadow_mode
        atexit.register(self.flush)
        self._record(
            {
                "kind": "init",
                "mode": self._mode,
                "family": self.family,
                "backend": self.planned_backend,
                "store_dir": self._store_dir,
                "template_ok": self._template_ok,
                "prefix": self._prefix,
                "suffix": self._suffix,
                "shadow": self._shadow_mode,
                "shadow_note": shadow_note,
                "identity": dict(self._identity),
                "hf_class": type(hf_tokenizer).__name__,
                "cpe_min_chars": self._cfg.cpe_min_chars,
                "cpe_k": self._cfg.cpe_k,
                "scheduler": (
                    None if self._sched is None else self._sched.stats()["policy"]
                ),
                "scheduler_note": self._sched_note,
            }
        )

    # -- construction helpers ------------------------------------------

    def _start_scheduler(self, config: SchedulerConfig) -> Scheduler | None:
        """Start the pool; on any failure keep the serial path and say why."""
        planner: Callable[[str, int], Sequence[tuple[int, int]]] | None = None
        try:
            from toktier import cpe

            if cpe.is_admitted(self._tok):
                subject = self._tok

                def planner(text: str, k: int) -> Sequence[tuple[int, int]]:
                    return cpe.plan_chunks(subject, text, k=k).spans

            else:
                self._sched_note = (
                    "chunking off: " + str(cpe.stats(self._tok).get("refusal"))
                )
        except Exception as ex:  # no chunking, but the pool is still worth having
            self._sched_note = f"chunking off: {type(ex).__name__}"
        store = self._cfg.store
        if store and self._cfg.per_process_store:
            store = os.path.join(store, f"pid{os.getpid()}")
        try:
            sched = Scheduler(
                config,
                family=self._cfg.family or self.family,
                repo_id=self._cfg.repo_id or getattr(self._hf, "name_or_path", None),
                device=self._cfg.device,
                store=store,
                planner=planner,
            )
            sched.start()
        except Exception as ex:
            self._sched_note = f"pool not started: {type(ex).__name__}: {ex}"[:300]
            return None
        return sched

    def _load_toktier(self) -> Any:
        import toktier

        store = self._cfg.store
        if store and self._cfg.per_process_store:
            store = os.path.join(store, f"pid{os.getpid()}")
        self._store_dir = store
        if self._cfg.family:
            return toktier.load(self._cfg.family, store=store, device=self._cfg.device)
        repo = self._cfg.repo_id or getattr(self._hf, "name_or_path", None)
        if not repo:
            raise ValueError(
                "AdapterConfig.family or repo_id is required when the wrapped "
                "tokenizer carries no name_or_path"
            )
        return toktier.from_pretrained(repo, store=store, device=self._cfg.device)

    def _default_shadow_factory(self) -> Any:
        """An independent copy of the supplied tokenizer for the shadow thread.

        A copy of the object SGLang supplied -- configuration, revision and
        tokens added after loading included -- so the asynchronous check
        judges against the same reference the synchronous one does. A
        fresh load by name could differ from the supplied object and would
        then certify agreement with the wrong tokenizer.
        """
        import copy

        try:
            return copy.deepcopy(self._hf)
        except Exception:
            import pickle

            return pickle.loads(pickle.dumps(self._hf))

    # -- identity ------------------------------------------------------

    @property
    def family(self) -> str | None:
        return None if self._tok is None else self._tok.family

    @property
    def planned_backend(self) -> str | None:
        return None if self._tok is None else self._tok.plan.backend

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def toktier(self) -> Any:
        """The underlying ``toktier.Tokenizer`` (``None`` in passthrough)."""
        return self._tok

    @property
    def wrapped(self) -> Any:
        return self._hf

    @property
    def config(self) -> AdapterConfig:
        return self._cfg

    @property
    def identity(self) -> dict[str, Any]:
        """The outcome of :func:`check_core_identity` at construction."""
        return dict(self._identity)

    @property
    def scheduler(self) -> Scheduler | None:
        """The running pool scheduler, or ``None`` (serial encodes)."""
        return self._sched if self._sched is not None and self._sched.active else None

    def __getattr__(self, name: str) -> Any:
        hf = self.__dict__.get("_hf")
        if hf is None:
            raise AttributeError(name)
        attr = getattr(hf, name)
        if name in _MUTATORS and callable(attr):
            # A forwarded call that changes what the supplied object
            # encodes: admission is re-established before the next
            # accelerated encode.
            def forwarded(*args: Any, **kwargs: Any) -> Any:
                try:
                    return attr(*args, **kwargs)
                finally:
                    self._stale = True

            return forwarded
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        # The wrapper's own state is private (underscore) or declared on
        # the class; any other assignment is the caller configuring the
        # tokenizer it was given (``tokenizer.chat_template = ...``) and
        # lands on the wrapped object, where the delegated methods read it.
        if name.startswith("_") or hasattr(type(self), name):
            object.__setattr__(self, name, value)
            return
        hf = self.__dict__.get("_hf")
        if hf is None:
            object.__setattr__(self, name, value)
        else:
            setattr(hf, name, value)
            self._stale = True

    def __delattr__(self, name: str) -> None:
        if name.startswith("_") or hasattr(type(self), name) or name in self.__dict__:
            object.__delattr__(self, name)
            return
        hf = self.__dict__.get("_hf")
        if hf is None:
            raise AttributeError(name)
        delattr(hf, name)
        self._stale = True

    def __len__(self) -> int:
        return len(self._hf)

    def __repr__(self) -> str:
        return (
            f"SGLangToktierTokenizer(mode={self._mode!r}, family={self.family!r}, "
            f"wrapped={type(self._hf).__name__})"
        )

    # -- admission after construction -------------------------------------

    def _admitted(self) -> bool:
        """Whether the toktier path may answer the next encode.

        Cheap on the common path (a fingerprint compare); the full
        comparison runs only after a forwarded mutation or a fingerprint
        change. Drift latches passthrough; an equivalent object rebuilds
        the shadow reference.
        """
        if self._tok is None:
            return False
        if self._cfg.identity != "require":
            return True
        with self._lock:
            if self._tok is None:
                return False
            if not self._stale:
                current = _live_fingerprint(self._hf)
                if current == self._fingerprint:
                    return True
            verdict = check_core_identity(self._hf, self._tok)
            self._stale = False
            if verdict["status"] == "equivalent":
                self._fingerprint = _live_fingerprint(self._hf)
                self._refresh_shadow_reference()
                return True
            self._latch_passthrough(verdict)
            return False

    def _latch_passthrough(self, verdict: dict[str, Any]) -> None:
        """The supplied object is no longer the tokenizer toktier loaded:
        delegate everything from now on (caller holds the lock)."""
        self._identity = dict(verdict)
        self._stats["identity"] = dict(verdict)
        self._mode = "passthrough"
        self._stats["mode"] = "passthrough"
        if self._sched is not None:
            with contextlib.suppress(Exception):
                self._sched.close()
            self._sched = None
        if self._tok is not None:
            with contextlib.suppress(Exception):
                self._tok.close()
            self._tok = None
        self._record({"kind": "identity_drift", **verdict})

    def _refresh_shadow_reference(self) -> None:
        """Rebuild the asynchronous shadow copy from the current object
        (caller holds the lock, so nothing is enqueued meanwhile)."""
        if self._shadow_mode != "async" or self._shadow_q is None:
            return
        self._shadow_q.join()
        try:
            self._shadow_ref = self._default_shadow_factory()
        except Exception:
            self._shadow_ref = self._hf
            self._shadow_mode = "sync"
            self._stats["shadow"] = "sync"

    # -- the tokenizer surface SGLang calls -----------------------------

    def __call__(
        self,
        text: Any = None,
        text_pair: Any = None,
        add_special_tokens: bool = True,
        return_token_type_ids: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        if (
            self._tok is None
            or text_pair is not None
            or return_token_type_ids
            or kwargs
            or text is None
        ):
            return self._delegate_call(
                text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                return_token_type_ids=return_token_type_ids,
                **kwargs,
            )
        if isinstance(text, str):
            return {"input_ids": self._encode_one(text, add_special_tokens)}
        if all(isinstance(t, str) for t in text):
            return {
                "input_ids": [self._encode_one(t, add_special_tokens) for t in text]
            }
        return self._delegate_call(text, add_special_tokens=add_special_tokens)

    def encode(self, text: Any, add_special_tokens: bool = True, **kwargs: Any) -> Any:
        if self._tok is None or kwargs or not isinstance(text, str):
            t0 = time.perf_counter()
            out = self._hf.encode(text, add_special_tokens=add_special_tokens, **kwargs)
            self._note_delegated(text, out, (time.perf_counter() - t0) * 1e3)
            return out
        return self._encode_one(text, add_special_tokens)

    def _delegate_call(self, text: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = self._hf(text, **kwargs)
        self._note_delegated(
            text,
            out.get("input_ids") if hasattr(out, "get") else None,
            (time.perf_counter() - t0) * 1e3,
        )
        return out

    def _note_delegated(self, text: Any, ids: Any, ms: float) -> None:
        self._stats["delegated"] += 1
        chars = (
            len(text)
            if isinstance(text, str)
            else sum(len(t) for t in text if isinstance(t, str))
            if isinstance(text, (list, tuple))
            else 0
        )
        n_ids = (
            len(ids)
            if isinstance(ids, (list, tuple)) and (not ids or isinstance(ids[0], int))
            else None
        )
        self._record(
            {
                "kind": "encode",
                "how": "hf_delegate",
                "chars": chars,
                "n_ids": n_ids,
                "ms": round(ms, 4),
                "key": None,
                "shadow": None,
            }
        )

    # -- the toktier path --------------------------------------------------

    def _resolve_key(self, key: Any) -> str | None:
        """The session key for this call: the explicit one, or the bound
        context variable; only ``"contextvar"`` mode carries keys."""
        if self._mode != "contextvar":
            return None
        if key is _KEY_FROM_CONTEXT:
            return SESSION_KEY.get()
        return key if isinstance(key, str) and key else None

    def _pool_kind(self, text: str, key: str | None) -> str | None:
        """Which pool call serves this request; ``None`` keeps it here.

        A keyed request takes the session path on its home worker; a
        request in ``"off"`` mode is stateless; a keyless request in
        ``"contextvar"`` mode keeps the content lookup the serial route
        gives it unless it is long enough to be chunked (then it is
        stateless, as for the in-process chunk-parallel encode); ``"auto"``
        mode is content lookup by design. An explicit
        ``long_request_encoder`` keeps precedence over the pool for the
        requests it was configured for.
        """
        sched = self._sched
        if sched is None or not sched.active:
            return None
        stateless = key is None and self._mode in ("off", "contextvar")
        if (
            stateless
            and self._cfg.long_request_encoder is not None
            and len(text) >= self._cfg.long_request_min_chars
        ):
            return None
        if key is not None:
            return "session"
        if self._mode == "off":
            return "free"
        if (
            self._mode == "contextvar"
            and sched.policy.planner is not None
            and len(text) >= sched.config.chunk_min_chars
        ):
            return "free"
        return "auto"

    def _serial_core(
        self, text: str, key: str | None
    ) -> tuple[list[int], str, dict[str, Any]]:
        """The in-process routes: ``(core ids, how, extra facts)``."""
        extra: dict[str, Any] = {}
        with self._lock:
            # A request that would be a stateless full encode: every
            # request in "off" mode, and a keyless one in "contextvar" mode.
            stateless = key is None and self._mode in ("off", "contextvar")
            if (
                stateless
                and self._cfg.long_request_encoder is not None
                and len(text) >= self._cfg.long_request_min_chars
            ):
                core = list(self._cfg.long_request_encoder(text))
                how = "long_request_encoder"
            elif (
                stateless
                and self._cfg.cpe_min_chars > 0
                and len(text) >= self._cfg.cpe_min_chars
            ):
                from toktier.cpe import chunked_encode

                encoded = chunked_encode(self._tok, text, k=self._cfg.cpe_k or None)
                core = list(encoded.ids)
                how = "cpe"
                extra = {
                    "cpe_k": encoded.k_requested,
                    "cpe_chunks": encoded.chunks,
                    "cpe_serial_reason": encoded.serial_reason,
                }
            elif key is not None:
                core = list(self._tok.encode(text, session=key).ids)
                how = "session"
            elif self._mode == "off":
                core = list(self._tok.encode(text, lookup="off").ids)
                how = "off"
            else:
                core = list(self._tok.encode(text).ids)
                how = "auto"
        return core, how, extra

    def _note_sched_fallback(self, result: SchedResult) -> None:
        self._stats["sched_fallback"] += 1
        self._record(
            {
                "kind": "sched_fallback",
                "error": result.error,
                "how": result.how,
                "sojourn_ms": round(result.sojourn_ms, 4),
            }
        )

    @staticmethod
    def _sched_extra(result: SchedResult) -> dict[str, Any]:
        return {
            "sched_k": result.k,
            "sched_chunks": result.chunks,
            "sched_wait_ms": round(result.wait_ms, 4),
            "sched_service_ms": round(result.service_ms, 4),
            "sched_sojourn_ms": round(result.sojourn_ms, 4),
            "sched_wids": list(result.wids),
        }

    def _encode_one(
        self, text: str, add_special_tokens: bool, key: Any = _KEY_FROM_CONTEXT
    ) -> list[int]:
        if not self._admitted():
            return list(
                self._delegate_call(text, add_special_tokens=add_special_tokens)[
                    "input_ids"
                ]
            )
        if add_special_tokens and not self._template_ok:
            t0 = time.perf_counter()
            out = list(self._hf(text, add_special_tokens=True)["input_ids"])
            self._note_delegated(text, out, (time.perf_counter() - t0) * 1e3)
            return out
        t0 = time.perf_counter()
        key = self._resolve_key(key)
        kind = self._pool_kind(text, key)
        if kind is not None:
            assert self._sched is not None
            limit = self._cfg.scheduler.worker_timeout_s if self._cfg.scheduler else 0
            result = self._sched.encode_sync(
                text, key, kind=kind, timeout=(2 * limit) if limit > 0 else None
            )
            if result.error is None and result.ids is not None:
                return self._complete(
                    text,
                    add_special_tokens,
                    result.ids,
                    "sched_" + result.how,
                    self._sched_extra(result),
                    key,
                    t0,
                )
            self._note_sched_fallback(result)
            if self._tok is None:  # passthrough latched while this call waited
                return list(
                    self._delegate_call(text, add_special_tokens=add_special_tokens)[
                        "input_ids"
                    ]
                )
        core, how, extra = self._serial_core(text, key)
        return self._complete(text, add_special_tokens, core, how, extra, key, t0)

    async def encode_async(
        self, text: str, add_special_tokens: bool = True, key: Any = _KEY_FROM_CONTEXT
    ) -> list[int]:
        """Encode one text with the pool, without blocking the event loop.

        Same answer as ``__call__``; the encode runs on a worker process
        and this coroutine waits for it, so other requests proceed
        meanwhile. Without a running pool (or for a request the pool
        does not serve) this is the synchronous encode, as before.
        ``key`` is the session key; by default the bound context
        variable is read, which the form Z hook binds around the
        request.
        """
        if not self._admitted():
            return list(
                self._delegate_call(text, add_special_tokens=add_special_tokens)[
                    "input_ids"
                ]
            )
        if add_special_tokens and not self._template_ok:
            return self._encode_one(text, add_special_tokens, key=key)
        t0 = time.perf_counter()
        key = self._resolve_key(key)
        kind = self._pool_kind(text, key)
        if kind is None:
            return self._encode_one(text, add_special_tokens, key=key)
        assert self._sched is not None
        result = await self._sched.encode_async(text, key, kind=kind)
        if result.error is None and result.ids is not None:
            return self._complete(
                text,
                add_special_tokens,
                result.ids,
                "sched_" + result.how,
                self._sched_extra(result),
                key,
                t0,
            )
        self._note_sched_fallback(result)
        if self._tok is None:  # passthrough latched while this call waited
            return list(
                self._delegate_call(text, add_special_tokens=add_special_tokens)[
                    "input_ids"
                ]
            )
        core, how, extra = self._serial_core(text, key)
        return self._complete(text, add_special_tokens, core, how, extra, key, t0)

    def _complete(
        self,
        text: str,
        add_special_tokens: bool,
        core: list[int],
        how: str,
        extra: dict[str, Any],
        key: str | None,
        t0: float,
    ) -> list[int]:
        """Template, counters, timing record and shadow check, one place."""
        ids = (self._prefix + core + self._suffix) if add_special_tokens else core
        ms = (time.perf_counter() - t0) * 1e3
        self._calls += 1
        st = self._stats
        st["calls"] += 1
        st["chars"] += len(text)
        st["ids"] += len(ids)
        st["ms_total"] += ms
        st["how"][how] = st["how"].get(how, 0) + 1
        rec: dict[str, Any] = {
            "kind": "encode",
            "how": how,
            "chars": len(text),
            "n_ids": len(ids),
            "ms": round(ms, 4),
            "key": key,
            "shadow": None,
            **extra,
        }
        if self._cfg.explain_every and self._calls % self._cfg.explain_every == 0:
            try:
                summary = self._tok.explain(summary=True)
                rec["path"] = summary.get("last_execution_path")
                rec["exec_backend"] = summary.get("last_execution_backend")
            except Exception as ex:  # diagnostics never change the answer
                rec["path"] = f"explain_error:{type(ex).__name__}"
        if self._shadow_mode != "off" and self._sample_shadow():
            ids = self._shadow(text, add_special_tokens, ids, key, rec)
        self._record(rec)
        return ids

    # -- shadow check -----------------------------------------------------

    def _sample_shadow(self) -> bool:
        rate = self._cfg.shadow_rate
        if rate >= 1.0:
            return True
        # deterministic sampling over the call counter keeps runs reproducible
        return (self._calls * 7919) % 10_000 < int(rate * 10_000)

    def _shadow(
        self,
        text: str,
        add_special_tokens: bool,
        ids: list[int],
        key: str | None,
        rec: dict[str, Any],
    ) -> list[int]:
        if self._shadow_mode == "sync":
            ref = list(
                self._shadow_ref(text, add_special_tokens=add_special_tokens)[
                    "input_ids"
                ]
            )
            self._stats["shadow_checked"] += 1
            if ref == ids:
                rec["shadow"] = "eq"
                return ids
            rec["shadow"] = "ne"
            self._on_mismatch(text, add_special_tokens, ids, ref, key)
            return ref
        assert self._shadow_q is not None
        try:
            self._shadow_q.put_nowait((text, add_special_tokens, ids, key))
            rec["shadow"] = "queued"
        except queue.Full:
            self._stats["shadow_dropped"] += 1
            rec["shadow"] = "dropped"
        return ids

    def _shadow_worker(self) -> None:
        assert self._shadow_q is not None
        while True:
            item = self._shadow_q.get()
            if item is None:
                self._shadow_q.task_done()
                return
            text, add_special_tokens, ids, key = item
            try:
                ref = list(
                    self._shadow_ref(text, add_special_tokens=add_special_tokens)[
                        "input_ids"
                    ]
                )
            except Exception as ex:
                self._record({"kind": "shadow_error", "error": repr(ex)[:200]})
                self._shadow_q.task_done()
                continue
            self._stats["shadow_checked"] += 1
            if ref != ids:
                self._on_mismatch(text, add_special_tokens, ids, ref, key)
                self._record(
                    {"kind": "shadow_async", "shadow": "ne", "chars": len(text)}
                )
            else:
                self._record(
                    {"kind": "shadow_async", "shadow": "eq", "chars": len(text)}
                )
            self._shadow_q.task_done()

    def _on_mismatch(
        self,
        text: str,
        add_special_tokens: bool,
        ids: list[int],
        ref: list[int],
        key: str | None,
    ) -> None:
        self._stats["shadow_mismatch"] += 1
        first = next(
            (i for i, (x, y) in enumerate(zip(ids, ref, strict=False)) if x != y),
            min(len(ids), len(ref)),
        )
        sample = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "key": key,
            "add_special_tokens": add_special_tokens,
            "chars": len(text),
            "n_ids": len(ids),
            "n_ref": len(ref),
            "first_diff": first,
            "ids_window": ids[max(0, first - 8) : first + 8],
            "ref_window": ref[max(0, first - 8) : first + 8],
            "text": text,
        }
        if self._cfg.shadow_dump_dir:
            d = Path(self._cfg.shadow_dump_dir)
            d.mkdir(parents=True, exist_ok=True)
            name = f"mismatch_{os.getpid()}_{self._stats['shadow_mismatch']}.json"
            (d / name).write_text(json.dumps(sample, ensure_ascii=False, indent=1))
        self._record(
            {k: v for k, v in sample.items() if k != "text"} | {"kind": "mismatch"}
        )

    # -- bookkeeping ------------------------------------------------------

    def _record(self, rec: dict[str, Any]) -> None:
        if self._log_fh is None:
            return
        rec.setdefault("ts", time.time())
        rec.setdefault("pid", os.getpid())
        with contextlib.suppress(Exception):
            self._log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def stats(self) -> dict[str, Any]:
        """Counters so far plus the toktier store view when available."""
        out = dict(self._stats)
        out["how"] = dict(self._stats["how"])
        out["ms_mean"] = (
            round(out["ms_total"] / out["calls"], 4) if out["calls"] else None
        )
        if self._tok is not None:
            try:
                report = self._tok.explain()
                out["store"] = report.get("store")
                out["fallback_counts"] = report.get("fallback_counts")
                if "cpe" in report:
                    out["cpe"] = report["cpe"]
            except Exception as ex:
                out["store"] = f"explain_error:{type(ex).__name__}"
        if self._sched is not None:
            out["scheduler"] = self._sched.stats()
        if self._sched_note is not None:
            out["scheduler_note"] = self._sched_note
        return out

    def flush(self) -> None:
        """Write the stats file (if configured) and flush the timing log."""
        if self._cfg.stats_file:
            try:
                p = Path(self._cfg.stats_file)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(
                    json.dumps(self.stats(), indent=1, ensure_ascii=False) + "\n"
                )
            except Exception:
                pass
        if self._log_fh is not None:
            with contextlib.suppress(Exception):
                self._log_fh.flush()

    def close(self) -> None:
        self.flush()
        if self._sched is not None:
            self._sched.close()
        if self._shadow_q is not None:
            self._shadow_q.put(None)
        if self._tok is not None:
            self._tok.close()


def wrap_tokenizer(
    hf_tokenizer: Any, config: AdapterConfig | None = None, **kwargs: Any
) -> Any:
    """Wrap an already loaded Hugging Face tokenizer.

    ``config`` defaults to :meth:`AdapterConfig.from_env`; keyword arguments
    override individual fields. Returns the wrapper (or the tokenizer itself
    when it is already wrapped).
    """
    if isinstance(hf_tokenizer, SGLangToktierTokenizer):
        return hf_tokenizer
    cfg = config if config is not None else AdapterConfig.from_env()
    if kwargs:
        cfg = replace(cfg, **kwargs)
    return SGLangToktierTokenizer(hf_tokenizer, cfg)


# -- form Z: attach without changing SGLang ---------------------------------


def attach_to_tokenizer_manager(
    manager_cls: Any,
    config: AdapterConfig | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> bool:
    """Patch a ``TokenizerManager``-shaped class in place (idempotent).

    After the original ``__init__`` the manager's ``tokenizer`` is wrapped
    (and the dynamic-batch tokenizer's reference to it updated); around
    ``_tokenize_one_request`` the request's session key is bound, and the
    manager's own tokenize timing (``created_time`` to
    ``tokenize_finish_time``) is appended to the timing log so the SGLang
    view and the adapter view of the same request sit side by side. When
    the wrapper runs the pool scheduler, ``_tokenize_texts`` is patched
    as well: a single string, or a list of strings, is awaited through
    :meth:`SGLangToktierTokenizer.encode_async` (keeping the method's
    ``(input_ids, token_type_ids)`` contract); cross-encoder inputs and
    every other shape take the original path.
    """
    if getattr(manager_cls, "_toktier_attached", False):
        return False
    cfg = config if config is not None else AdapterConfig.from_env()
    orig_init = manager_cls.__init__
    orig_tokenize = manager_cls._tokenize_one_request
    orig_tokenize_texts = getattr(manager_cls, "_tokenize_texts", None)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            return
        wrapper = wrap_tokenizer(tok, cfg)
        self.tokenizer = wrapper
        batcher = getattr(self, "async_dynamic_batch_tokenizer", None)
        if batcher is not None and getattr(batcher, "tokenizer", None) is tok:
            batcher.tokenizer = wrapper
        if log is not None:
            log(f"toktier sglang adapter attached in pid {os.getpid()}: {wrapper!r}")

    async def _tokenize_one_request(
        self: Any, obj: Any, *args: Any, **kwargs: Any
    ) -> Any:
        key = session_key_from_request(obj, cfg.rid_separator)
        with bind_session_key(key):
            out = await orig_tokenize(self, obj, *args, **kwargs)
        wrapper = getattr(self, "tokenizer", None)
        if isinstance(wrapper, SGLangToktierTokenizer):
            try:
                st = self.rid_to_state[obj.rid].time_stats
                wrapper._record(
                    {
                        "kind": "sglang_tokenize",
                        "rid": obj.rid,
                        "key": key,
                        "ms": round(
                            (st.tokenize_finish_time - st.created_time) * 1e3, 4
                        ),
                        "text_in": obj.text is not None,
                    }
                )
            except Exception:
                pass
        return out

    async def _tokenize_texts(
        self: Any, texts: Any, is_cross_encoder: bool = False, *args: Any, **kwargs: Any
    ) -> Any:
        wrapper = getattr(self, "tokenizer", None)
        if (
            isinstance(wrapper, SGLangToktierTokenizer)
            and wrapper.scheduler is not None
            and not is_cross_encoder
            and not args
            and not kwargs
            and not getattr(
                getattr(self, "model_config", None), "is_embedding_gemma", False
            )
        ):
            if isinstance(texts, str) and texts:
                return await wrapper.encode_async(texts), None
            if (
                isinstance(texts, list)
                and texts
                and all(isinstance(t, str) for t in texts)
            ):
                rows = await asyncio.gather(*(wrapper.encode_async(t) for t in texts))
                return list(rows), None
        assert orig_tokenize_texts is not None
        return await orig_tokenize_texts(self, texts, is_cross_encoder, *args, **kwargs)

    manager_cls.__init__ = __init__
    manager_cls._tokenize_one_request = _tokenize_one_request
    if orig_tokenize_texts is not None:
        manager_cls._tokenize_texts = _tokenize_texts
    manager_cls._toktier_attached = True
    return True
