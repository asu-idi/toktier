"""A process-pool scheduler for the SGLang adapter's encode path.

SGLang's ``TokenizerManager`` encodes every request on the thread of its
event loop, one after another, so the tokenizer is a serial stage of
the request path: while one long transcript is being encoded, every
other request waits, and a long request occupies one core however many
the host has. The certified CPU engine adds a second constraint of its
own -- its main engine is serialized inside one process, so parallel
encodes need one process each.

This module gives the adapter (``toktier.integrations.sglang``) a pool
of worker processes, each holding one loaded tokenizer, and a policy
that decides what each idle worker takes next:

* **shared pool, shortest-first.** Every request that carries no
  session state goes to a shared queue ordered by an estimate of its
  service time (``a + b * chars``, with a content coefficient from the
  share of non-ASCII characters and an online calibration of ``b``),
  aged linearly so that no request waits without bound
  (``policy="srpt"``; ``"fifo"`` keeps arrival order).
* **chunk tasks for long requests.** A stateless request of at least
  ``chunk_min_chars`` is cut at certified stable boundaries
  (``toktier.chunked.plan``: the same cuts the chunk-parallel encoder
  makes, so the concatenation is the serial encoding by construction)
  into ``k = min(workers, kmax, ceil(chars / chunk_target))`` tasks that
  any worker may take; the request completes when its last chunk does.
  A long request therefore neither monopolizes one worker nor holds
  the shorter ones behind it, and a document whose cut search finds no
  certified cut is encoded whole.
* **session affinity.** A request that carries a session key is served
  on the session path (``encode(text, session=key)``: the stored
  transcript is a prefix of the input and only the remainder is
  encoded) by its *home* worker, chosen from the key, because the
  session store is one directory with one writing process
  (``docs/serving.md`` Section 4). Its home is the only queue it can
  wait in; a worker serves its own queue and the shared queue in one
  priority order.
* **an asyncio face.** :meth:`Scheduler.encode_async` returns a future
  that a collector thread fills when the worker answers, so the event
  loop stays free while an encode runs and several requests are in
  flight at once. :meth:`Scheduler.encode_sync` is the same call for a
  caller without a loop.
* **serial fallback.** A worker that fails to start, exits, raises, or
  does not answer within ``worker_timeout_s`` makes the request fail
  *in the scheduler only*: the adapter then encodes it in-process on
  the route it would have used without the pool. Ids never depend on
  the pool; the pool only decides where and in what order the same
  encodes run.

The policy object (:class:`Policy`) is pure Python with no processes
behind it, so its rules -- ordering, chunking, homing -- are exercised
by the offline tests directly; :class:`Scheduler` adds the processes,
the queues and the clocks around it.

Configuration is a :class:`SchedulerConfig`, read from
``TOKTIER_SGLANG_SCHED_*`` by the adapter; ``workers=0`` (the default)
means no pool at all, and the adapter behaves as it did before this
module existed.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import heapq
import importlib
import multiprocessing
import os
import sys
import threading
import time
import zlib
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any

__all__ = [
    "Estimator",
    "Policy",
    "SchedResult",
    "Scheduler",
    "SchedulerConfig",
    "content_share",
    "parse_cores",
]

_ENV_PREFIX = "TOKTIER_SGLANG_SCHED_"
_POLICIES = ("srpt", "fifo")
_K_RULES = ("fixed", "idle")
_KINDS = ("free", "session", "auto")
_TASK_KINDS = ("free", "session", "auto", "chunk")
_CLOCK = time.perf_counter
#: The values an on/off setting accepts, compared without regard to case.
_FLAG_ON = ("1", "true", "yes")
_FLAG_OFF = ("0", "false", "no")


def _parse_flag(variable: str, value: str | None, default: bool) -> bool:
    """One on/off setting: unset or empty keeps ``default``, a value in
    :data:`_FLAG_ON` or :data:`_FLAG_OFF` (any case) sets it, and any other
    value raises ``ValueError`` naming ``variable``, so a mistyped setting
    is refused rather than read as on. Shared with the adapter's
    ``PER_PROCESS_STORE``."""
    if value is None or value == "":
        return default
    word = value.lower()
    if word in _FLAG_ON:
        return True
    if word in _FLAG_OFF:
        return False
    raise ValueError(
        f"{variable} must be one of {', '.join(_FLAG_ON)} (on) or "
        f"{', '.join(_FLAG_OFF)} (off), in any case (got {value!r})"
    )


def parse_cores(text: str) -> tuple[int, ...]:
    """``"0-3,8"`` -> ``(0, 1, 2, 3, 8)``; the empty string -> ``()``."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            out.extend(range(int(low), int(high) + 1))
        else:
            out.append(int(part))
    return tuple(out)


def content_share(text: str, sample: int = 512, slices: int = 4) -> float:
    """The share of non-ASCII characters, from a bounded sample.

    An all-ASCII string answers ``0.0`` in constant time (the
    interpreter knows); otherwise ``slices`` windows of ``sample``
    characters spread over the text are counted.
    """
    if not text or text.isascii():
        return 0.0
    length = len(text)
    if length <= sample * slices:
        counted = sum(1 for ch in text if ord(ch) >= 128)
        return counted / length
    step = length // slices
    counted = 0
    seen = 0
    for i in range(slices):
        window = text[i * step : i * step + sample]
        counted += sum(1 for ch in window if ord(ch) >= 128)
        seen += len(window)
    return counted / seen if seen else 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    """Everything the pool and the policy need.

    ``workers`` is the pool size (``0`` = no pool). ``cores`` pins worker
    ``i`` to ``cores[i % len(cores)]`` when given. ``policy`` orders the
    queues (``"srpt"`` by estimated service time with ``aging``
    milliseconds of priority per millisecond waited; ``"fifo"`` by
    arrival). Stateless requests of at least ``chunk_min_chars`` are cut
    into at most ``kmax`` chunks of about ``chunk_target`` characters,
    never more than there are workers; ``k_rule="idle"`` additionally
    caps the count by the number of idle workers at arrival. ``affinity``
    keeps keyed requests on their home worker (off: keyed requests are
    encoded stateless on any worker, and the store is not used).
    ``ms_fixed`` / ``ms_per_char`` / ``non_ascii_ratio`` /
    ``ms_per_char_session`` seed the service-time estimate; ``calibrate``
    lets observed service times move the per-character coefficients.
    ``worker_timeout_s`` bounds one task's answer (``0`` = no bound);
    ``start_timeout_s`` bounds the pool start. ``loader`` names a
    ``"module:attribute"`` callable ``(family, store, device) ->
    tokenizer`` for the workers to use instead of ``toktier.load``
    (``loader_path`` is prepended to their ``sys.path``); the tests use
    it, a deployment does not need it, and it is a constructor argument
    only (:meth:`from_env` does not read it).
    """

    workers: int = 0
    cores: tuple[int, ...] = ()
    policy: str = "srpt"
    aging: float = 1.0
    chunk_min_chars: int = 131_072
    chunk_target: int = 65_536
    kmax: int = 16
    k_rule: str = "fixed"
    affinity: bool = True
    ms_fixed: float = 0.2
    ms_per_char: float = 3.0e-5
    non_ascii_ratio: float = 5.0
    ms_per_char_session: float = 1.0e-5
    calibrate: bool = True
    calibrate_alpha: float = 0.05
    worker_timeout_s: float = 60.0
    start_timeout_s: float = 300.0
    rayon_threads: int = 1
    loader: str | None = None
    loader_path: str | None = None

    def __post_init__(self) -> None:
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if self.policy not in _POLICIES:
            raise ValueError(
                f"policy must be one of {_POLICIES!r}, not {self.policy!r}"
            )
        if self.k_rule not in _K_RULES:
            raise ValueError(f"k_rule must be one of {_K_RULES!r}, not {self.k_rule!r}")
        if self.aging < 0:
            raise ValueError("aging must be non-negative")
        if self.chunk_min_chars < 0 or self.chunk_target <= 0 or self.kmax < 1:
            raise ValueError(
                "chunk_min_chars must be non-negative, chunk_target and kmax positive"
            )
        if self.worker_timeout_s < 0 or self.start_timeout_s <= 0:
            raise ValueError("timeouts must be non-negative (start: positive)")
        if not 0.0 < self.calibrate_alpha <= 1.0:
            raise ValueError("calibrate_alpha must lie in (0, 1]")
        if any(core < 0 for core in self.cores):
            raise ValueError("cores must be non-negative")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> SchedulerConfig:
        """Read ``TOKTIER_SGLANG_SCHED_*`` from ``env``; unset names keep
        the defaults.

        ``env`` is required: the adapter reads the process environment
        once (:meth:`toktier.integrations.sglang.AdapterConfig.from_env`)
        and passes the same mapping here, so this module never reads the
        environment itself. Names map onto the fields: ``WORKERS``,
        ``CORES`` (``"0-3,8"``), ``POLICY``, ``AGING``,
        ``CHUNK_MIN_CHARS``, ``CHUNK_TARGET``, ``KMAX``, ``K_RULE``,
        ``AFFINITY``, ``MS_FIXED``, ``MS_PER_CHAR``, ``NON_ASCII_RATIO``,
        ``MS_PER_CHAR_SESSION``, ``CALIBRATE``, ``WORKER_TIMEOUT_S``,
        ``START_TIMEOUT_S``, ``RAYON_THREADS``. ``AFFINITY`` and
        ``CALIBRATE`` accept ``1``/``true``/``yes`` and ``0``/``false``/``no``
        in any case. An invalid value raises ``ValueError``. The test-only
        ``loader`` and ``loader_path`` fields keep their defaults here.
        """

        def get(name: str, default: Any) -> Any:
            value = env.get(_ENV_PREFIX + name)
            return default if value in (None, "") else value

        def flag(name: str, default: bool) -> bool:
            return _parse_flag(_ENV_PREFIX + name, env.get(_ENV_PREFIX + name), default)

        base = cls()
        return cls(
            workers=int(get("WORKERS", base.workers)),
            cores=parse_cores(str(get("CORES", ""))),
            policy=str(get("POLICY", base.policy)),
            aging=float(get("AGING", base.aging)),
            chunk_min_chars=int(get("CHUNK_MIN_CHARS", base.chunk_min_chars)),
            chunk_target=int(get("CHUNK_TARGET", base.chunk_target)),
            kmax=int(get("KMAX", base.kmax)),
            k_rule=str(get("K_RULE", base.k_rule)),
            affinity=flag("AFFINITY", base.affinity),
            ms_fixed=float(get("MS_FIXED", base.ms_fixed)),
            ms_per_char=float(get("MS_PER_CHAR", base.ms_per_char)),
            non_ascii_ratio=float(get("NON_ASCII_RATIO", base.non_ascii_ratio)),
            ms_per_char_session=float(
                get("MS_PER_CHAR_SESSION", base.ms_per_char_session)
            ),
            calibrate=flag("CALIBRATE", base.calibrate),
            worker_timeout_s=float(get("WORKER_TIMEOUT_S", base.worker_timeout_s)),
            start_timeout_s=float(get("START_TIMEOUT_S", base.start_timeout_s)),
            rayon_threads=int(get("RAYON_THREADS", base.rayon_threads)),
        )


# -- service-time estimate ---------------------------------------------------


class Estimator:
    """``T(kind, chars, share) = a + b[kind] * chars * (1 + ratio * share)``.

    ``share`` is the non-ASCII share of the text (:func:`content_share`);
    ``b`` is calibrated online per task kind when ``calibrate`` is set.
    Only the ordering of estimates matters to the policy, so a rough seed
    is enough; the calibration keeps the ratio between long and short
    requests close to what the machine shows.
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self.a = config.ms_fixed
        self.ratio = config.non_ascii_ratio
        self.b: dict[str, float] = {
            "free": config.ms_per_char,
            "chunk": config.ms_per_char,
            "session": config.ms_per_char_session,
            "auto": config.ms_per_char_session,
        }
        self.online = config.calibrate
        self.alpha = config.calibrate_alpha
        self.n_obs: dict[str, int] = dict.fromkeys(self.b, 0)
        self.abs_err_sum = 0.0
        self.err_n = 0

    def predict(self, kind: str, chars: int, share: float) -> float:
        return self.a + self.b[kind] * chars * (1.0 + self.ratio * share)

    def observe(self, kind: str, chars: int, share: float, service_ms: float) -> None:
        predicted = self.predict(kind, chars, share)
        self.abs_err_sum += abs(service_ms - predicted) / max(service_ms, 1e-3)
        self.err_n += 1
        self.n_obs[kind] += 1
        if not self.online or chars <= 0:
            return
        scale = chars * (1.0 + self.ratio * share)
        observed = max(0.0, (service_ms - self.a) / scale)
        self.b[kind] = (1.0 - self.alpha) * self.b[kind] + self.alpha * observed

    def summary(self) -> dict[str, Any]:
        return {
            "a_ms": self.a,
            "b_ms_per_char": dict(self.b),
            "non_ascii_ratio": self.ratio,
            "online": self.online,
            "n_obs": dict(self.n_obs),
            "rel_err_mean": (self.abs_err_sum / self.err_n) if self.err_n else None,
        }


# -- tasks and requests ------------------------------------------------------


@dataclass
class _Task:
    rid: int
    index: int
    kind: str  # free | session | auto | chunk
    text: str
    key: str | None
    chars: int
    share: float
    t_arr: float
    home: int | None = None
    estimate_ms: float = 0.0
    order: float = 0.0
    seq: int = 0
    wid: int = -1
    t_sent: float = 0.0


@dataclass
class _Request:
    rid: int
    kind: str  # free | session | auto | chunked
    key: str | None
    chars: int
    t_arr: float
    on_done: Callable[[SchedResult], None]
    tasks: list[_Task] = field(default_factory=list)
    parts: dict[int, bytes] = field(default_factory=dict)
    n_done: int = 0
    t_first: float | None = None
    t_end: float | None = None
    service_ms: float = 0.0
    wids: list[int] = field(default_factory=list)
    failed: str | None = None
    finished: bool = False


@dataclass(frozen=True)
class SchedResult:
    """What one request got back from the pool.

    ``ids`` is the core stream (``None`` when ``error`` names why the
    pool could not answer; the caller then encodes in-process).
    ``how`` is ``"session"``, ``"auto"``, ``"free"`` or ``"chunked"``; ``k`` the chunk
    count requested and ``chunks`` the count that ran; the three times
    are milliseconds from arrival to first dispatch (``wait_ms``), the
    sum of worker service (``service_ms``) and arrival to completion
    (``sojourn_ms``).
    """

    ids: list[int] | None
    error: str | None
    how: str
    k: int
    chunks: int
    wait_ms: float
    service_ms: float
    sojourn_ms: float
    wids: tuple[int, ...]


# -- the policy ----------------------------------------------------------------


class Policy:
    """Queues and rules, without processes.

    ``workers`` is the pool size; :meth:`set_alive` names which workers
    may be scheduled to (a dead worker's home requests are re-homed).
    ``planner(text, k)`` answers the chunk spans for a text, or fewer
    than ``k`` when certified cuts are missing; ``None`` disables
    chunking.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        workers: int,
        planner: Callable[[str, int], Sequence[tuple[int, int]]] | None,
        estimator: Estimator | None = None,
    ) -> None:
        self.config = config
        self.workers = workers
        self.planner = planner
        self.estimator = estimator if estimator is not None else Estimator(config)
        self.alive: list[int] = list(range(workers))
        self.local: dict[int, list[tuple[float, int, _Task]]] = {
            w: [] for w in range(workers)
        }
        self.shared: list[tuple[float, int, _Task]] = []
        self._seq = 0
        self.n_chunked = 0
        self.k_hist: dict[int, int] = {}
        self.chunks_total = 0
        self.plan_misses = 0
        self.depth_max = 0

    # -- rules ---------------------------------------------------------------

    def set_alive(self, alive: Sequence[int]) -> None:
        self.alive = sorted(alive)

    def home_of(self, key: str) -> int:
        """The worker that owns ``key``'s session state: a stable hash
        over the live workers."""
        if not self.alive:
            raise RuntimeError("no live worker")
        digest = zlib.crc32(key.encode("utf-8"))
        return self.alive[digest % len(self.alive)]

    def choose_k(self, chars: int, n_idle: int) -> int:
        cfg = self.config
        if self.planner is None or chars < cfg.chunk_min_chars:
            return 1
        k = ceil(chars / cfg.chunk_target)
        k = min(k, cfg.kmax, len(self.alive))
        if cfg.k_rule == "idle":
            k = min(k, max(1, n_idle))
        return max(1, k)

    def order_key(self, kind: str, estimate_ms: float, t_arr: float) -> float:
        """A time-invariant key that sorts like the aged priority.

        The aged priority is ``estimate - aging * (now - t_arr)``; the
        ``now`` term is common to every queued task, so the queues can
        be heaps keyed by ``estimate + aging * t_arr`` (milliseconds).
        """
        if self.config.policy == "fifo":
            return t_arr * 1e3
        return estimate_ms + self.config.aging * t_arr * 1e3

    # -- arrivals ------------------------------------------------------------

    def arrive(self, req: _Request, text: str, n_idle: int) -> list[_Task]:
        """Build ``req.tasks`` and queue them; returns the tasks."""
        share = content_share(text)
        est = self.estimator
        if req.kind in ("session", "auto"):
            assert req.key is not None
            home = self.home_of(req.key)
            task = _Task(
                req.rid, 0, req.kind, text, req.key, req.chars, share, req.t_arr
            )
            task.home = home
            task.estimate_ms = est.predict(req.kind, req.chars, share)
            req.tasks = [task]
        else:
            k = self.choose_k(req.chars, n_idle)
            spans: Sequence[tuple[int, int]] = ((0, req.chars),)
            if k > 1 and self.planner is not None:
                spans = tuple(self.planner(text, k))
                if len(spans) < k:
                    self.plan_misses += k - len(spans)
            if len(spans) > 1:
                req.kind = "chunked"
                self.n_chunked += 1
                self.k_hist[len(spans)] = self.k_hist.get(len(spans), 0) + 1
                self.chunks_total += len(spans)
                for index, (low, high) in enumerate(spans):
                    piece = text[low:high]
                    task = _Task(
                        req.rid,
                        index,
                        "chunk",
                        piece,
                        None,
                        high - low,
                        share,
                        req.t_arr,
                    )
                    task.estimate_ms = est.predict("chunk", high - low, share)
                    req.tasks.append(task)
            else:
                task = _Task(
                    req.rid, 0, "free", text, None, req.chars, share, req.t_arr
                )
                task.estimate_ms = est.predict("free", req.chars, share)
                req.tasks = [task]
        for task in req.tasks:
            self._push(task)
        return req.tasks

    def _push(self, task: _Task) -> None:
        self._seq += 1
        task.seq = self._seq
        task.order = self.order_key(task.kind, task.estimate_ms, task.t_arr)
        entry = (task.order, task.seq, task)
        if task.home is not None:
            heapq.heappush(self.local[task.home], entry)
        else:
            heapq.heappush(self.shared, entry)
        self.depth_max = max(self.depth_max, self.queued())

    def requeue(self, tasks: Sequence[_Task]) -> None:
        """Put tasks back (a worker died before answering them)."""
        for task in tasks:
            if task.home is not None and task.home not in self.alive:
                assert task.key is not None
                task.home = self.home_of(task.key)
            self._push(task)

    def next_for(self, wid: int) -> _Task | None:
        if wid not in self.alive:
            return None
        local = self.local[wid]
        shared = self.shared
        if local and (not shared or local[0][0] <= shared[0][0]):
            return heapq.heappop(local)[2]
        if shared:
            return heapq.heappop(shared)[2]
        return None

    def queued(self) -> int:
        return len(self.shared) + sum(len(q) for q in self.local.values())

    def summary(self) -> dict[str, Any]:
        return {
            "policy": self.config.policy,
            "aging": self.config.aging,
            "chunk_min_chars": self.config.chunk_min_chars,
            "chunk_target": self.config.chunk_target,
            "kmax": self.config.kmax,
            "k_rule": self.config.k_rule,
            "affinity": self.config.affinity,
            "chunking": self.planner is not None,
            "n_chunked_requests": self.n_chunked,
            "chunks_total": self.chunks_total,
            "plan_misses": self.plan_misses,
            "k_hist": {str(k): n for k, n in sorted(self.k_hist.items())},
            "queue_depth_max": self.depth_max,
            "alive": list(self.alive),
            "estimator": self.estimator.summary(),
        }


# -- the worker process ------------------------------------------------------


def _resolve_loader(spec: str) -> Callable[..., Any]:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"loader must be 'module:attribute', not {spec!r}")
    module = importlib.import_module(module_name)
    loader: Callable[..., Any] = getattr(module, attr)
    if not callable(loader):
        raise TypeError(f"{spec!r} is not callable")
    return loader


def _family_loader(family: str | None, repo_id: str | None, device: str) -> Any:
    """The default worker loader: ``toktier.load`` / ``from_pretrained``."""
    import toktier

    def load(store: str | None) -> Any:
        if family:
            return toktier.load(family, store=store, device=device)
        if repo_id:
            return toktier.from_pretrained(repo_id, store=store, device=device)
        raise ValueError("a family or a repo_id is required")

    return load


def _worker_main(
    wid: int,
    core: int | None,
    loader: str | None,
    loader_path: str | None,
    family: str | None,
    repo_id: str | None,
    device: str,
    store: str | None,
    env: Mapping[str, str],
    qin: Any,
    qres: Any,
) -> None:
    # The child sets the native libraries' thread-count variables before
    # anything loads them; it writes the environment and never reads it.
    for name, value in env.items():
        os.environ[name] = value
    if core is not None:
        with contextlib.suppress(Exception):
            os.sched_setaffinity(0, {core})
    if loader_path and loader_path not in sys.path:
        sys.path.insert(0, loader_path)
    try:
        if loader:
            tokenizer = _resolve_loader(loader)(family, store, device)
        else:
            tokenizer = _family_loader(family, repo_id, device)(store)
        encode_ids = getattr(tokenizer, "_encode_ids", None)
        if not callable(encode_ids):

            def encode_ids(text: str, **kwargs: Any) -> list[int]:
                return list(tokenizer.encode(text, **kwargs).ids)

        # A first call opens the engine before the pool is declared ready.
        encode_ids("warm-up text with numbers 12345 and symbols. " * 8, lookup="off")
    except BaseException as ex:  # the parent decides what a failed worker means
        qres.put(("failed", wid, os.getpid(), f"{type(ex).__name__}: {ex}"[:400]))
        return
    qres.put(("ready", wid, os.getpid(), None))
    while True:
        item = qin.get()
        if item[0] == "stop":
            break
        _, rid, index, kind, text, key = item
        t0 = _CLOCK()
        try:
            if kind == "session":
                ids = encode_ids(text, session=key)
            elif kind == "auto":
                ids = encode_ids(text)
            else:
                ids = encode_ids(text, lookup="off")
            payload = array("i", ids).tobytes()
            t1 = _CLOCK()
            qres.put(("done", rid, index, wid, t0, t1, payload, None))
        except BaseException as ex:
            t1 = _CLOCK()
            reason = f"{type(ex).__name__}: {ex}"[:400]
            qres.put(("done", rid, index, wid, t0, t1, None, reason))
    with contextlib.suppress(Exception):
        tokenizer.close()


# -- the scheduler -----------------------------------------------------------


class Scheduler:
    """The pool, the collector thread and the clocks around a :class:`Policy`.

    ``family`` / ``repo_id`` / ``device`` say what each worker loads
    (``loader`` overrides that with ``"module:attribute"``, a callable
    ``(family, store, device) -> tokenizer``, for tests; ``loader_path``
    is prepended to the worker's ``sys.path``). ``store`` is a directory
    under which worker ``i`` keeps its own session store ``w<i>`` (``None``:
    in-memory sessions per worker). ``planner`` is the chunk-span function
    (``None``: no chunking).
    """

    def __init__(
        self,
        config: SchedulerConfig,
        *,
        family: str | None = None,
        repo_id: str | None = None,
        device: str = "cpu",
        store: str | None = None,
        planner: Callable[[str, int], Sequence[tuple[int, int]]] | None = None,
        loader: str | None = None,
        loader_path: str | None = None,
    ) -> None:
        if config.workers < 1:
            raise ValueError("a scheduler needs at least one worker")
        self.config = config
        self.family = family
        self.repo_id = repo_id
        self.device = device
        self.store = store
        self.loader = loader if loader is not None else config.loader
        self.loader_path = (
            loader_path if loader_path is not None else config.loader_path
        )
        self.policy = Policy(config, config.workers, planner)
        self._lock = threading.Lock()
        self._procs: list[Any] = []
        self._qins: list[Any] = []
        self._qres: Any = None
        self._collector: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False
        self._closed = False
        self._idle: set[int] = set()
        self._inflight: dict[int, _Task] = {}
        self._reqs: dict[int, _Request] = {}
        self._next_rid = 0
        self._worker_pids: dict[int, int] = {}
        self._dead: dict[int, str] = {}
        self._stats: dict[str, Any] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "requeued_tasks": 0,
            "timeouts": 0,
            "by_how": {},
            "tasks_by_worker": {},
            "worker_service_ms": {},
            "start_s": None,
        }

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Spawn the workers and wait until every one has loaded.

        Raises ``RuntimeError`` when any worker fails to come up (the
        others are stopped); the caller keeps its serial path then.
        """
        if self._started:
            return
        cfg = self.config
        ctx = multiprocessing.get_context("spawn")
        self._qres = ctx.Queue()
        env = {
            "RAYON_NUM_THREADS": str(cfg.rayon_threads),
            "TOKENIZERS_PARALLELISM": "false",
        }
        t0 = _CLOCK()
        for wid in range(cfg.workers):
            core = cfg.cores[wid % len(cfg.cores)] if cfg.cores else None
            store = os.path.join(self.store, f"w{wid}") if self.store else None
            qin = ctx.Queue()
            proc = ctx.Process(
                target=_worker_main,
                args=(
                    wid,
                    core,
                    self.loader,
                    self.loader_path,
                    self.family,
                    self.repo_id,
                    self.device,
                    store,
                    env,
                    qin,
                    self._qres,
                ),
                name=f"toktier-sched-w{wid}",
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)
            self._qins.append(qin)
        ready = 0
        failures: list[str] = []
        deadline = t0 + cfg.start_timeout_s
        while ready + len(failures) < cfg.workers:
            remaining = deadline - _CLOCK()
            if remaining <= 0:
                failures.append("start timeout")
                break
            try:
                message = self._qres.get(timeout=min(remaining, 1.0))
            except Exception:
                if any(not p.is_alive() for p in self._procs):
                    failures.append("a worker exited before reporting")
                    break
                continue
            if message[0] == "ready":
                ready += 1
                self._worker_pids[message[1]] = message[2]
            elif message[0] == "failed":
                failures.append(f"worker {message[1]}: {message[3]}")
        if failures:
            self._terminate_all()
            raise RuntimeError("scheduler pool failed to start: " + "; ".join(failures))
        self._idle = set(range(cfg.workers))
        self._stats["start_s"] = round(_CLOCK() - t0, 3)
        self._started = True
        self._collector = threading.Thread(
            target=self._collect, name="toktier-sched-collector", daemon=True
        )
        self._collector.start()
        atexit.register(self.close)

    def _terminate_all(self) -> None:
        for qin in self._qins:
            with contextlib.suppress(Exception):
                qin.put(("stop",))
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                with contextlib.suppress(Exception):
                    proc.terminate()
                    proc.join(timeout=5)
            if proc.is_alive():
                with contextlib.suppress(Exception):
                    proc.kill()

    def close(self) -> None:
        """Stop the workers; pending requests fail with ``"closed"``."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        pending: list[_Request] = []
        with self._lock:
            for req in self._reqs.values():
                if not req.finished:
                    req.failed = "closed"
                    req.finished = True
                    pending.append(req)
            self._reqs.clear()
        for req in pending:
            self._finish(req, now=_CLOCK())
        self._terminate_all()
        if self._collector is not None and self._collector.is_alive():
            self._collector.join(timeout=5)

    @property
    def active(self) -> bool:
        return self._started and not self._closed and bool(self.policy.alive)

    @property
    def workers(self) -> int:
        return self.config.workers

    # -- submission ----------------------------------------------------------

    def submit(
        self,
        text: str,
        key: str | None,
        on_done: Callable[[SchedResult], None],
        *,
        kind: str = "free",
    ) -> int:
        """Queue one request; ``on_done`` runs on the collector thread.

        ``kind`` is ``"session"`` (needs ``key``; served on the home
        worker's session path), ``"free"`` (stateless, chunkable) or
        ``"auto"`` (content lookup on a home worker chosen from the
        text's start, so a later prefix-extending request finds the
        store entry there; not chunked).
        """
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS!r}, not {kind!r}")
        if not self.active:
            raise RuntimeError("scheduler is not running")
        now = _CLOCK()
        with self._lock:
            rid = self._next_rid
            self._next_rid += 1
            req = _Request(rid, "free", key, len(text), now, on_done)
            if kind == "session" and key is not None and self.config.affinity:
                req.kind = "session"
            elif kind == "auto":
                # Content lookup on a home chosen from the text's start, so
                # a prefix-extending request lands where the entry is.
                req.kind = "auto"
                req.key = "auto:" + str(zlib.crc32(text[:4096].encode("utf-8")))
            self._reqs[rid] = req
            self._stats["submitted"] += 1
            self.policy.arrive(req, text, len(self._idle))
            self._dispatch(now)
        return rid

    def encode_sync(
        self,
        text: str,
        key: str | None,
        *,
        kind: str = "free",
        timeout: float | None = None,
    ) -> SchedResult:
        """:meth:`submit` and wait for the answer on this thread."""
        done = threading.Event()
        box: list[SchedResult] = []

        def on_done(result: SchedResult) -> None:
            box.append(result)
            done.set()

        self.submit(text, key, on_done, kind=kind)
        if not done.wait(timeout):
            return SchedResult(None, "sync wait timeout", kind, 1, 0, 0.0, 0.0, 0.0, ())
        return box[0]

    async def encode_async(
        self, text: str, key: str | None, *, kind: str = "free"
    ) -> SchedResult:
        """:meth:`submit` with the answer delivered to the running loop."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SchedResult] = loop.create_future()

        def on_done(result: SchedResult) -> None:
            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(_set_result, future, result)

        self.submit(text, key, on_done, kind=kind)
        return await future

    # -- internals -----------------------------------------------------------

    def _dispatch(self, now: float) -> None:
        # Caller holds the lock.
        for wid in list(self._idle):
            task = self.policy.next_for(wid)
            if task is None:
                continue
            self._idle.discard(wid)
            task.wid = wid
            task.t_sent = now
            self._inflight[wid] = task
            req = self._reqs.get(task.rid)
            if req is not None and req.t_first is None:
                req.t_first = now
            self._qins[wid].put(
                ("task", task.rid, task.index, task.kind, task.text, task.key)
            )

    def _collect(self) -> None:
        assert self._qres is not None
        last_sweep = _CLOCK()
        while not self._stop.is_set():
            try:
                message = self._qres.get(timeout=0.25)
            except Exception:
                message = None
            now = _CLOCK()
            finished: list[_Request] = []
            with self._lock:
                if message is not None and message[0] == "done":
                    self._on_done(message, now, finished)
                if now - last_sweep >= 0.25:
                    last_sweep = now
                    self._sweep(now, finished)
                self._dispatch(_CLOCK())
            for req in finished:
                self._finish(req, now=now)

    def _on_done(
        self, message: tuple[Any, ...], now: float, finished: list[_Request]
    ) -> None:
        _, rid, index, wid, t0, t1, payload, error = message
        task = self._inflight.pop(wid, None)
        self._idle.add(wid)
        stats = self._stats
        by_worker = stats["tasks_by_worker"]
        by_worker[str(wid)] = by_worker.get(str(wid), 0) + 1
        stats["worker_service_ms"][str(wid)] = stats["worker_service_ms"].get(
            str(wid), 0.0
        ) + (t1 - t0) * 1e3
        req = self._reqs.get(rid)
        if task is not None and task.rid == rid and error is None:
            self.policy.estimator.observe(
                task.kind, task.chars, task.share, (t1 - t0) * 1e3
            )
        if req is None or req.finished:
            return  # a late answer for a request that already failed
        req.n_done += 1
        req.service_ms += (t1 - t0) * 1e3
        req.wids.append(wid)
        if error is not None:
            req.failed = f"worker {wid}: {error}"
        elif payload is not None:
            req.parts[index] = payload
        if req.failed is not None or req.n_done == len(req.tasks):
            req.t_end = t1
            req.finished = True
            del self._reqs[rid]
            finished.append(req)

    def _sweep(self, now: float, finished: list[_Request]) -> None:
        """Dead workers and overdue tasks (caller holds the lock)."""
        for wid, proc in enumerate(self._procs):
            if wid in self._dead or proc.is_alive():
                continue
            self._dead[wid] = f"exited with code {proc.exitcode}"
            self._idle.discard(wid)
            alive = [w for w in self.policy.alive if w != wid]
            self.policy.set_alive(alive)
            task = self._inflight.pop(wid, None)
            if task is not None:
                req = self._reqs.get(task.rid)
                if req is not None and not req.finished:
                    req.failed = f"worker {wid} died"
                    req.finished = True
                    del self._reqs[task.rid]
                    finished.append(req)
            # Its queued session tasks move to a new home; nothing else
            # was tied to it.
            stranded = [entry[2] for entry in self.policy.local.pop(wid, [])]
            if stranded and alive:
                self.policy.requeue(stranded)
                self._stats["requeued_tasks"] += len(stranded)
            elif stranded:
                for task in stranded:
                    req = self._reqs.get(task.rid)
                    if req is not None and not req.finished:
                        req.failed = "no live worker"
                        req.finished = True
                        del self._reqs[task.rid]
                        finished.append(req)
        limit = self.config.worker_timeout_s
        if limit > 0:
            for rid, req in list(self._reqs.items()):
                overdue = req.t_first is not None and now - req.t_first > limit
                if overdue and not req.finished:
                    req.failed = f"no answer within {limit:g} s"
                    req.finished = True
                    del self._reqs[rid]
                    self._stats["timeouts"] += 1
                    finished.append(req)

    def _finish(self, req: _Request, *, now: float) -> None:
        how = req.kind
        k = len(req.tasks)
        ids: list[int] | None = None
        if req.failed is None:
            try:
                buffer = array("i")
                for index in range(k):
                    buffer.frombytes(req.parts[index])
                ids = buffer.tolist()
            except Exception as ex:  # a missing part is a scheduler defect, not an id
                req.failed = f"assembly: {type(ex).__name__}"
        t_end = req.t_end if req.t_end is not None else now
        t_first = req.t_first if req.t_first is not None else t_end
        result = SchedResult(
            ids=ids,
            error=req.failed,
            how=how,
            k=k,
            chunks=k if how == "chunked" else 1,
            wait_ms=(t_first - req.t_arr) * 1e3,
            service_ms=req.service_ms,
            sojourn_ms=(t_end - req.t_arr) * 1e3,
            wids=tuple(req.wids),
        )
        with self._lock:
            if req.failed is None:
                self._stats["completed"] += 1
                self._stats["by_how"][how] = self._stats["by_how"].get(how, 0) + 1
            else:
                self._stats["failed"] += 1
        # A callback's failure must not stop the collector.
        with contextlib.suppress(Exception):
            req.on_done(result)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._stats)
            out["by_how"] = dict(self._stats["by_how"])
            out["tasks_by_worker"] = dict(self._stats["tasks_by_worker"])
            out["worker_service_ms"] = {
                k: round(v, 3) for k, v in self._stats["worker_service_ms"].items()
            }
            out["workers"] = self.config.workers
            out["cores"] = list(self.config.cores)
            out["worker_pids"] = dict(self._worker_pids)
            out["dead"] = dict(self._dead)
            out["inflight"] = len(self._inflight)
            out["queued"] = self.policy.queued()
            out["policy"] = self.policy.summary()
            out["active"] = self.active
        return out


def _set_result(future: asyncio.Future[SchedResult], result: SchedResult) -> None:
    if not future.done():
        future.set_result(result)
