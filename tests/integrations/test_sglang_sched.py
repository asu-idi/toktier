"""The pool scheduler behind the SGLang adapter, offline.

The policy tier needs no processes: ordering, chunk counts, homing and
re-homing are checked on :class:`Policy` directly. The pool tier spawns
real worker processes around the stub tokenizer of ``_sched_stub.py``
(one id per character, so any cut is exact) and checks the answers, the
asyncio face, the concurrency it buys, and every failure that must end
in the adapter's serial fallback: a loader that fails, a worker that
raises, a worker that exits, a worker that does not answer in time. The
adapter tier wires the pool into the wrapper and the manager hook. One
test at the end uses a real artifact when ``TOKTIER_TEST_ARTIFACTS``
names a directory of local artifacts (the convention of
``tests/test_chunked.py``) and judges pool ids -- session, stateless and
chunked -- against the serial encode of the same tokenizer.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from toktier.integrations.sglang import (
    AdapterConfig,
    SGLangToktierTokenizer,
    attach_to_tokenizer_manager,
    bind_session_key,
    tokenize_texts_with_pool,
    wrap_tokenizer,
)
from toktier.integrations.sglang_sched import (
    Estimator,
    Policy,
    Scheduler,
    SchedulerConfig,
    _Request,
    content_share,
    parse_cores,
)

from .test_sglang_adapter import _FakeToktier, _StubHf

_HERE = str(Path(__file__).resolve().parent)
_STUB = "_sched_stub:load"
_ARTIFACTS = os.environ.get("TOKTIER_TEST_ARTIFACTS")


def _even_split(text: str, k: int) -> Sequence[tuple[int, int]]:
    """Any cut is exact on the stub tokenizer; cut evenly."""
    length = len(text)
    bounds = [length * i // k for i in range(k + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(k) if bounds[i] < bounds[i + 1]]


def _config(**kw: Any) -> SchedulerConfig:
    base: dict[str, Any] = {
        "workers": 2,
        "loader": _STUB,
        "loader_path": _HERE,
        "start_timeout_s": 120.0,
        "chunk_min_chars": 32,
        "chunk_target": 16,
    }
    base.update(kw)
    return SchedulerConfig(**base)


@pytest.fixture
def pool() -> Iterator[Scheduler]:
    sched = Scheduler(_config(), planner=_even_split)
    sched.start()
    try:
        yield sched
    finally:
        sched.close()


# -- configuration ------------------------------------------------------------


def test_config_validation_and_env() -> None:
    assert SchedulerConfig().workers == 0
    with pytest.raises(ValueError):
        SchedulerConfig(workers=-1)
    with pytest.raises(ValueError):
        SchedulerConfig(policy="lifo")
    with pytest.raises(ValueError):
        SchedulerConfig(k_rule="queue")
    with pytest.raises(ValueError):
        SchedulerConfig(chunk_target=0)
    with pytest.raises(ValueError):
        SchedulerConfig(cores=(-1,))
    cfg = SchedulerConfig.from_env(
        {
            "TOKTIER_SGLANG_SCHED_WORKERS": "4",
            "TOKTIER_SGLANG_SCHED_CORES": "0-2,7",
            "TOKTIER_SGLANG_SCHED_POLICY": "fifo",
            "TOKTIER_SGLANG_SCHED_AFFINITY": "0",
            "TOKTIER_SGLANG_SCHED_CHUNK_MIN_CHARS": "1000",
            "TOKTIER_SGLANG_SCHED_KMAX": "3",
            "TOKTIER_SGLANG_SCHED_K_RULE": "idle",
            "TOKTIER_SGLANG_SCHED_WORKER_TIMEOUT_S": "0",
        }
    )
    assert cfg.workers == 4 and cfg.cores == (0, 1, 2, 7)
    assert cfg.policy == "fifo" and cfg.affinity is False
    assert cfg.chunk_min_chars == 1000 and cfg.kmax == 3 and cfg.k_rule == "idle"
    assert cfg.worker_timeout_s == 0
    assert parse_cores("") == ()
    # The scheduler reads only the mapping it is given; the adapter reads
    # the process environment once and passes it on.
    with pytest.raises(TypeError):
        SchedulerConfig.from_env()  # type: ignore[call-arg]
    # The worker loader is a constructor argument for tests, not a setting.
    loaded = SchedulerConfig.from_env(
        {
            "TOKTIER_SGLANG_SCHED_LOADER": "module:attribute",
            "TOKTIER_SGLANG_SCHED_LOADER_PATH": "/somewhere",
        }
    )
    assert loaded.loader is None and loaded.loader_path is None
    # The adapter reads the scheduler only when a worker count is set.
    assert AdapterConfig.from_env({}).scheduler is None
    adapter = AdapterConfig.from_env({"TOKTIER_SGLANG_SCHED_WORKERS": "2"})
    assert adapter.scheduler is not None and adapter.scheduler.workers == 2
    with pytest.raises(ValueError):
        AdapterConfig(scheduler=SchedulerConfig(workers=0))


def test_content_share_is_bounded_and_sampled() -> None:
    assert content_share("") == 0.0
    assert content_share("ascii only") == 0.0
    assert content_share("\u00e9" * 10) == 1.0
    mixed = ("a" + "\u00e9") * 100
    assert abs(content_share(mixed) - 0.5) < 1e-9
    long_text = ("abc" + "\u4e2d") * 100_000
    share = content_share(long_text, sample=64, slices=4)
    assert 0.2 <= share <= 0.3


# -- the policy, without processes ------------------------------------------------


def _req(
    rid: int, kind: str, chars: int, key: str | None = None, t: float = 0.0
) -> _Request:
    return _Request(rid, kind, key, chars, t, lambda result: None)


def test_srpt_orders_by_estimate_and_fifo_by_arrival() -> None:
    policy = Policy(SchedulerConfig(workers=2, policy="srpt"), 2, None)
    policy.arrive(_req(0, "free", 10_000, t=0.0), "x" * 10_000, 2)
    policy.arrive(_req(1, "free", 100, t=0.0001), "y" * 100, 2)
    first = policy.next_for(0)
    assert first is not None and first.rid == 1  # the short one, though later
    second = policy.next_for(1)
    assert second is not None and second.rid == 0
    assert policy.next_for(0) is None
    fifo = Policy(SchedulerConfig(workers=2, policy="fifo"), 2, None)
    fifo.arrive(_req(0, "free", 10_000, t=0.0), "x" * 10_000, 2)
    fifo.arrive(_req(1, "free", 100, t=0.0001), "y" * 100, 2)
    first = fifo.next_for(0)
    assert first is not None and first.rid == 0


def test_aging_lets_a_long_request_overtake_after_waiting() -> None:
    # 1 ms of priority per ms waited: a request 5 ms "longer" that arrived
    # 6 ms earlier sorts first.
    cfg = SchedulerConfig(workers=1, ms_fixed=0.0, ms_per_char=1.0)
    policy = Policy(cfg, 1, None)
    policy.arrive(_req(0, "free", 10, t=0.0), "x" * 10, 1)  # estimate 10 ms
    policy.arrive(_req(1, "free", 5, t=0.006), "y" * 5, 1)  # estimate 5 ms
    first = policy.next_for(0)
    assert first is not None and first.rid == 0
    strict = Policy(
        SchedulerConfig(workers=1, aging=0.0, ms_fixed=0.0, ms_per_char=1.0), 1, None
    )
    strict.arrive(_req(0, "free", 10, t=0.0), "x" * 10, 1)
    strict.arrive(_req(1, "free", 5, t=0.006), "y" * 5, 1)
    first = strict.next_for(0)
    assert first is not None and first.rid == 1


def test_chunk_count_rules() -> None:
    cfg = SchedulerConfig(workers=4, chunk_min_chars=100, chunk_target=50, kmax=3)
    policy = Policy(cfg, 4, _even_split)
    assert policy.choose_k(99, 4) == 1  # below the threshold
    assert policy.choose_k(100, 4) == 2
    assert policy.choose_k(1000, 4) == 3  # kmax binds before the pool size
    wide = Policy(
        SchedulerConfig(workers=2, chunk_min_chars=100, chunk_target=50), 2, _even_split
    )
    assert wide.choose_k(1000, 2) == 2  # the pool size binds
    idle = Policy(
        SchedulerConfig(workers=4, chunk_min_chars=100, chunk_target=50, k_rule="idle"),
        4,
        _even_split,
    )
    assert idle.choose_k(1000, 0) == 1
    assert idle.choose_k(1000, 2) == 2
    assert idle.choose_k(1000, 4) == 4
    none = Policy(cfg, 4, None)
    assert none.choose_k(10_000, 4) == 1  # no planner, no chunking


def test_long_request_becomes_chunk_tasks_and_short_ones_do_not() -> None:
    cfg = SchedulerConfig(workers=4, chunk_min_chars=100, chunk_target=50)
    policy = Policy(cfg, 4, _even_split)
    req = _req(0, "free", 200)
    tasks = policy.arrive(req, "x" * 200, 4)
    assert req.kind == "chunked" and len(tasks) == 4
    assert [t.kind for t in tasks] == ["chunk"] * 4
    assert [(t.index, t.chars) for t in tasks] == [(0, 50), (1, 50), (2, 50), (3, 50)]
    assert all(t.home is None for t in tasks)
    short = _req(1, "free", 20)
    assert [t.kind for t in policy.arrive(short, "y" * 20, 4)] == ["free"]
    assert short.kind == "free"
    # A planner that finds fewer cuts than asked: fewer chunks, counted.
    sparse = Policy(
        cfg, 4, lambda text, k: [(0, len(text) // 2), (len(text) // 2, len(text))]
    )
    req2 = _req(2, "free", 400)
    assert len(sparse.arrive(req2, "z" * 400, 4)) == 2
    assert sparse.plan_misses == 2 and sparse.k_hist == {2: 1}


def test_keyed_requests_are_homed_and_rehomed() -> None:
    policy = Policy(SchedulerConfig(workers=3), 3, None)
    homes = {policy.home_of(f"conv-{i}") for i in range(50)}
    assert homes == {0, 1, 2}
    key = "conv-7"
    home = policy.home_of(key)
    req = _req(0, "session", 10, key=key)
    (task,) = policy.arrive(req, "x" * 10, 3)
    assert task.home == home and task.kind == "session"
    other = next(w for w in range(3) if w != home)
    assert policy.next_for(other) is None  # only the home may take it
    assert policy.next_for(home) is task
    # The home dies: its queued tasks move to a live worker.
    (task2,) = policy.arrive(_req(1, "session", 10, key=key), "y" * 10, 3)
    policy.set_alive([w for w in range(3) if w != home])
    stranded = [entry[2] for entry in policy.local.pop(home)]
    policy.requeue(stranded)
    assert task2.home in policy.alive and task2.home == policy.home_of(key)
    assert policy.next_for(task2.home) is task2
    assert policy.next_for(home) is None  # a dead worker is never scheduled


def test_worker_serves_local_and_shared_in_one_order() -> None:
    cfg = SchedulerConfig(
        workers=2, ms_fixed=0.0, ms_per_char=1.0, ms_per_char_session=1.0
    )
    policy = Policy(cfg, 2, None)
    home = policy.home_of("k")
    policy.arrive(_req(0, "session", 100, key="k", t=0.0), "x" * 100, 2)  # local
    policy.arrive(_req(1, "free", 10, t=0.0), "y" * 10, 2)  # shared, 10 ms
    first = policy.next_for(home)
    assert first is not None and first.rid == 1  # the shorter shared one first
    second = policy.next_for(home)
    assert second is not None and second.rid == 0
    assert policy.queued() == 0


def test_estimator_calibrates_per_kind() -> None:
    est = Estimator(
        SchedulerConfig(workers=1, ms_fixed=0.0, ms_per_char=1.0, non_ascii_ratio=1.0)
    )
    assert est.predict("free", 100, 0.0) == 100.0
    assert est.predict("free", 100, 0.5) == 150.0
    for _ in range(200):
        est.observe("free", 100, 0.0, 10.0)  # the machine is 10x faster
    assert 9.0 < est.predict("free", 100, 0.0) < 11.0
    assert est.predict("chunk", 100, 0.0) == 100.0  # other kinds untouched
    frozen = Estimator(
        SchedulerConfig(workers=1, calibrate=False, ms_per_char=1.0, ms_fixed=0.0)
    )
    frozen.observe("free", 100, 0.0, 10.0)
    assert frozen.predict("free", 100, 0.0) == 100.0
    assert frozen.summary()["n_obs"]["free"] == 1


# -- the pool, with stub workers -----------------------------------------------------


def test_pool_answers_free_session_auto_and_chunked(pool: Scheduler) -> None:
    assert pool.active and pool.workers == 2
    free = pool.encode_sync("hello", None)
    assert free.error is None and free.ids == [ord(c) for c in "hello"]
    assert free.how == "free" and free.k == 1 and len(free.wids) == 1
    keyed = pool.encode_sync("hello world", "conv-1", kind="session")
    assert keyed.ids == [ord(c) for c in "hello world"] and keyed.how == "session"
    assert keyed.wids == (pool.policy.home_of("conv-1"),)
    again = pool.encode_sync("hello world, again", "conv-1", kind="session")
    assert again.wids == keyed.wids  # affinity: same home every time
    auto = pool.encode_sync("content lookup text", None, kind="auto")
    assert auto.ids == [ord(c) for c in "content lookup text"] and auto.how == "auto"
    long_text = "".join(chr(97 + i % 26) for i in range(200))
    chunked = pool.encode_sync(long_text, None)
    assert chunked.error is None and chunked.ids == [ord(c) for c in long_text]
    assert chunked.how == "chunked" and chunked.k == 2 and chunked.chunks == 2
    assert sorted(chunked.wids) == [0, 1]  # both workers took a chunk
    stats = pool.stats()
    assert stats["completed"] == 5 and stats["failed"] == 0
    assert stats["by_how"] == {"free": 1, "session": 2, "auto": 1, "chunked": 1}
    assert stats["policy"]["k_hist"] == {"2": 1}
    assert stats["worker_pids"].keys() == {0, 1}


def test_async_face_overlaps_requests(pool: Scheduler) -> None:
    async def run() -> tuple[list[Any], float]:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            pool.encode_async("slow one", None),
            pool.encode_async("slow two", None),
        )
        return list(results), time.perf_counter() - t0

    results, elapsed = asyncio.run(run())
    expected = [[ord(c) for c in "slow one"], [ord(c) for c in "slow two"]]
    assert [r.ids for r in results] == expected
    # Two 0.3 s encodes on two workers: well under the serial 0.6 s.
    assert elapsed < 0.55, elapsed
    assert {r.wids[0] for r in results} == {0, 1}


def test_worker_exception_is_reported_not_raised(pool: Scheduler) -> None:
    result = pool.encode_sync("this text goes boom", None)
    assert result.ids is None and result.error is not None
    assert "stub refuses" in result.error
    assert pool.active  # the worker is fine; only that text was refused
    assert pool.encode_sync("ok", None).ids == [111, 107]
    assert pool.stats()["failed"] == 1


def test_failing_loader_stops_the_pool_start() -> None:
    sched = Scheduler(_config(loader="_sched_stub:load_failing"))
    with pytest.raises(RuntimeError, match="always fails"):
        sched.start()
    assert not sched.active
    sched.close()


def test_dead_worker_is_retired_and_its_sessions_rehomed() -> None:
    sched = Scheduler(
        _config(loader="_sched_stub:load_exiting", worker_timeout_s=5.0)
    )
    sched.start()
    try:
        keys = (f"c{i}" for i in range(100))
        key = next(k for k in keys if sched.policy.home_of(k) == 0)
        assert sched.encode_sync("first", key, kind="session").wids == (0,)
        died = sched.encode_sync("please die", None)
        assert died.ids is None and died.error is not None and "died" in died.error
        deadline = time.perf_counter() + 5
        while sched.policy.alive != [1] and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert sched.policy.alive == [1] and sched.active
        moved = sched.encode_sync("second", key, kind="session")
        assert moved.error is None and moved.wids == (1,)  # re-homed
        assert sched.encode_sync("still served", None).wids == (1,)
        assert sched.stats()["dead"] == {0: "exited with code 3"}
    finally:
        sched.close()


def test_overdue_task_fails_the_request_and_the_pool_recovers() -> None:
    sched = Scheduler(_config(workers=1, worker_timeout_s=0.1))
    sched.start()
    try:
        late = sched.encode_sync("slow text", None)
        assert late.ids is None and late.error is not None
        assert "no answer within" in late.error
        assert sched.stats()["timeouts"] == 1
        # The worker finishes the slow text on its own; the pool goes on.
        assert sched.encode_sync("fine", None).ids == [ord(c) for c in "fine"]
    finally:
        sched.close()


def test_close_fails_pending_requests() -> None:
    sched = Scheduler(_config(workers=1))
    sched.start()
    seen: list[Any] = []
    sched.submit("slow text", None, seen.append)
    sched.submit("queued behind it", None, seen.append)
    sched.close()
    assert not sched.active
    assert len(seen) == 2 and all(r.error == "closed" for r in seen)
    with pytest.raises(RuntimeError):
        sched.submit("after close", None, seen.append)


# -- the adapter over the pool ------------------------------------------------


@pytest.fixture
def wrapped(monkeypatch: pytest.MonkeyPatch) -> Iterator[SGLangToktierTokenizer]:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    w = wrap_tokenizer(
        _StubHf(bos=1),
        AdapterConfig(session_key="contextvar", scheduler=_config()),
    )
    try:
        yield w
    finally:
        w.close()


def test_wrapper_routes_calls_through_the_pool(
    wrapped: SGLangToktierTokenizer,
) -> None:
    w = wrapped
    assert w.scheduler is not None and w.toktier is not None
    assert w.stats()["scheduler_note"].startswith("chunking off")  # no roster entry
    assert w("ab")["input_ids"] == [1, 97, 98]  # template applied over pool ids
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    with bind_session_key("conv"):
        assert w.encode("abc", add_special_tokens=False) == [97, 98, 99]
    assert w(["ab", "c"])["input_ids"] == [[1, 97, 98], [1, 99]]
    ids = asyncio.run(w.encode_async("abcd"))
    assert ids == [1, 97, 98, 99, 100]
    # Without a chunk planner (the fake has no roster entry) a long keyless
    # request keeps content lookup as well; the artifact tier covers the
    # chunked route.
    long_text = "x" * 64
    assert w.encode(long_text, add_special_tokens=False) == [ord("x")] * 64
    assert w.stats()["how"]["sched_auto"] == 6
    # Keyless requests in contextvar mode keep content lookup.
    assert w.stats()["how"] == {"sched_auto": 6, "sched_session": 1}
    assert w.toktier.calls == []  # the in-process subject encoded nothing
    assert w.stats()["scheduler"]["completed"] == 7


def test_wrapper_falls_back_serially_when_the_pool_cannot_answer(
    wrapped: SGLangToktierTokenizer,
) -> None:
    w = wrapped
    assert w.encode("boom", add_special_tokens=False) == [98, 111, 111, 109]
    assert w.stats()["sched_fallback"] == 1
    assert w.toktier.calls == [{}]  # the serial route (content lookup) answered
    assert w.stats()["how"] == {"auto": 1}
    with bind_session_key("conv"):
        assert asyncio.run(w.encode_async("boom", add_special_tokens=False)) == [
            98,
            111,
            111,
            109,
        ]
    assert w.toktier.calls[-1] == {"session": "conv"}
    assert w.stats()["sched_fallback"] == 2


def test_pool_start_failure_keeps_the_serial_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    w = wrap_tokenizer(
        _StubHf(),
        AdapterConfig(
            session_key="off", scheduler=_config(loader="_sched_stub:load_failing")
        ),
    )
    assert w.scheduler is None
    assert w.stats()["scheduler_note"].startswith("pool not started")
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    assert fake.calls == [{"lookup": "off"}] and w.stats()["how"] == {"off": 1}
    w.close()


def test_manager_hook_awaits_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    calls: list[Any] = []

    class Manager:
        def __init__(self) -> None:
            self.tokenizer: Any = _StubHf(bos=1)
            self.async_dynamic_batch_tokenizer = None
            self.rid_to_state: dict[str, Any] = {}
            self.model_config = None

        async def _tokenize_one_request(self, obj: Any) -> Any:
            ids, _ = await self._tokenize_texts(obj.text)
            return ids

        async def _tokenize_texts(
            self, texts: Any, is_cross_encoder: bool = False
        ) -> Any:
            calls.append((texts, is_cross_encoder))
            out = self.tokenizer(texts, return_token_type_ids=is_cross_encoder)
            return out["input_ids"], out.get("token_type_ids")

    assert attach_to_tokenizer_manager(
        Manager, AdapterConfig(session_key="contextvar", scheduler=_config())
    )
    m = Manager()
    try:
        assert isinstance(m.tokenizer, SGLangToktierTokenizer)
        assert m.tokenizer.scheduler is not None

        class Req:
            text = "hi"
            session_id = "conv"
            session_params = None
            cache_salt = None
            rid = "r1"

        assert asyncio.run(m._tokenize_one_request(Req())) == [1, 104, 105]
        assert calls == []  # the pool answered; the original was not called
        rows, types = asyncio.run(m._tokenize_texts(["ab", "c"]))
        assert rows == [[1, 97, 98], [1, 99]] and types is None
        # Cross-encoder inputs take the original path.
        asyncio.run(m._tokenize_texts([["q", "d"]], True))
        assert calls == [([["q", "d"]], True)]
        hows = m.tokenizer.stats()["how"]
        assert hows == {"sched_session": 1, "sched_auto": 2}
    finally:
        m.tokenizer.close()


def test_patched_manager_offers_texts_to_the_pool_first(
    wrapped: SGLangToktierTokenizer,
) -> None:
    """The form P hunk: ``_tokenize_texts`` asks the pool hook, then
    tokenizes as before when the hook declines."""
    calls: list[Any] = []

    class Manager:
        def __init__(self, tokenizer: Any, model_config: Any = None) -> None:
            self.tokenizer = tokenizer
            self.model_config = model_config

        async def _tokenize_texts(
            self, texts: Any, is_cross_encoder: bool = False
        ) -> Any:
            pooled = await tokenize_texts_with_pool(self, texts, is_cross_encoder)
            if pooled is not None:
                return pooled
            calls.append((texts, is_cross_encoder))
            out = self.tokenizer(texts, return_token_type_ids=is_cross_encoder)
            return out["input_ids"], out.get("token_type_ids")

    m = Manager(wrapped)
    assert asyncio.run(m._tokenize_texts("hi")) == ([1, 104, 105], None)
    assert asyncio.run(m._tokenize_texts(["ab", "c"])) == ([[1, 97, 98], [1, 99]], None)
    assert calls == []
    assert wrapped.stats()["how"] == {"sched_auto": 3}
    # Declined: cross-encoder pairs, EmbeddingGemma, a tokenizer without a pool.
    asyncio.run(m._tokenize_texts([["q", "d"]], True))
    gemma = type("ModelConfig", (), {"is_embedding_gemma": True})()
    asyncio.run(Manager(wrapped, gemma)._tokenize_texts("hi"))
    asyncio.run(Manager(_StubHf(bos=1))._tokenize_texts("hi"))
    assert [texts for texts, _ in calls] == [[["q", "d"]], "hi", "hi"]
    # Declined as well: an empty input and a row that is not a string.
    assert asyncio.run(tokenize_texts_with_pool(m, "")) is None
    assert asyncio.run(tokenize_texts_with_pool(m, ["ab", 7])) is None
    # The EmbeddingGemma request took the original path, where the wrapper
    # still serves it (through the pool, synchronously).
    assert wrapped.stats()["how"] == {"sched_auto": 4}


# -- artifact tier ------------------------------------------------------------


@pytest.mark.skipif(not _ARTIFACTS, reason="TOKTIER_TEST_ARTIFACTS is not set")
def test_pool_ids_equal_serial_on_a_real_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import toktier
    from toktier import Config, chunked
    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.paths import artifact_cache_dir

    family = "qwen3_8b"
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    entry = manifest.get(family)
    root = Path(_ARTIFACTS or "")
    candidates = [root / name for name in (entry.directory_name, family)]
    sources = [c for c in candidates if (c / "tokenizer.json").is_file()]
    if not sources:
        pytest.skip(f"{family} is not among the local artifacts")
    source = sources[0]
    home = tmp_path / "home"
    config = Config(home=home, offline=True)
    cache = artifact_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, cache / entry.directory_name)
    # The workers resolve the same home through the environment.
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    subject = toktier.load(family, config=config, device="cpu")
    assert chunked.is_admitted(subject)
    sched = Scheduler(
        SchedulerConfig(
            workers=2, chunk_min_chars=20_000, chunk_target=8_000, start_timeout_s=300.0
        ),
        family=family,
        store=str(tmp_path / "store"),
        planner=lambda text, k: chunked.plan(subject, text, k=k).spans,
    )
    sched.start()
    try:
        prose = " ".join(f"word{i} of a long, plain sentence." for i in range(2_000))
        code = "\n".join(f"def f{i}(x):\n    return x * {i} + 1" for i in range(1_000))
        mixed = "Mixed \u4e2d\u6587 text, caf\u00e9 \u0958/k " * 2_000
        for text in (prose, code, mixed):
            serial = list(subject.encode(text, lookup="off").ids)
            free = sched.encode_sync(text, None)
            assert free.error is None and free.ids == serial
            assert free.how == "chunked" and free.chunks >= 2
            keyed = sched.encode_sync(text, "conv", kind="session")
            assert keyed.ids == serial
            more = text + " and one more turn."
            longer = sched.encode_sync(more, "conv", kind="session")
            assert longer.ids == list(subject.encode(more).ids)
            assert longer.wids == keyed.wids
    finally:
        sched.close()
        subject.close()
