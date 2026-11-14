"""The SGLang adapter's lifecycle: what the wrapper says about itself, when
it gives way to the supplied tokenizer, and how its pool is kept and
closed.

None of these tests asks for different ids: each one checks that a call
returns what the supplied tokenizer (or the stub it stands for) returns,
whatever the wrapper did around it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import toktier
from toktier.integrations.sglang import (
    AdapterConfig,
    SGLangToktierTokenizer,
    wrap_tokenizer,
)

from .test_sglang_adapter import _FakeToktier, _StubHf

# -- the wrapper says whether it accelerates -------------------------------


class _ExternalToktier(_FakeToktier):
    """A family toktier knows only as external: the reference engine alone."""

    family = "external_0123456789abcdef"

    class plan:
        backend = "hf"


def test_accelerates_is_true_on_an_accelerated_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="off"))
    assert w.accelerates is True
    assert w.stats()["accelerates"] is True
    assert "accelerates=True" in repr(w)
    # is_fast keeps the Hugging Face meaning
    assert w.is_fast is True
    assert w.encode("ab", add_special_tokens=False) == [97, 98]


def test_accelerates_is_false_in_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="passthrough"))
    assert w.accelerates is False
    assert w.stats()["accelerates"] is False
    assert "accelerates=False" in repr(w)
    assert w.is_fast is True
    # a supplied object toktier declines: passthrough for good, not accelerated
    drifted = _StubHf()
    drifted.backend_tokenizer.add_tokens(["toktierprobeunique"])
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: _FakeToktier()
    )
    w2 = wrap_tokenizer(drifted, AdapterConfig(session_key="off"))
    assert w2.identity["status"] == "mismatch" and w2.accelerates is False


def test_accelerates_is_false_for_an_external_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = _ExternalToktier()
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: external
    )
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="auto"))
    # admitted and answering through toktier, but on the reference engine alone
    assert w.identity["status"] == "equivalent"
    assert w.planned_backend == "hf" and w.toktier is external
    assert w.accelerates is False
    assert w.stats()["accelerates"] is False
    assert "accelerates=False" in repr(w)
    assert w.encode("ab", add_special_tokens=False) == [97, 98]


def test_accelerates_follows_a_later_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    hf = _StubHf()
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off"))
    assert w.accelerates is True
    w.add_tokens(["toktierprobeunique"])
    assert w.encode("ab", add_special_tokens=False) == hf.encode(
        "ab", add_special_tokens=False
    )
    assert w.stats()["accelerates"] is False
    assert not w.accelerates


def test_the_start_up_record_carries_accelerates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import json

    log = tmp_path / "timing.jsonl"
    external = _ExternalToktier()
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: external
    )
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="off", timing_log=str(log)))
    w.flush()
    first = json.loads(log.read_text().splitlines()[0])
    assert first["kind"] == "init" and first["accelerates"] is False
    w.close()


# -- a tokenizer toktier cannot resolve: passthrough, not a stop ------------


def _offline_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """A fresh toktier home, offline: nothing is fetched, nothing is cached."""
    home = tmp_path / "toktier-home"
    home.mkdir()
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _saved_fast_tokenizer(directory: Any) -> Any:
    """A real fast tokenizer saved to and loaded from a local directory, as
    SGLang loads one for a local ``--tokenizer-path``."""
    from transformers import PreTrainedTokenizerFast

    from .test_sglang_adapter import _tiny_tokenizer

    PreTrainedTokenizerFast(tokenizer_object=_tiny_tokenizer()).save_pretrained(
        directory
    )
    return PreTrainedTokenizerFast.from_pretrained(str(directory))


def _assert_unresolved_passthrough(w: Any, hf: Any) -> None:
    assert isinstance(w, SGLangToktierTokenizer)
    assert w.toktier is None and w.accelerates is False
    assert w.identity["status"] == "unresolved"
    assert w.identity["reason"]
    stats = w.stats()
    assert stats["mode"] == "passthrough"
    assert stats["identity"]["status"] == "unresolved"
    for text in ("ab c", "abc ab", ""):
        assert w.encode(text, add_special_tokens=False) == hf.encode(
            text, add_special_tokens=False
        )
        assert w(text)["input_ids"] == hf(text)["input_ids"]


def _one_warning(caplog: pytest.LogCaptureFixture) -> str:
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "toktier.integrations.sglang" and r.levelname == "WARNING"
    ]
    assert len(lines) == 1, lines
    line: str = lines[0]
    assert "TOKTIER_SGLANG_FAMILY" in line and "TOKTIER_SGLANG_REPO" in line
    assert "toktier artifacts fetch" in line
    return line


def test_a_local_directory_is_served_in_passthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # The review's S4 and S5: --tokenizer-path names a local directory.
    _offline_home(monkeypatch, tmp_path)
    hf = _saved_fast_tokenizer(tmp_path / "model-dir")
    assert hf.name_or_path == str(tmp_path / "model-dir")
    caplog.set_level("WARNING", logger="toktier.integrations.sglang")
    w = wrap_tokenizer(hf, AdapterConfig.from_env({}))
    _assert_unresolved_passthrough(w, hf)
    assert "ArtifactNotFound" in w.identity["reason"]
    assert str(tmp_path / "model-dir") in _one_warning(caplog)


def test_a_certified_hub_id_with_an_empty_offline_cache_is_served_in_passthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # The review's S9: a certified repository, an offline host, toktier's own
    # artifact cache empty, and nothing named by the operator.
    _offline_home(monkeypatch, tmp_path)
    hf = _saved_fast_tokenizer(tmp_path / "model-dir")
    hf.name_or_path = "Qwen/Qwen3-8B"
    caplog.set_level("WARNING", logger="toktier.integrations.sglang")
    w = wrap_tokenizer(hf, AdapterConfig())
    _assert_unresolved_passthrough(w, hf)
    _one_warning(caplog)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("FAMILY", "qwen3_8b"),  # a shipped family whose files are not staged
        ("FAMILY", "qwen3_8bb"),  # a mistyped family
        ("REPO", "Qwen/Qwen3-8BB"),  # a mistyped repository
        ("REPO", "Qwen/Qwen3-8B"),  # a certified repository, files not staged
    ],
)
def test_a_named_family_or_repository_that_cannot_be_resolved_raises(
    setting: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """What the operator named and toktier cannot resolve stops the wrap,
    and the message names the setting; nothing named stays passthrough
    (the two tests above)."""
    from toktier.errors import ArtifactNotFound

    _offline_home(monkeypatch, tmp_path)
    hf = _saved_fast_tokenizer(tmp_path / "model-dir")
    caplog.set_level("WARNING", logger="toktier.integrations.sglang")
    variable = "TOKTIER_SGLANG_" + setting
    with pytest.raises(ArtifactNotFound, match=variable) as raised:
        wrap_tokenizer(hf, AdapterConfig.from_env({variable: value}))
    assert repr(value) in str(raised.value)
    assert raised.value.details["setting"] == variable
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_tokenizer_without_a_name_is_served_in_passthrough(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The review's S16: a tokenizer object that carries no name_or_path
    # (SGLang's MistralCommonTokenizer for a repository without
    # tokenizer.json) and no family or repository configured.
    hf = _StubHf()
    hf.name_or_path = None  # type: ignore[assignment]
    caplog.set_level("WARNING", logger="toktier.integrations.sglang")
    w = wrap_tokenizer(hf, AdapterConfig())
    _assert_unresolved_passthrough(w, hf)
    assert "name_or_path" in w.identity["reason"]
    _one_warning(caplog)


class _StrictCall(_StubHf):
    """A tokenizer whose call refuses keywords outside its own signature, as
    SGLang's MistralCommonBackend does (``text_pair`` and
    ``return_token_type_ids`` among them, even when ``None``)."""

    def __call__(
        self, text: Any, add_special_tokens: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs)} are not supported")
        return super().__call__(text, add_special_tokens=add_special_tokens)


def test_passthrough_forwards_only_the_keywords_the_caller_passed() -> None:
    hf = _StrictCall(bos=1)
    hf.name_or_path = None  # type: ignore[assignment]
    w = wrap_tokenizer(hf, AdapterConfig())
    assert w.identity["status"] == "unresolved"
    assert w("ab")["input_ids"] == [1, 97, 98]
    assert w(["ab", "c"], add_special_tokens=False)["input_ids"] == [[97, 98], [99]]
    assert w.encode("ab") == [1, 97, 98]
    # a keyword the caller does pass still reaches the tokenizer
    with pytest.raises(ValueError, match="return_token_type_ids"):
        w("ab", return_token_type_ids=True)


def test_an_invalid_setting_still_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from toktier.errors import ArtifactNotFound

    _offline_home(monkeypatch, tmp_path)
    hf = _saved_fast_tokenizer(tmp_path / "model-dir")
    monkeypatch.setenv("TOKTIER_SGLANG_SESSION_KEY", "bogus")
    with pytest.raises(ValueError, match="session_key"):
        wrap_tokenizer(hf)
    monkeypatch.delenv("TOKTIER_SGLANG_SESSION_KEY")
    monkeypatch.setenv("TOKTIER_SGLANG_REQUIRE_ARTIFACT", "maybe")
    with pytest.raises(ValueError, match="TOKTIER_SGLANG_REQUIRE_ARTIFACT"):
        wrap_tokenizer(hf)
    monkeypatch.delenv("TOKTIER_SGLANG_REQUIRE_ARTIFACT")
    # a family name toktier does not know is a setting, not a resolution
    with pytest.raises(ArtifactNotFound, match="unknown tokenizer family"):
        wrap_tokenizer(hf, AdapterConfig(family="no_such_family"))


def test_require_artifact_restores_the_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from toktier.errors import ArtifactNotFound

    _offline_home(monkeypatch, tmp_path)
    hf = _saved_fast_tokenizer(tmp_path / "model-dir")
    assert AdapterConfig.from_env({}).require_artifact is False
    strict = AdapterConfig.from_env({"TOKTIER_SGLANG_REQUIRE_ARTIFACT": "1"})
    assert strict.require_artifact is True
    with pytest.raises(ArtifactNotFound):
        wrap_tokenizer(hf, strict)
    nameless = _StubHf()
    nameless.name_or_path = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="name_or_path"):
        wrap_tokenizer(nameless, strict)


def test_other_load_errors_still_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    from toktier.errors import ArtifactHashMismatch

    def corrupt(self: Any) -> Any:
        raise ArtifactHashMismatch("cached artifact failed verification")

    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", corrupt)
    with pytest.raises(ArtifactHashMismatch):
        wrap_tokenizer(_StubHf(), AdapterConfig())


# -- the pool's processes and queues: closed with the pool, gone with the owner --

_SRC = str(Path(toktier.__file__).resolve().parents[1])
_HERE = str(Path(__file__).resolve().parent)


def _run_owner(script: str, *args: str, timeout: float = 300.0) -> Any:
    """Run ``script`` in a fresh interpreter that imports this source tree
    and the scheduler stub, with ResourceWarning shown and made an error."""
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_SRC, _HERE])
    return subprocess.run(
        [
            sys.executable,
            "-X",
            "dev",
            "-W",
            "error::ResourceWarning",
            "-c",
            script,
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_CYCLES = r"""
import os, sys, threading
from toktier.integrations.sglang_sched import Scheduler, SchedulerConfig

here = sys.argv[1]
cycles = int(sys.argv[2])

def config(workers):
    return SchedulerConfig(workers=workers, loader="_sched_stub:load",
                           loader_path=here, start_timeout_s=120.0)

def cycle(workers):
    sched = Scheduler(config(workers))
    sched.start()
    assert sched.encode_sync("ab", None, timeout=60).ids == [97, 98]
    sched.close()

def fds():
    return len(os.listdir("/proc/self/fd"))

cycle(2)  # the first start also starts multiprocessing's resource tracker
before_fds, before_threads = fds(), threading.active_count()
for _ in range(cycles):
    cycle(1)
print("FDS", before_fds, fds(), "THREADS", before_threads, threading.active_count())
if sys.argv[3] == "os_exit":
    sys.stdout.flush()
    os._exit(0)
"""


@pytest.mark.parametrize("ending", ["return", "os_exit"])
def test_closing_a_pool_leaves_no_pipe_thread_or_resource_warning(
    ending: str,
) -> None:
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("no /proc/self/fd on this platform")
    cycles = "50" if ending == "return" else "3"
    done = _run_owner(_CYCLES, _HERE, cycles, ending)
    assert done.returncode == 0, done.stderr[-3000:]
    line = next(x for x in done.stdout.splitlines() if x.startswith("FDS"))
    _, fds_before, fds_after, _, threads_before, threads_after = line.split()
    assert fds_after == fds_before, line
    assert threads_after == threads_before, line
    assert "ResourceWarning" not in done.stderr, done.stderr[-3000:]
    # multiprocessing's resource tracker reports semaphores a process left
    # behind when it exits without running its finalizers (os._exit)
    assert "leaked" not in done.stderr, done.stderr[-3000:]
    assert "resource_tracker" not in done.stderr, done.stderr[-3000:]


_SATURATED = r"""
import os, sys, threading, time
from toktier.integrations.sglang_sched import Scheduler, SchedulerConfig

here = sys.argv[1]
pad = "x" * (2 << 20)  # 2 MiB per task: far more than a pipe holds (64 KiB)
for delay in [float(x) for x in sys.argv[2].split(",")]:
    sched = Scheduler(SchedulerConfig(workers=2, loader="_sched_stub:load",
                                      loader_path=here, start_timeout_s=120.0))
    sched.start()
    seen = []
    for i in range(14):  # two in service, twelve queued in the parent
        text = f"slow {i} " + pad
        sched.submit(text, None, lambda r, text=text: seen.append((r, text)))
    time.sleep(delay)  # around the moment the first answers come back
    t0 = time.perf_counter()
    closer = threading.Thread(target=sched.close, daemon=True)
    closer.start()
    closer.join(60.0)
    took = time.perf_counter() - t0
    if closer.is_alive():
        print("HUNG", delay, flush=True)
        os._exit(3)
    answered = sum(
        1 for r, t in seen if r.error is None and r.ids == [ord(c) for c in t])
    closed = sum(1 for r, _ in seen if r.error == "closed")
    print("CLOSE", delay, round(took, 3), len(seen), answered, closed, flush=True)
sys.stdout.flush()
os._exit(0)
"""


def test_close_of_a_saturated_pool_with_large_tasks_returns() -> None:
    """``close()`` of a pool whose queued tasks are larger than a pipe:
    nothing is dispatched after the stop message, no feeder thread is
    joined without a bound, and every request is either answered or
    failed with ``"closed"``; over a range of delays around the first
    answers."""
    delays = "0.2,0.25,0.3,0.35,0.4,0.5,0.7"
    done = _run_owner(_SATURATED, _HERE, delays, timeout=600.0)
    assert done.returncode == 0, (done.stdout[-2000:], done.stderr[-3000:])
    rows = [x.split() for x in done.stdout.splitlines() if x.startswith("CLOSE")]
    assert len(rows) == len(delays.split(","))
    for _, _, took, seen, answered, closed in rows:
        assert float(took) < 20.0, rows
        assert int(seen) == 14 and int(answered) + int(closed) == 14, rows


_LATCH_UNDER_LOAD = r"""
import json, os, sys, threading, time
from toktier.integrations.sglang import (
    AdapterConfig, SGLangToktierTokenizer, wrap_tokenizer)
from toktier.integrations.sglang_sched import SchedulerConfig
from test_sglang_adapter import _FakeToktier, _StubHf

here, delay = sys.argv[1], float(sys.argv[2])
SGLangToktierTokenizer._load_toktier = lambda self: _FakeToktier()
hf = _StubHf()
w = wrap_tokenizer(hf, AdapterConfig(
    session_key="off",
    scheduler=SchedulerConfig(workers=2, loader="_sched_stub:load", loader_path=here,
                              start_timeout_s=120.0, chunk_min_chars=1 << 30)))
assert w.scheduler is not None
pad = "x" * (2 << 20)
outcomes = {}

def request(i):
    text = f"slow {i} " + pad
    try:
        ids = w.encode(text, add_special_tokens=False)
        same = ids == hf.encode(text, add_special_tokens=False)
        outcomes[i] = "answered" if same else "differs"
    except Exception as ex:
        outcomes[i] = "raised " + type(ex).__name__

threads = [threading.Thread(target=request, args=(i,), daemon=True) for i in range(16)]
for t in threads:
    t.start()
time.sleep(delay)
# the supplied tokenizer changes while the pool is saturated: the next call
# latches passthrough, which closes the pool on that call's thread
hf.backend_tokenizer.add_tokens(["toktierchangedunderload"])
box = {}

def notice():
    t0 = time.perf_counter()
    box["ids_equal"] = w.encode("ab", add_special_tokens=False) == hf.encode(
        "ab", add_special_tokens=False)
    box["seconds"] = round(time.perf_counter() - t0, 3)

noticer = threading.Thread(target=notice, daemon=True)
noticer.start()
noticer.join(60.0)
deadline = time.monotonic() + 30.0
for t in threads:
    t.join(max(0.0, deadline - time.monotonic()))
print("RESULT " + json.dumps({
    "returned": not noticer.is_alive(), **box, "mode": w.stats()["mode"],
    "pending": sum(t.is_alive() for t in threads),
    "outcomes": sorted(outcomes.values())}), flush=True)
os._exit(0)
"""


def _latch_under_load(delay: str) -> dict[str, Any]:
    import json

    done = _run_owner(_LATCH_UNDER_LOAD, _HERE, delay, timeout=300.0)
    assert done.returncode == 0, (done.stdout[-2000:], done.stderr[-3000:])
    line = next(x for x in done.stdout.splitlines() if x.startswith("RESULT "))
    result: dict[str, Any] = json.loads(line[len("RESULT ") :])
    return result


@pytest.mark.parametrize("delay", ["0.35", "0.8"])
def test_a_latch_while_the_pool_is_saturated_returns(delay: str) -> None:
    """The passthrough latch closes the pool under the wrapper's encode
    lock; with large tasks queued that close returns, so the call that
    noticed the change, and every request that was in flight, finish."""
    result = _latch_under_load(delay)
    assert result["returned"] and result["ids_equal"], result
    assert result["seconds"] < 20.0, result
    assert result["mode"] == "passthrough", result
    assert result["pending"] == 0 and len(result["outcomes"]) == 16, result


@pytest.mark.parametrize("delay", ["0.35", "0.8"])
def test_requests_in_flight_at_a_latch_are_answered_by_the_supplied_tokenizer(
    delay: str,
) -> None:
    """The latch fails the pool's pending requests while it holds the encode
    lock and then drops the toktier tokenizer; each of those requests, and
    any other admitted just before the latch, is answered by the supplied
    tokenizer rather than meeting the dropped one."""
    result = _latch_under_load(delay)
    assert result["returned"] and result["pending"] == 0, result
    assert result["outcomes"] == ["answered"] * 16, result


_WRAP_RAISES = r"""
import json, os, sys, threading, time
from toktier.integrations.sglang import (
    AdapterConfig, SGLangToktierTokenizer, wrap_tokenizer)
from toktier.integrations.sglang_sched import Scheduler, SchedulerConfig
from test_sglang_adapter import _FakeToktier, _StubHf

here, case, tmp = sys.argv[1], sys.argv[2], sys.argv[3]

def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return False

def children():
    found = set()
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/children") as fh:
                found |= {int(p) for p in fh.read().split()}
        except OSError:
            pass
    return sorted(p for p in found if alive(p))

def fds():
    return len(os.listdir("/proc/self/fd"))

pool = SchedulerConfig(workers=2, loader="_sched_stub:load", loader_path=here,
                       start_timeout_s=120.0)
warm = Scheduler(pool)  # the first pool starts multiprocessing's resource tracker
warm.start()
warm.close()
before = {"fds": fds(), "children": children(), "threads": threading.active_count()}
loaded = []

def load(self):
    loaded.append(_FakeToktier())
    return loaded[-1]

SGLangToktierTokenizer._load_toktier = load
original_start = threading.Thread.start
if case == "timing_log":
    blocker = os.path.join(tmp, "a-regular-file")
    open(blocker, "w").close()
    cfg = AdapterConfig(session_key="off", scheduler=pool,
                        timing_log=os.path.join(blocker, "timing.jsonl"))
else:  # a failure after the pool has started: the shadow thread cannot start
    def start(thread):
        if thread.name == "toktier-sglang-shadow":
            raise RuntimeError("can't start new thread (injected)")
        return original_start(thread)
    threading.Thread.start = start
    cfg = AdapterConfig(session_key="off", scheduler=pool, shadow="async")
try:
    wrap_tokenizer(_StubHf(), cfg)
    outcome = "returned"
except BaseException as ex:
    outcome = type(ex).__name__
threading.Thread.start = original_start
deadline = time.monotonic() + 5.0
while time.monotonic() < deadline and children() != before["children"]:
    time.sleep(0.1)
after = {"fds": fds(), "children": children(), "threads": threading.active_count()}
print("RESULT " + json.dumps({"outcome": outcome, "before": before, "after": after,
                              "loaded": len(loaded),
                              "closed": [t.closed for t in loaded]}), flush=True)
os._exit(0)
"""


@pytest.mark.parametrize("case", ["timing_log", "after_start"])
def test_a_wrap_that_raises_leaves_nothing_running(case: str, tmp_path: Any) -> None:
    """A timing log that cannot be opened stops the wrap before anything is
    started; a failure after the pool has started closes what was started.
    Either way no child process of the owner is alive afterwards and its
    descriptor and thread counts are what they were."""
    import json

    done = _run_owner(_WRAP_RAISES, _HERE, case, str(tmp_path), timeout=300.0)
    assert done.returncode == 0, (done.stdout[-2000:], done.stderr[-3000:])
    line = next(x for x in done.stdout.splitlines() if x.startswith("RESULT "))
    result = json.loads(line[len("RESULT ") :])
    expected = (
        ("FileExistsError", "NotADirectoryError")  # OSError of the parent path
        if case == "timing_log"
        else ("RuntimeError",)
    )
    assert result["outcome"] in expected, result
    assert result["after"] == result["before"], result
    if case == "timing_log":
        assert result["loaded"] == 0, result  # nothing was loaded or started
    else:
        assert result["loaded"] == 1 and result["closed"] == [True], result


def test_a_store_directory_that_cannot_be_created_stops_the_wrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The store directory is created by the wrap, so a path that cannot be
    created stops it (before anything is loaded) instead of failing the
    first request; one that can is there when the wrap returns."""
    loaded: list[Any] = []

    def load(self: Any) -> Any:
        loaded.append(_FakeToktier())
        return loaded[-1]

    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", load)
    blocker = tmp_path / "a-regular-file"
    blocker.write_text("")
    with pytest.raises(OSError):
        wrap_tokenizer(
            _StubHf(), AdapterConfig(session_key="off", store=str(blocker / "store"))
        )
    assert loaded == []
    w = wrap_tokenizer(
        _StubHf(), AdapterConfig(session_key="off", store=str(tmp_path / "store"))
    )
    try:
        assert (tmp_path / "store" / f"pid{os.getpid()}").is_dir()
        assert w.encode("ab", add_special_tokens=False) == [97, 98]
    finally:
        w.close()


_OWNER = r"""
import os, sys, threading, time
from toktier.integrations.sglang_sched import Scheduler, SchedulerConfig

here, ending, starter = sys.argv[1], sys.argv[2], sys.argv[3]
box = []

def start():
    sched = Scheduler(SchedulerConfig(workers=2, loader="_sched_stub:load",
                                      loader_path=here, start_timeout_s=120.0))
    sched.start()
    box.append(sched)

if starter == "thread":
    # the thread that spawned the workers ends while the owner lives on
    t = threading.Thread(target=start)
    t.start()
    t.join()
    time.sleep(1.5)
else:
    start()
sched = box[0]
assert sched.encode_sync("ab", None, timeout=60).ids == [97, 98]
assert sched.encode_sync("cd", None, timeout=60).ids == [99, 100]
assert not sched.stats()["dead"], sched.stats()["dead"]
print("PIDS", os.getpid(), *sched.stats()["worker_pids"].values(), flush=True)
if ending in ("os_exit", "return"):
    sys.stdin.readline()  # the test has seen the workers alive
if ending == "os_exit":
    os._exit(0)
if ending == "return":
    sys.exit(0)
time.sleep(600)
"""


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state not in ("Z", "X")


@pytest.mark.parametrize(
    ("ending", "starter"),
    [
        ("return", "main"),
        ("sigkill", "main"),
        ("sigterm", "main"),
        ("os_exit", "main"),
        ("sigkill", "thread"),
    ],
)
def test_pool_workers_leave_when_their_owner_is_gone(ending: str, starter: str) -> None:
    import signal
    import subprocess
    import sys
    import time

    if not os.path.isdir("/proc/self"):
        pytest.skip("no /proc on this platform")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_SRC, _HERE])
    owner = subprocess.Popen(
        [sys.executable, "-c", _OWNER, _HERE, ending, starter],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    workers: list[int] = []
    try:
        assert owner.stdout is not None
        line = owner.stdout.readline()
        assert line.startswith("PIDS"), (line, owner.stderr and owner.stderr.read())
        pids = [int(x) for x in line.split()[1:]]
        assert pids[0] == owner.pid
        workers = pids[1:]
        assert len(workers) == 2 and all(_alive(w) for w in workers)
        assert owner.stdin is not None
        if ending in ("os_exit", "return"):
            owner.stdin.write("go\n")
            owner.stdin.flush()
        if ending == "sigkill":
            owner.send_signal(signal.SIGKILL)
        elif ending == "sigterm":
            owner.send_signal(signal.SIGTERM)
        owner.wait(timeout=60)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(_alive(w) for w in workers):
            time.sleep(0.1)
        left = [w for w in workers if _alive(w)]
        assert left == [], f"workers alive 5 s after the owner ended: {left}"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
        for w in workers:
            if _alive(w):
                os.kill(w, signal.SIGKILL)


# -- tests the review asked for beside the fixes (P4, P5, P6) -----------------


def _stub_pool_config(**kw: Any) -> Any:
    from toktier.integrations.sglang_sched import SchedulerConfig

    return SchedulerConfig(
        workers=2,
        loader="_sched_stub:load",
        loader_path=_HERE,
        start_timeout_s=120.0,
        **kw,
    )


def test_the_pool_helper_takes_a_key_and_the_special_tokens_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behaviour only ``encode_texts_with_pool`` has (the module is imported
    here, not at the top, so without the helper only this test fails): an
    explicit key takes the session route of its home worker, and
    ``add_special_tokens=False`` leaves the template off."""
    import asyncio

    from toktier.integrations import sglang as sglang_module

    helper = sglang_module.encode_texts_with_pool
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: _FakeToktier()
    )
    w = wrap_tokenizer(
        _StubHf(bos=1),
        AdapterConfig(session_key="contextvar", scheduler=_stub_pool_config()),
    )
    try:
        assert w.scheduler is not None
        assert asyncio.run(helper(w, "ab", key="k1", add_special_tokens=False)) == [
            97,
            98,
        ]
        assert asyncio.run(helper(w, ["ab", "c"], key="k2")) == [[1, 97, 98], [1, 99]]
        assert w.stats()["how"] == {"sched_session": 3}
        assert asyncio.run(helper(w, "ab")) == [1, 97, 98]  # no key: content lookup
        assert w.stats()["how"] == {"sched_session": 3, "sched_auto": 1}
    finally:
        w.close()


def test_close_twice_and_from_two_threads_at_once() -> None:
    import threading

    from toktier.integrations.sglang_sched import Scheduler

    sched = Scheduler(_stub_pool_config())
    sched.start()
    seen: list[Any] = []
    sched.submit("slow one", None, seen.append)
    closers = [threading.Thread(target=sched.close) for _ in range(2)]
    for t in closers:
        t.start()
    for t in closers:
        t.join(30.0)
    assert not any(t.is_alive() for t in closers)
    sched.close()  # a third time: returns at once
    assert not sched.active and [r.error for r in seen] == ["closed"]
    with pytest.raises(RuntimeError, match="not running"):
        sched.submit("after", None, seen.append)


@pytest.mark.parametrize("face", ["sync", "async"])
def test_requests_in_flight_when_the_pool_closes_are_answered_in_process(
    face: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import threading
    import time

    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    w = wrap_tokenizer(
        _StubHf(), AdapterConfig(session_key="off", scheduler=_stub_pool_config())
    )
    texts = [f"slow {i}" for i in range(6)]
    got: dict[str, Any] = {}

    def sync_requests() -> None:
        threads = [
            threading.Thread(
                target=lambda t=t: got.__setitem__(
                    t, w.encode(t, add_special_tokens=False)
                )
            )
            for t in texts
        ]
        for t in threads:
            t.start()
        time.sleep(0.2)
        assert w._sched is not None
        w._sched.close()
        for t in threads:
            t.join(30.0)

    async def async_requests() -> None:
        tasks = [
            asyncio.ensure_future(w.encode_async(t, add_special_tokens=False))
            for t in texts
        ]
        await asyncio.sleep(0.2)
        assert w._sched is not None
        w._sched.close()
        for t, ids in zip(texts, await asyncio.gather(*tasks), strict=True):
            got[t] = ids

    try:
        if face == "sync":
            sync_requests()
        else:
            asyncio.run(async_requests())
        assert got == {t: [ord(c) for c in t] for t in texts}
        how = w.stats()["how"]
        assert how.get("sched_free", 0) + how.get("off", 0) == 6, how
        assert how.get("off", 0) >= 1 and w.stats()["sched_fallback"] == how["off"]
    finally:
        w.close()


def test_idle_workers_of_a_living_owner_stay_through_several_polls() -> None:
    """The parent check of an idle worker (once a second) must not take a
    living parent for gone."""
    import time

    from toktier.integrations.sglang_sched import Scheduler

    sched = Scheduler(_stub_pool_config())
    sched.start()
    try:
        pids = dict(sched.stats()["worker_pids"])
        time.sleep(3.5)
        for pid in pids.values():
            assert _alive(pid)
        assert sched.encode_sync("ab", None, timeout=30).ids == [97, 98]
        assert sched.encode_sync("cd", None, timeout=30).ids == [99, 100]
        assert sched.stats()["dead"] == {} and sched.stats()["worker_pids"] == pids
    finally:
        sched.close()
