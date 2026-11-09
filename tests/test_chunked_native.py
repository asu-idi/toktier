"""The native chunk encoder plans and encodes exactly as ``toktier.chunked`` does.

Artifact tier only (``TOKTIER_TEST_ARTIFACTS``, see ``test_chunked.py``): every
family found there is loaded once, and for each one the native runtime's
``chunked_plan`` is compared position for position with the Python
planner over the corpora, the plan-bound variants and the generated
normalization corpus; the native ``chunked.encode`` ids are judged
against a from-scratch oracle encode; and the ``Tokenizer.encode(...,
parallel=...)`` wrapper is exercised. ``TOKTIER_TEST_CHUNKED_ADVERSARIAL``
scales the generated corpus per family (default 400).

Until the source identities are rebound, the native extension's fast_cpu
source identity differs from the recorded one, so a family whose
certified plan lacks the corrected CPU route is reloaded under the
experimental policy (the waiver is recorded in
``explain()["experimental_waivers"]``); the readings here are about
equivalence with the Python planner, not about certification.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "4")

import toktier
from toktier import Config, ReasonCode, Tokenizer, chunked
from toktier.artifacts.manifest import ArtifactManifest
from toktier.artifacts.tables import ARTIFACT_MANIFEST
from toktier.paths import artifact_cache_dir

from .test_chunked import (
    _ADVERSARIAL,
    SIZES,
    TABLE_FAMILIES,
    TABLE_ROWS,
    _adversarial,
    _artifact_source,
    _generated_corpora,
    _reference,
    corpora,
    load_isolated,
)
from .test_chunked import CANDIDATE_FAMILIES as CERTIFIED_FAMILIES

#: Every family with a table row, certified or not: these are equivalence
#: readings of the two planners, taken with the evidence gate lifted.
CANDIDATE_FAMILIES = TABLE_FAMILIES


def available_families() -> list[str]:
    return [name for name in CANDIDATE_FAMILIES if _artifact_source(name)]


def _load(family: str, config: Config) -> Tokenizer:
    handle = toktier.load(family, config=config, device="cpu")
    chain = handle.explain()["fallback_chain"]
    if isinstance(chain, (list, tuple)) and "fast_cpu" in chain:
        return handle
    handle.close()
    return toktier.load(family, config=config, device="cpu", policy="experimental")


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Tokenizer]]:
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    handles: dict[str, Tokenizer] = {}
    for family in available_families():
        source = _artifact_source(family)
        assert source is not None
        home = tmp_path_factory.mktemp(f"chunked-native-{family}")
        config = Config(home=home, offline=True)
        cache = artifact_cache_dir(config)
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, cache / manifest.get(family).directory_name)
        handle = _load(family, config)
        if not chunked.native_available(handle):
            handle.close()
            pytest.skip(f"{family}: no native chunk encoder on this route")
        # These are equivalence readings of the two planners, not
        # certification: the evidence gate is lifted so the native path
        # runs for every admitted family, whatever the table records.
        chunked._planner_for(handle, evidence_gate=False)
        handles[family] = handle
    yield handles
    for handle in handles.values():
        handle.close()


def _family(loaded: dict[str, Tokenizer], family: str) -> Tokenizer:
    if family not in loaded:
        pytest.skip(f"{family}: artifact not available under TOKTIER_TEST_ARTIFACTS")
    return loaded[family]


def _plans_agree(
    tokenizer: Tokenizer, text: str, **options: object
) -> chunked.ChunkPlan:
    python = chunked.plan(tokenizer, text, **options)  # type: ignore[arg-type]
    native = chunked.plan(tokenizer, text, native=True, **options)  # type: ignore[arg-type]
    assert native == python, (options, native, python)
    return native


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_native_admission_matches_the_python_planner(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    tokenizer = _family(loaded, family)
    merged = chunked.admission(tokenizer)
    native = merged["native"]
    python = chunked.stats(tokenizer)
    assert isinstance(native, dict)
    for key in ("admitted", "refusal", "normalizer", "literals"):
        assert native[key] == python[key], (family, key, native, python)
    assert native["admitted"] is True
    assert merged["admitted"] is True and merged["native_available"] is True
    assert merged["admission_gate"] == "lifted_for_evidence_collection"
    assert merged["native_by_default"] is True
    assert merged["native_evidence_label"] == chunked._native_evidence_label(
        tuple(TABLE_ROWS[family]["evidence"]["native"])
    )


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_native_plan_equals_the_python_plan_on_corpora(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    tokenizer = _family(loaded, family)
    for size in SIZES:
        for text in corpora(size).values():
            for k in (None, 5, 8, 32):
                _plans_agree(tokenizer, text, k=k)
            _plans_agree(tokenizer, text, k=16, min_chunk=100)
            _plans_agree(tokenizer, text, k=16, min_chunk=size // 8)
            _plans_agree(tokenizer, text, k=64, scan_budget=1)
            _plans_agree(tokenizer, text, k=64, scan_budget=40)
            _plans_agree(tokenizer, text, k=4, scan_budget=100_000)
    generated = _generated_corpora(300_000)["transcript(generated)"]
    for k in (2, 5, 16, 64, 500, 5000):
        _plans_agree(tokenizer, generated, k=k)
    # The bounded-plan cases of the Python tests, on both planners.
    text = "a" * 110 + " " + "b" * 189
    assert _plans_agree(tokenizer, text, k=2, scan_budget=1).spans == ((0, 300),)
    assert _plans_agree(tokenizer, text, k=2, scan_budget=40).cuts == (110,)
    text = "ab 12 " * 50
    tight = _plans_agree(tokenizer, text, k=3, min_chunk=150)
    assert len(tight.spans) == 1 and tight.cut_misses == 2
    assert len(_plans_agree(tokenizer, text, k=3, min_chunk=100).spans) == 2


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_native_plan_and_ids_agree_on_the_adversarial_corpus(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    tokenizer = _family(loaded, family)
    reference = _reference(tokenizer)
    face = json.loads(tokenizer._backend.materialized_tokenizer_json())
    face_literals = [str(row["content"]) for row in face["added_tokens"]]
    rng = random.Random(sum(ord(c) for c in family) * 7919 + 0x5EED + 1)
    n_chunked = 0
    for index in range(_ADVERSARIAL):
        text = _adversarial(rng, rng.randint(300, 1500), face_literals)
        k = rng.choice((2, 3, 4, 6, 8))
        plan = _plans_agree(tokenizer, text, k=k)
        native = chunked.encode(tokenizer, text, k=k, native=True)
        python = chunked.encode(tokenizer, text, k=k)
        assert list(native.ids) == reference(text), (family, index, k)
        assert list(python.ids) == list(native.ids)
        for key in (
            "chunks",
            "cuts",
            "cut_misses",
            "literal_rejects",
            "norm_rejects",
            "serial_reason",
        ):
            assert getattr(native, key) == getattr(python, key), (family, index, key)
        assert native.chunks == len(plan.spans)
        n_chunked += native.chunks >= 2
    assert n_chunked >= _ADVERSARIAL // 4, (family, n_chunked, _ADVERSARIAL)


def _cases() -> list[object]:
    return [
        pytest.param(
            family,
            corpus,
            size,
            marks=[pytest.mark.slow] if size >= 1_000_000 else [],
        )
        for family in available_families()
        for size in SIZES
        for corpus in corpora(size)
    ]


@pytest.mark.parametrize(("family", "corpus", "size"), _cases())
def test_native_chunked_ids_equal_reference(
    loaded: dict[str, Tokenizer], family: str, corpus: str, size: int
) -> None:
    tokenizer = _family(loaded, family)
    text = corpora(size)[corpus]
    expected = _reference(tokenizer)(text)
    for k in (None, 8, 32):
        encoded = chunked.encode(tokenizer, text, k=k, native=True)
        assert list(encoded.ids) == expected, (family, corpus, size, k)
        assert encoded.serial_reason is None
        assert encoded.native_facts is not None
        assert encoded.chunks >= 2 if k is not None else True
        sources = encoded.native_facts["sources"]
        assert isinstance(sources, dict)
        assert sum(sources.values()) == encoded.chunks
    wrapped = tokenizer.encode(text, parallel=8)
    assert isinstance(wrapped, chunked.ChunkedEncoding)
    assert list(wrapped.ids) == expected and wrapped.chunks >= 2
    auto = tokenizer.encode(text, parallel="auto", lookup="off")
    assert isinstance(auto, chunked.ChunkedEncoding)
    assert list(auto.ids) == expected
    assert auto.k_requested == chunked.default_chunks(size)


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_native_serial_reasons_and_edges(
    loaded: dict[str, Tokenizer], family: str
) -> None:
    tokenizer = _family(loaded, family)
    reference = _reference(tokenizer)
    review = "a" * 127 + "\u0958" + "/k" + " " * 126
    native = chunked.encode(tokenizer, review, k=2, native=True)
    python = chunked.encode(tokenizer, review, k=2)
    assert list(native.ids) == reference(review)
    assert (native.chunks, native.cuts, native.serial_reason, native.norm_rejects) == (
        python.chunks,
        python.cuts,
        python.serial_reason,
        python.norm_rejects,
    )
    if chunked.stats(tokenizer)["normalizer"] == "nfc":
        assert native.serial_reason == ReasonCode.R_CHUNK_NO_CUT.value
    cjk = "\u4e2d" * 40_000
    encoded = chunked.encode(tokenizer, cjk, k=4, native=True)
    assert encoded.serial_reason == ReasonCode.R_CHUNK_NO_CUT.value
    assert encoded.chunks == 1 and encoded.cut_misses == 3
    assert list(encoded.ids) == reference(cjk)
    for text in ("", " ", "x", "\n", "a" * 300, "ab " * 100, " " * 300_000):
        encoded = chunked.encode(tokenizer, text, k=4, native=True)
        python = chunked.encode(tokenizer, text, k=4)
        assert list(encoded.ids) == reference(text), repr(text[:20])
        assert (encoded.chunks, encoded.cuts, encoded.serial_reason) == (
            python.chunks,
            python.cuts,
            python.serial_reason,
        ), repr(text[:20])
        if len(text) < 256:
            assert encoded.chunks == 1 and encoded.serial_reason is None
    with pytest.raises(UnicodeEncodeError):
        chunked.encode(tokenizer, "ab\ud800cd" * 200, k=2, native=True)
    with pytest.raises(TypeError):
        chunked.encode(tokenizer, b"bytes", k=2, native=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stateless"):
        tokenizer.encode("x" * 1000, parallel=2, session="s")
    with pytest.raises(ValueError, match="stateless"):
        tokenizer.encode("x" * 1000, parallel=2, lookup="auto")
    with pytest.raises(ValueError, match="positive"):
        tokenizer.encode("x" * 1000, parallel=0)
    with pytest.raises(ValueError, match="core stream"):
        tokenizer.encode("x" * 1000, parallel=2, add_special_tokens=True)
    plain = tokenizer.encode("x" * 1000)
    assert type(plain) is toktier.Encoding


@pytest.mark.parametrize("family", CANDIDATE_FAMILIES)
def test_native_pool_warm_and_ledger(loaded: dict[str, Tokenizer], family: str) -> None:
    tokenizer = _family(loaded, family)
    text = _generated_corpora(400_000)["prose(generated)"]
    size = chunked.warm(tokenizer, workers=3, text=text[:20_000])
    assert size >= 3
    assert chunked.warm(tokenizer, workers=2) == size
    native_facts = chunked.admission(tokenizer)["native"]
    assert isinstance(native_facts, dict) and native_facts["workers"] == size
    runtime = tokenizer._native_request_runtime()
    assert runtime is not None
    before = dict(runtime.runtime_stats()["execution_counts"])
    stats_before = chunked.stats(tokenizer)
    encoded = chunked.encode(tokenizer, text, k=6, native=True)
    after = dict(runtime.runtime_stats()["execution_counts"])
    assert encoded.chunks >= 2
    assert sum(after.values()) - sum(before.values()) == encoded.chunks
    stats_after = chunked.stats(tokenizer)
    assert stats_after["native_calls"] == int(str(stats_before["native_calls"])) + 1
    assert stats_after["applied"] == int(str(stats_before["applied"])) + 1
    assert stats_after["chunks_total"] == (
        int(str(stats_before["chunks_total"])) + encoded.chunks
    )
    last = stats_after["last"]
    assert isinstance(last, dict) and last["native"] is True
    assert isinstance(encoded.native_facts, dict)
    assert encoded.native_facts["cold_forks"] == 0 or size < 6
    assert encoded.native_facts["encode_ns"] > 0


# -- the evidence gate on the native path -------------------------------------


@pytest.mark.parametrize("family", CERTIFIED_FAMILIES)
def test_native_encoder_is_the_default_on_a_certified_row_when_gated(
    tmp_path_factory: pytest.TempPathFactory, family: str
) -> None:
    """On an ordinarily gated planner a certified row admits the native
    encoder as well: ``native=None`` takes it wherever it can run and
    ``native=True`` runs it without raising, whatever the row records for
    the native path, which ``native_evidence_label`` names
    (``native_evidence:adversarial_only`` while the corpus level of the
    native path is not recorded). The ids are the reference's on every
    path, and the native plan is the Python plan."""
    if not _artifact_source(family):
        pytest.skip(f"{family}: artifact not available under TOKTIER_TEST_ARTIFACTS")
    tokenizer = load_isolated(family, tmp_path_factory)
    try:
        if not chunked.native_available(tokenizer):
            # The default policy may route this build to the reference
            # engine only; the certified route is what the gate is about.
            tokenizer.close()
            tokenizer = load_isolated(family, tmp_path_factory, policy="experimental")
        if not chunked.native_available(tokenizer):
            pytest.skip(f"{family}: no native chunk encoder on this route")
        row = TABLE_ROWS[family]
        label = chunked._native_evidence_label(tuple(row["evidence"]["native"]))
        text = _generated_corpora(300_000)["prose(generated)"]
        expected = _reference(tokenizer)(text)
        facts = chunked.admission(tokenizer)
        assert facts["admission_gate"] == "evidence_table"
        assert facts["admitted"] is True and facts["evidence_status"] == "certified"
        assert facts["native_available"] is True
        assert facts["native_by_default"] is True
        assert facts["native_evidence_label"] == label
        auto = chunked.encode(tokenizer, text, k=4)
        assert auto.native_facts is not None
        last = chunked.stats(tokenizer)["last"]
        assert isinstance(last, dict) and last["native"] is True
        encoded = chunked.encode(tokenizer, text, k=4, native=True)
        assert encoded.native_facts is not None
        assert chunked.plan(tokenizer, text, k=4, native=True) == chunked.plan(
            tokenizer, text, k=4
        )
        assert list(auto.ids) == expected and auto.chunks >= 2
        assert list(encoded.ids) == expected
        forced = chunked.encode(tokenizer, text, k=4, native=False)
        assert forced.native_facts is None and list(forced.ids) == expected
        report = chunked.stats(tokenizer)
        assert report["native_evidence_label"] == label
        assert report["native_calls"] == 2
    finally:
        tokenizer.close()


@pytest.mark.parametrize("family", CERTIFIED_FAMILIES[:1])
def test_route_without_the_corrected_cpu_engine_is_serial_by_name(
    tmp_path_factory: pytest.TempPathFactory, family: str
) -> None:
    """``R_CHUNK_ROUTE_NOT_FAST_CPU``: on a plan whose starting backend is
    not the corrected CPU engine (here the reference-only route of
    ``policy="reference"``), the native encoder encodes serially through
    the normal route and says so when it is demanded; the ids are the
    plain encode's. The default ``native=None`` does not take it there,
    although the row is admitted: the encoder cannot run on that route,
    so the Python path cuts the document."""
    if not _artifact_source(family):
        pytest.skip(f"{family}: artifact not available under TOKTIER_TEST_ARTIFACTS")
    tokenizer = load_isolated(family, tmp_path_factory, policy="reference")
    try:
        chain = tokenizer.explain()["fallback_chain"]
        assert isinstance(chain, (list, tuple)) and "fast_cpu" not in chain
        runtime = tokenizer._native_request_runtime()
        if runtime is None or not callable(getattr(runtime, "chunked_encode", None)):
            pytest.skip(f"{family}: the reference route has no native runtime here")
        planner = chunked._planner_for(tokenizer)
        assert planner.admitted is True, planner.refusal
        assert chunked.native_available(tokenizer) is False
        text = _generated_corpora(300_000)["prose(generated)"]
        expected = _reference(tokenizer)(text)
        facts = chunked.admission(tokenizer)
        assert facts["native_available"] is False
        assert facts["native_by_default"] is False
        auto = chunked.encode(tokenizer, text, k=4)
        assert auto.native_facts is None and auto.serial_reason is None
        assert auto.chunks > 1 and list(auto.ids) == expected
        encoded = chunked.encode(tokenizer, text, k=4, native=True)
        assert encoded.serial_reason == ReasonCode.R_CHUNK_ROUTE_NOT_FAST_CPU.value
        assert encoded.chunks == 1 and encoded.cuts == ()
        assert encoded.native_facts is not None
        assert list(encoded.ids) == expected
        report = tokenizer.explain()["chunked"]
        assert isinstance(report, dict)
        assert report["fallback_counts"] == {"R_CHUNK_ROUTE_NOT_FAST_CPU": 1}
        assert report["native_calls"] == 1 and report["applied"] == 1
        facts = chunked.admission(tokenizer)
        assert facts["native"] is None
        assert "no corrected CPU engine" in str(facts["native_unavailable_reason"])
    finally:
        tokenizer.close()


@pytest.mark.skipif(
    os.environ.get("TOKTIER_TEST_CHUNKED_GPU_CHAIN") != "1",
    reason=(
        "GPU-headed chain: set TOKTIER_TEST_CHUNKED_GPU_CHAIN=1 on a host with a "
        "prebuilt-kernel GPU to cover R_CHUNK_ROUTE_NOT_FAST_CPU above the crossover"
    ),
)
@pytest.mark.parametrize("family", CERTIFIED_FAMILIES[:1])
def test_gpu_headed_chain_above_the_crossover_is_serial_by_name(
    tmp_path_factory: pytest.TempPathFactory, family: str
) -> None:
    """The same reason code on a GPU-headed plan: an input above the GPU
    crossover starts on the GPU backend, so the native chunk encoder
    encodes it serially through the normal route."""
    if not _artifact_source(family):
        pytest.skip(f"{family}: artifact not available under TOKTIER_TEST_ARTIFACTS")
    try:
        tokenizer = load_isolated(
            family, tmp_path_factory, device="cuda", gpu_delivery="prebuilt"
        )
    except Exception as error:
        pytest.skip(f"{family}: no GPU-headed plan on this host ({error})")
    try:
        chain = tokenizer.explain()["fallback_chain"]
        assert isinstance(chain, (list, tuple)) and chain and chain[0] == "gpu"
        if not chunked.native_available(tokenizer):
            pytest.skip(f"{family}: the GPU-headed route has no native runtime here")
        chunked._planner_for(tokenizer, evidence_gate=False)
        text = _generated_corpora(2_000_000)["prose(generated)"]
        encoded = chunked.encode(tokenizer, text, k=4, native=True)
        assert encoded.serial_reason == ReasonCode.R_CHUNK_ROUTE_NOT_FAST_CPU.value
        assert list(encoded.ids) == _reference(tokenizer)(text)
    finally:
        tokenizer.close()


#: Run in a fresh interpreter with ``RAYON_NUM_THREADS=1``: load the family
#: from the prepared home, name a worker count through ``warm``, encode on
#: the native path, and report the chunk-pool threads seen in
#: ``/proc/self/task`` and, per call, whether its rows overlapped in time:
#: rows run one after another take at least ``max + (rows - 1) * min`` of
#: the per-row times, so an encode phase shorter than that had two rows
#: running at once.
_POOL_PROBE = r"""
import hashlib, json, os, sys
from pathlib import Path
import toktier
from toktier import Config, chunked

family, home, text_path = sys.argv[1:4]
workers = int(sys.argv[4])
config = Config(home=Path(home), offline=True)
tokenizer = toktier.load(family, config=config, device="cpu")
chain = tokenizer.explain()["fallback_chain"]
if "fast_cpu" not in chain:
    tokenizer.close()
    tokenizer = toktier.load(family, config=config, device="cpu", policy="experimental")
chunked._planner_for(tokenizer, evidence_gate=False)
text = Path(text_path).read_text(encoding="utf-8")
default_before = chunked.admission(tokenizer)["native"]["default_workers"]
size = chunked.warm(tokenizer, workers=workers)
default_after = chunked.admission(tokenizer)["native"]["default_workers"]
serial = list(tokenizer.encode(text, lookup="off").ids)
chunk_counts = []
overlapped = []
for _ in range(4):
    encoded = chunked.encode(tokenizer, text, k=workers, native=True)
    assert list(encoded.ids) == serial
    chunk_counts.append(encoded.chunks)
    facts = encoded.native_facts
    rows = encoded.chunks
    floor = facts["worker_ns_max"] + (rows - 1) * facts["worker_ns_min"]
    overlapped.append(facts["encode_ns"] < floor)
threads = {}
for task in Path("/proc/self/task").iterdir():
    try:
        name = (task / "comm").read_text().strip()
        fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        continue
    # Keyed by thread id: Linux keeps 15 bytes of a thread name, so
    # "toktier-chunk-10" and "toktier-chunk-1" read the same.
    if name.startswith("toktier-chunk"):
        threads[task.name] = int(fields[11]) + int(fields[12])
print(json.dumps({
    "rayon_env": os.environ.get("RAYON_NUM_THREADS"),
    "default_before": default_before,
    "default_after": default_after,
    "warm_size": size,
    "chunks": chunk_counts,
    "overlapped": overlapped,
    "threads": threads,
    "ids_sha256": hashlib.sha256(json.dumps(serial).encode()).hexdigest(),
}))
"""


@pytest.mark.skipif(
    not Path("/proc/self/task").is_dir(), reason="needs the /proc thread view"
)
def test_named_workers_parallelize_with_a_single_thread_global_pool(
    loaded: dict[str, Tokenizer], tmp_path: Path
) -> None:
    """The chunk pool does not follow ``RAYON_NUM_THREADS``: with the global
    Rayon pool at one thread, ``warm(workers=N)`` builds an N-thread chunk
    pool, the rows of a chunked encode run at once, and the ids still equal
    the oracle's."""
    present = [name for name in CERTIFIED_FAMILIES if name in loaded]
    if not present:
        pytest.skip("no admitted family available under TOKTIER_TEST_ARTIFACTS")
    family = present[0]
    source = _artifact_source(family)
    assert source is not None
    home = tmp_path / "home"
    config = Config(home=home, offline=True)
    cache = artifact_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    shutil.copytree(source, cache / manifest.get(family).directory_name)
    text = _generated_corpora(1_200_000)["prose(generated)"]
    text_path = tmp_path / "text.txt"
    text_path.write_text(text, encoding="utf-8")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
    # At least 12, so the pool holds thread names that the kernel
    # truncates to the same 15 bytes, whatever this host's CPU count.
    workers = max(affinity + 2, 12)
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    argv = [sys.executable, "-c", _POOL_PROBE, family, str(home), str(text_path)]
    completed = subprocess.run(
        [*argv, str(workers)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    facts = json.loads(completed.stdout.strip().splitlines()[-1])
    assert facts["rayon_env"] == "1"
    assert 1 <= facts["default_before"] <= max(affinity, 1)
    assert facts["warm_size"] >= workers
    assert facts["default_after"] == workers
    assert min(facts["chunks"]) >= 2
    threads = facts["threads"]
    assert len(threads) == workers, threads
    assert any(facts["overlapped"]), facts
    expected = _reference(loaded[family])(text)
    digest = hashlib.sha256(json.dumps(expected).encode()).hexdigest()
    assert facts["ids_sha256"] == digest


#: A fresh interpreter that checks two facts of the chunk pool's life:
#: ``warm(workers=0)`` forks nothing and starts no thread, and a process
#: forked after the pool exists encodes in the child (the child builds
#: its own pool) instead of waiting on threads that live in the parent.
_POOL_LIFE_PROBE = r"""
import json, os, signal, sys
from pathlib import Path
import toktier
from toktier import Config, chunked

family, home, text_path = sys.argv[1:4]
config = Config(home=Path(home), offline=True)
tokenizer = toktier.load(family, config=config, device="cpu")
if "fast_cpu" not in tokenizer.explain()["fallback_chain"]:
    tokenizer.close()
    tokenizer = toktier.load(family, config=config, device="cpu", policy="experimental")
chunked._planner_for(tokenizer, evidence_gate=False)
text = Path(text_path).read_text(encoding="utf-8")
serial = list(tokenizer.encode(text, lookup="off").ids)

def chunk_threads():
    names = []
    for task in Path("/proc/self/task").iterdir():
        try:
            names.append((task / "comm").read_text().strip())
        except OSError:
            continue
    return sum(name.startswith("toktier-chunk") for name in names)

default_before = chunked.admission(tokenizer)["native"]["default_workers"]
zero = chunked.warm(tokenizer, workers=0)
default_after = chunked.admission(tokenizer)["native"]["default_workers"]
after_zero = {"returned": zero, "threads": chunk_threads(),
              "default_workers": default_after}
chunked.warm(tokenizer, workers=2)
parent_equal = list(chunked.encode(tokenizer, text, k=2, native=True).ids) == serial
pid = os.fork()
if pid == 0:
    signal.alarm(120)
    ok = list(chunked.encode(tokenizer, text, k=2, native=True).ids) == serial
    os._exit(0 if ok else 3)
_, status = os.waitpid(pid, 0)
print(json.dumps({"default_before": default_before, "after_zero": after_zero,
                  "parent_equal": parent_equal,
                  "child_status": os.waitstatus_to_exitcode(status)}))
"""


@pytest.mark.skipif(
    not Path("/proc/self/task").is_dir() or not hasattr(os, "fork"),
    reason="needs the /proc thread view and fork",
)
def test_warm_zero_is_inert_and_a_forked_child_builds_its_own_pool(
    loaded: dict[str, Tokenizer], tmp_path: Path
) -> None:
    """``warm(workers=0)`` names no worker: nothing is forked, no chunk
    thread starts and the default count is unchanged. A child forked after
    the pool exists does not inherit a pool without threads: its chunked
    encode returns the serial ids instead of waiting forever."""
    present = [name for name in CERTIFIED_FAMILIES if name in loaded]
    if not present:
        pytest.skip("no admitted family available under TOKTIER_TEST_ARTIFACTS")
    family = present[0]
    source = _artifact_source(family)
    assert source is not None
    home = tmp_path / "home"
    cache = artifact_cache_dir(Config(home=home, offline=True))
    cache.mkdir(parents=True, exist_ok=True)
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    shutil.copytree(source, cache / manifest.get(family).directory_name)
    text_path = tmp_path / "text.txt"
    text = _generated_corpora(200_000)["prose(generated)"]
    text_path.write_text(text, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-c", _POOL_LIFE_PROBE, family, str(home), str(text_path)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    facts = json.loads(completed.stdout.strip().splitlines()[-1])
    assert facts["after_zero"] == {
        "returned": 0,
        "threads": 0,
        "default_workers": facts["default_before"],
    }, facts
    assert facts["parent_equal"]
    # 0: the child's ids equal; 3: they differ; -14 (SIGALRM): it hung.
    assert facts["child_status"] == 0, facts
