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

import json
import os
import random
import shutil
from collections.abc import Iterator

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
    _adversarial,
    _artifact_source,
    _generated_corpora,
    _reference,
    corpora,
)

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
