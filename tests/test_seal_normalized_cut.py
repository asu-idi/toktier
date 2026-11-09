"""A session seal decides its boundary on the characters the reference reads.

Artifact tier only (``TOKTIER_TEST_ARTIFACTS``, see ``test_chunked.py``).
A session store seals its tail at a boundary the frozen class predicate
accepts and later encodes the text after it on its own. On a face whose
normalizer is NFC the reference pre-tokenizes the normalized text, so a seal
candidate also has to pass the chunk planner's normalized-cut condition
(``seal_cut.rs``). The specimen below puts a letter that NFC turns into a
letter and a combining mark right before a full stop, at the last boundary
under the seal ceiling, on a tail the reference seeded; without that
condition the store seals there and the next answer differs from the
reference. Until the source identities are rebound the families are loaded
under the experimental policy, as in ``test_literal_segments.py``.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import toktier
from toktier import Config, Tokenizer
from toktier.artifacts.manifest import ArtifactManifest
from toktier.artifacts.tables import ARTIFACT_MANIFEST
from toktier.paths import artifact_cache_dir

from .test_chunked import _artifact_source

tokenizers = pytest.importorskip("tokenizers")

#: Faces whose normalizer is NFC and whose letter class has no marks.
AFFECTED_FAMILIES = ("qwen3_8b", "mimo_v2_6")

FILL = "The committee met on Tuesday and agreed to the plan for the coming year. "
DELTA = "One more sentence arrives after the seed. "
LATER = "A reopened session continues the text. "


def _specimen(letter: str) -> str:
    return FILL * 60 + "ab " + letter + ".com" + "a" * 700


def _home(tmp_path_factory: pytest.TempPathFactory, family: str) -> Config:
    source = _artifact_source(family)
    if source is None:
        pytest.skip(f"{family}: artifact not available under TOKTIER_TEST_ARTIFACTS")
    assert source is not None
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    home = tmp_path_factory.mktemp(f"seal-{family}")
    config = Config(home=home, offline=True)
    cache = artifact_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, cache / manifest.get(family).directory_name)
    return config


def _load(config: Config, family: str, store: Path | None = None) -> Tokenizer:
    return toktier.load(
        family, config=config, device="cpu", policy="experimental", store=store
    )


@pytest.fixture(scope="module", params=AFFECTED_FAMILIES)
def face(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[str, Config, Tokenizer, object]]:
    family = request.param
    config = _home(tmp_path_factory, family)
    tokenizer = _load(config, family)
    assert "fast_cpu" in list(tokenizer.plan.fallback_chain), family
    oracle = tokenizers.Tokenizer.from_str(
        tokenizer._backend.materialized_tokenizer_json()
    )
    yield family, config, tokenizer, oracle
    tokenizer.close()


def _reference(oracle: object, text: str) -> list[int]:
    return list(oracle.encode(text, add_special_tokens=False).ids)  # type: ignore[attr-defined]


def test_the_specimen_is_not_stable_and_reaches_a_seal(
    face: tuple[str, Config, Tokenizer, object],
) -> None:
    _family, _config, tokenizer, oracle = face
    text = _specimen("\u0958")
    normalizer = oracle.normalizer  # type: ignore[attr-defined]
    assert normalizer.normalize_str(text) != text
    with tokenizer.session(session_id="seal-probe", text=text) as session:
        session.append(DELTA)
        assert list(session.ids) == _reference(oracle, text + DELTA)


@pytest.mark.parametrize("letter", ["\u0958", "\u095b", "\u09dc", "\u0915\u093c", "x"])
def test_session_append_equals_the_reference(
    face: tuple[str, Config, Tokenizer, object], letter: str
) -> None:
    _family, _config, tokenizer, oracle = face
    text = _specimen(letter)
    with tokenizer.session(session_id=f"seal-{ord(letter[0]):x}", text=text) as session:
        assert list(session.ids) == _reference(oracle, text)
        session.append(DELTA)
        assert list(session.ids) == _reference(oracle, text + DELTA)
        session.append(FILL * 4)
        assert list(session.ids) == _reference(oracle, text + DELTA + FILL * 4)


def test_the_default_call_twice_equals_the_reference(
    face: tuple[str, Config, Tokenizer, object],
) -> None:
    _family, _config, tokenizer, oracle = face
    text = _specimen("\u0958")
    assert list(tokenizer.encode(text).ids) == _reference(oracle, text)
    assert list(tokenizer.encode(text + DELTA).ids) == _reference(oracle, text + DELTA)


def test_a_persistent_reopen_equals_the_reference(
    face: tuple[str, Config, Tokenizer, object], tmp_path: Path
) -> None:
    family, config, _tokenizer, oracle = face
    store = tmp_path / "store"
    text = _specimen("\u0958")
    first = _load(config, family, store)
    try:
        with first.session(session_id="seal-persist", text=text) as session:
            session.append(DELTA)
            assert list(session.ids) == _reference(oracle, text + DELTA)
    finally:
        first.close()
    full = text + DELTA
    second = _load(config, family, store)
    try:
        with second.session(session_id="seal-persist", text=full) as session:
            assert list(session.ids) == _reference(oracle, full)
            session.append(LATER)
            assert list(session.ids) == _reference(oracle, full + LATER)
    finally:
        second.close()


def _code_points_nfc_changes(oracle: object) -> list[str]:
    normalizer = oracle.normalizer  # type: ignore[attr-defined]
    changed = []
    for value in range(sys.maxunicode + 1):
        if 0xD800 <= value <= 0xDFFF:
            continue
        character = chr(value)
        if normalizer.normalize_str(character) != character:
            changed.append(character)
    return changed


def test_every_code_point_nfc_changes_seals_where_the_reference_agrees(
    face: tuple[str, Config, Tokenizer, object],
) -> None:
    _family, _config, tokenizer, oracle = face
    changed = _code_points_nfc_changes(oracle)
    # Taken from the normalizer itself; Unicode has about a thousand.
    assert len(changed) > 500
    unequal = []
    for index, character in enumerate(changed):
        text = FILL * 60 + "ab " + character + ".com" + "a" * 700
        with tokenizer.session(session_id=f"cp-{index}", text=text) as session:
            session.append(DELTA)
            if list(session.ids) != _reference(oracle, text + DELTA):
                unequal.append(f"U+{ord(character):04X}")
    assert unequal == []


@pytest.mark.parametrize("family", AFFECTED_FAMILIES)
@pytest.mark.parametrize("with_reference", [True, False])
def test_the_python_owned_lane_applies_the_same_condition(
    family: str, with_reference: bool, tmp_path: Path
) -> None:
    """The callback lane (the store the facade owns under JIT delivery)
    decides seals with the face reference's normalizer; without one it
    certifies no boundary, so it never seals."""
    from toktier import _native, records
    from toktier.facade.store import EntryStore
    from toktier.repair.registry import pclass_table

    source = _artifact_source(family)
    if source is None:
        pytest.skip(f"{family}: artifact not available under TOKTIER_TEST_ARTIFACTS")
    assert source is not None
    path = source / "tokenizer.json"
    oracle = tokenizers.Tokenizer.from_file(str(path))

    def encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoding = oracle.encode(text, add_special_tokens=False)
        return list(encoding.ids), [tuple(item) for item in encoding.offsets]

    def run(text: str, name: str) -> int:
        directory = tmp_path / name
        store = EntryStore(
            fingerprint=bytes(range(32)),
            encode=encode,
            certified_bpe_witness=True,
            bpe_sync_pclass=pclass_table(),
            seal_reference=_native.ReferenceEngine(str(path))
            if with_reference
            else None,
            directory=directory,
        )
        assert store.encode_session("s", text) == _reference(oracle, text)
        assert store.encode_session("s", text + DELTA) == _reference(
            oracle, text + DELTA
        )
        grown = text + DELTA + FILL * 4
        assert store.encode_session("s", grown) == _reference(oracle, grown)
        (record,) = (directory / "entries").glob("*.rec")
        return records.decode_record(record.read_bytes()).stable_prefix_byte_length

    sealed_unstable = run(_specimen("\u0958"), "unstable")
    sealed_ascii = run(_specimen("x"), "ascii")
    if with_reference:
        assert sealed_ascii > 0
    else:
        assert sealed_ascii == 0 and sealed_unstable == 0
