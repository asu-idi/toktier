"""Added-token literals are segmented on the native route of real artifacts.

Artifact tier only (``TOKTIER_TEST_ARTIFACTS``, see ``test_chunked.py``):
for every family with a corrected CPU engine found there, chat-shaped
documents built from the face's own literals and a few literal-dense
strings (each literal alone, doubled, adjacent to another, between
letters, digits, white space and non-ASCII text, and cut short) are
encoded on the native ID route, as a batch, and as a session seed with
an append. Every answer must equal the reference library's encode of
the loader face. Where the face is served by literal segmentation, a
literal-bearing document whose text is stable under the face's
normalizer must have run segmented on the corrected CPU engine, with no
byte encoded by the reference. Until the source identities are rebound
the families are loaded under the experimental policy, as in
``test_chunked_native.py``; these are readings of the route, not of
certification.
"""

from __future__ import annotations

import json
import shutil
import unicodedata
from collections.abc import Iterator

import pytest

import toktier
from toktier import Config, Tokenizer
from toktier.artifacts.manifest import ArtifactManifest
from toktier.artifacts.tables import ARTIFACT_MANIFEST
from toktier.paths import artifact_cache_dir

from .test_chunked import _artifact_source

tokenizers = pytest.importorskip("tokenizers")

#: The families whose route carries the corrected CPU engine.
CPU_ENGINE_FAMILIES = (
    "deepseek_v3",
    "deepseek_v4_1_flash",
    "deepseek_v4_flash",
    "glm_5_2",
    "gpt_oss_120b",
    "llama_3_1_8b",
    "mimo_v2_6",
    "minimax_m3",
    "ministral_3_8b",
    "nemotron_3_nano_4b",
    "olmo_3_7b",
    "qwen3_5_08b",
    "qwen3_8b",
)

_BODIES = (
    "Hello there, could you summarise the plan for next week?",
    "Sure. The plan has three steps: 1) gather data, 2) fit, 3) report.",
    "caf\u00e9 \u4e2d\u6587 \u0928\u092e\u0938\u094d\u0924\u0947 "
    "na\u00efve r\u00e9sum\u00e9",
    "def f(x):\n    return x * 2  # doubled\n",
    "   leading and trailing spaces   ",
)
_NEIGHBOURS = ("", "a", "1", " ", "\n", ".", "\u4e2d", "\u00e9")


def _texts(literals: list[str]) -> list[str]:
    roles = literals[:6]
    out: list[str] = []
    for turns in range(1, 5):
        text = ""
        for turn in range(turns):
            opener = roles[turn % len(roles)]
            closer = roles[(turn + 1) % len(roles)]
            text += f"{opener}user\n{_BODIES[turn % len(_BODIES)]}{closer}\n"
        out.append(text)
    for index, literal in enumerate(literals[:24]):
        left = _NEIGHBOURS[index % len(_NEIGHBOURS)]
        right = _NEIGHBOURS[(index * 3 + 1) % len(_NEIGHBOURS)]
        out += [
            literal,
            literal + literal,
            f"{left}{literal}{right}",
            f"{literal}{literals[(index + 1) % len(literals)]}",
            f"text {index} {literal} more text {literal}",
            f"a{literal[:-1]}b" if len(literal) > 1 else f"a{literal}b",
        ]
    return out


@pytest.fixture(scope="module")
def faces(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Tokenizer]]:
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    handles: dict[str, Tokenizer] = {}
    for family in CPU_ENGINE_FAMILIES:
        source = _artifact_source(family)
        if source is None:
            continue
        home = tmp_path_factory.mktemp(f"literal-segments-{family}")
        config = Config(home=home, offline=True)
        cache = artifact_cache_dir(config)
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, cache / manifest.get(family).directory_name)
        handles[family] = toktier.load(
            family, config=config, device="cpu", policy="experimental"
        )
    if not handles:
        pytest.skip("no artifact under TOKTIER_TEST_ARTIFACTS")
    yield handles
    for handle in handles.values():
        handle.close()


@pytest.mark.parametrize("family", CPU_ENGINE_FAMILIES)
def test_literal_bearing_input_is_segmented_and_equals_the_reference(
    faces: dict[str, Tokenizer], family: str
) -> None:
    if family not in faces:
        pytest.skip(f"{family}: artifact not available under TOKTIER_TEST_ARTIFACTS")
    tokenizer = faces[family]
    face = tokenizer._backend.materialized_tokenizer_json()
    oracle = tokenizers.Tokenizer.from_str(face)
    document = json.loads(face)
    literals = [
        row["content"] for row in document.get("added_tokens") or [] if row["content"]
    ]
    assert literals, family
    has_nfc = (document.get("normalizer") or {}).get("type") == "NFC"
    tokenizer.encode("warm up the native request runtime", lookup="off")
    assert tokenizer._native_request_initialised
    runtime = tokenizer._native_request
    assert runtime is not None
    assert "fast_cpu" in list(runtime.fallback_chain), family
    facts = runtime.runtime_stats()["literal_segmentation"]
    served = facts["served"]
    texts = _texts(literals)

    def reference(text: str) -> list[int]:
        return list(oracle.encode(text, add_special_tokens=False).ids)

    segmented = 0
    for text in texts:
        ids = list(runtime.encode(text, lookup_auto=False))
        assert ids == reference(text), (family, text)
        last = runtime.runtime_stats()["last_execution"] or {}
        holds_literal = any(literal in text for literal in literals)
        stable = not has_nfc or unicodedata.normalize("NFC", text) == text
        if served and holds_literal and stable:
            assert last.get("path") == "gigatoken_segmented", (family, text, last)
            assert last["segments"]["bytes"]["hf"] == 0, (family, text, last)
            segmented += 1
    if served:
        assert segmented > 0, family
    rows = runtime.encode_batch(texts)
    assert [list(row) for row in rows] == [reference(text) for text in texts]
    # A session seeded with a literal-bearing transcript, then extended by
    # a turn whose opening literal is cut across the two appends.
    seed = texts[3]
    opener = literals[0]
    cut = max(1, len(opener) // 2)
    with tokenizer.session(session_id=f"segments-{family}", text=seed) as session:
        assert list(session.ids) == reference(seed)
        session.append(opener[:cut])
        session.append(opener[cut:] + "assistant\nok")
        assert list(session.ids) == reference(seed + opener + "assistant\nok")
