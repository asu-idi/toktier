"""GPU-headed state route on NFC faces: a device row is adopted only for the
text it encodes as given.

On a family whose normalizer is NFC, the device engine normalizes the input
itself and returns the ids of the normalized text. When the state route adopted
such a row because the tokens' byte lengths summed to the text length, a text
that is not NFC-stable but whose byte-length changes cancel (three decomposed
accents early, one U+0958 late: -1 -1 -1 +3) passed that check, the spans
rebuilt over the raw text were shifted between the two characters, the session
store sealed at a shifted cut, and the next append returned ids that differ
from the reference (one token repeated at the cut: "... agreed to to the ...").

These tests seed a session (and the default content store) on a
``[gpu, fast_cpu, hf]`` chain and compare the ids after every step with
``tokenizers`` on the same face. Either outcome of a fix is accepted: the
row is refused (``hf_span_guard``) or adopted with correct spans; only the ids
are asserted. The NFC-stable control asserts that the device path is really
taken, so a pass is not one where the reference answered every request.

Marked ``gpu``; skipped without torch, a CUDA device or the prebuilt delivery.
Artifacts: the GPU tier's ``artifact_handles`` fixture when this file lives in
``tests/gpu`` (``--artifact-manifest``), else a directory named by the
environment variable ``TOKTIER_TEST_ARTIFACTS`` that holds the shipped cache
layout (``<family>-<sha12>/tokenizer.json`` ...). The file stays ASCII.
"""

from __future__ import annotations

import os
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.gpu

NFC_FAMILIES = ("qwen3_8b", "qwen3_5_08b", "mimo_v2_6", "minimax_m3")
FILL = "The committee met on Tuesday and agreed to the plan for the coming year. "
DELTAS = (
    "One more sentence arrives after the seed. ",
    FILL * 4,
    "Final words here, and a short closing line.\n",
)
DECOMPOSED = (
    "cafe\u0301 re\u0301sume\u0301 "  # three accents, each 1 byte shorter after NFC
)
QA = "\u0958 "  # DEVANAGARI LETTER QA, 3 bytes longer after NFC (composition exclusion)


def _balanced(fill_repeats: int, tail_repeats: int = 2) -> str:
    head = fill_repeats * 6 // 10
    return (
        FILL * head
        + DECOMPOSED
        + FILL * (fill_repeats - head)
        + QA
        + FILL * tail_repeats
    )


def _cuda_or_skip() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device is available")


def _artifact_source(request: pytest.FixtureRequest, family: str) -> dict[str, Path]:
    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST

    entry = ArtifactManifest.load(ARTIFACT_MANIFEST).get(family)
    names = [f.name for f in entry.files]
    try:
        handles = request.getfixturevalue("artifact_handles")
    except pytest.FixtureLookupError:
        handles = None
    if handles is not None:
        if family not in handles:
            pytest.skip(f"the artifact manifest does not pin {family}")
        return {name: Path(handles[family].path(name)) for name in names}
    root = os.environ.get("TOKTIER_TEST_ARTIFACTS")
    if root is None or not root:
        pytest.skip(
            "no artifact source (artifact_handles fixture or TOKTIER_TEST_ARTIFACTS)"
        )
    directory = Path(str(root)) / entry.directory_name
    if not directory.is_dir():
        pytest.skip(f"{directory} is absent")
    return {name: directory / name for name in names}


@pytest.fixture(params=NFC_FAMILIES)
def nfc_face(request: pytest.FixtureRequest, tmp_path: Path) -> dict[str, Any]:
    _cuda_or_skip()
    import tokenizers

    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.config import Config
    from toktier.paths import artifact_cache_dir

    family = request.param
    files = _artifact_source(request, family)
    config = Config(home=tmp_path / "home", offline=True)
    entry = ArtifactManifest.load(ARTIFACT_MANIFEST).get(family)
    target = artifact_cache_dir(config) / entry.directory_name
    target.mkdir(parents=True)
    for name, source in files.items():
        shutil.copyfile(source, target / name)
    reference = tokenizers.Tokenizer.from_file(str(target / "tokenizer.json"))
    return {
        "family": family,
        "config": config,
        "reference": reference,
        "store": tmp_path / "store",
    }


def _load(face: dict[str, Any], crossover: int, persistent: bool = False) -> Any:
    import toktier
    from toktier.errors import ToktierError

    kwargs: dict[str, Any] = dict(
        policy="experimental",
        device="cuda",
        gpu_delivery="prebuilt",
        gpu_min_bytes=crossover,
        config=face["config"],
    )
    if persistent:
        kwargs["store"] = str(face["store"])
    try:
        tokenizer = toktier.load(face["family"], **kwargs)
    except ToktierError as error:  # e.g. no prebuilt kernel for this device
        pytest.skip(f"prebuilt GPU delivery unavailable: {error}")
    if tokenizer.explain().get("fallback_chain") != ["gpu", "fast_cpu", "hf"]:
        tokenizer.close()
        pytest.skip("this face does not plan a [gpu, fast_cpu, hf] chain")
    return tokenizer


def _ref(face: dict[str, Any], text: str) -> list[int]:
    return list(face["reference"].encode(text, add_special_tokens=False).ids)


def _seed_state(tokenizer: Any) -> dict[str, Any]:
    return dict(tokenizer.explain()["state_encode"]["last"] or {})


def _session_equal_after_appends(
    face: dict[str, Any], tokenizer: Any, text: str, name: str
) -> dict[str, Any]:
    with tokenizer.session(session_id=name, text=text) as session:
        assert list(session.ids) == _ref(face, text), (
            "seed ids differ from the reference"
        )
        state = _seed_state(tokenizer)
        full = text
        for step, delta in enumerate(DELTAS, start=1):
            session.append(delta)
            full += delta
            assert list(session.ids) == _ref(face, full), (
                f"ids differ from the reference after append {step}"
            )
    return state


def test_control_takes_the_device_state_path(nfc_face: dict[str, Any]) -> None:
    """Non-vacuity: an NFC-stable text of the same shape is seeded from the
    device row."""
    text = unicodedata.normalize("NFC", _balanced(100))
    tokenizer = _load(nfc_face, crossover=256)
    try:
        state = _session_equal_after_appends(nfc_face, tokenizer, text, "control")
        assert state.get("backend") == "gpu", state
        assert str(state.get("path", "")).startswith("accelerated"), state
    finally:
        tokenizer.close()


@pytest.mark.parametrize("persistent", [False, True], ids=["in_memory", "persistent"])
def test_balanced_nfc_changes_keep_session_ids_exact(
    nfc_face: dict[str, Any], persistent: bool
) -> None:
    text = _balanced(100)
    assert unicodedata.normalize("NFC", text) != text
    assert len(unicodedata.normalize("NFC", text).encode()) == len(text.encode())
    tokenizer = _load(nfc_face, crossover=256, persistent=persistent)
    try:
        _session_equal_after_appends(
            nfc_face, tokenizer, text, f"balanced-{persistent}"
        )
    finally:
        tokenizer.close()


def test_balanced_nfc_changes_at_the_default_crossover(
    nfc_face: dict[str, Any],
) -> None:
    """The same input class above the shipped 64 KiB crossover, with the late
    character in the session tail."""
    text = _balanced(950)
    assert len(text.encode()) > 64 * 1024
    tokenizer = _load(nfc_face, crossover=64 * 1024)
    try:
        _session_equal_after_appends(nfc_face, tokenizer, text, "balanced-default")
    finally:
        tokenizer.close()


def test_balanced_nfc_changes_through_the_default_call(
    nfc_face: dict[str, Any],
) -> None:
    text = _balanced(100) + " "
    tokenizer = _load(nfc_face, crossover=256)
    try:
        assert list(tokenizer.encode(text).ids) == _ref(nfc_face, text)
        grown = text + DELTAS[0]
        assert list(tokenizer.encode(grown).ids) == _ref(nfc_face, grown)
    finally:
        tokenizer.close()


def test_balanced_nfc_changes_inside_chat_literals(nfc_face: dict[str, Any]) -> None:
    """A chat-shaped document whose one text segment holds the balanced text."""
    import json

    tokenizer = _load(nfc_face, crossover=256)
    try:
        face_json = json.loads(nfc_face["reference"].to_str())
        literals = [
            r["content"]
            for r in face_json.get("added_tokens") or []
            if r.get("content")
        ]
        opening = next((c for c in literals if "im_start" in c), None) or next(
            (c for c in literals if "start" in c.lower()), None
        )
        closing = next((c for c in literals if "im_end" in c), None) or next(
            (c for c in literals if "end" in c.lower()), None
        )
        if opening is None or closing is None:
            pytest.skip("this face has no opening / closing literal")
        assert opening is not None and closing is not None
        text = opening + "user\n" + _balanced(100) + closing + "\n"
        _session_equal_after_appends(nfc_face, tokenizer, text, "balanced-chat")
    finally:
        tokenizer.close()
