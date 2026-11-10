"""The seal condition reaches a user of the JIT delivery.

Under ``gpu_delivery="jit"`` the facade serves stored state from its
Python-owned store (``facade/store.py`` over the callback encoder, wired by
``facade/api.py``), with the device on the seed path. That lane decides a
seal with the face reference's normalizer (``seal_cut.rs``); the wiring
lives outside every source identity, so it is checked here end to end: a
persistent session seeded with a letter that NFC turns into a letter and a
combining mark right before a full stop at the last boundary under the seal
ceiling, appends, a reopen in a second tokenizer object, and the default
call twice. Every answer equals ``tokenizers`` on the same face, the stored
record seals, and on the specimen the seal lies before the letter (the
ASCII control seals after it). On a build from before the seal change the
answers after the seed differ.

The scenario runs in a child process (the JIT kernel it loads does not
reach the other GPU tests). Marked ``gpu``; skipped without torch, a CUDA
device, the JIT toolchain or the artifacts (as in
``test_gpu_state_nfc_closure.py``). The file stays ASCII.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .test_facade_jit_lifetime import _child_environment
from .test_gpu_state_nfc_closure import _artifact_source, _cuda_or_skip

pytestmark = pytest.mark.gpu

FAMILIES = ("qwen3_8b", "mimo_v2_6")

_CHILD = r"""
import json, pathlib, sys
import tokenizers
import toktier
from toktier import records
from toktier.config import Config
home, root, family = sys.argv[1], sys.argv[2], sys.argv[3]
config = Config(home=home, offline=True)
FILL = "The committee met on Tuesday and agreed to the plan for the coming year. "
DELTA = "One more sentence arrives after the seed. "
LATER = " A reopened session continues the text."
def load(store=None):
    return toktier.load(family, config=config, device="cuda", gpu_delivery="jit",
                        gpu_min_bytes=256, policy="experimental", store=store)
def sealed(directory):
    return [records.decode_record(p.read_bytes()).stable_prefix_byte_length
            for p in sorted(pathlib.Path(directory, "entries").glob("*.rec"))]
out = {"cases": [], "toktier_file": toktier.__file__}
for name, letter in (("u0958", "\u0958"), ("u095b", "\u095b"), ("u09dc", "\u09dc"),
                     ("control_x", "x")):
    store = pathlib.Path(root) / name
    seed = FILL * 60 + "ab " + letter + ".com" + "a" * 700
    try:
        tok = load(str(store))
    except Exception as error:
        out["unavailable"] = f"{type(error).__name__}: {error}"
        break
    out["chain"] = list(tok.plan.fallback_chain)
    oracle = tokenizers.Tokenizer.from_str(tok._backend.materialized_tokenizer_json())
    ref = lambda text: list(oracle.encode(text, add_special_tokens=False).ids)
    steps = []
    with tok.session(session_id="s", text=seed) as session:
        steps.append(("seed", list(session.ids) == ref(seed)))
        session.append(DELTA)
        steps.append(("append", list(session.ids) == ref(seed + DELTA)))
        session.append(FILL * 4)
        grown = seed + DELTA + FILL * 4
        steps.append(("append2", list(session.ids) == ref(grown)))
    tok.close()
    tok = load(str(store))
    with tok.session(session_id="s", text=grown) as session:
        steps.append(("reopened", list(session.ids) == ref(grown)))
        session.append(LATER)
        steps.append(("append_after_reopen", list(session.ids) == ref(grown + LATER)))
    tok.close()
    tok = load()
    first = list(tok.encode(seed, add_special_tokens=False).ids) == ref(seed)
    grown_ids = tok.encode(seed + DELTA, add_special_tokens=False).ids
    second = list(grown_ids) == ref(seed + DELTA)
    steps.append(("default_call_twice", first and second))
    tok.close()
    out["cases"].append({"case": name, "steps": steps, "sealed": sealed(store),
                         "letter_at": len((FILL * 60 + "ab ").encode())})
print(json.dumps(out))
"""


@pytest.mark.parametrize("family", FAMILIES)
def test_a_jit_session_seals_where_the_reference_agrees(
    request: pytest.FixtureRequest, tmp_path: Path, family: str
) -> None:
    _cuda_or_skip()
    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.config import Config
    from toktier.paths import artifact_cache_dir

    config = Config(home=tmp_path / "home", offline=True)
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    target = artifact_cache_dir(config) / manifest.get(family).directory_name
    target.mkdir(parents=True)
    for name, source in _artifact_source(request, family).items():
        shutil.copyfile(source, target / name)
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(tmp_path / "home"),
            str(tmp_path / "stores"),
            family,
        ],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=3600,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    # The child tested the package of this tree, not an installed one.
    tree_source = str(Path(__file__).resolve().parents[2] / "src")
    assert result["toktier_file"].startswith(tree_source), result
    if "unavailable" in result:
        pytest.skip(f"the JIT delivery is unavailable: {result['unavailable']}")
    assert result["chain"] == ["gpu", "fast_cpu", "hf"], result
    assert len(result["cases"]) == 4, result
    by_case = {case["case"]: case for case in result["cases"]}
    for case in result["cases"]:
        for step, equal in case["steps"]:
            assert equal, (case["case"], step)
        (sealed,) = case["sealed"]
        assert sealed > 0, case
    # The specimen seals before the letter; the ASCII control seals after it.
    control = by_case["control_x"]
    assert control["sealed"][0] > control["letter_at"], control
    for name in ("u0958", "u095b", "u09dc"):
        case = by_case[name]
        assert case["sealed"][0] <= case["letter_at"], case
