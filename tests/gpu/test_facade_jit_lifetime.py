"""Tokenizers on the JIT delivery are released once their owner drops them.

Under ``gpu_delivery="jit"`` the facade serves stored state from its
Python-owned entry store, whose callbacks live in a native encoder that
the cycle collector cannot see into. A tokenizer that opened a session
there must still be released when the last reference to it goes, with or
without ``close()``. The scenario loads several tokenizers on the device,
opens a session on each (half of them over a store directory), appends,
checks every answer against ``tokenizers`` on the same face, drops them
and counts the survivors after a collection.

It runs in a child process (the JIT kernel it loads does not reach the
other GPU tests). Marked ``gpu``; skipped without torch, a CUDA device,
the JIT toolchain or the artifacts (as in ``test_gpu_jit_session_seal.py``).
The file stays ASCII.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .test_gpu_state_nfc_closure import _artifact_source, _cuda_or_skip

pytestmark = pytest.mark.gpu

FAMILIES = ("qwen3_8b", "mimo_v2_6")
LOADS = 4

_CHILD = r"""
import gc, json, pathlib, sys, weakref
import tokenizers
import toktier
from toktier import _native
from toktier.config import Config
home, root, family, policy = sys.argv[1:5]
loads = int(sys.argv[5])
config = Config(home=home, offline=True)
FILL = "The committee met on Tuesday and agreed to the plan for the coming year. "
DELTAS = (" One more sentence arrives after the seed.", " And a second one, 42.",
          " A third piece closes the turn.")
def load(store=None):
    options = dict(config=config, device="cuda", gpu_delivery="jit", gpu_min_bytes=256)
    if policy != "default":
        options["policy"] = policy
    if store is not None:
        options["store"] = store
    return toktier.load(family, **options)
out = {"equal": [], "alive": None, "toktier_file": toktier.__file__}
refs = []
oracle = None
for index in range(loads):
    try:
        tok = load(str(pathlib.Path(root) / f"s{index}") if index % 2 else None)
    except Exception as error:
        out["unavailable"] = f"{type(error).__name__}: {error}"
        break
    out["chain"] = list(tok.plan.fallback_chain)
    if "gpu" not in out["chain"]:
        tok.close()
        break
    if oracle is None:
        document = tok._backend.materialized_tokenizer_json()
        oracle = tokenizers.Tokenizer.from_str(document)
    ref = lambda text: list(oracle.encode(text, add_special_tokens=False).ids)
    text = f"[{index}] " + FILL * 80
    with tok.session(session_id=f"s{index}", text=text) as session:
        out["equal"].append(list(session.ids) == ref(text))
        for delta in DELTAS:
            text += delta
            update = session.append(delta)
            out["equal"].append(list(update.all_ids) == ref(text))
    store = tok._entry_store
    out.setdefault("callback_lane", []).append(
        store is not None and isinstance(store._encoder, _native.CallbackEncoder)
    )
    del store, session, update
    refs.append(weakref.ref(tok))
    if index % 2 == 0:
        tok.close()
    del tok
gc.collect()
out["loaded"] = len(refs)
out["alive"] = sum(1 for item in refs if item() is not None)
print(json.dumps(out))
"""


def _child_environment() -> dict[str, str]:
    """The child imports the package of this tree, as the test session does."""
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[2] / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not inherited else os.pathsep.join((source, inherited))
    )
    return environment


@pytest.mark.parametrize("policy", ("experimental", "default"))
@pytest.mark.parametrize("family", FAMILIES)
def test_jit_tokenizers_with_sessions_are_released(
    request: pytest.FixtureRequest, tmp_path: Path, family: str, policy: str
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
            policy,
            str(LOADS),
        ],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=3600,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    if "unavailable" in result:
        pytest.skip(f"the JIT delivery is unavailable: {result['unavailable']}")
    if "gpu" not in result.get("chain", []):
        assert policy == "default", result
        pytest.skip(
            "the default policy does not plan the device on this build: "
            f"{result.get('chain')}"
        )
    tree_source = str(Path(__file__).resolve().parents[2] / "src")
    assert result["toktier_file"].startswith(tree_source), result
    assert result["loaded"] == LOADS, result
    assert result["callback_lane"] == [True] * LOADS, result
    assert result["equal"] == [True] * (LOADS * 4), result
    assert result["alive"] == 0, result
