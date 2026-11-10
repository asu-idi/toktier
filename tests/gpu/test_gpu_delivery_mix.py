"""A prebuilt-delivery tokenizer opened after a JIT kernel was loaded in the
same process answers stored-state calls through the reference.

The prebuilt delivery's native request runtime does not start in a process
where a kernel was already loaded through the legacy (JIT) host; the facade
then serves the request from its Python-owned store. On a ``[gpu, fast_cpu,
hf]`` family that store asked the fast CPU backend for the callback-lane
repair components, whose engine had been built for the native runtime
without the live tokenizer object, and the first stored-state call raised an
internal ``RuntimeError`` naming neither cause nor remedy. It now records a
typed refusal (``session_repair`` reads ``reference_fallback`` with the
reason) and the call is answered by the reference.

The scenario runs in a child process, so the JIT kernel it loads does not
reach the other GPU tests of this session. Marked ``gpu``; skipped without
torch, a CUDA device, a JIT toolchain or the prebuilt delivery. Artifacts as
in ``test_gpu_state_nfc_closure.py``. The file stays ASCII.
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

FAMILIES = ("llama_3_1_8b", "qwen3_8b")

_CHILD = r"""
import json, sys
import tokenizers
import toktier
from toktier.config import Config
home, store = sys.argv[1], sys.argv[2]
config = Config(home=home, offline=True)
out = {"toktier_file": toktier.__file__}
try:
    first = toktier.load("llama_3_1_8b", config=config, device="cuda",
                         gpu_delivery="jit", gpu_min_bytes=256, policy="experimental")
    first.encode("warm up the kernel " * 64, lookup="off")
    out["jit_backend"] = first.explain()["runtime_policy"]["last_execution"][
        "executed_backend"]
except Exception as error:
    out["jit_unavailable"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out)); sys.exit(0)
try:
    tok = toktier.load("qwen3_8b", config=config, device="cuda",
                       gpu_delivery="prebuilt", gpu_min_bytes=256,
                       policy="experimental")
except Exception as error:
    out["prebuilt_unavailable"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out)); sys.exit(0)
out["chain"] = list(tok.plan.fallback_chain)
oracle = tokenizers.Tokenizer.from_str(tok._backend.materialized_tokenizer_json())
ref = lambda text: list(oracle.encode(text, add_special_tokens=False).ids)
long = "The committee met on Tuesday and agreed to the plan. " * 40
calls = []
def call(name, action):
    try:
        got, want = action()
        calls.append({"call": name, "equal": got == want})
    except Exception as error:
        calls.append({"call": name, "error": f"{type(error).__name__}: {error}"})
call("encode hello", lambda: (list(tok.encode("hello").ids), ref("hello")))
call("encode twice", lambda: (list(tok.encode(long + "x").ids), ref(long + "x")))
def session():
    with tok.session(session_id="mix", text=long) as s:
        s.append(" one more")
        return list(s.ids), ref(long + " one more")
call("session append", session)
persistent = toktier.load("qwen3_8b", config=config, device="cuda",
                          gpu_delivery="prebuilt", gpu_min_bytes=256,
                          policy="experimental", store=store)
def persisted():
    with persistent.session(session_id="mix", text=long) as s:
        s.append(" one more")
        return list(s.ids), ref(long + " one more")
call("persistent session append", persisted)
out["calls"] = calls
out["session_repair"] = tok.explain().get("session_repair")
print(json.dumps(out, default=str))
"""


def test_prebuilt_after_jit_answers_stored_state_through_the_reference(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    _cuda_or_skip()
    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.config import Config
    from toktier.paths import artifact_cache_dir

    config = Config(home=tmp_path / "home", offline=True)
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    for family in FAMILIES:
        target = artifact_cache_dir(config) / manifest.get(family).directory_name
        target.mkdir(parents=True)
        for name, source in _artifact_source(request, family).items():
            shutil.copyfile(source, target / name)
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tmp_path / "home"), str(tmp_path / "store")],
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
    if "jit_unavailable" in result:
        pytest.skip(f"the JIT delivery is unavailable: {result['jit_unavailable']}")
    if "prebuilt_unavailable" in result:
        pytest.skip(
            f"the prebuilt delivery is unavailable: {result['prebuilt_unavailable']}"
        )
    assert result["jit_backend"] == "gpu", result
    assert result["chain"] == ["gpu", "fast_cpu", "hf"], result
    assert result["calls"], result
    for row in result["calls"]:
        assert "error" not in row, row
        assert row["equal"], row
    repair = result["session_repair"]
    assert repair["status"] == "reference_fallback", repair
    assert "native request runtime" in repair["message"], repair
