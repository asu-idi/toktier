"""Importing toktier touches no accelerator, no network, no GPU probe.

Acceptance surface: ``import toktier`` (and the artifact subsystem) must
work on a machine with no GPU, no CUDA and no network, and must not pull
in torch even when torch is installed. The check runs in a subprocess
with an audit hook installed before the import, so a socket call or a
CUDA library load during import is a failure rather than a warning.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"

#: Modules that must not appear in ``sys.modules`` after the import.
FORBIDDEN_MODULES = (
    "torch",
    "ninja",
    "numpy",
    "huggingface_hub",
    "tokenizers",
    "transformers",
    "socket",
    "urllib.request",
    "ctypes",
)

PROBE = '''
import sys

blocked = []


def audit(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
        blocked.append(event)
    elif event in ("urllib.Request", "subprocess.Popen"):
        blocked.append(event)
    elif event == "ctypes.dlopen":
        name = str(args[0]).lower()
        if "cuda" in name or "nvidia" in name or "nvml" in name:
            blocked.append(event + ":" + name)


sys.addaudithook(audit)

import toktier
import toktier.artifacts
import toktier.config
import toktier.errors
import toktier.paths
import toktier.policy

assert toktier.API_VERSION == 1
assert toktier.RoutingPolicy.CERTIFIED.value == "certified"

loaded = sorted(name for name in sys.modules if "." not in name)
print("BLOCKED:" + ",".join(blocked))
print("MODULES:" + ",".join(sorted(sys.modules)))
'''


def run_probe() -> dict[str, set[str]]:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(REPO_ROOT),
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(SOURCE_ROOT),
            "HOME": "/nonexistent-toktier-home",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    result: dict[str, set[str]] = {}
    for line in completed.stdout.splitlines():
        key, _, value = line.partition(":")
        result[key] = {item for item in value.split(",") if item}
    return result


def test_import_is_free_of_torch_network_and_gpu_probes() -> None:
    observed = run_probe()

    assert observed["BLOCKED"] == set()
    for module in FORBIDDEN_MODULES:
        assert module not in observed["MODULES"], f"{module} was imported"
