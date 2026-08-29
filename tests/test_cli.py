"""Command-line interface acceptance tests.

Two kinds of artifact test live here, and the difference matters:

* the **shipped-manifest** tests run the commands exactly as an
  installed wheel does -- ``cli._artifact_manifest()`` is not replaced,
  so they fail if the manifest the package ships cannot resolve a
  family;
* the **synthetic-manifest** tests replace the manifest and the source
  with a small pair so that the fetch and mismatch paths can be driven
  with bytes a test can produce. They test the plumbing, not the
  configuration a user gets.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import NoReturn

import pytest

from toktier import __version__, cli
from toktier.artifacts import (
    ArtifactEntry,
    ArtifactFile,
    ArtifactManifest,
    HuggingFaceSource,
)
from toktier.engine.gpu.toolchain import JIT_TOOLCHAIN_CONSTRAINT, NvccFacts
from toktier.errors import BackendUnavailable

FAMILY = "demo_family"
REVISION = "a" * 40
GOOD = b'{"version": "1.0", "model": {}}\n'
BAD = b"corrupted bytes\n"

_DOCTOR_DEVICES = (
    (0, "NVIDIA Test GPU", "sm_120"),
    (1, "NVIDIA Test GPU 2", "sm_90"),
)
_DOCTOR_DRIVER_VERSION = "595.84"
#: (torch distribution version, torch runtime CUDA). Together with the
#: fixture's NVCC 13.0 this is one of the judged JIT triples.
_JUDGED_TORCH_FACTS = ("2.13.0+cu130", "13.0")


class StaticSource:
    """Serve one payload without reaching a network client."""

    name = "test"
    offline = False

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def fetch(
        self,
        entry: ArtifactEntry,
        artifact_file: ArtifactFile,
        destination: Path,
    ) -> None:
        del entry
        self.calls.append(artifact_file.name)
        destination.write_bytes(self.payload)


def _manifest() -> ArtifactManifest:
    return ArtifactManifest.from_mapping(
        {
            FAMILY: {
                "repo_id": "demo/demo",
                "revision": REVISION,
                "files": {
                    "tokenizer.json": {
                        "sha256": hashlib.sha256(GOOD).hexdigest(),
                        "size": len(GOOD),
                    }
                },
            }
        },
        source="<cli-test>",
    )


def _set_doctor_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    from toktier.backends import fast_cpu
    from toktier.backends.fast_cpu import FastCpuEngineFacts
    from toktier.engine.gpu import native as native_gpu
    from toktier.engine.gpu.native import NativeHostBuildFacts
    from toktier.repair import fastokens

    _set_doctor_device_probe(monkeypatch)

    def find_spec(name: str) -> object | None:
        assert name in {"torch", "cuda", "transformers", "ninja"}
        return object() if name in {"torch", "transformers"} else None

    def which(name: str) -> str:
        assert name == "nvcc"
        return "/opt/cuda/bin/nvcc"

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "nvcc: NVIDIA (R) Cuda compiler driver\n"
                "Cuda compilation tools, release 13.0, V13.0.88\n"
            ),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        fast_cpu,
        "fast_cpu_engine_facts",
        lambda: FastCpuEngineFacts(
            version="0.10.0+toktier.pinned.1",
            source_digest="f" * 64,
            build_flags=("profile=release", "opt-level=3"),
            toolchain="rustc 1.93.1 (test fixture)",
            config_digest="e" * 64,
        ),
    )
    monkeypatch.setattr(
        native_gpu,
        "native_host_build_facts",
        lambda: NativeHostBuildFacts(
            source_digest="a" * 64,
            build_flags=("profile=release", "opt-level=3"),
            toolchain="rustc 1.93.1 (test fixture)",
        ),
    )
    # The adapter resolves the engine by its import package; the doctor
    # fixture stands in for an environment where the upstream distribution
    # owns the bytes on disk and the registry lists no published wheel.
    def identity() -> fastokens.FastokensIdentity:
        owner = fastokens.DistributionOwner(
            name="fastokens",
            version=importlib.metadata.version("fastokens"),
            recorded=4,
            matching=4,
            missing=0,
            package_dir=Path("/site-packages/fastokens"),
        )
        return fastokens.FastokensIdentity(
            package_dir=Path("/site-packages/fastokens"),
            engine_digest="d" * 64,
            owners=(owner,),
            owner=owner,
        )

    monkeypatch.setattr(fastokens, "fastokens_identity", identity)
    monkeypatch.setattr(fastokens, "pinned_engine_entry", lambda: None)
    # The JIT judgement asks the installed torch for its two version
    # facts and asks the build system which compiler it would select.
    # Both are supplied here so the outcome does not depend on whether
    # the test machine happens to have torch installed.
    monkeypatch.setattr(cli, "_torch_build_facts", lambda: _JUDGED_TORCH_FACTS)
    monkeypatch.setattr(cli, "_jit_nvcc_report", cli._nvcc_report)
    # The nvcc search consults the loader's toolkit roots first; unset
    # them so the deterministic outcome is the PATH lookup above.
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)


def _set_doctor_device_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    devices: tuple[tuple[int, str, str], ...] = _DOCTOR_DEVICES,
    driver_version: str | None = _DOCTOR_DRIVER_VERSION,
) -> None:
    """Supply host facts without touching the test machine's accelerator."""
    from toktier.engine.gpu import host_probe
    from toktier.routing.probe import DeviceInfo

    observed = tuple(
        DeviceInfo(index=index, name=name, architecture=architecture)
        for index, name, architecture in devices
    )

    class StaticCudaHostProbe:
        def __init__(self, *, config: object, delivery: str) -> None:
            del config
            assert delivery in {"prebuilt", "jit"}

        def devices(self) -> tuple[DeviceInfo, ...]:
            return observed

        def driver_version(self) -> str | None:
            return driver_version

        def kernel_cache(self) -> NoReturn:
            raise AssertionError("doctor must not inspect or load a kernel")

    monkeypatch.setattr(host_probe, "CudaHostProbe", StaticCudaHostProbe)


_NVCC_CHECKED_VIA_PATH = [
    "CUDA_HOME: not set",
    "CUDA_PATH: not set",
    "PATH: /opt/cuda/bin/nvcc (found)",
]


def _shipped_prebuilt_digest() -> str:
    """Digest of the fatbin shipped in this source tree.

    The doctor probe reports the real package data, so the expectation
    is computed from the same shipped bytes rather than monkeypatched.
    """
    from toktier.kernels.prebuilt import fatbin_digest, fatbin_path

    return fatbin_digest(fatbin_path().read_bytes())


def _set_artifact_source(monkeypatch: pytest.MonkeyPatch, source: StaticSource) -> None:
    """Install the synthetic manifest and source (see the module docstring)."""
    manifest = _manifest()
    monkeypatch.setattr(cli, "_artifact_manifest", lambda: manifest)
    monkeypatch.setattr(cli, "HuggingFaceSource", lambda: source)


def _smallest_shipped_family() -> tuple[str, int]:
    """Family of the shipped manifest with the fewest bytes to hash."""
    manifest = cli._artifact_manifest()
    sizes = {
        family: sum(item.size or 0 for item in manifest.get(family).files)
        for family in manifest.families()
    }
    family = min(sizes, key=lambda name: (sizes[name], name))
    return family, sizes[family]


def test_doctor_human(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)

    exit_code = cli.main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == (
        f"python_version: {platform.python_version()}\n"
        "toktier_version: 1.2.3\n"
        "family: none\n"
        f"artifact_cache_dir: {home / 'cache' / 'artifacts'}\n"
        f"kernel_cache_dir: {home / 'cache' / 'kernels'}\n"
        f"store_state_dir: {home / 'state' / 'store'}\n"
        "directory_roots_usable: true\n"
        "directory_roots_problem: none\n"
        "configured_offline: true\n"
        "artifact_source: huggingface\n"
        "source_offline: false\n"
        "artifact_fetch_available: false\n"
        "tokenizers_version: 1.2.3\n"
        "transformers_version: 1.2.3\n"
        "certified_cpu_profile_ready: false\n"
        "torch_available: true\n"
        "ninja_available: false\n"
        "automatic_gpu_delivery: prebuilt\n"
        "automatic_gpu_min_bytes: 65536\n"
        "automatic_gpu_candidate: true\n"
        "automatic_routing_policy: supported\n"
        "automatic_gpu_eligible: true\n"
        "automatic_effective_backend: gpu\n"
        "jit_toolchain_satisfied: none\n"
        "jit_toolchain_observed: none\n"
        "jit_toolchain_constraint: none\n"
        "cuda_available: false "
        "(environment fact; not a certificate premise)\n"
        "cuda_hardware_present: true\n"
        "devices: 0: NVIDIA Test GPU (sm_120); "
        "1: NVIDIA Test GPU 2 (sm_90)\n"
        f"driver_version: {_DOCTOR_DRIVER_VERSION} "
        "(environment fact; not a certificate premise)\n"
        "automatic_gpu_delivery_certification: "
        "sm_120=certified; sm_90=certified\n"
        "prebuilt_fatbin_available: true\n"
        f"prebuilt_fatbin_digest: {_shipped_prebuilt_digest()}\n"
        "prebuilt_native_host_ready: true\n"
        f"prebuilt_host_source_digest: {'a' * 64}\n"
        "prebuilt_host_build_flags: profile=release; opt-level=3\n"
        "prebuilt_host_toolchain: rustc 1.93.1 (test fixture)\n"
        "gigatoken_available: true\n"
        "gigatoken_delivery: integrated\n"
        "gigatoken_module: toktier._native\n"
        "gigatoken_runtime_ready: true\n"
        "gigatoken_version: 0.10.0+toktier.pinned.1\n"
        "gigatoken_native_digest: none\n"
        f"gigatoken_source_digest: {'f' * 64}\n"
        "gigatoken_build_flags: profile=release; opt-level=3\n"
        "gigatoken_toolchain: rustc 1.93.1 (test fixture)\n"
        f"gigatoken_repair_config_digest: {'e' * 64}\n"
        "fastokens_available: true\n"
        "fastokens_distribution: fastokens\n"
        "fastokens_version: 1.2.3\n"
        f"fastokens_distribution_digest: {'d' * 64}\n"
        "fastokens_known_wheel: none\n"
        "fastokens_engine_assurance: upstream_build\n"
        "fastokens_exact_id_guarantee: false\n"
        "fastokens_policy: experimental\n"
        "fastokens_family_admitted: none\n"
        "fastokens_family_admission_reason: none\n"
        "fastokens_coinstalled: none\n"
        "fastokens_orphaned: none\n"
        "fastokens_advisory: none\n"
        "nvcc_available: true\n"
        "nvcc_path: /opt/cuda/bin/nvcc\n"
        "nvcc_resolved_path: /opt/cuda/bin/nvcc\n"
        "nvcc_release: 13.0\n"
        "nvcc_build: V13.0.88\n"
        "nvcc_error: none\n"
        f"nvcc_checked: {'; '.join(_NVCC_CHECKED_VIA_PATH)}\n"
    )
    assert captured.err == ""


def test_doctor_names_a_root_that_cannot_hold_private_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A probe that answers "what will actually run here" has to say so.

    The operation contract already refuses such a root with
    ``CONFIG_INVALID``; the diagnostic used to print the paths beneath it
    and say nothing.
    """
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("taken", encoding="utf-8")
    monkeypatch.setenv("TOKTIER_HOME", str(occupied))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)

    assert cli.main(["doctor", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["directory_roots_usable"] is False
    problem = report["directory_roots_problem"]
    assert str(occupied) in problem
    assert "is not a directory" in problem
    for name in ("artifact_cache_dir", "kernel_cache_dir", "store_state_dir"):
        assert name in problem


@pytest.mark.skipif(os.geteuid() == 0, reason="root may write anywhere")
def test_doctor_names_a_root_this_user_cannot_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    closed = tmp_path / "closed"
    closed.mkdir(mode=0o500)
    os.chmod(closed, 0o500)
    monkeypatch.setenv("TOKTIER_HOME", str(closed))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)

    assert cli.main(["doctor"]) == 0

    captured = capsys.readouterr()
    assert "directory_roots_usable: false\n" in captured.out
    assert "cannot be written by this user" in captured.out
    assert captured.err == ""


def test_doctor_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "0")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.0.0")
    _set_doctor_probes(monkeypatch)
    expected = {
        "python_version": platform.python_version(),
        "toktier_version": "2.0.0",
        "family": None,
        "artifact_cache_dir": str(home / "cache" / "artifacts"),
        "kernel_cache_dir": str(home / "cache" / "kernels"),
        "store_state_dir": str(home / "state" / "store"),
        "directory_roots_usable": True,
        "directory_roots_problem": None,
        "configured_offline": False,
        "artifact_source": "huggingface",
        "source_offline": False,
        "artifact_fetch_available": True,
        "tokenizers_version": "2.0.0",
        "transformers_version": "2.0.0",
        "certified_cpu_profile_ready": False,
        "torch_available": True,
        "ninja_available": False,
        "automatic_gpu_delivery": "prebuilt",
        "automatic_gpu_min_bytes": 65536,
        "automatic_gpu_candidate": True,
        "automatic_routing_policy": "supported",
        "automatic_gpu_eligible": True,
        "automatic_effective_backend": "gpu",
        "jit_toolchain_satisfied": None,
        "jit_toolchain_observed": None,
        "jit_toolchain_constraint": None,
        "cuda_available": False,
        "cuda_hardware_present": True,
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA Test GPU",
                "architecture": "sm_120",
            },
            {
                "index": 1,
                "name": "NVIDIA Test GPU 2",
                "architecture": "sm_90",
            },
        ],
        "driver_version": _DOCTOR_DRIVER_VERSION,
        "automatic_gpu_delivery_certification": {
            "sm_120": "certified",
            "sm_90": "certified",
        },
        "prebuilt_fatbin_available": True,
        "prebuilt_fatbin_digest": _shipped_prebuilt_digest(),
        "prebuilt_native_host_ready": True,
        "prebuilt_host_source_digest": "a" * 64,
        "prebuilt_host_build_flags": ["profile=release", "opt-level=3"],
        "prebuilt_host_toolchain": "rustc 1.93.1 (test fixture)",
        "gigatoken_available": True,
        "gigatoken_delivery": "integrated",
        "gigatoken_module": "toktier._native",
        "gigatoken_runtime_ready": True,
        "gigatoken_version": "0.10.0+toktier.pinned.1",
        "gigatoken_native_digest": None,
        "gigatoken_source_digest": "f" * 64,
        "gigatoken_build_flags": ["profile=release", "opt-level=3"],
        "gigatoken_toolchain": "rustc 1.93.1 (test fixture)",
        "gigatoken_repair_config_digest": "e" * 64,
        "fastokens_available": True,
        "fastokens_distribution": "fastokens",
        "fastokens_version": "2.0.0",
        "fastokens_distribution_digest": "d" * 64,
        "fastokens_known_wheel": None,
        "fastokens_engine_assurance": "upstream_build",
        "fastokens_exact_id_guarantee": False,
        "fastokens_policy": "experimental",
        "fastokens_family_admitted": None,
        "fastokens_family_admission_reason": None,
        "fastokens_coinstalled": None,
        "fastokens_orphaned": None,
        "fastokens_advisory": None,
        "nvcc_available": True,
        "nvcc_path": "/opt/cuda/bin/nvcc",
        "nvcc_resolved_path": "/opt/cuda/bin/nvcc",
        "nvcc_release": "13.0",
        "nvcc_build": "V13.0.88",
        "nvcc_error": None,
        "nvcc_checked": _NVCC_CHECKED_VIA_PATH,
    }

    exit_code = cli.main(["doctor", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (
        captured.out
        == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert json.loads(captured.out) == expected
    assert captured.err == ""


def test_doctor_certification_follows_the_selected_jit_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The architecture labels follow JIT, not the prebuilt row beside it."""
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)

    def find_spec(name: str) -> object | None:
        assert name in {"torch", "cuda", "transformers", "ninja"}
        return object() if name in {"torch", "transformers", "ninja"} else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["automatic_gpu_delivery"] == "jit"
    assert report["automatic_gpu_delivery_certification"] == {
        "sm_120": "certified_source",
        "sm_90": "uncertified",
    }


def _set_certified_oracle_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report the oracle pair the certified CPU profile is bound to."""
    versions = {"tokenizers": "0.22.2", "transformers": "4.57.6"}
    monkeypatch.setattr(cli, "_installed_version", versions.get)


def _select_jit_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ninja importable, which is what selects the JIT delivery."""

    def find_spec(name: str) -> object | None:
        assert name in {"torch", "cuda", "transformers", "ninja"}
        return object() if name in {"torch", "transformers", "ninja"} else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_doctor_reports_a_judged_jit_toolchain_as_eligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A judged triple: the observation is reported and the route is open."""
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)
    _select_jit_profile(monkeypatch)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["automatic_gpu_delivery"] == "jit"
    assert report["jit_toolchain_satisfied"] is True
    assert report["jit_toolchain_observed"] == (
        "NVCC 13.0 (V13.0.88; /opt/cuda/bin/nvcc) / torch CUDA 13.0 "
        "/ torch 2.13.0+cu130"
    )
    assert report["jit_toolchain_constraint"] == JIT_TOOLCHAIN_CONSTRAINT
    assert report["automatic_gpu_candidate"] is True
    assert report["automatic_gpu_eligible"] is True
    assert report["automatic_effective_backend"] == "gpu"


def _select_unjudged_jit_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """A compiler release no campaign has judged, beside a judged device."""
    monkeypatch.setattr(
        cli,
        "_jit_nvcc_report",
        lambda: NvccFacts(
            path="/usr/local/cuda/bin/nvcc",
            resolved_path="/usr/local/cuda-13.2/bin/nvcc",
            release="13.2",
            build="V13.2.86",
            error=None,
            checked=("torch CUDA_HOME: /usr/local/cuda/bin/nvcc (found)",),
        ),
    )


def test_doctor_reports_an_unjudged_jit_toolchain_as_eligible_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Architecture judged, compiler not: a coverage gap the default admits.

    ``SUPPORTED`` proceeds past an unjudged compiler/runtime pair and
    labels the route ``supported_untested``, so a request here does reach
    the GPU. The observation stays visible -- ``jit_toolchain_satisfied``
    is still ``false`` -- while the conclusion follows the plan the next
    request will get.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)
    _select_jit_profile(monkeypatch)
    _set_certified_oracle_versions(monkeypatch)
    _select_unjudged_jit_compiler(monkeypatch)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["automatic_gpu_delivery"] == "jit"
    assert report["automatic_gpu_delivery_certification"]["sm_120"] == (
        "certified_source"
    )
    assert report["automatic_gpu_candidate"] is True
    assert report["jit_toolchain_satisfied"] is False
    assert report["jit_toolchain_observed"] == (
        "NVCC 13.2 (V13.2.86; /usr/local/cuda/bin/nvcc) / torch CUDA 13.0 "
        "/ torch 2.13.0+cu130"
    )
    assert report["jit_toolchain_constraint"] == JIT_TOOLCHAIN_CONSTRAINT
    assert report["automatic_gpu_eligible"] is True
    assert report["automatic_effective_backend"] == "gpu"


def test_doctor_reports_an_unjudged_jit_toolchain_as_ineligible_under_certified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same machine under ``CERTIFIED``: the compiler premise refuses.

    This is the other half of the same rule. ``CERTIFIED`` has always
    refused a coverage gap, so the report has to say ``fast_cpu`` here,
    and the two answers have to differ only because the policy differs.
    """
    home = tmp_path / "toktier-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(
        "routing_policy = 'certified'\n", encoding="utf-8"
    )
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)
    _select_jit_profile(monkeypatch)
    _set_certified_oracle_versions(monkeypatch)
    _select_unjudged_jit_compiler(monkeypatch)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["automatic_gpu_candidate"] is True
    assert report["jit_toolchain_satisfied"] is False
    assert report["automatic_gpu_eligible"] is False
    assert report["automatic_effective_backend"] == "fast_cpu"


def test_doctor_reports_an_unjudged_architecture_the_way_the_policy_reads_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The other coverage gap: a device no campaign judged.

    The installation-level and the family-level answers both follow the
    policy in effect. Under the default the shipped kernel runs there and
    the route is labelled rather than refused; under ``CERTIFIED`` the
    same machine reads ``fast_cpu``.
    """
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    certified = {"tokenizers": "0.22.2", "transformers": "4.57.6"}
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: certified.get(name, "1.2.3")
    )
    _set_doctor_probes(monkeypatch)
    _set_doctor_device_probe(
        monkeypatch, devices=((0, "Unjudged device", "sm_61"),)
    )

    assert cli.main(["doctor", "--json", "--family", "qwen3_8b"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["automatic_gpu_delivery_certification"] == {
        "sm_61": "uncertified"
    }
    assert report["automatic_gpu_eligible"] is True
    assert report["automatic_effective_backend"] == "gpu"
    assert report["family"]["automatic_gpu_eligible"] is True
    assert report["family"]["automatic_effective_backend"] == "gpu"

    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(
        "routing_policy = 'certified'\n", encoding="utf-8"
    )

    assert cli.main(["doctor", "--json", "--family", "qwen3_8b"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["automatic_gpu_eligible"] is False
    assert report["automatic_effective_backend"] == "fast_cpu"
    assert report["family"]["automatic_gpu_eligible"] is False
    assert report["family"]["automatic_effective_backend"] == "fast_cpu"


def test_doctor_answers_under_the_reference_policy_the_way_the_plan_does(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A third premise, and the one this report used to leave out.

    ``REFERENCE`` is not a coverage question. Check 1 of the plan refuses
    every accelerated backend under it, unwaivably, so the same machine
    that reads ``gpu`` under the default has to read ``hf`` here -- at
    the installation level and for a named family -- and the report has
    to say which policy it applied.
    """
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    certified = {"tokenizers": "0.22.2", "transformers": "4.57.6"}
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: certified.get(name, "1.2.3")
    )
    _set_doctor_probes(monkeypatch)

    assert cli.main(["doctor", "--json", "--family", "qwen3_8b"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["automatic_routing_policy"] == "supported"
    assert report["automatic_gpu_eligible"] is True
    assert report["automatic_effective_backend"] == "gpu"

    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(
        "routing_policy = 'reference'\n", encoding="utf-8"
    )

    assert cli.main(["doctor", "--json", "--family", "qwen3_8b"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["automatic_routing_policy"] == "reference"
    # The machine did not change: torch is still installed and the
    # devices are still judged. Only the policy did.
    assert report["automatic_gpu_candidate"] is True
    assert report["automatic_gpu_delivery_certification"] == {
        "sm_120": "certified",
        "sm_90": "certified",
    }
    assert report["automatic_gpu_eligible"] is False
    assert report["automatic_effective_backend"] == "hf"
    assert report["family"]["automatic_gpu_eligible"] is False
    assert report["family"]["automatic_effective_backend"] == "hf"
    assert report["family"]["automatic_effective_backend_below_gpu_threshold"] == "hf"


def test_doctor_reports_the_reference_backend_without_a_certified_cpu_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No eligible GPU and no certified CPU profile leaves the oracle."""
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)
    _set_doctor_device_probe(monkeypatch, devices=(), driver_version=None)
    from toktier.backends import fast_cpu
    from toktier.backends.fast_cpu import FastCpuEngineFacts

    monkeypatch.setattr(fast_cpu, "fast_cpu_engine_facts", FastCpuEngineFacts)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["cuda_hardware_present"] is False
    assert report["automatic_gpu_candidate"] is True
    assert report["automatic_gpu_eligible"] is False
    assert report["gigatoken_runtime_ready"] is False
    assert report["automatic_effective_backend"] == "hf"


def test_doctor_reports_a_disabled_gpu_as_ineligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``TOKTIER_DISABLE_GPU`` closes the route the report describes."""
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setenv("TOKTIER_DISABLE_GPU", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["automatic_gpu_candidate"] is False
    assert report["automatic_gpu_eligible"] is False
    # The fixture's tokenizers/transformers versions are not the
    # certified pair, so the honest CPU answer is the reference oracle.
    assert report["automatic_effective_backend"] == "hf"


def test_doctor_separates_a_configuration_from_an_offline_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The case a single ``offline`` field used to hide.

    The configuration allows fetching and the hub client refuses to
    reach out, so the honest answer is that fetching is unavailable.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)

    exit_code = cli.main(["doctor", "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["configured_offline"] is False
    assert report["source_offline"] is True
    assert report["artifact_fetch_available"] is False
    assert report["artifact_source"] == "huggingface"


def test_doctor_nvcc_follows_the_build_system_search_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A toolkit under ``CUDA_HOME`` is found without ``nvcc`` on PATH.

    The JIT loader builds through ``torch.utils.cpp_extension``, which
    consults ``CUDA_HOME`` before the ``PATH``; the doctor answers the
    same question the same way.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    cuda_home = tmp_path / "cuda-home"
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n")
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    _set_doctor_device_probe(monkeypatch, devices=(), driver_version=None)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["nvcc_available"] is True
    assert report["nvcc_path"] == str(nvcc)
    assert report["nvcc_checked"] == [f"CUDA_HOME: {nvcc} (found)"]
    assert report["cuda_hardware_present"] is False
    assert report["devices"] == []
    assert report["driver_version"] is None
    assert report["automatic_gpu_delivery_certification"] == {}


def test_doctor_treats_a_set_cuda_home_as_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``CUDA_HOME`` without ``nvcc`` stops the search, as the build does.

    ``torch.utils.cpp_extension`` takes a set ``CUDA_HOME`` as the
    toolkit root without falling back to the ``PATH``, so reporting the
    ``PATH`` copy as available here would promise a build that the
    loader would not attempt with these settings.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvcc")
    empty_home = tmp_path / "cuda-home-empty"
    empty_home.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(empty_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    _set_doctor_device_probe(monkeypatch, devices=(), driver_version=None)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["nvcc_available"] is False
    assert report["nvcc_path"] is None
    assert report["nvcc_checked"] == [
        f"CUDA_HOME: {empty_home / 'bin' / 'nvcc'} (not found)"
    ]


# -- shipped manifest (nothing replaced) -------------------------------


def test_artifacts_verify_resolves_a_shipped_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``verify`` reaches the cache lookup for a family the package ships.

    With an empty cache the command still fails, but on the missing
    bytes rather than on the family: an empty manifest would refuse
    every family here, which is the failure this test exists for. The
    sentence also has to say enough to act on -- which directory was
    searched, which offline condition is in force, and a way forward --
    because the human form has no ``details`` to fall back on.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    family, _ = _smallest_shipped_family()
    searched = (
        tmp_path / "toktier-home" / "cache" / "artifacts"
    )

    exit_code = cli.main(["artifacts", "verify", family])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    message = captured.err
    assert message.startswith(
        "error ARTIFACT_NOT_FOUND: artifact file 'tokenizer.json' of "
        f"{family!r} is not in the cache at {searched}"
    )
    assert "fetching is disabled (offline: no_source)" in message
    assert f"toktier artifacts fetch {family}" in message
    assert "toktier artifacts import <bundle>" in message


def test_artifacts_fetch_refuses_a_family_the_package_does_not_ship(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown family fails on the name, before any source is asked."""
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))

    exit_code = cli.main(["artifacts", "fetch", "no_such_family"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == (
        "error ARTIFACT_NOT_FOUND: unknown tokenizer family 'no_such_family'\n"
    )


# -- synthetic manifest (plumbing) -------------------------------------


def test_artifacts_fetch_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    source = StaticSource(GOOD)
    _set_artifact_source(monkeypatch, source)
    directory = home / "cache" / "artifacts" / f"{FAMILY}-{REVISION[:12]}"

    exit_code = cli.main(["artifacts", "fetch", FAMILY, "--force"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"fetched {FAMILY}: {directory}\n"
    assert captured.err == ""
    assert source.calls == ["tokenizer.json"]
    assert (directory / "tokenizer.json").read_bytes() == GOOD


def test_artifacts_fetch_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    source = StaticSource(BAD)
    _set_artifact_source(monkeypatch, source)

    exit_code = cli.main(["artifacts", "fetch", FAMILY])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == (
        "error ARTIFACT_HASH_MISMATCH: content hash mismatch for "
        f"{FAMILY}/tokenizer.json\n"
    )
    assert source.calls == ["tokenizer.json", "tokenizer.json"]


def test_artifacts_fetch_reports_a_gated_repository_as_a_command_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The contract holds when the download client raises.

    This is the real fetch path -- the hub source inside the converting
    source the command builds -- with only the client call replaced, so
    no test reaches a network. A repository that needs credentials is
    the ordinary case: it must exit ``2`` with one report, in prose and
    in JSON alike, rather than print a traceback and exit ``1``.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))

    class GatedRepoError(Exception):
        """Stand-in for the client's own exception type."""

    def fetcher(
        *, repo_id: str, filename: str, revision: str, local_dir: str
    ) -> NoReturn:
        del local_dir
        raise GatedRepoError(
            "401 Client Error.\n\nCannot access gated repo for url "
            f"https://example.invalid/{repo_id}/resolve/{revision}/{filename}.\n"
            "Access is restricted. Please log in."
        )

    manifest = _manifest()
    monkeypatch.setattr(cli, "_artifact_manifest", lambda: manifest)
    monkeypatch.setattr(
        cli, "HuggingFaceSource", lambda: HuggingFaceSource(fetcher=fetcher)
    )

    exit_code = cli.main(["artifacts", "fetch", FAMILY])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    # One line, the frozen prose shape, naming the family and the client.
    assert captured.err.count("\n") == 1
    assert captured.err.startswith("error ARTIFACT_NOT_FOUND: ")
    assert "GatedRepoError" in captured.err
    assert "Traceback" not in captured.err

    exit_code = cli.main(["--json", "artifacts", "fetch", FAMILY])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "ARTIFACT_NOT_FOUND"
    assert error["details"]["family"] == FAMILY
    assert error["details"]["cause"] == "GatedRepoError"
    assert error["details"]["remedy"]
    # The flag is a promise about the command, wherever it is written.
    assert cli.main(["artifacts", "fetch", FAMILY, "--json"]) == 2
    assert json.loads(capsys.readouterr().err) == {"error": error}


def test_artifacts_export_then_import_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The air-gap recipe of the README, end to end.

    Fetch on one machine, export, import into a second machine's empty
    cache, then bind the installed bytes to the shipped digests with
    ``verify`` -- all through the command line, with no source configured
    on the second machine.
    """
    online_home = tmp_path / "online-home"
    offline_home = tmp_path / "offline-home"
    bundle = tmp_path / "demo.tar"
    monkeypatch.setenv("TOKTIER_HOME", str(online_home))
    source = StaticSource(GOOD)
    _set_artifact_source(monkeypatch, source)
    alias = f"{FAMILY}-{REVISION[:12]}"

    fetched = cli.main(["artifacts", "fetch", FAMILY])
    exported = cli.main(["artifacts", "export", FAMILY, "--out", str(bundle)])
    monkeypatch.setenv("TOKTIER_HOME", str(offline_home))
    imported = cli.main(["artifacts", "import", str(bundle)])
    verified = cli.main(["artifacts", "verify", FAMILY])

    captured = capsys.readouterr()
    installed = offline_home / "cache" / "artifacts" / alias
    assert (fetched, exported, imported, verified) == (0, 0, 0, 0)
    assert captured.out == (
        f"fetched {FAMILY}: {online_home / 'cache' / 'artifacts' / alias}\n"
        f"exported {FAMILY}: {bundle}\n"
        f"imported {alias}: {installed}\n"
        f"verified {FAMILY}: {installed}\n"
    )
    assert captured.err == ""
    assert source.calls == ["tokenizer.json"]
    assert (installed / "tokenizer.json").read_bytes() == GOOD


def test_artifacts_export_refuses_an_empty_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Export verifies first; with nothing cached there is nothing to pack."""
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    _set_artifact_source(monkeypatch, StaticSource(GOOD))
    bundle = tmp_path / "demo.tar"

    exit_code = cli.main(["artifacts", "export", FAMILY, "--out", str(bundle)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "ARTIFACT_NOT_FOUND" in captured.err
    assert not bundle.exists()


def test_artifacts_import_rejects_a_file_that_is_not_a_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    bundle = tmp_path / "not-a-bundle.tar"
    bundle.write_bytes(b"these bytes are not a tar archive\n")

    exit_code = cli.main(["artifacts", "import", str(bundle)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error BUNDLE_INVALID: ")
    # ``errors.md`` Section 4: without --json the report is one line. The
    # tar reader says what it tried one method per line; that belongs in
    # the envelope, not on standard error.
    assert captured.err.splitlines() == [captured.err.rstrip("\n")]
    cache = home / "cache" / "artifacts"
    assert not cache.exists() or not any(
        path for path in cache.iterdir() if not path.name.startswith(".")
    )


def test_artifacts_import_json_carries_what_the_tar_reader_tried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The detail folded out of the prose line is still delivered."""
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    bundle = tmp_path / "not-a-bundle.tar"
    bundle.write_bytes(b"these bytes are not a tar archive\n")

    exit_code = cli.main(["artifacts", "import", "--json", str(bundle)])

    captured = capsys.readouterr()
    assert exit_code == 2
    envelope = json.loads(captured.err)
    assert envelope["error"]["code"] == "BUNDLE_INVALID"
    assert "\n" not in envelope["error"]["message"]
    cause = envelope["error"]["details"]["cause"]
    assert "method gz" in cause and "method tar" in cause


def test_inspect_prints_one_family(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_artifact_source(monkeypatch, StaticSource(GOOD))
    digest = hashlib.sha256(GOOD).hexdigest()

    exit_code = cli.main(["inspect", FAMILY])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == (
        f"family: {FAMILY}\n"
        "repo_id: demo/demo\n"
        f"revision: {REVISION}\n"
        f"tokenizer.json: sha256 {digest} ({len(GOOD)} bytes)\n"
    )
    assert captured.err == ""


def test_inspect_json_matches_the_shipped_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``inspect --json`` over the manifest the package actually ships."""
    manifest = cli._artifact_manifest()

    exit_code = cli.main(["inspect", "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert sorted(report) == sorted(manifest.families())
    for family, block in report.items():
        entry = manifest.get(family)
        assert block["repo_id"] == entry.repo_id
        assert block["revision"] == entry.revision
        assert block["files"] == {
            item.name: {"sha256": item.sha256, "size": item.size}
            for item in entry.files
        }


def test_inspect_refuses_an_unknown_family(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["inspect", "no_such_family"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == (
        "error ARTIFACT_NOT_FOUND: unknown tokenizer family 'no_such_family'\n"
    )


def test_version_uses_package_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing_version(name: str) -> NoReturn:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_version)

    exit_code = cli.main(["version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"{__version__}\n"
    assert captured.err == ""


class _FakeJitTokenizer:
    def __init__(self, *, waivers: list[dict[str, object]] | None = None) -> None:
        self.encoded: list[tuple[str, str]] = []
        self.closed = False
        self._waivers = waivers or []

    def encode(self, text: str, *, lookup: str) -> object:
        self.encoded.append((text, lookup))
        return object()

    def explain(self) -> dict[str, object]:
        return {
            "kernel_delivery": "jit",
            "gpu_backend": {"loaded": True},
            "runtime_policy": {"last_execution": {"executed_backend": "gpu"}},
            "experimental_waivers": self._waivers,
        }

    def close(self) -> None:
        self.closed = True


def test_gpu_compile_follows_the_configured_policy_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The compile command asks for whatever this machine resolves.

    Since 0.2.6 that is ``SUPPORTED``, so an unjudged compiler pair
    compiles and runs here and the report says which label it earned
    rather than refusing first and offering an opt-in.
    """
    from toktier import facade

    calls: list[tuple[str, dict[str, object]]] = []
    tokenizer = _FakeJitTokenizer()

    def fake_load(family: str, **keywords: object) -> _FakeJitTokenizer:
        calls.append((family, keywords))
        return tokenizer

    monkeypatch.setattr(facade, "load", fake_load)

    exit_code = cli.main(["gpu", "compile", "qwen3_8b", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [
        (
            "qwen3_8b",
            {
                "device": "cuda",
                "policy": "supported",
                "gpu_delivery": "jit",
                "gpu_min_bytes": 0,
            },
        )
    ]
    assert tokenizer.encoded == [("TokTier JIT compile probe", "off")]
    assert tokenizer.closed is True
    assert json.loads(captured.out) == {
        "accepted_uncertified_jit": False,
        "certification_state": None,
        "experimental_waivers": [],
        "family": "qwen3_8b",
        "jit_ready": True,
        "kernel_delivery": "jit",
        "policy": "supported",
        "requested_uncertified_jit_opt_in": False,
        "supported_untested": [],
        "warning": None,
    }
    assert captured.err == ""


def _pin_certified_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the strict policy, the way a configuration file can.

    ``--accept-uncertified-jit`` is what remains for the policies that
    still refuse a combination nobody judged, so the tests that exercise
    the flag have to be under one of them.
    """
    from toktier.config import Config
    from toktier.policy import RoutingPolicy

    resolved = Config.resolve()
    monkeypatch.setattr(
        Config,
        "resolve",
        classmethod(
            lambda _cls, **_keywords: dataclasses.replace(
                resolved, routing_policy=RoutingPolicy.CERTIFIED
            )
        ),
    )


def test_gpu_compile_requires_a_loud_explicit_opt_in_for_unjudged_jit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from toktier import facade

    _pin_certified_policy(monkeypatch)

    waiver: dict[str, object] = {
        "backend": "gpu",
        "code": "R_UNCERTIFIED_ARTIFACT",
        "detail": {
            "cause": "toolchain_unverified",
            "observed": "CUDA 13.0 / torch 2.11.0+cu130",
        },
    }
    calls: list[tuple[str, dict[str, object]]] = []
    tokenizer = _FakeJitTokenizer(waivers=[waiver])

    def fake_load(family: str, **keywords: object) -> _FakeJitTokenizer:
        calls.append((family, keywords))
        if keywords["policy"] == "certified":
            raise BackendUnavailable(
                "unjudged JIT toolchain",
                details={"reason": waiver["detail"]},
            )
        return tokenizer

    monkeypatch.setattr(facade, "load", fake_load)

    exit_code = cli.main(
        [
            "gpu",
            "compile",
            "qwen3_8b",
            "--accept-uncertified-jit",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert [call[1]["policy"] for call in calls] == ["certified", "experimental"]
    report = json.loads(captured.out)
    assert report["accepted_uncertified_jit"] is True
    assert report["requested_uncertified_jit_opt_in"] is True
    assert report["policy"] == "experimental"
    assert report["experimental_waivers"] == [waiver]
    assert "UNCERTIFIED JIT OPT-IN" in captured.err
    assert "outside TokTier's certified exact-ID guarantee" in captured.err


def test_gpu_compile_risk_flag_cannot_waive_a_different_premise(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from toktier import facade

    calls: list[dict[str, object]] = []

    def fake_load(_family: str, **keywords: object) -> _FakeJitTokenizer:
        calls.append(keywords)
        raise BackendUnavailable(
            "uncertified architecture",
            details={
                "reason": {
                    "cause": "architecture_unverified",
                    "observed": "sm_130",
                }
            },
        )

    monkeypatch.setattr(facade, "load", fake_load)
    _pin_certified_policy(monkeypatch)

    exit_code = cli.main(["gpu", "compile", "qwen3_8b", "--accept-uncertified-jit"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert [call["policy"] for call in calls] == ["certified"]
    assert captured.out == ""
    assert "error BACKEND_UNAVAILABLE: uncertified architecture" in captured.err
    assert "UNCERTIFIED JIT OPT-IN" not in captured.err


class _FakeVerifyTokenizer:
    """A tokenizer stand-in for the local verification command."""

    def __init__(self, ids: dict[str, tuple[int, ...]], backend: str) -> None:
        self._ids = ids
        self._backend = backend
        self.closed = False

    def encode(self, text: str, *, lookup: str | None = None) -> object:
        import dataclasses as _dataclasses

        return _dataclasses.make_dataclass("E", ["ids"])(
            self._ids.get(text, (1, 2, 3))
        )

    def explain(self, *, summary: bool = False) -> dict[str, object]:
        return {"last_execution_backend": self._backend}

    def verification_key(self, engine: str) -> object:
        from toktier.verify_local import VerificationKey

        return VerificationKey(
            engine=engine,
            family="qwen3_8b",
            artifact_sha256="a" * 64,
            architecture="sm_100" if engine == "gpu" else None,
            delivery="prebuilt" if engine == "gpu" else None,
        )

    def close(self) -> None:
        self.closed = True


def _verify_loader(
    monkeypatch: pytest.MonkeyPatch, *, subject_backend: str
) -> None:
    from toktier import facade

    def fake_load(_family: str, **keywords: object) -> _FakeVerifyTokenizer:
        reference = keywords.get("policy") == "reference"
        return _FakeVerifyTokenizer(
            {}, "hf" if reference else subject_backend
        )

    monkeypatch.setattr(facade, "load", fake_load)


def test_verify_local_records_a_local_check_that_agreed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command the supported_untested label points at."""
    _verify_loader(monkeypatch, subject_backend="gpu")

    exit_code = cli.main(
        [
            "verify-local",
            "--family",
            "qwen3_8b",
            "--engine",
            "gpu",
            "--synthetic",
            "4",
            "--max-bytes",
            "128",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["documents"] == 4
    assert payload["input"] == "generated"
    entry = payload["engines"][0]
    assert entry["engine"] == "gpu"
    assert entry["status"] == "passed"
    assert entry["mismatches"] == 0
    assert entry["served_by_engine"] == 4
    # The record is on disk and reads back under its own key.
    assert entry["record_readable"] is True
    assert Path(entry["record_path"]).exists()


def test_verify_local_reports_a_disagreement_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from toktier import facade

    def fake_load(_family: str, **keywords: object) -> _FakeVerifyTokenizer:
        reference = keywords.get("policy") == "reference"
        return _FakeVerifyTokenizer(
            {} if reference else {"x": (9, 9)}, "hf" if reference else "gpu"
        )

    monkeypatch.setattr(facade, "load", fake_load)
    monkeypatch.setattr(
        cli, "_verify_documents", lambda _arguments: (["x"], "your text")
    )

    exit_code = cli.main(
        ["verify-local", "--family", "qwen3_8b", "--engine", "gpu"]
    )

    captured = capsys.readouterr()
    # A disagreement is a non-zero exit and a sentence, not a policy
    # change: the route keeps the label it already had.
    assert exit_code == 2
    assert "local verification failed on 1 of 1 documents" in captured.out
    assert "Nothing was changed automatically." in captured.out
    assert "certified" not in captured.out.replace("policy='certified'", "")


def test_verify_local_records_nothing_when_the_route_never_ran(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Agreement over documents the engine never served is not a pass."""
    from toktier import facade

    def fake_load(_family: str, **keywords: object) -> _FakeVerifyTokenizer:
        # Both sides answer as the reference engine: the route was
        # planned and then fell back for every document.
        return _FakeVerifyTokenizer({}, "hf")

    monkeypatch.setattr(facade, "load", fake_load)

    exit_code = cli.main(
        [
            "verify-local",
            "--family",
            "qwen3_8b",
            "--engine",
            "gpu",
            "--synthetic",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "was admitted and served none of the 3 documents" in captured.out
    assert "measured nothing about it and no record was written" in captured.out
    # The fake reports no execution path, and the sentence says so
    # rather than inventing one.
    assert "per-input reason (no path was recorded)" in captured.out
    assert "`explain()` on a tokenizer for the same input" in captured.out
    assert "says why the route did not run" not in captured.out
    assert "locally_verified" not in captured.out


def test_verify_local_points_a_route_the_plan_did_not_admit_at_doctor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zero coverage has two causes and the report says which.

    A route the plan did not admit is a fact about the plan and
    ``doctor`` explains it; a route the plan admitted and every document
    left again is a per-input matter, and pointing that reader at
    ``doctor`` sent them to an answer about something else. The first
    state is recognised from the plan's fallback chain.
    """
    from toktier import facade

    class _NotAdmitted(_FakeVerifyTokenizer):
        def __init__(self) -> None:
            super().__init__({}, "hf")

        def explain(self, *, summary: bool = False) -> dict[str, object]:
            if summary:
                return {
                    "last_execution_backend": "hf",
                    "last_execution_path": "hf_full",
                }
            return {"fallback_chain": ["hf"]}

    class _AddedTokens(_FakeVerifyTokenizer):
        def __init__(self) -> None:
            super().__init__({}, "hf")

        def explain(self, *, summary: bool = False) -> dict[str, object]:
            if summary:
                return {
                    "last_execution_backend": "hf",
                    "last_execution_path": "hf_added_token",
                }
            return {"fallback_chain": ["fast_cpu", "hf"]}

    subjects: list[_FakeVerifyTokenizer] = [_NotAdmitted(), _AddedTokens()]

    def fake_load(_family: str, **keywords: object) -> _FakeVerifyTokenizer:
        if keywords.get("policy") == "reference":
            return _FakeVerifyTokenizer({}, "hf")
        return subjects.pop(0)

    monkeypatch.setattr(facade, "load", fake_load)
    arguments = ["verify-local", "--family", "qwen3_8b", "--engine", "cpu"]

    assert cli.main([*arguments, "--synthetic", "2"]) == 0
    not_admitted = capsys.readouterr().out
    assert "the plan did not admit the cpu route" in not_admitted
    assert "served none of the 2 documents" in not_admitted
    assert (
        "`toktier doctor --family <family>` reports the plan's own reasons"
        in not_admitted
    )
    assert "explain()" not in not_admitted

    assert cli.main([*arguments, "--synthetic", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    (item,) = payload["engines"]
    assert item["status"] == "not_measured"
    assert item["route_admitted"] is True
    assert item["unserved_paths"] == [{"path": "hf_added_token", "documents": 2}]


def test_verify_local_says_what_a_partly_served_run_did_compare(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run the route served in part measured that part, and says so.

    Calling it "measured nothing" was the reading an evaluation
    objected to: those documents were compared and they agreed. What
    the run lacks is coverage, so it still writes no record.
    """
    from toktier import facade

    class _PartlyServed(_FakeVerifyTokenizer):
        """A subject that takes the accelerated route every other time."""

        def __init__(self) -> None:
            super().__init__({}, "fast_cpu")
            self._calls = 0

        def explain(self, *, summary: bool = False) -> dict[str, object]:
            self._calls += 1
            backend = "fast_cpu" if self._calls == 1 else "hf"
            return {"last_execution_backend": backend}

    def fake_load(_family: str, **keywords: object) -> _FakeVerifyTokenizer:
        if keywords.get("policy") == "reference":
            return _FakeVerifyTokenizer({}, "hf")
        return _PartlyServed()

    monkeypatch.setattr(facade, "load", fake_load)

    exit_code = cli.main(
        ["verify-local", "--family", "qwen3_8b", "--engine", "cpu", "--synthetic", "3"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "served 1 of 3 documents" in captured.out
    assert "the served ones compared equal" in captured.out
    assert "no record was written" in captured.out
    for word in ("measured nothing", "locally_verified", "toktier doctor"):
        assert word not in captured.out


def test_verify_local_names_an_engine_this_machine_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--engine both`` on a machine with no GPU still checks the CPU."""
    from toktier import facade

    def fake_load(_family: str, **keywords: object) -> _FakeVerifyTokenizer:
        if keywords.get("device") == "cuda":
            raise BackendUnavailable(
                "no usable device", details={"backend": "gpu"}
            )
        reference = keywords.get("policy") == "reference"
        return _FakeVerifyTokenizer({}, "hf" if reference else "fast_cpu")

    monkeypatch.setattr(facade, "load", fake_load)

    exit_code = cli.main(
        ["verify-local", "--family", "qwen3_8b", "--synthetic", "2", "--json"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert [item["engine"] for item in payload["engines"]] == ["cpu"]
    assert payload["skipped"] == [
        {
            "engine": "gpu",
            "status": "not_available",
            "reason": "BACKEND_UNAVAILABLE",
            "message": "no usable device",
        }
    ]


def test_verify_local_forgets_a_record_without_encoding_anything(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _verify_loader(monkeypatch, subject_backend="gpu")
    assert (
        cli.main(
            [
                "verify-local",
                "--family",
                "qwen3_8b",
                "--engine",
                "gpu",
                "--synthetic",
                "2",
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = cli.main(
        [
            "verify-local",
            "--family",
            "qwen3_8b",
            "--engine",
            "gpu",
            "--forget",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["engines"] == [
        {"engine": "gpu", "forgot_record": True, "status": "forgotten"}
    ]
    assert payload["documents"] == 0


def test_json_failure_emits_a_machine_readable_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--json`` covers the failure path, not only the success path."""
    from toktier import facade

    def fake_load(_family: str, **_keywords: object) -> NoReturn:
        raise BackendUnavailable(
            "device='cuda' requires an eligible GPU route, but the "
            "certified planner closed it; observed NVCC 13.2 / torch CUDA "
            "13.0 / torch 2.11.0+cu130; certified constraint: judged with "
            "NVCC 13.0 / torch CUDA 13.0 / torch 2.13.0+cu130",
            details={
                "backend": "gpu",
                "reason_code": "R_UNCERTIFIED_ARTIFACT",
                "reason": {
                    "cause": "architecture_unverified",
                    "observed": "sm_130",
                },
                "remedy": "toktier gpu compile qwen3_8b --accept-uncertified-jit",
            },
        )

    monkeypatch.setattr(facade, "load", fake_load)

    exit_code = cli.main(["gpu", "compile", "qwen3_8b", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    error = payload["error"]
    assert error["code"] == "BACKEND_UNAVAILABLE"
    assert "certified constraint" in error["message"]
    assert error["details"]["backend"] == "gpu"
    assert error["details"]["reason"]["observed"] == "sm_130"
    assert error["details"]["remedy"] == (
        "toktier gpu compile qwen3_8b --accept-uncertified-jit"
    )


def test_json_error_envelope_survives_an_unserialisable_detail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An open details mapping must not be able to break the envelope."""
    from toktier import facade

    def fake_load(_family: str, **_keywords: object) -> NoReturn:
        raise BackendUnavailable(
            "closed", details={"backend": "gpu", "path": Path("/tmp/kernel")}
        )

    monkeypatch.setattr(facade, "load", fake_load)

    exit_code = cli.main(["gpu", "compile", "qwen3_8b", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.err)
    assert payload["error"]["details"]["backend"] == "gpu"
    assert "kernel" in payload["error"]["details"]["path"]


def test_plain_failure_keeps_the_human_error_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--json`` the human line is unchanged."""
    from toktier import facade

    def fake_load(_family: str, **_keywords: object) -> NoReturn:
        raise BackendUnavailable("closed", details={"backend": "gpu"})

    monkeypatch.setattr(facade, "load", fake_load)

    exit_code = cli.main(["gpu", "compile", "qwen3_8b"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == "error BACKEND_UNAVAILABLE: closed\n"


def test_the_json_flag_is_accepted_before_or_after_every_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One flag, one meaning, wherever a caller puts it."""
    monkeypatch.setattr(cli, "_toktier_version", lambda: "9.9.9")

    for argv in (["--json", "version"], ["version", "--json"]):
        assert cli.main(argv) == 0
        assert json.loads(capsys.readouterr().out) == {"version": "9.9.9"}

    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out == "9.9.9\n"


def test_the_artifact_commands_report_machine_readably(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The commands that used to reject ``--json`` now answer with it."""
    exit_code = cli.main(["--json", "artifacts", "verify", "no_such_family"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "ARTIFACT_NOT_FOUND"
    assert error["details"]["family"] == "no_such_family"

    exit_code = cli.main(["inspect", "no_such_family", "--json"])
    assert exit_code == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == (
        "ARTIFACT_NOT_FOUND"
    )


def test_the_artifact_lifecycle_reports_the_same_facts_as_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each command's object carries what its prose line carried."""
    home = tmp_path / "home"
    bundle = tmp_path / "demo.tar"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    _set_artifact_source(monkeypatch, StaticSource(GOOD))
    alias = f"{FAMILY}-{REVISION[:12]}"
    directory = home / "cache" / "artifacts" / alias

    assert cli.main(["artifacts", "fetch", FAMILY, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "action": "fetched",
        "family": FAMILY,
        "directory": str(directory),
    }

    export = ["--json", "artifacts", "export", FAMILY, "--out", str(bundle)]
    assert cli.main(export) == 0
    assert json.loads(capsys.readouterr().out) == {
        "action": "exported",
        "family": FAMILY,
        "bundle": str(bundle),
    }

    assert cli.main(["artifacts", "verify", "--json", FAMILY]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "action": "verified",
        "family": FAMILY,
        "directory": str(directory),
    }

    offline = tmp_path / "offline"
    monkeypatch.setenv("TOKTIER_HOME", str(offline))
    assert cli.main(["artifacts", "import", str(bundle), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "action": "imported",
        "entry": alias,
        "directory": str(offline / "cache" / "artifacts" / alias),
    }


def test_doctor_answers_for_one_family_when_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A named family gets the exact answer the generic report cannot give.

    ``kimi_k3`` routes its CPU work to the reference engine by design.
    The installation-level report says a certified CPU fast path is
    ready here, which is true of the installation and not of that
    family: below the GPU threshold its inputs run on the reference
    engine.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    # The certified CPU profile needs the frozen oracle versions; with
    # them present the installation-level answer below the GPU threshold
    # is "fast_cpu", which is what the family answer has to contradict.
    certified = {"tokenizers": "0.22.2", "transformers": "4.57.6"}
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: certified.get(name, "1.2.3")
    )
    _set_doctor_probes(monkeypatch)

    assert cli.main(["doctor", "--json", "--family", "kimi_k3"]) == 0
    report = json.loads(capsys.readouterr().out)
    family = report["family"]

    assert report["automatic_effective_backend"] == "gpu"
    assert family["family"] == "kimi_k3"
    assert family["fast_cpu_status"] == "unsupported"
    assert family["certification_identity"] == "exact"
    assert family["automatic_effective_backend"] == "gpu"
    assert family["automatic_effective_backend_below_gpu_threshold"] == "hf"

    assert cli.main(["doctor", "--json", "--family", "qwen3_8b"]) == 0
    family = json.loads(capsys.readouterr().out)["family"]
    assert family["fast_cpu_status"] == "certified_source"
    assert family["automatic_effective_backend"] == "gpu"
    assert family["automatic_effective_backend_below_gpu_threshold"] == "fast_cpu"


def test_doctor_applies_the_family_premise_the_adapter_actually_has(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--family`` answers about the family, not only about the engine.

    The adapter reaches the families with a repair-table entry. A family
    outside that table is refused when a session asks for it, so the
    report may not print the same guarantee for it as for a family the
    adapter can open; ``engine_assurance`` keeps stating the
    engine-level fact either way.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)
    from toktier.repair import fastokens

    # A pinned engine whose bytes the registry lists: the engine-level
    # answer is `certified_pinned`, so the family premise is the only
    # thing that can move `exact_id_guarantee`.
    monkeypatch.setattr(
        fastokens,
        "assess",
        lambda identity, **kwargs: fastokens.AssuranceReport(
            assurance=fastokens.ASSURANCE_CERTIFIED_PINNED,
            reason=None,
            known_wheel={"filename": "toktier_fastokens-0.3.1.1.whl"},
            guard_active=True,
            guard_codepoints=154,
            basis={"statement": "guarded"},
            advisory=None,
            distribution="toktier-fastokens",
            version="0.3.1.1",
            engine_digest="d" * 64,
        ),
    )

    assert cli.main(["doctor", "--json", "--family", "qwen3_8b"]) == 0
    admitted = json.loads(capsys.readouterr().out)
    assert admitted["fastokens_family_admitted"] is True
    assert admitted["fastokens_exact_id_guarantee"] is True
    assert admitted["fastokens_family_admission_reason"] is None

    assert cli.main(["doctor", "--json", "--family", "hy3"]) == 0
    outside = json.loads(capsys.readouterr().out)
    assert outside["fastokens_family_admitted"] is False
    assert outside["fastokens_exact_id_guarantee"] is False
    reason = outside["fastokens_family_admission_reason"]
    assert isinstance(reason, str) and "repair-table entry" in reason
    # The engine-level fact is not rewritten by a family premise.
    assert outside["fastokens_engine_assurance"] == "certified_pinned"

    # Without a family the premise does not apply and says so.
    assert cli.main(["doctor", "--json"]) == 0
    generic = json.loads(capsys.readouterr().out)
    assert generic["fastokens_family_admitted"] is None
    assert generic["fastokens_exact_id_guarantee"] is True


def test_doctor_family_reports_the_plan_reasons_verify_local_points_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``verify-local`` sends a reader here, so the answer has to be here.

    A plan that admitted no accelerated route says why in its reasons,
    and the family block carries them with the detail the planner
    recorded -- the axis of a binding that did not verify included.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    certified = {"tokenizers": "0.22.2", "transformers": "4.57.6"}
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: certified.get(name, "1.2.3")
    )
    _set_doctor_probes(monkeypatch)
    # The planner reads the engine facts through its own probe module,
    # so the fixture's fast-CPU facts are placed where the planner looks
    # for them: a source digest that is not the bound one is exactly the
    # state the sentence under test is about.
    from toktier.backends.fast_cpu import FastCpuEngineFacts

    routing_probe = importlib.import_module("toktier.routing.probe")
    monkeypatch.setattr(
        routing_probe,
        "fast_cpu_engine_facts",
        lambda: FastCpuEngineFacts(
            version="0.10.0+toktier.pinned.1",
            source_digest="f" * 64,
            build_flags=("profile=release", "opt-level=3"),
            toolchain="rustc 1.93.1 (test fixture)",
            config_digest="e" * 64,
        ),
    )

    assert cli.main(["doctor", "--json", "--family", "qwen3_8b"]) == 0
    family = json.loads(capsys.readouterr().out)["family"]
    reasons = family["plan_reasons"]

    assert isinstance(reasons, list)
    binding = [
        item for item in reasons if item["code"] == "R_ENGINE_BINDING_MISMATCH"
    ]
    assert binding, reasons
    # The doctor fixture reports a fast-CPU engine whose source digest is
    # not the bound one, which is the state the sentence is about.
    assert binding[0]["backend"] == "fast_cpu"
    assert binding[0]["detail"]["axis"] == "source_digest"
    assert binding[0]["detail"]["observed_digest"] == "f" * 64
    assert binding[0]["detail"]["expected_digest"] != "f" * 64


def test_doctor_refuses_a_family_the_package_does_not_ship(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)

    exit_code = cli.main(["doctor", "--json", "--family", "qwen3-8b"])

    captured = capsys.readouterr()
    assert exit_code == 2
    error = json.loads(captured.err)["error"]
    assert error["code"] == "ARTIFACT_NOT_FOUND"
    assert error["details"]["suggestions"][0] == "qwen3_8b"


def test_an_unusable_root_reports_a_code_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``TOKTIER_HOME`` pointing at a regular file used to end the command
    with a ``NotADirectoryError`` traceback and no envelope at all."""
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("taken", encoding="utf-8")
    monkeypatch.setenv("TOKTIER_HOME", str(occupied))

    exit_code = cli.main(["--json", "artifacts", "verify", "qwen3_8b"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "CONFIG_INVALID"
    assert payload["error"]["details"]["cause"] == "NotADirectoryError"


def test_an_undeterminable_home_reports_a_code_not_a_guess(
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``doctor`` used to answer ``0`` and report ``/.cache/toktier``."""
    monkeypatch.setenv("HOME", "")
    monkeypatch.setenv("USERPROFILE", "")

    exit_code = cli.main(["--json", "doctor"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "CONFIG_INVALID"
    assert payload["error"]["details"]["field"] == "home"


@pytest.mark.parametrize(
    "argv",
    [
        ["--json", "artifacts", "check-conversion", "qwen3_8b"],
        ["artifacts", "check-conversion", "--json", "qwen3_8b"],
    ],
)
def test_check_conversion_refuses_inside_the_envelope(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """A family with no conversion recipe used to answer with one line of
    prose on either side of the flag: right exit code, no code, no
    envelope."""
    exit_code = cli.main(argv)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "UNSUPPORTED_CONFIG"
    assert payload["error"]["details"]["value"] == "qwen3_8b"
    assert "artifacts verify" in payload["error"]["details"]["remedy"]


def test_check_conversion_without_json_still_names_the_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["artifacts", "check-conversion", "qwen3_8b"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.splitlines() == [
        "error UNSUPPORTED_CONFIG: qwen3_8b: this family is downloaded "
        "whole, not converted"
    ]


def test_a_failed_conversion_gate_reports_a_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gate's own failure line carried no code either."""
    report = {
        "family": "kimi_k3",
        "converter": "kimi_tiktoken_v1",
        "upstream_repo": "moonshotai/Kimi-K3",
        "upstream_revision": "0" * 40,
        "upstream_inputs": [],
        "runs": 2,
        "deterministic": True,
        "observed_sha256": "a" * 64,
        "expected_sha256": "b" * 64,
        "added_tokens": 0,
        "added_tokens_special": 0,
        "added_tokens_first_id": 0,
        "identity_matches": False,
        "added_tokens_contiguous": True,
        "added_tokens_fully_described": True,
    }
    monkeypatch.setattr(cli, "recipe_for", lambda family: object())
    monkeypatch.setattr(
        cli, "conversion_report", lambda *args, **kwargs: report
    )

    exit_code = cli.main(["--json", "artifacts", "check-conversion", "kimi_k3"])

    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "ARTIFACT_HASH_MISMATCH"
    assert payload["error"]["details"]["failures"] == ["identity_matches"]
    assert payload["error"]["details"]["expected_sha256"] == "b" * 64
