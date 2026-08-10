"""The actual CUDA compiler is part of JIT identity and cache scope."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from toktier.engine.gpu.toolchain import (
    NvccFacts,
    jit_toolchain_satisfied,
    nvcc_facts,
)


def test_nvcc_version_parser_records_release_build_and_resolved_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("CUDA_HOME", str(nvcc.parents[1]))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    completed = subprocess.CompletedProcess(
        [str(nvcc), "--version"],
        0,
        stdout=(
            "nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Cuda compilation tools, release 13.2, V13.2.86\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: completed)

    facts = nvcc_facts()
    assert facts.path == str(nvcc)
    assert facts.resolved_path == str(nvcc.resolve())
    assert facts.release == "13.2"
    assert facts.build == "V13.2.86"
    assert facts.error is None


def test_missing_or_unparseable_nvcc_never_satisfies_jit_certificate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = NvccFacts(None, None, None, None, "nvcc was not found")
    assert not jit_toolchain_satisfied(
        torch_cuda="13.0",
        torch_version="2.13.0+cu130",
        nvcc=missing,
        ninja_present=True,
    )

    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("CUDA_HOME", str(nvcc.parents[1]))
    completed = subprocess.CompletedProcess(
        [str(nvcc), "--version"], 0, stdout="surprising output", stderr=""
    )
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: completed)
    observed = nvcc_facts()
    assert observed.release is None
    assert observed.error == "nvcc --version output was not recognized"


def test_compiler_release_is_an_independent_judgement_axis() -> None:
    accepted = NvccFacts(
        "/opt/cuda/bin/nvcc",
        "/opt/cuda-13.0/bin/nvcc",
        "13.0",
        "V13.0.88",
        None,
    )
    drifted = NvccFacts(
        "/opt/cuda/bin/nvcc",
        "/opt/cuda-13.2/bin/nvcc",
        "13.2",
        "V13.2.86",
        None,
    )
    assert jit_toolchain_satisfied(
        torch_cuda="13.0",
        torch_version="2.13.0+cu130",
        nvcc=accepted,
        ninja_present=True,
    )
    assert not jit_toolchain_satisfied(
        torch_cuda="13.0",
        torch_version="2.13.0+cu130",
        nvcc=drifted,
        ninja_present=True,
    )
    assert accepted.cache_tag(
        torch_cuda="13.0", torch_version="2.13.0+cu130"
    ) != drifted.cache_tag(
        torch_cuda="13.0", torch_version="2.13.0+cu130"
    )
