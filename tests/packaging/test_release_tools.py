"""Static release identity and artifact-verifier contracts."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFY_IDENTITY = ROOT / "tools" / "verify_release_identity.py"
VERIFY_ARTIFACTS = ROOT / "tools" / "verify_release_artifacts.py"


def _artifact_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_release_artifacts", VERIFY_ARTIFACTS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel_with_payload(path: Path, payload: bytes) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("toktier/_native.abi3.so", payload)
    return path


def test_release_identity_is_v021() -> None:
    subprocess.run(
        [sys.executable, str(VERIFY_IDENTITY), "--tag", "v0.2.1"],
        check=True,
    )


def test_release_identity_rejects_another_tag() -> None:
    completed = subprocess.run(
        [sys.executable, str(VERIFY_IDENTITY), "--tag", "v0.1.0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "must be 'v0.2.1'" in completed.stderr


def test_release_artifact_set_is_one_abi3_linux_wheel() -> None:
    source = VERIFY_ARTIFACTS.read_text(encoding="utf-8")
    assert (
        'EXPECTED_WHEEL = "toktier-0.2.1-cp310-abi3-manylinux_2_34_x86_64.whl"'
        in source
    )


@pytest.mark.parametrize("representation", ["hex", "bytes"])
def test_release_artifact_gate_rejects_identity_sentinel(
    tmp_path: Path, representation: str
) -> None:
    verifier = _artifact_verifier()
    sentinel = (
        verifier.IDENTITY_SENTINEL_HEX.encode("ascii")
        if representation == "hex"
        else verifier.IDENTITY_SENTINEL_BYTES
    )
    wheel = _wheel_with_payload(tmp_path / "sentinel.whl", b"prefix" + sentinel)

    with zipfile.ZipFile(wheel) as archive, pytest.raises(
        ValueError, match="contains the sentinel build identity"
    ):
        verifier._verify_no_identity_sentinel(archive, archive.namelist())


def test_release_artifact_gate_accepts_normal_identity(tmp_path: Path) -> None:
    verifier = _artifact_verifier()
    wheel = _wheel_with_payload(tmp_path / "normal.whl", b"\x7fELF" + b"a" * 64)

    with zipfile.ZipFile(wheel) as archive:
        verifier._verify_no_identity_sentinel(archive, archive.namelist())
