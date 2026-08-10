"""Static release identity and artifact-verifier contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERIFY_IDENTITY = ROOT / "tools" / "verify_release_identity.py"
VERIFY_ARTIFACTS = ROOT / "tools" / "verify_release_artifacts.py"


def test_release_identity_is_v020() -> None:
    subprocess.run(
        [sys.executable, str(VERIFY_IDENTITY), "--tag", "v0.2.0"],
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
    assert "must be 'v0.2.0'" in completed.stderr


def test_release_artifact_set_is_one_abi3_linux_wheel() -> None:
    source = VERIFY_ARTIFACTS.read_text(encoding="utf-8")
    assert (
        'EXPECTED_WHEEL = "toktier-0.2.0-cp310-abi3-manylinux_2_34_x86_64.whl"'
        in source
    )
