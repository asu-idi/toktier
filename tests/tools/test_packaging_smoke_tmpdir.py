"""The packaging smoke script places its scratch venv under ``TMPDIR``.

The script builds a venv (and, without an argument, a wheel) in a
scratch directory. It takes that directory from ``TMPDIR`` when set, so a
host with a small ``/tmp`` can direct it elsewhere. The test runs the
script's own ``mktemp`` line in a shell, with and without ``TMPDIR``,
without building or installing anything.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = ROOT / "tools" / "run_packaging_smoke.sh"


def _mktemp_line() -> str:
    lines = [
        line
        for line in SMOKE_SCRIPT.read_text(encoding="utf-8").splitlines()
        if re.match(r"^venv_dir=\$\(mktemp -d ", line)
    ]
    assert len(lines) == 1, lines
    return lines[0]


def _scratch_dir(environment: dict[str, str]) -> Path:
    script = _mktemp_line() + '\nprintf "%s" "$venv_dir"\nrmdir "$venv_dir"\n'
    result = subprocess.run(
        ["sh", "-eu", "-c", script],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout)


def test_the_scratch_venv_goes_under_tmpdir(tmp_path: Path) -> None:
    environment = {key: value for key, value in os.environ.items() if key != "TMPDIR"}
    environment["TMPDIR"] = str(tmp_path)
    created = _scratch_dir(environment)
    assert created.parent == tmp_path
    assert created.name.startswith("toktier-packaging-smoke.")
    assert not created.exists()


def test_without_tmpdir_the_scratch_venv_goes_under_tmp() -> None:
    environment = {key: value for key, value in os.environ.items() if key != "TMPDIR"}
    created = _scratch_dir(environment)
    assert created.parent == Path("/tmp")
    assert not created.exists()


def test_the_script_parses() -> None:
    subprocess.run(["sh", "-n", str(SMOKE_SCRIPT)], check=True)
