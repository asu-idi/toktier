"""Run conformance probes against an installed ``toktier`` distribution."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

import pytest

_NOT_INSTALLED = 5
_DISCOVER_INSTALLED_PACKAGE = f"""
import importlib.metadata
import json

try:
    distribution = importlib.metadata.distribution("toktier")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit({_NOT_INSTALLED})

import toktier

print(json.dumps({{
    "origin": toktier.__file__,
    "version": distribution.version,
}}))
"""


@dataclass(frozen=True)
class InstalledPackage:
    """Isolated interpreter for exercising installed-package behavior."""

    executable: str
    origin: str
    version: str

    def json_output(self, source: str, *arguments: str) -> object:
        """Run a probe with repository paths isolated and decode its JSON."""

        completed = subprocess.run(
            [self.executable, "-I", "-c", source, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            pytest.fail(
                "installed-package conformance probe failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return json.loads(completed.stdout)


@pytest.fixture(scope="session", autouse=True)
def installed_package() -> InstalledPackage:
    """Require a distribution and exclude pytest's source-path injection."""

    completed = subprocess.run(
        [sys.executable, "-I", "-c", _DISCOVER_INSTALLED_PACKAGE],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode == _NOT_INSTALLED:
        pytest.skip(
            "toktier is not installed; installed-package conformance tests skipped"
        )
    if completed.returncode != 0:
        pytest.fail(
            "the installed toktier distribution could not be imported\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    discovered = json.loads(completed.stdout)
    return InstalledPackage(
        executable=sys.executable,
        origin=discovered["origin"],
        version=discovered["version"],
    )
