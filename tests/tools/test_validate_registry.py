from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPOSITORY_ROOT / "tools" / "validate_registry.py"
FIXTURES = Path(__file__).with_name("fixtures")


class ValidateRegistryCliTests(unittest.TestCase):
    def run_fixture(
        self, fixture: str, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(TOOL), *arguments, str(FIXTURES / fixture)],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_golden_registry_passes(self) -> None:
        result = self.run_fixture("valid_registry.json")

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "schema: PASS\nroot_digest: PASS\n")
        self.assertEqual(result.stderr, "")

    def test_tampered_digest_fails(self) -> None:
        result = self.run_fixture("tampered_digest_registry.json")

        self.assertEqual(result.returncode, 1)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "schema: PASS")
        self.assertTrue(lines[1].startswith("root_digest: FAIL: expected sha256:"))
        self.assertEqual(len(lines), 2)

    def test_schema_violation_fails(self) -> None:
        result = self.run_fixture(
            "schema_violation_registry.json", "--schema", "registry"
        )

        self.assertEqual(result.returncode, 1)
        lines = result.stdout.splitlines()
        self.assertTrue(lines[0].startswith("schema: FAIL:"))
        self.assertEqual(lines[1], "root_digest: PASS")
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
