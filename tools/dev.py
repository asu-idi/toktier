# Run the repository's local checks without changing tracked files or using the network.
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scan_common import DECLINED, vendored_source_archive

ROOT = Path(__file__).resolve().parents[1]

Command = tuple[str, ...]

CHECK_COMMANDS: tuple[Command, ...] = (
    ("ruff", "check", "--no-cache", "."),
    ("mypy", "--no-site-packages", "--no-incremental", "."),
    (sys.executable, "tools/scan_non_ascii.py"),
    (sys.executable, "tools/scan_name_residue.py"),
    (sys.executable, "tools/scan_secrets.py"),
    (sys.executable, "tools/scan_version_constants.py"),
    (sys.executable, "tools/verify_carryover.py", "--check"),
    ("cargo", "fmt", "--check"),
    (
        "cargo",
        "clippy",
        "--offline",
        "--locked",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ),
)

TEST_CORE_COMMANDS: tuple[Command, ...] = (
    ("pytest", "-p", "no:cacheprovider"),
    ("cargo", "test", "--offline", "--locked"),
)

# The packaging suite asserts the release's import-surface promises
# (import hygiene, offline behavior, no GPU probe at import) against the
# source tree, offline. The isolated-venv variant that additionally
# exercises a wheel install is tools/run_packaging_smoke.sh; it creates
# a fresh virtual environment and therefore needs an index or a wheel
# cache, which this offline command deliberately does not.
TEST_PACKAGING_COMMANDS: tuple[Command, ...] = (
    ("pytest", "-p", "no:cacheprovider", "tests/packaging"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run toktier development tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Run Python and Rust static checks.")
    subparsers.add_parser("test-core", help="Run Python and Rust core tests.")
    subparsers.add_parser(
        "test-packaging",
        help=(
            "Run the packaging test suite (tests/packaging); a non-zero "
            "exit is a failed gate."
        ),
    )
    return parser


def run_commands(commands: tuple[Command, ...]) -> int:
    with tempfile.TemporaryDirectory(prefix="toktier-dev-") as temporary_directory:
        environment = os.environ.copy()
        environment["CARGO_NET_OFFLINE"] = "true"
        environment["CARGO_TARGET_DIR"] = os.path.join(
            temporary_directory, "cargo-target"
        )
        environment["PIP_NO_INDEX"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["RUSTUP_AUTO_INSTALL"] = "0"

        for command in commands:
            print(f"+ {shlex.join(command)}", flush=True)
            try:
                result = subprocess.run(command, check=False, env=environment)
            except FileNotFoundError:
                print(f"{command[0]}: command not found", file=sys.stderr)
                return 127
            if result.returncode != 0:
                return result.returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    command = str(arguments.command)
    if command == "test-packaging":
        if vendored_source_archive(ROOT):
            # The published source archive carries the sources, the
            # evidence and the vendored dependencies, and deliberately
            # not `tests/`. Handing the missing directory to pytest gets
            # "collected 0 items" and pytest's usage exit, which reads
            # like a broken checkout rather than a suite that was never
            # shipped here.
            print(
                "declined: the packaging suite lives in tests/, which the "
                "published source archive does not carry. Nothing was run. "
                "This check runs from a repository checkout.",
                file=sys.stderr,
            )
            return DECLINED
        return run_commands(TEST_PACKAGING_COMMANDS)
    if command == "check":
        return run_commands(CHECK_COMMANDS)
    return run_commands(TEST_CORE_COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
