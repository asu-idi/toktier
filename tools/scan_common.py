# Shared helpers for repository hygiene scanners.
import os
from collections.abc import Iterator
from fnmatch import fnmatchcase
from pathlib import Path

SKIPPED_NAMES = {".git", ".venv", "build", "dist", "target"}


def load_allowlist(path: Path) -> tuple[str, ...]:
    patterns = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        pattern = raw_line.strip()
        if pattern and not pattern.startswith("#"):
            patterns.append(pattern)
    return tuple(patterns)


def is_allowed(relative_path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(relative_path, pattern) for pattern in patterns)


def iter_files(root: Path) -> Iterator[tuple[Path, str]]:
    for directory, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in SKIPPED_NAMES
        )
        for file_name in sorted(file_names):
            if file_name in SKIPPED_NAMES:
                continue
            path = Path(directory) / file_name
            yield path, path.relative_to(root).as_posix()


def read_text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="replace")
