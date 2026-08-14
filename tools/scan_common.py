# Shared helpers for the repository-side tools.
import json
import os
from collections.abc import Iterator
from fnmatch import fnmatchcase
from pathlib import Path

SKIPPED_NAMES = {".git", ".venv", "build", "dist", "target"}

#: Exit status a tool uses when it has nothing to check in this tree.
#: Distinct from 0, which would read as "the check passed", and from the
#: failure codes, which would read as "the check found something".
DECLINED = 3

#: Written at the root of a published source archive, and only there.
ARCHIVE_MANIFEST_NAME = "SOURCE-MANIFEST.json"
ARCHIVE_MANIFEST_SCHEMA = "toktier.rust_source_archive.v1"


def vendored_source_archive(root: Path) -> bool:
    """Whether ``root`` is an unpacked published source archive.

    ``tools/build_rust_source_archive.py`` writes ``SOURCE-MANIFEST.json``
    at the root of what it builds and nowhere else, so a file of that
    name carrying the archive's own schema tag is what tells the two
    trees apart. Tools that verify the repository against itself ask
    this before running, so that "not applicable here" is something they
    say rather than something a reader infers from a failure.
    """
    manifest = root / ARCHIVE_MANIFEST_NAME
    if not manifest.is_file():
        return False
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(document, dict)
        and document.get("schema") == ARCHIVE_MANIFEST_SCHEMA
    )


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
