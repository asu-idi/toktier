# Report non-ASCII source and configuration text outside its allowlist.
from pathlib import Path

from scan_common import is_allowed, iter_files, load_allowlist, read_text

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tools" / "scan_allowlist_non_ascii.txt"
SCANNED_SUFFIXES = {
    ".cuh",
    ".cu",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".yaml",
    ".yml",
}


def main() -> int:
    patterns = load_allowlist(ALLOWLIST)
    found = False
    for path, relative_path in iter_files(ROOT):
        if path.suffix.lower() not in SCANNED_SUFFIXES or is_allowed(
            relative_path, patterns
        ):
            continue
        text = read_text(path)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(ord(character) > 127 for character in line):
                print(f"{relative_path}:{line_number}:{line}")
                found = True
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
