# Report legacy internal-name residue in repository text files.
import re
from pathlib import Path

from scan_common import is_allowed, iter_files, load_allowlist, read_text

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tools" / "scan_allowlist_name_residue.txt"
LEGACY_NAME = re.compile("lo" + "pt", re.IGNORECASE)


def main() -> int:
    patterns = load_allowlist(ALLOWLIST)
    found = False
    for path, relative_path in iter_files(ROOT):
        if is_allowed(relative_path, patterns):
            continue
        text = read_text(path)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if LEGACY_NAME.search(line):
                print(f"{relative_path}:{line_number}:{line}")
                found = True
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
