# Report internal-name residue in repository text files.
#
# Two names are refused anywhere in the tree, case-insensitively: the
# legacy internal project name and the internal name of the research
# prototype. Both patterns are built by concatenation so that this file
# does not match itself.
import re
from pathlib import Path

from scan_common import is_allowed, iter_files, load_allowlist, read_text

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tools" / "scan_allowlist_name_residue.txt"
LEGACY_NAME = re.compile("lo" + "pt", re.IGNORECASE)
PROTOTYPE_NAME = re.compile("inc" + "tok", re.IGNORECASE)
FORBIDDEN_NAMES = (LEGACY_NAME, PROTOTYPE_NAME)


def scan(root: Path, patterns: tuple[str, ...] = ()) -> list[str]:
    """Return ``path:line:text`` for every line naming a forbidden name."""
    findings = []
    for path, relative_path in iter_files(root):
        if is_allowed(relative_path, patterns):
            continue
        text = read_text(path)
        if text is None:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(name.search(line) for name in FORBIDDEN_NAMES):
                findings.append(f"{relative_path}:{line_number}:{line}")
    return findings


def main() -> int:
    findings = scan(ROOT, load_allowlist(ALLOWLIST))
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
