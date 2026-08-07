# Report common credential patterns in repository text files.
import re
from pathlib import Path

from scan_common import is_allowed, iter_files, load_allowlist, read_text

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tools" / "scan_allowlist_secrets.txt"
API_KEY_NAME = "api" + r"[_-]?" + "key"
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-{5}BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-{5}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[po]_[A-Za-z0-9]{36,}\b"),
    re.compile(
        r"\b"
        + API_KEY_NAME
        + r"\b\s*=\s*(?:\"[^\"\r\n]+\"|'[^'\r\n]+'|[^\s#;]+)",
        re.IGNORECASE,
    ),
)


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
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                print(f"{relative_path}:{line_number}:{line}")
                found = True
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
