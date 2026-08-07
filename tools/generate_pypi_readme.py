#!/usr/bin/env python3
"""Generate the PyPI long description from the canonical README.

PyPI renders metadata outside the repository, so relative links and images do
not resolve there. The repository README remains the sole editable source;
this tool rewrites its relative targets to immutable project URLs and reduces
the light/dark ``picture`` element to one portable image.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "README.md"
OUTPUT = ROOT / "README.pypi.md"
REPOSITORY = "https://github.com/asu-idi/toktier"
RELEASE_REF = "v0.1.0"
RAW = f"https://raw.githubusercontent.com/asu-idi/toktier/{RELEASE_REF}"

_PICTURE = re.compile(r"<picture>.*?</picture>", re.DOTALL)
_MARKDOWN_LINK = re.compile(r"(?P<image>!?)\[(?P<label>[^]]*)]\((?P<target>[^)]+)\)")
_HTML_TARGET = re.compile(r'(?P<attribute>src|srcset|href)="(?P<target>[^"]+)"')


def _external(target: str) -> bool:
    return target.startswith(("https://", "http://", "mailto:", "#"))


def _url(target: str, *, image: bool) -> str:
    if _external(target):
        return target
    clean = target.removeprefix("./")
    if image:
        return f"{RAW}/{clean}"
    if clean.endswith("/"):
        return f"{REPOSITORY}/tree/{RELEASE_REF}/{clean.rstrip('/')}"
    return f"{REPOSITORY}/blob/{RELEASE_REF}/{clean}"


def render(source: str) -> str:
    source = _PICTURE.sub(
        "![Latency head-to-head: TokTier versus full re-encode across three "
        "workloads of a 4M-character session]"
        f"({RAW}/docs/figures/hero_session_vs_reencode.svg)",
        source,
        count=1,
    )

    def markdown(match: re.Match[str]) -> str:
        marker = match.group("image")
        target = match.group("target")
        # Preserve optional Markdown titles while rewriting the URL portion.
        url, separator, title = target.partition(' "')
        rendered = _url(url, image=bool(marker))
        suffix = f' "{title}' if separator else ""
        return f"{marker}[{match.group('label')}]({rendered}{suffix})"

    source = _MARKDOWN_LINK.sub(markdown, source)

    def html(match: re.Match[str]) -> str:
        attribute = match.group("attribute")
        target = match.group("target")
        return f'{attribute}="{_url(target, image=attribute != "href")}"'

    return _HTML_TARGET.sub(html, source).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    payload = render(SOURCE.read_text(encoding="utf-8"))
    if arguments.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != payload:
            print(f"error: {OUTPUT} is not generated from {SOURCE}", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
