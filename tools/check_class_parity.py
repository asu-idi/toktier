#!/usr/bin/env python3
"""Ask whether this tree's character classes answer the way the judge does.

The certification compares TokTier's ids with the ids the frozen Hugging
Face reference engine produces, and that engine cuts text on Unicode
character classes it reads from Oniguruma. The fast CPU pre-tokenizer
reads the same classes from ICU4X property data. Nothing said the two
had to carry the same Unicode version, and in August 2026 they did not:
ICU4X 2.1 moved its data to Unicode 17.0 while Oniguruma stayed on 16.0,
so 4,804 code points -- the whole of CJK Extension J among them -- were
letters or Han on one side and unassigned on the other. The workspace
manifest now pins the property data to the version the judge carries.

This is the gate that keeps the pin honest. It runs the exhaustive
comparison that lives with the tables it is about, in
``crates/toktier-gigatoken-core/src/pretokenize/unicode.rs``: every
scalar value, every class the shipped pre-tokenizer patterns name, both
sides. CI reaches the same test through the workspace test run; this
command exists so a release battery, or a person, can ask the one
question by itself.

Usage::

    python tools/check_class_parity.py
    python tools/check_class_parity.py --check
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The test module that holds the comparison, and the crate that owns the
#: tables it compares.
CRATE = "toktier-gigatoken-core"
FILTER = "pretokenize::unicode::class_parity"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="accepted for symmetry with the other record checks; this "
        "command only ever checks",
    )
    parser.parse_args()
    command = [
        "cargo",
        "test",
        "--locked",
        "--offline",
        "--package",
        CRATE,
        FILTER,
    ]
    print(f"+ {shlex.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    if completed.returncode != 0:
        print(
            "error: the character classes of this tree do not answer the way "
            "the judge does; the output above names the class and the first "
            "code points that differ",
            file=sys.stderr,
        )
        return completed.returncode
    print("character-class parity with the judge: check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
