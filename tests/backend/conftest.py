"""Test session setup.

Two responsibilities:

- Put the package on the import path without requiring an install, so
  the suite runs in a checkout.
- Set the oracle's threading environment before it is ever imported.
  The oracle package reads these once, at import, so a later assignment
  would be silently ignored; the values keep a test run from
  oversubscribing the machine and keep timings comparable.

Optional inputs, all honest about their absence:

``TOKTIER_TEST_ARTIFACTS``
    Directory holding one subdirectory per artifact, each with a
    ``tokenizer.json``. When unset, parity sampling runs against a small
    synthetic artifact and says so in its report line.
``TOKTIER_TEST_CORPUS``
    Directory of UTF-8 text files, or a JSONL file with a ``text``
    member, used to build the parity document pool. When unset, the pool
    is generated and labeled as generated.
``TOKTIER_TEST_PARITY_DOCS``
    Documents per family (default 1000).

These are development inputs, not part of the frozen environment
variable set of ``docs/contracts/config.md``.

Source text in this repository is ASCII, so the non-ASCII probe cases
below are written as escapes rather than as literal characters.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# This tier's helper module (_support) must be importable by the test
# modules under pytest's importlib import mode.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

DEFAULT_PARITY_DOCS = 1000
DOC_CHARS = 2048
_REVISION_SUFFIX = re.compile(r"-[0-9a-f]{12}$")

#: Fixed cases appended to every parity pool. Cheap, and each one has
#: bitten a tokenizer wrapper somewhere: empty and whitespace-only
#: input, precomposed against combining marks, a zero-width joiner
#: sequence, mixed scripts, text that looks like a special token, and
#: long repetitive runs.
EDGE_DOCUMENTS = (
    "",
    " ",
    "\n\n\t  \r\n",
    "hello world",
    "\u00e9 e\u0301",
    "\U0001f469\u200d\U0001f4bb",
    "a\u4e2d\U0001f642b",
    "\ud55c\uae00 \u30c6\u30b9\u30c8 \u6d4b\u8bd5",
    "<|endoftext|>",
    "[CLS] x [SEP]",
    "def f(x):\n    return x + 1\n" * 8,
    "aaaa" * 512,
    "0123456789 " * 64,
)

#: Alphabet of the generated pool: ASCII text and punctuation, plus a
#: few multi-byte code points so that byte-level and character-level
#: paths are both exercised.
_GENERATED_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " \n\t.,;:'\"(){}[]<>/\\|-_=+*&^%$#@!?~`"
    "\u4e2d\u6587\u6d4b\u8bd5\u3042\u3044\u3046\ud55c\uae00"
    "\u00e9\u00e8\u00fc\u00df\u0301\u200d\U0001f642\U0001f469"
)


def _iter_corpus_files(root: Path) -> Iterator[str]:
    if root.is_file():
        with root.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text")
                if isinstance(text, str) and text:
                    yield text
        return
    for path in sorted(root.iterdir()):
        if path.is_file():
            yield path.read_text(encoding="utf-8", errors="replace")


def _slice_documents(sources: Iterator[str], wanted: int) -> list[str]:
    documents: list[str] = []
    for text in sources:
        for start in range(0, len(text), DOC_CHARS):
            chunk = text[start : start + DOC_CHARS]
            if chunk:
                documents.append(chunk)
            if len(documents) >= wanted:
                return documents
    return documents


def _generated_documents(wanted: int) -> list[str]:
    """A deterministic pool used when no corpus is provided."""
    import random

    rng = random.Random(20260805)
    documents: list[str] = []
    for _ in range(wanted):
        length = rng.randint(16, DOC_CHARS)
        documents.append(
            "".join(rng.choice(_GENERATED_ALPHABET) for _ in range(length))
        )
    return documents


@pytest.fixture(scope="session")
def artifact_root() -> Path | None:
    """Directory of local artifacts, or ``None`` when unavailable."""
    raw = os.environ.get("TOKTIER_TEST_ARTIFACTS")
    if not raw:
        return None
    root = Path(raw)
    return root if root.is_dir() else None


@pytest.fixture(scope="session")
def local_artifact_dirs(artifact_root: Path | None) -> list[tuple[str, Path]]:
    """(family, directory) pairs for every local artifact found."""
    if artifact_root is None:
        return []
    found: list[tuple[str, Path]] = []
    for path in sorted(artifact_root.iterdir()):
        if not path.is_dir() or not (path / "tokenizer.json").is_file():
            continue
        found.append((_REVISION_SUFFIX.sub("", path.name), path))
    return found


@pytest.fixture(scope="session")
def parity_documents() -> tuple[list[str], str]:
    """(documents, provenance) for parity sampling."""
    wanted = int(os.environ.get("TOKTIER_TEST_PARITY_DOCS", DEFAULT_PARITY_DOCS))
    raw = os.environ.get("TOKTIER_TEST_CORPUS")
    documents: list[str] = []
    provenance = "generated"
    if raw:
        root = Path(raw)
        if root.exists():
            documents = _slice_documents(_iter_corpus_files(root), wanted)
            provenance = "corpus"
    if len(documents) < wanted:
        if documents:
            provenance = "corpus+generated"
        documents.extend(_generated_documents(wanted - len(documents)))
    documents.extend(EDGE_DOCUMENTS)
    return documents, provenance


def pytest_sessionfinish() -> None:
    """Report the parity readings collected during the session."""
    import _support

    for reading in _support.PARITY_READINGS:
        print(
            "parity: family={family} documents={documents} "
            "mismatches={mismatches} pool={provenance}".format(**reading)
        )
