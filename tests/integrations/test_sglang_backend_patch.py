"""The SGLang backend patches wrap the tokenizer in the TokenizerManager only.

SGLang's scheduler, TP worker and detokenizer processes load their own
tokenizer through ``get_tokenizer`` with the same ``tokenizer_backend``
argument, so a wrap inside ``get_tokenizer`` would give each of them a
wrapper (and, with the worker pool on, a pool) of its own. The patches
therefore leave ``get_tokenizer`` alone and wrap in
``TokenizerManager.init_tokenizer_and_processor``, before the dynamic-batch
tokenizer is built: one wrapper, and at most one pool, per tokenizer
process. These tests read the shipped patch files; they need no SGLang.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PATCHES = Path(__file__).resolve().parents[2] / "patches"
MANAGER = "python/sglang/srt/managers/tokenizer_manager.py"
GET_TOKENIZER = "python/sglang/srt/utils/hf_transformers/tokenizer.py"
OPTION_FILES = {
    "sglang-0.5.19-toktier-backend.patch": "python/sglang/srt/server_args.py",
    "sglang-0.5.21-toktier-backend.patch": (
        "python/sglang/srt/arg_groups/fields/serving.py"
    ),
}
WRAP_CALL = "self.tokenizer = self._wrap_tokenizer_with_toktier(self.tokenizer)"
BATCHER_COMMENT = "# Initialize async dynamic batch tokenizer if enabled"

_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


def _hunks(text: str) -> dict[str, list[list[str]]]:
    """``{path: [hunk body, ...]}``, each body checked against its header."""
    files: dict[str, list[list[str]]] = {}
    lines = text.split("\n")
    path: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :]
            files[path] = []
            continue
        match = _HUNK.match(line)
        if match is None:
            continue
        assert path is not None, "a hunk before any file header"
        old_left = int(match.group(1) or 1)
        new_left = int(match.group(2) or 1)
        body: list[str] = []
        while old_left or new_left:
            row = lines[i]
            i += 1
            tag = row[:1]
            assert tag in (" ", "-", "+"), f"{path}: unexpected hunk line {row!r}"
            old_left -= tag in (" ", "-")
            new_left -= tag in (" ", "+")
            assert old_left >= 0 and new_left >= 0, (
                f"{path}: hunk longer than its header"
            )
            body.append(row)
        files[path].append(body)
    return files


def _added(body: list[str]) -> list[str]:
    return [row[1:] for row in body if row.startswith("+")]


@pytest.fixture(scope="module")
def patches() -> dict[str, dict[str, list[list[str]]]]:
    found = sorted(p.name for p in PATCHES.glob("sglang-*-toktier-backend.patch"))
    assert found == sorted(OPTION_FILES)
    out = {}
    for name in found:
        text = (PATCHES / name).read_text(encoding="utf-8")
        assert text.isascii()
        out[name] = _hunks(text)
    return out


def test_patches_leave_get_tokenizer_unchanged(
    patches: dict[str, dict[str, list[list[str]]]],
) -> None:
    for name, files in patches.items():
        assert set(files) == {OPTION_FILES[name], MANAGER}, name
        assert GET_TOKENIZER not in files


def test_patches_wrap_once_before_the_batcher_is_built(
    patches: dict[str, dict[str, list[list[str]]]],
) -> None:
    for name, files in patches.items():
        hunks = files[MANAGER]
        added = [row.strip() for body in hunks for row in _added(body)]
        # One wrap site, one call of the adapter's wrap_tokenizer.
        assert added.count(WRAP_CALL) == 1, name
        assert sum("wrap_tokenizer(" in row for row in added) == 1, name
        (hunk,) = [body for body in hunks if any(WRAP_CALL in r for r in body)]
        at = next(k for k, row in enumerate(hunk) if WRAP_CALL in row)
        before = [row for row in hunk[:at] if row.startswith(" ")]
        after = [row for row in hunk[at:] if row.startswith(" ")]
        # After the tokenizer is loaded (the text branch's get_tokenizer
        # call closes just above) and before the dynamic-batch tokenizer.
        assert any(
            "tokenizer_backend=get_serving().tokenizer_backend" in r for r in before
        )
        assert BATCHER_COMMENT in after[0], name
        # Only for this backend, and only when a tokenizer was loaded.
        guard = next(r for r in _added(hunk) if r.strip().startswith("if "))
        assert 'tokenizer_backend == "toktier"' in guard
        assert "self.tokenizer is not None" in guard


def test_both_patches_carry_the_same_manager_code(
    patches: dict[str, dict[str, list[list[str]]]],
) -> None:
    def code(files: dict[str, list[list[str]]]) -> list[str]:
        return [r for body in files[MANAGER] for r in _added(body) if r.strip()]

    first, second = (code(files) for files in patches.values())
    assert first == second
