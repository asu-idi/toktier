"""The store-fingerprint history table is recomputed at its release tags.

``tools/store_history.py`` regenerates every row of
``src/toktier/routing/tables/store_fingerprint_history.v1.json`` from what
the named release shipped, and ``generate_registry.py --release-check``
refuses a table that differs. These tests pin that the shipped table equals
its recomputation (where the tags are present), that a changed row or digest
is named, that a missing tag is refused outside a shallow clone, and that a
tree without the table has nothing to check.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import store_history as tool  # noqa: E402

from toktier._jcs import canonical_json  # noqa: E402


def _has_tags(*tags: str) -> bool:
    return all(
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "--verify",
                "--quiet",
                f"{tag}^{{commit}}",
            ],
            capture_output=True,
            check=False,
        ).returncode
        == 0
        for tag in tags
    )


def _shipped() -> dict[str, Any]:
    table: dict[str, Any] = json.loads(tool.HISTORY_PATH.read_text(encoding="utf-8"))
    return table


def _tags() -> list[str]:
    return [f"v{row['release']}" for row in _shipped()["releases"]]


requires_tags = pytest.mark.skipif(
    not tool.HISTORY_PATH.exists() or not _has_tags(*_tags()),
    reason="the release tags the history table names are not in this checkout",
)


@requires_tags
def test_the_shipped_table_equals_its_recomputation() -> None:
    assert tool.history_problems() == ([], [])
    regenerated = tool.history_document(ROOT, _tags())
    assert regenerated == _shipped()


@requires_tags
def test_a_changed_row_or_digest_is_named(tmp_path: Path) -> None:
    table = _shipped()
    changed = copy.deepcopy(table)
    first = changed["releases"][0]
    sha = sorted(first["artifacts"])[0]
    first["artifacts"][sha]["repair_parameters"] = "none"
    path = tmp_path / "history.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    problems, _notes = tool.history_problems(path)
    assert any(
        f"the {first['release']} row differs" in problem and "artifacts" in problem
        for problem in problems
    )
    # The same change with the root digest refreshed over it is named twice:
    # by the row and by the digest, which no longer equals the recomputed one.
    body = {key: value for key, value in changed.items() if key != "root_digest"}
    changed["root_digest"] = (
        "sha256:"
        + hashlib.sha256(tool.HISTORY_DOMAIN_TAG + canonical_json(body)).hexdigest()
    )
    path.write_text(json.dumps(changed), encoding="utf-8")
    problems, _notes = tool.history_problems(path)
    assert any(f"the {first['release']} row differs" in problem for problem in problems)
    assert any("root digest" in problem for problem in problems)
    changed["note"] = "another note"
    path.write_text(json.dumps(changed), encoding="utf-8")
    problems, _notes = tool.history_problems(path)
    assert any("note differs" in problem for problem in problems)


def test_a_missing_tag_is_refused(tmp_path: Path) -> None:
    table = {
        "schema": tool.SCHEMA,
        "releases": [{"release": "0.0.0-no-such-release"}],
        "root_digest": "sha256:0",
    }
    path = tmp_path / "history.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    problems, notes = tool.history_problems(path)
    shallow = (
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--is-shallow-repository"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        == "true"
    )
    if shallow:
        assert problems == [] and "shallow clone" in notes[0]
    else:
        assert notes == []
        assert any("cannot be recomputed" in problem for problem in problems)


def test_no_table_has_nothing_to_check(tmp_path: Path) -> None:
    assert tool.history_problems(tmp_path / "absent.json") == ([], [])
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"releases": "none"}), encoding="utf-8")
    assert tool.history_problems(path) == (
        ["empty.json: the table names no releases"],
        [],
    )
