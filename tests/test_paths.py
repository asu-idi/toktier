"""Private-directory creation: modes, and the error contract around it.

Acceptance surface: ``docs/contracts/config.md`` section 5 (0700 for
every directory this layer creates) and ``docs/contracts/errors.md``
section 4 (a failed command reports a code, never a traceback).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from toktier.errors import ConfigInvalid
from toktier.paths import DIRECTORY_MODE, ensure_private_dir, private_dir_problem


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_every_directory_created_here_is_owner_only(tmp_path: Path) -> None:
    """Including the intermediate ones.

    ``Path.mkdir(parents=True, mode=...)`` applies the mode to the last
    component only, which used to leave a fresh cache root as
    ``0775/0775/0700``.
    """
    leaf = tmp_path / "home" / "cache" / "artifacts" / ".locks"

    assert ensure_private_dir(leaf) == leaf

    for created in (leaf, *list(leaf.parents)[:3]):
        assert _mode(created) == DIRECTORY_MODE, created


def test_the_umask_does_not_get_a_vote(tmp_path: Path) -> None:
    previous = os.umask(0o077)
    try:
        os.umask(0o200)
        leaf = tmp_path / "masked" / "cache"
        ensure_private_dir(leaf)
    finally:
        os.umask(previous)

    assert _mode(leaf) == DIRECTORY_MODE
    assert _mode(leaf.parent) == DIRECTORY_MODE


def test_a_directory_already_there_keeps_the_operator_s_mode(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "operator"
    existing.mkdir(mode=0o755)
    os.chmod(existing, 0o755)

    ensure_private_dir(existing / "cache")

    assert _mode(existing) == 0o755
    assert _mode(existing / "cache") == DIRECTORY_MODE


def test_a_root_that_is_a_file_is_a_configuration_answer(
    tmp_path: Path,
) -> None:
    """Not a ``NotADirectoryError`` from three frames further in."""
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("taken", encoding="utf-8")

    with pytest.raises(ConfigInvalid) as caught:
        ensure_private_dir(occupied / "cache" / "artifacts")

    assert caught.value.code == "CONFIG_INVALID"
    assert caught.value.details["cause"] == "NotADirectoryError"
    assert "TOKTIER_HOME" in str(caught.value.details["remedy"])


def test_a_final_path_that_is_a_file_is_named_as_such(tmp_path: Path) -> None:
    occupied = tmp_path / "leaf"
    occupied.write_text("taken", encoding="utf-8")

    with pytest.raises(ConfigInvalid) as caught:
        ensure_private_dir(occupied)

    assert caught.value.details["value"] == str(occupied)
    assert "not a directory" in str(caught.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root may write anywhere")
def test_a_root_this_user_cannot_write_is_a_configuration_answer(
    tmp_path: Path,
) -> None:
    closed = tmp_path / "closed"
    closed.mkdir(mode=0o500)
    os.chmod(closed, 0o500)

    with pytest.raises(ConfigInvalid) as caught:
        ensure_private_dir(closed / "cache")

    assert caught.value.details["cause"] == "PermissionError"
    assert caught.value.details["cause_message"]


# -- the reading half, for diagnostics ----------------------------------


def test_a_location_that_can_hold_private_state_has_no_problem(
    tmp_path: Path,
) -> None:
    assert private_dir_problem(tmp_path / "home" / "cache") is None
    ensure_private_dir(tmp_path / "home" / "cache")
    assert private_dir_problem(tmp_path / "home" / "cache") is None


def test_a_root_that_is_a_file_is_named_without_creating_anything(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("taken", encoding="utf-8")

    problem = private_dir_problem(occupied / "cache" / "artifacts")

    assert problem is not None
    assert str(occupied) in problem
    assert "is not a directory" in problem
    # Nothing was made on the way to finding out.
    assert occupied.is_file()
    assert occupied.read_text(encoding="utf-8") == "taken"
    assert "cache" not in {path.name for path in tmp_path.iterdir()}


def test_a_symlink_to_a_file_is_named_the_same_way(tmp_path: Path) -> None:
    occupied = tmp_path / "target"
    occupied.write_text("taken", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(occupied)

    problem = private_dir_problem(link / "cache")

    assert problem is not None
    assert "is not a directory" in problem


@pytest.mark.skipif(os.geteuid() == 0, reason="root may write anywhere")
def test_a_root_this_user_cannot_write_is_named(tmp_path: Path) -> None:
    closed = tmp_path / "closed"
    closed.mkdir(mode=0o500)
    os.chmod(closed, 0o500)

    problem = private_dir_problem(closed / "cache")

    assert problem is not None
    assert "cannot be written by this user" in problem


def test_the_reading_and_the_creating_halves_agree(tmp_path: Path) -> None:
    """Whatever one refuses, the other names, and the other way round."""
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("taken", encoding="utf-8")
    leaf = occupied / "cache" / "artifacts"

    assert private_dir_problem(leaf) is not None
    with pytest.raises(ConfigInvalid):
        ensure_private_dir(leaf)

    fine = tmp_path / "home" / "cache"
    assert private_dir_problem(fine) is None
    assert ensure_private_dir(fine) == fine
