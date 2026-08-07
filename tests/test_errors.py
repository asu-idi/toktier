"""Structured error codes and the ``.details`` payload.

Acceptance surface: the frozen, append-only code table of
``docs/contracts/errors.md``, the ``.code`` attribute,
and the read-only machine-readable ``.details`` mapping.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from toktier import errors
from toktier.errors import ERROR_CLASSES_BY_CODE, ConfigInvalid, ToktierError

CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "contracts" / "errors.md"

_TABLE_ROW = re.compile(r"^\| `(?P<name>\w+)` \| `(?P<code>[A-Z][A-Z0-9_]*)` \|")


def contract_rows() -> dict[str, str]:
    """Class name to code, as written in the frozen contract table."""
    rows: dict[str, str] = {}
    for line in CONTRACT.read_text(encoding="utf-8").splitlines():
        match = _TABLE_ROW.match(line)
        if match is not None:
            rows[match.group("name")] = match.group("code")
    return rows


def test_code_table_matches_the_contract_document() -> None:
    rows = contract_rows()

    assert rows, "no code table rows found in the contract document"
    assert {code for code in rows.values()} == set(ERROR_CLASSES_BY_CODE)
    for name, code in rows.items():
        cls = getattr(errors, name)
        assert code == cls.CODE
        assert ERROR_CLASSES_BY_CODE[code] is cls


@pytest.mark.parametrize("code", sorted(ERROR_CLASSES_BY_CODE))
def test_every_error_is_an_exception_with_its_message(code: str) -> None:
    cls = ERROR_CLASSES_BY_CODE[code]
    error = cls("something went wrong")

    assert isinstance(error, Exception)
    assert str(error) == "something went wrong"


@pytest.mark.parametrize("code", sorted(ERROR_CLASSES_BY_CODE))
def test_details_default_to_an_empty_mapping(code: str) -> None:
    error = ERROR_CLASSES_BY_CODE[code]("no detail")

    assert dict(error.details) == {}


def test_details_are_copied_from_the_callers_mapping() -> None:
    payload = {"field": "offline", "value": "maybe", "source": "TOKTIER_OFFLINE"}
    error = ConfigInvalid("bad boolean", details=payload)

    payload["field"] = "mutated"
    assert error.details == {
        "field": "offline",
        "value": "maybe",
        "source": "TOKTIER_OFFLINE",
    }


def test_catching_the_base_class_and_switching_on_the_code() -> None:
    """The documented forward-compatible consumer pattern."""
    try:
        raise ConfigInvalid("bad boolean", details={"field": "offline"})
    except ToktierError as error:
        assert error.code == "CONFIG_INVALID"
        assert error.details.get("field") == "offline"
