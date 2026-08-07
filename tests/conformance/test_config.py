"""Conformance checks for frozen configuration resolution rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

LONG_TERM_ENVIRONMENT = (
    "TOKTIER_HOME",
    "TOKTIER_OFFLINE",
    "TOKTIER_LOG_LEVEL",
    "TOKTIER_DISABLE_GPU",
    "TOKTIER_DIAGNOSTICS",
)

BOOLEAN_ENVIRONMENT_FIELDS = (
    ("TOKTIER_OFFLINE", "offline"),
    ("TOKTIER_DISABLE_GPU", "disable_gpu"),
    ("TOKTIER_DIAGNOSTICS", "diagnostics"),
)


def _clear_long_term_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LONG_TERM_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def test_explicit_file_environment_and_default_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    installed_package: Any,
) -> None:
    """Exercise precedence layers three through six on one shared field."""

    _clear_long_term_environment(monkeypatch)
    home = tmp_path / "configured-home"
    home.mkdir()
    config_file = home / "config.toml"
    # Sentinels must be valid logging level names: the implementation may
    # validate log_level values, and the contract does not forbid that.
    config_file.write_text('log_level = "INFO"\n', encoding="utf-8")
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_LOG_LEVEL", "ERROR")

    explicit_and_file = installed_package.json_output(
        """
import json
from toktier.config import Config

print(json.dumps({
    "explicit": Config.resolve(log_level="DEBUG").log_level,
    "file": Config.resolve().log_level,
}))
"""
    )
    assert explicit_and_file == {"explicit": "DEBUG", "file": "INFO"}

    config_file.unlink()
    environment_value = installed_package.json_output(
        """
import json
from toktier.config import Config

print(json.dumps(Config.resolve().log_level))
"""
    )
    assert environment_value == "ERROR"

    monkeypatch.delenv("TOKTIER_LOG_LEVEL")
    default_value = installed_package.json_output(
        """
import json
from toktier.config import Config

print(json.dumps(Config.resolve().log_level))
"""
    )
    assert default_value == "WARNING"


def test_top_precedence_layers_have_the_frozen_call_shapes(
    installed_package: Any,
) -> None:
    """Layers one and two belong to Tokenizer.

    The ``add_special_tokens`` default follows the operative 0.x facade
    contract (``facade.md`` Section 2: core stream first, default
    ``False``); ``api.md`` Section 3 records ``True`` for the 1.0 shape
    and the delta is listed in ``facade.md`` Section 7.
    """

    observed = installed_package.json_output(
        """
import inspect
import json
import toktier

constructor = inspect.signature(toktier.Tokenizer).parameters
encode = inspect.signature(toktier.Tokenizer.encode).parameters
print(json.dumps({
    "config_default_is_none": constructor["config"].default is None,
    "policy_keyword_only": (
        constructor["policy"].kind is inspect.Parameter.KEYWORD_ONLY
    ),
    "policy_default_is_none": constructor["policy"].default is None,
    "special_tokens_keyword_only": (
        encode["add_special_tokens"].kind is inspect.Parameter.KEYWORD_ONLY
    ),
    "special_tokens_default": encode["add_special_tokens"].default,
}))
"""
    )
    assert observed == {
        "config_default_is_none": True,
        "policy_keyword_only": True,
        "policy_default_is_none": True,
        "special_tokens_keyword_only": True,
        "special_tokens_default": False,
    }


def test_environment_is_captured_once_per_config_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    installed_package: Any,
) -> None:
    _clear_long_term_environment(monkeypatch)
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "snapshot-home"))
    monkeypatch.setenv("TOKTIER_LOG_LEVEL", "INFO")
    monkeypatch.setenv("TOKTIER_OFFLINE", "yes")

    observed = installed_package.json_output(
        """
import json
import os
from toktier.config import Config

captured = Config.resolve()
os.environ["TOKTIER_LOG_LEVEL"] = "ERROR"
os.environ["TOKTIER_OFFLINE"] = "no"
reconstructed = Config.resolve()
print(json.dumps({
    "captured": [captured.log_level, captured.offline],
    "reconstructed": [reconstructed.log_level, reconstructed.offline],
}))
"""
    )
    assert observed == {
        "captured": ["INFO", True],
        "reconstructed": ["ERROR", False],
    }


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    (
        ("1", True),
        ("TrUe", True),
        ("YeS", True),
        ("oN", True),
        ("0", False),
        ("FaLsE", False),
        ("nO", False),
        ("OfF", False),
    ),
)
def test_documented_boolean_environment_values_are_case_insensitive(
    raw_value: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    installed_package: Any,
) -> None:
    _clear_long_term_environment(monkeypatch)
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "boolean-home"))
    for environment_name, _ in BOOLEAN_ENVIRONMENT_FIELDS:
        monkeypatch.setenv(environment_name, raw_value)

    observed = installed_package.json_output(
        """
import json
from toktier.config import Config

config = Config.resolve()
print(json.dumps({
    "offline": config.offline,
    "disable_gpu": config.disable_gpu,
    "diagnostics": config.diagnostics,
}))
"""
    )
    assert observed == {
        field_name: expected for _, field_name in BOOLEAN_ENVIRONMENT_FIELDS
    }


@pytest.mark.parametrize(
    "environment_name",
    tuple(name for name, _ in BOOLEAN_ENVIRONMENT_FIELDS),
)
def test_unrecognized_boolean_environment_value_raises_config_invalid(
    environment_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    installed_package: Any,
) -> None:
    _clear_long_term_environment(monkeypatch)
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "invalid-boolean-home"))
    monkeypatch.setenv(environment_name, "sometimes")

    observed = installed_package.json_output(
        """
import json
from toktier.config import Config
from toktier.errors import ConfigInvalid

try:
    Config.resolve()
except ConfigInvalid as error:
    print(json.dumps({"code": error.code}))
else:
    raise AssertionError("unrecognized boolean value was accepted")
"""
    )
    assert observed == {"code": "CONFIG_INVALID"}


def test_five_long_term_environment_variable_names_and_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    installed_package: Any,
) -> None:
    _clear_long_term_environment(monkeypatch)
    home = tmp_path / "long-term-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "true")
    monkeypatch.setenv("TOKTIER_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TOKTIER_DISABLE_GPU", "on")
    monkeypatch.setenv("TOKTIER_DIAGNOSTICS", "1")

    observed = installed_package.json_output(
        """
import json
from toktier.config import Config

config = Config.resolve()
print(json.dumps({
    "home": str(config.home),
    "cache_dir": str(config.cache_dir),
    "state_dir": str(config.state_dir),
    "offline": config.offline,
    "log_level": config.log_level,
    "disable_gpu": config.disable_gpu,
    "diagnostics": config.diagnostics,
}))
"""
    )
    assert observed == {
        "home": str(home),
        "cache_dir": str(home / "cache"),
        "state_dir": str(home / "state"),
        "offline": True,
        "log_level": "DEBUG",
        "disable_gpu": True,
        "diagnostics": True,
    }
