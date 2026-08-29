"""Configuration precedence, strict parsing, and immutability.

Acceptance surface: the frozen precedence chain of
``docs/contracts/config.md`` (per-call argument > constructor argument >
explicit field > configuration file > environment variable > default),
the read-once environment rule, strict boolean parsing, and the
fail-closed configuration file.
"""

from __future__ import annotations

import ast
import dataclasses
import re
import sys
from pathlib import Path

import pytest

from toktier.config import (
    ENV_DIAGNOSTICS,
    ENV_HOME,
    ENV_LOG_LEVEL,
    ENV_OFFLINE,
    Config,
)
from toktier.errors import ConfigInvalid, ToktierError
from toktier.policy import RoutingPolicy

needs_tomllib = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="configuration file support needs the standard library TOML parser",
)


def write_config_file(home: Path, body: str) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "contracts" / "config.md"

_FIELD_ROW = re.compile(
    r"^\| `(?P<field>\w+)` \| (?P<type>[^|]+?) \| (?P<default>[^|]+?) \|$"
)

#: Stands for a default the table states in prose rather than as a value:
#: the directory layout of Section 5, which ``test_paths.py`` covers.
_DERIVED = object()


def _section(number: str) -> list[str]:
    """The lines of one numbered section of the contract document."""
    lines = CONTRACT.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"## {number}. "))
    rest = lines[start + 1 :]
    end = next((i for i, line in enumerate(rest) if line.startswith("## ")), len(rest))
    return rest[:end]


def contract_defaults() -> dict[str, object]:
    """Field name to documented default, as written in the field table."""
    defaults: dict[str, object] = {}
    for line in _section("7"):
        match = _FIELD_ROW.match(line)
        if match is None:
            continue
        written = match.group("default").strip()
        if not (written.startswith("`") and written.endswith("`")):
            defaults[match.group("field")] = _DERIVED
            continue
        literal = written[1:-1]
        if match.group("type").strip() == "`RoutingPolicy`":
            defaults[match.group("field")] = RoutingPolicy[literal]
        else:
            defaults[match.group("field")] = ast.literal_eval(literal)
    return defaults


# -- layer 6: built-in defaults ---------------------------------------


def test_documented_defaults_are_the_defaults(isolated_home: Path) -> None:
    """The frozen field table is read back against a resolved config.

    The table is where a reader looks up what happens when nothing is
    set, so it is worth checking rather than trusting: a default that
    moves in a release and a row that stays behind is exactly the drift
    this compares away.
    """
    documented = contract_defaults()
    config = Config.resolve()

    assert set(documented) == {field.name for field in dataclasses.fields(Config)}
    # The three directory rows are prose; every other row states a value.
    stated = {
        name: value for name, value in documented.items() if value is not _DERIVED
    }
    assert set(stated) == {
        "offline",
        "log_level",
        "disable_gpu",
        "diagnostics",
        "routing_policy",
    }
    for name, expected in stated.items():
        assert getattr(config, name) == expected, name


def test_defaults_when_nothing_is_set(isolated_home: Path) -> None:
    config = Config.resolve()

    assert config.home is None
    assert config.offline is False
    assert config.disable_gpu is False
    assert config.diagnostics is False
    assert config.cache_dir.name == "toktier"
    assert config.state_dir.name == "toktier"
    assert config.cache_dir != config.state_dir


def test_cache_and_state_are_distinct_directories(isolated_home: Path) -> None:
    config = Config.resolve()

    assert config.cache_dir != config.state_dir
    assert not str(config.state_dir).startswith(str(config.cache_dir))


# -- layer 5: environment ---------------------------------------------


@pytest.mark.parametrize("word", ["maybe", "2", "", "TRUE!", "y"])
def test_strict_boolean_rejects_anything_else(
    word: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_DIAGNOSTICS, word)

    with pytest.raises(ConfigInvalid) as caught:
        Config.resolve()

    error = caught.value
    assert error.code == "CONFIG_INVALID"
    assert error.details["field"] == "diagnostics"
    assert error.details["value"] == word
    assert error.details["source"] == ENV_DIAGNOSTICS
    assert isinstance(error, ToktierError)


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_LOG_LEVEL, "chatty")

    with pytest.raises(ConfigInvalid) as caught:
        Config.resolve()

    assert caught.value.details["field"] == "log_level"
    assert caught.value.details["source"] == ENV_LOG_LEVEL


# -- layer 4: configuration file --------------------------------------


@needs_tomllib
def test_configuration_file_beats_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "toktier-home"
    write_config_file(home, "offline = false\n")
    monkeypatch.setenv(ENV_HOME, str(home))
    monkeypatch.setenv(ENV_OFFLINE, "1")

    config = Config.resolve()

    assert config.offline is False


@needs_tomllib
def test_explicit_field_beats_configuration_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "toktier-home"
    write_config_file(home, "offline = false\n")
    monkeypatch.setenv(ENV_HOME, str(home))

    assert Config.resolve(offline=True).offline is True


@needs_tomllib
def test_constructor_none_defers_to_the_layer_below(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constructor argument that was not supplied must not win.

    This is how layer 2 is wired: callers forward their optional
    argument, and ``None`` means "not provided".
    """
    home = tmp_path / "toktier-home"
    write_config_file(home, "routing_policy = 'reference'\n")
    monkeypatch.setenv(ENV_HOME, str(home))

    deferred = Config.resolve(routing_policy=None)
    explicit = Config.resolve(routing_policy=RoutingPolicy.REQUIRE_ACCELERATED)

    assert deferred.routing_policy is RoutingPolicy.REFERENCE
    assert explicit.routing_policy is RoutingPolicy.REQUIRE_ACCELERATED


@needs_tomllib
def test_configuration_file_accepts_the_auto_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "toktier-home"
    write_config_file(home, "routing_policy = 'auto'\n")
    monkeypatch.setenv(ENV_HOME, str(home))

    assert Config.resolve().routing_policy is RoutingPolicy.CERTIFIED


@needs_tomllib
def test_configuration_file_cannot_select_experimental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uncertified output is reachable only from an explicit call site."""
    home = tmp_path / "toktier-home"
    write_config_file(home, "routing_policy = 'experimental'\n")
    monkeypatch.setenv(ENV_HOME, str(home))

    with pytest.raises(ConfigInvalid) as caught:
        Config.resolve()

    assert caught.value.details["field"] == "routing_policy"
    # The same value set in code is accepted: the call site is visible.
    assert (
        Config(routing_policy=RoutingPolicy.EXPERIMENTAL).routing_policy
        is RoutingPolicy.EXPERIMENTAL
    )


@needs_tomllib
def test_unknown_configuration_file_key_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "toktier-home"
    write_config_file(home, "offlien = true\n")
    monkeypatch.setenv(ENV_HOME, str(home))

    with pytest.raises(ConfigInvalid) as caught:
        Config.resolve()

    assert caught.value.details["value"] == ["offlien"]


@needs_tomllib
def test_configuration_file_cannot_set_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "toktier-home"
    write_config_file(home, "home = '/elsewhere'\n")
    monkeypatch.setenv(ENV_HOME, str(home))

    with pytest.raises(ConfigInvalid) as caught:
        Config.resolve()

    assert caught.value.details["field"] == "home"


@needs_tomllib
def test_unparsable_configuration_file_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "toktier-home"
    write_config_file(home, "offline = \n")
    monkeypatch.setenv(ENV_HOME, str(home))

    with pytest.raises(ConfigInvalid) as caught:
        Config.resolve()

    assert caught.value.details["field"] == "config_file"


@needs_tomllib
def test_bad_boolean_in_configuration_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "toktier-home"
    write_config_file(home, "offline = 'sometimes'\n")
    monkeypatch.setenv(ENV_HOME, str(home))

    with pytest.raises(ConfigInvalid) as caught:
        Config.resolve()

    assert caught.value.details["field"] == "offline"


def test_configuration_file_is_read_on_every_supported_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3.11 and newer use tomllib; 3.10 uses the tomli dependency."""
    home = tmp_path / "toktier-home"
    write_config_file(home, "offline = true\n")
    monkeypatch.setenv(ENV_HOME, str(home))

    assert Config.resolve().offline is True


def test_absent_configuration_file_is_fine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "toktier-home"
    home.mkdir()
    monkeypatch.setenv(ENV_HOME, str(home))

    assert Config.resolve().offline is False


# -- layer 3: explicit fields and immutability ------------------------


def test_direct_construction_derives_directories_from_home(
    tmp_path: Path,
) -> None:
    config = Config(home=tmp_path / "toktier-home")

    assert config.cache_dir == tmp_path / "toktier-home" / "cache"
    assert config.state_dir == tmp_path / "toktier-home" / "state"


def test_resolve_derives_directories_from_home(tmp_path: Path) -> None:
    config = Config.resolve(home=tmp_path / "toktier-home")

    assert config.cache_dir == tmp_path / "toktier-home" / "cache"
    assert config.state_dir == tmp_path / "toktier-home" / "state"


def test_explicit_directories_are_kept(tmp_path: Path) -> None:
    config = Config(home=tmp_path / "h", cache_dir=tmp_path / "c")

    assert config.cache_dir == tmp_path / "c"
    assert config.state_dir == tmp_path / "h" / "state"


def test_explicit_platform_default_cache_directory_is_kept(tmp_path: Path) -> None:
    platform_default = Config().cache_dir
    home = tmp_path / "h"

    config = Config(home=home, cache_dir=platform_default)

    assert config.cache_dir == platform_default
    assert config.cache_dir != home / "cache"


def test_string_paths_are_accepted(tmp_path: Path) -> None:
    """Typed callers pass a ``Path``; a string is coerced, not misread."""
    config = Config(home=str(tmp_path / "toktier-home"))  # type: ignore[arg-type]

    assert config.home == tmp_path / "toktier-home"


def test_config_is_immutable(tmp_path: Path) -> None:
    config = Config(home=tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.offline = True  # type: ignore[misc]


def test_replace_returns_a_new_object(tmp_path: Path) -> None:
    config = Config(home=tmp_path)
    derived = config.replace(offline=True)

    assert config.offline is False
    assert derived.offline is True
    assert derived.home == config.home
    assert derived is not config


def test_replace_home_keeps_resolved_directories(tmp_path: Path) -> None:
    config = Config(home=tmp_path / "first")

    derived = config.replace(home=tmp_path / "second")

    assert derived.home == tmp_path / "second"
    assert derived.cache_dir == config.cache_dir
    assert derived.state_dir == config.state_dir


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ConfigInvalid) as caught:
        Config.resolve(cache_size=17)

    assert caught.value.details["field"] == "config"


def test_home_treats_an_empty_value_as_unset(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path, tmp_path: Path
) -> None:
    """Three states of one variable, because two of them look alike.

    An empty value is unset, the way the XDG variables already read it
    and the way the directory contract states it. Reading it as "set to
    the empty path" would put the whole footprint on a relative path
    under whatever the working directory happened to be.
    """
    unset = Config.resolve()
    assert unset.home is None
    assert unset.cache_dir.is_absolute()

    monkeypatch.setenv(ENV_HOME, "")
    empty = Config.resolve()
    assert empty.home is None
    assert empty.cache_dir == unset.cache_dir
    assert empty.state_dir == unset.state_dir
    assert empty.cache_dir.is_absolute()

    named = tmp_path / "named-home"
    monkeypatch.setenv(ENV_HOME, str(named))
    home = Config.resolve()
    assert home.home == named
    assert home.cache_dir == named / "cache"
    assert home.state_dir == named / "state"


def test_an_undeterminable_home_is_refused_rather_than_guessed(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """No home, no guess.

    With ``HOME`` empty and no XDG override, ``~`` expands to nothing and
    the platform conventions land at the filesystem root
    (``/.cache/toktier``). Reporting that as this machine's layout, and
    then meeting a bare ``PermissionError`` on the way to creating it,
    describes a machine that does not exist. The remedy is named while
    naming it still helps.
    """
    monkeypatch.setenv("HOME", "")
    monkeypatch.setenv("USERPROFILE", "")

    with pytest.raises(ConfigInvalid) as caught:
        Config.resolve()

    assert caught.value.details["field"] == "home"
    assert ENV_HOME in str(caught.value.details["remedy"])


def test_an_xdg_override_needs_no_home(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path, tmp_path: Path
) -> None:
    """The refusal is about having nowhere to hang the layout, not about
    ``HOME`` itself: an explicit XDG root answers the question."""
    monkeypatch.setenv("HOME", "")
    monkeypatch.setenv("USERPROFILE", "")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))

    config = Config.resolve()

    assert config.cache_dir == tmp_path / "xdg-cache" / "toktier"
    assert config.state_dir == tmp_path / "xdg-state" / "toktier"
