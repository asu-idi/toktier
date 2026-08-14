"""Immutable configuration with a fixed precedence chain.

Contract reference: ``docs/contracts/config.md``. Frozen rules:

- ``Config`` is immutable; all fields are resolved at construction.
- Precedence (highest first): per-call method argument > constructor
  argument > explicit ``Config`` field > configuration file >
  environment variable > built-in default.
- Environment variables are read exactly once, when a ``Config`` is
  constructed. Later environment changes never affect existing objects.
- Long-term environment variables: ``TOKTIER_HOME``,
  ``TOKTIER_OFFLINE``, ``TOKTIER_LOG_LEVEL``, ``TOKTIER_DISABLE_GPU``,
  ``TOKTIER_DIAGNOSTICS``. There is no variable to skip hash
  verification or bypass certification checks; these are deliberately
  not configuration.

The two highest layers live with the caller and reach this module as
overrides: a per-call keyword argument wins over everything, and a
constructor argument is forwarded to :meth:`Config.resolve`, where
``None`` means "not provided, defer to the layer below". Layers three
to six (explicit field, configuration file, environment, default) are
resolved here.

Only the standard library is imported at module level. ``platformdirs``
is a declared dependency and is used for the platform-conventional
directories when it is importable; an XDG-shaped fallback keeps this
module usable in a bare standard-library environment.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from .errors import ConfigInvalid
from .policy import RoutingPolicy

__all__ = [
    "ENV_DIAGNOSTICS",
    "ENV_DISABLE_GPU",
    "ENV_HOME",
    "ENV_LOG_LEVEL",
    "ENV_OFFLINE",
    "Config",
]

ENV_HOME = "TOKTIER_HOME"
ENV_OFFLINE = "TOKTIER_OFFLINE"
ENV_LOG_LEVEL = "TOKTIER_LOG_LEVEL"
ENV_DISABLE_GPU = "TOKTIER_DISABLE_GPU"
ENV_DIAGNOSTICS = "TOKTIER_DIAGNOSTICS"

#: Application name used for the platform-conventional directories.
_APP_NAME = "toktier"

# Omitted-path sentinel; compare by identity, never by value.
_OMITTED = Path("")

_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})

#: Standard logging level names accepted for ``log_level``.
_LOG_LEVELS = frozenset(
    {"CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"}
)

#: Keys accepted in the configuration file; unknown keys fail closed.
_FILE_KEYS = frozenset(
    {"home", "offline", "log_level", "disable_gpu", "diagnostics", "routing_policy"}
)

#: Boolean-valued knobs shared by the environment and the file.
_BOOL_FIELDS = ("offline", "disable_gpu", "diagnostics")


def _parse_bool(raw: str, *, field_name: str, source: str) -> bool:
    """Strict boolean parsing; anything unrecognized raises ConfigInvalid."""
    word = raw.strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise ConfigInvalid(
        f"cannot interpret {raw!r} as a boolean for {field_name}",
        details={"field": field_name, "value": raw, "source": source},
    )


def _coerce_bool(value: object, *, field_name: str, source: str) -> bool:
    """Accept a real boolean, or a string parsed by the strict rules."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _parse_bool(value, field_name=field_name, source=source)
    raise ConfigInvalid(
        f"cannot interpret {value!r} as a boolean for {field_name}",
        details={"field": field_name, "value": repr(value), "source": source},
    )


def _coerce_path(value: object, *, field_name: str, source: str) -> Path:
    """Accept a path-like value; anything else is a configuration error."""
    if isinstance(value, Path):
        return value
    if isinstance(value, (str, os.PathLike)):
        return Path(os.fspath(value))
    raise ConfigInvalid(
        f"cannot interpret {value!r} as a path for {field_name}",
        details={"field": field_name, "value": repr(value), "source": source},
    )


def _coerce_log_level(value: object, *, source: str) -> str:
    """Validate a standard logging level name, normalized to upper case."""
    if isinstance(value, str):
        name = value.strip().upper()
        if name in _LOG_LEVELS:
            return name
    raise ConfigInvalid(
        f"{value!r} is not a standard logging level name",
        details={"field": "log_level", "value": repr(value), "source": source},
    )


def _coerce_policy(
    value: object, *, source: str, allow_experimental: bool
) -> RoutingPolicy:
    """Resolve a routing policy value, honoring the ``auto`` alias.

    ``allow_experimental`` is false for values that arrive from the
    configuration file: ``EXPERIMENTAL`` is reachable only through an
    explicit construction parameter, never through a file or an
    environment variable (``routing.md`` section 1).
    """
    if isinstance(value, (RoutingPolicy, str)):
        try:
            policy = RoutingPolicy.coerce(value)
        except ValueError as exc:
            raise ConfigInvalid(
                f"unknown routing policy {value!r}",
                details={
                    "field": "routing_policy",
                    "value": repr(value),
                    "source": source,
                },
            ) from exc
    else:
        raise ConfigInvalid(
            f"unknown routing policy {value!r}",
            details={
                "field": "routing_policy",
                "value": repr(value),
                "source": source,
            },
        )
    if policy is RoutingPolicy.EXPERIMENTAL and not allow_experimental:
        raise ConfigInvalid(
            "the experimental routing policy is only selectable through an "
            "explicit construction parameter, not from configuration",
            details={
                "field": "routing_policy",
                "value": policy.value,
                "source": source,
            },
        )
    return policy


def _require_user_home(variable: str) -> None:
    """Refuse to guess when this process has no home directory.

    Without ``TOKTIER_HOME`` every convention in ``config.md`` section 5
    hangs off the user's home directory. When that cannot be determined
    -- an empty ``HOME`` in a container, a login without a passwd entry
    -- ``~`` expands to nothing and the conventions land at the
    filesystem root (``/.cache/toktier``). Reporting such a path as the
    layout, and then failing to create it with a bare ``PermissionError``
    somewhere further in, states something about this machine that is not
    true. Say so here instead, while there is still a remedy to name.

    ``variable`` is the XDG override that would have made a home
    unnecessary; a set one has already been taken, so this is only
    reached when there is nothing else to fall back on.
    """
    home = os.path.expanduser("~")
    if home and home != "~" and Path(home).is_absolute() and Path(home).parts[1:]:
        return
    raise ConfigInvalid(
        "no home directory could be determined for this user, so the "
        "platform-conventional cache and state directories cannot be "
        "resolved",
        details={
            "field": "home",
            "value": home,
            "source": "environment",
            "remedy": (
                f"set {ENV_HOME} to a directory this user can write "
                f"(or {variable} for the platform convention alone)"
            ),
        },
    )


def _platform_dir(kind: str) -> Path:
    """Platform-conventional user directory for ``kind`` (cache/state)."""
    module: ModuleType | None
    if not os.environ.get("XDG_CACHE_HOME" if kind == "cache" else "XDG_STATE_HOME"):
        _require_user_home(
            "XDG_CACHE_HOME" if kind == "cache" else "XDG_STATE_HOME"
        )
    try:
        module = importlib.import_module("platformdirs")
    except ModuleNotFoundError:
        module = None
    if module is not None:
        if kind == "cache":
            resolved = module.user_cache_dir(_APP_NAME, appauthor=False)
        else:
            resolved = module.user_state_dir(_APP_NAME, appauthor=False)
        return Path(cast("str", resolved))
    # Standard-library fallback: the XDG layout, which is what
    # platformdirs itself resolves to on Linux.
    if kind == "cache":
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    else:
        base = os.environ.get("XDG_STATE_HOME") or str(
            Path.home() / ".local" / "state"
        )
    return Path(base) / _APP_NAME


def _default_cache_dir(home: Path | None) -> Path:
    if home is not None:
        return home / "cache"
    return _platform_dir("cache")


def _default_state_dir(home: Path | None) -> Path:
    if home is not None:
        return home / "state"
    return _platform_dir("state")


def _load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file with tomllib or its Python 3.10 compatibility parser."""
    try:
        toml_module = importlib.import_module("tomllib")
    except ModuleNotFoundError:  # pragma: no cover - version-dependent
        toml_module = importlib.import_module("tomli")
    decode_error = cast("type[Exception]", toml_module.TOMLDecodeError)
    try:
        with open(path, "rb") as handle:
            data = toml_module.load(handle)
    except decode_error as exc:
        raise ConfigInvalid(
            f"cannot parse configuration file: {exc}",
            details={"field": "config_file", "value": str(path), "source": "file"},
        ) from exc
    if not isinstance(data, dict):  # pragma: no cover - tomllib always maps
        raise ConfigInvalid(
            "configuration file must contain a table",
            details={"field": "config_file", "value": str(path), "source": "file"},
        )
    return cast("dict[str, Any]", data)


def _read_config_file(home: Path | None) -> dict[str, Any]:
    """Read and validate ``config.toml`` under the resolved home.

    Unknown keys raise :class:`ConfigInvalid` (fail closed rather than
    silently ignoring typos). Values are coerced with the same strict
    rules the environment uses, so a typo fails at construction rather
    than becoming a silent default.

    Without an explicit home there is no file location to read: the
    configuration file lives under ``TOKTIER_HOME`` (``config.md``
    sections 5 and 6).
    """
    if home is None:
        return {}
    path = home / "config.toml"
    if not path.is_file():
        return {}
    data = _load_toml(path)
    source = str(path)
    unknown = set(data) - _FILE_KEYS
    if unknown:
        raise ConfigInvalid(
            f"unknown configuration file keys: {sorted(unknown)}",
            details={
                "field": "config_file",
                "value": sorted(unknown),
                "source": source,
            },
        )
    if "home" in data:
        raise ConfigInvalid(
            "'home' cannot be set from the configuration file "
            "(it decides where the file lives)",
            details={"field": "home", "value": data["home"], "source": source},
        )
    values: dict[str, Any] = {}
    for field_name in _BOOL_FIELDS:
        if field_name in data:
            values[field_name] = _coerce_bool(
                data[field_name], field_name=field_name, source=source
            )
    if "log_level" in data:
        values["log_level"] = _coerce_log_level(data["log_level"], source=source)
    if "routing_policy" in data:
        values["routing_policy"] = _coerce_policy(
            data["routing_policy"], source=source, allow_experimental=False
        )
    return values


@dataclass(frozen=True)
class Config:
    """Immutable resolved configuration (frozen field names).

    Instances never change after construction; derive modified copies
    with :meth:`replace`. A ``Tokenizer`` captures its ``Config`` at
    construction.
    """

    #: Root of toktier's on-disk footprint; ``None`` means platform
    #: conventions decide cache/state locations independently.
    home: Path | None = None
    #: Rebuildable cache (artifacts, built kernels).
    cache_dir: Path = dataclasses.field(default_factory=lambda: _OMITTED)
    #: Persistent state (session store). Deleting state loses sessions;
    #: state is not a cache.
    state_dir: Path = dataclasses.field(default_factory=lambda: _OMITTED)
    #: Never touch the network; artifacts must already be local and
    #: verified.
    offline: bool = False
    #: Standard logging level name for the library logger.
    log_level: str = "WARNING"
    #: Exclude GPU backends at plan time.
    disable_gpu: bool = False
    #: Enable extended diagnostics collection.
    diagnostics: bool = False
    #: Default routing policy.
    routing_policy: RoutingPolicy = RoutingPolicy.CERTIFIED

    def __post_init__(self) -> None:
        """Normalize and validate the resolved field values.

        Direct construction is part of the precedence chain (layer 3),
        so the same checks apply here as to values read from a file or
        the environment. Path-like values become :class:`Path`, the
        logging level is validated, and omitted ``cache_dir``/``state_dir``
        values follow an explicit ``home``.
        """
        home: Path | None = None
        if self.home is not None:
            home = _coerce_path(self.home, field_name="home", source="config")
            object.__setattr__(self, "home", home)

        cache_dir: object = self.cache_dir
        if cache_dir is _OMITTED:
            cache_dir = _default_cache_dir(home)
        else:
            cache_dir = _coerce_path(
                cache_dir, field_name="cache_dir", source="config"
            )
        object.__setattr__(self, "cache_dir", cache_dir)

        state_dir: object = self.state_dir
        if state_dir is _OMITTED:
            state_dir = _default_state_dir(home)
        else:
            state_dir = _coerce_path(
                state_dir, field_name="state_dir", source="config"
            )
        object.__setattr__(self, "state_dir", state_dir)

        object.__setattr__(
            self, "log_level", _coerce_log_level(self.log_level, source="config")
        )
        for field_name in _BOOL_FIELDS:
            flag: object = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                _coerce_bool(flag, field_name=field_name, source="config"),
            )
        policy: object = self.routing_policy
        object.__setattr__(
            self,
            "routing_policy",
            _coerce_policy(policy, source="config", allow_experimental=True),
        )

    def replace(self, **changes: Any) -> Config:
        """Return a new ``Config`` with the given fields replaced."""
        return dataclasses.replace(self, **changes)

    @classmethod
    def resolve(cls, **overrides: Any) -> Config:
        """Build a ``Config`` through the frozen precedence chain.

        ``overrides`` occupy the explicit-field layer (layer 3); the
        configuration file (layer 4), environment (layer 5), and
        built-in defaults (layer 6) fill the rest. An override whose
        value is ``None`` is treated as "not provided", which is how a
        constructor argument (layer 2) defers to the layers below. The
        environment is read exactly once, here.
        """
        env: Mapping[str, str] = os.environ

        # Layer 5: environment (read once).
        env_values: dict[str, Any] = {}
        # An empty value counts as unset, which is what the XDG variables
        # below already do (`or` in :func:`_platform_dir`) and what the
        # directory contract states. Reading it as "set to the empty
        # path" would place the whole footprint on a relative path under
        # the working directory.
        if env.get(ENV_HOME):
            env_values["home"] = Path(env[ENV_HOME])
        if ENV_OFFLINE in env:
            env_values["offline"] = _parse_bool(
                env[ENV_OFFLINE], field_name="offline", source=ENV_OFFLINE
            )
        if ENV_LOG_LEVEL in env:
            env_values["log_level"] = _coerce_log_level(
                env[ENV_LOG_LEVEL], source=ENV_LOG_LEVEL
            )
        if ENV_DISABLE_GPU in env:
            env_values["disable_gpu"] = _parse_bool(
                env[ENV_DISABLE_GPU], field_name="disable_gpu", source=ENV_DISABLE_GPU
            )
        if ENV_DIAGNOSTICS in env:
            env_values["diagnostics"] = _parse_bool(
                env[ENV_DIAGNOSTICS], field_name="diagnostics", source=ENV_DIAGNOSTICS
            )

        # Home resolves first: it decides where the configuration file
        # lives, so it can only come from an override or the
        # environment, never from the file itself.
        raw_home = overrides.get("home", env_values.get("home"))
        home = (
            _coerce_path(raw_home, field_name="home", source="resolve")
            if raw_home is not None
            else None
        )

        # Layer 4: configuration file under the resolved home.
        file_values = _read_config_file(home)

        merged: dict[str, Any] = {}
        merged.update(env_values)
        merged.update(file_values)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        merged["home"] = home

        unknown = set(merged) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ConfigInvalid(
                f"unknown configuration fields: {sorted(unknown)}",
                details={
                    "field": "config",
                    "value": sorted(unknown),
                    "source": "resolve",
                },
            )
        return cls(**merged)
