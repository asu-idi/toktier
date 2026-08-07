from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[2] / "src"


def test_config_reads_toktier_environment_once_at_construction(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """docs/contracts/config.md permits TOKTIER_* reads only at construction."""
    monkeypatch.syspath_prepend(str(SRC))
    from toktier import Config

    first_home = tmp_path / "first"
    second_home = tmp_path / "second"
    initial = {
        "TOKTIER_HOME": str(first_home),
        "TOKTIER_OFFLINE": "1",
        "TOKTIER_LOG_LEVEL": "INFO",
        "TOKTIER_DISABLE_GPU": "1",
        "TOKTIER_DIAGNOSTICS": "1",
    }
    changed = {
        "TOKTIER_HOME": str(second_home),
        "TOKTIER_OFFLINE": "0",
        "TOKTIER_LOG_LEVEL": "ERROR",
        "TOKTIER_DISABLE_GPU": "0",
        "TOKTIER_DIAGNOSTICS": "0",
    }
    for name, value in initial.items():
        monkeypatch.setenv(name, value)

    config = Config.resolve()
    for name, value in changed.items():
        monkeypatch.setenv(name, value)

    assert (config.home, config.offline, config.log_level) == (
        first_home,
        True,
        "INFO",
    )
    assert (config.disable_gpu, config.diagnostics) == (True, True)
    assert config.cache_dir == first_home / "cache"
    assert config.state_dir == first_home / "state"

    refreshed = Config.resolve()
    assert (refreshed.home, refreshed.offline, refreshed.log_level) == (
        second_home,
        False,
        "ERROR",
    )
    assert (refreshed.disable_gpu, refreshed.diagnostics) == (False, False)
