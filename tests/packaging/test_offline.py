import importlib
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast

SRC = Path(__file__).resolve().parents[2] / "src"
pytest: Any = importlib.import_module("pytest")


def _set_offline(monkeypatch: Any, home: Path) -> None:
    monkeypatch.syspath_prepend(str(SRC))
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _require_artifacts_module() -> None:
    try:
        importlib.import_module("toktier.artifacts")
    except ModuleNotFoundError as exc:
        if exc.name != "toktier.artifacts":
            raise
        pytest.skip("toktier.artifacts implementation lane has not merged yet")


def test_offline_config_construction_succeeds(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _set_offline(monkeypatch, tmp_path)
    from toktier import Config

    config = Config.resolve()

    assert config.offline is True
    assert config.home == tmp_path
    assert config.cache_dir == tmp_path / "cache"


def test_offline_missing_artifact_never_downloads(
    monkeypatch: Any, tmp_path: Path
) -> None:
    empty_home = tmp_path / "empty-home"
    (empty_home / "cache").mkdir(parents=True)
    _set_offline(monkeypatch, empty_home)
    socket_attempts: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class RejectSocket(socket.socket):
        def __new__(
            cls, *args: object, **kwargs: object
        ) -> NoReturn:
            socket_attempts.append((args, kwargs))
            raise AssertionError("offline artifact resolution attempted network access")

    monkeypatch.setattr(socket, "socket", RejectSocket)
    _require_artifacts_module()

    from toktier.errors import ArtifactNotFound

    toktier = importlib.import_module("toktier")
    config_type = toktier.Config
    tokenizer_type = getattr(toktier, "Tokenizer", None)
    if tokenizer_type is None:
        pytest.skip("Tokenizer is not merged yet (backend lane)")
    tokenizer_factory = cast(Callable[..., object], tokenizer_type)
    config = config_type.resolve()

    with pytest.raises(ArtifactNotFound) as raised:
        tokenizer_factory("qwen3_8b", config=config)

    assert raised.value.code == "ARTIFACT_NOT_FOUND"
    assert not socket_attempts
