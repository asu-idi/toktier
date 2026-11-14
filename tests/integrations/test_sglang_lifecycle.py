"""The SGLang adapter's lifecycle: what the wrapper says about itself, when
it gives way to the supplied tokenizer, and how its pool and its encoder
state are kept and closed.

None of these tests asks for different ids: each one checks that a call
returns what the supplied tokenizer (or the stub it stands for) returns,
whatever the wrapper did around it.
"""

from __future__ import annotations

from typing import Any

import pytest

from toktier.integrations.sglang import (
    AdapterConfig,
    SGLangToktierTokenizer,
    wrap_tokenizer,
)

from .test_sglang_adapter import _FakeToktier, _StubHf

# -- the wrapper says whether it accelerates -------------------------------


class _ExternalToktier(_FakeToktier):
    """A family toktier knows only as external: the reference engine alone."""

    family = "external_0123456789abcdef"

    class plan:
        backend = "hf"


def test_accelerates_is_true_on_an_accelerated_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="off"))
    assert w.accelerates is True
    assert w.stats()["accelerates"] is True
    assert "accelerates=True" in repr(w)
    # is_fast keeps the Hugging Face meaning
    assert w.is_fast is True
    assert w.encode("ab", add_special_tokens=False) == [97, 98]


def test_accelerates_is_false_in_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="passthrough"))
    assert w.accelerates is False
    assert w.stats()["accelerates"] is False
    assert "accelerates=False" in repr(w)
    assert w.is_fast is True
    # a supplied object toktier declines: passthrough for good, not accelerated
    drifted = _StubHf()
    drifted.backend_tokenizer.add_tokens(["toktierprobeunique"])
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: _FakeToktier()
    )
    w2 = wrap_tokenizer(drifted, AdapterConfig(session_key="off"))
    assert w2.identity["status"] == "mismatch" and w2.accelerates is False


def test_accelerates_is_false_for_an_external_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = _ExternalToktier()
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: external
    )
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="auto"))
    # admitted and answering through toktier, but on the reference engine alone
    assert w.identity["status"] == "equivalent"
    assert w.planned_backend == "hf" and w.toktier is external
    assert w.accelerates is False
    assert w.stats()["accelerates"] is False
    assert "accelerates=False" in repr(w)
    assert w.encode("ab", add_special_tokens=False) == [97, 98]


def test_accelerates_follows_a_later_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    hf = _StubHf()
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off"))
    assert w.accelerates is True
    w.add_tokens(["toktierprobeunique"])
    assert w.encode("ab", add_special_tokens=False) == hf.encode(
        "ab", add_special_tokens=False
    )
    assert w.stats()["accelerates"] is False
    assert not w.accelerates


def test_the_start_up_record_carries_accelerates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import json

    log = tmp_path / "timing.jsonl"
    external = _ExternalToktier()
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: external
    )
    w = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="off", timing_log=str(log)))
    w.flush()
    first = json.loads(log.read_text().splitlines()[0])
    assert first["kind"] == "init" and first["accelerates"] is False
    w.close()


# -- a tokenizer toktier cannot resolve: passthrough, not a stop ------------


def _offline_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """A fresh toktier home, offline: nothing is fetched, nothing is cached."""
    home = tmp_path / "toktier-home"
    home.mkdir()
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _saved_fast_tokenizer(directory: Any) -> Any:
    """A real fast tokenizer saved to and loaded from a local directory, as
    SGLang loads one for a local ``--tokenizer-path``."""
    from transformers import PreTrainedTokenizerFast

    from .test_sglang_adapter import _tiny_tokenizer

    PreTrainedTokenizerFast(tokenizer_object=_tiny_tokenizer()).save_pretrained(
        directory
    )
    return PreTrainedTokenizerFast.from_pretrained(str(directory))


def _assert_unresolved_passthrough(w: Any, hf: Any) -> None:
    assert isinstance(w, SGLangToktierTokenizer)
    assert w.toktier is None and w.accelerates is False
    assert w.identity["status"] == "unresolved"
    assert w.identity["reason"]
    stats = w.stats()
    assert stats["mode"] == "passthrough"
    assert stats["identity"]["status"] == "unresolved"
    for text in ("ab c", "abc ab", ""):
        assert w.encode(text, add_special_tokens=False) == hf.encode(
            text, add_special_tokens=False
        )
        assert w(text)["input_ids"] == hf(text)["input_ids"]


def _one_warning(caplog: pytest.LogCaptureFixture) -> str:
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "toktier.integrations.sglang" and r.levelname == "WARNING"
    ]
    assert len(lines) == 1, lines
    line: str = lines[0]
    assert "TOKTIER_SGLANG_FAMILY" in line and "TOKTIER_SGLANG_REPO" in line
    assert "toktier artifacts fetch" in line
    return line


def test_a_local_directory_is_served_in_passthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # The review's S4 and S5: --tokenizer-path names a local directory.
    _offline_home(monkeypatch, tmp_path)
    hf = _saved_fast_tokenizer(tmp_path / "model-dir")
    assert hf.name_or_path == str(tmp_path / "model-dir")
    caplog.set_level("WARNING", logger="toktier.integrations.sglang")
    w = wrap_tokenizer(hf, AdapterConfig.from_env({}))
    _assert_unresolved_passthrough(w, hf)
    assert "ArtifactNotFound" in w.identity["reason"]
    assert str(tmp_path / "model-dir") in _one_warning(caplog)


def test_a_certified_hub_id_with_an_empty_offline_cache_is_served_in_passthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # The review's S9: a certified repository, an offline host, toktier's own
    # artifact cache empty. The same holds for a configured family.
    _offline_home(monkeypatch, tmp_path)
    hf = _saved_fast_tokenizer(tmp_path / "model-dir")
    hf.name_or_path = "Qwen/Qwen3-8B"
    caplog.set_level("WARNING", logger="toktier.integrations.sglang")
    w = wrap_tokenizer(hf, AdapterConfig())
    _assert_unresolved_passthrough(w, hf)
    _one_warning(caplog)
    caplog.clear()
    w2 = wrap_tokenizer(hf, AdapterConfig(family="qwen3_8b"))
    _assert_unresolved_passthrough(w2, hf)
    _one_warning(caplog)


def test_a_tokenizer_without_a_name_is_served_in_passthrough(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The review's S16: a tokenizer object that carries no name_or_path
    # (SGLang's MistralCommonTokenizer for a repository without
    # tokenizer.json) and no family or repository configured.
    hf = _StubHf()
    hf.name_or_path = None  # type: ignore[assignment]
    caplog.set_level("WARNING", logger="toktier.integrations.sglang")
    w = wrap_tokenizer(hf, AdapterConfig())
    _assert_unresolved_passthrough(w, hf)
    assert "name_or_path" in w.identity["reason"]
    _one_warning(caplog)


class _StrictCall(_StubHf):
    """A tokenizer whose call refuses keywords outside its own signature, as
    SGLang's MistralCommonBackend does (``text_pair`` and
    ``return_token_type_ids`` among them, even when ``None``)."""

    def __call__(
        self, text: Any, add_special_tokens: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        if kwargs:
            raise ValueError(f"Kwargs {list(kwargs)} are not supported")
        return super().__call__(text, add_special_tokens=add_special_tokens)


def test_passthrough_forwards_only_the_keywords_the_caller_passed() -> None:
    hf = _StrictCall(bos=1)
    hf.name_or_path = None  # type: ignore[assignment]
    w = wrap_tokenizer(hf, AdapterConfig())
    assert w.identity["status"] == "unresolved"
    assert w("ab")["input_ids"] == [1, 97, 98]
    assert w(["ab", "c"], add_special_tokens=False)["input_ids"] == [[97, 98], [99]]
    assert w.encode("ab") == [1, 97, 98]
    # a keyword the caller does pass still reaches the tokenizer
    with pytest.raises(ValueError, match="return_token_type_ids"):
        w("ab", return_token_type_ids=True)


def test_an_invalid_setting_still_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from toktier.errors import ArtifactNotFound

    _offline_home(monkeypatch, tmp_path)
    hf = _saved_fast_tokenizer(tmp_path / "model-dir")
    monkeypatch.setenv("TOKTIER_SGLANG_SESSION_KEY", "bogus")
    with pytest.raises(ValueError, match="session_key"):
        wrap_tokenizer(hf)
    monkeypatch.delenv("TOKTIER_SGLANG_SESSION_KEY")
    monkeypatch.setenv("TOKTIER_SGLANG_REQUIRE_ARTIFACT", "maybe")
    with pytest.raises(ValueError, match="TOKTIER_SGLANG_REQUIRE_ARTIFACT"):
        wrap_tokenizer(hf)
    monkeypatch.delenv("TOKTIER_SGLANG_REQUIRE_ARTIFACT")
    # a family name toktier does not know is a setting, not a resolution
    with pytest.raises(ArtifactNotFound, match="unknown tokenizer family"):
        wrap_tokenizer(hf, AdapterConfig(family="no_such_family"))


def test_require_artifact_restores_the_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from toktier.errors import ArtifactNotFound

    _offline_home(monkeypatch, tmp_path)
    hf = _saved_fast_tokenizer(tmp_path / "model-dir")
    assert AdapterConfig.from_env({}).require_artifact is False
    strict = AdapterConfig.from_env({"TOKTIER_SGLANG_REQUIRE_ARTIFACT": "1"})
    assert strict.require_artifact is True
    with pytest.raises(ArtifactNotFound):
        wrap_tokenizer(hf, strict)
    nameless = _StubHf()
    nameless.name_or_path = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="name_or_path"):
        wrap_tokenizer(nameless, strict)


def test_other_load_errors_still_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    from toktier.errors import ArtifactHashMismatch

    def corrupt(self: Any) -> Any:
        raise ArtifactHashMismatch("cached artifact failed verification")

    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", corrupt)
    with pytest.raises(ArtifactHashMismatch):
        wrap_tokenizer(_StubHf(), AdapterConfig())
