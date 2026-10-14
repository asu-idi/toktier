"""Unit tests for the SGLang adapter that need neither sglang nor an artifact.

The toktier path itself is exercised end to end by the PLAN/184 smoke
(shadow check against the Hugging Face stream on a live server); these
tests pin the pure-Python contract: delegation, the session-key plumbing,
the special-token template derivation, the identity gate, attribute
forwarding, the shadow reference, and the manager attachment. One test
at the end uses a real artifact when ``TOKTIER_TEST_ARTIFACTS`` names a
directory of local artifacts (the convention of ``tests/test_chunked.py``).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import toktier
from toktier import Config
from toktier.artifacts.manifest import ArtifactManifest
from toktier.artifacts.tables import ARTIFACT_MANIFEST
from toktier.integrations.sglang import (
    SESSION_KEY,
    AdapterConfig,
    SGLangToktierTokenizer,
    _derive_special_template,
    attach_to_tokenizer_manager,
    bind_session_key,
    check_core_identity,
    session_key_from_request,
    wrap_tokenizer,
)
from toktier.paths import artifact_cache_dir

_ARTIFACTS = os.environ.get("TOKTIER_TEST_ARTIFACTS")


def _tiny_tokenizer() -> Any:
    """A real (Rust) fast tokenizer with a five-entry vocabulary."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(
        models.BPE(
            vocab={"<unk>": 0, "a": 1, "b": 2, "ab": 3, "c": 4},
            merges=[("a", "b")],
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


class _StubHf:
    """A tiny tokenizer: one id per character; optional BOS/EOS template.

    Its ``backend_tokenizer`` is a real fast tokenizer whose serialization
    the fake toktier below reports as its loader face, so the identity
    gate finds the two equivalent unless a test changes one of them.
    """

    name_or_path = "stub/model"
    eos_token_id = 999
    is_fast = True

    def __init__(self, bos: int | None = None, eos: int | None = None) -> None:
        self._bos, self._eos = bos, eos
        self.backend_tokenizer = _tiny_tokenizer()

    split_special_tokens = False

    def _core(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def add_tokens(self, tokens: list[str]) -> int:
        added: int = self.backend_tokenizer.add_tokens(tokens)
        return added

    def __call__(
        self, text: Any, add_special_tokens: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        if isinstance(text, str):
            return {"input_ids": self._one(text, add_special_tokens), "kwargs": kwargs}
        return {
            "input_ids": [self._one(t, add_special_tokens) for t in text],
            "kwargs": kwargs,
        }

    def _one(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = self._core(text)
        if add_special_tokens:
            ids = (
                ([self._bos] if self._bos is not None else [])
                + ids
                + ([self._eos] if self._eos is not None else [])
            )
        return ids

    def encode(
        self, text: str, add_special_tokens: bool = True, **kwargs: Any
    ) -> list[int]:
        return self._one(text, add_special_tokens)

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "".join(chr(i) for i in ids)


def test_config_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError):
        AdapterConfig(session_key="nope")
    with pytest.raises(ValueError):
        AdapterConfig(shadow="sometimes")
    with pytest.raises(ValueError):
        AdapterConfig(shadow_rate=1.5)


def test_config_from_env_maps_names(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "TOKTIER_SGLANG_SESSION_KEY": "contextvar",
        "TOKTIER_SGLANG_SHADOW": "1",
        "TOKTIER_SGLANG_SHADOW_RATE": "0.05",
        "TOKTIER_SGLANG_FAMILY": "qwen3_8b",
        "TOKTIER_SGLANG_EXPLAIN_EVERY": "10",
    }
    cfg = AdapterConfig.from_env(env)
    assert cfg.session_key == "contextvar"
    assert cfg.shadow == "sync"
    assert cfg.shadow_rate == 0.05
    assert cfg.family == "qwen3_8b"
    assert cfg.explain_every == 10
    assert AdapterConfig.from_env({}).session_key == "auto"


def test_special_template_derivation() -> None:
    assert _derive_special_template(_StubHf()) == ([], [], True)
    assert _derive_special_template(_StubHf(bos=1)) == ([1], [], True)
    assert _derive_special_template(_StubHf(bos=1, eos=2)) == ([1], [2], True)


def test_passthrough_wrapper_delegates_everything() -> None:
    hf = _StubHf(bos=1)
    w = wrap_tokenizer(hf, AdapterConfig(session_key="passthrough"))
    assert isinstance(w, SGLangToktierTokenizer)
    assert w.is_fast is True
    assert w.eos_token_id == 999
    assert w("ab")["input_ids"] == [1, 97, 98]
    assert w(["ab", "c"])["input_ids"] == [[1, 97, 98], [1, 99]]
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    assert w.decode([97]) == "a"
    assert w.stats()["delegated"] == 3
    assert wrap_tokenizer(w) is w


class _Req:
    def __init__(self, **kw: Any) -> None:
        self.session_id = None
        self.session_params = None
        self.cache_salt = None
        self.rid = None
        self.__dict__.update(kw)


def test_session_key_precedence() -> None:
    assert session_key_from_request(_Req(session_id="s1", cache_salt="c1")) == "s1"
    assert session_key_from_request(_Req(cache_salt="c1")) == "c1"
    assert (
        session_key_from_request(_Req(rid="conv7-turn3"), rid_separator="-turn")
        == "conv7"
    )
    assert session_key_from_request(_Req(rid="conv7-turn3")) is None
    # server-side continuation sends a delta, not a transcript: no key
    assert (
        session_key_from_request(_Req(session_id="s1", session_params={"id": "s1"}))
        is None
    )


def test_bind_session_key_restores() -> None:
    assert SESSION_KEY.get() is None
    with bind_session_key("k"):
        assert SESSION_KEY.get() == "k"
    assert SESSION_KEY.get() is None


def test_attach_to_manager_wraps_and_binds() -> None:
    hf = _StubHf()
    seen: dict[str, str | None] = {}

    class Manager:
        def __init__(self) -> None:
            self.tokenizer: Any = hf
            self.async_dynamic_batch_tokenizer = None
            self.rid_to_state: dict[str, Any] = {}

        async def _tokenize_one_request(self, obj: Any) -> Any:
            seen["key"] = SESSION_KEY.get()
            return self.tokenizer(obj.text)["input_ids"]

    assert attach_to_tokenizer_manager(
        Manager, AdapterConfig(session_key="passthrough")
    )
    assert (
        attach_to_tokenizer_manager(Manager, AdapterConfig(session_key="passthrough"))
        is False
    )
    m = Manager()
    assert isinstance(m.tokenizer, SGLangToktierTokenizer)
    ids = asyncio.run(
        m._tokenize_one_request(_Req(text="hi", session_id="conv", rid="r"))
    )
    assert ids == [104, 105]
    assert seen["key"] == "conv"
    assert SESSION_KEY.get() is None


# -- chunk-parallel encode wiring (0.3.0) -----------------------------------


class _Ids:
    def __init__(self, ids: tuple[int, ...]) -> None:
        self.ids = ids


class _FakeToktier:
    """Stands in for ``toktier.Tokenizer`` inside the wrapper."""

    family = "fake_family"

    class plan:  # mirrors the attribute the wrapper reads
        backend = "fast_cpu"

    class _backend:  # the loader face the identity gate reads
        @staticmethod
        def materialized_tokenizer_json() -> str:
            face: str = _tiny_tokenizer().to_str()
            return face

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def encode(self, text: str, **kwargs: object) -> _Ids:
        self.calls.append(dict(kwargs))
        return _Ids(tuple(ord(c) for c in text))

    def explain(self, summary: bool = False) -> dict[str, object]:
        return {"store": None, "fallback_counts": {}, "chunked": {"calls": 1}}

    def close(self) -> None:
        self.closed = True


class _Chunked:
    def __init__(self, text: str, k: int | None) -> None:
        self.ids = tuple(ord(c) for c in text)
        self.chunks = 2
        self.k_requested = k or 4
        self.serial_reason = None


def test_config_chunked_fields_from_env_and_validation() -> None:
    cfg = AdapterConfig.from_env(
        {"TOKTIER_SGLANG_CHUNKED_MIN_CHARS": "4000", "TOKTIER_SGLANG_CHUNKED_K": "4"}
    )
    assert cfg.chunked_min_chars == 4000 and cfg.chunked_k == 4
    assert AdapterConfig.from_env({}).chunked_min_chars == 0
    assert AdapterConfig.from_env({}).chunked_k == 0
    with pytest.raises(ValueError):
        AdapterConfig(chunked_min_chars=-1)
    with pytest.raises(ValueError):
        AdapterConfig(chunked_k=-1)


def test_chunked_serves_long_stateless_requests_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    seen: list[tuple[str, int | None]] = []

    def chunked(tok: object, text: str, *, k: int | None = None) -> _Chunked:
        assert tok is fake
        seen.append((text, k))
        return _Chunked(text, k)

    monkeypatch.setattr("toktier.chunked.encode", chunked)
    long_text = "x" * 20
    # "off": short stays on the stateless route, long is chunked.
    w = wrap_tokenizer(
        _StubHf(), AdapterConfig(session_key="off", chunked_min_chars=10, chunked_k=4)
    )
    assert w.encode("short", add_special_tokens=False) == [ord(c) for c in "short"]
    assert fake.calls == [{"lookup": "off"}]
    assert w.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert seen == [(long_text, 4)]
    assert w.stats()["how"] == {"off": 1, "chunked": 1}
    assert w.stats()["chunked"] == {"calls": 1}
    # "contextvar": a keyed request stays on the session path whatever its
    # length; a keyless long one is chunked (the table, since chunked_k is 0).
    fake.calls.clear()
    seen.clear()
    w2 = wrap_tokenizer(
        _StubHf(), AdapterConfig(session_key="contextvar", chunked_min_chars=10)
    )
    with bind_session_key("conv"):
        assert w2.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert fake.calls == [{"session": "conv"}]
    assert w2.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert seen == [(long_text, None)]
    assert w2.encode("short", add_special_tokens=False) == [ord(c) for c in "short"]
    assert fake.calls[-1] == {}
    assert w2.stats()["how"] == {"session": 1, "chunked": 1, "auto": 1}
    # "auto": content lookup by design; chunking never fires.
    fake.calls.clear()
    seen.clear()
    w3 = wrap_tokenizer(
        _StubHf(), AdapterConfig(session_key="auto", chunked_min_chars=10)
    )
    assert w3.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert seen == [] and fake.calls == [{}]
    # Off by default: nothing changes for a wrapper without the option.
    seen.clear()
    w4 = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="off"))
    assert w4.encode(long_text, add_special_tokens=False) == [ord("x")] * 20
    assert seen == []


def test_explicit_long_request_encoder_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    monkeypatch.setattr(
        "toktier.chunked.encode",
        lambda *a, **k: pytest.fail("chunking must not run behind an explicit hook"),
    )
    w = wrap_tokenizer(
        _StubHf(),
        AdapterConfig(
            session_key="off",
            long_request_encoder=lambda text: [1, 2, 3],
            long_request_min_chars=5,
            chunked_min_chars=5,
        ),
    )
    assert w.encode("x" * 8, add_special_tokens=False) == [1, 2, 3]
    assert w.stats()["how"] == {"long_request_encoder": 1}


# -- identity gate (release-candidate review, finding 2) ----------------------


def test_identity_equivalent_accelerates_and_drift_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    hf = _StubHf()
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off"))
    assert w.identity == {"status": "equivalent", "reason": None, "sections": []}
    assert w.family == "fake_family" and w.toktier is fake
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    assert fake.calls == [{"lookup": "off"}]
    assert w.stats()["mode"] == "off" and w.stats()["identity"]["status"] == (
        "equivalent"
    )
    # A token added after loading: the supplied object is no longer the
    # tokenizer toktier loaded, so the wrapper delegates for good.
    drifted = _StubHf()
    drifted.backend_tokenizer.add_tokens(["toktierprobeunique"])
    fake2 = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake2)
    w2 = wrap_tokenizer(drifted, AdapterConfig(session_key="off"))
    assert w2.identity == {
        "status": "mismatch",
        "reason": "core configuration differs in added_tokens",
        "sections": ["added_tokens"],
    }
    assert w2.toktier is None and w2.family is None and fake2.closed
    assert w2.stats()["mode"] == "passthrough"
    assert w2.encode("ab", add_special_tokens=False) == drifted.encode(
        "ab", add_special_tokens=False
    )
    assert w2("ab")["input_ids"] == drifted("ab")["input_ids"]
    assert w2.stats()["delegated"] == 2 and fake2.calls == []


def test_identity_compares_every_core_section() -> None:
    from tokenizers import normalizers

    fake = _FakeToktier()
    same = _StubHf()
    assert check_core_identity(same, fake)["status"] == "equivalent"
    normalized = _StubHf()
    normalized.backend_tokenizer.normalizer = normalizers.NFC()
    assert check_core_identity(normalized, fake)["sections"] == ["normalizer"]
    truncating = _StubHf()
    truncating.backend_tokenizer.enable_truncation(4)
    assert check_core_identity(truncating, fake)["sections"] == ["truncation"]
    padding = _StubHf()
    padding.backend_tokenizer.enable_padding(length=8)
    assert check_core_identity(padding, fake)["sections"] == ["padding"]
    both = _StubHf()
    both.backend_tokenizer.normalizer = normalizers.NFC()
    both.backend_tokenizer.add_tokens(["x"])
    assert check_core_identity(both, fake)["sections"] == ["normalizer", "added_tokens"]


def test_identity_unavailable_delegates_and_skip_mode_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    hf = _StubHf()
    del hf.backend_tokenizer
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off"))
    assert w.identity["status"] == "unavailable" and w.toktier is None
    assert w.identity["reason"] == (
        "the supplied tokenizer has no serializable fast backend"
    )
    fake2 = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake2)
    w2 = wrap_tokenizer(hf, AdapterConfig(session_key="off", identity="skip"))
    assert w2.identity["status"] == "skipped" and w2.toktier is fake2
    with pytest.raises(ValueError):
        AdapterConfig(identity="maybe")
    assert AdapterConfig.from_env({"TOKTIER_SGLANG_IDENTITY": "skip"}).identity == (
        "skip"
    )
    assert AdapterConfig.from_env({}).identity == "require"
    # passthrough never loads toktier, so there is nothing to compare
    w3 = wrap_tokenizer(_StubHf(), AdapterConfig(session_key="passthrough"))
    assert w3.identity["status"] == "skipped"


# -- attribute writes (release-candidate review, finding 3) ------------------


def test_attribute_writes_follow_attribute_reads() -> None:
    hf: Any = _StubHf()
    hf.chat_template = "old"
    w = wrap_tokenizer(hf, AdapterConfig(session_key="passthrough"))
    w.chat_template = "new"
    assert hf.chat_template == "new" and w.chat_template == "new"
    assert "chat_template" not in w.__dict__
    w.brand_new = 1
    assert hf.brand_new == 1 and w.brand_new == 1
    del w.brand_new
    assert not hasattr(hf, "brand_new") and not hasattr(w, "brand_new")
    w._private = 2  # wrapper state stays on the wrapper
    assert w.__dict__["_private"] == 2 and not hasattr(hf, "_private")
    with pytest.raises(AttributeError):
        w.family = "x"  # a read-only wrapper property


# -- shadow reference (release-candidate review, finding 4) -------------------


def _wait_for_shadow(w: SGLangToktierTokenizer, count: int) -> None:
    deadline = time.perf_counter() + 5.0
    while w.stats()["shadow_checked"] < count and time.perf_counter() < deadline:
        time.sleep(0.01)


def test_async_shadow_judges_against_a_copy_of_the_supplied_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    hf: Any = _StubHf()
    # A live difference the identity gate does not see (the serialized
    # backend is unchanged): only a copy of this very object carries it.
    hf._core = lambda text: [1] * len(text)
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off", shadow="async"))
    assert w.identity["status"] == "equivalent"
    assert w._shadow_ref is not hf and isinstance(w._shadow_ref, _StubHf)
    assert w.stats()["shadow"] == "async"
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    _wait_for_shadow(w, 1)
    assert w.stats()["shadow_checked"] == 1 and w.stats()["shadow_mismatch"] == 1
    w.close()


def test_async_shadow_without_a_copy_checks_synchronously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Uncopyable(_StubHf):
        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            raise TypeError("no copy")

        def __reduce__(self) -> Any:
            raise TypeError("no pickle")

    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    hf = _Uncopyable()
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off", shadow="async"))
    assert w.stats()["shadow"] == "sync" and w._shadow_ref is hf
    assert w._shadow_q is None
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    assert w.stats()["shadow_checked"] == 1 and w.stats()["shadow_mismatch"] == 0


# -- a real artifact: the reviewed add_tokens sequence ------------------------


@pytest.fixture(scope="module")
def qwen3_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Config]:
    artifacts = _ARTIFACTS
    if not artifacts:
        pytest.skip("TOKTIER_TEST_ARTIFACTS is not set")
    root = Path(artifacts or "")
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    entry = manifest.get("qwen3_8b")
    source: Path | None = None
    for name in (entry.directory_name, "qwen3_8b"):
        candidate = root / name
        if (candidate / "tokenizer.json").is_file():
            source = candidate
    if source is None:
        pytest.skip("qwen3_8b artifact not available under TOKTIER_TEST_ARTIFACTS")
    assert source is not None
    home = tmp_path_factory.mktemp("sglang-identity")
    config = Config(home=home, offline=True)
    cache = artifact_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, cache / entry.directory_name)
    yield config


def test_real_tokenizer_drift_after_add_tokens_is_delegated(
    qwen3_home: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tokenizers
    from transformers import PreTrainedTokenizerFast

    subject = toktier.load("qwen3_8b", config=qwen3_home, device="cpu")
    face = subject._backend.materialized_tokenizer_json()
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizers.Tokenizer.from_str(face))
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: subject)
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off"))
    assert w.identity["status"] == "equivalent" and w.family == "qwen3_8b"
    probe = "toktierprobeunique in a sentence"
    assert w.encode(probe, add_special_tokens=False) == hf.encode(
        probe, add_special_tokens=False
    )
    assert w.stats()["how"] == {"off": 1}
    # release-candidate review, finding 2: the literal becomes one token on
    # the supplied object; the wrapper must answer with that object.
    hf.add_tokens(["toktierprobeunique"])
    second = toktier.load("qwen3_8b", config=qwen3_home, device="cpu")
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: second)
    w2 = wrap_tokenizer(hf, AdapterConfig(session_key="off"))
    assert w2.identity["status"] == "mismatch"
    assert w2.identity["sections"] == ["added_tokens"]
    expected = hf.encode("toktierprobeunique", add_special_tokens=False)
    assert len(expected) == 1
    assert w2.encode("toktierprobeunique", add_special_tokens=False) == expected
    assert (
        w2("toktierprobeunique")["input_ids"] == hf("toktierprobeunique")["input_ids"]
    )
    assert w2.stats()["mode"] == "passthrough" and w2.stats()["delegated"] == 2
    w.close()


# -- identity gate, second review: split_special_tokens, post_processor, drift ----


def test_split_special_tokens_is_part_of_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: _FakeToktier()
    )
    flagged = _StubHf()
    flagged.split_special_tokens = True
    w = wrap_tokenizer(flagged, AdapterConfig(session_key="off"))
    assert w.identity["status"] == "mismatch"
    assert "split_special_tokens" in str(w.identity["reason"])
    assert w.toktier is None and w.stats()["mode"] == "passthrough"
    # The backend-level flag alone (what transformers sets on the Rust
    # object before an encode) is caught as well.
    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: _FakeToktier()
    )
    backend_flagged = _StubHf()
    backend_flagged.backend_tokenizer.encode_special_tokens = True
    w2 = wrap_tokenizer(backend_flagged, AdapterConfig(session_key="off"))
    assert w2.identity["status"] == "mismatch" and w2.toktier is None


def test_post_processor_is_part_of_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenizers import processors

    monkeypatch.setattr(
        SGLangToktierTokenizer, "_load_toktier", lambda self: _FakeToktier()
    )
    doubled = _StubHf()
    doubled.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="$A $A"
    )
    w = wrap_tokenizer(doubled, AdapterConfig(session_key="off"))
    assert w.identity["status"] == "mismatch"
    assert w.identity["sections"] == ["post_processor"]
    assert w.toktier is None and w.stats()["mode"] == "passthrough"
    assert check_core_identity(_StubHf(), _FakeToktier())["status"] == "equivalent"


def test_drift_after_construction_latches_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # (a) a mutation forwarded through the wrapper
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    hf = _StubHf()
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off"))
    assert w.encode("ab", add_special_tokens=False) == [97, 98] and fake.calls
    w.add_tokens(["toktierprobeunique"])
    expected = hf.encode("ab", add_special_tokens=False)
    assert w.encode("ab", add_special_tokens=False) == expected
    assert w.identity["status"] == "mismatch"
    assert w.identity["sections"] == ["added_tokens"]
    assert w.toktier is None and fake.closed and w.stats()["mode"] == "passthrough"
    assert w.stats()["delegated"] == 1 and len(fake.calls) == 1
    # (b) a forwarded assignment of the flag
    fake2 = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake2)
    hf2 = _StubHf()
    w2 = wrap_tokenizer(hf2, AdapterConfig(session_key="off"))
    assert w2.encode("ab", add_special_tokens=False) == [97, 98]
    w2.split_special_tokens = True
    assert hf2.split_special_tokens is True  # forwarded, as before
    assert w2("ab")["input_ids"] == hf2("ab")["input_ids"]
    assert w2.identity["status"] == "mismatch" and w2.toktier is None
    # (c) a change made on the retained object, behind the wrapper's back
    fake3 = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake3)
    hf3 = _StubHf()
    w3 = wrap_tokenizer(hf3, AdapterConfig(session_key="off"))
    assert w3.encode("ab", add_special_tokens=False) == [97, 98]
    hf3.backend_tokenizer.add_tokens(["toktierprobeunique"])
    expected3 = hf3.encode("ab", add_special_tokens=False)
    assert w3.encode("ab", add_special_tokens=False) == expected3
    assert w3.identity["status"] == "mismatch" and w3.toktier is None
    # (d) an async wrapper over a pool: the pool is closed with the latch
    fake4 = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake4)
    hf4 = _StubHf()
    w4 = wrap_tokenizer(hf4, AdapterConfig(session_key="off", shadow="async"))
    assert w4.stats()["shadow"] == "async" and w4._shadow_ref is not hf4
    assert asyncio.run(w4.encode_async("ab", add_special_tokens=False)) == [97, 98]
    hf4.backend_tokenizer.post_processor = None
    hf4.backend_tokenizer.add_tokens(["toktierprobeunique"])
    assert asyncio.run(w4.encode_async("ab", add_special_tokens=False)) == hf4.encode(
        "ab", add_special_tokens=False
    )
    assert w4.identity["status"] == "mismatch" and w4.toktier is None
    w4.close()


def test_equivalent_change_refreshes_the_shadow_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeToktier()
    monkeypatch.setattr(SGLangToktierTokenizer, "_load_toktier", lambda self: fake)
    hf = _StubHf()
    w = wrap_tokenizer(hf, AdapterConfig(session_key="off", shadow="async"))
    old_ref = w._shadow_ref
    assert old_ref is not hf
    w.chat_template = "a template"  # forwarded; does not touch the id stream
    assert getattr(hf, "chat_template", None) == "a template"
    assert w.encode("ab", add_special_tokens=False) == [97, 98]
    assert w.identity["status"] == "equivalent" and w.toktier is fake
    assert w._shadow_ref is not old_ref
    assert getattr(w._shadow_ref, "chat_template", None) == "a template"
    w.close()


def test_byte_level_trim_offsets_is_not_an_identity_fact() -> None:
    # transformers 5 sets trim_offsets=True on the live ByteLevel pre-tokenizer
    # (and post-processor); offsets change, ids do not: still equivalent.
    from tokenizers import pre_tokenizers, processors

    fake = _FakeToktier()
    live = _StubHf()
    plain = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    live.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), plain]
    )
    live.backend_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    face_shape = _StubHf()
    face_shape.backend_tokenizer.pre_tokenizer = live.backend_tokenizer.pre_tokenizer
    face_shape.backend_tokenizer.post_processor = live.backend_tokenizer.post_processor

    class _Face(_FakeToktier):
        class _backend:  # the artifact face: the same components, trim_offsets off
            @staticmethod
            def materialized_tokenizer_json() -> str:
                text: str = face_shape.backend_tokenizer.to_str()
                return text

    assert check_core_identity(live, _Face())["status"] == "equivalent"
    trimmed = _StubHf()
    trimmed.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False, trim_offsets=True
            ),
        ]
    )
    trimmed.backend_tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    assert check_core_identity(trimmed, _Face())["status"] == "equivalent"
    # a fact that does change ids (the byte-level regex) is still caught
    regex = _StubHf()
    with_regex = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    regex.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), with_regex]
    )
    regex.backend_tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    assert check_core_identity(regex, _Face())["sections"] == ["pre_tokenizer"]
    del fake
