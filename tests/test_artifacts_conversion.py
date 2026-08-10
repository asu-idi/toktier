"""Locally converted artifacts: pinned inputs, determinism, identity."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from toktier.artifacts.conversion import (
    CONVERSIONS,
    ConversionRecipe,
    ConvertingSource,
    conversion_report,
    convert_kimi_tiktoken,
    recipe_for,
)
from toktier.artifacts.manifest import ArtifactEntry, ArtifactFile
from toktier.artifacts.sources import LocalDirectorySource
from toktier.artifacts.store import ArtifactStore
from toktier.config import Config
from toktier.errors import ArtifactNotFound, RegistryInvalid

# A rank file small enough to keep in a test but shaped like the real
# one: every single byte, then a handful of multi-byte tokens whose
# decompositions become the merge list.
_EXTRA_TOKENS = (b"ab", b"cd", b"abcd", b"th", b"the")

_UPSTREAM_MODULE = '''"""Stub of the upstream tokenizer module."""

class KimiTokenizer:
    pat_str = "|".join([
        r"""{pattern}""",
    ])
'''


def _rank_file(tokens: tuple[bytes, ...]) -> bytes:
    lines = []
    for rank, token in enumerate(tokens):
        lines.append(base64.b64encode(token) + b" " + str(rank).encode())
    return b"\n".join(lines) + b"\n"


def _tokens() -> tuple[bytes, ...]:
    return tuple(bytes([value]) for value in range(256)) + _EXTRA_TOKENS


def _configuration(base_count: int) -> bytes:
    decoder = {
        str(base_count): {"content": "[BOS]", "special": True},
        str(base_count + 1): {"content": "<|open|>", "special": False},
    }
    return (
        json.dumps({"added_tokens_decoder": decoder, "auto_map": {"drop": "me"}})
    ).encode()


@pytest.fixture()
def upstream(tmp_path: Path) -> Path:
    from toktier.artifacts.conversion import KIMI_PATTERN

    directory = tmp_path / "upstream"
    directory.mkdir()
    tokens = _tokens()
    (directory / "tiktoken.model").write_bytes(_rank_file(tokens))
    (directory / "tokenization_kimi.py").write_text(
        _UPSTREAM_MODULE.format(pattern=KIMI_PATTERN), encoding="utf-8"
    )
    (directory / "tokenizer_config.json").write_bytes(
        _configuration(len(tokens))
    )
    return directory


def _recipe(directory: Path) -> ConversionRecipe:
    inputs = []
    for name in ("tiktoken.model", "tokenization_kimi.py", "tokenizer_config.json"):
        payload = (directory / name).read_bytes()
        inputs.append(
            ArtifactFile(
                name=name,
                sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
            )
        )
    return ConversionRecipe(
        family="kimi_test",
        inputs=tuple(inputs),
        converter="kimi_tiktoken_v1",
        upstream_license="LICENSE",
    )


def _payloads(directory: Path) -> dict[str, bytes]:
    return {
        name: (directory / name).read_bytes()
        for name in (
            "tiktoken.model",
            "tokenization_kimi.py",
            "tokenizer_config.json",
        )
    }


def _entry(directory: Path, *, sha256: str, size: int | None) -> ArtifactEntry:
    return ArtifactEntry(
        family="kimi_test",
        repo_id="stub/Kimi-Test",
        revision="0123456789abcdef",
        files=(ArtifactFile(name="tokenizer.json", sha256=sha256, size=size),),
        local_dir=str(directory),
    )


def test_conversion_is_deterministic_and_well_formed(upstream: Path) -> None:
    first = convert_kimi_tiktoken(_payloads(upstream))
    second = convert_kimi_tiktoken(_payloads(upstream))
    assert first == second

    document = json.loads(first)
    assert first.endswith(b"\n")
    assert document["normalizer"] is None
    assert document["model"]["ignore_merges"] is True
    assert len(document["model"]["vocab"]) == 256 + len(_EXTRA_TOKENS)
    # Merges are the two-way decompositions, ordered by the rank of the
    # token each one produces.
    assert document["model"]["merges"] == [
        ["a", "b"],
        ["c", "d"],
        ["ab", "cd"],
        ["t", "h"],
        ["th", "e"],
    ]
    added = document["added_tokens"]
    assert [item["id"] for item in added] == list(range(261, 261 + 256))
    assert added[0]["content"] == "[BOS]" and added[0]["special"] is True
    assert added[1]["content"] == "<|open|>" and added[1]["special"] is False
    assert added[2]["content"] == "<|reserved_token_263|>"


def test_upstream_pattern_drift_is_refused(upstream: Path) -> None:
    payloads = _payloads(upstream)
    payloads["tokenization_kimi.py"] = payloads["tokenization_kimi.py"].replace(
        b"\\p{Han}", b"\\p{Hani}"
    )
    with pytest.raises(RegistryInvalid, match="pre-tokenizer pattern"):
        convert_kimi_tiktoken(payloads)


def test_ranks_must_be_a_dense_range(upstream: Path) -> None:
    payloads = _payloads(upstream)
    payloads["tiktoken.model"] = _rank_file(_tokens()[:-1]) + (
        base64.b64encode(b"zz") + b" 9999\n"
    )
    with pytest.raises(RegistryInvalid, match="dense range"):
        convert_kimi_tiktoken(payloads)


def test_unreadable_rank_line_is_refused(upstream: Path) -> None:
    payloads = _payloads(upstream)
    payloads["tiktoken.model"] += b"not-a-pair\n"
    with pytest.raises(RegistryInvalid, match="token/rank pair"):
        convert_kimi_tiktoken(payloads)


def test_report_binds_the_conversion_to_the_pinned_digest(
    upstream: Path,
) -> None:
    payload = convert_kimi_tiktoken(_payloads(upstream))
    digest = hashlib.sha256(payload).hexdigest()
    entry = _entry(upstream, sha256=digest, size=len(payload))
    report = conversion_report(
        entry, _recipe(upstream), LocalDirectorySource(), repeats=3
    )
    assert report["runs"] == 3
    assert report["deterministic"] is True
    assert report["identity_matches"] is True
    assert report["observed_sha256"] == digest
    assert report["added_tokens"] == 256
    assert report["added_tokens_contiguous"] is True
    assert report["added_tokens_fully_described"] is True
    assert [item["name"] for item in report["upstream_inputs"]] == [
        "tiktoken.model",
        "tokenization_kimi.py",
        "tokenizer_config.json",
    ]


def test_report_reports_a_pin_that_does_not_match(upstream: Path) -> None:
    entry = _entry(upstream, sha256="0" * 64, size=1)
    report = conversion_report(entry, _recipe(upstream), LocalDirectorySource())
    assert report["deterministic"] is True
    assert report["identity_matches"] is False


def test_an_upstream_input_that_drifted_is_refused(upstream: Path) -> None:
    recipe = _recipe(upstream)
    tampered = ConversionRecipe(
        family=recipe.family,
        inputs=(
            ArtifactFile(name="tiktoken.model", sha256="1" * 64, size=None),
            *recipe.inputs[1:],
        ),
        converter=recipe.converter,
        upstream_license=recipe.upstream_license,
    )
    entry = _entry(upstream, sha256="0" * 64, size=1)
    with pytest.raises(ArtifactNotFound, match="does not match the digest"):
        conversion_report(entry, tampered, LocalDirectorySource())


def test_an_upstream_input_of_the_wrong_length_is_refused(
    upstream: Path,
) -> None:
    recipe = _recipe(upstream)
    tampered = ConversionRecipe(
        family=recipe.family,
        inputs=(
            ArtifactFile(
                name="tiktoken.model",
                sha256=recipe.inputs[0].sha256,
                size=(recipe.inputs[0].size or 0) + 1,
            ),
            *recipe.inputs[1:],
        ),
        converter=recipe.converter,
        upstream_license=recipe.upstream_license,
    )
    entry = _entry(upstream, sha256="0" * 64, size=1)
    with pytest.raises(ArtifactNotFound, match="unexpected byte length"):
        conversion_report(entry, tampered, LocalDirectorySource())


def test_the_store_installs_a_converted_artifact(
    upstream: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from toktier.artifacts.manifest import ArtifactManifest

    payload = convert_kimi_tiktoken(_payloads(upstream))
    digest = hashlib.sha256(payload).hexdigest()
    entry = _entry(upstream, sha256=digest, size=len(payload))
    manifest = ArtifactManifest(entries={entry.family: entry})
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "home"))
    source = ConvertingSource(
        LocalDirectorySource(), conversions={entry.family: _recipe(upstream)}
    )
    assert source.name == "local_dir"
    store = ArtifactStore(manifest, config=Config.resolve(), source=source)
    artifact = store.ensure(entry.family)
    assert artifact.path("tokenizer.json").read_bytes() == payload
    # A second call is served from the verified cache.
    assert store.verify(entry.family).family == entry.family


def test_the_store_refuses_a_conversion_that_missed_its_pin(
    upstream: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.errors import ArtifactHashMismatch

    entry = _entry(upstream, sha256="0" * 64, size=None)
    manifest = ArtifactManifest(entries={entry.family: entry})
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "home"))
    store = ArtifactStore(
        manifest,
        config=Config.resolve(),
        source=ConvertingSource(
            LocalDirectorySource(), conversions={entry.family: _recipe(upstream)}
        ),
    )
    with pytest.raises(ArtifactHashMismatch):
        store.ensure(entry.family)


def test_families_without_a_recipe_pass_through(
    upstream: Path, tmp_path: Path
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "tokenizer.json").write_bytes(b"{}")
    entry = ArtifactEntry(
        family="plain_family",
        repo_id="stub/Plain",
        revision="0123456789abcdef",
        files=(
            ArtifactFile(
                name="tokenizer.json",
                sha256=hashlib.sha256(b"{}").hexdigest(),
                size=2,
            ),
        ),
        local_dir=str(plain),
    )
    source = ConvertingSource(
        LocalDirectorySource(), conversions={"kimi_test": _recipe(upstream)}
    )
    assert source.recipe("plain_family") is None
    destination = tmp_path / "out.json"
    source.fetch(entry, entry.files[0], destination)
    assert destination.read_bytes() == b"{}"


def test_a_converted_family_produces_only_its_one_file(
    upstream: Path, tmp_path: Path
) -> None:
    entry = _entry(upstream, sha256="0" * 64, size=None)
    source = ConvertingSource(
        LocalDirectorySource(), conversions={entry.family: _recipe(upstream)}
    )
    with pytest.raises(ArtifactNotFound, match="produces"):
        source.fetch(
            entry,
            ArtifactFile(name="vocab.txt", sha256="0" * 64),
            tmp_path / "out",
        )


def test_the_shipped_recipe_names_the_pinned_upstream_inputs() -> None:
    recipe = recipe_for("kimi_k3")
    assert recipe is not None
    assert recipe is CONVERSIONS["kimi_k3"]
    assert recipe.converter == "kimi_tiktoken_v1"
    assert recipe.input_names() == (
        "tiktoken.model",
        "tokenization_kimi.py",
        "tokenizer_config.json",
    )
    for item in recipe.inputs:
        assert len(item.sha256) == 64
        assert item.size is not None and item.size > 0


def test_every_recipe_names_a_family_the_release_knows() -> None:
    """A recipe must name a family this release has an identity for.

    The pinned output digest lives in the artifact manifest entry, so a
    recipe is only *runnable* once the family is pinned there. Before
    that it still has to name a family the release recognises, which the
    verified-sibling table records for every canonical anchor.
    """
    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST, SIBLING_ALIASES

    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    aliases = json.loads(SIBLING_ALIASES.read_text(encoding="utf-8"))
    known = {row["canonical_family"] for row in aliases["aliases"]}
    for family in CONVERSIONS:
        assert family in known or family in manifest.entries, (
            f"{family} has a conversion recipe but this release has no "
            "identity for that family"
        )
        entry = manifest.entries.get(family)
        if entry is not None:
            assert [item.name for item in entry.files] == ["tokenizer.json"]
