"""Loader-face materialization: subset extraction, digest, and verification.

Contract reference: ``docs/contracts/registry.md`` Section 1 (the exact
artifact identity and its configuration-side extension),
``docs/contracts/facade.md`` Section 5 (reference = the loader face).

Since 0.2.8 the reference backend executes the loader face: the artifact
document when the configuration sidecar declares no added token beyond
it, and the serialization of the live loader object when it does. These
tests pin the three shared answers -- which tokens are
configuration-only, their canonical digest, and the fail-closed
verification against a certification record's claim -- and the
construction behavior of the reference backend on both kinds of
artifact directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import _support as support
import pytest

from toktier.backends.hf import HfBackend
from toktier.backends.loader_face import (
    CONFIG_ADDED_TOKENS_MISMATCH,
    config_added_token_rows,
    config_added_tokens_sha256,
    config_only_added_tokens,
    load_live_tokenizer,
    verify_declared_config_added_tokens,
)
from toktier.errors import ArtifactHashMismatch

tokenizers = pytest.importorskip("tokenizers")

#: The one extra literal of the loader-face fixtures. The declared id
#: matches the id the loader actually assigns (the artifact vocabulary
#: holds ids 0..255, so the first added token lands on 256).
EXTRA = "<x_extra>"
EXTRA_ID = 256


def _decoder_row(content: str, **flags: bool) -> dict[str, Any]:
    row = {
        "content": content,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    }
    row.update(flags)
    return row


def _artifact_with_sidecar(
    tmp_path: Path,
    *,
    decoder: dict[str, Any] | None,
    tokenizer_class: str = "PreTrainedTokenizerFast",
    added_tokens: tuple[dict[str, Any], ...] = (),
) -> support.StubArtifact:
    document = support.byte_level_document(added_tokens=added_tokens)
    artifact = support.write_artifact(tmp_path / "artifact", document)
    sidecar: dict[str, Any] = {"tokenizer_class": tokenizer_class}
    if decoder is not None:
        sidecar["added_tokens_decoder"] = decoder
    (artifact.root / "tokenizer_config.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return artifact


# ---------------------------------------------------------------------
# subset extraction
# ---------------------------------------------------------------------


def test_no_sidecar_means_no_configuration_side_tokens(tmp_path: Path) -> None:
    artifact = support.write_artifact(
        tmp_path / "artifact", support.byte_level_document()
    )
    assert config_added_token_rows(artifact.root) == []
    assert config_only_added_tokens(artifact.root) == []


def test_decoder_covered_by_artifact_is_not_configuration_only(
    tmp_path: Path,
) -> None:
    """A sidecar that only restates the artifact's tokens adds nothing."""
    token = {
        "id": 300,
        "content": EXTRA,
        "special": True,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
    }
    artifact = _artifact_with_sidecar(
        tmp_path,
        decoder={"300": _decoder_row(EXTRA)},
        added_tokens=(token,),
    )
    assert config_added_token_rows(artifact.root) == []


def test_configuration_only_rows_are_id_ascending_with_flags(
    tmp_path: Path,
) -> None:
    artifact = _artifact_with_sidecar(
        tmp_path,
        decoder={
            "257": _decoder_row("<x_b>"),
            "256": _decoder_row(EXTRA, single_word=True),
        },
    )
    rows = config_added_token_rows(artifact.root)
    assert [row["id"] for row in rows] == [256, 257]
    assert rows[0]["content"] == EXTRA
    assert rows[0]["single_word"] is True
    assert rows[0]["special"] is True
    assert rows[1]["content"] == "<x_b>"
    assert config_only_added_tokens(artifact.root) == [EXTRA, "<x_b>"]


def test_canonical_digest_is_frozen(tmp_path: Path) -> None:
    """The canonical form is pinned: id-ascending rows, sorted keys, no
    whitespace, literal non-ASCII, SHA-256 over the UTF-8 text.

    The golden value is computed from the spelled-out form here, so a
    serialization change in the helper cannot pass silently.
    """
    import hashlib

    rows = [
        {
            "id": 256,
            "content": EXTRA,
            "special": True,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        }
    ]
    expected_payload = (
        '[{"content":"<x_extra>","id":256,"lstrip":false,"normalized":false,'
        '"rstrip":false,"single_word":false,"special":true}]'
    )
    expected = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
    assert config_added_tokens_sha256(rows) == expected
    # Order of the input list does not matter; the ids do.
    assert config_added_tokens_sha256(list(reversed(rows))) == expected


# ---------------------------------------------------------------------
# claim verification
# ---------------------------------------------------------------------


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    artifact = _artifact_with_sidecar(
        tmp_path, decoder={"256": _decoder_row(EXTRA)}
    )
    return config_added_token_rows(artifact.root)


def test_no_claim_verifies_nothing(tmp_path: Path) -> None:
    verify_declared_config_added_tokens(
        family="t", observed_rows=_rows(tmp_path), declared=None
    )


def test_matching_claim_passes(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    verify_declared_config_added_tokens(
        family="t",
        observed_rows=rows,
        declared={
            "sha256": config_added_tokens_sha256(rows),
            "count": len(rows),
            "source": "tokenizer_config.json",
        },
    )


def test_undeclared_observed_subset_fails_closed(tmp_path: Path) -> None:
    """A certified record without the section claims an empty subset."""
    with pytest.raises(ArtifactHashMismatch) as caught:
        verify_declared_config_added_tokens(
            family="t",
            observed_rows=_rows(tmp_path),
            declared={"sha256": None, "count": 0},
        )
    assert caught.value.details["reason"] == CONFIG_ADDED_TOKENS_MISMATCH


def test_declared_subset_missing_on_disk_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ArtifactHashMismatch) as caught:
        verify_declared_config_added_tokens(
            family="t",
            observed_rows=[],
            declared={"sha256": "0" * 64, "count": 7},
        )
    assert caught.value.details["observed_count"] == 0


# ---------------------------------------------------------------------
# reference backend construction over the two kinds of directory
# ---------------------------------------------------------------------


def test_reference_extracts_configuration_side_literal(tmp_path: Path) -> None:
    """With a loadable sidecar the reference executes the loader face."""
    pytest.importorskip("transformers")
    artifact = _artifact_with_sidecar(
        tmp_path, decoder={str(EXTRA_ID): _decoder_row(EXTRA)}
    )
    handle = support.local_artifact(artifact.root, family="tiny")
    backend = HfBackend.open(handle)
    assert backend.encode(f"a{EXTRA}b", add_special_tokens=False) == [
        97,
        EXTRA_ID,
        98,
    ]
    assert backend.encode("ab", add_special_tokens=False) == [97, 98]
    # The materialized document is the loader face, so a second engine
    # built from it (the facade's decode oracle) answers identically.
    live = tokenizers.Tokenizer.from_str(backend.materialized_tokenizer_json())
    assert [int(i) for i in live.encode(f"a{EXTRA}b").ids] == [97, EXTRA_ID, 98]
    backend.close()


def test_reference_without_sidecar_runs_the_artifact_document(
    tmp_path: Path,
) -> None:
    artifact = support.write_artifact(
        tmp_path / "artifact", support.byte_level_document()
    )
    backend = HfBackend.open(artifact)
    crate = tokenizers.Tokenizer.from_file(str(artifact.root / "tokenizer.json"))
    text = f"a{EXTRA}b"
    assert backend.encode(text, add_special_tokens=False) == [
        int(i) for i in crate.encode(text, add_special_tokens=False).ids
    ]
    assert backend.materialized_tokenizer_json() == (
        artifact.root / "tokenizer.json"
    ).read_text(encoding="utf-8")
    backend.close()


def test_reference_claim_mismatch_fails_closed(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    artifact = _artifact_with_sidecar(
        tmp_path, decoder={str(EXTRA_ID): _decoder_row(EXTRA)}
    )
    handle = support.local_artifact(artifact.root, family="tiny")

    @dataclass
    class Claimed:
        family: str = handle.family
        root: Path = handle.root
        artifact_sha256: str = handle.artifact_sha256
        files: dict[str, str] = field(
            default_factory=lambda: dict(handle.files)
        )
        config_added_tokens_claim: dict[str, Any] = field(
            default_factory=lambda: {"sha256": None, "count": 0}
        )

        def path(self, name: str) -> Path:
            return self.root / name

    with pytest.raises(ArtifactHashMismatch) as caught:
        HfBackend.open(Claimed())
    assert caught.value.details["reason"] == CONFIG_ADDED_TOKENS_MISMATCH


# ---------------------------------------------------------------------
# both scanning surfaces read the loader face
# ---------------------------------------------------------------------


def test_router_and_fallback_counter_answer_alike(tmp_path: Path) -> None:
    """The routing scanner and the execution ledger see the same face.

    On 0.2.7 the router was built from the artifact file while execution
    saw the loader face, so a configuration-side literal was denied by
    one surface and extracted by the other. Built from the materialized
    loader face, the router routes such an input to the reference and
    the ledger counts it -- one answer on both surfaces.
    """
    pytest.importorskip("transformers")
    from toktier.frontend.added import AddedTokenFrontend
    from toktier.policy import (
        BACKEND_FAST_CPU,
        BACKEND_REFERENCE,
        ReasonCode,
        RoutePlan,
        RoutingPolicy,
    )
    from toktier.routing.added_route import AddedTokenRouter
    from toktier.routing.execute import RoutedExecutor

    artifact = _artifact_with_sidecar(
        tmp_path, decoder={str(EXTRA_ID): _decoder_row(EXTRA)}
    )
    backend = HfBackend.open(support.local_artifact(artifact.root, family="t"))
    document = json.loads(backend.materialized_tokenizer_json())
    router = AddedTokenRouter(
        AddedTokenFrontend(
            {
                "family": "t",
                "normalizer": document.get("normalizer"),
                "added_tokens": document.get("added_tokens") or [],
            }
        )
    )
    accelerated = support.FakeBackend(BACKEND_FAST_CPU, base=1000)
    executor = RoutedExecutor(
        RoutePlan(
            policy=RoutingPolicy.CERTIFIED,
            backend=BACKEND_FAST_CPU,
            fallback_chain=(BACKEND_FAST_CPU, BACKEND_REFERENCE),
        ),
        {BACKEND_FAST_CPU: accelerated, BACKEND_REFERENCE: backend},
        added_router=router,
    )
    text = f"a{EXTRA}b"
    assert router.holds_literal(text) is True
    assert executor.encode(text) == [97, EXTRA_ID, 98]
    assert accelerated.calls == []
    assert executor.fallback_counts == {ReasonCode.R_INPUT_ADDED_TOKEN.value: 1}
    # And the negative direction agrees as well.
    assert router.holds_literal("plain") is False
    executor.encode("plain")
    assert accelerated.calls == ["plain"]
    backend.close()


# ---------------------------------------------------------------------
# artifacts with no resolvable loader class (Hy4-preview shape)
# ---------------------------------------------------------------------


def test_unknown_loader_class_falls_back_to_the_file_only_face(
    tmp_path: Path,
) -> None:
    """No loader class and no configuration-side token: file-only face.

    ``tencent/Hy4-preview`` ships ``tokenizer_class: TokenizersBackend``,
    which the pinned ``transformers`` cannot resolve. The loader face
    degrades to the file-only face there, and only because the
    configuration declares no added token beyond the artifact are the
    two provably the same function -- which is exactly the condition the
    fallback verifies before taking that path.
    """
    pytest.importorskip("transformers")
    artifact = _artifact_with_sidecar(
        tmp_path, decoder=None, tokenizer_class="SomeUnknownBackend"
    )
    live: Any = load_live_tokenizer(artifact.root)
    crate = tokenizers.Tokenizer.from_file(str(artifact.root / "tokenizer.json"))
    text = "some text"
    assert live.encode(text, add_special_tokens=False) == [
        int(i) for i in crate.encode(text, add_special_tokens=False).ids
    ]
    # The reference backend never needed the loader here at all.
    backend = HfBackend.open(support.local_artifact(artifact.root, family="t"))
    assert backend.encode(text, add_special_tokens=False) == live.encode(
        text, add_special_tokens=False
    )
    backend.close()


def test_unknown_loader_class_with_config_tokens_propagates(
    tmp_path: Path,
) -> None:
    """The file-only fallback is refused when it would drop a token."""
    pytest.importorskip("transformers")
    artifact = _artifact_with_sidecar(
        tmp_path,
        decoder={str(EXTRA_ID): _decoder_row(EXTRA)},
        tokenizer_class="SomeUnknownBackend",
    )
    with pytest.raises(Exception) as caught:
        load_live_tokenizer(artifact.root)
    # The original loader error propagates; the fallback that would have
    # silently dropped the configuration-side token is not taken.
    assert not isinstance(caught.value, ArtifactHashMismatch)
