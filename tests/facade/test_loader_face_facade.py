"""The facade over an artifact whose sidecar declares extra added tokens.

Contract reference: ``docs/contracts/facade.md`` Section 5 (reference =
the loader face), ``docs/contracts/routing.md`` Section 5.2
(``R_INPUT_ADDED_TOKEN``).

Since 0.2.8 every face the facade serves reads the same loader-face
document: the reference backend, the decode oracle, and the added-token
router. These tests pin that agreement on a tiny artifact whose
``tokenizer_config.json`` declares one added token beyond the artifact
file -- the shape on which 0.2.7's two certified routes could answer
differently.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from toktier.backends.hf import oracle_version
from toktier.backends.loader_face import (
    config_added_token_rows,
    config_added_tokens_sha256,
    loader_face_sha256,
)
from toktier.facade import api as facade_api
from toktier.policy import BACKEND_FAST_CPU, BACKEND_REFERENCE, ReasonCode
from toktier.routing.registry_view import (
    ArtifactRecord,
    BackendEntry,
    OracleRecord,
    RegistryView,
)

from .conftest import Rig, build_rig

EXTRA = "<x_extra>"
EXTRA_ID = 256

#: Local artifacts for the artifact tier (the convention of
#: ``tests/test_chunked.py``): one subdirectory per artifact.
_ARTIFACTS = os.environ.get("TOKTIER_TEST_ARTIFACTS")


def _sidecar_rig(tmp_path: Path) -> Rig:
    rig = build_rig(tmp_path)
    sidecar = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "added_tokens_decoder": {
            str(EXTRA_ID): {
                "content": EXTRA,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
        },
    }
    (rig.artifact_path.parent / "tokenizer_config.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return rig


@pytest.fixture
def sidecar_rig(tmp_path: Path) -> Rig:
    pytest.importorskip("transformers")
    return _sidecar_rig(tmp_path)


def test_default_and_reference_policies_agree_on_the_literal(
    sidecar_rig: Rig,
) -> None:
    """One loader face, every policy: the literal maps to its id."""
    text = f"a{EXTRA}b"
    with_default = sidecar_rig.tokenizer()
    with_reference = sidecar_rig.tokenizer(policy="reference")
    try:
        default_ids = list(with_default.encode(text).ids)
        reference_ids = list(with_reference.encode(text).ids)
        assert default_ids == reference_ids == [97, EXTRA_ID, 98]
        assert list(with_default.encode("ab").ids) == [97, 98]
    finally:
        with_default.close()
        with_reference.close()


def test_router_and_execution_answer_alike_about_the_literal(
    sidecar_rig: Rig,
) -> None:
    """The two scanning surfaces agree (FINDING follow-up, 0.2.8).

    On 0.2.7 the facade's Python router was built from the artifact
    file's added tokens while execution saw the loader face, so the
    router could deny a literal that execution then extracted. Both now
    read the loader face.
    """
    tokenizer = sidecar_rig.tokenizer()
    try:
        assert tokenizer._added_router.holds_literal(f"a{EXTRA}b") is True
        assert tokenizer._added_router.holds_literal("plain") is False
        plan = tokenizer._added_router.plan_for(f"a{EXTRA}b")
        assert plan is not None
        assert (EXTRA, EXTRA_ID) in list(plan)
    finally:
        tokenizer.close()


def test_decode_oracle_reads_the_same_document(sidecar_rig: Rig) -> None:
    tokenizer = sidecar_rig.tokenizer()
    try:
        assert tokenizer.decode([97, EXTRA_ID, 98], skip_special_tokens=False) == (
            f"a{EXTRA}b"
        )
    finally:
        tokenizer.close()


def test_loader_face_matches_the_pinned_loader(sidecar_rig: Rig) -> None:
    """The facade's ids equal AutoTokenizer's own on the same directory."""
    import transformers

    loader = transformers.AutoTokenizer.from_pretrained(
        str(sidecar_rig.artifact_path.parent),
        use_fast=True,
        local_files_only=True,
    )
    tokenizer = sidecar_rig.tokenizer()
    try:
        for text in (f"a{EXTRA}b", "plain text", f"{EXTRA}", f"x{EXTRA}"):
            assert list(tokenizer.encode(text).ids) == loader.encode(
                text, add_special_tokens=False
            )
    finally:
        tokenizer.close()


def test_seal_guard_covers_configuration_side_literals(sidecar_rig: Rig) -> None:
    """The session seal guard reads the loader face's longest literal."""
    tokenizer = sidecar_rig.tokenizer()
    try:
        assert tokenizer._seal_end_guard_chars >= len(EXTRA)
    finally:
        tokenizer.close()


def test_certification_block_reports_the_two_annotations(
    sidecar_rig: Rig,
) -> None:
    """``explain()`` carries the claim and carry-over keys, honest nulls here.

    The rig artifact has no certification record, so both read ``None``:
    the absence of a claim, not a claim of absence. The shipped-table
    values are pinned by ``tests/registry/test_carryover_sections.py``.
    """
    tokenizer = sidecar_rig.tokenizer()
    try:
        certification = tokenizer.explain()["certification"]
        assert isinstance(certification, dict)
        assert certification["config_added_tokens"] is None
        assert certification["carryover"] is None
    finally:
        tokenizer.close()


# -- the record's face claim (0.3.0) ------------------------------------


def _record_registry(rig: Rig, loader_face_sha256: str | None) -> RegistryView:
    """A record for the rig artifact with an eligible fast CPU entry.

    The entry's engine binding is a fixture value, so the engine check
    (plan check 9) refuses it on any machine; what matters here is that
    the face check (check 8) is reached first, which needs an eligible
    entry, the installed oracle inside the certified set, and nothing
    earlier in the order refusing. ``loader_face_sha256`` is the face
    claim under test, or ``None`` for a record that makes none. The
    record declares the rig's configuration-side token, as a certified
    record for such an artifact must (an exact record without the claim
    asserts an empty subset and the load fails closed before any plan).
    """
    installed = oracle_version()
    assert installed is not None
    rows = config_added_token_rows(rig.artifact_path.parent)
    claim = (
        {
            "sha256": config_added_tokens_sha256(rows),
            "count": len(rows),
            "source": "tokenizer_config.json",
        }
        if rows
        else None
    )
    return RegistryView(
        artifacts=(
            ArtifactRecord(
                artifact_sha256=rig.artifact_sha256,
                family=rig.family,
                pipeline_id="pipeline",
                added_frontend_id="frontend",
                oracle_id="oracle",
                suite_version="test",
                evidence_id="test",
                config_added_tokens=claim,
                backends={
                    BACKEND_FAST_CPU: BackendEntry(
                        status="certified_source",
                        source_digest="a" * 64,
                        build_flags=("profile=release",),
                        toolchain="rustc (test fixture)",
                        engine="gigatoken",
                        engine_version="0.0.0+test",
                        engine_delivery="integrated",
                        engine_module="toktier._native",
                        config_id="toktier-fast-repair-v1",
                        config_digest="b" * 64,
                    )
                },
                loader_face_sha256=loader_face_sha256,
            ),
        ),
        oracles=(
            OracleRecord(
                oracle_id="oracle",
                package="tokenizers",
                certified_versions=(installed,),
                semantic_id="test",
            ),
        ),
    )


def _probe_digest(report: dict[str, object]) -> str:
    probe = report["probe"]
    assert isinstance(probe, dict)
    digest = probe["loader_face_sha256"]
    assert isinstance(digest, str)
    return digest


def test_a_face_the_record_did_not_judge_is_reported(
    sidecar_rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record names the file's digest; the loader materialized another face.

    The identity still resolves (the bytes are the certified ones), the
    accelerated routes close, the state is reference-only with the
    reason naming both digests, and the ids are those of the face the
    installed loader built -- which is what the reference executes.
    """
    registry = _record_registry(sidecar_rig, sidecar_rig.artifact_sha256)
    monkeypatch.setattr(facade_api, "shipped_registry", lambda: registry)
    tokenizer = sidecar_rig.tokenizer(device="cpu")
    try:
        face = tokenizer._backend.materialized_tokenizer_json()
        observed = loader_face_sha256(face)
        assert observed != sidecar_rig.artifact_sha256
        report = tokenizer.explain()
        assert report["planned_backend"] == BACKEND_REFERENCE
        certification = report["certification"]
        assert isinstance(certification, dict)
        assert certification["identity"] == "exact"
        assert certification["state"] == "reference_only"
        assert certification["effective_verdict"] == "unverified"
        reasons = report["plan_reasons"]
        assert isinstance(reasons, list)
        face_reasons = [
            reason
            for reason in reasons
            if reason["code"] == ReasonCode.R_LOADER_FACE_MISMATCH.value
        ]
        assert len(face_reasons) == 1
        assert face_reasons[0]["backend"] == BACKEND_FAST_CPU
        assert face_reasons[0]["detail"] == {
            "expected_digest": sidecar_rig.artifact_sha256,
            "observed_digest": observed,
        }
        probe = report["probe"]
        assert isinstance(probe, dict)
        assert probe["loader_face_sha256"] == observed

        import tokenizers

        installed_face = tokenizers.Tokenizer.from_str(face)
        for text in ("ab cd", f"ab {EXTRA} cd"):
            encoded = installed_face.encode(text, add_special_tokens=False)
            assert list(tokenizer.encode(text).ids) == [int(i) for i in encoded.ids]
    finally:
        tokenizer.close()


def test_the_judged_face_changes_nothing(
    sidecar_rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record naming the face that was materialized plans as one without the claim."""
    reports: list[dict[str, object]] = []
    for recorded in (False, True):
        claim = _probe_digest(reports[0]) if recorded else None
        registry = _record_registry(sidecar_rig, claim)
        monkeypatch.setattr(facade_api, "shipped_registry", lambda r=registry: r)
        tokenizer = sidecar_rig.tokenizer(device="cpu")
        try:
            reports.append(tokenizer.explain())
        finally:
            tokenizer.close()
    without, with_face = reports
    for key in (
        "plan_reasons",
        "certification",
        "planned_backend",
        "fallback_chain",
        "probe",
    ):
        assert with_face[key] == without[key], key
    reasons = with_face["plan_reasons"]
    assert isinstance(reasons, list)
    assert all(
        reason["code"] != ReasonCode.R_LOADER_FACE_MISMATCH.value for reason in reasons
    )


def test_shipped_records_name_the_face_the_pinned_loader_builds(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Artifact tier: every cached packaged artifact loads on its recorded face.

    The shipped ``loader_face_sha256`` values were written by the
    maintainer tool under the pinned loader; this loads each artifact
    the way a user does and checks that the face the facade materialized
    is the recorded one -- no face reason, no reference-only state from
    it -- so the tool and the backend agree on real artifacts.
    """
    artifacts = _ARTIFACTS
    if not artifacts:
        pytest.skip("set TOKTIER_TEST_ARTIFACTS to a directory of local artifacts")
    assert artifacts is not None
    pytest.importorskip("transformers")
    from toktier import Config, load
    from toktier.artifacts.manifest import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.paths import artifact_cache_dir
    from toktier.routing.tables import SUPPORT_REGISTRY

    registry = json.loads(SUPPORT_REGISTRY.read_text(encoding="utf-8"))
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    checked: list[str] = []
    for row in registry["artifacts"]:
        family = str(row["family"])
        entry = manifest.entries.get(family)
        if entry is None:
            continue
        source = Path(artifacts) / entry.directory_name
        # A directory missing any manifest file cannot be loaded offline;
        # it is left out rather than failed, like an absent artifact.
        if not all((source / item.name).is_file() for item in entry.files):
            continue
        home = tmp_path_factory.mktemp(f"face-{family}")
        config = Config(home=home, offline=True)
        cache = artifact_cache_dir(config)
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, cache / entry.directory_name)
        tokenizer = load(family, config=config, device="cpu")
        try:
            report = tokenizer.explain()
        finally:
            tokenizer.close()
        probe = report["probe"]
        assert isinstance(probe, dict)
        assert probe["loader_face_sha256"] == row["loader_face_sha256"], family
        reasons = report["plan_reasons"]
        assert isinstance(reasons, list)
        assert all(
            reason["code"] != ReasonCode.R_LOADER_FACE_MISMATCH.value
            for reason in reasons
        ), family
        certification = report["certification"]
        assert isinstance(certification, dict)
        assert certification["state"] != "reference_only", family
        checked.append(family)
    if not checked:
        pytest.skip("no packaged artifact available under TOKTIER_TEST_ARTIFACTS")
