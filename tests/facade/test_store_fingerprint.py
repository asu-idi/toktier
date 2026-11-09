"""The store fingerprint's source identity (``toktier.facade.fingerprint``).

A store owned by the native request runtime is keyed on the
version-normalized (v2) source identities of the fast CPU engine and the
native request host when the shipped registry binds them, so a rebuild
that changes only the version number keeps opening it; any other change
of a bound component refuses it. The Python entry store stays on the
build's own (v1) digests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from toktier.errors import SessionStateMismatch
from toktier.facade import api as facade_api
from toktier.facade import fingerprint as fp
from toktier.repair.registry import RepairFamily

from .conftest import Rig

_FC1, _FC1_NEXT, _FC2 = "1" * 64, "2" * 64, "a" * 64
_NH1, _NH1_NEXT, _NH2 = "3" * 64, "4" * 64, "b" * 64


def _registry(
    pairs: list[tuple[str, str]],
    hosts: list[tuple[str, str]],
    status: str = "certified",
) -> dict[str, Any]:
    artifacts = []
    for index, (v1, v2) in enumerate(pairs):
        artifacts.append(
            {
                "family": f"f{index}",
                "backends": {
                    "fast_cpu": {
                        "status": "certified_source",
                        "source_digest": v1,
                        "source_digest_v2": v2,
                    }
                },
            }
        )
    for index, (v1, v2) in enumerate(hosts):
        artifacts.append(
            {
                "family": f"g{index}",
                "backends": {
                    "gpu": {
                        "status": "certified",
                        "deliveries": {
                            "prebuilt": {
                                "status": status,
                                "host_source_digest": v1,
                                "host_source_digest_v2": v2,
                            }
                        },
                    }
                },
            }
        )
    return {"artifacts": artifacts}


def test_the_v2_pair_is_bound_only_by_agreeing_certified_entries() -> None:
    document = _registry([(_FC1, _FC2), (_FC1, _FC2)], [(_NH1, _NH2)])
    assert fp.bound_source_v2(document, fast_cpu_v1=_FC1, native_host_v1=_NH1) == (
        _FC2,
        _NH2,
    )
    # Another build's engine digest, or host digest: nothing is bound.
    assert (
        fp.bound_source_v2(document, fast_cpu_v1=_FC1_NEXT, native_host_v1=_NH1) is None
    )
    assert (
        fp.bound_source_v2(document, fast_cpu_v1=_FC1, native_host_v1=_NH1_NEXT) is None
    )
    assert fp.bound_source_v2(document, fast_cpu_v1=None, native_host_v1=_NH1) is None
    # Entries that disagree on v2 bind nothing.
    split = _registry([(_FC1, _FC2), (_FC1, "c" * 64)], [(_NH1, _NH2)])
    assert fp.bound_source_v2(split, fast_cpu_v1=_FC1, native_host_v1=_NH1) is None
    # A host binding that is not certified binds nothing.
    untested = _registry([(_FC1, _FC2)], [(_NH1, _NH2)], status="supported_untested")
    assert fp.bound_source_v2(untested, fast_cpu_v1=_FC1, native_host_v1=_NH1) is None


def test_only_a_native_owned_store_takes_the_v2_axis() -> None:
    document = _registry([(_FC1, _FC2)], [(_NH1, _NH2)])
    native = fp.source_binding(
        native_owned=True, fast_cpu_v1=_FC1, native_host_v1=_NH1, document=document
    )
    assert native == fp.SourceBinding(fp.AXIS_SOURCE_V2, _FC2, _NH2)
    python = fp.source_binding(
        native_owned=False, fast_cpu_v1=_FC1, native_host_v1=_NH1, document=document
    )
    assert python == fp.SourceBinding(fp.AXIS_SOURCE_V1, _FC1, _NH1)
    unbound = fp.source_binding(
        native_owned=True, fast_cpu_v1=_FC1_NEXT, native_host_v1=_NH1, document=document
    )
    assert unbound.axis == fp.AXIS_SOURCE_V1


def _install_build(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
    *,
    fast_cpu_v1: str,
    native_host_v1: str,
    document: dict[str, Any],
) -> None:
    from toktier.engine.gpu import native as native_module

    engine = replace(tokenizer._snapshot.fast_cpu_engine, source_digest=fast_cpu_v1)
    monkeypatch.setattr(
        tokenizer, "_snapshot", replace(tokenizer._snapshot, fast_cpu_engine=engine)
    )
    facts = native_module.NativeHostBuildFacts(
        source_digest=native_host_v1,
        build_flags=("profile=release",),
        toolchain="rustc test",
    )
    monkeypatch.setattr(native_module, "native_host_build_facts", lambda: facts)
    monkeypatch.setattr(fp, "shipped_registry_document", lambda: document)


def test_a_version_only_rebuild_keeps_the_native_store_fingerprint(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = rig.tokenizer()
    before = _registry([(_FC1, _FC2)], [(_NH1, _NH2)])
    _install_build(
        monkeypatch, tokenizer, fast_cpu_v1=_FC1, native_host_v1=_NH1, document=before
    )
    native_old = tokenizer._semantic_fingerprint(native_owned=True)
    python_old = tokenizer._semantic_fingerprint(native_owned=False)
    # The next release: new v1 digests (the version moved), same v2.
    after = _registry([(_FC1_NEXT, _FC2)], [(_NH1_NEXT, _NH2)])
    _install_build(
        monkeypatch,
        tokenizer,
        fast_cpu_v1=_FC1_NEXT,
        native_host_v1=_NH1_NEXT,
        document=after,
    )
    assert tokenizer._semantic_fingerprint(native_owned=True) == native_old
    assert tokenizer._semantic_fingerprint(native_owned=False) != python_old
    # Engine or host code changed: v2 moves, so does the fingerprint.
    for moved in (
        _registry([(_FC1_NEXT, "c" * 64)], [(_NH1_NEXT, _NH2)]),
        _registry([(_FC1_NEXT, _FC2)], [(_NH1_NEXT, "d" * 64)]),
    ):
        _install_build(
            monkeypatch,
            tokenizer,
            fast_cpu_v1=_FC1_NEXT,
            native_host_v1=_NH1_NEXT,
            document=moved,
        )
        assert tokenizer._semantic_fingerprint(native_owned=True) != native_old


def test_loader_face_and_repair_parameters_are_bound(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = rig.tokenizer()
    base = tokenizer._semantic_fingerprint(native_fast_cpu=True, native_owned=True)
    monkeypatch.setattr(tokenizer, "_loader_face_sha256", "e" * 64)
    assert (
        tokenizer._semantic_fingerprint(native_fast_cpu=True, native_owned=True) != base
    )
    monkeypatch.undo()
    assert (
        tokenizer._semantic_fingerprint(native_fast_cpu=True, native_owned=True) == base
    )
    spec = RepairFamily(
        family=rig.family,
        artifact_sha256=rig.artifact_sha256,
        margin=4,
        effective_l_max=2,
        has_normalizer=False,
        source_table_sha256="f" * 64,
    )
    monkeypatch.setattr(facade_api, "family_spec", lambda *_args: spec)
    with_spec = tokenizer._semantic_fingerprint(native_fast_cpu=True, native_owned=True)
    assert with_spec != base
    monkeypatch.setattr(
        facade_api, "family_spec", lambda *_args: replace(spec, effective_l_max=3)
    )
    assert (
        tokenizer._semantic_fingerprint(native_fast_cpu=True, native_owned=True)
        != with_spec
    )
    assert fp.repair_parameters_component(None) == "none"


def test_this_build_binds_the_shipped_v2_pair(rig: Rig) -> None:
    """On a build whose digests the shipped registry records, the native
    store carries the registry's v2 pair, and ``explain()`` says so."""
    from toktier.engine.gpu.native import native_host_build_facts

    document = fp.shipped_registry_document()
    tokenizer = rig.tokenizer(store=rig.store_path())
    tokenizer.encode("hello world", session="s")
    owners = tokenizer.explain()["store_fingerprint"]["owners"]  # type: ignore[index]
    pair = fp.bound_source_v2(
        document,
        fast_cpu_v1=tokenizer._snapshot.fast_cpu_engine.source_digest,
        native_host_v1=native_host_build_facts().source_digest,
    )
    if pair is None:
        # A development build between a source change and the rebind of
        # the registry: nothing is bound, which the axis tests cover.
        pytest.skip("this build's source digests are not the registry's")
        return
    assert owners["native"] == {
        "axis": fp.AXIS_SOURCE_V2,
        "fast_cpu": pair[0],
        "native_host": pair[1],
    }


def _fingerprint_v1(tokenizer: Any) -> str:
    """The preimage releases up to 0.3.1 wrote (reference-only session)."""
    from toktier._oracle import ORACLE_PACKAGE, oracle_version

    engine = tokenizer._snapshot.fast_cpu_engine
    digest = hashlib.sha256(b"toktier.facade.fingerprint.v1\0")
    for component in (
        tokenizer._artifact_sha256,
        ORACLE_PACKAGE,
        oracle_version() or "",
        facade_api._ENGINE_ID,
        facade_api._SESSION_SEMVER,
        "hf",
        "",
        "",
        "",
        engine.version or "",
        engine.source_digest or "",
        "\x1f".join(engine.build_flags),
        engine.toolchain or "",
        engine.config_digest or "",
    ):
        raw = component.encode("utf-8")
        digest.update(len(raw).to_bytes(4, "little"))
        digest.update(raw)
    return digest.hexdigest()


def test_a_store_written_under_the_earlier_form_is_refused(rig: Rig) -> None:
    """Without a history row naming this build identity, refused as before."""
    probe = rig.tokenizer()
    earlier = _fingerprint_v1(probe)
    probe.close()
    directory = rig.store_path()
    (directory / "entries").mkdir(parents=True)
    (directory / "meta.json").write_text(
        json.dumps({"format": 1, "fingerprint": earlier})
    )
    tokenizer = rig.tokenizer(store=directory)
    with pytest.raises(SessionStateMismatch) as caught:
        tokenizer.encode("hello", session="s")
    assert caught.value.details["stored_fingerprint"] == earlier


# -- predecessor stores ------------------------------------------------------


def test_the_shipped_history_table_verifies_and_names_both_releases() -> None:
    rows = fp.store_history()
    assert [row.release for row in rows] == ["0.3.0", "0.3.1"]
    for row in rows:
        assert len(row.fast_cpu_source_digest) == 64
        assert len(row.fast_cpu_source_digest_v2) == 64
        assert len(row.native_host_source_digest_v2) == 64
        assert row.artifacts


def test_a_tampered_history_table_is_refused(tmp_path: Path) -> None:
    from toktier.errors import RegistryInvalid

    source = fp._history_path().read_text(encoding="utf-8")
    copy = tmp_path / "history.json"
    copy.write_text(source.replace('"0.3.1"', '"0.3.9"', 1), encoding="utf-8")
    with pytest.raises(RegistryInvalid, match="digest does not match"):
        fp.read_store_history(copy)


def _predecessor_rig(
    rig: Rig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    row_v2: str = _FC2,
    loader_face: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """A build bound to (_FC2, _NH2) and a history row for release 9.9.9."""
    tokenizer = rig.tokenizer()
    document = _registry([(_FC1_NEXT, _FC2)], [(_NH1_NEXT, _NH2)])
    row = fp.PredecessorRelease(
        release="9.9.9",
        fast_cpu_source_digest=_FC1,
        fast_cpu_source_digest_v2=row_v2,
        native_host_source_digest_v2=_NH2,
        artifacts={
            rig.artifact_sha256: (
                loader_face or tokenizer._loader_face_sha256,
                "none",
            )
        },
    )
    tokenizer.close()
    monkeypatch.setattr(fp, "store_history", lambda: (row,))
    return document, row


def _bind(
    monkeypatch: pytest.MonkeyPatch, tokenizer: Any, document: dict[str, Any]
) -> None:
    _install_build(
        monkeypatch,
        tokenizer,
        fast_cpu_v1=_FC1_NEXT,
        native_host_v1=_NH1_NEXT,
        document=document,
    )


def _write_predecessor_store(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, document: dict[str, Any]
) -> tuple[Any, bytes]:
    """Sessions written under the predecessor's v1 key, as it would have."""
    writer = rig.tokenizer(store=rig.store_path())
    _bind(monkeypatch, writer, document)
    key = writer._semantic_fingerprint_v1(
        native_fast_cpu=False, fast_cpu_source_digest=_FC1
    )
    monkeypatch.setattr(writer, "_native_store_fingerprint", lambda _fast: key)
    with writer.session(session_id="chat") as session:
        session.append("hello there")
        session.append(" and more")
    revision = writer.store_session_revision("chat")
    writer.close()
    return revision, key


def test_a_predecessor_store_with_this_build_identity_opens_under_its_key(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, reference: Any
) -> None:
    document, _row = _predecessor_rig(rig, monkeypatch)
    revision, key = _write_predecessor_store(rig, monkeypatch, document)
    reader = rig.tokenizer(store=rig.store_path())
    _bind(monkeypatch, reader, document)
    with reader.session(session_id="chat", text="hello there and more") as session:
        assert session.revision == revision
        session.append(" again")
        assert list(session.ids) == reference("hello there and more again")
    native = reader.explain()["store_fingerprint"]["owners"]["native"]  # type: ignore[index]
    assert native["opened_as_release"] == "9.9.9"
    meta = json.loads((rig.store_path() / "meta.json").read_text())
    assert meta["fingerprint"] == key.hex()  # nothing rewritten


@pytest.mark.parametrize("change", ["row_v2", "loader_face"])
def test_a_predecessor_store_is_refused_when_the_row_does_not_match(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    keywords: dict[str, Any] = (
        {"row_v2": "c" * 64} if change == "row_v2" else {"loader_face": "e" * 64}
    )
    document, _row = _predecessor_rig(rig, monkeypatch, **keywords)
    _write_predecessor_store(rig, monkeypatch, document)
    reader = rig.tokenizer(store=rig.store_path())
    _bind(monkeypatch, reader, document)
    with pytest.raises(SessionStateMismatch):
        reader.encode("hello there and more", session="chat")


def test_the_python_entry_store_never_consults_the_history(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    document, _row = _predecessor_rig(rig, monkeypatch)
    binding = fp.source_binding(
        native_owned=False,
        fast_cpu_v1=_FC1_NEXT,
        native_host_v1=_NH1_NEXT,
        document=document,
    )
    assert binding.axis == fp.AXIS_SOURCE_V1
    assert (
        fp.predecessor_releases(
            binding,
            artifact_sha256=rig.artifact_sha256,
            loader_face_sha256="x",
            repair_parameters="none",
        )
        == []
    )
