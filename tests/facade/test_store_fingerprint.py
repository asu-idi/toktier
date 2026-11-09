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


def _planner_binds_this_build(tokenizer: Any) -> bool:
    """The planner's own answer, independent of the store binder.

    True when the executing fast CPU engine verifies against an eligible
    fast CPU entry of the shipped registry (plan check 9: no
    ``R_ENGINE_BINDING_MISMATCH`` axis) and the native host verifies
    against an eligible prebuilt delivery (no ``host_source`` mismatch of
    check 13).
    """
    from toktier.engine.gpu.native import native_host_build_facts
    from toktier.policy import BACKEND_FAST_CPU, BACKEND_GPU
    from toktier.routing.plan import (
        fast_cpu_binding_mismatches,
        prebuilt_host_binding_mismatches,
    )
    from toktier.routing.registry_load import shipped_registry

    registry = shipped_registry()
    engine = tokenizer._snapshot.fast_cpu_engine
    fast_cpu = any(
        not fast_cpu_binding_mismatches(entry, engine)
        for entry in registry.eligible_entries(BACKEND_FAST_CPU)
    )
    host = native_host_build_facts()
    host_bound = any(
        entry.for_delivery("prebuilt").status in {"certified", "certified_source"}
        and entry.for_delivery("prebuilt").host_source_digest is not None
        and not any(
            axis == "host_source"
            for axis, _code, _detail in prebuilt_host_binding_mismatches(
                entry.for_delivery("prebuilt"),
                host_source_digest=host.source_digest,
                host_build_flags=host.build_flags,
                host_toolchain=host.toolchain,
                driver_version=None,
            )
        )
        for entry in registry.eligible_entries(BACKEND_GPU)
    )
    return fast_cpu and host_bound


def test_this_build_binds_the_shipped_v2_pair(rig: Rig) -> None:
    """On a build whose digests the shipped registry records, the native
    store carries the registry's v2 pair, and ``explain()`` says so.

    Whether this build is bound is decided by the planner's own binding
    checks, not by the function under test: on a build the planner
    certifies by binding, a binder answering ``None`` fails here.
    """
    from toktier.engine.gpu.native import native_host_build_facts

    document = fp.shipped_registry_document()
    tokenizer = rig.tokenizer(store=rig.store_path())
    tokenizer.encode("hello world", session="s")
    owners = tokenizer.explain()["store_fingerprint"]["owners"]  # type: ignore[index]
    if not _planner_binds_this_build(tokenizer):
        # A development build between a source change and the rebind of
        # the registry: the planner binds nothing either, which the axis
        # tests cover.
        pytest.skip("the planner does not bind this build's source digests")
        return
    pair = fp.bound_source_v2(
        document,
        fast_cpu_v1=tokenizer._snapshot.fast_cpu_engine.source_digest,
        native_host_v1=native_host_build_facts().source_digest,
    )
    assert pair is not None, "the planner binds this build; the store binder does not"
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


@pytest.mark.parametrize("owner", ["native", "python"])
def test_a_store_of_another_release_is_refused_plainly_and_rebuilt(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, reference: Any, owner: str
) -> None:
    """A store whose fingerprint this build does not write (an earlier
    release's, here the v1 form) is never served from: the refusal says
    it comes from another release or build and how to go on, and once the
    directory is removed the store is rebuilt and the ids are the
    reference's."""
    import shutil

    probe = rig.tokenizer()
    earlier = _fingerprint_v1(probe)
    probe.close()
    directory = rig.store_path()
    (directory / "entries").mkdir(parents=True)
    (directory / "meta.json").write_text(
        json.dumps({"format": 1, "fingerprint": earlier})
    )
    tokenizer = rig.tokenizer(store=directory)
    if owner == "python":
        monkeypatch.setattr(
            tokenizer, "_construct_native_request_runtime", lambda: None
        )
    with pytest.raises(SessionStateMismatch) as caught:
        tokenizer.encode("hello there", session="chat")
    assert "written by another toktier release or build" in str(caught.value)
    assert "Remove the directory" in str(caught.value)
    assert set(caught.value.details) == {"expected_fingerprint", "stored_fingerprint"}
    assert caught.value.details["stored_fingerprint"] == earlier
    tokenizer.close()
    shutil.rmtree(directory)
    rebuilt = rig.tokenizer(store=directory)
    if owner == "python":
        monkeypatch.setattr(rebuilt, "_construct_native_request_runtime", lambda: None)
    with rebuilt.session(session_id="chat", text="hello there") as session:
        session.append(" again")
        assert list(session.ids) == reference("hello there again")
    assert json.loads((directory / "meta.json").read_text())["fingerprint"] != earlier
    rebuilt.close()


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


# -- the two owners of one build ----------------------------------------------
#
# On a bound build the native request runtime keys its store on the v2 axis
# and the Python entry store on the exact build (v1 axis). The native runtime
# also opens the store this same build's Python owner wrote, under its stored
# value; the Python owner refuses a store the native runtime holds, with its
# own message.

_OWNERS = ("native", "python", "python_callback")


def _owner_tokenizer(
    rig: Rig,
    monkeypatch: pytest.MonkeyPatch,
    document: dict[str, Any],
    owner: str,
    directory: Path,
) -> Any:
    tokenizer = rig.tokenizer(store=directory)
    _bind(monkeypatch, tokenizer, document)
    if owner != "native":
        monkeypatch.setattr(
            tokenizer, "_construct_native_request_runtime", lambda: None
        )
    if owner == "python_callback":
        monkeypatch.setattr(
            tokenizer, "_native_repair_encoder", lambda reference=None: None
        )
    return tokenizer


def _bound_document() -> dict[str, Any]:
    return _registry([(_FC1_NEXT, _FC2)], [(_NH1_NEXT, _NH2)])


def _write_session(tokenizer: Any) -> tuple[int | None, str]:
    text = "hello there"
    with tokenizer.session(session_id="chat", text=text) as session:
        session.append(" and more")
        text += " and more"
    revision = tokenizer.store_session_revision("chat")
    tokenizer.close()
    return revision, text


def _assert_owner_refusal(error: SessionStateMismatch, directory: Path) -> None:
    """The refused direction: its own message, the two documented details."""
    assert "held by this build's native request runtime" in str(error)
    stored = json.loads((directory / "meta.json").read_text())["fingerprint"]
    assert set(error.details) == {"expected_fingerprint", "stored_fingerprint"}
    assert error.details["stored_fingerprint"] == stored
    assert error.details["expected_fingerprint"] != stored


def test_the_native_runtime_opens_the_store_this_build_s_python_owner_wrote(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, reference: Any
) -> None:
    document = _bound_document()
    directory = rig.store_path()
    writer = _owner_tokenizer(rig, monkeypatch, document, "python", directory)
    revision, text = _write_session(writer)
    written = json.loads((directory / "meta.json").read_text())["fingerprint"]
    reader = _owner_tokenizer(rig, monkeypatch, document, "native", directory)
    with reader.session(session_id="chat", text=text) as session:
        assert session.revision == revision
        session.append(" again")
        assert list(session.ids) == reference(text + " again")
    native = reader.explain()["store_fingerprint"]["owners"]["native"]
    assert native["opened_as_owner"] == "python"
    assert native["opened_axis"] == fp.AXIS_SOURCE_V1
    # Opened under the stored value: nothing rewritten.
    meta = json.loads((directory / "meta.json").read_text())
    assert meta["fingerprint"] == written
    reader.close()


def test_the_python_owner_refuses_the_store_this_build_s_native_runtime_holds(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _bound_document()
    directory = rig.store_path()
    writer = _owner_tokenizer(rig, monkeypatch, document, "native", directory)
    _write_session(writer)
    reader = _owner_tokenizer(rig, monkeypatch, document, "python", directory)
    with pytest.raises(SessionStateMismatch, match="native request runtime") as caught:
        reader.encode("hello there and more", session="chat")
    _assert_owner_refusal(caught.value, directory)


def test_an_unbound_build_shares_one_key_between_its_owners(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, reference: Any
) -> None:
    """Nothing bound: one exact-build key, both directions open."""
    unbound = _registry([(_FC1, _FC2)], [(_NH1, _NH2)])  # not this build's v1
    directory = rig.store_path()
    writer = _owner_tokenizer(rig, monkeypatch, unbound, "native", directory)
    revision, text = _write_session(writer)
    reader = _owner_tokenizer(rig, monkeypatch, unbound, "python", directory)
    with reader.session(session_id="chat", text=text) as session:
        assert session.revision == revision
        assert list(session.ids) == reference(text)
    reader.close()


class _FailingRuntime:
    def __init__(self, *_args: Any, **_keywords: Any) -> None:
        raise RuntimeError("injected: the native request runtime did not start")


@pytest.mark.parametrize("writer_owner", ["native", "python"])
def test_the_initialisation_fallback_with_a_persistent_store(
    rig: Rig,
    monkeypatch: pytest.MonkeyPatch,
    reference: Any,
    writer_owner: str,
) -> None:
    """The facade's own fallback when the native runtime fails to start.

    A store its Python owner wrote is served on the fallback path; a store
    the native runtime holds is refused there with the specific message
    (the direction that stays refused), and the guard is recorded.
    """
    from toktier import _native

    document = _bound_document()
    directory = rig.store_path()
    writer = _owner_tokenizer(rig, monkeypatch, document, writer_owner, directory)
    revision, text = _write_session(writer)
    monkeypatch.setattr(_native, "NativeRuntime", _FailingRuntime)
    reader = rig.tokenizer(store=directory)
    _bind(monkeypatch, reader, document)
    if writer_owner == "python":
        with reader.session(session_id="chat", text=text) as session:
            assert session.revision == revision
            session.append(" again")
            assert list(session.ids) == reference(text + " again")
    else:
        with pytest.raises(SessionStateMismatch) as caught:
            reader.encode(text, session="chat")
        _assert_owner_refusal(caught.value, directory)
    guard = reader.explain()["native_request_guard"]
    assert guard["reason"] == "native_request_initialisation_guard"  # type: ignore[index]
    reader.close()


@pytest.mark.parametrize("reader_owner", _OWNERS)
@pytest.mark.parametrize("writer_owner", _OWNERS)
def test_the_owner_matrix(
    rig: Rig,
    monkeypatch: pytest.MonkeyPatch,
    reference: Any,
    writer_owner: str,
    reader_owner: str,
) -> None:
    """Every (writer, reader) owner pair of one bound build either opens --
    revision kept, ids equal to the reference at every step -- or is
    refused with the specific message; only the Python owner reading the
    native runtime's store is refused."""
    document = _bound_document()
    directory = rig.store_path()
    writer = _owner_tokenizer(rig, monkeypatch, document, writer_owner, directory)
    revision, text = _write_session(writer)
    reader = _owner_tokenizer(rig, monkeypatch, document, reader_owner, directory)
    refused = writer_owner == "native" and reader_owner != "native"
    if refused:
        with pytest.raises(SessionStateMismatch) as caught:
            reader.encode(text, session="chat")
        _assert_owner_refusal(caught.value, directory)
    else:
        with reader.session(session_id="chat", text=text) as session:
            assert session.revision == revision
            assert list(session.ids) == reference(text)
            for piece in (" reader one", "\n\nreader two " + "y" * 300, " z"):
                session.append(piece)
                text += piece
                assert list(session.ids) == reference(text)
    reader.close()
