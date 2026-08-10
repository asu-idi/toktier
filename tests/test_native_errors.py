"""The native binding boundary raises the public structured errors.

The ``toktier._native`` extension defines no exception hierarchy of its
own (decision 0004): a failure crossing the pyo3 boundary is an instance
of the ``toktier.errors`` classes -- caught by ``toktier.ToktierError``,
carrying the frozen ``.code``, and exposing ``.details`` as a read-only
mapping, exactly like a Python-raised error.

The extension is built from ``crates/toktier-py``. Set ``TOKTIER_PY_SO``
to point at a prebuilt shared object; otherwise an existing
``target/{debug,release}`` build is used. The module skips cleanly when
neither is present: building here would fold a minutes-long ``cargo
build`` into a routine pytest run, so the build stays an explicit,
visible step (``cargo build -p toktier-py``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# The isolated-home fixture clears TOKTIER_* per test; capture the
# override while the outer environment is still visible.
_SO_OVERRIDE = os.environ.get("TOKTIER_PY_SO")

# Resolve the package from this tree ahead of any installed copy, so the
# extension binds to the classes under test.
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def _shared_object() -> Path:
    if _SO_OVERRIDE:
        return Path(_SO_OVERRIDE)
    candidates = [
        REPOSITORY_ROOT / "target/debug/lib_native.so",
        REPOSITORY_ROOT / "target/release/lib_native.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # No implicit build: a cargo build inside a test run is a hidden,
    # minutes-long step on a cold checkout. Build the extension
    # explicitly (cargo build -p toktier-py) or set TOKTIER_PY_SO.
    message = (
        "toktier._native is not prebuilt; run `cargo build -p toktier-py` "
        "or set TOKTIER_PY_SO to a shared object"
    )
    pytest.skip(message)
    raise RuntimeError(message)  # pragma: no cover - skip always raises


@pytest.fixture(scope="session")
def native_module() -> ModuleType:
    shared_object = _shared_object()
    spec = importlib.util.spec_from_file_location("toktier._native", shared_object)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    ids = list(range(len(text)))
    spans = [(index, index + 1) for index in range(len(text))]
    return ids, spans


def test_native_failure_is_caught_as_the_public_toktier_error(
    native_module: ModuleType,
) -> None:
    import toktier
    from toktier import errors

    engine = native_module.CallbackEncoder(0, _encode)
    store = native_module.SessionStore()
    key = store.register_fingerprint(b"\x11" * 32, 0)
    handle, revision, _token_count = store.put(key, "hello world", engine)

    with pytest.raises(toktier.ToktierError) as caught:
        store.append(handle, "!", revision + 17, engine)

    error = caught.value
    assert isinstance(error, errors.SessionRevisionConflict)
    assert error.code == "SESSION_REVISION_CONFLICT"
    assert isinstance(error.details, Mapping)
    assert error.details["expected_revision"] == revision + 17
    assert error.details["stored_revision"] == revision
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], error.details)["attempted_mutation"] = True


def test_native_corruption_failure_is_public_with_read_only_details(
    native_module: ModuleType,
) -> None:
    from toktier import errors

    engine = native_module.CallbackEncoder(0, _encode)
    store = native_module.SessionStore()
    key = store.register_fingerprint(b"\x22" * 32, 0)

    with pytest.raises(errors.ToktierError) as caught:
        store.import_session(key, b"garbage-record", engine)

    error = caught.value
    assert isinstance(
        error, (errors.StoreCorrupt, errors.StoreFormatUnsupported)
    )
    assert error.code in ("STORE_CORRUPT", "STORE_FORMAT_UNSUPPORTED")
    assert isinstance(error.details, Mapping)
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], error.details)["attempted_mutation"] = True


def test_native_reexports_are_the_public_error_classes(
    native_module: ModuleType,
) -> None:
    from toktier import errors

    for name in (
        "ToktierError",
        "StoreCorrupt",
        "StoreFormatUnsupported",
        "SessionStateMismatch",
        "SessionRevisionConflict",
        "ConfigInvalid",
    ):
        assert getattr(native_module, name) is getattr(errors, name)


def test_plain_argument_misuse_stays_a_standard_exception(
    native_module: ModuleType,
) -> None:
    engine = native_module.CallbackEncoder(0, _encode)
    store = native_module.SessionStore()
    key = store.register_fingerprint(b"\x33" * 32, 0)
    store.put(key, "text", engine)

    with pytest.raises(KeyError):
        store.revision(10**9)
