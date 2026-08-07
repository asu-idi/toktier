"""Conformance checks for the frozen structured-error table."""

from __future__ import annotations

import re
from typing import Any

import pytest

ERROR_TABLE = (
    ("ArtifactNotFound", "ARTIFACT_NOT_FOUND"),
    ("ArtifactHashMismatch", "ARTIFACT_HASH_MISMATCH"),
    ("UncertifiedTokenizer", "UNCERTIFIED_TOKENIZER"),
    ("OracleVersionUnsupported", "ORACLE_VERSION_UNSUPPORTED"),
    ("BackendUnavailable", "BACKEND_UNAVAILABLE"),
    ("BackendExecutionFault", "BACKEND_EXECUTION_FAULT"),
    ("KernelIncompatible", "KERNEL_INCOMPATIBLE"),
    ("CudaDriverTooOld", "CUDA_DRIVER_TOO_OLD"),
    ("StoreCorrupt", "STORE_CORRUPT"),
    ("StoreFormatUnsupported", "STORE_FORMAT_UNSUPPORTED"),
    ("SessionStateMismatch", "SESSION_STATE_MISMATCH"),
    ("SessionRevisionConflict", "SESSION_REVISION_CONFLICT"),
    ("ConfigInvalid", "CONFIG_INVALID"),
    ("UnsupportedConfig", "UNSUPPORTED_CONFIG"),
    ("RegistryInvalid", "REGISTRY_INVALID"),
    ("BundleInvalid", "BUNDLE_INVALID"),
)

_ERROR_PROBE = """
import json
import sys
from collections.abc import Mapping

from toktier import errors

observed = {}
for class_name in sys.argv[1].split(","):
    error_class = getattr(errors, class_name)
    payload = {"unknown_future_key": 17}
    error = error_class("conformance probe", details=payload)
    try:
        error.details["attempted_mutation"] = True
    except TypeError:
        read_only = True
    else:
        read_only = False
    observed[class_name] = {
        "toktier_subclass": issubclass(error_class, errors.ToktierError),
        "code": error.code,
        "details_mapping": isinstance(error.details, Mapping),
        "details_equal": error.details == payload,
        "details_read_only": read_only,
    }

print(json.dumps(observed, sort_keys=True))
"""


@pytest.fixture(scope="module")
def error_observations(installed_package: Any) -> dict[str, dict[str, object]]:
    class_names = ",".join(class_name for class_name, _ in ERROR_TABLE)
    observed = installed_package.json_output(_ERROR_PROBE, class_names)
    assert isinstance(observed, dict)
    return observed


@pytest.mark.parametrize(("class_name", "expected_code"), ERROR_TABLE)
def test_frozen_error_is_importable_and_has_structured_read_only_state(
    class_name: str,
    expected_code: str,
    error_observations: dict[str, dict[str, object]],
) -> None:
    observed = error_observations[class_name]

    assert observed["toktier_subclass"] is True
    assert observed["code"] == expected_code
    assert re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+", expected_code)
    assert observed["code"] != class_name
    assert observed["details_mapping"] is True
    assert observed["details_equal"] is True
    assert observed["details_read_only"] is True


def test_code_registry_matches_the_frozen_table_exactly(
    installed_package: Any,
) -> None:
    """The installed registry carries these codes and no others."""
    observed = installed_package.json_output(
        """
import json
from toktier.errors import ERROR_CLASSES_BY_CODE

print(json.dumps({
    code: cls.__name__ for code, cls in ERROR_CLASSES_BY_CODE.items()
}))
"""
    )
    assert observed == {code: class_name for class_name, code in ERROR_TABLE}
