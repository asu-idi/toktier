"""Golden vectors for the version-normalized source identities."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
FIXTURES = Path(__file__).with_name("fixtures") / "identity_v2"
sys.path.insert(0, str(TOOLS))

import compute_identity_v2  # noqa: E402
import update_rust_package_identity  # noqa: E402
from source_identity_common import IDENTITIES  # noqa: E402

IDENTITY_NAMES = tuple(compute_identity_v2.DOMAINS)


def _materialize(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(FIXTURES / name, root)
    for identity in IDENTITIES.values():
        for relative in identity.files:
            path = root / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
        for tree in identity.trees:
            marker = root / tree / "identity-v2-fixture.txt"
            if not marker.exists():
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("fixture\n", encoding="utf-8")
    return root


def _digests(root: Path) -> dict[str, str]:
    return {
        name: compute_identity_v2.source_digest(name, root) for name in IDENTITY_NAMES
    }


def _golden() -> dict[str, dict[str, str]]:
    value = json.loads((FIXTURES / "golden.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v2_uses_exactly_the_v1_coverage() -> None:
    assert sum(map(len, compute_identity_v2.INTERNAL_DEPENDENCIES.values())) == 11
    assert len(compute_identity_v2.WORKSPACE_PACKAGES) == 7
    for name in IDENTITY_NAMES:
        assert (
            compute_identity_v2.source_paths(ROOT, name)
            == IDENTITIES[name].source_paths()
        )


def test_pure_version_bump_golden_vectors_are_equal(tmp_path: Path) -> None:
    before = _digests(_materialize(tmp_path, "version_0_2_0"))
    after = _digests(_materialize(tmp_path, "version_0_2_1"))
    golden = _golden()

    assert before == golden["version_0_2_0"]
    assert after == golden["version_0_2_1"]
    assert before == after


def test_dev_dependency_version_golden_vectors_are_different(
    tmp_path: Path,
) -> None:
    before = _digests(_materialize(tmp_path, "version_0_2_0"))
    after = _digests(_materialize(tmp_path, "dev_dependency_version"))
    golden = _golden()

    assert after == golden["dev_dependency_version"]
    assert all(before[name] != after[name] for name in IDENTITY_NAMES)


def test_review_diff_has_only_the_19_enumerated_rows(tmp_path: Path) -> None:
    root = _materialize(tmp_path, "dev_dependency_version")
    diff = compute_identity_v2.normalization_diff(root, IDENTITY_NAMES)
    removed = [
        line
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]

    assert len(removed) == len(added) == 19
    assert any('version = "=0.2.0", features' in line for line in diff.splitlines())
    assert not any('version = "=0.0.0", features' in line for line in diff.splitlines())


def test_next_rust_package_record_carries_v1_and_v2() -> None:
    document = update_rust_package_identity.document()

    for name in IDENTITY_NAMES:
        assert document[f"{name}_source_sha256_v2"] == (
            compute_identity_v2.source_digest(name)
        )
