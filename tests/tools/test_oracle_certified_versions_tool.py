"""The maintainer tool that widens the certified reference versions on evidence.

Contract reference: ``docs/contracts/registry.md`` Section 2 (oracle
version policy) and ``docs/contracts/fingerprint.md`` Section 7 (widening
a behavior class requires certification evidence). What is checked here
is the agreement the set rests on: the embedded reference is always a
member and names the same release as the Rust constants; a release is
added only when every compatibility reading the tool asks for exists,
names that release, and covers what the certification reading it mirrors
covers; a reading that fails, or one that attests a release outside the
set, is reported by ``--check``; the written document carries a verifying
root digest and both copies are the same bytes; and the shipped tree
passes its own check.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
for entry in (str(ROOT), str(TOOLS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import update_oracle_certified_versions as tool  # noqa: E402
from registry_common import (  # noqa: E402
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    root_digest,
)

VERSION = "0.23.3"
IDENTITIES = {"fast_cpu_v2": "f" * 64, "native_host_v2": "e" * 64}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_tree(tmp_path: Path) -> Path:
    """A tree with the shipped registry, schema, metadata and readings,
    and a complete set of compatibility readings for ``VERSION`` whose
    identities are the ones the tests inject."""
    root = tmp_path / "tree"
    for relative in (
        "tables/support_registry.json",
        "src/toktier/routing/tables/support_registry.v1.json",
        "schemas/support_registry.schema.json",
        "pyproject.toml",
        str(tool.CPU_READING),
        *(str(tool.gpu_reading(arch)) for arch in tool.GPU_ARCHITECTURES),
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (root / tool.COMPAT_DIRECTORY).mkdir(parents=True)
    cpu = _read(root / tool.CPU_READING)
    cpu["oracle"] = {"package": "tokenizers", "version": VERSION}
    cpu["engine"]["source_digest_v2"] = IDENTITIES["fast_cpu_v2"]
    _write(tool.compat_path(root, tool.CPU_READING, VERSION), cpu)
    for arch in tool.GPU_ARCHITECTURES:
        gpu = _read(root / tool.gpu_reading(arch))
        gpu["oracle"] = {"package": "tokenizers", "version": VERSION}
        gpu["native_host_build_facts"]["host_source_digest_v2"] = IDENTITIES[
            "native_host_v2"
        ]
        _write(tool.compat_path(root, tool.gpu_reading(arch), VERSION), gpu)
    return root


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def test_the_shipped_tree_passes_its_own_check() -> None:
    assert tool.check_problems() == []


def test_the_embedded_reference_names_the_rust_constants() -> None:
    """One release, stated in three places that have to agree."""
    assert tool.EMBEDDED_REFERENCE_VERSION == "0.22.2"
    lib = (ROOT / "crates/toktier/src/lib.rs").read_text(encoding="utf-8")
    jit = (ROOT / "crates/toktier/src/jit.rs").read_text(encoding="utf-8")
    assert re.search(
        r'pub const ORACLE: &str = "tokenizers=='
        + re.escape(tool.EMBEDDED_REFERENCE_VERSION)
        + '";',
        lib,
    )
    assert re.search(
        r'const FROZEN_ORACLE_VERSION: &str = "'
        + re.escape(tool.EMBEDDED_REFERENCE_VERSION)
        + '";',
        jit,
    )
    registry = _read(ROOT / "tables/support_registry.json")
    assert tool.EMBEDDED_REFERENCE_VERSION in tool.certified_versions(registry)


def test_the_metadata_range_admits_every_certified_version() -> None:
    floor, ceiling = tool.metadata_range(ROOT)
    assert floor == tool.EMBEDDED_REFERENCE_VERSION
    registry = _read(ROOT / "tables/support_registry.json")
    for version in tool.certified_versions(registry):
        assert tool.version_key(floor) <= tool.version_key(version)
        assert tool.version_key(version) < tool.version_key(ceiling)


def test_versions_sort_numerically() -> None:
    assert sorted(["0.23.3", "0.22.2", "0.23.10", "0.23.2"], key=tool.version_key) == [
        "0.22.2",
        "0.23.2",
        "0.23.3",
        "0.23.10",
    ]


def test_a_release_is_added_on_complete_evidence(tmp_path: Path) -> None:
    root = build_tree(tmp_path)
    before = tool.certified_versions(_read(tool.registry_path(root)))
    assert VERSION not in before

    generated = tool.add_versions(root, [VERSION], IDENTITIES)

    assert tool.certified_versions(generated) == sorted(
        [*before, VERSION], key=tool.version_key
    )
    written = _read(tool.registry_path(root))
    assert written == generated
    assert written["root_digest"] == root_digest(
        {k: v for k, v in written.items() if k != "root_digest"}, REGISTRY_DOMAIN_TAG
    )
    assert (
        tool.packaged_path(root).read_bytes() == tool.registry_path(root).read_bytes()
    )
    # The semantic id names the behavior class and does not move with the set.
    for record in tool.oracle_records(written):
        assert record["semantic_id"] == "tokenizers.0.22.2"
    assert tool.check_problems(root, IDENTITIES) == []


def test_missing_or_failing_evidence_refuses_the_release(tmp_path: Path) -> None:
    root = build_tree(tmp_path)
    sm89 = tool.compat_path(root, tool.gpu_reading("sm_89"), VERSION)
    sm89.unlink()
    with pytest.raises(GenerationError, match="compatibility reading is missing"):
        tool.add_versions(root, [VERSION], IDENTITIES)

    root = build_tree(tmp_path / "second")
    cpu = tool.compat_path(root, tool.CPU_READING, VERSION)
    document = _read(cpu)
    document["mismatches"] = 1
    document["rows"][0]["mismatches"] = 1
    _write(cpu, document)
    with pytest.raises(GenerationError, match="mismatches"):
        tool.add_versions(root, [VERSION], IDENTITIES)

    root = build_tree(tmp_path / "third")
    cpu = tool.compat_path(root, tool.CPU_READING, VERSION)
    document = _read(cpu)
    document["oracle"]["version"] = "0.23.2"
    _write(cpu, document)
    with pytest.raises(GenerationError, match="file name says"):
        tool.add_versions(root, [VERSION], IDENTITIES)

    root = build_tree(tmp_path / "fourth")
    with pytest.raises(GenerationError, match="not this tree's fast CPU engine"):
        tool.add_versions(root, [VERSION], {**IDENTITIES, "fast_cpu_v2": "0" * 64})

    root = build_tree(tmp_path / "fifth")
    with pytest.raises(GenerationError, match="outside the package metadata range"):
        tool.add_versions(root, ["0.24.0"], IDENTITIES)


def test_check_reports_a_reading_outside_the_set_and_a_set_without_evidence(
    tmp_path: Path,
) -> None:
    root = build_tree(tmp_path)
    # Evidence present, set not widened: the readings attest a release the
    # registry does not certify.
    problems = tool.check_problems(root, IDENTITIES)
    assert any("attests 0.23.3" in problem for problem in problems)

    # Set widened, then the evidence changed under it.
    tool.add_versions(root, [VERSION], IDENTITIES)
    assert tool.check_problems(root, IDENTITIES) == []
    cpu = tool.compat_path(root, tool.CPU_READING, VERSION)
    document = _read(cpu)
    document["documents"] -= 1
    _write(cpu, document)
    problems = tool.check_problems(root, IDENTITIES)
    assert any("documents" in problem for problem in problems)

    # A hand-widened set with no evidence at all.
    root = build_tree(tmp_path / "second")
    shutil.rmtree(root / tool.COMPAT_DIRECTORY)
    registry = _read(tool.registry_path(root))
    widened = tool.widened_document(copy.deepcopy(registry), [VERSION])
    for path in (tool.registry_path(root), tool.packaged_path(root)):
        path.write_bytes(tool.serialise_document(widened))
    problems = tool.check_problems(root, IDENTITIES)
    assert sum("compatibility reading is missing" in p for p in problems) == 1 + len(
        tool.GPU_ARCHITECTURES
    )

    # Copies that differ.
    root = build_tree(tmp_path / "third")
    tool.packaged_path(root).write_bytes(b"{}\n")
    assert any(
        "not the serialised registry" in problem
        for problem in tool.check_problems(root, IDENTITIES)
    )
