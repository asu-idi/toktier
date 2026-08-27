"""Static release identity and artifact-verifier contracts."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFY_IDENTITY = ROOT / "tools" / "verify_release_identity.py"
VERIFY_ARTIFACTS = ROOT / "tools" / "verify_release_artifacts.py"


def _artifact_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_release_artifacts", VERIFY_ARTIFACTS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel_with_payload(path: Path, payload: bytes) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("toktier/_native.abi3.so", payload)
    return path


def test_release_identity_is_v026() -> None:
    subprocess.run(
        [sys.executable, str(VERIFY_IDENTITY), "--tag", "v0.2.6"],
        check=True,
    )


def test_release_identity_rejects_another_tag() -> None:
    completed = subprocess.run(
        [sys.executable, str(VERIFY_IDENTITY), "--tag", "v0.1.0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "must be 'v0.2.6'" in completed.stderr


def test_release_artifact_set_is_one_abi3_linux_wheel() -> None:
    source = VERIFY_ARTIFACTS.read_text(encoding="utf-8")
    assert (
        'EXPECTED_WHEEL = "toktier-0.2.6-cp310-abi3-manylinux_2_34_x86_64.whl"'
        in source
    )


def test_release_artifact_gate_requires_the_pinned_fastokens_extra() -> None:
    """The extra names the published distribution, never the upstream one."""
    import json

    import tomllib

    source = VERIFY_ARTIFACTS.read_text(encoding="utf-8")
    assert "tools/fastokens_binding.json" in source
    assert "requires the upstream fastokens distribution" in source
    binding = json.loads((ROOT / "tools/fastokens_binding.json").read_bytes())
    pinned = binding["distribution"]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = project["project"]["optional-dependencies"]["fastokens"]
    assert extra == [f"{pinned['name']}=={pinned['version']}"]


@pytest.mark.parametrize("representation", ["hex", "bytes"])
def test_release_artifact_gate_rejects_identity_sentinel(
    tmp_path: Path, representation: str
) -> None:
    verifier = _artifact_verifier()
    sentinel = (
        verifier.IDENTITY_SENTINEL_HEX.encode("ascii")
        if representation == "hex"
        else verifier.IDENTITY_SENTINEL_BYTES
    )
    wheel = _wheel_with_payload(tmp_path / "sentinel.whl", b"prefix" + sentinel)

    with zipfile.ZipFile(wheel) as archive, pytest.raises(
        ValueError, match="contains the sentinel build identity"
    ):
        verifier._verify_no_identity_sentinel(archive, archive.namelist())


def test_release_artifact_gate_accepts_normal_identity(tmp_path: Path) -> None:
    verifier = _artifact_verifier()
    wheel = _wheel_with_payload(tmp_path / "normal.whl", b"\x7fELF" + b"a" * 64)

    with zipfile.ZipFile(wheel) as archive:
        verifier._verify_no_identity_sentinel(archive, archive.namelist())


def _archive_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_rust_source_archive", ROOT / "tools" / "build_rust_source_archive.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_source_archive_carries_what_its_readme_links_to() -> None:
    """The README travels inside the archive, so its local links have to
    resolve from there. Four of them used to be dead: the translation and
    the three evidence manifests, each named as the machine-readable
    record behind a claim the same README makes.

    This asserts the declared inputs rather than building the archive,
    which vendors the whole dependency graph.
    """
    builder = _archive_builder()
    declared = set(builder.ROOT_FILES)
    trees = set(builder.TREES)

    assert "README.zh-CN.md" in declared
    assert "evidence" in trees
    # `generate_sibling_aliases.py --check` is in the same README block.
    assert "data" in trees

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in (
        "README.zh-CN.md",
        "evidence/evidence_manifest.json",
        "evidence/evidence_manifest_added_families.json",
        "evidence/evidence_manifest_kimi_band.json",
    ):
        assert target in readme, target
        head = target.split("/")[0]
        assert head in declared or head in trees, target


def test_the_readme_scopes_its_repository_only_self_checks() -> None:
    """Two of the seven commands cannot run from the archive. Saying so is
    the difference between a documented boundary and a dead end."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Two are repository-only" in readme
    for command in ("generate_registry.py --release-check", "dev.py test-packaging"):
        _, _, after = readme.partition("Two are repository-only")
        assert command in after.split("The prerequisites are stated here")[0]


# -- the two repository-only checks, from the archive --------------------


def _archive_shaped(root: Path) -> Path:
    """A tree that carries the marker the archive builder writes."""
    (root / "SOURCE-MANIFEST.json").write_text(
        '{"schema": "toktier.rust_source_archive.v1", "files": [], '
        '"root_digest": "sha256:0"}\n',
        encoding="utf-8",
    )
    return root


def _tools_module(name: str) -> ModuleType:
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    return importlib.import_module(name)


def test_the_archive_is_told_apart_by_its_own_manifest(tmp_path: Path) -> None:
    scan_common = _tools_module("scan_common")

    assert scan_common.vendored_source_archive(ROOT) is False
    assert scan_common.vendored_source_archive(tmp_path) is False
    assert scan_common.vendored_source_archive(_archive_shaped(tmp_path)) is True

    # A file of that name carrying something else is not the marker.
    (tmp_path / "SOURCE-MANIFEST.json").write_text("{}", encoding="utf-8")
    assert scan_common.vendored_source_archive(tmp_path) is False
    (tmp_path / "SOURCE-MANIFEST.json").write_text("not json", encoding="utf-8")
    assert scan_common.vendored_source_archive(tmp_path) is False


def test_the_release_check_declines_in_the_archive_rather_than_borrowing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """It used to read the extension of whatever wheel was installed.

    That answers a question about that installation while looking like an
    answer about this tree, and the README calls the command
    repository-only.
    """
    generate_registry = _tools_module("generate_registry")
    monkeypatch.setattr(
        generate_registry, "REPOSITORY_ROOT", _archive_shaped(tmp_path)
    )
    borrowed: list[str] = []
    monkeypatch.setattr(
        generate_registry,
        "_adopt_installed_native_extension",
        lambda: borrowed.append("adopted"),
    )

    assert generate_registry.main(["--release-check"]) == 3

    captured = capsys.readouterr()
    assert captured.err.startswith("declined: ")
    assert "Nothing was checked" in captured.err
    assert captured.out == ""
    assert borrowed == []


def test_test_packaging_declines_in_the_archive_rather_than_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``tests/`` is not carried there; pytest's usage exit is not an answer."""
    dev = _tools_module("dev")
    monkeypatch.setattr(dev, "ROOT", _archive_shaped(tmp_path))
    monkeypatch.setattr(
        dev,
        "run_commands",
        lambda commands: pytest.fail("the suite was run from the archive"),
    )

    assert dev.main(["test-packaging"]) == 3

    captured = capsys.readouterr()
    assert captured.err.startswith("declined: ")
    assert "Nothing was run" in captured.err


def test_a_repository_checkout_still_runs_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decline is the archive's answer, not everyone's."""
    dev = _tools_module("dev")
    ran: list[tuple[tuple[str, ...], ...]] = []

    def record(commands: tuple[tuple[str, ...], ...]) -> int:
        ran.append(commands)
        return 0

    monkeypatch.setattr(dev, "run_commands", record)

    assert dev.main(["test-packaging"]) == 0
    assert ran == [dev.TEST_PACKAGING_COMMANDS]
