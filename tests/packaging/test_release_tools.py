"""Static release identity and artifact-verifier contracts."""

from __future__ import annotations

import importlib
import importlib.util
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFY_IDENTITY = ROOT / "tools" / "verify_release_identity.py"
VERIFY_ARTIFACTS = ROOT / "tools" / "verify_release_artifacts.py"
SMOKE_SCRIPT = ROOT / "tools" / "run_packaging_smoke.sh"


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
        [sys.executable, str(VERIFY_IDENTITY), "--tag", "v0.2.8"],
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
    assert "must be 'v0.2.8'" in completed.stderr


def test_release_artifact_set_is_one_abi3_linux_wheel() -> None:
    source = VERIFY_ARTIFACTS.read_text(encoding="utf-8")
    assert (
        'EXPECTED_WHEEL = "toktier-0.2.8-cp310-abi3-manylinux_2_34_x86_64.whl"'
        in source
    )


def test_the_packaging_smoke_counts_match_the_shipped_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The smoke script's row counts have to be the table's own counts.

    The script runs on the release path (the publish workflow calls it in
    the build job), and nothing used to read its two sibling-registry
    assertions, so a wave that carried a new count into the prose left the
    script asserting the old one. Reading the literals back and comparing
    them with the shipped table keeps the two in step.
    """
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from toktier.artifacts import shipped_sibling_aliases

    source = SMOKE_SCRIPT.read_text(encoding="utf-8")
    total = re.search(r"assert len\(aliases\.records\) == (\d+)", source)
    packaged = re.search(
        r"assert sum\(record\.canonical_packaged for record in aliases\.records\)"
        r" == (\d+)",
        source,
    )
    assert total is not None and packaged is not None, source

    registry = shipped_sibling_aliases()
    assert int(total.group(1)) == len(registry.records)
    assert int(packaged.group(1)) == sum(
        record.canonical_packaged for record in registry.records
    )


def test_release_artifact_gate_requires_the_pinned_fastokens_extra() -> None:
    """The extra names the published distribution, never the upstream one."""
    import json

    # The package targets 3.10, where the parser is the tomli backport.
    toml_module = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    toml = importlib.import_module(toml_module)

    source = VERIFY_ARTIFACTS.read_text(encoding="utf-8")
    assert "tools/fastokens_binding.json" in source
    assert "requires the upstream fastokens distribution" in source
    binding = json.loads((ROOT / "tools/fastokens_binding.json").read_bytes())
    pinned = binding["distribution"]
    project = toml.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = project["project"]["optional-dependencies"]["fastokens"]
    assert extra == [f"{pinned['name']}=={pinned['version']}"]


def test_requirement_key_reads_both_marker_spellings() -> None:
    """A marker quoted either way has to compare equal.

    PEP 508 leaves the quoting to whoever writes the metadata, and the wheel
    builder picks single quotes. Matching one spelling literally once read a
    correct wheel as one that had lost the pinned extra.
    """
    verifier = _artifact_verifier()
    single = "toktier-fastokens==0.3.1.1 ; extra == 'fastokens'"
    double = 'toktier-fastokens==0.3.1.1 ; extra == "fastokens"'
    assert verifier._requirement_key(single) == verifier._requirement_key(double)
    for spelling in (single, double):
        key = verifier._requirement_key(spelling)
        assert key.startswith("toktier-fastokens==0.3.1.1")
        assert 'extra=="fastokens"' in key


#: The `METADATA` a real release wheel carries, taken verbatim from one
#: `maturin build --release` produced for this tree. The header block is
#: trimmed to what the gate reads; every `Requires-Dist` and
#: `Provides-Extra` line is exactly as the builder wrote it, single-quoted
#: markers and all. It is a literal on purpose: rebuilding these lines
#: from `pyproject.toml` would test the gate against the same source the
#: gate already reads, which is how the marker-quoting defect survived
#: having a test. When a release moves a pin, this fixture is refreshed
#: from the new wheel and the failure here is the reminder.
REAL_WHEEL_METADATA = b"""\
Metadata-Version: 2.4
Name: toktier
Version: 0.2.7
Classifier: Programming Language :: Rust
Requires-Dist: tokenizers==0.22.2
Requires-Dist: transformers==4.57.6
Requires-Dist: huggingface-hub>=0.30
Requires-Dist: platformdirs>=3.0
Requires-Dist: tomli>=2.0 ; python_full_version < '3.11'
Requires-Dist: toktier-fastokens==0.3.1.1 ; extra == 'fastokens'
Requires-Dist: torch~=2.11 ; extra == 'gpu'
Requires-Dist: numpy>=1.24 ; extra == 'gpu'
Requires-Dist: torch~=2.11 ; extra == 'gpu-jit'
Requires-Dist: ninja~=1.11 ; extra == 'gpu-jit'
Requires-Dist: numpy>=1.24 ; extra == 'gpu-jit'
Provides-Extra: fastokens
Provides-Extra: gpu
Provides-Extra: gpu-jit
Summary: Correctness-first tokenization with a certified fast path.

Long description.
"""


def test_the_gate_accepts_a_real_wheels_metadata() -> None:
    """The fixture the fastokens check never had.

    Its test read this tool's source and the pyproject table, both of
    which said the right thing while the matching logic read a correct
    wheel as one that had lost the pinned extra. Feeding the builder's
    own text through the matching logic is what tells the two apart.
    """
    import json

    verifier = _artifact_verifier()

    verifier.verify_metadata(REAL_WHEEL_METADATA)

    # The fixture is a snapshot, so it says out loud which release it is
    # a snapshot of; a moved pin fails here with a reason rather than
    # inside the matcher.
    binding = json.loads((ROOT / "tools/fastokens_binding.json").read_bytes())
    pinned = binding["distribution"]
    assert (
        f"{pinned['name']}=={pinned['version']} ; extra == 'fastokens'".encode()
        in REAL_WHEEL_METADATA
    ), "refresh REAL_WHEEL_METADATA from a wheel built after the pin moved"


@pytest.mark.parametrize(
    ("edit", "complaint"),
    [
        (
            (
                b"Requires-Dist: toktier-fastokens==0.3.1.1 ; extra == 'fastokens'",
                b"Requires-Dist: fastokens==0.3.1 ; extra == 'fastokens'",
            ),
            "requires the upstream fastokens distribution",
        ),
        (
            (
                b"Requires-Dist: toktier-fastokens==0.3.1.1 ; extra == 'fastokens'",
                b"Requires-Dist: toktier-fastokens ; extra == 'fastokens'",
            ),
            "the fastokens extra does not require",
        ),
        (
            (b"Requires-Dist: tokenizers==0.22.2\n", b""),
            "does not pin tokenizers==0.22.2",
        ),
        (
            (b"Provides-Extra: gpu-jit\n", b""),
            "does not expose both GPU delivery profiles",
        ),
    ],
)
def test_the_gate_refuses_a_real_wheels_metadata_once_edited(
    edit: tuple[bytes, bytes], complaint: str
) -> None:
    """Each refusal is exercised against the same real text."""
    verifier = _artifact_verifier()
    before, after = edit
    assert before in REAL_WHEEL_METADATA
    damaged = REAL_WHEEL_METADATA.replace(before, after)

    with pytest.raises(ValueError, match=complaint):
        verifier.verify_metadata(damaged)


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


def _recorded_refresh(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> list[list[str]]:
    """Run the refresh tool with every subprocess replaced by a record."""
    refresh = _tools_module("refresh_dependency_judgement")
    ran: list[list[str]] = []

    def record(command: list[str]) -> int:
        ran.append(command)
        return 0

    monkeypatch.setattr(refresh, "run", record)
    monkeypatch.setattr(refresh, "legal_digest_problems", lambda *, rewrite: [])
    assert refresh.main(argv) == 0
    return ran


def test_the_refresh_updates_the_lock_offline_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The write mode's first step is a lockfile edit, and its scope matters.

    An unrestricted ``cargo update`` reaches the network and moves every
    transitive third-party version that has published since. A 0.2.7
    release wave met that: twelve unrelated packages lifted, a seven-line
    lock diff turned into thirty-seven, and the version-normalised source
    identities moved with them. The narrow operation this tool is for is
    the workspace's own members against what is already cached.
    """
    ran = _recorded_refresh(monkeypatch, [])

    assert ran[0] == ["cargo", "update", "--workspace", "--offline"]
    assert ran[1] == ["cargo", "fetch", "--locked"]


def test_the_refresh_reaches_the_network_only_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wide update is still available, by name."""
    ran = _recorded_refresh(monkeypatch, ["--allow-network-update"])

    assert ran[0] == ["cargo", "update"]


def test_the_refresh_checks_without_touching_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = _recorded_refresh(monkeypatch, ["--check"])

    assert not any(command[:2] == ["cargo", "update"] for command in ran)
    assert all("--check" in command for command in ran)


def test_the_refresh_refuses_a_flag_pair_that_means_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--check`` writes nothing, so widening its writes is not a request."""
    refresh = _tools_module("refresh_dependency_judgement")

    with pytest.raises(SystemExit):
        refresh.main(["--check", "--allow-network-update"])


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
    # `pyproject.toml` travels in the archive and its `readme` key names
    # this file, so an archive without it points at something it does
    # not carry, and a reader cannot check the PyPI front page there.
    assert "README.pypi.md" in declared
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
