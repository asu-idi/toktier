"""The artifact manifest that ships inside the package.

``toktier artifacts fetch`` and ``toktier artifacts verify`` resolve a
family against this file, so it is part of the product rather than a
development convenience: if it cannot pin a family, both commands are
unusable wherever the package is installed.

The generator owns the rules (which families belong in it, what the
deterministic form is, which digests it has to agree with); these tests
check the shipped file through the library that reads it and through
that generator's own ``--check``, rather than restating the rules a
third time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from toktier.artifacts import ArtifactManifest
from toktier.artifacts.tables import ARTIFACT_MANIFEST

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tables" / "support_registry.json"

#: The one file the reference backend opens; the GPU tables are exported
#: from the same bytes.
RUNTIME_FILE = "tokenizer.json"


def _generator() -> ModuleType:
    """Import the generator the way its own tooling does."""
    tools = str(ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import generate_artifact_manifest

    return generate_artifact_manifest


def test_the_shipped_manifest_pins_every_family_it_offers() -> None:
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)

    assert manifest.families(), "the package ships no artifact identities"
    for family in manifest.families():
        entry = manifest.get(family)
        assert entry.repo_id and entry.revision, family
        names = [item.name for item in entry.files]
        assert RUNTIME_FILE in names, family
        for item in entry.files:
            # The parser already rejects a missing or malformed digest;
            # the recorded length is this manifest's own requirement, so
            # a truncated transfer fails before anything is hashed.
            assert item.size is not None, f"{family}/{item.name}"
            assert item.size > 0, f"{family}/{item.name}"


@pytest.mark.skipif(
    not REGISTRY.is_file(),
    reason="cross-registry verification activates with certification evidence",
)
def test_the_shipped_manifest_passes_its_generator_check() -> None:
    """Deterministic form, library parse, agreement with the registry."""
    assert _generator().main(["--check", "--out", str(ARTIFACT_MANIFEST)]) == 0


@pytest.mark.skipif(
    not REGISTRY.is_file(),
    reason="cross-registry verification activates with certification evidence",
)
def test_the_generator_check_rejects_an_edited_digest(tmp_path: Path) -> None:
    """The green check above has teeth: a changed digest fails it."""
    document = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    family = sorted(document)[0]
    document[family]["files"][RUNTIME_FILE]["sha256"] = "0" * 64
    edited = tmp_path / "artifact_manifest.v1.json"
    edited.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    assert _generator().main(["--check", "--out", str(edited)]) == 1


def test_the_generator_check_reports_a_missing_manifest(tmp_path: Path) -> None:
    assert _generator().main(["--check", "--out", str(tmp_path / "absent.json")]) == 1


@pytest.mark.parametrize("command", ["fetch", "verify"])
def test_every_shipped_family_is_reachable_from_the_command_line(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No shipped family is refused on its name by either command.

    The cache is empty, so both commands fail; what this checks is
    *how*. ``HF_HUB_OFFLINE`` keeps ``fetch`` from reaching out: the hub
    source refuses before any client is imported.
    """
    from toktier import cli

    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    families = cli._artifact_manifest().families()

    for family in families:
        assert cli.main(["artifacts", command, family]) == 2
        message = capsys.readouterr().err
        assert "unknown tokenizer family" not in message
        assert f"of {family!r} is not in the cache" in message
