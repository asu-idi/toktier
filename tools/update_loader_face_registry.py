#!/usr/bin/env python3
"""Record the loader-face digest of every certified artifact in the registry.

Since 0.2.8 the certification subject is the loader face: the tokenizer
JSON document the pinned loader materializes from the verified artifact
directory (``docs/contracts/facade.md`` Section 5). For most artifacts
that document is the artifact file itself; for an artifact whose
``tokenizer_config.json`` declares added tokens beyond the file, it is
the pinned ``transformers`` release's serialization of the live object.
A different loader release can materialize a different face from the
same verified bytes -- rebuilding the pre-tokenizer from its own class
defaults, say -- and the artifact identity alone does not see that.

This tool writes the digest of that document (``loader_face_sha256``,
``docs/contracts/registry.md`` Section 1) on every artifact record of
``tables/support_registry.json`` and its packaged copy, computed here
from the artifact directories cached on this machine under the pinned
loader. At load the facade recomputes the digest of the face it
materialized and compares it with the record; a difference closes the
accelerated routes and is reported as ``R_LOADER_FACE_MISMATCH``
(``docs/contracts/routing.md`` Section 5.1). The chunk-parallel
admission table carries the same digest per row, written by its own
campaign tool, and compares it the same way.

The digest is taken over the UTF-8 encoding of the exact text the
reference backend executes, by the one function both readers use
(``toktier.backends.loader_face.loader_face_sha256``), and the text is
produced by the same rule the reference backend applies: the artifact
file when the configuration-side added-token subset is empty, the live
object's serialization otherwise. Writing requires the pinned
``transformers`` to be the installed one, because the recorded face is
the pinned loader's face; a different installed loader is refused
rather than recorded.

``--check`` recomputes the digests from the cached artifacts and
compares them with the shipped table. Artifacts not cached on this
machine are listed and skipped (exit code 3 when any was skipped and
nothing mismatched), so the check says what it could not conclude
rather than passing over it. ``tools/check_loader_face_alignment.py
--fingerprints`` performs the same comparison as part of the standing
loader-face gate. After writing, ``tools/sync_rust_package_data.py``
carries the packaged copy into the Rust crate's data payload.

Usage::

    python tools/update_loader_face_registry.py
    python tools/update_loader_face_registry.py --check
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from registry_common import (
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    load_json,
    schema_violations,
    serialise_document,
    with_root_digest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "tables" / "support_registry.json"
PACKAGED_REGISTRY = (
    REPOSITORY_ROOT
    / "src"
    / "toktier"
    / "routing"
    / "tables"
    / "support_registry.v1.json"
)
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "support_registry.schema.json"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"

#: The artifact-record member this tool owns.
FIELD = "loader_face_sha256"
#: Exit code of ``--check`` when a registry artifact is not cached here
#: and nothing that was cached mismatched: not a pass, not a failure.
ABSENT = 3

_LOADER_PIN = re.compile(r'"transformers==(?P<version>[^"]+)"')


def pinned_loader_version(pyproject: Path = PYPROJECT_PATH) -> str:
    """The ``transformers`` release the package metadata pins."""
    match = _LOADER_PIN.search(pyproject.read_text(encoding="utf-8"))
    if match is None:
        raise GenerationError(f"{pyproject} does not pin transformers")
    return match["version"]


def installed_loader_version() -> str | None:
    """The installed ``transformers`` release, or ``None`` when absent."""
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover - stdlib on every supported Python
        return None
    try:
        return version("transformers")
    except Exception:
        return None


def loader_face_text(root: Path) -> str:
    """The tokenizer JSON text the reference backend executes for ``root``.

    This is the rule of ``toktier.backends.hf.HfBackend.open``: when the
    configuration sidecar declares no added token beyond the artifact
    file, the verified document is executed as written, so the face is
    the file's text; otherwise the live loader object is materialized
    once and its serialization is the face. The two branches are kept
    apart here for the same reason they are kept apart there: the
    common case never imports the loader, and the file-only face is the
    file, not a re-serialization of it.
    """
    from toktier.backends.loader_face import (
        config_added_token_rows,
        live_tokenizer_json,
        load_live_tokenizer,
    )
    from toktier.backends.protocol import TOKENIZER_FILE

    text = (root / TOKENIZER_FILE).read_text(encoding="utf-8")
    document = json.loads(text)
    rows = config_added_token_rows(root, document if isinstance(document, dict) else {})
    if not rows:
        return text
    return live_tokenizer_json(load_live_tokenizer(root))


def observed_loader_faces(
    registry: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """Digest of the face of every registry artifact cached on this machine.

    Returns the digests by family and the families whose artifact is not
    cached. A cached artifact whose file digest is not the one the
    registry records is refused: the face would belong to other bytes.
    """
    from toktier import Config
    from toktier.artifacts import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.backends.loader_face import loader_face_sha256
    from toktier.backends.protocol import TOKENIZER_FILE
    from toktier.paths import artifact_cache_dir

    config = Config(offline=True)
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    cache = artifact_cache_dir(config)
    observed: dict[str, str] = {}
    absent: list[str] = []
    for row in registry.get("artifacts", []):
        if not isinstance(row, dict):
            raise GenerationError("support registry has a non-object artifact row")
        family = str(row.get("family"))
        entry = manifest.entries.get(family)
        if entry is None:
            absent.append(family)
            continue
        root = cache / entry.directory_name
        path = root / TOKENIZER_FILE
        if not path.is_file():
            absent.append(family)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row.get("artifact_sha256"):
            raise GenerationError(
                f"{family}: the cached {path} has sha256 {digest}, the registry "
                f"records {row.get('artifact_sha256')}"
            )
        observed[family] = loader_face_sha256(loader_face_text(root))
    return observed, absent


def augmented_document(
    registry: dict[str, Any], observed: dict[str, str]
) -> dict[str, Any]:
    """Return ``registry`` with the observed digests written on its rows."""
    completed = copy.deepcopy(registry)
    for row in completed.get("artifacts", []):
        family = str(row.get("family"))
        if family in observed:
            row[FIELD] = observed[family]
    return with_root_digest(completed, REGISTRY_DOMAIN_TAG)


def check_problems(
    registry: dict[str, Any], observed: dict[str, str]
) -> list[str]:
    """Rows whose recorded digest is missing or differs from the observed one."""
    problems: list[str] = []
    for row in registry.get("artifacts", []):
        family = str(row.get("family"))
        if family not in observed:
            continue
        recorded = row.get(FIELD)
        if recorded is None:
            problems.append(f"{family}: no {FIELD} is recorded")
        elif recorded != observed[family]:
            problems.append(
                f"{family}: the registry records {FIELD} {recorded}, the pinned "
                f"loader materializes {observed[family]}"
            )
    return problems


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialise_document(document))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record or check the loader-face digest of every certified artifact."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute the digests from the cached artifacts and compare",
    )
    arguments = parser.parse_args(argv)

    pinned = pinned_loader_version()
    installed = installed_loader_version()
    if installed != pinned:
        print(
            f"error: the recorded face is the pinned loader's "
            f"(transformers=={pinned}); installed: {installed}",
            file=sys.stderr,
        )
        return 2
    registry = load_json(DEFAULT_REGISTRY)
    if not isinstance(registry, dict):
        raise GenerationError("support registry is not a JSON object")
    observed, absent = observed_loader_faces(registry)

    if arguments.check:
        problems = check_problems(registry, observed)
        reading = {
            "checked": sorted(observed),
            "absent": absent,
            "problems": problems,
        }
        print(json.dumps(reading, ensure_ascii=True))
        if problems:
            return 1
        if absent or not observed:
            return ABSENT
        payload = serialise_document(registry)
        for path in (DEFAULT_REGISTRY, PACKAGED_REGISTRY):
            if not path.is_file() or path.read_bytes() != payload:
                print(f"error: {path} is not the serialised registry", file=sys.stderr)
                return 1
        return 0

    if absent:
        raise GenerationError(
            "cannot record the loader face of artifacts that are not cached "
            "here: " + ", ".join(absent)
        )
    generated = augmented_document(registry, observed)
    violations = schema_violations(generated, load_json(SCHEMA_PATH))
    if violations:
        raise GenerationError(
            "augmented registry violates its schema:\n  " + "\n  ".join(violations)
        )
    _write(DEFAULT_REGISTRY, generated)
    _write(PACKAGED_REGISTRY, generated)
    print(
        f"updated {DEFAULT_REGISTRY} and {PACKAGED_REGISTRY} "
        f"({len(observed)} artifact records, transformers=={installed})"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GenerationError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
