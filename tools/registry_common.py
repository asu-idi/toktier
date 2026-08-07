# Shared helpers for the generated registry and evidence documents.
"""Canonical JSON, root digests, schema validation and check-mode plumbing.

Every generated table in this repository is produced by tooling only
(``docs/contracts/registry.md`` Section 7). This module holds the parts that
both generators need, so that the digest rule exists exactly once:

* ``canonical_json`` delegates to the repository's complete RFC 8785
  implementation and translates its failures into ``GenerationError``.
* ``root_digest`` implements the frozen construction of
  ``docs/contracts/registry.md`` Section 6.
* ``serialise_document`` is the on-disk form: pretty printed, ASCII only, one
  trailing newline, deterministic for a given document.
* ``verify_file`` and ``check_regenerated`` implement ``--check``.

Everything here reads only this repository; the maintainer generation
tooling builds on these helpers so that the digest rule cannot drift.
"""

import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from toktier._jcs import (  # noqa: E402
    CanonicalizationError as _JCSCanonicalizationError,
)
from toktier._jcs import canonical_json as _canonical_json  # noqa: E402

REGISTRY_DOMAIN_TAG = b"toktier.registry.v1\x00"
EVIDENCE_DOMAIN_TAG = b"toktier.evidence.v1\x00"
PIPELINE_DOMAIN_TAG = b"toktier.pipeline.v1\x00"
ADDED_FRONTEND_DOMAIN_TAG = b"toktier.added_frontend.v1\x00"

#: Digest stand-in for a binding that another lane owns. It is a syntactically
#: valid sha256 that cannot be a real digest of anything anyone shipped, so a
#: loader comparing against it always closes the accelerated path. Generators
#: that emit it must say so on stderr.
PLACEHOLDER_SHA256 = "0" * 64

class GenerationError(Exception):
    """A generator could not produce a document it can stand behind."""


def canonical_json(value: object) -> bytes:
    """Return the RFC 8785 canonical form of ``value`` as UTF-8 bytes."""
    try:
        return _canonical_json(value)
    except _JCSCanonicalizationError as error:
        raise GenerationError(str(error)) from error


def root_digest(document: Mapping[str, Any], domain_tag: bytes) -> str:
    """Return the ``sha256:`` root digest of ``document``.

    The ``root_digest`` member is removed (not blanked) before hashing, as
    required by ``docs/contracts/registry.md`` Section 6.
    """
    without_digest = {
        key: value for key, value in document.items() if key != "root_digest"
    }
    digest = hashlib.sha256(domain_tag + canonical_json(without_digest)).hexdigest()
    return f"sha256:{digest}"


def with_root_digest(
    document: Mapping[str, Any], domain_tag: bytes
) -> dict[str, Any]:
    """Return a copy of ``document`` whose ``root_digest`` member is correct."""
    completed = dict(document)
    completed["root_digest"] = root_digest(document, domain_tag)
    return completed


def serialise_document(document: Mapping[str, Any]) -> bytes:
    """Return the deterministic on-disk form of a generated document."""
    text = json.dumps(document, indent=2, ensure_ascii=True, sort_keys=False)
    return (text + "\n").encode("ascii")


def normalise_root(value: object) -> Path | None:
    """Return a filesystem path, treating an empty value as "not given"."""
    if value is None:
        return None
    text = str(value)
    return Path(text) if text else None


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_of_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    """Read a JSON document, with the path named in any parse failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise GenerationError(f"cannot read {path}: {error}") from error
    except ValueError as error:
        raise GenerationError(f"cannot parse {path}: {error}") from error


def schema_violations(
    document: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[str]:
    """Return schema violation messages for ``document`` (empty when valid).

    ``jsonschema`` is imported through :func:`importlib.import_module` so that
    the rest of this repository, which does not depend on it, still type checks
    and imports without it installed.
    """
    try:
        jsonschema = importlib.import_module("jsonschema")
    except ModuleNotFoundError as error:  # pragma: no cover - environment issue
        raise GenerationError(
            "the jsonschema package is required to validate generated "
            "documents; install it with 'pip install jsonschema'"
        ) from error
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    messages = []
    violations = sorted(validator.iter_errors(document), key=lambda item: item.path)
    for violation in violations:
        location = "/".join(str(part) for part in violation.absolute_path) or "<root>"
        messages.append(f"{location}: {violation.message}")
    return messages


def verify_file(
    path: Path, schema: Mapping[str, Any], domain_tag: bytes
) -> list[str]:
    """Check a generated file on its own terms.

    This is the part of ``--check`` that needs nothing but the file itself: the
    serialisation is the deterministic one, the document validates against the
    schema, and the recorded root digest matches the document. It therefore
    still catches a hand edit when the sources that produced the file are not
    reachable (for example in CI).
    """
    problems = []
    if not path.exists():
        return [f"{path}: file is missing"]
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        return [f"{path}: cannot parse as JSON: {error}"]
    if not isinstance(document, dict):
        return [f"{path}: top level value is not an object"]
    if serialise_document(document) != raw:
        problems.append(
            f"{path}: file is not in the deterministic serialised form "
            "(regenerate with the tool rather than editing by hand)"
        )
    problems.extend(
        f"{path}: {message}" for message in schema_violations(document, schema)
    )
    recorded = document.get("root_digest")
    expected = root_digest(document, domain_tag)
    if recorded != expected:
        problems.append(
            f"{path}: root digest mismatch "
            f"(recorded {recorded!r}, computed {expected!r})"
        )
    return problems


def check_regenerated(
    path: Path,
    regenerated: Mapping[str, Any],
    schema: Mapping[str, Any],
    domain_tag: bytes,
) -> list[str]:
    """Compare a freshly generated document with the file on disk.

    ``generated_by`` is taken from the existing file for the comparison: the
    tool version, source commit and timestamp describe *when* the file was
    written, not what it asserts, and a moved HEAD must not be reported as a
    hand edit. Everything else has to match byte for byte.
    """
    problems = verify_file(path, schema, domain_tag)
    if not path.exists():
        return problems
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        return problems
    replayed = dict(regenerated)
    if "generated_by" in existing:
        replayed["generated_by"] = existing["generated_by"]
    replayed = with_root_digest(replayed, domain_tag)
    if canonical_json(replayed) != canonical_json(existing):
        problems.append(
            f"{path}: regenerated content differs from the file on disk; "
            "run the generator without --check to refresh it"
        )
    return problems


def write_document(
    path: Path,
    document: Mapping[str, Any],
    schema: Mapping[str, Any],
    domain_tag: bytes,
) -> dict[str, Any]:
    """Validate, digest and write a generated document. Returns the document."""
    completed = with_root_digest(document, domain_tag)
    violations = schema_violations(completed, schema)
    if violations:
        raise GenerationError(
            "generated document does not satisfy its schema:\n  "
            + "\n  ".join(violations)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialise_document(completed))
    return completed


def git_commit(repository: Path, paths: Iterable[str] = ()) -> str | None:
    """Return the commit of ``repository`` (optionally: touching ``paths``).

    Read-only. Returns ``None`` when the repository or the commit cannot be
    determined, so callers can fall back to an explicit command line value
    rather than inventing one.
    """
    command = ["git", "-C", str(repository), "log", "-1", "--format=%H"]
    selected = list(paths)
    if selected:
        command.append("--")
        command.extend(selected)
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return None
    return commit
