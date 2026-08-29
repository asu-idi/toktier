"""Export, import, and source access for air-gapped artifact bundles.

A bundle is a tar archive with one ``bundle_manifest.json`` member and
the artifact files named by that manifest.  The manifest has this shape::

    {
      "alias": "family-revision",
      "files": [
        {"path": "tokenizer.json", "sha256": "...", "size": 123}
      ],
      "root_digest": "sha256:..."
    }

The root digest follows the construction frozen for registry documents:
remove ``root_digest``, RFC-8785-canonicalize the remaining object, prepend
the bundle domain tag, and hash the result with sha256.

Imports first validate and extract into a private sibling directory.  Only
after every member and digest has passed is that directory fsynced and
renamed into the artifact cache, so a rejected bundle leaves no partial
artifact behind.

Re-importing the same bundle into a cache that already holds its alias is
idempotent, on the one condition the Rust face has always stated: the
visible tree still authenticates as exactly these contents.  The installed
tree is re-read against the manifest -- every declared path, byte count and
SHA-256, and no undeclared file -- and only then is the freshly staged copy
discarded and the existing directory returned untouched.  A tree that does
not authenticate is a conflict, not a success, and is reported as one.

Error mapping (``docs/contracts/errors.md``, decision 0004): violations
of the bundle archive format -- the tar container and the embedded
bundle manifest -- raise ``BundleInvalid`` (``BUNDLE_INVALID``);
content-hash failures of the artifact files raise
``ArtifactHashMismatch``; a missing bundle file or requested member
raises ``ArtifactNotFound``; and an alias the cache already holds with
other contents raises ``AliasConflict`` (``ALIAS_CONFLICT``, added in
0.2.8), which is about the installed tree rather than about a bundle
that could not be found.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

from .._jcs import canonical_json
from ..errors import (
    AliasConflict,
    ArtifactHashMismatch,
    ArtifactNotFound,
    BundleInvalid,
    one_line,
)
from ..paths import FILE_MODE, ensure_private_dir
from .manifest import ArtifactEntry, ArtifactFile

__all__ = ["AirgapBundleSource", "export_bundle", "import_bundle"]

BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
BUNDLE_ROOT_DOMAIN = b"toktier.bundle.v1\0"
MAX_BUNDLE_MEMBERS = 4096
MAX_BUNDLE_UNCOMPRESSED_SIZE = 8 * 1024 * 1024 * 1024

_READ_CHUNK = 1024 * 1024
_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
_ROOT_DIGEST_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_MANIFEST_KEYS = frozenset({"alias", "files", "root_digest"})
_FILE_KEYS = frozenset({"path", "sha256", "size"})


@dataclass(frozen=True)
class _BundleFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class _BundleManifest:
    alias: str
    files: tuple[_BundleFile, ...]


@dataclass(frozen=True)
class _ExportFile:
    bundle_file: _BundleFile
    source: Path


class AirgapBundleSource:
    """Use a validated local bundle through the ``ArtifactSource`` protocol."""

    name = "airgap_bundle"

    def __init__(
        self,
        bundle: str | os.PathLike[str],
        *,
        offline: bool = False,
    ) -> None:
        self._bundle = Path(bundle)
        self._offline = offline

    @property
    def offline(self) -> bool:
        return self._offline

    def fetch(
        self,
        entry: ArtifactEntry,
        artifact_file: ArtifactFile,
        destination: Path,
    ) -> None:
        if self._offline:
            raise ArtifactNotFound(
                f"air-gap bundle source for {entry.family!r} is offline",
                details={
                    "family": entry.family,
                    "searched": [str(self._bundle)],
                    "offline": True,
                },
            )

        staging = Path(
            tempfile.mkdtemp(
                prefix=".toktier-airgap-source-", dir=str(destination.parent)
            )
        )
        try:
            manifest = _extract_verified_bundle(self._bundle, staging)
            relative = _normalize_path(
                artifact_file.name,
                bundle=self._bundle,
                what="artifact file path",
            )
            source = staging / relative
            if not source.is_file():
                raise ArtifactNotFound(
                    f"bundle has no file named {artifact_file.name!r}",
                    details={
                        "family": entry.family,
                        "searched": [item.path for item in manifest.files],
                        "offline": False,
                    },
                )
            shutil.copyfile(source, destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def export_bundle(
    bundle: str | os.PathLike[str],
    alias: str,
    files: Mapping[str, str | os.PathLike[str]],
) -> Path:
    """Atomically write a verified air-gap bundle.

    ``files`` maps bundle-relative POSIX paths to files in a verified
    artifact directory.  ``alias`` is the cache-relative directory name
    that :func:`import_bundle` installs.
    """
    bundle_path = Path(bundle)
    checked_alias = _normalize_alias(alias, bundle=bundle_path)
    export_files = _prepare_export_files(files, bundle=bundle_path)
    manifest_without_root: dict[str, object] = {
        "alias": checked_alias,
        "files": [
            {
                "path": item.bundle_file.path,
                "sha256": item.bundle_file.sha256,
                "size": item.bundle_file.size,
            }
            for item in export_files
        ],
    }
    manifest = dict(manifest_without_root)
    manifest["root_digest"] = _root_digest(manifest_without_root)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    ensure_private_dir(bundle_path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{bundle_path.name}.",
        suffix=".tmp",
        dir=str(bundle_path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    validation = Path(
        tempfile.mkdtemp(prefix=".toktier-bundle-export-", dir=str(bundle_path.parent))
    )
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            _add_bytes(archive, BUNDLE_MANIFEST_NAME, manifest_bytes)
            for item in export_files:
                _add_file(archive, item.bundle_file, item.source)

        # Validate the exact archive bytes before they become visible.  This
        # also catches a source file that changed while the tar was written.
        _extract_verified_bundle(temporary, validation)
        _install_file(temporary, bundle_path)
    finally:
        _remove_file(temporary)
        shutil.rmtree(validation, ignore_errors=True)
    return bundle_path


def import_bundle(
    bundle: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
) -> Path:
    """Verify ``bundle`` and atomically install its alias into ``cache_root``.

    A second import of the same bundle into the same cache is idempotent
    when the installed tree still authenticates as exactly these
    contents; the already installed directory is returned and its bytes
    are not touched.  A tree that holds the alias but does not
    authenticate is reported instead of overwritten.
    """
    bundle_path = Path(bundle)
    root = Path(cache_root)
    root_existed = root.is_dir()
    ensure_private_dir(root)
    staging = Path(
        tempfile.mkdtemp(prefix=".toktier-bundle-import-", dir=str(root))
    )
    staging_removed = False
    try:
        manifest = _extract_verified_bundle(bundle_path, staging)
        target = root / manifest.alias
        ensure_private_dir(target.parent)
        _sync_tree(staging)
        if target.exists() or target.is_symlink():
            # Re-import is idempotent exactly when the visible tree still
            # authenticates as this bundle, which is the condition the
            # Rust face states and now the condition both faces apply.
            _authenticate_installed_alias(target, manifest)
            shutil.rmtree(staging, ignore_errors=True)
            staging_removed = True
            return target
        try:
            os.replace(staging, target)
        except OSError as exc:
            raise ArtifactNotFound(
                f"cannot install bundle alias {manifest.alias!r} "
                f"into the cache",
                details={
                    "family": manifest.alias,
                    "searched": [str(target)],
                    "cause": "install_failed",
                    "cause_message": str(exc),
                },
            ) from exc
        staging_removed = True
        _sync_directory(target.parent)
        return target
    finally:
        if not staging_removed:
            shutil.rmtree(staging, ignore_errors=True)
            if not root_existed:
                with suppress(OSError):
                    root.rmdir()


def _authenticate_installed_alias(
    target: Path, manifest: _BundleManifest
) -> None:
    """Re-read an installed alias against the manifest that claims it.

    The check mirrors the Rust ``verify_installed``: nothing in the tree
    is a symbolic or special file, every file in it is declared, every
    declared file is present, and each one still has its declared byte
    count and SHA-256.  The first path that does not authenticate is
    named, in sorted order, so a reader is told which file to look at
    rather than only that something differs.
    """
    if target.is_symlink() or not target.is_dir():
        raise _alias_conflict(
            target, manifest, path=target, failure="not_a_directory"
        )
    declared = {item.path: item for item in manifest.files}
    observed: set[str] = set()
    unexpected: list[tuple[str, str, Path]] = []
    pending = [target]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as scan:
            children = list(scan)
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(target).as_posix()
            if child.is_symlink():
                unexpected.append((relative, "symlink", path))
            elif child.is_dir(follow_symlinks=False):
                pending.append(path)
            elif not child.is_file(follow_symlinks=False):
                unexpected.append((relative, "special_file", path))
            elif relative in declared:
                observed.add(relative)
            else:
                unexpected.append((relative, "undeclared_file", path))
    if unexpected:
        relative, failure, path = min(unexpected)
        raise _alias_conflict(target, manifest, path=path, failure=failure)
    for item in sorted(manifest.files, key=lambda file: file.path):
        path = target / PurePosixPath(item.path)
        if item.path not in observed:
            raise _alias_conflict(
                target, manifest, path=path, failure="missing_file"
            )
        observed_sha256, observed_size = _sha256_file(path)
        if observed_size != item.size:
            raise _alias_conflict(
                target,
                manifest,
                path=path,
                failure="size_mismatch",
                extra={
                    "expected_size": item.size,
                    "observed_size": observed_size,
                },
            )
        if observed_sha256 != item.sha256:
            raise _alias_conflict(
                target,
                manifest,
                path=path,
                failure="hash_mismatch",
                extra={
                    "expected_sha256": item.sha256,
                    "observed_sha256": observed_sha256,
                },
            )


def _alias_conflict(
    target: Path,
    manifest: _BundleManifest,
    *,
    path: Path,
    failure: str,
    extra: Mapping[str, object] | None = None,
) -> AliasConflict:
    """Build the error for an alias the cache holds with other contents."""
    details: dict[str, object] = {
        "family": manifest.alias,
        "searched": [str(target)],
        "cause": "alias_conflict",
        "failure": failure,
        "path": str(path),
        "remedy": (
            f"remove {target} and import again, or import into another "
            f"cache root; the bundle itself verified"
        ),
    }
    details.update(extra or {})
    return AliasConflict(
        f"the cache holds the bundle alias {manifest.alias!r} with other "
        f"contents: {failure} at {path}",
        details=details,
    )


def _prepare_export_files(
    files: Mapping[str, str | os.PathLike[str]], *, bundle: Path
) -> tuple[_ExportFile, ...]:
    if not isinstance(files, Mapping) or not files:
        raise _invalid(bundle, "files must be a non-empty mapping")
    prepared: list[_ExportFile] = []
    seen: set[str] = set()
    for raw_path, raw_source in files.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise _invalid(bundle, "file paths must be non-empty strings")
        path = _normalize_path(raw_path, bundle=bundle, what="file path")
        if path == ".":
            raise _invalid(bundle, "file path must name a file")
        if path == BUNDLE_MANIFEST_NAME:
            raise _invalid(bundle, f"file path {path!r} is reserved", member=path)
        if path in seen:
            raise _invalid(bundle, f"duplicate path {path!r}", member=path)
        seen.add(path)
        source = Path(raw_source)
        if not source.is_file():
            raise ArtifactNotFound(
                f"artifact file missing while exporting bundle: {source}",
                details={
                    "family": None,
                    "searched": [str(source)],
                    "offline": True,
                },
            )
        observed_sha256, observed_size = _sha256_file(source)
        prepared.append(
            _ExportFile(
                bundle_file=_BundleFile(
                    path=path,
                    sha256=observed_sha256,
                    size=observed_size,
                ),
                source=source,
            )
        )
    return tuple(sorted(prepared, key=lambda item: item.bundle_file.path))


def _extract_verified_bundle(bundle: Path, destination: Path) -> _BundleManifest:
    if not bundle.is_file():
        raise ArtifactNotFound(
            f"air-gap bundle not found: {bundle}",
            details={"family": None, "searched": [str(bundle)], "offline": True},
        )
    try:
        with tarfile.open(bundle, mode="r:*") as archive:
            members = _scan_members(archive, bundle=bundle)
            manifest = _read_manifest(archive, members, bundle=bundle)
            _check_member_set(members, manifest, bundle=bundle)
            for bundle_file in manifest.files:
                member = members[bundle_file.path]
                _extract_and_verify(
                    archive,
                    member,
                    bundle_file,
                    destination=destination,
                    bundle=bundle,
                )
            return manifest
    except (ArtifactHashMismatch, ArtifactNotFound, BundleInvalid):
        raise
    except (tarfile.TarError, OSError) as exc:
        raise _invalid(
            bundle, "cannot read tar archive", cause=str(exc)
        ) from exc


def _scan_members(
    archive: tarfile.TarFile, *, bundle: Path
) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    total_size = 0
    for count, member in enumerate(archive, start=1):
        if count > MAX_BUNDLE_MEMBERS:
            raise _invalid(
                bundle,
                f"archive has more than {MAX_BUNDLE_MEMBERS} members",
            )
        total_size += member.size
        if total_size > MAX_BUNDLE_UNCOMPRESSED_SIZE:
            raise _invalid(
                bundle,
                "archive exceeds the 8 GiB uncompressed-size limit",
            )
        path = _normalize_path(member.name, bundle=bundle, what="archive member")
        if path in members:
            raise _invalid(bundle, f"duplicate path {path!r}", member=path)
        if member.issym() or member.islnk():
            kind = "symbolic" if member.issym() else "hard"
            raise _invalid(
                bundle,
                f"{kind} link member {path!r} is not allowed",
                member=path,
            )
        members[path] = member
    return members


def _read_manifest(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    *,
    bundle: Path,
) -> _BundleManifest:
    member = members.get(BUNDLE_MANIFEST_NAME)
    if member is None or not member.isfile():
        raise _invalid(
            bundle,
            f"{BUNDLE_MANIFEST_NAME} is missing",
            member=BUNDLE_MANIFEST_NAME,
        )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise _invalid(
            bundle,
            f"cannot read {BUNDLE_MANIFEST_NAME}",
            member=BUNDLE_MANIFEST_NAME,
        )
    try:
        raw_bytes = extracted.read()
    finally:
        extracted.close()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid(
            bundle,
            f"cannot parse {BUNDLE_MANIFEST_NAME}",
            member=BUNDLE_MANIFEST_NAME,
            cause=str(exc),
        ) from exc
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
        raise _invalid(
            bundle,
            "bundle manifest must contain exactly alias, files, and root_digest",
            member=BUNDLE_MANIFEST_NAME,
        )

    root_digest = raw.get("root_digest")
    if not isinstance(root_digest, str) or not _ROOT_DIGEST_PATTERN.match(root_digest):
        raise _invalid(
            bundle,
            "root_digest must be sha256 followed by 64 hex digits",
            member=BUNDLE_MANIFEST_NAME,
        )
    digest_input = dict(raw)
    del digest_input["root_digest"]
    try:
        expected_root = _root_digest(digest_input)
    except ValueError as exc:
        raise _invalid(
            bundle,
            "cannot canonicalize bundle manifest",
            member=BUNDLE_MANIFEST_NAME,
            cause=str(exc),
        ) from exc
    if root_digest != expected_root:
        raise _invalid(
            bundle,
            f"root digest mismatch: expected {expected_root}, found {root_digest}",
            member=BUNDLE_MANIFEST_NAME,
        )

    alias = raw.get("alias")
    if not isinstance(alias, str) or not alias:
        raise _invalid(bundle, "alias must be a non-empty string")
    checked_alias = _normalize_alias(alias, bundle=bundle)
    raw_files = raw.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise _invalid(bundle, "files must be a non-empty list")
    files: list[_BundleFile] = []
    seen: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, dict) or set(raw_file) != _FILE_KEYS:
            raise _invalid(
                bundle,
                f"files[{index}] must contain exactly path, sha256, and size",
            )
        raw_path = raw_file.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise _invalid(bundle, f"files[{index}].path must be a non-empty string")
        path = _normalize_path(raw_path, bundle=bundle, what="manifest file path")
        if path == ".":
            raise _invalid(bundle, f"files[{index}].path must name a file")
        if path in seen:
            raise _invalid(bundle, f"duplicate path {path!r}", member=path)
        seen.add(path)
        sha256 = raw_file.get("sha256")
        if not isinstance(sha256, str) or not _SHA256_PATTERN.match(sha256):
            raise _invalid(
                bundle,
                f"files[{index}].sha256 must be 64 lower-case hex digits",
            )
        size = raw_file.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise _invalid(
                bundle,
                f"files[{index}].size must be a non-negative integer",
            )
        files.append(_BundleFile(path=path, sha256=sha256, size=size))
    return _BundleManifest(alias=checked_alias, files=tuple(files))


def _check_member_set(
    members: Mapping[str, tarfile.TarInfo],
    manifest: _BundleManifest,
    *,
    bundle: Path,
) -> None:
    archive_files = {
        path
        for path, member in members.items()
        if member.isfile() and path != BUNDLE_MANIFEST_NAME
    }
    declared_files = {item.path for item in manifest.files}
    missing = sorted(declared_files - archive_files)
    extra = sorted(archive_files - declared_files)
    if missing:
        raise _invalid(bundle, f"manifest files are missing from archive: {missing}")
    if extra:
        raise _invalid(bundle, f"archive files are missing from manifest: {extra}")


def _extract_and_verify(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    bundle_file: _BundleFile,
    *,
    destination: Path,
    bundle: Path,
) -> None:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise _invalid(
            bundle,
            f"cannot read artifact file {bundle_file.path!r}",
            member=bundle_file.path,
        )
    target = destination / bundle_file.path
    ensure_private_dir(target.parent)
    digest = hashlib.sha256()
    size = 0
    try:
        with open(target, "wb") as output:
            while True:
                chunk = extracted.read(_READ_CHUNK)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
    finally:
        extracted.close()
    os.chmod(target, FILE_MODE)
    observed_sha256 = digest.hexdigest()
    if size != bundle_file.size or observed_sha256 != bundle_file.sha256:
        raise ArtifactHashMismatch(
            f"content hash mismatch for bundle file {bundle_file.path}",
            details={
                "expected_sha256": bundle_file.sha256,
                "observed_sha256": observed_sha256,
                "expected_size": bundle_file.size,
                "observed_size": size,
                "path": f"{bundle}!{bundle_file.path}",
                "remedy": "re-export the air-gap bundle from a verified cache",
            },
        )


def _normalize_alias(alias: object, *, bundle: Path) -> str:
    if not isinstance(alias, str) or not alias:
        raise _invalid(bundle, "alias must be a non-empty string")
    normalized = _normalize_path(alias, bundle=bundle, what="alias")
    if normalized == ".":
        raise _invalid(bundle, "alias must name a cache directory")
    return normalized


def _normalize_path(name: str, *, bundle: Path, what: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise _invalid(
            bundle,
            f"{what} {name!r} must not traverse directories",
            member=name,
        )
    return path.as_posix()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = _regular_member(name, len(data))
    archive.addfile(member, io.BytesIO(data))


def _add_file(
    archive: tarfile.TarFile, bundle_file: _BundleFile, source: Path
) -> None:
    member = _regular_member(bundle_file.path, bundle_file.size)
    with open(source, "rb") as handle:
        archive.addfile(member, cast(BinaryIO, handle))


def _regular_member(name: str, size: int) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = FILE_MODE
    member.mtime = 0
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    return member


def _root_digest(document_without_root: object) -> str:
    canonical = canonical_json(document_without_root)
    return "sha256:" + hashlib.sha256(BUNDLE_ROOT_DOMAIN + canonical).hexdigest()


def _install_file(temporary: Path, target: Path) -> None:
    handle = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)
    os.chmod(temporary, FILE_MODE)
    os.replace(temporary, target)
    _sync_directory(target.parent)


def _sync_tree(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    ordered = sorted(
        directories,
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in ordered:
        _sync_directory(directory)
    _sync_directory(root)


def _sync_directory(directory: Path) -> None:
    try:
        handle = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory fds
        return
    try:
        os.fsync(handle)
    except OSError:  # pragma: no cover - filesystems without directory fsync
        pass
    finally:
        os.close(handle)


def _remove_file(path: Path) -> None:
    with suppress(OSError):
        path.unlink()


def _invalid(
    bundle: Path,
    failure: str,
    *,
    member: str | None = None,
    cause: str | None = None,
) -> BundleInvalid:
    """Build a ``BUNDLE_INVALID`` error with structured details.

    ``member`` names the archive or manifest member implicated, when one
    is; ``cause`` carries the underlying I/O or parse error text.
    """
    details: dict[str, object] = {"path": str(bundle), "failure": failure}
    if member is not None:
        details["member"] = member
    if cause is not None:
        details["cause"] = cause
    message = f"invalid air-gap bundle: {failure}"
    if cause is not None:
        # The tar reader reports what it tried as several lines, one per
        # compression method. The prose report is one line
        # (``errors.md`` Section 4), so it is folded here and kept whole
        # in ``details["cause"]`` for ``--json`` to carry.
        message = f"{message}: {one_line(cause)}"
    return BundleInvalid(message, details=details)
