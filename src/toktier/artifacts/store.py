"""Verified artifact cache: fetch, verify, install.

Every file that reaches the cache has been checked against the
manifest's per-file sha256. The installation path is deliberate:

1. take a per-artifact lock, so two processes cannot race on the same
   directory;
2. download into a private temporary file whose name carries the
   process id (concurrent first fetches must not collide);
3. check the declared size, then the sha256 -- both, not one;
4. ``fsync`` the file, rename it into place atomically, ``fsync`` the
   directory;
5. record a verified marker so a later run can skip re-hashing when
   nothing on disk moved.

Digest mismatch policy (adoption matrix, tier B):

- **online**: the suspect bytes are moved to a quarantine directory and
  the file is fetched once more; if that also fails to verify, the call
  raises :class:`~toktier.errors.ArtifactHashMismatch`. Nothing that
  failed verification is ever left in place of a good file.
- **offline**: the call raises immediately, carrying the expected and
  observed digests, the path, and a remediation command.

"Offline" is not one condition but three: the configuration says so
(``TOKTIER_OFFLINE`` or ``Config.offline``), no source was supplied, or
the source itself reports that it cannot reach out (for example the hub
client with ``HF_HUB_OFFLINE`` set). Any one of them disables fetching.
:func:`fetch_availability` is the single place where the three are
combined, and :class:`FetchAvailability` keeps them separately readable
so a diagnostic can say *which* one is in force instead of printing one
word that means three different things.

This module is dependency-free (standard library only).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..config import Config
from ..errors import ArtifactHashMismatch, ArtifactNotFound
from ..paths import DIRECTORY_MODE, FILE_MODE, artifact_cache_dir, ensure_private_dir
from .manifest import ArtifactEntry, ArtifactFile, ArtifactManifest
from .sources import ArtifactSource

try:  # pragma: no cover - platform dependent
    import fcntl

    _HAVE_FCNTL = True
except ModuleNotFoundError:  # pragma: no cover - non-POSIX platforms
    _HAVE_FCNTL = False

__all__ = [
    "ArtifactStore",
    "FetchAvailability",
    "VerifiedArtifact",
    "fetch_availability",
    "sha256_file",
]

#: Reason ids of :attr:`FetchAvailability.reasons`. Stable strings: they
#: reach diagnostics and error details.
REASON_CONFIGURED_OFFLINE = "configured_offline"
REASON_NO_SOURCE = "no_source"
REASON_SOURCE_OFFLINE = "source_offline"

#: Name of the verified marker written inside an artifact directory.
MARKER_NAME = ".toktier-verified.json"

#: Marker format version. A marker this reader does not understand is
#: ignored, which costs a re-hash and never accepts unverified bytes.
MARKER_FORMAT = 1

#: Initial fetch plus exactly one re-fetch after a mismatch.
FETCH_ATTEMPTS = 2

_READ_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class FetchAvailability:
    """Whether artifact bytes may be fetched, and why not when they cannot.

    Three independent conditions disable fetching, and collapsing them
    into a single ``offline`` flag loses the only fact a user needs: a
    configuration that says ``offline = false`` still cannot fetch when
    the source itself is offline. Each condition is therefore recorded
    on its own, and :attr:`available` is the one derived answer.
    """

    #: ``Config.offline``: the caller asked for no network at all.
    configured_offline: bool
    #: Name of the configured source, or ``None`` when there is none.
    source_name: str | None
    #: The source's own reachability. ``False`` when no source is
    #: configured: with nothing to ask, this condition does not apply
    #: and :attr:`source_configured` is the field that says so.
    source_offline: bool

    @property
    def source_configured(self) -> bool:
        """True when a source object was supplied at all."""
        return self.source_name is not None

    @property
    def available(self) -> bool:
        """True when a fetch may be attempted."""
        return not self.reasons

    @property
    def offline(self) -> bool:
        """True when no fetch may happen, for any of the three reasons."""
        return not self.available

    @property
    def reasons(self) -> tuple[str, ...]:
        """Every reason fetching is disabled, in a stable order."""
        reasons: list[str] = []
        if self.configured_offline:
            reasons.append(REASON_CONFIGURED_OFFLINE)
        if self.source_name is None:
            reasons.append(REASON_NO_SOURCE)
        elif self.source_offline:
            reasons.append(REASON_SOURCE_OFFLINE)
        return tuple(reasons)


def fetch_availability(
    config: Config, source: ArtifactSource | None
) -> FetchAvailability:
    """Decide whether bytes can be fetched right now.

    This is the only place the three offline conditions are combined;
    the store, the command line and any other diagnostic read the
    answer from here rather than re-deriving it.
    """
    return FetchAvailability(
        configured_offline=bool(config.offline),
        source_name=None if source is None else source.name,
        source_offline=False if source is None else bool(source.offline),
    )


@dataclass(frozen=True)
class VerifiedArtifact:
    """An artifact whose files are present and hash-verified."""

    family: str
    revision: str
    directory: Path
    files: Mapping[str, Path]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    def path(self, name: str) -> Path:
        """Path of one verified file."""
        try:
            return self.files[name]
        except KeyError as exc:
            raise ArtifactNotFound(
                f"artifact {self.family!r} has no file named {name!r}",
                details={"family": self.family, "searched": sorted(self.files)},
            ) from exc


def sha256_file(path: Path) -> tuple[str, int]:
    """Return the sha256 hex digest and byte length of a file."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


class ArtifactStore:
    """Fetches artifacts into the cache and keeps them verified."""

    def __init__(
        self,
        manifest: ArtifactManifest,
        *,
        config: Config | None = None,
        source: ArtifactSource | None = None,
    ) -> None:
        self._manifest = manifest
        self._config = config if config is not None else Config.resolve()
        self._source = source
        self._root = artifact_cache_dir(self._config)

    # -- properties ----------------------------------------------------

    @property
    def manifest(self) -> ArtifactManifest:
        return self._manifest

    @property
    def root(self) -> Path:
        """Cache subtree holding verified artifacts."""
        return self._root

    @property
    def quarantine_root(self) -> Path:
        """Where bytes that failed verification are kept for inspection."""
        return self._root / ".quarantine"

    @property
    def availability(self) -> FetchAvailability:
        """Whether this store may fetch, and why not when it may not."""
        return fetch_availability(self._config, self._source)

    @property
    def offline(self) -> bool:
        """True when no fetch may happen for any reason.

        Shorthand for ``not self.availability.available``; read
        :attr:`availability` when the reason matters.
        """
        return self.availability.offline

    def directory(self, family: str) -> Path:
        """Cache directory of one artifact (may not exist yet)."""
        return self._root / self._manifest.get(family).directory_name

    # -- operations ----------------------------------------------------

    def ensure(self, family: str) -> VerifiedArtifact:
        """Return the artifact, fetching and verifying what is missing."""
        return self._resolve(family, rehash=False)

    def verify(self, family: str) -> VerifiedArtifact:
        """Re-hash every file of a cached artifact, ignoring the marker."""
        return self._resolve(family, rehash=True)

    def _resolve(self, family: str, *, rehash: bool) -> VerifiedArtifact:
        entry = self._manifest.get(family)
        directory = self._root / entry.directory_name
        # Fast path: an artifact that is already verified needs no lock
        # and writes nothing, so a warm cache stays usable when the
        # cache directory is read-only.
        if not rehash and self._marker_is_current(directory, entry):
            return self._verified(entry, directory)
        with self._lock(entry):
            # Re-check under the lock: another process may have just
            # finished the same work.
            if rehash or not self._marker_is_current(directory, entry):
                ensure_private_dir(directory)
                for artifact_file in entry.files:
                    self._ensure_file(entry, artifact_file, directory)
                self._write_marker(directory, entry)
        return self._verified(entry, directory)

    def _verified(
        self, entry: ArtifactEntry, directory: Path
    ) -> VerifiedArtifact:
        return VerifiedArtifact(
            family=entry.family,
            revision=entry.revision,
            directory=directory,
            files={item.name: directory / item.name for item in entry.files},
        )

    # -- per-file work -------------------------------------------------

    def _ensure_file(
        self, entry: ArtifactEntry, artifact_file: ArtifactFile, directory: Path
    ) -> None:
        target = _resolve_within(directory, artifact_file.name)
        # One reading of the fetch conditions for the whole file, so a
        # single call cannot act on two different answers.
        availability = self.availability
        quarantined: list[str] = []
        if target.is_file():
            observed_sha, observed_size = sha256_file(target)
            if _matches(artifact_file, observed_sha, observed_size):
                return
            if availability.offline:
                raise self._mismatch(
                    entry,
                    artifact_file,
                    target,
                    observed_sha,
                    observed_size,
                    availability=availability,
                    quarantined=[],
                    attempts=0,
                )
            # Online: the cached copy is suspect. Move it aside so that a
            # failed repair cannot leave unverified bytes in place.
            quarantined.append(str(self._quarantine(entry, target)))

        if availability.offline:
            raise ArtifactNotFound(
                f"artifact file {artifact_file.name!r} of {entry.family!r} is not "
                "in the cache and fetching is disabled (offline)",
                details={
                    "family": entry.family,
                    "searched": [str(target)],
                    "offline": True,
                    "offline_reasons": list(availability.reasons),
                },
            )

        source = self._source
        if source is None:  # pragma: no cover - the offline check covers this
            raise ArtifactNotFound(
                f"no artifact source configured for {entry.family!r}",
                details={
                    "family": entry.family,
                    "searched": [str(target)],
                    "offline": True,
                    "offline_reasons": [REASON_NO_SOURCE],
                },
            )
        target.parent.mkdir(parents=True, exist_ok=True, mode=DIRECTORY_MODE)
        last_sha: str | None = None
        last_size: int | None = None
        for attempt in range(FETCH_ATTEMPTS):
            temporary = directory / _temporary_name(artifact_file.name, attempt)
            try:
                source.fetch(entry, artifact_file, temporary)
                observed_sha, observed_size = sha256_file(temporary)
                if _matches(artifact_file, observed_sha, observed_size):
                    _install(temporary, target)
                    return
                last_sha, last_size = observed_sha, observed_size
                quarantined.append(str(self._quarantine(entry, temporary)))
            finally:
                _remove_quietly(temporary)
        raise self._mismatch(
            entry,
            artifact_file,
            target,
            last_sha,
            last_size,
            availability=availability,
            quarantined=quarantined,
            attempts=FETCH_ATTEMPTS,
        )

    def _mismatch(
        self,
        entry: ArtifactEntry,
        artifact_file: ArtifactFile,
        path: Path,
        observed_sha: str | None,
        observed_size: int | None,
        *,
        availability: FetchAvailability,
        quarantined: list[str],
        attempts: int,
    ) -> ArtifactHashMismatch:
        offline = availability.offline
        if offline:
            remedy = (
                f"delete {path} and run 'toktier artifacts fetch {entry.family}' "
                "on a host with network access"
            )
        else:
            remedy = f"toktier artifacts fetch {entry.family} --force"
        details: dict[str, Any] = {
            "family": entry.family,
            "file": artifact_file.name,
            "expected_sha256": artifact_file.sha256,
            "observed_sha256": observed_sha,
            "path": str(path),
            "remedy": remedy,
            "offline": offline,
            "offline_reasons": list(availability.reasons),
            "attempts": attempts,
            "quarantined": quarantined,
        }
        if artifact_file.size is not None:
            details["expected_size"] = artifact_file.size
            details["observed_size"] = observed_size
        return ArtifactHashMismatch(
            f"content hash mismatch for {entry.family}/{artifact_file.name}",
            details=details,
        )

    def _quarantine(self, entry: ArtifactEntry, path: Path) -> Path:
        """Move bytes that failed verification out of the way."""
        directory = ensure_private_dir(self.quarantine_root / entry.directory_name)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        flat = path.name.lstrip(".").replace("/", "__")
        destination = directory / f"{stamp}-{os.getpid()}-{flat}"
        counter = 0
        while destination.exists():
            counter += 1
            destination = directory / f"{stamp}-{os.getpid()}-{counter}-{flat}"
        os.replace(path, destination)
        return destination

    # -- verified marker -----------------------------------------------

    def _marker_is_current(self, directory: Path, entry: ArtifactEntry) -> bool:
        """True when the marker still describes what is on disk.

        The marker is an optimization only: anything unexpected means a
        full re-hash, never an acceptance.
        """
        marker_path = directory / MARKER_NAME
        try:
            raw = marker_path.read_text(encoding="utf-8")
        except OSError:
            return False
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(data, dict):
            return False
        if data.get("format") != MARKER_FORMAT:
            return False
        if data.get("family") != entry.family or data.get("revision") != entry.revision:
            return False
        recorded = data.get("files")
        if not isinstance(recorded, dict):
            return False
        if set(recorded) != {item.name for item in entry.files}:
            return False
        for artifact_file in entry.files:
            state = recorded.get(artifact_file.name)
            if not isinstance(state, dict):
                return False
            if state.get("sha256") != artifact_file.sha256:
                return False
            if (
                artifact_file.size is not None
                and state.get("size") != artifact_file.size
            ):
                return False
            path = directory / artifact_file.name
            try:
                stat = path.stat()
            except OSError:
                return False
            if state.get("size") != stat.st_size:
                return False
            if state.get("mtime_ns") != stat.st_mtime_ns:
                return False
        return True

    def _write_marker(self, directory: Path, entry: ArtifactEntry) -> None:
        files: dict[str, dict[str, Any]] = {}
        for artifact_file in entry.files:
            stat = (directory / artifact_file.name).stat()
            files[artifact_file.name] = {
                "sha256": artifact_file.sha256,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        payload = {
            "format": MARKER_FORMAT,
            "family": entry.family,
            "revision": entry.revision,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": files,
        }
        temporary = directory / f"{MARKER_NAME}.{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _install(temporary, directory / MARKER_NAME)

    # -- locking -------------------------------------------------------

    @contextmanager
    def _lock(self, entry: ArtifactEntry) -> Iterator[None]:
        """Hold an exclusive per-artifact lock for the critical section."""
        lock_dir = ensure_private_dir(self._root / ".locks")
        lock_path = lock_dir / f"{entry.directory_name}.lock"
        handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, FILE_MODE)
        try:
            if _HAVE_FCNTL:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if _HAVE_FCNTL:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


def _matches(
    artifact_file: ArtifactFile, observed_sha: str, observed_size: int
) -> bool:
    """Both checks, in the cheap-first order: size, then digest."""
    if artifact_file.size is not None and artifact_file.size != observed_size:
        return False
    return artifact_file.sha256 == observed_sha


def _resolve_within(directory: Path, name: str) -> Path:
    """Join a manifest file name to the artifact directory, safely."""
    candidate = (directory / name).resolve(strict=False)
    root = directory.resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise ArtifactNotFound(
            f"artifact file name {name!r} escapes its directory",
            details={"searched": [str(directory)], "file": name},
        )
    return directory / name


def _temporary_name(name: str, attempt: int) -> str:
    """Temporary file name carrying the process id.

    Two workers doing a first fetch of the same file at the same time
    must not write the same temporary path; the process id in the name
    is what keeps them apart.
    """
    flat = name.replace("/", "__")
    return f".{flat}.{os.getpid()}.{attempt}.tmp"


def _install(temporary: Path, target: Path) -> None:
    """fsync, atomically rename, then fsync the directory."""
    handle = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)
    os.chmod(temporary, FILE_MODE)
    os.replace(temporary, target)
    try:
        directory_handle = os.open(target.parent, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory fds
        return
    try:
        os.fsync(directory_handle)
    except OSError:  # pragma: no cover - filesystems without directory fsync
        pass
    finally:
        os.close(directory_handle)


def _remove_quietly(path: Path) -> None:
    with suppress(OSError):
        path.unlink()
