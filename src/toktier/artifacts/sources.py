"""Artifact sources: where the bytes of an artifact can come from.

A source produces bytes at a destination path and nothing else. It does
not verify, install, or cache: hashing, atomic installation and the
mismatch policy live in :mod:`toktier.artifacts.store`, so there is one
place where the verification rules are written down.

Four sources are provided (``ARCHITECTURE.md`` section 1.4): the
upstream hub, a local directory, an internal mirror and an air-gapped
bundle.  Mirror and bundle implementations live in their focused
modules and are re-exported here with the original protocol surface.

Nothing in this module imports ``huggingface_hub`` at import time; the
hub client is loaded lazily inside the fetch call, so importing toktier
touches no network stack.

A source's ``offline`` property reports **only that source's own**
reachability. It is not the answer to "can toktier fetch": the
configuration and the presence of a source decide that as well, and
:func:`toktier.artifacts.store.fetch_availability` is the single place
where the three are combined.
"""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Protocol, cast

from ..errors import ArtifactNotFound, ToktierError, one_line
from .bundle import AirgapBundleSource
from .manifest import ArtifactEntry, ArtifactFile
from .mirror import MirrorSource

__all__ = [
    "AirgapBundleSource",
    "ArtifactSource",
    "HubFetcher",
    "HuggingFaceSource",
    "LocalDirectorySource",
    "MirrorSource",
]

#: Environment variable of the hub client that switches it offline. It
#: is not a toktier configuration knob; it is read once, when a
#: :class:`HuggingFaceSource` is constructed, and only makes the source
#: refuse to reach out.
ENV_HF_HUB_OFFLINE = "HF_HUB_OFFLINE"

_HF_OFFLINE_TRUE = frozenset({"1", "true", "yes", "on"})


class ArtifactSource(Protocol):
    """Produces the bytes of one artifact file at a destination path."""

    @property
    def name(self) -> str:
        """Short identifier of this source, used in diagnostics."""

    @property
    def offline(self) -> bool:
        """True when this source cannot produce bytes right now.

        An offline source is never called: the store reports a missing
        artifact instead, and a digest mismatch becomes an immediate
        hard error rather than a re-fetch. This flag covers this source
        alone; the effective answer is
        :func:`toktier.artifacts.store.fetch_availability`.
        """

    def fetch(
        self,
        entry: ArtifactEntry,
        artifact_file: ArtifactFile,
        destination: Path,
    ) -> None:
        """Write the file's bytes to ``destination``.

        ``destination`` is a private temporary path chosen by the
        caller; the source must not touch the final location. Failure
        to produce the file raises :class:`ArtifactNotFound`, including
        a failure raised inside a client this package does not own: it
        is classified here, at the boundary, rather than left to escape
        as whatever type that client happens to use.
        """


class HubFetcher(Protocol):
    """The single hub call this package uses (injectable for tests)."""

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: str,
    ) -> str:
        """Download one file and return the local path it landed in."""


class HuggingFaceSource:
    """Fetches artifact files from the upstream hub.

    The hub client is imported lazily and can be replaced by an
    injected ``fetcher``, which is what the test suite does: no test
    reaches the network.
    """

    name = "huggingface"

    def __init__(
        self,
        *,
        fetcher: HubFetcher | None = None,
        offline: bool | None = None,
    ) -> None:
        self._fetcher = fetcher
        # Read once, at construction, like every other environment
        # value in this package.
        self._offline = _hub_offline_from_env() if offline is None else offline

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
                f"artifact {entry.family!r} is not present locally and the hub "
                "source is offline",
                details={
                    "family": entry.family,
                    "searched": [f"{entry.repo_id}@{entry.revision}"],
                    "offline": True,
                },
            )
        fetcher = self._fetcher if self._fetcher is not None else _load_hub_fetcher()
        with tempfile.TemporaryDirectory(
            prefix=".toktier-hub-", dir=str(destination.parent)
        ) as staging:
            # The hub client is a third party, and everything it can
            # raise -- a gated or private repository, an expired token,
            # a name that moved, a socket that failed -- means the same
            # thing here: these bytes did not arrive. The protocol above
            # says a source reports that as ``ArtifactNotFound``, so the
            # client's own exception type is carried in ``details``
            # rather than escaping as a traceback and taking the exit
            # code and the ``--json`` envelope with it.
            try:
                downloaded = fetcher(
                    repo_id=entry.repo_id,
                    filename=artifact_file.name,
                    revision=entry.revision,
                    local_dir=staging,
                )
            except ToktierError:
                # Already one of ours, with its own code; passing it
                # through keeps that code instead of overwriting it.
                raise
            except Exception as error:
                raise ArtifactNotFound(
                    f"the hub did not deliver {artifact_file.name!r} of "
                    f"{entry.family!r} from {entry.repo_id}@{entry.revision}: "
                    f"{type(error).__name__}: {one_line(str(error))}",
                    details={
                        "family": entry.family,
                        "searched": [f"{entry.repo_id}@{entry.revision}"],
                        "offline": False,
                        "cause": type(error).__name__,
                        "cause_message": str(error),
                        "remedy": (
                            "make the repository reachable from this machine "
                            "(credentials for a gated one, or network access), "
                            "or supply the file from a local directory, a "
                            "mirror, or an air-gap bundle"
                        ),
                    },
                ) from error
            source_path = Path(downloaded)
            if not source_path.is_file():
                raise ArtifactNotFound(
                    f"hub returned no file for {entry.family}/{artifact_file.name}",
                    details={
                        "family": entry.family,
                        "searched": [str(source_path)],
                        "offline": False,
                    },
                )
            shutil.copyfile(source_path, destination)


class LocalDirectorySource:
    """Reads artifact files from a directory tree on this machine.

    The directory for an entry is its ``local_dir`` when the manifest
    names one (resolved against ``root`` if it is relative), otherwise
    ``root/<directory_name>``.
    """

    name = "local_dir"
    offline = False

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self._root = Path(root) if root is not None else None

    def directory(self, entry: ArtifactEntry) -> Path:
        if entry.local_dir is not None:
            candidate = Path(entry.local_dir)
            if candidate.is_absolute() or self._root is None:
                return candidate
            return self._root / candidate
        if self._root is None:
            raise ArtifactNotFound(
                f"no local directory configured for {entry.family!r}",
                details={"family": entry.family, "searched": [], "offline": False},
            )
        return self._root / entry.directory_name

    def fetch(
        self,
        entry: ArtifactEntry,
        artifact_file: ArtifactFile,
        destination: Path,
    ) -> None:
        source_path = self.directory(entry) / artifact_file.name
        if not source_path.is_file():
            raise ArtifactNotFound(
                f"local artifact file missing: {source_path}",
                details={
                    "family": entry.family,
                    "searched": [str(source_path)],
                    "offline": False,
                },
            )
        shutil.copyfile(source_path, destination)


def _hub_offline_from_env() -> bool:
    """Read the hub client's own offline switch, tolerantly."""
    raw = os.environ.get(ENV_HF_HUB_OFFLINE)
    if raw is None:
        return False
    return raw.strip().lower() in _HF_OFFLINE_TRUE


def _load_hub_fetcher() -> HubFetcher:
    """Import the hub client lazily and adapt its download function."""
    try:
        hub = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        raise ArtifactNotFound(
            "the huggingface_hub package is required to fetch artifacts from "
            "the hub",
            details={
                "family": None,
                "searched": [],
                "offline": False,
                "missing": "huggingface_hub",
            },
        ) from exc
    download = cast("HubFetcher", hub.hf_hub_download)
    return download
