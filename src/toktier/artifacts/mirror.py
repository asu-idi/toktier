"""Artifact retrieval from an HTTP-compatible repository mirror.

The mirror uses the same repository-relative URL shape as the upstream
Hugging Face endpoint::

    <base>/<repo_id>/resolve/<revision>/<artifact path>

Like every :class:`~toktier.artifacts.sources.ArtifactSource`, this source
only writes bytes to the private temporary destination supplied by the
store.  Per-file size and sha256 verification, quarantine, the single
re-fetch, and atomic installation therefore remain centralized in
``ArtifactStore``.

The default URL fetcher is imported lazily so importing :mod:`toktier`
does not load a network stack.  Tests and embedded users can inject the
small ``MirrorFetcher`` callable instead.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import quote, urljoin

from ..errors import ArtifactNotFound
from .manifest import ArtifactEntry, ArtifactFile

__all__ = ["MirrorFetcher", "MirrorSource"]


class MirrorFetcher(Protocol):
    """Fetch one URL into a caller-owned temporary destination."""

    def __call__(self, *, url: str, destination: Path) -> None:
        """Write the response body to ``destination``."""


class _URLRetrieve(Protocol):
    def __call__(self, url: str, filename: str) -> object: ...


class MirrorSource:
    """Fetch artifact files from a configured repository mirror."""

    name = "mirror"

    def __init__(
        self,
        base_url: str,
        *,
        fetcher: MirrorFetcher | None = None,
        offline: bool = False,
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty string")
        self._base_url = base_url.rstrip("/") + "/"
        self._fetcher = fetcher
        self._offline = offline

    @property
    def offline(self) -> bool:
        return self._offline

    def url(self, entry: ArtifactEntry, artifact_file: ArtifactFile) -> str:
        """Return the mirror URL for one manifest-pinned artifact file."""
        relative = "/".join(
            (
                quote(entry.repo_id.lstrip("/"), safe="/"),
                "resolve",
                quote(entry.revision, safe=""),
                quote(artifact_file.name, safe="/"),
            )
        )
        return urljoin(self._base_url, relative)

    def fetch(
        self,
        entry: ArtifactEntry,
        artifact_file: ArtifactFile,
        destination: Path,
    ) -> None:
        url = self.url(entry, artifact_file)
        if self._offline:
            raise ArtifactNotFound(
                f"artifact {entry.family!r} is not present locally and the "
                "mirror source is offline",
                details={
                    "family": entry.family,
                    "searched": [url],
                    "offline": True,
                },
            )

        fetcher = self._fetcher if self._fetcher is not None else _fetch_url
        try:
            fetcher(url=url, destination=destination)
        except OSError as exc:
            raise ArtifactNotFound(
                f"mirror artifact file unavailable: {url}",
                details={
                    "family": entry.family,
                    "searched": [url],
                    "offline": False,
                },
            ) from exc
        if not destination.is_file():
            raise ArtifactNotFound(
                f"mirror returned no file for {entry.family}/{artifact_file.name}",
                details={
                    "family": entry.family,
                    "searched": [url],
                    "offline": False,
                },
            )


def _fetch_url(*, url: str, destination: Path) -> None:
    """Load ``urllib.request`` lazily and retrieve one URL."""
    request = importlib.import_module("urllib.request")
    retrieve = cast("_URLRetrieve", request.urlretrieve)
    retrieve(url, str(destination))
