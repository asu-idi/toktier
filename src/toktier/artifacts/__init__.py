"""Artifact resolution: manifests, sources, and the verified cache.

An artifact is a tokenizer's files, identified by the sha256 of each
file rather than by a repository name or a revision alone. This package
resolves a family id to those files, fetches what is missing, verifies
what it fetched, and installs it atomically into the cache.

Importing this package pulls in no network client and no accelerator
runtime; the hub client is loaded lazily, inside the fetch call.
"""

from __future__ import annotations

from ..paths import artifact_cache_dir, kernel_cache_dir, store_state_dir
from .bundle import export_bundle, import_bundle
from .manifest import ArtifactEntry, ArtifactFile, ArtifactManifest
from .mirror import MirrorFetcher
from .sibling_aliases import (
    SiblingAliasRecord,
    SiblingAliasRegistry,
    load_sibling_aliases,
    shipped_sibling_aliases,
)
from .sources import (
    AirgapBundleSource,
    ArtifactSource,
    HubFetcher,
    HuggingFaceSource,
    LocalDirectorySource,
    MirrorSource,
)
from .store import ArtifactStore, VerifiedArtifact, sha256_file

__all__ = [
    "AirgapBundleSource",
    "ArtifactEntry",
    "ArtifactFile",
    "ArtifactManifest",
    "ArtifactSource",
    "ArtifactStore",
    "HubFetcher",
    "HuggingFaceSource",
    "LocalDirectorySource",
    "MirrorFetcher",
    "MirrorSource",
    "SiblingAliasRecord",
    "SiblingAliasRegistry",
    "VerifiedArtifact",
    "artifact_cache_dir",
    "export_bundle",
    "import_bundle",
    "kernel_cache_dir",
    "load_sibling_aliases",
    "sha256_file",
    "shipped_sibling_aliases",
    "store_state_dir",
]
