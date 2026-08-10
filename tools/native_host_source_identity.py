#!/usr/bin/env python3
"""Canonical source identity of the Rust native prebuilt-GPU host.

The domain and path set live in :mod:`source_identity_common`, one
table for all three identities; this wrapper keeps the established
command-line and import surface.
"""

from __future__ import annotations

from pathlib import Path

from source_identity_common import IDENTITIES

_IDENTITY = IDENTITIES["native_host"]

DOMAIN = _IDENTITY.domain
FILES = _IDENTITY.files
TREES = _IDENTITY.trees


def source_paths() -> tuple[Path, ...]:
    """Return the exact, sorted source set hashed by ``toktier-py/build.rs``."""
    return _IDENTITY.source_paths()


def source_digest() -> str:
    """Bare SHA-256 over path-bound native-host source bytes."""
    return _IDENTITY.source_digest()


if __name__ == "__main__":
    print(source_digest())
