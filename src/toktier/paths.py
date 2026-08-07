"""On-disk layout: cache (rebuildable) versus state (not rebuildable).

Contract reference: ``docs/contracts/config.md`` section 5. The
distinction is deliberate and user-visible:

- **Cache** holds fetched artifacts and built kernels. Deleting it costs
  re-download or re-build time, never data.
- **State** holds the session store. Deleting it loses sessions; no tool
  or document may describe it as "just a cache".

Both roots come from an immutable :class:`~toktier.config.Config`, which
resolved them once at construction (``TOKTIER_HOME`` when set, otherwise
the platform conventions). Nothing here re-reads the environment.

This module is dependency-free (standard library only).
"""

from __future__ import annotations

from pathlib import Path

from .config import Config

__all__ = [
    "DIRECTORY_MODE",
    "FILE_MODE",
    "artifact_cache_dir",
    "ensure_private_dir",
    "kernel_cache_dir",
    "store_state_dir",
]

#: Directories toktier creates are owner-only.
DIRECTORY_MODE = 0o700

#: Files toktier creates are owner-only.
FILE_MODE = 0o600


def artifact_cache_dir(config: Config) -> Path:
    """Cache subtree for verified tokenizer artifacts."""
    return config.cache_dir / "artifacts"


def kernel_cache_dir(config: Config) -> Path:
    """Cache subtree for built kernels and generated lookup tables."""
    return config.cache_dir / "kernels"


def store_state_dir(config: Config) -> Path:
    """State subtree for the session store. Not a cache: deleting it
    loses sessions."""
    return config.state_dir / "store"


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) with owner-only permissions.

    An existing directory is left as the operator configured it; only
    directories created here get the restrictive mode.
    """
    if path.is_dir():
        return path
    path.mkdir(parents=True, exist_ok=True, mode=DIRECTORY_MODE)
    return path
