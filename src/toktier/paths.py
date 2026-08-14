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

import os
from pathlib import Path

from .config import Config
from .errors import ConfigInvalid

__all__ = [
    "DIRECTORY_MODE",
    "FILE_MODE",
    "artifact_cache_dir",
    "ensure_private_dir",
    "kernel_cache_dir",
    "private_dir_problem",
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


def private_dir_problem(path: Path) -> str | None:
    """Why ``path`` cannot become a private directory, or ``None``.

    The reading half of :func:`ensure_private_dir`: it reaches the same
    judgement about a configured location without creating anything, so a
    diagnostic can report the answer an operation would get instead of
    printing the path as though it were fine. Being a question about the
    filesystem it is answered as of now, and a root can still change
    between this call and the operation that follows it.
    """
    if path.is_dir():
        return None
    if path.exists() or path.is_symlink():
        return f"{path} exists and is not a directory"
    for ancestor in path.parents:
        if ancestor.is_dir():
            if not os.access(ancestor, os.W_OK | os.X_OK):
                return f"{ancestor} cannot be written by this user"
            return None
        if ancestor.exists() or ancestor.is_symlink():
            return f"{ancestor} exists and is not a directory"
    return f"{path} has no existing parent directory"


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) with owner-only permissions.

    An existing directory is left as the operator configured it; only
    directories created here get the restrictive mode -- and *every*
    directory created here does, the intermediate ones included.
    ``Path.mkdir(parents=True, mode=...)`` applies the mode to the final
    component alone and leaves the parents at the process umask, which
    ``config.md`` section 5 does not offer; the components are therefore
    created one at a time. The mode is set explicitly after creation so
    the result does not depend on the umask either.

    A location that cannot become a private directory is a statement
    about the configured root, not an accident of the moment, so it is
    reported as :class:`~toktier.errors.ConfigInvalid` rather than
    escaping as the operating system's own exception. ``cause`` names
    that exception so nothing is lost.
    """
    if path.is_dir():
        return path
    if path.exists() or path.is_symlink():
        raise ConfigInvalid(
            f"{path} exists and is not a directory, so it cannot hold "
            "toktier's private state",
            details={
                "field": "path",
                "value": str(path),
                "source": "filesystem",
                "remedy": (
                    "point TOKTIER_HOME (or the matching cache/state "
                    "variable) at a directory"
                ),
            },
        )
    try:
        for component in [*reversed(path.parents), path]:
            try:
                component.mkdir(mode=DIRECTORY_MODE)
            except FileExistsError:
                # Already there: another process, or the operator. Its
                # mode is not ours to change.
                continue
            os.chmod(component, DIRECTORY_MODE)
    except OSError as exc:
        failed = Path(exc.filename) if exc.filename else path
        raise ConfigInvalid(
            f"cannot create toktier's private directory {failed}: "
            f"{exc.strerror or exc}",
            details={
                "field": "path",
                "value": str(path),
                "source": "filesystem",
                "cause": type(exc).__name__,
                "cause_message": str(exc),
                "remedy": (
                    "set TOKTIER_HOME to a directory this user can write"
                ),
            },
        ) from exc
    return path
