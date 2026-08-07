"""Access to the oracle package, in one place.

The oracle is the Hugging Face ``tokenizers`` package: it defines
correct output, and every certification reading was taken against a
specific version of it (``docs/contracts/registry.md`` Section 2).

Two rules are implemented here so they cannot drift between callers:

- The version is read from installed distribution metadata, so probing
  can report it without importing the package.
- The import itself happens through :func:`import_oracle`, late and in
  one place, so that importing ``toktier`` neither loads the oracle nor
  pays for it. A missing oracle is reported as
  ``ORACLE_VERSION_UNSUPPORTED``: without it there is no correct output
  to produce, which is an error rather than a fallback.

The module is otherwise dependency-free (standard library only).
"""

from __future__ import annotations

from typing import Any

from .errors import OracleVersionUnsupported

__all__ = ["ORACLE_PACKAGE", "import_oracle", "oracle_version"]

#: The package whose behavior defines correct output. The registry names
#: it per record and binds a semantic id for it; package metadata
#: deliberately does not pin the version.
ORACLE_PACKAGE = "tokenizers"


def oracle_version() -> str | None:
    """Installed version of the oracle package, or ``None`` if absent."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(ORACLE_PACKAGE)
    except PackageNotFoundError:
        return None


def import_oracle() -> Any:
    """Import and return the oracle package.

    Raises:
        OracleVersionUnsupported: the package is absent or fails at
            import level, so reference execution is impossible.
    """
    from importlib import import_module

    try:
        return import_module(ORACLE_PACKAGE)
    except ImportError as exc:
        raise OracleVersionUnsupported(
            f"the {ORACLE_PACKAGE} package is required for the reference "
            "backend and could not be imported",
            details={
                "package": ORACLE_PACKAGE,
                "installed": oracle_version(),
                "certified": None,
            },
        ) from exc
