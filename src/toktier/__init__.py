"""Public contract types for the toktier package."""

from __future__ import annotations

from .config import Config
from .errors import ToktierError
from .policy import ReasonCode, RoutePlan, RoutingPolicy
from .session import SessionUpdate

__all__ = [
    "API_VERSION",
    "Config",
    "ReasonCode",
    "RoutePlan",
    "RoutingPolicy",
    "SessionUpdate",
    "ToktierError",
    "__version__",
]

#: Public API version axis (see docs/contracts/versioning.md).
API_VERSION: int = 1


def _distribution_version() -> str:
    """Read the package version from installed distribution metadata."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("toktier")
    except PackageNotFoundError:  # pragma: no cover - checkout-only runs
        return "0.0.0"


def __getattr__(name: str) -> str:
    """Serve the package version axis on demand."""
    if name == "__version__":
        return _distribution_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
