"""toktier: correctness-first, high-throughput tokenization toolkit.

The frozen public contracts live in ``docs/contracts/``; this package
provides their importable, typed counterparts. :func:`load` is the entry
point: it resolves and verifies a family's artifact and returns a
:class:`Tokenizer` with encode, decode, session and content-lookup
paths, every one of which returns ids equal to a from-scratch reference
encode.
"""

from __future__ import annotations

from .config import Config
from .errors import ToktierError
from .facade import Encoding, Tokenizer, load
from .policy import ReasonCode, RoutePlan, RoutingPolicy
from .session import SessionUpdate

__all__ = [
    "API_VERSION",
    "Config",
    "Encoding",
    "ReasonCode",
    "RoutePlan",
    "RoutingPolicy",
    "SessionUpdate",
    "Tokenizer",
    "ToktierError",
    "__version__",
    "load",
]

#: Public API version axis (see docs/contracts/versioning.md).
API_VERSION: int = 1


def _distribution_version() -> str:
    """Package version axis, read from installed metadata.

    Read lazily (module ``__getattr__``): the metadata machinery pulls
    in modules that must stay off the import path of ``import toktier``.
    A source tree that is not installed has no distribution metadata;
    the placeholder says so instead of imitating a release number.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("toktier")
    except PackageNotFoundError:  # pragma: no cover - checkout-only runs
        return "0.0.0"


def __getattr__(name: str) -> str:
    """Serve the package version axis (versioning.md axis 1) on demand."""
    if name == "__version__":
        return _distribution_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
