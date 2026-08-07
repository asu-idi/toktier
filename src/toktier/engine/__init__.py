"""Backend implementations behind the routing layer.

Subpackages here implement the backends the routing layer chooses
between. They do not read configuration and they do not decide policy:
they are handed what they need and report what they observed.

``toktier.engine.gpu``
    The CUDA backend. Requires the ``gpu-jit`` extra (torch and ninja);
    importing this package does not import it.
"""

from __future__ import annotations

__all__: list[str] = []
