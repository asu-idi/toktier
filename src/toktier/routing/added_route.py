"""Input-level routing for added-token literals.

Contract reference: ``docs/contracts/routing.md`` Section 5.2
(``R_INPUT_ADDED_TOKEN``): "Input contains an added-token literal; this
input is routed to the reference frontend path. Part of the certified
pipeline design, not a correctness incident."

Open item recorded here rather than decided here
------------------------------------------------

Two readings of that clause exist, and they differ in throughput, not in
correctness:

1. **Whole input to the reference backend** (implemented). An input
   holding a literal runs entirely on the reference path.
2. **Split and splice.** The frontend extracts the literals, the
   segments between them run on the accelerated backend, and the ids are
   joined in order. This is what the prototype pipeline did, and it is
   what its routed-input counter measured.

Reading 1 is implemented because it is the conservative one and because
the accelerated backend is not part of this lane, so nothing is lost
today. :meth:`AddedTokenRouter.plan_for` exposes the split so reading 2
becomes a small change once the mainline settles the wording; the
splicing helper itself is already ported
(:func:`toktier.frontend.added.assemble`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

__all__ = ["AddedTokenRouter", "LiteralScanner"]


class LiteralScanner(Protocol):
    """The part of the added-token frontend routing depends on."""

    def scan(self, text: str) -> Sequence[tuple[str, int | None]] | None:
        """Ordered spans of the document, or ``None`` when it holds none."""


class AddedTokenRouter:
    """Answers whether an input takes the added-token path."""

    def __init__(self, frontend: LiteralScanner) -> None:
        self._frontend = frontend

    def plan_for(self, text: str) -> Sequence[tuple[str, int | None]] | None:
        """The literal split of ``text``, or ``None`` when it holds none."""
        return self._frontend.scan(text)

    def holds_literal(self, text: str) -> bool:
        """Whether ``text`` holds at least one added-token literal."""
        return self._frontend.scan(text) is not None
