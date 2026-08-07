"""Session value objects shared by the public surface.

Contract reference: ``docs/contracts/api.md`` Section 5.1. The
``SessionUpdate`` shape is frozen: all three fields speak about the
pre-postprocessor core token stream, and the splice invariant
``all_ids == old_ids[:replace_from] + replacement_ids`` is enforced at
construction so an inconsistent update cannot be built, only refused.

This module is dependency-free (standard library only).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["SessionUpdate"]


@dataclass(frozen=True)
class SessionUpdate:
    """One append's effect on the core token stream (immutable).

    ``replace_from`` is a zero-based token index into the stream held
    before the append; tokens at indices ``>= replace_from`` were
    replaced. A full re-encode reports ``replace_from == 0``.
    """

    replace_from: int
    replacement_ids: Sequence[int]
    all_ids: Sequence[int]

    def __post_init__(self) -> None:
        replacement = tuple(self.replacement_ids)
        everything = tuple(self.all_ids)
        object.__setattr__(self, "replacement_ids", replacement)
        object.__setattr__(self, "all_ids", everything)
        if not 0 <= self.replace_from <= len(everything):
            raise ValueError(
                "replace_from must index into the resulting stream"
            )
        if everything[self.replace_from :] != replacement:
            raise ValueError(
                "all_ids must equal the kept prefix plus replacement_ids"
            )
