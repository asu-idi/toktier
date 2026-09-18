"""A tokenizer stand-in for the scheduler tests' worker processes.

One id per character, so any cut is exact and the concatenation of chunk
rows equals the whole; the session path remembers the last text per key
and reports through ``calls`` how it was reached. Loaded by name in the
spawned worker (``tests/integrations`` is put on its ``sys.path``).
"""

from __future__ import annotations

import os
from typing import Any


class _Ids:
    def __init__(self, ids: list[int]) -> None:
        self.ids = tuple(ids)


class StubTokenizer:
    family = "stub_family"

    def __init__(self, store: str | None) -> None:
        self.store = store
        self.sessions: dict[str, str] = {}
        self.calls = 0
        if store:
            os.makedirs(store, exist_ok=True)

    def _encode_ids(self, text: str, **kwargs: Any) -> list[int]:
        self.calls += 1
        if "boom" in text:
            raise ValueError("stub refuses this text")
        if "slow" in text:
            import time

            time.sleep(0.3)
        key = kwargs.get("session")
        if key is not None:
            self.sessions[key] = text
        return [ord(c) for c in text]

    def encode(self, text: str, **kwargs: Any) -> _Ids:
        return _Ids(self._encode_ids(text, **kwargs))

    def close(self) -> None:
        pass


def load(family: str | None, store: str | None, device: str) -> StubTokenizer:
    return StubTokenizer(store)


def load_failing(family: str | None, store: str | None, device: str) -> StubTokenizer:
    raise RuntimeError("this loader always fails")


def load_exiting(family: str | None, store: str | None, device: str) -> StubTokenizer:
    """Loads, then exits the process on the first 'die' text (dead-worker test)."""
    tokenizer = StubTokenizer(store)
    original = tokenizer._encode_ids

    def encode_ids(text: str, **kwargs: Any) -> list[int]:
        if "die" in text:
            os._exit(3)
        return original(text, **kwargs)

    tokenizer._encode_ids = encode_ids  # type: ignore[method-assign]
    return tokenizer
