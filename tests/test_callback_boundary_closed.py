"""A callback encoder certifies a session boundary only through the class
table and the seal rule.

``CallbackEncoder`` once accepted a Python ``boundary_cb`` and, when it was
built without the frozen class table, returned whatever cut that callback
named, with neither the class predicate nor ``SealCutRule`` applied. No
production caller passes one (the facade's store passes ``None``). The path
is closed: such an encoder certifies no boundary, so the store never seals
on its word, and the callback is never asked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

native = pytest.importorskip("toktier._native")

FILL = "The committee met on Tuesday and agreed to the plan for the coming year. "


def _encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    # One token per character: every character boundary is a token boundary.
    return [ord(c) % 50_000 for c in text], [(i, i + 1) for i in range(len(text))]


def test_a_boundary_callback_alone_never_seals(tmp_path: Path) -> None:
    from toktier import records
    from toktier.facade.store import EntryStore

    asked: list[tuple[int, int]] = []

    def boundary(
        text: str, ids: list[int], spans: object, floor: int, ceil: int
    ) -> tuple[int, int] | None:
        asked.append((floor, ceil))
        cut = max(floor, ceil - 1)
        return (cut, cut) if cut > 0 else None

    witness = native.WITNESS_BPE_SYNC_TRANSITION
    encoder = native.CallbackEncoder(witness, _encode, None, boundary)
    store = EntryStore(
        fingerprint=bytes(range(32)),
        encode=_encode,
        certified_bpe_witness=True,
        native_encoder=encoder,
        seal_end_guard_chars=8,
        directory=tmp_path,
    )
    text = FILL * 80
    for _ in range(4):
        text += FILL * 20
        assert store.encode_session("s", text) == _encode(text)[0]
    (record,) = (tmp_path / "entries").glob("*.rec")
    assert records.decode_record(record.read_bytes()).stable_prefix_byte_length == 0
    assert asked == []
