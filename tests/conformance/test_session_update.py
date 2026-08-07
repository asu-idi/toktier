"""Property-style checks for the frozen SessionUpdate splice invariant."""

from __future__ import annotations

from typing import Any


def test_session_update_splice_invariant_over_constructed_instances(
    installed_package: Any,
) -> None:
    observed = installed_package.json_output(
        """
import json
import random
import toktier

random_source = random.Random(0x70C71E)
for _ in range(128):
    old_ids = tuple(
        random_source.randrange(0, 2**32)
        for _ in range(random_source.randrange(0, 33))
    )
    replace_from = random_source.randrange(0, len(old_ids) + 1)
    replacement_ids = tuple(
        random_source.randrange(0, 2**32)
        for _ in range(random_source.randrange(0, 17))
    )
    expected_all_ids = old_ids[:replace_from] + replacement_ids
    update = toktier.SessionUpdate(
        replace_from=replace_from,
        replacement_ids=replacement_ids,
        all_ids=expected_all_ids,
    )
    assert update.replace_from == replace_from
    assert list(update.replacement_ids) == list(replacement_ids)
    assert list(update.all_ids) == (
        list(old_ids[: update.replace_from]) + list(update.replacement_ids)
    )

inconsistent_refused = False
try:
    toktier.SessionUpdate(replace_from=0, replacement_ids=(1,), all_ids=(2,))
except ValueError:
    inconsistent_refused = True
print(json.dumps({
    "available": True,
    "inconsistent_refused": inconsistent_refused,
}))
"""
    )
    assert observed == {"available": True, "inconsistent_refused": True}
