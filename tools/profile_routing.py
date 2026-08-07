"""Profile the allocation-heavy v0.1 route gate against the Rust selector.

This is a control-plane microprofile, not a tokenizer throughput benchmark.
It models the common no-added-literal case: the former executor materialized
UTF-8 once for the size crossover and the Python frontend materialized and
translated it again for its necessary-condition gate. The native selector
borrows CPython's cached UTF-8 view and performs both decisions together.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from toktier import _native

_SIZES = (32, 1500, 65_536, 4_000_000)
_FIRST_BYTE = ord("<")
_PAIR_SECOND = ord("|")
_FIRST_GATE = bytes(1 if value == _FIRST_BYTE else 0 for value in range(256))


def _text(size: int) -> str:
    seed = "agent turn 123: plain payload without control literals. "
    return (seed * ((size + len(seed) - 1) // len(seed)))[:size]


def _legacy(text: str) -> tuple[int, bool]:
    """The two allocation sites present in the v0.1 no-hit route."""
    input_bytes = len(text.encode("utf-8"))
    raw = text.encode("utf-8")
    if b"\x01" not in raw.translate(_FIRST_GATE):
        return input_bytes, False
    return input_bytes, bytes((_FIRST_BYTE, _PAIR_SECOND)) in raw


def _measure(call: Callable[[], object], *, iterations: int, repeats: int) -> float:
    readings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _iteration in range(iterations):
            call()
        elapsed = time.perf_counter_ns() - started
        readings.append(elapsed / iterations / 1000.0)
    return statistics.median(readings)


def _iterations(size: int) -> int:
    if size <= 32:
        return 200_000
    if size <= 1500:
        return 100_000
    if size <= 65_536:
        return 5000
    return 100


def profile(*, repeats: int) -> dict[str, object]:
    selector = _native.RouteSelector(
        (65_536, 0, 0),
        2,
        True,
        2,
        ((_FIRST_BYTE, _PAIR_SECOND),),
    )
    rows: list[dict[str, object]] = []
    for size in _SIZES:
        text = _text(size)
        expected = _legacy(text)
        observed = selector.route(text)
        if expected[0] != observed[0] or expected[1] != observed[3]:
            raise RuntimeError(
                f"routing profile setup disagrees at {size}: {expected} != {observed}"
            )
        iterations = _iterations(size)
        legacy_us = _measure(
            partial(_legacy, text),
            iterations=iterations,
            repeats=repeats,
        )
        native_us = _measure(
            partial(selector.route, text),
            iterations=iterations,
            repeats=repeats,
        )
        rows.append(
            {
                "characters": len(text),
                "utf8_bytes": len(text.encode("utf-8")),
                "iterations_per_repeat": iterations,
                "legacy_us_p50": round(legacy_us, 6),
                "native_us_p50": round(native_us, 6),
                "speedup": round(legacy_us / native_us, 3),
            }
        )
    return {
        "schema": "toktier.routing_profile.v1",
        "caliber": "threshold plus no-hit added-token necessary-condition gate",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repeats": repeats,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    report = profile(repeats=arguments.repeats)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.json_output is not None:
        arguments.json_output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
