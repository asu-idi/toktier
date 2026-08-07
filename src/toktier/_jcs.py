"""JSON Canonicalization Scheme (JCS) serialization.

This module implements RFC 8785 using only the Python standard library.  It is
the canonical JSON operation required by the frozen pipeline fingerprint in
``docs/contracts/fingerprint.md``.

RFC 8785: https://www.rfc-editor.org/rfc/rfc8785
"""

from __future__ import annotations

import math
from collections.abc import Mapping

__all__ = ["CanonicalizationError", "canonical_json"]

_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: '\\"',
    0x5C: "\\\\",
}


class CanonicalizationError(ValueError):
    """A value cannot be represented by RFC 8785 canonical JSON."""


def canonical_json(value: object) -> bytes:
    """Return the RFC 8785 canonical representation of *value* as UTF-8."""
    parts: list[str] = []
    _serialize(value, parts, "$")
    return "".join(parts).encode("utf-8")


def _serialize(value: object, parts: list[str], path: str) -> None:
    if value is None:
        parts.append("null")
    elif isinstance(value, bool):
        parts.append("true" if value else "false")
    elif isinstance(value, (int, float)):
        parts.append(_serialize_number(value, path))
    elif isinstance(value, str):
        parts.append(_quote(value, path))
    elif isinstance(value, (list, tuple)):
        parts.append("[")
        for index, item in enumerate(value):
            if index:
                parts.append(",")
            _serialize(item, parts, f"{path}[{index}]")
        parts.append("]")
    elif isinstance(value, Mapping):
        keys = list(value)
        for key in keys:
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"object key is not a string at {path}: {key!r}"
                )
            _validate_string(key, f"object key at {path}")
        keys.sort(key=lambda key: key.encode("utf-16-be"))

        parts.append("{")
        for index, key in enumerate(keys):
            if index:
                parts.append(",")
            parts.append(_quote(key, f"object key at {path}"))
            parts.append(":")
            _serialize(value[key], parts, f"{path}[{key!r}]")
        parts.append("}")
    else:
        raise CanonicalizationError(
            f"value of type {type(value).__name__} is not JSON-serializable at {path}"
        )


def _serialize_number(value: int | float, path: str) -> str:
    try:
        number = float(value)
    except OverflowError as error:
        raise CanonicalizationError(
            f"number is outside the IEEE 754 binary64 range at {path}"
        ) from error

    if not math.isfinite(number):
        raise CanonicalizationError(
            f"NaN and Infinity are not permitted by RFC 8785 at {path}: {number!r}"
        )
    if number == 0:
        return "0"

    sign = "-" if number < 0 else ""
    rendered = repr(abs(number))
    significand, exponent_marker, raw_exponent = rendered.partition("e")
    exponent = int(raw_exponent) if exponent_marker else 0
    exponent_text = f"e{exponent:+d}" if exponent_marker else ""

    first, dot, last = significand.partition(".")
    if last == "0":
        dot = ""
        last = ""

    # Python and ECMAScript select the same shortest decimal that round-trips
    # to the binary64 value.  Their display thresholds differ: ECMAScript uses
    # fixed notation from 1e-6 through values below 1e21.
    if exponent_marker and 0 < exponent < 21:
        digits = first + last
        decimal_point = exponent + 1
        if decimal_point < len(digits):
            first = digits[:decimal_point]
            dot = "."
            last = digits[decimal_point:]
        else:
            first = digits + "0" * (decimal_point - len(digits))
            dot = ""
            last = ""
        exponent_text = ""
    elif exponent_marker and -7 < exponent < 0:
        last = "0" * (-exponent - 1) + first + last
        first = "0"
        dot = "."
        exponent_text = ""

    return sign + first + dot + last + exponent_text


def _quote(value: str, path: str) -> str:
    _validate_string(value, path)
    parts = ['"']
    for character in value:
        code_point = ord(character)
        escape = _ESCAPES.get(code_point)
        if escape is not None:
            parts.append(escape)
        elif code_point < 0x20:
            parts.append(f"\\u{code_point:04x}")
        else:
            parts.append(character)
    parts.append('"')
    return "".join(parts)


def _validate_string(value: str, path: str) -> None:
    for character in value:
        code_point = ord(character)
        if 0xD800 <= code_point <= 0xDFFF:
            raise CanonicalizationError(
                f"lone surrogate U+{code_point:04X} is not valid Unicode at {path}"
            )
