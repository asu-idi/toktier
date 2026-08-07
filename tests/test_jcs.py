"""RFC 8785 and reference JCS vectors."""

from __future__ import annotations

import json
import struct
from typing import cast

import pytest

from toktier._jcs import CanonicalizationError, canonical_json


def _binary64(hexadecimal: str) -> float:
    return cast(float, struct.unpack(">d", bytes.fromhex(hexadecimal))[0])


def test_reference_array_vector() -> None:
    value = [56, {"d": True, "10": None, "1": []}]

    assert canonical_json(value) == b'[56,{"1":[],"10":null,"d":true}]'


def test_rfc_8785_literals_numbers_and_string_example() -> None:
    value = json.loads(
        r'''{
          "numbers": [333333333.33333329, 1E30, 4.50,
                      2e-3, 0.000000000000000000000000001],
          "string": "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/",
          "literals": [null, true, false]
        }'''
    )
    expected = (
        r'''{"literals":[null,true,false],"numbers":'''
        r'''[333333333.3333333,1e+30,4.5,0.002,1e-27],'''
        r'''"string":"€$\u000f\nA'B\"\\\\\"/"}'''
    ).encode()

    assert canonical_json(value) == expected


@pytest.mark.parametrize(
    ("ieee754", "expected"),
    [
        ("0000000000000000", b"0"),
        ("8000000000000000", b"0"),
        ("0000000000000001", b"5e-324"),
        ("8000000000000001", b"-5e-324"),
        ("7fefffffffffffff", b"1.7976931348623157e+308"),
        ("ffefffffffffffff", b"-1.7976931348623157e+308"),
        ("4340000000000000", b"9007199254740992"),
        ("c340000000000000", b"-9007199254740992"),
        ("4430000000000000", b"295147905179352830000"),
        ("7fffffffffffffff", None),
        ("7ff0000000000000", None),
        ("44b52d02c7e14af5", b"9.999999999999997e+22"),
        ("44b52d02c7e14af6", b"1e+23"),
        ("44b52d02c7e14af7", b"1.0000000000000001e+23"),
        ("444b1ae4d6e2ef4e", b"999999999999999700000"),
        ("444b1ae4d6e2ef4f", b"999999999999999900000"),
        ("444b1ae4d6e2ef50", b"1e+21"),
        ("3eb0c6f7a0b5ed8c", b"9.999999999999997e-7"),
        ("3eb0c6f7a0b5ed8d", b"0.000001"),
        ("41b3de4355555553", b"333333333.3333332"),
        ("41b3de4355555554", b"333333333.33333325"),
        ("41b3de4355555555", b"333333333.3333333"),
        ("41b3de4355555556", b"333333333.3333334"),
        ("41b3de4355555557", b"333333333.33333343"),
        ("becbf647612f3696", b"-0.0000033333333333333333"),
        ("43143ff3c1cb0959", b"1424953923781206.2"),
    ],
)
def test_rfc_8785_appendix_b_number_serialization(
    ieee754: str, expected: bytes | None
) -> None:
    value = _binary64(ieee754)
    if expected is None:
        with pytest.raises(CanonicalizationError, match="NaN and Infinity"):
            canonical_json(value)
    else:
        assert canonical_json(value) == expected

    if ieee754 == "4430000000000000":
        assert canonical_json(2**68) == expected


def test_rfc_8785_utf16_property_sorting() -> None:
    value = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "😀": "Emoji: Grinning Face",
        "\u0080": "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }
    expected = (
        '{"\\r":"Carriage Return","1":"One","'
        "\u0080"
        '":"Control","ö":"Latin Small Letter O With Diaeresis",'
        '"€":"Euro Sign","😀":"Emoji: Grinning Face",'
        '"דּ":"Hebrew Letter Dalet With Dagesh"}'
    ).encode()

    assert canonical_json(value) == expected


def test_reference_french_and_unicode_vectors() -> None:
    french = {
        "sin": "ignore locale",
        "pêche": "but canonicalization MUST",
        "péché": "is wrong according to French",
        "peach": "This sorting order",
    }
    assert canonical_json(french) == (
        '{"peach":"This sorting order",'
        '"péché":"is wrong according to French",'
        '"pêche":"but canonicalization MUST","sin":"ignore locale"}'
    ).encode()
    assert canonical_json({"Unnormalized Unicode": "A\u030a"}) == (
        '{"Unnormalized Unicode":"A\u030a"}'.encode()
    )


def test_rfc_8785_rejects_lone_surrogates() -> None:
    for value in ("\ud800", "\udead", {"\udfff": "object key"}):
        with pytest.raises(CanonicalizationError, match="lone surrogate"):
            canonical_json(value)


def test_canonical_json_round_trip_properties() -> None:
    values = [
        None,
        True,
        False,
        0,
        -17,
        10**20,
        10**21,
        -0.0,
        0.5,
        1e-7,
        "€\x00\n😀",
        {"nested": [None, True, 4.5, {"é": "A\u030a"}]},
    ]

    for value in values:
        encoded = canonical_json(value)
        decoded = json.loads(encoded)
        assert decoded == value
        assert canonical_json(decoded) == encoded
