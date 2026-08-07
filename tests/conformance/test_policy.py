"""Conformance checks for the frozen routing-policy enum."""

from __future__ import annotations

from typing import Any

import pytest

_POLICY_PROBE = """
import json

from toktier import Config, RoutingPolicy

default = Config().routing_policy
print(json.dumps({
    "iterated_members": sorted(member.name for member in RoutingPolicy),
    "declared_members": sorted(RoutingPolicy.__members__),
    "default_is_certified": default is RoutingPolicy.CERTIFIED,
    "default_is_experimental": default is RoutingPolicy.EXPERIMENTAL,
    "auto_is_certified": RoutingPolicy("auto") is RoutingPolicy.CERTIFIED,
}))
"""


@pytest.fixture(scope="module")
def policy_observation(installed_package: Any) -> dict[str, object]:
    observed = installed_package.json_output(_POLICY_PROBE)
    assert isinstance(observed, dict)
    return observed


def test_routing_policy_has_exactly_the_four_frozen_members(
    policy_observation: dict[str, object],
) -> None:
    expected_members = sorted(
        {
            "CERTIFIED",
            "REFERENCE",
            "REQUIRE_ACCELERATED",
            "EXPERIMENTAL",
        }
    )
    assert policy_observation["iterated_members"] == expected_members
    assert policy_observation["declared_members"] == expected_members


def test_certified_is_the_default_routing_policy(
    policy_observation: dict[str, object],
) -> None:
    assert policy_observation["default_is_certified"] is True
    assert policy_observation["default_is_experimental"] is False


def test_auto_tier_is_an_alias_for_certified(
    policy_observation: dict[str, object],
) -> None:
    assert policy_observation["auto_is_certified"] is True
