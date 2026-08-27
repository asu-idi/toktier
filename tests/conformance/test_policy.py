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
    "default_is_supported": default is RoutingPolicy.SUPPORTED,
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


def test_routing_policy_has_exactly_the_five_frozen_members(
    policy_observation: dict[str, object],
) -> None:
    """0.2.6 appends `SUPPORTED` to the four values v1 froze.

    `docs/contracts/routing.md` section 1.1 records the addition and says
    the `(default)` marker in the v1 table names what v1 shipped.
    """

    expected_members = sorted(
        {
            "CERTIFIED",
            "REFERENCE",
            "REQUIRE_ACCELERATED",
            "EXPERIMENTAL",
            "SUPPORTED",
        }
    )
    assert policy_observation["iterated_members"] == expected_members
    assert policy_observation["declared_members"] == expected_members


def test_supported_is_the_default_routing_policy(
    policy_observation: dict[str, object],
) -> None:
    """The default moved from `CERTIFIED` to `SUPPORTED` in 0.2.6."""

    assert policy_observation["default_is_supported"] is True
    assert policy_observation["default_is_certified"] is False
    assert policy_observation["default_is_experimental"] is False


def test_auto_tier_is_an_alias_for_certified(
    policy_observation: dict[str, object],
) -> None:
    assert policy_observation["auto_is_certified"] is True
