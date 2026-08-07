"""Routing policy enum semantics and the immutable route plan.

Acceptance surface: the frozen enum of ``docs/contracts/routing.md``
(four policies, the ``auto`` alias for ``CERTIFIED``, the append-only
``R_*`` namespace) and the ``RoutePlan`` value object.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from toktier.policy import (
    BACKEND_GPU,
    BACKEND_REFERENCE,
    PlanReason,
    ReasonCode,
    RoutePlan,
    RoutingPolicy,
)

CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "contracts" / "routing.md"

_REASON_ROW = re.compile(r"^\| `(R_[A-Z0-9_]+)` \|")
_POLICY_ROW = re.compile(r"^\| `([A-Z][A-Z_]*)`(?: \(default\))? \|")


def contract_text() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def contract_policies() -> set[str]:
    text = contract_text()
    start = text.index("## 1. `RoutingPolicy`")
    end = text.index("## 2. Three phases")
    return {
        match.group(1)
        for line in text[start:end].splitlines()
        if (match := _POLICY_ROW.match(line)) is not None
    }


def contract_reason_codes() -> set[str]:
    return {
        match.group(1)
        for line in contract_text().splitlines()
        if (match := _REASON_ROW.match(line)) is not None
    }


def test_policy_members_match_the_contract_document() -> None:
    assert contract_policies() == {member.name for member in RoutingPolicy}


def test_reason_codes_match_the_contract_document() -> None:
    documented = contract_reason_codes()

    assert documented, "no reason code rows found in the contract document"
    assert documented == {member.name for member in ReasonCode}
    for member in ReasonCode:
        assert member.value == member.name


@pytest.mark.parametrize("spelling", ["auto", "AUTO", " Auto "])
def test_auto_is_an_alias_for_certified(spelling: str) -> None:
    assert RoutingPolicy(spelling) is RoutingPolicy.CERTIFIED
    assert RoutingPolicy.coerce(spelling) is RoutingPolicy.CERTIFIED


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("certified", RoutingPolicy.CERTIFIED),
        ("reference", RoutingPolicy.REFERENCE),
        ("require_accelerated", RoutingPolicy.REQUIRE_ACCELERATED),
        ("experimental", RoutingPolicy.EXPERIMENTAL),
        ("REFERENCE", RoutingPolicy.REFERENCE),
    ],
)
def test_policy_values_parse(spelling: str, expected: RoutingPolicy) -> None:
    assert RoutingPolicy(spelling) is expected
    assert RoutingPolicy.coerce(spelling) is expected


def test_coerce_passes_a_policy_through() -> None:
    assert RoutingPolicy.coerce(RoutingPolicy.EXPERIMENTAL) is (
        RoutingPolicy.EXPERIMENTAL
    )


@pytest.mark.parametrize("spelling", ["fastest", "", "certifed"])
def test_unknown_policy_names_are_rejected(spelling: str) -> None:
    with pytest.raises(ValueError):
        RoutingPolicy(spelling)


def test_route_plan_records_the_chain_and_reasons() -> None:
    plan = RoutePlan(
        policy=RoutingPolicy.CERTIFIED,
        backend=BACKEND_REFERENCE,
        fallback_chain=(BACKEND_REFERENCE,),
        reasons=(
            PlanReason(
                code=ReasonCode.R_UNCERTIFIED_ARTIFACT,
                backend=BACKEND_GPU,
                detail={"artifact_sha256": "0" * 64},
            ),
        ),
    )

    assert plan.backend == BACKEND_REFERENCE
    assert plan.reason_codes() == (ReasonCode.R_UNCERTIFIED_ARTIFACT,)
    assert plan.reasons[0].detail["artifact_sha256"] == "0" * 64


def test_route_plan_is_immutable() -> None:
    plan = RoutePlan(
        policy=RoutingPolicy.REFERENCE,
        backend=BACKEND_REFERENCE,
        fallback_chain=(BACKEND_REFERENCE,),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.backend = BACKEND_GPU  # type: ignore[misc]


def test_plan_reason_detail_is_sealed() -> None:
    detail = {"sm": "sm_90"}
    reason = PlanReason(
        code=ReasonCode.R_SM_UNCERTIFIED, backend=BACKEND_GPU, detail=detail
    )

    detail["sm"] = "sm_80"
    assert reason.detail["sm"] == "sm_90"
    with pytest.raises(TypeError):
        reason.detail["sm"] = "sm_80"  # type: ignore[index]


def test_fallback_chain_must_end_with_the_reference_backend() -> None:
    with pytest.raises(ValueError):
        RoutePlan(
            policy=RoutingPolicy.CERTIFIED,
            backend=BACKEND_GPU,
            fallback_chain=(BACKEND_GPU,),
        )


def test_fallback_chain_must_not_be_empty() -> None:
    with pytest.raises(ValueError):
        RoutePlan(
            policy=RoutingPolicy.CERTIFIED,
            backend=BACKEND_REFERENCE,
            fallback_chain=(),
        )


def test_selected_backend_must_be_in_the_chain() -> None:
    with pytest.raises(ValueError):
        RoutePlan(
            policy=RoutingPolicy.CERTIFIED,
            backend=BACKEND_GPU,
            fallback_chain=(BACKEND_REFERENCE,),
        )


def test_selected_backend_must_head_the_chain() -> None:
    """A plan executes chain[0]; it must be the backend it reports."""
    with pytest.raises(ValueError):
        RoutePlan(
            policy=RoutingPolicy.CERTIFIED,
            backend=BACKEND_REFERENCE,
            fallback_chain=(BACKEND_GPU, BACKEND_REFERENCE),
        )


def test_fallback_chain_must_not_repeat_a_backend() -> None:
    with pytest.raises(ValueError):
        RoutePlan(
            policy=RoutingPolicy.CERTIFIED,
            backend=BACKEND_GPU,
            fallback_chain=(BACKEND_GPU, BACKEND_GPU, BACKEND_REFERENCE),
        )
    with pytest.raises(ValueError):
        RoutePlan(
            policy=RoutingPolicy.CERTIFIED,
            backend=BACKEND_REFERENCE,
            fallback_chain=(BACKEND_REFERENCE, BACKEND_REFERENCE),
        )
