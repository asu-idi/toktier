"""Routing policy, reason codes, and the immutable route plan.

Contract reference: ``docs/contracts/routing.md``. The enums and the
``RoutePlan`` shape defined here are frozen public contract; reason
codes are append-only. Guiding rule: we prefer a miss over a wrong
result, and an uncertified configuration runs as reference.

This module is dependency-free (standard library only).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = [
    "BACKEND_FASTOKENS",
    "BACKEND_FAST_CPU",
    "BACKEND_GPU",
    "BACKEND_REFERENCE",
    "POLICY_ALIAS_AUTO",
    "PlanReason",
    "ReasonCode",
    "RoutePlan",
    "RoutingPolicy",
]

#: Backend identifier of the reference backend (pinned HF tokenizers
#: path). Always present; always last in every fallback chain.
BACKEND_REFERENCE: str = "hf"

#: Backend identifier of the CUDA kernel backend (JIT delivery in the
#: first release).
BACKEND_GPU: str = "gpu"

#: Corrected, evidence-bound Gigatoken build used by the certified CPU
#: full-encode and session repair paths.
BACKEND_FAST_CPU: str = "fast_cpu"

#: Fastokens adapter.  This identifier is deliberately outside the set of
#: backends a certified policy may select; it is reachable only through an
#: explicit EXPERIMENTAL request.
BACKEND_FASTOKENS: str = "fastokens"

#: Convenience alias accepted wherever a policy is named as a string
#: (``tier="auto"``); it selects :attr:`RoutingPolicy.CERTIFIED` and
#: introduces no additional behavior.
POLICY_ALIAS_AUTO: str = "auto"


class RoutingPolicy(enum.Enum):
    """The routing policies (frozen enum, appended to in 0.2.6).

    ``tier="auto"``, where accepted, is a convenience alias for
    :attr:`CERTIFIED`. The alias is resolved by :meth:`coerce`, and by
    the enum call form (``RoutingPolicy("auto")``) which delegates to
    it, so a constructor argument and a configuration value spell the
    alias the same way. The alias keeps pointing at :attr:`CERTIFIED`;
    the default is :attr:`SUPPORTED`, so asking for ``auto`` asks for
    the stricter of the two.
    """

    #: Default since 0.2.6. Everything :attr:`CERTIFIED` admits, and in
    #: addition a device architecture or compiler toolchain no
    #: certification campaign has judged, as long as the shipped kernel
    #: loads and runs there and every certified constraint verifies.
    #: Such a route is reported as ``supported_untested`` rather than as
    #: certified, and as ``locally_verified`` once a local check has
    #: compared it with the reference engine on that machine.
    SUPPORTED = "supported"

    #: Accelerated paths only where the support registry certifies them
    #: for this exact configuration; otherwise reference. The default
    #: through 0.2.5, and the way back to that behaviour since.
    CERTIFIED = "certified"

    #: Always run the reference backend; no accelerated code is planned
    #: or executed.
    REFERENCE = "reference"

    #: Like SUPPORTED, but raise the specific cause error at plan time
    #: if no accelerated path is eligible.
    REQUIRE_ACCELERATED = "require_accelerated"

    #: Permits uncertified accelerated paths. Outputs under this policy
    #: are not covered by the certification claims. Never the default.
    EXPERIMENTAL = "experimental"

    @classmethod
    def _missing_(cls, value: object) -> RoutingPolicy | None:
        """Resolve the ``auto`` alias and case-insensitive spellings.

        Returning ``None`` keeps the standard ``ValueError`` for values
        that name no policy.
        """
        if not isinstance(value, str):
            return None
        word = value.strip().lower()
        if word == POLICY_ALIAS_AUTO:
            return cls.CERTIFIED
        for member in cls:
            if member.value == word:
                return member
        return None

    def admits_unjudged_device(self) -> bool:
        """Whether this policy runs a device or toolchain nobody judged.

        The engines still have to be the judged ones and every bound
        digest still has to verify; what this waives is coverage, not
        integrity.
        """
        return self in (RoutingPolicy.SUPPORTED, RoutingPolicy.REQUIRE_ACCELERATED)

    @classmethod
    def coerce(cls, value: RoutingPolicy | str) -> RoutingPolicy:
        """Return the policy named by ``value``.

        Accepts a :class:`RoutingPolicy`, one of the frozen value
        strings, or the ``auto`` alias for :attr:`CERTIFIED`. Raises
        ``ValueError`` for anything else; callers that need a
        structured error wrap it (see :mod:`toktier.config`).
        """
        if isinstance(value, RoutingPolicy):
            return value
        return cls(value)


class ReasonCode(enum.Enum):
    """Fallback reason codes (``R_*``), frozen namespace, append-only.

    Consumers must tolerate unknown codes: switch on the codes you know
    and pass the rest through as opaque diagnostics.
    """

    # ---- plan-time reasons -------------------------------------------
    #: Policy is REFERENCE; accelerated options were not considered.
    R_POLICY_REFERENCE = "R_POLICY_REFERENCE"
    #: No eligible registry identity for this backend.
    R_UNCERTIFIED_ARTIFACT = "R_UNCERTIFIED_ARTIFACT"
    #: Installed oracle version is outside the certified set;
    #: acceleration off, reference still runs (reference-only state).
    R_ORACLE_MISMATCH = "R_ORACLE_MISMATCH"
    #: Required backend package/extra is not importable.
    R_BACKEND_UNAVAILABLE = "R_BACKEND_UNAVAILABLE"
    #: GPU use disabled by configuration.
    R_GPU_DISABLED = "R_GPU_DISABLED"
    #: A performed device probe found no usable CUDA device.
    R_NO_GPU_DETECTED = "R_NO_GPU_DETECTED"
    #: Device enumeration was not performed: the integrating caller
    #: supplied no device probe, i.e. it adopts no accelerator runtime
    #: on this path. Says nothing about the machine's hardware; distinct
    #: from ``R_NO_GPU_DETECTED``, which reports a probe that ran.
    R_ACCELERATOR_NOT_ADOPTED = "R_ACCELERATOR_NOT_ADOPTED"
    #: CUDA driver below the certified minimum.
    R_DRIVER_TOO_OLD = "R_DRIVER_TOO_OLD"
    #: Device architecture has no certified kernel entry.
    R_SM_UNCERTIFIED = "R_SM_UNCERTIFIED"
    #: Built or cached kernel digest does not match the registry-bound
    #: digest.
    R_KERNEL_DIGEST_MISMATCH = "R_KERNEL_DIGEST_MISMATCH"
    #: JIT kernel build attempted at load and failed.
    R_KERNEL_BUILD_FAILED = "R_KERNEL_BUILD_FAILED"
    #: The installed external engine differs from the version or native
    #: binary digest bound by the support registry.
    R_ENGINE_BINDING_MISMATCH = "R_ENGINE_BINDING_MISMATCH"

    # ---- run-time reasons --------------------------------------------
    #: Input contains an added-token literal; routed to the reference
    #: frontend path (part of the certified pipeline design).
    R_INPUT_ADDED_TOKEN = "R_INPUT_ADDED_TOKEN"
    #: Input is deliberately smaller than the configured GPU crossover;
    #: execution starts at the next eligible backend in the same plan.
    R_INPUT_BELOW_GPU_THRESHOLD = "R_INPUT_BELOW_GPU_THRESHOLD"
    #: A per-input guard premise on an accelerated path (a guarded
    #: fast-CPU input, or a state-seed closure/span premise) could not
    #: be proved; the input was routed to reference. Event detail always
    #: identifies the guard stage.
    R_INPUT_GUARD_ROUTED = "R_INPUT_GUARD_ROUTED"
    #: A session append found no certified safe cut point; the
    #: accumulated text was fully re-encoded.
    R_SESSION_NO_SAFE_CUT = "R_SESSION_NO_SAFE_CUT"
    #: An accelerated engine failed to open or execute; the input was
    #: re-run on the next backend in the fallback chain.
    R_EXEC_FAULT = "R_EXEC_FAULT"
    #: A core-stream-only accelerated backend was bypassed before execution
    #: because the request asked for postprocessing; routed to reference.
    R_INPUT_POSTPROCESS_ROUTED = "R_INPUT_POSTPROCESS_ROUTED"


@dataclass(frozen=True)
class PlanReason:
    """One recorded reason: a code plus optional structured detail."""

    code: ReasonCode
    #: The backend the reason speaks about (a backend identifier).
    backend: str
    #: Machine-readable facts (paths, digests, versions). Keys are
    #: informational; unknown keys must be tolerated.
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A reason is a value object: the detail mapping is copied and
        # sealed so that a recorded reason cannot be edited after the
        # fact by whoever handed the mapping in.
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))


@dataclass(frozen=True)
class RoutePlan:
    """Immutable routing plan (frozen shape).

    Produced by the pure planning function from a probe snapshot, the
    policy, the registry, and the configuration. A ``Tokenizer`` holds
    exactly one plan for its lifetime; environment changes after
    construction never alter an existing plan.
    """

    #: The policy the plan was computed under.
    policy: RoutingPolicy
    #: The selected backend identifier; always the head of the chain.
    backend: str
    #: Ordered fallback chain: starts with the selected backend, lists
    #: each backend once, and ends with the reference backend.
    fallback_chain: tuple[str, ...]
    #: Plan-time reasons: one entry per accelerated option considered
    #: and not selected.
    reasons: tuple[PlanReason, ...] = ()

    def __post_init__(self) -> None:
        chain = tuple(self.fallback_chain)
        if not chain:
            raise ValueError("fallback_chain must not be empty")
        if chain[-1] != BACKEND_REFERENCE:
            raise ValueError(
                "fallback_chain must end with the reference backend "
                f"{BACKEND_REFERENCE!r}"
            )
        if chain[0] != self.backend:
            raise ValueError(
                f"the selected backend {self.backend!r} must be the head of "
                f"the fallback chain {chain!r}: execution starts there, and "
                "a plan must execute the backend it reports"
            )
        if len(set(chain)) != len(chain):
            raise ValueError(
                f"fallback_chain {chain!r} must not repeat a backend"
            )
        # Uniqueness plus the terminal rule imply exactly one reference
        # entry, so the executor's walk always ends at the reference
        # backend and nowhere else.
        # Accept any ordered iterable of reasons and store a tuple, so a
        # plan handed a list is still an immutable value object.
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "fallback_chain", chain)

    def reason_codes(self) -> tuple[ReasonCode, ...]:
        """Codes of the recorded plan-time reasons, in order."""
        return tuple(reason.code for reason in self.reasons)
