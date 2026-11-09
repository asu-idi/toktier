"""What ``add_special_tokens=True`` adds, read from the reference document.

``add_special_tokens=True`` means the post-processor of the document the
reference engine executes (the loader face): the facade answers with
exactly the ids that document's post-processor produces around the core
stream. This module classifies that post-processor into one of three
kinds, as a pure function of the document:

``identity``
    The post-processor adds no token to a single sequence (absent,
    ``ByteLevel``, a template whose single form is the sequence alone,
    or a ``Sequence`` of those). ``True`` returns the core stream.
``prefix``
    The single form is one special token followed by the sequence, the
    special token carrying exactly one id. ``True`` returns that id
    followed by the core stream.
``reference``
    Anything else (a suffix, both sides, several special tokens, the
    BERT and RoBERTa processors, unknown or nested shapes, truncation or
    padding). ``True`` runs on the reference engine as before.

For the first two kinds the reference computes ``prefix_ids ++ core``
for every valid Unicode scalar string: the post-processing step is
truncation (absent), then the processor, then padding (absent);
``ByteLevel`` rewrites offsets only; a template emits the special
token's ids for a special-token piece and the core encoding for the
sequence piece. A ``Sequence`` is admitted only with at most one
template member and every other member ``ByteLevel``, so the template
receives exactly one encoding and its pair form never runs.

The classification is then checked once at construction on the
reference engine itself (:func:`self_check`): a handful of probes,
including an added token alone and inside text and a non-ASCII string,
must satisfy ``encode(p, True) == prefix_ids + encode(p, False)``. Any
difference, or any error while checking, demotes the kind to
``reference``, which can only cost speed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "KIND_IDENTITY",
    "KIND_PREFIX",
    "KIND_REFERENCE",
    "SpecialTokenPlan",
    "classify_postprocessor",
    "plan_special_tokens",
    "self_check",
    "self_check_probes",
]

KIND_IDENTITY = "identity"
KIND_PREFIX = "prefix"
KIND_REFERENCE = "reference"

#: Where ``add_special_tokens=True`` is served, per kind (``explain()``).
SERVED_BY = {
    KIND_IDENTITY: "core_stream",
    KIND_PREFIX: "facade_prefix",
    KIND_REFERENCE: "reference",
}

_U32_MAX = 2**32 - 1

#: Probes every face is checked on, beside its own added-token probes.
#: The same probe set the Rust facade checks (``crates/toktier``), so the
#: two faces classify alike: three ASCII texts, one non-ASCII text, and the
#: face's first added token alone and inside text.
_FIXED_PROBES = (
    "",
    "a",
    "hello world",
    "caf\u00e9 \u4e2d\u6587 \U0001f600 na\u00efve",
)


def _piece(value: object, kind: str) -> Mapping[str, object] | None:
    """The body of a one-key template piece of ``kind``, or ``None``."""
    if not isinstance(value, Mapping) or len(value) != 1 or kind not in value:
        return None
    body = value[kind]
    if not isinstance(body, Mapping) or body.get("type_id") != 0:
        return None
    return body


def _is_sequence_a(value: object) -> bool:
    body = _piece(value, "Sequence")
    return body is not None and body.get("id") == "A"


def _special_token_ids(
    processor: Mapping[str, object], piece: object
) -> list[int] | None:
    """Ids of a special-token piece, looked up by the piece's own id."""
    body = _piece(piece, "SpecialToken")
    if body is None:
        return None
    key = body.get("id")
    table = processor.get("special_tokens")
    if not isinstance(key, str) or not isinstance(table, Mapping):
        return None
    entry = table.get(key)
    if not isinstance(entry, Mapping) or entry.get("id") != key:
        return None
    ids = entry.get("ids")
    if (
        not isinstance(ids, list)
        or len(ids) != 1
        or type(ids[0]) is not int
        or not 0 <= ids[0] <= _U32_MAX
    ):
        return None
    return [ids[0]]


def _classify_template(processor: Mapping[str, object]) -> tuple[str, list[int]]:
    single = processor.get("single")
    if not isinstance(single, list):
        return KIND_REFERENCE, []
    if len(single) == 1 and _is_sequence_a(single[0]):
        return KIND_IDENTITY, []
    if len(single) == 2 and _is_sequence_a(single[1]):
        ids = _special_token_ids(processor, single[0])
        if ids is not None:
            return KIND_PREFIX, ids
    return KIND_REFERENCE, []


def classify_postprocessor(document: Mapping[str, object]) -> tuple[str, list[int]]:
    """``(kind, prefix_ids)`` for one tokenizer document.

    A pure function of the document; :func:`self_check` is the separate
    construction-time witness on the engine that executes it.
    """
    for section in ("truncation", "padding"):
        if document.get(section) is not None:
            return KIND_REFERENCE, []
    processor = document.get("post_processor")
    if processor is None:
        return KIND_IDENTITY, []
    if not isinstance(processor, Mapping):
        return KIND_REFERENCE, []
    kind = processor.get("type")
    if kind == "ByteLevel":
        return KIND_IDENTITY, []
    if kind == "TemplateProcessing":
        return _classify_template(processor)
    if kind == "Sequence":
        members = processor.get("processors")
        if not isinstance(members, list):
            return KIND_REFERENCE, []
        types = [
            member.get("type") if isinstance(member, Mapping) else None
            for member in members
        ]
        templates = [
            member
            for member, member_type in zip(members, types, strict=True)
            if member_type == "TemplateProcessing"
        ]
        if len(templates) > 1 or any(
            member_type not in ("ByteLevel", "TemplateProcessing")
            for member_type in types
        ):
            return KIND_REFERENCE, []
        if not templates:
            return KIND_IDENTITY, []
        return _classify_template(templates[0])
    return KIND_REFERENCE, []


def self_check_probes(document: Mapping[str, object]) -> list[str]:
    """The fixed probes plus the face's first added token, alone and in text."""
    probes = list(_FIXED_PROBES)
    rows = document.get("added_tokens") or []
    contents = (
        [
            row["content"]
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("content"), str)
        ]
        if isinstance(rows, list)
        else []
    )
    first = next((str(content) for content in contents if content), None)
    if first is not None:
        probes.extend((first, f"a {first} b"))
    return probes


@dataclass(frozen=True)
class SpecialTokenPlan:
    """The facade's fixed answer for ``add_special_tokens=True``."""

    kind: str
    prefix_ids: tuple[int, ...] = ()
    #: The kind the document alone gave, before the self-check.
    document_kind: str = KIND_REFERENCE
    self_check: Mapping[str, object] = field(default_factory=dict)

    @property
    def served_by(self) -> str:
        return SERVED_BY[self.kind]

    @property
    def facade_served(self) -> bool:
        """Whether ``True`` is answered by the facade over the core stream."""
        return self.kind != KIND_REFERENCE

    def report(self) -> dict[str, object]:
        return {
            "postprocessor_class": self.kind,
            "document_class": self.document_kind,
            "prefix_ids": list(self.prefix_ids),
            "served_by": self.served_by,
            "self_check": dict(self.self_check),
        }


def self_check(
    encode: Callable[[str, bool], Sequence[int]],
    prefix_ids: Sequence[int],
    probes: Sequence[str],
) -> dict[str, object]:
    """Evaluate ``encode(p, True) == prefix_ids + encode(p, False)``.

    ``encode`` is the reference engine of the document being classified.
    Returns ``{"passed": bool, "probes": n}`` and, on a failure, the
    first failing probe index and the error type when one was raised.
    """
    prefix = [int(value) for value in prefix_ids]
    for index, probe in enumerate(probes):
        try:
            with_special = [int(value) for value in encode(probe, True)]
            core = [int(value) for value in encode(probe, False)]
        except Exception as error:  # any failure demotes the kind
            return {
                "passed": False,
                "probes": len(probes),
                "failed_probe": index,
                "error": type(error).__name__,
            }
        if with_special != prefix + core:
            return {
                "passed": False,
                "probes": len(probes),
                "failed_probe": index,
                "message": f"probe {index} ({len(probe.encode('utf-8'))} bytes) "
                "answered otherwise than the class on the reference engine",
            }
    return {"passed": True, "probes": len(probes)}


def plan_special_tokens(
    document: Mapping[str, object],
    encode: Callable[[str, bool], Sequence[int]],
) -> SpecialTokenPlan:
    """Classify ``document`` and confirm the kind on its reference engine."""
    kind, prefix_ids = classify_postprocessor(document)
    if kind == KIND_REFERENCE:
        return SpecialTokenPlan(
            kind=KIND_REFERENCE,
            document_kind=KIND_REFERENCE,
            self_check={"passed": None, "probes": 0},
        )
    check = self_check(encode, prefix_ids, self_check_probes(document))
    if check["passed"] is not True:
        return SpecialTokenPlan(
            kind=KIND_REFERENCE, document_kind=kind, self_check=check
        )
    return SpecialTokenPlan(
        kind=kind,
        prefix_ids=tuple(prefix_ids),
        document_kind=kind,
        self_check=check,
    )
