"""A local check of an accelerated route against the reference engine.

Since 0.2.6 a device architecture or compiler toolchain no certification
campaign has judged runs under the default ``SUPPORTED`` policy and is
labelled ``supported_untested`` (``docs/contracts/routing.md`` Section
1.1). This module is the other half of that: a way for the person
running it to measure the combination on their own machine, on their own
text, and to have the answer remembered until something it depended on
changes.

What a record is not: a certificate. It says who compared what, on which
device, over how many documents; it never says ``certified``; it is
written by whoever ran the command; and it stops applying as soon as the
driver, the toolchain, the kernel, the source identity or the family
artifact moves. Nothing runs the check automatically -- a first-run
canary would be a default behaviour, and this release deliberately adds
none.

The documents are the caller's. ``--synthetic`` builds them from rules
rather than from a corpus, so no text of anyone else's travels with this
package and none is fetched to run a check.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .config import Config
from .errors import ToktierError
from .paths import ensure_private_dir

__all__ = [
    "FRAGMENTS",
    "RECORD_SCHEMA",
    "TOOL_VERSION",
    "Comparison",
    "VerificationKey",
    "VerificationRecord",
    "compare",
    "forget_record",
    "generate",
    "input_digest",
    "is_locally_verified",
    "read_record",
    "record_path",
    "split_documents",
    "verify_cache_dir",
    "write_record",
]

#: The record format. A record written under another schema is not read.
RECORD_SCHEMA = "toktier.local_verification.v1"

#: The version of the comparison itself. A change to what the check does
#: -- which cases it generates, what it compares -- retires the records
#: taken under the older one, because they answer a different question.
TOOL_VERSION = "1"

_KEY_DOMAIN = b"toktier.local_verification_key.v1\0"

#: This package's face of the check. The Rust binary keeps its own
#: records in the same directory under keys that name its own engines;
#: the field is here so a reader of a record can tell which face took it,
#: and so neither face can read the other's answer as its own.
_FACE = "python"


@dataclass(frozen=True)
class VerificationKey:
    """Everything a record is about.

    Any of it changing makes the record describe a different
    combination, so a record is filed under all of it and read back only
    when every field still matches. That is the whole of the expiry
    rule: nothing has to be swept, because a moved driver, toolchain,
    kernel, source identity or artifact is simply a different key.
    """

    #: ``gpu`` or ``cpu``: which accelerated route was compared.
    engine: str
    #: The family, and the exact artifact bytes that were tokenized.
    family: str
    artifact_sha256: str
    #: The device architecture and delivery, for a GPU route.
    architecture: str | None = None
    delivery: str | None = None
    #: The image that ran: the shipped fatbin's digest under prebuilt
    #: delivery, the kernel source digest under JIT.
    image_digest: str | None = None
    #: The generated class tables the kernel reads, and the flags a JIT
    #: product was built with: two more inputs to what actually ran.
    class_table_digest: str | None = None
    build_flags: tuple[str, ...] = ()
    #: The compiler the route was built with.
    toolchain: str | None = None
    #: The driver the device was opened through. It is an environment
    #: fact rather than a certificate premise, and a record still stops
    #: applying when it moves: the measurement was taken through it.
    driver_version: str | None = None
    #: The source identity of the host that selects and launches, and of
    #: the engine itself.
    host_source_digest: str | None = None
    engine_source_digest: str | None = None
    #: The repair configuration the CPU fast path is bound to.
    config_digest: str | None = None
    #: The version of the check itself.
    tool_version: str = TOOL_VERSION
    face: str = _FACE

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "artifact_sha256": self.artifact_sha256,
            "build_flags": list(self.build_flags),
            "class_table_digest": self.class_table_digest,
            "config_digest": self.config_digest,
            "delivery": self.delivery,
            "driver_version": self.driver_version,
            "engine": self.engine,
            "engine_source_digest": self.engine_source_digest,
            "face": self.face,
            "family": self.family,
            "host_source_digest": self.host_source_digest,
            "image_digest": self.image_digest,
            "tool_version": self.tool_version,
            "toolchain": self.toolchain,
        }

    def digest(self) -> str:
        """The name this key files its record under."""
        rendered = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(_KEY_DOMAIN + rendered).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> VerificationKey | None:
        if not isinstance(value, dict):
            return None
        try:
            return cls(
                engine=str(value["engine"]),
                family=str(value["family"]),
                artifact_sha256=str(value["artifact_sha256"]),
                architecture=_optional_str(value.get("architecture")),
                delivery=_optional_str(value.get("delivery")),
                image_digest=_optional_str(value.get("image_digest")),
                class_table_digest=_optional_str(
                    value.get("class_table_digest")
                ),
                build_flags=tuple(
                    str(flag) for flag in value.get("build_flags", ())
                ),
                toolchain=_optional_str(value.get("toolchain")),
                driver_version=_optional_str(value.get("driver_version")),
                host_source_digest=_optional_str(value.get("host_source_digest")),
                engine_source_digest=_optional_str(
                    value.get("engine_source_digest")
                ),
                config_digest=_optional_str(value.get("config_digest")),
                tool_version=str(value.get("tool_version", "")),
                face=str(value.get("face", "")),
            )
        except (KeyError, TypeError):
            return None


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


@dataclass(frozen=True)
class VerificationRecord:
    """What one local check found.

    A check that ran and disagreed leaves a record too. A reader is
    better served by "this was measured and it did not agree" than by
    silence, and the route keeps the label it would have had without any
    check: running the tool never makes a combination more restricted
    than not running it.
    """

    key: VerificationKey
    status: str
    documents: int
    bytes: int
    mismatches: int
    input: str
    input_digest: str
    taken_at: int
    first_mismatch: tuple[int, int] | None = None
    schema: str = RECORD_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "documents": self.documents,
            "first_mismatch": (
                None
                if self.first_mismatch is None
                else list(self.first_mismatch)
            ),
            "input": self.input,
            "input_digest": self.input_digest,
            "key": self.key.to_dict(),
            "mismatches": self.mismatches,
            "schema": self.schema,
            "status": self.status,
            "taken_at": self.taken_at,
        }

    @classmethod
    def from_dict(cls, value: object) -> VerificationRecord | None:
        if not isinstance(value, dict):
            return None
        key = VerificationKey.from_dict(value.get("key"))
        if key is None:
            return None
        first = value.get("first_mismatch")
        try:
            return cls(
                key=key,
                status=str(value["status"]),
                documents=int(value["documents"]),
                bytes=int(value["bytes"]),
                mismatches=int(value["mismatches"]),
                input=str(value["input"]),
                input_digest=str(value["input_digest"]),
                taken_at=int(value["taken_at"]),
                first_mismatch=(
                    (int(first[0]), int(first[1]))
                    if isinstance(first, list | tuple) and len(first) == 2
                    else None
                ),
                schema=str(value.get("schema", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None


def verify_cache_dir(config: Config) -> Path:
    """Where records live: beside the other caches, in a directory of
    their own, owner-only like the rest."""
    return config.cache_dir / "device-verify"


def record_path(config: Config, key: VerificationKey) -> Path:
    return verify_cache_dir(config) / f"{key.digest()}.json"


def read_record(
    config: Config, key: VerificationKey
) -> VerificationRecord | None:
    """The record for one combination, when one is on disk and still
    describes it.

    Reading never creates the directory: asking what was measured must
    not be an operation on the machine.
    """
    path = record_path(config, key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    record = VerificationRecord.from_dict(payload)
    if record is None or record.schema != RECORD_SCHEMA or record.key != key:
        return None
    return record


def is_locally_verified(config: Config, key: VerificationKey) -> bool:
    """Whether this combination carries a check that passed."""
    record = read_record(config, key)
    return record is not None and record.status == "passed"


def write_record(config: Config, record: VerificationRecord) -> Path:
    """Write one record, replacing whatever this combination had before."""
    directory = ensure_private_dir(verify_cache_dir(config))
    path = directory / f"{record.key.digest()}.json"
    text = json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    return path


def forget_record(config: Config, key: VerificationKey) -> bool:
    """Forget one combination's record. ``False`` when there was none."""
    path = record_path(config, key)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as error:  # pragma: no cover - filesystem specific
        raise ToktierError(
            f"the local verification record could not be removed: {error}",
            details={"path": str(path)},
        ) from error
    return True


# ---------------------------------------------------------------------
# The generated documents.
# ---------------------------------------------------------------------


class _Sequence:
    """A deterministic stream of bits from a seed, so the same command
    generates the same documents on any machine."""

    _MASK = (1 << 64) - 1
    _GAMMA = 0x9E3779B97F4A7C15

    def __init__(self, seed: int) -> None:
        self._state = (seed ^ self._GAMMA) & self._MASK

    def next(self) -> int:
        self._state = (self._state + self._GAMMA) & self._MASK
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return value ^ (value >> 31)

    def pick(self, choices: Sequence[str]) -> str:
        return choices[self.next() % len(choices)]


#: The fragments the generator assembles documents from.
#:
#: Every one is written here rather than taken from a corpus, and each is
#: a shape a fallback or a divergence has actually been found on:
#: variation selectors after an emoji or a Han character; combining marks
#: in non-canonical order; seams between scripts; newline and punctuation
#: seams; the whitespace variants regex dialects disagree about; long
#: repeats that cross a chunk boundary; and the three code points FINDING
#: 044 measured, which separate a Unicode 16 table from a Unicode 17 one.
FRAGMENTS: tuple[str, ...] = (
    "the quick brown fox jumps over the lazy dog",
    "Cargo builds, tests, and packages 148 crates.",
    '{"key": [1, 2.5, true, null], "nested": {"a": "b"}}',
    'fn main() {\n\tlet total = 0usize;\r\n\tprintln!("{total}");\n}',
    "0123456789 3.14159 -42 1e-9 0x2A",
    "-- ... --- ,,, ;;; ??? !!! (((())))",
    "\U0001f600\ufe0f\U0001f601\ufe0e emoji with selectors",
    "\u4e2d\u6587\ufe0f\u6df7\u6392mixed with ASCII",
    "e\u0301a\u034d\u08cb combining marks out of order",
    "spaces\u00a0and\u3000separators\u180ebetween words",
    "\U00010940\U00010941 sidetic letters after 16.0",
    "x\U000323b0\U000323b1y extension J han",
    "\u0295Bear \u0294Bear pharyngeal letters",
    "A" * 52,
    "\n\n   \t\t  \r\n \r\n",
    "https://example.invalid/path?query=1&other=2#fragment",
)

_SEPARATORS: tuple[str, ...] = (" ", "\n", "", "\t", ", ", "\r\n")


def generate(count: int, max_bytes: int, seed: int) -> list[str]:
    """``count`` documents of at most ``max_bytes`` each, from ``seed``.

    Rules only: no text of anyone else's travels with this package, and
    nothing is fetched.
    """
    sequence = _Sequence(seed)
    documents: list[str] = []
    for index in range(count):
        # Each document opens on a different fragment, so even a small
        # run reaches every shape above.
        document = FRAGMENTS[index % len(FRAGMENTS)]
        size = len(document.encode("utf-8"))
        while size < max_bytes:
            separator = sequence.pick(_SEPARATORS)
            fragment = sequence.pick(FRAGMENTS)
            added = len((separator + fragment).encode("utf-8"))
            if size + added > max_bytes:
                break
            document += separator + fragment
            size += added
        documents.append(document)
    return documents


def split_documents(text: str) -> list[str]:
    """One document per non-empty line, or the whole text as one when it
    holds no line break."""
    lines = [line for line in text.splitlines() if line]
    if not lines and text:
        return [text]
    return lines


def input_digest(documents: Sequence[str]) -> str:
    """A digest of the documents a record was taken over, so two runs on
    different text are told apart without the text being stored."""
    accumulator = hashlib.sha256()
    for document in documents:
        encoded = document.encode("utf-8")
        accumulator.update(len(encoded).to_bytes(8, "little"))
        accumulator.update(encoded)
    return accumulator.hexdigest()


# ---------------------------------------------------------------------
# The comparison.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """What comparing two engines over one set of documents found."""

    documents: int
    bytes: int
    mismatches: int
    first_mismatch: tuple[int, int] | None
    #: Documents the engine under test actually served. A route that
    #: silently fell back to the reference would otherwise compare the
    #: judge with itself and report agreement it never measured.
    served: int

    @property
    def passed(self) -> bool:
        return self.mismatches == 0 and self.served == self.documents


def compare(
    subject: object,
    judge: object,
    documents: Sequence[str],
    *,
    expected_backend: str,
) -> Comparison:
    """Encode every document on both tokenizers and compare id by id.

    ``subject`` and ``judge`` are open tokenizers; the comparison is the
    same criterion the certification battery uses -- every id of every
    document, in order. ``expected_backend`` is the backend the subject
    is supposed to be exercising, checked per document so a route that
    fell back mid-run is counted rather than mistaken for agreement.
    """
    encode_subject = subject.encode  # type: ignore[attr-defined]
    encode_judge = judge.encode  # type: ignore[attr-defined]
    explain_subject = subject.explain  # type: ignore[attr-defined]
    total_bytes = 0
    mismatches = 0
    served = 0
    first: tuple[int, int] | None = None
    for index, document in enumerate(documents):
        total_bytes += len(document.encode("utf-8"))
        mine = encode_subject(document, lookup="off").ids
        theirs = encode_judge(document, lookup="off").ids
        summary = explain_subject(summary=True)
        if summary.get("last_execution_backend") == expected_backend:
            served += 1
        if mine != theirs:
            mismatches += 1
            if first is None:
                shared = min(len(mine), len(theirs))
                position = next(
                    (
                        offset
                        for offset in range(shared)
                        if mine[offset] != theirs[offset]
                    ),
                    shared,
                )
                first = (index, position)
    return Comparison(
        documents=len(documents),
        bytes=total_bytes,
        mismatches=mismatches,
        first_mismatch=first,
        served=served,
    )


def record_for(
    key: VerificationKey,
    comparison: Comparison,
    *,
    documents: Sequence[str],
    source: str,
) -> VerificationRecord:
    """The record one comparison leaves behind."""
    return VerificationRecord(
        key=replace(key),
        status="passed" if comparison.passed else "failed",
        documents=comparison.documents,
        bytes=comparison.bytes,
        mismatches=comparison.mismatches,
        first_mismatch=comparison.first_mismatch,
        input=source,
        input_digest=input_digest(documents),
        taken_at=int(time.time()),
    )
