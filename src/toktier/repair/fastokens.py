"""Explicit experimental full-reencode callback backed by Fastokens.

Two questions are kept apart here, and the report answers both:

* **Admission** -- how the adapter is selected. It is never automatic: the
  caller must request ``repair_backend="fastokens"`` together with the
  ``EXPERIMENTAL`` policy, and ``certification`` reads ``experimental`` for
  that reason alone. Nothing in this module changes that.
* **Assurance** -- what is known about the engine that is installed. The
  toktier project publishes a pinned build of Fastokens (``toktier-fastokens``)
  and took its readings on specific published wheels. The adapter resolves
  which bytes it is about to run, by the import package rather than by a
  distribution name, and reports ``engine_assurance``: ``certified_pinned``
  when the bytes are a published wheel listed in the shipped registry, the
  Unicode guard is active, the reference is the judged one and the family is
  in the evidence; otherwise one value naming the premise that does not hold.
  ``exact_id_guarantee`` is ``true`` only in the first case, and then in the
  guarded sense: ids equal the pinned reference, or the request was routed to
  that reference by the guard.

Every premise is a runtime comparison of bytes and versions against the
digest-verified registry; a missing registry node, an unknown digest or a
missing guard reads as the weaker state, never as the stronger one.

It re-encodes the complete session tail on every append and reports that fact.
If its token-byte spans cannot be reconstructed, or the guard fires, it
answers from the HF reference callback instead of committing malformed session
state.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from ..errors import BackendUnavailable
from ..policy import BACKEND_FASTOKENS
from .gigatoken import (
    ReferenceEncode,
    WindowUnsupported,
    _byte_lengths_from_hf,
    _spans_from_ids,
)
from .registry import RepairFamily

__all__ = [
    "ASSURANCE_CERTIFIED_PINNED",
    "ASSURANCE_FAMILY_OUTSIDE_EVIDENCE",
    "ASSURANCE_GUARD_DISABLED",
    "ASSURANCE_ORACLE_MISMATCH",
    "ASSURANCE_UNRECOGNIZED_BUILD",
    "ASSURANCE_UNVERIFIABLE",
    "ASSURANCE_UPSTREAM_BUILD",
    "CONFIG_ID",
    "GUARD_ID",
    "AssuranceReport",
    "DistributionOwner",
    "FastokensFullRepair",
    "FastokensIdentity",
    "assess",
    "compile_unicode_guard",
    "family_admitted",
    "fastokens_distribution_identity",
    "fastokens_identity",
    "pinned_engine_entry",
]

#: The configuration identity of this adapter. It is bound into the session
#: fingerprint, so a change of what the adapter does (v2 added the Unicode
#: guard) retires stored sessions from the earlier meaning.
CONFIG_ID = "toktier-fastokens-full-experimental-v2"
#: The guard the registry node carries the code-point set for.
GUARD_ID = "toktier-fastokens-unicode-skew-guard-v1"
IMPORT_NAME = "fastokens"
PINNED_DISTRIBUTION = "toktier-fastokens"
UPSTREAM_DISTRIBUTION = "fastokens"

ASSURANCE_CERTIFIED_PINNED = "certified_pinned"
ASSURANCE_UNRECOGNIZED_BUILD = "unrecognized_build"
ASSURANCE_UPSTREAM_BUILD = "upstream_build"
ASSURANCE_GUARD_DISABLED = "guard_disabled"
ASSURANCE_ORACLE_MISMATCH = "oracle_mismatch"
ASSURANCE_FAMILY_OUTSIDE_EVIDENCE = "family_outside_evidence"
ASSURANCE_UNVERIFIABLE = "unverifiable"

#: Domain of the engine digest: the ``fastokens/`` files, sorted by their
#: distribution-relative path, each bound by name and content hash. Unchanged
#: since v1 so digests measured earlier still compare.
_DIGEST_DOMAIN = b"toktier.fastokens.distribution.v1\0"
_GUARD_SET_DOMAIN = b"toktier.fastokens.unicode_guard.v1\0"

_REINSTALL_COMMAND = (
    'pip uninstall -y fastokens toktier-fastokens && pip install "toktier[fastokens]"'
)
_PUBLISHED_WHEEL_COMMAND = (
    'pip install --only-binary toktier-fastokens "toktier[fastokens]"'
)


# ---------------------------------------------------------------------------
# Identity: which bytes would be imported, and whose are they


@dataclass(frozen=True)
class DistributionOwner:
    """One installed distribution whose RECORD names ``fastokens/`` files."""

    name: str
    version: str
    #: Number of ``fastokens/`` entries in its RECORD.
    recorded: int
    #: Entries whose on-disk bytes equal the RECORD hash.
    matching: int
    #: Entries absent from disk.
    missing: int
    #: Directory the RECORD resolves the package to, or ``None`` when the
    #: files are gone.
    package_dir: Path | None

    @property
    def owns_bytes(self) -> bool:
        return self.recorded > 0 and self.matching == self.recorded

    @property
    def orphaned(self) -> bool:
        return self.recorded > 0 and self.missing == self.recorded

    @property
    def label(self) -> str:
        return f"{self.name} {self.version}"


@dataclass(frozen=True)
class FastokensIdentity:
    """What ``import fastokens`` would run, and which distribution it belongs to.

    ``package_dir`` is where the import system locates the package (``None``
    when it is not importable). ``engine_digest`` hashes the files in that
    directory. ``owners`` are the installed distributions whose RECORD names
    ``fastokens/`` files; ``owner`` is the one whose recorded bytes are the
    ones on disk, when exactly such a one exists. ``shadowed`` says the
    import location is not the directory any installed distribution recorded,
    so no metadata describes the bytes that would run.
    """

    package_dir: Path | None
    engine_digest: str | None
    owners: tuple[DistributionOwner, ...] = ()
    owner: DistributionOwner | None = None
    shadowed: bool = False
    hash_error: str | None = None

    @property
    def available(self) -> bool:
        return self.package_dir is not None

    @property
    def distribution(self) -> str | None:
        return self.owner.name if self.owner is not None else None

    @property
    def version(self) -> str | None:
        return self.owner.version if self.owner is not None else None

    @property
    def coinstalled(self) -> tuple[DistributionOwner, ...]:
        """Distributions other than the owner that also record these files.

        Empty when the import is shadowed: the recorded distribution is then
        not sharing files with another one, it is being bypassed, and the
        assurance reason names it together with the two locations.
        """
        if self.shadowed:
            return ()
        return tuple(
            candidate
            for candidate in self.owners
            if candidate is not self.owner and not candidate.orphaned
        )

    @property
    def orphaned(self) -> tuple[DistributionOwner, ...]:
        return tuple(candidate for candidate in self.owners if candidate.orphaned)

    @property
    def verifiable(self) -> bool:
        """The digest describes the bytes an installed distribution recorded."""
        return (
            self.package_dir is not None
            and self.engine_digest is not None
            and not self.shadowed
            and any(not candidate.orphaned for candidate in self.owners)
        )

    @property
    def imported_tree_matches_record(self) -> bool:
        return self.owner is not None and self.verifiable


def _relative_name(package_dir: Path, path: Path) -> str:
    return f"{IMPORT_NAME}/{path.relative_to(package_dir).as_posix()}"


def _hash_tree(package_dir: Path) -> str:
    """Engine digest of the files under ``package_dir`` (v1 domain)."""
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    files = sorted(
        (
            path
            for path in package_dir.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.relative_to(package_dir).parts
        ),
        key=lambda path: _relative_name(package_dir, path),
    )
    for path in files:
        name = _relative_name(package_dir, path).encode()
        digest.update(len(name).to_bytes(4, "little"))
        digest.update(name)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _record_entries(text: str) -> list[tuple[str, str]]:
    """``(path, sha256 hex)`` for the ``fastokens/`` files a RECORD names."""
    entries: list[tuple[str, str]] = []
    for row in csv.reader(text.splitlines()):
        if len(row) < 2 or not row[0]:
            continue
        parts = row[0].split("/")
        if parts[0] != IMPORT_NAME or "__pycache__" in parts or len(parts) < 2:
            continue
        algorithm, _, encoded = row[1].partition("=")
        if algorithm != "sha256" or not encoded:
            continue
        padded = encoded + "=" * (-len(encoded) % 4)
        try:
            raw = base64.urlsafe_b64decode(padded)
        except ValueError:
            continue
        entries.append((row[0], raw.hex()))
    return entries


def _owners() -> tuple[DistributionOwner, ...]:
    # Keep importlib.metadata off ``import toktier``: it pulls in the socket
    # module on CPython.  Probing an explicitly requested optional backend
    # may load it; importing the public package may not.
    from importlib.metadata import distributions

    found: list[DistributionOwner] = []
    for dist in distributions():
        try:
            record = dist.read_text("RECORD")
        except OSError:
            record = None
        if not record:
            continue
        entries = _record_entries(record)
        if not entries:
            continue
        matching = missing = 0
        package_dir: Path | None = None
        for relative, expected in entries:
            path = Path(str(dist.locate_file(relative)))
            try:
                observed = hashlib.sha256(path.read_bytes()).hexdigest()
            except FileNotFoundError:
                missing += 1
                continue
            except OSError:
                continue
            if package_dir is None:
                package_dir = path.parents[len(relative.split("/")) - 2]
            if observed == expected:
                matching += 1
        name = str(dist.metadata["Name"] or dist.name)
        found.append(
            DistributionOwner(
                name=name,
                version=str(dist.version),
                recorded=len(entries),
                matching=matching,
                missing=missing,
                package_dir=package_dir,
            )
        )
    found.sort(
        key=lambda owner: (owner.name != PINNED_DISTRIBUTION, owner.name.lower())
    )
    return tuple(found)


def _locate_package() -> Path | None:
    from importlib.util import find_spec

    try:
        spec = find_spec(IMPORT_NAME)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin and spec.origin != "namespace":
        return Path(spec.origin).resolve().parent
    locations = list(spec.submodule_search_locations or ())
    return Path(locations[0]).resolve() if locations else None


def fastokens_identity() -> FastokensIdentity:
    """Resolve the Fastokens engine by its import package, without importing it."""
    package_dir = _locate_package()
    owners = _owners()
    if package_dir is None:
        return FastokensIdentity(package_dir=None, engine_digest=None, owners=owners)
    try:
        engine_digest: str | None = _hash_tree(package_dir)
        hash_error: str | None = None
    except OSError as error:
        engine_digest = None
        hash_error = str(error)
    recorded_here = tuple(
        candidate
        for candidate in owners
        if candidate.package_dir is not None
        and candidate.package_dir.resolve() == package_dir
    )
    shadowed = any(not candidate.orphaned for candidate in owners) and not recorded_here
    owner: DistributionOwner | None = None
    full = [candidate for candidate in recorded_here if candidate.owns_bytes]
    if full:
        # Two RECORDs matching fully means byte-identical payloads; the
        # pinned distribution sorts first and is the one reported.
        owner = full[0]
    return FastokensIdentity(
        package_dir=package_dir,
        engine_digest=engine_digest,
        owners=owners,
        owner=owner,
        shadowed=shadowed,
        hash_error=hash_error,
    )


def fastokens_distribution_identity() -> tuple[str | None, str | None]:
    """Installed version and content digest, without importing Fastokens.

    Kept for callers of the 0.2.x signature. The version is the owning
    distribution's when one owns the bytes on disk; the digest is that of the
    package the import system would load.
    """
    identity = fastokens_identity()
    if not identity.available:
        orphans = identity.orphaned
        return (orphans[0].version if orphans else None), None
    return identity.version, identity.engine_digest


# ---------------------------------------------------------------------------
# Registry node, guard, and the assurance decision


@lru_cache(maxsize=1)
def _shipped_entry() -> dict[str, Any] | None:
    from ..routing.registry_load import load_registry_document
    from ..routing.tables import SUPPORT_REGISTRY

    document = load_registry_document(SUPPORT_REGISTRY)
    distributions = document.get("engine_distributions")
    if not isinstance(distributions, dict):
        return None
    entry = distributions.get(IMPORT_NAME)
    if not isinstance(entry, dict) or entry.get("backend") != BACKEND_FASTOKENS:
        return None
    return entry


def family_admitted(family: str, artifact_sha256: str | None) -> bool:
    """Whether the adapter can be opened for this family at all.

    A session names a family and an artifact, and the adapter runs the
    certified repair table's parameters for that exact pair, so a pair
    with no entry in that table is one the adapter refuses to open
    (``UnsupportedConfig``) rather than one it answers about. This is a
    narrower question than the evidence premise behind
    ``family_outside_evidence``: the readings cover fifteen families,
    the repair table reaches eleven, and the difference is the set this
    answer is about.
    """
    from .registry import family_spec

    if artifact_sha256 is None:
        return False
    return family_spec(family, artifact_sha256) is not None


def pinned_engine_entry() -> Mapping[str, Any] | None:
    """The ``engine_distributions.fastokens`` node of the shipped registry.

    Read from the digest-verified document; ``None`` when the registry
    carries no such node, which every consumer treats as "no published wheel
    is known" rather than as permission.
    """
    return _shipped_entry()


def _parse_codepoint(text: object) -> int | None:
    if not isinstance(text, str) or not text.startswith("U+") or len(text) < 6:
        return None
    try:
        value = int(text[2:], 16)
    except ValueError:
        return None
    return value if 0 <= value <= 0x10FFFF else None


def compile_unicode_guard(entry: Mapping[str, Any] | None) -> re.Pattern[str] | None:
    """The guard pattern from the registry node, or ``None`` when it cannot be.

    The set is checked against the digest the node records for it, so a node
    whose ranges and digest disagree yields no guard; the adapter then reports
    ``guard_disabled`` rather than running a set it cannot vouch for.
    """
    if entry is None:
        return None
    guard = entry.get("guard")
    if not isinstance(guard, Mapping) or guard.get("id") != GUARD_ID:
        return None
    ranges = guard.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        return None
    codepoints: list[int] = []
    pieces: list[str] = []
    previous = -1
    for pair in ranges:
        if not isinstance(pair, list) or len(pair) != 2:
            return None
        low, high = _parse_codepoint(pair[0]), _parse_codepoint(pair[1])
        if low is None or high is None or low > high or low <= previous:
            return None
        codepoints.extend(range(low, high + 1))
        piece = re.escape(chr(low))
        if high > low:
            piece += "-" + re.escape(chr(high))
        pieces.append(piece)
        previous = high
    if guard.get("codepoints") != len(codepoints):
        return None
    joined = "\n".join(f"U+{value:04X}" for value in codepoints).encode("ascii")
    digest = hashlib.sha256(_GUARD_SET_DOMAIN + joined).hexdigest()
    if digest != guard.get("set_sha256"):
        return None
    return re.compile("[" + "".join(pieces) + "]")


@dataclass(frozen=True)
class AssuranceReport:
    """The assurance decision and everything the report says about it."""

    assurance: str
    reason: str | None
    known_wheel: dict[str, str] | None
    guard_active: bool
    guard_codepoints: int
    basis: dict[str, Any] | None
    advisory: str | None
    distribution: str | None
    version: str | None
    engine_digest: str | None
    identity: FastokensIdentity | None = field(default=None, compare=False, repr=False)

    @property
    def exact_id_guarantee(self) -> bool:
        return self.assurance == ASSURANCE_CERTIFIED_PINNED


def _statement(entry: Mapping[str, Any], wheel: Mapping[str, Any]) -> str:
    evidence = entry.get("evidence") or {}
    oracle = entry.get("oracle") or {}
    gate3 = evidence.get("gate3") or {}
    return (
        "true means: for the families this evidence covers, the ids this "
        "adapter returns equal the pinned reference (tokenizers "
        f"{oracle.get('version')}), or the request was routed to that "
        "reference by the adapter's Unicode guard. The evidence was taken on "
        f"the wheel toktier published (engine digest {wheel.get('engine_digest')}), "
        f"at {evidence.get('visible_cpus')} visible CPUs for the large run and "
        f"across {gate3.get('topologies')} CPU topologies in the topology gate; "
        "other builds of the same source are not covered."
    )


def _basis(entry: Mapping[str, Any], wheel: Mapping[str, Any]) -> dict[str, Any]:
    evidence = dict(entry.get("evidence") or {})
    return {
        "statement": _statement(entry, wheel),
        "evidence_id": evidence.get("evidence_id"),
        "suite_version": evidence.get("suite_version"),
        "known_wheel": {
            "filename": str(wheel.get("filename")),
            "sha256": str(wheel.get("sha256")),
            "engine_digest": str(wheel.get("engine_digest")),
        },
        "oracle": dict(entry.get("oracle") or {}),
        "families": evidence.get("families"),
        "docs_per_family": evidence.get("docs_per_family"),
        "comparisons": evidence.get("comparisons"),
        "mismatch_guarded": evidence.get("mismatch_guarded"),
        "mismatch_raw": evidence.get("mismatch_raw"),
        "engine_error": evidence.get("engine_error"),
        "routed_reference_per_family": evidence.get("routed_reference_per_family"),
        "visible_cpus": evidence.get("visible_cpus"),
        "gate2": dict(evidence.get("gate2") or {}),
        "gate3": dict(evidence.get("gate3") or {}),
        "gate4": dict(evidence.get("gate4") or {}),
    }


def _advisory(identity: FastokensIdentity) -> str | None:
    others = identity.coinstalled
    if not others:
        return None
    named = ", ".join(f"'{other.label}'" for other in others)
    if identity.owner is not None:
        return (
            f"the distribution {named} is also installed and its RECORD names the "
            f"same files; the bytes on disk belong to {identity.owner.label}. "
            "Uninstalling either distribution removes the shared files; to keep "
            f"only the pinned build, run: {_REINSTALL_COMMAND}"
        )
    return (
        f"the distributions {named} are installed and their RECORDs name the same "
        "files; the bytes on disk match neither completely. Uninstalling either "
        "distribution removes the shared files; to keep only the pinned build, "
        f"run: {_REINSTALL_COMMAND}"
    )


def assess(
    identity: FastokensIdentity,
    *,
    entry: Mapping[str, Any] | None,
    guard: re.Pattern[str] | None,
    oracle_version: str | None,
    family: str | None,
    artifact_sha256: str | None,
) -> AssuranceReport:
    """Decide ``engine_assurance`` from machine-checkable premises only.

    ``family=None`` is the environment-level answer ``doctor`` gives, where
    the family premise is not applicable; the session report always names a
    family.
    """
    known: Mapping[str, Any] | None = None
    if entry is not None and identity.engine_digest is not None:
        for wheel in entry.get("known_wheels") or ():
            if not isinstance(wheel, Mapping):
                continue
            if wheel.get("engine_digest") == identity.engine_digest:
                known = wheel
                break
    guard_codepoints = 0
    if entry is not None and isinstance(entry.get("guard"), Mapping):
        guard_codepoints = int((entry["guard"]).get("codepoints") or 0)
    known_wheel = (
        {"filename": str(known["filename"]), "sha256": str(known["sha256"])}
        if known is not None
        else None
    )

    def report(
        assurance: str, reason: str | None, basis: dict[str, Any] | None
    ) -> AssuranceReport:
        return AssuranceReport(
            assurance=assurance,
            reason=reason,
            known_wheel=known_wheel,
            guard_active=guard is not None,
            guard_codepoints=guard_codepoints if guard is not None else 0,
            basis=basis,
            advisory=_advisory(identity),
            distribution=identity.distribution,
            version=identity.version,
            engine_digest=identity.engine_digest,
            identity=identity,
        )

    if not identity.verifiable:
        return report(
            ASSURANCE_UNVERIFIABLE,
            (
                "the bytes the import system would run are not the ones an "
                "installed distribution recorded, so the engine digest cannot be "
                "verified"
            ),
            None,
        )
    if known is None:
        if identity.owner is not None and identity.owner.name == UPSTREAM_DISTRIBUTION:
            return report(
            ASSURANCE_UPSTREAM_BUILD,
            (
                    "the installed engine is the upstream fastokens distribution; "
                    "toktier's readings were taken on its own pinned build and do "
                    "not carry over"
                ),
            None,
        )
        return report(
            ASSURANCE_UNRECOGNIZED_BUILD,
            (
                "the engine digest of the installed fastokens package is not among "
                "the wheels toktier published for this release (a wheel built on "
                "another host or toolchain usually differs), so the pinned readings "
                "do not apply; install the published wheel "
                f"({_PUBLISHED_WHEEL_COMMAND}) to run the certified bytes"
            ),
            None,
        )
    assert entry is not None
    if guard is None:
        return report(
            ASSURANCE_GUARD_DISABLED,
            (
                "the Unicode guard is not active in this process; without it the "
                "pinned readings do not apply"
            ),
            None,
        )
    judged = str((entry.get("oracle") or {}).get("version"))
    if oracle_version != judged:
        return report(
            ASSURANCE_ORACLE_MISMATCH,
            (
                f"the installed reference (tokenizers {oracle_version}) is not the "
                f"one the pinned evidence was judged against ({judged})"
            ),
            None,
        )
    if family is not None:
        covered = any(
            isinstance(row, Mapping)
            and row.get("family") == family
            and row.get("artifact_sha256") == artifact_sha256
            for row in entry.get("families") or ()
        )
        if not covered:
            return report(
                ASSURANCE_FAMILY_OUTSIDE_EVIDENCE,
                "no pinned-build reading is on file for this family",
                None,
            )
    return report(
            ASSURANCE_CERTIFIED_PINNED,
            None,
            _basis(entry, known),
        )


def _fail_closed(identity: FastokensIdentity | None) -> AssuranceReport:
    """The report a directly constructed adapter carries: no premise verified."""
    return AssuranceReport(
        assurance=ASSURANCE_UNRECOGNIZED_BUILD,
        reason=(
            "the engine was not resolved against the shipped registry, so the "
            "pinned readings do not apply"
        ),
        known_wheel=None,
        guard_active=False,
        guard_codepoints=0,
        basis=None,
        advisory=None,
        distribution=identity.distribution if identity is not None else None,
        version=identity.version if identity is not None else None,
        engine_digest=identity.engine_digest if identity is not None else None,
        identity=identity,
    )


# ---------------------------------------------------------------------------
# The adapter


class _Encoding(Protocol):
    @property
    def ids(self) -> Sequence[int]: ...


class _Tokenizer(Protocol):
    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> _Encoding: ...


class _Factory(Protocol):
    @staticmethod
    def from_file(path: str) -> _Tokenizer: ...


class _FastTokenizer(Protocol):
    def get_vocab(self) -> Mapping[str, int]: ...


class FastokensFullRepair:
    """Full-session Fastokens callback, admitted only under EXPERIMENTAL."""

    def __init__(
        self,
        *,
        spec: RepairFamily,
        engine: _Tokenizer,
        engine_version: str,
        engine_digest: str,
        hf_tokenizer: _FastTokenizer,
        reference_encode: ReferenceEncode,
        assurance: AssuranceReport | None = None,
        unicode_guard: re.Pattern[str] | None = None,
    ) -> None:
        self.spec = spec
        self._engine = engine
        self._engine_version = engine_version
        self._engine_digest = engine_digest
        self._reference_encode = reference_encode
        self._byte_lengths = tuple(_byte_lengths_from_hf(hf_tokenizer))
        self._assurance = assurance if assurance is not None else _fail_closed(None)
        self._guard = unicode_guard
        self._path_counts: dict[str, int] = {}
        self._last: dict[str, object] | None = None

    @classmethod
    def open(
        cls,
        *,
        spec: RepairFamily,
        tokenizer_path: Path,
        hf_tokenizer: _FastTokenizer,
        reference_encode: ReferenceEncode,
    ) -> FastokensFullRepair:
        """Load Fastokens from the already verified tokenizer artifact.

        Refuses when the package is not importable, when its metadata is
        orphaned, and when the bytes the import system would run are not the
        ones an installed distribution recorded (a shadowing copy on
        ``sys.path``): a digest that names other bytes than the ones running
        would let one build's readings vouch for another's.
        """
        from importlib import import_module

        from .._oracle import oracle_version

        identity = fastokens_identity()
        if not identity.available:
            orphans = identity.orphaned
            if orphans:
                named = ", ".join(orphan.label for orphan in orphans)
                raise BackendUnavailable(
                    "the experimental Fastokens repair backend is recorded as "
                    f"installed ({named}) but its files are missing; uninstalling "
                    "one of two distributions that share the fastokens package "
                    f"removes the files of both. Reinstall: {_REINSTALL_COMMAND}",
                    details={
                        "backend": BACKEND_FASTOKENS,
                        "extra": "fastokens",
                        "orphaned": [orphan.label for orphan in orphans],
                    },
                )
            raise BackendUnavailable(
                "the experimental Fastokens repair backend is not installed",
                details={"backend": BACKEND_FASTOKENS, "extra": "fastokens"},
            )
        if not identity.verifiable:
            recorded = ", ".join(
                f"{candidate.label} at {candidate.package_dir}"
                for candidate in identity.owners
                if candidate.package_dir is not None
            )
            raise BackendUnavailable(
                "the Fastokens engine could not be verified: the import system "
                f"resolves fastokens to {identity.package_dir}"
                + (
                    f", while the installed distribution records it at {recorded}"
                    if recorded
                    else ", which no installed distribution records"
                )
                + (
                    f"; hashing failed ({identity.hash_error})"
                    if identity.hash_error
                    else ""
                ),
                details={
                    "backend": BACKEND_FASTOKENS,
                    "stage": "identity",
                    "engine_assurance": ASSURANCE_UNVERIFIABLE,
                    "imported_from": str(identity.package_dir),
                    "recorded_at": [
                        str(candidate.package_dir)
                        for candidate in identity.owners
                        if candidate.package_dir is not None
                    ],
                },
            )
        try:
            module = import_module(IMPORT_NAME)
            factory: _Factory = module.Tokenizer
            engine = factory.from_file(str(tokenizer_path))
        except (ImportError, AttributeError) as error:
            raise BackendUnavailable(
                "the experimental Fastokens repair backend is not installed",
                details={"backend": BACKEND_FASTOKENS, "extra": "fastokens"},
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise BackendUnavailable(
                f"Fastokens could not load the verified artifact: {error}",
                details={"backend": BACKEND_FASTOKENS, "stage": "engine_load"},
            ) from error
        module_file = getattr(module, "__file__", None)
        imported_from = (
            Path(module_file).resolve().parent if module_file is not None else None
        )
        if imported_from != identity.package_dir:
            raise BackendUnavailable(
                "the Fastokens engine could not be verified: the module imported "
                f"from {module_file} while the identity was taken at "
                f"{identity.package_dir}",
                details={
                    "backend": BACKEND_FASTOKENS,
                    "stage": "identity",
                    "engine_assurance": ASSURANCE_UNVERIFIABLE,
                },
            )
        entry = pinned_engine_entry()
        guard = compile_unicode_guard(entry)
        report = assess(
            identity,
            entry=entry,
            guard=guard,
            oracle_version=oracle_version(),
            family=spec.family,
            artifact_sha256=spec.artifact_sha256,
        )
        try:
            return cls(
                spec=spec,
                engine=engine,
                engine_version=identity.version or "",
                engine_digest=str(identity.engine_digest),
                hf_tokenizer=hf_tokenizer,
                reference_encode=reference_encode,
                assurance=report,
                unicode_guard=guard,
            )
        except (AttributeError, TypeError, ValueError, WindowUnsupported) as error:
            raise BackendUnavailable(
                f"Fastokens cannot use this verified artifact: {error}",
                details={
                    "backend": BACKEND_FASTOKENS,
                    "stage": "span_table",
                    "family": spec.family,
                },
            ) from error

    @property
    def config_id(self) -> str:
        return CONFIG_ID

    @property
    def assurance(self) -> AssuranceReport:
        return self._assurance

    def _count(self, path: str) -> None:
        self._path_counts[path] = self._path_counts.get(path, 0) + 1

    def _reference_full(
        self, text: str, *, reason: str, detail: object | None = None
    ) -> tuple[list[int], list[tuple[int, int]], int, str]:
        ids, spans = self._reference_encode(text)
        path = f"hf_full_fastokens_{reason}"
        self._count(path)
        self._last = {
            "path": path,
            "reason": reason,
            "detail": detail,
            "input_chars": len(text),
        }
        return ids, spans, 0, path

    def __call__(
        self,
        tail_text: str,
        tail_ids: list[int],
        tail_spans: Sequence[tuple[int, int]],
        delta: str,
    ) -> tuple[list[int], list[tuple[int, int]], int, str]:
        text = tail_text + delta
        if not delta:
            path = "fastokens_full_experimental_noop"
            self._count(path)
            self._last = {"path": path, "kept_tokens": len(tail_ids)}
            return list(tail_ids), list(tail_spans), len(tail_ids), path
        if self._guard is not None:
            hit = self._guard.search(text)
            if hit is not None:
                # The pinned reference does not reorder these combining marks
                # the way the engine does; the whole request is answered by
                # the reference, which is what the evidence counted as
                # routed.
                return self._reference_full(
                    text,
                    reason="unicode_skew_guard",
                    detail={
                        "codepoint": f"U+{ord(hit.group(0)):04X}",
                        "position": hit.start(),
                    },
                )
        try:
            ids = [
                int(value)
                for value in self._engine.encode(
                    text, add_special_tokens=False
                ).ids
            ]
            spans = _spans_from_ids(ids, self._byte_lengths, text)
        except (
            OSError,
            RuntimeError,
            UnicodeError,
            ValueError,
            WindowUnsupported,
        ) as error:
            return self._reference_full(
                text,
                reason="guard",
                detail={"error": type(error).__name__, "message": str(error)},
            )
        path = "fastokens_full_experimental"
        self._count(path)
        self._last = {
            "path": path,
            "input_chars": len(text),
            "delta_chars": len(delta),
            "kept_tokens": 0,
        }
        return ids, spans, 0, path

    def stats(self) -> dict[str, object]:
        report = self._assurance
        return {
            "backend": BACKEND_FASTOKENS,
            "engine": "fastokens",
            "engine_distribution": report.distribution,
            "engine_version": self._engine_version,
            "engine_digest": self._engine_digest,
            "config_id": self.config_id,
            "certification": "experimental",
            "engine_assurance": report.assurance,
            "exact_id_guarantee": report.exact_id_guarantee,
            "assurance_reason": report.reason,
            "guarantee_basis": dict(report.basis) if report.basis is not None else None,
            "known_wheel": dict(report.known_wheel) if report.known_wheel else None,
            "unicode_guard": {
                "id": GUARD_ID,
                "codepoints": report.guard_codepoints,
                "active": report.guard_active,
            },
            "advisory": report.advisory,
            "mode": "full_reencode",
            "family": self.spec.family,
            "artifact_sha256": self.spec.artifact_sha256,
            "path_counts": dict(sorted(self._path_counts.items())),
            "last": dict(self._last) if self._last is not None else None,
        }
