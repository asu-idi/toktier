r"""Added-token frontend: literal extraction ahead of a backend.

Contract reference: ``docs/contracts/registry.md`` Section 1 (the
added-frontend capability identity), ``docs/contracts/fingerprint.md``
Sections 3 and 6 (the added-token table encoding), and
``docs/contracts/routing.md`` Section 5.2 (``R_INPUT_ADDED_TOKEN``).

What this layer does
--------------------

The oracle's first encoding step extracts added-token literals from the
raw text before the pipeline proper runs. This module performs that same
extraction ahead of a backend, so an accelerated backend only ever sees
the segments between literals, and each literal maps straight to its id.

Two stages, cheap first:

1. **Prefilter** -- a sound over-approximation over the UTF-8 bytes of
   the document: a first-byte gate, then a two-byte-prefix gate. A
   document with no candidate costs one linear scan and takes the
   unchanged path.
2. **Exact layer** -- runs only for candidates, and does not implement
   matching itself. It builds a "splitter-only" tokenizer (a WordLevel
   model holding a single sentinel entry, carrying the family's added
   tokens verbatim) and lets the oracle's own added-vocabulary
   implementation produce the leftmost-longest split. The flags
   (``single_word`` / ``lstrip`` / ``rstrip`` / ``normalized``) are
   therefore honored by the same code that defines correct behavior,
   not by a second matcher written here.

Two faces, in the oracle's order
--------------------------------

Phase A extracts ``normalized=False`` tokens from the raw text; phase B
extracts ``normalized=True`` tokens from the normalized form of the
remaining segments. A ``normalized=True`` token combined with a
normalizer that is not the identity would require mapping coordinates
back through normalization, which no reading covers, so that
combination is refused at construction (``UNSUPPORTED_CONFIG``) instead
of being approximated.

Tables
------

Tables are keyed by their own content hash (``content_sha256``) and
record the digest of the artifact they were exported from, so a table
that no longer belongs to the artifact in use is refused rather than
applied. Entries that exist only in the tokenizer configuration and not
in the artifact are recorded in the table but take no part in matching,
which keeps this layer's view identical to the oracle's.

Port notes (prototype -> this package)
--------------------------------------

- The sentinel string is an internal constant with no external contract;
  it was renamed to the current package name.
- The prefilter is implemented with the standard library
  (``bytes.translate`` plus a compiled literal alternation) rather than
  with array lookup tables. Same decision on every input -- it is an
  over-approximation either way -- and it keeps the core package free of
  an array dependency.
- There is no environment switch for this layer. Whether it is engaged
  is a routing decision made by the executor; a switch that can change
  output must not exist in ambient form.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from .._oracle import import_oracle
from ..errors import ArtifactHashMismatch, UnsupportedConfig

__all__ = [
    "ADDED_FRONTEND_DOMAIN_TAG",
    "SEG_ID",
    "SEG_TOKEN",
    "TABLE_VERSION",
    "AddedTokenFrontend",
    "AddedTokenFrontendProtocol",
    "AddedTokenPlan",
    "added_frontend_fingerprint",
    "assemble",
    "load_table",
    "table_content_sha256",
]

#: Sentinel entry of the splitter model. It carries control bytes, so it
#: cannot equal any real added-token content; if a document happened to
#: contain it, the sentinel id still means "encode this span as ordinary
#: text", so a collision stays benign.
SEG_TOKEN = "\x00TOKTIER_SEG\x00"

#: Sentinel id, chosen above every real vocabulary size.
SEG_ID = 4_200_000_000

#: Table format identifier recorded in every exported table.
TABLE_VERSION = "added-v1"

#: Domain tag of the added-frontend capability identity
#: (registry.md Section 1, identity 3).
ADDED_FRONTEND_DOMAIN_TAG = b"toktier.added_frontend.v1\x00"

#: The seven per-token sub-fields, in the frozen order of
#: fingerprint.md Section 6.
_FLAG_FIELDS = ("special", "single_word", "lstrip", "rstrip", "normalized")

#: One planned span: the text, and the token id if it is a literal.
AddedTokenPlan = list[tuple[str, "int | None"]]
"""Ordered spans of one planned document."""


@runtime_checkable
class AddedTokenFrontendProtocol(Protocol):
    """The frontend surface an encoder consumes: scan, then assemble.

    This is the one contract shared by the concrete frontend below and
    by every backend that takes a frontend by injection (for example
    the GPU encoders). A backend never decides whether the frontend is
    engaged; that is a routing decision recorded in the plan.
    """

    def scan(self, text: str) -> AddedTokenPlan | None:
        """Return a split plan, or ``None`` when there is no literal."""

    def assemble(
        self,
        plan: Sequence[tuple[str, int | None]],
        encode_segment: Callable[[str], Sequence[int]],
    ) -> list[int]:
        """Concatenate literal ids and encoded segments, in order."""


class _CrateEncoding(Protocol):
    @property
    def ids(self) -> Sequence[int]:
        """Token ids of one encoded sequence."""

    @property
    def offsets(self) -> Sequence[tuple[int, int]]:
        """Character spans of one encoded sequence."""


class _CrateTokenizer(Protocol):
    def encode(
        self, sequence: str, add_special_tokens: bool = True
    ) -> _CrateEncoding:
        """Encode one sequence."""

    def token_to_id(self, token: str) -> int | None:
        """Resolve a token string to the id the crate assigned it."""


# ---------------------------------------------------------------------
# table handling
# ---------------------------------------------------------------------


def _canonical_body(table: Mapping[str, Any]) -> str:
    body = {key: table[key] for key in sorted(table) if key != "content_sha256"}
    return json.dumps(body, ensure_ascii=False, sort_keys=True)


def table_content_sha256(table: Mapping[str, Any]) -> str:
    """Content hash of a table, excluding the recorded hash itself."""
    return hashlib.sha256(_canonical_body(table).encode("utf-8")).hexdigest()


def load_table(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and verify one added-token table.

    Two checks, both fail-closed: the table's own content hash, and --
    when the caller knows it -- the digest of the artifact the table was
    exported from. A table that drifted from its artifact describes a
    different added vocabulary than the one being encoded, which is
    exactly the case that must never be applied silently.
    """
    table_path = Path(path)
    table = cast(dict[str, Any], json.loads(table_path.read_text(encoding="utf-8")))
    recorded = table.get("content_sha256")
    observed = table_content_sha256(table)
    if recorded != observed:
        raise ArtifactHashMismatch(
            "added-token table does not match its recorded content hash",
            details={
                "expected_sha256": recorded,
                "observed_sha256": observed,
                "path": str(table_path),
                "remedy": "re-export the table from the artifact",
            },
        )
    if expected_artifact_sha256 is not None:
        bound = table.get("tokenizer_json_sha256")
        if bound != expected_artifact_sha256:
            raise ArtifactHashMismatch(
                "added-token table was exported from a different artifact",
                details={
                    "expected_sha256": expected_artifact_sha256,
                    "observed_sha256": bound,
                    "path": str(table_path),
                    "remedy": "re-export the table from the artifact in use",
                },
            )
    return table


# ---------------------------------------------------------------------
# capability fingerprint
# ---------------------------------------------------------------------


def _encode_string(value: str) -> bytes:
    return b"\x01" + value.encode("utf-8")


def _encode_bool(value: bool) -> bytes:
    return b"\x01\x01" if value else b"\x01\x00"


def _encode_u64(value: int) -> bytes:
    return b"\x01" + struct.pack("<Q", value)


def _encode_added_tokens(added_tokens: Sequence[Mapping[str, Any]]) -> bytes:
    """List encoding of the added-token table (fingerprint.md 3 and 6).

    Element order is the artifact's insertion order, which is part of the
    binding because extraction behavior can depend on it.
    """
    parts = [b"\x01", struct.pack("<I", len(added_tokens))]
    for token in added_tokens:
        parts.append(_encode_string(str(token["content"])))
        parts.append(_encode_u64(int(token["id"])))
        for flag in _FLAG_FIELDS:
            parts.append(_encode_bool(bool(token[flag])))
    return b"".join(parts)


def added_frontend_fingerprint(
    added_tokens: Sequence[Mapping[str, Any]],
) -> str:
    """Added-frontend capability identity for a table.

    Interface alignment note: the preimage is the domain tag followed by
    the Section 6 list encoding, including the list presence byte and
    element count. If another lane materializes the shared Section 6
    encoder for the semantic fingerprint's field ``0x0003``, the two must
    produce identical bytes; this is the one place to reconcile.
    """
    preimage = ADDED_FRONTEND_DOMAIN_TAG + _encode_added_tokens(added_tokens)
    return hashlib.sha256(preimage).hexdigest()


# ---------------------------------------------------------------------
# frontend
# ---------------------------------------------------------------------


def _is_identity_normalizer(normalizer: Any) -> bool:
    """True when a normalizer section cannot change the text."""
    if normalizer is None:
        return True
    return (
        isinstance(normalizer, dict)
        and normalizer.get("type") == "Sequence"
        and not normalizer.get("normalizers")
    )


def _splitter_document(
    added_tokens: Sequence[Mapping[str, Any]], normalizer: Any
) -> str:
    return json.dumps(
        {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": list(added_tokens),
            "normalizer": normalizer,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {SEG_TOKEN: SEG_ID},
                "unk_token": SEG_TOKEN,
            },
        }
    )


def _build_splitter(
    added_tokens: Sequence[Mapping[str, Any]], normalizer: Any
) -> _CrateTokenizer:
    document = _splitter_document(added_tokens, normalizer)
    return cast(_CrateTokenizer, import_oracle().Tokenizer.from_str(document))


class AddedTokenFrontend:
    """Added-token literal extraction for one family."""

    def __init__(
        self,
        table: Mapping[str, Any],
        *,
        allow_nonidentity_norm_face: bool = False,
    ) -> None:
        self._family = str(table.get("family", "unknown"))
        tokens = [dict(token) for token in table["added_tokens"]]
        self._added_tokens: tuple[Mapping[str, Any], ...] = tuple(tokens)
        raw_face = [token for token in tokens if not token["normalized"]]
        norm_face = [token for token in tokens if token["normalized"]]
        normalizer = table.get("normalizer")
        identity_normalizer = _is_identity_normalizer(normalizer)
        if norm_face and not identity_normalizer and not allow_nonidentity_norm_face:
            raise UnsupportedConfig(
                f"{self._family}: an added token declared normalized=True "
                "together with a normalizer that is not the identity would "
                "require mapping spans back through normalization, which no "
                "reading covers",
                details={
                    "option": "added_tokens.normalized",
                    "value": self._family,
                    "reason": "coordinate mapping through normalization is "
                    "not certified",
                },
            )

        self._splitter_raw: _CrateTokenizer | None = None
        self._remap_raw: dict[int, int] = {}
        self._splitter_norm: _CrateTokenizer | None = None
        self._remap_norm: dict[int, int] = {}
        if raw_face:
            splitter = _build_splitter(raw_face, None)
            self._splitter_raw = splitter
            self._remap_raw = self._build_remap(splitter, raw_face)
        if norm_face:
            splitter = _build_splitter(norm_face, normalizer)
            self._splitter_norm = splitter
            self._remap_norm = self._build_remap(splitter, norm_face)

        # With a normalizer that is not the identity, a normalized-face
        # pattern need not appear verbatim in the raw bytes, so a byte
        # scan is no longer a necessary condition for a match. That
        # combination (reachable only through the probe constructor)
        # skips the prefilter and always runs the exact layer: slower,
        # never wrong.
        self._prefilter_exact = (not norm_face) or identity_normalizer
        self._build_prefilter([str(token["content"]) for token in tokens])

    # -- construction helpers ------------------------------------------

    @classmethod
    def from_table_file(
        cls,
        path: str | Path,
        *,
        expected_artifact_sha256: str | None = None,
    ) -> AddedTokenFrontend:
        """Build a frontend from a verified table file."""
        return cls(
            load_table(path, expected_artifact_sha256=expected_artifact_sha256)
        )

    @staticmethod
    def _build_remap(
        splitter: _CrateTokenizer, tokens: Sequence[Mapping[str, Any]]
    ) -> dict[int, int]:
        """Map splitter-assigned ids back to the artifact's ids.

        Deserializing the splitter re-assigns added-token ids, so the
        mapping is rebuilt from the token content rather than assumed.
        """
        remap: dict[int, int] = {}
        for token in tokens:
            content = str(token["content"])
            assigned = splitter.token_to_id(content)
            if assigned is None or assigned == SEG_ID:
                raise UnsupportedConfig(
                    "added-token content is not recoverable from the "
                    "splitter tokenizer",
                    details={
                        "option": "added_tokens.content",
                        "value": content,
                        "reason": "the oracle did not assign the literal an id",
                    },
                )
            remap[int(assigned)] = int(token["id"])
        return remap

    def _build_prefilter(self, contents: Iterable[str]) -> None:
        first_bytes: set[int] = set()
        single_bytes: set[int] = set()
        pair_prefixes: set[bytes] = set()
        for content in contents:
            encoded = content.encode("utf-8")
            if not encoded:
                # An empty literal has no first byte to gate on, so the
                # cheap stage cannot rule anything out: fall back to
                # always running the exact layer rather than build a
                # filter that could say no.
                self._prefilter_exact = False
                continue
            first_bytes.add(encoded[0])
            if len(encoded) >= 2:
                pair_prefixes.add(encoded[:2])
            else:
                single_bytes.add(encoded[0])
        self._first_gate = bytes(
            1 if index in first_bytes else 0 for index in range(256)
        )
        self._single_gate = bytes(
            1 if index in single_bytes else 0 for index in range(256)
        )
        self._has_single = bool(single_bytes)
        self._pair_pattern = (
            re.compile(b"|".join(re.escape(prefix) for prefix in sorted(pair_prefixes)))
            if pair_prefixes
            else None
        )

    # -- identity ------------------------------------------------------

    @property
    def family(self) -> str:
        """Family id this frontend was built for."""
        return self._family

    @property
    def added_tokens(self) -> tuple[Mapping[str, Any], ...]:
        """The added-token table, in artifact insertion order."""
        return self._added_tokens

    def capability_fingerprint(self) -> str:
        """Added-frontend capability identity of this table."""
        return added_frontend_fingerprint(self._added_tokens)

    def _native_prefilter_prefixes(self) -> tuple[tuple[int, int], ...] | None:
        """Frozen one-/two-byte necessary conditions for the Rust router.

        ``None`` means this frontend cannot soundly reject a document before
        the exact layer (for example normalized-face literals under a
        non-identity normalizer). A second value of ``-1`` denotes a
        one-byte literal. This is private plumbing, not a second matcher: a
        positive result still runs :meth:`scan` and only a negative result is
        final.
        """
        if not self._prefilter_exact:
            return None
        prefixes: set[tuple[int, int]] = set()
        for token in self._added_tokens:
            encoded = str(token["content"]).encode("utf-8")
            if not encoded:
                return None
            prefixes.add((encoded[0], encoded[1] if len(encoded) > 1 else -1))
        return tuple(sorted(prefixes))

    # -- scanning ------------------------------------------------------

    def _prefilter_hit(self, raw: bytes) -> bool:
        if not self._prefilter_exact:
            return True
        if not raw:
            return False
        if b"\x01" not in raw.translate(self._first_gate):
            return False
        if self._has_single and b"\x01" in raw.translate(self._single_gate):
            return True
        if self._pair_pattern is None:
            return False
        return self._pair_pattern.search(raw) is not None

    def _split_phase(
        self,
        splitter: _CrateTokenizer,
        remap: Mapping[int, int],
        text: str,
    ) -> Iterator[tuple[str, int | None]]:
        encoding = splitter.encode(text, add_special_tokens=False)
        for token_id, (start, end) in zip(encoding.ids, encoding.offsets, strict=True):
            if int(token_id) == SEG_ID:
                yield text[start:end], None
            else:
                yield text[start:end], remap[int(token_id)]

    def scan(self, text: str) -> AddedTokenPlan | None:
        """Plan the literal split of one document.

        Returns ``None`` when the document holds no added-token literal
        (the caller keeps its unchanged path), otherwise the ordered
        spans: ``(text, None)`` for ordinary text and ``(literal, id)``
        for a literal.
        """
        if not text or not self._prefilter_hit(text.encode("utf-8")):
            return None
        if self._splitter_raw is not None:
            parts: AddedTokenPlan = list(
                self._split_phase(self._splitter_raw, self._remap_raw, text)
            )
        else:
            parts = [(text, None)]
        if self._splitter_norm is not None:
            merged: AddedTokenPlan = []
            for segment, token_id in parts:
                if token_id is not None or not segment:
                    merged.append((segment, token_id))
                else:
                    merged.extend(
                        self._split_phase(
                            self._splitter_norm, self._remap_norm, segment
                        )
                    )
            parts = merged
        if all(token_id is None for _, token_id in parts):
            return None
        return parts

    def assemble(
        self,
        plan: Sequence[tuple[str, int | None]],
        encode_segment: Callable[[str], Sequence[int]],
    ) -> list[int]:
        """Join a scan plan; delegates to the module-level :func:`assemble`."""
        return assemble(plan, encode_segment)


def assemble(
    plan: Sequence[tuple[str, int | None]],
    encode_segment: Callable[[str], Sequence[int]],
) -> list[int]:
    """Join a scan plan: literals map directly, segments are encoded."""
    out: list[int] = []
    for segment, token_id in plan:
        if token_id is not None:
            out.append(token_id)
        elif segment:
            out.extend(int(value) for value in encode_segment(segment))
    return out
