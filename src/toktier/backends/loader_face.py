"""Loader-face materialization, shared by the reference and fast CPU backends.

Contract reference: ``docs/contracts/registry.md`` Section 1 (the exact
artifact identity and its configuration-side extension) and
``docs/contracts/facade.md`` Section 5 (reference = the loader face).

Since 0.2.8 the reference is the **loader face**: the tokenizer object the
pinned loader (``transformers.AutoTokenizer``) materializes from the verified
artifact directory. The artifact identity covers the tokenizer file plus the
declared configuration-side added tokens, so the certification subject and
the executed object describe the same added-token vocabulary. For an
artifact whose ``tokenizer_config.json`` declares no added token beyond the
artifact file, the artifact document already is the loader face's added-token
vocabulary and is executed as written, without importing ``transformers``.
A directory carrying no loader configuration file at all is materialized as
the file-only face directly, never through a tokenizer class inferred from
the directory path (:data:`LOADER_CONFIGURATION_FILES`), so the loader face
is the same object wherever the verified bytes sit.

This module is the one place that answers three questions the two backends
must answer identically:

- Which added-token literals exist only in the configuration sidecar
  (:func:`config_added_token_rows`)?
- What is the canonical digest of that subset
  (:func:`config_added_tokens_sha256`), the value a registry record declares
  under ``config_added_tokens``?
- Does the subset this machine observes match what the certification record
  declared (:func:`verify_declared_config_added_tokens`)? A mismatch is
  fail-closed: the running configuration would not be the one the
  certification readings were taken against.

The ``transformers`` import stays inside :func:`load_live_tokenizer`, so
importing a backend module continues to load neither the loader nor the
oracle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..errors import ArtifactHashMismatch, UnsupportedConfig
from .protocol import TOKENIZER_FILE

__all__ = [
    "CONFIG_ADDED_TOKENS_MISMATCH",
    "LOADER_CONFIGURATION_FILES",
    "TOKENIZER_CONFIG_FILE",
    "config_added_token_rows",
    "config_added_tokens_sha256",
    "config_only_added_tokens",
    "live_tokenizer_json",
    "load_live_tokenizer",
    "verify_declared_config_added_tokens",
]

#: Name of the loader-side configuration sidecar; the file that can
#: declare added tokens the artifact itself does not carry.
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

#: Configuration files the pinned loader reads to choose a tokenizer
#: class from content. When neither is present the loader's remaining
#: rule infers a class from substrings of the directory *path*, which
#: would make the materialized face depend on where the bytes sit
#: rather than on what they are. Such a directory is therefore
#: materialized as the file-only face directly (see
#: :func:`load_live_tokenizer`).
LOADER_CONFIGURATION_FILES = ("config.json", TOKENIZER_CONFIG_FILE)

#: Named reason recorded when the observed configuration-side added-token
#: subset does not match the subset the certification record declared.
CONFIG_ADDED_TOKENS_MISMATCH = "config_added_tokens_mismatch"

#: Per-token flags that participate in added-token matching semantics.
#: The set matches the frontend's frozen flag fields
#: (``toktier.frontend.added``); every row of the canonical form carries
#: all of them, so two subsets differing only in a flag hash differently.
_ROW_FLAGS = ("special", "single_word", "lstrip", "rstrip", "normalized")


def config_added_token_rows(
    root: Path, artifact: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Added tokens declared only in the configuration sidecar, id-ascending.

    Each row carries ``id``, ``content`` and the matching-semantics flags.
    These are the tokens a ``tokenizer.json``-only construction cannot see;
    an empty list means the artifact document and the loader face agree on
    the added-token vocabulary.
    """
    config_path = root / TOKENIZER_CONFIG_FILE
    if not config_path.is_file():
        return []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder = (
        config.get("added_tokens_decoder") if isinstance(config, dict) else None
    )
    if not isinstance(decoder, dict):
        return []
    declared: list[tuple[int, Mapping[str, Any]]] = []
    for key, item in decoder.items():
        if isinstance(item, Mapping) and "content" in item:
            try:
                token_id = int(key)
            except (TypeError, ValueError):
                continue
            declared.append((token_id, item))
    if not declared:
        return []
    if artifact is None:
        loaded = json.loads((root / TOKENIZER_FILE).read_text(encoding="utf-8"))
        artifact = loaded if isinstance(loaded, Mapping) else {}
    raw_added_tokens = artifact.get("added_tokens")
    added_tokens = raw_added_tokens if isinstance(raw_added_tokens, list) else ()
    carried = {
        token.get("content") for token in added_tokens if isinstance(token, dict)
    }
    model = artifact.get("model")
    vocabulary = model.get("vocab", {}) if isinstance(model, Mapping) else {}
    rows: list[dict[str, Any]] = [
        {
            "id": token_id,
            "content": str(item["content"]),
            **{flag: bool(item.get(flag, False)) for flag in _ROW_FLAGS},
        }
        for token_id, item in declared
        if item["content"] not in carried and item["content"] not in vocabulary
    ]
    rows.sort(key=lambda row: int(row["id"]))
    return rows


def config_only_added_tokens(
    root: Path, artifact: Mapping[str, Any] | None = None
) -> list[str]:
    """Added-token literals declared only in the configuration sidecar.

    The list gates the loading fallback in :func:`load_live_tokenizer`:
    the fallback may only be taken when this list is empty, because a
    fallback that silently dropped an added token would encode
    differently from the loader face.
    """
    return [
        str(row["content"]) for row in config_added_token_rows(root, artifact)
    ]


def config_added_tokens_sha256(rows: list[dict[str, Any]]) -> str:
    """Canonical digest of a configuration-side added-token subset.

    The canonical form is the id-ascending row list serialized with sorted
    keys, no whitespace, and literal non-ASCII; the digest is the SHA-256
    of that UTF-8 text. Registry records declare this value under
    ``config_added_tokens.sha256``, and the loading paths recompute it
    from the files they are about to execute.
    """
    ordered = sorted(rows, key=lambda row: int(row["id"]))
    payload = json.dumps(
        ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_declared_config_added_tokens(
    *,
    family: str,
    observed_rows: list[dict[str, Any]],
    declared: Mapping[str, Any] | None,
) -> None:
    """Fail closed when the observed subset differs from the declared one.

    ``declared`` is the certification record's ``config_added_tokens``
    claim for this exact artifact: a mapping with ``sha256`` and ``count``
    when the record declares configuration-side added tokens, an empty
    claim (count 0) when the record declares none, and ``None`` when no
    record makes a claim at all -- then there is nothing to verify and an
    uncertified artifact still loads. A mismatch in either direction means
    the loader face on this machine is not the certified subject, so the
    open fails rather than serving ids nobody judged
    (reason ``config_added_tokens_mismatch``).
    """
    if declared is None:
        return
    declared_count = int(declared.get("count") or 0)
    declared_sha = declared.get("sha256")
    observed_count = len(observed_rows)
    observed_sha = (
        config_added_tokens_sha256(observed_rows) if observed_rows else None
    )
    if observed_count == declared_count and observed_sha == declared_sha:
        return
    raise ArtifactHashMismatch(
        f"{family}: the configuration-side added tokens this machine "
        "observes do not match the subset the certification record "
        "declares",
        details={
            "reason": CONFIG_ADDED_TOKENS_MISMATCH,
            "expected_sha256": declared_sha,
            "observed_sha256": observed_sha,
            "expected_count": declared_count,
            "observed_count": observed_count,
            "file": TOKENIZER_CONFIG_FILE,
            "remedy": (
                "re-fetch the artifact directory; a configuration sidecar "
                "that no longer declares the recorded added-token subset "
                "is never accepted on a certified artifact"
            ),
        },
    )


def live_tokenizer_json(hf_tokenizer: object) -> str:
    """Serialize the already materialized fast HF tokenizer."""
    backend = getattr(hf_tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        to_str = getattr(hf_tokenizer, "to_str", None)
    if not callable(to_str):
        raise UnsupportedConfig(
            "the live fast tokenizer cannot serialize its backend",
            details={
                "option": "hf_tokenizer",
                "value": type(hf_tokenizer).__name__,
                "reason": "serializable live backend required",
            },
        )
    data = to_str()
    if not isinstance(data, str):
        raise UnsupportedConfig(
            "the live fast tokenizer returned a non-text serialization",
            details={
                "option": "hf_tokenizer",
                "value": type(data).__name__,
                "reason": "tokenizer JSON text required",
            },
        )
    return data


def load_live_tokenizer(root: Path) -> object:
    """Materialize the live HF tokenizer from a verified directory.

    Uses the base installation's pinned ``transformers`` with local files
    only: the artifact was verified by the artifacts layer, and nothing
    here reaches the network. Loading through ``transformers`` is what
    makes configuration-only added tokens visible; the engine is then
    handed the live object, never a path.

    A directory carrying none of the loader configuration files
    (:data:`LOADER_CONFIGURATION_FILES`) is materialized as a
    ``PreTrainedTokenizerFast`` over the artifact file directly, without
    consulting ``AutoTokenizer``. With no configuration to read, the
    loader's remaining resolution rule infers a tokenizer class from
    substrings of the directory path, so the face it builds would depend
    on where the bytes sit rather than on what they are. Such a
    directory also cannot declare a configuration-side added token, so
    the file-only face already is the loader face -- the same degenerate
    form the fallback below produces, reached without the detour.

    When a configuration file is present but the pinned loader cannot
    construct the object, the documented fallback is a
    ``PreTrainedTokenizerFast`` over the artifact file alone. The case
    this fallback exists for is a configuration naming a loader class
    the installed ``transformers`` does not know (a ``tokenizer_class``
    from a newer release, say), but the cause of the construction
    failure is not classified: every failure reaches the same branch.
    The one condition tested before the fallback is taken is the one the
    equivalence rests on -- the configuration-side added-token subset is
    empty, which is exactly when the file-only face and the loader face
    are provably the same function, since a subset the artifact file
    does not carry is what a file-only construction cannot see. Over an
    artifact with a non-empty subset the original loading error
    propagates instead (and surfaces as a recoverable fault, so the
    input runs on the reference backend).
    ``docs/contracts/facade.md`` Section 5 states the same rule.
    """
    from importlib import import_module

    transformers = import_module("transformers")
    if not any((root / name).is_file() for name in LOADER_CONFIGURATION_FILES):
        tokenizer: object = transformers.PreTrainedTokenizerFast(
            tokenizer_file=str(root / TOKENIZER_FILE)
        )
        return _require_fast(tokenizer)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(root), use_fast=True, local_files_only=True
        )
    except Exception:
        if config_only_added_tokens(root):
            raise
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_file=str(root / TOKENIZER_FILE)
        )
    return _require_fast(tokenizer)


def _require_fast(tokenizer: object) -> object:
    """The engine consumes the fast backend object; refuse anything else."""
    if not getattr(tokenizer, "is_fast", False):
        raise UnsupportedConfig(
            "the loaded tokenizer object is not a fast tokenizer; the "
            "engine consumes the fast backend object",
            details={
                "option": "tokenizer",
                "value": type(tokenizer).__name__,
                "reason": "fast tokenizer object required",
            },
        )
    return tokenizer
