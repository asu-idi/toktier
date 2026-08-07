"""Added-token frontend behavior.

Contract reference: ``docs/contracts/fingerprint.md`` Sections 3 and 6,
``docs/contracts/registry.md`` Section 1.

The differential shape is the one the prototype battery used: the
reference is the whole artifact encoding a document, and the subject is
the frontend's split with each segment encoded by the same artifact with
its added tokens removed. If the split is right, the two are id-equal;
if it is wrong anywhere, they are not.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import _support as support
import pytest

from toktier import _native
from toktier.errors import ArtifactHashMismatch, UnsupportedConfig
from toktier.frontend.added import (
    AddedTokenFrontend,
    added_frontend_fingerprint,
    assemble,
    load_table,
    table_content_sha256,
)

tokenizers = pytest.importorskip("tokenizers")

BIG_ID = 300_000


def added(content: str, index: int, **flags: bool) -> dict[str, Any]:
    """One constructed added-token entry with all flags spelled out."""
    entry = {
        "content": content,
        "id": BIG_ID + index,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": False,
    }
    entry.update(flags)
    return entry


#: (name, normalizer, added tokens, documents). Ported and condensed
#: from the prototype flag-probe battery: word boundaries, whitespace
#: attachment, the normalized face, special against ordinary, and
#: leftmost-longest overlap.
PROBES = [
    (
        "single_word",
        None,
        [added("cat", 0, single_word=True), added("dog", 1)],
        [
            "a cat b",
            "concatenate",
            "cat",
            "cat-dog",
            "cat9",
            "9cat",
            " cat ",
            "cat.",
            ".cat",
            "cat\n",
            "catcat",
            "dogcat",
            "cat dog cat",
            "cats",
            "scat",
            "_cat_",
        ],
    ),
    (
        "lstrip_rstrip",
        None,
        [
            added("<lp>", 0, lstrip=True),
            added("<rp>", 1, rstrip=True),
            added("<bp>", 2, lstrip=True, rstrip=True),
        ],
        [
            "x <lp>",
            "x\t\n <lp> y",
            "<lp>",
            "  <lp>",
            "<rp> x",
            "<rp>   y",
            "<rp>",
            "a<rp>  \n b",
            " <bp> ",
            "a  <bp>\t\tb",
            "<bp>",
            "x<lp><lp>y",
            "w  <bp>  <bp>  w",
        ],
    ),
    (
        "normalized_face",
        {"type": "Lowercase"},
        [
            added("QQ", 0, normalized=True),
            added("<nn>", 1, normalized=True),
            added("<RAW>", 2),
        ],
        [
            "a QQ b",
            "a qq b",
            "aQQb",
            "aqqb",
            "QQ",
            "qq",
            "Qq",
            "<nn>",
            "<NN>",
            "a<nn>B",
            "<RAW>",
            "<RAW>qq<RAW>",
            "MIXED Qq <RAW> QQ case",
        ],
    ),
    (
        "special_vs_ordinary",
        None,
        [added("<sp>", 0, special=True), added("<or>", 1)],
        [
            "a<sp>b",
            "a<or>b",
            "<sp><or>",
            "text <sp> tail",
            "x<sp",
            "<or",
            "<sp>>",
            "<<or>",
        ],
    ),
    (
        "leftmost_longest",
        None,
        [
            added("<t", 0),
            added("<t>", 1),
            added("<tt>", 2),
            added("ab", 3),
            added("abc", 4),
            added("bc", 5),
        ],
        [
            "<t>",
            "<tt>",
            "<t",
            "x<t>y",
            "x<tt>y",
            "<t<t>",
            "<t><tt>",
            "ab",
            "abc",
            "abcd",
            "aabc",
            "xabcbc",
            "ababc",
            "abcbc",
            "zab",
            "abz",
            "aabbcc",
        ],
    ),
]


def _pair(
    added_tokens: Sequence[dict[str, Any]], normalizer: Any
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """(full tokenizer, tokenizer without added tokens, effective table).

    The oracle re-assigns ids when it deserializes a constructed
    artifact, so the table records the ids the oracle actually assigned:
    what is under test is the split, not id bookkeeping.
    """
    document = support.byte_level_document(added_tokens, normalizer=normalizer)
    full = tokenizers.Tokenizer.from_str(json.dumps(document))
    without = json.loads(json.dumps(document))
    without["added_tokens"] = []
    bare = tokenizers.Tokenizer.from_str(json.dumps(without))
    effective = [
        {**token, "id": full.token_to_id(token["content"])} for token in added_tokens
    ]
    assert all(token["id"] is not None for token in effective)
    return full, bare, effective


def _differential(
    frontend: AddedTokenFrontend, full: Any, bare: Any, documents: Sequence[str]
) -> list[tuple[str, list[int], list[int]]]:
    """Documents where the frontend split does not reproduce the oracle."""
    bad: list[tuple[str, list[int], list[int]]] = []
    for text in documents:
        plan = frontend.scan(text)
        if plan is None:
            plan = [(text, None)]
        subject = assemble(
            plan, lambda segment: bare.encode(segment, add_special_tokens=False).ids
        )
        reference = list(full.encode(text, add_special_tokens=False).ids)
        if subject != reference:
            bad.append((text, subject, reference))
    return bad


@pytest.mark.parametrize(
    ("name", "normalizer", "added_tokens", "documents"),
    PROBES,
    ids=[probe[0] for probe in PROBES],
)
def test_flag_probes_reproduce_the_oracle(
    name: str,
    normalizer: Any,
    added_tokens: list[dict[str, Any]],
    documents: list[str],
) -> None:
    """Every flag combination splits exactly as the oracle does."""
    full, bare, effective = _pair(added_tokens, normalizer)
    frontend = AddedTokenFrontend(
        {"family": f"probe_{name}", "normalizer": normalizer,
         "added_tokens": effective},
        allow_nonidentity_norm_face=True,
    )
    assert _differential(frontend, full, bare, documents) == []


def test_prefilter_never_vetoes_a_real_literal() -> None:
    """The cheap stage is an over-approximation, never a filter.

    A document that the oracle would split has to reach the exact layer.
    The pool mixes literal fragments, near misses and random bytes.
    """
    import random

    added_tokens = [
        added("<|end|>", 0, special=True),
        added("[MASK]", 1),
        added("Z", 2),
        added("\u00e9\u00e8", 3),
    ]
    full, bare, effective = _pair(added_tokens, None)
    frontend = AddedTokenFrontend(
        {"family": "prefilter", "normalizer": None, "added_tokens": effective}
    )
    prefixes = frontend._native_prefilter_prefixes()
    assert prefixes is not None
    native_selector = _native.RouteSelector((0, 0), 1, False, 2, prefixes)
    fragments = [
        "<|end|>",
        "<|end",
        "|end|>",
        "[MASK]",
        "[MAS",
        "MASK]",
        "Z",
        "z",
        "\u00e9\u00e8",
        "\u00e9",
        "\u00e8",
        "abc",
        " ",
        "\n",
        "\U0001f642",
        "\u4e2d\u6587",
    ]
    rng = random.Random(7)
    documents = ["".join(rng.choice(fragments) for _ in range(rng.randint(0, 12)))
                 for _ in range(2000)]
    documents.extend(fragments)
    documents.append("")

    literal_ids = {token["id"] for token in effective}
    for text in documents:
        oracle_ids = set(full.encode(text, add_special_tokens=False).ids)
        oracle_found = bool(oracle_ids & literal_ids)
        if oracle_found:
            assert native_selector.route(text)[3] is True, text
            assert frontend.scan(text) is not None, text
    assert _differential(frontend, full, bare, documents) == []


def test_nonidentity_normalizer_with_normalized_face_is_refused() -> None:
    """An uncertified coordinate mapping is refused, not approximated."""
    with pytest.raises(UnsupportedConfig) as caught:
        AddedTokenFrontend(
            {
                "family": "refused",
                "normalizer": {"type": "Lowercase"},
                "added_tokens": [added("QQ", 0, normalized=True)],
            }
        )
    assert caught.value.code == "UNSUPPORTED_CONFIG"


def test_identity_normalizer_shapes_are_accepted() -> None:
    """An empty normalizer sequence is the identity, and is allowed."""
    _, _, effective = _pair([added("<x>", 0, normalized=True)], None)
    frontend = AddedTokenFrontend(
        {
            "family": "identity",
            "normalizer": {"type": "Sequence", "normalizers": []},
            "added_tokens": effective,
        }
    )
    assert frontend.scan("a<x>b") is not None


def test_assemble_joins_in_order() -> None:
    """Literals map straight to ids; segments go through the backend."""
    plan = [("ab", None), ("<x>", 7), ("", None), ("c", None)]
    assert assemble(plan, lambda segment: [len(segment)]) == [2, 7, 1]


# ---------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------


def _table(tmp_path: Path, **overrides: Any) -> Path:
    table: dict[str, Any] = {
        "table_version": "added-v1",
        "family": "test_family",
        "tokenizer_json_sha256": "a" * 64,
        "normalizer": None,
        "added_tokens": [added("<x>", 0)],
    }
    table.update(overrides)
    table["content_sha256"] = table_content_sha256(table)
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def test_table_round_trips_and_binds_to_its_artifact(tmp_path: Path) -> None:
    """A good table loads and can name the artifact it came from."""
    path = _table(tmp_path)
    loaded = load_table(path, expected_artifact_sha256="a" * 64)
    assert loaded["family"] == "test_family"


def test_damaged_table_is_refused(tmp_path: Path) -> None:
    """A table whose content hash does not match is never applied."""
    path = _table(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["added_tokens"][0]["id"] = 999
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ArtifactHashMismatch) as caught:
        load_table(path)
    assert caught.value.code == "ARTIFACT_HASH_MISMATCH"


def test_table_from_another_artifact_is_refused(tmp_path: Path) -> None:
    """A table that drifted from its artifact describes a different vocabulary."""
    path = _table(tmp_path)
    with pytest.raises(ArtifactHashMismatch):
        load_table(path, expected_artifact_sha256="b" * 64)


# ---------------------------------------------------------------------
# capability fingerprint
# ---------------------------------------------------------------------


def test_fingerprint_matches_the_specified_encoding() -> None:
    """The preimage is built exactly as fingerprint.md Sections 3 and 6."""
    import hashlib
    import struct

    token = added("<x>", 0, special=True)
    expected_preimage = b"toktier.added_frontend.v1\x00"
    expected_preimage += b"\x01" + struct.pack("<I", 1)
    expected_preimage += b"\x01" + b"<x>"
    expected_preimage += b"\x01" + struct.pack("<Q", token["id"])
    expected_preimage += b"\x01\x01"  # special
    expected_preimage += b"\x01\x00"  # single_word
    expected_preimage += b"\x01\x00"  # lstrip
    expected_preimage += b"\x01\x00"  # rstrip
    expected_preimage += b"\x01\x00"  # normalized
    assert (
        added_frontend_fingerprint([token])
        == hashlib.sha256(expected_preimage).hexdigest()
    )


def test_fingerprint_binds_flags_ids_and_order() -> None:
    """Anything that can change extraction changes the identity."""
    first = added("<a>", 0)
    second = added("<b>", 1)
    base = added_frontend_fingerprint([first, second])
    assert added_frontend_fingerprint([second, first]) != base
    assert added_frontend_fingerprint([{**first, "lstrip": True}, second]) != base
    assert added_frontend_fingerprint([{**first, "id": 5}, second]) != base
    assert added_frontend_fingerprint([first, second]) == base


# ---------------------------------------------------------------------
# frozen artifacts, when they are available
# ---------------------------------------------------------------------


def _frozen_tables() -> list[tuple[str, Path, Path]]:
    """(family, table path, artifact directory) for local frozen data."""
    tables_root = os.environ.get("TOKTIER_TEST_ADDED_TABLES")
    artifacts_root = os.environ.get("TOKTIER_TEST_ARTIFACTS")
    if not tables_root or not artifacts_root:
        return []
    import re

    revision = re.compile(r"-[0-9a-f]{12}$")
    directories = {
        revision.sub("", path.name): path
        for path in sorted(Path(artifacts_root).iterdir())
        if path.is_dir() and (path / "tokenizer.json").is_file()
    }
    found: list[tuple[str, Path, Path]] = []
    for table_path in sorted(Path(tables_root).glob("*.json")):
        family = table_path.stem
        directory = directories.get(family)
        if directory is not None:
            found.append((family, table_path, directory))
    return found


FROZEN_TABLES = _frozen_tables()


@pytest.mark.skipif(not FROZEN_TABLES, reason="no local added-token tables")
@pytest.mark.parametrize(
    ("family", "table_path", "directory"),
    FROZEN_TABLES,
    ids=[item[0] for item in FROZEN_TABLES],
)
def test_frozen_table_reproduces_the_oracle(
    family: str,
    table_path: Path,
    directory: Path,
    parity_documents: tuple[list[str], str],
) -> None:
    """Each shipped table splits its own artifact exactly as the oracle does."""
    import hashlib

    raw = (directory / "tokenizer.json").read_bytes()
    table = load_table(
        table_path, expected_artifact_sha256=hashlib.sha256(raw).hexdigest()
    )
    frontend = AddedTokenFrontend(table)
    assert frontend.capability_fingerprint()

    full = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    document = json.loads(raw.decode("utf-8"))
    document["added_tokens"] = []
    bare = tokenizers.Tokenizer.from_str(json.dumps(document))

    documents, _ = parity_documents
    sample = list(documents[:200])
    sample.extend(
        token["content"] for token in table["added_tokens"][:20]
    )
    sample.extend(
        "left " + str(token["content"]) + " right"
        for token in table["added_tokens"][:20]
    )
    bad = _differential(frontend, full, bare, sample)
    assert bad == [], f"{family}: {len(bad)} documents differ"
