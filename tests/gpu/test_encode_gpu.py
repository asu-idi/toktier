"""End-to-end GPU tests: kernel output against the reference tokenizer.

Every test here is marked ``gpu`` and is skipped unless a CUDA device,
torch and the frozen artifacts are all present. They are the suite the
mainline runs on hardware.

Run them with::

    pytest tests/gpu -m gpu \\
        --artifact-manifest /path/to/tokenizer_manifest.json \\
        --class-table-dir /path/to/generated/tables

The reference is the frozen artifact read directly by the reference
tokenizer package, with ``add_special_tokens=False``: that is the same
comparison the certification runs used, so a disagreement here is a
disagreement with the published numbers.

Sample texts are written with escapes rather than literal characters so
the file stays ASCII, which the repository's hygiene scan requires.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

pytestmark = pytest.mark.gpu


#: Small but deliberately awkward inputs. Each line exercises something
#: the splitter rules treat specially.
SAMPLES: tuple[str, ...] = (
    "hello world",
    "The quick brown fox jumps over the lazy dog. 1234567890",
    "it's a test, isn't it? we've seen 'em all: 'tis, 'Twas, '\u017fx",
    "def main():\n    return {'a': 1, 'b': [2, 3]}\n\n\n",
    "trailing spaces   \nand a tab\there\r\n\r\nCRLF runs\r\r\n",
    "\u4e2d\u6587\u6df7\u6392 english \u65e5\u672c\u8a9e "
    "\u0410\u043b\u0444\u0430\u0432\u0438\u0442",
    "combining: e\u0301 a\u030a o\u0308, precomposed: \u00e9 \u00e5 \u00f6",
    "numbers 1 12 123 1234 12345 and 0.5 and 1,000,000",
    "emoji \U0001f600\U0001f469\u200d\U0001f4bb symbols \u2192\u2264\u221e",
    "   ",
    "\n",
    "a",
    "\x00\x01 control bytes \x7f",
    "mixed\u00a0nbsp\u3000ideographic\u2028line separator",
)


def _reference_encoder(handle: Any) -> Any:
    """The reference tokenizer over the same verified bytes."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(handle.path("tokenizer.json")))

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False).ids)

    return encode


def _e2e_families(engine: Any, handles: dict[str, Any]) -> list[str]:
    return [
        name
        for name in engine.families.names()
        if engine.families.supports_e2e(name) and name in handles
    ]


def _samples_without_added_literals(encoder: Any) -> list[str]:
    """Drop samples that contain an added-token literal.

    With the added-token frontend switched off, such a document is the
    routing layer's business: it goes to the reference backend and is
    counted, and comparing the kernel against the reference on it would
    be comparing two different code paths.
    """
    literals = [literal for literal in encoder.added if literal]
    return [
        text
        for text in SAMPLES
        if text and not any(literal in text for literal in literals)
    ]


# -- loader and certificate ---------------------------------------------


def test_engine_builds_the_kernel_once(gpu_engine: Any) -> None:
    from toktier.engine.gpu.loader import KernelLoader

    assert KernelLoader.is_loaded()
    assert KernelLoader.certificate_void() is False
    binding = gpu_engine.binding_set()
    # Bound fields in schema spelling: bare 64-hex digests.
    assert len(binding["source_digest"]) == 64
    assert len(binding["family_table_digest"]) == 64
    assert binding["build_flags"] == ["-O3"]
    assert binding["build_flags_digest"].startswith("sha256:")
    assert binding["toolchain_facts"]["device_capability"].startswith("sm_")
    assert binding["certificate_void"] is False


def test_class_tables_verify(gpu_engine: Any) -> None:
    """Every table the routing data names resolves and hashes cleanly."""
    observed = gpu_engine.class_tables.observed_digests()
    for table_id, digest in observed.items():
        if digest is None:
            pytest.skip(f"generated table {table_id} is not present")
        assert digest.startswith("sha256:")


# -- correctness ---------------------------------------------------------


def test_encode_matches_the_reference(
    gpu_engine: Any, artifact_handles: dict[str, Any]
) -> None:
    families = _e2e_families(gpu_engine, artifact_handles)
    assert families, "no certified end-to-end family is present in the manifest"
    for family in families:
        encoder = gpu_engine.encoder(family, kind="eager")
        reference = _reference_encoder(artifact_handles[family])
        for text in _samples_without_added_literals(encoder):
            assert encoder.encode(text) == reference(text), (family, text)


def test_delivery_forms_agree(
    gpu_engine: Any, artifact_handles: dict[str, Any]
) -> None:
    """Eager, fused and graph forms are delivery choices, not semantics."""
    for family in _e2e_families(gpu_engine, artifact_handles):
        encoders = {
            kind: gpu_engine.encoder(family, kind=kind)
            for kind in ("eager", "fused", "graph")
        }
        for text in _samples_without_added_literals(encoders["eager"]):
            ids = {
                kind: encoder.encode(text) for kind, encoder in encoders.items()
            }
            assert ids["eager"] == ids["fused"] == ids["graph"], (family, text)


def test_numpy_delivery_matches_list_delivery(
    gpu_engine: Any, artifact_handles: dict[str, Any]
) -> None:
    for family in _e2e_families(gpu_engine, artifact_handles):
        encoder = gpu_engine.encoder(family, kind="graph")
        for text in _samples_without_added_literals(encoder):
            assert encoder.encode_np(text).tolist() == encoder.encode(text)


# -- the batched channel -------------------------------------------------


def test_batched_matches_per_document(
    gpu_engine: Any, artifact_handles: dict[str, Any]
) -> None:
    """The batched channel's equivalence argument, measured.

    The argument is that piece starts are forced at document boundaries
    and that merging is closed inside a piece. This checks the
    conclusion element by element, including a batch whose documents
    have been ordered to put a combining mark right after a boundary.
    """
    for family in _e2e_families(gpu_engine, artifact_handles):
        encoder = gpu_engine.encoder(family, kind="eager")
        batched = gpu_engine.batched(family)
        docs = [
            text for text in _samples_without_added_literals(encoder) if text
        ]
        rows = batched.encode_batch(docs)
        assert len(rows) == len(docs)
        for text, row in zip(docs, rows, strict=True):
            assert row.tolist() == encoder.encode(text), (family, text)


def test_batched_boundary_stress(
    gpu_engine: Any, artifact_handles: dict[str, Any]
) -> None:
    """Adjacent documents that would compose or merge if concatenated."""
    tricky = [
        "e",
        "\u0301combining lead",
        "trailing space ",
        " leading space",
        "abc",
        "def",
        "\r",
        "\n",
        "1",
        "2",
        "3",
    ]
    for family in _e2e_families(gpu_engine, artifact_handles):
        encoder = gpu_engine.encoder(family, kind="eager")
        batched = gpu_engine.batched(family)
        rows = batched.encode_batch(tricky)
        for text, row in zip(tricky, rows, strict=True):
            assert row.tolist() == encoder.encode(text), (family, text)


def test_ragged_shape_satisfies_the_contract(
    gpu_engine: Any, artifact_handles: dict[str, Any]
) -> None:
    """The frozen invariants of the ragged batch output."""
    import numpy as np

    for family in _e2e_families(gpu_engine, artifact_handles):
        batched = gpu_engine.batched(family)
        docs = [text for text in SAMPLES if text]
        values, offsets = batched.encode_batch_ragged(docs)
        assert values.dtype == np.uint32
        assert offsets.dtype == np.int64
        assert offsets.shape == (len(docs) + 1,)
        assert int(offsets[0]) == 0
        assert int(offsets[-1]) == values.size
        assert bool((np.diff(offsets) >= 0).all())
        rows = batched.encode_batch(docs)
        for index, row in enumerate(rows):
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            assert values[lo:hi].tolist() == row.tolist()


def test_digest_convention_is_the_published_one(
    gpu_engine: Any, artifact_handles: dict[str, Any]
) -> None:
    """Document digests reproduce the published construction exactly."""
    import numpy as np

    family = _e2e_families(gpu_engine, artifact_handles)[0]
    batched = gpu_engine.batched(family)
    docs = [text for text in SAMPLES if text]
    digests = batched.digest_batch(docs)
    rows = batched.encode_batch(docs)
    for digest, row in zip(digests, rows, strict=True):
        expected = hashlib.sha256(
            np.asarray(row, dtype="<u4").tobytes()
        ).digest()
        assert digest == expected
        assert len(digest) == 32


# -- honest refusals -----------------------------------------------------


def test_split_only_family_refuses_end_to_end(gpu_engine: Any) -> None:
    """A split-only band must refuse, not silently produce something."""
    from toktier.errors import UncertifiedTokenizer

    split_only = [
        name
        for name in gpu_engine.families.names()
        if not gpu_engine.families.supports_e2e(name)
    ]
    for family in split_only:
        with pytest.raises(UncertifiedTokenizer):
            gpu_engine.encoder(family)


def test_unknown_family_refuses(gpu_engine: Any) -> None:
    from toktier.errors import UncertifiedTokenizer

    with pytest.raises(UncertifiedTokenizer):
        gpu_engine.encoder("not_a_real_family")


def test_unknown_encoder_kind_refuses(
    gpu_engine: Any, artifact_handles: dict[str, Any]
) -> None:
    from toktier.errors import UnsupportedConfig

    family = _e2e_families(gpu_engine, artifact_handles)[0]
    with pytest.raises(UnsupportedConfig):
        gpu_engine.encoder(family, kind="turbo")
