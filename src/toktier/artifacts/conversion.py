"""Artifacts that are converted locally rather than downloaded whole.

Most families publish ``tokenizer.json`` upstream, so fetching one is a
download plus a digest check. One family does not: the Kimi lineage
publishes a tiktoken rank file and its own Python tokenizer, and the
executable Hugging Face artifact is derived from those bytes.

This module keeps that derivation honest by making it a *pinned*
computation rather than a redistribution:

1. every upstream input is fetched at a pinned repository revision and
   checked against a pinned sha256 before it is read;
2. the conversion is deterministic and depends on nothing but those
   bytes (no clock, no environment, no set iteration order);
3. the produced ``tokenizer.json`` is handed to
   :mod:`toktier.artifacts.store`, which verifies it against the digest
   the shipped artifact manifest pins -- the same check every downloaded
   artifact passes. A converter that drifted by one byte fails there.

The upstream inputs are read from a private temporary directory and are
never installed into the cache: this package distributes no upstream
file, only the recipe that reproduces the certified artifact.

Which families are derived, and from which pinned inputs, is shipped
data (``tables/artifact_conversions.v1.json``): this module implements
the transformations that table selects from and names no family itself,
so the routing data stays the only place a family is named.

This module is dependency-free (standard library only) and imports no
tokenizer runtime: the conversion is a data transformation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ArtifactNotFound, RegistryInvalid
from .manifest import ArtifactEntry, ArtifactFile
from .sources import ArtifactSource
from .tables import ARTIFACT_CONVERSIONS

__all__ = [
    "CONVERSIONS",
    "ConversionRecipe",
    "ConvertingSource",
    "conversion_report",
    "convert_kimi_tiktoken",
    "load_conversions",
    "recipe_for",
]

#: File the conversion produces; the only file a converted family pins.
TOKENIZER_JSON = "tokenizer.json"

#: Number of reserved ids appended above the tiktoken rank space. Kimi
#: publishes the named ones in ``tokenizer_config.json`` and leaves the
#: rest reserved; both spellings land in ``added_tokens``.
KIMI_RESERVED_TOKENS = 256

#: The pre-tokenizer pattern of the Kimi lineage. It is carried here as
#: a constant *and* re-derived from the pinned upstream
#: ``tokenization_kimi.py`` at conversion time, so the constant can never
#: quietly disagree with the file it came from.
KIMI_PATTERN = "|".join(
    (
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    )
)

#: Where the pattern lives in the upstream module, as a pair of regular
#: expressions: the assignment block, then each raw-string branch in it.
_UPSTREAM_PATTERN_BLOCK = re.compile(
    r'pat_str = "\|"\.join\(\(?\[(.*?)\]\)?\)', re.DOTALL
)
_UPSTREAM_PATTERN_BRANCH = re.compile(r'r"""(.*?)"""', re.DOTALL)


@dataclass(frozen=True)
class ConversionRecipe:
    """How one family's artifact is derived from pinned upstream bytes.

    The repository and revision are not repeated here: they come from
    the artifact manifest entry being fetched, so there is one place
    that says where the upstream bytes are frozen.
    """

    #: Canonical family id this recipe produces.
    family: str
    #: Upstream files the conversion reads, with their pinned digests.
    inputs: tuple[ArtifactFile, ...]
    #: Name of the deterministic transformation, for diagnostics.
    converter: str
    #: Upstream licence file name, recorded so the note in the docs can
    #: be checked against the repository it describes.
    upstream_license: str

    def input_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.inputs)


def _bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte-to-character bijection used by byte-level BPE.

    Written with ordinals rather than literal characters so the source
    stays plain ASCII; the mapping is the usual one.
    """
    printable = (
        list(range(0x21, 0x7E + 1))  # '!' through '~'
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    mapped = list(printable)
    overflow = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            mapped.append(256 + overflow)
            overflow += 1
    return {value: chr(code) for value, code in zip(printable, mapped, strict=True)}


def _load_tiktoken_ranks(payload: bytes) -> dict[bytes, int]:
    """Parse a tiktoken rank file, preserving its line order.

    The file is one ``<base64 token> <rank>`` pair per line. Insertion
    order is the file order, and the conversion depends on it: the merge
    list is built by walking the ranks and is sorted stably afterwards.
    """
    ranks: dict[bytes, int] = {}
    for number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise RegistryInvalid(
                "the upstream rank file has a line that is not a "
                "token/rank pair",
                details={"line": number},
            )
        token, rank = fields
        try:
            decoded = base64.b64decode(token, validate=True)
            value = int(rank)
        except (ValueError, TypeError) as error:
            raise RegistryInvalid(
                "the upstream rank file has an unreadable token/rank pair",
                details={"line": number, "failure": str(error)},
            ) from error
        ranks[decoded] = value
    return ranks


def _upstream_pattern(module_source: str) -> str:
    """Rebuild the pre-tokenizer pattern from the upstream module text.

    The module is parsed as text and never imported or executed.
    """
    block = _UPSTREAM_PATTERN_BLOCK.search(module_source)
    if block is None:
        raise RegistryInvalid(
            "the upstream tokenizer module does not define the pattern in "
            "the expected form"
        )
    branches = _UPSTREAM_PATTERN_BRANCH.findall(block.group(1))
    if not branches:
        raise RegistryInvalid("the upstream pattern block lists no branches")
    return "|".join(branches)


def convert_kimi_tiktoken(sources: Mapping[str, bytes]) -> bytes:
    """Derive the Kimi ``tokenizer.json`` from pinned upstream bytes.

    ``sources`` maps each recipe input name to its verified content.
    The return value is the exact byte string the artifact manifest
    pins; the caller does not reformat it.

    Structure, and why each piece is what it is (the choices follow the
    already certified tiktoken-family artifacts this package ships):

    * the pre-tokenizer is a ``Split`` on the upstream pattern followed
      by a byte-level step with the pattern disabled, so the split is
      done once, by the upstream pattern;
    * the model is byte-level BPE with ``ignore_merges``: tiktoken looks
      a whole piece up before it merges anything, and ``ignore_merges``
      is the equivalent of that lookup;
    * merges are the exhaustive two-way decompositions of every
      multi-byte token, ordered by the rank of the token they produce,
      which is the same order tiktoken applies them in;
    * the reserved ids above the rank space become ``added_tokens``,
      taking their names and special flags from the upstream
      configuration and a reserved spelling otherwise.
    """
    pattern = _upstream_pattern(
        sources["tokenization_kimi.py"].decode("utf-8")
    )
    if pattern != KIMI_PATTERN:
        raise RegistryInvalid(
            "the upstream pre-tokenizer pattern differs from the pattern "
            "this release converts with",
            details={
                "upstream_sha256": hashlib.sha256(pattern.encode()).hexdigest(),
                "recipe_sha256": hashlib.sha256(
                    KIMI_PATTERN.encode()
                ).hexdigest(),
            },
        )

    ranks = _load_tiktoken_ranks(sources["tiktoken.model"])
    base_count = len(ranks)
    if sorted(ranks.values()) != list(range(base_count)):
        raise RegistryInvalid("the upstream ranks are not a dense range")
    missing = [value for value in range(256) if bytes([value]) not in ranks]
    if missing:
        raise RegistryInvalid(
            "the upstream ranks do not cover every single byte",
            details={"missing_bytes": missing[:8]},
        )

    table = _bytes_to_unicode()

    def spell(token: bytes) -> str:
        return "".join(table[value] for value in token)

    vocabulary = {spell(token): rank for token, rank in ranks.items()}
    if len(vocabulary) != base_count:
        raise RegistryInvalid("the byte-to-character spelling is not injective")

    merges: list[tuple[int, str, str]] = []
    for token, rank in ranks.items():
        if len(token) == 1:
            continue
        for cut in range(1, len(token)):
            left, right = token[:cut], token[cut:]
            if left in ranks and right in ranks:
                merges.append((rank, spell(left), spell(right)))
    merges.sort(key=lambda item: item[0])
    unreachable = base_count - 256 - len({rank for rank, _, _ in merges})

    configuration = json.loads(sources["tokenizer_config.json"])
    named = {
        int(key): value
        for key, value in configuration["added_tokens_decoder"].items()
    }
    added: list[dict[str, Any]] = []
    for identifier in range(base_count, base_count + KIMI_RESERVED_TOKENS):
        entry = named.get(identifier)
        added.append(
            {
                "id": identifier,
                "content": (
                    entry["content"]
                    if entry
                    else f"<|reserved_token_{identifier}|>"
                ),
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": bool(entry["special"]) if entry else True,
            }
        )

    document = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": KIMI_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocabulary,
            "merges": [[left, right] for _, left, right in merges],
        },
    }
    if unreachable:
        raise RegistryInvalid(
            "some upstream ranks have no two-way decomposition",
            details={"unreachable_tokens": unreachable},
        )
    return (json.dumps(document, ensure_ascii=False) + "\n").encode("utf-8")


#: Converters by name, so a recipe names its transformation as data.
_CONVERTERS = {"kimi_tiktoken_v1": convert_kimi_tiktoken}


def _parse_input(family: str, raw: Any, *, position: int) -> ArtifactFile:
    """One upstream input of a shipped recipe, or a closed failure."""
    if not isinstance(raw, Mapping):
        raise RegistryInvalid(
            "a conversion input must be a table",
            details={"family": family, "position": position},
        )
    name, digest, size = raw.get("name"), raw.get("sha256"), raw.get("size")
    if not isinstance(name, str) or not name:
        raise RegistryInvalid(
            "a conversion input must name the upstream file it reads",
            details={"family": family, "position": position},
        )
    if not isinstance(digest, str) or len(digest) != 64:
        raise RegistryInvalid(
            "a conversion input must pin a sha256 of its upstream file",
            details={"family": family, "file": name},
        )
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise RegistryInvalid(
            "a conversion input must pin the byte length of its upstream file",
            details={"family": family, "file": name},
        )
    return ArtifactFile(name=name, sha256=digest, size=size)


def _parse_recipes(document: Any, *, source: Path) -> dict[str, ConversionRecipe]:
    """Validate the shipped conversion table (fail closed)."""
    if not isinstance(document, Mapping):
        raise RegistryInvalid(
            "the conversion table must be a JSON object",
            details={"source": str(source)},
        )
    raw_conversions = document.get("conversions")
    if not isinstance(raw_conversions, Mapping):
        raise RegistryInvalid(
            "the conversion table must carry a 'conversions' object",
            details={"source": str(source)},
        )
    recipes: dict[str, ConversionRecipe] = {}
    for family, raw in raw_conversions.items():
        if not isinstance(family, str) or not family:
            raise RegistryInvalid(
                "family ids must be non-empty strings",
                details={"source": str(source)},
            )
        if not isinstance(raw, Mapping):
            raise RegistryInvalid(
                "a conversion entry must be a table",
                details={"family": family, "source": str(source)},
            )
        converter = raw.get("converter")
        if converter not in _CONVERTERS:
            raise RegistryInvalid(
                "a conversion entry must name a transformation this "
                "release implements",
                details={
                    "family": family,
                    "converter": converter,
                    "implemented": sorted(_CONVERTERS),
                },
            )
        licence = raw.get("upstream_license")
        if not isinstance(licence, str) or not licence:
            raise RegistryInvalid(
                "a conversion entry must name the upstream licence file",
                details={"family": family},
            )
        raw_inputs = raw.get("inputs")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise RegistryInvalid(
                "a conversion entry must list the upstream inputs it reads",
                details={"family": family},
            )
        inputs = tuple(
            _parse_input(family, item, position=position)
            for position, item in enumerate(raw_inputs)
        )
        if len({item.name for item in inputs}) != len(inputs):
            raise RegistryInvalid(
                "a conversion entry must not list an upstream file twice",
                details={"family": family},
            )
        recipes[family] = ConversionRecipe(
            family=family,
            inputs=inputs,
            converter=str(converter),
            upstream_license=licence,
        )
    return recipes


def load_conversions(path: str | Path = ARTIFACT_CONVERSIONS) -> dict[
    str, ConversionRecipe
]:
    """Read and validate a conversion table from disk."""
    table_path = Path(path)
    try:
        raw_text = table_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RegistryInvalid(
            f"cannot read the conversion table: {error}",
            details={"source": str(table_path)},
        ) from error
    try:
        document = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise RegistryInvalid(
            f"cannot parse the conversion table: {error}",
            details={"source": str(table_path)},
        ) from error
    return _parse_recipes(document, source=table_path)


#: Recipes by family, read from the shipped table. The table is the only
#: place a converted family is named; this module implements the
#: transformations that table selects from.
CONVERSIONS: dict[str, ConversionRecipe] = load_conversions()


def recipe_for(family: str) -> ConversionRecipe | None:
    """The conversion recipe of ``family``, or ``None`` when it has none."""
    return CONVERSIONS.get(family)


def _verify(artifact_file: ArtifactFile, payload: bytes, *, family: str) -> None:
    """Refuse upstream bytes that are not the pinned ones."""
    observed = hashlib.sha256(payload).hexdigest()
    if artifact_file.size is not None and artifact_file.size != len(payload):
        raise ArtifactNotFound(
            f"upstream input {artifact_file.name!r} of {family!r} has an "
            "unexpected byte length",
            details={
                "family": family,
                "file": artifact_file.name,
                "expected_size": artifact_file.size,
                "observed_size": len(payload),
            },
        )
    if observed != artifact_file.sha256:
        raise ArtifactNotFound(
            f"upstream input {artifact_file.name!r} of {family!r} does not "
            "match the digest this release converts from",
            details={
                "family": family,
                "file": artifact_file.name,
                "expected_sha256": artifact_file.sha256,
                "observed_sha256": observed,
            },
        )


def _upstream_entry(
    entry: ArtifactEntry, recipe: ConversionRecipe
) -> ArtifactEntry:
    """The manifest entry describing the recipe's upstream inputs."""
    return ArtifactEntry(
        family=entry.family,
        repo_id=entry.repo_id,
        revision=entry.revision,
        files=recipe.inputs,
        local_dir=entry.local_dir,
    )


def read_upstream_inputs(
    entry: ArtifactEntry,
    recipe: ConversionRecipe,
    source: ArtifactSource,
    *,
    workspace: Path,
) -> dict[str, bytes]:
    """Fetch and verify every upstream input of one recipe."""
    upstream = _upstream_entry(entry, recipe)
    payloads: dict[str, bytes] = {}
    for artifact_file in recipe.inputs:
        destination = workspace / artifact_file.name.replace("/", "__")
        source.fetch(upstream, artifact_file, destination)
        payload = destination.read_bytes()
        _verify(artifact_file, payload, family=entry.family)
        payloads[artifact_file.name] = payload
    return payloads


class ConvertingSource:
    """Produces converted artifacts, delegating everything else.

    The wrapped source supplies the bytes; this class decides, per
    family, whether those bytes are the artifact itself or the inputs a
    pinned conversion reads. Families without a recipe pass straight
    through, so wrapping is safe for the whole manifest.
    """

    def __init__(
        self,
        base: ArtifactSource,
        *,
        conversions: Mapping[str, ConversionRecipe] | None = None,
    ) -> None:
        self._base = base
        self._conversions = (
            dict(CONVERSIONS) if conversions is None else dict(conversions)
        )

    @property
    def name(self) -> str:
        """The wrapped source's name.

        A source name answers "where do the bytes come from", and that
        answer is unchanged by wrapping: the upstream inputs still come
        from the wrapped source. Which families are derived rather than
        downloaded is a property of the manifest, reported by
        ``artifacts check-conversion`` rather than folded into this name.
        """
        return self._base.name

    @property
    def offline(self) -> bool:
        return bool(self._base.offline)

    @property
    def base(self) -> ArtifactSource:
        return self._base

    def recipe(self, family: str) -> ConversionRecipe | None:
        return self._conversions.get(family)

    def fetch(
        self,
        entry: ArtifactEntry,
        artifact_file: ArtifactFile,
        destination: Path,
    ) -> None:
        recipe = self._conversions.get(entry.family)
        if recipe is None:
            self._base.fetch(entry, artifact_file, destination)
            return
        if artifact_file.name != TOKENIZER_JSON:
            raise ArtifactNotFound(
                f"the {entry.family!r} conversion produces "
                f"{TOKENIZER_JSON!r}, not {artifact_file.name!r}",
                details={
                    "family": entry.family,
                    "file": artifact_file.name,
                    "converter": recipe.converter,
                },
            )
        destination.write_bytes(self.convert(entry, recipe))

    def convert(
        self, entry: ArtifactEntry, recipe: ConversionRecipe
    ) -> bytes:
        """Run one recipe end to end, discarding the upstream inputs."""
        with tempfile.TemporaryDirectory(prefix=".toktier-convert-") as staging:
            payloads = read_upstream_inputs(
                entry, recipe, self._base, workspace=Path(staging)
            )
        return _CONVERTERS[recipe.converter](payloads)


def conversion_report(
    entry: ArtifactEntry,
    recipe: ConversionRecipe,
    source: ArtifactSource,
    *,
    repeats: int = 2,
) -> dict[str, Any]:
    """Run the conversion gate and report what it observed.

    Three claims are checked, all of them locally reproducible:

    * **determinism** -- the conversion is run ``repeats`` times from the
      same pinned inputs and every run must produce the same bytes;
    * **identity** -- those bytes must be the digest the shipped artifact
      manifest pins for this family;
    * **added tokens** -- the reserved-id block must be the contiguous,
      fully described table the certified artifact carries.

    The upstream-equivalence campaign behind the pinned digest is
    recorded evidence, not something this gate re-runs: it needs the
    upstream tokenizer runtime, which this package does not depend on.
    Binding the produced bytes to the pinned digest is what ties this
    installation to that campaign.
    """
    converting = ConvertingSource(source, conversions={recipe.family: recipe})
    digests: list[str] = []
    payload = b""
    for _ in range(max(1, repeats)):
        payload = converting.convert(entry, recipe)
        digests.append(hashlib.sha256(payload).hexdigest())
    expected = entry.file(TOKENIZER_JSON)
    document = json.loads(payload)
    added = document.get("added_tokens") or []
    identifiers = [int(item["id"]) for item in added]
    contiguous = bool(identifiers) and identifiers == list(
        range(identifiers[0], identifiers[0] + len(identifiers))
    )
    described = all(
        set(item)
        >= {
            "id",
            "content",
            "special",
            "single_word",
            "lstrip",
            "rstrip",
            "normalized",
        }
        for item in added
    )
    return {
        "family": entry.family,
        "converter": recipe.converter,
        "upstream_repo": entry.repo_id,
        "upstream_revision": entry.revision,
        "upstream_inputs": [
            {"name": item.name, "sha256": item.sha256, "size": item.size}
            for item in recipe.inputs
        ],
        "runs": len(digests),
        "deterministic": len(set(digests)) == 1,
        "observed_sha256": digests[0],
        "expected_sha256": expected.sha256,
        "observed_size": len(payload),
        "expected_size": expected.size,
        "identity_matches": (
            digests[0] == expected.sha256
            and (expected.size is None or expected.size == len(payload))
        ),
        "added_tokens": len(added),
        "added_tokens_special": sum(1 for item in added if item["special"]),
        "added_tokens_contiguous": contiguous,
        "added_tokens_fully_described": described,
        "added_tokens_first_id": identifiers[0] if identifiers else None,
        "normalizer": document.get("normalizer"),
    }
