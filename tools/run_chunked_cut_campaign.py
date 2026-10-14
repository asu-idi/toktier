#!/usr/bin/env python3
"""Judge the chunk-parallel path document by document, four ways.

This is the tool that produces the evidence behind
``src/toktier/routing/tables/chunked_admission.v1.json``. For every
document of a corpus and every family asked for it compares four id
sequences:

* the Python chunked encode, ``toktier.chunked.encode(..., native=False)``;
* the native chunked encode, ``toktier.chunked.encode(..., native=True)``;
* the serial route, ``Tokenizer.encode(text, lookup="off")``;
* the oracle package on the loader face, ``tokenizers.Tokenizer.from_str``.

Chunking is forced: ``k = max(2, ceil(chars / chunk_chars))`` with a small
``min_chunk``, so every document crosses the cut predicate (the shipped
chunk table would keep short documents in one chunk, which judges nothing).
Documents outside the length gate are skipped and counted. Any inequality
is a finding: the document is saved and the process ends with status 3.

Three subcommands:

``run``
    One work unit of the corpus campaign: the files of one shard and,
    optionally, one part of it (``--part-docs`` accepted documents,
    ``--part-index`` passes over the earlier parts without encoding). The
    unit's shard JSON records what ran (versions, identities, host, job,
    arguments, the files read), the document stream it judged (counts and
    a digest), the planner counters per family and the four mismatch
    columns.
``adversarial``
    The adversarial normalization battery: ``--strings`` generated strings
    per family from ``tools/chunked_adversarial_corpus.py`` (the generator
    the chunked tests use), judged the same four ways.
``summarize``
    Merges unit JSONs (and a battery JSON) into
    ``readings/chunked_cuts_campaign.json``, the readings file
    ``tools/update_chunked_admission.py`` derives the table from. A unit
    may be split over several files, one per family group; the files of
    one unit must report the same document stream and disjoint families,
    and the documents are counted once.

Document order comes from a content listing, never from scanning a
directory: a ``parquet`` or ``texts`` source is ``kind:<root>:<listing>``,
where the listing names the files in the order the campaign uses, one
relative path per line (``sha256sum`` lines are accepted). A shard reads the
listing positions ``p`` with ``p % num_shards == shard_index``, per source,
sources in command-line order; a selected file that is missing under the
root stops the unit, while files the shard does not select need not be
present. ``jsonl:<file>`` takes every ``num_shards``-th line.

Evidence collection runs with the admission table's step lifted for the
tokenizers under judgement (``toktier.chunked._planner_for(tokenizer,
evidence_gate=False)``): the families a campaign is meant to admit have no
row yet. The roster and normalizer steps still apply, and every unit JSON
records ``admission_gate`` per family. With ``--native on|both`` the tool
insists on a plan that starts on the corrected CPU engine and stops
otherwise (status 4): a native call on a reference-only route is encoded
serially under ``R_CHUNK_ROUTE_NOT_FAST_CPU``, which is not evidence for
the native path.

Example (one unit of 100,000 documents, both chunked paths)::

    python tools/run_chunked_cut_campaign.py run \\
        --source texts:/corpus/texts:/corpus/lists/texts.txt \\
        --source parquet:/corpus/parquet:/corpus/lists/parquet.txt \\
        --families qwen3_8b,llama_3_1_8b --shard-index 0 --num-shards 128 \\
        --part-docs 100000 --part-index 3 --native both --native-workers 1 \\
        --mismatch-dir out/mismatch --out out/shard_0000_part3_g0.json

Status: 0 all equal; 2 a family could not be loaded or admitted, or a
source is unusable; 3 at least one inequality; 4 the native arm was asked
for on a route that cannot run it. ``RAYON_NUM_THREADS=1`` and
``TOKENIZERS_PARALLELISM=false`` are applied when unset, for one
single-threaded process per core.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import importlib
import json
import os
import platform
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOL_NAME = "tools/run_chunked_cut_campaign.py"
TOOL_VERSION = "1.0.0"
SHARD_SCHEMA = "toktier.chunked_cuts_shard.v1"
BATTERY_SCHEMA = "toktier.chunked_cuts_adversarial.v1"
READINGS_SCHEMA = "toktier.chunked_cuts_campaign.v1"
MISMATCH_COLUMNS = (
    "python_vs_oracle",
    "native_vs_oracle",
    "python_vs_native",
    "serial_vs_oracle",
)
#: The chunked source identity, recomputed from the installed package
#: (``tools/source_identity_common.py``, ``TOOL_SIDE_IDENTITIES["chunked"]``):
#: the same domain and path-bound digest, with the package files standing in
#: for their repository paths.
CHUNKED_IDENTITY_DOMAIN = b"toktier.chunked.source.v1\0"
CHUNKED_IDENTITY_FILES = (
    ("src/toktier/chunked.py", "chunked.py"),
    (
        "src/toktier/repair/tables/fast_repair_families.v1.json",
        "repair/tables/fast_repair_families.v1.json",
    ),
    (
        "src/toktier/repair/tables/repair_pclass.v1.zlib",
        "repair/tables/repair_pclass.v1.zlib",
    ),
)
SLURM_KEYS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
ENV_KEYS = ("RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

STATUS_OK = 0
STATUS_SETUP = 2
STATUS_MISMATCH = 3
STATUS_NATIVE_ROUTE = 4


class SetupError(Exception):
    """A family could not be loaded or admitted, or an argument is unusable."""


class NativeRouteError(SetupError):
    """The native arm was asked for on a route that cannot run it."""


# -- identities and environment ------------------------------------------------


def tool_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def installed_chunked_identity() -> str:
    """``sha256:<digest>`` of the chunked sources as installed."""
    import toktier

    package = Path(toktier.__file__).resolve().parent
    digest = hashlib.sha256(CHUNKED_IDENTITY_DOMAIN)
    for rendered_name, installed in CHUNKED_IDENTITY_FILES:
        rendered = rendered_name.encode()
        content = (package / installed).read_bytes()
        digest.update(len(rendered).to_bytes(8, "little"))
        digest.update(rendered)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def native_facts() -> tuple[str | None, dict[str, Any] | None]:
    """The native extension's file digest and its fast_cpu build facts."""
    try:
        native = importlib.import_module("toktier._native")
    except ImportError:
        return None, None
    path = getattr(native, "__file__", None)
    file_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    facts = getattr(native, "fast_cpu_build_facts", None)
    build: dict[str, Any] | None = None
    if callable(facts):
        with contextlib.suppress(Exception):
            build = dict(facts())
    return file_digest, build


def _version(module: str) -> str | None:
    try:
        return str(getattr(importlib.import_module(module), "__version__", None))
    except ImportError:
        return None


def manifest(
    *, argv: list[str], started: float, tag: str | None, policy: str | None
) -> dict[str, Any]:
    import toktier

    extension_sha256, build = native_facts()
    return {
        "toktier_version": toktier.__version__,
        "toktier_file": toktier.__file__,
        "native_extension_sha256": extension_sha256,
        "chunked_identity": installed_chunked_identity(),
        "fast_cpu_source_digest": (build or {}).get("source_digest"),
        "fast_cpu_build_facts": build,
        "policy": policy,
        "tokenizers": _version("tokenizers"),
        "transformers": _version("transformers"),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "hostname": platform.node(),
        "pid": os.getpid(),
        "slurm": {key: os.environ.get(key) for key in SLURM_KEYS},
        "env": {key: os.environ.get(key) for key in ENV_KEYS},
        "argv": list(argv),
        "tag": tag,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }


# -- corpus sources -------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One ``--source`` of a unit, resolved to the files this shard reads."""

    spec: str
    kind: str
    root: Path | None
    listing: Path | None
    listing_sha256: str | None
    files: tuple[Path, ...]

    def record(self) -> dict[str, Any]:
        root = self.root
        return {
            "spec": self.spec,
            "kind": self.kind,
            "root": None if root is None else str(root),
            "listing": None if self.listing is None else str(self.listing),
            "listing_sha256": self.listing_sha256,
            "files": [
                str(path.relative_to(root)) if root is not None else str(path)
                for path in self.files
            ],
        }


def parse_listing(listing: Path) -> list[str]:
    """Relative paths in listing order; ``sha256sum`` lines are accepted."""
    names: list[str] = []
    for raw in listing.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _space, rest = line.partition(" ")
        if _SHA256_HEX.match(head) and rest:
            line = rest.strip().lstrip("*")
        names.append(line)
    if not names:
        raise SetupError(f"{listing}: the listing names no file")
    return names


def select_sources(specs: Iterable[str], index: int, total: int) -> list[Source]:
    """Resolve every ``--source`` to the files of shard ``index`` of ``total``."""
    if total < 1 or not 0 <= index < total:
        raise SetupError("--shard-index must lie in [0, --num-shards)")
    sources: list[Source] = []
    for spec in specs:
        kind, _separator, rest = spec.partition(":")
        if kind == "jsonl":
            path = Path(rest)
            if not path.is_file():
                raise SetupError(f"{spec!r}: {path} is not a file")
            sources.append(Source(spec, kind, None, None, None, (path,)))
            continue
        if kind not in ("parquet", "texts"):
            raise SetupError(f"unknown source {spec!r} (parquet:/texts:/jsonl:)")
        root_text, separator, listing_text = rest.rpartition(":")
        if not separator or not root_text or not listing_text:
            raise SetupError(
                f"{spec!r}: a {kind} source is {kind}:<root>:<listing>; the "
                "listing fixes the file order, a directory is never scanned"
            )
        root = Path(root_text)
        listing = Path(listing_text)
        if not listing.is_file():
            raise SetupError(f"{spec!r}: listing {listing} is not a file")
        names = parse_listing(listing)
        selected = [
            name for position, name in enumerate(names) if position % total == index
        ]
        missing = [name for name in selected if not (root / name).is_file()]
        if missing:
            shown = ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
            raise SetupError(
                f"{spec!r}: {len(missing)} of the {len(selected)} files shard {index} "
                f"needs are not under {root}: {shown}"
            )
        sources.append(
            Source(
                spec,
                kind,
                root,
                listing,
                hashlib.sha256(listing.read_bytes()).hexdigest(),
                tuple(root / name for name in selected),
            )
        )
    return sources


def iter_parquet(files: Iterable[Path]) -> Iterator[tuple[str, str]]:
    parquet = importlib.import_module("pyarrow.parquet")
    for path in files:
        handle = parquet.ParquetFile(path)
        names = list(handle.schema_arrow.names)
        column = "text" if "text" in names else names[0]
        for batch in handle.iter_batches(batch_size=2048, columns=[column]):
            for value in batch.column(0).to_pylist():
                if value:
                    yield path.name, str(value)


def iter_texts(files: Iterable[Path]) -> Iterator[tuple[str, str]]:
    for path in files:
        text = path.read_text(encoding="utf-8")
        for part in text.split("\n\n"):
            if part:
                yield path.name, part


def iter_jsonl(
    path: Path, index: int, total: int, field: str
) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position % total != index:
                continue
            try:
                value = json.loads(line).get(field)
            except (ValueError, AttributeError):
                continue
            if value:
                yield path.name, str(value)


def documents(
    sources: Iterable[Source], index: int, total: int, jsonl_field: str
) -> Iterator[tuple[str, str]]:
    for source in sources:
        if source.kind == "parquet":
            yield from iter_parquet(source.files)
        elif source.kind == "texts":
            yield from iter_texts(source.files)
        else:
            yield from iter_jsonl(source.files[0], index, total, jsonl_field)


# -- the judge ----------------------------------------------------------------


def _zero_counters() -> dict[str, Any]:
    return {
        "docs": 0,
        "chars": 0,
        "tokens": 0,
        "chunks": 0,
        "chunked_docs": 0,
        "cut_misses": 0,
        "literal_rejects": 0,
        "norm_rejects": 0,
        "serial_fallbacks": 0,
        "serial_reasons": {},
        "native": None,
        "mismatches": {column: None for column in MISMATCH_COLUMNS},
    }


def _zero_native() -> dict[str, Any]:
    return {
        "chunks": 0,
        "chunked_docs": 0,
        "serial_fallbacks": 0,
        "serial_reasons": {},
        "plan_disagreements": 0,
    }


class Family:
    """One loaded tokenizer, its oracle, and its counters."""

    def __init__(self, name: str, tokenizer: Any, *, native: str) -> None:
        import tokenizers

        from toktier import chunked

        self.name = name
        self.tokenizer = tokenizer
        self.native = native
        face = tokenizer._backend.materialized_tokenizer_json()
        self.face_sha256 = hashlib.sha256(face.encode("utf-8")).hexdigest()
        self.oracle = tokenizers.Tokenizer.from_str(face)
        # The table step is lifted for the tokenizer under judgement; the
        # roster and normalizer steps are not.
        chunked._planner_for(tokenizer, evidence_gate=False)
        self.admission: dict[str, Any] = dict(chunked.admission(tokenizer))
        self.counters = _zero_counters()
        if native != "off":
            self.counters["native"] = _zero_native()
        python_runs = native != "on"
        native_runs = native != "off"
        self.counters["mismatches"] = {
            "python_vs_oracle": 0 if python_runs else None,
            "native_vs_oracle": 0 if native_runs else None,
            "python_vs_native": 0 if python_runs and native_runs else None,
            "serial_vs_oracle": 0,
        }

    @property
    def artifact_sha256(self) -> str:
        return str(self.tokenizer._artifact_sha256)

    @property
    def normalizer(self) -> str | None:
        value = self.admission.get("normalizer")
        return None if value is None else str(value)

    def reference(self, text: str) -> list[int]:
        return [int(i) for i in self.oracle.encode(text, add_special_tokens=False).ids]

    def judge(self, text: str, k: int, min_chunk: int) -> dict[str, Any] | None:
        """Judge one document; return the finding, or ``None`` when all equal."""
        from toktier import chunked

        counters = self.counters
        mismatches = counters["mismatches"]
        serial = list(self.tokenizer.encode(text, lookup="off").ids)
        oracle = self.reference(text)
        python_ids: list[int] | None = None
        native_ids: list[int] | None = None
        python_encoded = native_encoded = None
        if self.native != "on":
            python_encoded = chunked.encode(
                self.tokenizer, text, k=k, min_chunk=min_chunk, native=False
            )
            python_ids = list(python_encoded.ids)
        if self.native != "off":
            native_encoded = chunked.encode(
                self.tokenizer, text, k=k, min_chunk=min_chunk, native=True
            )
            native_ids = list(native_encoded.ids)
        lead = python_encoded if python_encoded is not None else native_encoded
        assert lead is not None
        counters["docs"] += 1
        counters["chars"] += len(text)
        counters["tokens"] += len(lead.ids)
        counters["chunks"] += lead.chunks
        counters["chunked_docs"] += 1 if lead.chunks >= 2 else 0
        counters["cut_misses"] += lead.cut_misses
        counters["literal_rejects"] += lead.literal_rejects
        counters["norm_rejects"] += lead.norm_rejects
        if lead.serial_reason:
            counters["serial_fallbacks"] += 1
            reasons = counters["serial_reasons"]
            reasons[lead.serial_reason] = reasons.get(lead.serial_reason, 0) + 1
        if native_encoded is not None:
            native = counters["native"]
            native["chunks"] += native_encoded.chunks
            native["chunked_docs"] += 1 if native_encoded.chunks >= 2 else 0
            if native_encoded.serial_reason:
                native["serial_fallbacks"] += 1
                reasons = native["serial_reasons"]
                reasons[native_encoded.serial_reason] = (
                    reasons.get(native_encoded.serial_reason, 0) + 1
                )
            if python_encoded is not None and (
                native_encoded.chunks,
                native_encoded.cuts,
                native_encoded.serial_reason,
            ) != (
                python_encoded.chunks,
                python_encoded.cuts,
                python_encoded.serial_reason,
            ):
                native["plan_disagreements"] += 1
        unequal: list[str] = []
        if python_ids is not None and python_ids != oracle:
            mismatches["python_vs_oracle"] += 1
            unequal.append("python_vs_oracle")
        if native_ids is not None and native_ids != oracle:
            mismatches["native_vs_oracle"] += 1
            unequal.append("native_vs_oracle")
        if (
            python_ids is not None
            and native_ids is not None
            and python_ids != native_ids
        ):
            mismatches["python_vs_native"] += 1
            unequal.append("python_vs_native")
        if serial != oracle:
            mismatches["serial_vs_oracle"] += 1
            unequal.append("serial_vs_oracle")
        if not unequal:
            return None
        shown = python_ids if python_ids is not None else native_ids
        assert shown is not None
        first = next(
            (i for i, (a, b) in enumerate(zip(shown, oracle, strict=False)) if a != b),
            min(len(shown), len(oracle)),
        )
        return {
            "family": self.name,
            "unequal": unequal,
            "k": k,
            "min_chunk": min_chunk,
            "chars": len(text),
            "chunks": lead.chunks,
            "cuts": list(lead.cuts),
            "serial_reason": lead.serial_reason,
            "first_difference": first,
            "text": text,
        }

    def report(self) -> dict[str, Any]:
        from toktier import chunked

        return dict(chunked.stats(self.tokenizer))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.tokenizer.close()


def load_families(
    names: list[str], *, native: str, native_workers: int | None, policy: str | None
) -> dict[str, Family]:
    """Load every family, lift the table step, and check what the arms need."""
    import toktier
    from toktier import chunked

    loaded: dict[str, Family] = {}
    try:
        for name in names:
            options: dict[str, Any] = {"device": "cpu"}
            if policy is not None:
                options["policy"] = policy
            tokenizer = toktier.load(name, **options)
            family = Family(name, tokenizer, native=native)
            loaded[name] = family
            if not family.admission.get("admitted"):
                raise SetupError(
                    f"{name}: not admitted to the chunked path even with the "
                    f"table step lifted ({family.admission.get('refusal')})"
                )
            if native != "off":
                planned = tokenizer.explain().get("planned_backend")
                runs = chunked.native_available(tokenizer)
                if not runs or planned != "fast_cpu":
                    raise NativeRouteError(
                        f"{name}: the native arm needs a plan that starts on the "
                        f"corrected CPU engine (planned backend {planned!r}, native "
                        f"encoder {'can' if runs else 'cannot'} run on this route); "
                        "a different policy may be needed, and a "
                        "reference-only route would only record "
                        "R_CHUNK_ROUTE_NOT_FAST_CPU"
                    )
                if native_workers is not None:
                    chunked.warm(tokenizer, workers=native_workers)
    except Exception:
        for family in loaded.values():
            family.close()
        raise
    return loaded


def save_finding(directory: Path | None, finding: dict[str, Any]) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(finding["text"].encode("utf-8")).hexdigest()[:16]
    name = f"{finding['family']}_{key}_{'_'.join(finding['unequal'])}.json"
    (directory / name).write_text(
        json.dumps(finding, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def write_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n")
    os.replace(temporary, path)


def family_rows(families: dict[str, Family]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "artifact_sha256": family.artifact_sha256,
            "loader_face_sha256": family.face_sha256,
            "normalizer": family.normalizer,
            "admission": family.admission,
            **family.counters,
            "report": family.report(),
        }
        for name, family in families.items()
    }


def _mismatch_totals(families: dict[str, Family]) -> dict[str, int | None]:
    totals: dict[str, int | None] = {}
    for column in MISMATCH_COLUMNS:
        values = [f.counters["mismatches"][column] for f in families.values()]
        totals[column] = (
            None if all(v is None for v in values) else sum(v or 0 for v in values)
        )
    return totals


# -- run ----------------------------------------------------------------------


class Stream:
    """Counts and the digest of the document stream a unit judges.

    The digest covers the judged (accepted) documents in order, each as the
    8-byte little-endian length of its UTF-8 bytes followed by the bytes,
    so a unit's digest equals the corpus plan's part digest for the same
    (shard, part), and the whole-shard digest when no part is selected.
    ``docs_seen`` and ``docs_skipped`` count the part window only, as the
    corpus plan does.
    """

    def __init__(self) -> None:
        self.docs_seen = 0
        self.docs_skipped = 0
        self.docs_passed_over = 0
        self.docs_judged = 0
        self.chars = 0
        self._digest = hashlib.sha256()

    def judged(self, text: str) -> None:
        self.docs_judged += 1
        self.chars += len(text)
        payload = text.encode("utf-8")
        self._digest.update(len(payload).to_bytes(8, "little"))
        self._digest.update(payload)

    def facts(self) -> dict[str, Any]:
        return {
            "docs_seen": self.docs_seen,
            "docs_skipped": self.docs_skipped,
            "docs_passed_over": self.docs_passed_over,
            "docs_judged": self.docs_judged,
            "chars": self.chars,
            "doc_stream_sha256": self._digest.hexdigest(),
        }


def run_unit(
    families: dict[str, Family],
    stream_of_documents: Iterable[tuple[str, str]],
    *,
    min_chars: int,
    max_chars: int,
    chunk_chars: int,
    min_chunk: int,
    max_docs: int,
    skip_docs: int,
    time_budget_s: float,
    report_every: int,
    mismatch_dir: Path | None,
    flush: Callable[[Stream, bool, str], None],
    started: float,
) -> tuple[Stream, str]:
    """Judge one unit; return the stream facts and why it stopped."""
    stream = Stream()
    stopped = "source_exhausted"
    for _source, text in stream_of_documents:
        stream.docs_seen += 1
        length = len(text)
        if length < min_chars or length > max_chars:
            stream.docs_skipped += 1
            continue
        if stream.docs_passed_over < skip_docs:
            stream.docs_passed_over += 1
            if stream.docs_passed_over == skip_docs:
                stream.docs_seen = 0
                stream.docs_skipped = 0
            continue
        k = max(2, -(-length // chunk_chars))
        for family in families.values():
            finding = family.judge(text, k, min_chunk)
            if finding is not None:
                save_finding(mismatch_dir, finding)
                print(
                    f"finding: {family.name} {finding['unequal']} chars={length} "
                    f"k={k} first_difference={finding['first_difference']}",
                    file=sys.stderr,
                    flush=True,
                )
        stream.judged(text)
        if report_every and stream.docs_judged % report_every == 0:
            elapsed = time.time() - started
            totals = _mismatch_totals(families)
            print(
                f"[run] judged={stream.docs_judged} families={len(families)} "
                f"mismatches={totals} {elapsed:.0f}s "
                f"({stream.docs_judged / max(elapsed, 1e-9):.1f} docs/s)",
                flush=True,
            )
            flush(stream, False, "running")
        if stream.docs_judged >= max_docs:
            stopped = "max_docs"
            break
        if time_budget_s > 0 and time.time() - started > time_budget_s:
            stopped = "time_budget"
            break
    return stream, stopped


def command_run(arguments: argparse.Namespace) -> int:
    started = time.time()
    names = [name for name in arguments.families.split(",") if name]
    if not names:
        raise SetupError("--families names no family")
    if arguments.part_index is not None and arguments.part_docs is None:
        raise SetupError("--part-index needs --part-docs")
    skip_docs = 0
    max_docs = arguments.max_docs
    if arguments.part_docs is not None:
        skip_docs = (arguments.part_index or 0) * arguments.part_docs
        max_docs = arguments.part_docs
    sources = select_sources(
        arguments.source, arguments.shard_index, arguments.num_shards
    )
    families = load_families(
        names,
        native=arguments.native,
        native_workers=arguments.native_workers,
        policy=arguments.policy,
    )
    out = Path(arguments.out)
    mismatch_dir = Path(arguments.mismatch_dir) if arguments.mismatch_dir else None
    header = manifest(
        argv=sys.argv, started=started, tag=arguments.tag, policy=arguments.policy
    )
    unit = {
        "sources": [source.record() for source in sources],
        "shard_index": arguments.shard_index,
        "num_shards": arguments.num_shards,
        "part_index": arguments.part_index,
        "part_docs": arguments.part_docs,
        "max_docs": max_docs,
        "min_chars": arguments.min_chars,
        "max_chars": arguments.max_chars,
        "chunk_chars": arguments.chunk_chars,
        "min_chunk": arguments.min_chunk,
        "native": arguments.native,
        "native_workers": arguments.native_workers,
    }

    def flush(stream: Stream, final: bool, stopped: str) -> None:
        totals = _mismatch_totals(families)
        document = {
            "schema": SHARD_SCHEMA,
            "tool": {
                "path": TOOL_NAME,
                "version": TOOL_VERSION,
                "sha256": tool_sha256(),
            },
            "manifest": {
                **header,
                "admission_gate": "lifted_for_evidence_collection",
                "elapsed_s": round(time.time() - started, 1),
                "final": final,
                "stopped_by": stopped,
            },
            "unit": unit,
            "stream": stream.facts(),
            "families": family_rows(families),
            "totals": {
                "docs": stream.docs_judged,
                "checks": stream.docs_judged * len(families),
                "mismatches": totals,
            },
        }
        write_atomic(out, document)

    try:
        stream, stopped = run_unit(
            families,
            documents(
                sources,
                arguments.shard_index,
                arguments.num_shards,
                arguments.jsonl_field,
            ),
            min_chars=arguments.min_chars,
            max_chars=arguments.max_chars,
            chunk_chars=arguments.chunk_chars,
            min_chunk=arguments.min_chunk,
            max_docs=max_docs,
            skip_docs=skip_docs,
            time_budget_s=arguments.time_budget_s,
            report_every=arguments.report_every,
            mismatch_dir=mismatch_dir,
            flush=flush,
            started=started,
        )
        flush(stream, True, stopped)
    finally:
        for family in families.values():
            family.close()
    totals = _mismatch_totals(families)
    unequal = sum(value or 0 for value in totals.values())
    elapsed = time.time() - started
    checks = stream.docs_judged * len(families)
    print(
        f"[run] done judged={stream.docs_judged} checks={checks} mismatches={totals} "
        f"stopped_by={stopped} {elapsed:.0f}s -> {out}",
        flush=True,
    )
    return STATUS_MISMATCH if unequal else STATUS_OK


# -- adversarial ----------------------------------------------------------------


def command_adversarial(arguments: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    corpus = importlib.import_module("chunked_adversarial_corpus")
    started = time.time()
    names = [name for name in arguments.families.split(",") if name]
    if not names or arguments.strings < 1:
        raise SetupError("--families and a positive --strings are required")
    families = load_families(
        names,
        native=arguments.native,
        native_workers=arguments.native_workers,
        policy=arguments.policy,
    )
    out = Path(arguments.out)
    mismatch_dir = Path(arguments.mismatch_dir) if arguments.mismatch_dir else None
    header = manifest(
        argv=sys.argv, started=started, tag=arguments.tag, policy=arguments.policy
    )
    strings: dict[str, int] = {}
    chars: dict[str, int] = {}
    try:
        for name, family in families.items():
            face = json.loads(family.tokenizer._backend.materialized_tokenizer_json())
            literals = [
                str(row.get("content", ""))
                for row in face.get("added_tokens") or []
                if isinstance(row, dict)
            ]
            rng = corpus.battery_rng(name, arguments.seed_offset)
            strings[name] = 0
            chars[name] = 0
            for _index in range(arguments.strings):
                text = corpus.adversarial(rng, rng.randint(300, 1500), literals)
                k = corpus.battery_k(rng)
                finding = family.judge(text, k, arguments.min_chunk)
                strings[name] += 1
                chars[name] += len(text)
                if finding is not None:
                    save_finding(mismatch_dir, finding)
                    print(
                        f"finding: {name} {finding['unequal']} chars={len(text)} k={k}",
                        file=sys.stderr,
                        flush=True,
                    )
            print(
                f"[adversarial] {name}: strings={strings[name]} "
                f"mismatches={family.counters['mismatches']} "
                f"{time.time() - started:.0f}s",
                flush=True,
            )
        rows = {}
        for name, family in families.items():
            counters = family.counters
            rows[name] = {
                "artifact_sha256": family.artifact_sha256,
                "loader_face_sha256": family.face_sha256,
                "normalizer": family.normalizer,
                "admission": family.admission,
                "strings": strings[name],
                "chars": chars[name],
                "chunked": counters["chunked_docs"],
                "chunks": counters["chunks"],
                "cut_misses": counters["cut_misses"],
                "literal_rejects": counters["literal_rejects"],
                "norm_rejects": counters["norm_rejects"],
                "serial_fallbacks": counters["serial_fallbacks"],
                "serial_reasons": counters["serial_reasons"],
                "native": counters["native"],
                "mismatches": counters["mismatches"],
                "report": family.report(),
            }
        totals = _mismatch_totals(families)
        document = {
            "schema": BATTERY_SCHEMA,
            "tool": {
                "path": TOOL_NAME,
                "version": TOOL_VERSION,
                "sha256": tool_sha256(),
            },
            "manifest": {
                **header,
                "admission_gate": "lifted_for_evidence_collection",
                "elapsed_s": round(time.time() - started, 1),
                "final": True,
                "stopped_by": "strings",
            },
            "generator": corpus.GENERATOR,
            "battery": {
                "strings_per_family": arguments.strings,
                "min_chunk": arguments.min_chunk,
                "seed_offset": arguments.seed_offset,
                "native": arguments.native,
            },
            "families": rows,
            "totals": {
                "strings": sum(strings.values()),
                "checks": sum(strings.values()),
                "mismatches": totals,
            },
        }
        write_atomic(out, document)
    finally:
        for family in families.values():
            family.close()
    unequal = sum(value or 0 for value in totals.values())
    print(f"[adversarial] done mismatches={totals} -> {out}", flush=True)
    return STATUS_MISMATCH if unequal else STATUS_OK


# -- summarize ------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SetupError(f"{path}: not a JSON object")
    return document


def _merge_counts(
    target: dict[str, Any], source: dict[str, Any], keys: Iterable[str]
) -> None:
    for key in keys:
        target[key] = int(target.get(key, 0)) + int(source.get(key, 0))


def _merge_reasons(target: dict[str, int], source: dict[str, Any]) -> None:
    for reason, count in source.items():
        target[str(reason)] = target.get(str(reason), 0) + int(count)


def _merge_mismatches(target: dict[str, int | None], source: dict[str, Any]) -> None:
    for column in MISMATCH_COLUMNS:
        value = source.get(column)
        if value is None:
            continue
        current = target.get(column)
        target[column] = int(value) if current is None else current + int(value)


def _agree(values: set[Any], what: str) -> Any:
    if len(values) != 1:
        raise SetupError(f"the units disagree on {what}: {sorted(map(str, values))}")
    return next(iter(values))


def _unit_key(unit: dict[str, Any]) -> tuple[Any, ...]:
    spec = unit["unit"]
    return (
        spec.get("shard_index"),
        spec.get("num_shards"),
        spec.get("part_index"),
        spec.get("part_docs"),
    )


def group_units(
    paths: list[Path], units: list[dict[str, Any]]
) -> dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]]:
    """Group unit files by work unit and check that a group is one unit.

    The files of one unit (one per family group) must have judged the same
    document stream, byte for byte, and disjoint families.
    """
    groups: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = {}
    for path, unit in zip(paths, units, strict=True):
        groups.setdefault(_unit_key(unit), []).append((path, unit))
    for key, members in groups.items():
        streams = {
            json.dumps(unit["stream"], sort_keys=True) for _path, unit in members
        }
        if len(streams) != 1:
            names = ", ".join(str(path) for path, _unit in members)
            raise SetupError(
                f"unit {key}: its files did not judge the same document stream "
                f"({names}); a unit is the same documents for every family group"
            )
        seen: dict[str, Path] = {}
        for path, unit in members:
            for family in unit["families"]:
                if family in seen:
                    raise SetupError(
                        f"unit {key}: {family} is judged in both {seen[family]} and "
                        f"{path}; the family groups of a unit must be disjoint"
                    )
                seen[family] = path
    return groups


def _merged_row(name: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": name,
        "artifact_sha256": row["artifact_sha256"],
        "loader_face_sha256": row.get("loader_face_sha256"),
        "normalizer": row["normalizer"],
        "paths": set(),
        "docs": 0,
        "chars": 0,
        "tokens": 0,
        "chunks": 0,
        "chunked_docs": 0,
        "cut_misses": 0,
        "literal_rejects": 0,
        "norm_rejects": 0,
        "serial_fallbacks": 0,
        "serial_reasons": {},
        "native": None,
        "mismatches": {column: None for column in MISMATCH_COLUMNS},
    }


def merge_families(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-family readings rows merged over every unit file."""
    families: dict[str, dict[str, Any]] = {}
    for unit in units:
        arm = unit["unit"].get("native", "off")
        for name, row in unit["families"].items():
            merged = families.setdefault(name, _merged_row(name, row))
            if merged["artifact_sha256"] != row["artifact_sha256"]:
                raise SetupError(f"{name}: the units judged different artifacts")
            if merged["loader_face_sha256"] != row.get("loader_face_sha256"):
                raise SetupError(f"{name}: the units judged different loader faces")
            if merged["normalizer"] != row["normalizer"]:
                raise SetupError(f"{name}: the units disagree on the normalizer")
            _merge_counts(
                merged,
                row,
                (
                    "docs",
                    "chars",
                    "tokens",
                    "chunks",
                    "chunked_docs",
                    "cut_misses",
                    "literal_rejects",
                    "norm_rejects",
                    "serial_fallbacks",
                ),
            )
            _merge_reasons(merged["serial_reasons"], row.get("serial_reasons") or {})
            _merge_mismatches(merged["mismatches"], row["mismatches"])
            if arm != "on":
                merged["paths"].add("python")
            if arm != "off":
                merged["paths"].add("native")
                native = row.get("native") or _zero_native()
                if merged["native"] is None:
                    merged["native"] = _zero_native()
                _merge_counts(
                    merged["native"],
                    native,
                    (
                        "chunks",
                        "chunked_docs",
                        "serial_fallbacks",
                        "plan_disagreements",
                    ),
                )
                _merge_reasons(
                    merged["native"]["serial_reasons"],
                    native.get("serial_reasons") or {},
                )
    rows = []
    for name in sorted(families):
        merged = dict(families[name])
        merged["paths"] = [p for p in ("python", "native") if p in merged["paths"]]
        rows.append(merged)
    return rows


def battery_rows(
    battery: dict[str, Any],
    identity: Any,
    tool_digest: Any,
    judged: list[dict[str, Any]],
) -> dict[str, Any]:
    """The readings' adversarial section from a battery JSON."""
    if battery.get("schema") != BATTERY_SCHEMA:
        raise SetupError(f"the battery file is not a {BATTERY_SCHEMA} document")
    if not battery["manifest"].get("final"):
        raise SetupError("the battery did not finish")
    if battery["manifest"].get("chunked_identity") != identity:
        raise SetupError(
            "the battery and the corpus units were recorded under different "
            "chunked identities"
        )
    if battery["tool"].get("sha256") != tool_digest:
        raise SetupError("the battery and the corpus units ran different tool bytes")
    arm = battery["battery"].get("native", "off")
    paths = [
        p for p, runs in (("python", arm != "on"), ("native", arm != "off")) if runs
    ]
    faces = {row["family"]: row["loader_face_sha256"] for row in judged}
    rows = []
    for name in sorted(battery["families"]):
        row = battery["families"][name]
        if name in faces and faces[name] != row.get("loader_face_sha256"):
            raise SetupError(
                f"{name}: the battery and the corpus units judged different "
                "loader faces"
            )
        entry: dict[str, Any] = {
            "family": name,
            "artifact_sha256": row["artifact_sha256"],
            "loader_face_sha256": row.get("loader_face_sha256"),
            "normalizer": row["normalizer"],
            "paths": paths,
            "strings": int(row["strings"]),
            "chars": int(row["chars"]),
            "chunked": int(row["chunked"]),
            "chunks": int(row["chunks"]),
            "cut_misses": int(row["cut_misses"]),
            "literal_rejects": int(row["literal_rejects"]),
            "norm_rejects": int(row["norm_rejects"]),
            "mismatches": {
                column: row["mismatches"].get(column) for column in MISMATCH_COLUMNS
            },
        }
        # The plan comparison runs only when both paths judged a string;
        # the readings carry its count then, and leave it out otherwise.
        native = row.get("native")
        if paths == ["python", "native"] and isinstance(native, dict):
            entry["plan_disagreements"] = int(native["plan_disagreements"])
        rows.append(entry)
    return {"generator": str(battery.get("generator")), "families": rows}


def command_summarize(arguments: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    registry_common = importlib.import_module("registry_common")
    paths = sorted(Path(p) for pattern in arguments.shards for p in glob.glob(pattern))
    if not paths:
        raise SetupError("--shards matched no file")
    units = [_load_json(path) for path in paths]
    for path, unit in zip(paths, units, strict=True):
        if unit.get("schema") != SHARD_SCHEMA:
            raise SetupError(f"{path}: not a {SHARD_SCHEMA} document")
        if not unit["manifest"].get("final"):
            raise SetupError(f"{path}: the unit did not finish (final is false)")
    manifests = [unit["manifest"] for unit in units]
    identity = _agree(
        {m.get("chunked_identity") for m in manifests}, "chunked_identity"
    )
    tool_digest = _agree(
        {unit["tool"].get("sha256") for unit in units}, "the tool digest"
    )
    gate = _agree({m.get("admission_gate") for m in manifests}, "admission_gate")
    if gate != "lifted_for_evidence_collection":
        raise SetupError(f"unexpected admission_gate {gate!r} in the units")
    unit_specs = [unit["unit"] for unit in units]
    groups = group_units(paths, units)
    rows = merge_families(units)

    # Documents are counted once per unit, whatever the number of family
    # groups it was split over (group_units checked they are the same).
    unit_streams = [members[0][1]["stream"] for members in groups.values()]
    corpus_docs = sum(int(stream["docs_judged"]) for stream in unit_streams)
    corpus_chars = sum(int(stream["chars"]) for stream in unit_streams)
    hosts = sorted({str(m.get("hostname")) for m in manifests if m.get("hostname")})
    job_ids = sorted(
        {
            str(m["slurm"][key])
            for m in manifests
            for key in ("SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID")
            if m.get("slurm", {}).get(key)
        }
        | set(arguments.job_ids or [])
    )

    adversarial: dict[str, Any] = {"generator": "not run", "families": []}
    if arguments.adversarial:
        battery = _load_json(Path(arguments.adversarial))
        adversarial = battery_rows(battery, identity, tool_digest, rows)

    not_reasoned = []
    for item in arguments.not_reasoned or []:
        family, _sep, rest = item.partition(":")
        digest, _sep, refusal = rest.partition(":")
        if not (family and digest and refusal):
            raise SetupError("--not-reasoned takes family:artifact_sha256:refusal")
        not_reasoned.append(
            {"family": family, "artifact_sha256": digest, "refusal": refusal}
        )

    os_name = arguments.os
    if os_name is None:
        os_name = _agree({m.get("os") for m in manifests}, "os")

    def agreed(field: str) -> Any:
        return _agree({u.get(field) for u in unit_specs}, field)

    document = {
        "schema": READINGS_SCHEMA,
        "schema_version": 1,
        "generated_by": {
            "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "source_commit": arguments.source_commit,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "campaign": {
            "run_id": arguments.run_id,
            "date": arguments.date,
            "tag": arguments.tag,
        },
        "tool_sha256": tool_digest,
        "chunked_identity": identity,
        "fast_cpu_source_digest": _agree(
            {m.get("fast_cpu_source_digest") for m in manifests},
            "fast_cpu_source_digest",
        ),
        "fast_cpu_source_digest_v2": arguments.fast_cpu_source_digest_v2,
        "toktier_version": str(
            _agree({m.get("toktier_version") for m in manifests}, "toktier")
        ),
        "tokenizers": str(
            _agree({m.get("tokenizers") for m in manifests}, "tokenizers")
        ),
        "transformers": _agree(
            {m.get("transformers") for m in manifests}, "transformers"
        ),
        "python": str(_agree({m.get("python") for m in manifests}, "python")),
        "os": str(os_name),
        "hosts": hosts,
        "job_ids": job_ids,
        "core_hours": arguments.core_hours,
        "k_rule": {
            "rule": "max(2, ceil(chars / chunk_chars))",
            "chunk_chars": int(agreed("chunk_chars")),
            "min_chunk": int(agreed("min_chunk")),
        },
        "length_gate": {
            "min_chars": int(agreed("min_chars")),
            "max_chars": int(agreed("max_chars")),
        },
        "shards": {
            "planned": arguments.expected_units or len(groups),
            "final": len(groups),
        },
        "corpus": [
            {
                "corpus_id": arguments.corpus_id,
                "docs": corpus_docs,
                "chars": corpus_chars,
            }
        ],
        "families": rows,
        "adversarial": adversarial,
        "not_reasoned": not_reasoned,
        "notes": arguments.notes,
    }
    completed = registry_common.with_root_digest(
        document, registry_common.CHUNKED_READINGS_DOMAIN_TAG
    )
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(registry_common.serialise_document(completed))
    unequal = sum(value or 0 for row in rows for value in row["mismatches"].values())
    unequal += sum(
        value or 0
        for row in adversarial["families"]
        for value in row["mismatches"].values()
    )
    print(
        f"[summarize] {len(units)} files, {len(groups)} units, {len(rows)} families, "
        f"{corpus_docs} documents, mismatches={unequal} -> {out}; validate with "
        "tools/validate_registry.py and derive the table with "
        "tools/update_chunked_admission.py",
        flush=True,
    )
    return STATUS_MISMATCH if unequal else STATUS_OK


# -- command line ---------------------------------------------------------------


def _add_judge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--families", required=True, help="comma-separated family names"
    )
    parser.add_argument(
        "--native",
        choices=("off", "on", "both"),
        default="off",
        help="which chunked paths run: python (off), native (on), or both",
    )
    parser.add_argument(
        "--native-workers",
        type=int,
        default=None,
        help="fork the native chunk worker pool with this many workers up front",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="routing policy passed to toktier.load (recorded; experimental lets a "
        "tree whose identities are not yet rebound plan the corrected CPU route)",
    )
    parser.add_argument("--min-chunk", type=int, default=64)
    parser.add_argument("--mismatch-dir", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge the chunk-parallel path four ways and record the readings."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="judge one unit of the corpus campaign")
    run.add_argument(
        "--source",
        action="append",
        required=True,
        help="parquet:<root>:<listing>, texts:<root>:<listing> or jsonl:<file>",
    )
    run.add_argument("--jsonl-field", default="text")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--max-docs", type=int, default=100_000)
    run.add_argument("--part-docs", type=int, default=None)
    run.add_argument("--part-index", type=int, default=None)
    run.add_argument("--min-chars", type=int, default=512)
    run.add_argument("--max-chars", type=int, default=400_000)
    run.add_argument("--chunk-chars", type=int, default=8_000)
    run.add_argument("--time-budget-s", type=float, default=0.0)
    run.add_argument("--report-every", type=int, default=20_000)
    _add_judge_arguments(run)
    run.set_defaults(handler=command_run)

    battery = commands.add_parser("adversarial", help="judge the adversarial battery")
    battery.add_argument(
        "--strings", type=int, required=True, help="strings per family"
    )
    battery.add_argument("--seed-offset", type=int, default=0)
    _add_judge_arguments(battery)
    battery.set_defaults(handler=command_adversarial)

    summarize = commands.add_parser("summarize", help="merge units into the readings")
    summarize.add_argument("--shards", action="append", required=True, help="glob(s)")
    summarize.add_argument("--adversarial", default=None, help="battery JSON")
    summarize.add_argument("--out", required=True)
    summarize.add_argument("--corpus-id", required=True)
    summarize.add_argument("--run-id", required=True)
    summarize.add_argument("--date", required=True, help="YYYY-MM-DD")
    summarize.add_argument("--source-commit", required=True)
    summarize.add_argument("--expected-units", type=int, default=None)
    summarize.add_argument("--core-hours", type=float, default=None)
    summarize.add_argument("--fast-cpu-source-digest-v2", default=None)
    summarize.add_argument("--job-ids", nargs="*", default=None)
    summarize.add_argument("--not-reasoned", nargs="*", default=None)
    summarize.add_argument("--os", default=None)
    summarize.add_argument("--tag", default=None)
    summarize.add_argument("--notes", default="")
    summarize.set_defaults(handler=command_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    arguments = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = arguments.handler
    try:
        return handler(arguments)
    except NativeRouteError as error:
        print(f"error: {error}", file=sys.stderr)
        return STATUS_NATIVE_ROUTE
    except SetupError as error:
        print(f"error: {error}", file=sys.stderr)
        return STATUS_SETUP


if __name__ == "__main__":
    raise SystemExit(main())
