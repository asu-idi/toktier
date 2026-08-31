#!/usr/bin/env python3
# Standing loader-face alignment gate over real artifacts.
"""Compare TokTier's served ids with the pinned loader's own ids.

Contract reference: ``docs/contracts/facade.md`` Section 5 (reference =
the loader face). Serving stacks that take their ids from
``transformers.AutoTokenizer`` (SGLang's ``get_tokenizer`` among them)
define the ecosystem baseline. This gate loads a real artifact from the
local cache, encodes a fixed probe set on TokTier's default policy and
on ``policy="reference"``, and requires both to equal
``AutoTokenizer.encode`` id for id. The probe set is the one the
two-face divergence was found with: the seven configuration-only
literals of ``qwen3_5_08b``, one artifact-file added token, and one
plain-text control.

The pytest suite deliberately never reads the developer's cache, so this
gate lives here instead: it is meant for the release gates and any
machine that holds the real artifact. It refuses to conclude anything
when the artifact is absent (exit code 3, distinct from a mismatch).

``--loaderless-dir`` optionally names a frozen artifact directory whose
``tokenizer_class`` the pinned ``transformers`` cannot resolve (the
``tencent/Hy4-preview`` shape). There the loader face degrades to the
file-only face, and only because such a directory declares no
configuration-side added token are the two provably the same function;
the gate re-verifies that premise, then compares the reference backend
and the fallback loader object with the oracle on a CJK-heavy probe set.

``--fingerprints`` adds the second computation behind the capability
identities. Since 0.2.9 the registry records both capability
fingerprints on the loader-face document
(``docs/contracts/registry.md`` Section 1), and the values are written
once by maintainer tooling. This step recomputes them here, on every
cached family, and requires them to equal the ids the shipped registry
records -- so a moved loader, a changed artifact, or a hand-edited table
is caught by the release gates rather than by a reader. Families whose
artifact is not cached on this machine are reported and skipped, exactly
like the probe comparison above.

Usage::

    python tools/check_loader_face_alignment.py
    python tools/check_loader_face_alignment.py --loaderless-dir <dir>
    python tools/check_loader_face_alignment.py --fingerprints
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

#: Absent-artifact exit code: nothing was checked, nothing failed.
ABSENT = 3

FAMILY = "qwen3_5_08b"
QWEN_PROBES = (
    "hello <|audio_start|> world",
    "a<tts_text_bos_single>b",
    "<|audio_pad|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "x<|audio_end|>y",
    "hello <|im_start|> world",
    "plain text with no literals",
)

LOADERLESS_PROBES = (
    "The quick brown fox jumps over the lazy dog.",
    "腾讯混元大模型支持中文分词。",
    "你好！！！世界……「引号」【括号】",
    "こんにちは世界 안녕하세요 세계",
    "a  b\t\tc\n\nd     e",
    "1234567890123 and 42",
    "中文abc中文123中文",
)


@dataclasses.dataclass
class _Handle:
    """A minimal verified-artifact handle over a frozen directory."""

    family: str
    root: Path
    artifact_sha256: str
    files: dict[str, str]

    def path(self, name: str) -> Path:
        return self.root / name


def _loader(root: Path) -> Any:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(
        str(root), use_fast=True, local_files_only=True
    )


def check_family(family: str) -> tuple[int, dict[str, object]]:
    from toktier import Config, load
    from toktier.artifacts import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.paths import artifact_cache_dir

    config = Config(offline=True)
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    entry = manifest.get(family)
    root = artifact_cache_dir(config) / entry.directory_name
    reading: dict[str, object] = {"family": family, "root": str(root)}
    if not (root / "tokenizer.json").is_file() or not (
        root / "tokenizer_config.json"
    ).is_file():
        reading["skipped"] = "artifact or sidecar not cached"
        return ABSENT, reading
    loader = _loader(root)
    with_default = load(family, config=config)
    with_reference = load(family, config=config, policy="reference")
    mismatches = []
    try:
        for text in QWEN_PROBES:
            expected = loader.encode(text, add_special_tokens=False)
            observed_default = list(with_default.encode(text).ids)
            observed_reference = list(with_reference.encode(text).ids)
            if observed_default != expected or observed_reference != expected:
                mismatches.append(
                    {
                        "text": text,
                        "loader": expected,
                        "default": observed_default,
                        "reference": observed_reference,
                    }
                )
    finally:
        with_default.close()
        with_reference.close()
    reading["probes"] = len(QWEN_PROBES)
    reading["mismatches"] = mismatches
    return (1 if mismatches else 0), reading


def check_loaderless(root: Path) -> tuple[int, dict[str, object]]:
    import tokenizers

    from toktier.backends.hf import HfBackend
    from toktier.backends.loader_face import (
        config_added_token_rows,
        load_live_tokenizer,
    )

    reading: dict[str, object] = {"loaderless_root": str(root)}
    if not (root / "tokenizer.json").is_file():
        reading["skipped"] = "no tokenizer.json in the named directory"
        return ABSENT, reading
    rows = config_added_token_rows(root)
    if rows:
        reading["error"] = (
            "the directory declares configuration-side added tokens; the "
            "file-only degradation premise does not hold here"
        )
        return 1, reading
    live: Any = load_live_tokenizer(root)
    crate = tokenizers.Tokenizer.from_file(str(root / "tokenizer.json"))
    digest = hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
    backend = HfBackend.open(
        _Handle(
            family="loaderless_regression",
            root=root,
            artifact_sha256=digest,
            files={"tokenizer.json": digest},
        )
    )
    mismatches = []
    try:
        for text in LOADERLESS_PROBES:
            expected = [
                int(i) for i in crate.encode(text, add_special_tokens=False).ids
            ]
            observed_reference = backend.encode(text, add_special_tokens=False)
            observed_live = [
                int(i) for i in live.encode(text, add_special_tokens=False)
            ]
            if observed_reference != expected or observed_live != expected:
                mismatches.append(
                    {
                        "text": text,
                        "oracle": expected,
                        "reference": observed_reference,
                        "fallback_loader": observed_live,
                    }
                )
    finally:
        backend.close()
    reading["fallback_loader"] = type(live).__name__
    reading["probes"] = len(LOADERLESS_PROBES)
    reading["mismatches"] = mismatches
    return (1 if mismatches else 0), reading


def check_fingerprints() -> tuple[int, dict[str, object]]:
    """Recompute the capability ids of every cached family on the face.

    The registry records the values a maintainer run computed; this
    recomputes them from the artifact directories present on this
    machine and compares. A family whose artifact is not cached cannot
    be checked and is listed rather than passed over silently.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from artifact_identity import added_frontend_fingerprint, pipeline_fingerprint

    from toktier import Config
    from toktier.artifacts import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.backends.loader_face import (
        live_tokenizer_json,
        load_live_tokenizer,
    )
    from toktier.paths import artifact_cache_dir
    from toktier.routing.tables import SUPPORT_REGISTRY

    registry = json.loads(SUPPORT_REGISTRY.read_text(encoding="utf-8"))
    pipelines = {
        row["pipeline_id"]: row["pipeline_fingerprint"]
        for row in registry["pipelines"]
    }
    frontends = {
        row["added_frontend_id"]: row["added_frontend_fingerprint"]
        for row in registry["added_frontends"]
    }
    config = Config(offline=True)
    manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
    cache = artifact_cache_dir(config)

    checked: list[str] = []
    absent: list[str] = []
    mismatches: list[dict[str, object]] = []
    for row in registry["artifacts"]:
        family = str(row["family"])
        entry = manifest.entries.get(family)
        if entry is None:
            absent.append(family)
            continue
        root = cache / entry.directory_name
        if not (root / "tokenizer.json").is_file():
            absent.append(family)
            continue
        face = json.loads(live_tokenizer_json(load_live_tokenizer(root)))
        observed = {
            "pipeline": pipeline_fingerprint(face),
            "added_frontend": added_frontend_fingerprint(face),
        }
        recorded = {
            "pipeline": pipelines.get(row["pipeline_id"]),
            "added_frontend": frontends.get(row["added_frontend_id"]),
        }
        if observed != recorded:
            mismatches.append(
                {"family": family, "recorded": recorded, "recomputed": observed}
            )
        else:
            checked.append(family)

    reading: dict[str, object] = {
        "checked": checked,
        "absent": absent,
        "mismatches": mismatches,
    }
    if mismatches:
        return 1, reading
    if not checked:
        reading["skipped"] = "no artifact of the registry is cached here"
        return ABSENT, reading
    return 0, reading


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Loader-face alignment gate over real artifacts."
    )
    parser.add_argument("--family", default=FAMILY)
    parser.add_argument(
        "--loaderless-dir",
        type=Path,
        default=None,
        help="Frozen artifact directory with an unresolvable loader class.",
    )
    parser.add_argument(
        "--fingerprints",
        action="store_true",
        help=(
            "Recompute the capability fingerprints of every cached family "
            "on the loader face and compare them with the shipped registry."
        ),
    )
    arguments = parser.parse_args(argv)
    codes = []
    code, reading = check_family(arguments.family)
    codes.append(code)
    print(json.dumps(reading, ensure_ascii=False))
    if arguments.loaderless_dir is not None:
        code, reading = check_loaderless(arguments.loaderless_dir)
        codes.append(code)
        print(json.dumps(reading, ensure_ascii=False))
    if arguments.fingerprints:
        code, reading = check_fingerprints()
        codes.append(code)
        print(json.dumps(reading, ensure_ascii=False))
    if 1 in codes:
        return 1
    if ABSENT in codes:
        return ABSENT
    return 0


if __name__ == "__main__":
    sys.exit(main())
