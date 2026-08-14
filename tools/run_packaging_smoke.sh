#!/bin/sh

set -eu

root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
venv_dir=$(mktemp -d /tmp/toktier-packaging-smoke.XXXXXX)
trap 'rm -rf "$venv_dir"' 0 HUP INT TERM

wheel=${1:-}
if [ -z "$wheel" ]; then
    mkdir "$venv_dir/dist"
    maturin build --locked --release --out "$venv_dir/dist"
    wheel=$(find "$venv_dir/dist" -maxdepth 1 -name '*.whl' -print)
fi

python3 "$root/tools/verify_release_artifacts.py" "$wheel"
python3 -m venv "$venv_dir/runtime"
"$venv_dir/runtime/bin/python" -m pip install \
    --disable-pip-version-check --quiet "$wheel"

cd "$venv_dir"
TOKTIER_REPOSITORY_ROOT="$root" "$venv_dir/runtime/bin/python" - <<'PY'
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import toktier
import tokenizers
import transformers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from toktier.artifacts import shipped_sibling_aliases
from toktier.backends.fast_cpu import (
    ENGINE_DELIVERY,
    ENGINE_MODULE,
    fast_cpu_engine_facts,
)

repository = Path(os.environ["TOKTIER_REPOSITORY_ROOT"]).resolve()
installed = Path(toktier.__file__).resolve()
assert repository not in installed.parents, installed
assert toktier.__version__ == "0.2.5"
assert tokenizers.__version__ == "0.22.2"
assert transformers.__version__ == "4.57.6"
assert importlib.util.find_spec("gigatoken") is None
assert ENGINE_MODULE not in sys.modules
assert callable(toktier.from_pretrained)
aliases = shipped_sibling_aliases()
assert len(aliases.records) == 211
assert sum(record.canonical_packaged for record in aliases.records) == 204

facts = fast_cpu_engine_facts()
binding = json.loads(
    (repository / "tools" / "fast_cpu_binding.json").read_text(encoding="utf-8")
)
assert ENGINE_MODULE == "toktier._native"
assert ENGINE_DELIVERY == "integrated"
assert facts.binary_digest is None
assert facts.source_digest == binding["source_digest"]
assert list(facts.build_flags) == binding["build_flags"]
assert facts.toolchain == binding["toolchain"]
assert ENGINE_MODULE in sys.modules
assert not (installed.parent / "_vendor" / "gigatoken_rs.abi3.so").exists()
assert not (installed.parent / "_vendor" / "gigatoken_build.json").exists()

from toktier import _native
from toktier.repair.registry import pclass_table

alphabet = pre_tokenizers.ByteLevel.alphabet()
reference = Tokenizer(
    models.BPE(vocab={token: index for index, token in enumerate(alphabet)}, merges=[])
)
reference.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False, use_regex=True
)
reference.decoder = decoders.ByteLevel()
tokenizer_json = reference.to_str().encode("utf-8")
engine = _native.CallbackEncoder.native_fast_cpu(
    tokenizer_json,
    "synthetic_byte_level",
    hashlib.sha256(tokenizer_json).hexdigest(),
    4,
    2,
    False,
    pclass_table(),
)
texts = ["hello", "世界"]
actual = engine.encode_batch(texts)
expected = [reference.encode(text).ids for text in texts]
assert actual == expected
print(f"wheel runtime verified: {installed}")
PY

printf '%s\n' "Packaging smoke: PASS"
