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
from toktier.backends.fast_cpu import ENGINE_MODULE, fast_cpu_engine_facts

repository = Path(os.environ["TOKTIER_REPOSITORY_ROOT"]).resolve()
installed = Path(toktier.__file__).resolve()
assert repository not in installed.parents, installed
assert toktier.__version__ == "0.1.0"
assert tokenizers.__version__ == "0.22.2"
assert transformers.__version__ == "4.57.6"
assert importlib.util.find_spec("gigatoken") is None
assert ENGINE_MODULE not in sys.modules

facts = fast_cpu_engine_facts()
assert facts.binary_digest == (
    "9a701047dafa1cdebc168851d0548a0ca"
    "af08d0523d70911cc7a24112ccf92a3"
)
assert ENGINE_MODULE not in sys.modules

from toktier._vendor import gigatoken_rs

alphabet = pre_tokenizers.ByteLevel.alphabet()
reference = Tokenizer(
    models.BPE(vocab={token: index for index, token in enumerate(alphabet)}, merges=[])
)
reference.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False, use_regex=True
)
reference.decoder = decoders.ByteLevel()
engine = gigatoken_rs.load_hf_json(reference.to_str())
texts = ["hello", "世界"]
actual = engine.encode_batch_list(texts, parallel=False)
expected = [reference.encode(text).ids for text in texts]
assert actual == expected

manifest_path = installed.parent / "_vendor" / "gigatoken_build.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
native = installed.parent / "_vendor" / manifest["native_file"]
assert hashlib.sha256(native.read_bytes()).hexdigest() == manifest["native_sha256"]
print(f"wheel runtime verified: {installed}")
PY

printf '%s\n' "Packaging smoke: PASS"
