#!/usr/bin/env python3
"""Record workspace source identities consumed by an unpacked Rust crate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fast_cpu_source_identity
import native_host_source_identity
import rust_api_source_identity
from compute_identity_v2 import source_digest as source_digest_v2

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "crates/toktier/data/build/source_identity.json"


def document() -> dict[str, object]:
    return {
        "schema": "toktier.rust_package_source_identity.v1",
        "identity_rule": (
            "content-addressed source sets; independent of repository history"
        ),
        "rust_api_source_sha256": rust_api_source_identity.source_digest(),
        "rust_api_source_sha256_v2": source_digest_v2("rust_api"),
        "fast_cpu_source_sha256": fast_cpu_source_identity.source_digest(),
        "fast_cpu_source_sha256_v2": source_digest_v2("fast_cpu"),
        "native_host_source_sha256": native_host_source_identity.source_digest(),
        "native_host_source_sha256_v2": source_digest_v2("native_host"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    expected = (json.dumps(document(), indent=2, sort_keys=True) + "\n").encode()
    observed = TARGET.read_bytes() if TARGET.is_file() else None
    if arguments.check:
        if observed != expected:
            print(f"error: {TARGET} is missing or stale")
            return 1
        print(f"{TARGET}: check passed")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(expected)
    TARGET.chmod(0o644)
    print(f"{TARGET}: updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
