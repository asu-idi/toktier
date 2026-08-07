"""Command-line interface acceptance tests.

Two kinds of artifact test live here, and the difference matters:

* the **shipped-manifest** tests run the commands exactly as an
  installed wheel does -- ``cli._artifact_manifest()`` is not replaced,
  so they fail if the manifest the package ships cannot resolve a
  family;
* the **synthetic-manifest** tests replace the manifest and the source
  with a small pair so that the fetch and mismatch paths can be driven
  with bytes a test can produce. They test the plumbing, not the
  configuration a user gets.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
from pathlib import Path
from typing import NoReturn

import pytest

from toktier import __version__, cli
from toktier.artifacts import (
    ArtifactEntry,
    ArtifactFile,
    ArtifactManifest,
)

FAMILY = "demo_family"
REVISION = "a" * 40
GOOD = b'{"version": "1.0", "model": {}}\n'
BAD = b"corrupted bytes\n"


class StaticSource:
    """Serve one payload without reaching a network client."""

    name = "test"
    offline = False

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def fetch(
        self,
        entry: ArtifactEntry,
        artifact_file: ArtifactFile,
        destination: Path,
    ) -> None:
        del entry
        self.calls.append(artifact_file.name)
        destination.write_bytes(self.payload)


def _manifest() -> ArtifactManifest:
    return ArtifactManifest.from_mapping(
        {
            FAMILY: {
                "repo_id": "demo/demo",
                "revision": REVISION,
                "files": {
                    "tokenizer.json": {
                        "sha256": hashlib.sha256(GOOD).hexdigest(),
                        "size": len(GOOD),
                    }
                },
            }
        },
        source="<cli-test>",
    )


def _set_doctor_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    from toktier.backends import fast_cpu
    from toktier.backends.fast_cpu import FastCpuEngineFacts
    from toktier.repair import fastokens

    def find_spec(name: str) -> object | None:
        assert name in {"torch", "cuda", "transformers"}
        return object() if name in {"torch", "transformers"} else None

    def which(name: str) -> str:
        assert name == "nvcc"
        return "/opt/cuda/bin/nvcc"

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(
        fast_cpu,
        "fast_cpu_engine_facts",
        lambda: FastCpuEngineFacts(
            version="0.10.0+toktier.pinned.1",
            binary_digest="f" * 64,
            config_digest="e" * 64,
        ),
    )
    monkeypatch.setattr(
        fastokens,
        "fastokens_distribution_identity",
        lambda: (importlib.metadata.version("fastokens"), "d" * 64),
    )
    # The nvcc search consults the loader's toolkit roots first; unset
    # them so the deterministic outcome is the PATH lookup above.
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)


_NVCC_CHECKED_VIA_PATH = [
    "CUDA_HOME: not set",
    "CUDA_PATH: not set",
    "PATH: /opt/cuda/bin/nvcc (found)",
]


def _set_artifact_source(
    monkeypatch: pytest.MonkeyPatch, source: StaticSource
) -> None:
    """Install the synthetic manifest and source (see the module docstring)."""
    manifest = _manifest()
    monkeypatch.setattr(cli, "_artifact_manifest", lambda: manifest)
    monkeypatch.setattr(cli, "HuggingFaceSource", lambda: source)


def _smallest_shipped_family() -> tuple[str, int]:
    """Family of the shipped manifest with the fewest bytes to hash."""
    manifest = cli._artifact_manifest()
    sizes = {
        family: sum(item.size or 0 for item in manifest.get(family).files)
        for family in manifest.families()
    }
    family = min(sizes, key=lambda name: (sizes[name], name))
    return family, sizes[family]


def test_doctor_human(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)

    exit_code = cli.main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == (
        f"python_version: {platform.python_version()}\n"
        "toktier_version: 1.2.3\n"
        f"artifact_cache_dir: {home / 'cache' / 'artifacts'}\n"
        f"kernel_cache_dir: {home / 'cache' / 'kernels'}\n"
        f"store_state_dir: {home / 'state' / 'store'}\n"
        "configured_offline: true\n"
        "artifact_source: huggingface\n"
        "source_offline: false\n"
        "artifact_fetch_available: false\n"
        "torch_available: true\n"
        "cuda_available: false\n"
        "gigatoken_available: true\n"
        "gigatoken_delivery: vendored\n"
        "gigatoken_module: toktier._vendor.gigatoken_rs\n"
        "gigatoken_runtime_ready: true\n"
        "gigatoken_version: 0.10.0+toktier.pinned.1\n"
        f"gigatoken_native_digest: {'f' * 64}\n"
        f"gigatoken_repair_config_digest: {'e' * 64}\n"
        "fastokens_available: true\n"
        "fastokens_version: 1.2.3\n"
        f"fastokens_distribution_digest: {'d' * 64}\n"
        "fastokens_policy: experimental\n"
        "fastokens_exact_id_guarantee: false\n"
        "nvcc_available: true\n"
        "nvcc_path: /opt/cuda/bin/nvcc\n"
        f"nvcc_checked: {'; '.join(_NVCC_CHECKED_VIA_PATH)}\n"
    )
    assert captured.err == ""


def test_doctor_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    monkeypatch.setenv("TOKTIER_OFFLINE", "0")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.0.0")
    _set_doctor_probes(monkeypatch)
    expected = {
        "python_version": platform.python_version(),
        "toktier_version": "2.0.0",
        "artifact_cache_dir": str(home / "cache" / "artifacts"),
        "kernel_cache_dir": str(home / "cache" / "kernels"),
        "store_state_dir": str(home / "state" / "store"),
        "configured_offline": False,
        "artifact_source": "huggingface",
        "source_offline": False,
        "artifact_fetch_available": True,
        "torch_available": True,
        "cuda_available": False,
        "gigatoken_available": True,
        "gigatoken_delivery": "vendored",
        "gigatoken_module": "toktier._vendor.gigatoken_rs",
        "gigatoken_runtime_ready": True,
        "gigatoken_version": "0.10.0+toktier.pinned.1",
        "gigatoken_native_digest": "f" * 64,
        "gigatoken_repair_config_digest": "e" * 64,
        "fastokens_available": True,
        "fastokens_version": "2.0.0",
        "fastokens_distribution_digest": "d" * 64,
        "fastokens_policy": "experimental",
        "fastokens_exact_id_guarantee": False,
        "nvcc_available": True,
        "nvcc_path": "/opt/cuda/bin/nvcc",
        "nvcc_checked": _NVCC_CHECKED_VIA_PATH,
    }

    exit_code = cli.main(["doctor", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ) + "\n"
    assert json.loads(captured.out) == expected
    assert captured.err == ""


def test_doctor_separates_a_configuration_from_an_offline_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The case a single ``offline`` field used to hide.

    The configuration allows fetching and the hub client refuses to
    reach out, so the honest answer is that fetching is unavailable.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    _set_doctor_probes(monkeypatch)

    exit_code = cli.main(["doctor", "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["configured_offline"] is False
    assert report["source_offline"] is True
    assert report["artifact_fetch_available"] is False
    assert report["artifact_source"] == "huggingface"


def test_doctor_nvcc_follows_the_build_system_search_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A toolkit under ``CUDA_HOME`` is found without ``nvcc`` on PATH.

    The JIT loader builds through ``torch.utils.cpp_extension``, which
    consults ``CUDA_HOME`` before the ``PATH``; the doctor answers the
    same question the same way.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    cuda_home = tmp_path / "cuda-home"
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\n")
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["nvcc_available"] is True
    assert report["nvcc_path"] == str(nvcc)
    assert report["nvcc_checked"] == [f"CUDA_HOME: {nvcc} (found)"]


def test_doctor_treats_a_set_cuda_home_as_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``CUDA_HOME`` without ``nvcc`` stops the search, as the build does.

    ``torch.utils.cpp_extension`` takes a set ``CUDA_HOME`` as the
    toolkit root without falling back to the ``PATH``, so reporting the
    ``PATH`` copy as available here would promise a build that the
    loader would not attempt with these settings.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    monkeypatch.setenv("TOKTIER_OFFLINE", "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvcc")
    empty_home = tmp_path / "cuda-home-empty"
    empty_home.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(empty_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    exit_code = cli.main(["doctor", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["nvcc_available"] is False
    assert report["nvcc_path"] is None
    assert report["nvcc_checked"] == [
        f"CUDA_HOME: {empty_home / 'bin' / 'nvcc'} (not found)"
    ]


# -- shipped manifest (nothing replaced) -------------------------------


def test_artifacts_verify_resolves_a_shipped_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``verify`` reaches the cache lookup for a family the package ships.

    With an empty cache the command still fails, but on the missing
    bytes rather than on the family: an empty manifest would refuse
    every family here, which is the failure this test exists for.
    """
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    family, _ = _smallest_shipped_family()

    exit_code = cli.main(["artifacts", "verify", family])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == (
        "error ARTIFACT_NOT_FOUND: artifact file 'tokenizer.json' of "
        f"{family!r} is not in the cache and fetching is disabled (offline)\n"
    )


def test_artifacts_fetch_refuses_a_family_the_package_does_not_ship(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown family fails on the name, before any source is asked."""
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))

    exit_code = cli.main(["artifacts", "fetch", "no_such_family"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == (
        "error ARTIFACT_NOT_FOUND: unknown tokenizer family 'no_such_family'\n"
    )


# -- synthetic manifest (plumbing) -------------------------------------


def test_artifacts_fetch_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    source = StaticSource(GOOD)
    _set_artifact_source(monkeypatch, source)
    directory = home / "cache" / "artifacts" / f"{FAMILY}-{REVISION[:12]}"

    exit_code = cli.main(["artifacts", "fetch", FAMILY, "--force"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"fetched {FAMILY}: {directory}\n"
    assert captured.err == ""
    assert source.calls == ["tokenizer.json"]
    assert (directory / "tokenizer.json").read_bytes() == GOOD


def test_artifacts_fetch_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    source = StaticSource(BAD)
    _set_artifact_source(monkeypatch, source)

    exit_code = cli.main(["artifacts", "fetch", FAMILY])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == (
        "error ARTIFACT_HASH_MISMATCH: content hash mismatch for "
        f"{FAMILY}/tokenizer.json\n"
    )
    assert source.calls == ["tokenizer.json", "tokenizer.json"]


def test_artifacts_export_then_import_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The air-gap recipe of the README, end to end.

    Fetch on one machine, export, import into a second machine's empty
    cache, then bind the installed bytes to the shipped digests with
    ``verify`` -- all through the command line, with no source configured
    on the second machine.
    """
    online_home = tmp_path / "online-home"
    offline_home = tmp_path / "offline-home"
    bundle = tmp_path / "demo.tar"
    monkeypatch.setenv("TOKTIER_HOME", str(online_home))
    source = StaticSource(GOOD)
    _set_artifact_source(monkeypatch, source)
    alias = f"{FAMILY}-{REVISION[:12]}"

    fetched = cli.main(["artifacts", "fetch", FAMILY])
    exported = cli.main(["artifacts", "export", FAMILY, "--out", str(bundle)])
    monkeypatch.setenv("TOKTIER_HOME", str(offline_home))
    imported = cli.main(["artifacts", "import", str(bundle)])
    verified = cli.main(["artifacts", "verify", FAMILY])

    captured = capsys.readouterr()
    installed = offline_home / "cache" / "artifacts" / alias
    assert (fetched, exported, imported, verified) == (0, 0, 0, 0)
    assert captured.out == (
        f"fetched {FAMILY}: {online_home / 'cache' / 'artifacts' / alias}\n"
        f"exported {FAMILY}: {bundle}\n"
        f"imported {alias}: {installed}\n"
        f"verified {FAMILY}: {installed}\n"
    )
    assert captured.err == ""
    assert source.calls == ["tokenizer.json"]
    assert (installed / "tokenizer.json").read_bytes() == GOOD


def test_artifacts_export_refuses_an_empty_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Export verifies first; with nothing cached there is nothing to pack."""
    monkeypatch.setenv("TOKTIER_HOME", str(tmp_path / "toktier-home"))
    _set_artifact_source(monkeypatch, StaticSource(GOOD))
    bundle = tmp_path / "demo.tar"

    exit_code = cli.main(["artifacts", "export", FAMILY, "--out", str(bundle)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "ARTIFACT_NOT_FOUND" in captured.err
    assert not bundle.exists()


def test_artifacts_import_rejects_a_file_that_is_not_a_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "toktier-home"
    monkeypatch.setenv("TOKTIER_HOME", str(home))
    bundle = tmp_path / "not-a-bundle.tar"
    bundle.write_bytes(b"these bytes are not a tar archive\n")

    exit_code = cli.main(["artifacts", "import", str(bundle)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error BUNDLE_INVALID: ")
    cache = home / "cache" / "artifacts"
    assert not cache.exists() or not any(
        path for path in cache.iterdir() if not path.name.startswith(".")
    )


def test_inspect_prints_one_family(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_artifact_source(monkeypatch, StaticSource(GOOD))
    digest = hashlib.sha256(GOOD).hexdigest()

    exit_code = cli.main(["inspect", FAMILY])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == (
        f"family: {FAMILY}\n"
        "repo_id: demo/demo\n"
        f"revision: {REVISION}\n"
        f"tokenizer.json: sha256 {digest} ({len(GOOD)} bytes)\n"
    )
    assert captured.err == ""


def test_inspect_json_matches_the_shipped_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``inspect --json`` over the manifest the package actually ships."""
    manifest = cli._artifact_manifest()

    exit_code = cli.main(["inspect", "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert sorted(report) == sorted(manifest.families())
    for family, block in report.items():
        entry = manifest.get(family)
        assert block["repo_id"] == entry.repo_id
        assert block["revision"] == entry.revision
        assert block["files"] == {
            item.name: {"sha256": item.sha256, "size": item.size}
            for item in entry.files
        }


def test_inspect_refuses_an_unknown_family(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["inspect", "no_such_family"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == (
        "error ARTIFACT_NOT_FOUND: unknown tokenizer family 'no_such_family'\n"
    )


def test_version_uses_package_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing_version(name: str) -> NoReturn:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_version)

    exit_code = cli.main(["version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"{__version__}\n"
    assert captured.err == ""
