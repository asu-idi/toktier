"""Command-line interface for toktier."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
from collections.abc import Callable, Sequence
from typing import NoReturn, cast

from . import __version__
from .artifacts import (
    ArtifactEntry,
    ArtifactManifest,
    ArtifactSource,
    ArtifactStore,
    HuggingFaceSource,
    VerifiedArtifact,
    export_bundle,
    import_bundle,
)
from .artifacts.store import fetch_availability
from .artifacts.tables import ARTIFACT_MANIFEST
from .config import Config
from .errors import ToktierError
from .paths import artifact_cache_dir, kernel_cache_dir, store_state_dir

_USAGE_ERROR = 64
_TOKTIER_ERROR = 2


class _UsageError(Exception):
    """Signal an argparse usage failure without terminating the process."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self._print_message(f"{self.prog}: error: {message}\n", sys.stderr)
        raise _UsageError


def _toktier_version() -> str:
    try:
        return importlib.metadata.version("toktier")
    except importlib.metadata.PackageNotFoundError:
        return __version__


def _nvcc_search() -> tuple[str | None, list[str]]:
    """Locate ``nvcc`` the way the JIT loader's build system does.

    The kernel loader delegates compilation to the torch extension
    build system, which resolves the CUDA toolkit in this order: the
    ``CUDA_HOME`` environment variable, then ``CUDA_PATH``, then the
    ``nvcc`` on ``PATH``, then the conventional ``/usr/local/cuda``
    root. This check walks the same order without importing torch and
    records every location it consulted, so the report names where it
    looked rather than answering from a single ``PATH`` lookup that the
    build system does not limit itself to.

    An explicitly set ``CUDA_HOME``/``CUDA_PATH`` is authoritative for
    the build system -- it stops the search whether or not ``nvcc`` is
    present under it -- and the same is true here.

    Environment caliber: the two variables read here are the CUDA
    toolchain's own, not toktier configuration; they are observed and
    reported, never stored and never acted on (``config.md`` Section 3
    keeps toktier's environment surface to the frozen five, read once
    by ``Config``).
    """
    checked: list[str] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if not root:
            checked.append(f"{variable}: not set")
            continue
        candidate = os.path.join(root, "bin", "nvcc")
        if os.path.isfile(candidate):
            checked.append(f"{variable}: {candidate} (found)")
            return candidate, checked
        checked.append(f"{variable}: {candidate} (not found)")
        return None, checked
    from_path = shutil.which("nvcc")
    if from_path is not None:
        checked.append(f"PATH: {from_path} (found)")
        return from_path, checked
    checked.append("PATH: not found")
    default = "/usr/local/cuda/bin/nvcc"
    if os.path.isfile(default):
        checked.append(f"default: {default} (found)")
        return default, checked
    checked.append(f"default: {default} (not found)")
    return None, checked


def _doctor_report(
    config: Config, *, source: ArtifactSource | None
) -> dict[str, object]:
    # Fetch availability is reported field by field. A single "offline"
    # line would answer three different questions with one word, and the
    # interesting case is exactly the one it hides: a configuration that
    # is not offline in front of a source that is.
    availability = fetch_availability(config, source)
    # Looking up torch.cuda would import its parent package. The top-level
    # cuda package is the CUDA probe that preserves this command's no-import
    # guarantee for torch.
    nvcc_path, nvcc_checked = _nvcc_search()
    from .backends.fast_cpu import ENGINE_MODULE, fast_cpu_engine_facts

    fast_cpu = fast_cpu_engine_facts()
    from .repair.fastokens import fastokens_distribution_identity

    fastokens_version, fastokens_digest = fastokens_distribution_identity()
    return {
        "python_version": platform.python_version(),
        "toktier_version": _toktier_version(),
        "artifact_cache_dir": str(artifact_cache_dir(config)),
        "kernel_cache_dir": str(kernel_cache_dir(config)),
        "store_state_dir": str(store_state_dir(config)),
        "configured_offline": availability.configured_offline,
        "artifact_source": availability.source_name or "none",
        "source_offline": availability.source_offline,
        "artifact_fetch_available": availability.available,
        "torch_available": importlib.util.find_spec("torch") is not None,
        "cuda_available": importlib.util.find_spec("cuda") is not None,
        "gigatoken_available": fast_cpu.version is not None,
        "gigatoken_delivery": "vendored",
        "gigatoken_module": ENGINE_MODULE,
        "gigatoken_runtime_ready": (
            fast_cpu.version is not None
            and importlib.util.find_spec("transformers") is not None
        ),
        "gigatoken_version": fast_cpu.version,
        "gigatoken_native_digest": fast_cpu.binary_digest,
        "gigatoken_repair_config_digest": fast_cpu.config_digest,
        "fastokens_available": fastokens_version is not None,
        "fastokens_version": fastokens_version,
        "fastokens_distribution_digest": fastokens_digest,
        "fastokens_policy": "experimental",
        "fastokens_exact_id_guarantee": False,
        "nvcc_available": nvcc_path is not None,
        "nvcc_path": nvcc_path,
        "nvcc_checked": nvcc_checked,
    }


def _print_doctor_human(report: dict[str, object]) -> None:
    for name, value in report.items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, list):
            rendered = "; ".join(str(item) for item in value)
        elif value is None:
            rendered = "none"
        else:
            rendered = str(value)
        print(f"{name}: {rendered}")


def _artifact_manifest() -> ArtifactManifest:
    """Return the artifact identities shipped with this package.

    The file is generated by ``tools/generate_artifact_manifest.py`` and
    installed inside the package, so a wheel resolves the same families
    a source checkout does. It is read on each call: the command runs
    once per process, and a cached manifest would only add a way for a
    long-lived process to keep stale digests.
    """
    return ArtifactManifest.load(ARTIFACT_MANIFEST)


def _artifact_store(
    config: Config, *, source: ArtifactSource | None
) -> ArtifactStore:
    return ArtifactStore(_artifact_manifest(), config=config, source=source)


def _print_artifact(action: str, artifact: VerifiedArtifact) -> None:
    print(f"{action} {artifact.family}: {artifact.directory}")


def _doctor(arguments: argparse.Namespace) -> int:
    # The same source ``artifacts fetch`` would use, constructed the same
    # way. It reads its environment and imports no hub client, so the
    # report describes the fetch path this machine would actually take.
    report = _doctor_report(Config.resolve(), source=HuggingFaceSource())
    if arguments.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        _print_doctor_human(report)
    return 0


def _artifacts_fetch(arguments: argparse.Namespace) -> int:
    config = Config.resolve()
    store = _artifact_store(config, source=HuggingFaceSource())
    if arguments.force:
        artifact = store.verify(arguments.family)
    else:
        artifact = store.ensure(arguments.family)
    _print_artifact("fetched", artifact)
    return 0


def _artifacts_verify(arguments: argparse.Namespace) -> int:
    store = _artifact_store(Config.resolve(), source=None)
    artifact = store.verify(arguments.family)
    _print_artifact("verified", artifact)
    return 0


def _artifacts_export(arguments: argparse.Namespace) -> int:
    # Re-hash before exporting (no source, so nothing is fetched): the
    # bundle records the digests of the bytes it packs, and this check
    # ties those bytes to the shipped manifest first. Bytes that fail
    # verification are never exported.
    store = _artifact_store(Config.resolve(), source=None)
    artifact = store.verify(arguments.family)
    entry = store.manifest.get(arguments.family)
    bundle = export_bundle(
        arguments.out,
        entry.directory_name,
        {item.name: artifact.path(item.name) for item in entry.files},
    )
    print(f"exported {artifact.family}: {bundle}")
    return 0


def _artifacts_import(arguments: argparse.Namespace) -> int:
    # The bundle is validated and every file digest-checked before the
    # atomic install; `artifacts verify FAMILY` afterwards binds the
    # installed bytes to the digests the shipped manifest pins.
    target = import_bundle(arguments.bundle, artifact_cache_dir(Config.resolve()))
    print(f"imported {target.name}: {target}")
    return 0


def _inspect_entry(entry: ArtifactEntry) -> dict[str, object]:
    return {
        "family": entry.family,
        "repo_id": entry.repo_id,
        "revision": entry.revision,
        "files": {
            item.name: {"sha256": item.sha256, "size": item.size}
            for item in entry.files
        },
    }


def _inspect(arguments: argparse.Namespace) -> int:
    """Print the identity the package pins, from the shipped manifest.

    Reads no cache and no network: the answer is what the package would
    verify against, which is the value a deployment check needs. With a
    family it prints that entry; without one, every family and its
    per-file digests.
    """
    manifest = _artifact_manifest()
    if arguments.family is not None:
        entries = [manifest.get(arguments.family)]
    else:
        entries = [manifest.get(family) for family in sorted(manifest.families())]
    if arguments.json:
        report: object = (
            _inspect_entry(entries[0])
            if arguments.family is not None
            else {entry.family: _inspect_entry(entry) for entry in entries}
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    for entry in entries:
        if arguments.family is not None:
            print(f"family: {entry.family}")
            print(f"repo_id: {entry.repo_id}")
            print(f"revision: {entry.revision}")
            for item in entry.files:
                print(f"{item.name}: sha256 {item.sha256} ({item.size} bytes)")
        else:
            for item in entry.files:
                print(f"{entry.family}: {item.name} sha256 {item.sha256}")
    return 0


def _version(arguments: argparse.Namespace) -> int:
    del arguments
    print(_toktier_version())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="toktier")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="show environment diagnostics")
    doctor.add_argument("--json", action="store_true", help="emit JSON")
    doctor.set_defaults(handler=_doctor)

    artifacts = commands.add_parser("artifacts", help="manage artifacts")
    artifact_commands = artifacts.add_subparsers(
        dest="artifact_command", required=True
    )

    fetch = artifact_commands.add_parser("fetch", help="fetch an artifact")
    fetch.add_argument("family", metavar="FAMILY")
    fetch.add_argument("--force", action="store_true", help="re-hash cached files")
    fetch.set_defaults(handler=_artifacts_fetch)

    verify = artifact_commands.add_parser("verify", help="verify an artifact")
    verify.add_argument("family", metavar="FAMILY")
    verify.set_defaults(handler=_artifacts_verify)

    export = artifact_commands.add_parser(
        "export", help="write a verified artifact as an air-gap bundle"
    )
    export.add_argument("family", metavar="FAMILY")
    export.add_argument(
        "--out",
        metavar="BUNDLE",
        required=True,
        help="path of the tar bundle to write",
    )
    export.set_defaults(handler=_artifacts_export)

    import_ = artifact_commands.add_parser(
        "import", help="verify a bundle and install it into the cache"
    )
    import_.add_argument("bundle", metavar="BUNDLE")
    import_.set_defaults(handler=_artifacts_import)

    inspect = commands.add_parser(
        "inspect", help="show the artifact identity the package pins"
    )
    inspect.add_argument("family", metavar="FAMILY", nargs="?")
    inspect.add_argument("--json", action="store_true", help="emit JSON")
    inspect.set_defaults(handler=_inspect)

    version = commands.add_parser("version", help="show the toktier version")
    version.set_defaults(handler=_version)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the toktier command-line interface."""
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
    except _UsageError:
        return _USAGE_ERROR

    handler = cast(Callable[[argparse.Namespace], int], arguments.handler)
    try:
        return handler(arguments)
    except ToktierError as error:
        print(f"error {error.code}: {error}", file=sys.stderr)
        return _TOKTIER_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
