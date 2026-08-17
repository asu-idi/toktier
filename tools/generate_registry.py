# Verify tables/support_registry.json, the machine-generated certification registry.
"""Check the shipped certification registry.

The registry states exactly what has been certified and nothing more, and it
is produced by maintainer tooling from the recorded campaign readings
(``docs/contracts/registry.md`` Section 7). The release-facing native CPU and
GPU parity summaries ship under ``readings/``; the larger archival campaign
inputs remain outside the distribution. What ships here is the verification
surface: the serialisation is deterministic, the document validates against
its schema, and the recorded root digest matches the document. A hand edit
fails ``--check``.

Kernel identities: the GPU certificate binds the digest of the shipped
kernel sources (computed here through the same torch-free helper the
loader uses) and the combined class-table digest from the routing data.
The class-table binding is real only once ``generate_class_tables.py``
has backfilled every table digest; before that the registry carries
``PLACEHOLDER_SHA256`` for it. The placeholder is all zeroes, so a
loader comparing a real build against it always closes the accelerated
path: the failure mode is a refused fast path, never an unbacked one.
``--release-check`` refuses to pass while any placeholder, unrecorded
digest, or missing packaged table remains.

Packaged copy: the registry also ships inside the package
(``src/toktier/routing/tables/support_registry.v1.json``) so installed
wheels can report certification statuses. ``--check`` verifies that the
packaged copy is byte-identical to the repository copy, so both stay
covered by the same discipline.

Delivery refinements: a record that carries ``deliveries`` sub-entries
is checked against the shipped artifacts of both delivery modes. The
``jit`` sub-entry must restate the top-level (JIT-era) view verbatim,
and the ``prebuilt`` sub-entry must bind the packaged fatbin, each embedded
architecture image, and the executing Rust host's domain-separated source
digest, exact rustc identity, and release flags. Certified sm_89/sm_120 rows
must also have current hardware parity readings bound to those same facts.
All identity helpers are torch-free, so this check still needs no GPU stack.

Usage::

    python tools/generate_registry.py --check
    python tools/generate_registry.py --release-check
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from compute_identity_v2 import source_digest as source_digest_v2
from native_host_source_identity import source_digest as native_host_source_digest
from registry_common import (
    PLACEHOLDER_SHA256,
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    load_json,
    sha256_of_file,
    verify_file,
    write_document,
)
from scan_common import DECLINED, vendored_source_archive
from source_identity_common import coverage_problems as source_coverage_problems
from update_fast_cpu_registry import augmented_document

# registry_common puts src/ on sys.path; the kernel identity helpers are
# torch-free by design so this check needs no GPU stack.
from toktier.engine.gpu.class_tables import (
    class_table_digest as combined_class_table_digest,
)
from toktier.engine.gpu.families import (
    DEFAULT_FAMILY_TABLE_PATH,
    KernelFamilyTable,
)
from toktier.engine.gpu.toolchain import JIT_TOOLCHAIN_CONSTRAINT
from toktier.kernels import kernel_source_digest
from toktier.kernels.bindings import bare_sha256

TOOL_NAME = "tools/generate_registry.py"
# One tool identity, two halves: the maintainer-side generator writes the
# registry and stamps this version into its ``generated_by``; this
# verification half moves in lockstep with it. 1.2.0 is the generator
# version that produced the shipped document (it grew the delivery
# refinements this half checks).
TOOL_VERSION = "1.2.0"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "tables" / "support_registry.json"
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "support_registry.schema.json"

#: Installed copy of the registry, shipped as package data so wheels can
#: report certification statuses without a source checkout. It must stay
#: byte-identical to the repository copy; ``--check`` verifies that.
PACKAGED_OUTPUT = (
    REPOSITORY_ROOT
    / "src"
    / "toktier"
    / "routing"
    / "tables"
    / "support_registry.v1.json"
)

#: Which architectures the prebuilt delivery is certified for, and where
#: each one's campaign reading lives. This mapping *is* the certified
#: device list: an architecture enters it only when a reading of the
#: shipped fatbin on that hardware exists, and everything else the fatbin
#: carries an image for is listed as experimental instead. Every reading
#: has to name the same native host, because one prebuilt row binds one
#: host identity for all of them.
PREBUILT_HARDWARE_READINGS = {
    "sm_80": REPOSITORY_ROOT / "readings" / "gpu_native_frontend_sm80_parity.json",
    "sm_89": REPOSITORY_ROOT / "readings" / "gpu_native_frontend_sm89_parity.json",
    "sm_90": REPOSITORY_ROOT / "readings" / "gpu_native_frontend_sm90_parity.json",
    "sm_120": REPOSITORY_ROOT / "readings" / "gpu_native_frontend_sm120_parity.json",
}

ORACLE_PACKAGE = "tokenizers"


#: Where the extension the check read came from, once one was adopted.
#: Reported in failure messages so a mismatch names the file it judged.
_ADOPTED_NATIVE_EXTENSION: Path | None = None


def _installed_native_extension() -> Path | None:
    """A ``toktier._native`` extension installed outside this source tree.

    ``registry_common`` puts ``src`` first on ``sys.path`` so the tools
    read this repository's Python. A source tree that has never been
    built has no extension there, which is the ordinary shape of a
    snapshot checked out beside an installed wheel. The wheel's
    extension is a legitimate subject for this check -- its identity is
    still compared exactly against the current source set -- so it is
    worth finding rather than refusing on setup grounds.
    """
    from importlib.machinery import EXTENSION_SUFFIXES

    source_package = (REPOSITORY_ROOT / "src" / "toktier").resolve()
    for entry in sys.path:
        if not entry:
            continue
        directory = Path(entry).resolve() / "toktier"
        if directory == source_package or not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("_native*")):
            if candidate.is_file() and any(
                candidate.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES
            ):
                return candidate
    return None


def _adopt_installed_native_extension() -> Path | None:
    """Bind ``toktier._native`` to an installed extension, saying so.

    Returns the path adopted, or ``None`` when there is none to adopt.
    Nothing about the comparison changes: the adopted extension has to
    carry exactly the identity the current source set hashes to.
    """
    global _ADOPTED_NATIVE_EXTENSION
    import importlib.util
    from importlib.machinery import ExtensionFileLoader

    path = _installed_native_extension()
    if path is None:
        return None
    loader = ExtensionFileLoader("toktier._native", str(path))
    spec = importlib.util.spec_from_loader("toktier._native", loader)
    if spec is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    import toktier

    sys.modules["toktier._native"] = module
    toktier._native = module
    _ADOPTED_NATIVE_EXTENSION = path
    print(
        f"note: no compiled extension in {REPOSITORY_ROOT / 'src' / 'toktier'}; "
        f"reading native-host identity from the installed {path}",
        file=sys.stderr,
    )
    return path


def native_host_bindings(*, include_v2: bool = False) -> dict[str, Any]:
    """Source/build facts of the Rust host that executes prebuilt requests.

    The facts are compile-time constants of the extension, so this needs
    a built extension to read: this source tree's, or -- when it has
    none -- the one an installed wheel provides. Either way the digest
    must equal the one the current source set hashes to; adopting an
    installed extension changes where the identity is read, never
    whether it has to match.
    """
    from toktier.engine.gpu.native import native_host_build_facts

    facts = native_host_build_facts()
    if facts.source_digest is None:
        _adopt_installed_native_extension()
        facts = native_host_build_facts()
    source = native_host_source_digest()
    if facts.source_digest is None:
        raise GenerationError(
            "no native host identity could be read: this source tree has "
            f"no compiled extension in {REPOSITORY_ROOT / 'src' / 'toktier'} "
            "and no installed toktier._native was found on sys.path. This "
            "check binds the shipped registry to the extension that "
            "executes prebuilt GPU requests, so it needs one of the two: "
            "build the extension (maturin develop, or maturin build "
            "--locked and place it in src/toktier), or run this check in "
            "an environment where the matching toktier wheel is installed"
        )
    if facts.source_digest != source:
        read_from = (
            f" (read from {_ADOPTED_NATIVE_EXTENSION})"
            if _ADOPTED_NATIVE_EXTENSION is not None
            else ""
        )
        raise GenerationError(
            "the loaded native host was not built from the current source set"
            f"{read_from} (loaded={facts.source_digest!r}, current={source})"
        )
    if not facts.build_flags or facts.toolchain is None:
        raise GenerationError("the loaded native host exposes incomplete build facts")
    bindings = {
        "host_source_digest": source,
        "host_build_flags": list(facts.build_flags),
        "host_toolchain": facts.toolchain,
    }
    if include_v2:
        bindings["host_source_digest_v2"] = source_digest_v2("native_host")
    return bindings


def kernel_bindings() -> tuple[str, str | None]:
    """The kernel identities a ``certified_source`` record binds.

    The source digest covers the shipped CUDA sources and is always
    computable. The combined class-table digest is real only once the
    table generator has backfilled every table digest into the routing
    data; before that it is ``None`` and the registry carries the
    placeholder, which every loader treats as a failed verification.
    """
    source = bare_sha256(kernel_source_digest())
    table = KernelFamilyTable.load()
    specs = list(table.class_tables())
    if any(spec.sha256 is None for spec in specs):
        return source, None
    return source, bare_sha256(combined_class_table_digest(specs))


def kernel_binding_problems(registry_path: Path) -> list[str]:
    """Compare the registry's kernel bindings with the shipped sources."""
    problems: list[str] = []
    source_digest, class_digest = kernel_bindings()
    document = load_json(registry_path)
    for artifact in document.get("artifacts", []):
        family = artifact.get("family", "<unknown>")
        entry = (artifact.get("backends") or {}).get("gpu")
        if not isinstance(entry, dict):
            continue
        if entry.get("source_digest") != source_digest:
            problems.append(
                f"{family}/gpu: source_digest does not match the shipped "
                f"kernel sources ({source_digest})"
            )
        if (
            entry.get("status") == "certified_source"
            and entry.get("toolchain") != JIT_TOOLCHAIN_CONSTRAINT
        ):
            problems.append(
                f"{family}/gpu: toolchain constraint does not bind the "
                "actual NVCC/runtime/torch triple"
            )
        jit = (entry.get("deliveries") or {}).get("jit")
        if (
            isinstance(jit, dict)
            and jit.get("status") == "certified_source"
            and jit.get("toolchain") != JIT_TOOLCHAIN_CONSTRAINT
        ):
            problems.append(
                f"{family}/gpu/jit: toolchain constraint does not bind "
                "the actual NVCC/runtime/torch triple"
            )
        recorded_class = entry.get("class_table_digest")
        expected_class = (
            class_digest if class_digest is not None else PLACEHOLDER_SHA256
        )
        if recorded_class != expected_class:
            problems.append(
                f"{family}/gpu: class_table_digest does not match the "
                f"shipped routing data ({expected_class})"
            )
    return problems


def sync_jit_toolchain_constraint(registry_path: Path) -> None:
    """Synchronize the explicit compiler tuple and recompute the root.

    The campaign rows already name the two judged CUDA/PyTorch
    environments. This maintenance operation makes the previously
    implicit meaning of ``CUDA`` explicit as actual NVCC plus the torch
    runtime label; it does not add a toolchain, architecture, or reading.
    """
    if registry_path.resolve() != DEFAULT_OUTPUT.resolve():
        raise GenerationError(
            "--sync-jit-toolchain is only valid for the shipped registry"
        )
    schema = load_json(SCHEMA_PATH)
    document = load_json(registry_path)
    changed = 0
    for artifact in document.get("artifacts", []):
        entry = (artifact.get("backends") or {}).get("gpu")
        if not isinstance(entry, dict) or entry.get("status") != "certified_source":
            continue
        entry["toolchain"] = JIT_TOOLCHAIN_CONSTRAINT
        jit = (entry.get("deliveries") or {}).get("jit")
        if isinstance(jit, dict):
            jit["toolchain"] = JIT_TOOLCHAIN_CONSTRAINT
        changed += 1
    if changed == 0:
        raise GenerationError("no certified-source GPU rows were found")
    write_document(registry_path, document, schema, REGISTRY_DOMAIN_TAG)
    PACKAGED_OUTPUT.write_bytes(registry_path.read_bytes())


def prebuilt_binding_problems(registry_path: Path) -> list[str]:
    """Compare delivery refinements with the shipped fatbin artifacts.

    Records without ``deliveries`` are exactly the JIT-era shape and
    are fully covered by ``kernel_binding_problems``; this helper only
    runs when a record refines per delivery, and then requires the
    shipped prebuilt artifacts to back every claim.
    """
    from toktier.kernels.prebuilt import (
        fatbin_digest,
        fatbin_path,
        load_manifest,
    )

    document = load_json(registry_path)
    host_problem: str | None
    try:
        host_bindings = native_host_bindings()
    except GenerationError as error:
        host_bindings = {}
        host_problem = str(error)
    else:
        host_problem = None
    refined: list[tuple[str, dict[str, Any]]] = []
    for artifact in document.get("artifacts", []):
        family = artifact.get("family", "<unknown>")
        entry = (artifact.get("backends") or {}).get("gpu")
        if isinstance(entry, dict) and isinstance(entry.get("deliveries"), dict):
            refined.append((family, entry))
    if not refined:
        return []

    packaged = fatbin_path()
    if not packaged.is_file():
        return [
            "the registry records a prebuilt delivery but no fatbin is "
            f"packaged ({packaged})"
        ]
    shipped = bare_sha256(fatbin_digest(packaged.read_bytes()))
    try:
        manifest = load_manifest()
    except (OSError, ValueError) as exc:
        return [
            "the registry records a prebuilt delivery but the build "
            f"manifest is unreadable: {exc}"
        ]
    manifest_architectures = {
        str(arch): bare_sha256(str(info["digest"]))
        for arch, info in manifest["architectures"].items()
    }
    embedded = {arch for arch in manifest_architectures if arch.startswith("sm_")}

    problems: list[str] = []
    if host_problem is not None:
        problems.append(host_problem)
    jit_view_keys = (
        "status",
        "source_digest",
        "build_flags",
        "toolchain",
        "class_table_digest",
        "devices",
    )
    for family, entry in refined:
        deliveries = entry["deliveries"]
        jit = deliveries.get("jit")
        if not isinstance(jit, dict):
            problems.append(
                f"{family}/gpu: deliveries carry no jit sub-entry, so the "
                "top-level view restates nothing"
            )
        else:
            for key in jit_view_keys:
                if jit.get(key) != entry.get(key):
                    problems.append(
                        f"{family}/gpu: deliveries.jit.{key} does not "
                        "restate the top-level view"
                    )
        prebuilt = deliveries.get("prebuilt")
        if not isinstance(prebuilt, dict):
            continue
        if prebuilt.get("binary_digest") != shipped:
            problems.append(
                f"{family}/gpu: deliveries.prebuilt.binary_digest does "
                f"not match the packaged fatbin ({shipped})"
            )
        for key, expected in host_bindings.items():
            if prebuilt.get(key) != expected:
                problems.append(
                    f"{family}/gpu: deliveries.prebuilt.{key} does not "
                    "match the loaded native request host"
                )
        if prebuilt.get("host_source_digest_v2") not in {
            None,
            source_digest_v2("native_host"),
        }:
            problems.append(
                f"{family}/gpu: deliveries.prebuilt.host_source_digest_v2 "
                "does not match the normalized native request host"
            )
        recorded = {
            str(key): str(value)
            for key, value in (prebuilt.get("architecture_digests") or {}).items()
        }
        if recorded != manifest_architectures:
            problems.append(
                f"{family}/gpu: deliveries.prebuilt.architecture_digests "
                "do not match the shipped build manifest"
            )
        listed = set(prebuilt.get("devices") or ()) | set(
            prebuilt.get("devices_experimental") or ()
        )
        if listed != embedded:
            problems.append(
                f"{family}/gpu: deliveries.prebuilt lists devices "
                f"{sorted(listed)}, but the fatbin embeds images for "
                f"{sorted(embedded)}"
            )
    return problems


def prebuilt_hardware_evidence_problems(registry_path: Path) -> list[str]:
    """Bind every certified prebuilt architecture to a current hardware run."""
    from toktier.kernels.prebuilt import fatbin_digest, fatbin_path, load_manifest

    document = load_json(registry_path)
    artifact_rows: dict[str, dict[str, Any]] = {}
    certified_devices: set[str] = set()
    for artifact in document.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        entry = (artifact.get("backends") or {}).get("gpu")
        prebuilt = (
            (entry.get("deliveries") or {}).get("prebuilt")
            if isinstance(entry, dict)
            else None
        )
        if isinstance(prebuilt, dict) and prebuilt.get("status") == "certified":
            family = str(artifact.get("family"))
            artifact_rows[family] = artifact
            certified_devices.update(
                str(value) for value in prebuilt.get("devices") or ()
            )
    if not artifact_rows:
        return []

    try:
        manifest = load_manifest()
        shipped_digest = bare_sha256(fatbin_digest(fatbin_path().read_bytes()))
        architecture_digests = {
            str(architecture): bare_sha256(str(facts["digest"]))
            for architecture, facts in manifest["architectures"].items()
        }
    except (OSError, KeyError, TypeError, ValueError) as error:
        return [f"prebuilt hardware evidence cannot read shipped identities: {error}"]

    problems: list[str] = []
    try:
        host_bindings = native_host_bindings()
    except GenerationError as error:
        return [str(error)]
    unknown = certified_devices - set(PREBUILT_HARDWARE_READINGS)
    if unknown:
        problems.append(
            "certified prebuilt architectures have no configured hardware reading: "
            + ", ".join(sorted(unknown))
        )
    for architecture in sorted(certified_devices & set(PREBUILT_HARDWARE_READINGS)):
        path = PREBUILT_HARDWARE_READINGS[architecture]
        if not path.is_file():
            problems.append(
                f"{architecture}: hardware parity reading is missing ({path})"
            )
            continue
        try:
            reading = load_json(path)
        except (OSError, ValueError) as error:
            problems.append(
                f"{architecture}: hardware parity reading is unreadable: {error}"
            )
            continue
        rows = reading.get("rows")
        by_family = {
            str(row.get("family")): row for row in rows or () if isinstance(row, dict)
        }
        expected_documents = sum(
            int(row.get("documents", 0)) for row in by_family.values()
        )
        expected_characters = sum(
            int(row.get("characters", 0)) for row in by_family.values()
        )
        recorded_host = reading.get("native_host_build_facts")
        expected_host = dict(host_bindings)
        if isinstance(recorded_host, dict) and "host_source_digest_v2" in recorded_host:
            expected_host["host_source_digest_v2"] = source_digest_v2("native_host")
        if (
            reading.get("schema") != "toktier.gpu.native_frontend_parity.v1"
            or reading.get("architecture") != architecture
            # What the reading covers has to be in the reading. Both
            # scales are certified and both report zero mismatches, so
            # the only thing that distinguished them was an unlabelled
            # document count -- which says nothing on its own about the
            # protocol the campaign followed.
            or reading.get("scale") not in {"full", "spot"}
            or reading.get("fatbin_digest") != shipped_digest
            or reading.get("architecture_digest")
            != architecture_digests.get(architecture)
            or recorded_host != expected_host
            or reading.get("families") != len(artifact_rows)
            or int(reading.get("documents", -1)) != expected_documents
            or int(reading.get("characters", -1)) != expected_characters
            or int(reading.get("mismatches", -1)) != 0
            or reading.get("all_ids_equal_hf") is not True
            or reading.get("one_python_to_rust_call_per_batch") is not True
            or reading.get("gil_released") is not True
            or set(by_family) != set(artifact_rows)
        ):
            problems.append(f"{architecture}: hardware parity reading summary drifted")
            continue
        for family, artifact in artifact_rows.items():
            row = by_family[family]
            if (
                row.get("artifact_sha256") != artifact.get("artifact_sha256")
                or int(row.get("documents", 0)) < 1
                or int(row.get("mismatches", -1)) != 0
                or row.get("all_ids_equal_hf") is not True
                or row.get("one_python_to_rust_call_per_batch") is not True
                or row.get("gil_released") is not True
            ):
                problems.append(
                    f"{architecture}/{family}: hardware parity row did not pass"
                )
    return problems


def sync_prebuilt_bindings(registry_path: Path) -> None:
    """Bind the shipped fatbin only after both judged architectures pass."""
    from toktier.kernels.prebuilt import fatbin_digest, fatbin_path, load_manifest

    if registry_path.resolve() != DEFAULT_OUTPUT.resolve():
        raise GenerationError("--sync-prebuilt is only valid for the shipped registry")
    document = load_json(registry_path)
    # Validate the readings against the current registry roster and current
    # binary before changing a single certification field.
    problems = prebuilt_hardware_evidence_problems(registry_path)
    if problems:
        raise GenerationError(
            "prebuilt hardware evidence failed:\n  " + "\n  ".join(problems)
        )
    manifest = load_manifest()
    host_bindings = native_host_bindings(include_v2=True)
    kernel_source, class_table = kernel_bindings()
    if class_table is None:
        raise GenerationError(
            "cannot sync prebuilt bindings while a class-table digest is absent"
        )
    shipped = bare_sha256(fatbin_digest(fatbin_path().read_bytes()))
    architecture_digests = {
        str(architecture): bare_sha256(str(facts["digest"]))
        for architecture, facts in manifest["architectures"].items()
    }
    embedded = sorted(
        architecture
        for architecture in architecture_digests
        if architecture.startswith("sm_")
    )
    certified = sorted(PREBUILT_HARDWARE_READINGS)
    experimental = sorted(set(embedded) - set(certified))
    changed = 0
    for artifact in document.get("artifacts", []):
        entry = (artifact.get("backends") or {}).get("gpu")
        if not isinstance(entry, dict):
            continue
        deliveries = entry.get("deliveries")
        if not isinstance(deliveries, dict):
            continue
        prebuilt = deliveries.get("prebuilt")
        if not isinstance(prebuilt, dict):
            continue
        # The hardware readings above exercised this exact fatbin, which was
        # built from the current source and generated class tables. Keep the
        # top-level compatibility view and its JIT restatement on the same
        # source identity as the binary being admitted.
        entry["source_digest"] = kernel_source
        entry["class_table_digest"] = class_table
        jit = deliveries.get("jit")
        if not isinstance(jit, dict):
            raise GenerationError(
                f"{artifact.get('family', '<unknown>')}: no JIT delivery view"
            )
        jit["source_digest"] = kernel_source
        jit["class_table_digest"] = class_table
        prebuilt.update(
            {
                "status": "certified",
                "binary_digest": shipped,
                "toolchain": str(manifest["toolchain"]),
                "devices": certified,
                "devices_experimental": experimental,
                "architecture_digests": architecture_digests,
                **host_bindings,
            }
        )
        changed += 1
    if changed == 0:
        raise GenerationError("no prebuilt GPU delivery rows were found")
    write_document(registry_path, document, load_json(SCHEMA_PATH), REGISTRY_DOMAIN_TAG)
    PACKAGED_OUTPUT.write_bytes(registry_path.read_bytes())


def packaged_copy_problems(registry_path: Path) -> list[str]:
    """The installed copy must be byte-identical to the repository copy.

    The runtime reports certification statuses out of the packaged copy
    (``toktier.routing.tables``); two copies that drift would let the
    repository checks pass while an installed wheel reports something
    else. Byte identity keeps the packaged file covered by exactly the
    checks that cover the repository file.
    """
    if registry_path.resolve() != DEFAULT_OUTPUT.resolve():
        # A custom --out is not the shipped registry; the packaged-copy
        # invariant binds the default output only.
        return []
    if not PACKAGED_OUTPUT.is_file():
        return [
            f"the packaged registry copy is missing ({PACKAGED_OUTPUT}); "
            f"copy {DEFAULT_OUTPUT} there byte-identically"
        ]
    if PACKAGED_OUTPUT.read_bytes() != registry_path.read_bytes():
        return [
            f"the packaged registry copy ({PACKAGED_OUTPUT}) differs from "
            f"{registry_path}; the two must be byte-identical"
        ]
    return []


def fast_cpu_binding_problems(registry_path: Path) -> list[str]:
    """Require the fast CPU rows to equal the checked shipped binding."""
    if registry_path.resolve() != DEFAULT_OUTPUT.resolve():
        return []
    try:
        document = load_json(registry_path)
        binding = load_json(REPOSITORY_ROOT / "tools" / "fast_cpu_binding.json")
        expected = augmented_document(document, binding)
    except (GenerationError, OSError, KeyError, TypeError, ValueError) as error:
        return [f"fast CPU binding cannot be verified: {error}"]
    if document != expected:
        return [
            "fast CPU registry entries differ from the checked binding; "
            "run tools/update_fast_cpu_registry.py"
        ]
    return []


def rust_api_binding_problems(registry_path: Path) -> list[str]:
    """Require the public Rust host row to equal its checked evidence binding."""
    if registry_path.resolve() != DEFAULT_OUTPUT.resolve():
        return []
    try:
        from update_rust_api_registry import augmented_document

        document = load_json(registry_path)
        binding = load_json(REPOSITORY_ROOT / "tools" / "rust_api_binding.json")
        expected = augmented_document(document, binding)
    except (GenerationError, OSError, KeyError, TypeError, ValueError) as error:
        return [f"Rust API binding cannot be verified: {error}"]
    if document != expected:
        return [
            "Rust API runtime-build row differs from the checked binding; "
            "run tools/update_rust_api_registry.py"
        ]
    return []


def release_problems(registry_path: Path) -> list[str]:
    """Refusals that apply to a release, on top of ``--check``.

    A release must not ship placeholder identities, unrecorded table
    digests, or routing data that names tables the package does not
    carry: every one of those produces documents that cannot certify
    the implementation they select.
    """
    problems: list[str] = []
    document = load_json(registry_path)
    digest_fields = ("binary_digest", "source_digest", "class_table_digest")
    for artifact in document.get("artifacts", []):
        family = artifact.get("family", "<unknown>")
        for backend_id, entry in (artifact.get("backends") or {}).items():
            views = [(f"{family}/{backend_id}", entry)]
            for name, sub in (entry.get("deliveries") or {}).items():
                views.append((f"{family}/{backend_id}/deliveries.{name}", sub))
            for label, view in views:
                for field in digest_fields:
                    if view.get(field) == PLACEHOLDER_SHA256:
                        problems.append(f"{label}: {field} is the placeholder digest")
    families_document = load_json(DEFAULT_FAMILY_TABLE_PATH)
    packaged_dir = DEFAULT_FAMILY_TABLE_PATH.parent
    for table_id, spec in families_document["class_tables"].items():
        checks = [("sha256", spec.get("file"))]
        if spec.get("meta_file") or spec.get("meta_sha256"):
            checks.append(("meta_sha256", spec.get("meta_file")))
        for digest_key, file_name in checks:
            recorded = spec.get(digest_key)
            if not recorded:
                problems.append(
                    f"class table {table_id}: no {digest_key} recorded in "
                    "the routing data"
                )
                continue
            if not file_name:
                problems.append(
                    f"class table {table_id}: {digest_key} is recorded but "
                    "no file is named"
                )
                continue
            packaged = packaged_dir / str(file_name)
            if not packaged.is_file():
                problems.append(
                    f"class table {table_id}: {file_name} is not packaged "
                    f"under {packaged_dir}"
                )
                continue
            observed = f"sha256:{sha256_of_file(packaged)}"
            if observed != recorded:
                problems.append(
                    f"class table {table_id}: packaged {file_name} has "
                    f"{observed}, the routing data records {recorded}"
                )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the shipped support registry table."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the serialisation, the schema and the root digest.",
    )
    parser.add_argument(
        "--release-check",
        action="store_true",
        help=(
            "Release gate: --check plus a refusal of placeholder "
            "identities, unrecorded table digests, and missing packaged "
            "tables."
        ),
    )
    parser.add_argument(
        "--sync-jit-toolchain",
        action="store_true",
        help=(
            "Rewrite JIT rows with the exact NVCC/runtime/torch constraint "
            "and refresh the registry root."
        ),
    )
    parser.add_argument(
        "--sync-prebuilt",
        action="store_true",
        help=(
            "Bind the shipped fatbin and per-architecture images after the "
            "configured sm_89 and sm_120 hardware readings pass."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if vendored_source_archive(REPOSITORY_ROOT):
        # This tool compares the repository's generated registry with the
        # repository's own sources and its built extension. A published
        # source archive is neither: it carries no extension, and the one
        # an installed wheel provides belongs to whatever else is on this
        # machine. Borrowing it would answer a question about that wheel
        # while looking like an answer about this tree.
        print(
            "declined: this check reads the repository's own source tree "
            "and its built extension, and this is the published source "
            "archive. Nothing was checked. The archive's registry copy is "
            "verified by tools/validate_registry.py, which does run here; "
            "this check runs from a repository checkout.",
            file=sys.stderr,
        )
        return DECLINED
    if arguments.sync_jit_toolchain:
        sync_jit_toolchain_constraint(arguments.out)
    if arguments.sync_prebuilt:
        sync_prebuilt_bindings(arguments.out)
    if not (
        arguments.check
        or arguments.release_check
        or arguments.sync_jit_toolchain
        or arguments.sync_prebuilt
    ):
        raise GenerationError(
            "this tool verifies the shipped registry (--check or "
            "--release-check); the registry itself is generated by the "
            "maintainers from the recorded campaign readings"
        )
    schema = load_json(SCHEMA_PATH)
    # Source-identity coverage: the identity path lists are transcribed
    # in two build scripts and one tools-side table, and routing-core is
    # covered by named files rather than a whole tree. These refusals
    # catch the transcriptions drifting apart or a new routing-core file
    # staying outside the fast_cpu digest.
    problems = source_coverage_problems()
    problems += verify_file(arguments.out, schema, REGISTRY_DOMAIN_TAG)
    if not problems:
        problems = kernel_binding_problems(arguments.out)
        problems += prebuilt_binding_problems(arguments.out)
        problems += prebuilt_hardware_evidence_problems(arguments.out)
        problems += fast_cpu_binding_problems(arguments.out)
        problems += rust_api_binding_problems(arguments.out)
        problems += packaged_copy_problems(arguments.out)
    if arguments.release_check:
        problems = list(problems) + release_problems(arguments.out)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"{arguments.out}: check passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GenerationError as error:  # pragma: no cover - command line surface
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
