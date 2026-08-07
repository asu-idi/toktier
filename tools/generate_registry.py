# Verify tables/support_registry.json, the machine-generated certification registry.
"""Check the shipped certification registry.

The registry states exactly what has been certified and nothing more, and it
is produced by maintainer tooling from the recorded campaign readings
(``docs/contracts/registry.md`` Section 7); those generation inputs are not
part of this repository. What ships here is the verification surface: the
serialisation is the deterministic one, the document validates against its
schema, and the recorded root digest matches the document. A hand edit fails
``--check``.

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
and the ``prebuilt`` sub-entry must bind the digest of the packaged
fatbin and the per-architecture image digests of its build manifest.
Both helpers are torch-free, so this check still needs no GPU stack.

Usage::

    python tools/generate_registry.py --check
    python tools/generate_registry.py --release-check
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from registry_common import (
    PLACEHOLDER_SHA256,
    REGISTRY_DOMAIN_TAG,
    GenerationError,
    load_json,
    sha256_of_file,
    verify_file,
)
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

ORACLE_PACKAGE = "tokenizers"


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
    refined: list[tuple[str, dict[str, Any]]] = []
    for artifact in document.get("artifacts", []):
        family = artifact.get("family", "<unknown>")
        entry = (artifact.get("backends") or {}).get("gpu")
        if isinstance(entry, dict) and isinstance(
            entry.get("deliveries"), dict
        ):
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
    embedded = {
        arch for arch in manifest_architectures if arch.startswith("sm_")
    }

    problems: list[str] = []
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
        recorded = {
            str(key): str(value)
            for key, value in (
                prebuilt.get("architecture_digests") or {}
            ).items()
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
                        problems.append(
                            f"{label}: {field} is the placeholder digest"
                        )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not (arguments.check or arguments.release_check):
        raise GenerationError(
            "this tool verifies the shipped registry (--check or "
            "--release-check); the registry itself is generated by the "
            "maintainers from the recorded campaign readings"
        )
    schema = load_json(SCHEMA_PATH)
    problems = verify_file(arguments.out, schema, REGISTRY_DOMAIN_TAG)
    if not problems:
        problems = kernel_binding_problems(arguments.out)
        problems += prebuilt_binding_problems(arguments.out)
        problems += fast_cpu_binding_problems(arguments.out)
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
