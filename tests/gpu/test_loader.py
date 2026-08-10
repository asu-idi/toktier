"""Host tests for the single-loader, single-flag-set rule.

Contract reference: ``docs/contracts/registry.md`` Section 3.2. A
``certified_source`` certificate covers exactly one kernel build
configuration per process. These tests check both halves of that: the
package contains one loading call site, and the loader refuses a second,
divergent build at runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from toktier.engine.gpu.loader import (
    DEFAULT_BUILD_FLAGS,
    EXTENSION_NAME,
    BuildFlags,
    KernelLoader,
    ToolchainFacts,
)
from toktier.errors import KernelIncompatible

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "toktier"


@pytest.fixture(autouse=True)
def _clean_loader() -> object:
    KernelLoader._reset_for_tests()
    yield
    KernelLoader._reset_for_tests()


# -- static: one call site ---------------------------------------------


def _python_sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_only_one_extension_load_call_site() -> None:
    """``cpp_extension.load`` is called from exactly one module.

    Two call sites with different flags in one process would produce two
    builds of the same kernel, which is precisely the condition the
    certificate cannot survive. The prototype this was ported from
    had four; the released package has one.
    """
    call_sites = [
        path
        for path in _python_sources()
        if "cpp_extension" in path.read_text(encoding="utf-8")
    ]
    assert [path.name for path in call_sites] == ["loader.py"]


def test_extension_name_is_used_once() -> None:
    """The extension name is defined once and never spelled again."""
    literals: list[Path] = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        if f'"{EXTENSION_NAME}"' in text and path.name != "loader.py":
            literals.append(path)
    assert not literals, literals


# -- flags --------------------------------------------------------------


def test_flag_digest_is_stable_and_order_sensitive() -> None:
    assert BuildFlags().digest() == BuildFlags().digest()
    assert (
        BuildFlags(cuda_cflags=("-O3",)).digest()
        != BuildFlags(cuda_cflags=("-O2",)).digest()
    )
    assert (
        BuildFlags(cuda_cflags=("-O3", "-lineinfo")).digest()
        != BuildFlags(cuda_cflags=("-lineinfo", "-O3")).digest()
    )


def test_defines_reach_the_compiler_arguments() -> None:
    flags = BuildFlags(defines=(("TOKTIER_TPB", "128"),))
    assert "-DTOKTIER_TPB=128" in flags.as_cuda_flags()
    assert "-DTOKTIER_TPB=128" in flags.as_host_flags()
    assert flags.digest() != DEFAULT_BUILD_FLAGS.digest()


def test_default_flags_match_the_certified_configuration() -> None:
    """The default build is the one the judgement runs used."""
    assert DEFAULT_BUILD_FLAGS.cuda_cflags == ("-O3",)
    assert DEFAULT_BUILD_FLAGS.cflags == ()
    assert DEFAULT_BUILD_FLAGS.defines == ()


# -- runtime rule -------------------------------------------------------


def test_second_divergent_flag_set_is_refused_and_voids_the_certificate() -> None:
    """A divergent request raises and marks the process certificate void.

    Simulated without a GPU by seeding the loader with an already-loaded
    build: the divergence check runs before anything is compiled, which
    is the point -- the refusal does not depend on being able to build.
    """
    import types

    KernelLoader._state.module = types.ModuleType("fake_extension")
    KernelLoader._state.flags = DEFAULT_BUILD_FLAGS

    assert KernelLoader.certificate_void() is False
    same = KernelLoader.get(flags=DEFAULT_BUILD_FLAGS)
    assert same is KernelLoader._state.module

    with pytest.raises(KernelIncompatible) as caught:
        KernelLoader.get(flags=BuildFlags(cuda_cflags=("-O2",)))
    assert caught.value.code == "KERNEL_INCOMPATIBLE"
    assert caught.value.details["reason_code"] == "R_KERNEL_DIGEST_MISMATCH"
    assert KernelLoader.certificate_void() is True
    assert KernelLoader.void_reason()

    # Once void it stays void, even for the flag set that did load.
    KernelLoader.get(flags=DEFAULT_BUILD_FLAGS)
    assert KernelLoader.certificate_void() is True


def test_binding_set_is_computable_without_a_gpu() -> None:
    """The certificate inputs a verifier needs are host-computable."""
    binding = KernelLoader.binding_set(class_table_digest="sha256:" + "0" * 64)
    assert binding["delivery"] == "jit"
    # Bound fields are spelled the way the registry schema spells them:
    # bare 64-hex digests and one flat flag list.
    assert len(binding["source_digest"]) == 64
    assert int(binding["source_digest"], 16) >= 0
    assert binding["build_flags"] == ["-O3"]
    assert binding["build_flags_digest"].startswith("sha256:")
    assert binding["class_table_digest"] == "0" * 64
    assert binding["family_table_digest"] is None  # none was supplied
    assert binding["certificate_void"] is False
    assert "toolchain_facts" not in binding  # nothing has been built yet


def test_binding_flags_encode_as_the_judged_flat_list() -> None:
    """The canonical flag encoding matches what the judged build recorded.

    The registry generator records the judged build as the flat list
    ``["-O3"]`` (the ``extra_cuda_cflags`` of the judgement loader), so
    the loader's encoding of the default flag set must produce exactly
    that; host flags and definitions extend the list without changing
    the certified default.
    """
    assert DEFAULT_BUILD_FLAGS.as_binding_flags() == ("-O3",)
    rich = BuildFlags(
        cuda_cflags=("-O3",),
        cflags=("-fno-fast-math",),
        defines=(("TOKTIER_TPB", "128"),),
    )
    assert rich.as_binding_flags() == (
        "-O3",
        "-DTOKTIER_TPB=128",
        "host:-fno-fast-math",
    )


def test_producer_bindings_round_trip_through_the_registry_shape() -> None:
    """Producer and consumer share one binding representation.

    A binding set the loader produces, serialized into the registry
    entry shape and parsed back by the shared reader, must come back
    equal -- this is exactly the drift that once hid behind a prefixed
    digest name and a structured flag mapping.
    """
    from toktier.kernels.bindings import CertifiedSourceBindings

    produced = KernelLoader.certified_source_bindings(
        class_table_digest="sha256:" + "1" * 64
    )
    parsed = CertifiedSourceBindings.from_mapping(produced.as_mapping())
    assert parsed == produced


def test_build_directory_separates_flag_sets(tmp_path: Path) -> None:
    from toktier.engine.gpu.loader import _resolve_build_dir

    first = _resolve_build_dir(tmp_path, None, DEFAULT_BUILD_FLAGS)
    second = _resolve_build_dir(tmp_path, None, BuildFlags(cuda_cflags=("-O2",)))
    assert first != second
    assert first.parent == tmp_path / "kernels"


def test_build_directory_separates_actual_compiler_identities(
    tmp_path: Path,
) -> None:
    from toktier.engine.gpu.loader import _resolve_build_dir

    def facts(release: str, build: str) -> ToolchainFacts:
        return ToolchainFacts(
            torch_version="2.13.0+cu130",
            cuda_version="13.0",
            nvcc_path="/usr/local/cuda/bin/nvcc",
            nvcc_resolved_path=f"/usr/local/cuda-{release}/bin/nvcc",
            nvcc_release=release,
            nvcc_build=build,
            nvcc_error=None,
            jit_toolchain_satisfied=release == "13.0",
            device_name="test",
            device_capability="sm_120",
            driver_version="595.84",
        )

    judged = _resolve_build_dir(
        tmp_path,
        None,
        DEFAULT_BUILD_FLAGS,
        toolchain=facts("13.0", "V13.0.88"),
    )
    drifted = _resolve_build_dir(
        tmp_path,
        None,
        DEFAULT_BUILD_FLAGS,
        toolchain=facts("13.2", "V13.2.86"),
    )
    assert judged != drifted
    assert judged.parent == drifted.parent == tmp_path / "kernels"


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Identify docstring constants, which are prose rather than code."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def test_no_module_hardcodes_a_build_directory() -> None:
    """Build products go under the cache directory, never a fixed path.

    Documentation may name the environment variable the loader
    deliberately does not use, and may quote a machine path in an
    explanation; executable code may do neither. The check therefore
    skips docstrings and looks at the remaining string constants.
    """
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        prose = _docstring_node_ids(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in prose
            ):
                assert "TORCH_EXTENSIONS_DIR" not in node.value, path
                assert not node.value.startswith("/mnt/"), (path, node.value)
                assert not node.value.startswith("/scratch/"), (path, node.value)


def test_package_never_reads_the_environment() -> None:
    """No module in the package touches ``os.environ`` or ``getenv``.

    Every former environment flag became an explicit argument. The five
    long-term variables the contract keeps are read once, by the
    configuration object, which lives outside this package.
    """
    offenders: list[tuple[str, str]] = []
    for path in _python_sources():
        if path.parts[-2:] == ("toktier", "config.py"):
            continue  # the configuration object is where env is read
        if path.parts[-2:] == ("artifacts", "sources.py"):
            # The hub source honors HF_HUB_OFFLINE exactly once, at
            # construction, as the artifacts contract documents.
            continue
        if path.parts[-3:] == ("engine", "gpu", "toolchain.py"):
            # JIT certification must identify the compiler the build
            # system selects. CUDA_HOME/CUDA_PATH are compiler-selection
            # inputs already honored by that system; the shared probe
            # observes them, never treats them as TokTier configuration.
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {
                "environ",
                "getenv",
            }:
                offenders.append((str(path), node.attr))
            if isinstance(node, ast.Name) and node.id in {"environ", "getenv"}:
                offenders.append((str(path), node.id))
    assert not offenders, offenders
