"""Prebuilt (fatbin) delivery: identities, loader rules, launcher basics.

Host tests need neither a GPU nor torch: they cover the digest domains,
the shipped-artifact consistency (fatbin vs manifest), and the loader's
delivery selection rules (explicit requests never substitute; ``auto``
falls back with the reason recorded). The GPU test at the end is marked
``gpu`` like the rest of the suite.

Two of them do need one more thing: the compiled ``toktier._native``
extension, because they assert on identity facts the Cargo build script
embeds into it. ``pytest.ini`` runs the suite against ``src`` rather
than an install, so a source tree that has never been built (an
unpacked archive, for instance) has no extension there. An installed
wheel beside such a tree does carry one, and it is a legitimate subject
for these assertions, so the two tests look for it the same way
``tools/generate_registry.py`` does before deciding they have no
premise. Adopting an installed extension changes only where the
identity is read: it still has to equal, exactly, the digest the
current source set hashes to. With neither extension available the two
skip with that reason rather than failing, which is the honest report:
the premise for the assertion is absent, not violated.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from toktier.engine.gpu import prebuilt as prebuilt_pkg
from toktier.engine.gpu.loader import BuildFlags, KernelLoader
from toktier.engine.gpu.native import native_host_build_facts
from toktier.engine.gpu.prebuilt import (
    PrebuiltUnavailable,
    shipped_fatbin_digest,
)
from toktier.errors import KernelIncompatible
from toktier.kernels import kernel_source_digest
from toktier.kernels.prebuilt import (
    FATBIN_NAME,
    PREBUILT_DIR,
    cubin_digest,
    fatbin_digest,
    load_manifest,
    prebuilt_source_digest,
)


@pytest.fixture(autouse=True)
def _clean_loader() -> Any:
    KernelLoader._reset_for_tests()
    yield
    KernelLoader._reset_for_tests()


# -- digest domains ----------------------------------------------------


def test_prebuilt_source_digest_is_its_own_domain() -> None:
    """The prebuilt lineage digest never collides with the JIT digest.

    Both cover ``pretok_kernel.cu``; the prebuilt one adds the wrapper
    and a distinct domain tag, so equal values would mean a broken
    domain separation.
    """
    prebuilt = prebuilt_source_digest()
    assert prebuilt.startswith("sha256:")
    assert prebuilt != kernel_source_digest()
    assert prebuilt == prebuilt_source_digest()  # deterministic


def test_cubin_digest_binds_the_architecture() -> None:
    """The same bytes under two architecture labels differ by digest."""
    payload = b"\x00\x01\x02cubin"
    assert cubin_digest("sm_89", payload) != cubin_digest("sm_120", payload)
    assert cubin_digest("sm_89", payload) != fatbin_digest(payload)


# -- shipped artifact consistency --------------------------------------

_FATBIN_SHIPPED = (PREBUILT_DIR / FATBIN_NAME).is_file()

def _readable_native_extension() -> bool:
    """Whether identity facts can be read from a compiled extension.

    This source tree's own build answers first. When it has none, the
    repository's registry generator already knows how to bind an
    installed wheel's extension instead; reusing that helper keeps one
    lookup rule (skip this tree, accept only real extension suffixes,
    load through ``ExtensionFileLoader``) rather than a second copy that
    could drift from it. The helper only changes where the facts come
    from -- the assertions below still require exact equality with the
    current source set.
    """
    if native_host_build_facts().source_digest:
        return True
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    try:
        from generate_registry import _adopt_installed_native_extension
    except ImportError:  # pragma: no cover - tooling absent from a slice
        return False
    if _adopt_installed_native_extension() is None:
        return False
    return bool(native_host_build_facts().source_digest)


#: Whether identity facts are readable at all: from this source tree's
#: compiled ``toktier._native``, or failing that from an installed
#: wheel's. The build script embeds those facts, so tests that read them
#: have no premise without one of the two.
_NATIVE_BUILT = _readable_native_extension()

_NEEDS_NATIVE = pytest.mark.skipif(
    not _NATIVE_BUILT,
    reason=(
        "no compiled toktier._native in this source tree and none found "
        "in an installed toktier on sys.path; build it into src/toktier "
        "(maturin develop, or maturin build --locked and place the "
        "extension there), or run against an environment where the "
        "matching wheel is installed, and re-run"
    ),
)


def test_shipped_digest_reports_presence_honestly() -> None:
    digest = shipped_fatbin_digest()
    if _FATBIN_SHIPPED:
        assert digest is not None and digest.startswith("sha256:")
    else:
        assert digest is None


@_NEEDS_NATIVE
def test_native_host_source_identity_matches_the_loaded_extension() -> None:
    """The verifier and Cargo build script hash exactly the same source set."""
    from toktier import _native

    root = Path(__file__).resolve().parents[2]
    observed = subprocess.run(
        [sys.executable, str(root / "tools/native_host_source_identity.py")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    facts = _native.native_host_build_facts()
    assert facts["source_digest"] == observed
    assert facts["build_flags"]
    assert facts["toolchain"]


@pytest.mark.skipif(
    not _FATBIN_SHIPPED, reason="no fatbin in this source tree"
)
def test_shipped_fatbin_matches_its_manifest() -> None:
    """Package data honesty: bytes, manifest and source lineage agree."""
    manifest = load_manifest()
    data = (PREBUILT_DIR / FATBIN_NAME).read_bytes()
    assert fatbin_digest(data) == manifest["fatbin"]["digest"]
    assert manifest["sources"]["prebuilt_source_digest"] == (
        prebuilt_source_digest()
    )
    assert manifest["sources"]["jit_kernel_source_digest"] == (
        kernel_source_digest()
    )
    # Every certified-relevant architecture ships as a real device image.
    kinds = {
        arch: entry["kind"] for arch, entry in manifest["architectures"].items()
    }
    for arch in ("sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100",
                 "sm_120"):
        assert kinds.get(arch) == "cubin", arch
    assert kinds.get(manifest["ptx_fallback"]) == "ptx"
    # The launcher resolves kernels through the manifest symbol map.
    for logical in (
        "k_utf8_decode",
        "k_classify_rs0",
        "k_classify_rs1",
        "k_classify_rs2",
        "k_bpe_thread_cap32",
        "k_o2k_rules",
        "tk_select_scatter",
        "tk_carrier_scatter",
        "tk_carrier_gather",
        "tk_ds_constants_dump",
    ):
        assert logical in manifest["kernels"], logical


# -- loader delivery rules ---------------------------------------------


def test_unknown_delivery_is_refused() -> None:
    with pytest.raises(KernelIncompatible):
        KernelLoader.get(delivery="wheel")


def test_explicit_prebuilt_never_substitutes(monkeypatch: Any) -> None:
    """delivery='prebuilt' raises when unavailable instead of JITing."""

    def refuse(device: str | None = None) -> Any:
        raise PrebuiltUnavailable("R_DRIVER_TOO_OLD", "driver too old")

    monkeypatch.setattr(prebuilt_pkg, "load_prebuilt_extension", refuse)
    monkeypatch.setattr(
        "toktier.engine.gpu.loader._import_torch", lambda: object()
    )
    with pytest.raises(KernelIncompatible) as excinfo:
        KernelLoader.get(delivery="prebuilt")
    assert "driver too old" in str(excinfo.value)
    assert not KernelLoader.is_loaded()


def test_auto_falls_back_to_jit_with_reason(monkeypatch: Any) -> None:
    """auto -> JIT on refusal, and the reason is recorded, not dropped."""

    def refuse(device: str | None = None) -> Any:
        raise PrebuiltUnavailable("R_DRIVER_TOO_OLD", "driver too old")

    module = object()
    monkeypatch.setattr(prebuilt_pkg, "load_prebuilt_extension", refuse)
    monkeypatch.setattr(
        "toktier.engine.gpu.loader._import_torch", lambda: object()
    )
    monkeypatch.setattr(
        "toktier.engine.gpu.loader._compile",
        lambda torch, build_dir, flags: module,
    )
    monkeypatch.setattr(
        "toktier.engine.gpu.loader._toolchain_facts",
        lambda torch, device, **_kwargs: None,
    )
    got = KernelLoader.get(cache_dir=Path("/tmp/toktier-test"), delivery="auto")
    assert got is module
    assert KernelLoader.delivery() == "jit"
    reason = KernelLoader.prebuilt_fallback_reason()
    assert reason is not None and "R_DRIVER_TOO_OLD" in reason
    binding = KernelLoader.binding_set()
    assert binding["delivery"] == "jit"
    assert "R_DRIVER_TOO_OLD" in binding["prebuilt_fallback_reason"]


def test_custom_flags_need_the_jit_delivery(monkeypatch: Any) -> None:
    """The fatbin serves exactly the shipped configuration."""
    monkeypatch.setattr(
        "toktier.engine.gpu.loader._import_torch", lambda: object()
    )
    flags = BuildFlags(cuda_cflags=("-O2",))
    with pytest.raises(KernelIncompatible) as excinfo:
        KernelLoader.get(delivery="prebuilt", flags=flags)
    assert "R_PREBUILT_FLAGS_UNSERVABLE" in str(
        excinfo.value.details.get("reason_code")
    )


def test_divergent_delivery_request_voids_certificate(
    monkeypatch: Any,
) -> None:
    """A second, different explicit delivery is a certificate breach."""
    module = object()

    class FakeLoad:
        extension = module
        fatbin_digest = "sha256:" + "d" * 64
        manifest: ClassVar[dict[str, Any]] = {
            "toolchain": "cuda 13.2",
            "architectures": {},
            "sources": {},
        }
        device_architecture = "sm_120"
        architecture_embedded = True

    monkeypatch.setattr(
        prebuilt_pkg, "load_prebuilt_extension", lambda device=None: FakeLoad()
    )
    monkeypatch.setattr(
        "toktier.engine.gpu.loader._import_torch", lambda: object()
    )
    monkeypatch.setattr(
        "toktier.engine.gpu.loader._toolchain_facts",
        lambda torch, device, **_kwargs: None,
    )
    got = KernelLoader.get(delivery="prebuilt")
    assert got is module
    assert KernelLoader.delivery() == "prebuilt"
    binding = KernelLoader.binding_set()
    assert binding["delivery"] == "prebuilt"
    assert binding["binary_digest"] == "d" * 64
    with pytest.raises(KernelIncompatible):
        KernelLoader.get(delivery="jit")
    assert KernelLoader.certificate_void()
    # auto accepts whatever is loaded; it never breaches.
    KernelLoader._reset_for_tests()
    monkeypatch.setattr(
        prebuilt_pkg, "load_prebuilt_extension", lambda device=None: FakeLoad()
    )
    got = KernelLoader.get(delivery="auto")
    assert KernelLoader.delivery() == "prebuilt"
    assert KernelLoader.get(delivery="auto") is got


# -- launcher geometry helpers -----------------------------------------


@_NEEDS_NATIVE
def test_rust_prebuilt_host_publishes_one_process_delivery() -> None:
    manifest = {
        "toolchain": "cuda 13.2",
        "architectures": {
            "sm_120": {"digest": "sha256:" + "a" * 64},
        },
        "sources": {"prebuilt_source_digest": "sha256:" + "b" * 64},
    }
    digest = "sha256:" + "d" * 64
    KernelLoader.note_native_prebuilt_loaded(
        manifest=manifest,
        fatbin_digest=digest,
        architecture="sm_120",
    )

    assert KernelLoader.is_loaded()
    assert KernelLoader.delivery() == "prebuilt"
    binding = KernelLoader.binding_set()
    assert binding["binary_digest"] == "d" * 64
    assert binding["host_source_digest"]
    assert binding["host_build_flags"]
    assert binding["host_toolchain"]
    assert binding["prebuilt"]["device_architecture"] == "sm_120"
    assert binding["prebuilt"]["architecture_embedded"] is True

    # Re-publishing the same immutable identity is idempotent; a second
    # architecture in the same process invalidates the single-load premise.
    KernelLoader.note_native_prebuilt_loaded(
        manifest=manifest,
        fatbin_digest=digest,
        architecture="sm_120",
    )
    with pytest.raises(KernelIncompatible):
        KernelLoader.note_native_prebuilt_loaded(
            manifest=manifest,
            fatbin_digest=digest,
            architecture="sm_90",
        )
    assert KernelLoader.certificate_void()



def test_launcher_grid_and_scalar_packing() -> None:
    torch = pytest.importorskip("torch")  # noqa: F841 - import gate only
    from toktier.engine.gpu.prebuilt.launcher import _b, _grid, _i, _u

    assert _grid(0) == 0
    assert _grid(1) == 1
    assert _grid(256) == 1
    assert _grid(257) == 2
    assert _grid(65, 64) == 2
    assert _i(3).value == 3
    assert _u(0xFFFFFFFF).value == 0xFFFFFFFF
    assert _b(True).value is True and _b(False).value is False


# -- GPU smoke ---------------------------------------------------------


@pytest.mark.gpu
def test_prebuilt_utf8_roundtrip_on_device() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from toktier.engine.gpu.prebuilt import load_prebuilt_extension

    try:
        load = load_prebuilt_extension()
    except PrebuiltUnavailable as exc:
        pytest.skip(f"prebuilt unavailable: {exc}")
    text = "prebuilt smoke — 你好 🎉 café\r\n"
    data = torch.frombuffer(
        bytearray(text.encode()), dtype=torch.uint8
    ).cuda()
    cp = load.extension.utf8_to_cp(data)
    assert cp.cpu().tolist() == [ord(c) for c in text]
