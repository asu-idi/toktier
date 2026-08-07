"""Host tests over the kernel source and its digest.

These run without a GPU, without torch and without nvcc. They exist
because several release properties of the kernel are decidable by
reading it, and a property that can be checked cheaply on every commit
should not wait for hardware.
"""

from __future__ import annotations

import re

import pytest

from toktier.kernels import (
    KERNEL_DIR,
    KERNEL_SOURCES,
    kernel_source_digest,
    kernel_source_paths,
)

SOURCE = (KERNEL_DIR / "pretok_kernel.cu").read_text(encoding="utf-8")


def strip_comments(source: str) -> str:
    """Remove C and C++ comments, leaving string literals intact.

    Several checks below are about what the kernel *does*, so they have
    to look at code rather than prose: a comment that explains why an
    environment switch is absent must not count as one being present.
    """
    out: list[str] = []
    index, size = 0, len(source)
    while index < size:
        char = source[index]
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            while index < size:
                if source[index] == "\\":
                    out.append(source[index : index + 2])
                    index += 2
                    continue
                out.append(source[index])
                if source[index] == quote:
                    index += 1
                    break
                index += 1
            continue
        if source.startswith("//", index):
            while index < size and source[index] != "\n":
                index += 1
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            index = end + 2 if end >= 0 else size
            continue
        out.append(char)
        index += 1
    return "".join(out)


CODE = strip_comments(SOURCE)


def test_sources_exist() -> None:
    for path in kernel_source_paths():
        assert path.is_file(), path


def test_digest_is_deterministic_and_prefixed() -> None:
    first = kernel_source_digest()
    assert first == kernel_source_digest()
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


def test_digest_changes_with_content(tmp_path: object) -> None:
    """A one-byte edit must move the digest.

    Checked structurally rather than by editing the shipped file: the
    digest folds in each source's length and bytes under a domain tag, so
    two different contents cannot share a preimage by construction. This
    test pins the shape of that construction.
    """
    import hashlib

    expected = hashlib.sha256()
    expected.update(b"toktier.kernel_source.v1\x00")
    for name, path in zip(KERNEL_SOURCES, kernel_source_paths(), strict=True):
        data = path.read_bytes()
        expected.update(name.encode("utf-8"))
        expected.update(b"\x00")
        expected.update(len(data).to_bytes(8, "little"))
        expected.update(data)
    assert kernel_source_digest() == f"sha256:{expected.hexdigest()}"


def test_kernel_reads_no_environment_variables() -> None:
    """No environment switch may reach the kernel.

    A switch that changes how the kernel merges is a switch that can
    change output, and the configuration contract forbids exposing those
    as environment variables. The kernel therefore calls neither
    ``getenv`` nor any relative of it.
    """
    for pattern in ("getenv", "_dupenv_s", "environ"):
        assert pattern not in CODE, pattern


def test_non_lossless_parallel_merge_is_absent() -> None:
    """The non-exact parallel plateau merge is gone from the source.

    It was a prototype-only mode: exactness for it would need an offline
    vocabulary certificate, and known counterexamples exist where
    equal-rank merges interact. Rather than leaving it switched off, the
    released kernel does not contain it.
    """
    for pattern in ("par_merge", "par_ok", "PAR_MERGE"):
        assert pattern not in CODE, pattern
    assert "plateau" not in SOURCE  # not even described as available


def test_non_monotone_merge_guard_is_present() -> None:
    """The guard that keeps batched merging exact must still be there."""
    assert "unsafe_bits" in CODE
    assert "ub[(unsigned)rank >> 5]" in CODE


def test_build_macros_use_the_package_prefix() -> None:
    """Build-time macros carry the package prefix, not a legacy one."""
    macros = set(re.findall(r"^#define\s+([A-Z][A-Z0-9_]*)", CODE, re.M))
    tunables = {name for name in macros if not name.startswith("TOKTIER_")}
    assert not tunables, tunables
    for expected in ("TOKTIER_TPB", "TOKTIER_SHORT_MAX"):
        assert expected in macros


def test_source_is_ascii() -> None:
    """Shipped source is ASCII, so no toolchain has to guess an encoding."""
    offenders = [
        (number, line)
        for number, line in enumerate(SOURCE.splitlines(), start=1)
        if any(ord(char) > 127 for char in line)
    ]
    assert not offenders, offenders[:5]


@pytest.mark.parametrize(
    "entry",
    [
        "pretok_starts",
        "pretok_starts_batched",
        "pretok_starts_ds",
        "pretok_starts_batched_ds",
        "pretok_starts_laguna",
        "pretok_starts_batched_laguna",
        "pretok_starts_o200k",
        "pretok_starts_batched_o200k",
        "pretok_starts_kimi",
        "o200k_win_extents",
        "o200k_win_apply",
        "bpe_encode",
        "encode_fused",
        "encode_fused_ds",
        "encode_fused_laguna",
        "encode_fused_o200k",
        "utf8_to_cp",
        "utf8_to_cp_bo",
        "nfc_qc_scan",
        "ds_constants",
    ],
)
def test_extension_entry_is_registered(entry: str) -> None:
    """Every entry the Python layer calls is registered by the module."""
    assert f'm.def("{entry}"' in CODE
