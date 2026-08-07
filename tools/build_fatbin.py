#!/usr/bin/env python3
"""Build the prebuilt-delivery fatbin and its build manifest.

One ``nvcc -fatbin`` invocation compiles ``src/toktier/kernels/
prebuilt_unit.cu`` (which includes the pristine ``pretok_kernel.cu``)
for every target architecture and embeds a ``compute_75`` PTX image as
the forward-JIT fallback for architectures not on the list. The build
manifest records everything a verifier needs to tie the shipped bytes
to this build: toolchain version, complete ``nvcc`` argument list,
per-architecture image digests, the fatbin digest, the source lineage
digests, the kernel symbol map and the build host facts.

The script verifies, before writing anything into the package, that
every expected kernel entry point is present in every embedded image;
a missing kernel fails the build rather than shipping a fatbin that
would fail at load time.

Usage::

    python tools/build_fatbin.py            # build into the package tree
    python tools/build_fatbin.py --check    # verify shipped fatbin + manifest

Run inside an environment with torch installed (only its header tree is
used; no GPU is needed to build).
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
KERNELS_DIR = SRC_ROOT / "toktier" / "kernels"
UNIT = KERNELS_DIR / "prebuilt_unit.cu"

sys.path.insert(0, str(SRC_ROOT))

from toktier.kernels import kernel_source_digest  # noqa: E402
from toktier.kernels.prebuilt import (  # noqa: E402
    FATBIN_NAME,
    MANIFEST_NAME,
    PREBUILT_DIR,
    cubin_digest,
    fatbin_digest,
    prebuilt_source_digest,
)

MANIFEST_SCHEMA = "toktier_prebuilt_build_manifest_v1"

#: Real device images embedded in the fatbin, in `sm_<cc>` spelling.
SM_TARGETS = ("sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120")

#: PTX fallback image for architectures not listed above (driver JIT).
PTX_TARGET = "compute_75"

#: Kernel entry points every embedded image must contain. Keys are the
#: logical names the launcher uses; values match the demangled symbol
#: (up to the parameter list).
EXPECTED_KERNELS = {
    # RS-templated pretokenization stages (0=cl100k, 1=deepseek, 2=laguna).
    "k_classify_rs0": "void k_classify<0>",
    "k_classify_rs1": "void k_classify<1>",
    "k_classify_rs2": "void k_classify<2>",
    "k_runinfo_rs0": "void k_runinfo<0>",
    "k_runinfo_rs1": "void k_runinfo<1>",
    "k_runinfo_rs2": "void k_runinfo<2>",
    "k_rules_rs0": "void k_rules<0>",
    "k_rules_rs1": "void k_rules<1>",
    "k_rules_rs2": "void k_rules<2>",
    # UTF-8 decode and shared helpers.
    "k_utf8_decode": "k_utf8_decode",
    "k_dso_seed": "k_dso_seed",
    # DeepSeek prepass.
    "k_ds_seed_n": "k_ds_seed_n",
    "k_ds_bmask": "k_ds_bmask",
    # Laguna stage-0 prepass.
    "k_lag_bmask": "k_lag_bmask",
    # BPE stage.
    "k_bpe_thread_cap32": "void k_bpe_thread<32>",
    "k_bpe_warp": "k_bpe_warp",
    "k_bpe_long": "k_bpe_long",
    "k_bpe_compact": "k_bpe_compact",
    "k_memo_insert": "k_memo_insert",
    # Fused-path glue.
    "k_pb_sentinel": "k_pb_sentinel",
    "k_dispatch_flags": "k_dispatch_flags",
    # NFC quick check.
    "k_nfc_qc": "k_nfc_qc",
    # o200k / kimi splitter group.
    "k_o2k_heads": "k_o2k_heads",
    "k_o2k_runinfo1": "k_o2k_runinfo1",
    "k_o2k_runinfo2": "k_o2k_runinfo2",
    "k_o2k_rules": "k_o2k_rules",
    "k_o2k_win_extents": "k_o2k_win_extents",
    "k_o2k_win_clear": "k_o2k_win_clear",
    "k_o2k_win_mark": "k_o2k_win_mark",
    "k_o2k_meta3": "k_o2k_meta3",
    # Launcher-support kernels (extern "C", unmangled).
    "tk_carrier_scatter": "tk_carrier_scatter",
    "tk_carrier_gather": "tk_carrier_gather",
    "tk_select_scatter": "tk_select_scatter",
    "tk_ds_constants_dump": "tk_ds_constants_dump",
}


def log(message: str) -> None:
    print(f"[build_fatbin] {message}", file=sys.stderr, flush=True)


def find_tool(name: str, cuda_home: Path) -> str:
    candidate = cuda_home / "bin" / name
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(name)
    if found is None:
        raise SystemExit(f"error: {name} was not found (looked in {candidate})")
    return found


def torch_include_paths() -> list[str]:
    try:
        from torch.utils import cpp_extension
    except ImportError as exc:  # pragma: no cover - environment surface
        raise SystemExit(
            "error: torch is required to locate its header tree "
            f"(import failed: {exc})"
        ) from exc
    import sysconfig

    paths = list(cpp_extension.include_paths(device_type="cuda"))
    paths.append(sysconfig.get_paths()["include"])
    return paths


def nvcc_version(nvcc: str) -> str:
    text = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"release ([\d.]+), (V[\d.]+)", text)
    if match is None:
        raise SystemExit(f"error: cannot parse nvcc version from: {text!r}")
    return f"cuda {match.group(1)} ({match.group(2)})"


def build_argv(nvcc: str, includes: list[str], out: Path) -> list[str]:
    argv = [
        nvcc,
        "-fatbin",
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
    ]
    for path in includes:
        argv.append(f"-I{path}")
    argv += [
        "-DTORCH_EXTENSION_NAME=toktier_pretok_cuda",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ]
    for target in SM_TARGETS:
        cc = target.split("_", 1)[1]
        argv.append(f"-gencode=arch=compute_{cc},code=sm_{cc}")
    ptx_cc = PTX_TARGET.split("_", 1)[1]
    argv.append(f"-gencode=arch=compute_{ptx_cc},code=compute_{ptx_cc}")
    argv += [str(UNIT), "-o", str(out)]
    return argv


def demangle(cxxfilt: str, symbols: list[str]) -> dict[str, str]:
    """Mangled -> demangled map through one c++filt invocation."""
    proc = subprocess.run(
        [cxxfilt],
        input="\n".join(symbols) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    lines = proc.stdout.splitlines()
    if len(lines) != len(symbols):
        raise SystemExit("error: c++filt returned an unexpected line count")
    return dict(zip(symbols, lines, strict=True))


def image_symbols(cuobjdump: str, fatbin: Path) -> dict[str, list[str]]:
    """Entry-point symbols per embedded image, keyed by architecture."""
    text = subprocess.run(
        [cuobjdump, "-symbols", str(fatbin)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    images: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        match = re.match(r"\s*arch = (sm_\d+)", line)
        if match:
            current = images.setdefault(match.group(1), [])
            continue
        if current is not None and "STT_FUNC" in line and "STO_ENTRY" in line:
            current.append(line.split()[-1])
    return images


def ptx_symbols(cuobjdump: str, fatbin: Path) -> list[str]:
    """Entry symbols of the embedded PTX image (dumped as text)."""
    text = subprocess.run(
        [cuobjdump, "-ptx", str(fatbin)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return re.findall(r"\.visible \.entry (\w+)", text)


def extract_images(cuobjdump: str, fatbin: Path) -> dict[str, bytes]:
    """Per-architecture image bytes: cubins via -xelf, PTX via -xptx."""
    images: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for flag in ("-xelf", "-xptx"):
            subprocess.run(
                [cuobjdump, flag, "all", str(fatbin)],
                check=True,
                capture_output=True,
                cwd=tmp,
            )
        for path in sorted(Path(tmp).iterdir()):
            match = re.search(r"\.(sm_\d+)\.(cubin|ptx)$", path.name)
            if not match:
                continue
            arch = match.group(1)
            key = (
                arch
                if match.group(2) == "cubin"
                else f"compute_{arch.split('_')[1]}"
            )
            images[key] = path.read_bytes()
    return images


def build_symbol_map(
    cuobjdump: str, cxxfilt: str, fatbin: Path
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """(logical name -> mangled symbol, per-arch missing-kernel report)."""
    per_image = image_symbols(cuobjdump, fatbin)
    ptx_syms = ptx_symbols(cuobjdump, fatbin)
    if ptx_syms:
        per_image.setdefault(PTX_TARGET, ptx_syms)
    all_symbols = sorted({s for syms in per_image.values() for s in syms})
    demangled = demangle(cxxfilt, all_symbols)

    def logical_of(mangled: str) -> str | None:
        pretty = demangled[mangled].split("(")[0].strip()
        for logical, prefix in EXPECTED_KERNELS.items():
            if pretty == prefix:
                return logical
        return None

    symbol_map: dict[str, str] = {}
    for mangled in all_symbols:
        logical = logical_of(mangled)
        if logical is None:
            continue
        previous = symbol_map.setdefault(logical, mangled)
        if previous != mangled:
            raise SystemExit(
                f"error: logical kernel {logical} maps to two symbols "
                f"({previous} and {mangled})"
            )
    missing_report: dict[str, list[str]] = {}
    for arch in (*SM_TARGETS, PTX_TARGET):
        symbols = set(per_image.get(arch, ()))
        missing = [
            logical
            for logical, mangled in symbol_map.items()
            if mangled not in symbols
        ]
        absent = [k for k in EXPECTED_KERNELS if k not in symbol_map]
        missing_report[arch] = sorted(missing + absent)
    return symbol_map, missing_report


def driver_facts() -> dict[str, object]:
    # Host names are deliberately omitted; the GPU models and driver
    # versions, which carry evidence, are written.
    facts: dict[str, object] = {
        "platform": platform.platform(),
    }
    try:
        text = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        rows = [line.split(", ") for line in text.splitlines() if line]
        facts["gpus"] = [
            {"name": r[0], "compute_cap": r[1], "driver_version": r[2]}
            for r in rows
        ]
    except (OSError, subprocess.SubprocessError):
        facts["gpus"] = []
    return facts


def portable_argv(argv: list[str]) -> list[str]:
    """The recorded compiler invocation, with machine-local roots masked.

    The flags, defines and gencode targets carry the build semantics and
    are recorded verbatim. Include roots and the input path only say
    where this particular machine kept its packages and checkout; the
    recorded form replaces the environment prefix with
    ``<site-packages>`` and the checkout prefix with the repository-
    relative path, so the manifest states the build without stating the
    builder's directory layout.
    """
    roots: list[tuple[str, str]] = []
    try:
        import torch

        site_root = str(Path(torch.__file__).resolve().parents[1])
        roots.append((site_root, "<site-packages>"))
    except ImportError:  # pragma: no cover - check-only environments
        pass
    roots.append((str(REPO_ROOT), "."))
    recorded: list[str] = []
    for token in argv:
        for root, label in roots:
            if root and root in token:
                token = token.replace(root + "/", label + "/").replace(
                    root, label
                )
        recorded.append(token)
    return recorded


def build_manifest(
    argv: list[str],
    nvcc_release: str,
    fatbin_bytes: bytes,
    images: dict[str, bytes],
    symbol_map: dict[str, str],
) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "built_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "toolchain": nvcc_release,
        "nvcc_argv": portable_argv(argv),
        "build_host": driver_facts(),
        "architectures": {
            arch: {
                "kind": "ptx" if arch.startswith("compute_") else "cubin",
                "size": len(data),
                "digest": cubin_digest(arch, data),
            }
            for arch, data in sorted(images.items())
        },
        "ptx_fallback": PTX_TARGET,
        "fatbin": {
            "file": FATBIN_NAME,
            "size": len(fatbin_bytes),
            "digest": fatbin_digest(fatbin_bytes),
        },
        "sources": {
            "prebuilt_source_digest": prebuilt_source_digest(),
            "jit_kernel_source_digest": kernel_source_digest(),
        },
        "kernels": symbol_map,
        "notes": [
            "sub-sm_80 images substitute a shuffle reduction for the "
            "__reduce_min_sync builtin (see prebuilt_unit.cu); "
            "architectures below sm_80 are experimental-tier deliveries",
        ],
    }


def run_build(cuda_home: Path) -> int:
    nvcc = find_tool("nvcc", cuda_home)
    cuobjdump = find_tool("cuobjdump", cuda_home)
    cxxfilt = shutil.which("c++filt") or find_tool("cu++filt", cuda_home)
    includes = torch_include_paths()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / FATBIN_NAME
        argv = build_argv(nvcc, includes, out)
        log("nvcc: " + " ".join(argv))
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
            return proc.returncode
        symbol_map, missing = build_symbol_map(cuobjdump, cxxfilt, out)
        problems = {a: m for a, m in missing.items() if m}
        if problems:
            for arch, kernels in problems.items():
                log(f"missing kernels in {arch}: {kernels}")
            return 1
        fatbin_bytes = out.read_bytes()
        images = extract_images(cuobjdump, out)
        expected_images = set(SM_TARGETS) | {PTX_TARGET}
        if set(images) != expected_images:
            log(
                "embedded images do not match the target list: "
                f"got {sorted(images)}, expected {sorted(expected_images)}"
            )
            return 1
        manifest = build_manifest(
            build_argv("nvcc", includes, Path(FATBIN_NAME)),
            nvcc_version(nvcc),
            fatbin_bytes,
            images,
            symbol_map,
        )
        PREBUILT_DIR.mkdir(parents=True, exist_ok=True)
        (PREBUILT_DIR / FATBIN_NAME).write_bytes(fatbin_bytes)
        (PREBUILT_DIR / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
        )
    log(
        f"wrote {PREBUILT_DIR / FATBIN_NAME} "
        f"({len(fatbin_bytes):,} bytes, {len(images)} images) and manifest"
    )
    return 0


def run_check() -> int:
    from toktier.kernels.prebuilt import load_manifest

    problems: list[str] = []
    try:
        manifest = load_manifest()
    except (OSError, ValueError) as exc:
        log(f"manifest unreadable: {exc}")
        return 1
    fatbin_file = PREBUILT_DIR / FATBIN_NAME
    if not fatbin_file.is_file():
        problems.append("fatbin file is not present")
    else:
        observed = fatbin_digest(fatbin_file.read_bytes())
        recorded = manifest["fatbin"]["digest"]
        if observed != recorded:
            problems.append(
                f"fatbin digest mismatch: shipped {observed}, "
                f"manifest {recorded}"
            )
    if manifest["sources"]["prebuilt_source_digest"] != prebuilt_source_digest():
        problems.append("prebuilt source digest drifted since the build")
    if manifest["sources"]["jit_kernel_source_digest"] != kernel_source_digest():
        problems.append("JIT kernel source digest drifted since the build")
    for problem in problems:
        log(f"check: {problem}")
    if not problems:
        log("check passed")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=Path("/usr/local/cuda"),
        help="CUDA toolkit root holding bin/nvcc (default: /usr/local/cuda).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the shipped fatbin and manifest instead of building.",
    )
    arguments = parser.parse_args(argv)
    if arguments.check:
        return run_check()
    return run_build(arguments.cuda_home)


if __name__ == "__main__":
    raise SystemExit(main())
