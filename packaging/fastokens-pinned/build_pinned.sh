#!/usr/bin/env bash
# Rebuild the pinned fastokens distribution that the toktier project publishes
# as toktier-fastokens.
#
# The build starts from a clean checkout of upstream fastokens tag v0.3.1,
# applies the patch series in PATCHES/ in order, checks that the resulting
# tree hash is the expected one, rewrites only the *distribution* metadata
# (name, version, description, readme, license files), and then builds a
# wheel and an sdist with a pinned Rust toolchain and a pinned maturin.
#
# The import package name is NOT changed: the distribution still installs a
# top-level ``fastokens`` package, so it is a drop-in for code that imports it.
#
# Two trees. Patches 0001-0005 change code and yield tree aa192428...; that is
# the tree the published 0.3.1.1 wheel and its readings were built from. Patch
# 0006 adds one notice comment line per modified file (Apache-2.0 section
# 4(b)) and yields tree aaa5fb94.... The script applies the full series by
# default; APPLY_NOTICE_PATCH=0 stops after 0005. The two trees do not compile
# to the same bytes (source line numbers reach the binary through diagnostics
# metadata), so a wheel's engine digest always says which tree it came from.
#
# Reproducibility. The compiled extension depends on the absolute path of the
# dependency source root (vendored crates or the cargo registry); with the
# same toolchain, the same tree and the same paths it reproduces byte for
# byte. Pass a fixed TOKTIER_FASTOKENS_BUILD_ROOT to reproduce a build.
#
# Nothing here uploads anything. The script only writes into its build root.

set -euo pipefail

# ---------------------------------------------------------------- parameters

DIST_NAME="${DIST_NAME:-toktier-fastokens}"        # PyPI distribution name
DIST_VERSION="${DIST_VERSION:-0.3.1.1}"            # PEP 440; PyPI rejects local versions
IMPORT_NAME="fastokens"                            # unchanged on purpose
PATCH_SET_ID="${PATCH_SET_ID:-toktier patch set 1}"
# Home-Page in the wheel metadata. maturin 1.14.1 fills this from the crate's
# `homepage` field and ignores `[project.urls]` in this mixed layout, so the
# rewrite below edits the crate field; leaving it alone would publish upstream's
# project page as this distribution's home page.
HOMEPAGE="${HOMEPAGE:-https://github.com/asu-idi/toktier}"

# The upstream project moved from crusoecloud/fastokens to Atero-ai/fastokens;
# the v0.3.1 Cargo.toml itself names the latter.
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/Atero-ai/fastokens.git}"
UPSTREAM_TAG="v0.3.1"
UPSTREAM_COMMIT="fe854299553524f2156a22036a2cb4d1f2ef4d97"
# Tree hash after the five code patches: the tree the published 0.3.1.1
# wheel was built from and every reading quoted for it was taken on.
CODE_TREE_SHA="aa1924284ec4abaedcc8ed5823ee17e7959c55c5"
# Tree hash after the notice patch 0006 on top of the five.
NOTICED_TREE_SHA="aaa5fb94ea62b9379d03074640e267c8d837d649"
# 1 (default): apply 0006 as well and expect NOTICED_TREE_SHA; 0: stop after
# 0005 and expect CODE_TREE_SHA.
APPLY_NOTICE_PATCH="${APPLY_NOTICE_PATCH:-1}"

RUSTC_VERSION="${RUSTC_VERSION:-rustc 1.93.1 (01f6ddf75 2026-02-11)}"
CARGO_VERSION="${CARGO_VERSION:-cargo 1.93.1 (083ac5135 2025-12-15)}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.93.1}"
MATURIN_VERSION="${MATURIN_VERSION:-1.14.1}"

# sha256 of each patch, in application order. Checked before anything is applied.
PATCH_FILES=(
    "0001-F040-fix-bpe-resolve-out-of-vocabulary-characters.patch"
    "0002-F041-fix-pcre2-align-unicode-class-semantics.patch"
    "0003-F042-fix-scan-chunk-boundaries-inside-pretokens.patch"
    "0004-F045-fix-scan-accept-long-s-in-contractions.patch"
    "0005-F046-fix-split-rescan-tail-after-merge.patch"
    "0006-notices-mark-modified-files.patch"
)
PATCH_SHA256=(
    "41814e13e60286371ce74ec2a22ae84517a2f99fb315f7fa2044c4af3568c583"
    "e780c21e03a05bc619f67f3cb0df4d12e28f03e9cf3264c4325446531af640ce"
    "aa724b35c3bf4da666be507cef79c9737deb3209c97d393ba5056e228c408549"
    "0b5fff18801cde5dc70310c3e72625c4094c0e874abe883bba9cd5898935d5ad"
    "d7b6f6e603ae0852e8d00fc6fc8a294ff87e520001a886c53e7e79fb5c557542"
    "ee2bb9cfc88603a665d4d50ec916a72ea2f2bebea686104a0e9a98aba54e784b"
)
if [[ "$APPLY_NOTICE_PATCH" == "1" ]]; then
    PATCHED_TREE_SHA="$NOTICED_TREE_SHA"
else
    PATCH_FILES=("${PATCH_FILES[@]:0:5}")
    PATCH_SHA256=("${PATCH_SHA256[@]:0:5}")
    PATCHED_TREE_SHA="$CODE_TREE_SHA"
fi

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEGAL_DIR="$KIT_DIR"
BUILD_ROOT="${TOKTIER_FASTOKENS_BUILD_ROOT:-$(mktemp -d)}"
SOURCE_DIR="$BUILD_ROOT/source"
VENV_DIR="$BUILD_ROOT/venv"
OUT_DIR="${OUT_DIR:-$BUILD_ROOT/dist}"

# Offline knobs, used on hosts without outbound network (for example the SOL
# lightwork partition). SOURCE_TARBALL replaces the git clone; VENDOR_TARBALL
# supplies a `cargo vendor` tree; MATURIN_BIN / PYTHON_BIN reuse an existing
# interpreter instead of building a venv.
SOURCE_TARBALL="${SOURCE_TARBALL:-}"
VENDOR_TARBALL="${VENDOR_TARBALL:-}"
VENDOR_DIR="${VENDOR_DIR:-}"
MATURIN_BIN="${MATURIN_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Interpreter maturin builds against. abi3 means the tag is cp39-abi3 either way,
# but maturin refuses to run against a python older than 3.7, and `python3` on a
# Rocky 8 compute node is 3.6 -- so name it explicitly rather than let maturin
# pick whatever is first on PATH.
INTERPRETER="${INTERPRETER:-$PYTHON_BIN}"
CARGO_OFFLINE="${CARGO_OFFLINE:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
BUILD_JOBS="${BUILD_JOBS:-16}"

step() { printf '\n=== %s ===\n' "$*"; }
rc_of() { local rc=$?; printf 'rc=%d  (%s)\n' "$rc" "$1"; return 0; }

step "parameters"
cat <<EOF
dist_name        = $DIST_NAME
dist_version     = $DIST_VERSION
import_name      = $IMPORT_NAME
patch_set        = $PATCH_SET_ID
upstream         = $UPSTREAM_URL @ $UPSTREAM_TAG ($UPSTREAM_COMMIT)
patches          = ${#PATCH_FILES[@]} (APPLY_NOTICE_PATCH=$APPLY_NOTICE_PATCH)
patched_tree     = $PATCHED_TREE_SHA
build_root       = $BUILD_ROOT
out_dir          = $OUT_DIR
host             = $(uname -srm) / $(getconf GNU_LIBC_VERSION 2>/dev/null || echo 'glibc ?')
date_utc         = $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

if [[ -e "$SOURCE_DIR" ]]; then
    echo "error: build root is not fresh: $BUILD_ROOT" >&2
    exit 2
fi
mkdir -p "$BUILD_ROOT" "$OUT_DIR"

# ------------------------------------------------------------- toolchain pin

step "toolchain"
if command -v rustup >/dev/null 2>&1; then
    RUSTC_BIN="$(rustup which --toolchain "$RUST_TOOLCHAIN" rustc)"
    CARGO_BIN="$(rustup which --toolchain "$RUST_TOOLCHAIN" cargo)"
else
    RUSTC_BIN="$(command -v rustc)"
    CARGO_BIN="$(command -v cargo)"
fi
observed_rustc="$("$RUSTC_BIN" --version)"
observed_cargo="$("$CARGO_BIN" --version)"
echo "rustc: $observed_rustc"
echo "cargo: $observed_cargo"
if [[ "$observed_rustc" != "$RUSTC_VERSION" ]]; then
    echo "error: this build is pinned to $RUSTC_VERSION" >&2
    exit 2
fi
if [[ "$observed_cargo" != "$CARGO_VERSION" ]]; then
    echo "error: this build is pinned to $CARGO_VERSION" >&2
    exit 2
fi
# Pin the toolchain for every rustc that cargo spawns as well, so the build
# never falls through to whatever `stable` happens to point at.
if command -v rustup >/dev/null 2>&1; then
    export RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN"
fi
export RUSTC="$RUSTC_BIN"

# ------------------------------------------------------------------- sources

step "clean upstream source"
if [[ -n "$SOURCE_TARBALL" ]]; then
    echo "source tarball: $SOURCE_TARBALL"
    sha256sum "$SOURCE_TARBALL"
    mkdir -p "$SOURCE_DIR"
    tar xzf "$SOURCE_TARBALL" -C "$SOURCE_DIR"
    # A tarball is expected to already carry the patched tree; the git checks
    # below are skipped and the caller must state its provenance.
    SOURCE_IS_GIT=0
else
    git clone --quiet "$UPSTREAM_URL" "$SOURCE_DIR"
    git -C "$SOURCE_DIR" checkout --quiet --detach "$UPSTREAM_COMMIT"
    echo "checked out $(git -C "$SOURCE_DIR" rev-parse HEAD)"
    test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
    SOURCE_IS_GIT=1
fi
rc_of "source"

if [[ "$SOURCE_IS_GIT" == "1" ]]; then
    step "patch series"
    for i in "${!PATCH_FILES[@]}"; do
        p="$LEGAL_DIR/PATCHES/${PATCH_FILES[$i]}"
        printf '%s  %s\n' "${PATCH_SHA256[$i]}" "$p" | sha256sum --check --status
        git -C "$SOURCE_DIR" apply --check "$p"
        git -C "$SOURCE_DIR" apply "$p"
        echo "applied ${PATCH_FILES[$i]}  sha256=${PATCH_SHA256[$i]}"
    done
    rc_of "patches"

    step "patched tree hash"
    git -C "$SOURCE_DIR" add -A
    observed_tree="$(git -C "$SOURCE_DIR" write-tree)"
    echo "observed tree = $observed_tree"
    echo "expected tree = $PATCHED_TREE_SHA"
    if [[ "$observed_tree" != "$PATCHED_TREE_SHA" ]]; then
        echo "error: patched tree does not match the certified revision" >&2
        exit 3
    fi
    rc_of "tree hash"
fi

# ------------------------------------------------------- distribution rename

step "distribution metadata"
cp "$LEGAL_DIR/LICENSE-fastokens" "$SOURCE_DIR/LICENSE"
cp "$LEGAL_DIR/NOTICE-fastokens-pinned" "$SOURCE_DIR/NOTICE"
cp "$LEGAL_DIR/CHANGES-toktier.md" "$SOURCE_DIR/CHANGES-toktier.md"
cp "$LEGAL_DIR/README-dist.md" "$SOURCE_DIR/README-dist.md"
cp "$LEGAL_DIR/THIRD_PARTY_LICENSES-fastokens.txt" "$SOURCE_DIR/THIRD_PARTY_LICENSES-fastokens.txt"
mkdir -p "$SOURCE_DIR/PATCHES"
for i in "${!PATCH_FILES[@]}"; do
    cp "$LEGAL_DIR/PATCHES/${PATCH_FILES[$i]}" "$SOURCE_DIR/PATCHES/"
done

DIST_NAME="$DIST_NAME" DIST_VERSION="$DIST_VERSION" \
PATCH_SET_ID="$PATCH_SET_ID" HOMEPAGE="$HOMEPAGE" \
"$PYTHON_BIN" - "$SOURCE_DIR" <<'PY'
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
name = os.environ["DIST_NAME"]
version = os.environ["DIST_VERSION"]
patch_set = os.environ["PATCH_SET_ID"]
homepage = os.environ["HOMEPAGE"]

# ---- pyproject.toml: distribution name, version, description, legal files.
# `module-name`, `python-source` and `manifest-path` are left exactly as
# upstream wrote them, so the installed import package is still `fastokens`.
path = root / "pyproject.toml"
text = path.read_text(encoding="utf-8")
assert 'name = "fastokens"' in text, "unexpected pyproject layout"
assert 'dynamic = ["version"]' in text, "unexpected pyproject layout"
assert 'module-name = "fastokens._native"' in text, "unexpected pyproject layout"
text = text.replace('name = "fastokens"', f'name = "{name}"')
description = (
    f"Pinned build of fastokens 0.3.1 with the {patch_set} "
    "(independent of the upstream project)"
)
text = text.replace(
    'dynamic = ["version"]',
    "\n".join(
        [
            f'version = "{version}"',
            f'description = "{description}"',
            'readme = "README-dist.md"',
            'license = "Apache-2.0"',
            'license-files = [',
            '    "LICENSE",',
            '    "NOTICE",',
            '    "NOTICES.txt",',
            '    "CHANGES-toktier.md",',
            '    "THIRD_PARTY_LICENSES-fastokens.txt",',
            '    "PATCHES/*.patch",',
            ']',
        ]
    ),
)
# The same files ride along in the sdist at the top level. They are not added
# to the wheel payload: `license-files` already places them under
# ``dist-info/licenses`` and site-packages should stay clean.
text = text.replace(
    '[tool.maturin]',
    '[tool.maturin]\n'
    'include = [\n'
    '    { path = "NOTICE", format = "sdist" },\n'
    '    { path = "CHANGES-toktier.md", format = "sdist" },\n'
    '    { path = "THIRD_PARTY_LICENSES-fastokens.txt", format = "sdist" },\n'
    '    { path = "PATCHES/*.patch", format = "sdist" },\n'
    ']',
    1,
)
path.write_text(text, encoding="utf-8")
print(text)

# ---- Cargo.toml: the crate `homepage` is what maturin writes into the
# wheel's Home-Page field. Left alone it would name upstream's project page
# as this distribution's home page.
path = root / "Cargo.toml"
text = path.read_text(encoding="utf-8")
old = 'homepage = "https://github.com/Atero-ai/fastokens"'
assert old in text, "unexpected Cargo.toml layout"
text = text.replace(old, f'homepage = "{homepage}"', 1)
path.write_text(text, encoding="utf-8")
print(f"crate homepage -> {homepage}")
PY
rc_of "metadata"

# ------------------------------------------------------------------ vendoring

if [[ -n "$VENDOR_TARBALL" || -n "$VENDOR_DIR" ]]; then
    step "vendored crates"
    if [[ -n "$VENDOR_TARBALL" ]]; then
        sha256sum "$VENDOR_TARBALL"
        tar xzf "$VENDOR_TARBALL" -C "$BUILD_ROOT"
        VENDOR_DIR="$BUILD_ROOT/vendor"
    fi
    echo "vendor dir: $VENDOR_DIR ($(ls "$VENDOR_DIR" | wc -l) crates)"
    # The source replacement goes in a build-root-local CARGO_HOME, NOT in the
    # source tree. `.cargo/config.toml` ships inside the sdist, so an override
    # written there would put this host's absolute vendor path into the
    # published source distribution, where it cannot resolve.
    export CARGO_HOME="$BUILD_ROOT/cargo"
    mkdir -p "$CARGO_HOME"
    cat > "$CARGO_HOME/config.toml" <<CFG
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "$VENDOR_DIR"
CFG
    echo "CARGO_HOME=$CARGO_HOME (source replacement written there)"
    CARGO_OFFLINE=1
    rc_of "vendor"
fi

OFFLINE_FLAG=()
[[ "$CARGO_OFFLINE" == "1" ]] && OFFLINE_FLAG=(--offline)

# --------------------------------------------------------------------- build

export CARGO_TARGET_DIR="$BUILD_ROOT/target"

step "cargo build --locked --release"
( cd "$SOURCE_DIR" && "$CARGO_BIN" build --locked --release -j "$BUILD_JOBS" "${OFFLINE_FLAG[@]}" )
rc_of "cargo build"

if [[ "$RUN_TESTS" == "1" ]]; then
    step "upstream test suite (cargo test --release)"
    ( cd "$SOURCE_DIR" && "$CARGO_BIN" test --locked --release -j "$BUILD_JOBS" "${OFFLINE_FLAG[@]}" 2>&1 ) \
        | tee "$BUILD_ROOT/cargo_test.log" \
        | grep -E "^test result|FAILED|panicked" || true
    rc_of "cargo test"
fi

step "maturin"
if [[ -z "$MATURIN_BIN" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --quiet --disable-pip-version-check \
        "maturin==$MATURIN_VERSION"
    MATURIN_BIN="$VENV_DIR/bin/maturin"
fi
"$MATURIN_BIN" --version
test "$("$MATURIN_BIN" --version)" = "maturin $MATURIN_VERSION"

echo "interpreter: $INTERPRETER ($("$INTERPRETER" -V 2>&1))"
( cd "$SOURCE_DIR" && "$MATURIN_BIN" build --locked --release \
    --manifest-path python/Cargo.toml --out "$OUT_DIR" \
    -i "$INTERPRETER" "${OFFLINE_FLAG[@]}" )
rc_of "maturin build"

( cd "$SOURCE_DIR" && "$MATURIN_BIN" sdist \
    --manifest-path python/Cargo.toml --out "$OUT_DIR" )
rc_of "maturin sdist"

# ------------------------------------------------------------------ digests

step "artifacts"
( cd "$OUT_DIR" && sha256sum ./*.whl ./*.tar.gz )
echo "build_root=$BUILD_ROOT"
echo "out_dir=$OUT_DIR"
