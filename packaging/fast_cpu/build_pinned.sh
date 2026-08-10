#!/usr/bin/env bash
# Rebuild the exact corrected Gigatoken distribution certified by TokTier.
set -euo pipefail

UPSTREAM_URL="https://github.com/marcelroed/gigatoken"
UPSTREAM_COMMIT="34a1599f0c0ae7d7cd0d1c530e6522320158b360"
PATCH_SHA256="19345bea9a7a5440c3aa1dcebe19a6090685e07e1d0ddce65f773c8dbbcfb506"
RUSTC_VERSION="rustc 1.99.0-nightly (7608eb7b0 2026-08-05)"
RUST_TOOLCHAIN_DATE="nightly-2026-08-06"
CARGO_TOOLCHAIN="1.93.1"
CARGO_VERSION="cargo 1.93.1 (083ac5135 2025-12-15)"
MATURIN_VERSION="1.14.1"
DATAGEN_VERSION="2.2.0"
LICENSE_BUNDLE_SHA256="49d4311b36c8c1886f1752e3dd0c47b66520923fb70b38e7d822039029cb80d7"

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_PATH="$RECIPE_DIR/gigatoken-toktier-pinned-1.patch"
BUILD_ROOT="${TOKTIER_GIGATOKEN_BUILD_ROOT:-$(mktemp -d)}"
SOURCE_DIR="$BUILD_ROOT/source"
VENV_DIR="$BUILD_ROOT/venv"
DATA_DIR="$BUILD_ROOT/icu4x-properties-u16"
WHEEL_DIR="$BUILD_ROOT/wheels"
TOOLS_DIR="$BUILD_ROOT/tools"

if [[ -e "$SOURCE_DIR" || -e "$WHEEL_DIR" ]]; then
    echo "error: build root is not fresh: $BUILD_ROOT" >&2
    echo "choose a new TOKTIER_GIGATOKEN_BUILD_ROOT" >&2
    exit 2
fi

RUST_TOOLCHAIN="nightly"
if [[ "$(rustup run "$RUST_TOOLCHAIN" rustc --version)" != "$RUSTC_VERSION" ]]; then
    rustup toolchain install --profile minimal "$RUST_TOOLCHAIN_DATE"
    RUST_TOOLCHAIN="$RUST_TOOLCHAIN_DATE"
fi
if [[ "$(rustup run "$RUST_TOOLCHAIN" rustc --version)" != "$RUSTC_VERSION" ]]; then
    echo "error: this certified build requires $RUSTC_VERSION" >&2
    echo "observed: $(rustup run "$RUST_TOOLCHAIN" rustc --version)" >&2
    exit 2
fi
if [[ "$(rustup run "$CARGO_TOOLCHAIN" cargo --version)" != "$CARGO_VERSION" ]]; then
    echo "error: this certified build requires $CARGO_VERSION" >&2
    exit 2
fi
RUSTC_BIN="$(rustup which --toolchain "$RUST_TOOLCHAIN" rustc)"
BUILD_ENV=(
    env
    "RUSTUP_TOOLCHAIN=$CARGO_TOOLCHAIN"
    "RUSTC=$RUSTC_BIN"
    "RUSTC_BOOTSTRAP=1"
)

printf '%s  %s\n' "$PATCH_SHA256" "$PATCH_PATH" | sha256sum --check --status
git clone --quiet "$UPSTREAM_URL" "$SOURCE_DIR"
git -C "$SOURCE_DIR" checkout --quiet --detach "$UPSTREAM_COMMIT"
git -C "$SOURCE_DIR" apply --check "$PATCH_PATH"
git -C "$SOURCE_DIR" apply "$PATCH_PATH"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --quiet --disable-pip-version-check \
    "maturin==$MATURIN_VERSION"

DATAGEN_BIN=""
if command -v icu4x-datagen >/dev/null 2>&1 \
    && [[ "$(icu4x-datagen --version)" == "icu4x-datagen $DATAGEN_VERSION" ]]; then
    DATAGEN_BIN="$(command -v icu4x-datagen)"
else
    RUSTUP_TOOLCHAIN="$CARGO_TOOLCHAIN" cargo install --quiet --locked \
        --root "$TOOLS_DIR" \
        icu4x-datagen --version "$DATAGEN_VERSION"
    DATAGEN_BIN="$TOOLS_DIR/bin/icu4x-datagen"
fi

"${BUILD_ENV[@]}" cargo metadata --locked --format-version 1 \
    --manifest-path "$SOURCE_DIR/Cargo.toml" > "$BUILD_ROOT/metadata.json"
"$VENV_DIR/bin/python" "$RECIPE_DIR/generate_license_bundle.py" \
    "$BUILD_ROOT/metadata.json" "$BUILD_ROOT/THIRD_PARTY_LICENSES-gigatoken.txt"
printf '%s  %s\n' "$LICENSE_BUNDLE_SHA256" \
    "$BUILD_ROOT/THIRD_PARTY_LICENSES-gigatoken.txt" | sha256sum --check --status
# The repository file at this historical path now accounts for the complete
# integrated toktier._native closure. The digest check above remains the
# reproducibility gate for this standalone lineage build.
"$VENV_DIR/bin/python" - "$BUILD_ROOT/metadata.json" \
    "$BUILD_ROOT/property-markers.txt" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
manifests = [
    pathlib.Path(package["manifest_path"])
    for package in metadata["packages"]
    if package["name"] == "icu_properties_data" and package["version"] == "2.2.0"
]
if len(manifests) != 1:
    raise SystemExit(f"expected one icu_properties_data 2.2.0 package, found {len(manifests)}")
data_dir = manifests[0].parent / "data"
markers = [
    "".join(word.capitalize() for word in path.name.removesuffix(".rs.data").split("_"))
    for path in sorted(data_dir.glob("*.rs.data"))
]
if not markers:
    raise SystemExit("icu_properties_data supplied no data markers")
pathlib.Path(sys.argv[2]).write_text("\n".join(markers) + "\n", encoding="utf-8")
PY

mapfile -t PROPERTY_MARKERS < "$BUILD_ROOT/property-markers.txt"
"$DATAGEN_BIN" --format baked --out "$DATA_DIR" --overwrite \
    --ucd-tag 16.0.0 --icuexport-tag release-77-1 \
    --markers "${PROPERTY_MARKERS[@]}"

mkdir -p "$WHEEL_DIR"
ICU4X_DATA_DIR="$DATA_DIR" \
    "${BUILD_ENV[@]}" "$VENV_DIR/bin/maturin" build --locked --release \
    --manifest-path "$SOURCE_DIR/Cargo.toml" --out "$WHEEL_DIR"

"$VENV_DIR/bin/python" - "$WHEEL_DIR" <<'PY'
import hashlib
import pathlib
import zipfile
import sys

wheels = list(pathlib.Path(sys.argv[1]).glob("gigatoken-*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"expected one wheel, found {len(wheels)}")
wheel = wheels[0]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    notice = [name for name in names if name.endswith("/licenses/NOTICE-TOKTIER.md")]
    natives = [name for name in names if name.endswith("gigatoken_rs.abi3.so")]
    if len(notice) != 1 or len(natives) != 1:
        raise SystemExit("wheel must carry one modification notice and one native module")
    native_sha = hashlib.sha256(archive.read(natives[0])).hexdigest()
wheel_sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
print(f"wheel={wheel}")
print(f"wheel_sha256={wheel_sha}")
print(f"native_sha256={native_sha}")
print(f"notice={notice[0]}")
PY

echo "build_root=$BUILD_ROOT"
