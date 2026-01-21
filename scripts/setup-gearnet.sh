#!/usr/bin/env bash
set -euo pipefail

# --- Conda env activation (so gdown is available) ---
GEARNET_ENV_NAME="${GEARNET_ENV_NAME:-text2struct-gearnet}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Load your conda module / init conda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Activating conda env: $GEARNET_ENV_NAME"
set +u
conda activate "$GEARNET_ENV_NAME"
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/eval-gearnet/vendor"
PROTEINA_DIR="$VENDOR_DIR/proteina"
CACHE_DIR="$REPO_ROOT/eval-gearnet/cache"
DATA_DIR="$CACHE_DIR/proteina_additional_files"

PROTEINA_REF="${PROTEINA_REF:-main}"
PROTEINA_URL="${PROTEINA_URL:-https://github.com/NVIDIA-Digital-Bio/proteina.git}"

ZIP_FILE="proteina_additional_files.zip"
NGC_URL="https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/clara/proteina_additional_files/1.0/files?redirect=true&path=proteina_additional_files.zip"

mkdir -p "$VENDOR_DIR" "$CACHE_DIR"

echo "== GearNet setup =="

if [[ -d "$PROTEINA_DIR/.git" ]]; then
  git -C "$PROTEINA_DIR" fetch --tags --force
  git -C "$PROTEINA_DIR" checkout "$PROTEINA_REF" || git -C "$PROTEINA_DIR" checkout -f FETCH_HEAD
else
  git clone --depth 1 --branch "$PROTEINA_REF" "$PROTEINA_URL" "$PROTEINA_DIR" \
    || (git clone "$PROTEINA_URL" "$PROTEINA_DIR" && git -C "$PROTEINA_DIR" checkout "$PROTEINA_REF")
fi

echo "Proteina ref: $(git -C "$PROTEINA_DIR" rev-parse HEAD)"

cd "$CACHE_DIR"

W1="$DATA_DIR/metric_factory/model_weights/gearnet_ca.pth"
W2="$DATA_DIR/pdb_raw/cath_label_mapping.pt"

if [[ ! -f "$W1" ]] || [[ ! -f "$W2" ]]; then
  echo "Downloading proteina_additional_files from NGC..."
  curl -L --fail "$NGC_URL" -o "$ZIP_FILE"
  unzip -q "$ZIP_FILE"
  rm -f "$ZIP_FILE"
  rm -rf __MACOSX
else
  echo "proteina_additional_files already present, skipping download"
fi

echo "Files:"
ls -lh "$W1" "$W2"

# --- CATH node names cache (for human-readable labels) ---

echo
echo "Export for Proteina-compatible paths:"
echo "  export DATA_PATH=\"$DATA_DIR\""
echo "Done."
