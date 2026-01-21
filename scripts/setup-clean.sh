#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Conda env activation (so gdown is available) ---
CLEAN_ENV_NAME="${CLEAN_ENV_NAME:-text2struct-clean}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Load your conda module / init conda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Activating conda env: $CLEAN_ENV_NAME"
set +u
conda activate "$CLEAN_ENV_NAME"
set -u

if ! command -v gdown >/dev/null 2>&1; then
  echo "ERROR: gdown not found in env '$CLEAN_ENV_NAME'. Did you create the env from envs/clean.yml?"
  exit 1
fi

# --- Paths ---
VENDOR_DIR="$REPO_ROOT/eval-clean/vendor"
CLEAN_DIR="$VENDOR_DIR/CLEAN"

# Pin a tag/commit for reproducibility (override if you want)
CLEAN_REF="${CLEAN_REF:-v1.0.1}"
CLEAN_URL="${CLEAN_URL:-https://github.com/tttianhao/CLEAN.git}"

# IMPORTANT: CLEAN expects esm at repo-root ./esm (relative paths)
ESM_URL="${ESM_URL:-https://github.com/facebookresearch/esm.git}"
ESM_DIR="$CLEAN_DIR/esm"

# Download pretrained bundle by default
DOWNLOAD_PRETRAINED="${DOWNLOAD_PRETRAINED:-1}"
CLEAN_PRETRAIN_FUZZY_URL="${CLEAN_PRETRAIN_FUZZY_URL:-https://drive.google.com/file/d/1kwYd4VtzYuMvJMWXy6Vks91DSUAOcKpZ/view?usp=sharing}"

mkdir -p "$VENDOR_DIR"

echo "== CLEAN setup =="
echo "REPO_ROOT:           $REPO_ROOT"
echo "VENDOR_DIR:          $VENDOR_DIR"
echo "CLEAN_URL:           $CLEAN_URL"
echo "CLEAN_REF:           $CLEAN_REF"
echo "CLEAN_DIR:           $CLEAN_DIR"
echo "ESM_URL:             $ESM_URL"
echo "ESM_DIR:             $ESM_DIR"
echo "DOWNLOAD_PRETRAINED: $DOWNLOAD_PRETRAINED"
echo

# --- Clone/update CLEAN ---
if [[ -d "$CLEAN_DIR/.git" ]]; then
  echo "Found existing CLEAN repo. Updating + checking out $CLEAN_REF ..."
  git -C "$CLEAN_DIR" fetch --tags --force
  git -C "$CLEAN_DIR" checkout "$CLEAN_REF" || git -C "$CLEAN_DIR" checkout -f FETCH_HEAD
else
  echo "Cloning CLEAN into: $CLEAN_DIR"
  git clone "$CLEAN_URL" "$CLEAN_DIR"
  git -C "$CLEAN_DIR" fetch --tags --force
  git -C "$CLEAN_DIR" checkout "$CLEAN_REF" || true
fi

echo
echo "CLEAN ref:"
git -C "$CLEAN_DIR" rev-parse HEAD

# --- Ensure canonical directories CLEAN code uses ---
mkdir -p "$CLEAN_DIR/data/esm_data"
mkdir -p "$CLEAN_DIR/data/pretrained"

# Optional convenience: keep app/data pointing to canonical dirs (harmless)
mkdir -p "$CLEAN_DIR/app/data"
ln -sfn ../../data/esm_data "$CLEAN_DIR/app/data/esm_data"
ln -sfn ../../data/pretrained "$CLEAN_DIR/app/data/pretrained"

# --- Clone ESM to repo-root ./esm (CLEAN code expects ./esm/scripts/extract.py) ---
if [[ -d "$ESM_DIR/.git" ]]; then
  echo
  echo "Found existing ESM repo at: $ESM_DIR"
else
  echo
  echo "Cloning ESM into: $ESM_DIR"
  git clone "$ESM_URL" "$ESM_DIR"
fi

# Optional convenience: keep app/esm pointing to canonical root esm
mkdir -p "$CLEAN_DIR/app"
ln -sfn ../esm "$CLEAN_DIR/app/esm"

# --- Pretrained weights download + normalization ---
if [[ "$DOWNLOAD_PRETRAINED" == "1" ]]; then
  echo
  echo "Downloading CLEAN pretrained bundle into canonical: $CLEAN_DIR/data/pretrained"

  # If already present (expected files), skip download
  if [[ -f "$CLEAN_DIR/data/pretrained/split100.pth" && -f "$CLEAN_DIR/data/pretrained/gmm_ensumble.pkl" ]]; then
    echo "Pretrained weights already present (split100.pth + gmm_ensumble.pkl). Skipping download."
  else
    # Clean any previous messy extraction
    rm -f "$CLEAN_DIR/data/pretrained/pretrained.zip" || true

    tmp_zip="$CLEAN_DIR/data/pretrained/pretrained.zip"
    tmp_unpack="$CLEAN_DIR/data/pretrained/_unpack_tmp"
    rm -rf "$tmp_unpack" || true
    mkdir -p "$tmp_unpack"

    echo "Downloading to: $tmp_zip"
    gdown --fuzzy "$CLEAN_PRETRAIN_FUZZY_URL" -O "$tmp_zip"

    echo "Unzipping into temp dir..."
    unzip -o "$tmp_zip" -d "$tmp_unpack" >/dev/null

    # Find the payload directory (zip may contain a folder)
    # Grab the first directory that contains split100.pth OR any .pth
    payload_dir=""
    if [[ -f "$tmp_unpack/split100.pth" ]]; then
      payload_dir="$tmp_unpack"
    else
      # search for split100.pth
      found="$(find "$tmp_unpack" -maxdepth 4 -name "split100.pth" -print | head -n 1 || true)"
      if [[ -n "$found" ]]; then
        payload_dir="$(dirname "$found")"
      else
        # fallback: directory containing any .pth
        found_any="$(find "$tmp_unpack" -maxdepth 4 -name "*.pth" -print | head -n 1 || true)"
        if [[ -n "$found_any" ]]; then
          payload_dir="$(dirname "$found_any")"
        fi
      fi
    fi

    if [[ -z "$payload_dir" ]]; then
      echo "ERROR: Could not locate pretrained .pth files after unzip."
      echo "Contents of $tmp_unpack:"
      find "$tmp_unpack" -maxdepth 3 -print
      exit 1
    fi

    echo "Using pretrained payload dir: $payload_dir"

    # Move/copy canonical expected files into $CLEAN_DIR/data/pretrained
    # (Use cp -n to avoid overwriting if some exist)
    for f in split100.pth split70.pth gmm_ensumble.pkl 100.pt 70.pt; do
      if [[ -f "$payload_dir/$f" ]]; then
        cp -f "$payload_dir/$f" "$CLEAN_DIR/data/pretrained/$f"
      fi
    done

    # sanity check
    if [[ ! -f "$CLEAN_DIR/data/pretrained/split100.pth" ]]; then
      echo "ERROR: split100.pth not found after extraction."
      echo "Available files in payload:"
      ls -lah "$payload_dir" || true
      exit 1
    fi

    echo "Done downloading + installing pretrained assets into $CLEAN_DIR/data/pretrained"
    rm -rf "$tmp_unpack" || true
  fi
else
  echo
  echo "Skipping pretrained download. To fetch pretrained weights later:"
  echo "  DOWNLOAD_PRETRAINED=1 bash scripts/setup-clean.sh"
fi

# --- Final preflight checks for reproducibility ---
echo
echo "Preflight checks..."
if [[ ! -f "$CLEAN_DIR/esm/scripts/extract.py" ]]; then
  echo "ERROR: Missing $CLEAN_DIR/esm/scripts/extract.py (ESM not set up correctly)."
  exit 1
fi
if [[ ! -f "$CLEAN_DIR/data/pretrained/split100.pth" ]]; then
  echo "ERROR: Missing $CLEAN_DIR/data/pretrained/split100.pth (pretrained weights not installed)."
  exit 1
fi

# Convenience symlink (matches p2rank style)
ln -sfn "$CLEAN_DIR" "$VENDOR_DIR/clean"

echo
echo "Done."
echo "CLEAN dir: $CLEAN_DIR"
