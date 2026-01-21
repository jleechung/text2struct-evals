#!/usr/bin/env bash
set -euo pipefail

# Configuration
ENV_NAME="text2struct-clean"
RUN_NAME="testrun"
MANIFEST="test_data/manifest.csv"
BATCH_SIZE=8
OVERWRITE=true  # true|false

# CLEAN knobs (not setup-determined)
MAX_EC=10
CHAIN_POLICY="longest"  # longest|first|all
MAX_LEN=1022

# Execution
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Initialize conda first."
  exit 1
fi

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

if [[ -f "$REPO_ROOT/$MANIFEST" ]]; then
  MANIFEST="$REPO_ROOT/$MANIFEST"
fi

# setup-clean.sh installs here
CLEAN_DIR="$REPO_ROOT/eval-clean/vendor/CLEAN"

extra=()
if [[ "$OVERWRITE" == "true" ]]; then
  extra+=(--overwrite)
fi

python eval-clean/run_clean.py \
  --manifest "$MANIFEST" \
  --run_name "$RUN_NAME" \
  --output_root "$REPO_ROOT/results" \
  --log_root "$REPO_ROOT/logs" \
  --batch_size "$BATCH_SIZE" \
  --clean_dir "$CLEAN_DIR" \
  --max_ec "$MAX_EC" \
  --chain_policy "$CHAIN_POLICY" \
  --max_len "$MAX_LEN" \
  "${extra[@]}"
