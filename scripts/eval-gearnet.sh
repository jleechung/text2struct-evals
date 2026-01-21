#!/usr/bin/env bash
set -euo pipefail

# Configuration
ENV_NAME="text2struct-gearnet"
RUN_NAME="testrun"
MANIFEST="test_data/manifest.csv"
BATCH_SIZE=8
OVERWRITE=true  # true|false

# GearNet knobs (not setup-determined)
NUM_WORKERS=8
TOP_K=5
BB_MODEL=false  # true|false (backbone model vs CA-only)

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

# setup-gearnet.sh installs Proteina additional files here
export DATA_PATH="$REPO_ROOT/eval-gearnet/cache/proteina_additional_files"

extra=()
if [[ "$OVERWRITE" == "true" ]]; then
  extra+=(--overwrite)
fi
if [[ "$BB_MODEL" == "true" ]]; then
  extra+=(--bb_model)
fi

python eval-gearnet/run_gearnet.py \
  --manifest "$MANIFEST" \
  --run_name "$RUN_NAME" \
  --output_root "$REPO_ROOT/results" \
  --log_root "$REPO_ROOT/logs" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --top_k "$TOP_K" \
  "${extra[@]}"
