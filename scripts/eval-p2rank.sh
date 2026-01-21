#!/usr/bin/env bash
set -euo pipefail

# Configuration
ENV_NAME="text2struct-p2rank"
RUN_NAME="testrun"
MANIFEST="test_data/manifest.csv"
BATCH_SIZE=8
OVERWRITE=true  # true|false

# P2Rank knobs (not setup-determined)
THREADS=1
PROFILE="alphafold"
VISUALIZATIONS=false  # true|false
TOP_K=5
TOP_RES_K=50

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

# setup-p2rank.sh installs here
P2RANK_BIN="$REPO_ROOT/eval-p2rank/vendor/p2rank/prank"

extra=()
if [[ "$OVERWRITE" == "true" ]]; then
  extra+=(--overwrite)
fi
if [[ "$VISUALIZATIONS" == "true" ]]; then
  extra+=(--visualizations)
fi

python eval-p2rank/run_p2rank.py \
  --manifest "$MANIFEST" \
  --run_name "$RUN_NAME" \
  --output_root "$REPO_ROOT/results" \
  --log_root "$REPO_ROOT/logs" \
  --batch_size "$BATCH_SIZE" \
  --prank_bin "$P2RANK_BIN" \
  --threads "$THREADS" \
  --profile "$PROFILE" \
  --top_k "$TOP_K" \
  --top_res_k "$TOP_RES_K" \
  "${extra[@]}"
