#!/usr/bin/env bash
set -euo pipefail

# Set once (as requested)
BASE_PATH="/gscratch/h2lab/jxlee"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Conda + env activation
# shellcheck disable=SC1091
source "$BASE_PATH/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-text2struct-dssp}"

# Run config
RUN_NAME="${RUN_NAME:-testrun}"
MANIFEST="${MANIFEST:-$REPO_ROOT/test_data/manifest.csv}"
BATCH_SIZE="${BATCH_SIZE:-10}"

python eval-dssp/run_dssp.py \
  --manifest "$MANIFEST" \
  --run_name "$RUN_NAME" \
  --output_root "$REPO_ROOT/results" \
  --log_root "$REPO_ROOT/logs" \
  --batch_size "$BATCH_SIZE" \
  --overwrite
