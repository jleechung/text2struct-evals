#!/usr/bin/env bash
set -euo pipefail

# Configuration
ENV_NAME="text2struct-dssp"
RUN_NAME="testrun"
MANIFEST="test_data/manifest.csv"
BATCH_SIZE=8
OVERWRITE=true  # true|false

# Execution
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Initialize conda first."
  exit 1
fi

# conda activate can reference unset vars; avoid nounset during activation
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

if [[ -f "$REPO_ROOT/$MANIFEST" ]]; then
  MANIFEST="$REPO_ROOT/$MANIFEST"
fi

extra=()
if [[ "$OVERWRITE" == "true" ]]; then
  extra+=(--overwrite)
fi

python eval-dssp/run_dssp.py \
  --manifest "$MANIFEST" \
  --run_name "$RUN_NAME" \
  --output_root "$REPO_ROOT/results" \
  --log_root "$REPO_ROOT/logs" \
  --batch_size "$BATCH_SIZE" \
  "${extra[@]}"
