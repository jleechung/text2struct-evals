#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/gscratch/h2lab/jxlee"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# conda
source "$BASE_PATH/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-text2struct-p2rank}"

RUN_NAME="${RUN_NAME:-testrun}"
MANIFEST="${MANIFEST:-$REPO_ROOT/test_data/manifest.csv}"
BATCH_SIZE="${BATCH_SIZE:-10}"
OVERWRITE="${OVERWRITE:-0}"

P2RANK_BIN="${P2RANK_BIN:-$REPO_ROOT/eval-p2rank/vendor/p2rank/prank}"

extra=()
if [[ "$OVERWRITE" == "1" ]]; then
  extra+=(--overwrite)
fi

python eval-p2rank/run_p2rank.py \
  --manifest "$MANIFEST" \
  --run_name "$RUN_NAME" \
  --output_root "$REPO_ROOT/results" \
  --log_root "$REPO_ROOT/logs" \
  --batch_size "$BATCH_SIZE" \
  --prank_bin "$P2RANK_BIN" \
  "${extra[@]}"
