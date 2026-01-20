#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/gscratch/h2lab/jxlee"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# conda activation can reference unset vars in activate.d hooks; disable nounset temporarily
set +u
source "$BASE_PATH/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-text2struct-thermompnn}"
set -u

RUN_NAME="${RUN_NAME:-testrun}"
MANIFEST="${MANIFEST:-$REPO_ROOT/test_data/manifest.csv}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OVERWRITE="${OVERWRITE:-0}"
GPU="${GPU:-}"   # e.g. GPU=0

extra=()
if [[ "$OVERWRITE" == "1" ]]; then
  extra+=(--overwrite)
fi
if [[ -n "$GPU" ]]; then
  extra+=(--gpu "$GPU")
fi

python eval-thermompnn/run_thermompnn.py \
  --manifest "$MANIFEST" \
  --run_name "$RUN_NAME" \
  --output_root "$REPO_ROOT/results" \
  --log_root "$REPO_ROOT/logs" \
  --batch_size "$BATCH_SIZE" \
  "${extra[@]}"
