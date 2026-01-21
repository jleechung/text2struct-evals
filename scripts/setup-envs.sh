#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Initialize conda first."
  exit 1
fi

# conda activation scripts can reference unset vars; avoid nounset issues
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u

shopt -s nullglob
ENV_FILES=(envs/*.yml envs/*.yaml)
shopt -u nullglob

if [[ ${#ENV_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No env files found under envs/ (*.yml or *.yaml)."
  exit 1
fi

echo "Found ${#ENV_FILES[@]} env file(s):"
printf "  - %s\n" "${ENV_FILES[@]}"
echo

# Helper: check if env exists
env_exists() {
  local env_name="$1"
  conda env list | awk '{print $1}' | grep -qx "$env_name"
}

for f in "${ENV_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    continue
  fi

  env_name="$(grep -E '^[[:space:]]*name:[[:space:]]*' "$f" | head -n 1 | sed -E 's/^[[:space:]]*name:[[:space:]]*//')"
  if [[ -z "$env_name" ]]; then
    echo "ERROR: Could not parse env name from $f (missing 'name:')."
    exit 1
  fi

  echo "==> $f  (name: $env_name)"

  if env_exists "$env_name"; then
    echo "Updating existing env: $env_name"
    conda env update -f "$f" --prune
  else
    echo "Creating env: $env_name"
    conda env create -f "$f"
  fi

  echo
done

echo "Done."
