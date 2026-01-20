#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/eval-thermompnn/vendor"
THERMO_DIR="$VENDOR_DIR/ThermoMPNN"

# Pin a tag/commit for reproducibility (override if you want)
THERMO_REF="${THERMO_REF:-v1.0.0}"
THERMO_URL="${THERMO_URL:-https://github.com/Kuhlman-Lab/ThermoMPNN.git}"

mkdir -p "$VENDOR_DIR"

if [[ -d "$THERMO_DIR/.git" ]]; then
  echo "ThermoMPNN already cloned: $THERMO_DIR"
  echo "Updating to ref: $THERMO_REF"
  git -C "$THERMO_DIR" fetch --tags --force
  git -C "$THERMO_DIR" checkout "$THERMO_REF" || git -C "$THERMO_DIR" checkout -f FETCH_HEAD
else
  echo "Cloning ThermoMPNN into: $THERMO_DIR"
  git clone --depth 1 --branch "$THERMO_REF" "$THERMO_URL" "$THERMO_DIR" \
    || (git clone "$THERMO_URL" "$THERMO_DIR" && git -C "$THERMO_DIR" checkout "$THERMO_REF")
fi

echo
echo "ThermoMPNN ref:"
git -C "$THERMO_DIR" rev-parse HEAD
echo "Done."
