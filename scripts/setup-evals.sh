#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# DSSP usually needs no setup script (just the conda env), so we skip it here.

echo "==> Setting up P2Rank"
bash scripts/setup-p2rank.sh

echo "==> Setting up GearNet / Proteina"
bash scripts/setup-gearnet.sh

echo "==> Setting up ThermoMPNN"
bash scripts/setup-thermompnn.sh

echo "==> Setting up CLEAN"
bash scripts/setup-clean.sh

echo "Done."
