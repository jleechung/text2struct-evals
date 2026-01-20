#!/usr/bin/env bash
set -euo pipefail

# Reproducible P2Rank setup:
# - downloads a pinned release tarball
# - extracts into eval-p2rank/vendor/p2rank_<version>/
# - creates/updates symlink eval-p2rank/vendor/p2rank -> p2rank_<version>
#
# Usage:
#   bash scripts/setup-p2rank.sh
#   P2RANK_VERSION=2.5 bash scripts/setup-p2rank.sh
#
# Assumes: curl, tar are available.

P2RANK_VERSION="${P2RANK_VERSION:-2.5.1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/eval-p2rank/vendor"

TARBALL="p2rank_${P2RANK_VERSION}.tar.gz"
URL="https://github.com/rdk/p2rank/releases/download/${P2RANK_VERSION}/${TARBALL}"

mkdir -p "$VENDOR_DIR"
cd "$VENDOR_DIR"

echo "== P2Rank setup =="
echo "REPO_ROOT: $REPO_ROOT"
echo "VENDOR_DIR: $VENDOR_DIR"
echo "VERSION:    $P2RANK_VERSION"
echo "URL:        $URL"
echo

# If already installed, do nothing (idempotent)
if [[ -d "p2rank_${P2RANK_VERSION}" ]]; then
  echo "Found existing directory: $VENDOR_DIR/p2rank_${P2RANK_VERSION}"
else
  echo "Downloading $TARBALL ..."
  curl -L --fail -o "$TARBALL" "$URL"

  echo "Extracting $TARBALL ..."
  tar -xzf "$TARBALL"
fi

# Create stable symlink
ln -sfn "p2rank_${P2RANK_VERSION}" p2rank

echo "Symlink set:"
ls -l "$VENDOR_DIR/p2rank"
echo

# Quick sanity: locate prank launcher
if [[ -x "$VENDOR_DIR/p2rank/prank" ]]; then
  echo "Found prank: $VENDOR_DIR/p2rank/prank"
  echo "Try: eval-p2rank/vendor/p2rank/prank -v"
else
  echo "WARNING: prank launcher not found at expected path:"
  echo "  $VENDOR_DIR/p2rank/prank"
  echo "Listing top-level contents:"
  ls -la "$VENDOR_DIR/p2rank" || true
fi

echo
echo "Done."
