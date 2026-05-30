#!/usr/bin/env bash
# Stop the Gemma 4 cluster stack. Counterpart to launch-gemma4-stack.sh.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "▶ stopping sparkrun proxy…"
sparkrun proxy stop || true

stop_recipe() {
  local recipe=$1 name=$2
  echo "▶ stopping $name…"
  sparkrun stop "$recipe" || echo "  (already stopped or not found)"
}

stop_recipe "$REPO_DIR/recipes/gemma4-26b-a4b-nvfp4-cluster.yaml"          "Gemma4-26B-A4B"
stop_recipe "$REPO_DIR/recipes/qwen3.6-27b-nvfp4-mtp-cluster-ray.yaml"    "Qwen27B-NVFP4-MTP"

echo
echo "✓ gemma4 stack stopped. Cluster state:"
sparkrun cluster status || true
