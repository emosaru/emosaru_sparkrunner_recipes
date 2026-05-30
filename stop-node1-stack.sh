#!/usr/bin/env bash
# Stop the node 1 stack: proxy first, then each model.
# Counterpart to launch-node1-stack.sh.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "▶ stopping sparkrun proxy…"
sparkrun proxy stop || true

stop_recipe() {
  local recipe=$1 name=$2
  echo "▶ stopping $name…"
  sparkrun stop "$recipe" || echo "  (already stopped or not found)"
}

stop_recipe "$REPO_DIR/recipes/qwen3-embedding-0.6B-multi.yaml"        "0.6B embedding"
stop_recipe "$REPO_DIR/recipes/qwen3-1.7b-fp8-multi.yaml"              "1.7B classifier"
stop_recipe "$REPO_DIR/recipes/qwen3.6-35b-a3b-fp8-mtp-node1.yaml"    "35B-A3B-FP8-MTP"

echo
echo "✓ node 1 stack stopped. Cluster state:"
sparkrun cluster status || true
