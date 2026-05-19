#!/usr/bin/env bash
# Stop the unified cluster stack: proxy first, then each model.
# Counterpart to launch-unified-stack.sh.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "▶ stopping sparkrun proxy…"
sparkrun proxy stop || true

# `sparkrun stop <recipe>` returns non-zero if the workload isn't running;
# we treat that as a soft failure so the script keeps unwinding the rest.
stop_recipe() {
  local recipe=$1 name=$2
  echo "▶ stopping $name…"
  sparkrun stop "$recipe" || echo "  (already stopped or not found)"
}

stop_recipe "$REPO_DIR/recipes/qwen3-embedding-0.6B-multi.yaml"  "0.6B embedding"
stop_recipe "$REPO_DIR/recipes/qwen3-1.7b-fp8-multi.yaml"        "1.7B classifier"
stop_recipe "$REPO_DIR/recipes/qwen3.6-35b-a3b-fp8-mtp.yaml"     "35B-A3B-MTP"
stop_recipe "$REPO_DIR/recipes/qwen3.6-27b-nvfp4-mtp-cluster.yaml" "27B-NVFP4-MTP"

echo
echo "✓ unified stack stopped. Cluster state:"
sparkrun cluster status || true
