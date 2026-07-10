#!/usr/bin/env bash
# Stop the 122B FP8 stack: proxy first, then 1.7B, then the cluster.
# Counterpart to launch-122b-fp8-stack.sh.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "▶ stopping sparkrun proxy…"
sparkrun proxy stop || true

stop_recipe() {
  local recipe=$1 name=$2
  echo "▶ stopping $name…"
  sparkrun stop "$recipe" || echo "  (already stopped or not found)"
}

stop_recipe "$REPO_DIR/recipes/qwen3-1.7b-fp8-multi.yaml"              "1.7B classifier"
stop_recipe "$REPO_DIR/recipes/qwen3.5-122b-fp8-cluster-nothink.yaml"  "122B-FP8"

echo
echo "✓ 122B FP8 stack stopped. Cluster state:"
sparkrun cluster status || true
