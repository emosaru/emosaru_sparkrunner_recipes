#!/usr/bin/env bash
# Stop the DeepSeek-V4-Flash standalone stack. Counterpart to launch-deepseek4-flash-stack.sh.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "▶ stopping sparkrun proxy…"
sparkrun proxy stop || true

echo "▶ stopping DeepSeek4-Flash-FP8-MTP…"
sparkrun stop "$REPO_DIR/recipes/deepseek4-flash-fp8-mtp-vllm.yaml" || echo "  (already stopped or not found)"

echo
echo "✓ deepseek4-flash stack stopped. Cluster state:"
sparkrun cluster status || true
