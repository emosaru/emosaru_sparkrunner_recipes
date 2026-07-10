#!/usr/bin/env bash
# Stop the gpt-oss standalone stack. Counterpart to launch-gpt-oss-stack.sh.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "▶ stopping sparkrun proxy…"
sparkrun proxy stop || true

echo "▶ stopping gpt-oss-120b-MXFP4…"
sparkrun stop "$REPO_DIR/recipes/gpt-oss-120b-mxfp4-cluster-ray.yaml" || echo "  (already stopped or not found)"

echo
echo "✓ gpt-oss stack stopped. Cluster state:"
sparkrun cluster status || true
