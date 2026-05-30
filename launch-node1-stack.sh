#!/usr/bin/env bash
# Launch a self-contained LLM stack on node 1 (192.168.1.237) only.
# Three endpoints, no 27B slice, no NCCL.
#
# Counterpart to launch-unified-stack.sh for when node 1 runs independently.
#
#   Node 1 — 192.168.1.237
#     OS                          ~27 GB
#     35B-A3B-FP8-MTP (TP=1)     ~77.4 GB  (util 0.64, 128k ctx; 256k may work)
#     1.7B classifier             ~6 GB     (util 0.05)
#     0.6B embedding              ~8.5 GB   (util 0.07)
#     ─────────────────────────────────────
#     Claimed                     ~119 GB
#
# Launch order: 35B first (largest CUDA-graph spike), then small models.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODE1=192.168.1.237

wait_ready() {
  local host=$1 port=$2 name=$3 timeout=${4:-1800}
  local url="http://$host:$port/v1/models"
  local deadline=$((SECONDS + timeout))
  echo "  ⏳ waiting for $name at $url (timeout ${timeout}s)…"
  while (( SECONDS < deadline )); do
    if curl -fsS -m 3 "$url" >/dev/null 2>&1; then
      echo "  ✓ $name ready"
      return 0
    fi
    sleep 5
  done
  echo "  ✗ $name not ready after ${timeout}s" >&2
  return 1
}

VLLM_CACHE_MOUNT="-v /home/mebell/.cache/vllm:/tmp/.cache/vllm -v /home/mebell/.cache/flashinfer:/tmp/.cache/flashinfer"

echo "▶ 35B-A3B-FP8-MTP (TP=1, node 1 only, port 8000)…"
sparkrun run "$REPO_DIR/recipes/qwen3.6-35b-a3b-fp8-mtp-node1.yaml" \
  --hosts "$NODE1" --port 8000 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE1" 8000 "35B-A3B-FP8-MTP" 1800

echo "▶ 1.7B classifier (node 1, port 8002)…"
sparkrun run "$REPO_DIR/recipes/qwen3-1.7b-fp8-multi.yaml" \
  --hosts "$NODE1" --port 8002 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE1" 8002 "1.7B classifier" 900

echo "▶ 0.6B embedding (node 1, port 8001)…"
sparkrun run "$REPO_DIR/recipes/qwen3-embedding-0.6B-multi.yaml" \
  --hosts "$NODE1" --port 8001 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE1" 8001 "0.6B embedding" 600

echo "▶ starting sparkrun proxy (LiteLLM, auto-discovers all 3 endpoints)…"
sparkrun proxy start

echo
echo "✓ node 1 stack up. Cluster state:"
sparkrun cluster status || true
