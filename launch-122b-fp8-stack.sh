#!/usr/bin/env bash
# Launch Qwen3.5-122B-A10B FP8 across both Sparks (TP=2, Ray) plus a 1.7B
# classifier pinned to node 1.
#
# Topology and per-node memory budget (Spark unified ~121 GB, OS ~27 GB):
#
#   Node 0 — 192.168.1.236
#     OS                             ~27 GB
#     122B-FP8 slice (TP=2, util=0.70) ~84.7 GB  (head node; API served here)
#     ──────────────────────────────────────────
#     Claimed                         ~111.7 GB
#     Free                              ~9.3 GB
#
#   Node 1 — 192.168.1.237
#     OS                             ~27 GB
#     122B-FP8 slice (TP=2, util=0.70) ~84.7 GB
#     1.7B classifier (util=0.05)       ~6 GB
#     ──────────────────────────────────────────
#     Claimed                         ~117.7 GB  (tight; comparable to node1-stack)
#     Free                              ~3.3 GB
#
# Launch order: 122B first (larger CUDA-graph spike across both nodes), then
# 1.7B on node 1 once the cluster is up and memory is settled.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODE0=192.168.1.236
NODE1=192.168.1.237
BOTH="$NODE0,$NODE1"

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

echo "▶ Qwen3.5-122B-FP8 (TP=2, Ray, both nodes, port 8000)…"
sparkrun run "$REPO_DIR/recipes/qwen3.5-122b-fp8-cluster-nothink.yaml" \
  --hosts "$BOTH" --port 8000 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE0" 8000 "122B-FP8" 2400

echo "▶ 1.7B classifier (pinned to node 1, port 8002)…"
sparkrun run "$REPO_DIR/recipes/qwen3-1.7b-fp8-multi.yaml" \
  --hosts "$NODE1" --port 8002 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE1" 8002 "1.7B classifier" 900

echo "▶ starting sparkrun proxy (LiteLLM, auto-discovers both endpoints)…"
sparkrun proxy start

echo
echo "✓ 122B FP8 stack up. Cluster state:"
sparkrun cluster status || true
