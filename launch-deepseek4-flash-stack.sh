#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash standalone across the 2-Spark cluster.
#
# DeepSeek-V4-Flash FP8 MTP as a standalone model on port 8000.
# TP=2 via direct vllm runtime (NCCL, one GPU per node).
#
# Memory budget per node (Spark unified ~121 GB, OS ~27 GB):
#
#   Node 0 — 192.168.1.236
#     OS                                 ~27 GB
#     DeepSeek4-Flash slice (TP=2)      ~96.8 GB  (util 0.80)
#     ──────────────────────────────────────────────────────
#     Claimed                           ~123.8 GB
#
#   Node 1 — 192.168.1.237
#     OS                                 ~27 GB
#     DeepSeek4-Flash slice (TP=2)      ~96.8 GB  (util 0.80)
#     ──────────────────────────────────────────────────────
#     Claimed                           ~123.8 GB
#
# KV cache: ~4x max_model_len (256k) with MLA FP8. Verify from vLLM startup
# log: "GPU blocks: N" — N × 16 / 256000 should be ≈ 4.

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

echo "▶ DeepSeek-V4-Flash FP8 MTP (TP=2, both nodes, port 8000)…"
sparkrun run "$REPO_DIR/recipes/deepseek4-flash-fp8-mtp-vllm.yaml" \
  --hosts "$BOTH" --port 8000 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE0" 8000 "DeepSeek4-Flash-FP8-MTP" 1800

echo "▶ starting sparkrun proxy (LiteLLM, auto-discovers endpoint)…"
sparkrun proxy start

echo
echo "✓ deepseek4-flash stack up. Cluster state:"
sparkrun cluster status || true
