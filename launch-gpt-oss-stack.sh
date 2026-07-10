#!/usr/bin/env bash
# Launch gpt-oss-120b standalone across the 2-Spark cluster.
#
# gpt-oss-120b MXFP4 MoE (native quantization from OpenAI) as a standalone
# model on port 8001. TP=2 via the ray backend (runtime: vllm-distributed).
# Eagle3 speculative decoding (nvidia/gpt-oss-120b-Eagle3-v3, 7 tokens/step).
#
# Memory budget per node (Spark unified ~121 GB, OS ~27 GB):
#
#   Node 0 — 192.168.1.236
#     OS                           ~27 GB
#     gpt-oss-120b slice (TP=2)   ~66.6 GB  (util 0.55, ~33 GB weights + KV)
#     ──────────────────────────────────────
#     Claimed                      ~93.6 GB
#     Free                         ~27.4 GB
#
#   Node 1 — 192.168.1.237
#     OS                           ~27 GB
#     gpt-oss-120b slice (TP=2)   ~66.6 GB  (util 0.55)
#     ──────────────────────────────────────
#     Claimed                      ~93.6 GB
#     Free                         ~27.4 GB
#
# SM121 note: VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 is set in the recipe to
# route MoE matmuls through FlashInfer and avoid the SM80 Marlin fallback that
# produces null content on first token (vllm#37030). Verify on first deploy.

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

echo "▶ gpt-oss-120b MXFP4 (TP=2, ray, both nodes, port 8001)…"
sparkrun run "$REPO_DIR/recipes/gpt-oss-120b-mxfp4-cluster-ray.yaml" \
  --hosts "$BOTH" --port 8001 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE0" 8001 "gpt-oss-120b-MXFP4" 1800

echo "▶ starting sparkrun proxy (LiteLLM, auto-discovers endpoint)…"
sparkrun proxy start

echo
echo "✓ gpt-oss stack up. Cluster state:"
sparkrun cluster status || true
