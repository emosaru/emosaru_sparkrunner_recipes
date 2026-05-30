#!/usr/bin/env bash
# Launch the Gemma 4 cluster stack across the 2-Spark cluster.
#
# Gemma 4 26B A4B MoE FP8 (on-the-fly quantization from BF16 safetensors) as
# the hermes main LLM (port 8000) and Qwen3.6-27B NVFP4-MTP as the opus model
# (port 8003). Both run TP=2 across both nodes via the ray backend (runtime:
# vllm-distributed) so their NCCL process groups are isolated — avoids the
# communicator collision that occurs with two concurrent vllm (direct) TP=2
# groups sharing the same GPU pair.
#
# NOTE: NVFP4 was attempted for Gemma4 but is not viable on the Spark (SM 12.1):
#   - FlashInfer/FlashAttn reject Gemma4's multimodal bidirectional attention
#   - Marlin MoE fails at runtime (K=352 has no valid thread config at 101KB shm)
#   - EMULATION backend OOMs during profiling at viable utilization levels
#   - flashinfer_b12x (native SM12.1) rejects GELU_TANH MoE activation
#   FP8 uses the TRITON backend which works correctly on SM 12.1.
#
# Memory budget per node (Spark unified ~121 GB, OS ~27 GB):
#
#   Node 0 — 192.168.1.236 (hermes + classifier + embedding)
#     OS                          ~27 GB
#     Gemma4 FP8 slice (TP=2)     ~48.4 GB  (util 0.40, ~13.7 GB weights + KV)
#     Qwen27B slice (TP=2)        ~38.7 GB  (util 0.32, MTP-3)
#     ──────────────────────────────────────
#     Claimed                     ~114.1 GB
#     Free                        ~6.9 GB
#
#   Node 1 — 192.168.1.237
#     OS                          ~27 GB
#     Gemma4 FP8 slice (TP=2)     ~48.4 GB  (util 0.40)
#     Qwen27B slice (TP=2)        ~38.7 GB  (util 0.32)
#     ──────────────────────────────────────
#     Claimed                     ~114.1 GB
#     Free                        ~6.9 GB
#
# Launch order: Qwen27B first (denser model, larger CUDA-graph spike), then
# Gemma4, then the single-node small models on node 0.

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

echo "▶ Qwen3.6-27B-NVFP4-MTP (TP=2, ray, both nodes, port 8003)…"
sparkrun run "$REPO_DIR/recipes/qwen3.6-27b-nvfp4-mtp-cluster-ray.yaml" \
  --hosts "$BOTH" --port 8003 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE0" 8003 "Qwen27B-NVFP4-MTP" 1800

echo "▶ Gemma4-26B-A4B-it FP8 (TP=2, ray, both nodes, port 8000)…"
sparkrun run "$REPO_DIR/recipes/gemma4-26b-a4b-fp8-cluster.yaml" \
  --hosts "$BOTH" --port 8000 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE0" 8000 "Gemma4-26B-A4B-FP8" 1800

echo "▶ starting sparkrun proxy (LiteLLM, auto-discovers all 2 endpoints)…"
sparkrun proxy start

echo
echo "✓ gemma4 stack up. Cluster state:"
sparkrun cluster status || true
