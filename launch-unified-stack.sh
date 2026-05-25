#!/usr/bin/env bash
# Launch the unified cluster stack across the 2-Spark cluster.
#
# Replaces the agentic-coding + llm-support-bot stack files with a single
# script that drives `sparkrun run` directly, so we can pin single-node models
# to specific hosts via --hosts (something the stackctl stack file format does
# not currently expose).
#
# Topology and per-node memory budget (Spark unified ~121 GB, OS ~27 GB).
#
# Key motivation for this layout: the previous design ran both 27B and 35B as
# TP=2 clusters across both nodes. Two NCCL communicator groups sharing the
# same GPU pair caused NCCL errors under concurrent load. Moving the 35B to
# TP=1 (single-node) eliminates the NCCL group collision — only the 27B uses
# NCCL now.
#
#   Node 0 — 192.168.1.236 (this machine: hermes + classifier + embedding)
#     OS                         ~27 GB
#     Hermes agent               ~5 GB   (external, steady)
#     27B-NVFP4-MTP slice (TP=2) ~39 GB  (util 0.32, this node's half)
#     1.7B classifier            ~6 GB   (util 0.05, pinned here)
#     0.6B embedding             ~8.5 GB (util 0.07, pinned here)
#     ─────────────────────────────────
#     Claimed                    ~85.5 GB
#     Free for ad-hoc loading    ~35.5 GB
#
#   Node 1 — 192.168.1.237 (35B host — no ComfyUI, full budget given to 35B)
#     OS                         ~27 GB
#     27B-NVFP4-MTP slice (TP=2) ~39 GB  (util 0.32, this node's half)
#     35B-A3B-MTP (TP=1, solo)   ~56 GB  (util 0.46, 256k context; NVFP4 ~17.5 GB vs FP8 ~35 GB)
#     ─────────────────────────────────
#     Claimed                    ~122 GB  (full node)
#
# Launch order: 27B first (largest CUDA-graph spike, uses both nodes), then
# 35B single-node on node 1, then the two small models on node 0.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODE0=192.168.1.236   # node 0: main / hermes / classifier / embedding
NODE1=192.168.1.237   # node 1: ComfyUI host / 35B solo
BOTH="$NODE0,$NODE1"

# Wait until a model's /v1/models endpoint responds 2xx, or fail after timeout.
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

# Mount host-side caches so compile/JIT results survive container restarts.
# Both paths must exist on all nodes before launching (mkdir -p on each host).
# The flashinfer mount is required because Docker auto-creates /tmp/.cache as
# root when bind-mounting the vllm subdirectory, leaving the sibling
# /tmp/.cache/flashinfer unwritable by the mebell worker process (uid 1000).
VLLM_CACHE_MOUNT="-v /home/mebell/.cache/vllm:/tmp/.cache/vllm -v /home/mebell/.cache/flashinfer:/tmp/.cache/flashinfer"

echo "▶ 27B-NVFP4-MTP (TP=2, both nodes, port 8003)…"
sparkrun run "$REPO_DIR/recipes/qwen3.6-27b-nvfp4-mtp-cluster.yaml" \
  --hosts "$BOTH" --port 8003 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE0" 8003 "27B-NVFP4-MTP" 1800

echo "▶ 35B-A3B-MTP (TP=1, node 1 only, port 8000)…"
sparkrun run "$REPO_DIR/recipes/qwen3.6-35b-a3b-nvfp4-mtp-colocated.yaml" \
  --hosts "$NODE1" --port 8000 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE1" 8000 "35B-A3B-MTP" 1800

echo "▶ 1.7B classifier (pinned to $NODE0, port 8002)…"
sparkrun run "$REPO_DIR/recipes/qwen3-1.7b-fp8-multi.yaml" \
  --hosts "$NODE0" --port 8002 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE0" 8002 "1.7B classifier" 900

echo "▶ 0.6B embedding (pinned to $NODE0, port 8001)…"
sparkrun run "$REPO_DIR/recipes/qwen3-embedding-0.6B-multi.yaml" \
  --hosts "$NODE0" --port 8001 --ensure --no-follow \
  --executor-args "$VLLM_CACHE_MOUNT"
wait_ready "$NODE0" 8001 "0.6B embedding" 600

echo "▶ starting sparkrun proxy (LiteLLM, auto-discovers all 4 endpoints)…"
sparkrun proxy start

echo
echo "✓ unified stack up. Cluster state:"
sparkrun cluster status || true
