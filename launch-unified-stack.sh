#!/usr/bin/env bash
# Launch the unified cluster stack across the 2-Spark cluster.
#
# Replaces the agentic-coding + llm-support-bot stack files with a single
# script that drives `sparkrun run` directly, so we can pin single-node models
# to specific hosts via --hosts (something the stackctl stack file format does
# not currently expose).
#
# Topology and per-node memory budget (Spark unified ~121 GB, OS ~27 GB).
# Cluster recipes run at FULL 256k context (max_model_len=262144); the
# concession is that ComfyUI on node 2 is squeezed (see below) — that was
# the explicit user trade-off.
#
#   Node 1 — 192.168.1.236 (this machine: hermes + ad-hoc + classifier)
#     OS                        ~27 GB
#     Hermes agent              ~5 GB   (external, steady)
#     27B-NVFP4-MTP slice (TP=2) ~39 GB  (util 0.32, this node's half)
#     35B-A3B-MTP slice (TP=2)  ~29 GB  (util 0.24, this node's half)
#     1.7B classifier           ~6 GB   (util 0.05, pinned here)
#     ────────────────────────────────
#     Claimed                   ~106 GB
#     Free for ad-hoc loading   ~15 GB  (room for a ~3-7B FP8 spin)
#
#   Node 2 — 192.168.1.237 (ComfyUI host + embedding)
#     OS                        ~27 GB
#     27B-NVFP4-MTP slice (TP=2) ~39 GB
#     35B-A3B-MTP slice (TP=2)  ~29 GB
#     0.6B embedding            ~8.5 GB (util 0.07, pinned here)
#     ────────────────────────────────
#     Claimed                   ~103.5 GB
#     Budget for ComfyUI        ~17.5 GB (down from ~25 GB pre-cluster)
#
# Launch order matters: 27B is dense and has the largest CUDA-graph capture
# spike — load it first while memory is still cold. 35B-A3B (MoE) is next.
# Tiny single-node models last; they're cheap.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODE1=192.168.1.236   # main / hermes
NODE2=192.168.1.237   # ComfyUI host
BOTH="$NODE1,$NODE2"

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

echo "▶ 27B-NVFP4-MTP (TP=2, both nodes, port 8003)…"
sparkrun run "$REPO_DIR/recipes/qwen3.6-27b-nvfp4-mtp-cluster.yaml" \
  --hosts "$BOTH" --port 8003 --ensure --no-follow
wait_ready "$NODE1" 8003 "27B-NVFP4-MTP" 1800

echo "▶ 35B-A3B-MTP (TP=2, both nodes, port 8000)…"
sparkrun run "$REPO_DIR/recipes/qwen3.6-35b-a3b-fp8-mtp.yaml" \
  --hosts "$BOTH" --port 8000 --ensure --no-follow
wait_ready "$NODE1" 8000 "35B-A3B-MTP" 1800

echo "▶ 1.7B classifier (pinned to $NODE1, port 8002)…"
sparkrun run "$REPO_DIR/recipes/qwen3-1.7b-fp8-multi.yaml" \
  --hosts "$NODE1" --port 8002 --ensure --no-follow
wait_ready "$NODE1" 8002 "1.7B classifier" 900

echo "▶ 0.6B embedding (pinned to $NODE2, port 8001)…"
sparkrun run "$REPO_DIR/recipes/qwen3-embedding-0.6B-multi.yaml" \
  --hosts "$NODE2" --port 8001 --ensure --no-follow
wait_ready "$NODE2" 8001 "0.6B embedding" 600

echo "▶ starting sparkrun proxy (LiteLLM, auto-discovers all 4 endpoints)…"
sparkrun proxy start

echo
echo "✓ unified stack up. Cluster state:"
sparkrun cluster status || true
