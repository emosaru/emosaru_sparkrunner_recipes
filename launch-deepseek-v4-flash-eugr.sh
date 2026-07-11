#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash (eugr's stock recipe) directly via eugr's
# spark-vllm-docker tooling — NOT via sparkrun.
#
# Same rationale as launch-qwen36-35b-nvfp4-eugr.sh: run eugr's own recipe
# through eugr's scripts + a fresh `eugr/spark-vllm:latest` image so the runtime
# and checkpoint stay in sync. See memory/eugr_direct_container_workflow.md.
#
# DeepSeek-V4-Flash is cluster_only (needs both Sparks) and heavy (util 0.8,
# max_model_len 500000) — it claims essentially the whole cluster on port 8000.
# Stop any other model on the cluster (e.g. the qwen nvfp4 instance) first.
#
# eugr documents DSV4F with --no-ray (PyTorch distributed backend), which also
# fits full context better; run-recipe strips the recipe's ray backend line in
# that mode. --no-ray is the default here; pass RAY_FLAG=--ray to override.
#
# Usage:
#   ./launch-deepseek-v4-flash-eugr.sh                 # launch (daemon) + wait ready
#   REFRESH_IMAGE=1 ./launch-deepseek-v4-flash-eugr.sh # pull+distribute fresh image first
#   DOWNLOAD_MODEL=1 ./launch-deepseek-v4-flash-eugr.sh# hf-download + distribute model first
#   RAY_FLAG=--ray ./launch-deepseek-v4-flash-eugr.sh  # force Ray backend instead of no-ray

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
RECIPE="${RECIPE:-deepseek-v4-flash}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
PORT="${PORT:-8000}"
RAY_FLAG="${RAY_FLAG:---no-ray}"     # eugr's documented/default backend for DSV4F
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

if [[ ! -d "$EUGR_DIR" ]]; then
  echo "✗ eugr repo not found at $EUGR_DIR" >&2
  echo "  git clone https://github.com/eugr/spark-vllm-docker.git $EUGR_DIR" >&2
  exit 1
fi

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

cd "$EUGR_DIR"

if [[ "${REFRESH_IMAGE:-0}" == "1" ]]; then
  echo "▶ refreshing eugr/spark-vllm:latest and distributing to $WORKER…"
  ./build-and-copy.sh -c "$WORKER"
fi

if [[ "${DOWNLOAD_MODEL:-0}" == "1" ]]; then
  echo "▶ downloading $MODEL and distributing across the cluster…"
  ./hf-download.sh "$MODEL" -c
fi

echo "▶ launching $RECIPE (cluster, $RAY_FLAG, daemon) on $HEAD + $WORKER…"
# -d = daemon: run-recipe returns after start; we poll the endpoint below.
./run-recipe.sh "$RECIPE" $RAY_FLAG -d "$@"

wait_ready "$HEAD" "$PORT" "DeepSeek-V4-Flash" "$READY_TIMEOUT"

echo
echo "✓ up. Endpoint: http://$HEAD:$PORT/v1"
echo "  logs:  ssh $HEAD 'docker logs -f vllm_node'"
echo "  stop:  $SCRIPT_DIR/stop-deepseek-v4-flash-eugr.sh"
