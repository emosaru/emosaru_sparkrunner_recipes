#!/usr/bin/env bash
# Launch Qwen3.6-35B-A3B-NVFP4 (eugr's stock recipe) directly via eugr's
# spark-vllm-docker tooling — NOT via sparkrun.
#
# Why this path instead of sparkrun: sparkrun ran the recipe against the local
# `vllm-node` image, which drifts stale and fails at weight load with
#   ValueError: no module or parameter named 'lm_head.input_scale'
# against the current nvidia/Qwen3.6-35B-A3B-NVFP4 checkpoint. Running through
# eugr's own scripts + a fresh `eugr/spark-vllm:latest` image keeps the runtime
# and checkpoint in sync. See memory/eugr_direct_container_workflow.md.
#
# TP=2 across both Sparks via Ray (recipe pins --distributed-executor-backend
# ray). Cluster nodes / interfaces come from EUGR_DIR/.env. Serves on port 8000.
#
# Usage:
#   ./launch-qwen36-35b-nvfp4-eugr.sh                 # launch (daemon) + wait ready
#   REFRESH_IMAGE=1 ./launch-qwen36-35b-nvfp4-eugr.sh # pull+distribute fresh image first
#   ./launch-qwen36-35b-nvfp4-eugr.sh --no-ray        # extra flags pass through to run-recipe.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
RECIPE="${RECIPE:-qwen3.6-35b-a3b-nvfp4}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
PORT="${PORT:-8000}"
RAY_FLAG="${RAY_FLAG:---ray}"        # recipe hardcodes ray backend; keep --ray unless overridden
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

echo "▶ launching $RECIPE (TP=2, Ray, daemon) on $HEAD + $WORKER…"
# -d = daemon: run-recipe returns after start; we poll the endpoint below.
./run-recipe.sh "$RECIPE" $RAY_FLAG -d "$@"

wait_ready "$HEAD" "$PORT" "Qwen3.6-35B-A3B-NVFP4" "$READY_TIMEOUT"

echo
echo "✓ up. Endpoint: http://$HEAD:$PORT/v1"
echo "  logs:  ssh $HEAD 'docker logs -f vllm_node'"
echo "  stop:  $SCRIPT_DIR/stop-qwen36-35b-nvfp4-eugr.sh"
