#!/usr/bin/env bash
# Launch the 4-model multi-tenant stack on a single Spark, then start the proxy.
#
# vLLM's --gpu-memory-utilization is "fraction of TOTAL device memory the
# engine must leave free system-wide" — NOT this instance's share. So with
# co-tenants, each recipe sets a CUMULATIVE value = sum of all prior launches
# + this model's own share. This requires strict serial loading: each model
# must be fully ready (its memory claim settled) before the next starts.
#
# Launch order is LARGEST → SMALLEST so each big model loads when the most
# contiguous memory is free (small co-tenants can later squeeze into fragments).
#
# Cumulative budget (Spark ~119 GB unified, ~12% safety margin):
#   1. 35B-A3B FP8 256k      cumulative 0.43   (own 0.43, ~51 GB)
#   2. 27B prismaquant 128k  cumulative 0.78   (own 0.35, ~42 GB)
#   3. 1.7B classifier       cumulative 0.82   (own 0.04,  ~5 GB)
#   4. embedding 0.6B        cumulative 0.88   (own 0.06,  ~7 GB)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPES_DIR="$REPO_DIR/recipes"

PROXY_PORT="${PROXY_PORT:-4000}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1800}"   # 30 min per model

# (recipe-name, port) — strict launch order; cumulative shares depend on it.
STAGES=(
  "qwen3.6-35b-a3b-fp8-multi 8000"
  "qwen3.6-27b-prismaquant-multi 8003"
  "qwen3-1.7b-fp8-multi 8002"
  "qwen3-embedding-0.6B-multi 8001"
)

wait_ready() {
  local name=$1 port=$2 deadline=$(( SECONDS + READY_TIMEOUT_SECONDS ))
  printf "  waiting for :%s " "$port"
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "ready"
      return 0
    fi
    # Detect crash: if no vllm process is left in this recipe's container, abort early.
    if ! docker ps --format '{{.Names}}' | xargs -I{} docker exec {} pgrep -f "port ${port}" >/dev/null 2>&1; then
      :   # best-effort; keep polling — pgrep across all containers is noisy
    fi
    printf "."
    sleep 5
  done
  echo
  echo "  ✗ ${name} never came up on :${port} within ${READY_TIMEOUT_SECONDS}s" >&2
  echo "  inspect: docker ps; docker exec <container> tail -100 /tmp/sparkrun_serve.log" >&2
  return 1
}

for stage in "${STAGES[@]}"; do
  read -r recipe port <<<"$stage"
  recipe_path="$RECIPES_DIR/${recipe}.yaml"
  [[ -f "$recipe_path" ]] || { echo "✗ recipe not found: $recipe_path" >&2; exit 1; }

  echo
  echo "▶ [$recipe] launching (port $port)…"
  sparkrun run "$recipe_path" --ensure --no-follow
  wait_ready "$recipe" "$port"
done

echo
echo "▶ Starting sparkrun proxy on ${PROXY_HOST}:${PROXY_PORT}…"
sparkrun proxy start --host "$PROXY_HOST" --port "$PROXY_PORT"

echo
echo "✓ Stack up."
echo "  Proxy:     http://${PROXY_HOST}:${PROXY_PORT}"
echo "  Status:    sparkrun status"
echo "  Models:    sparkrun proxy models"
echo "  Shutdown:  $REPO_DIR/shutdown-multi.sh"
