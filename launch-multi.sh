#!/usr/bin/env bash
# Launch the 4-model multi-tenant stack on a single Spark, then start the proxy.
# Recipes co-budgeted to share unified memory: 35B-A3B (0.42) + 27B (0.40) + 1.7B (0.04) + embed (0.07) ≈ 0.93.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPES_DIR="$REPO_DIR/recipes"

PROXY_PORT="${PROXY_PORT:-4000}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"

RECIPES=(
  "qwen3-embedding-0.6B-multi"   # smallest first — fastest to come up, frees the foreground sooner
  "qwen3-1.7b-fp8-multi"
  "qwen3.6-27b-fp8-multi"
  "qwen3.6-35b-a3b-fp8-multi"
)

echo "▶ Launching ${#RECIPES[@]} models via sparkrun (detached)…"
for recipe in "${RECIPES[@]}"; do
  recipe_path="$RECIPES_DIR/${recipe}.yaml"
  if [[ ! -f "$recipe_path" ]]; then
    echo "  ✗ recipe not found: $recipe_path" >&2
    exit 1
  fi
  echo "  • $recipe"
  sparkrun run "$recipe_path" --ensure --no-follow
done

echo
echo "▶ Waiting for endpoints to report ready…"
declare -A PORTS=(
  [qwen3-embedding-0.6B-multi]=8001
  [qwen3-1.7b-fp8-multi]=8002
  [qwen3.6-27b-fp8-multi]=8003
  [qwen3.6-35b-a3b-fp8-multi]=8000
)
for recipe in "${RECIPES[@]}"; do
  port="${PORTS[$recipe]}"
  printf "  • %s (:%s) " "$recipe" "$port"
  # Up to 30 minutes for the big models to load weights.
  for _ in $(seq 1 360); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "ready"
      break
    fi
    printf "."
    sleep 5
  done
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
