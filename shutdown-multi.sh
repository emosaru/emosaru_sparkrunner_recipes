#!/usr/bin/env bash
# Shut down the multi-model stack: stop the proxy, then stop each model.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPES_DIR="$REPO_DIR/recipes"

RECIPES=(
  "qwen3.6-35b-a3b-fp8-multi"
  "qwen3.6-27b-prismaquant-multi"
  "qwen3-1.7b-fp8-multi"
  "qwen3-embedding-0.6B-multi"
)

echo "▶ Stopping sparkrun proxy…"
sparkrun proxy stop || echo "  (proxy already stopped or not running)"

echo
echo "▶ Stopping models…"
for recipe in "${RECIPES[@]}"; do
  recipe_path="$RECIPES_DIR/${recipe}.yaml"
  echo "  • $recipe"
  sparkrun stop "$recipe_path" || echo "    (not running)"
done

echo
echo "✓ Stack down. Verify with: sparkrun status"
