#!/usr/bin/env bash
# Launch the llm-support-bot multi-tenant stack (35B + 1.7B + 0.6B embedding)
# on a single Spark via stackctl. See stacks/llm-support-bot.yaml.
#
# Per-instance memory targets on Spark (~121 GB unified):
#   1. 35B-A3B FP8     util 0.36  ~43 GB
#   2. 1.7B classifier util 0.05  ~6 GB
#   3. embedding 0.6B  util 0.05  ~6 GB
# Sum ~0.46 → ~55 GB models; ~27 GB system/process overhead → ~82 GB total
# with comfortable headroom.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$REPO_DIR/launcher/target/release/stackctl"

if [[ ! -x "$BIN" ]]; then
  echo "▶ building stackctl…"
  (cd "$REPO_DIR/launcher" && cargo build --release)
fi

exec "$BIN" up "$REPO_DIR/stacks/llm-support-bot.yaml" "$@"
