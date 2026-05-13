#!/usr/bin/env bash
# Launch the alternative 122B agentic-coding stack: Qwen3.5-122B-A10B NVFP4
# (TRT-LLM) + Nemotron-3-Nano 30B-A3B NVFP4 on a single Spark, then start the
# managed proxy on both ports.
#
# Per-instance memory targets on Spark (~121 GB unified):
#   1. 122B NVFP4 trtllm   util 0.60  ~73 GB  — launched first
#   2. Nemotron-3-Nano     util 0.20  ~24 GB
# Sum 0.80, ~97 GB models + ~25 GB framework overhead → ~122 GB used.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$REPO_DIR/launcher/target/release/stackctl"

if [[ ! -x "$BIN" ]]; then
  echo "▶ building stackctl…"
  (cd "$REPO_DIR/launcher" && cargo build --release)
fi

exec "$BIN" up "$REPO_DIR/stacks/agentic-coding-122b.yaml" "$@"
