#!/usr/bin/env bash
# Launch the agentic-coding stack (35B + 27B) on a single Spark, start the
# managed LiteLLM proxy with our Claude Code aliases, then attach a TUI.
#
# All orchestration lives in the stackctl Rust launcher; the stack recipe is
# stacks/agentic-coding.yaml.
#
# Per-instance memory targets on Spark (~121 GB unified):
#   1. 27B prismaquant   util 0.40  ~48 GB  — launched first (dense, larger
#      CUDA-graph capture spike than the MoE 35B-A3B; needs all memory free
#      during load to fit).
#   2. 35B-A3B FP8       util 0.38  ~46 GB  — trimmed from 0.40; 27B's load
#      consumes ~74 GB once overhead is counted, leaving the 35B startup
#      check tight at util=0.40. Sum 0.78, under earlyoom's 80% line.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$REPO_DIR/launcher/target/release/stackctl"

if [[ ! -x "$BIN" ]]; then
  echo "▶ building stackctl…"
  (cd "$REPO_DIR/launcher" && cargo build --release)
fi

exec "$BIN" up "$REPO_DIR/stacks/agentic-coding.yaml" "$@"
