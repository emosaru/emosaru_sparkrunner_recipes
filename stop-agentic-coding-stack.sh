#!/usr/bin/env bash
# Stop the agentic-coding stack via stackctl: proxy down, then each model.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$REPO_DIR/launcher/target/release/stackctl"

if [[ ! -x "$BIN" ]]; then
  echo "▶ building stackctl…"
  (cd "$REPO_DIR/launcher" && cargo build --release)
fi

exec "$BIN" down "$REPO_DIR/stacks/agentic-coding.yaml" "$@"
