#!/usr/bin/env bash
# Stop the co-located dual NVFP4-MTP quants launched by launch-qwen-dual-nvfp4-eugr.sh.
#
# Each model runs in its own named container with its own Ray cluster, so we stop
# each by name via eugr's `launch-cluster.sh --name <name> stop` (which stops the
# container on the head + every worker in EUGR_DIR/.env and tears down that model's
# Ray). Belt-and-suspenders direct docker stop on both nodes afterward.
#
# Usage:
#   ./stop-qwen-dual-nvfp4-eugr.sh            # stop both
#   ONLY=27b ./stop-qwen-dual-nvfp4-eugr.sh   # stop just one (35b|27b)

set -uo pipefail

EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME_35B="${NAME_35B:-vllm_qwen35}"
NAME_27B="${NAME_27B:-vllm_qwen27}"
ONLY="${ONLY:-both}"                          # both | 35b | 27b

stop_one() {
  local name=$1 label=$2
  echo "▶ stopping $label ($name) across the cluster…"
  if [[ -d "$EUGR_DIR" ]]; then
    ( cd "$EUGR_DIR" && ./launch-cluster.sh --name "$name" stop ) \
      || echo "  (launcher stop returned non-zero; falling back to direct docker stop)"
  else
    echo "  eugr repo not at $EUGR_DIR; falling back to direct docker stop"
  fi
  for host in "$HEAD" "$WORKER"; do
    ssh "$host" "docker stop $name >/dev/null 2>&1 && docker rm $name >/dev/null 2>&1" 2>/dev/null \
      && echo "  ✓ $host: $name stopped" \
      || echo "  · $host: $name already stopped"
  done
}

[[ "$ONLY" == "both" || "$ONLY" == "35b" ]] && stop_one "$NAME_35B" "Qwen3.6-35B-A3B-NVFP4-MTP"
[[ "$ONLY" == "both" || "$ONLY" == "27b" ]] && stop_one "$NAME_27B" "Qwen3.6-27B-NVFP4-MTP"

echo
echo "✓ dual NVFP4-MTP co-location stopped."
