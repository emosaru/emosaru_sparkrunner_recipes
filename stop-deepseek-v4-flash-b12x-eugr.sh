#!/usr/bin/env bash
# Stop the eugr B12X DeepSeek-V4-Flash production stack.
# Counterpart to launch-deepseek-v4-flash-b12x-eugr.sh.
#
# Uses the same RoCE cluster config as the launcher so launch-cluster.sh resolves the
# SAME node list it started (eugr's own .env points at the management net).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
CONFIG="${CONFIG:-$SCRIPT_DIR/env/eugr-roce.env}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME="${NAME:-vllm_node}"

echo "▶ stopping $NAME across the cluster…"
if [[ -d "$EUGR_DIR" && -f "$CONFIG" ]]; then
  ( cd "$EUGR_DIR" && ./launch-cluster.sh --config "$CONFIG" --name "$NAME" stop ) \
    || echo "  (launcher stop returned non-zero; falling back to direct docker stop)"
else
  echo "  eugr repo or config missing; falling back to direct docker stop"
fi

# Belt-and-suspenders: the container must be gone on BOTH nodes, or the next launch
# fails on a stale name / leaked GPU memory.
for host in "$HEAD" "$WORKER"; do
  ssh -o BatchMode=yes "$host" "docker stop $NAME >/dev/null 2>&1 && docker rm $NAME >/dev/null 2>&1" 2>/dev/null \
    && echo "  ✓ $host: $NAME stopped" \
    || echo "  · $host: $NAME already stopped"
done

echo
for host in "$HEAD" "$WORKER"; do
  echo "  $host free: $(ssh -o BatchMode=yes "$host" "free -g | awk 'NR==2{print \$7\" GiB available\"}'" 2>/dev/null || echo '?')"
done
echo "✓ DeepSeek-V4-Flash (eugr B12X) stopped."
