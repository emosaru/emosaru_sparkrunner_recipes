#!/usr/bin/env bash
# Stop the DeepSeek-V4-Flash-Vision-Exp TRIAL stack.
# Counterpart to launch-deepseek-v4-flash-vision-exp-b12x-eugr.sh.
#
# Uses env/eugr-roce-vision.env so launch-cluster.sh resolves the SAME node list and
# the SAME container name ('vllm_node_vision') it started. This script will NOT touch
# the production container 'vllm_node' — that is the whole point of the separate name.
#
# Usage:
#   ./stop-deepseek-v4-flash-vision-exp-b12x-eugr.sh                 # stop the trial
#   RESTORE_PROD=1 ./stop-deepseek-v4-flash-vision-exp-b12x-eugr.sh  # + relaunch production

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
CONFIG="${CONFIG:-$SCRIPT_DIR/env/eugr-roce-vision.env}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME="${NAME:-vllm_node_vision}"

echo "▶ stopping $NAME across the cluster…"
if [[ -d "$EUGR_DIR" && -f "$CONFIG" ]]; then
  ( cd "$EUGR_DIR" && ./launch-cluster.sh --config "$CONFIG" --name "$NAME" stop ) \
    || echo "  (launcher stop returned non-zero; falling back to direct docker stop)"
else
  echo "  eugr repo or config missing; falling back to direct docker stop"
fi

# The container must be gone on BOTH nodes, or the next launch fails on a stale name
# and — more importantly — production cannot reclaim the GPU memory.
for host in "$HEAD" "$WORKER"; do
  ssh -o BatchMode=yes "$host" "docker stop $NAME >/dev/null 2>&1 && docker rm $NAME >/dev/null 2>&1" 2>/dev/null \
    && echo "  ✓ $host: $NAME stopped" \
    || echo "  · $host: $NAME already stopped"
done

echo
for host in "$HEAD" "$WORKER"; do
  echo "  $host free: $(ssh -o BatchMode=yes "$host" "free -g | awk 'NR==2{print \$7\" GiB available\"}'" 2>/dev/null || echo '?')"
done
echo "✓ DeepSeek-V4-Flash-Vision-Exp (trial) stopped."

if [[ "${RESTORE_PROD:-0}" == "1" ]]; then
  echo
  echo "▶ restoring production (DeepSeek-V4-Flash-0731, eugr B12X) …"
  # Deliberately sequential and gated: the trial's memory must actually be back before
  # production's memory profiling runs, or the pool comes up short.
  exec "$SCRIPT_DIR/launch-deepseek-v4-flash-b12x-eugr.sh"
else
  echo
  echo "  Nothing is serving :8000 now."
  echo "  NOTE 2026-09-11: Vision-Exp IS production, and this script just stopped it."
  echo "  Restore it with:"
  echo "    $SCRIPT_DIR/launch-deepseek-v4-flash-vision-exp-b12x-eugr.sh"
  echo "  launch-deepseek-v4-flash-b12x-eugr.sh will NOT boot — its 0731 checkpoint was"
  echo "  deleted 2026-09-11 (re-download revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062)."
fi
