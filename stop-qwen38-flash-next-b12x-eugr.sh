#!/usr/bin/env bash
# Stop the Qwen3.8-Flash-Next-NVFP4 trial stack.
# Counterpart to launch-qwen38-flash-next-b12x-eugr.sh.
#
# Uses the same RoCE cluster config as the launcher so launch-cluster.sh resolves the
# SAME node list it started (eugr's own .env points at the management net).
#
# ── WHY THIS ONE SAVES LOGS AND THE OLDER STOP SCRIPTS DO NOT ─────────────────
# memory/earlyoom_head_kill.md: the Vision-Exp stop script `docker rm`s on both nodes,
# which DESTROYED the container logs. When that trial died after a 5-day soak the entire
# diagnosis had to be reconstructed from host-side journalctl/syslog/sar. This model has
# never booted here, so its logs are the most valuable artifact it produces. Logs are
# dumped to .flashnext-logs/ before removal; set SAVE_LOGS=0 to skip.
#
# Usage:
#   ./stop-qwen38-flash-next-b12x-eugr.sh            # save logs, then stop + remove
#   SAVE_LOGS=0 ./stop-qwen38-flash-next-b12x-eugr.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
CONFIG="${CONFIG:-$SCRIPT_DIR/env/eugr-roce-flashnext.env}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME="${NAME:-vllm_node_flashnext}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/.flashnext-logs}"

# ── capture logs BEFORE anything is removed ───────────────────────────────────
if [[ "${SAVE_LOGS:-1}" == "1" ]]; then
  stamp="$(date +%Y%m%dT%H%M%S)"
  mkdir -p "$LOG_DIR"
  for host in "$HEAD" "$WORKER"; do
    out="$LOG_DIR/${stamp}_${host##*.}_${NAME}.log"
    if ssh -o BatchMode=yes "$host" "docker logs $NAME" > "$out" 2>&1 && [[ -s "$out" ]]; then
      echo "  ✓ saved $host logs -> ${out#$SCRIPT_DIR/} ($(wc -l < "$out") lines)"
    else
      rm -f "$out"
      echo "  · $host: no logs to save (container already gone?)"
    fi
  done
  echo
fi

echo "▶ stopping $NAME across the cluster…"
if [[ -d "$EUGR_DIR" && -f "$CONFIG" ]]; then
  ( cd "$EUGR_DIR" && ./launch-cluster.sh --config "$CONFIG" --name "$NAME" stop ) \
    || echo "  (launcher stop returned non-zero; falling back to direct docker stop)"
else
  echo "  eugr repo or config missing; falling back to direct docker stop"
fi

# Belt-and-suspenders: the container must be gone on BOTH nodes, or the next launch
# fails on a stale name / leaked GPU memory. A half-dead cluster is the documented
# failure mode here — the head can die while the worker spins at 100% GPU forever —
# so never assume the launcher's stop reached both.
# 2026-09-11: this used to be `docker stop && docker rm` with `|| echo "already stopped"`,
# which MASKED a real failure — the containers stayed Up, the next launch `exec`d into the
# stale one, and it died on leftover distributed state (gloo "Connection closed"). Force-remove
# and then VERIFY absence; never infer success from an exit code we deliberately swallowed.
for host in "$HEAD" "$WORKER"; do
  ssh -o BatchMode=yes "$host" "docker rm -f $NAME >/dev/null 2>&1; ! docker ps -a --format '{{.Names}}' | grep -qx '$NAME'" 2>/dev/null \
    && echo "  ✓ $host: $NAME removed" \
    || { echo "  ✗ $host: $NAME STILL PRESENT — the next launch would reuse it and fail." >&2
         echo "    ssh $host docker rm -f $NAME" >&2; }
done

echo
for host in "$HEAD" "$WORKER"; do
  echo "  $host free: $(ssh -o BatchMode=yes "$host" "free -g | awk 'NR==2{print \$7\" GiB available\"}'" 2>/dev/null || echo '?')"
done
echo "✓ Qwen3.8-Flash-Next (eugr B12X) stopped."
echo
echo "  ▶ RESTORE PRODUCTION:"
echo "    $SCRIPT_DIR/launch-deepseek-v4-flash-vision-exp-b12x-eugr.sh"
echo "  (launch-deepseek-v4-flash-b12x-eugr.sh will NOT boot — its 0731 checkpoint was"
echo "   deleted 2026-09-11; it would need a 156 GiB re-download at revision"
echo "   7872f01b1d1fe23eabc4c98b48bffcef5a386062.)"
