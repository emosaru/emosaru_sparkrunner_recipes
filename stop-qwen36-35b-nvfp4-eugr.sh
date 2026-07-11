#!/usr/bin/env bash
# Stop Qwen3.6-35B-A3B-NVFP4 launched via eugr's tooling.
# Counterpart to launch-qwen36-35b-nvfp4-eugr.sh.
#
# eugr's `launch-cluster.sh stop` stops the vllm_node container on the head node
# and every worker in EUGR_DIR/.env (CLUSTER_NODES), then tears down Ray.

set -uo pipefail

EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME="${NAME:-vllm_node}"

echo "▶ stopping $NAME across the cluster via launch-cluster.sh stop…"
if [[ -d "$EUGR_DIR" ]]; then
  ( cd "$EUGR_DIR" && ./launch-cluster.sh --name "$NAME" stop ) || echo "  (launcher stop returned non-zero; falling back to direct docker stop)"
else
  echo "  eugr repo not at $EUGR_DIR; falling back to direct docker stop"
fi

# Belt-and-suspenders: ensure the container is gone on both nodes even if the
# launcher couldn't (e.g. .env drift).
for host in "$HEAD" "$WORKER"; do
  ssh "$host" "docker stop $NAME >/dev/null 2>&1 && docker rm $NAME >/dev/null 2>&1" 2>/dev/null \
    && echo "  ✓ $host: $NAME stopped" \
    || echo "  · $host: $NAME already stopped"
done

echo
echo "✓ Qwen3.6-35B-A3B-NVFP4 (eugr) stopped."
