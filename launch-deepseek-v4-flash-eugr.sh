#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash-0731 (the OFFICIAL release) directly via eugr's
# spark-vllm-docker tooling — NOT via sparkrun.
#
# Same rationale as launch-qwen36-35b-nvfp4-eugr.sh: run through eugr's scripts +
# a fresh `eugr/spark-vllm:latest` image so runtime and checkpoint stay in sync.
# See memory/eugr_direct_container_workflow.md.
#
# ── WHY OUR OWN RECIPE, NOT eugr's by-name `deepseek-v4-flash` ────────────────────
# The official build lives at a DIFFERENT repo id: deepseek-ai/DeepSeek-V4-Flash-0731
# (released 2026-07-31). eugr's built-in recipe still hardcodes the PREVIEW id
# `deepseek-ai/DeepSeek-V4-Flash`, and run-recipe.py has no --model override — so we
# run our own pinned recipe by PATH: recipes/deepseek-v4-flash-mtp-eugr.yaml. Read
# that recipe's header before tuning anything. Architecture is identical to preview
# (0731 was only re-post-trained), so this is a model-id swap.
#
# ONE ~167 GB DOWNLOAD FEEDS BOTH PATHS: 0731 bakes the DSpark speculation module in
# (its weight map is byte-identical to the -DSpark checkpoint). The DSpark launcher
# reuses this same resident checkpoint — no second download.
#
# DeepSeek-V4-Flash is cluster_only (needs both Sparks) and heavy (util 0.8,
# max_model_len 500000) — it claims essentially the whole cluster on port 8000.
# Stop any other model on the cluster (e.g. the qwen nvfp4 instance) first.
#
# eugr documents DSV4F with --no-ray (PyTorch distributed backend), which also
# fits full context better; run-recipe strips the recipe's ray backend line in
# that mode. --no-ray is the default here; pass RAY_FLAG=--ray to override.
#
# INTERCONNECT (NCCL/RoCE): vLLM's multiproc executor can bind the mgmt/Wi-Fi NIC
# instead of the QSFP direct link, silently falling NCCL back to TCP (~19x slower
# all-reduce). Pin the QSFP interface with ETH_IF=/IB_IF= (eugr --eth-if/--ib-if)
# and verify with:  docker logs vllm_node | grep -iE 'NCCL|IB|Socket|via'
#
# Usage:
#   ./launch-deepseek-v4-flash-eugr.sh                 # launch (daemon) + wait ready
#   REFRESH_IMAGE=1 ./launch-deepseek-v4-flash-eugr.sh # pull+distribute fresh image first
#   DOWNLOAD_MODEL=1 ./launch-deepseek-v4-flash-eugr.sh# hf-download 0731 + distribute first
#   RAY_FLAG=--ray ./launch-deepseek-v4-flash-eugr.sh  # force Ray backend instead of no-ray
#   ETH_IF=enP2p1s0 ./launch-deepseek-v4-flash-eugr.sh # pin QSFP iface (NCCL/RoCE audit)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
# Our local pinned-0731 recipe, run by PATH (eugr's by-name recipe is preview).
RECIPE="${RECIPE:-$SCRIPT_DIR/recipes/deepseek-v4-flash-mtp-eugr.yaml}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME="${NAME:-vllm_node}"             # matches stop-deepseek-v4-flash-eugr.sh
PORT="${PORT:-8000}"
RAY_FLAG="${RAY_FLAG:---no-ray}"     # eugr's documented/default backend for DSV4F
ETH_IF="${ETH_IF:-}"                  # empty = eugr .env / auto-detect; set to pin QSFP (NCCL/RoCE)
IB_IF="${IB_IF:-}"                    # empty = eugr .env / auto-detect
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

if [[ ! -f "$RECIPE" ]]; then
  echo "✗ recipe not found: $RECIPE" >&2
  exit 1
fi

if [[ ! -d "$EUGR_DIR" ]]; then
  echo "✗ eugr repo not found at $EUGR_DIR" >&2
  echo "  git clone https://github.com/eugr/spark-vllm-docker.git $EUGR_DIR" >&2
  exit 1
fi

wait_ready() {
  local host=$1 port=$2 name=$3 timeout=${4:-1800}
  local url="http://$host:$port/v1/models"
  local deadline=$((SECONDS + timeout))
  echo "  ⏳ waiting for $name at $url (timeout ${timeout}s)…"
  while (( SECONDS < deadline )); do
    if curl -fsS -m 3 "$url" >/dev/null 2>&1; then
      echo "  ✓ $name ready"
      return 0
    fi
    sleep 5
  done
  echo "  ✗ $name not ready after ${timeout}s" >&2
  return 1
}

cd "$EUGR_DIR"

if [[ "${REFRESH_IMAGE:-0}" == "1" ]]; then
  echo "▶ refreshing eugr/spark-vllm:latest and distributing to $WORKER…"
  ./build-and-copy.sh -c "$WORKER"
fi

if [[ "${DOWNLOAD_MODEL:-0}" == "1" ]]; then
  echo "▶ downloading $MODEL and distributing across the cluster…"
  ./hf-download.sh "$MODEL" -c
fi

echo "▶ launching $RECIPE (cluster, $RAY_FLAG, daemon) on $HEAD + $WORKER…"
# -d = daemon: run-recipe returns after start; we poll the endpoint below.
launch_args=( "$RECIPE" $RAY_FLAG -d --name "$NAME" --port "$PORT" )
[[ -n "$ETH_IF" ]] && launch_args+=( --eth-if "$ETH_IF" )
[[ -n "$IB_IF" ]] && launch_args+=( --ib-if "$IB_IF" )
./run-recipe.sh "${launch_args[@]}" "$@"

wait_ready "$HEAD" "$PORT" "DeepSeek-V4-Flash-0731" "$READY_TIMEOUT"

echo
echo "✓ up. Endpoint: http://$HEAD:$PORT/v1"
echo "  logs:  ssh $HEAD 'docker logs -f $NAME'"
echo "  stop:  $SCRIPT_DIR/stop-deepseek-v4-flash-eugr.sh"
echo
echo "  First launch on 0731 — verify before trusting the operating point:"
echo "    • served id is …-0731:  curl -s http://$HEAD:$PORT/v1/models"
echo "    • MTP draft actually loaded (vLLM #43456 silent 0%): "
echo "        curl -s http://$HEAD:$PORT/metrics | grep spec_decode_num_accepted_tokens_total"
echo "      must be > 0 after a few requests, else the draft head mis-keyed → draft-off speed."
echo "    • NCCL on RoCE not TCP:  ssh $HEAD \"docker logs $NAME 2>&1 | grep -iE 'NCCL|IB|Socket|via'\""
