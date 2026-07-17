#!/usr/bin/env bash
# Co-locate BOTH of our custom NVFP4-MTP quants on the 2-Spark cluster using the
# STOCK eugr image directly (eugr/spark-vllm -> vllm-node), via eugr's
# spark-vllm-docker tooling — NOT the emosaru fork image and NOT sparkrun.
#
#   • emosaru/Qwen3.6-35B-A3B-NVFP4-MTP  (MoE, Marlin)  — recipe qwen3.6-35b-a3b-nvfp4-eugr.yaml
#   • emosaru/Qwen3.6-27B-NVFP4-MTP      (dense)        — recipe qwen3.6-27b-nvfp4-eugr.yaml
#
# Both run TP=2 across .236 + .237. Two TP=2 instances sharing the same GPU pair
# is exactly the case that collided under the direct/mp backend (two NCCL
# communicator groups stomping each other's GPU state). The fix is Ray: each model
# gets its OWN Ray cluster — distinct container name (--name) AND distinct Ray head
# port (--master-port). eugr runs `ray start --head --port $MASTER_PORT` inside the
# named container on the host network, so two distinct (name, master-port) pairs =
# two fully independent Ray clusters = independent placement groups and NCCL groups.
# Neither model can touch the other's GPU commands/state.
#
# MEMORY BALANCING IS DONE HERE, NOT IN THE RECIPES. The recipes ship a solo-sane
# gpu_memory_utilization (0.8); this script OVERRIDES both via
# --gpu-memory-utilization so the split lives in ONE place. Tune GPU_MEM_35B /
# GPU_MEM_27B below (env-overridable). vLLM reads util as a fraction of each node's
# ~121 GB pool, so the two fractions plus OS/hermes/ComfyUI overhead must stay < 1.
#
# Defaults: 35B=0.40, 27B=0.32  (sum 0.72 -> ~87 GB/node for the two models,
# leaving ~34 GB/node for OS+hermes(.236)+ComfyUI(.237)+headroom). These mirror the
# per-model 0.32 operating points that were validated on this cluster; nudge up for
# more KV/throughput headroom if a node has room, down if a launch OOMs.
#
# The 35B is launched and confirmed ready FIRST, then the 27B — so the 27B's memory
# profiling sees the 35B already resident and its util fraction is honored cleanly.
#
# Usage:
#   ./launch-qwen-dual-nvfp4-eugr.sh                    # launch both (daemon) + wait ready
#   REFRESH_IMAGE=1 ./launch-qwen-dual-nvfp4-eugr.sh    # pull+distribute fresh eugr image first
#   DOWNLOAD_MODELS=1 ./launch-qwen-dual-nvfp4-eugr.sh  # hf-download + distribute both quants first
#   GPU_MEM_35B=0.44 GPU_MEM_27B=0.30 ./launch-qwen-dual-nvfp4-eugr.sh   # rebalance the split
#   ONLY=35b ./launch-qwen-dual-nvfp4-eugr.sh           # bring up just one side (35b|27b)
#
# Stop with: ./stop-qwen-dual-nvfp4-eugr.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"

HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"

# --- 35B (MoE) knobs ---
RECIPE_35B="${RECIPE_35B:-$SCRIPT_DIR/recipes/qwen3.6-35b-a3b-nvfp4-eugr.yaml}"
MODEL_35B="${MODEL_35B:-emosaru/Qwen3.6-35B-A3B-NVFP4-MTP}"
NAME_35B="${NAME_35B:-vllm_qwen35}"
PORT_35B="${PORT_35B:-8000}"
MASTER_PORT_35B="${MASTER_PORT_35B:-29501}"   # Ray head (GCS) port for the 35B cluster
GPU_MEM_35B="${GPU_MEM_35B:-0.40}"

# --- 27B (dense) knobs ---
RECIPE_27B="${RECIPE_27B:-$SCRIPT_DIR/recipes/qwen3.6-27b-nvfp4-eugr.yaml}"
MODEL_27B="${MODEL_27B:-emosaru/Qwen3.6-27B-NVFP4-MTP}"
NAME_27B="${NAME_27B:-vllm_qwen27}"
PORT_27B="${PORT_27B:-8003}"
MASTER_PORT_27B="${MASTER_PORT_27B:-29502}"   # distinct Ray head port -> isolated cluster
GPU_MEM_27B="${GPU_MEM_27B:-0.32}"

READY_TIMEOUT="${READY_TIMEOUT:-1800}"
ONLY="${ONLY:-both}"                          # both | 35b | 27b

if [[ ! -d "$EUGR_DIR" ]]; then
  echo "✗ eugr repo not found at $EUGR_DIR" >&2
  echo "  git clone https://github.com/eugr/spark-vllm-docker.git $EUGR_DIR" >&2
  exit 1
fi
for r in "$RECIPE_35B" "$RECIPE_27B"; do
  [[ -f "$r" ]] || { echo "✗ recipe not found: $r" >&2; exit 1; }
done

# Sanity-check the memory split before doing anything expensive.
sum_ok="$(awk -v a="$GPU_MEM_35B" -v b="$GPU_MEM_27B" 'BEGIN{ s=a+b; printf (s<0.90)?"ok":"high"; }')"
if [[ "$sum_ok" == "high" ]]; then
  echo "⚠ GPU_MEM_35B ($GPU_MEM_35B) + GPU_MEM_27B ($GPU_MEM_27B) is high for co-location;" >&2
  echo "  leave room for OS (~0.20) + hermes/ComfyUI. Continuing, but a node may OOM." >&2
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

# Launch one model onto its OWN Ray cluster. Memory util is overridden here, not
# in the recipe, so the split is balanced in one place.
launch_one() {
  local recipe=$1 name=$2 port=$3 master_port=$4 util=$5 label=$6
  echo "▶ launching $label"
  echo "    recipe:      $recipe"
  echo "    container:   $name   port: $port   ray head: $master_port"
  echo "    gpu-mem-util: $util (override — recipe default ignored)"
  # -d daemon; --ray starts the per-model Ray cluster and injects the ray backend.
  ( cd "$EUGR_DIR" && ./run-recipe.sh "$recipe" \
      --ray -d \
      --name "$name" \
      --port "$port" \
      --master-port "$master_port" \
      --gpu-memory-utilization "$util" )
}

cd "$EUGR_DIR"

if [[ "${REFRESH_IMAGE:-0}" == "1" ]]; then
  echo "▶ refreshing eugr/spark-vllm:latest and distributing to $WORKER…"
  ./build-and-copy.sh -c "$WORKER"
fi

if [[ "${DOWNLOAD_MODELS:-0}" == "1" ]]; then
  echo "▶ downloading both quants and distributing across the cluster…"
  ./hf-download.sh "$MODEL_35B" -c
  ./hf-download.sh "$MODEL_27B" -c
fi

echo
echo "═══ dual-model co-location plan (stock eugr image, isolated Ray clusters) ═══"
echo "  35B: $NAME_35B  ->  http://$HEAD:$PORT_35B/v1   util $GPU_MEM_35B   ray :$MASTER_PORT_35B"
echo "  27B: $NAME_27B  ->  http://$HEAD:$PORT_27B/v1   util $GPU_MEM_27B   ray :$MASTER_PORT_27B"
echo "  nodes: $HEAD (head) + $WORKER (worker), TP=2 each"
echo

# 35B first, then 27B, so the 27B profiles memory against a resident 35B.
if [[ "$ONLY" == "both" || "$ONLY" == "35b" ]]; then
  launch_one "$RECIPE_35B" "$NAME_35B" "$PORT_35B" "$MASTER_PORT_35B" "$GPU_MEM_35B" "Qwen3.6-35B-A3B-NVFP4-MTP"
  wait_ready "$HEAD" "$PORT_35B" "Qwen3.6-35B-A3B-NVFP4-MTP" "$READY_TIMEOUT"
  echo
fi

if [[ "$ONLY" == "both" || "$ONLY" == "27b" ]]; then
  launch_one "$RECIPE_27B" "$NAME_27B" "$PORT_27B" "$MASTER_PORT_27B" "$GPU_MEM_27B" "Qwen3.6-27B-NVFP4-MTP"
  wait_ready "$HEAD" "$PORT_27B" "Qwen3.6-27B-NVFP4-MTP" "$READY_TIMEOUT"
  echo
fi

echo "✓ up."
[[ "$ONLY" == "both" || "$ONLY" == "35b" ]] && echo "  35B: http://$HEAD:$PORT_35B/v1   logs: ssh $HEAD 'docker logs -f $NAME_35B'"
[[ "$ONLY" == "both" || "$ONLY" == "27b" ]] && echo "  27B: http://$HEAD:$PORT_27B/v1   logs: ssh $HEAD 'docker logs -f $NAME_27B'"
echo "  stop:  $SCRIPT_DIR/stop-qwen-dual-nvfp4-eugr.sh"
