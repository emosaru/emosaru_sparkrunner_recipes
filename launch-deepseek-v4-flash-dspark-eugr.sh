#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash-0731 with DSpark speculation — the DSpark variant of
# launch-deepseek-v4-flash-eugr.sh — directly via eugr's spark-vllm-docker
# tooling, NOT via sparkrun. See memory/eugr_direct_container_workflow.md.
#
# Recipe: recipes/deepseek-v4-flash-dspark-eugr.yaml (our port of eugr's PENDING
# PR #319 — https://github.com/eugr/spark-vllm-docker/pull/319, branch
# feat/deepseek-v4-flash-dspark). Read that recipe's header before tuning anything;
# it documents where we deviate from the PR and why.
#
# BECAUSE #319 IS UNMERGED, this script fetches the PR's mod (mods/fix-dspark-sm120-topk)
# into the eugr checkout before launching. That mod is REQUIRED on GB10/SM120 — without
# it the DSpark draft's topk=256 sparse-MLA decode has no compiled FlashInfer bucket and
# warmup crashes on `num_tokens > 64`. The fetch writes ONE new directory into
# $EUGR_DIR/mods/ and does NOT switch that repo's branch or touch tracked files.
# Once #319 merges, the mod arrives on main and the fetch no-ops.
#
# THE IMAGE MUST CARRY vLLM #46995 (the DSpark method), or startup dies with
# "unknown speculative method 'dspark'". Our other eugr recipes don't need this, so a
# `vllm-node` that works for them can still be too old here — REBUILD_VLLM=1 forces it.
# CHECKED 2026-07-17: the installed vllm-node (image 5c033d442a9a, vllm
# 0.25.1.dev24+g96bb89286, built 2026-07-10) ALREADY registers the dspark method, so
# no rebuild is needed today. Re-check after any image refresh with:
#   docker run --rm --entrypoint bash vllm-node -c \
#     'python3 -c "import inspect;from vllm.config.speculative import SpeculativeConfig as S;print(\"dspark\" in inspect.getsource(S))"'
#
# MODEL IS NOW THE SAME 0731 CHECKPOINT AS THE MTP RECIPE. The official 0731 build
# bakes the DSpark module in (weight map byte-identical to the old standalone
# -DSpark repo), so there is no longer a separate DSpark download: if you already
# ran launch-deepseek-v4-flash-eugr.sh with DOWNLOAD_MODEL=1, 0731 is resident on
# both nodes and this launches without DOWNLOAD_MODEL. Only pass DOWNLOAD_MODEL=1 if
# 0731 has never been fetched on this cluster (~167 GB).
#
# DSpark is heavy and cluster_only: TP=2 across BOTH Sparks, util 0.85, 200k context —
# it claims essentially the whole cluster on port 8000. STOP any other model first
# (./stop-qwen-dual-nvfp4-eugr.sh, ./stop-deepseek-v4-flash-eugr.sh). Distinct
# container name + Ray port from the MTP instance, so the two stop scripts never
# cross-fire — but they cannot co-reside; there isn't the memory for both.
#
# Backend is --ray: unlike the MTP sibling (which defaults --no-ray, eugr's documented
# DSV4F backend), PR #319's recipe pins the ray backend and eugr's ~4 accept-length /
# ~53 tok/s numbers come from that path. Pass RAY_FLAG=--no-ray to A/B it — run-recipe
# strips the recipe's ray backend line in that mode.
#
# Usage:
#   ./launch-deepseek-v4-flash-dspark-eugr.sh                  # launch (daemon) + wait ready
#                                                              # (0731 usually already resident from the MTP launch)
#   REBUILD_VLLM=1 ./launch-deepseek-v4-flash-dspark-eugr.sh   # force-rebuild vLLM wheels first (needs #46995)
#   REFRESH_IMAGE=1 ./launch-deepseek-v4-flash-dspark-eugr.sh  # rebuild+distribute image (no vLLM wheel rebuild)
#   DOWNLOAD_MODEL=1 ./launch-deepseek-v4-flash-dspark-eugr.sh # hf-download 0731 (~167 GB) ONLY if never fetched
#   GPU_MEM=0.75 ./launch-deepseek-v4-flash-dspark-eugr.sh     # override the memory split
#   PR_REFRESH=1 ./launch-deepseek-v4-flash-dspark-eugr.sh     # re-fetch the mod (PR moved / force update)
#   RAY_FLAG=--no-ray ./launch-deepseek-v4-flash-dspark-eugr.sh
#   ETH_IF=enP2p1s0 ./launch-deepseek-v4-flash-dspark-eugr.sh  # pin QSFP iface (NCCL/RoCE audit)
#
# Stop with: ./stop-deepseek-v4-flash-dspark-eugr.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
RECIPE="${RECIPE:-$SCRIPT_DIR/recipes/deepseek-v4-flash-dspark-eugr.yaml}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME="${NAME:-vllm_dsv4_dspark}"       # distinct from the MTP instance's vllm_node
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29503}"    # distinct Ray head port (qwen dual uses 29501/29502)
GPU_MEM="${GPU_MEM:-}"                 # empty = use the recipe default (0.85, per PR #319)
RAY_FLAG="${RAY_FLAG:---ray}"          # PR #319 pins the ray backend; see header
ETH_IF="${ETH_IF:-}"                   # empty = eugr .env / auto-detect; set to pin QSFP (NCCL/RoCE)
IB_IF="${IB_IF:-}"                     # empty = eugr .env / auto-detect
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

# PR #319 — source of the recipe this launcher runs and of the SM120 mod below.
PR_NUM="${PR_NUM:-319}"
PR_REMOTE="${PR_REMOTE:-origin}"
MOD_PATH="mods/fix-dspark-sm120-topk"

if [[ ! -d "$EUGR_DIR" ]]; then
  echo "✗ eugr repo not found at $EUGR_DIR" >&2
  echo "  git clone https://github.com/eugr/spark-vllm-docker.git $EUGR_DIR" >&2
  exit 1
fi
[[ -f "$RECIPE" ]] || { echo "✗ recipe not found: $RECIPE" >&2; exit 1; }

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
  echo "    logs: ssh $host 'docker logs --tail 100 $NAME'" >&2
  return 1
}

# Pull just the SM120 topk mod out of the pending PR, without switching the eugr
# checkout's branch. Idempotent: skips when the mod is already present.
ensure_pr_mod() {
  if [[ -d "$EUGR_DIR/$MOD_PATH" && "${PR_REFRESH:-0}" != "1" ]]; then
    echo "▶ $MOD_PATH already present in $EUGR_DIR — skipping PR #$PR_NUM fetch (PR_REFRESH=1 to force)"
    return 0
  fi
  echo "▶ fetching $MOD_PATH from eugr PR #$PR_NUM (unmerged) into $EUGR_DIR…"
  ( cd "$EUGR_DIR"
    git fetch --quiet "$PR_REMOTE" "pull/$PR_NUM/head" || {
      echo "  ✗ could not fetch pull/$PR_NUM/head from $PR_REMOTE" >&2
      echo "    if #$PR_NUM has merged, the mod is on main: git -C $EUGR_DIR pull" >&2
      exit 1
    }
    # Checkout ONLY the mod dir from the PR head — no branch switch, no tracked-file churn.
    git checkout FETCH_HEAD -- "$MOD_PATH" || {
      echo "  ✗ $MOD_PATH not found in PR #$PR_NUM head — did the PR restructure?" >&2
      exit 1
    }
    chmod +x "$MOD_PATH/run.sh" 2>/dev/null || true
    echo "  ✓ $MOD_PATH staged from PR #$PR_NUM ($(git rev-parse --short FETCH_HEAD))"
    echo "  · note: this leaves $MOD_PATH staged in $EUGR_DIR's index; it is PR content,"
    echo "    not local work — 'git -C $EUGR_DIR restore --staged $MOD_PATH' to unstage."
  )
}

cd "$EUGR_DIR"

if [[ "${REBUILD_VLLM:-0}" == "1" ]]; then
  echo "▶ rebuilding vLLM wheels + image and distributing to $WORKER (needs vLLM #46995 for dspark)…"
  ./build-and-copy.sh --rebuild-vllm -c "$WORKER"
elif [[ "${REFRESH_IMAGE:-0}" == "1" ]]; then
  echo "▶ refreshing eugr/spark-vllm:latest and distributing to $WORKER…"
  ./build-and-copy.sh -c "$WORKER"
fi

if [[ "${DOWNLOAD_MODEL:-0}" == "1" ]]; then
  echo "▶ downloading $MODEL (~167 GB) and distributing across the cluster…"
  ./hf-download.sh "$MODEL" -c
fi

ensure_pr_mod

echo
echo "═══ DeepSeek-V4-Flash-0731 + DSpark (stock eugr image, port of PR #$PR_NUM) ═══"
echo "  recipe:    $RECIPE"
echo "  model:     $MODEL"
echo "  container: $NAME   port: $PORT   ray head: $MASTER_PORT"
echo "  nodes:     $HEAD (head) + $WORKER (worker), TP=2 $RAY_FLAG"
echo "  gpu-mem:   ${GPU_MEM:-recipe default (0.85)}"
echo

# -d = daemon: run-recipe returns after start; we poll the endpoint below.
launch_args=( "$RECIPE" $RAY_FLAG -d --name "$NAME" --port "$PORT" --master-port "$MASTER_PORT" )
[[ -n "$GPU_MEM" ]] && launch_args+=( --gpu-memory-utilization "$GPU_MEM" )
[[ -n "$ETH_IF" ]] && launch_args+=( --eth-if "$ETH_IF" )
[[ -n "$IB_IF" ]] && launch_args+=( --ib-if "$IB_IF" )
./run-recipe.sh "${launch_args[@]}" "$@"

wait_ready "$HEAD" "$PORT" "DeepSeek-V4-Flash-0731 (DSpark)" "$READY_TIMEOUT"

echo
echo "✓ up. Endpoint: http://$HEAD:$PORT/v1"
echo "  logs:  ssh $HEAD 'docker logs -f $NAME'"
echo "  stop:  $SCRIPT_DIR/stop-deepseek-v4-flash-dspark-eugr.sh"
echo
echo "  First launch — check these in the log before trusting the operating point:"
echo "    • 'GPU KV cache size' — a full 200k fp8 request needs ~2.1 GiB; if KV is"
echo "      near that, lower max_model_len rather than climbing util past 0.85."
echo "    • '[fix-dspark-sm120-topk] applied' — the mod ran; absent = expect a"
echo "      'num_tokens > 64' crash during sparse-MLA warmup."
echo "    • accept length — eugr reports ~4 on code workloads, ~53 tok/s."
