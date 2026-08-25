#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash-0731 on eugr's B12X stack — OUR PRODUCTION CONFIG.
#
# Promoted 2026-08-24, replacing tonyd2wild's stack (~/ds4-tony). Full audit:
# memory/eugr_b12x_bakeoff_2026_08_24.md
#
# ── WHY WE MIGRATED (correctness, not capacity) ───────────────────────────────
# eugr reserves KV blocks for a SECOND in-flight batch under async scheduling:
#   max_in_flight_tokens = max_concurrent_batches * max_num_batched_tokens
# and max_concurrent_batches == 2 when async_scheduling is on (config/vllm.py:604).
# tony's older vLLM uses max_num_batched_tokens directly, with NO multiplier, while
# still running async_scheduling: True — i.e. tony UNDER-reserves. The upstream
# docstring gives the reason: out-of-window blocks are freed on a processed-token
# basis, so a second in-flight batch transiently keeps its blocks. That is why we
# moved, not the pool numbers.
#
# ── WHY THIS RECIPE / THIS mnbt ───────────────────────────────────────────────
# Config matrix 2026-08-24 (len x async x mnbt, 6 of 8 configs). Winner:
#   max_model_len: auto   async_scheduling: default (ON)   max_num_batched_tokens: 4096
#   -> 2,027,903 KV tokens at the FULL 1,048,576 context, 1.93x concurrency
#   -> past tony (2,001,845) and past the community best (1,888,842)
#   -> every perf point inside run-to-run spread of the shipped mnbt-8192 default
#
# mnbt 4096 wins TWICE and costs nothing measurable:
#   1. it halves the binding KV group (the cr=4 compressor state cache, block_size 4,
#      whose per-request blocks are cdiv(sliding_window-1+in_flight, 4) -- purely
#      in-flight-driven, no context term), and
#   2. it raises AVAILABLE KV ~25% (smaller prefill batch -> smaller activation peak
#      at memory-profiling time): 13.11/13.27/13.93 GiB at 8192 vs 16.90/16.66/17.32
#      at 4096, six boots, no overlap.
# The old "mnbt 4096 throttles prefill" belief did NOT reproduce: 105k cold-prefill
# TTFT was 49.0s at 4096 vs 48.6s at 8192.
#
# DO NOT set --max-model-len. `auto` resolves to the full 1,048,576 and TRUNCATING IT
# BUYS NOTHING: available KV was 13.93 GiB at 524,288 vs 13.76 GiB at 1M (truncated is
# marginally HIGHER), and deep-context TTFT matched within 1%. The `GPU KV cache size`
# line is int(max_concurrency * max_model_len) -- a projection, not a pool measurement --
# so a shorter length only shrinks the headline number. Ignore that line as a capacity
# signal; use `Available KV cache memory` + max_concurrency instead.
#
# DO NOT pass --no-async-scheduling. It does enlarge the pool (1.66x at matched mnbt),
# but the throughput cost is UNRESOLVED (inside the 17% c2 run-to-run spread) and it
# gives up the very reservation that motivated this migration.
#
# ── OPERATING LIMITS (measured, config-independent) ───────────────────────────
# Deep context does NOT benefit from concurrency. At ~105k prompt the aggregate is
# flat from c2 onward (~6-8.5 tok/s) while TTFT climbs to ~121s and per-stream decode
# collapses 47 -> 21 -> 5 tok/s across c1/c2/c4. CAP DEEP-CONTEXT CONCURRENCY AT 2.
# This held in every config tested; no config change fixes it.
#
# NOT YET SOAKED: longest validation run was ~10 min. No evidence about stability,
# fragmentation, or preemption under hours of real traffic. Watch the first day.
#
# Usage:
#   ./launch-deepseek-v4-flash-b12x-eugr.sh              # launch (daemon) + wait ready
#   REFRESH_IMAGE=1 ./launch-deepseek-v4-flash-b12x-eugr.sh   # pull a fresh b12x image first
#   PORT=8010 ./launch-deepseek-v4-flash-b12x-eugr.sh    # side-by-side on another port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
# OUR fork of eugr's shipped mnbt-4096 recipe, run by ABSOLUTE PATH from this repo.
# Single delta vs upstream: it adds --served-model-name so the bare alias
# `deepseek-ai/DeepSeek-V4-Flash` is served alongside the canonical -0731 id. The retired
# tony stack served both; without the alias every client that hardcoded the base id 404s.
# Upstream defaults it otherwise carries: gmu 0.85, max_model_len auto, block_size 256,
# max_num_seqs 8, max_cudagraph_capture_size 64, kv-cache-dtype fp8, DSpark 5/probabilistic.
# BEING A FORK, upstream fixes do NOT flow in automatically -- the recipe header carries
# the re-diff command to run after an eugr pull.
RECIPE="${RECIPE:-$SCRIPT_DIR/recipes/deepseek-v4-flash-b12x-eugr.yaml}"
IMAGE="${IMAGE:-eugr/spark-vllm-b12x:latest}"
CONFIG="${CONFIG:-$SCRIPT_DIR/env/eugr-roce.env}"   # RoCE fabric; see that file's header
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME="${NAME:-vllm_node}"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

[[ -d "$EUGR_DIR" ]] || { echo "✗ eugr repo not found at $EUGR_DIR" >&2
  echo "  git clone https://github.com/eugr/spark-vllm-docker.git $EUGR_DIR" >&2; exit 1; }
[[ -f "$RECIPE" ]] || { echo "✗ recipe not found: $RECIPE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "✗ cluster config not found: $CONFIG" >&2; exit 1; }

# Refuse to fight another server for the port AND the cluster's memory. The most
# likely occupant is the old tony stack, which claims the whole cluster.
if curl -fsS -m 3 "http://$HEAD:$PORT/v1/models" >/dev/null 2>&1; then
  echo "✗ something is already serving http://$HEAD:$PORT/v1 — refusing to start." >&2
  echo "  If it is the retired tony stack:  ~/ds4-tony/stop-deepseek-v4-flash-dspark.sh" >&2
  echo "  If it is this stack:              $SCRIPT_DIR/stop-deepseek-v4-flash-b12x-eugr.sh" >&2
  echo "  Or run side by side:              PORT=8010 $0   (needs free memory)" >&2
  exit 1
fi

# The B12X image must exist on BOTH nodes — TP=2 loads it on each.
for host in "$HEAD" "$WORKER"; do
  if ! ssh -o BatchMode=yes "$host" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
    echo "✗ $host is missing $IMAGE" >&2
    echo "  ssh $host docker pull $IMAGE" >&2
    exit 1
  fi
done

if [[ "${REFRESH_IMAGE:-0}" == "1" ]]; then
  echo "▶ pulling fresh $IMAGE on both nodes…"
  for host in "$HEAD" "$WORKER"; do ssh -o BatchMode=yes "$host" "docker pull '$IMAGE'"; done
fi

echo "▶ launching DeepSeek-V4-Flash-0731 (eugr B12X, mnbt 4096, len auto) …"
echo "  recipe:    $RECIPE"
echo "  image:     $IMAGE"
echo "  config:    $CONFIG   (RoCE fabric)"
echo "  container: $NAME     port: $PORT"
( cd "$EUGR_DIR" && ./run-recipe.sh "$RECIPE" -d --config "$CONFIG" -t "$IMAGE" --port "$PORT" "$@" )

echo "  ⏳ waiting for http://$HEAD:$PORT/v1/models (timeout ${READY_TIMEOUT}s, ~5 min typical)…"
deadline=$((SECONDS + READY_TIMEOUT))
while (( SECONDS < deadline )); do
  if curl -fsS -m 5 "http://$HEAD:$PORT/v1/models" >/dev/null 2>&1; then
    echo "  ✓ ready"
    echo
    echo "✓ up. Endpoint: http://$HEAD:$PORT/v1"
    echo "  served ids: $(curl -fsS -m 5 "http://$HEAD:$PORT/v1/models" | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || echo '?')"
    echo "  KV pool:    $(ssh -o BatchMode=yes "$HEAD" "docker logs $NAME 2>&1 | grep -E 'Available KV cache memory|GPU KV cache size' | tail -2 | tr '\n' ' '" 2>/dev/null)"
    echo "  logs:       ssh $HEAD 'docker logs -f $NAME'"
    echo "  stop:       $SCRIPT_DIR/stop-deepseek-v4-flash-b12x-eugr.sh"
    echo
    echo "  Expected at this config: ~16.9 GiB available, ~2.0M KV tokens, ~1.9x @ 1,048,576."
    echo "  Serves BOTH the canonical -0731 id and the bare 'deepseek-ai/DeepSeek-V4-Flash'"
    echo "  alias (restored for clients carried over from the retired tony stack)."
    exit 0
  fi
  if ssh -o BatchMode=yes "$HEAD" "docker logs $NAME 2>&1 | tail -40" 2>/dev/null \
      | grep -qE "Traceback|Engine core initialization failed|CUDA out of memory"; then
    echo "  ✗ engine error during boot — last 30 lines:" >&2
    ssh -o BatchMode=yes "$HEAD" "docker logs $NAME 2>&1 | tail -30" >&2
    exit 1
  fi
  sleep 10
done
echo "  ✗ not ready after ${READY_TIMEOUT}s" >&2
exit 1
