#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next-NVFP4 on eugr's B12X stack — TRIAL, replaces :8000.
#
# ── READ THIS FIRST: the 2026-08-26 evaluation is OBSOLETE ────────────────────
# memory/qwen38_flash_next_eval_2026_08_26.md concluded "the working path is SGLang,
# not vLLM" and declined to migrate, largely because standing up a second serving
# stack would strand all our recipes, mods and launch scripts. Re-checked 2026-09-11:
#
#   * vLLM PR #53896 (model support)  MERGED 2026-08-31, shipped in v0.29.0 (2026-09-09)
#   * SGLang PR #36497                CLOSED UNMERGED 2026-09-08   <- the dead path
#   * eugr shipped qwen3.8-flash-next-nvfp4-{cluster,solo}.yaml on 2026-09-08
#   * local-inference-lab (the org b12x is built from) published the NVFP4 quant
#     — this was the exact "watch this signal" trigger the old eval named, and it fired
#
# So this runs on our EXISTING toolchain. The old eval's remaining objections are
# quality-fit, not engineering: NL2Repo-Bench (closest proxy to our agentic-coding
# workload) still favoured DeepSeek-V4-Flash-0731, 54.2 vs 48.1. That is the thing to
# re-measure, and it is the reason this is labelled a trial and not a promotion.
#
# ── WHAT IS AND IS NOT KNOWN ABOUT THIS CONFIG ────────────────────────────────
# NOT MEASURED BY US AT ALL. No boot, no benchmark, no correctness gate has ever been
# run against this model on this cluster. Everything below is upstream default or
# third-party report. Treat the first run as an experiment, not a deployment:
#   * eugr's defaults (TP=2, len 262144, mnbt 4096, MTP 4, gmu 0.7) are unvalidated here.
#   * The nearest datapoint is NVIDIA forum topic 381428 (2x GB10, TP=2, RoCE): 47 tok/s
#     typical / 70.2 peak single-stream — parity with our DS4F ~47 tok/s. But that was
#     SGLang with different spec settings. It does NOT transfer to this stack as a number.
#   * Two things could fail at boot and both are documented in the recipe header:
#     fp8 KV under the QSA path, and whether multimodality actually works.
#
# ── gmu 0.7 IS DELIBERATE — DO NOT RAISE IT TO MATCH PRODUCTION ───────────────
# Production DS4F runs 0.85. This runs eugr's 0.7 for two independent reasons:
#   1. memory/earlyoom_head_kill.md — the host earlyoom daemon (`-m 2 -s 80`, still
#      ACTIVE on both nodes as of 2026-09-11) SIGTERMs the vLLM head at ~2.49 GB
#      available and lists vllm as *preferred* prey. At 0.85 the margin was ~2.7 GB and
#      routine Ubuntu maintenance was enough to tip it; it killed the Vision-Exp trial
#      after a 5-day soak. 0.7 buys back ~18 GB of that margin.
#   2. The forum reports GB10 page cache starves the GPU allocator above ~0.82.
# DEFAULT IS NOW 0.8 (set in the -noMoeB12x recipe), matching every measurement in
# memory/flashnext_bakeoff_2026_09_24.md and the field's 0.80-0.835. It leaves 7-13 GiB
# host headroom vs DS4F's 3 GiB. To back OFF use `GMU=0.75 ./this`; to push up, note the
# forum's ~0.82 page-cache ceiling above and measure before trusting it.
#
# ── THIS REPLACES PRODUCTION ON :8000 ─────────────────────────────────────────
# Not a co-location. As of 2026-09-11 production is DeepSeek-V4-Flash-Vision-Exp, and it must
# be stopped before this can start; the script refuses otherwise. Note there is no longer a
# second DS4F checkpoint on disk — 0731 was deleted — so the only way back to a DeepSeek stack
# is relaunching Vision-Exp. Do not experiment in a way that could damage that checkpoint.
#
# Usage:
#   ./stop-deepseek-v4-flash-vision-exp-b12x-eugr.sh   # stop PRODUCTION to free the cluster
#   ./launch-qwen38-flash-next-b12x-eugr.sh            # launch (daemon) + wait ready
#   REFRESH_IMAGE=1 ./launch-qwen38-flash-next-b12x-eugr.sh   # pull a fresh b12x image
#   DOWNLOAD_MODEL=1 ./launch-qwen38-flash-next-b12x-eugr.sh  # fetch checkpoint to both nodes
#   GMU=0.75 ./launch-qwen38-flash-next-b12x-eugr.sh          # back off from the 0.8 default
#   PORT=8010 ./launch-qwen38-flash-next-b12x-eugr.sh         # side-by-side (needs free memory)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
# ── context length: native (262,144) by default; YaRN (1M) is OPT-IN AND CURRENTLY BROKEN ──
# YARN=0 (DEFAULT) = native 262,144. YARN=1 selects the RoPE-scaled 1M recipe.
#
# *** YARN=1 DOES NOT BOOT while MTP speculative decoding is on. *** Verified 2026-09-11:
#     ValueError: QSA sequence length 1000000 exceeds the configured limit 262144
# Cause: `mtp.py:_make_draft_vllm_config()` swaps in `speculative_config.draft_model_config`,
# and --hf-overrides is applied to the TARGET ModelConfig only — it never reaches the draft.
# So the main model correctly resolves max_seq_len=1000000 while the MTP draft's QSA layer is
# still built at 262144 and rejects the workload. It fails at memory-profiling time, loudly.
# See memory/qwen38_flash_next_yarn_blocked_by_mtp.md for the untried ways out (the most
# promising is baking rope_parameters into a local copy of the checkpoint's config.json, so
# BOTH configs read the scaled values from disk rather than from --hf-overrides).
#
# Default flipped to 0 on 2026-09-11 at the user's call, so a bare invocation does the thing
# that actually works. Everything for YaRN is kept and wired — only the default changed.
#
# Independently, even once it boots, the vendor warns that every open-source framework
# implements STATIC YaRN: the factor applies at ALL lengths and "potentially impact[s]
# performance on shorter texts" — and short-context agentic coding is our dominant workload.
#
# YARN_FACTOR tunes the scaling (vendor: "modify the factor as needed"). The recipe ships 4.0.
# Setting it ALSO recomputes --max-model-len to match, because changing one without the other
# is the easy way to get a silently wrong config:
#     YARN_FACTOR=2.0 -> 524,288    YARN_FACTOR=4.0 -> 1,048,576
YARN="${YARN:-0}"
if [[ "$YARN" == "1" ]]; then
  # NOTE: the YaRN recipe still carries --moe-backend b12x and so has the SAME boot
  # failure documented below. YaRN does not boot on this stack anyway (the overrides
  # never reach draft_model_config while MTP is on), so it has not been reworked.
  RECIPE="${RECIPE:-$SCRIPT_DIR/recipes/qwen3.8-flash-next-nvfp4-cluster-yarn-eugr.yaml}"
else
  # ── DEFAULT IS THE -noMoeB12x RECIPE, AND THAT IS DELIBERATE ─────────────────
  # The plain `-cluster-eugr.yaml` recipe CANNOT BOOT with the nvidia checkpoint.
  # `--moe-backend` is GLOBAL and there is no moe_backend_per_quant, so one value has
  # to serve both the main model's NVFP4 routed experts and nvidia's block-FP8 MTP
  # experts. Measured 2026-09-24 on nightly-20260923, b12x fails both ways:
  #   TP=2 +EP -> ValueError: NvFp4 MoE backend 'B12X' does not support the deployment
  #               configuration ... FusedMoEParallelConfig(ep_size=2, use_ep=True)
  #   TP=1     -> ValueError: moe_backend='b12x' is not supported for FP8 MoE
  # Dropping the flag lets each quant resolve its own backend (FLASHINFER_CUTLASS) and
  # removes the EP prohibition. --gdn-decode-kernel b12x and --linear-backend b12x are
  # KEPT, so 2 of the 3 b12x kernels still run.
  # The old recipe is still reachable for comparison:
  #   RECIPE="$SCRIPT_DIR/recipes/qwen3.8-flash-next-nvfp4-cluster-eugr.yaml" $0
  RECIPE="${RECIPE:-$SCRIPT_DIR/recipes/qwen3.8-flash-next-nvfp4-cluster-noMoeB12x-eugr.yaml}"
fi
# PINNED, not :latest. `eugr/spark-vllm-b12x:latest` locally still resolves to the
# 2026-09-10 production image (retagged prod-20260910), which predates the b12x fork
# absorbing PR #55513 and so needs the retired fix-block-fp8-mtp-modelopt mod.
# nightly-20260923 ships that fix natively (_BLOCK_FP8_MOE_ALGOS present) and is the
# image every measurement in memory/flashnext_bakeoff_2026_09_24.md was taken on.
IMAGE="${IMAGE:-eugr/spark-vllm-b12x:nightly-20260923}"
CONFIG="${CONFIG:-$SCRIPT_DIR/env/eugr-roce-flashnext.env}"
# Our corrected fork of eugr's mods/drop-caches. Upstream's is INERT: its loop body
# `echo 3 > /proc/sys/vm/drop_caches >> $LOG 2>&1` redirects stdout twice, so "3" lands
# in the logfile and the proc file is never written — it syncs but never drops. Two
# GB10 reasons we want a working one: a warm page cache starves the GPU allocator during
# model load on unified memory, and writeback backlog eats the host margin earlyoom
# polices (memory/earlyoom_head_kill.md). DROP_CACHES=0 disables.
DROP_CACHES="${DROP_CACHES:-1}"
DROP_CACHES_MOD="${DROP_CACHES_MOD:-$SCRIPT_DIR/mods/drop-caches-fixed}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME="${NAME:-vllm_node_flashnext}"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"   # matches Vision-Exp: 206-shard load + AOT compile

MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
MODEL_CACHE_DIR="models--${MODEL//\//--}"
CKPT_GIB=124                             # nvidia quant, measured from the HF API 2026-09-11: 123.6 GiB
                                         # (was 99 for the local-inference-lab quant, deleted 2026-09-11 —
                                         #  its 4-bit MTP head cost spec-decode acceptance)
NEED_GIB=$((CKPT_GIB + 6))               # + margin for in-flight blobs and logs

# Containers that own the whole cluster. Flash-Next cannot co-exist with either.
CONFLICTS=("vllm_node" "vllm_node_vision")

[[ -d "$EUGR_DIR" ]] || { echo "✗ eugr repo not found at $EUGR_DIR" >&2
  echo "  git clone https://github.com/eugr/spark-vllm-docker.git $EUGR_DIR" >&2; exit 1; }
[[ -f "$RECIPE" ]] || { echo "✗ recipe not found: $RECIPE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "✗ cluster config not found: $CONFIG" >&2; exit 1; }
if [[ "$DROP_CACHES" == "1" ]]; then
  [[ -x "$DROP_CACHES_MOD/run.sh" ]] || { echo "✗ drop-caches mod missing/not executable: $DROP_CACHES_MOD/run.sh" >&2
    echo "  (set DROP_CACHES=0 to launch without it)" >&2; exit 1; }
fi

# ── 1. port / stack conflicts ─────────────────────────────────────────────────
if [[ "${FORCE:-0}" != "1" ]] && curl -fsS -m 3 "http://$HEAD:$PORT/v1/models" >/dev/null 2>&1; then
  echo "✗ something is already serving http://$HEAD:$PORT/v1 — refusing to start." >&2
  echo "  Flash-Next REPLACES that stack; it cannot run beside it." >&2
  echo "  PRODUCTION (Vision-Exp):  $SCRIPT_DIR/stop-deepseek-v4-flash-vision-exp-b12x-eugr.sh" >&2
  echo "  Taking production down for a trial is a deliberate act — be sure that is what" >&2
  echo "  you mean, and see the restore line printed by that stop script." >&2
  exit 1
fi
for host in "$HEAD" "$WORKER"; do
  for c in "${CONFLICTS[@]}"; do
    if ssh -o BatchMode=yes "$host" "docker ps --format '{{.Names}}' | grep -qx '$c'" 2>/dev/null; then
      echo "✗ $host still runs '$c' — it holds the cluster's memory." >&2
      echo "  Stop it first, then re-run this script." >&2
      exit 1
    fi
  done
done

# ── 2. image present, and NEW ENOUGH to actually contain the architecture ─────
# Our b12x image as of 2026-08-23 does NOT have qwen4_exp at all (verified: registry
# reports zero Qwen4Exp entries, and vllm/models/qwen4_exp is absent). A date check
# would be a guess, so ask the image directly.
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

echo "▶ checking $IMAGE actually registers Qwen4Exp…"
ARCHS=$(ssh -o BatchMode=yes "$HEAD" "docker run --rm --entrypoint python3 '$IMAGE' -c '
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS, _TEXT_GENERATION_MODELS
m = {**_MULTIMODAL_MODELS, **_TEXT_GENERATION_MODELS}
print(\",\".join(sorted(k for k in m if \"Qwen4Exp\" in k)))' 2>/dev/null | tail -1" 2>/dev/null || true)
if [[ -z "$ARCHS" ]]; then
  echo "✗ $IMAGE does not support Qwen3.8-Flash-Next (no Qwen4Exp architectures)." >&2
  echo "  Model support merged upstream 2026-08-31; eugr's b12x image needs to be at" >&2
  echo "  least the 2026-09-10 'latest' build. Refresh on BOTH nodes:" >&2
  echo "      REFRESH_IMAGE=1 $0" >&2
  echo "  or, to also re-tag it as the recipe's 'vllm-node-b12x' container:" >&2
  echo "      ( cd $EUGR_DIR && ./build-and-copy.sh --exp-b12x -c $WORKER )" >&2
  exit 1
fi
echo "  ✓ image registers: $ARCHS"

# ── 3. checkpoint: present on both nodes, or enough disk to fetch it ──────────
missing=()
for host in "$HEAD" "$WORKER"; do
  if ! ssh -o BatchMode=yes "$host" "[ -d ~/.cache/huggingface/hub/$MODEL_CACHE_DIR ]" 2>/dev/null; then
    missing+=("$host")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "▶ checkpoint $MODEL absent on: ${missing[*]}"
  short=()
  for host in "${missing[@]}"; do
    free_gib=$(ssh -o BatchMode=yes "$host" "df -BG --output=avail / | tail -1 | tr -dc '0-9'" 2>/dev/null || echo 0)
    echo "    $host: ${free_gib} GiB free, needs ~${NEED_GIB} GiB"
    (( free_gib < NEED_GIB )) && short+=("$host:${free_gib}")
  done
  if [[ ${#short[@]} -gt 0 ]]; then
    echo "✗ not enough disk for the ${CKPT_GIB} GiB checkpoint on: ${short[*]}" >&2
    echo "  Each node needs its OWN full copy — there is no shared/NFS HF cache." >&2
    echo "  Largest reclaimable items (see memory/cluster_topology.md before deleting):" >&2
    echo "    ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp  157 GiB" >&2
    echo "      ^ the trial this model replaces; deleting it keeps DS4F-0731 as rollback" >&2
    echo "    ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731        156 GiB" >&2
    echo "      ^ DO NOT delete: this is production's rollback target" >&2
    echo "    docker builder prune -af   (safe; was 154 GB of dead cache on .236 once)" >&2
    exit 1
  fi
  if [[ "${DOWNLOAD_MODEL:-0}" != "1" ]]; then
    echo "✗ refusing to download ${CKPT_GIB} GiB implicitly." >&2
    echo "  Re-run with:  DOWNLOAD_MODEL=1 $0" >&2
    echo "  or fetch manually: ( cd $EUGR_DIR && ./hf-download.sh --config $CONFIG $MODEL -c )" >&2
    exit 1
  fi
  echo "▶ downloading $MODEL to head and copying to worker (~${CKPT_GIB} GiB, slow)…"
  # NOTE the argument order: eugr's `-c` greedily consumes every following non-flag
  # token as a copy-host, so `-c "$MODEL"` would swallow the model name and then die
  # with "model name required". Model positional FIRST, bare `-c` LAST (which makes it
  # fall back to COPY_HOSTS from --config, i.e. the RoCE address).
  ( cd "$EUGR_DIR" && ./hf-download.sh --config "$CONFIG" "$MODEL" -c )
fi

# ── 4. launch ─────────────────────────────────────────────────────────────────
OVERRIDES=()
[[ -n "${GMU:-}" ]] && OVERRIDES+=(--gpu-memory-utilization "$GMU")

# YARN_FACTOR: run-recipe only accepts a fixed override allow-list (port/host/tp/gmu/
# max_model_len) — yarn_factor is NOT in it — so render a temp recipe with the factor
# swapped, and drive the matching length through the override that IS supported.
# LOG_REQUESTS=1 appends --enable-log-requests (vLLM default False, arg_utils.py:2946), which
# logs each request's prompt and SamplingParams. This is the only way to see what a CLIENT
# actually sent after a proxy has rewritten it — e.g. LiteLLM runs with drop_params:true and
# silently discards params it does not recognise. NOTE: it logs PROMPT TEXT to the container
# log; enable it for an investigation, not permanently.
if [[ "${LOG_REQUESTS:-0}" == "1" ]]; then
  TMP_LOG="$(mktemp -t flashnext-logreq-XXXXXX.yaml)"
  sed -E "s|^(    --gpu-memory-utilization \{gpu_memory_utilization\})|    --enable-log-requests \\\n\1|" "$RECIPE" > "$TMP_LOG"
  grep -q -- "--enable-log-requests" "$TMP_LOG" \
    || { echo "✗ failed to inject --enable-log-requests" >&2; exit 1; }
  RECIPE="$TMP_LOG"
  echo "  ▸ LOG_REQUESTS=1 — prompts WILL be written to the container log (temporary use only)"
fi

# LOAD_FORMAT overrides --load-format. The recipe ships `b12x` (O_DIRECT fast loader), but
# that loader is tuned for local-inference-lab's own quant and does NOT implement every
# conversion: nvidia's FP8-block-scaled MTP experts trip
#   NotImplementedError: strided dtype conversion: mtp.layers.0.mlp.experts.0.down_proj.weight
# `auto` uses the stock safetensors loader (slower to load, identical weights afterwards).
# Load format only affects HOW weights are read -- the b12x moe/linear/gdn BACKENDS are
# separate flags and are unaffected.
if [[ -n "${LOAD_FORMAT:-}" ]]; then
  TMP_LF="$(mktemp -t flashnext-lf-XXXXXX.yaml)"
  sed -E "s|^    --load-format .*|    --load-format ${LOAD_FORMAT} \\\\|" "$RECIPE" > "$TMP_LF"
  grep -qE "^    --load-format ${LOAD_FORMAT} " "$TMP_LF" \
    || { echo "✗ failed to set --load-format ${LOAD_FORMAT}" >&2; exit 1; }
  RECIPE="$TMP_LF"
  echo "  ▸ LOAD_FORMAT=${LOAD_FORMAT} (temp recipe $TMP_LF)"
fi

# MOE_BACKEND overrides --moe-backend. The recipe ships `b12x`, but the b12x NVFP4 MoE
# kernel does NOT support expert parallelism:
#   ValueError: NvFp4 MoE backend 'B12X' does not support the deployment configuration
#               ... FusedMoEParallelConfig(tp_size=1, ep_size=2, ...)
# and nvidia's block-FP8 MTP REQUIRES expert parallelism (its 128-blocks do not divide the
# TP-sliced 320 expert width). Those two constraints are mutually exclusive, so serving
# nvidia's quant with MTP means giving up the b12x MoE kernel for the main model.
if [[ -n "${MOE_BACKEND:-}" ]]; then
  TMP_MB="$(mktemp -t flashnext-moe-XXXXXX.yaml)"
  sed -E "s|^    --moe-backend .*|    --moe-backend ${MOE_BACKEND} \\\\|" "$RECIPE" > "$TMP_MB"
  grep -qE "^    --moe-backend ${MOE_BACKEND} " "$TMP_MB" \
    || { echo "✗ failed to set --moe-backend ${MOE_BACKEND}" >&2
         grep -qE '^    --moe-backend ' "$RECIPE" \
           || echo "  CAUSE: $(basename "$RECIPE") has NO --moe-backend line to replace." \
                   "The default -noMoeB12x recipe deliberately omits it (b12x cannot serve" \
                   "nvidia's block-FP8 MTP, see the recipe header). MOE_BACKEND only works" \
                   "on a recipe that already sets the flag, e.g." \
                   "RECIPE=.../qwen3.8-flash-next-nvfp4-cluster-eugr.yaml" >&2
         exit 1; }
  RECIPE="$TMP_MB"
  echo "  ▸ MOE_BACKEND=${MOE_BACKEND} (temp recipe $TMP_MB)"
fi

# SPEC_TOKENS overrides num_speculative_tokens. Not in run-recipe's override allow-list
# (port/host/tp/gmu/max_model_len only), so render a temp recipe with the value swapped.
# eugr ships 4; our own notes call 3 the value validated against the QSA ring-capacity
# constraint, and qsa.py caps at 4 (_QSA_MAX_SPECULATIVE_TOKENS), so 4 is legal but at the
# edge. Measured acceptance decays hard by position (pos3 ~10%), hence the A/B.
if [[ -n "${SPEC_TOKENS:-}" ]]; then
  TMP_SPEC="$(mktemp -t flashnext-spec-XXXXXX.yaml)"
  sed -E "s/^  num_speculative_tokens: .*/  num_speculative_tokens: ${SPEC_TOKENS}/" "$RECIPE" > "$TMP_SPEC"
  grep -qE "^  num_speculative_tokens: ${SPEC_TOKENS}\$" "$TMP_SPEC" \
    || { echo "✗ failed to set num_speculative_tokens=${SPEC_TOKENS}" >&2; exit 1; }
  RECIPE="$TMP_SPEC"
  echo "  ▸ SPEC_TOKENS=${SPEC_TOKENS} (temp recipe $TMP_SPEC)"
fi

if [[ -n "${YARN_FACTOR:-}" ]]; then
  [[ "$YARN" == "1" ]] || { echo "✗ YARN_FACTOR set but YARN=0 — nothing to scale." >&2; exit 1; }
  TMP_RECIPE="$(mktemp -t flashnext-yarn-XXXXXX.yaml)"
  sed -E "s/^  yarn_factor: .*/  yarn_factor: ${YARN_FACTOR}/" "$RECIPE" > "$TMP_RECIPE"
  grep -qE "^  yarn_factor: ${YARN_FACTOR}\$" "$TMP_RECIPE" \
    || { echo "✗ failed to set yarn_factor=${YARN_FACTOR} in $TMP_RECIPE" >&2; exit 1; }
  YARN_LEN="$(python3 -c "print(int(262144*float('${YARN_FACTOR}')))")"
  OVERRIDES+=(--max-model-len "$YARN_LEN")
  RECIPE="$TMP_RECIPE"
  echo "  ▸ YARN_FACTOR=${YARN_FACTOR} -> max_model_len ${YARN_LEN} (temp recipe $TMP_RECIPE)"
fi

echo
if [[ "$YARN" == "1" ]]; then
  CTX_DESC="len ${YARN_LEN:-1000000} via YaRN x${YARN_FACTOR:-4.0}"
  FORK_DESC="our fork; deltas = --served-model-name + YaRN rope override"
else
  CTX_DESC="len 262144 native (no rope scaling)"
  FORK_DESC="our fork; drops --moe-backend b12x, gmu 0.8, +served-model-name && usage flags"
fi
echo "▶ launching Qwen3.8-Flash-Next-NVFP4 (eugr B12X, TP=2, MTP 4, $CTX_DESC) …"
echo "  recipe:    $RECIPE   ($FORK_DESC)"
echo "  model:     $MODEL"
echo "  image:     $IMAGE"
echo "  config:    $CONFIG   (RoCE fabric, container $NAME, master port 29503)"
echo "  port:      $PORT     gmu: ${GMU:-0.8 (recipe default — see header)}"
[[ "$DROP_CACHES" == "1" ]] && echo "  mods:      drop-caches-fixed (our fork; upstream's is inert — see its PROVENANCE.md)"
[[ ${#OVERRIDES[@]} -gt 0 ]] && echo "  overrides: ${OVERRIDES[*]}"
echo
if [[ "$YARN" == "1" ]]; then
  echo "  ⚠ YaRN IS ON (context ${YARN_LEN:-1000000}) — AND IS KNOWN TO FAIL AT BOOT."
  echo "    Expect: ValueError: QSA sequence length ... exceeds the configured limit 262144."
  echo "    --hf-overrides never reaches speculative_config.draft_model_config, so the MTP"
  echo "    draft's QSA stays at 262144. Unset YARN (or YARN=0) for the working config."
  echo "    Static YaRN also scales at ALL lengths, so the vendor warns it can COST quality on"
  echo "    SHORT prompts — which is most of our agentic-coding traffic. If you are not"
  echo "    actually using >262k, relaunch with YARN=0 for the native 262,144 config."
  echo "    Tune down instead of off with e.g. YARN_FACTOR=2.0 (524,288)."
  echo
fi
echo "  ⚠ TRIAL STACK, NEVER BOOTED HERE. Unlike the DeepSeek launchers, no number in"
echo "    this script was measured on our hardware. Expect to debug the first boot."
echo "  ⚠ It does NOT answer to DeepSeek model ids. Clients hardcoding"
echo "    'deepseek-ai/DeepSeek-V4-Flash*' will 404 — that is deliberate, so nobody"
echo "    silently benchmarks Flash-Next believing it is DeepSeek."
echo
MODS=()
[[ "$DROP_CACHES" == "1" ]] && MODS+=(--apply-mod "$DROP_CACHES_MOD")
# Port of vLLM PR #55513 (block-FP8 MTP in ModelOpt mixed checkpoints). REQUIRED by the
# nvidia quant: without it MTP dies at load on a missing w2_weight_scale_inv. Merged
# upstream 2026-09-08 but NOT in v0.29.0 (release cut before the merge) and not in the
# b12x fork. Retire when a b12x image ships it — see the mod's PROVENANCE.md.
BLOCKFP8_MOD="${BLOCKFP8_MOD:-$SCRIPT_DIR/mods/fix-block-fp8-mtp-modelopt}"
if [[ "${BLOCK_FP8_MTP_FIX:-1}" == "1" ]]; then
  [[ -x "$BLOCKFP8_MOD/run.sh" ]] || { echo "✗ block-fp8-mtp mod missing: $BLOCKFP8_MOD/run.sh" >&2; exit 1; }
  MODS+=(--apply-mod "$BLOCKFP8_MOD")
fi

# ── REDUCED-VOCABULARY MTP DRAFTING — OPT-IN, NEVER BOOTED ───────────────────
# Restricts the DRAFT head's projection to the lowest N token ids. The win is bandwidth:
# the draft head is read once per speculative position per decode step, and at TP=2 with
# N=65536 rank 0 projects 65,536 of its 124,160 org rows while rank 1 skips its matmul
# entirely — 26.4% of the full head's rows. Correctness is structural: masked ids are
# never PROPOSED, and the target still verifies the full vocabulary, so an out-of-set
# token is rejected like any other bad draft and the emitted text is unchanged.
#
# Community measurement (independently re-measured 2026-09-16, TP2+EP, mtp=3, 262k):
# +8.4% mean throughput, "entirely step time, acceptance unchanged". Expect tok/step to
# DIP slightly and tok/s to RISE — judge this on tok/s, not on acceptance.
#
# *** WE HAVE NEVER BOOTED THIS. *** Static validation only (anchors, idempotency,
# byte-compile, shard arithmetic). See mods/mtp-reduced-draft-vocab/PROVENANCE.md for
# what is verified, the safety rails, and the one known trade-off (it bypasses the b12x
# BF16 vocabulary projection, which only applies at batch size 1).
#
#   DRAFT_VOCAB=65536 ./launch-qwen38-flash-next-b12x-eugr.sh
#
# Unset or 0 leaves drafting at the full 248,320 ids. The mod is inert without the env
# var, so applying it and not setting DRAFT_VOCAB changes nothing.
DRAFT_VOCAB_MOD="${DRAFT_VOCAB_MOD:-$SCRIPT_DIR/mods/mtp-reduced-draft-vocab}"
if [[ "${DRAFT_VOCAB:-0}" != "0" ]]; then
  [[ "$DRAFT_VOCAB" =~ ^[0-9]+$ ]] || {
    echo "✗ DRAFT_VOCAB must be a positive integer (got '$DRAFT_VOCAB')" >&2; exit 1; }
  [[ -x "$DRAFT_VOCAB_MOD/run.sh" ]] || {
    echo "✗ reduced-draft-vocab mod missing: $DRAFT_VOCAB_MOD/run.sh" >&2; exit 1; }
  MODS+=(--apply-mod "$DRAFT_VOCAB_MOD")
  OVERRIDES+=(-e "VLLM_MTP_DRAFT_VOCAB_SIZE=$DRAFT_VOCAB")
  echo "  ⚠ reduced-vocab MTP drafting ON (lowest $DRAFT_VOCAB ids) — UNPROVEN on our"
  echo "    hardware. Confirm the 'Reduced-vocab MTP drafting ENABLED' line on rank 0,"
  echo "    then A/B decode tok/s against a DRAFT_VOCAB=0 boot."
fi
( cd "$EUGR_DIR" && ./run-recipe.sh "$RECIPE" -d \
    --config "$CONFIG" -t "$IMAGE" --port "$PORT" \
    "${MODS[@]}" "${OVERRIDES[@]}" "$@" )

echo "  ⏳ waiting for http://$HEAD:$PORT/v1/models (timeout ${READY_TIMEOUT}s)…"
deadline=$((SECONDS + READY_TIMEOUT))
LAUNCH_EPOCH=0   # seconds-ago window for `docker logs --since`, grows as we wait
while (( SECONDS < deadline )); do
  if curl -fsS -m 5 "http://$HEAD:$PORT/v1/models" >/dev/null 2>&1; then
    echo "  ✓ ready"
    echo
    echo "✓ up. Endpoint: http://$HEAD:$PORT/v1"
    echo "  served ids: $(curl -fsS -m 5 "http://$HEAD:$PORT/v1/models" | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || echo '?')"
    echo "  KV pool:    $(ssh -o BatchMode=yes "$HEAD" "docker logs $NAME 2>&1 | grep -E 'Available KV cache memory|GPU KV cache size' | tail -2 | tr '\n' ' '" 2>/dev/null)"
    echo "  logs:       ssh $HEAD 'docker logs -f $NAME'"
    echo "  stop:       $SCRIPT_DIR/stop-qwen38-flash-next-b12x-eugr.sh"
    echo
    if [[ "$DROP_CACHES" == "1" ]]; then
      dc="$(ssh -o BatchMode=yes "$HEAD" "docker exec $NAME tail -1 /tmp/drop_caches.log 2>/dev/null" 2>/dev/null)"
      case "$dc" in
        *dropped*) echo "  drop-caches: ✓ active ($dc)" ;;
        3)         echo "  drop-caches: ✗ INERT — a bare '3' means upstream's broken mod ran, not ours." ;;
        *)         echo "  drop-caches: ? no 'dropped' line yet (first tick is +60s): ${dc:-<empty>}" ;;
      esac
    fi
    echo "  BEFORE TRUSTING THIS RUN, verify the two open questions from the recipe header:"
    echo "    1. fp8 KV survived the QSA path (no NotImplementedError in the log), and"
    echo "    2. an actual image request works — a clean boot does not prove multimodality."
    echo "  Then A/B against DS4F-0731 on NL2Repo-style agentic-coding work, which is the"
    echo "  benchmark the old eval says this model LOSES (48.1 vs 54.2). That is the decision."
    exit 0
  fi
  # Only look at THIS run's output. eugr `exec`s into a long-lived container, so `docker logs`
  # accumulates across launches — on 2026-09-11 a stale YaRN traceback from a previous run made
  # this abort a boot that was actually progressing. --since pins it to the current attempt.
  #
  # And match FATAL lines only. The old pattern grepped for bare "Traceback", which also appears
  # inside NON-FATAL torch-inductor warnings (e.g. "Cubin file saved by TritonBundler not found",
  # which vLLM recovers from and then logs "torch.compile took 0.06 s"). That aborted three
  # healthy boots on 2026-09-11. vLLM warnings carry W<date>/WARNING; real failures carry an
  # " ERROR " level token, so require that or an explicit fatal phrase.
  if ssh -o BatchMode=yes "$HEAD" "docker logs --since ${LAUNCH_EPOCH}s $NAME 2>&1 | tail -60" 2>/dev/null \
      | grep -qE "Engine core initialization failed|encountered a fatal error|Executor failed|CUDA out of memory|\) ERROR .*(Error|Exception)"; then
    echo "  ✗ engine error during boot — last 40 lines:" >&2
    ssh -o BatchMode=yes "$HEAD" "docker logs $NAME 2>&1 | tail -40" >&2
    echo >&2
    echo "  Known first-boot suspects for THIS model:" >&2
    echo "    * NotImplementedError near QSA/KV  -> fp8 KV unsupported; set --kv-cache-dtype auto" >&2
    echo "    * mamba-cache-mode                 -> must be 'align'; 'all' is rejected" >&2
    echo "    * CUDA OOM at load                 -> lower with GMU=0.65 $0" >&2
    echo "  Container logs are destroyed by the stop script — capture them NOW if you need them." >&2
    exit 1
  fi
  sleep 10
  LAUNCH_EPOCH=$((LAUNCH_EPOCH + 10))
done
echo "  ✗ not ready after ${READY_TIMEOUT}s" >&2
echo "  Check: ssh $HEAD 'docker logs --tail 100 $NAME'" >&2
exit 1
