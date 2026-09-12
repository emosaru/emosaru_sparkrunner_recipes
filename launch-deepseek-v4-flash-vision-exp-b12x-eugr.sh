#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash-Vision-Exp on eugr's B12X stack — TRIAL, NOT PRODUCTION.
#
# Counterpart to launch-deepseek-v4-flash-b12x-eugr.sh. Same mechanism (eugr
# run-recipe.sh, RoCE config, our recipe by absolute path), different checkpoint,
# plus the vendored native-vision mod.
#
# ── THIS TAKES PRODUCTION DOWN. THERE IS NO WAY AROUND IT. ───────────────────
# Non-KV is ~83.4 GiB/node; production already sits at ~121.6 GiB/node. Two DS4F-class
# checkpoints do not fit on this cluster at any port or any gmu. PORT=8010 does NOT
# make this co-resident — it only avoids the port check. This script therefore REFUSES
# to start while anything is serving the production port, and prints the stop command.
#
# ── WHY THE CONFIG LOOKS LIKE PRODUCTION ─────────────────────────────────────
# The recipe is a near-clone of the production 0731 fork so an A/B measures the
# CHECKPOINT, not a config rewrite. Forced deltas only: model id, DSpark k=6 (k=5 is
# rejected — n_predict is 3 on this checkpoint), and the vision mod (which also fixes
# rms_norm_eps 1e-20 -> 1e-6 for the b12x fused MHC kernel). Full rationale in the
# recipe header.
#
# ── WHAT TO EXPECT (so you do not misread a healthy run as a broken one) ─────
#   * KV pool SMALLER than production's 1,995,672 tokens — the ViT is replicated on
#     every TP rank, not sharded. Expected.
#   * Decode ~10-17% behind 0731 — this checkpoint's DSpark head accepts ~0.46 vs
#     0.56-0.59. Checkpoint property. Do not tune for it.
#   * If boot dies at determine_available_memory, the vision mod did not apply —
#     check the mod output in the container logs before blaming memory.
#   * If it OOMs, back off with GMU=0.82 (upstream vision recipe's value) rather than
#     editing the recipe, so exactly one variable moves.
#
# ── IT ANSWERS TO ALL THREE IDS — INCLUDING PRODUCTION'S ─────────────────────
# Serves deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (canonical, first) PLUS both
# production ids: -0731 and the bare deepseek-ai/DeepSeek-V4-Flash. So every existing
# client keeps working with no reconfiguration — and every existing client is answered
# by Vision-Exp without knowing it. Nothing 404s to warn you. Before attributing any
# benchmark number to 0731, check /v1/models data[0] or the `model` field on the
# response. Vision-Exp also decodes ~10-17% slower, which stale clients feel as an
# unexplained latency regression.
#
# ── VISION IS NOT PROVEN BY A CLEAN BOOT ─────────────────────────────────────
# This checkpoint declares the same architecture string as 0731. A lineage that fails
# open serves text-only and SILENTLY DISCARDS IMAGES — it looks perfectly healthy.
# This script therefore gates readiness on an actual multimodal capability check, and
# --smoke-image sends a real image. Never conclude "vision works" from /v1/models.
#
# Usage:
#   ./launch-deepseek-v4-flash-vision-exp-b12x-eugr.sh                # launch + wait ready
#   GMU=0.82 ./launch-deepseek-v4-flash-vision-exp-b12x-eugr.sh       # back off memory
#   ./launch-deepseek-v4-flash-vision-exp-b12x-eugr.sh --smoke-image  # + real image test
#   FORCE=1 ./launch-...sh    # skip the production-is-running refusal (you meant it)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUGR_DIR="${EUGR_DIR:-$HOME/code/spark-vllm-docker-eugr}"
# ── RECIPE / IMAGE ARE A MATCHED PAIR — DO NOT SET THEM INDEPENDENTLY ─────────
# There are now TWO working combinations and they are NOT interchangeable:
#
#   recipes/...-0910-eugr.yaml  + eugr/spark-vllm-b12x:latest (2026-09-10)  <- DEPLOYED
#       native vision, NO vision mod, --attention-backend B12X
#   recipes/...-b12x-eugr.yaml  + vllm-node-b12x:latest       (2026-08-23)  <- rollback
#       vendored vision mod REQUIRED, --attention-backend B12X_MLA_SPARSE
#
# Mixing them fails in two different ways, both of which cost us production time:
#   * old recipe on the new image -> `B12X_MLA_SPARSE` is not in the new image's backend
#     enum (it was consolidated to `B12X`), so the engine refuses to start.
#   * vision mod on the new image -> anchor-drift FATAL, because the new image implements
#     natively what the mod patches in.
# So IMAGE is DERIVED from RECIPE below, and both invariants are VERIFIED against the real
# image before launch rather than assumed. Setting IMAGE explicitly still works and still
# gets validated.
RECIPE="${RECIPE:-$SCRIPT_DIR/recipes/deepseek-v4-flash-vision-exp-b12x-0910-eugr.yaml}"
case "$RECIPE" in
  *-0910-eugr.yaml) DEFAULT_IMAGE="eugr/spark-vllm-b12x:latest" ;;   # native vision
  *)                DEFAULT_IMAGE="vllm-node-b12x:latest" ;;         # needs the vendored mod
esac
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
VISION_MOD="${VISION_MOD:-$SCRIPT_DIR/mods/deepseek-v4-vision-exp}"
# drop-caches: OUR fork — eugr's mods/drop-caches is INERT (a double stdout redirection writes
# "3" to its logfile instead of /proc/sys/vm/drop_caches, so it syncs but never drops). Two
# GB10 reasons we want it: a warm page cache starves the GPU allocator during model load on
# unified memory, and writeback backlog eats the host margin earlyoom polices
# (memory/earlyoom_head_kill.md). The 2026-09-12 deployment runs WITH it, so it is on by
# default here — a bare launch must reproduce what is actually serving. DROP_CACHES=0 disables.
DROP_CACHES="${DROP_CACHES:-1}"
DROP_CACHES_MOD="${DROP_CACHES_MOD:-$SCRIPT_DIR/mods/drop-caches-fixed}"
CONFIG="${CONFIG:-$SCRIPT_DIR/env/eugr-roce-vision.env}"
HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
NAME="${NAME:-vllm_node_vision}"
PORT="${PORT:-8000}"
PROD_NAME="${PROD_NAME:-vllm_node}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"   # > production's 1800: vision adds processor init
MODEL_ID="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"

SMOKE_IMAGE=0
PASSTHRU=()
for arg in "$@"; do
  case "$arg" in
    --smoke-image) SMOKE_IMAGE=1 ;;
    *) PASSTHRU+=("$arg") ;;
  esac
done

[[ -d "$EUGR_DIR"    ]] || { echo "✗ eugr repo not found at $EUGR_DIR" >&2
  echo "  git clone https://github.com/eugr/spark-vllm-docker.git $EUGR_DIR" >&2; exit 1; }
[[ -f "$RECIPE"      ]] || { echo "✗ recipe not found: $RECIPE" >&2; exit 1; }
[[ -f "$CONFIG"      ]] || { echo "✗ cluster config not found: $CONFIG" >&2; exit 1; }
[[ -f "$VISION_MOD/run.sh" ]] || { echo "✗ vision mod missing run.sh: $VISION_MOD" >&2; exit 1; }

# ── Refuse to fight production for the cluster's memory ──────────────────────
if [[ "${FORCE:-0}" != "1" ]] && curl -fsS -m 3 "http://$HEAD:$PORT/v1/models" >/dev/null 2>&1; then
  echo "✗ something is already serving http://$HEAD:$PORT/v1 — refusing to start." >&2
  echo "  Vision-Exp CANNOT co-exist with the production stack: ~83.4 GiB/node of" >&2
  echo "  non-KV weights each, and production already holds ~121.6 GiB/node." >&2
  echo "  Stop production first:  $SCRIPT_DIR/stop-deepseek-v4-flash-b12x-eugr.sh" >&2
  echo "  Restore it after:       $SCRIPT_DIR/launch-deepseek-v4-flash-b12x-eugr.sh" >&2
  exit 1
fi
for host in "$HEAD" "$WORKER"; do
  if ssh -o BatchMode=yes "$host" "docker ps --format '{{.Names}}' | grep -qx '$PROD_NAME'" 2>/dev/null; then
    echo "✗ $host still runs the production container '$PROD_NAME'." >&2
    echo "  Stop it first: $SCRIPT_DIR/stop-deepseek-v4-flash-b12x-eugr.sh" >&2
    exit 1
  fi
done

for host in "$HEAD" "$WORKER"; do
  if ! ssh -o BatchMode=yes "$host" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
    echo "✗ $host is missing $IMAGE" >&2; echo "  ssh $host docker pull $IMAGE" >&2; exit 1
  fi
done

OVERRIDES=()
[[ -n "${GMU:-}" ]] && OVERRIDES+=(--gpu-memory-utilization "$GMU")

echo "▶ launching DeepSeek-V4-Flash-Vision-Exp (eugr B12X + native vision mod) …"
echo "  recipe:     $RECIPE"
echo "  vision mod: $VISION_MOD  (vendored — see its PROVENANCE.md)"
echo "  image:      $IMAGE"
echo "  config:     $CONFIG   (RoCE fabric, container $NAME)"
echo "  port:       $PORT     DSpark k=6 (k=5 is rejected on this checkpoint)"
[[ ${#OVERRIDES[@]} -gt 0 ]] && echo "  overrides:  ${OVERRIDES[*]}"
echo
echo "  ⚠ THIS STACK ANSWERS TO PRODUCTION'S MODEL IDS."
echo "    Serving: Vision-Exp (canonical) + deepseek-ai/DeepSeek-V4-Flash-0731"
echo "             + deepseek-ai/DeepSeek-V4-Flash"
echo "    Existing clients will be answered by Vision-Exp with no error and no warning."
echo "    Any number you collect while this is up is a Vision-Exp number."
echo
echo "  NOTE: a clean boot does NOT prove vision works — this checkpoint shares 0731's"
echo "        architecture string and a fail-open lineage would serve text-only."
echo

# ── VERIFY THE RECIPE/IMAGE PAIRING AGAINST THE REAL IMAGE ───────────────────
# One probe container reports the two facts that matter; we then check the recipe against
# them. This catches a mispairing no matter how the script was invoked.
echo "▶ verifying recipe/image compatibility…"
WANT_BACKEND="$(python3 -c "
import yaml,re,sys
r=yaml.safe_load(open('$RECIPE'))
c=r['command'].format(**r.get('defaults',{}))
m=re.search(r'--attention-backend\s+(\S+)', c)
print(m.group(1) if m else '')
" 2>/dev/null)"

PROBE="$(ssh -o BatchMode=yes "$HEAD" "docker run --rm --memory=2g --entrypoint bash '$IMAGE' -c '
V=/usr/local/lib/python3.12/dist-packages/vllm
echo -n \"NATIVE_VISION=\"; grep -c IMAGE_PLACEHOLDER \$V/tokenizers/deepseek_v4_encoding.py 2>/dev/null || echo 0
f=\$(grep -rl \"class AttentionBackendEnum\" \$V --include=*.py 2>/dev/null | head -1)
echo -n \"BACKENDS=\"; grep -oE \"^[[:space:]]+B12X[A-Z_0-9]* *=\" \"\$f\" 2>/dev/null | tr -d \" =\" | tr \"\\n\" \",\"
echo
'" 2>/dev/null)"
NATIVE_VISION="$(sed -n 's/^NATIVE_VISION=//p' <<<"$PROBE" | tr -dc '0-9')"
BACKENDS="$(sed -n 's/^BACKENDS=//p' <<<"$PROBE")"
[[ -n "${NATIVE_VISION:-}" ]] || { echo "✗ could not probe $IMAGE (is it present on $HEAD?)" >&2; exit 1; }

# 1. the recipe's attention backend must exist in THIS image
if [[ -n "$WANT_BACKEND" ]] && [[ ",$BACKENDS," != *",$WANT_BACKEND,"* ]]; then
  echo "✗ RECIPE/IMAGE MISMATCH: recipe asks for --attention-backend '$WANT_BACKEND'," >&2
  echo "  but $IMAGE only provides: ${BACKENDS%,}" >&2
  echo "  You have paired the $( [[ "$NATIVE_VISION" -gt 0 ]] && echo NEW || echo OLD ) image with the wrong recipe." >&2
  echo "  Use:  recipes/deepseek-v4-flash-vision-exp-0910-eugr.yaml  with eugr/spark-vllm-b12x:latest" >&2
  echo "    or: recipes/deepseek-v4-flash-vision-exp-b12x-eugr.yaml  with vllm-node-b12x:latest" >&2
  exit 1
fi

# 2. the vendored vision mod is applied ONLY when the image lacks native vision
MODS=()
if [[ "$DROP_CACHES" == "1" ]]; then
  [[ -x "$DROP_CACHES_MOD/run.sh" ]] || { echo "✗ drop-caches mod missing: $DROP_CACHES_MOD/run.sh" >&2
    echo "  (DROP_CACHES=0 to launch without it)" >&2; exit 1; }
  MODS+=(--apply-mod "$DROP_CACHES_MOD")
  echo "  ✓ drop-caches-fixed will be applied"
fi
if [[ "${NATIVE_VISION:-0}" -gt 0 ]]; then
  echo "  ✓ $IMAGE has NATIVE vision — not applying the vendored mod (it would FATAL on drift)"
else
  [[ -f "$VISION_MOD/run.sh" ]] || { echo "✗ image lacks native vision and the vendored mod is missing: $VISION_MOD" >&2; exit 1; }
  echo "  ✓ $IMAGE lacks native vision — applying vendored mod $(basename "$VISION_MOD")"
  MODS+=(--apply-mod "$VISION_MOD")
fi
echo "  ✓ backend '$WANT_BACKEND' available in this image"

( cd "$EUGR_DIR" && ./run-recipe.sh "$RECIPE" -d \
    --config "$CONFIG" -t "$IMAGE" --port "$PORT" \
    "${MODS[@]}" \
    "${OVERRIDES[@]}" "${PASSTHRU[@]}" )

echo "  ⏳ waiting for http://$HEAD:$PORT/v1/models (timeout ${READY_TIMEOUT}s)…"
deadline=$((SECONDS + READY_TIMEOUT))
ready=0
while (( SECONDS < deadline )); do
  if curl -fsS -m 5 "http://$HEAD:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ssh -o BatchMode=yes "$HEAD" "docker logs $NAME 2>&1 | tail -60" 2>/dev/null \
      | grep -qE "Traceback|Engine core initialization failed|CUDA out of memory|no module or parameter named"; then
    echo "  ✗ engine error during boot — last 40 lines:" >&2
    ssh -o BatchMode=yes "$HEAD" "docker logs $NAME 2>&1 | tail -40" >&2
    echo >&2
    echo "  Triage:" >&2
    echo "   · 'no module or parameter named aligner'  -> vision mod did NOT apply" >&2
    echo "   · ValueError near determine_available_memory / b12x_mhc_pre" >&2
    echo "        -> rms_norm_eps override missing; confirm the mod ran on BOTH nodes" >&2
    echo "   · CUDA out of memory -> retry with GMU=0.82" >&2
    exit 1
  fi
  sleep 10
done
(( ready == 1 )) || { echo "  ✗ not ready after ${READY_TIMEOUT}s" >&2; exit 1; }
echo "  ✓ endpoint up"

# ── Gate 1: the mod actually applied, on BOTH nodes ──────────────────────────
# Check the PATCHED FILE inside each container, not `docker logs`. The mod runs
# during launch-cluster.sh's apply_mod_to_container, BEFORE the serving process
# starts, so its success line goes to the launcher's stdout and never appears in
# the container's log stream — grepping docker logs for it always fails.
echo "  ▶ verifying vision mod applied on both nodes…"
mod_ok=1
MODEL_PY=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py
for host in "$HEAD" "$WORKER"; do
  if ssh -o BatchMode=yes "$host" \
      "docker exec $NAME grep -q 'vision-exp-hotfix' $MODEL_PY && docker exec $NAME test -f /opt/dspark-patches/vision_exp/apply.py" 2>/dev/null; then
    echo "    ✓ $host: model.py carries the hotfix marker and vision_exp/ is staged"
  else
    echo "    ✗ $host: vision mod NOT present in the running container" >&2; mod_ok=0
  fi
done

# ── Gate 2: the served model is registered MULTIMODAL, not text-only ─────────
echo "  ▶ verifying the model is registered multimodal (fail-open guard)…"
mm=$(ssh -o BatchMode=yes "$HEAD" "docker exec $NAME python3 -c \"
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM as C
from vllm.model_executor.models.interfaces import supports_multimodal
print(supports_multimodal(C))\" 2>/dev/null | tr -d '[:space:]'" 2>/dev/null || true)
if [[ "$mm" == "True" ]]; then
  echo "    ✓ supports_multimodal == True"
else
  echo "    ✗ supports_multimodal == '${mm:-<no answer>}' — this server may be TEXT-ONLY" >&2
  mod_ok=0
fi

# ── Gate 3: all three ids are served, and the canonical one is first ─────────
echo "  ▶ verifying all three model ids resolve…"
if ! curl -fsS -m 10 "http://$HEAD:$PORT/v1/models" -o /tmp/.vision_models.$$ 2>/dev/null; then
  echo "    ✗ could not read /v1/models" >&2; mod_ok=0
else
  python3 - /tmp/.vision_models.$$ "$MODEL_ID" <<'PY' || mod_ok=0
import json, sys
data = json.load(open(sys.argv[1]))["data"]
ids = [m["id"] for m in data]
want = [sys.argv[2],
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "deepseek-ai/DeepSeek-V4-Flash"]
ok = True
for w in want:
    if w in ids:
        print(f"    ✓ {w}")
    else:
        print(f"    ✗ MISSING: {w}"); ok = False
if ids and ids[0] != want[0]:
    print(f"    ✗ data[0] is {ids[0]!r}, expected the canonical id first"); ok = False
extra = [i for i in ids if i not in want]
if extra:
    print(f"    · also serving: {', '.join(extra)}")
raise SystemExit(0 if ok else 1)
PY
  rm -f /tmp/.vision_models.$$
fi

# Each alias must also actually ANSWER, not merely be listed.
echo "  ▶ verifying each id answers a request…"
for mid in "$MODEL_ID" "deepseek-ai/DeepSeek-V4-Flash-0731" "deepseek-ai/DeepSeek-V4-Flash"; do
  code=$(curl -fsS -m 120 -o /dev/null -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$mid\",\"max_tokens\":1,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
      "http://$HEAD:$PORT/v1/chat/completions" 2>/dev/null || echo "000")
  if [[ "$code" == "200" ]]; then echo "    ✓ $mid -> 200"
  else echo "    ✗ $mid -> HTTP $code" >&2; mod_ok=0; fi
done

echo
echo "✓ up. Endpoint: http://$HEAD:$PORT/v1"
echo "  served ids: $(curl -fsS -m 5 "http://$HEAD:$PORT/v1/models" | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || echo '?')"
echo "  ⚠ the two 0731 ids above are ALIASES onto Vision-Exp — not the 0731 checkpoint."
echo "  KV pool:    $(ssh -o BatchMode=yes "$HEAD" "docker logs $NAME 2>&1 | grep -E 'Available KV cache memory|GPU KV cache size|Maximum concurrency' | tail -3 | tr '\n' ' '" 2>/dev/null)"
echo "  logs:       ssh $HEAD 'docker logs -f $NAME'"
echo "  stop:       $SCRIPT_DIR/stop-deepseek-v4-flash-vision-exp-b12x-eugr.sh"
echo
echo "  Expected: a SMALLER pool than production's ~2.0M tokens (ViT is replicated per"
echo "  TP rank). Judge capacity by 'Available KV cache memory' + max_concurrency, never"
echo "  by the 'GPU KV cache size' headline (that is concurrency x max_model_len)."

if (( SMOKE_IMAGE == 1 )); then
  echo
  echo "  ▶ image smoke test (a real 8x8 PNG, in a USER message — the only legal role)…"
  python3 - "$HEAD" "$PORT" "$MODEL_ID" <<'PY' || echo "    ✗ image smoke test FAILED (see above)"
import base64, json, struct, sys, urllib.request, zlib
host, port, model = sys.argv[1], sys.argv[2], sys.argv[3]
# 8x8 solid red PNG, built inline so the test needs no fixture on disk.
def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))
b64 = base64.b64encode(png(8, 8, (255, 0, 0))).decode()
body = {"model": model, "max_tokens": 24, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What colour is this image? Answer with one word."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}
req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=180))
except urllib.error.HTTPError as e:
    print("    ✗ HTTP", e.code, e.read().decode()[:300]); print("      A 400 here usually means the image never reached a multimodal path."); raise SystemExit(1)
txt = r["choices"][0]["message"]["content"]
print("    model replied:", repr(txt.strip()[:120]))
print("    ✓ image ACCEPTED and answered correctly" if "red" in txt.lower()
      else "    ✗ image accepted but answer is wrong — suspect a text-only fail-open path")
raise SystemExit(0 if "red" in txt.lower() else 1)
PY
fi

if (( mod_ok == 0 )); then
  echo
  echo "✗ SERVER IS UP BUT THE VISION GATES DID NOT PASS — treat it as TEXT-ONLY." >&2
  exit 2
fi
