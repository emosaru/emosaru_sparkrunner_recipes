#!/usr/bin/env bash
# stress_cron.sh — 24h day-scale stability monitor for DeepSeek-V4-Flash on 2x GB10.
#
# Crashes historically appeared after ~1 day of uptime under Hermes cron-burst load. A single
# ~20-min soak/stress pass proved the ACUTE failures (the #40969 hang, OOM-under-concurrency) are
# fixed, but NOT a slow leak (the al-engr pinned-host-alloc leak in vLLM >=0.23 that silently
# hard-freezes over hours). Signature = available unified RAM trending DOWN across a day.
#
# This runs a short concurrency stress every 30 min for 24h and appends a one-line summary to
# bench_results/stress_24h.log so the mem_avail_min trend is visible. Self-removes its own cron
# line after 24h. Install: `*/30 * * * * /home/mebell/code/.../stress_cron.sh >> .../stress_cron.out 2>&1`
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin

REPO=/home/mebell/code/emosaru_sparkrunner_recipes
RESULTS=$REPO/bench_results
LOG=$RESULTS/stress_24h.log
MARKER=$RESULTS/.stress_24h_start
BASE=http://192.168.1.236:8000
mkdir -p "$RESULTS"

# Establish the 24h window on first run.
if [ ! -f "$MARKER" ]; then
  date +%s > "$MARKER"
  echo "$(date -u +%FT%TZ)  [START] 24h day-scale stress monitor begins (every 30 min)" >> "$LOG"
fi
START=$(cat "$MARKER" 2>/dev/null || echo 0)
ELAPSED=$(( $(date +%s) - START ))

# After 24h: log final, remove our cron line, clean the marker, stop.
if [ "$ELAPSED" -ge 86400 ]; then
  echo "$(date -u +%FT%TZ)  [DONE] 24h window complete — removing cron line" >> "$LOG"
  ( crontab -l 2>/dev/null | grep -v 'stress_cron.sh' | crontab - ) || true
  rm -f "$MARKER"
  exit 0
fi

# Skip (don't fail) if the endpoint isn't serving this tick.
if ! curl -fsS -m 5 "$BASE/v1/models" >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ)  [DOWN] endpoint not serving — skipped" >> "$LOG"
  exit 0
fi

# Short concurrency stress (c=8,16 x 60s). timeout guards against a wedge hanging the tick.
LABEL="cron-$(date +%m%d-%H%M)"
timeout 600 python3 "$REPO/soak_bench.py" --base-url "$BASE" --label "$LABEL" --mode stress \
  --stress-levels 8,16 --stress-duration 60 --stress-sizes 500,4000,8000 \
  --max-tokens 128 --hang-timeout 120 >/dev/null 2>&1
rc=$?

# Summarize the JSON this run wrote into a single trend line.
JSON=$(ls -t "$RESULTS/${LABEL}-stress-"*.json 2>/dev/null | head -1)
HRS=$(awk "BEGIN{printf \"%.1f\", $ELAPSED/3600}")
python3 - "$JSON" "$HRS" "$rc" >> "$LOG" <<'PY'
import json, sys, time
ts = time.strftime('%Y-%m-%dT%H:%M:%S')
jsonf, hrs, rc = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2], sys.argv[3]
if rc == "124":
    print(f"{ts}  [+{hrs}h] [TIMEOUT] stress run exceeded 600s (possible wedge)"); sys.exit()
try:
    d = json.load(open(jsonf))["results"]["stress"]
    mrs = [l["resources"].get("mem_avail_min") for l in d["levels"]]
    mrs = [m for m in mrs if m is not None]
    minram = min(mrs) if mrs else None
    parts = " ".join(
        f"c{l['concurrency']}={l['verdict']}({l['ok']}ok/{l['fail']}f,{l['agg_tps']}tps,ram{l['resources'].get('mem_avail_min')})"
        for l in d["levels"])
    print(f"{ts}  [+{hrs}h] [{d['verdict']}] min_ram={minram}GB  {parts}")
except Exception as e:
    print(f"{ts}  [+{hrs}h] [PARSE-ERR] {type(e).__name__}: {e} (json={jsonf})")
PY
