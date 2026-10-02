#!/usr/bin/env bash
# flashnext-supervisor — keep Qwen3.8-Flash-Next serving across the known engine deaths.
#
# WHY THIS EXISTS
# The engine dies with `CUDA error: an illegal memory access was encountered` (GPU-level:
# `Xid 31 ... ACCESS_TYPE_VIRT_WRITE`) under cumulative long-context traffic. On the
# current best config (CKPT_POLICY=aligned, the launcher default) real-workload MTTF was
# ~3h / ~394 requests. No fix exists: the faulting subsystem is partly fork-only, no
# upstream PR applies, and a speculative kernel patch could not be confirmed and would
# risk silent state corruption. See memory/flashnext_cuda_illegal_access_2026_09_25.md.
#
# Recovery takes ~5 minutes, so at ~3h MTTF a supervisor makes this largely invisible.
# THIS IS NOT A FIX. It converts a hard outage into a ~5 minute gap.
#
# CRITICAL: the container stays `Up` with a dead engine, so `docker ps` is useless as a
# health signal. This polls the HTTP endpoint, which is the only reliable check.
#
# USAGE
#   scripts/flashnext-supervisor.sh start     # detach and supervise
#   scripts/flashnext-supervisor.sh status
#   scripts/flashnext-supervisor.sh stop      # stop supervising (leaves the engine up)
#   scripts/flashnext-supervisor.sh run       # run in the foreground (debugging)
#
# TO TAKE THE STACK DOWN YOURSELF, stop the supervisor FIRST:
#   scripts/flashnext-supervisor.sh stop && ./stop-qwen38-flash-next-b12x-eugr.sh
# Otherwise it will helpfully undo your shutdown. A sentinel file is also honoured:
# `touch .flashnext-logs/supervisor.pause` makes it idle without exiting.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGDIR="$REPO/.flashnext-logs"
LOG="$LOGDIR/supervisor.log"
PIDFILE="$LOGDIR/supervisor.pid"
PAUSE="$LOGDIR/supervisor.pause"

HEAD="${HEAD:-192.168.1.236}"
WORKER="${WORKER:-192.168.1.237}"
PORT="${PORT:-8000}"
URL="http://$HEAD:$PORT/v1/models"

POLL_SECS="${POLL_SECS:-30}"
# Require several consecutive failures: a single curl can fail during a long prefill or a
# transient hiccup, and relaunching a HEALTHY engine would be worse than waiting.
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"
CURL_TIMEOUT="${CURL_TIMEOUT:-15}"
# Relaunch-storm guard: if we restart this many times inside the window, stop and leave it
# down. A boot-time failure must not loop forever unattended.
MAX_RESTARTS="${MAX_RESTARTS:-4}"
RESTART_WINDOW_SECS="${RESTART_WINDOW_SECS:-7200}"

mkdir -p "$LOGDIR"

# Writes to stderr only. `start` redirects stderr into $LOG, so teeing here would
# duplicate every line; in foreground `run` mode you see it directly.
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

healthy() { curl -fsS -m "$CURL_TIMEOUT" "$URL" 2>/dev/null | grep -q "Flash-Next"; }

capture() {
  # Perishable: `docker logs` dies with the container, and the stop script removes it.
  # Always grab BOTH nodes and BOTH kernel logs -- the real fault is often only the Xid,
  # with no `illegal memory access` in the Python log at all.
  local d="$LOGDIR/crash-$(date +%Y%m%dT%H%M%S)-supervisor"
  mkdir -p "$d"
  docker logs vllm_node_flashnext >"$d/236_head.log" 2>&1 || true
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" \
      "docker logs vllm_node_flashnext" >"$d/237_worker.log" 2>&1 || true
  journalctl -k --since "3 hours ago" 2>/dev/null | grep -i xid >"$d/236_kernel.log" 2>&1 || true
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" \
      "journalctl -k --since '3 hours ago' 2>/dev/null | grep -i xid" >"$d/237_kernel.log" 2>&1 || true
  {
    echo "captured by flashnext-supervisor at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "requests served this boot: $(grep -c 'POST /v1/chat/completions' "$d/236_head.log" 2>/dev/null | head -1)"
    echo "boot:  $(grep -m1 'Using V2 Model Runner' "$d/236_head.log" 2>/dev/null | grep -oE '[0-9]+-[0-9]+ [0-9:]+' || echo '?')"
    echo "signature:"
    grep -oE "illegal memory access|RuntimeError: cancelled|EngineDeadError|TimeoutError" \
      "$d/236_head.log" 2>/dev/null | sort | uniq -c | sed 's/^/  /' || true
    echo "in-flight at death:"
    grep -o 'num_computed_tokens=\[[^]]*\]' "$d/236_head.log" 2>/dev/null | tail -2 | sed 's/^/  /' || true
    echo "  (NOTE: this is just the request in flight -- it is NOT a causal length."
    echo "   The 'exact multiple of 2848' theory was tested and retracted.)"
    echo "Xid (head):   $(grep -oE 'faulted @ 0x[0-9a-f_]+' "$d/236_kernel.log" 2>/dev/null | tail -1)"
    echo "Xid (worker): $(grep -oE 'faulted @ 0x[0-9a-f_]+' "$d/237_kernel.log" 2>/dev/null | tail -1)"
  } >"$d/SUMMARY.txt" 2>&1
  log "captured diagnostics -> $d"
}

supervise() {
  log "supervisor starting: url=$URL poll=${POLL_SECS}s threshold=$FAIL_THRESHOLD"
  log "  guard: max $MAX_RESTARTS restarts per ${RESTART_WINDOW_SECS}s"
  local fails=0
  local -a restarts=()

  while :; do
    if [[ -f "$PAUSE" ]]; then
      log "paused (remove $PAUSE to resume)"
      sleep "$POLL_SECS"; continue
    fi

    if healthy; then
      [[ $fails -gt 0 ]] && log "recovered after $fails failed poll(s)"
      fails=0
      sleep "$POLL_SECS"; continue
    fi

    fails=$((fails + 1))
    log "health check FAILED ($fails/$FAIL_THRESHOLD)"
    if [[ $fails -lt $FAIL_THRESHOLD ]]; then
      sleep "$POLL_SECS"; continue
    fi

    # Prune restart history outside the window, then enforce the storm guard.
    local now; now=$(date +%s)
    local -a kept=()
    for ts in "${restarts[@]:-}"; do
      [[ -n "$ts" ]] && (( now - ts < RESTART_WINDOW_SECS )) && kept+=("$ts")
    done
    restarts=("${kept[@]:-}")
    local count=0
    for ts in "${restarts[@]:-}"; do [[ -n "$ts" ]] && count=$((count + 1)); done
    if (( count >= MAX_RESTARTS )); then
      log "ABORT: $count restarts within ${RESTART_WINDOW_SECS}s — not looping. Engine left"
      log "       down for inspection. Fix, then: $0 start"
      capture
      rm -f "$PIDFILE"
      exit 1
    fi

    log "engine confirmed dead — capturing, then relaunching"
    capture
    "$REPO/stop-qwen38-flash-next-b12x-eugr.sh" >>"$LOG" 2>&1 || log "stop script returned nonzero (continuing)"
    # Bare launch: the committed defaults are the config we want (CKPT_POLICY=aligned).
    # The launcher blocks until the endpoint answers or READY_TIMEOUT expires.
    if "$REPO/launch-qwen38-flash-next-b12x-eugr.sh" >>"$LOG" 2>&1; then
      log "relaunch returned ok"
    else
      log "relaunch returned NONZERO — will re-check and retry under the storm guard"
    fi
    restarts+=("$(date +%s)")
    fails=0
    if healthy; then log "engine healthy again"; else log "still not healthy after relaunch"; fi
    sleep "$POLL_SECS"
  done
}

case "${1:-status}" in
  run) supervise ;;
  start)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE"))"; exit 0
    fi
    rm -f "$PAUSE"
    nohup "$0" run >>"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "supervisor started (pid $(cat "$PIDFILE")); log: $LOG"
    ;;
  stop)
    if [[ -f "$PIDFILE" ]]; then
      pid=$(cat "$PIDFILE")
      kill "$pid" 2>/dev/null && echo "stopped supervisor (pid $pid)" || echo "pid $pid not running"
      rm -f "$PIDFILE"
    else
      echo "no pidfile; nothing to stop"
    fi
    ;;
  status)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
      echo "supervisor: RUNNING (pid $(cat "$PIDFILE"))"
    else
      echo "supervisor: not running"
    fi
    [[ -f "$PAUSE" ]] && echo "state: PAUSED ($PAUSE present)"
    if healthy; then echo "engine: UP"; else echo "engine: DOWN"; fi
    echo "restarts logged: $(grep -c 'engine confirmed dead' "$LOG" 2>/dev/null | head -1)"
    tail -3 "$LOG" 2>/dev/null | sed 's/^/  /'
    ;;
  *) echo "usage: $0 {start|stop|status|run}" >&2; exit 2 ;;
esac
