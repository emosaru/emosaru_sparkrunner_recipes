#!/bin/bash
# Drop the FS page cache every 60s — CORRECTED FORK of eugr's mods/drop-caches.
#
# ── WHY THIS FORK EXISTS ──────────────────────────────────────────────────────
# Upstream's loop body is:
#     sync; echo 3 > /proc/sys/vm/drop_caches >> "$LOG" 2>&1
# stdout is redirected TWICE and the LAST redirection wins, so `echo 3` writes "3"
# into the LOG FILE and /proc/sys/vm/drop_caches is merely opened (O_WRONLY|O_TRUNC)
# and never written. Opening it has no effect — dropping requires an actual write.
# Net upstream behaviour: `sync` runs (dirty pages do get flushed), but NO CACHE IS
# EVER DROPPED. Verified 2026-09-11 on this cluster: /tmp/drop_caches.log contained
# exactly the byte "3", and a direct `echo 3 > /proc/sys/vm/drop_caches` inside the
# same container succeeded — so it is the redirection, not privilege.
#
# Fix: write to the proc file, and send only diagnostics to the log.
set -u
LOG="/tmp/drop_caches.log"
PIDFILE="/tmp/drop_caches.pid"

# Fail loudly at apply time if we cannot actually drop — a silently inert memory mod is
# worse than none, because it gets trusted. This is the check upstream lacks.
if ! sync || ! echo 3 > /proc/sys/vm/drop_caches 2>>"$LOG"; then
  echo "drop-caches-fixed: CANNOT write /proc/sys/vm/drop_caches (needs privileged container)." | tee -a "$LOG" >&2
  exit 1
fi

nohup bash -c '
  while true; do
    sync
    if echo 3 > /proc/sys/vm/drop_caches 2>>"'"$LOG"'"; then
      echo "$(date -Is) dropped" >> "'"$LOG"'"
    else
      echo "$(date -Is) DROP FAILED" >> "'"$LOG"'"
    fi
    sleep 60
  done
' >/dev/null 2>&1 &

echo $! > "$PIDFILE"
echo "drop-caches-fixed: started loop PID $(cat "$PIDFILE"); log $LOG (expect one 'dropped' line/min)"
