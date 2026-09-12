# drop-caches-fixed

Fork of `spark-vllm-docker-eugr/mods/drop-caches` (upstream unchanged as of 2026-09-11).

**Upstream bug:** the loop body is
`sync; echo 3 > /proc/sys/vm/drop_caches >> "$LOG" 2>&1`.
Bash applies both stdout redirections in order, so fd 1 ends up on `$LOG`; the literal
`3` is written to the log and the proc file is opened but never written. Caches are
therefore never dropped. The `sync` does still run, so the dirty-writeback benefit is
real upstream — only the drop is missing.

**Evidence (2026-09-11, this cluster):** `/tmp/drop_caches.log` inside
`vllm_node_flashnext` contained exactly `3` (via `cat -A`: `3$`), while a direct
`echo 3 > /proc/sys/vm/drop_caches` in the same container returned success — ruling out
a permissions cause.

**Changes:** write to the proc file for real; log timestamped `dropped` / `DROP FAILED`
lines instead of the payload; and abort at apply time if the first drop fails, so an
inert memory mod cannot be silently trusted.

**Why we care:** both reasons are GB10-specific — a warm page cache starves the GPU
allocator during model load on unified memory, and writeback backlog eats the host
margin that `earlyoom` polices (see `memory/earlyoom_head_kill.md`, which cost us the
serving engine twice).

Worth reporting upstream.
