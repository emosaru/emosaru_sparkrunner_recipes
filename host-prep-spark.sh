#!/usr/bin/env bash
# host-prep-spark.sh — Phase-1 unified-memory / OOM stability hardening for the DGX Spark nodes.
#
# Run on BOTH nodes (192.168.1.236 AND 192.168.1.237) as root, immediately BEFORE launching the
# DeepSeek-V4-Flash stack (or any large 2-node model). Idempotent; safe to re-run. Intended to
# run while the target stack is DOWN — it drops page cache and raises the free-page reserve, which
# is disruptive to reclaim mid-serving.
#
# Fixes the unified-memory OOM / earlyoom crash class documented for GB10:
#   - reserve a protected free-page pool so page cache can't consume the last allocation margin
#   - drop clean page cache accumulated from prior ~148 GB model loads
#   - bias the kernel against large dirty-page buildup during weight load
#   - stop earlyoom from reaping the vLLM EngineCore (the min_free reserve replaces its guard)
# Refs: tobias-weiss OOM debugging; conselara SM121 gotchas; NVIDIA forum 363989. See
# /home/mebell/.claude/plans/deepseek-v4-flash-stability-plan.md (Phase 1).
set -euo pipefail

MIN_FREE_KB="${MIN_FREE_KB:-3145728}"        # 3 GiB protected reserve (DS-V4-Flash needs the headroom for KV; raise to 5242880=5 GiB if not memory-constrained)
EXPECTED_DRIVER="${EXPECTED_DRIVER:-580.159.03}"   # node 0 as of 2026-07-09; BOTH nodes must match

if [ "$(id -u)" -ne 0 ]; then
  echo "re-exec under sudo…"; exec sudo -E "$0" "$@"
fi

echo "▶ host-prep-spark on $(hostname) ($(date -u +%FT%TZ))"

# 1) Driver sanity — a cross-node mismatch is a ~2.4x perf + stability problem; 590.x has a
#    reported CUDAGraph deadlock on GB10.
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
echo "  driver: $drv (expected $EXPECTED_DRIVER)"
[ "$drv" = "$EXPECTED_DRIVER" ] || echo "  ⚠ driver != expected — ensure BOTH nodes match EXACTLY."
case "$drv" in 590.*) echo "  ⚠ 590.x has a reported CUDAGraph deadlock on GB10 — prefer the 580.x line.";; esac

echo "  free BEFORE prep:"; free -g | sed 's/^/    /'

# 2) Emergency free-page reserve (tobias-weiss: single 'most impactful' setting).
sysctl -w vm.min_free_kbytes="$MIN_FREE_KB"
# 3) Keep dirty pages from competing with GPU allocations during model load.
sysctl -w vm.dirty_ratio=5
sysctl -w vm.dirty_background_ratio=2
sysctl -w vm.vfs_cache_pressure=200

# 4) Drop clean page cache (recovers memory eaten by prior model loads).
sync
echo 3 > /proc/sys/vm/drop_caches

# 5) Stop earlyoom — it can kill the vLLM EngineCore under transient pressure. The min_free_kbytes
#    reserve above is the safer replacement guard. Add `systemctl disable earlyoom` to persist.
if systemctl is-active --quiet earlyoom 2>/dev/null; then
  systemctl stop earlyoom
  echo "  earlyoom stopped (run 'sudo systemctl disable earlyoom' to persist across reboot)."
else
  echo "  earlyoom not active."
fi

# 6) Clear stale shm / zombie IPC that can wedge a restart.
rm -f /dev/shm/psm_* /dev/shm/sem.mp-* 2>/dev/null || true

echo "  free AFTER prep:"; free -g | sed 's/^/    /'
echo "✓ host-prep-spark done on $(hostname)"

# =============================================================================================
# OPTIONAL — NCCL/RoCE pinning. Enable ONLY if NCCL is falling back to TCP (symptom: ~12 tok/s
# multi-node, or shm_broadcast/all-reduce stalls). NOT enabled by default: the current network
# path starts and runs, and a wrong iface silently degrades to slow TCP.
#
# Inspected on node 0 (192.168.1.236) — two cabled RoCE links on SEPARATE /24s (best practice),
# second (f1) ports DOWN/uncabled:
#   enp1s0f0np0   192.168.0.236   HCA rocep1s0f0      (cabled)      | enp1s0f1np1  DOWN
#   enP2p1s0f0np0 192.168.2.236   HCA roceP2p1s0f0    (cabled)      | enP2p1s0f1np1 DOWN
# Confirm node 1 has matching 192.168.0.237 / 192.168.2.237, then export in the CONTAINER env
# (pin to ONE cabled HCA — the empty GID on the uncabled f1 port stalls the NCCL handshake):
#   NCCL_IB_DISABLE=0
#   NCCL_IB_HCA=rocep1s0f0
#   NCCL_IB_GID_INDEX=3              # verify RoCEv2/IPv4 entry: `show_gids | grep rocep1s0f0`
#   NCCL_SOCKET_IFNAME=enp1s0f0np0
#   GLOO_SOCKET_IFNAME=enp1s0f0np0
#   TP_SOCKET_IFNAME=enp1s0f0np0
#   NCCL_IGNORE_CPU_AFFINITY=1
#   NCCL_IB_TIMEOUT=22
# =============================================================================================
