#!/bin/bash
# bound-accepted-token-state-lookups — port of vLLM PR #50021 (OPEN, unmerged).
#
# Bounds every block-table column that is derived from a per-request accepted-token
# count before it is turned into an address, across the GDN/KDA/Mamba spec-decode
# kernels. Upstream's own description of the failure mode:
#
#   "speculative decoding generates per-request accepted-token counts that multiple GPU
#    kernels convert into array indices without bounds checking; stale, zero, or
#    oversized counts cause out-of-bounds reads, dereferencing wild addresses and
#    faulting the GPU"
#
# ...and, critically, from the patched comment itself: an out-of-range column becomes a
# wild address "rather than MERELY A WRONG COPY". That is the single mechanism behind
# both of our symptoms — silent recurrent-state corruption (degenerate constant-token
# reasoning loops) when the stray column lands inside the cache, and
# `CUDA error: an illegal memory access was encountered` when it lands outside.
# See memory/flashnext_cuda_illegal_access_2026_09_25.md.
#
# This is a DEFENSIVE masking patch: out-of-range columns load an `other=-1` sentinel
# and are then rejected by `<= 0` guards, so they fall into pre-existing
# invalid-state paths instead of producing addresses. No stream-ordering or
# synchronisation changes, so it cannot deadlock or reorder; worst case it is inert.
#
# Pure Python/Triton — 7 files, no CUDA/C++ — which is why this is a mod and NOT an
# image rebuild. Both Sparks were at 92%/96% disk when this was written; a 36 GB
# rebuild did not fit, and does not need to.
set -euo pipefail

TAG="[bound-accepted-token-state-lookups]"
PATCH_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pr50021-bound-accepted-token-lookups.patch"
[ -f "$PATCH_FILE" ] || { echo "$TAG ERROR: patch file missing: $PATCH_FILE" >&2; exit 1; }

VLLM_DIR="$(python3 -c 'import os,vllm; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
[ -n "$VLLM_DIR" ] && [ -d "$VLLM_DIR" ] || { echo "$TAG ERROR: cannot locate the vllm package" >&2; exit 1; }
SITE_DIR="$(dirname "$VLLM_DIR")"
SENTINEL="$VLLM_DIR/v1/worker/mamba_utils.py"
[ -f "$SENTINEL" ] || { echo "$TAG ERROR: $SENTINEL not found" >&2; exit 1; }

# Idempotent, and forward-compatible: `dst_col_ok` is the bound introduced by #50021,
# so its presence means the guard is already in place — whether from a previous run of
# this mod or because upstream finally merged the PR. Either way, do nothing.
if grep -q "dst_col_ok" "$SENTINEL"; then
  echo "$TAG already bounded (dst_col_ok present) — nothing to do."
  exit 0
fi

command -v patch >/dev/null || { echo "$TAG ERROR: no 'patch' binary in this image" >&2; exit 1; }

# Refuse to half-apply. Dry-run first: if the base has drifted, say so and change
# nothing, rather than leaving the kernels in a mixed state.
if ! patch -p1 --dry-run --directory="$SITE_DIR" < "$PATCH_FILE" >/tmp/bound_accepted_dryrun.log 2>&1; then
  echo "$TAG ERROR: patch does not apply to this vLLM build — base has drifted." >&2
  echo "$TAG This port was generated against v0.1.dev21504+g57fdda71b (eugr nightly-20260923)." >&2
  echo "$TAG Re-port against the new base before using it; dry-run output:" >&2
  sed 's/^/  /' /tmp/bound_accepted_dryrun.log >&2
  exit 1
fi

patch -p1 --directory="$SITE_DIR" < "$PATCH_FILE"

# Drop stale bytecode for the touched modules so the patched source is what loads.
# (.pyc invalidation is mtime+size based and patch changes both, but these are Triton
# kernel modules and a silently stale one would be indistinguishable from an inert mod.)
while read -r rel; do
  d="$SITE_DIR/$(dirname "$rel")"; b="$(basename "$rel" .py)"
  rm -f "$d/__pycache__/$b."*.pyc 2>/dev/null || true
done < <(grep '^+++ ' "$PATCH_FILE" | sed 's|^+++ work/||; s|\t.*||')

grep -q "dst_col_ok" "$SENTINEL" || { echo "$TAG ERROR: applied but sentinel missing — aborting." >&2; exit 1; }
echo "$TAG applied vLLM PR #50021 (ported): accepted-token-derived block-table columns are now bounded."
