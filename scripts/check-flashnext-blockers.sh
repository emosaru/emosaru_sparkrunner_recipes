#!/usr/bin/env bash
# Are the upstream bugs that keep Qwen3.8-Flash-Next off our production stack fixed yet?
#
# WHY THIS EXISTS (2026-09-18): we decided NOT to migrate Flash-Next onto the "community"
# stack because it carried open GB10/SM12x bugs in prefix caching and MTP.
#
# *** CORRECTED 2026-09-24: BOTH "BLOCKERS" WERE ALREADY FIXED, AND NEITHER ISSUE WAS
# *** EVER CLOSED. Watching issue STATE was therefore watching a signal that will never
# *** fire. This script now tracks the FIX PRs, which are merged, and says so.
#
#   #54173 (prefix caching illegal-access on GB10 sm_121)
#       -> fixed incidentally by PR #53906, merged 2026-09-03. Root cause: on hybrid
#          models whose QSA indexer registers a CircularBufferSpec ring, block_size took
#          the ring's 4/8-token size instead of the ~1,600-token Mamba block, so a
#          resumed Mamba state index was seeded past the block table.
#   #55357 (MTP episodic 0% acceptance + repetition collapse)
#       -> fixed by PR #55375, merged 2026-09-05. Prefill state slots were passed as the
#          strided column view state_indices_tensor_p[:, 0] while the PLE short-conv
#          kernel loads the slot with unit stride.
#
# Our b12x images are built from main and postdate both merges (dev20610 @ 2026-09-10,
# dev21504 @ 2026-09-23), which is why we have never reproduced either bug and why our
# prefix-cache hit rate is healthy. Verified empirically 2026-09-24: prefix caching on
# Flash-Next works (identical prompt 14,465 cached, 6.4x speedup; extension reuse works
# at 12,288-token granularity), and a 93k needle passes.
#
# So these four entries are now INFORMATIONAL. The decision they were watching for has
# already been made on measured data — see memory/flashnext_bakeoff_2026_09_24.md.
#
# Read-only: queries the GitHub API and prints. Changes nothing, touches no node.
set -euo pipefail

# Fix PRs whose MERGED state is the thing that actually mattered.
FIXES=(
  "53906:fixes #54173 (prefix caching / Mamba state index block size) — merged 2026-09-03"
  "55375:fixes #55357 (PLE conv state index strides under MTP)      — merged 2026-09-05"
)
printf '%s\n' "Flash-Next fix PRs — $(date -Is)"
printf '%s\n' "──────────────────────────────────────────────────────────"
for entry in "${FIXES[@]}"; do
  num="${entry%%:*}"; why="${entry#*:}"
  if json=$(gh api "repos/vllm-project/vllm/pulls/${num}" 2>/dev/null); then
    merged=$(jq -r '.merged' <<<"$json"); at=$(jq -r '.merged_at' <<<"$json")
    printf '  %s PR #%s merged=%s (%s)\n      %s\n\n' \
      "$([[ $merged == true ]] && echo ✔ || echo ✗)" "$num" "$merged" "$at" "$why"
  else
    printf '  PR #%s QUERY FAILED\n\n' "$num"
  fi
done

printf '%s\n' "Tracked issues (still OPEN upstream, but fixed in main — do not read state as a blocker):"

ISSUES=(
  "54173:FIXED by #53906 — prefix caching crashes on GB10 sm_121 (our silicon, our workload:
         growing multi-turn shared-prefix requests). Only workaround is disabling APC,
         at ~23.5s re-prefill per turn on 55K. This is why several community Flash-Next
         recipes run --no-enable-prefix-caching."
  "55357:FIXED by #55375 — MTP episodic 0% draft acceptance + repetition collapse lasting hours.
         Reported on SM120 (RTX PRO 6000), not our SM121 — related silicon, not identical."
  "55122:OPEN, nice-to-have — deterministic persistent_topk on GB10. Without it identical greedy
         requests can differ. Would also let us drop a local topk workaround."
  "54426:OPEN, nice-to-have — fp8_e4m3 KV on the QSA path; still an RFC seeking corroboration.
         We already get fp8 KV from b12x, so this only matters if we ever leave it."
)

changed=0
for entry in "${ISSUES[@]}"; do
  num="${entry%%:*}"
  why="${entry#*:}"
  if ! json=$(gh api "repos/vllm-project/vllm/issues/${num}" 2>/dev/null); then
    printf '  #%s  QUERY FAILED (network, rate limit, or auth)\n\n' "$num"
    continue
  fi
  state=$(jq -r '.state' <<<"$json")
  title=$(jq -r '.title' <<<"$json")
  updated=$(jq -r '.updated_at' <<<"$json")
  mark="·"
  if [[ "$state" == "closed" ]]; then
    mark="✔"
    changed=1
  fi
  printf '  %s #%s  [%s]  last updated %s\n' "$mark" "$num" "$state" "$updated"
  printf '      %s\n' "$title"
  printf '      %s\n\n' "$why"
done

if [[ "$changed" == "1" ]]; then
  echo "  (An issue was closed upstream — bookkeeping only; both fixes already landed.)"
fi
cat <<'MSG'
  REMINDER: issue state here is NOT the migration signal. Both blockers are fixed in
  main and shipped in our images. The Flash-Next decision was settled on measured data
  2026-09-24 (dual TP=2+EP: 43.7 prose / 60.5 code tok/s, 2.50M KV, 9.52x, 5/5 vision).
  See memory/flashnext_bakeoff_2026_09_24.md.
MSG
