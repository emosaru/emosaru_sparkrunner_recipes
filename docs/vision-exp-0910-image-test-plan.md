# Test plan — DeepSeek-V4-Flash-Vision-Exp on the 2026-09-10 B12X image

**Status: NOT RUN.** Written 2026-09-12. Production is serving Vision-Exp on the **2026-08-23**
image (`vllm-node-b12x:latest`) via `deepseek-v4-flash-vision-exp-b12x-eugr.yaml` + our vendored
`mods/deepseek-v4-vision-exp`. Nothing here has been executed.

Candidate: `recipes/deepseek-v4-flash-vision-exp-b12x-0910-eugr.yaml`.

## Why bother

Primarily **stability, not speed**: the new image implements Vision-Exp natively, letting us drop
a vendored 5-file mod that monkeypatches model internals and hard-fails on every image refresh.
It already caused one failed production restore (2026-09-11). Secondary, unmeasured: new
`b12x_indexer.py` and `fi_moe.py` modules, ~480 vLLM commits, and eugr's claim that the b12x
loader is "faster and more memory efficient than InstantTensor on DGX Spark".

## Preconditions

- [ ] A real maintenance window. This takes production down; both stacks need the whole cluster.
- [ ] Both images present on BOTH nodes (verified 2026-09-11):
      `vllm-node-b12x:latest` = 7dc02f162929 (2026-08-23, **rollback**)
      `eugr/spark-vllm-b12x:latest` = 693a1d778e99 (2026-09-10, candidate)
- [ ] **Quiesce the LiteLLM proxy** (192.168.1.32, ~4.6 req/min) or accept that every
      throughput number is contended. `scripts/bench-flashnext.py` gates on
      `vllm:num_requests_running` and discards contended samples — do not bypass that.
- [ ] Capture a same-session BASELINE on the OLD image first (below). Today's 1.60M-token KV
      reading is NOT a usable baseline — the historical figure was 1.90M and we don't know why
      it differs, so measure both stacks back-to-back in one window.

## Step 0 — baseline on the CURRENT image (do not skip)

```bash
python3 scripts/verify-vision-exp-correctness.py          # expect 13/13
python3 scripts/bench-flashnext.py --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
        --label "BASELINE 0823 image" --reps 3
```
Record: 13/13 gates, decode tok/s median, DSpark accepted%/tok-step, `Available KV cache memory`,
`GPU KV cache size`, max concurrency, boot wall-clock.

## Step 1 — switch

```bash
./stop-deepseek-v4-flash-vision-exp-b12x-eugr.sh
# VERIFY both nodes are actually clear — the stop script has left half-dead clusters 3x:
for h in 192.168.1.236 192.168.1.237; do ssh $h "docker ps -a --format '{{.Names}}' | grep vision"; done

( cd ~/code/spark-vllm-docker-eugr && ./run-recipe.sh \
    ~/code/emosaru_sparkrunner_recipes/recipes/deepseek-v4-flash-vision-exp-b12x-0910-eugr.yaml -d \
    --config ~/code/emosaru_sparkrunner_recipes/env/eugr-roce-vision.env \
    -t eugr/spark-vllm-b12x:latest --port 8000 \
    --apply-mod ~/code/emosaru_sparkrunner_recipes/mods/drop-caches-fixed )
```
**Do NOT pass `--apply-mod .../deepseek-v4-vision-exp`** — it will FATAL on anchor drift, which
is the expected and correct behaviour now that vision is native.

## Tests, in order — stop at the first hard failure

| # | test | how | pass criterion | kills the upgrade? |
|---|---|---|---|---|
| 1 | **boots at all** | launch above | reaches `/v1/models` | YES |
| 2 | **correctness** | `verify-vision-exp-correctness.py` | **13/13**, incl. vision, tool calls, 32k needle, DSpark acceptance >15% | YES |
| 3 | **KV pool** | `Available KV cache memory`, `GPU KV cache size` | **>= baseline** | YES if materially worse |
| 4 | **decode + DSpark** | `bench-flashnext.py --reps 3` (gated) | tok/step and tok/s **>= baseline** | YES if worse |
| 5 | **image prefix-cache bust** | see below | — | no, informational |
| 6 | **concurrency** | c=1/2/4 sweep | aggregate scales, no `reason="capacity"` waits | no |
| 7 | **soak** | 24h+ under real proxy traffic | no earlyoom kill, no silent half-death | gate before promoting |

### Test 5 detail — the one genuinely new question
Does the new image still hoist images to the front of their message?
(`memory/vision_image_prefix_cache_bust.md`: ~16x TTFT, 0% vs 98.9% cached on a 28.7k context.)

Send the SAME ~28k-token context + one 560x560 image twice:
- layout A: image in the same message as the context
- layout B: image alone in a trailing message

Compare `usage.prompt_tokens_details.cached_tokens` and TTFT. On the old image both layouts
behave identically (proving client-side ordering is ignored). **If layout A now caches, the new
image fixes a real 16x penalty on image turns** — that would be the single biggest practical win
here, worth more than any throughput delta.

## Variables to A/B only AFTER a clean pass

1. **`max_num_batched_tokens` 4096 (candidate) vs 8192 (eugr's native).** Ours is measured on
   the OLD image (+25% available KV, 2026-08-24 bakeoff). Re-confirm on the new one.
2. `max_cudagraph_capture_size` 48 (candidate, eugr's) vs 64 (production).
3. `VLLM_USE_MEGA_AOT_ARTIFACT` "1" (candidate) vs "-1" (production).
4. eugr's experimental b12x loader vs `instanttensor` (changelog claims faster + leaner).

Change ONE at a time. Bundling the checkpoint and mtp change on 2026-09-11 produced a
confounded result that had to be re-run.

## Rollback

```bash
./stop-deepseek-v4-flash-vision-exp-b12x-eugr.sh
IMAGE=vllm-node-b12x:latest ./launch-deepseek-v4-flash-vision-exp-b12x-eugr.sh
```
Known-good, verified 2026-09-11: 13/13 gates, 4/4 vision, DSpark 2.87 tok/step, 1,598,902 KV tokens.
**The `IMAGE=` override is mandatory** — the launcher still defaults to `eugr/spark-vllm-b12x:latest`,
which is now the NEW image and will fail on vision-mod drift. Fixing that default is a prerequisite
of any promotion decision (either way, the two stacks no longer share an image).

## Promotion criteria

Promote only if: tests 1-4 pass at or above baseline, test 7 soaks 24h clean, and the vendored
vision mod can be retired. If only test 5 improves but 3/4 regress, consider keeping the old image
and instead fixing the cache bust client-side by isolating images in a trailing message — that
already gets the 16x win today with zero risk.
