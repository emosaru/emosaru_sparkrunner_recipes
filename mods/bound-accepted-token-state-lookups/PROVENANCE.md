# bound-accepted-token-state-lookups

> **MEASURED 2026-10-01 — THIS IS NOT A FIX. KEPT FOR THE RECORD; LEAVE IT OFF.**
>
> A/B'd against `scripts/flashnext_corruption_ab.py --mode mixed`, one variable at a time:
>
> | arm | boots | result |
> |---|---|---|
> | baseline (stock) | 2 | both dead — 32 min, and request 28 of 32 |
> | `BOUND_ACCEPTED=1` (this mod) | 2 | one survived 48 requests; **the other died in round 1** |
>
> The mod was verified genuinely live both times (`dst_col_ok` present on BOTH ranks), so
> this is not a deployment miss. The surviving boot was luck.
>
> **Why it cannot work:** the fault is in `v1/worker/gpu/boundary_checkpoint.py`
> (`wait_for_boundary_checkpoint_copies` -> `wait_for_copies`), and #50021 patches
> `mamba_utils.py`, `causal_conv1d.py`, `mamba_ssm.py`, FLA and KDA — never
> `boundary_checkpoint.py`. That file **does not exist in upstream vLLM at all**
> (authenticated code search: 0 hits for the filename, 0 for
> `wait_for_boundary_checkpoint_copies`, 0 for `NUM_BOUNDARY_CHECKPOINT_SLOTS`, against 3
> for the `mamba_utils.py` control), so no upstream PR can address it.
>
> **The actual fix is `CKPT_POLICY=aligned`** (now the launcher default), which disables
> that fork-only subsystem: 96 requests clean, 35+ min under load, 0 Xid, at unchanged
> throughput. See `memory/flashnext_cuda_illegal_access_2026_09_25.md`.
>
> One loose end, if anyone revisits this: the GPU fault is
> `Xid 31 ... ACCESS_TYPE_VIRT_WRITE` — a wild WRITE — and this port deliberately scoped
> the `dest_block_id <= 0` guard to the block-table-load branch rather than applying it
> unconditionally as upstream does. That scoping was never tested in isolation.

Port of **[vLLM PR #50021](https://github.com/vllm-project/vllm/pull/50021)** —
*"Bound accepted-token state lookups in GDN/KDA spec decode"*.
**Status upstream: OPEN / unmerged** (opened 2026-09-05, still open 2026-09-30).

## Why this mod exists

It is the only candidate fix that matches our stack on every axis we actually run:
hybrid **GDN** model, **MTP** speculative decoding, **prefix caching**, and
`--mamba-cache-mode align` (upstream validated under exactly that flag).

Upstream's root cause, verbatim:

> speculative decoding generates per-request accepted-token counts that multiple GPU
> kernels convert into array indices **without bounds checking**; stale, zero, or
> oversized counts cause out-of-bounds reads, dereferencing wild addresses and faulting
> the GPU

And from the patched comment itself — the sentence that makes this *one* bug rather than
two:

> an out-of-range column becomes a wild (and possibly misaligned) address **rather than
> merely a wrong copy**

That is both of our symptoms from a single defect. When a stray column lands inside the
cache it is "merely a wrong copy" — silent recurrent-state corruption, which surfaces as
the degenerate constant-token reasoning loops we saw (deterministic and byte-identical
across greedy runs, because the wrong state is deterministically wrong). When it lands
outside, it faults — the four `CUDA error: an illegal memory access was encountered`
deaths. See `memory/flashnext_cuda_illegal_access_2026_09_25.md`.

Upstream reports it as `unspecified launch failure` / Xid 13 / Xid 31 within 7-10
requests under `--enforce-eager`; ours reports `cudaErrorIllegalAddress` at an async sync
point. Same wild-address family, different reporting site.

## Why a mod and not an image rebuild

All 7 changed files are **pure Python/Triton** — no CUDA or C++ — so patching the
installed package is equivalent to rebuilding. This matters practically: when this was
written .236 was at **92% disk / 74 GB free** and .237 at **96% / 37 GB free**, with
`earlyoom -m 2 --prefer '(vllm|...)'` armed and only ~7 GB RAM available on the serving
head. A 36 GB image rebuild did not fit and would have risked the head (it has been
killed twice — `memory/earlyoom_head_kill.md`).

eugr *does* support building PRs in (`Dockerfile` `ARG VLLM_PRS`, driven by
`build-and-copy.sh --vllm-pr`), and that is the right route once disk allows. This mod is
the zero-downtime equivalent.

## The port

24 of the PR's 25 hunks applied to our build unchanged. One failed:
`vllm/v1/worker/mamba_utils.py` hunk 1, in `_copy_mamba_state_block`. The PR expects

```python
dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)
```

but our build (post-dating the PR by ~3 weeks) wraps that load in a
`if destination_block_id is None:` branch, so the context did not match. The fix itself is
directly portable — `block_table_stride_req` is already a parameter of our kernel
signature, so no plumbing was needed.

**One deliberate deviation:** upstream places the `if dest_block_id <= 0: return` guard
unconditionally after the load. We scope it to the block-table-load branch only, because
the caller-supplied `destination_block_id` path did not exist when the PR was written and
gating it would be an unreviewed behaviour change. If the PR merges, take upstream's
version over this port.

The shipped `.patch` was regenerated from the fully-ported tree, so it applies as a single
clean unit (18 hunks / 7 files) rather than needing the reject dance.

## What is verified, and what is not

Verified (2026-09-30, against `eugr/spark-vllm-b12x:nightly-20260923`,
vLLM `v0.1.dev21504+g57fdda71b`):

* All 7 source files exist in the image and the patch applies with **exit 0**, no fuzz.
* All patched files pass `py_compile`.
* End-to-end run of `run.sh` in a throwaway container from that image: applies, is
  **idempotent** on a second run, and the bound is present in the kernel source
  (`dst_col_ok = (dst_col >= 0) & (dst_col < block_table_stride_req)`).
* Refuses to half-apply: dry-runs first and aborts with the dry-run log if the base has
  drifted.
* Forward-compatible: keys idempotency off `dst_col_ok`, so if upstream merges #50021 the
  mod becomes a no-op instead of conflicting.

**NOT verified: we have never booted the engine with this applied.** It is static and
container-level validation only. Nothing here demonstrates that it fixes the crash — it
demonstrates that the patch is correctly applied and cannot silently no-op.

## How to judge it

`BOUND_ACCEPTED=1 ./launch-qwen38-flash-next-b12x-eugr.sh`, then confirm the
"accepted-token-derived block-table columns are now bounded" line on **both** ranks.

Judge on **MTTF** and on whether long-context output stays coherent. It should **not**
change throughput — this is index masking, not a kernel rewrite. Note that MTTF across the
four deaths was 2h29m / 47h / 4min / ~70h, so uptime alone proves little; the fast
discriminator is the deterministic degeneration at ~173.6k prompt tokens.

Important, from [issue #54173](https://github.com/vllm-project/vllm/issues/54173): the
trigger is a **cache-hit resume where a new request extends a cached prefix** — *identical
repeated prompts do not reproduce it*. Any A/B must send prompt A, then a longer A+B.

## Ruled out while getting here

* **#55506 / #57807** (batch-row vs request-slot block-table indexing) — #55506 states the
  bug *"requires PP>1; it does not occur with tensor parallelism alone"*. We run TP=2,
  PP=1. Their headline symptoms (14-33% constant-token loops, acceptance 50%→26%) are
  PP>1 repro figures, not corroboration of our case.
* **#58434** (merged 2026-09-25, padded prompt tails misclassified for hybrid models) —
  fixes a regression from #52988, and we **predate** #52988: both runners still classify
  spec rows by `num_scheduled_tokens == draft_len + 1`
  (`gpu_model_runner.py:2299`, and the V2 path we run at `mamba_hybrid.py:330`).
  Upgrading to `nightly-20260926+` would import the regression and its fix together.
* **#53142 / #53798 / #55507 / #55601** (align seed uses `cache_config.block_size` instead
  of the mamba block size) — **already fixed in our image**;
  `mamba_hybrid.py:124` seeds with `cache_config.mamba_block_size`.
