# mtp-reduced-draft-vocab

Reduced-vocabulary MTP drafting for Qwen3.8-Flash-Next. Restricts the **draft** head's
projection to the lowest `VLLM_MTP_DRAFT_VOCAB_SIZE` token ids (suggested 65,536 of 248,320).

**Status: WRITTEN 2026-09-18, NEVER BOOTED.** Static validation only (below). Validating it
for real requires stopping production — see "How to validate".

## Why — bandwidth, not acceptance
The draft head is read once per speculative position per decode step (3-4 reads per step at
our `num_speculative_tokens`), so its row count is the dominant cost of drafting. Shrinking
the rows shrinks that cost. Acceptance is expected to *fall slightly* (the community measures
~75% -> ~68%, i.e. ~6% of tokens fall outside the set); the net is still positive because step
time falls more than acceptance does.

## Correctness is structural, not empirical
Masked ids get `-inf`, so the draft never **proposes** them. The **target** model's own
`lm_head` is untouched and still verifies over the full vocabulary, so a token outside the set
is rejected exactly like any other bad draft. Emitted text is unchanged. Poor coverage costs
speed, never correctness. This is the same argument the upstream community makes, and it is a
property of rejection sampling rather than a measured claim.

## Design — why it is an `_apply_head` override on the draft's own LogitsProcessor
`Qwen3_8FlashNextMTP.compute_logits` calls `self.logits_processor(self.lm_head, hidden)`, and
`LogitsProcessor._get_logits` does `_apply_head` -> TP gather -> vocab-padding trim. We replace
**only `_apply_head`**, and only on the **draft head's own LogitsProcessor instance**:
* the TP gather and the padding trim are inherited untouched, so shapes and the collective are
  exactly as before — no reimplementation of the vocab-parallel all-gather;
* the main model has a different LogitsProcessor instance and is completely unaffected.

The reduced set is a contiguous id **prefix**, which is what makes this cheap: because each
vocab-parallel shard owns a contiguous ascending id range, a prefix of ids is a prefix of
*rows*. The projection is `F.linear(hidden, weight[:k])` — a view, no gather, no resident copy.

Verified against the image's real `VocabParallelEmbedding._get_indices` (vocab 248,320, TP=2):

| rank | org ids | k @ N=65536 | effect |
|---|---|---|---|
| 0 | [0, 124160) | 65,536 of 124,160 | reads 52.8% of its shard |
| 1 | [124160, 248320) | **0** | **skips its matmul entirely** |

Total draft-head rows read drops to **26.4%** of the full head. (The community's 47k-id set
gives ~19% by the same arithmetic — consistent with their reported 1.18 GiB -> 0.22 GiB.)

## Safety rails — it disables itself rather than risk wrong output
Falls back to the stock `_apply_head` (logging a warning) when any assumption fails:
* draft head is not a `VocabParallelEmbedding`
* draft head is **quantized** (`VLLM_MTP_NVFP4_LM_HEAD`) — a quantized head is not a plain
  row-sliceable matrix (scales, swizzling)
* no 2-D `weight`
* **added vocabulary (LoRA) present** — added rows sit after the org rows and the padding, so a
  row prefix would mask them incorrectly
* per call: `embedding_bias is not None`, or a `head_dtype` differing from the hidden dtype

If `N` already covers a rank's whole shard, it logs and does nothing on that rank.

## Known trade-off
The override bypasses the **b12x BF16 vocabulary projection** fast path inside the stock
`_apply_head`. That path requires `flat.shape[0] == 1` (a single token in the batch) and an
exact weight-shape match against a pre-resolved plan, so it applies only at batch size 1. At
that batch size we trade the b12x kernel for a 3.8x smaller read. **Unmeasured** — if a
single-stream A/B shows a regression, that is the first thing to suspect.

## Configuration
Off unless `VLLM_MTP_DRAFT_VOCAB_SIZE` is set in the container environment; applying the mod
without it is a no-op by design. An invalid value raises at load rather than silently serving
a different config than intended.

Why the **lowest ids** specifically: the tokenizer assigns ids in BPE merge order, so low ids
are the frequent tokens — the same frequency proxy the published single-Spark recipe uses. A
tuned, non-contiguous id set (the dual-Spark repo ships a 47k code-tuned list) would cover
better, but non-contiguous ids are not a row prefix and would need a materialized gathered
head. That is a possible v2; see "Licensing".

## Licensing
Written from the **publicly described mechanism**, not from anyone's source. No third-party
code or vocabulary file is vendored here — deliberately: `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks`
is **AGPL-3.0-or-later**, so copying its overlay or its `draft_vocab_en_code_47k.txt` into this
repo would raise licensing questions we do not need. If we ever want a tuned id set, derive one
from our own traffic.

## Static validation performed (2026-09-18, eugr/spark-vllm-b12x:latest, run on .237)
* applies cleanly; **idempotent** (second run is a no-op)
* patched module parses (`ast.parse`) and byte-compiles (`py_compile`, `doraise=True`)
* every anchor asserted present **and unique** — fails loudly if the fork moves
* backs up to `mtp.py.pre-reduced-draft-vocab`
* shard arithmetic checked against the image's real `_get_indices` (table above)
* the module has no `logger` of its own, so the helper installs a private `_rdv_logger`

## How to validate for real
Needs a Flash-Next launch, which replaces production on :8000. Then:
1. Boot with `VLLM_MTP_DRAFT_VOCAB_SIZE` **unset** — confirm the no-op path and get a baseline.
2. Boot with it set to 65536 — confirm the "ENABLED" log line appears on **rank 0** and the
   "keeps its whole shard"/skip path on rank 1.
3. Compare `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` and decode tok/s.
   Expect **tok/step to dip slightly** and **tok/s to rise**; judge on tok/s.
4. Gate every sample on `vllm:num_requests_running` — :8000 is never idle
   (see memory/qwen38_flash_next_cache_and_spec_rates.md).

Retire this mod if a b12x image ships reduced-vocab drafting natively.
