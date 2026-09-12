# Provenance — vendored, not authored here

Vendored VERBATIM from `vavrusa/sparkrun-recipes` at commit `fa90eb92bcd0d0e657c935effb227d395c309fce`
(`mods/deepseek-v4-vision-exp/`), fetched 2026-09-01. That is itself a port of
`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` (`patches/hotfix-dsv4-vision-exp.py`
+ `patches/vision_exp/`) onto the eugr b12x runtime.

Vendored rather than referenced so a launch cannot depend on a third-party repo staying
up or unchanged. Upstream is a **1-commit, 0-star repo created 2026-09-01** — treat it as
unproven code we have read, not as a maintained dependency.

## LOCAL DELTAS (we are no longer byte-identical to upstream)

**2026-09-01 — `vision_exp/apply.py`: made the three monkeypatch wrappers signature-agnostic.**
Upstream pinned exact signatures matching its Aug-10 b12x image:
`moe_init(self, vllm_config, prefix="")`, `model_init(self, *, vllm_config, prefix="")`,
`lm_init(self, *, vllm_config, prefix="")`. Our b12x build (2026-08-23) added a third parameter
to `DeepseekV4MoE.__init__` — `use_sequence_parallel: bool = False` — and
`DeepseekV4DecoderLayer` passes it by keyword. Result, on first launch:

    TypeError: apply_vision_exp.<locals>.moe_init() got an unexpected keyword argument 'use_sequence_parallel'

Engine core init failed; the server never came up. All three wrappers now forward
`*args, **kwargs` verbatim and extract `vllm_config` defensively, so they tolerate added or
reordered parameters. Verified against all three call shapes (positional+kwarg, upstream's
2-arg, all-keyword).

**Why this matters beyond the one fix:** the hotfix is a TEXT-substitution patch, so its
anchors matching says nothing about RUNTIME signatures matching. Our dry-run and the
`supports_multimodal` check both passed while this was still broken. Expect more of this class
of drift the further our b12x build moves from upstream's Aug-10 pin.

**2026-09-01 — `vision_exp/processor.py`: lock around the HF fast-tokenizer encode.**
Live defect, not theoretical: **7 of 326 chat completions (2.1%) returned HTTP 500** with
`RuntimeError: Already borrowed`, every one on the image path, all from the workstation during
normal 2-request concurrency. vLLM preprocesses multimodal inputs in a THREAD POOL
(`renderers/base.py::_process_multimodal_async`); the Rust fast tokenizer has interior
mutability and `encode()` -> `set_truncation_and_padding()` -> `no_truncation()` takes a mutable
borrow, so two concurrent image requests race. Now serialized with a module-level `Lock`.
Cheap relative to the ViT work that follows. **Takes effect on next launch** (mods apply at
container start). If "Already borrowed" ever recurs despite the lock, the competing borrow is
coming from OUTSIDE our processor and the fix escalates to a thread-local tokenizer clone.

## Re-sync

    curl -sL https://codeload.github.com/vavrusa/sparkrun-recipes/tar.gz/refs/heads/main -o /tmp/vm.tgz
    tar xzf /tmp/vm.tgz -C /tmp
    diff -ru mods/deepseek-v4-vision-exp /tmp/sparkrun-recipes-main/mods/deepseek-v4-vision-exp \
      --exclude PROVENANCE.md

## What we verified ourselves (2026-09-01, throwaway container, eugr/spark-vllm-b12x:latest @ 2026-08-23)

Upstream pins `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:20260810`; we run a build
13 days newer. The mod is fail-closed on anchor drift, and against OUR image it reports:

    vision-exp hotfix model.py               : applied
    vision-exp hotfix deepseek_v4_encoding.py: skipped-no-placeholder
    vision-exp hotfix dspark.py              : applied

and afterwards, in-process:

    supports_multimodal(DeepseekV4ForCausalLM) == True     # SupportsMultiModal in the MRO

NOT verified: real 2-GPU boot, KV pool size, end-to-end image correctness. The dry run
was CPU-only.

## Known fidelity gaps (both upstream, not introduced here)

* `bias_vl` is loaded but **never applied at routing** — confirmed by reading
  `vision_exp/apply.py`: name-mapped (line 32) and allocated as a Parameter (line 236),
  never consulted in a routing path. Image tokens route on the TEXT bias.
  MiaAI-Lab's docstring claims image rows route with it; that claim is optimistic.
* Image-span attention is **causal**, not the reference's bidirectional within-span
  attention. Upstream vLLM PR #54566 admits the same gap.
* Text-only requests are unaffected by both.
