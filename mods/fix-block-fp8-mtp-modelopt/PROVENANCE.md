# fix-block-fp8-mtp-modelopt

Port of **vLLM PR #55513** — "[Bugfix][Model] Fix block FP8 MTP in ModelOpt mixed
checkpoints" — merged upstream **2026-09-08**, Python-only (5 files).

## Why we carry it
`eugr/spark-vllm-b12x` is built from local-inference-lab's vLLM fork, which has not rebased
onto the merge, and the fork renamed the model tree (`vllm/models/qwen4_exp` upstream ->
`vllm/models/qwen3_8_flash_next`), so the PR does not apply by path.

Without it, `nvidia/Qwen3.8-Flash-Next-NVFP4` + MTP fails at weight load:

```
AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'
                for checkpoint weight 'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'
```

MIXED_PRECISION had no branch constructing a **block-scaled** FP8 MoE method, so the expert
layer never allocated the `*_weight_scale_inv` parameters nvidia's block-FP8 MTP carries.
(The old local-inference-lab quant did not hit this — its MTP experts are W4A16_NVFP4.)

## What is ported, and what is not
Only the `modelopt.py` half:
1. import `Fp8Config`, `Fp8MoEMethod`
2. `_BLOCK_FP8_MOE_ALGOS = ("FP8_PB_WO", "FP8_BLOCK_SCALES")` — same layout, two names
3. build `self.fp8_block_config` on the mixed config from the layers' `group_size`
4. route `RoutedExperts` with a block-FP8 algo to `Fp8MoEMethod(self.fp8_block_config, layer)`

**Not ported, deliberately:**
* the PR's `mtp.py` hunk — **the fork already has it** (`_remap_mtp_quantized_layers` is
  already called from `_make_draft_vllm_config`)
* the PR's `amd/mtp.py` hunk — irrelevant on CUDA
* the PR's `has_blocked_weights()` hunk — that method does not exist on the fork's mixed config

## Verified against this build before writing
* `Fp8MoEMethod.__init__(self, quant_config: Fp8Config, layer: RoutedExperts)` — positional,
  matching upstream's call site
* it sets `weight_scale_name = "weight_scale_inv"` when `weight_block_size` is set, which is
  exactly the missing parameter
* every anchor asserted unique; the mod **fails loudly** if the fork moves, and is idempotent
* backs up to `modelopt.py.pre-55513` and byte-compiles after patching

Retire this mod as soon as a b12x image ships the upstream fix — check with
`grep -c weight_scale_inv .../quantization/modelopt.py` inside the image.

## Edit 5 is OURS, not part of the PR
`--moe-backend b12x` is set globally by the recipe. Once edit 4 routes block-FP8 experts to
`Fp8MoEMethod`, that method resolves its backend through `fused_moe/oracle/fp8.py`, which
rejects `b12x`:

```
ValueError: moe_backend='b12x' is not supported for FP8 MoE.
Expected one of ['triton', 'deep_gemm', 'cutlass', ...]
```

That oracle is consulted **only for FP8 MoE layers**, so mapping `b12x -> TRITON` there
affects just the small block-FP8 MTP draft experts. The main model's NVFP4 routed experts
resolve through a different oracle and keep the b12x kernel path. Triton is the safe general
backend on SM121.

Upstream never hit this because upstream does not default to `--moe-backend b12x`; eugr's
recipe does, and eugr never ran a block-FP8 checkpoint through it.
