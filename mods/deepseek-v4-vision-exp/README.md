# DeepSeek-V4-Flash-Vision-Exp native vision mod

Adds **native image input** for `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` to a
vLLM that otherwise serves the text-only `DeepseekV4ForCausalLM`. Ported from
[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
(`patches/hotfix-dsv4-vision-exp.py` + `patches/vision_exp/`).

## Why a patch is required

The Vision-Exp checkpoint declares the **same**
`architectures: ["DeepseekV4ForCausalLM"]` string as the text-only 0731 model
while carrying 316 extra tensors — a 32-block ViT, a 2-layer aligner, four
learned image embeddings, and a `bias_vl` MoE routing bias on every layer.
vLLM's `DeepseekV4ForCausalLM` is text-only, so loading the vision checkpoint
fails with `ValueError: There is no module or parameter named 'aligner'`.

DeepSeek ships only a reference implementation (`inference/` in their repo —
"a readable reference implementation rather than a production serving engine"),
so there is nothing to configure: the ViT + aligner and the multimodal
processor must be added to vLLM's vendored `deepseek_v4/nvidia/model.py`.

## Mechanism

`run.sh` stages `vision_exp/` to `/opt/dspark-patches/vision_exp` (the overlay
package root — the hotfix's `DEFAULT_PATCHES` points there) and runs
`hotfix-dsv4-vision-exp.py`, which:

1. **Injects a fail-closed import hook** at the end of `nvidia/model.py` that
   constructs `vision` / `aligner` / `image_{start,end,newline,pad}` on
   `DeepseekV4Model` when `config.vision_n_layers > 0`, maps the `vision.*` /
   `aligner.*` / `image_*` / `bias_vl` weights, and registers a vLLM multimodal
   processor (`vision_exp/processor.py`).
2. **Remaps DSpark draft `ffn.gate.bias_vl`** → `e_score_correction_bias_vl` in
   `dspark.py` (the draft loader only rewrote names ending in `.ffn.gate.bias`).
3. Clears `__pycache__` and the stale `$VLLM_CACHE_ROOT/modelinfos/` cache (the
   disk-cached multimodal inspection — a prior text-only boot leaves a stale
   entry that silently reuses the "text-only" classification).

The whole thing is idempotent and fail-closed on anchor drift.

## Recipe usage

Point the recipe at the **vision checkpoint** and add the mod:

```yaml
model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
mods:
  - mods/deepseek-v4-vision-exp
```

`served_model_name` should stay `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.

## b12x fused-MHC `rms_norm_eps` fix (eugr b12x images only)

The Vision-Exp checkpoint ships `rms_norm_eps: 1e-20` (0731 used `1e-6`). With
`VLLM_USE_B12X_MHC=1`, the b12x fused MHC Gram **decode** kernel hard-requires
`rms_eps == 1e-6` (`b12x.norm.mhc._impl._supports_fused_mhc_gram`); 1e-20 falls
through to a `ValueError` during `determine_available_memory` ("b12x_mhc_pre is
served only by the fused Gram kernel"). Lineages that route MHC through a
TileLang kernel (`mhc_pre_tilelang`) accept any eps and are unaffected.

`apply_vision_exp`'s `model_init` hook forces `config.rms_norm_eps = 1e-6`
before `DeepseekV4Model.__init__` reads it, **guarded on `vision_n_layers > 0`**
so text-only checkpoints are untouched. The DSpark MTP draft is built from the
same shared `hf_config` (speculative `target_model_config.hf_config`), so it
inherits the override.

**Correctness of the override:** RMSNorm computes
`x * rsqrt(mean(x²) + eps)`; eps is a div-by-zero stabilizer. On realistic
DeepSeek V4 hidden states (hidden_size 4096, bf16, variance O(1)), RMSNorm with
`eps=1e-6` vs `eps=1e-20` produced **bit-identical bf16 output** (0/131072
elements differ). So the override is exact for bf16, not merely approximately
equal.

To instead keep `rms_norm_eps=1e-20` exactly, set `VLLM_USE_B12X_MHC=0` in the
recipe env — this routes MHC to the TileLang path (no eps constraint) but drops
the b12x fused decode kernel.

## Notes / limitations

- **Images in `user` messages only** — `system` / `assistant` images return
  HTTP 400 (official Chat Completions restriction).
- **Max 384 image tokens** per image (`vision_max_n_token`).
- **DSpark k must be a multiple of 3.** Vision-Exp has
  `num_nextn_predict_layers=3` (0731 had 1), so k=5 is rejected
  (`k % 3 == 0` when `k > n_predict`). Use `num_speculative_tokens: 6` — the
  smallest k that is also ≥ dspark block size 5.
- **Fidelity gap (documented, not silent):** image tokens use the standard
  causal sparse attention pattern — the reference's bidirectional attention
  *within* each `[IMAGE_START, IMAGE_END]` span is not ported. Expect some
  quality loss on image-heavy prompts.
- `bias_vl` is loaded but not applied at routing time — image tokens route
  through the text bias. Known limitation in the reference ports; text-only
  requests are byte-identical to 0731.
