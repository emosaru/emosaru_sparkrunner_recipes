#!/bin/bash
# run.sh — DeepSeek-V4-Flash-Vision-Exp native vision mod
#
# Ported from MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
# (patches/hotfix-dsv4-vision-exp.py + patches/vision_exp/). Adds native image
# input to a vLLM that serves deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.
#
# Target layout (Anemll / eugr-style DeepSeek-V4 fork):
#   model.py at /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/
#     nvidia/model.py with DeepseekV4ForCausalLM / DeepseekV4MoE /
#     DeepseekV4Model classes (apply.py's monkeypatch anchors), plus
#     nvidia/dspark.py and tokenizers/deepseek_v4_encoding.py. If the encoding
#     file has no IMAGE_PLACEHOLDER the encoding patch is skipped gracefully.
#
# What it does:
#   1. Stage patches/vision_exp -> /opt/dspark-patches/vision_exp (the injected
#      import hook reads from there).
#   2. Run hotfix-dsv4-vision-exp.py: injects a fail-closed import hook into
#      nvidia/model.py that builds the ViT+Aligner, maps vision./aligner./
#      image_*/bias_vl weights, registers a vLLM multimodal processor, makes
#      embed_input_ids scatter image-block embeddings, and remaps the DSpark
#      draft ffn.gate.bias_vl -> e_score_correction_bias_vl in dspark.py.
#   3. Clear __pycache__ and the stale $VLLM_CACHE_ROOT/modelinfos/ cache (the
#      disk-cached multimodal inspection keyed by module+class — a prior
#      text-only boot would leave a stale entry).
#
# Notes: images go in user messages only (official restriction). Max 384 image
# tokens per image. num_nextn_predict_layers=3 on Vision-Exp -> DSpark k must be
# a multiple of 3 (k=6 recommended; k=5 is rejected).
set -euo pipefail

PREFIX="[deepseek-v4-vision-exp]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="/usr/local/lib/python3.12/dist-packages/vllm"
MODEL="$SITE/models/deepseek_v4/nvidia/model.py"
DSPARK="$SITE/models/deepseek_v4/nvidia/dspark.py"
ENCODING="$SITE/tokenizers/deepseek_v4_encoding.py"
PATCHES_DIR="/opt/dspark-patches"
HOTFIX="$MOD_DIR/hotfix-dsv4-vision-exp.py"

echo "=== deepseek-v4-vision-exp native vision mod ==="

if [ ! -d "$SITE" ]; then
    echo "$PREFIX vLLM tree not found at $SITE" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "$PREFIX nvidia/model.py not found at $MODEL" >&2
    exit 1
fi
if [ ! -f "$DSPARK" ]; then
    echo "$PREFIX nvidia/dspark.py not found at $DSPARK" >&2
    exit 1
fi
if [ ! -f "$HOTFIX" ] || [ ! -d "$MOD_DIR/vision_exp" ]; then
    echo "$PREFIX hotfix or vision_exp package missing under $MOD_DIR" >&2
    exit 1
fi

# Stage the vision_exp package where the injected import hook looks for it.
mkdir -p "$PATCHES_DIR"
rm -rf "$PATCHES_DIR/vision_exp"
cp -r "$MOD_DIR/vision_exp" "$PATCHES_DIR/vision_exp"
echo "$PREFIX staged vision_exp -> $PATCHES_DIR/vision_exp"

# Run the hotfix. It patches model.py + dspark.py in place (idempotent,
# fail-closed on anchor drift) and clears the encoding.py only if that file
# actually has placeholder logic (no-op on lineages without it).
python3 "$HOTFIX"

# Clear stale bytecode + the disk-cached multimodal inspection.
find "$SITE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
if [ -d "$CACHE_ROOT/modelinfos" ]; then
    rm -rf "$CACHE_ROOT/modelinfos"
    echo "$PREFIX cleared stale $CACHE_ROOT/modelinfos"
fi

echo "=== OK: DeepSeek-V4-Flash-Vision-Exp native vision installed ==="
