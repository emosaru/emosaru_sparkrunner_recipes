#!/bin/bash
set -e

VLLM_MODELS=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models

# Modelopt NVFP4 checkpoints exclude MTP layers from quantization via
# "re:.*mtp.*" in config.json, so those weights are BF16. vLLM's
# is_layer_excluded uses fnmatch (not regex), so "re:.*mtp.*" never
# matches the "mtp.layers.*" prefix and NVFP4 is incorrectly applied,
# causing a shape assertion on load. This patch adds the correct glob
# in Qwen3_5MTP.__init__ so the draft model falls back to BF16 linears.
echo "Patching qwen3_5_mtp.py: fix MTP TP>=2 weight loader for NVFP4 checkpoints"
cp qwen3_5_mtp.py "$VLLM_MODELS/qwen3_5_mtp.py"
