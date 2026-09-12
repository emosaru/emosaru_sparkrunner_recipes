#!/bin/bash
# Port of vLLM PR #55513 "[Bugfix][Model] Fix block FP8 MTP in ModelOpt mixed checkpoints"
# onto the b12x fork's tree. Merged upstream 2026-09-08; the b12x fork has not rebased.
#
# WITHOUT THIS, nvidia/Qwen3.8-Flash-Next-NVFP4 + MTP dies at load with:
#   AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'
#                   for checkpoint weight 'mtp...down_proj.weight_scale_inv'
# because MIXED_PRECISION had no branch that builds a BLOCK-scaled FP8 MoE method, so the
# expert layer never created the *_weight_scale_inv parameters the checkpoint carries.
#
# ONLY the modelopt.py half of the PR is ported: the fork ALREADY has the mtp.py half
# (_remap_mtp_quantized_layers is already called from _make_draft_vllm_config), and the
# PR's amd/mtp.py hunk is irrelevant on CUDA.
#
# Upstream calls Fp8MoEMethod(self.fp8_block_config, layer) positionally; verified this
# build's signature is __init__(self, quant_config: Fp8Config, layer: RoutedExperts), and
# that it sets weight_scale_name="weight_scale_inv" when weight_block_size is set — which
# is exactly the missing parameter. The fork has no has_blocked_weights() on the mixed
# config, so that hunk of the PR is skipped.
set -euo pipefail

python3 - <<'PYEOF'
import py_compile, sys, shutil, os

M = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"
src = open(M).read()

if "_BLOCK_FP8_MOE_ALGOS" in src:
    print("fix-block-fp8-mtp-modelopt: already applied, nothing to do")
    sys.exit(0)

def edit(s, old, new, what):
    if old not in s:
        raise SystemExit(f"fix-block-fp8-mtp-modelopt: ANCHOR MISSING ({what}) — "
                         "the fork moved; re-derive this mod against the new image.")
    if s.count(old) != 1:
        raise SystemExit(f"fix-block-fp8-mtp-modelopt: anchor not unique ({what}), "
                         f"found {s.count(old)}")
    return s.replace(old, new, 1)

# 1. import Fp8Config / Fp8MoEMethod
src = edit(src,
  "from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod",
  "from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod\n"
  "from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod",
  "imports")

# 2. the algo alias tuple (FP8_PB_WO is ModelOpt's canonical 2D block-FP8 name;
#    composed Flash-Next checkpoints emit FP8_BLOCK_SCALES for the same layout)
src = edit(src,
  '    # MIXED_PRECISION,\n    "MIXED_PRECISION",\n]',
  '    # MIXED_PRECISION,\n    "MIXED_PRECISION",\n]\n\n'
  '# Ported from vLLM PR #55513: both names describe the same 2D block-FP8 layout.\n'
  '_BLOCK_FP8_MOE_ALGOS = ("FP8_PB_WO", "FP8_BLOCK_SCALES")',
  "algo tuple")

# 3. build a block-FP8 Fp8Config on the mixed-precision config
src = edit(src,
  "        self.mxfp8_config = mxfp8_config\n\n    def get_name(self) -> QuantizationMethods:\n"
  '        return "modelopt_mixed"',
  "        self.mxfp8_config = mxfp8_config\n\n"
  "        # PR #55513: one shared block size across all block-FP8 MoE layers.\n"
  "        block_sizes = {\n"
  '            int(layer_info.get("group_size", 128))\n'
  "            for layer_info in quantized_layers.values()\n"
  '            if layer_info.get("quant_algo", "").upper() in _BLOCK_FP8_MOE_ALGOS\n'
  "        }\n"
  "        if len(block_sizes) > 1:\n"
  "            raise ValueError(\n"
  '                "MIXED_PRECISION currently requires all block-FP8 MoE layers "\n'
  '                f"to use one group_size, got {sorted(block_sizes)}."\n'
  "            )\n"
  "        block_size = next(iter(block_sizes), 128)\n"
  "        self.fp8_block_config = Fp8Config(\n"
  "            is_checkpoint_fp8_serialized=True,\n"
  '            activation_scheme="dynamic",\n'
  "            weight_block_size=[block_size, block_size],\n"
  "        )\n\n"
  "    def get_name(self) -> QuantizationMethods:\n"
  '        return "modelopt_mixed"',
  "mixed __init__")

# 4. route block-FP8 routed-experts to the block-capable Fp8MoEMethod
src = edit(src,
  '        if isinstance(layer, RoutedExperts):\n            if quant_algo == "FP8":\n'
  "                return ModelOptFp8MoEMethod(",
  "        if isinstance(layer, RoutedExperts):\n"
  "            # PR #55513: block-scaled FP8 experts need Fp8MoEMethod, which creates the\n"
  "            # *_weight_scale_inv params that block checkpoints (e.g. nvidia's MTP) carry.\n"
  "            if quant_algo in _BLOCK_FP8_MOE_ALGOS:\n"
  "                return Fp8MoEMethod(self.fp8_block_config, layer)\n"
  '            if quant_algo == "FP8":\n'
  "                return ModelOptFp8MoEMethod(",
  "RoutedExperts branch")

shutil.copy2(M, M + ".pre-55513")
open(M, "w").write(src)
py_compile.compile(M, doraise=True)
print("fix-block-fp8-mtp-modelopt: applied 4 edits to modelopt.py (backup: .pre-55513)")

# ---------------------------------------------------------------------------
# 5. FP8-MoE backend oracle: accept the b12x backend name.
#
# NOT part of PR #55513 — this is ours, and it is needed only because the recipe sets
# --moe-backend b12x GLOBALLY. Once edit 4 routes the MTP's block-FP8 experts to
# Fp8MoEMethod, that method asks the FP8 oracle to resolve the backend and the oracle
# rejects 'b12x':
#   ValueError: moe_backend='b12x' is not supported for FP8 MoE.
# This mapping lives in oracle/fp8.py and is consulted ONLY for FP8 MoE layers, so
# mapping b12x -> TRITON here affects just the small block-FP8 MTP draft experts. The
# main model's NVFP4 routed experts resolve through a different oracle and keep the
# b12x kernel path untouched. Triton is the safe general backend on SM121.
O = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/fp8.py"
osrc = open(O).read()
if "# b12x-compat" in osrc:
    print("fix-block-fp8-mtp-modelopt: fp8 oracle already patched")
else:
    anchor = '        "hpc": Fp8MoeBackend.HPC,\n    }'
    if anchor not in osrc or osrc.count(anchor) != 1:
        raise SystemExit("fix-block-fp8-mtp-modelopt: fp8 oracle anchor missing/not unique")
    osrc = osrc.replace(anchor,
        '        "hpc": Fp8MoeBackend.HPC,\n'
        '        # b12x-compat: the b12x MoE kernels do not implement block-FP8, but the\n'
        '        # recipe sets --moe-backend b12x globally. Only FP8 MoE layers (the\n'
        '        # block-FP8 MTP draft experts) reach this oracle; NVFP4 experts keep b12x.\n'
        '        "b12x": Fp8MoeBackend.TRITON,\n'
        '        "flashinfer_b12x": Fp8MoeBackend.TRITON,\n    }', 1)
    shutil.copy2(O, O + ".pre-55513")
    open(O, "w").write(osrc)
    py_compile.compile(O, doraise=True)
    print("fix-block-fp8-mtp-modelopt: mapped b12x -> TRITON in the FP8 MoE oracle")
PYEOF
