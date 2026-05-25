#!/usr/bin/env python3
"""
NVFP4 quantization of Qwen/Qwen3.6-35B-A3B for DGX Spark (GB10, SM 121a).

Produces a checkpoint with:
  - Text transformer (MoE) layers quantized to NVFP4
  - MTP head kept in BF16 and grafted into the export

Architecture (verified from Qwen/Qwen3.6-35B-A3B-FP8 config.json):
  - class: Qwen3_5MoeForConditionalGeneration (model_type: qwen3_5_moe)
  - 40 layers: 30 linear_attention (GatedDeltaNet) + 10 full_attention (every 4th)
  - MoE: 256 experts, 8 active per token (~3B active params). Expert and router
    layers are the dominant weight mass.
  - Has vision tower (vision_config present) — *visual* exclusions are required,
    not optional guards.
  - MTP: mtp_num_hidden_layers=1, same structure as 27B.
  - KV cache: only 10 full_attention layers carry KV, with 2 KV heads, head_dim=256.
    At FP8 KV, 256k context = ~2.7 GB KV total — trivially fits in any reasonable budget.
  - hidden_size=2048 (text trunk; expert FFNs make up the 35B parameter count).

KV cache benefit:
  At TP=1 on node 1 with util=0.46 (~55.7 GB budget):
    FP8:   weights+workspace ~45-47 GB → ~8-10 GB KV → 64k context max
    NVFP4: weights+workspace ~25-28 GB → ~28-30 GB KV → 256k context possible

Calibration data choice:
  neuralmagic/calibration (LLM config) is the recommended baseline for NVFP4
  calibration. It provides a broad, diverse token distribution (web text, code,
  conversation) that covers the activation range of a general-purpose agentic
  model without overfitting to any single domain. For coding + tool-calling use,
  the distribution match is sufficient: NVFP4 uses group-wise activation scaling
  (group_size=16) which adapts locally per tensor group, so domain-specific
  calibration data provides diminishing returns. The most impactful variable is
  sample count: 512 samples (vs the 27B default of 20) gives better statistical
  coverage of MoE expert activations across different routing paths.

  If you want to supplement with code-heavy calibration, pass --n-samples 400
  for neuralmagic/calibration then build a second forward_loop over a code
  dataset (e.g. bigcode/starcoderdata Python subset) and call mtq.quantize twice.

InstantTensor compatibility:
  This script exports standard HF safetensors via modelopt's export_hf_checkpoint.
  The exported checkpoint is fully compatible with --load-format instanttensor
  (the instanttensor package reads standard safetensors with lower peak memory
  via unified-memory-aware streaming). No post-export conversion step is needed.

Requirements:
  nvidia-modelopt >= 0.41 (0.44+ recommended)
  transformers >= 4.51
  datasets, huggingface_hub, torch >= 2.5

Run inside a modelopt-capable environment, e.g. the eugr nightly vLLM container
with modelopt installed (same environment used for the 27B quant).

Exclusion rationale (verified against Qwen's FP8 quantization_config)
-------------------
*lm_head*, *output_layer*, *router*, *mlp.gate.*  — NVFP4_DEFAULT_CFG defaults
*visual*, *model.visual*                           — vision tower (27 ViT blocks +
                                                     merger + patch_embed); confirmed
                                                     present in 35B-A3B config
*mtp*                                              — MTP head; grafted back in BF16
*shared_expert_gate*                               — MoE shared-expert gate Linear
                                                     (mlp.shared_expert_gate per layer).
                                                     NOT covered by *mlp.gate.* (different
                                                     name). Qwen excluded it in FP8 quant;
                                                     we do the same.
*linear_attn*                                      — GatedDeltaNet full block. The 35B
                                                     uses in_proj_a/b/ba (not in_proj_qkvz
                                                     as in 27B) but the vLLM NVFP4 loading
                                                     path for combined input projections is
                                                     still unvalidated. Qwen's FP8 excluded
                                                     in_proj_a/b/ba and kept out_proj;
                                                     we exclude the full block conservatively
                                                     until NVFP4+vLLM is validated on this
                                                     naming convention.
"""
import argparse
import copy
import glob
import json
import logging
import os
import sys
from pathlib import Path

import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def build_quant_cfg() -> dict:
    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)

    extra_disabled = [
        "*visual*",              # vision tower — confirmed present in 35B-A3B
        "*model.visual*",        # alternate vision prefix
        "*mtp*",                 # MTP head — grafted back post-quantization in BF16
        "*shared_expert_gate*",  # MoE shared-expert gate; not in NVFP4_DEFAULT_CFG
                                 # (*mlp.gate.* doesn't match shared_expert_gate)
        "*linear_attn*",         # GatedDeltaNet full block — conservative exclusion;
                                 # 35B uses in_proj_a/b/ba (vs 27B in_proj_qkvz) but
                                 # NVFP4 vLLM loading of these combined projections is
                                 # unvalidated; out_proj could be quantized once confirmed
    ]
    for pattern in extra_disabled:
        cfg["quant_cfg"].append({"quantizer_name": pattern, "enable": False})

    return cfg


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calibration_loader(tokenizer, *, dataset: str, n_samples: int, seq_len: int):
    from datasets import load_dataset

    log.info("Loading calibration dataset %s (%d samples, max_len=%d)", dataset, n_samples, seq_len)
    ds = load_dataset(dataset, "LLM", split=f"train[:{n_samples}]")

    def forward_loop(model):
        model.eval()
        for i, sample in enumerate(ds):
            text = sample.get("text") or sample.get("content") or str(sample)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
                padding=False,
            ).to(next(model.parameters()).device)
            with torch.no_grad():
                model(**inputs)
            if (i + 1) % 50 == 0:
                log.info("  calibrated %d/%d samples", i + 1, n_samples)

    return forward_loop


# ---------------------------------------------------------------------------
# MTP graft helpers (identical logic to 27B script)
# ---------------------------------------------------------------------------

def extract_mtp_state_dict(model, model_id: str) -> dict:
    """Extract MTP tensors for grafting back after export.

    For Qwen3 MoE, AutoModelForCausalLM drops mtp.* during from_pretrained.
    Path 3 (direct safetensors read) is the reliable fallback.
    """
    # 1. Try modelopt's official helper
    try:
        from modelopt.torch.export.unified_export_hf import load_mtp_weights
        mtp_sd = load_mtp_weights(model)
        n = sum(len(v) for v in mtp_sd.values()) if isinstance(mtp_sd, dict) else len(mtp_sd)
        if n > 0:
            log.info("MTP state dict extracted via load_mtp_weights (%d tensors)", n)
            return mtp_sd
    except (ImportError, Exception):
        pass

    # 2. Try model state dict
    mtp_sd = {k: v.clone().to(torch.bfloat16)
               for k, v in model.state_dict().items()
               if "mtp" in k.lower()}
    if mtp_sd:
        log.info("MTP state dict extracted from model.state_dict (%d tensors)", len(mtp_sd))
        return mtp_sd

    # 3. Read directly from safetensors shards — most reliable for Qwen3 MoE
    log.info("mtp.* not in model state_dict — reading directly from safetensors shards")
    import safetensors.torch as st
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(model_id, ignore_patterns=["*.bin"])
    shards = sorted(glob.glob(os.path.join(local_path, "*.safetensors")))
    mtp_sd = {}
    for shard in shards:
        with st.safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    mtp_sd[key] = f.get_tensor(key).to(torch.bfloat16)

    log.info("MTP state dict extracted from safetensors shards (%d tensors)", len(mtp_sd))
    return mtp_sd


def graft_mtp_into_export(export_dir: Path, mtp_sd: dict) -> None:
    """Write MTP tensors into the exported checkpoint as a dedicated shard."""
    import safetensors.torch as st

    index_path = export_dir / "model.safetensors.index.json"
    if not index_path.exists():
        log.warning("No model.safetensors.index.json found — skipping MTP graft")
        return

    shard_name = "model-mtp.safetensors"
    shard_path = export_dir / shard_name
    st.save_file({k: v.contiguous() for k, v in mtp_sd.items()}, str(shard_path))
    log.info("Wrote MTP shard: %s (%d tensors)", shard_name, len(mtp_sd))

    with open(index_path) as f:
        index = json.load(f)
    for key in mtp_sd:
        index["weight_map"][key] = shard_name
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    log.info("Updated model.safetensors.index.json with mtp.* entries")


# ---------------------------------------------------------------------------
# config.json ignore list for vLLM
# ---------------------------------------------------------------------------

VLLM_IGNORE_PATTERNS = [
    "re:.*lm_head.*",
    "re:.*output_layer.*",
    "re:.*router.*",
    "re:.*mlp\\.gate\\..*",
    "re:.*shared_expert_gate.*",
    "re:.*linear_attn.*",
    "re:.*visual.*",
    "re:.*model\\.visual.*",
    "re:.*mtp.*",
]


def patch_config_json(export_dir: Path) -> None:
    config_path = export_dir / "config.json"
    if not config_path.exists():
        log.warning("config.json not found at %s — skipping quantization_config patch", config_path)
        return

    with open(config_path) as f:
        config = json.load(f)

    qcfg = config.setdefault("quantization_config", {})
    qcfg["quant_type"] = "nvfp4"
    qcfg["kv_cache_quant_type"] = "nvfp4"
    existing = set(qcfg.get("ignore", []))
    qcfg["ignore"] = sorted(existing | set(VLLM_IGNORE_PATTERNS))

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("config.json patched with quantization_config.ignore (%d patterns)", len(qcfg["ignore"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="NVFP4 quantization of Qwen3.6-35B-A3B for DGX Spark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",       default="Qwen/Qwen3.6-35B-A3B",
                   help="HuggingFace model ID or local path (use BF16 source, not FP8)")
    p.add_argument("--export-dir",  default="./qwen36-35b-a3b-nvfp4-mtp",
                   help="Output directory for the quantized checkpoint")
    p.add_argument("--dataset",     default="neuralmagic/calibration",
                   help="Calibration dataset (HuggingFace path, must have 'LLM' config)")
    p.add_argument("--n-samples",   type=int, default=512,
                   help="Number of calibration samples (512 recommended for MoE expert coverage)")
    p.add_argument("--seq-len",     type=int, default=8192,
                   help="Max sequence length for calibration tokens")
    p.add_argument("--batch-size",  type=int, default=1,
                   help="Calibration batch size (keep at 1 on Spark unified memory)")
    p.add_argument("--push-to-hub", default=None, metavar="HF_REPO_ID",
                   help="If set, push the quantized checkpoint to this HuggingFace repo")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Verify modelopt version
    try:
        import modelopt
        version = tuple(int(x) for x in modelopt.__version__.split(".")[:2])
        if version < (0, 41):
            log.error("nvidia-modelopt >= 0.41 required (found %s)", modelopt.__version__)
            sys.exit(1)
        log.info("nvidia-modelopt %s", modelopt.__version__)
    except ImportError:
        log.error("nvidia-modelopt not installed")
        sys.exit(1)

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export.unified_export_hf import export_hf_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Try the multimodal MoE class first (preserves any vision tower + MTP head).
    # Qwen3.6-35B-A3B is a text-only MoE in practice, but the class lookup is
    # harmless if it's registered. If not found, fall back to AutoModelForCausalLM.
    # In either case, MTP heads are extracted via direct safetensors reads (path 3)
    # since the CausalLM mapping drops mtp.* during from_pretrained.
    ModelClass = None
    for cls_name in ("Qwen3_5MoeForConditionalGeneration", "Qwen3MoeForCausalLM"):
        try:
            import transformers
            ModelClass = getattr(transformers, cls_name)
            log.info("Using %s", cls_name)
            break
        except AttributeError:
            continue

    if ModelClass is None:
        ModelClass = AutoModelForCausalLM
        log.info("Using AutoModelForCausalLM (specific MoE class not found in this transformers build)")

    # 1. Load model in BF16
    log.info("Loading %s in BF16 ...", args.model)
    model = ModelClass.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 2. Extract MTP before quantization
    mtp_state_dict = extract_mtp_state_dict(model, args.model)
    if not mtp_state_dict:
        log.error("MTP state dict is empty — verify model has MTP heads before continuing")
        sys.exit(1)
    log.info("Captured %d MTP tensors for graft", len(mtp_state_dict))

    # 3. Build quantization config
    quant_cfg = build_quant_cfg()
    log.info("Quantization config built (NVFP4, MoE-safe: routers+gates excluded by default)")

    # 4. Calibrate + quantize
    forward_loop = build_calibration_loader(
        tokenizer,
        dataset=args.dataset,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
    )
    log.info("Quantizing %d-B MoE — calibration ~10-20 min, export ~5-10 min ...", 35)
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    log.info("Quantization complete")

    # 5. Export
    log.info("Exporting to %s ...", export_dir)
    try:
        export_hf_checkpoint(
            model,
            export_dir=str(export_dir),
            extra_state_dict=mtp_state_dict,
        )
    except TypeError:
        export_hf_checkpoint(model, export_dir=str(export_dir))

    # Verify MTP made it in; graft manually if not
    index_path = export_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        mtp_present = any(k.startswith("mtp.") for k in index.get("weight_map", {}))
    else:
        mtp_present = False

    if not mtp_present:
        log.warning("MTP tensors missing from export — grafting manually")
        graft_mtp_into_export(export_dir, mtp_state_dict)
    else:
        log.info("MTP tensors confirmed in export index")

    log.info("Export complete")

    # 6. Patch config.json
    patch_config_json(export_dir)

    # 7. Copy tokenizer
    tokenizer.save_pretrained(str(export_dir))
    log.info("Tokenizer saved")

    # 8. Optional push to Hub
    if args.push_to_hub:
        log.info("Pushing to HuggingFace Hub: %s", args.push_to_hub)
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=args.push_to_hub,
            repo_type="model",
        )
        log.info("Upload complete: https://huggingface.co/%s", args.push_to_hub)

    log.info("")
    log.info("Done. Quantized checkpoint: %s", export_dir.resolve())
    log.info("")
    log.info("Serve on Spark (colocated on node 1, TP=1):")
    log.info("  ENABLE_NVFP4_SM100=0 VLLM_MARLIN_USE_ATOMIC_ADD=1 \\")
    log.info("  vllm serve %s \\", export_dir.resolve())
    log.info("    --quantization modelopt \\")
    log.info("    --kv-cache-dtype fp8 \\")
    log.info("    --load-format instanttensor \\")
    log.info("    --gpu-memory-utilization 0.46 \\")
    log.info("    --max-model-len 262144 \\")
    log.info("    --max-num-batched-tokens 16384 \\")
    log.info("    --enable-auto-tool-choice --tool-call-parser qwen3_xml \\")
    log.info("    --reasoning-parser qwen3 \\")
    log.info("    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'")
    log.info("")
    log.info("If KV alloc fails at 262144, drop --max-model-len to 131072.")


if __name__ == "__main__":
    main()
