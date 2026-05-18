#!/usr/bin/env python3
"""
NVFP4 quantization of Qwen/Qwen3.6-27B targeting DGX Spark (GB10, SM 121a).

Produces a checkpoint with:
  - Text transformer layers quantized to NVFP4 / FP8 KV
  - Vision tower kept in BF16 (multimodal capability preserved)
  - MTP head kept in BF16 and grafted into the export

Requirements:
  nvidia-modelopt >= 0.41 (0.43+ recommended)
  transformers >= 4.51
  datasets, huggingface_hub, torch >= 2.5

Run inside a modelopt-capable environment, e.g. the DGX Spark's nvcr.io/nvidia/nemo container
or the eugr nightly vLLM container with modelopt installed.

Exclusion rationale
-------------------
*lm_head*, *output_layer*, *router*, *mlp.gate.*   — NVFP4_DEFAULT_CFG defaults, always excluded
*visual*, *model.visual*                            — vision tower; BF16 cost is ~1.4 GB, must
                                                      stay for multimodal inference
*mtp*                                               — MTP head; grafted back post-quantization
*linear_attn*  (full block)                         — conservative: avoids the vLLM in_proj_qkvz
                                                      naming bug (#40252) that silently zeros the
                                                      layer and produces garbage output.
                                                      Specifically: conv1d is a correctness failure
                                                      on long context; the combined in_proj_qkvz /
                                                      in_proj_ba tensor names differ from what
                                                      older vLLM quantization_config ignore patterns
                                                      expect, causing silent misloads.
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

    # modelopt 0.44+ uses a list of rule dicts; append our additions at the end
    # so they take precedence over any earlier enabling rules.
    # Note: *linear_attn.conv1d* and *mixer.conv1d* are already in the defaults;
    # we add the full *linear_attn* block for the conservative vLLM-safety reason.
    extra_disabled = [
        "*visual*",        # vision tower — keep multimodal capabilities
        "*model.visual*",  # alternate vision tower prefix
        "*mtp*",           # MTP head — grafted back post-quantization in BF16
        "*linear_attn*",   # GatedDeltaNet full block — avoids vLLM in_proj_qkvz
                           # naming bug that silently zeros layers (#40252)
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
    # neuralmagic/calibration requires 'LLM' config; trust_remote_code removed in datasets 3+
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
            if (i + 1) % 5 == 0:
                log.info("  calibrated %d/%d samples", i + 1, n_samples)

    return forward_loop


# ---------------------------------------------------------------------------
# MTP graft helpers
# ---------------------------------------------------------------------------

def extract_mtp_state_dict(model, model_id: str) -> dict:
    """
    Extract MTP tensors for grafting back after export.

    Qwen3_5ForConditionalGeneration drops mtp.* keys during from_pretrained
    because MTP is not part of its architecture definition — so neither
    load_mtp_weights() nor model.state_dict() will find them. The reliable
    path is to read them directly from the original safetensors shards.
    """
    # 1. Try modelopt's official helper (modelopt >= 0.41, may return 0 tensors)
    try:
        from modelopt.torch.export.unified_export_hf import load_mtp_weights
        mtp_sd = load_mtp_weights(model)
        n = sum(len(v) for v in mtp_sd.values()) if isinstance(mtp_sd, dict) else len(mtp_sd)
        if n > 0:
            log.info("MTP state dict extracted via load_mtp_weights (%d tensors)", n)
            return mtp_sd
    except (ImportError, Exception):
        pass

    # 2. Try model state dict (works if the model class retained MTP)
    mtp_sd = {k: v.clone().to(torch.bfloat16)
               for k, v in model.state_dict().items()
               if "mtp" in k.lower()}
    if mtp_sd:
        log.info("MTP state dict extracted from model.state_dict (%d tensors)", len(mtp_sd))
        return mtp_sd

    # 3. Read directly from safetensors shards — the model class drops mtp.*
    #    during from_pretrained, but the tensors are present in the files.
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
    """
    Write MTP tensors into the exported checkpoint as a dedicated shard and
    update model.safetensors.index.json to include them.

    Used as a fallback when export_hf_checkpoint's extra_state_dict param
    is unavailable or silently drops the tensors (modelopt + transformers>=5.0
    experimental path).
    """
    import safetensors.torch as st

    index_path = export_dir / "model.safetensors.index.json"
    if not index_path.exists():
        log.warning("No model.safetensors.index.json found — skipping MTP graft")
        return

    # Write a dedicated mtp shard
    shard_name = "model-mtp.safetensors"
    shard_path = export_dir / shard_name
    st.save_file({k: v.contiguous() for k, v in mtp_sd.items()}, str(shard_path))
    log.info("Wrote MTP shard: %s (%d tensors)", shard_name, len(mtp_sd))

    # Update the index
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
    p = argparse.ArgumentParser(description="NVFP4 quantization of Qwen3.6-27B for DGX Spark")
    p.add_argument("--model",        default="Qwen/Qwen3.6-27B",
                   help="HuggingFace model ID or local path")
    p.add_argument("--export-dir",   default="./qwen36-27b-nvfp4-mtp",
                   help="Output directory for the quantized checkpoint")
    p.add_argument("--dataset",      default="neuralmagic/calibration",
                   help="Calibration dataset (HuggingFace path)")
    p.add_argument("--n-samples",    type=int, default=20,
                   help="Number of calibration samples")
    p.add_argument("--seq-len",      type=int, default=8192,
                   help="Max sequence length for calibration")
    p.add_argument("--batch-size",   type=int, default=1,
                   help="Calibration batch size (keep at 1 on Spark unified memory)")
    p.add_argument("--push-to-hub",  default=None, metavar="HF_REPO_ID",
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
    from transformers import AutoTokenizer

    # Qwen3_5ForConditionalGeneration is the multimodal class that preserves
    # both the vision tower and the MTP head. AutoModelForCausalLM silently
    # drops both.
    try:
        from transformers import Qwen3_5ForConditionalGeneration as ModelClass
        log.info("Using Qwen3_5ForConditionalGeneration (preserves vision + MTP)")
    except ImportError:
        log.error("Qwen3_5ForConditionalGeneration not found — update transformers to >= 4.51")
        sys.exit(1)

    # 1. Load model in BF16
    log.info("Loading %s in BF16 ...", args.model)
    model = ModelClass.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 2. Extract MTP before quantization (quantize() modifies model in-place).
    #    Qwen3_5ForConditionalGeneration drops mtp.* during from_pretrained, so
    #    extract_mtp_state_dict falls through to reading the safetensors shards directly.
    mtp_state_dict = extract_mtp_state_dict(model, args.model)
    if not mtp_state_dict:
        log.error("MTP state dict is empty — verify model has MTP heads before continuing")
        sys.exit(1)
    log.info("Captured %d MTP tensors for graft", len(mtp_state_dict))

    # 3. Build quantization config
    quant_cfg = build_quant_cfg()
    log.info("Quantization config built (NVFP4, conservative linear_attn exclusion)")

    # 4. Calibrate + quantize
    forward_loop = build_calibration_loader(
        tokenizer,
        dataset=args.dataset,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
    )
    log.info("Quantizing — this takes 10–20 min on a single Spark ...")
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    log.info("Quantization complete")

    # 5. Export. Try extra_state_dict first (modelopt >= 0.41); if the export
    #    silently drops the MTP tensors (known issue with transformers >= 5.0),
    #    graft them in manually as a dedicated shard afterwards.
    log.info("Exporting to %s ...", export_dir)
    try:
        export_hf_checkpoint(
            model,
            export_dir=str(export_dir),
            extra_state_dict=mtp_state_dict,
        )
    except TypeError:
        # extra_state_dict not supported in this modelopt build
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

    # 6. Patch config.json with vLLM-compatible quantization_config.ignore list
    patch_config_json(export_dir)

    # 7. Copy tokenizer files
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
    log.info("Serve on Spark with:")
    log.info("  ENABLE_NVFP4_SM100=0 vllm serve %s \\", export_dir.resolve())
    log.info("    --quantization modelopt \\")
    log.info("    --kv-cache-dtype fp8 \\")
    log.info("    --gpu-memory-utilization 0.85 \\")
    log.info("    --max-model-len 131072 \\")
    log.info("    --max-num-seqs 2 \\")
    log.info("    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'")


if __name__ == "__main__":
    main()
