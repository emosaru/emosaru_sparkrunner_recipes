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
import json
import logging
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
    overrides = {
        # Vision tower — keep multimodal capabilities
        "*visual*":              {"enable": False},
        "*model.visual*":        {"enable": False},
        # MTP head — grafted back post-quantization in BF16
        "*mtp*":                 {"enable": False},
        # GatedDeltaNet / linear_attn — full-block conservative exclusion.
        # conv1d causes recurrence drift on long context (correctness failure);
        # full exclusion avoids vLLM loading-name mismatch bug on the matmul layers.
        "*linear_attn*":         {"enable": False},
        "*linear_attn.conv1d*":  {"enable": False},
        "*mixer.conv1d*":        {"enable": False},
    }
    cfg["quant_cfg"] = {**cfg["quant_cfg"], **overrides}
    return cfg


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calibration_loader(tokenizer, *, dataset: str, n_samples: int, seq_len: int):
    from datasets import load_dataset

    log.info("Loading calibration dataset %s (%d samples, max_len=%d)", dataset, n_samples, seq_len)
    ds = load_dataset(dataset, split=f"train[:{n_samples}]", trust_remote_code=True)

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

def extract_mtp_state_dict(model) -> dict:
    """
    Pull MTP tensors from the loaded BF16 model before quantization modifies it.
    Falls back to modelopt's load_mtp_weights() if available (modelopt >= 0.41).
    """
    # Preferred: use modelopt's official helper
    try:
        from modelopt.torch.export.unified_export_hf import load_mtp_weights
        mtp_sd = load_mtp_weights(model)
        n = sum(len(v) for v in mtp_sd.values()) if isinstance(mtp_sd, dict) else len(mtp_sd)
        log.info("MTP state dict extracted via load_mtp_weights (%d tensors)", n)
        return mtp_sd
    except ImportError:
        pass

    # Fallback: manual extraction
    mtp_sd = {k: v.clone().to(torch.bfloat16)
               for k, v in model.state_dict().items()
               if "mtp" in k.lower()}
    log.info("MTP state dict extracted manually (%d tensors)", len(mtp_sd))
    return mtp_sd


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

    # 2. Extract MTP before quantization (quantize() modifies model in-place)
    mtp_state_dict = extract_mtp_state_dict(model)
    if not mtp_state_dict:
        log.error("MTP state dict is empty — verify model has MTP heads before continuing")
        sys.exit(1)
    mtp_count = sum(len(v) for v in mtp_state_dict.values()) if isinstance(mtp_state_dict, dict) else len(mtp_state_dict)
    log.info("Captured %d MTP tensors for graft", mtp_count)

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

    # 5. Export with MTP grafted back in BF16
    log.info("Exporting to %s (with MTP graft) ...", export_dir)
    export_hf_checkpoint(
        model,
        export_dir=str(export_dir),
        extra_state_dict=mtp_state_dict,
    )
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
