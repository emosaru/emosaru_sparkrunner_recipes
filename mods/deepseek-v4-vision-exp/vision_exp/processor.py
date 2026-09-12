"""vLLM 0.25 multimodal processor for DeepSeek-V4-Flash-Vision-Exp.

Images only. The checkpoint has no video encoder; GIF is a still frame via PIL.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import time
from threading import Lock, local
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    TimingContext,
)

from .image_processor import (
    IMAGE_PAD,
    IMAGE_PLACEHOLDER,
    as_pil,
    build_image_block,
    pil_to_patches,
    vision_args_from_config,
)


def _image_token_id(tokenizer) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        token_id = convert(IMAGE_PLACEHOLDER)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return int(token_id)
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    if IMAGE_PLACEHOLDER in vocab:
        return int(vocab[IMAGE_PLACEHOLDER])
    # Vision-Exp added-token id (vocab_size 129280; placeholder is in the tail).
    return 129264



# LOCAL DELTA (emosaru, 2026-09-01): PER-THREAD tokenizer clones.
#
# vLLM preprocesses multimodal inputs in a THREAD POOL
# (renderers/base.py::_process_multimodal_async -> concurrent.futures.thread).
# The HF fast tokenizer is a Rust object with interior mutability: every encode()
# reaches set_truncation_and_padding() -> self._tokenizer.no_truncation(), which
# takes a MUTABLE borrow. Concurrent users race and one dies with
#
#     RuntimeError: Already borrowed
#
# surfacing as HTTP 500. Measured live 2026-09-01: 7/326 requests (2.1%) under
# light use, and 5/12 under a deliberate 12-way concurrent image burst.
#
# A LOCK IS NOT SUFFICIENT — tried and measured: with a mod-local lock held, the
# burst still failed 5/12, because the competing borrow comes from vLLM's OWN use
# of the same shared tokenizer, which our lock cannot serialize. The only robust
# fix is to stop sharing: hand each thread its own clone.
#
# copy.deepcopy of this tokenizer is cheap (measured 0.23s, one-off per thread)
# and verified to produce byte-identical encodings. Clone creation is itself
# guarded + retried, since the deepcopy reads the shared object and can lose the
# same race while doing so.
_TLS = local()
_CLONE_LOCK = Lock()


def _thread_local_tokenizer(shared):
    """Per-thread tokenizer clone; falls back to the shared object if cloning fails."""
    tok = getattr(_TLS, "tok", None)
    if tok is not None:
        return tok
    for attempt in range(5):
        try:
            with _CLONE_LOCK:
                tok = copy.deepcopy(shared)
            _TLS.tok = tok
            return tok
        except RuntimeError:
            # lost the borrow race while cloning — back off and retry
            time.sleep(0.02 * (attempt + 1))
    return shared


def _collect_images(mm_data: Mapping[str, object]) -> list[Any]:
    for key in ("images", "image"):
        value = mm_data.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
    return []


class DeepseekV4VisionExpProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 16}

    def get_hf_processor(self, **kwargs: object):
        raise RuntimeError(
            "DeepSeek-V4-Flash-Vision-Exp has no Hugging Face processor; "
            "the vLLM Vision-Exp processor handles images directly."
        )

    def get_max_image_tokens(self) -> int:
        return int(getattr(self.get_hf_config(), "vision_max_n_token", 384))


class DeepseekV4VisionExpDummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV4VisionExpProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_overrides = mm_options.get("image")
        # Oversized square; official safe_resize caps at vision_max_n_token.
        return {
            "image": self._get_dummy_images(
                width=2048,
                height=2048,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class DeepseekV4VisionExpMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV4VisionExpProcessingInfo]
):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], Any, bool]:
        # compress_pad depends on the expanded token position of each image,
        # so a content-only processor cache would reuse the wrong layout.
        if inputs.mm_data_items.get_count("image", strict=False) > 0:
            return self._apply_hf_processor(inputs, timing_ctx)
        return super()._cached_apply_hf_processor(inputs, timing_ctx)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = _thread_local_tokenizer(self.info.get_tokenizer())
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            raise RuntimeError("Vision-Exp processor requires tokenizer.encode")
        prompt_ids = list(encode(prompt, add_special_tokens=False))
        images = _collect_images(mm_data)
        if not images:
            return BatchFeature(
                data={"input_ids": torch.tensor([prompt_ids], dtype=torch.long)}
            )

        args = vision_args_from_config(self.info.get_hf_config())
        image_token_id = _image_token_id(tokenizer)
        max_tokens = args.vision_max_n_token
        pixel_rows: list[torch.Tensor] = []
        n_vit_h_rows: list[int] = []
        n_vit_w_rows: list[int] = []
        types_rows: list[torch.Tensor] = []
        perm_rows: list[torch.Tensor] = []
        num_token_rows: list[int] = []
        max_patches = 1

        expanded_len = 0
        image_iter = iter(images)
        for tok in prompt_ids:
            if tok != image_token_id:
                expanded_len += 1
                continue
            pil = as_pil(next(image_iter))
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = pil_to_patches(pil, args)
            types, perm = build_image_block(n_llm_h, n_llm_w, expanded_len)
            if types.numel() > max_tokens:
                raise ValueError(
                    f"Image token block length {types.numel()} exceeds "
                    f"vision_max_n_token={max_tokens}"
                )
            pixel_rows.append(patches)
            n_vit_h_rows.append(int(n_vit_h))
            n_vit_w_rows.append(int(n_vit_w))
            types_rows.append(types)
            perm_rows.append(perm)
            num_token_rows.append(int(types.numel()))
            max_patches = max(max_patches, int(patches.shape[0]))
            expanded_len += int(types.numel())

        leftover = list(image_iter)
        if leftover:
            raise ValueError(
                f"Found {len(images)} images but only "
                f"{len(images) - len(leftover)} {IMAGE_PLACEHOLDER!r} placeholders"
            )
        if len(pixel_rows) != prompt_ids.count(image_token_id):
            raise ValueError(
                f"Found {prompt_ids.count(image_token_id)} image placeholders "
                f"but processed {len(pixel_rows)} images"
            )

        padded_pixels = []
        for patches in pixel_rows:
            if patches.shape[0] < max_patches:
                pad = patches.new_zeros(
                    max_patches - patches.shape[0], *patches.shape[1:]
                )
                patches = torch.cat([patches, pad], dim=0)
            padded_pixels.append(patches.to(torch.bfloat16))

        def _pad_1d(rows: list[torch.Tensor], fill: int, width: int) -> torch.Tensor:
            out = []
            for row in rows:
                if row.numel() < width:
                    row = torch.cat(
                        [
                            row,
                            torch.full((width - row.numel(),), fill, dtype=row.dtype),
                        ]
                    )
                out.append(row[:width])
            return torch.stack(out, dim=0)

        return BatchFeature(
            data={
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
                "pixel_values": torch.stack(padded_pixels, dim=0),
                "n_vit_h": torch.tensor(n_vit_h_rows, dtype=torch.int32),
                "n_vit_w": torch.tensor(n_vit_w_rows, dtype=torch.int32),
                "types": _pad_1d(types_rows, IMAGE_PAD, max_tokens),
                "perm": _pad_1d(perm_rows, -1, max_tokens),
                "num_tokens": torch.tensor(num_token_rows, dtype=torch.int32),
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            n_vit_h=MultiModalFieldConfig.batched("image"),
            n_vit_w=MultiModalFieldConfig.batched("image"),
            types=MultiModalFieldConfig.batched("image"),
            perm=MultiModalFieldConfig.batched("image"),
            num_tokens=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ) -> Sequence[PromptUpdate]:
        image_token_id = _image_token_id(self.info.get_tokenizer())

        def get_replacement(item_idx: int):
            item = out_mm_kwargs["image"][item_idx]
            raw = item["num_tokens"]
            num_tokens = raw.data if hasattr(raw, "data") else raw
            if hasattr(num_tokens, "reshape"):
                num_tokens = int(num_tokens.reshape(-1)[0].item())
            else:
                num_tokens = int(num_tokens)
            if num_tokens <= 0:
                raise ValueError(f"Image {item_idx} produced 0 vision tokens")
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]


def register_vision_exp_processor(model_cls):
    return MULTIMODAL_REGISTRY.register_processor(
        DeepseekV4VisionExpMultiModalProcessor,
        info=DeepseekV4VisionExpProcessingInfo,
        dummy_inputs=DeepseekV4VisionExpDummyInputsBuilder,
    )(model_cls)
