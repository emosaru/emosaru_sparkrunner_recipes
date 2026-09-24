#!/bin/bash
# Reduced-vocabulary MTP drafting for Qwen3.8-Flash-Next.
#
# Restricts the MTP DRAFT head's projection to the lowest N token ids (default 65,536
# of 248,320). The win is BANDWIDTH, not acceptance: the draft head is read once per
# speculative position per decode step, so shrinking the rows it touches shrinks the
# dominant cost of drafting.
#
# CORRECTNESS IS STRUCTURAL, NOT EMPIRICAL. Masked ids get -inf, so the draft never
# PROPOSES them. The TARGET model's own lm_head is untouched and still verifies over
# the full vocabulary, so a token outside the set is rejected exactly like any other
# bad draft. Emitted text is unchanged; only speed moves. Poor coverage costs
# acceptance, never correctness.
#
# OFF unless VLLM_MTP_DRAFT_VOCAB_SIZE is set in the container environment. Applying
# this mod without that variable is a no-op, by design.
#
# *** UNTESTED: written 2026-09-18 against eugr/spark-vllm-b12x:latest, NEVER BOOTED. ***
# Validating it requires stopping production. See PROVENANCE.md.
set -euo pipefail

python3 - <<'PYEOF'
import os, py_compile, shutil, sys

# The b12x fork renamed the model tree between images: it was
# `vllm/models/qwen3_8_flash_next/mtp.py` on the 2026-09-10 image and is
# `vllm/models/qwen4_exp/nvidia/mtp.py` on nightly-20260923. Resolve across both
# rather than pinning one, and fail loudly if neither is present.
_BASE = "/usr/local/lib/python3.12/dist-packages/vllm/models/"
_CANDIDATES = ("qwen4_exp/nvidia/mtp.py", "qwen3_8_flash_next/mtp.py")
M = next((_BASE + c for c in _CANDIDATES if os.path.isfile(_BASE + c)), None)
if M is None:
    raise SystemExit("mtp-reduced-draft-vocab: no MTP module found at any known path "
                     f"({', '.join(_CANDIDATES)}) — re-derive against this image.")
print(f"mtp-reduced-draft-vocab: patching {M}")
src = open(M).read()

if "_install_reduced_draft_vocab" in src:
    print("mtp-reduced-draft-vocab: already applied, nothing to do")
    sys.exit(0)

def edit(s, old, new, what):
    if old not in s:
        raise SystemExit(f"mtp-reduced-draft-vocab: ANCHOR MISSING ({what}) — the "
                         "fork moved; re-derive this mod against the new image.")
    if s.count(old) != 1:
        raise SystemExit(f"mtp-reduced-draft-vocab: anchor not unique ({what}), "
                         f"found {s.count(old)}")
    return s.replace(old, new, 1)

HELPER = '''
from vllm.logger import init_logger

# Private name: this module defines no `logger` of its own, and a bare `logger`
# would collide if the fork later adds one.
_rdv_logger = init_logger(__name__)

_REDUCED_VOCAB_ENV = "VLLM_MTP_DRAFT_VOCAB_SIZE"


def _install_reduced_draft_vocab(logits_processor, lm_head) -> None:
    """Restrict the MTP draft head's projection to the lowest N token ids.

    Installs a replacement ``_apply_head`` on the DRAFT head's own
    LogitsProcessor instance, so the gather/vocab-trim logic in ``_get_logits``
    is inherited untouched and the main model is unaffected.

    The reduced set is a contiguous id PREFIX, which makes it a contiguous row
    prefix of every vocab-parallel shard: the projection reads ``weight[:k]``
    with no gather and no resident copy. At TP=2 with N=65536 the whole set
    lands on rank 0, so rank 1 skips its matmul entirely (k == 0).
    """
    import os

    raw = os.environ.get(_REDUCED_VOCAB_ENV, "").strip()
    if not raw:
        return
    try:
        n_keep = int(raw)
    except ValueError:
        raise ValueError(
            f"{_REDUCED_VOCAB_ENV}={raw!r} is not an integer"
        ) from None
    if n_keep <= 0:
        raise ValueError(f"{_REDUCED_VOCAB_ENV} must be positive, got {n_keep}")

    import torch.nn.functional as F

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    def _off(reason: str) -> None:
        _rdv_logger.warning(
            "Reduced-vocab MTP drafting DISABLED: %s", reason
        )

    if not isinstance(lm_head, VocabParallelEmbedding):
        return _off("draft head is not a VocabParallelEmbedding")
    # A quantized head is not a plain row-sliceable matrix (scales, swizzling),
    # so the fast path is only safe on an unquantized one.
    if not isinstance(
        lm_head.quant_method,
        (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
    ):
        return _off("draft head is quantized (e.g. VLLM_MTP_NVFP4_LM_HEAD)")
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.ndim != 2:
        return _off("draft head has no 2-D weight")

    si = lm_head.shard_indices
    if (si.added_vocab_end_index - si.added_vocab_start_index) > 0:
        # Added-vocab rows sit after the org rows and the padding, so a row
        # prefix would mask them incorrectly.
        return _off("added vocabulary (LoRA) is present")

    start = si.org_vocab_start_index
    n_org = si.org_vocab_end_index - start
    k = min(max(n_keep - start, 0), n_org)
    if k >= n_org:
        _rdv_logger.info(
            "Reduced-vocab MTP drafting: rank keeps its whole shard "
            "(N=%d covers ids [%d, %d)); nothing to do.",
            n_keep, start, si.org_vocab_end_index,
        )
        return

    rows = lm_head.num_embeddings_per_partition
    orig_apply_head = logits_processor._apply_head
    head_dtype = logits_processor.head_dtype

    def _apply_head(lm_head_arg, hidden_states, embedding_bias):
        # Delegate the cases this fast path does not model.
        if embedding_bias is not None or (
            head_dtype is not None and head_dtype != hidden_states.dtype
        ):
            return orig_apply_head(lm_head_arg, hidden_states, embedding_bias)
        w = lm_head_arg.weight
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = flat.new_full((flat.shape[0], rows), float("-inf"), dtype=w.dtype)
        if k > 0:
            out[:, :k] = F.linear(flat.to(w.dtype), w[:k])
        return out.reshape(*hidden_states.shape[:-1], rows)

    logits_processor._apply_head = _apply_head
    _rdv_logger.info(
        "Reduced-vocab MTP drafting ENABLED: drafting over the lowest %d ids; "
        "this rank projects %d of its %d org rows (shard ids [%d, %d)). "
        "Correctness is unaffected — the target verifies the full vocabulary.",
        n_keep, k, n_org, start, si.org_vocab_end_index,
    )


'''

# 1. module-level helper, inserted ahead of the first module-level def
src = edit(src, "def _mtp_api() -> Any:", HELPER.lstrip("\n") + "def _mtp_api() -> Any:",
           "helper insertion point")

# 2. install it on the DRAFT head's own LogitsProcessor, right after it is built
src = edit(src,
  "        self.logits_processor = LogitsProcessor(\n"
  "            config.vocab_size,\n"
  "            lm_head=self.lm_head,\n"
  "        )\n",
  "        self.logits_processor = LogitsProcessor(\n"
  "            config.vocab_size,\n"
  "            lm_head=self.lm_head,\n"
  "        )\n"
  "        _install_reduced_draft_vocab(self.logits_processor, self.lm_head)\n",
  "logits_processor construction")

# The module defines no logger of its own, so the helper brings one.
if "_rdv_logger = init_logger(__name__)" not in src:
    raise SystemExit("mtp-reduced-draft-vocab: logger not installed; re-derive.")
if src.count("_rdv_logger = init_logger(__name__)") != 1:
    raise SystemExit("mtp-reduced-draft-vocab: duplicate logger install; re-derive.")

shutil.copyfile(M, M + ".pre-reduced-draft-vocab")
open(M, "w").write(src)
py_compile.compile(M, doraise=True)
print("mtp-reduced-draft-vocab: applied to", M)
PYEOF
