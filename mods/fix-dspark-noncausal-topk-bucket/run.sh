#!/bin/bash
# fix-dspark-noncausal-topk-bucket — restore the DSpark draft's full sparse-attention
# candidate set on SM120 (DGX Spark / GB10).
#
# Problem: vLLM sizes the DSpark draft's non-causal index list as
#     noncausal_index_width = cdiv(window_size + num_speculative_tokens, 128) * 128
# (vllm/v1/attention/backends/mla/sparse_swa.py). DeepSeek-V4-Flash has
# sliding_window=128, so with ANY num_speculative_tokens >= 1 this rounds 133 -> 256.
# But FlashInfer's SM120 decode_dsv4 kernel is only compiled for topk buckets
# {128, 512, 1024} (_DECODE_DSV4_DISPATCH in flashinfer/mla/_sparse_mla_sm120.py).
# 256 has no bucket, so mods/fix-dspark-sm120-topk rescues the dispatch by truncating
# indices[..., :128] — which silently DROPS the tail of the index list. That tail is
# exactly the non-causal part: the query/speculative tokens each draft position needs
# in order to see the other speculated tokens. Result: correct output (the target
# verifies every proposal) but collapsed accept-length, worst at deep positions.
#
# Measured on this cluster (DeepSeek-V4-Flash-0731, TP=2, num_speculative_tokens=5),
# per-position acceptance with the truncation active:
#     0.79 / 0.57 / 0.40 / 0.21 / 0.067   (mean accept length ~2.7-3.0)
#
# Fix: round the index width UP to the nearest bucket the kernel actually has
# (256 -> 512) instead of up to the nearest multiple of 128. The draft then keeps all
# ~133 real candidates and dispatches natively; fix-dspark-sm120-topk becomes a no-op.
#
# Why this is safe:
#   * The fill kernel (_compute_dspark_noncausal_swa_indices_kernel) already writes the
#     FULL index_width and pads every slot past swa_len with -1, and swa_lens is passed
#     to the kernel as topk_length. A wider buffer just gets more -1 padding.
#   * FlashInfer allocates its split-K scratch from indices.shape[-1] at call time
#     (_maybe_allocate_decode_scratch), so num_splits sizes itself correctly (4 -> 8).
#     There is no caller-supplied scratch to keep in sync.
#   * 512 is still a multiple of 128, satisfying the kernel's B_TOPK head padding.
#   * (num_heads, 512) is in _DECODE_DSV4_DISPATCH for num_heads in {8, 16, 32};
#     DSV4's 64 attention heads at TP=2 give 32 local heads.
#   * Extra cost is a 2x-wider sparse attention across the 3 draft layers only, plus
#     ~8 MB for the index buffer. The draft is tiny next to the target's verify pass.
#
# Apply this ALONGSIDE mods/fix-dspark-sm120-topk (different files, any order); keep
# that one as a fallback for shapes this rounding cannot reach.
set -e

F="$(python3 -c 'import os,vllm; print(os.path.join(os.path.dirname(vllm.__file__),"v1","attention","backends","mla","sparse_swa.py"))' 2>/dev/null)"
[ -f "$F" ] || { echo "[fix-dspark-noncausal-topk-bucket] ERROR: sparse_swa.py not found ($F)" >&2; exit 1; }

python3 - "$F" <<'PY'
import sys
f = sys.argv[1]
s = open(f).read()
if "fix-dspark-noncausal-topk-bucket" in s:
    print("[fix-dspark-noncausal-topk-bucket] already applied"); raise SystemExit(0)

anchor = """        self.noncausal_index_width = (
            cdiv(self.window_size + self.num_speculative_tokens, 128) * 128
            if self.is_dspark
            else 0
        )
"""
if anchor not in s:
    raise SystemExit(
        "[fix-dspark-noncausal-topk-bucket] ERROR: anchor not found — vLLM version drift"
    )

replacement = """        # fix-dspark-noncausal-topk-bucket: round the draft's non-causal index width
        # up to a topk bucket FlashInfer's SM120 decode_dsv4 actually has compiled
        # ({128,512,1024}), not merely up to a multiple of 128. The stock expression
        # yields 256 (window 128 + spec tokens), which has no SM120 bucket and forces
        # a lossy truncation to 128 at dispatch, gutting draft accept-length. Slots
        # past swa_len are -1 padded by the fill kernel and bounded by topk_length,
        # so a wider buffer is inert; FlashInfer sizes its split-K scratch from the
        # index width at call time, so nothing else needs resizing.
        _DSV4_TOPK_BUCKETS = (128, 512, 1024)
        _needed = cdiv(self.window_size + self.num_speculative_tokens, 128) * 128
        _bucket = [b for b in _DSV4_TOPK_BUCKETS if b >= _needed]
        self.noncausal_index_width = (
            (min(_bucket) if _bucket else _needed) if self.is_dspark else 0
        )
"""
open(f, "w").write(s.replace(anchor, replacement, 1))
print(
    "[fix-dspark-noncausal-topk-bucket] applied: draft index width now rounds to a "
    "compiled SM120 bucket (256 -> 512)"
)
PY

python3 -c "import ast; ast.parse(open('$F').read()); print('[fix-dspark-noncausal-topk-bucket] verified (AST valid)')"
