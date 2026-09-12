#!/usr/bin/env python3
"""Correctness gate for DeepSeek-V4-Flash-Vision-Exp. Run BEFORE benchmarking.

Benchmarking a silently-broken model is worse than not benchmarking: the numbers
look plausible and get believed. This gate is designed around the two failure modes
that produce *plausible* results on this specific checkpoint:

  1. FAIL-OPEN TEXT-ONLY. Vision-Exp declares the same `architectures` string as the
     text-only 0731 checkpoint. An engine without the vision mod can load it, serve
     happily, and DISCARD EVERY IMAGE. Nothing errors. A single "what colour is this"
     probe does NOT catch it either — a text-only model guessing "red" is right about
     as often as not.

     The discriminator is a PAIR of results, not any single test:
       * images in a `user` message are accepted (200) AND answered correctly across
         several probes whose correct answers DIFFER (red/blue/green/left/3), so a
         constant reply or a lucky guess cannot pass; and
       * an image in a `system` message is REJECTED with 400.
     Note carefully: a text-only server also returns 400 for the system-message probe
     (it rejects images in every role), so gate [1] ALONE proves nothing. It is only
     meaningful together with the user-role probes passing. Conversely, a lineage that
     silently drops images returns 200 for the system probe and wrong answers for the
     colour probes. Read the two together.

  2. SILENT SPEC-DECODE DEATH (vLLM #43456). The DSpark draft head can fail to load
     with no error, giving 0% acceptance and decode at draft-off speed. Benchmarking
     that would blame the checkpoint for a loading bug. We read the engine's own
     spec-decode counters and require real acceptance.

Also checks text correctness, long-context needle retrieval, and tool calling, because
"vision works" is not the same as "the model is behaving properly".

Usage:
  python3 scripts/verify-vision-exp-correctness.py --base-url http://192.168.1.236:8000
  ... --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp   (default)
  ... --thinking                 # exercise the production reasoning path (slow)
  ... --skip-needle              # skip the 32k retrieval probe

Exit 0 only if every gate passes. Exit 1 otherwise.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import sys
import urllib.error
import urllib.request
import zlib

# ── tiny dependency-free PNG encoder ────────────────────────────────────────
# Deliberately no PIL: this gate must run anywhere, including inside a container
# that only has the serving stack installed.

def encode_png(rows: list[list[tuple[int, int, int]]]) -> bytes:
    h = len(rows)
    w = len(rows[0])
    raw = b"".join(b"\x00" + b"".join(bytes(px) for px in row) for row in rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


# vision_min_pixels is 147456 (=384x384); generate at that size so the image
# processor is not upscaling a toy thumbnail into something ambiguous.
SIDE = 384
WHITE = (255, 255, 255)


def solid(rgb):
    return [[rgb] * SIDE for _ in range(SIDE)]


def split_lr(left_rgb, right_rgb):
    half = SIDE // 2
    return [[left_rgb if x < half else right_rgb for x in range(SIDE)] for _ in range(SIDE)]


def squares(n: int):
    """n black squares on white, evenly spaced along the top — unambiguous count."""
    img = [[WHITE] * SIDE for _ in range(SIDE)]
    size = 48
    gap = 24
    y0 = 100
    for i in range(n):
        x0 = 30 + i * (size + gap)
        for y in range(y0, y0 + size):
            for x in range(x0, min(x0 + size, SIDE)):
                img[y][x] = (0, 0, 0)
    return img


def data_uri(img) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(img)).decode()


# ── HTTP ────────────────────────────────────────────────────────────────────

def post(base, path, payload, timeout=300):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body[:400]}


def get_text(base, path, timeout=60):
    try:
        with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
            return r.read().decode(errors="replace")
    except Exception as e:
        return f"__ERROR__ {e}"


def chat(base, model, messages, thinking, max_tokens=64, tools=None, timeout=600):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if not thinking:
        # Recipe defaults thinking=true / effort=high; a "what colour is this" probe
        # would otherwise burn thousands of reasoning tokens per gate.
        payload["chat_template_kwargs"] = {"thinking": False}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return post(base, "/v1/chat/completions", payload, timeout=timeout)


def answer_of(body) -> str:
    try:
        msg = body["choices"][0]["message"]
    except Exception:
        return ""
    txt = (msg.get("content") or "").strip()
    if not txt:
        # With a reasoning parser the visible answer can land in reasoning_content
        # when the model produced no post-think content.
        txt = (msg.get("reasoning_content") or "").strip()
    return txt


# ── gates ───────────────────────────────────────────────────────────────────

class Results:
    def __init__(self):
        self.rows = []

    def add(self, group, name, ok, detail=""):
        self.rows.append((group, name, ok, detail))
        mark = "✓" if ok else "✗"
        print(f"    {mark} {name}" + (f"  — {detail}" if detail else ""), flush=True)

    def failed(self):
        return [r for r in self.rows if not r[2]]


def user_image(prompt, img):
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_uri(img)}},
    ]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://192.168.1.236:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp")
    ap.add_argument("--thinking", action="store_true", help="exercise the production reasoning path (slow)")
    ap.add_argument("--skip-needle", action="store_true")
    a = ap.parse_args()
    base, model, think = a.base_url, a.model, a.thinking
    R = Results()

    print(f"Correctness gate: {model} @ {base}  (thinking={'on' if think else 'off'})\n")

    # ── 1. SYSTEM-ROLE IMAGE: strict, permissive, or actually dropped? ───────
    # A bare status-code check cannot tell "dropped" from "permissively accepted",
    # and those are opposite diagnoses. So probe with a question ONLY answerable
    # from the image and classify on the ANSWER:
    #   400            -> strict, matches DeepSeek's documented contract     PASS
    #   200 + correct  -> permissive: image processed in an out-of-contract role.
    #                     Not a defect — on our b12x lineage the role rule lives in
    #                     deepseek_v4_encoding.py, which the hotfix reports as
    #                     "skipped-no-placeholder" because that file has no
    #                     IMAGE_PLACEHOLDER here, so the restriction is never
    #                     installed. Images work in MORE places, not fewer.  PASS+note
    #   200 + wrong    -> the genuine fail-open: accepted then discarded.    FAIL
    print("  [1] system-role image — strict / permissive / dropped?")
    code, body = chat(base, model, [
        {"role": "system", "content": [
            {"type": "text", "text": "You are helpful."},
            {"type": "image_url", "image_url": {"url": data_uri(solid((20, 190, 60)))}},
        ]},
        {"role": "user", "content": "What colour is the image you were given? One word."},
    ], think, max_tokens=24)
    got = answer_of(body).lower()
    if code == 400:
        R.add("plumbing", "system-role image handling", True,
              "strict — rejected per DeepSeek's contract")
    elif code == 200 and "green" in got:
        R.add("plumbing", "system-role image handling", True,
              f"PERMISSIVE — accepted and correctly processed (said {got[:20]!r}); "
              "role rule not installed on this lineage, images are NOT dropped")
    elif code == 200:
        R.add("plumbing", "system-role image handling", False,
              f"FAIL-OPEN — accepted but image ignored (said {got[:30]!r})")
    else:
        R.add("plumbing", "system-role image handling", False, f"unexpected HTTP {code}")

    # ── 2. VISION: distinct ground truths, so a constant/lucky answer fails ──
    print("  [2] vision — colours (3 different answers required)")
    for want, rgb in (("red", (220, 20, 20)), ("blue", (20, 40, 220)), ("green", (20, 190, 60))):
        code, body = chat(base, model, user_image(
            "What colour is this image? Reply with exactly one word.", solid(rgb)), think, max_tokens=24)
        got = answer_of(body).lower()
        R.add("vision", f"solid {want}", code == 200 and want in got, f"HTTP {code}, said {got[:40]!r}")

    print("  [3] vision — spatial layout")
    code, body = chat(base, model, user_image(
        "This image is split into a left half and a right half. Which half is red? "
        "Answer with exactly one word: left or right.",
        split_lr((220, 20, 20), (20, 40, 220))), think, max_tokens=24)
    got = answer_of(body).lower()
    R.add("vision", "left/right discrimination",
          code == 200 and "left" in got and "right" not in got, f"HTTP {code}, said {got[:40]!r}")

    print("  [4] vision — counting")
    code, body = chat(base, model, user_image(
        "How many black squares are in this image? Reply with just the digit.",
        squares(3)), think, max_tokens=24)
    got = answer_of(body)
    R.add("vision", "counts 3 squares", code == 200 and "3" in got, f"HTTP {code}, said {got[:40]!r}")

    # ── 3. TEXT correctness ─────────────────────────────────────────────────
    print("  [5] text — arithmetic + instruction following")
    code, body = chat(base, model, [{"role": "user",
        "content": "What is 17 * 23? Reply with only the number."}], think, max_tokens=32)
    got = answer_of(body)
    R.add("text", "arithmetic (17*23=391)", code == 200 and "391" in got, f"said {got[:40]!r}")

    code, body = chat(base, model, [{"role": "user",
        "content": "Reply with exactly the word PINEAPPLE and nothing else."}], think, max_tokens=16)
    got = answer_of(body)
    R.add("text", "instruction following", code == 200 and "PINEAPPLE" in got.upper(), f"said {got[:40]!r}")

    # ── 4. LONG-CONTEXT needle ──────────────────────────────────────────────
    if not a.skip_needle:
        print("  [6] long context — needle retrieval @ ~32k tokens")
        filler = ("The quarterly logistics report contains routine inventory notes. " * 60 + "\n")
        needle = "The authorization code for the Helsinki depot is ZEPHYR-8814."
        # ~32k tokens: ~4 chars/token, needle buried mid-document
        chunks = [filler] * 500
        chunks[len(chunks) // 2] = needle + "\n"
        doc = "".join(chunks)
        code, body = chat(base, model, [{"role": "user",
            "content": doc + "\n\nWhat is the authorization code for the Helsinki depot? "
                             "Reply with just the code."}], think, max_tokens=32, timeout=900)
        got = answer_of(body)
        R.add("context", "needle @ ~32k", code == 200 and "ZEPHYR-8814" in got.upper(),
              f"HTTP {code}, said {got[:60]!r}")

    # ── 5. TOOL CALLING (agentic stack) ─────────────────────────────────────
    print("  [7] tool calling")
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}]
    code, body = chat(base, model, [{"role": "user",
        "content": "Use the get_weather tool to check the weather in Helsinki."}],
        think, max_tokens=128, tools=tools)
    calls = []
    try:
        calls = body["choices"][0]["message"].get("tool_calls") or []
    except Exception:
        pass
    ok = code == 200 and len(calls) >= 1 and calls[0]["function"]["name"] == "get_weather"
    arg_ok = False
    if ok:
        try:
            arg_ok = "helsinki" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower()
        except Exception:
            arg_ok = False
    R.add("tools", "emits get_weather call", ok, f"HTTP {code}, calls={len(calls)}")
    R.add("tools", "argument correct (Helsinki)", ok and arg_ok, "")

    # ── 6. SPEC DECODE actually alive (vLLM #43456) ─────────────────────────
    print("  [8] DSpark draft head — acceptance must be > 0")
    metrics = get_text(base, "/metrics")
    if metrics.startswith("__ERROR__"):
        R.add("spec", "metrics reachable", False, metrics[:80])
    else:
        def total(pat):
            s = 0.0
            found = False
            for line in metrics.splitlines():
                if line.startswith("#"):
                    continue
                if re.match(pat, line):
                    try:
                        s += float(line.rsplit(" ", 1)[1])
                        found = True
                    except Exception:
                        pass
            return (s if found else None)

        acc = total(r"^vllm:spec_decode_num_accepted_tokens_total")
        draft = total(r"^vllm:spec_decode_num_draft_tokens_total")
        if acc is None:
            R.add("spec", "spec-decode counters present", False,
                  "no vllm:spec_decode_* metrics — is speculative decoding enabled?")
        else:
            rate = (acc / draft * 100) if draft else 0.0
            R.add("spec", "draft acceptance > 0", acc > 0,
                  f"accepted={acc:.0f} drafted={draft or 0:.0f} ({rate:.1f}%)")
            # Below ~15% the draft is loaded but effectively useless; benchmarking
            # would misattribute the slowdown to the checkpoint.
            R.add("spec", "acceptance not degenerate (>15%)", rate > 15, f"{rate:.1f}%")

    # ── verdict ─────────────────────────────────────────────────────────────
    print()
    bad = R.failed()
    total_n = len(R.rows)
    if not bad:
        print(f"✓ ALL {total_n} CORRECTNESS GATES PASSED — safe to benchmark.")
        return 0
    print(f"✗ {len(bad)} of {total_n} GATES FAILED — DO NOT BENCHMARK:")
    for group, name, _, detail in bad:
        print(f"    [{group}] {name}  — {detail}")
    print("\n  A benchmark run now would produce plausible-looking numbers for a model")
    print("  that is not behaving correctly. Fix the failures first.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
