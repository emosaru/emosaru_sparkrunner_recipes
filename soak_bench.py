#!/usr/bin/env python3
"""
soak_bench.py — Phase-2 stability soak + performance/quality harness for the
DeepSeek-V4-Flash 2x GB10 stability plan. Complements bench.py (which covers raw
concurrency throughput); this adds the three things Phase 2 needs:

  1. SOAK   — sequential stability gate. Fires N sequential long-context requests and
              detects the #40969 wedge (silent hang after ~6 requests: 100% SM, zero
              decode, no exception). Distinguishes a per-request failure from a WEDGED
              engine via a liveness probe + GPU-utilisation signature.
  2. STRESS — concurrency stability gate. Sustained concurrent bursts across a
              concurrency ramp — models the Hermes cron-job load that historically
              preceded day-scale crashes. Tracks min available unified RAM (the OOM /
              KV-exhaustion signal), error types, and a post-burst liveness probe.
  3. LATENCY— single-stream TTFT + decode tok/s (streaming), with a long-context
              "needle" correctness check (catches FP8-KV / sparse-MLA long-context
              regressions). This is the real prize-sizer for eager vs PIECEWISE vs FULL.
  4. TOOLS  — tool-call reliability %. The bake-off discriminator: gpt-oss-120b emits
              invalid JSON here (~84% success); DS-V4-Flash / Qwen3-Coder should be ~100%.

Doubles as the model bake-off: point --base-url at each endpoint and set --label.

Examples
  # Phase-1 gate after relaunch (must PASS = survives past the old ~6-req hang):
  python3 soak_bench.py --base-url http://192.168.1.236:8000 --label ds4-piecewise --mode soak
  # Full A/B for one config:
  python3 soak_bench.py --base-url http://192.168.1.236:8000 --label ds4-piecewise --mode all
  # Model bake-off (run per endpoint, then diff the JSON):
  python3 soak_bench.py --base-url http://192.168.1.236:8000 --label qwen122b --mode all
"""

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Prompt construction — long context with an embedded needle for a correctness gate
# ---------------------------------------------------------------------------
_FILLER = (
    "In distributed LLM inference on unified-memory accelerators, the serving engine must "
    "balance prefill and decode work across tensor-parallel ranks while keeping the KV cache "
    "resident. Speculative decoding drafts multiple tokens per step and verifies them in a "
    "single forward pass, raising decode throughput without altering the output distribution. "
    "Sparse attention indexers prune the key/value set each query attends to, trading a small "
    "accuracy cost for large long-context savings. "
)
_NEEDLE_CODE = "SPARK-7731-QMX"


def build_prompt(approx_tokens: int, nonce: str = "") -> tuple[str, str]:
    """Return (prompt, needle). ~1.3 tokens/word; needle buried near the middle.

    `nonce` is prepended so each measured request does a genuinely COLD prefill —
    without it, --enable-prefix-caching turns a repeated prompt into a cache hit
    (zeroing TTFT and defeating the soak's mixed-batch repro of #40969)."""
    words_needed = int(approx_tokens / 1.3)
    unit = _FILLER.split()
    body = []
    while len(body) < words_needed:
        body.extend(unit)
    mid = len(body) // 2
    body[mid:mid] = f"IMPORTANT: the deployment codeword is {_NEEDLE_CODE}. Remember it.".split()
    doc = " ".join(body[:words_needed])
    head = f"[session {nonce}] " if nonce else ""
    prompt = (
        f"{head}Read the following technical notes carefully, then answer the question at the end.\n\n"
        f"{doc}\n\n"
        "Question: First state the deployment codeword mentioned in the notes, then explain "
        "in detail how speculative decoding raises throughput without changing the output "
        "distribution. Begin your answer with the codeword."
    )
    return prompt, _NEEDLE_CODE


def _nonce(i: int) -> str:
    return f"{time.time_ns()}-{i}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_terminal_cmd",
            "description": "Run a shell command on the host and return its stdout.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The shell command"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute file path"}},
                "required": ["path"],
            },
        },
    },
]
_TOOL_REQUIRED = {"run_terminal_cmd": "command", "read_file": "path"}


# ---------------------------------------------------------------------------
# GPU-utilisation sampler (local node) — corroborates the wedge signature
# ---------------------------------------------------------------------------
class GPUSampler:
    """Samples utilization.gpu + available RAM every 1s in the background. On GB10 the KV
    cache lives in the same unified pool, so min available RAM is the OOM-proximity signal
    (nvidia-smi memory.used is N/A here)."""

    def __init__(self):
        self._util: list[int] = []
        self._avail: list[int] = []   # GB available (local node)
        self._running = False
        self._task = None

    async def start(self):
        self._util.clear()
        self._avail.clear()
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> dict:
        self._running = False
        if self._task:
            await self._task
        if not self._util:
            return {"util_avg": None, "util_max": None, "mem_avail_min": None, "n": 0}
        return {
            "util_avg": round(statistics.mean(self._util), 1),
            "util_max": max(self._util),
            "util_last5": self._util[-5:],
            "mem_avail_min": min(self._avail) if self._avail else None,
            "mem_avail_last": self._avail[-1] if self._avail else None,
            "n": len(self._util),
        }

    async def _loop(self):
        while self._running:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
                if vals:
                    self._util.append(max(vals))
            except Exception:
                pass
            m = free_mem_gb()
            if m.get("available") is not None:
                self._avail.append(m["available"])
            await asyncio.sleep(1.0)


def free_mem_gb() -> dict:
    try:
        out = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            if line.lower().startswith("mem:"):
                p = line.split()
                return {"total": int(p[1]) // 1024, "used": int(p[2]) // 1024,
                        "available": int(p[-1]) // 1024}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def discover_model(client: httpx.AsyncClient, base: str) -> str:
    r = await client.get(f"{base}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


async def wait_ready(base: str, timeout: int = 60) -> bool:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(f"{base}/v1/models", timeout=5)).status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(3)
    return False


async def stream_chat(client, base, model, prompt, max_tokens, hard_timeout, extra=None):
    """Streaming chat request. Returns dict with ttft_s, decode_tps, tokens, text, ok, error.

    Pass extra={"ignore_eos": True} for benchmarking so exactly max_tokens are generated —
    a fixed decode window gives a stable tok/s (a short natural answer does not)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra:
        payload.update(extra)
    t0 = time.monotonic()
    ttft = None
    content = []
    full = []
    n_chunks = 0
    usage_tokens = None
    try:
        async with client.stream("POST", f"{base}/v1/chat/completions",
                                 json=payload, timeout=hard_timeout) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens")
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    # DS-V4-Flash streams thinking tokens under `reasoning`; other reasoning
                    # models use `reasoning_content`; final answer is under `content`.
                    piece = (delta.get("content") or delta.get("reasoning")
                             or delta.get("reasoning_content"))
                    if piece:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        n_chunks += 1
                        full.append(piece)
                        if delta.get("content"):
                            content.append(delta["content"])
        t_end = time.monotonic()
        tokens = usage_tokens if usage_tokens is not None else n_chunks
        gen_window = max(t_end - t0 - (ttft or 0), 1e-6)
        decode_tps = (tokens - 1) / gen_window if tokens and tokens > 1 else 0.0
        return {"ok": True, "ttft_s": round(ttft or 0, 3), "decode_tps": round(decode_tps, 1),
                "tokens": tokens, "text": "".join(content), "full_text": "".join(full),
                "total_s": round(t_end - t0, 2), "error": None}
    except Exception as e:
        return {"ok": False, "ttft_s": None, "decode_tps": None, "tokens": 0,
                "text": "", "full_text": "", "total_s": round(time.monotonic() - t0, 2),
                "error": f"{type(e).__name__}: {e}"}


async def liveness_probe(client, base, model) -> bool:
    """Tiny request to tell a per-request failure apart from a WEDGED engine."""
    try:
        r = await client.post(f"{base}/v1/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5, "temperature": 0.0, "stream": False,
        }, timeout=30)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
async def mode_soak(base, model, n, prompt_tokens, max_tokens, hang_timeout) -> dict:
    print(f"\n── SOAK: {n} sequential requests, ~{prompt_tokens} tok prompt, "
          f"hang-timeout {hang_timeout}s ──", flush=True)
    gpu = GPUSampler()
    await gpu.start()
    mem_start = free_mem_gb()
    ttfts, tpss, results = [], [], []
    wedged_at = None
    async with httpx.AsyncClient() as client:
        for i in range(1, n + 1):
            # fresh nonce → cold prefill every request → real mixed prefill+decode batches;
            # ignore_eos → sustained decode work per request (heavier, stable tok/s)
            prompt, _ = build_prompt(prompt_tokens, nonce=_nonce(i))
            r = await stream_chat(client, base, model, prompt, max_tokens, hang_timeout,
                                  extra={"ignore_eos": True})
            results.append({"i": i, "ok": r["ok"], "ttft_s": r["ttft_s"],
                            "decode_tps": r["decode_tps"], "tokens": r["tokens"],
                            "total_s": r["total_s"], "error": r["error"]})
            if r["ok"]:
                ttfts.append(r["ttft_s"]); tpss.append(r["decode_tps"])
                print(f"  req {i:>2}/{n}: {r['tokens']:>4} tok  ttft {r['ttft_s']:.2f}s  "
                      f"{r['decode_tps']:.1f} tok/s", flush=True)
            else:
                print(f"  req {i:>2}/{n}: ✗ {r['error']}  (t={r['total_s']}s) — probing liveness…",
                      flush=True)
                alive = await liveness_probe(client, base, model)
                if not alive:
                    wedged_at = i
                    print(f"  ⛔ ENGINE WEDGED at request {i} (liveness probe failed). "
                          f"This is the #40969 signature.", flush=True)
                    break
                print("  ↳ engine still alive; treating as transient request failure.", flush=True)
    gpu_stats = await gpu.stop()
    mem_end = free_mem_gb()
    passed = wedged_at is None and all(x["ok"] for x in results)
    verdict = "PASS" if passed else ("WEDGED" if wedged_at else "DEGRADED")
    print(f"  → SOAK {verdict}: {sum(x['ok'] for x in results)}/{len(results)} ok"
          + (f", wedged@{wedged_at}" if wedged_at else "")
          + f"; gpu util avg {gpu_stats.get('util_avg')} max {gpu_stats.get('util_max')}", flush=True)
    return {
        "mode": "soak", "verdict": verdict, "requests": n,
        "ok_count": sum(x["ok"] for x in results), "wedged_at": wedged_at,
        "ttft_p50": round(statistics.median(ttfts), 3) if ttfts else None,
        "decode_tps_p50": round(statistics.median(tpss), 1) if tpss else None,
        "gpu": gpu_stats, "mem_start": mem_start, "mem_end": mem_end, "per_request": results,
    }


async def mode_stress(base, model, levels, duration, max_tokens, hang_timeout, prompt_sizes) -> dict:
    """Sustained concurrent bursts across a concurrency ramp — models the Hermes cron-job
    load that historically preceded crashes. Keeps C requests in flight for `duration` s per
    level, mixing prompt sizes (mixed prefill+decode batches), and watches for the OOM /
    KV-exhaustion / wedge failure mode: tracks min available unified RAM, GPU util, error
    types, and a post-burst liveness probe."""
    print(f"\n── STRESS: concurrency ramp {levels}, {duration}s/level, "
          f"prompt-size mix {prompt_sizes} tok ──", flush=True)
    levels_out = []
    async with httpx.AsyncClient() as client:
        for C in levels:
            print(f"\n  ▶ concurrency={C} sustained for {duration}s "
                  f"(recipe max_num_seqs caps in-engine batch; excess queues)…", flush=True)
            sampler = GPUSampler()
            await sampler.start()
            st = {"ok": 0, "fail": 0, "tokens": 0, "lat": [], "ttft": [], "errors": {}}
            ctr = {"i": 0}
            stop_at = time.monotonic() + duration

            async def worker():
                while time.monotonic() < stop_at:
                    ctr["i"] += 1
                    i = ctr["i"]
                    ptok = prompt_sizes[i % len(prompt_sizes)]
                    prompt, _ = build_prompt(ptok, nonce=_nonce(i))
                    r = await stream_chat(client, base, model, prompt, max_tokens,
                                          hang_timeout, extra={"ignore_eos": True})
                    if r["ok"]:
                        st["ok"] += 1
                        st["tokens"] += r["tokens"]
                        st["lat"].append(r["total_s"])
                        if r["ttft_s"] is not None:
                            st["ttft"].append(r["ttft_s"])
                    else:
                        st["fail"] += 1
                        k = (r["error"] or "unknown").split(":")[0]
                        st["errors"][k] = st["errors"].get(k, 0) + 1

            w0 = time.monotonic()
            await asyncio.gather(*[worker() for _ in range(C)])
            wall = time.monotonic() - w0
            res = await sampler.stop()
            alive = await liveness_probe(client, base, model)
            total = st["ok"] + st["fail"]
            sr = round(100 * st["ok"] / total, 1) if total else 0.0
            agg = round(st["tokens"] / wall, 1) if wall > 0 else 0.0
            wedged = not alive
            verdict = "WEDGED" if wedged else ("DEGRADED" if st["fail"] else "PASS")
            p95 = (round(sorted(st["lat"])[int(len(st["lat"]) * 0.95)], 2)
                   if len(st["lat"]) > 1 else None)
            print(f"    → {verdict}: {st['ok']} ok / {st['fail']} fail ({sr}%), "
                  f"{agg} tok/s aggregate, {total/wall:.2f} req/s", flush=True)
            print(f"      min avail RAM {res.get('mem_avail_min')} GB, "
                  f"gpu util avg {res.get('util_avg')}/max {res.get('util_max')}, "
                  f"p95 lat {p95}s", flush=True)
            if st["errors"]:
                print(f"      errors: {st['errors']}", flush=True)
            levels_out.append({
                "concurrency": C, "verdict": verdict, "ok": st["ok"], "fail": st["fail"],
                "success_pct": sr, "agg_tps": agg, "req_per_s": round(total / wall, 2),
                "p50_lat": round(statistics.median(st["lat"]), 2) if st["lat"] else None,
                "p95_lat": p95,
                "ttft_p50": round(statistics.median(st["ttft"]), 2) if st["ttft"] else None,
                "errors": st["errors"], "resources": res, "wedged": wedged,
            })
            if wedged:
                print(f"  ⛔ ENGINE WEDGED at concurrency {C} — stopping ramp.", flush=True)
                break
    overall = ("WEDGED" if any(l["wedged"] for l in levels_out)
               else "DEGRADED" if any(l["fail"] for l in levels_out) else "PASS")
    print(f"  → STRESS overall: {overall}", flush=True)
    return {"mode": "stress", "verdict": overall, "duration_s": duration,
            "prompt_sizes": prompt_sizes, "levels": levels_out}


async def mode_latency(base, model, prompt_tokens_list, max_tokens, timeout) -> dict:
    print(f"\n── LATENCY: single-stream TTFT + decode tok/s + needle check "
          f"@ contexts {prompt_tokens_list} ──", flush=True)
    rows = []
    async with httpx.AsyncClient() as client:
        # throwaway warmup to trigger compile / graph capture (NOT the target prompt,
        # so it doesn't pre-cache it and zero the measured TTFT)
        await stream_chat(client, base, model, "Warmup. Reply with OK.", 8, timeout)
        for ptok in prompt_tokens_list:
            # unique nonce → cold prefill → meaningful TTFT despite prefix caching
            prompt, needle = build_prompt(ptok, nonce=_nonce(ptok))
            # natural stop: now that every reasoning token is captured, a normal generation
            # gives a stable decode window AND lets the model finish and emit the codeword
            r = await stream_chat(client, base, model, prompt, max_tokens, timeout)
            # reasoning models may state the answer in reasoning_content → check full text
            needle_ok = needle.lower() in r["full_text"].lower() if r["ok"] else None
            rows.append({"prompt_tokens": ptok, "ok": r["ok"], "ttft_s": r["ttft_s"],
                         "decode_tps": r["decode_tps"], "tokens": r["tokens"],
                         "needle_ok": needle_ok, "error": r["error"]})
            status = "✓" if r["ok"] else "✗"
            nk = {True: "needle✓", False: "needle✗", None: "needle?"}[needle_ok]
            print(f"  ~{ptok:>6} tok ctx: {status} ttft "
                  f"{r['ttft_s']}s  {r['decode_tps']} tok/s  {nk}", flush=True)
    return {"mode": "latency", "rows": rows}


async def mode_tools(base, model, trials, timeout) -> dict:
    print(f"\n── TOOLS: {trials} tool-call reliability trials ──", flush=True)
    ask = ("List the files in /etc, then read /etc/hostname. "
           "You must use the provided tools to do this.")
    valid = 0
    detail = []
    async with httpx.AsyncClient() as client:
        for i in range(1, trials + 1):
            try:
                r = await client.post(f"{base}/v1/chat/completions", json={
                    "model": model, "messages": [{"role": "user", "content": ask}],
                    "tools": TOOLS, "tool_choice": "auto",
                    "max_tokens": 512, "temperature": 0.0, "stream": False,
                }, timeout=timeout)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                ok = False
                for c in calls:
                    fn = c.get("function", {})
                    name = fn.get("name")
                    try:
                        args = json.loads(fn.get("arguments", ""))
                    except (json.JSONDecodeError, TypeError):
                        args = None
                    if name in _TOOL_REQUIRED and isinstance(args, dict) \
                            and _TOOL_REQUIRED[name] in args:
                        ok = True
                        break
                valid += ok
                detail.append({"i": i, "ok": ok, "n_calls": len(calls)})
                print(f"  trial {i:>2}/{trials}: {'✓ valid tool_call' if ok else '✗ invalid/none'} "
                      f"({len(calls)} call(s))", flush=True)
            except Exception as e:
                detail.append({"i": i, "ok": False, "error": f"{type(e).__name__}: {e}"})
                print(f"  trial {i:>2}/{trials}: ✗ {type(e).__name__}: {e}", flush=True)
    rate = round(100 * valid / trials, 1) if trials else 0.0
    print(f"  → tool-call reliability: {valid}/{trials} = {rate}%", flush=True)
    return {"mode": "tools", "trials": trials, "valid": valid, "reliability_pct": rate,
            "detail": detail}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser(description="Phase-2 soak + benchmark + bake-off harness")
    ap.add_argument("--base-url", help="e.g. http://192.168.1.236:8000")
    ap.add_argument("--host", default="192.168.1.236")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", help="model id (auto-discovered if omitted)")
    ap.add_argument("--label", default="run", help="tag for output file / comparisons")
    ap.add_argument("--mode", default="soak",
                    choices=["soak", "stress", "latency", "tools", "all"])
    ap.add_argument("--requests", type=int, default=24, help="soak: sequential requests (>6)")
    ap.add_argument("--prompt-tokens", type=int, default=4000, help="soak: approx prompt tokens")
    ap.add_argument("--stress-levels", default="4,8,16",
                    help="stress: concurrency ramp, comma-separated (models cron-burst load)")
    ap.add_argument("--stress-duration", type=int, default=90,
                    help="stress: seconds of sustained load per concurrency level")
    ap.add_argument("--stress-sizes", default="500,4000,8000",
                    help="stress: prompt-size mix in approx tokens, comma-separated")
    ap.add_argument("--latency-contexts", default="4000,16000,64000,128000",
                    help="latency: comma-separated approx context sizes")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tool-trials", type=int, default=12)
    ap.add_argument("--hang-timeout", type=int, default=180, help="soak: per-request wedge timeout")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-dir", default="bench_results")
    args = ap.parse_args()

    base = (args.base_url or f"http://{args.host}:{args.port}").rstrip("/")
    print(f"Target: {base}  (label={args.label}, mode={args.mode})")
    if not await wait_ready(base, timeout=60):
        print(f"✗ endpoint {base} not ready after 60s"); return 1
    async with httpx.AsyncClient() as client:
        model = args.model or await discover_model(client, base)
    print(f"Model: {model}")

    contexts = [int(x) for x in args.latency_contexts.split(",") if x.strip()]
    report = {"label": args.label, "base_url": base, "model": model,
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": {}}

    if args.mode in ("soak", "all"):
        report["results"]["soak"] = await mode_soak(
            base, model, args.requests, args.prompt_tokens, args.max_tokens, args.hang_timeout)
    if args.mode in ("stress", "all"):
        report["results"]["stress"] = await mode_stress(
            base, model, [int(x) for x in args.stress_levels.split(",") if x.strip()],
            args.stress_duration, args.max_tokens, args.hang_timeout,
            [int(x) for x in args.stress_sizes.split(",") if x.strip()])
    if args.mode in ("latency", "all"):
        report["results"]["latency"] = await mode_latency(
            base, model, contexts, args.max_tokens, args.timeout)
    if args.mode in ("tools", "all"):
        report["results"]["tools"] = await mode_tools(
            base, model, args.tool_trials, args.timeout)

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.label)
    out = out_dir / f"{safe}-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n✓ results → {out}")

    # Verdict banner for the stability gates
    s = report["results"].get("soak")
    st = report["results"].get("stress")
    lines = []
    if s:
        lines.append(f"  SOAK   {s['verdict']:<8} ({s['ok_count']}/{s['requests']} ok"
                     + (f", wedged@{s['wedged_at']}" if s['wedged_at'] else "") + ")")
    if st:
        worst = "; ".join(f"c{l['concurrency']}:{l['verdict']}" for l in st["levels"])
        minmem = min((l["resources"].get("mem_avail_min")
                      for l in st["levels"]
                      if l["resources"].get("mem_avail_min") is not None), default=None)
        lines.append(f"  STRESS {st['verdict']:<8} ({worst}; min avail RAM {minmem} GB)")
    if lines:
        print(f"\n{'='*60}\n  GATE [{args.label}]\n" + "\n".join(lines) + f"\n{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
