#!/usr/bin/env python3
"""Benchmark Flash-Next decode speed + MTP acceptance + prefix-cache reporting.

WHY THIS EXISTS: :8000 carries live traffic from the LiteLLM proxy (192.168.1.32,
~4.6 req/min). On 2026-09-11 that silently contaminated every "single-stream" number
taken by hand -- a reading of 10.9 tok/s looked like a 5x regression but was simply
3 concurrent requests, while the engine's own aggregate was a healthy 63-104 tok/s.
Metric DELTAS around a request are worse: they count other clients' drafts too.

So every sample here is gated: we poll vllm:num_requests_running throughout the request
and DISCARD the sample if anything else was in flight. A contended sample is not a
slow sample, it is a meaningless one.

Usage:
  scripts/bench-flashnext.py [--base URL] [--model ID] [--api-key KEY] [--label NAME]
                             [--reps N] [--no-gate]
--no-gate is for measuring THROUGH a proxy, where /metrics is not reachable; results
are then labelled unisolated and must not be compared against gated ones.
"""
import argparse, json, re, threading, time, urllib.request, statistics, sys

def _get(url, key=None, timeout=20):
    r = urllib.request.Request(url)
    if key: r.add_header("Authorization", f"Bearer {key}")
    return urllib.request.urlopen(r, timeout=timeout).read().decode()

def metrics(base):
    out = {}
    for l in _get(base + "/metrics").splitlines():
        if l.startswith("#"): continue
        m = re.match(r'(vllm:[a-z_0-9]+)\{([^}]*)\}\s+([0-9.e+-]+)', l)
        if m:
            k = m.group(1)
            p = re.search(r'position="(\d+)"', m.group(2))
            if p: k += f"[{p.group(1)}]"
            out[k] = out.get(k, 0) + float(m.group(3))
    return out

def chat(base, model, prompt, mx, key=None, temp=0.0):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mx, "temperature": temp}
    r = urllib.request.Request(base + "/v1/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    if key: r.add_header("Authorization", f"Bearer {key}")
    t = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=900))
    return d, time.time() - t

PROMPTS = [
    ("code",  "Write a complete Python implementation of a red-black tree with insert, delete and search. Include docstrings.", 400),
    ("prose", "Explain how TCP congestion control works, in detail.", 400),
    ("short", "Write a Python function to reverse a linked list. Code only.", 200),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.1.236:8000")
    ap.add_argument("--metrics-base", default=None, help="where to read /metrics (default: --base)")
    ap.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--label", default="direct")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    mb = a.metrics_base or a.base
    gate = not a.no_gate

    print(f"=== {a.label}: {a.base} model={a.model} gate={'on' if gate else 'OFF (unisolated)'} ===")
    rows = []
    for name, prompt, mx in PROMPTS:
        for rep in range(a.reps):
            running = []
            stop = threading.Event()
            def watch():
                while not stop.is_set():
                    try:
                        d = _get(mb + "/metrics", timeout=5)
                        m = re.search(r'vllm:num_requests_running\{[^}]*\}\s+([0-9.]+)', d)
                        if m: running.append(float(m.group(1)))
                    except Exception: pass
                    stop.wait(0.3)
            if gate:
                m0 = metrics(mb)
                threading.Thread(target=watch, daemon=True).start()
            d, el = chat(a.base, a.model, prompt, mx, a.api_key)
            if gate:
                stop.set(); time.sleep(0.5)
                m1 = metrics(mb)
            u = d["usage"]; comp = u["completion_tokens"]
            peak = max(running) if running else -1
            clean = (peak <= 1) if gate else None
            rec = {"name": name, "rep": rep, "tok": comp, "s": el, "tps": comp/el,
                   "peak": peak, "clean": clean,
                   "ptd": u.get("prompt_tokens_details")}
            if gate:
                dr = m1.get('vllm:spec_decode_num_drafts_total',0)-m0.get('vllm:spec_decode_num_drafts_total',0)
                dt = m1.get('vllm:spec_decode_num_draft_tokens_total',0)-m0.get('vllm:spec_decode_num_draft_tokens_total',0)
                ac = m1.get('vllm:spec_decode_num_accepted_tokens_total',0)-m0.get('vllm:spec_decode_num_accepted_tokens_total',0)
                rec.update(drafts=dr, dtok=dt, acc=ac,
                           accpct=(100*ac/dt if dt else 0), tps_step=(1+ac/dr if dr else 0))
            rows.append(rec)
            flag = "" if clean is None else ("" if clean else "  <-- CONTENDED, DISCARDED")
            extra = f" | acc {rec.get('accpct',0):.1f}% tok/step {rec.get('tps_step',0):.2f}" if gate else ""
            print(f"  {name:5s} r{rep}: {comp:4d} tok {el:6.1f}s {comp/el:6.1f} tok/s{extra}{flag}")

    good = [r for r in rows if r["clean"] is not False]
    print(f"\n  --- summary ({len(good)}/{len(rows)} samples usable) ---")
    if good:
        print(f"  decode tok/s: median {statistics.median(r['tps'] for r in good):.1f}  "
              f"min {min(r['tps'] for r in good):.1f}  max {max(r['tps'] for r in good):.1f}")
        if gate:
            dt = sum(r.get("dtok",0) for r in good); ac = sum(r.get("acc",0) for r in good)
            dr = sum(r.get("drafts",0) for r in good)
            if dt: print(f"  MTP: {ac:.0f}/{dt:.0f} draft tokens accepted = {100*ac/dt:.1f}%  "
                         f"| tokens/step {1+ac/dr:.2f}")
    ptd = [r["ptd"] for r in rows]
    print(f"  prompt_tokens_details seen: {ptd[-1]!r}")
    if all(p is None for p in ptd):
        print("  ^ NULL on every response: client cannot see cache hits "
              "(needs --enable-prompt-tokens-details + --enable-force-include-usage)")

if __name__ == "__main__":
    sys.exit(main())
