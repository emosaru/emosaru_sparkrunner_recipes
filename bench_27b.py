#!/usr/bin/env python3
"""
Throughput benchmark for 27B-MTP (port 8003).
Measures: TTFT, decode tok/s, and concurrent request throughput.
"""
import json, math, sys, time, statistics, threading, argparse
import httpx


def safe_stdev(vals):
    finite = [v for v in vals if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    mean = sum(finite) / len(finite)
    return (sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)) ** 0.5


def safe_mean(vals):
    finite = [v for v in vals if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")

BASE_URL = "http://192.168.1.236:8003"
MODEL    = "Qwen/Qwen3.6-27B-FP8"

# Set by main() from CLI args; functions read these globals.
_base_url = BASE_URL
_model    = MODEL

PROMPTS = {
    "short":  "Explain the difference between TCP and UDP in one paragraph.",
    "medium": (
        "You are a senior distributed systems engineer. "
        "Describe in detail the trade-offs between consistency, availability, and "
        "partition tolerance in distributed databases (CAP theorem), with concrete "
        "examples of how real systems like Cassandra, DynamoDB, and PostgreSQL position "
        "themselves along those axes. Include discussion of PACELC as an extension."
    ),
    "long": (
        "Write a comprehensive 1000-word technical blog post aimed at ML engineers "
        "explaining how speculative decoding works, why it improves throughput without "
        "sacrificing quality, what the key hyperparameters are (draft length, acceptance "
        "threshold), how Multi-Token Prediction (MTP) differs from classic draft-model "
        "speculative decoding, and what workloads benefit most. Include a worked numeric "
        "example showing how a 4-draft-token speculation with 75% acceptance rate "
        "translates to a concrete tok/s improvement."
    ),
}

MAX_TOKENS = 512


def stream_generate(prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
    """Stream one completion; return timing stats."""
    payload = {
        "model": _model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}

    t_start = time.perf_counter()
    t_first = None
    out_tokens = 0
    in_tokens = 0

    with httpx.Client(timeout=300) as client:
        with client.stream("POST", f"{_base_url}/v1/chat/completions",
                           json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                # usage is usually in the last chunk
                usage = chunk.get("usage") or {}
                if usage.get("prompt_tokens"):
                    in_tokens = usage["prompt_tokens"]
                if usage.get("completion_tokens"):
                    out_tokens = usage["completion_tokens"]

                delta_obj = chunk["choices"][0]["delta"]
                # Qwen3 reasoning parser emits reasoning_content before content;
                # capture TTFT on whichever arrives first.
                delta = delta_obj.get("content", "") or delta_obj.get("reasoning", "")
                if delta and t_first is None:
                    t_first = time.perf_counter()

    t_end = time.perf_counter()

    ttft        = (t_first - t_start) if t_first else float("nan")
    total_time  = t_end - t_start
    decode_time = (t_end - t_first) if t_first else float("nan")
    # fall back to a token count estimate if usage wasn't returned
    if out_tokens == 0:
        out_tokens = max_tokens  # rough upper bound
    decode_tps  = out_tokens / decode_time if decode_time > 0 else float("nan")

    return {
        "ttft_s":       ttft,
        "total_s":      total_time,
        "decode_s":     decode_time,
        "out_tokens":   out_tokens,
        "in_tokens":    in_tokens,
        "decode_tps":   decode_tps,
    }


def run_single_stream_suite(reps: int = 3):
    print("\n=== Single-stream latency & decode tok/s ===")
    print(f"{'Scenario':<10} {'reps':<5} {'TTFT (s)':<12} {'decode tok/s':<15} {'out_tok':<10}")
    print("-" * 55)

    for name, prompt in PROMPTS.items():
        ttfts, tpss, toks = [], [], []
        for _ in range(reps):
            r = stream_generate(prompt)
            ttfts.append(r["ttft_s"])
            tpss.append(r["decode_tps"])
            toks.append(r["out_tokens"])
        print(
            f"{name:<10} {reps:<5} "
            f"{safe_mean(ttfts):.3f}±{safe_stdev(ttfts):.3f}   "
            f"{safe_mean(tpss):.1f}±{safe_stdev(tpss):.1f}       "
            f"{safe_mean(toks):.0f}"
        )


def run_concurrent_suite(concurrency_levels=(1, 2, 4, 8)):
    print("\n=== Concurrent throughput (medium prompt, 512 out tok) ===")
    print(f"{'concurrency':<14} {'total tok/s':<14} {'median TTFT':<14} {'p95 TTFT'}")
    print("-" * 56)

    prompt = PROMPTS["medium"]

    for n in concurrency_levels:
        results = [None] * n
        errors  = []

        def worker(idx):
            try:
                results[idx] = stream_generate(prompt)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        wall = time.perf_counter() - t0

        if errors:
            print(f"{n:<14} ERROR: {errors[0]}")
            continue

        all_tokens = sum(r["out_tokens"] for r in results if r)
        total_tps  = all_tokens / wall
        ttfts      = sorted(r["ttft_s"] for r in results if r)
        p50 = statistics.median(ttfts)
        p95 = ttfts[int(len(ttfts) * 0.95)] if len(ttfts) >= 2 else ttfts[-1]

        print(f"{n:<14} {total_tps:<14.1f} {p50:<14.3f} {p95:.3f}")


def check_alive():
    try:
        r = httpx.get(f"{_base_url}/v1/models", timeout=10)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        print(f"Endpoint up. Models: {models}")
        return True
    except Exception as e:
        print(f"ERROR: can't reach {BASE_URL}: {e}", file=sys.stderr)
        return False


def main():
    global _base_url, _model
    ap = argparse.ArgumentParser(description="27B throughput benchmark")
    ap.add_argument("--base-url",    default=BASE_URL, metavar="URL",
                    help=f"vLLM base URL (default: {BASE_URL})")
    ap.add_argument("--model",       default=MODEL, metavar="MODEL_ID",
                    help=f"model ID sent in requests (default: {MODEL})")
    ap.add_argument("--reps",        type=int, default=3,
                    help="repetitions per single-stream scenario (default 3)")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="concurrency levels to test (default: 1 2 4 8)")
    ap.add_argument("--skip-single", action="store_true",
                    help="skip single-stream suite")
    ap.add_argument("--skip-conc",   action="store_true",
                    help="skip concurrent suite")
    args = ap.parse_args()
    _base_url = args.base_url
    _model    = args.model

    print(f"Target: {_base_url}  model: {_model}")
    if not check_alive():
        sys.exit(1)

    if not args.skip_single:
        run_single_stream_suite(reps=args.reps)
    if not args.skip_conc:
        run_concurrent_suite(concurrency_levels=args.concurrency)

    print("\nDone.")


if __name__ == "__main__":
    main()
