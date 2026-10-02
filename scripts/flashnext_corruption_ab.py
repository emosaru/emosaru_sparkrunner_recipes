#!/usr/bin/env python3
"""Deterministic reproducer / A-B harness for the Flash-Next long-context corruption.

This is the only reproducer we have for the five+ `CUDA error: an illegal memory access`
deaths. In `--mode mixed` it kills the stock config in ~20 minutes, which is what made the
root cause findable at all: the fault is in `v1/worker/gpu/boundary_checkpoint.py`, a
fork-only subsystem, and `CKPT_POLICY=aligned` disables it (96 requests clean vs stock
dying twice at ~28). See memory/flashnext_cuda_illegal_access_2026_09_25.md.

THE THREE INGREDIENTS, ALL OF WHICH WERE MISSING FROM EARLIER "CLEAN" TESTS
--------------------------------------------------------------------------
1. **Long generations.** The suspect state copy only fires when decode CROSSES a mamba
   block boundary, and the block is 2848 tokens. The first version of this harness
   generated 206-290 tokens -- ZERO crossings -- and came back clean on all 7 steps up to
   178,909 prompt tokens while proving nothing. `boundary_crossings` is now reported per
   request and the run warns if the total is zero. Never trust a clean run with 0.
2. **Mixed-length concurrency.** Sequential long requests stayed clean even WITH
   crossings. Deaths need 4-7 concurrent requests of mixed length -- the shape of real
   agent traffic, and of vllm#55894's and our own crash #1's load.
3. **Cache-hit resume that EXTENDS a cached prefix.** Per vllm#54173, identical repeated
   prompts do not reproduce. Each step here takes a longer prefix of one corpus, so step
   N+1's prompt literally starts with step N's.

READING THE OUTPUT
------------------
* Corruption = a run of repeated identical tokens (the observed `4 4 4 5 5 5 ... 0 0 0`),
  which is what the verdict keys on. The `rep` column is that run length.
* `finish_reason=length` with empty `content` is NORMAL, not corruption: thinking is ON by
  default and a long task exhausts max_tokens inside the reasoning block. Treating it as a
  signal produced a false positive once.
* The thinking stream comes back as `reasoning`, NOT the documented `reasoning_content`.
  Reading the wrong key made this harness score a perfectly coherent 4096-token answer as
  CORRUPT. Both are checked now.
* `prompt` should track `target`. If it drifts far, the chars-per-token calibration was
  poisoned and the run is untrustworthy (see CPT below).

USAGE
    python3 scripts/flashnext_corruption_ab.py --label stock --mode mixed --rounds 8
    python3 scripts/flashnext_corruption_ab.py --compare stock aligned

Results land in .flashnext-logs/ab/<label>.json so two engine configs can be diffed after
the fact. Run against a QUIET server: a contending client (192.168.1.32 has historically
polled this endpoint) perturbs the cache and muddies attribution. And ALWAYS stop the
stack before relaunching for a new arm -- containers are reused otherwise, and a mod that
patched the writable layer survives into what you think is a clean baseline.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
OUTDIR = REPO / ".flashnext-logs" / "ab"

NEEDLE_ID = "QX-4417-QUETZAL"
NEEDLE = (
    f"OPERATIONAL NOTE {NEEDLE_ID}: the designated fallback coordinator for this "
    f"cluster is node two-three-seven, and the escalation codeword is MARMALADE."
)
NEEDLE_ANSWER_BITS = ("237", "two-three-seven", "MARMALADE", "marmalade")

# The suspect kernel is the align-mode mamba STATE COPY, which only fires when a request
# crosses a mamba block boundary (`needs_copy = aligned_new_computed >= ...` in
# mamba_utils). The mamba block is 2848 tokens on this model, so a 250-token generation
# crosses ZERO boundaries and cannot exercise the copy at all -- which is exactly why the
# first version of this harness came back clean on all 7 steps. Generate long enough to
# cross several, and aim prompts just under a boundary so the first one is crossed early.
MAMBA_BLOCK = 2848
MAX_TOKENS = 4096
COUNT_TO = 1500  # ~1500-2500 output tokens, enough for 1-2 boundary crossings


def build_corpus(min_chars: int) -> str:
    """Deterministic, varied, non-repetitive filler from real source text.

    Repetitive filler is the thing upstream says does not reproduce, and it also gives
    unrealistically good prefix-cache behaviour. Real source has the token diversity of
    an actual agent context.
    """
    chunks: list[str] = []
    seen = 0
    # Host paths (this harness runs on the host, not in the container). The eugr tree is
    # ~1.9M chars of varied Python/Markdown, enough for the longest step here.
    roots = [
        pathlib.Path.home() / "code" / "spark-vllm-docker-eugr",
        REPO,
    ]
    files: list[pathlib.Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(p for p in root.rglob("*.py") if p.is_file()))
            files.extend(sorted(p for p in root.rglob("*.md") if p.is_file()))
    if not files:
        # Fall back to generated prose rather than silently producing a tiny prompt.
        rng = random.Random(1717)
        words = [f"token{rng.randrange(100000)}" for _ in range(20000)]
        while seen < min_chars:
            chunk = " ".join(rng.sample(words, 400))
            chunks.append(chunk)
            seen += len(chunk)
        return "\n".join(chunks)

    rng = random.Random(1717)
    rng.shuffle(files)
    for p in files:
        if seen >= min_chars:
            break
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if len(text) < 200:
            continue
        chunks.append(f"\n\n===== FILE: {p} =====\n{text}")
        seen += len(text)
    return "".join(chunks)


# One corpus, built once. Steps take successively longer PREFIXES of it, so step N+1's
# prompt literally starts with step N's prompt -> a cache-hit resume that extends.
_CORPUS: str | None = None


def corpus(min_chars: int) -> str:
    global _CORPUS
    if _CORPUS is None or len(_CORPUS) < min_chars:
        _CORPUS = build_corpus(max(min_chars, 900_000))
    return _CORPUS


def make_prompt(target_chars: int) -> str:
    body = corpus(target_chars + 10_000)[:target_chars]
    return (
        "You are reviewing a large codebase dump. Read it and answer the two questions "
        "at the end precisely and briefly.\n\n"
        f"{NEEDLE}\n\n"
        f"{body}\n\n"
        "--- END OF DUMP ---\n"
        "Question 1: Restate operational note "
        f"{NEEDLE_ID} exactly: which node is the fallback coordinator, and what is the "
        "escalation codeword?\n"
        "Question 2: Compute 17 * 23 + 4 and give just the number.\n"
        f"Question 3: Then output every integer from 1 to {COUNT_TO} in order, separated "
        "by single spaces, with nothing after the final number.\n"
        "Answer Q1 and Q2 in one short line each, then Q3."
    )


def counting_report(text: str) -> dict:
    """How far the model counted correctly: Q3 asks for the integers 1..COUNT_TO.

    The count's main job is to force a long generation so decode crosses mamba block
    boundaries, which is what the suspect align state copy keys on.

    The model interleaves its own headers ("Chunk 7 601-700:"), which inject stray
    numbers, so requiring the run to start at the first literal 1 is too brittle. Take
    the LONGEST run of adjacent integers ascending by exactly 1 instead: headers merely
    break it, and the real count body is by far the longest run. A mid-count state
    corruption shows up as a longest run far shorter than the tokens generated would
    allow, which is a diagnostic to read alongside the repeated-run detector.
    """
    nums = [int(m) for m in re.findall(r"\b\d{1,5}\b", text)]
    best = cur = 0
    best_end = 0
    for i, v in enumerate(nums):
        if i and v == nums[i - 1] + 1:
            cur += 1
        else:
            cur = 1
        if cur > best:
            best, best_end = cur, v
    return {
        "longest_ascending": best,
        "reached": best_end,
        "complete": best_end >= COUNT_TO,
    }


def degeneration_report(text: str) -> dict:
    """Detect the constant-token loop signature without relying on an exact shape.

    The observed failure was a run of repeating short tokens ('4 4 4 ... 0 0 0'), so the
    tests are: longest run of one repeated whitespace-separated token, and the fraction
    of the output made up of the single most common token.
    """
    toks = text.split()
    if not toks:
        return {"longest_run": 0, "top_token_share": 0.0, "n_tokens": 0}
    longest = cur = 1
    for a, b in zip(toks, toks[1:]):
        cur = cur + 1 if a == b else 1
        longest = max(longest, cur)
    counts: dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    return {
        "longest_run": longest,
        "top_token_share": round(max(counts.values()) / len(toks), 3),
        "n_tokens": len(toks),
    }


def post(url: str, payload: dict, timeout: int) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:2000]
    except Exception as e:  # noqa: BLE001 - want the reason text, whatever it is
        return 0, f"{type(e).__name__}: {e}"


import threading

# chars-per-token, recalibrated from real responses.
# MUST only be calibrated from LONG requests and under a lock: in mixed mode the
# concurrent ~900-token requests have a very different chars/token ratio, and letting
# them write this global poisoned it badly enough that a later round asked for 150,744
# tokens and sent 34,414. Any run where the `prompt` column drifts far from `target` is
# untrustworthy -- that column is the check.
CPT = [3.6]
_CPT_LOCK = threading.Lock()
CPT_MIN_TARGET = 10_000  # below this, a request's ratio is not representative


def one_request(url: str, model: str, target: int, timeout: int) -> dict:
    with _CPT_LOCK:
        cpt = CPT[0]
    prompt = make_prompt(int(target * cpt))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,  # determinism: a repeat must be byte-identical
    }
    t0 = time.time()
    status, body = post(url, payload, timeout)
    dt = time.time() - t0

    if status != 200 or not isinstance(body, dict):
        return {"target": target, "http_status": status, "error": str(body)[:4000],
                "seconds": round(dt, 1), "verdict": f"HTTP{status}"}

    usage = body.get("usage") or {}
    ptok = usage.get("prompt_tokens") or 0
    otok = usage.get("completion_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    # NOTE: this build returns the thinking stream as `reasoning`, NOT the
    # `reasoning_content` the OpenAI-compatible docs suggest, and thinking is ON by
    # default (usage.completion_tokens_details.reasoning_tokens confirms it). Reading the
    # wrong key made an earlier version of this harness see empty strings and score a
    # perfectly coherent response as CORRUPT. Check both.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    finish = choice.get("finish_reason")
    if ptok and target >= CPT_MIN_TARGET:
        with _CPT_LOCK:
            CPT[0] = (target * cpt) / ptok

    rep_c = degeneration_report(content)
    rep_r = degeneration_report(reasoning)
    worst_run = max(rep_c["longest_run"], rep_r["longest_run"])
    worst_share = max(rep_c["top_token_share"], rep_r["top_token_share"])
    # The count is emitted inside the thinking stream, so score both.
    count = counting_report(content + "\n" + reasoning)

    both = content + "\n" + reasoning
    needle_ok = any(b in both for b in NEEDLE_ANSWER_BITS)
    math_ok = "395" in both
    # `finish_reason=length` with empty content is NOT corruption by itself: with thinking
    # on, a long task legitimately exhausts max_tokens inside the reasoning block and
    # never reaches the content. Only a DIVERGENCE is corruption -- a number emitted out
    # of order means the tokens after it came from state that no longer matches what the
    # model was emitting. Truncation mid-count is fine as long as the run is ordered.
    truncated = finish == "length"
    # The signature that actually characterises this bug is a run of repeated identical
    # tokens ("4 4 4 4 5 5 5 ... 0 0 0"), so that is what decides the verdict. The count
    # is a long-generation driver and a diagnostic, not a pass/fail gate: the model's own
    # chunk headers legitimately break strict sequence checks.
    degenerate = worst_run >= 8 or worst_share >= 0.30

    if degenerate:
        verdict = "CORRUPT"
    elif truncated:
        verdict = f"trunc@{count['reached']}"
    elif not needle_ok or not math_ok:
        verdict = f"wrong(n={needle_ok},m={math_ok})"
    else:
        verdict = "ok"

    # Boundary crossings this request's DECODE phase went through -- the thing the align
    # state copy keys on. Zero means this request could not have exercised it.
    crossings = (ptok + otok) // MAMBA_BLOCK - ptok // MAMBA_BLOCK

    return {"target": target, "prompt_tokens": ptok,
            "length_drift": round(abs(ptok - target) / max(target, 1), 3),
            "cached_tokens": cached,
            "completion_tokens": otok, "finish_reason": finish,
            "boundary_crossings": crossings,
            "needle_ok": needle_ok, "math_ok": math_ok, "truncated": truncated,
            "count": count, "content_report": rep_c, "reasoning_report": rep_r,
            "verdict": verdict, "seconds": round(dt, 1),
            "worst_run": worst_run, "worst_share": worst_share,
            "content_head": content[:400],
            "content_tail": content[-400:],
            "reasoning_tail": reasoning[-400:] if reasoning else ""}


def fmt(r: dict) -> str:
    if "http_status" in r:
        return (f"{r['target']:>10} {'-':>10} {'-':>8} {'-':>8} {'-':>4} {'-':>6} "
                f" *** HTTP {r['http_status']}: {str(r.get('error'))[:120]}")
    c = r["count"]
    return (f"{r['target']:>10} {r['prompt_tokens']:>10} {str(r['cached_tokens']):>8} "
            f"{r['completion_tokens']:>8} {r['boundary_crossings']:>4} "
            f"{c['longest_ascending']:>6} {r['worst_run']:>4} "
            f"{str(r['finish_reason']):>8}  {r['verdict']}")


HEADER = (f"{'target':>10} {'prompt':>10} {'cached':>8} {'out':>8} {'xing':>4} "
          f"{'ascend':>6} {'rep':>4} {'finish':>8}  verdict")


def boundary_targets(steps: list[int]) -> list[int]:
    """Nudge each target just under a mamba block boundary.

    Puts the request ~200 tokens short of a multiple of 2848 so the first boundary is
    crossed early in decode, and the long count then crosses more.
    """
    out = []
    for s in steps:
        k = max(1, round(s / MAMBA_BLOCK))
        out.append(k * MAMBA_BLOCK - 200)
    return out


def run_session(args) -> int:
    """Grow ONE conversation turn by turn, the way a real agent session does.

    This exists because the single-stream `boundary` sweep came back 11/11 clean at the
    exact lengths that kill a resumed agent session (199,171 and 204,369 prompt tokens),
    while the user's session died in 10 requests at 199,360. The difference is not length
    and not concurrency: in the sweep, position N arrives via one big PREFILL; in a real
    session, position N was built incrementally, with the assistant's own DECODE-produced
    tokens folded into the next prompt and a cache-hit resume on every turn. For a
    recurrent model those are not equivalent -- the retained state at a block boundary
    differs depending on whether that boundary was crossed during prefill or during
    decode.

    So: seed with a large document, then take many short turns on top, appending each real
    response. Every turn is a genuine cache-hit resume over a context that grew by decode.
    """
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    seed_chars = int(args.seed_tokens * CPT[0])
    msgs = [{"role": "user", "content": make_prompt(seed_chars)}]
    print(f"label={args.label}  mode=session  seed~{args.seed_tokens} tok  "
          f"turns={args.turns}  (mamba block={MAMBA_BLOCK})")
    print(f"{'turn':>5} {'prompt':>9} {'cached':>9} {'out':>6} {'xing':>4} {'blk_rem':>8} "
          f"{'rep':>4} {'finish':>8}  verdict")
    results: list[dict] = []
    for turn in range(args.turns):
        payload = {"model": args.model, "messages": msgs,
                   "max_tokens": args.turn_tokens, "temperature": 0}
        t0 = time.time()
        status, body = post(url, payload, args.timeout)
        dt = time.time() - t0
        if status != 200 or not isinstance(body, dict):
            print(f"{turn:>5} {'-':>9} {'-':>9} {'-':>6} {'-':>4} {'-':>8} {'-':>4} "
                  f"  *** HTTP {status}: {str(body)[:110]}")
            results.append({"turn": turn, "http_status": status,
                            "error": str(body)[:2000], "seconds": round(dt, 1)})
            break
        usage = body.get("usage") or {}
        ptok = usage.get("prompt_tokens") or 0
        otok = usage.get("completion_tokens") or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        choice = (body.get("choices") or [{}])[0]
        m = choice.get("message") or {}
        content = m.get("content") or ""
        reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
        finish = choice.get("finish_reason")
        rep = max(degeneration_report(content)["longest_run"],
                  degeneration_report(reasoning)["longest_run"])
        share = max(degeneration_report(content)["top_token_share"],
                    degeneration_report(reasoning)["top_token_share"])
        crossings = (ptok + otok) // MAMBA_BLOCK - ptok // MAMBA_BLOCK
        verdict = "CORRUPT" if (rep >= 8 or share >= 0.30) else (
            "trunc" if finish == "length" else "ok")
        print(f"{turn:>5} {ptok:>9} {str(cached):>9} {otok:>6} {crossings:>4} "
              f"{ptok % MAMBA_BLOCK:>8} {rep:>4} {str(finish):>8}  {verdict}")
        results.append({"turn": turn, "prompt_tokens": ptok, "cached_tokens": cached,
                        "completion_tokens": otok, "boundary_crossings": crossings,
                        "block_remainder": ptok % MAMBA_BLOCK, "worst_run": rep,
                        "worst_share": share, "finish_reason": finish,
                        "verdict": verdict, "seconds": round(dt, 1),
                        "content_head": content[:300]})
        # Append the REAL response, then a short new user turn -> the context grows the
        # way an agent session grows, and the next request resumes over it.
        msgs.append({"role": "assistant", "content": content or "(thinking only)"})
        msgs.append({"role": "user", "content":
                     f"Turn {turn + 2}: in two sentences, name one more risk in the dump "
                     f"not already mentioned, then restate note {NEEDLE_ID} verbatim."})
        if ptok > args.max_prompt:
            print(f"  reached {ptok} > --max-prompt {args.max_prompt}; stopping")
            break

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "mode": "session",
                               "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               "results": results}, indent=2))
    bad = [r for r in results if r.get("verdict") == "CORRUPT"
           or r.get("http_status") in (0, 500)]
    print(f"\nwrote {out}")
    print(f"VERDICT[{args.label}]: {len(bad)} corrupt/dead over {len(results)} turns; "
          f"final prompt {results[-1].get('prompt_tokens', 0) if results else 0} tok")
    return 1 if bad else 0


def run(args) -> int:
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    steps = [int(s) for s in args.steps.split(",")]
    if args.mode in ("boundary", "mixed"):
        steps = boundary_targets(steps)
    results: list[dict] = []

    print(f"label={args.label}  mode={args.mode}  rounds={args.rounds}")
    print(f"steps={steps}  (mamba block={MAMBA_BLOCK}, count_to={COUNT_TO})")
    print(HEADER)

    dead = False
    for rnd in range(args.rounds):
        if dead:
            break
        if args.mode == "mixed":
            # The shape every upstream report and our own crash #1 share: long prompts
            # chunk-prefilled ALONGSIDE short requests. Sequential long requests alone
            # came back clean, so concurrency is the variable being added here.
            import concurrent.futures as cf
            longs = steps
            shorts = [900] * args.shorts
            with cf.ThreadPoolExecutor(max_workers=len(longs) + len(shorts)) as ex:
                futs = {ex.submit(one_request, url, args.model, t, args.timeout): t
                        for t in longs + shorts}
                for f in cf.as_completed(futs):
                    r = f.result()
                    r["round"] = rnd
                    results.append(r)
                    print(fmt(r))
                    if r.get("http_status") in (0, 500):
                        dead = True
        else:
            for t in steps:
                r = one_request(url, args.model, t, args.timeout)
                r["round"] = rnd
                results.append(r)
                print(fmt(r))
                if r.get("http_status") in (0, 500):
                    print("    engine appears dead -- stopping")
                    dead = True
                    break

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{args.label}.json"
    out.write_text(json.dumps({
        "label": args.label, "mode": args.mode, "model": args.model, "steps": steps,
        "rounds": args.rounds, "mamba_block": MAMBA_BLOCK,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": results,
    }, indent=2))
    print(f"\nwrote {out}")

    bad = [r for r in results if r.get("verdict", "").startswith("CORRUPT")
           or r.get("http_status") in (0, 500)]
    xings = sum(r.get("boundary_crossings", 0) for r in results)
    print(f"VERDICT[{args.label}]: {len(bad)} corrupt/dead of {len(results)} requests; "
          f"{xings} total mamba block-boundary crossings exercised")
    if xings == 0:
        print("  WARNING: zero boundary crossings -- this run could NOT exercise the "
              "align state copy, so a clean result proves nothing.")
    return 1 if bad else 0


def compare(a: str, b: str) -> int:
    pa = OUTDIR / f"{a}.json"
    pb = OUTDIR / f"{b}.json"
    for p in (pa, pb):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 2
    da, db = json.loads(pa.read_text()), json.loads(pb.read_text())
    ra = {r["target"]: r for r in da["results"]}
    rb = {r["target"]: r for r in db["results"]}
    print(f"{'target_tok':>10} {a[:18]:>20} {b[:18]:>20}")
    for t in sorted(set(ra) | set(rb)):
        va = ra.get(t, {}).get("verdict") or f"HTTP {ra.get(t, {}).get('http_status')}"
        vb = rb.get(t, {}).get("verdict") or f"HTTP {rb.get(t, {}).get('http_status')}"
        flag = "  <-- CHANGED" if va != vb else ""
        print(f"{t:>10} {va:>20} {vb:>20}{flag}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--label", default="run")
    # Walks from well below the reported threshold up past where all five deaths sat.
    # Each step extends the previous prompt, so each is a cache-hit resume.
    ap.add_argument("--steps", default="40000,100000,131072,140000,150000,165000,180000")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--mode", choices=("sweep", "boundary", "mixed", "session"),
                    default="mixed",
                    help="sweep: sequential, targets as given (came back clean -- kept "
                         "for reference). boundary: sequential, targets nudged just "
                         "under a mamba block boundary. mixed: boundary targets run "
                         "CONCURRENTLY with short requests, which is the shape our "
                         "crash #1 and upstream #55894/#56736 all describe.")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--shorts", type=int, default=4,
                    help="concurrent short (~900 token) requests in mixed mode")
    # session mode: simulate a real agent conversation growing by decode
    ap.add_argument("--seed-tokens", type=int, default=150000,
                    help="session mode: size of the opening document turn")
    ap.add_argument("--turns", type=int, default=60)
    ap.add_argument("--turn-tokens", type=int, default=1200,
                    help="max_tokens per session turn")
    ap.add_argument("--max-prompt", type=int, default=215000,
                    help="session mode: stop once the prompt exceeds this")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.mode == "session":
        return run_session(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
