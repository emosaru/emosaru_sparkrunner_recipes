"""Probe whether a prompt of EXACTLY k*2848 tokens faults.

Both aligned deaths sat on an exact multiple of the 2848-token mamba block:
  death #7  185,120 = 65 x 2848
  death #8  199,360 = 70 x 2848
and the boundary sweep (which landed at arbitrary remainders) was 11/11 clean. So bisect
the character count until prompt_tokens lands exactly on the multiple, then send it.
"""
import importlib.util, json, sys
s = importlib.util.spec_from_file_location("h", "scripts/flashnext_corruption_ab.py")
h = importlib.util.module_from_spec(s); s.loader.exec_module(h)
URL = "http://127.0.0.1:8000/v1/chat/completions"

def probe(chars, max_tokens=8):
    p = h.make_prompt(chars)
    st, b = h.post(URL, {"model": "Qwen/Qwen3.8-Flash-Next",
                         "messages": [{"role": "user", "content": p}],
                         "max_tokens": max_tokens, "temperature": 0}, 1200)
    if st != 200 or not isinstance(b, dict):
        return None, (st, str(b)[:200])
    return b["usage"]["prompt_tokens"], b

for target in (int(sys.argv[1]) if len(sys.argv) > 1 else 199360,):
    lo, hi = int(target*3.0), int(target*4.4)
    got = None
    for it in range(14):
        mid = (lo+hi)//2
        pt, info = probe(mid)
        if pt is None:
            print(f"  probe died: {info}"); sys.exit(2)
        print(f"  iter{it}: chars={mid} -> prompt_tokens={pt} (target {target}, delta {pt-target})")
        if pt == target:
            got = mid; break
        if pt < target: lo = mid + 1
        else: hi = mid - 1
        if hi <= lo: break
    if got is None:
        print(f"  could not land exactly; closest search exhausted"); sys.exit(1)
    print(f"\n*** EXACT HIT: {target} tokens = {target//2848} x 2848. Now generating long. ***")
    pt, b = probe(got, max_tokens=4096)
    if pt is None:
        print(f"  *** FAULTED on the long generation: {b}"); sys.exit(3)
    ch = b["choices"][0]; m = ch["message"]
    txt = (m.get("content") or "") + "\n" + (m.get("reasoning") or "")
    rep = h.degeneration_report(txt)
    print(f"  prompt={pt} out={b['usage']['completion_tokens']} finish={ch['finish_reason']}")
    print(f"  degeneration: {rep}")
    print(f"  VERDICT: {'CORRUPT' if rep['longest_run']>=8 or rep['top_token_share']>=0.30 else 'clean'}")
