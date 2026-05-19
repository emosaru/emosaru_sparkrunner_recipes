#!/usr/bin/env python3
"""
Throughput benchmark for the unified cluster stack.
Runs at concurrency=1 and concurrency=8, monitors RoCE bandwidth throughout.
Usage: python3 bench.py
"""

import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Cluster config
# ---------------------------------------------------------------------------
NODE1 = "192.168.1.236"

ENDPOINTS = [
    {"name": "27B-NVFP4-MTP",  "host": NODE1, "port": 8003},
    {"name": "35B-A3B-FP8-MTP", "host": NODE1, "port": 8000},
    {"name": "1.7B-FP8",        "host": NODE1, "port": 8002},
]

CONCURRENCIES = [1, 8]
REQUESTS_C1  = 20   # sequential single-stream
REQUESTS_C8  = 64   # concurrent burst

# Fixed input — ~512 tokens across all models
PROMPT = (
    "You are an expert ML engineer. "
    "Describe in detail the architecture of a transformer model, "
    "covering self-attention, multi-head attention, positional encoding, "
    "feed-forward layers, layer normalization, and residual connections. "
    "Then explain how speculative decoding with a draft model improves "
    "throughput without changing output distribution. "
) * 6

MAX_TOKENS = 256

# ---------------------------------------------------------------------------
# RoCE monitoring (sysfs counters — 4-byte words → bytes)
# ---------------------------------------------------------------------------
ROCE_NICS = ["rocep1s0f0", "roceP2p1s0f0"]
_CPATH = "/sys/class/infiniband/{nic}/ports/1/counters/{c}"


def _read_roce():
    rx = tx = 0
    for nic in ROCE_NICS:
        try:
            rx += int(Path(_CPATH.format(nic=nic, c="port_rcv_data")).read_text()) * 4
            tx += int(Path(_CPATH.format(nic=nic, c="port_xmit_data")).read_text()) * 4
        except OSError:
            pass
    return rx, tx


class RoCEMonitor:
    """Samples RoCE counters every 0.5 s; reports avg/peak Gbps on stop."""

    def __init__(self):
        self._samples: list[float] = []  # combined RX+TX Gbps per sample
        self._running = False
        self._task = None

    async def start(self):
        self._samples.clear()
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> dict:
        self._running = False
        if self._task:
            await self._task
        if not self._samples:
            return {"avg_gbps": 0.0, "peak_gbps": 0.0}
        return {
            "avg_gbps":  round(statistics.mean(self._samples), 2),
            "peak_gbps": round(max(self._samples), 2),
        }

    async def _loop(self):
        interval = 0.5
        prev_rx, prev_tx = _read_roce()
        while self._running:
            await asyncio.sleep(interval)
            rx, tx = _read_roce()
            bits = ((rx - prev_rx) + (tx - prev_tx)) * 8
            gbps = bits / interval / 1e9
            self._samples.append(gbps)
            prev_rx, prev_tx = rx, tx


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------
async def discover_model_id(client: httpx.AsyncClient, host: str, port: int) -> str:
    resp = await client.get(f"http://{host}:{port}/v1/models", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


async def wait_ready(host: str, port: int, name: str, timeout: int = 1800):
    url = f"http://{host}:{port}/v1/models"
    deadline = time.monotonic() + timeout
    print(f"  ⏳ waiting for {name} at :{port}…", flush=True)
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(url, timeout=5)
                if r.status_code == 200:
                    print(f"  ✓ {name} ready", flush=True)
                    return
            except Exception:
                pass
            await asyncio.sleep(5)
    raise TimeoutError(f"{name} not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
async def single_request(
    client: httpx.AsyncClient,
    host: str,
    port: int,
    model_id: str,
) -> tuple[float, int]:
    """Returns (latency_s, output_tokens)."""
    payload = {
        "model": model_id,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.monotonic()
    resp = await client.post(
        f"http://{host}:{port}/v1/completions",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    elapsed = time.monotonic() - t0
    out_tokens = resp.json()["usage"]["completion_tokens"]
    return elapsed, out_tokens


async def run_concurrency(
    host: str,
    port: int,
    model_id: str,
    concurrency: int,
    n_requests: int,
) -> dict:
    """Run n_requests at the given concurrency, return throughput stats."""
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    token_counts: list[int] = []

    async with httpx.AsyncClient() as client:

        async def bounded(i: int):
            async with sem:
                lat, toks = await single_request(client, host, port, model_id)
                latencies.append(lat)
                token_counts.append(toks)
                print(f"    req {i+1}/{n_requests}: {toks} tok in {lat:.1f}s "
                      f"({toks/lat:.1f} tok/s)", flush=True)

        wall_t0 = time.monotonic()
        await asyncio.gather(*[bounded(i) for i in range(n_requests)])
        wall_elapsed = time.monotonic() - wall_t0

    total_tokens = sum(token_counts)
    return {
        "concurrency":      concurrency,
        "n_requests":       n_requests,
        "wall_s":           round(wall_elapsed, 2),
        "total_out_tokens": total_tokens,
        "throughput_tps":   round(total_tokens / wall_elapsed, 1),
        "avg_lat_s":        round(statistics.mean(latencies), 2),
        "p50_lat_s":        round(statistics.median(latencies), 2),
        "p95_lat_s":        round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
    }


async def benchmark_endpoint(ep: dict) -> dict:
    name, host, port = ep["name"], ep["host"], ep["port"]
    print(f"\n{'='*60}")
    print(f"  {name}  ({host}:{port})")
    print(f"{'='*60}")

    async with httpx.AsyncClient() as client:
        model_id = await discover_model_id(client, host, port)
    print(f"  model_id: {model_id}")

    results = []
    for c in CONCURRENCIES:
        n = REQUESTS_C1 if c == 1 else REQUESTS_C8
        print(f"\n  ── concurrency={c}, {n} requests ──", flush=True)

        mon = RoCEMonitor()
        await mon.start()
        stats = await run_concurrency(host, port, model_id, c, n)
        roce = await mon.stop()

        stats["roce"] = roce
        results.append(stats)

        print(f"  throughput : {stats['throughput_tps']} tok/s")
        print(f"  avg latency: {stats['avg_lat_s']}s  p95: {stats['p95_lat_s']}s")
        print(f"  RoCE avg   : {roce['avg_gbps']} Gbps  peak: {roce['peak_gbps']} Gbps")

    return {"endpoint": name, "model_id": model_id, "runs": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    print("Waiting for all endpoints…")
    await asyncio.gather(*[
        wait_ready(ep["host"], ep["port"], ep["name"]) for ep in ENDPOINTS
    ])

    all_results = []
    # Benchmark one model at a time so RoCE samples aren't mixed
    for ep in ENDPOINTS:
        r = await benchmark_endpoint(ep)
        all_results.append(r)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    hdr = f"{'Model':<20} {'C':>3}  {'tok/s':>7}  {'avg_lat':>7}  {'RoCE avg':>9}  {'RoCE peak':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        for run in r["runs"]:
            print(
                f"{r['endpoint']:<20} {run['concurrency']:>3}  "
                f"{run['throughput_tps']:>7.1f}  "
                f"{run['avg_lat_s']:>7.2f}s  "
                f"{run['roce']['avg_gbps']:>8.2f}G  "
                f"{run['roce']['peak_gbps']:>9.2f}G"
            )

    out_path = Path(__file__).parent / "bench_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
