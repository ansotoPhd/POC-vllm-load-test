#!/usr/bin/env python3
"""Mixed workload: one giant prefill + a stream of small requests.

This is where max-num-batched-tokens matters most. While a 200k prompt is
prefilling, we fire small requests at a steady rate and measure THEIR TTFT.
With small chunks the small requests slip between prefill chunks (low TTFT);
with a huge batch budget the big prefill monopolises each engine step and the
small requests stall (high TTFT = decode/prefill starvation).

Usage:
  mixed_workload.py --base-url http://host:port [--big-tokens 200000]
      [--small-tokens 256] [--small-rate 1.0] [--small-count 30]
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from load_test import (  # noqa: E402
    build_prompt, do_request, discover_model, snapshot_metrics,
    system_prompt_for, pct,
)


async def main_async(a: argparse.Namespace) -> int:
    base = a.base_url.rstrip("/")
    metrics_url = f"{base}/metrics"
    model = a.model or await discover_model(base, None, 30)
    system = system_prompt_for("text")

    big_prompt = build_prompt(a.big_tokens, 0, shared_prefix=False, salt=f"mxb{a.salt}-")
    small_prompts = [build_prompt(a.small_tokens, i, shared_prefix=False,
                                  salt=f"mxs{a.salt}-") for i in range(a.small_count)]

    print(f"Model: {model}")
    print(f"Big prefill: ~{a.big_tokens} tok | small: {a.small_count} x ~{a.small_tokens} "
          f"tok @ {a.small_rate}/s, {a.small_max_tokens} out\n")

    limits = httpx.Limits(max_connections=a.small_count + 8,
                          max_keepalive_connections=a.small_count + 8)
    small_results: list[dict] = []
    big_done = asyncio.Event()

    async with httpx.AsyncClient(timeout=a.timeout, limits=limits) as client:
        t0 = time.perf_counter()

        async def big() -> None:
            r = await do_request(client, base, model, big_prompt, system,
                                 16, "chat", "text", 0.0, True)
            print(f"  [big] prefill done: TTFT={ (r.start - t0 + (r.ttft or 0)):.1f}s "
                  f"ok={r.ok}")
            big_done.set()

        async def small(idx: int, launch_at: float) -> None:
            await asyncio.sleep(launch_at)
            during = not big_done.is_set()
            r = await do_request(client, base, model, small_prompts[idx], system,
                                 a.small_max_tokens, "chat", "text", 0.0, True)
            small_results.append({
                "idx": idx,
                "t_sent": r.start - t0,
                "ttft": r.ttft,
                "latency": r.end - r.start,
                "during_big": during,
                "ok": r.ok,
            })

        big_task = asyncio.create_task(big())
        # Stagger the small requests across the expected big-prefill window.
        smalls = [small(i, i / a.small_rate) for i in range(a.small_count)]
        await asyncio.gather(big_task, *smalls)

    snap = await snapshot_metrics(httpx.AsyncClient(timeout=30), metrics_url)  # noqa

    during = [s for s in small_results if s["during_big"] and s["ok"] and s["ttft"]]
    after = [s for s in small_results if not s["during_big"] and s["ok"] and s["ttft"]]

    def summarize(label, rows):
        if not rows:
            print(f"  {label:<28} (no samples)")
            return
        ttfts = [s["ttft"] for s in rows]
        print(f"  {label:<28} n={len(rows):>3}  "
              f"TTFT avg={statistics.mean(ttfts)*1000:>7.0f}ms  "
              f"p50={pct(ttfts,50)*1000:>7.0f}ms  "
              f"p99={pct(ttfts,99)*1000:>7.0f}ms")

    print("\nSmall-request TTFT, split by whether the big prefill was in flight:")
    summarize("DURING big prefill", during)
    summarize("AFTER big prefill", after)
    if during and after:
        ratio = statistics.mean([s["ttft"] for s in during]) / \
            statistics.mean([s["ttft"] for s in after])
        print(f"\n  -> small requests were {ratio:.1f}x slower (TTFT) during the big prefill")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--big-tokens", type=int, default=200000)
    p.add_argument("--small-tokens", type=int, default=256)
    p.add_argument("--small-max-tokens", type=int, default=64)
    p.add_argument("--small-rate", type=float, default=1.0, help="small reqs per second")
    p.add_argument("--small-count", type=int, default=30)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--salt", default="0")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
