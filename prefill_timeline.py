#!/usr/bin/env python3
"""Show HOW vLLM processes concurrent large-prompt requests.

Fires N big requests at once and records, per request, when it was sent and
when its first token arrived (TTFT) -- all relative to a common t0. In
parallel it polls /metrics to capture the Running/Waiting/KV time series.
The result makes the scheduling pattern (serialized prefill vs overlapped)
visible directly instead of inferring it from aggregates.

Usage:
  prefill_timeline.py --base-url http://host:port [--num 4] \
      [--prompt-tokens 200000] [--max-tokens 16] [--interval 1.0]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from load_test import (  # noqa: E402
    build_prompt, do_request, discover_model, snapshot_metrics,
    system_prompt_for, _count_tokens,
)


async def main_async(a: argparse.Namespace) -> int:
    base = a.base_url.rstrip("/")
    metrics_url = f"{base}/metrics"
    model = a.model or await discover_model(base, None, 30)
    system = system_prompt_for("text")

    print(f"Model: {model}")
    print(f"Building {a.num} cold prompts of ~{a.prompt_tokens} tokens...")
    # Unique prefix per request => no prefix-cache short-circuit (cold prefill).
    prompts = [build_prompt(a.prompt_tokens, i, shared_prefix=False,
                            salt=f"tl{a.salt}-") for i in range(a.num)]
    print(f"  ~{_count_tokens(prompts[0])} tokens/req\n")

    limits = httpx.Limits(max_connections=a.num + 4,
                          max_keepalive_connections=a.num + 4)
    series: list[tuple[float, float, float, float]] = []
    rows: list[dict] = []
    stop = asyncio.Event()

    async with httpx.AsyncClient(timeout=a.timeout, limits=limits) as client:

        t0 = time.perf_counter()

        async def poll() -> None:
            while not stop.is_set():
                snap = await snapshot_metrics(client, metrics_url)
                if snap:
                    series.append((
                        time.perf_counter() - t0,
                        snap.get("vllm:num_requests_running", 0.0),
                        snap.get("vllm:num_requests_waiting", 0.0),
                        snap.get("vllm:kv_cache_usage_perc", 0.0) * 100,
                    ))
                try:
                    await asyncio.wait_for(stop.wait(), timeout=a.interval)
                except asyncio.TimeoutError:
                    pass

        async def fire(idx: int) -> None:
            r = await do_request(client, base, model, prompts[idx], system,
                                 a.max_tokens, "chat", "text", 0.0, True)
            rows.append({
                "idx": idx,
                "sent_s": r.start - t0,
                "ttft_s": (r.start - t0 + r.ttft) if r.ttft is not None else None,
                "end_s": r.end - t0,
                "ok": r.ok,
                "prompt_tokens": r.prompt_tokens,
                "error": r.error[:120],
            })

        poller = asyncio.create_task(poll())
        await asyncio.gather(*(fire(i) for i in range(a.num)))
        stop.set()
        await poller

    rows.sort(key=lambda x: (x["ttft_s"] is None, x["ttft_s"] or 0))

    print("Per-request timeline (seconds from t0):")
    print(f"  {'req':>4}{'sent':>8}{'first-token':>14}{'done':>9}"
          f"{'gap vs prev TTFT':>18}{'ok':>4}")
    prev = None
    for r in rows:
        ttft = r["ttft_s"]
        gap = "" if (ttft is None or prev is None) else f"+{ttft - prev:.1f}s"
        print(f"  {r['idx']:>4}{r['sent_s']:>8.1f}"
              f"{(f'{ttft:.1f}' if ttft is not None else 'n/a'):>14}"
              f"{r['end_s']:>9.1f}{gap:>18}{('Y' if r['ok'] else 'N'):>4}")
        if ttft is not None:
            prev = ttft

    print("\nServer gauge time series (Running / Waiting / KV%):")
    print(f"  {'t(s)':>7}{'run':>5}{'wait':>6}{'kv%':>7}   timeline")
    # Sub-sample to keep it readable (~25 rows max).
    step = max(1, len(series) // 25)
    for i in range(0, len(series), step):
        t, run, wait, kv = series[i]
        bar = "█" * int(run) + "·" * int(wait)
        print(f"  {t:>7.1f}{run:>5.0f}{wait:>6.0f}{kv:>7.1f}   {bar}")

    out = Path(__file__).parent / "timeline_results.json"
    out.write_text(json.dumps({"rows": rows, "series": series}, indent=2))
    print(f"\nWrote {out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--prompt-tokens", type=int, default=200000)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--salt", default="0")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
