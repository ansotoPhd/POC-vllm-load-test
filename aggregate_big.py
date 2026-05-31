#!/usr/bin/env python3
"""Aggregate bench_big_results/*.json into large-input tables."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).parent / "bench_big_results")


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else []


def gv(r, *path, default=None):
    cur = r
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def ms(x):
    return "n/a" if x is None else f"{x * 1000:.0f}ms"


def num(x, f="{:.0f}"):
    return "n/a" if x is None else f.format(x)


# 1) Prefill scaling by input size
print("### 1) PREFILL SCALING by input size (sequential, cold, 1 req)")
print(f"  {'size':>8}{'prompt tok':>12}{'TTFT(cli)':>11}{'prefill(srv)':>13}"
      f"{'eff tok/s':>11}{'KV% pk':>8}")
for sz in [50000, 100000, 150000, 200000]:
    rows = load(f"01_size_{sz}")
    if not rows:
        continue
    r = rows[0]
    ptok = gv(r, "server", "prompt_tokens")
    pref = gv(r, "server", "prefill_avg_s")
    eff = (ptok / pref) if (ptok and pref) else None
    print(f"  {sz:>8}{num(ptok):>12}{ms(gv(r,'ttft_avg_ms') and gv(r,'ttft_avg_ms')/1000):>11}"
          f"{ms(pref):>13}{num(eff):>11}"
          f"{num((gv(r,'server','kv_peak') or 0)*100,'{:.1f}'):>8}")

# 2) Concurrency at 200k cold
print("\n### 2) CONCURRENCY at ~200k COLD (max_tokens 16)")
print(f"  {'run':>10}{'req/s':>8}{'TTFT p90':>11}{'lat p90':>11}{'prefill(srv)':>13}"
      f"{'KV% pk':>8}{'wait pk':>9}{'preempt':>9}{'fail':>6}")
for label, fname in [("c=2", "02_c2_cold"), ("c=4", "03_c4_cold"),
                     ("c=8", "04_c8_cold"), ("c=4 warm", "05_c4_warm")]:
    rows = load(fname)
    if not rows:
        continue
    r = rows[0]
    print(f"  {label:>10}{num(gv(r,'req_per_s'),'{:.2f}'):>8}"
          f"{ms(gv(r,'ttft_p90_ms') and gv(r,'ttft_p90_ms')/1000):>11}"
          f"{ms(gv(r,'latency_p90_ms') and gv(r,'latency_p90_ms')/1000):>11}"
          f"{ms(gv(r,'server','prefill_avg_s')):>13}"
          f"{num((gv(r,'server','kv_peak') or 0)*100,'{:.1f}'):>8}"
          f"{num(gv(r,'server','waiting_peak')):>9}"
          f"{num(gv(r,'server','preemptions')):>9}"
          f"{str(gv(r,'failed',default=0)):>6}")

# 3) cold vs warm at 200k
print("\n### 3) 200k COLD vs WARM (concurrency 4)")
print(f"  {'run':>10}{'req/s':>8}{'tok/s':>8}{'TTFT p90':>11}{'hit%':>8}"
      f"{'KVcomp/req':>12}{'prefill(srv)':>13}")
for label, fname in [("cold", "03_c4_cold"), ("warm", "05_c4_warm")]:
    rows = load(fname)
    if not rows:
        continue
    r = rows[0]
    print(f"  {label:>10}{num(gv(r,'req_per_s'),'{:.2f}'):>8}"
          f"{num(gv(r,'out_tok_per_s')):>8}"
          f"{ms(gv(r,'ttft_p90_ms') and gv(r,'ttft_p90_ms')/1000):>11}"
          f"{num((gv(r,'server','prefix_hit_rate') or 0)*100,'{:.1f}'):>8}"
          f"{num(gv(r,'server','prefill_kv_computed_avg')):>12}"
          f"{ms(gv(r,'server','prefill_avg_s')):>13}")
print()
