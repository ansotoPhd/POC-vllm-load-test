#!/usr/bin/env python3
"""Aggregate bench_results/*.json into comparison tables."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).parent / "bench_results")


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else []


def g(row, *path, default=None):
    cur = row
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def num(x, fmt="{:.0f}"):
    return "n/a" if x is None else fmt.format(x)


def sweep_table(title, names, cols):
    print(f"\n### {title}")
    header = "  " + "".join(f"{c[0]:>{c[2]}}" for c in cols)
    print(header)
    for fname in names:
        for row in load(fname):
            vals = []
            for _, fn, w in cols:
                vals.append(f"{fn(row):>{w}}")
            print("  " + "".join(vals))


COLS = [
    ("level", lambda r: g(r, "label").replace("CONCURRENCY = ", "c=")
        .replace("SEQUENTIAL (concurrency=1)", "seq"), 8),
    ("req/s", lambda r: num(g(r, "req_per_s"), "{:.2f}"), 9),
    ("tok/s", lambda r: num(g(r, "out_tok_per_s"), "{:.0f}"), 8),
    ("lat p50", lambda r: num(g(r, "latency_p50_ms"), "{:.0f}") + "ms", 10),
    ("lat p99", lambda r: num(g(r, "latency_p99_ms"), "{:.0f}") + "ms", 10),
    ("ttft p90", lambda r: num(g(r, "ttft_p90_ms"), "{:.0f}") + "ms", 10),
    ("run pk", lambda r: num(g(r, "server", "running_peak")), 7),
    ("wait pk", lambda r: num(g(r, "server", "waiting_peak")), 8),
    ("kv%", lambda r: num((g(r, "server", "kv_peak") or 0) * 100, "{:.1f}"), 7),
    ("preempt", lambda r: num(g(r, "server", "preemptions")), 8),
    ("fail", lambda r: str(g(r, "failed", default=0)), 6),
]

sweep_table("1) STREAMING sweep (cache off, 512-tok prompt, 128 out)",
            ["01_seq_stream", "02_sweep_stream"], COLS)
sweep_table("2) NO-STREAM sweep (cache off, 512-tok prompt, 128 out)",
            ["03_sweep_nostream"], COLS)

# Big-prompt cache comparison
print("\n### 3) LARGE PROMPT 8k, concurrency 8 -- prefix cache effect")
bcols = [
    ("run", lambda n: n, 18),
    ("req/s", lambda r: num(g(r, "req_per_s"), "{:.2f}"), 9),
    ("tok/s", lambda r: num(g(r, "out_tok_per_s"), "{:.0f}"), 8),
    ("ttft p90", lambda r: num(g(r, "ttft_p90_ms"), "{:.0f}") + "ms", 11),
    ("hit%", lambda r: num((g(r, "server", "prefix_hit_rate") or 0) * 100, "{:.1f}"), 8),
    ("KVcomp/req", lambda r: num(g(r, "server", "prefill_kv_computed_avg")), 11),
    ("prefill", lambda r: num((g(r, "server", "prefill_avg_s") or 0) * 1000) + "ms", 9),
    ("decode", lambda r: num((g(r, "server", "decode_avg_s") or 0) * 1000) + "ms", 9),
]
print("  " + "".join(f"{c[0]:>{c[1+1]}}" if False else f"{c[0]:>{c[2]}}" for c in bcols))
for label, fname in [("cold", "04_big_cold"), ("warm(+warmup)", "05_big_warm"),
                     ("cache-on no-warm", "06_big_cacheon_nowarm")]:
    rows = load(fname)
    if not rows:
        continue
    r = rows[0]
    line = [f"{label:>18}"]
    for _, fn, w in bcols[1:]:
        line.append(f"{fn(r):>{w}}")
    print("  " + "".join(line))

# Completion types
print("\n### 4) COMPLETION TYPES (concurrency 8, 512-tok prompt, streaming)")
ccols = [
    ("type", None, 8),
    ("req/s", lambda r: num(g(r, "req_per_s"), "{:.2f}"), 9),
    ("tok/s", lambda r: num(g(r, "out_tok_per_s"), "{:.0f}"), 8),
    ("lat p50", lambda r: num(g(r, "latency_p50_ms"), "{:.0f}") + "ms", 10),
    ("ttft p90", lambda r: num(g(r, "ttft_p90_ms"), "{:.0f}") + "ms", 11),
    ("decode", lambda r: num((g(r, "server", "decode_avg_s") or 0) * 1000) + "ms", 9),
]
print("  " + "".join(f"{c[0]:>{c[2]}}" for c in ccols))
for ct in ["text", "json", "tool", "tools"]:
    rows = load(f"07_ctype_{ct}")
    if not rows:
        continue
    r = rows[0]
    line = [f"{ct:>8}"]
    for _, fn, w in ccols[1:]:
        line.append(f"{fn(r):>{w}}")
    print("  " + "".join(line))
print()
