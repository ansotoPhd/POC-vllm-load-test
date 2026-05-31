#!/usr/bin/env python3
"""Compare two bench_big result dirs (baseline vs new config)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "bench_big_results_baseline"
NEW = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "bench_big_results"
BASE_LABEL = sys.argv[3] if len(sys.argv) > 3 else "baseline"
NEW_LABEL = sys.argv[4] if len(sys.argv) > 4 else "new"


def load(d, name):
    p = d / f"{name}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    return rows[0] if rows else None


def gv(r, *path):
    cur = r
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def fnum(x, f="{:.0f}"):
    return "n/a" if x is None else f.format(x)


def delta(b, n, lower_better=True):
    if b is None or n is None or b == 0:
        return ""
    chg = (n - b) / b * 100
    arrow = "↓" if (chg < 0) == lower_better else "↑"
    good = "✓" if (chg < 0) == lower_better else "✗"
    return f"{arrow}{abs(chg):.0f}% {good}"


def row(name, label, getter, f="{:.0f}", suffix="", lower_better=True):
    b = load(BASE, name)
    n = load(NEW, name)
    bv, nv = getter(b), getter(n)
    print(f"  {label:<26}{fnum(bv,f)+suffix:>14}{fnum(nv,f)+suffix:>14}"
          f"{delta(bv,nv,lower_better):>14}")


print(f"\n{'metric':<28}{BASE_LABEL:>14}{NEW_LABEL:>14}{'change':>14}")
print("=" * 70)

print("\n# Single-request prefill (sequential, cold)")
for sz in [50000, 100000, 150000, 200000]:
    row(f"01_size_{sz}", f"{sz//1000}k TTFT",
        lambda r: gv(r, "ttft_avg_ms"), "{:.0f}", "ms")
print("  -- server prefill_time --")
for sz in [50000, 100000, 150000, 200000]:
    row(f"01_size_{sz}", f"{sz//1000}k prefill(srv)",
        lambda r: (gv(r, "server", "prefill_avg_s") or 0) * 1000, "{:.0f}", "ms")

print("\n# 200k concurrency (cold) -- TTFT p90")
for label, fname in [("c=2", "02_c2_cold"), ("c=4", "03_c4_cold"), ("c=8", "04_c8_cold")]:
    row(fname, f"{label} TTFT p90",
        lambda r: gv(r, "ttft_p90_ms"), "{:.0f}", "ms")
print("  -- Waiting peak --")
for label, fname in [("c=2", "02_c2_cold"), ("c=4", "03_c4_cold"), ("c=8", "04_c8_cold")]:
    row(fname, f"{label} Waiting pk",
        lambda r: gv(r, "server", "waiting_peak"), "{:.0f}", "", lower_better=True)
print("  -- KV% peak --")
for label, fname in [("c=2", "02_c2_cold"), ("c=4", "03_c4_cold"), ("c=8", "04_c8_cold")]:
    row(fname, f"{label} KV% pk",
        lambda r: (gv(r, "server", "kv_peak") or 0) * 100, "{:.1f}", "%", lower_better=False)

print("\n# 200k warm (conc 4)")
row("05_c4_warm", "warm TTFT p90", lambda r: gv(r, "ttft_p90_ms"), "{:.0f}", "ms")
row("05_c4_warm", "warm req/s", lambda r: gv(r, "req_per_s"), "{:.2f}", "", lower_better=False)
print()
