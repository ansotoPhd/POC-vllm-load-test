#!/usr/bin/env bash
# Performance battery for a deployed vLLM instance.
#
# Runs a matrix of configurations (streaming on/off, prefix-cache cold/warm,
# concurrency sweep, completion types) against one server, scraping vLLM
# /metrics on every run, and writes one JSON per run into bench_results/.
#
# Usage: bench.sh <base-url>
set -euo pipefail

BASE_URL="${1:?usage: bench.sh <base-url>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../../.venv/bin/python"
SCRIPT="$HERE/load_test.py"
OUT="$HERE/bench_results"
SALT="bench-$(date +%s)-"   # fresh salt => guaranteed cold start for cold runs
mkdir -p "$OUT"

run() {  # run <name> <args...>
  local name="$1"; shift
  echo
  echo "########## $name ##########"
  "$PY" "$SCRIPT" --base-url "$BASE_URL" --metrics --json-out "$OUT/$name.json" "$@"
}

COMMON="--prompt-tokens 512 --max-tokens 128"

# 1) Sequential baseline (best-case latency, no contention).
run 01_seq_stream   --sequential --num-requests 20 $COMMON --salt "${SALT}seq-"

# 2) Capacity curve, STREAMING, cache off (the core saturation sweep).
run 02_sweep_stream --concurrency 4,8,16,32,64 --num-requests 128 $COMMON --salt "${SALT}sw-"

# 3) Same curve, NO streaming (full response at once; latency/throughput only).
run 03_sweep_nostream --no-stream --concurrency 4,8,16,32,64 --num-requests 128 $COMMON --salt "${SALT}swn-"

# 4) Large prompt (8k), concurrency 8 -- prefix-cache effect:
run 04_big_cold       --cache off --salt "${SALT}bigcold-" --concurrency 8 --num-requests 24 --prompt-tokens 8000 --max-tokens 64
run 05_big_warm       --cache on --warmup 1 --concurrency 8 --num-requests 24 --prompt-tokens 8000 --max-tokens 64
run 06_big_cacheon_nowarm --cache on --concurrency 8 --num-requests 24 --prompt-tokens 8000 --max-tokens 64

# 5) Completion-type overhead at concurrency 8, streaming.
for ct in text json tool tools; do
  run "07_ctype_${ct}" --completion-type "$ct" --concurrency 8 --num-requests 48 $COMMON --salt "${SALT}ct${ct}-"
done

echo
echo "All runs done. JSON in $OUT/"
