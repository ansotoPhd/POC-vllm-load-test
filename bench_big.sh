#!/usr/bin/env bash
# Large-input battery for a deployed vLLM instance (~50k-200k token prompts).
#
# Focus: how prefill scales with input size, the KV/concurrency cliff at 200k,
# and the prefix-cache payoff for huge contexts. Each run is a SEPARATE
# invocation with its own salt so cold runs stay genuinely cold (the comma
# concurrency-sweep would reuse prompts across levels and cache them).
#
# Usage: bench_big.sh <base-url>
set -euo pipefail

BASE_URL="${1:?usage: bench_big.sh <base-url>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../../.venv/bin/python"
SCRIPT="$HERE/load_test.py"
OUT="$HERE/bench_big_results"
SALT="big-$(date +%s)-"
mkdir -p "$OUT"

run() {  # run <name> <args...>
  local name="$1"; shift
  echo
  echo "########## $name ##########"
  "$PY" "$SCRIPT" --base-url "$BASE_URL" --metrics --json-out "$OUT/$name.json" \
    --max-tokens 16 --timeout 600 "$@"
}

# 1) Prefill scaling by input size (sequential, cold, 1 req each).
for sz in 50000 100000 150000 200000; do
  run "01_size_${sz}" --sequential --num-requests 1 \
    --cache off --salt "${SALT}sz${sz}-" --prompt-tokens "$sz"
done

# 2) Concurrency at 200k, COLD -- separate salts so each level is truly cold.
run 02_c2_cold --cache off --salt "${SALT}c2-" --concurrency 2 --num-requests 4 --prompt-tokens 200000
run 03_c4_cold --cache off --salt "${SALT}c4-" --concurrency 4 --num-requests 8 --prompt-tokens 200000
# conc 8 x 200k cannot all fit in KV (~13% each) -> expect Waiting>0 / preemptions.
run 04_c8_cold --cache off --salt "${SALT}c8-" --concurrency 8 --num-requests 8 --prompt-tokens 200000

# 3) 200k WARM (shared prompt + warmup) at conc 4 -- prefix-cache payoff.
run 05_c4_warm --cache on --warmup 1 --concurrency 4 --num-requests 8 --prompt-tokens 200000

echo
echo "All large-input runs done. JSON in $OUT/"
