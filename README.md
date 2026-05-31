# POC-vllm-load-test

Toolkit for benchmarking and load-testing vLLM OpenAI-compatible inference servers,
with a dashboard to launch runs and visualize results.

## Goals

- Measure latency, TTFT (time to first token), inter-token latency and throughput
  under sequential and concurrent request patterns.
- Sweep concurrency levels and prompt/output token sizes.
- Explore mixed prefill/decode workloads (e.g. a huge prefill alongside small
  requests) to study chunked prefill and `max-num-batched-tokens` effects.
- Provide a frontend/dashboard to launch test runs and visualize the metrics
  recovered from vLLM and the final results.

## Status

Early scaffolding — work in progress.
