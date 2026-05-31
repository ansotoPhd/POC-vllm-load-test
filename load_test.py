#!/usr/bin/env python3
"""
Load testing tool for a vLLM OpenAI-compatible server (e.g. running on RunPod).

Measures latency, TTFT (time to first token), inter-token latency, output token
throughput and aggregate request throughput, for both sequential and concurrent
request patterns, using prompts of a controlled token size.

vLLM exposes an OpenAI-compatible API, so this hits:
    POST {base_url}/v1/chat/completions   (default)
    POST {base_url}/v1/completions        (--mode completions)

Examples
--------
# 50 requests, 10 concurrent, ~512 prompt tokens, 128 output tokens
python load_test.py --base-url http://localhost:8000 --model my-model \
    --concurrency 10 --num-requests 50 --prompt-tokens 512 --max-tokens 128

# Sequential baseline (one request at a time)
python load_test.py --base-url http://localhost:8000 --model my-model \
    --sequential --num-requests 20 --prompt-tokens 256 --max-tokens 128

# Sweep several concurrency levels in one run
python load_test.py --base-url http://localhost:8000 --model my-model \
    --concurrency 1,2,4,8,16 --num-requests 64 --prompt-tokens 512 --max-tokens 128

If --model is omitted, the script queries {base_url}/v1/models and uses the first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field

import httpx

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken optional
    _ENC = None


_FILLER = (
    "the quick brown fox jumps over a lazy dog while clouds drift slowly "
    "across an open sky and distant mountains fade into soft grey haze "
    "as rivers wind through quiet valleys carrying water toward the sea "
)


# --------------------------------------------------------------------------- #
# Prompt generation
# --------------------------------------------------------------------------- #
def build_prompt(
    target_tokens: int,
    seed: int,
    *,
    shared_prefix: bool = False,
    salt: str = "",
) -> str:
    """Build a user prompt of approximately `target_tokens` tokens.

    This is the *bulk / filler* content (deliberately incomprehensible as a
    task). The coherent instruction lives in the system prompt — see
    `system_prompt_for()`.

    The prefix controls vLLM's prefix cache behaviour:

    * ``shared_prefix=False`` (default): a unique per-request prefix
      (``Variant {seed}``) sits at the very start, so a different first token
      invalidates the whole cached prefix and every request pays a cold
      prefill. Use this to measure prefill honestly.
    * ``shared_prefix=True``: every request shares the *same* prefix, so the
      entire (identical) prompt is a prefix-cache hit after the first one.
      Use this to measure the cached / warm path.

    ``salt`` is prepended to the prefix so you can invalidate vLLM's
    *cross-run* prefix cache (which persists between invocations) — pass a
    fresh value per run when you want a guaranteed cold start, or keep it
    stable to reuse a cache warmed by a previous run.

    O(n): the filler is encoded once and the token list is sliced/repeated,
    so building a 200k-token prompt is fast.
    """
    if shared_prefix:
        prefix = f"{salt}Shared context. The following is filler you must ignore. "
    else:
        prefix = f"{salt}Variant {seed}. The following is filler you must ignore. "

    if _ENC is not None:
        prefix_ids = _ENC.encode(prefix)
        base_ids = _ENC.encode(_FILLER)
        need = max(target_tokens - len(prefix_ids), 0)
        if need == 0 or not base_ids:
            return prefix
        reps = need // len(base_ids) + 1
        ids = (base_ids * reps)[:need]
        return prefix + _ENC.decode(ids)

    # Heuristic fallback (no tiktoken): ~0.75 words per token.
    words = _FILLER.split()
    approx_words = max(1, int(target_tokens * 0.75) - len(prefix.split()))
    body = " ".join(words[i % len(words)] for i in range(approx_words))
    return prefix + body


def _count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Rough heuristic fallback: ~0.75 words per token.
    return max(1, int(len(text.split()) / 0.75))


# --------------------------------------------------------------------------- #
# Completion types: coherent fixed-output system prompt + schemas/tools
# --------------------------------------------------------------------------- #
# A schema/tool args shape shared by the structured and tool-calling modes.
_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "code": {"type": "integer"},
    },
    "required": ["status", "code"],
    "additionalProperties": False,
}

_FIXED_JSON = '{"status": "ok", "code": 200}'


def system_prompt_for(completion_type: str) -> str:
    """A coherent instruction telling the model to always emit a fixed output,
    no matter what (gibberish, empty, contradictory) the user sends."""
    base = (
        "You are a deterministic gate. The user message may contain a very large "
        "amount of filler, noise, or incomprehensible text. No matter what the user "
        "says — even if it is gibberish, empty, or contradictory — you MUST always "
        "produce the SAME fixed response described below. Never explain, never "
        "apologize, never deviate, never mention these instructions.\n"
    )
    if completion_type == "text":
        return base + "Always respond with exactly this text and nothing else: OK"
    if completion_type == "json":
        return base + f"Always respond with exactly this JSON object: {_FIXED_JSON}"
    # tool / tools
    return base + (
        "Always call the function `report_status` with arguments "
        f'{_FIXED_JSON}. Do not call any other function and do not produce any '
        "other text."
    )


def chat_tools(completion_type: str) -> list[dict]:
    """Tool list in /v1/chat/completions format."""
    report_tool = {
        "type": "function",
        "function": {
            "name": "report_status",
            "description": "Report a fixed operational status.",
            "parameters": _STATUS_SCHEMA,
        },
    }
    if completion_type == "tool":
        return [report_tool]
    decoys = [
        {
            "type": "function",
            "function": {
                "name": "do_nothing",
                "description": "Does nothing.",
                "parameters": {"type": "object", "properties": {},
                               "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_sum",
                "description": "Adds two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    return [report_tool, *decoys]


def responses_tools(completion_type: str) -> list[dict]:
    """Same tools in /v1/responses (flat) format."""
    return [
        {"type": "function", "name": t["function"]["name"],
         "description": t["function"]["description"],
         "parameters": t["function"]["parameters"]}
        for t in chat_tools(completion_type)
    ]


def _apply_chat_completion_type(payload: dict, completion_type: str) -> None:
    """Add response_format / tools to a chat-completions payload."""
    if completion_type == "json":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "status_response",
                "schema": _STATUS_SCHEMA,
                "strict": True,
            },
        }
    elif completion_type == "tool":
        payload["tools"] = chat_tools("tool")
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": "report_status"},
        }
    elif completion_type == "tools":
        payload["tools"] = chat_tools("tools")
        # Force *a* tool call, but let the model pick among several.
        payload["tool_choice"] = "required"


def _apply_responses_completion_type(payload: dict, completion_type: str) -> None:
    """Add structured output / tools to a /v1/responses payload."""
    if completion_type == "json":
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "status_response",
                "schema": _STATUS_SCHEMA,
                "strict": True,
            }
        }
    elif completion_type == "tool":
        payload["tools"] = responses_tools("tool")
        payload["tool_choice"] = {"type": "function", "name": "report_status"}
    elif completion_type == "tools":
        payload["tools"] = responses_tools("tools")
        payload["tool_choice"] = "required"


# --------------------------------------------------------------------------- #
# Per-request result
# --------------------------------------------------------------------------- #
@dataclass
class RequestResult:
    ok: bool
    status: int = 0
    error: str = ""
    start: float = 0.0
    end: float = 0.0
    ttft: float | None = None          # seconds to first token
    token_times: list[float] = field(default_factory=list)  # absolute ts per chunk
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def latency(self) -> float:
        return self.end - self.start

    @property
    def inter_token_latencies(self) -> list[float]:
        if len(self.token_times) < 2:
            return []
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]


# --------------------------------------------------------------------------- #
# Single request
# --------------------------------------------------------------------------- #
async def do_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int,
    mode: str,
    completion_type: str,
    temperature: float,
    stream: bool,
) -> RequestResult:
    payload: dict = {}
    if mode == "chat":
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        _apply_chat_completion_type(payload, completion_type)
    else:  # responses
        url = f"{base_url}/v1/responses"
        payload = {
            "model": model,
            "instructions": system,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        _apply_responses_completion_type(payload, completion_type)

    if not stream:
        return await _do_request_nostream(client, url, payload, mode)

    res = RequestResult(ok=False)
    res.start = time.perf_counter()
    try:
        async with client.stream("POST", url, json=payload) as resp:
            res.status = resp.status_code
            if resp.status_code != 200:
                body = await resp.aread()
                res.error = body.decode("utf-8", "replace")[:300]
                res.end = time.perf_counter()
                return res

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if mode == "chat":
                    _consume_chat_chunk(chunk, now, res)
                else:
                    _consume_responses_event(chunk, now, res)

        res.end = time.perf_counter()
        res.ok = True
        # Fallback completion token estimate if server didn't send usage.
        if res.completion_tokens == 0 and res.token_times:
            res.completion_tokens = len(res.token_times)
        return res
    except Exception as exc:  # noqa: BLE001
        res.end = time.perf_counter()
        res.error = f"{type(exc).__name__}: {exc}"
        return res


def _consume_chat_chunk(chunk: dict, now: float, res: RequestResult) -> None:
    """Parse one /v1/chat/completions SSE chunk into `res`."""
    usage = chunk.get("usage")
    if usage:
        res.prompt_tokens = usage.get("prompt_tokens", res.prompt_tokens)
        res.completion_tokens = usage.get("completion_tokens", res.completion_tokens)

    choices = chunk.get("choices") or []
    if not choices:
        return
    delta = choices[0].get("delta") or {}
    # Text content, reasoning tokens (thinking models emit these first and they
    # do consume decode compute), or streamed tool-call argument fragments.
    piece = (
        delta.get("content")
        or delta.get("reasoning_content")
        or delta.get("reasoning")
    )
    if not piece:
        for tc in delta.get("tool_calls") or []:
            piece = (tc.get("function") or {}).get("arguments")
            if piece:
                break
    if piece:
        if res.ttft is None:
            res.ttft = now - res.start
        res.token_times.append(now)


def _consume_responses_event(event: dict, now: float, res: RequestResult) -> None:
    """Parse one /v1/responses SSE event into `res`.

    The Responses API emits typed events; text arrives as
    `response.output_text.delta`, tool-call args as
    `response.function_call_arguments.delta`, and the final usage in
    `response.completed`.
    """
    etype = event.get("type", "")
    if etype in (
        "response.output_text.delta",
        "response.function_call_arguments.delta",
    ):
        if event.get("delta"):
            if res.ttft is None:
                res.ttft = now - res.start
            res.token_times.append(now)
    elif etype in ("response.completed", "response.incomplete"):
        usage = (event.get("response") or {}).get("usage") or {}
        if usage:
            res.prompt_tokens = usage.get("input_tokens", res.prompt_tokens)
            res.completion_tokens = usage.get("output_tokens", res.completion_tokens)


async def _do_request_nostream(
    client: httpx.AsyncClient, url: str, payload: dict, mode: str
) -> RequestResult:
    """Non-streaming request: full response at once. No TTFT available."""
    res = RequestResult(ok=False)
    res.start = time.perf_counter()
    try:
        resp = await client.post(url, json=payload)
        res.end = time.perf_counter()
        res.status = resp.status_code
        if resp.status_code != 200:
            res.error = resp.text[:300]
            return res

        body = resp.json()
        if mode == "chat":
            usage = body.get("usage") or {}
            res.prompt_tokens = usage.get("prompt_tokens", 0)
            res.completion_tokens = usage.get("completion_tokens", 0)
            if res.completion_tokens == 0:
                choices = body.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    text = (msg.get("content") or msg.get("reasoning_content")
                            or msg.get("reasoning") or "")
                    for tc in msg.get("tool_calls") or []:
                        text += (tc.get("function") or {}).get("arguments", "")
                    res.completion_tokens = _count_tokens(text)
        else:  # responses
            usage = body.get("usage") or {}
            res.prompt_tokens = usage.get("input_tokens", 0)
            res.completion_tokens = usage.get("output_tokens", 0)
            if res.completion_tokens == 0:
                res.completion_tokens = _count_tokens(_extract_responses_text(body))
        res.ok = True
        return res
    except Exception as exc:  # noqa: BLE001
        res.end = time.perf_counter()
        res.error = f"{type(exc).__name__}: {exc}"
        return res


def _extract_responses_text(body: dict) -> str:
    """Pull concatenated output text from a /v1/responses non-stream body."""
    # Convenience field when present.
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    parts = []
    for item in body.get("output") or []:
        # function_call items carry their JSON args here.
        if item.get("type") == "function_call" and item.get("arguments"):
            parts.append(item["arguments"])
        for content in item.get("content") or []:
            if content.get("type") in ("output_text", "text") and content.get("text"):
                parts.append(content["text"])
    return "".join(parts)


# --------------------------------------------------------------------------- #
# vLLM /metrics (Prometheus) scraping
# --------------------------------------------------------------------------- #
# Cumulative counters and histogram sum/count we diff across a run.
_COUNTER_METRICS = {
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
}
# Histograms: we read _sum and _count and diff both to get a per-run average.
_HIST_METRICS = {
    "vllm:time_to_first_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prefill_kv_computed_tokens",
}
# Point-in-time gauges we sample by polling during the run.
_GAUGE_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
}


def parse_metrics(text: str) -> dict:
    """Parse the Prometheus text exposition into the values we care about.

    Returns a dict with: counters & histogram _sum/_count keyed by metric name
    (summed across label sets), gauges keyed by name, and request_success_total
    broken down under "success" -> {finished_reason: count}.
    """
    out: dict[str, float] = {}
    success: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            nameblock, val = line.rsplit(" ", 1)
            v = float(val)
        except ValueError:
            continue
        if "{" in nameblock:
            name = nameblock[: nameblock.index("{")]
            labels = nameblock[nameblock.index("{") + 1 : nameblock.rindex("}")]
        else:
            name, labels = nameblock, ""

        if name == "vllm:request_success_total":
            reason = "unknown"
            for part in labels.split(","):
                if part.startswith("finished_reason="):
                    reason = part.split("=", 1)[1].strip('"')
                    break
            success[reason] = success.get(reason, 0.0) + v
        elif name in _COUNTER_METRICS or name in _GAUGE_METRICS:
            out[name] = out.get(name, 0.0) + v
        elif name.endswith(("_sum", "_count")):
            base = name.rsplit("_", 1)[0]
            if base in _HIST_METRICS:
                out[name] = out.get(name, 0.0) + v
    out["success"] = success  # type: ignore[assignment]
    return out


async def snapshot_metrics(client: httpx.AsyncClient, metrics_url: str) -> dict | None:
    try:
        resp = await client.get(metrics_url)
        if resp.status_code != 200:
            return None
        return parse_metrics(resp.text)
    except Exception:  # noqa: BLE001
        return None


def _avg(after: dict, before: dict, base: str) -> float | None:
    """Per-run average from histogram deltas: Δsum / Δcount."""
    dc = after.get(f"{base}_count", 0.0) - before.get(f"{base}_count", 0.0)
    ds = after.get(f"{base}_sum", 0.0) - before.get(f"{base}_sum", 0.0)
    return ds / dc if dc > 0 else None


def diff_metrics(before: dict, after: dict, gauges: dict) -> dict:
    """Build a server-side summary from before/after counters + sampled gauges."""
    q = after.get("vllm:prefix_cache_queries_total", 0.0) - before.get(
        "vllm:prefix_cache_queries_total", 0.0)
    h = after.get("vllm:prefix_cache_hits_total", 0.0) - before.get(
        "vllm:prefix_cache_hits_total", 0.0)
    pt = after.get("vllm:prompt_tokens_total", 0.0) - before.get(
        "vllm:prompt_tokens_total", 0.0)
    ptc = after.get("vllm:prompt_tokens_cached_total", 0.0) - before.get(
        "vllm:prompt_tokens_cached_total", 0.0)
    gen = after.get("vllm:generation_tokens_total", 0.0) - before.get(
        "vllm:generation_tokens_total", 0.0)
    preempt = after.get("vllm:num_preemptions_total", 0.0) - before.get(
        "vllm:num_preemptions_total", 0.0)
    succ = {
        r: after["success"].get(r, 0.0) - before.get("success", {}).get(r, 0.0)
        for r in after.get("success", {})
    }
    return {
        "prefix_hit_rate": (h / q) if q > 0 else None,
        "prefix_queried_tokens": q,
        "prefix_hit_tokens": h,
        "prompt_tokens": pt,
        "prompt_tokens_cached": ptc,
        "generation_tokens": gen,
        "preemptions": preempt,
        "ttft_avg_s": _avg(after, before, "vllm:time_to_first_token_seconds"),
        "queue_avg_s": _avg(after, before, "vllm:request_queue_time_seconds"),
        "prefill_avg_s": _avg(after, before, "vllm:request_prefill_time_seconds"),
        "decode_avg_s": _avg(after, before, "vllm:request_decode_time_seconds"),
        "e2e_avg_s": _avg(after, before, "vllm:e2e_request_latency_seconds"),
        "prefill_kv_computed_avg": _avg(
            after, before, "vllm:request_prefill_kv_computed_tokens"),
        "finished": {r: c for r, c in succ.items() if c},
        "running_peak": gauges.get("running_peak"),
        "running_avg": gauges.get("running_avg"),
        "waiting_peak": gauges.get("waiting_peak"),
        "waiting_avg": gauges.get("waiting_avg"),
        "kv_peak": gauges.get("kv_peak"),
        "kv_avg": gauges.get("kv_avg"),
    }


# --------------------------------------------------------------------------- #
# Runner for one concurrency level
# --------------------------------------------------------------------------- #
async def run_level(
    base_url: str,
    model: str,
    prompts: list[str],
    system: str,
    concurrency: int,
    max_tokens: int,
    mode: str,
    completion_type: str,
    temperature: float,
    timeout: float,
    api_key: str | None,
    stream: bool,
    metrics_url: str | None = None,
    metrics_interval: float = 0.5,
) -> tuple[list[RequestResult], float, dict | None]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    limits = httpx.Limits(
        max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4
    )
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    server_metrics: dict | None = None

    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers) as client:

        async def worker(p: str) -> None:
            async with sem:
                r = await do_request(
                    client, base_url, model, p, system, max_tokens, mode,
                    completion_type, temperature, stream
                )
                results.append(r)

        # Background gauge poller (running / waiting / KV usage are 0 at rest,
        # so we must sample them while the load is in flight).
        samples: list[tuple[float, float, float]] = []
        stop_poll = asyncio.Event()

        async def poll() -> None:
            while not stop_poll.is_set():
                snap = await snapshot_metrics(client, metrics_url)  # type: ignore[arg-type]
                if snap:
                    samples.append((
                        snap.get("vllm:num_requests_running", 0.0),
                        snap.get("vllm:num_requests_waiting", 0.0),
                        snap.get("vllm:kv_cache_usage_perc", 0.0),
                    ))
                try:
                    await asyncio.wait_for(stop_poll.wait(), timeout=metrics_interval)
                except asyncio.TimeoutError:
                    pass

        before = await snapshot_metrics(client, metrics_url) if metrics_url else None
        poller = asyncio.create_task(poll()) if metrics_url else None

        wall_start = time.perf_counter()
        await asyncio.gather(*(worker(p) for p in prompts))
        wall = time.perf_counter() - wall_start

        if poller:
            stop_poll.set()
            await poller
        after = await snapshot_metrics(client, metrics_url) if metrics_url else None

        if before and after:
            gauges: dict = {}
            if samples:
                run = [s[0] for s in samples]
                wait = [s[1] for s in samples]
                kv = [s[2] for s in samples]
                gauges = {
                    "running_peak": max(run), "running_avg": statistics.mean(run),
                    "waiting_peak": max(wait), "waiting_avg": statistics.mean(wait),
                    "kv_peak": max(kv), "kv_avg": statistics.mean(kv),
                }
            server_metrics = diff_metrics(before, after, gauges)

    return results, wall, server_metrics


# --------------------------------------------------------------------------- #
# Cache warmup
# --------------------------------------------------------------------------- #
async def warmup(
    base_url: str,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int,
    mode: str,
    completion_type: str,
    temperature: float,
    timeout: float,
    api_key: str | None,
    stream: bool,
    n: int,
) -> None:
    """Send `n` warmup requests sequentially before the timed run.

    Only useful when running with a shared prefix (``--cache on``): the first
    request populates vLLM's prefix cache so the timed requests measure the
    warm path instead of paying a one-off cold prefill. Results are discarded.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        for i in range(n):
            t0 = time.perf_counter()
            r = await do_request(
                client, base_url, model, prompt, system, max_tokens, mode,
                completion_type, temperature, stream,
            )
            dt = time.perf_counter() - t0
            status = "ok" if r.ok else f"FAILED [{r.status}] {r.error[:120]}"
            print(f"  warmup {i + 1}/{n}: {status} ({dt:.2f}s)")


# --------------------------------------------------------------------------- #
# Stats / reporting
# --------------------------------------------------------------------------- #
def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def fmt_ms(x: float) -> str:
    return "n/a" if x != x else f"{x * 1000:.1f}ms"  # x!=x => NaN


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _fmt_s(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1000:.1f}ms"


def print_server_metrics(m: dict) -> None:
    """Print the vLLM-side view scraped from /metrics for this run."""
    print(f"\n  {'-' * 64}")
    print("  vLLM server metrics (/metrics, deltas over this run)")
    print(f"  {'-' * 64}")
    print(f"  Prefix cache:    hit {_fmt_pct(m['prefix_hit_rate'])} of "
          f"{int(m['prefix_queried_tokens'])} queried tokens "
          f"({int(m['prefix_hit_tokens'])} cached)")
    if m["prompt_tokens"]:
        cached_pct = m["prompt_tokens_cached"] / m["prompt_tokens"]
        print(f"  Prompt tokens:   {int(m['prompt_tokens'])} prefilled, "
              f"{int(m['prompt_tokens_cached'])} from cache ({_fmt_pct(cached_pct)})")
    if m["prefill_kv_computed_avg"] is not None:
        print(f"  KV computed:     {m['prefill_kv_computed_avg']:.0f} new tokens/req "
              f"(uncached prefill work)")
    print(f"  Gen tokens:      {int(m['generation_tokens'])} decoded")
    # Server-side latency split (per-request averages from histograms).
    parts = []
    for label, key in (("queue", "queue_avg_s"), ("prefill", "prefill_avg_s"),
                       ("decode", "decode_avg_s"), ("ttft", "ttft_avg_s"),
                       ("e2e", "e2e_avg_s")):
        if m[key] is not None:
            parts.append(f"{label} {_fmt_s(m[key])}")
    if parts:
        print(f"  Latency split:   {' | '.join(parts)}")
    if m["preemptions"]:
        print(f"  Preemptions:     {int(m['preemptions'])}  (KV pressure -> recompute)")
    if m["running_peak"] is not None:
        print(f"  Running reqs:    peak {m['running_peak']:.0f} | "
              f"avg {m['running_avg']:.1f}")
        print(f"  Waiting reqs:    peak {m['waiting_peak']:.0f} | "
              f"avg {m['waiting_avg']:.1f}")
        print(f"  KV cache usage:  peak {_fmt_pct(m['kv_peak'])} | "
              f"avg {_fmt_pct(m['kv_avg'])}")
    if m["finished"]:
        reasons = ", ".join(f"{r}={int(c)}" for r, c in m["finished"].items())
        print(f"  Finished:        {reasons}")


def report(label: str, results: list[RequestResult], wall: float,
           server_metrics: dict | None = None) -> dict:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    latencies = [r.latency for r in ok]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    itls = [itl for r in ok for itl in r.inter_token_latencies]
    completion_tokens = sum(r.completion_tokens for r in ok)
    prompt_tokens = sum(r.prompt_tokens for r in ok)

    out_tok_per_s = completion_tokens / wall if wall > 0 else 0.0
    req_per_s = len(ok) / wall if wall > 0 else 0.0

    print(f"\n{'=' * 68}")
    print(f"  {label}")
    print(f"{'=' * 68}")
    print(f"  Requests:        {len(ok)} ok / {len(failed)} failed / {len(results)} total")
    print(f"  Wall time:       {wall:.2f}s")
    print(f"  Throughput:      {req_per_s:.2f} req/s | {out_tok_per_s:.1f} out-tok/s")
    if prompt_tokens:
        print(f"  Prompt tokens:   {prompt_tokens} total ({prompt_tokens / max(len(ok),1):.0f}/req)")
    if completion_tokens:
        print(f"  Output tokens:   {completion_tokens} total ({completion_tokens / max(len(ok),1):.0f}/req)")

    if latencies:
        print(f"  Latency (e2e):   "
              f"avg {fmt_ms(statistics.mean(latencies))} | "
              f"p50 {fmt_ms(pct(latencies, 50))} | "
              f"p90 {fmt_ms(pct(latencies, 90))} | "
              f"p99 {fmt_ms(pct(latencies, 99))} | "
              f"max {fmt_ms(max(latencies))}")
    if ttfts:
        print(f"  TTFT:            "
              f"avg {fmt_ms(statistics.mean(ttfts))} | "
              f"p50 {fmt_ms(pct(ttfts, 50))} | "
              f"p90 {fmt_ms(pct(ttfts, 90))} | "
              f"p99 {fmt_ms(pct(ttfts, 99))}")
    if itls:
        print(f"  Inter-token:     "
              f"avg {fmt_ms(statistics.mean(itls))} | "
              f"p50 {fmt_ms(pct(itls, 50))} | "
              f"p99 {fmt_ms(pct(itls, 99))}")

    if failed:
        # Show a couple of distinct error samples.
        seen = set()
        print("  Errors:")
        for r in failed:
            key = (r.status, r.error[:80])
            if key in seen:
                continue
            seen.add(key)
            print(f"    [{r.status}] {r.error[:160]}")
            if len(seen) >= 3:
                break

    if server_metrics:
        print_server_metrics(server_metrics)

    summary = {
        "label": label,
        "ok": len(ok),
        "failed": len(failed),
        "wall_s": round(wall, 3),
        "req_per_s": round(req_per_s, 3),
        "out_tok_per_s": round(out_tok_per_s, 2),
        "latency_avg_ms": round(statistics.mean(latencies) * 1000, 1) if latencies else None,
        "latency_p50_ms": round(pct(latencies, 50) * 1000, 1) if latencies else None,
        "latency_p90_ms": round(pct(latencies, 90) * 1000, 1) if latencies else None,
        "latency_p99_ms": round(pct(latencies, 99) * 1000, 1) if latencies else None,
        "ttft_avg_ms": round(statistics.mean(ttfts) * 1000, 1) if ttfts else None,
        "ttft_p90_ms": round(pct(ttfts, 90) * 1000, 1) if ttfts else None,
    }
    if server_metrics:
        summary["server"] = server_metrics
    return summary


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #
async def discover_model(base_url: str, api_key: str | None, timeout: float) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        resp = await client.get(f"{base_url}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if not models:
            raise RuntimeError("No models reported by /v1/models")
        return models[0]["id"]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main_async(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    api_key = args.api_key

    model = args.model
    if not model:
        try:
            model = await discover_model(base_url, api_key, args.timeout)
            print(f"Discovered model: {model}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not discover model from {base_url}/v1/models: {exc}",
                  file=sys.stderr)
            print("Pass --model explicitly.", file=sys.stderr)
            return 2

    # Coherent system prompt that pins a fixed output regardless of user content.
    system = args.system_prompt or system_prompt_for(args.completion_type)

    # Build the prompt set once (reused across concurrency levels for comparability).
    cache_on = args.cache == "on"
    print(f"Building {args.num_requests} prompts of ~{args.prompt_tokens} tokens "
          f"({'tiktoken' if _ENC else 'heuristic'} counting)...")
    if cache_on:
        # Identical prompt for every request -> full prefix-cache hits after
        # the first. Build once and reuse.
        shared = build_prompt(args.prompt_tokens, 0, shared_prefix=True, salt=args.salt)
        prompts = [shared] * args.num_requests
    else:
        prompts = [
            build_prompt(args.prompt_tokens, i, shared_prefix=False, salt=args.salt)
            for i in range(args.num_requests)
        ]
    actual = _count_tokens(prompts[0])
    sys_tokens = _count_tokens(system)
    print(f"  Prompt[0]: ~{actual} user tokens + ~{sys_tokens} system tokens "
          f"= ~{actual + sys_tokens} total")
    print(f"  Completion type: {args.completion_type}")
    cache_desc = (
        "on (shared prefix -> prefix-cache hits)" if cache_on
        else "off (unique per-request prefix -> cold prefill)"
    )
    if args.salt:
        cache_desc += f", salt={args.salt!r}"
    print(f"  Prefix cache:    {cache_desc}")

    # Concurrency levels.
    if args.sequential:
        levels = [1]
    else:
        levels = [int(c) for c in str(args.concurrency).split(",") if c.strip()]

    stream = not args.no_stream

    # vLLM /metrics scraping (server-side view).
    metrics_url = None
    if args.metrics:
        metrics_url = args.metrics_url or f"{base_url}/metrics"
        async with httpx.AsyncClient(timeout=args.timeout) as c:
            probe = await snapshot_metrics(c, metrics_url)
        if probe is None:
            print(f"\nWarning: could not read {metrics_url}; "
                  f"server metrics disabled.")
            metrics_url = None
        else:
            print(f"  Server metrics:  scraping {metrics_url} "
                  f"every {args.metrics_interval}s")

    # Optional cache warmup before the timed run(s).
    if args.warmup > 0:
        if not cache_on:
            print(f"\nNote: --warmup {args.warmup} with --cache off; warmup uses "
                  f"prompt[0] only, so it won't warm the unique per-request prefixes.")
        print(f"\n>>> Warmup: {args.warmup} request(s) to populate the cache...")
        await warmup(
            base_url, model, prompts[0], system, args.max_tokens, args.mode,
            args.completion_type, args.temperature, args.timeout, api_key,
            stream, args.warmup,
        )

    summaries = []
    for level in levels:
        label = (
            f"SEQUENTIAL (concurrency=1)" if (args.sequential and level == 1)
            else f"CONCURRENCY = {level}"
        )
        print(f"\n>>> Running {label}: {args.num_requests} requests, "
              f"max_tokens={args.max_tokens}, mode={args.mode}, "
              f"type={args.completion_type}, stream={'on' if stream else 'off'}")

        results, wall, server_metrics = await run_level(
            base_url, model, prompts, system, level, args.max_tokens,
            args.mode, args.completion_type, args.temperature, args.timeout,
            api_key, stream, metrics_url, args.metrics_interval,
        )
        summaries.append(report(label, results, wall, server_metrics))

    if len(summaries) > 1:
        print(f"\n{'#' * 68}")
        print("  SWEEP SUMMARY")
        print(f"{'#' * 68}")
        print(f"  {'level':<22} {'req/s':>8} {'tok/s':>9} {'lat p50':>10} "
              f"{'lat p99':>10} {'ttft p90':>10} {'fail':>6}")
        for s in summaries:
            print(f"  {s['label']:<22} {s['req_per_s']:>8.2f} {s['out_tok_per_s']:>9.1f} "
                  f"{(str(s['latency_p50_ms']) + 'ms'):>10} "
                  f"{(str(s['latency_p99_ms']) + 'ms'):>10} "
                  f"{(str(s['ttft_p90_ms']) + 'ms'):>10} {s['failed']:>6}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote JSON summary to {args.json_out}")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load test a vLLM OpenAI-compatible server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", required=True,
                   help="Base URL of the vLLM server, e.g. http://localhost:8000 "
                        "or https://xxxx-8000.proxy.runpod.net")
    p.add_argument("--model", default=None,
                   help="Model id. If omitted, discovered via /v1/models.")
    p.add_argument("--api-key", default=None,
                   help="Bearer token (if the server requires one).")
    p.add_argument("--mode", choices=["chat", "responses"], default="chat",
                   help="Which OpenAI-compatible endpoint to hit: "
                        "chat -> /v1/chat/completions, responses -> /v1/responses.")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable streaming (request the full response at once). "
                        "TTFT/inter-token metrics are unavailable in this mode.")
    p.add_argument("--completion-type", choices=["text", "json", "tool", "tools"],
                   default="text",
                   help="Output type to force: text, json (structured/guided "
                        "JSON schema), tool (one forced function), tools (several "
                        "tools, one call required). A coherent system prompt pins "
                        "the model to a fixed output for each type.")
    p.add_argument("--system-prompt", default=None,
                   help="Override the built-in fixed-output system prompt.")
    p.add_argument("--num-requests", type=int, default=50,
                   help="Total number of requests to send (per concurrency level).")
    p.add_argument("--concurrency", default="10",
                   help="Concurrency level, or comma-separated list to sweep "
                        "(e.g. 1,2,4,8,16). Ignored if --sequential.")
    p.add_argument("--sequential", action="store_true",
                   help="Send requests one at a time (concurrency=1).")
    p.add_argument("--prompt-tokens", type=int, default=512,
                   help="Approximate prompt size in tokens.")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Max output tokens to generate per request.")
    p.add_argument("--cache", choices=["off", "on"], default="off",
                   help="Prefix-cache behaviour. off (default): every request "
                        "gets a unique prefix so vLLM's prefix cache misses and "
                        "you measure cold prefill. on: all requests share an "
                        "identical prompt so requests after the first are "
                        "prefix-cache hits (warm path).")
    p.add_argument("--salt", default="",
                   help="String prepended to every prompt prefix to invalidate "
                        "vLLM's cross-run prefix cache (which persists between "
                        "invocations). Pass a fresh value for a guaranteed cold "
                        "start, or keep it stable to reuse a previously warmed cache.")
    p.add_argument("--warmup", type=int, default=0,
                   help="Number of warmup requests to send sequentially before "
                        "the timed run (results discarded). Useful with --cache on "
                        "to populate the prefix cache first.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-request timeout in seconds.")
    p.add_argument("--metrics", action="store_true",
                   help="Scrape vLLM's Prometheus /metrics before/after each run "
                        "and poll gauges during it, to report the server-side view "
                        "(prefix-cache hit rate, KV-computed tokens, queue/prefill/"
                        "decode split, Running/Waiting/KV%%, preemptions).")
    p.add_argument("--metrics-url", default=None,
                   help="Override the metrics endpoint (default: <base-url>/metrics).")
    p.add_argument("--metrics-interval", type=float, default=0.5,
                   help="Seconds between gauge samples while load is in flight.")
    p.add_argument("--json-out", default=None,
                   help="Optional path to write a JSON summary of results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rc = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
