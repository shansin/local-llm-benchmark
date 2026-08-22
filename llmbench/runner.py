"""Running prompts against Ollama and extracting timing metrics."""

from __future__ import annotations

import json
import random
import string
import time
import uuid
from dataclasses import replace
from typing import Any, TypedDict

import requests

from llmbench import telemetry
from llmbench.config import PERF_PROBE_PROMPT, GenerationParams

# Rough characters-per-token ratio, used only to aim a probe at a length.
CHARS_PER_TOKEN = 4

# Fresh per process, so probe prompts are never text this Ollama has already cached.
_PROBE_SALT = uuid.uuid4().hex


class GenResult(TypedDict):
    """One generation, with the metrics measured for it.

    `ttft` is measured client-side from request to first streamed token — the
    number a user actually waits. The server-reported components are kept
    alongside it: `prefill_time` (prompt evaluation) and `load_time` (model
    load), so a cold first prompt can be told apart from a slow one.

    `thinking` holds the chain of thought when Ollama streams it on its own
    channel. It has to be captured: a reasoning model that puts everything
    there and never reaches a visible answer is indistinguishable, from
    `response` alone, from a model that returned nothing — and the second
    reading is the one that quietly lands in the scores.

    `truncated` marks a generation that stopped because it ran out of room
    rather than because the model was finished. Those are not bad answers, they
    are absent ones, and a benchmark that averages them in as zeros is
    reporting its own context setting as a property of the model.
    """

    response: str
    thinking: str
    done_reason: str
    truncated: bool
    tokens_per_sec: float
    ttft: float
    prefill_time: float
    load_time: float
    total_time: float
    eval_count: int
    prompt_eval_speed: float
    prompt_eval_count: int
    seed: int | None
    error: str | None
    gpu: dict[str, float] | None


def _failed(marker: str, error: str, seed: int | None = None) -> GenResult:
    return {
        "response": marker,
        "thinking": "",
        "done_reason": "",
        "truncated": False,
        "tokens_per_sec": 0.0,
        "ttft": 0.0,
        "prefill_time": 0.0,
        "load_time": 0.0,
        "total_time": 0.0,
        "eval_count": 0,
        "prompt_eval_speed": 0.0,
        "prompt_eval_count": 0,
        "seed": seed,
        "error": error,
        "gpu": None,
    }


def metrics_from_response(
    data: dict[str, Any],
    text: str | None = None,
    client_ttft: float | None = None,
    thinking: str = "",
    num_ctx: int | None = None,
) -> GenResult:
    """Convert an Ollama /api/generate payload into a GenResult."""
    eval_duration = data.get("eval_duration", 0)
    eval_count = data.get("eval_count", 0)
    load_duration = data.get("load_duration", 0)
    prompt_eval_duration = data.get("prompt_eval_duration", 0)
    total_duration = data.get("total_duration", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)

    prefill_time = prompt_eval_duration / 1e9
    return {
        "response": data.get("response", "") if text is None else text,
        "thinking": thinking or str(data.get("thinking") or ""),
        "done_reason": str(data.get("done_reason") or ""),
        "truncated": _hit_the_wall(data, eval_count, prompt_eval_count, num_ctx),
        "tokens_per_sec": (eval_count / eval_duration * 1e9) if eval_duration > 0 else 0.0,
        # Without a streamed measurement, the best available estimate of the
        # user-visible wait is load + prefill.
        "ttft": client_ttft if client_ttft is not None else (load_duration / 1e9 + prefill_time),
        "prefill_time": prefill_time,
        "load_time": load_duration / 1e9,
        "total_time": total_duration / 1e9,
        "eval_count": eval_count,
        "prompt_eval_speed": (
            (prompt_eval_count / prompt_eval_duration * 1e9) if prompt_eval_duration > 0 else 0.0
        ),
        "prompt_eval_count": prompt_eval_count,
        "seed": None,
        "error": None,
        "gpu": None,
    }


def _rejected_thinking(exc: Exception) -> bool:
    """Is this failure Ollama saying the model has no thinking channel?"""
    return "think" in str(exc).lower()


def _hit_the_wall(
    data: dict[str, Any], eval_count: int, prompt_eval_count: int, num_ctx: int | None
) -> bool:
    """Did this generation stop for lack of room rather than because it finished?

    Ollama says so directly with `done_reason: "length"`. The context check is a
    belt-and-braces second reading for builds that omit the reason: a
    generation whose prompt and output together fill the window did not choose
    to stop. The margin absorbs the handful of tokens the template adds.
    """
    if str(data.get("done_reason") or "").lower() == "length":
        return True
    return bool(num_ctx and eval_count and prompt_eval_count + eval_count >= num_ctx - 16)


def _stream_generate(
    base_url: str, payload: dict[str, Any], timeout: int
) -> tuple[str, str, dict[str, Any], float | None]:
    """POST a streaming request, returning (text, thinking, final_payload, ttft)."""
    start = time.monotonic()
    resp = requests.post(
        f"{base_url}/api/generate", json={**payload, "stream": True}, timeout=timeout, stream=True
    )
    resp.raise_for_status()

    chunks: list[str] = []
    thoughts: list[str] = []
    ttft: float | None = None
    final: dict[str, Any] = {}

    for line in resp.iter_lines():
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("error"):
            raise requests.exceptions.HTTPError(str(obj["error"]))
        piece = obj.get("response", "")
        # Reasoning models stream their chain of thought on a separate channel.
        # It is kept apart from the answer but not thrown away: it is the only
        # evidence of what a model that never reached an answer spent its
        # context on.
        thought = obj.get("thinking", "")
        if thought:
            thoughts.append(thought)
        if piece:
            if ttft is None:
                ttft = time.monotonic() - start
            chunks.append(piece)
        if obj.get("done"):
            final = obj
        elif time.monotonic() - start > timeout:
            # iter_lines' timeout is per-read; this bounds the whole generation.
            resp.close()
            raise requests.exceptions.ReadTimeout("exceeded total prompt timeout")

    return "".join(chunks), "".join(thoughts), final, ttft


def run_prompt(
    base_url: str,
    model_name: str,
    prompt_text: str,
    timeout: int,
    params: GenerationParams | None = None,
    retries: int = 3,
    think: bool | None = None,
) -> GenResult:
    """Run a prompt against a model and return its response plus metrics.

    Never raises on a transport or HTTP failure: the failure is recorded as a
    result so that one bad model cannot abort a multi-hour run. Connection
    errors are retried with backoff; timeouts are not — a model that cannot
    finish in the allotted time is a finding, not a flake.
    """
    params = params or GenerationParams()
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt_text,
        "options": params.to_options(),
    }
    if think is not None:
        payload["think"] = think

    last_error = "unknown"
    for attempt in range(retries):
        try:
            text, thinking, final, ttft = _stream_generate(base_url, payload, timeout)
        except requests.exceptions.Timeout:
            print("timeout, skipping.")
            return _failed("[TIMEOUT]", "timeout", params.seed)
        except requests.exceptions.ConnectionError as exc:
            last_error = type(exc).__name__
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
        except requests.exceptions.RequestException as exc:
            if "think" in payload and _rejected_thinking(exc):
                # Not every model supports the thinking channel. Asking for it
                # and being refused is a fact about the model, not a run
                # failure: drop the option and measure it as it comes.
                print("(no thinking channel) ", end="", flush=True)
                payload.pop("think")
                continue
            detail = type(exc).__name__
            print(f"failed ({detail}), skipping.")
            return _failed(f"[ERROR: {detail}]", detail, params.seed)
        else:
            result = metrics_from_response(
                final,
                text=text,
                client_ttft=ttft,
                thinking=thinking,
                num_ctx=params.num_ctx,
            )
            result["seed"] = params.seed
            return result

    print(f"failed ({last_error}) after {retries} attempts, skipping.")
    return _failed(f"[ERROR: {last_error}]", last_error, params.seed)


def warm_up(
    base_url: str, model_name: str, timeout: int, params: GenerationParams | None = None
) -> tuple[float, dict[str, float]]:
    """Load a model into memory before timing it.

    Returns (cold-load seconds, GPU usage). Without this, whichever prompt runs
    first absorbs the entire model load and is not comparable with the rest.

    The memory figures come from Ollama's own accounting once the model is
    resident, which attributes memory to *this* model. GPU utilisation is
    sampled separately during the load.
    """
    params = replace(params or GenerationParams(), num_predict=1)
    with telemetry.measure(interval=0.25) as monitor:
        result = run_prompt(base_url, model_name, "hi", timeout, params, retries=1)

    usage = {"mean_utilization": monitor.usage.mean_utilization}
    usage.update(telemetry.loaded_model_footprint(base_url, model_name))
    return result["load_time"], usage


def run_perf_probe(
    base_url: str,
    model_name: str,
    timeout: int,
    params: GenerationParams,
    repeats: int,
    predict_tokens: int,
    prompt_text: str = PERF_PROBE_PROMPT,
) -> list[GenResult]:
    """Measure throughput with a short, fixed-length generation.

    Deriving throughput from the benchmark answers themselves conflates speed
    with verbosity and costs a full generation per sample. A capped probe is
    cheap enough to repeat properly.

    GPU memory is sampled while the probe runs — peak VRAM is what decides
    whether a model fits on the card, and it can only be seen from outside.
    """
    probe_params = replace(params, num_predict=predict_tokens)
    results = []
    for i in range(repeats):
        # A unique body per sample, so the reported prefill speed reflects work
        # actually done rather than Ollama's prompt cache.
        prompt = f"{prompt_text}\n\n{build_probe_prompt(256, nonce=f'probe-{i}')}"
        with telemetry.measure() as monitor:
            result = run_prompt(base_url, model_name, prompt, timeout, probe_params.for_repeat(i))
        result["gpu"] = monitor.usage.as_dict()
        results.append(result)
    return results


def build_probe_prompt(target_tokens: int, nonce: str = "") -> str:
    """Build a prompt of roughly `target_tokens` input tokens, unique per call.

    Length is approximate by design — the actual input length comes back from
    Ollama as `prompt_eval_count`, and that measured value is what gets reported.

    The body is unique word soup, not repeated filler. Ollama reuses cached KV
    for text it has seen, and it reports `prompt_eval_duration` for only the
    part it actually computed while still counting the whole prompt in
    `prompt_eval_count` — so a reused prompt yields a throughput figure that
    measures the cache, not the model. Observed in practice as a "prefill" of
    111,000 tok/s against a true rate near 5,500. Varying only a prefix is not
    enough; the whole body has to be new.

    Uniqueness comes from a per-process salt as well as `nonce`, so a second
    run against a still-warm Ollama measures cold prefill too. The words are
    drawn from a seeded RNG, so a single run's samples stay self-consistent.
    """
    rng = random.Random(f"{_PROBE_SALT}:{nonce}:{target_tokens}")
    target_chars = max(target_tokens, 1) * CHARS_PER_TOKEN
    words = []
    length = 0
    while length < target_chars:
        word = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9)))
        words.append(word)
        length += len(word) + 1
    return (
        "Read the following reference material.\n\n"
        + " ".join(words)
        + "\n\nSummarise the material above in one short paragraph."
    )


def run_prefill_sweep(
    base_url: str,
    model_name: str,
    timeout: int,
    params: GenerationParams,
    lengths: list[int],
    repeats: int = 2,
    predict_tokens: int = 64,
) -> dict[str, list[GenResult]]:
    """Measure prefill speed at several input lengths.

    Prompt-ingestion cost scales with input length in ways that single-length
    benchmarking hides entirely — it is where quantisation and KV-cache
    differences show up. Lengths that would not fit in the configured context
    window are skipped rather than silently truncated by Ollama.
    """
    sweep_params = replace(params, num_predict=predict_tokens)
    budget = params.num_ctx - predict_tokens
    results: dict[str, list[GenResult]] = {}

    for length in lengths:
        if length > budget:
            continue
        results[str(length)] = [
            run_prompt(
                base_url,
                model_name,
                # A distinct nonce per sample, so no repeat is served from the
                # prompt cache — that would measure the cache, not the model.
                build_probe_prompt(length, nonce=f"{length}-{i}"),
                timeout,
                sweep_params.for_repeat(i),
            )
            for i in range(repeats)
        ]
    return results
