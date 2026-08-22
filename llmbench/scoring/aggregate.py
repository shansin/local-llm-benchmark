"""Aggregating repeated measurements into per-model summaries."""

from __future__ import annotations

from typing import Any

from llmbench.scoring.extract import extraction_for, looks_like_leaked_reasoning
from llmbench.stats import iqr, mean, median, percentile, stdev

Repeats = list[dict[str, Any]]


def successful(results: Repeats) -> Repeats:
    """Drop timed-out and errored generations; they have no metrics to average."""
    return [r for r in results if not r.get("error")]


def perf_summary(results: Repeats) -> dict[str, float]:
    """Summarise throughput/latency across repeats.

    Reports median and spread rather than a mean: a single stalled sample
    should not be able to move the headline number.
    """
    ok = successful(results)
    tps = [r["tokens_per_sec"] for r in ok]
    ttft = [r["ttft"] for r in ok]
    prefill = [r.get("prompt_eval_speed", 0.0) for r in ok]
    return {
        "tps_median": median(tps),
        "tps_iqr": iqr(tps),
        "ttft_p50": median(ttft),
        "ttft_p90": percentile(ttft, 90),
        "prefill_median": median(prefill),
        "gen_time_median": median([r["total_time"] for r in ok]),
        "tokens_median": median([float(r["eval_count"]) for r in ok]),
        "samples": float(len(ok)),
        "failures": float(len(results) - len(ok)),
    }


def completeness(results_by_task: dict[str, Repeats]) -> dict[str, float]:
    """Count the generations that produced no answer, and why.

    Quality tables average scores; they cannot show that a score is low because
    the answer was absent. These three counts are what separates "answered
    badly" from "never answered", and the second is usually a fact about the
    run's configuration rather than about the model.
    """
    total = truncated = empty = leaked = errors = 0
    for repeats in results_by_task.values():
        for result in repeats:
            total += 1
            if result.get("error"):
                errors += 1
                continue
            extraction = extraction_for(result)
            if result.get("truncated") or not extraction.complete:
                truncated += 1
            if extraction.empty:
                empty += 1
            elif looks_like_leaked_reasoning(extraction):
                leaked += 1

    return {
        "total": float(total),
        "truncated": float(truncated),
        "empty": float(empty),
        "leaked_reasoning": float(leaked),
        "errors": float(errors),
    }


def peak_vram(results: Repeats) -> float:
    """Highest VRAM delta observed across repeats, in MiB. 0.0 when unmeasured."""
    deltas = [
        float(r["gpu"]["peak_delta_mib"])
        for r in successful(results)
        if isinstance(r.get("gpu"), dict) and "peak_delta_mib" in r["gpu"]
    ]
    return max(deltas) if deltas else 0.0


def combined_perf(
    results_by_category: dict[str, Repeats], categories: list[str]
) -> dict[str, float]:
    """Perf summary pooled over every answer the model generated.

    Used when no dedicated perf probe ran, so the report still has throughput
    numbers — measured from the benchmark answers themselves, as before.
    """
    pooled: Repeats = []
    for category in categories:
        pooled.extend(results_by_category.get(category, []))
    summary = perf_summary(pooled)
    summary["total_tokens"] = sum(r["eval_count"] for r in successful(pooled))
    return summary


def blend_repeats(
    objective: list[dict[str, Any]], judge: list[dict[str, Any]], weight: float = 0.6
) -> list[dict[str, Any]]:
    """Pair up objective and judge verdicts repeat by repeat and blend them."""
    from llmbench.scoring.objective import blended_score

    count = max(len(objective), len(judge))
    blended = []
    for i in range(count):
        obj = objective[i].get("score") if i < len(objective) else None
        jud = judge[i].get("score") if i < len(judge) else None
        blended.append({"score": blended_score(obj, jud, weight)})
    return blended


def task_score_stats(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and spread of scores for one task across repeats.

    Works for judge verdicts, objective results, and blended scores alike —
    they all carry a nullable `score`.
    """
    values = [float(v["score"]) for v in verdicts if v.get("score") is not None]
    return {
        "mean": mean(values) if values else None,
        "std": stdev(values),
        "n": len(values),
        "n_attempted": len(verdicts),
    }


def model_score_stats(
    scores_by_task: dict[str, list[dict[str, Any]]],
    task_keys: list[str],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Quality over a set of tasks — a whole model, or one category of it.

    Unscored tasks are excluded from the average rather than counted as zero —
    a judge that fails to answer should not look like a model that answered
    badly. `repeat_std` is the average within-task spread across repeats: the
    noise floor that any gap between models has to clear.
    """
    scored: list[tuple[float, float, float]] = []  # (mean, std, weight)
    for key in task_keys:
        stats = task_score_stats(scores_by_task.get(key, []))
        if stats["mean"] is not None:
            weight = (weights or {}).get(key, 1.0)
            scored.append((stats["mean"], stats["std"], weight))

    if not scored:
        return {"mean": None, "repeat_std": 0.0, "n_scored": 0, "n_total": len(task_keys)}

    total_weight = sum(w for _, _, w in scored)
    return {
        "mean": sum(m * w for m, _, w in scored) / total_weight,
        "repeat_std": mean([s for _, s, _ in scored]),
        "n_scored": len(scored),
        "n_total": len(task_keys),
    }
