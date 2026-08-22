"""The canonical machine-readable result file.

`results.md` is a *view*. This is the source of truth: everything needed to
rebuild any report, compare runs, or re-analyse old results without re-running
anything. It keeps the summary statistics *and* the per-repeat raw records they
came from, because a summary you cannot check is a summary you cannot trust.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmbench.scoring.aggregate import (
    Repeats,
    blend_repeats,
    completeness,
    model_score_stats,
    perf_summary,
    task_score_stats,
)
from llmbench.sysinfo import get_cpu_info, get_gpu_info, get_ram_info
from llmbench.tasks import Task, categories_of, group_by_category

RESULTS_FILE = "results.json"
SCHEMA_VERSION = 1


def build_document(
    run_name: str,
    all_results: dict[str, dict[str, Repeats]],
    all_details: dict[str, Any],
    judge_scores: dict[str, Any],
    judge_models: list[str],
    tasks: list[Task],
    perf_results: dict[str, Repeats] | None = None,
    cold_loads: dict[str, float] | None = None,
    gen_params: dict[str, Any] | None = None,
    repeats: int = 1,
    total_runtime: float | None = None,
    objective_scores: dict[str, Any] | None = None,
    objective_weight: float = 0.6,
    prefill_results: dict[str, dict[str, Repeats]] | None = None,
    model_vram: dict[str, dict[str, float]] | None = None,
    answer_tags: bool = False,
    think: bool | None = None,
) -> dict[str, Any]:
    """Assemble the full result document."""
    perf_results = perf_results or {}
    cold_loads = cold_loads or {}
    objective_scores = objective_scores or {}
    prefill_results = prefill_results or {}
    model_vram = model_vram or {}

    categories = categories_of(tasks)
    by_category = group_by_category(tasks)
    task_keys = [t.key for t in tasks]
    weights = {t.key: t.weight for t in tasks}

    models = []
    for model_name, results in all_results.items():
        judge = judge_scores.get(model_name, {})
        objective = objective_scores.get(model_name, {})
        blended = {
            key: blend_repeats(objective.get(key, []), judge.get(key, []), objective_weight)
            for key in task_keys
        }

        per_task = {}
        for task in tasks:
            per_task[task.key] = {
                "category": task.category,
                "difficulty": task.difficulty,
                "weight": task.weight,
                "objective": task_score_stats(objective.get(task.key, [])),
                "judge": task_score_stats(judge.get(task.key, [])),
                "blended": task_score_stats(blended.get(task.key, [])),
                "performance": perf_summary(results.get(task.key, [])),
                # The raw records behind every number above.
                "repeats": results.get(task.key, []),
                "judge_verdicts": judge.get(task.key, []),
                "objective_results": objective.get(task.key, []),
            }

        models.append(
            {
                "name": model_name,
                "details": all_details.get(model_name, {}),
                "memory": model_vram.get(model_name, {}),
                "cold_load_seconds": cold_loads.get(model_name, 0.0),
                "throughput": perf_summary(perf_results.get(model_name, [])),
                "throughput_samples": perf_results.get(model_name, []),
                "prefill_scaling": {
                    length: perf_summary(samples)
                    for length, samples in prefill_results.get(model_name, {}).items()
                },
                # How many generations were scorable at all. A quality mean
                # cannot distinguish a wrong answer from an absent one; this can.
                "completeness": completeness(results),
                "overall": model_score_stats(blended, task_keys, weights),
                "by_category": {
                    category: model_score_stats(
                        blended, [t.key for t in by_category[category]], weights
                    )
                    for category in categories
                },
                "tasks": per_task,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "run": run_name,
        "host": {
            "cpu": get_cpu_info(),
            "ram": get_ram_info(),
            "gpus": get_gpu_info(),
        },
        # Everything a later run has to match before its numbers can be
        # compared with these. A run judged by a different model, or sampled
        # with a different context window, is measuring on a different ruler.
        "config": {
            "generation": gen_params or {},
            "repeats": repeats,
            "objective_weight": objective_weight,
            "judges": judge_models,
            # Both change the text models were asked to produce, so both change
            # what a score means.
            "answer_tags": answer_tags,
            "think": think,
        },
        "tasks": [
            {
                "id": t.id,
                "category": t.category,
                "difficulty": t.difficulty,
                "weight": t.weight,
                "prompt": t.prompt,
                "checks": [c.get("type") for c in t.checks],
            }
            for t in tasks
        ],
        "categories": categories,
        "models": models,
        "total_runtime_seconds": total_runtime,
    }


def write_json(run_dir: Path, document: dict[str, Any]) -> None:
    (run_dir / RESULTS_FILE).write_text(json.dumps(document, indent=2, default=str))


def load_json(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / RESULTS_FILE
    if not path.exists():
        return None
    document: dict[str, Any] = json.loads(path.read_text())
    return document
