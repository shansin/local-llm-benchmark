"""Markdown report writers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from llmbench.config import OBJECTIVE_WEIGHT, GenerationParams
from llmbench.scoring.aggregate import (
    Repeats,
    blend_repeats,
    combined_perf,
    completeness,
    model_score_stats,
    perf_summary,
    task_score_stats,
)
from llmbench.scoring.extract import extraction_for, looks_like_leaked_reasoning
from llmbench.stats import mean
from llmbench.sysinfo import get_cpu_info, get_gpu_info, get_ram_info
from llmbench.tasks import Task, categories_of, group_by_category
from llmbench.util import format_duration

Scores = dict[str, list[dict[str, Any]]]


def _table(columns: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(columns) + " |\n"
    head += "|" + "|".join("-" * (len(c) + 2) for c in columns) + "|\n"
    return head + "".join("| " + " | ".join(r) + " |\n" for r in rows)


def _fmt_mean(stats: dict[str, Any]) -> str:
    """Render a score as mean ± spread, or an em dash when never scored."""
    if stats["mean"] is None:
        return "—"
    if stats.get("std", 0.0) > 0:
        return f"{stats['mean']:.1f} ±{stats['std']:.1f}"
    return f"{stats['mean']:.1f}"


def _fmt_vram(usage: dict[str, float]) -> str:
    """Model memory, flagging any part that did not fit on the GPU."""
    vram = usage.get("vram_mib")
    if not vram:
        return "—"
    spilled = usage.get("offloaded_mib", 0.0)
    return f"{vram:.0f} (+{spilled:.0f} RAM)" if spilled > 1 else f"{vram:.0f}"


def _num_ctx_of(gen: GenerationParams | dict[str, Any] | None) -> int | None:
    if gen is None:
        return None
    values = gen if isinstance(gen, dict) else gen.to_options()
    ctx = values.get("num_ctx")
    return int(ctx) if ctx else None


def _params_line(gen: GenerationParams | dict[str, Any] | None) -> str:
    if gen is None:
        return ""
    values = gen if isinstance(gen, dict) else gen.to_options()
    rendered = ", ".join(f"{k}={v}" for k, v in values.items())
    return f"- **Generation params:** `{rendered}`\n"


def write_model_benchmark(
    model_dir: Path,
    model_info: dict[str, Any],
    results: dict[str, Repeats],
    tasks: list[Task],
    scores: Scores | None = None,
    perf: Repeats | None = None,
    cold_load: float | None = None,
) -> None:
    """Write the per-model aggregate report plus one file per task."""
    scores = scores or {}
    details = model_info.get("details", {})
    md = f"# Benchmark: {model_info['name']}\n\n"
    md += f"- **Parameters:** {details.get('parameter_size', '?')}\n"
    md += f"- **Quantization:** {details.get('quantization_level', '?')}\n"
    md += f"- **Family:** {details.get('family', '?')}\n"
    if cold_load is not None:
        md += f"- **Cold load:** {cold_load:.2f}s\n"
    md += "\n"

    if perf:
        p = perf_summary(perf)
        md += "## Throughput (fixed probe)\n\n"
        md += _table(
            ["Tokens/s", "IQR", "Prefill tok/s", "TTFT p50 (s)", "TTFT p90 (s)", "Samples"],
            [
                [
                    f"{p['tps_median']:.1f}",
                    f"{p['tps_iqr']:.1f}",
                    f"{p['prefill_median']:.1f}",
                    f"{p['ttft_p50']:.2f}",
                    f"{p['ttft_p90']:.2f}",
                    f"{p['samples']:.0f}",
                ]
            ],
        )
        md += "\n"

    md += "## Per-task measurements\n\n"
    rows = []
    for task in tasks:
        repeats = results.get(task.key, [])
        p = perf_summary(repeats)
        rows.append(
            [
                task.category,
                task.id,
                task.difficulty,
                f"{p['tps_median']:.1f}",
                f"{p['gen_time_median']:.2f}",
                f"{p['tokens_median']:.0f}",
                _fmt_mean(task_score_stats(scores.get(task.key, []))),
            ]
        )
    md += _table(
        [
            "Category",
            "Task",
            "Difficulty",
            "Tokens/s",
            "Gen Time (s)",
            "Tokens",
            "Score",
        ],
        rows,
    )

    md += "\n## Responses\n\n"
    for task in tasks:
        repeats = results.get(task.key, [])
        md += f"### {task.category} / {task.id}\n\n"
        truncated = f"{task.prompt[:200]}{'...' if len(task.prompt) > 200 else ''}"
        md += f"**Prompt:** {truncated}\n\n"
        if repeats:
            md += f"**Response:**\n\n{repeats[0]['response']}\n\n"
        md += "---\n\n"

    (model_dir / "aggregate_benchmark.md").write_text(md)

    for task in tasks:
        repeats = results.get(task.key, [])
        verdicts = scores.get(task.key, [])
        cat_md = f"# {task.category} / {task.id}\n\n"
        cat_md += f"**Prompt:** {task.prompt}\n\n"
        for i, r in enumerate(repeats):
            verdict = verdicts[i] if i < len(verdicts) else {}
            score = verdict.get("score")
            cat_md += f"## Repeat {i + 1} (seed {r.get('seed', '?')})\n\n"
            cat_md += f"- Score: {'—' if score is None else f'{score}/10'}"
            if verdict.get("reason"):
                cat_md += f" — {verdict['reason']}"
            cat_md += "\n"
            cat_md += f"- Tokens/s: {r['tokens_per_sec']:.1f}\n"
            cat_md += f"- Prefill tok/s: {r.get('prompt_eval_speed', 0.0):.1f}\n"
            cat_md += f"- TTFT: {r['ttft']:.2f}s\n"
            cat_md += f"- Gen Time: {r['total_time']:.2f}s\n"
            cat_md += f"- Output Tokens: {r['eval_count']}\n"
            cat_md += _answer_provenance(r)
            cat_md += "\n"
            extraction = extraction_for(r)
            if extraction.reasoning:
                # The reasoning is kept, folded away. It is what a model that
                # never reached an answer spent its context on, and reading it
                # is the fastest way to tell a stuck model from a slow one.
                cat_md += (
                    "<details><summary>Reasoning "
                    f"({len(extraction.reasoning)} chars, not scored)</summary>\n\n"
                    f"{extraction.reasoning}\n\n</details>\n\n"
                )
            cat_md += f"**Answer (as scored):**\n\n{extraction.answer or '_(none)_'}\n\n"
        (model_dir / f"{task.id}.md").write_text(cat_md)


def _answer_provenance(result: dict[str, Any]) -> str:
    """One line on how this response was read, when that is not obvious.

    Silent by default: on a model that simply answers the prompt there is
    nothing to explain, and a note on every response would train the reader to
    skip the ones that matter.
    """
    extraction = extraction_for(result)
    notes = []
    if result.get("discarded_reasoning"):
        notes.append(
            "**reasoning discarded** — tokens were decoded onto neither channel; "
            "the budget was spent before the answer began"
        )
    elif result.get("truncated") or not extraction.complete:
        notes.append("**truncated** — stopped at the context limit, not by choice")
    if extraction.source not in ("verbatim",):
        notes.append(f"answer read from `{extraction.source}`")
    if looks_like_leaked_reasoning(extraction):
        notes.append("**opens with planning prose** — checks are measuring the reasoning")
    return "".join(f"- {note}\n" for note in notes)


def _system_info_section() -> str:
    gpus = get_gpu_info()
    md = "## System Info\n\n"
    md += f"- **CPU:** {get_cpu_info() or 'Not detected'}\n"
    md += f"- **RAM:** {get_ram_info() or 'Not detected'}\n"
    md += f"- **GPUs:** {'; '.join(gpus) if gpus else 'Not detected'}\n"
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    md += f"- **Ollama GPUs:** CUDA_VISIBLE_DEVICES={cuda_vis}\n"
    return md


def _performance_section(
    all_results: dict[str, dict[str, Repeats]],
    all_details: dict[str, Any],
    perf_results: dict[str, Repeats],
    cold_loads: dict[str, float],
    task_keys: list[str],
    model_vram: dict[str, dict[str, float]],
) -> str:
    probe_used = any(perf_results.get(m) for m in all_results)
    md = f"## Performance ({'fixed probe' if probe_used else 'benchmark answers'})\n\n"
    rows = []
    for model_name, results in all_results.items():
        details = all_details.get(model_name, {})
        probe = perf_results.get(model_name)
        p = perf_summary(probe) if probe else combined_perf(results, task_keys)
        rows.append(
            [
                model_name,
                details.get("parameter_size", "?"),
                details.get("quantization_level", "?"),
                f"{p['tps_median']:.1f}",
                f"±{p['tps_iqr']:.1f}",
                f"{p['prefill_median']:.1f}",
                f"{p['ttft_p50']:.2f}",
                f"{p['ttft_p90']:.2f}",
                f"{cold_loads.get(model_name, 0.0):.1f}",
                _fmt_vram(model_vram.get(model_name, {})),
            ]
        )
    return md + _table(
        [
            "Model",
            "Params",
            "Quant",
            "Tokens/s",
            "IQR",
            "Prefill tok/s",
            "TTFT p50",
            "TTFT p90",
            "Cold load (s)",
            "VRAM (MiB)",
        ],
        rows,
    )


def _prefill_section(prefill_results: dict[str, dict[str, Repeats]]) -> str:
    """Prefill speed against input length.

    Prompt ingestion is where quantisation and KV-cache differences show up,
    and a single-length measurement hides it completely.
    """
    lengths = sorted(
        {int(length) for runs in prefill_results.values() for length in runs}, reverse=False
    )
    if not lengths:
        return ""

    md = "\n## Prefill scaling (tok/s by input length)\n\n"
    md += (
        "Measured input tokens are reported by Ollama, so these are actual lengths, not targets. "
        "Lengths beyond the configured `num_ctx` are skipped rather than silently truncated.\n\n"
    )
    rows = []
    for model_name, runs in prefill_results.items():
        cells = [model_name]
        for length in lengths:
            samples = runs.get(str(length), [])
            summary = perf_summary(samples)
            cells.append(f"{summary['prefill_median']:.0f}" if summary["samples"] else "—")
        rows.append(cells)
    return md + _table(["Model", *[f"~{n} tok" for n in lengths]], rows)


def _completeness_section(all_results: dict[str, dict[str, Repeats]], num_ctx: int | None) -> str:
    """How many generations produced a scorable answer at all.

    Without this, an absent answer and a wrong answer are the same number in
    the quality table. They are not the same finding: one is about the model,
    the other is usually about the context window this run was configured with.
    """
    stats = {model: completeness(results) for model, results in all_results.items()}
    if not any(
        s["truncated"] or s["empty"] or s["errors"] or s["leaked_reasoning"] for s in stats.values()
    ):
        return ""

    md = "\n## Answer completeness\n\n"
    md += (
        "Generations that produced nothing to score, and generations whose answer "
        "arrived buried in undelimited reasoning. Both distort scores in ways the "
        "quality table cannot show.\n\n"
    )
    rows = [
        [
            model,
            f"{s['total']:.0f}",
            f"{s['truncated']:.0f}",
            f"{s['discarded_reasoning']:.0f}",
            f"{s['empty']:.0f}",
            f"{s['leaked_reasoning']:.0f}",
            f"{s['errors']:.0f}",
        ]
        for model, s in stats.items()
    ]
    md += _table(
        [
            "Model",
            "Generations",
            "Truncated",
            "…discarded",
            "No answer",
            "Leaked reasoning",
            "Errors",
        ],
        rows,
    )

    if any(s["truncated"] for s in stats.values()):
        limit = f"`num_ctx={num_ctx}`" if num_ctx else "the configured context window"
        md += (
            f"\n**Truncated** generations hit {limit} and stopped mid-thought. They are "
            "scored as missing answers, because that is what they are — not as wrong "
            "ones. Raise `NUM_CTX` and re-run before reading those rows as quality.\n"
        )
    if any(s["discarded_reasoning"] for s in stats.values()):
        md += (
            "\n**…discarded** is the share of those truncations that raising `NUM_CTX` "
            "cannot fix. Tokens were decoded that arrived on neither the answer nor the "
            "`thinking` channel, so the budget was spent before the answer began. The "
            "preflight chooses a thinking mode that avoids this on a trivial prompt; a "
            "nonzero count here means the model behaved differently at task length, "
            "which is itself a finding about the model.\n"
        )
    if any(s["leaked_reasoning"] for s in stats.values()):
        md += (
            "\n**Leaked reasoning** counts answers that open with first-person planning "
            "prose despite the instruction to put the final answer inside "
            "`<answer></answer>` — the model ignored the delimiter it was asked for. "
            "Deterministic checks on those responses may be measuring the planning "
            "notes rather than the answer, so read that model's objective scores with "
            "suspicion.\n"
        )
    return md


def _excluded_section(excluded: dict[str, str]) -> str:
    """Models the preflight kept out of the run, with the evidence."""
    if not excluded:
        return ""
    md = "\n## Excluded models\n\n"
    md += (
        "These models produced no scorable answer to a trivial preflight prompt under "
        "any thinking mode, so benchmarking them would have measured the harness "
        "rather than the model. They have no leaderboard row.\n\n"
    )
    md += _table(["Model", "Preflight evidence"], [[m, note] for m, note in excluded.items()])
    return md


def write_results(
    run_dir: Path,
    all_results: dict[str, dict[str, Repeats]],
    all_details: dict[str, Any],
    judge_scores: dict[str, Scores],
    judge_model: str,
    tasks: list[Task],
    perf_results: dict[str, Repeats] | None = None,
    cold_loads: dict[str, float] | None = None,
    gen_params: GenerationParams | dict[str, Any] | None = None,
    total_runtime: float | None = None,
    objective_scores: dict[str, Scores] | None = None,
    prefill_results: dict[str, dict[str, Repeats]] | None = None,
    model_vram: dict[str, dict[str, float]] | None = None,
    excluded: dict[str, str] | None = None,
) -> None:
    """Write results.md: performance, quality by category, and a per-task breakdown."""
    perf_results = perf_results or {}
    cold_loads = cold_loads or {}
    objective_scores = objective_scores or {}
    categories = categories_of(tasks)
    by_category = group_by_category(tasks)
    task_keys = [t.key for t in tasks]
    weights = {t.key: t.weight for t in tasks}

    def blended_for(model_name: str) -> Scores:
        """Per-task blended verdicts for one model."""
        judge = judge_scores.get(model_name, {})
        objective = objective_scores.get(model_name, {})
        return {
            key: blend_repeats(objective.get(key, []), judge.get(key, []), OBJECTIVE_WEIGHT)
            for key in task_keys
        }

    md = f"# Benchmark Results — {run_dir.name}\n\n"
    md += _system_info_section()
    md += _params_line(gen_params)
    md += f"- **Tasks:** {len(tasks)} across {len(categories)} categories\n"
    if total_runtime is not None:
        md += f"- **Total Benchmark Runtime:** {format_duration(total_runtime)}\n"
    md += "\n"

    md += _performance_section(
        all_results, all_details, perf_results, cold_loads, task_keys, model_vram or {}
    )
    md += _prefill_section(prefill_results or {})
    md += _excluded_section(excluded or {})
    md += _completeness_section(all_results, _num_ctx_of(gen_params))

    # ---- Quality by category ----
    md += "\n## Quality\n\n"
    md += (
        f"Blended score: {OBJECTIVE_WEIGHT:.0%} deterministic checks, "
        f"{1 - OBJECTIVE_WEIGHT:.0%} judge ({judge_model}). "
        "Tasks with no checks are scored by the judge alone. Each prompt is sampled "
        "once at temperature 0, so treat small differences between models as ties.\n\n"
    )
    rows = []
    unscored_seen = False
    for model_name in all_results:
        scores = blended_for(model_name)
        cells = [model_name]
        for category in categories:
            keys = [t.key for t in by_category[category]]
            stats = model_score_stats(scores, keys, weights)
            if stats["mean"] is None:
                unscored_seen = True
                cells.append("—")
            else:
                cells.append(f"{stats['mean']:.1f}")
        overall = model_score_stats(scores, task_keys, weights)
        if overall["mean"] is None:
            cells.append("—")
        elif overall["n_scored"] < overall["n_total"]:
            cells.append(f"**{overall['mean']:.2f}**\\*")
        else:
            cells.append(f"**{overall['mean']:.2f}**")
        rows.append(cells)

    md += _table(["Model", *[c.title() for c in categories], "Avg Score"], rows)

    if unscored_seen:
        md += (
            "\n\\* — some tasks were never scored (judge timeout, error, or unparseable output). "
            "They are excluded from the average rather than counted as zero.\n"
        )

    # ---- Judge calibration ----
    model_names = list(all_results.keys())
    checked = [t for t in tasks if t.checks]
    if checked and any(objective_scores.get(m) for m in model_names):
        md += "\n## Judge calibration\n\n"
        md += (
            "On tasks with verifiable checks, how far the judge's score sits from the measured "
            "one. A large positive number means the judge is scoring answers higher than they "
            "deserve — the single most useful thing to know about an LLM-as-judge setup.\n\n"
        )
        cal_rows = []
        for model_name in model_names:
            judge = judge_scores.get(model_name, {})
            objective = objective_scores.get(model_name, {})
            cells = [model_name]
            for category in categories:
                keys = [t.key for t in by_category[category] if t.checks]
                deltas = []
                for key in keys:
                    j = task_score_stats(judge.get(key, []))["mean"]
                    o = task_score_stats(objective.get(key, []))["mean"]
                    if j is not None and o is not None:
                        deltas.append(j - o)
                cells.append(f"{mean(deltas):+.1f}" if deltas else "—")
            cal_rows.append(cells)
        md += _table(["Model", *[c.title() for c in categories]], cal_rows)

    # ---- Per-task breakdown ----
    md += "\n## Per-task scores\n\n"
    md += "Each cell is `blended (objective / judge)` averaged over repeats.\n\n"
    task_rows = []
    for task in tasks:
        row = [task.category, task.id, task.difficulty]
        for model_name in model_names:
            blended = task_score_stats(blended_for(model_name).get(task.key, []))
            cell = _fmt_mean(blended)
            if task.checks:
                obj = task_score_stats(objective_scores.get(model_name, {}).get(task.key, []))
                jud = task_score_stats(judge_scores.get(model_name, {}).get(task.key, []))
                obj_text = "—" if obj["mean"] is None else f"{obj['mean']:.1f}"
                jud_text = "—" if jud["mean"] is None else f"{jud['mean']:.1f}"
                cell += f" ({obj_text} / {jud_text})"
            row.append(cell)
        task_rows.append(row)
    md += _table(["Category", "Task", "Difficulty", *model_names], task_rows)

    # ---- Judge details ----
    md += "\n## Judge Details\n\n"
    for model_name, scores in judge_scores.items():
        md += f"### {model_name}\n\n"
        for task in tasks:
            verdicts = scores.get(task.key, [])
            stats = task_score_stats(verdicts)
            md += f"- **{task.category}/{task.id}:** {_fmt_mean(stats)}"
            reasons = [v.get("reason", "") for v in verdicts if v.get("reason")]
            if reasons:
                md += f" — {reasons[0]}"
            md += "\n"
            for entry in objective_scores.get(model_name, {}).get(task.key, [])[:1]:
                for result in entry.get("checks", []):
                    md += f"    - `{result['type']}`: {result['detail']}\n"
        md += "\n"

    (run_dir / "results.md").write_text(md)
