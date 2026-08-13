"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from llmbench.config import JUDGE_PARAMS, Config, GenerationParams
from llmbench.models import Model, get_models
from llmbench.report.compare import compare_documents, leaderboard, load_run, render_comparison
from llmbench.report.html import write_html
from llmbench.report.jsonout import build_document, write_json
from llmbench.report.markdown import write_model_benchmark, write_results
from llmbench.runner import run_perf_probe, run_prefill_sweep, run_prompt, warm_up
from llmbench.scoring.aggregate import perf_summary
from llmbench.scoring.judge import judge_response
from llmbench.scoring.objective import objective_score, run_checks
from llmbench.state import STATE_FILE, load_state, save_state
from llmbench.tasks import TaskError, categories_of, group_by_category, load_tasks
from llmbench.util import format_duration, safe_dirname

SUBCOMMANDS = ("run", "list-models", "validate", "compare")


def select_models(models: list[Model], config: Config) -> list[Model]:
    """Let the user select which models to benchmark."""
    if config.benchmark_models:
        if config.benchmark_models.lower() == "all":
            return models
        selected_names = [n.strip() for n in config.benchmark_models.split(",")]
        selected = [m for m in models if m["name"] in selected_names]
        if selected:
            missing = sorted(set(selected_names) - {m["name"] for m in selected})
            if missing:
                print(f"Warning: models not found, skipping: {', '.join(missing)}")
            return selected
        print(
            f"Warning: none of the .env models found ({config.benchmark_models}), "
            "falling back to interactive selection."
        )

    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        details = m.get("details", {})
        params = details.get("parameter_size", "?")
        quant = details.get("quantization_level", "?")
        print(f"  {i:2d}. {m['name']:<35s} ({params}, {quant})")

    print("\nEnter model numbers to benchmark (comma-separated), or 'all': ", end="")
    choice = input().strip()
    if choice.lower() == "all":
        return models
    indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
    return [models[i] for i in indices if 0 <= i < len(models)]


def select_judges(models: list[Model], config: Config) -> list[Model]:
    """Let the user select the judge panel. One judge is a panel of one."""
    if config.judge_model:
        wanted = [n.strip() for n in config.judge_model.split(",") if n.strip()]
        by_name = {m["name"]: m for m in models}
        chosen = [by_name[n] for n in wanted if n in by_name]
        missing = [n for n in wanted if n not in by_name]
        if missing:
            print(f"Warning: judge model(s) not found: {', '.join(missing)}")
        if chosen:
            return chosen
        print("Warning: no configured judge model found, falling back to interactive selection.")

    print("\nSelect judge model(s) (comma-separated numbers): ", end="")
    choice = input().strip()
    indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
    chosen = [models[i] for i in indices if 0 <= i < len(models)]
    if chosen:
        return chosen
    print("Invalid selection, using first model as judge.")
    return [models[0]]


def cmd_list_models(args: argparse.Namespace, config: Config) -> int:
    """Print the models Ollama has available, smallest first."""
    models = get_models(config.base_url)
    if not models:
        print("No models found. Is Ollama running?")
        return 1
    for m in models:
        details = m.get("details", {})
        print(
            f"{m['name']:<40s} {details.get('parameter_size', '?'):>8s} "
            f"{details.get('quantization_level', '?')}"
        )
    return 0


def cmd_validate(args: argparse.Namespace, config: Config) -> int:
    """Lint the task set: schema, unique ids, referenced files, criteria."""
    try:
        tasks = load_tasks(config.tasks_dir)
    except TaskError as exc:
        print(f"INVALID: {exc}")
        return 1

    print(f"Loaded {len(tasks)} tasks from {config.tasks_dir}\n")

    warnings = 0
    for category, group in group_by_category(tasks).items():
        difficulties = ", ".join(sorted({t.difficulty for t in group}))
        print(f"  {category:<14s} {len(group)} task(s)  [{difficulties}]")
        if len(group) < 3:
            print(f"    warning: {len(group)} task(s) is a thin sample for a category score")
            warnings += 1
        for task in group:
            if not task.criteria:
                print(f"    warning: {task.id} has no criteria; the judge scores it unguided")
                warnings += 1
            if not task.checks:
                print(f"    note:    {task.id} has no objective checks; judge-only scoring")

    print(
        f"\nOK — {len(tasks)} tasks, {len(categories_of(tasks))} categories, {warnings} warning(s)"
    )
    return 0


def cmd_compare(args: argparse.Namespace, config: Config) -> int:
    """Diff two runs, or build a leaderboard across all of them."""
    if args.all:
        runs = (
            sorted(p for p in config.output_dir.iterdir() if p.is_dir())
            if (config.output_dir.exists())
            else []
        )
        documents = [d for d in (load_run(p) for p in runs) if d]
        if not documents:
            print(
                f"No runs with results.json found in {config.output_dir}. "
                "Only runs from this version write one; re-run or --resume an older run "
                "to generate it."
            )
            return 1
        print(leaderboard(documents))
        return 0

    if not args.runs or len(args.runs) != 2:
        print("Usage: benchmark.py compare <before_run> <after_run>   (or --all)")
        return 1

    before, after = (load_run(p) for p in args.runs)
    for path, document in zip(args.runs, (before, after), strict=True):
        if document is None:
            print(f"Error: no results.json in '{path}' (older runs predate it).")
            return 1

    assert before is not None and after is not None
    print(render_comparison(compare_documents(before, after)))
    return 0


def cmd_run(args: argparse.Namespace, config: Config) -> int:  # noqa: PLR0915
    print("=" * 60)
    print("  Local LLM Benchmark Tool")
    print("=" * 60)

    try:
        tasks = load_tasks(config.tasks_dir)
    except TaskError as exc:
        print(f"Error: {exc}")
        return 1
    task_keys = [t.key for t in tasks]
    categories = categories_of(tasks)

    perf_results: dict[str, Any] = {}
    cold_loads: dict[str, float] = {}
    objective_scores: dict[str, Any] = {}
    prefill_results: dict[str, Any] = {}
    model_vram: dict[str, Any] = {}

    if args.resume:
        run_dir = args.resume
        if not run_dir.exists():
            print(f"Error: resume folder '{run_dir}' not found.")
            return 1
        state = load_state(run_dir)
        if state is None:
            print(f"Error: no {STATE_FILE} in '{run_dir}'.")
            return 1
        print(f"\nResuming from: {run_dir}")
        selected = state["selected"]
        # Pre-panel checkpoints stored a single "judge" object.
        judges = state.get("judges") or [state["judge"]]
        all_results = state.get("all_results", {})
        all_details = state.get("all_details", {})
        judge_scores = state.get("judge_scores", {})
        perf_results = state.get("perf_results", {})
        cold_loads = state.get("cold_loads", {})
        objective_scores = state.get("objective_scores", {})
        prefill_results = state.get("prefill_results", {})
        model_vram = state.get("model_vram", {})
        elapsed_prior = state.get("elapsed_seconds", 0.0)
        if state.get("task_keys") and state["task_keys"] != task_keys:
            print("Warning: the task set changed since the original run.")
            added = sorted(set(task_keys) - set(state["task_keys"]))
            removed = sorted(set(state["task_keys"]) - set(task_keys))
            if added:
                print(f"  Added:   {', '.join(added)}")
            if removed:
                print(f"  Removed: {', '.join(removed)} (their results stay in the checkpoint)")
        # The seeds and sampling settings of the original run govern the resumed
        # one; mixing settings inside a single run would make it uninterpretable.
        if state.get("repeats"):
            config = replace(config, quality_repeats=state["repeats"])
        if state.get("gen_params"):
            config = replace(config, gen=GenerationParams(**state["gen_params"]))
    else:
        print("\nFetching models from Ollama...")
        models = get_models(config.base_url)
        if not models:
            print("No models found. Is Ollama running?")
            return 1

        selected_models = select_models(models, config)
        if not selected_models:
            print("No models selected.")
            return 1
        selected = [{"name": m["name"], "details": m.get("details", {})} for m in selected_models]
        print(f"\nBenchmarking: {', '.join(m['name'] for m in selected)}")

        judge_models = select_judges(models, config)
        judges = [{"name": m["name"], "details": m.get("details", {})} for m in judge_models]
        print(f"Judge panel: {', '.join(j['name'] for j in judges)}")
        overlap = {j["name"] for j in judges} & {m["name"] for m in selected}
        if overlap and not config.allow_self_judge:
            print(
                f"Note: {', '.join(sorted(overlap))} both compete and judge. "
                "Their votes on their own answers are excluded."
            )
        elif overlap:
            print(
                f"Warning: ALLOW_SELF_JUDGE is on, so {', '.join(sorted(overlap))} "
                "will score their own answers. Those scores are not independent."
            )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = config.output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}
        all_details = {}
        judge_scores = {}
        elapsed_prior = 0.0

    repeats = config.quality_repeats
    print(f"Loaded {len(tasks)} tasks across {len(categories)} categories: {', '.join(categories)}")
    print(f"Sampling: {repeats} repeat(s) per prompt, {config.gen.to_options()}")
    if repeats < 2:
        print(
            "Note: with 1 repeat there is no noise estimate; gaps between models are not "
            "distinguishable from sampling variation."
        )

    session_start = time.monotonic()

    def current_elapsed() -> float:
        return elapsed_prior + (time.monotonic() - session_start)

    def checkpoint() -> None:
        save_state(
            run_dir,
            {
                "selected": selected,
                "judges": judges,
                "judge": judges[0],  # kept so older tooling can still read the file
                "allow_self_judge": config.allow_self_judge,
                "categories": categories,
                "task_keys": task_keys,
                "gen_params": config.gen.to_options(),
                "repeats": repeats,
                "all_results": all_results,
                "all_details": all_details,
                "judge_scores": judge_scores,
                "objective_scores": objective_scores,
                "objective_weight": config.objective_weight,
                "perf_results": perf_results,
                "prefill_results": prefill_results,
                "model_vram": model_vram,
                "cold_loads": cold_loads,
                "elapsed_seconds": current_elapsed(),
            },
        )

    # ---- Generation ----
    for mi, model in enumerate(selected, 1):
        model_name = model["name"]
        model_dir = run_dir / safe_dirname(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        results = all_results.setdefault(model_name, {})
        all_details[model_name] = model.get("details", {})

        complete = all(len(results.get(k, [])) >= repeats for k in task_keys)
        probe_done = not config.perf_probe or (
            len(perf_results.get(model_name, [])) > 0
            and (not config.prefill_sweep or model_name in prefill_results)
        )
        if complete and probe_done:
            print(f"\n[{mi}/{len(selected)}] {model_name}: already complete, skipping.")
            continue

        print(f"\n{'─' * 50}")
        print(f"[{mi}/{len(selected)}] Benchmarking: {model_name}")
        print(f"{'─' * 50}")

        if model_name not in cold_loads:
            print("  warming up...", end=" ", flush=True)
            load_seconds, gpu = warm_up(
                config.base_url, model_name, config.prompt_timeout, config.gen
            )
            cold_loads[model_name] = load_seconds
            model_vram[model_name] = gpu
            footprint = gpu.get("vram_mib", 0.0)
            note = f", {footprint:.0f} MiB VRAM" if footprint else ""
            if gpu.get("offloaded_mib"):
                note += f" (+{gpu['offloaded_mib']:.0f} MiB spilled to RAM)"
            print(f"loaded in {load_seconds:.1f}s{note}")
            checkpoint()

        if config.perf_probe and not probe_done:
            print(f"  throughput probe ({config.perf_repeats}x)...", end=" ", flush=True)
            perf_results[model_name] = run_perf_probe(
                config.base_url,
                model_name,
                config.prompt_timeout,
                config.gen,
                config.perf_repeats,
                config.perf_predict_tokens,
            )
            summary = perf_summary(perf_results[model_name])
            print(f"{summary['tps_median']:.1f} tok/s (IQR ±{summary['tps_iqr']:.1f})")
            checkpoint()

            if config.prefill_sweep and model_name not in prefill_results:
                lengths = [n for n in config.prefill_sweep if n <= config.gen.num_ctx - 64]
                skipped = [n for n in config.prefill_sweep if n not in lengths]
                print(f"  prefill sweep at {lengths} tokens...", end=" ", flush=True)
                if skipped:
                    print(f"(skipping {skipped}: beyond NUM_CTX={config.gen.num_ctx})", end=" ")
                prefill_results[model_name] = run_prefill_sweep(
                    config.base_url,
                    model_name,
                    config.prompt_timeout,
                    config.gen,
                    lengths,
                    config.prefill_repeats,
                )
                speeds = [
                    f"{perf_summary(runs)['prefill_median']:.0f}"
                    for runs in prefill_results[model_name].values()
                ]
                print(f"{' / '.join(speeds)} tok/s")
                checkpoint()

        for ti, task in enumerate(tasks, 1):
            done = results.setdefault(task.key, [])
            for rep in range(len(done), repeats):
                label = f"  [{ti}/{len(tasks)}] {task.category}/{task.id} [{rep + 1}/{repeats}]..."
                print(label, end=" ", flush=True)
                result = run_prompt(
                    config.base_url,
                    model_name,
                    task.prompt,
                    config.prompt_timeout,
                    config.gen.for_repeat(rep),
                    retries=config.retries,
                )
                done.append(result)
                if result["error"] is None:
                    print(
                        f"done ({result['tokens_per_sec']:.1f} tok/s, {result['total_time']:.1f}s)"
                    )
                checkpoint()

        write_model_benchmark(
            model_dir,
            model,
            results,
            tasks,
            judge_scores.get(model_name, {}),
            perf_results.get(model_name),
            cold_loads.get(model_name),
        )
        print(f"  Saved to {model_dir}/aggregate_benchmark.md")
        checkpoint()

    # ---- Objective checks ----
    checked_tasks = [t for t in tasks if t.checks]
    if checked_tasks:
        print(f"\n{'═' * 50}")
        print(f"  Objective checks on {len(checked_tasks)} task(s)")
        if not config.code_exec:
            print("  Code execution disabled (CODE_EXEC=0); those checks are skipped.")
        print(f"{'═' * 50}")

        for model_name, results in all_results.items():
            model_checks = objective_scores.setdefault(model_name, {})
            pending = [
                t
                for t in checked_tasks
                if len(model_checks.get(t.key, [])) < len(results.get(t.key, []))
            ]
            if not pending:
                print(f"\n  Checking: {model_name} — already complete, skipping.")
                continue
            print(f"\n  Checking: {model_name}")
            for task in checked_tasks:
                answers = results.get(task.key, [])
                done = model_checks.setdefault(task.key, [])
                for rep in range(len(done), len(answers)):
                    print(
                        f"    {task.category}/{task.id} [{rep + 1}/{len(answers)}]...",
                        end=" ",
                        flush=True,
                    )
                    checks = run_checks(task, answers[rep]["response"], config.code_exec)
                    score = objective_score(checks)
                    done.append({"score": score, "checks": checks})
                    print("—" if score is None else f"{score:.1f}/10")
                    checkpoint()

    # ---- Judging ----
    judge_names = [j["name"] for j in judges]
    print(f"\n{'═' * 50}")
    print(f"  Judging responses with: {', '.join(judge_names)}")
    if len(judge_names) > 1:
        print("  Panel verdict is the median across judges.")
    print(f"{'═' * 50}")

    for model_name, results in all_results.items():
        model_scores = judge_scores.setdefault(model_name, {})
        if all(len(model_scores.get(k, [])) >= len(results.get(k, [])) for k in task_keys):
            print(f"\n  Judging: {model_name} — already complete, skipping.")
            continue
        print(f"\n  Judging: {model_name}")
        for task in tasks:
            answers = results.get(task.key, [])
            verdicts = model_scores.setdefault(task.key, [])
            for rep in range(len(verdicts), len(answers)):
                print(
                    f"    {task.category}/{task.id} [{rep + 1}/{len(answers)}]...",
                    end=" ",
                    flush=True,
                )
                verdict = judge_response(
                    config.base_url,
                    judge_names,
                    task.category,
                    task.prompt,
                    answers[rep]["response"],
                    task.criteria,
                    timeout=config.judge_timeout,
                    params=JUDGE_PARAMS,
                    model_under_test=model_name,
                    allow_self_judge=config.allow_self_judge,
                )
                verdicts.append(verdict)
                score = verdict["score"]
                print(f"{score:.1f}/10" if score is not None else "unscored")
                checkpoint()

    # Judge scores are written into the per-model reports, so refresh them.
    for model in selected:
        model_name = model["name"]
        write_model_benchmark(
            run_dir / safe_dirname(model_name),
            model,
            all_results.get(model_name, {}),
            tasks,
            judge_scores.get(model_name, {}),
            perf_results.get(model_name),
            cold_loads.get(model_name),
        )

    total_runtime = current_elapsed()
    write_results(
        run_dir,
        all_results,
        all_details,
        judge_scores,
        ", ".join(judge_names),
        tasks,
        perf_results,
        cold_loads,
        config.gen,
        repeats,
        total_runtime,
        objective_scores,
        config.objective_weight,
        prefill_results,
        model_vram,
    )
    # results.json is the canonical record; markdown and HTML are views of it.
    document = build_document(
        run_dir.name,
        all_results,
        all_details,
        judge_scores,
        judge_names,
        tasks,
        perf_results,
        cold_loads,
        config.gen.to_options(),
        repeats,
        total_runtime,
        objective_scores,
        config.objective_weight,
        prefill_results,
        model_vram,
    )
    write_json(run_dir, document)
    write_html(run_dir, document)

    checkpoint()
    print(f"\n{'═' * 50}")
    print(f"  Results saved to: {run_dir / 'results.md'}")
    print(f"  Data:             {run_dir / 'results.json'}")
    print(f"  Report:           {run_dir / 'report.html'}")
    print(f"  Total runtime: {format_duration(total_runtime)}")
    print(f"{'═' * 50}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark", description="Local LLM Benchmark Tool")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run the benchmark (default)")
    run.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a previous run from an output folder (e.g. ./output/2026-04-11_23-43-10)",
    )
    run.add_argument(
        "--repeats", type=int, default=None, help="Times to answer each prompt (default 3)"
    )
    run.add_argument(
        "--no-perf-probe", action="store_true", help="Skip the dedicated throughput probe"
    )
    run.add_argument(
        "--quick",
        action="store_true",
        help="1 repeat, no throughput probe — fastest way to smoke-test a change",
    )
    run.add_argument(
        "--no-code-exec",
        action="store_true",
        help="Do not execute model-generated code; coding tasks fall back to judge-only scoring",
    )
    run.set_defaults(func=cmd_run)

    listing = sub.add_parser("list-models", help="List models available from Ollama")
    listing.set_defaults(func=cmd_list_models)

    validate = sub.add_parser("validate", help="Check the task set for problems")
    validate.set_defaults(func=cmd_validate)

    compare = sub.add_parser("compare", help="Diff two runs, or rank across all of them")
    compare.add_argument("runs", nargs="*", type=Path, help="Two run directories: before, after")
    compare.add_argument(
        "--all", action="store_true", help="Best score per model across every run in OUTPUT_DIR"
    )
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `benchmark.py --resume X` predates subcommands; keep it working.
    if not argv or argv[0] not in SUBCOMMANDS + ("-h", "--help"):
        argv.insert(0, "run")

    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace, Config], int] = getattr(args, "func", cmd_run)
    return func(args, Config.from_env().apply_cli(args))
