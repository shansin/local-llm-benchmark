#!/usr/bin/env python3
"""Local LLM Benchmark Tool — benchmarks Ollama models across prompt categories."""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", "./prompts"))
CRITERIA_DIR = Path(os.getenv("CRITERIA_DIR", "./prompts_criteria"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
PROMPT_TIMEOUT = int(os.getenv("PROMPT_TIMEOUT", "2700"))


def get_models():
    """Fetch available models from Ollama, excluding embedding models."""
    resp = requests.get(f"{BASE_URL}/api/tags")
    resp.raise_for_status()
    models = resp.json().get("models", [])
    # Filter out embedding models
    filtered = []
    for m in models:
        name = m["name"].lower()
        families = [f.lower() for f in (m.get("details", {}).get("families") or [])]
        if "embed" in name or any("embed" in f for f in families):
            continue
        filtered.append(m)
    def param_sort_key(m):
        size = m.get("details", {}).get("parameter_size", "0")
        match = re.match(r"([\d.]+)\s*([BKMGT]?)", str(size).upper())
        if not match:
            return 0
        val, unit = float(match.group(1)), match.group(2)
        return val * {"B": 1e9, "M": 1e6, "K": 1e3, "G": 1e9, "T": 1e12, "": 1}.get(unit, 1)

    return sorted(filtered, key=param_sort_key)


def get_model_details(model_name):
    """Fetch detailed model info."""
    resp = requests.post(f"{BASE_URL}/api/show", json={"name": model_name})
    resp.raise_for_status()
    return resp.json()


def select_models(models):
    """Let user select which models to benchmark."""
    env_models = os.getenv("BENCHMARK_MODELS", "").strip()
    if env_models:
        if env_models.lower() == "all":
            return models
        selected_names = [n.strip() for n in env_models.split(",")]
        selected = [m for m in models if m["name"] in selected_names]
        if selected:
            return selected
        print(f"Warning: none of the .env models found ({env_models}), falling back to interactive selection.")

    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        details = m.get("details", {})
        params = details.get("parameter_size", "?")
        quant = details.get("quantization_level", "?")
        print(f"  {i:2d}. {m['name']:<35s} ({params}, {quant})")

    print(f"\nEnter model numbers to benchmark (comma-separated), or 'all': ", end="")
    choice = input().strip()
    if choice.lower() == "all":
        return models
    indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
    return [models[i] for i in indices if 0 <= i < len(models)]


def select_judge(models):
    """Let user select which model to use as judge."""
    env_judge = os.getenv("JUDGE_MODEL", "").strip()
    if env_judge:
        for m in models:
            if m["name"] == env_judge:
                return m
        print(f"Warning: judge model '{env_judge}' not found, falling back to interactive selection.")

    print("\nSelect judge model (enter number): ", end="")
    choice = input().strip()
    idx = int(choice) - 1
    if 0 <= idx < len(models):
        return models[idx]
    print("Invalid selection, using first model as judge.")
    return models[0]


def load_txt_dir(directory, required=False):
    """Load all .txt files from a directory into a {stem: content} dict."""
    if not directory.exists():
        if required:
            print(f"Error: directory '{directory}' not found.")
            sys.exit(1)
        return {}
    files = {f.stem: f.read_text().strip() for f in sorted(directory.glob("*.txt"))}
    if required and not files:
        print(f"Error: no .txt files found in '{directory}'.")
        sys.exit(1)
    return files


def run_prompt(model_name, prompt_text):
    """Run a prompt against a model and return response + metrics."""
    try:
        resp = requests.post(
            f"{BASE_URL}/api/generate",
            json={"model": model_name, "prompt": prompt_text, "stream": False},
            timeout=PROMPT_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.ReadTimeout:
        print("timeout, skipping.")
        return {
            "response": "[TIMEOUT]",
            "tokens_per_sec": 0, "ttft": 0, "total_time": 0,
            "eval_count": 0, "prompt_eval_speed": 0, "prompt_eval_count": 0,
        }
    data = resp.json()

    eval_duration = data.get("eval_duration", 0)
    eval_count = data.get("eval_count", 0)
    load_duration = data.get("load_duration", 0)
    prompt_eval_duration = data.get("prompt_eval_duration", 0)
    total_duration = data.get("total_duration", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)

    tokens_per_sec = (eval_count / eval_duration * 1e9) if eval_duration > 0 else 0
    ttft = (load_duration + prompt_eval_duration) / 1e9
    total_time = total_duration / 1e9
    prompt_eval_speed = (prompt_eval_count / prompt_eval_duration * 1e9) if prompt_eval_duration > 0 else 0

    return {
        "response": data.get("response", ""),
        "tokens_per_sec": tokens_per_sec,
        "ttft": ttft,
        "total_time": total_time,
        "eval_count": eval_count,
        "prompt_eval_speed": prompt_eval_speed,
        "prompt_eval_count": prompt_eval_count,
    }


def judge_response(judge_model, category, prompt_text, response_text, expected_text=""):
    """Ask the judge model to score a response."""
    expected_section = ""
    eval_instruction = "Evaluate based on accuracy, completeness, clarity, and relevance."
    if expected_text:
        expected_section = f"\nExpected answer / evaluation criteria:\n{expected_text}\n\n"
        eval_instruction += " Use the expected answer and evaluation criteria above as your primary scoring guide."

    judge_prompt = f"""You are an expert evaluator. Score the following AI response on a scale of 1 to 10 (10 = excellent).

Category: {category}

Original prompt:
{prompt_text}
{expected_section}
AI response:
{response_text}

{eval_instruction} Respond with EXACTLY this format:
Score: <number>
Reason: <one line justification>"""

    resp = requests.post(
        f"{BASE_URL}/api/generate",
        json={"model": judge_model, "prompt": judge_prompt, "stream": False},
        timeout=600,
    )
    resp.raise_for_status()
    judge_text = resp.json().get("response", "")

    # Parse score
    match = re.search(r"Score:\s*(\d+)", judge_text)
    score = int(match.group(1)) if match else 0
    score = min(max(score, 1), 10)  # clamp to 1-10

    reason_match = re.search(r"Reason:\s*(.+)", judge_text)
    reason = reason_match.group(1).strip() if reason_match else judge_text.strip()[:100]

    return score, reason


def get_gpu_info():
    """Get system GPU info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []


def get_cpu_info():
    """Get CPU model name."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_ram_info():
    """Get total system RAM in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024 / 1024:.1f} GB"
    except OSError:
        pass
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return f"{int(result.stdout.strip()) / 1024 / 1024 / 1024:.1f} GB"
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def compute_perf_stats(results, categories):
    """Compute average performance stats across categories."""
    n = len(categories)
    return {
        "avg_tps": sum(results[c]["tokens_per_sec"] for c in categories) / n,
        "avg_ttft": sum(results[c]["ttft"] for c in categories) / n,
        "avg_time": sum(results[c]["total_time"] for c in categories) / n,
        "total_tokens": sum(results[c]["eval_count"] for c in categories),
    }


def write_model_benchmark(model_dir, model_info, results, prompts):
    """Write per-model benchmark.md."""
    details = model_info.get("details", {})
    md = f"# Benchmark: {model_info['name']}\n\n"
    md += f"- **Parameters:** {details.get('parameter_size', '?')}\n"
    md += f"- **Quantization:** {details.get('quantization_level', '?')}\n"
    md += f"- **Family:** {details.get('family', '?')}\n\n"

    # Metrics summary
    md += "## Performance Summary\n\n"
    md += "| Category | Tokens/s | TTFT (s) | Gen Time (s) | Output Tokens |\n"
    md += "|----------|----------|----------|--------------|---------------|\n"
    for category in prompts:
        r = results[category]
        md += f"| {category} | {r['tokens_per_sec']:.1f} | {r['ttft']:.2f} | {r['total_time']:.2f} | {r['eval_count']} |\n"

    # Averages
    stats = compute_perf_stats(results, list(prompts.keys()))
    n = len(prompts)
    md += f"| **Average** | **{stats['avg_tps']:.1f}** | **{stats['avg_ttft']:.2f}** | **{stats['avg_time']:.2f}** | **{stats['total_tokens'] / n:.0f}** |\n"

    # Prompt responses
    md += "\n## Responses\n\n"
    for category, prompt_text in prompts.items():
        r = results[category]
        md += f"### {category.title()}\n\n"
        md += f"**Prompt:** {prompt_text[:200]}{'...' if len(prompt_text) > 200 else ''}\n\n"
        md += f"**Response:**\n\n{r['response']}\n\n---\n\n"

    (model_dir / "aggregate_benchmark.md").write_text(md)

    # Write separate file per category
    for category, prompt_text in prompts.items():
        r = results[category]
        cat_md = f"# {category.title()}\n\n"
        cat_md += f"**Prompt:** {prompt_text}\n\n"
        cat_md += f"**Metrics:**\n"
        cat_md += f"- Tokens/s: {r['tokens_per_sec']:.1f}\n"
        cat_md += f"- TTFT: {r['ttft']:.2f}s\n"
        cat_md += f"- Gen Time: {r['total_time']:.2f}s\n"
        cat_md += f"- Output Tokens: {r['eval_count']}\n\n"
        cat_md += f"**Response:**\n\n{r['response']}\n"
        (model_dir / f"{category}.md").write_text(cat_md)


def format_duration(seconds):
    """Format seconds into a human-readable duration string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def write_results(run_dir, all_results, all_details, judge_scores, judge_model, prompts, total_runtime=None):
    """Write the final results.md with merged performance + quality table."""
    categories = list(prompts.keys())

    md = f"# Benchmark Results — {run_dir.name}\n\n"

    # System info
    gpus = get_gpu_info()
    cpu = get_cpu_info()
    ram = get_ram_info()
    md += "## System Info\n\n"
    md += f"- **CPU:** {cpu or 'Not detected'}\n"
    md += f"- **RAM:** {ram or 'Not detected'}\n"
    if gpus:
        md += f"- **GPUs:** {'; '.join(gpus)}\n"
    else:
        md += "- **GPUs:** Not detected\n"
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    md += f"- **Ollama GPUs:** CUDA_VISIBLE_DEVICES={cuda_vis}\n"
    if total_runtime is not None:
        md += f"- **Total Benchmark Runtime:** {format_duration(total_runtime)}\n"
    md += "\n"

    # Merged table
    md += f"## Results (Judge: {judge_model})\n\n"

    # Build header
    header = "| Model | Params | Quant | Tokens/s | TTFT (s) | Gen Time (s) | Tokens |"
    sep = "|-------|--------|-------|----------|----------|--------------|--------|"
    for cat in categories:
        header += f" {cat.title()} |"
        sep += "--------|"
    header += " Avg Score |"
    sep += "-----------|"
    md += header + "\n" + sep + "\n"

    for model_name, results in all_results.items():
        stats = compute_perf_stats(results, categories)
        details = all_details.get(model_name, {})
        params = details.get("parameter_size", "?")
        quant = details.get("quantization_level", "?")

        row = f"| {model_name} | {params} | {quant} | {stats['avg_tps']:.1f} | {stats['avg_ttft']:.2f} | {stats['avg_time']:.2f} | {stats['total_tokens']} |"
        scores = judge_scores.get(model_name, {})
        total_score = 0
        for cat in categories:
            score = scores.get(cat, {}).get("score", 0)
            total_score += score
            row += f" {score}/10 |"
        avg_score = total_score / len(categories) if categories else 0
        row += f" **{avg_score:.1f}** |"
        md += row + "\n"

    # Judge details
    md += "\n## Judge Details\n\n"
    for model_name, scores in judge_scores.items():
        md += f"### {model_name}\n\n"
        for cat in categories:
            info = scores.get(cat, {})
            md += f"- **{cat.title()}:** {info.get('score', 0)}/10 — {info.get('reason', 'N/A')}\n"
        md += "\n"

    (run_dir / "results.md").write_text(md)


def main():
    print("=" * 60)
    print("  Local LLM Benchmark Tool")
    print("=" * 60)

    # 1. Get models
    print("\nFetching models from Ollama...")
    models = get_models()
    if not models:
        print("No models found. Is Ollama running?")
        sys.exit(1)

    # 2. Select models to benchmark
    selected = select_models(models)
    if not selected:
        print("No models selected.")
        sys.exit(1)
    print(f"\nBenchmarking: {', '.join(m['name'] for m in selected)}")

    # 3. Select judge model
    judge = select_judge(models)
    print(f"Judge model: {judge['name']}")

    # 4. Load prompts and expected answers
    prompts = load_txt_dir(PROMPTS_DIR, required=True)
    expected = load_txt_dir(CRITERIA_DIR)
    categories = list(prompts.keys())
    print(f"Loaded {len(prompts)} prompt categories: {', '.join(categories)}")
    if expected:
        print(f"Loaded expected answers for: {', '.join(expected.keys())}")

    # 5. Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # 6. Benchmark each model
    bench_start = time.monotonic()
    all_results = {}
    all_details = {}
    for mi, model in enumerate(selected, 1):
        model_name = model["name"]
        print(f"\n{'─' * 50}")
        print(f"[{mi}/{len(selected)}] Benchmarking: {model_name}")
        print(f"{'─' * 50}")

        model_dir = run_dir / model_name.replace(":", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for ci, (category, prompt_text) in enumerate(prompts.items(), 1):
            print(f"  [{ci}/{len(prompts)}] {category}...", end=" ", flush=True)
            result = run_prompt(model_name, prompt_text)
            results[category] = result
            print(f"done ({result['tokens_per_sec']:.1f} tok/s, {result['total_time']:.1f}s)")

        all_results[model_name] = results
        all_details[model_name] = model.get("details", {})

        # Write per-model benchmark
        write_model_benchmark(model_dir, model, results, prompts)
        print(f"  Saved to {model_dir}/aggregate_benchmark.md")

    # 7. Judge scoring
    print(f"\n{'═' * 50}")
    print(f"  Judging responses with: {judge['name']}")
    print(f"{'═' * 50}")

    judge_scores = {}
    for model_name, results in all_results.items():
        print(f"\n  Judging: {model_name}")
        judge_scores[model_name] = {}
        for category in categories:
            print(f"    {category}...", end=" ", flush=True)
            score, reason = judge_response(
                judge["name"], category, prompts[category], results[category]["response"],
                expected.get(category, ""),
            )
            judge_scores[model_name][category] = {"score": score, "reason": reason}
            print(f"{score}/10")

    # 8. Write results
    total_runtime = time.monotonic() - bench_start
    write_results(run_dir, all_results, all_details, judge_scores, judge["name"], prompts, total_runtime)
    print(f"\n{'═' * 50}")
    print(f"  Results saved to: {run_dir / 'results.md'}")
    print(f"  Total runtime: {format_duration(total_runtime)}")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    main()
