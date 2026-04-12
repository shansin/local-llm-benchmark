# Local LLM Benchmark

A tool for benchmarking locally-running Ollama models across multiple prompt categories, with automated LLM-as-judge scoring and detailed markdown output.

## How it works

1. Fetches available models from your local Ollama instance (embedding models are automatically excluded)
2. You select which models to benchmark and which model acts as judge
3. Each model runs all prompts; response time, token throughput, and output are recorded
4. The judge model scores every response 1–10 against optional per-category evaluation criteria
5. Results are written to a timestamped directory under `output/`

## Prompt categories

| Category | Description |
|----------|-------------|
| `coding` | Implement a Python algorithm with type hints, edge cases, and tests |
| `reasoning` | Solve a multi-constraint logic puzzle step by step |
| `knowledge` | Explain a factual topic (nuclear fission vs. fusion) concisely |
| `instruction` | Follow a multi-step task with math, number conversion, and ASCII art |
| `writing` | Write a short story with specific narrative constraints |

Prompts live in `prompts/*.txt`. Evaluation criteria (used to guide the judge) live in `prompts_criteria/*.txt`. Both directories use the filename stem as the category name.

## Output structure

```
output/
└── 2026-04-06_17-02-25/
    ├── results.md                        # combined leaderboard + judge details
    └── model-name/
        ├── aggregate_benchmark.md        # per-model perf summary + all responses
        ├── coding.md
        ├── reasoning.md
        ├── knowledge.md
        ├── instruction.md
        └── writing.md
```

`results.md` contains a merged table of performance metrics and judge scores:

| Model | Params | Quant | Tokens/s | TTFT (s) | Gen Time (s) | Tokens | Coding | … | Avg Score |
|-------|--------|-------|----------|----------|--------------|--------|--------|---|-----------|
| qwen3.5:27b | 27.8B | Q4_K_M | 27.0 | 1.64 | 351.45 | 45602 | 10/10 | … | **9.2** |

## Setup

**Prerequisites:** [Ollama](https://ollama.com) running locally, [uv](https://docs.astral.sh/uv/) installed.

```bash
# Install dependencies
uv sync
```

## Configuration

Copy `.env.example` to `.env` (or just set environment variables directly):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OUTPUT_DIR` | `./output` | Directory for benchmark results |
| `BENCHMARK_MODELS` | *(interactive)* | Comma-separated model names to benchmark, or `all` |
| `JUDGE_MODEL` | *(interactive)* | Model name to use as judge |
| `PROMPT_TIMEOUT` | `2700` | Per-prompt request timeout (seconds) |
| `JUDGE_TIMEOUT` | `1800` | Per-judge request timeout (seconds); timeouts/errors record a 0 score |

`PROMPTS_DIR` and `CRITERIA_DIR` are fixed to `./prompts` and `./prompts_criteria` respectively.

**Example `.env`:**
```env
OLLAMA_BASE_URL=http://localhost:11434
BENCHMARK_MODELS=qwen3.5:27b,gemma4:26b
JUDGE_MODEL=qwen3.5:27b
```

If `BENCHMARK_MODELS` or `JUDGE_MODEL` are not set (or the named models aren't found), the tool falls back to an interactive selection prompt.

## Running

```bash
./start_benchmark.sh
```

Or directly:

```bash
uv run benchmark.py
```

### Resuming a crashed or interrupted run

Each run writes a `state.json` checkpoint into its output folder after every
completed prompt and judge call. To resume, pass `--resume` with the run folder:

```bash
./start_benchmark.sh --resume ./output/2026-04-11_23-43-10
```

Already-completed prompts and judge scores are reused from the checkpoint; only
the remaining work runs. Total benchmark runtime is accumulated across sessions.

## Adding prompts

1. Add a `.txt` file to `prompts/` — the filename stem becomes the category name.
2. Optionally add a matching `.txt` file to `prompts_criteria/` with the expected answer or scoring rubric. The judge model uses this as its primary scoring guide.

## Performance metrics

| Metric | Description |
|--------|-------------|
| **Tokens/s** | Generation throughput (output tokens per second) |
| **TTFT** | Time to first token — load time + prompt evaluation time (seconds) |
| **Gen Time** | Total generation time (seconds) |
| **Output Tokens** | Number of tokens generated |
| **Score** | Judge score 1–10 per category; average shown in final column |
