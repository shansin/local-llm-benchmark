# Local LLM Benchmark

A tool for benchmarking locally-running Ollama models across multiple prompt categories, with automated LLM-as-judge scoring and detailed markdown output.

## How it works

1. Fetches available models from your local Ollama instance (embedding models are automatically excluded)
2. You select which models to benchmark and which model acts as judge
3. Each model is warmed up, then measured with a short fixed-length **throughput probe**
4. Each model answers every prompt **`QUALITY_REPEATS` times** at temperature 0 with a distinct seed per repeat
5. The judge model scores every response 1–10 against optional per-category evaluation criteria
6. Results are written to a timestamped directory under `output/`

### Reading the numbers

Sampling is pinned (temperature 0, fixed seeds, explicit `num_ctx`) so a run can be repeated
exactly; the parameters used are recorded in `results.md`.

Because each prompt is answered several times, the report carries a **noise floor** — the widest
within-task spread across repeats. *Differences in Avg Score smaller than the noise floor are ties,
not rankings.* Running with `--repeats 1` produces no noise estimate at all, and the report says so.

Throughput is measured by a dedicated probe (`PERF_PREDICT_TOKENS` tokens, repeated
`PERF_REPEATS` times, reported as median ± IQR) rather than inferred from the benchmark answers,
which would conflate speed with verbosity. Every probe sample uses a distinct prompt prefix —
re-sending identical text lets Ollama serve it from its prompt cache, which measures the cache
rather than the model.

**VRAM** comes from Ollama's own per-model accounting (`/api/ps`), not from a `nvidia-smi` delta:
on a machine keeping several models warm, a global delta attributes nothing. If part of a model
spills to system RAM the report shows it as `5180 (+2400 RAM)` — that is usually the explanation
for a model that is unexpectedly slow.

**Prefill scaling** measures prompt-ingestion speed at several input lengths, which is where
quantisation and KV-cache differences appear and where a single-length measurement shows nothing.

## Tasks

25 tasks across 5 categories, 5 per category, spread across easy/medium/hard so the set does not
saturate at the top:

| Category | Measures |
|----------|----------|
| `coding` | Implementing algorithms correctly — scored by running real tests against the generated code |
| `reasoning` | Multi-constraint logic puzzles with one correct answer |
| `knowledge` | Factual explanation under a length limit, with precise criteria for what counts as accurate |
| `instruction` | Following exact, sometimes adversarial, formatting and negative constraints |
| `writing` | Craft under constraint — dramatic irony, dialogue-only scenes, register control |

Tasks live in `tasks/<category>/<id>.toml`:

```toml
category = "coding"
difficulty = "medium"
prompt = """..."""
criteria = """..."""      # guides the judge

[[checks]]                # deterministic, judge-independent
type = "code_exec"
suite = "suites/merge_intervals_test.py"
```

Run `uv run benchmark.py validate` to check the set for problems.

### Scoring

Each response gets up to three numbers, all reported separately:

- **Objective** — from the task's `checks`. `code_exec` runs the benchmark's own pytest suite
  against the generated code (partial credit per test passed); other checks cover exact answers,
  word counts, valid JSON, and forbidden patterns.
- **Judge** — a panel of LLMs rating four named dimensions (accuracy, completeness,
  instruction following, clarity) from 1–10 against the task's criteria. Judging runs at
  temperature 0 with a fixed seed and a JSON schema, and the panel verdict is the **median**, so
  one model's taste cannot decide the leaderboard. **A model never scores its own answers** unless
  `ALLOW_SELF_JUDGE=1`; if that leaves no independent judge, the task is left unscored rather than
  scored by an interested party. Every individual vote is kept in `state.json`.
- **Blended** — `OBJECTIVE_WEIGHT` of the first plus the rest from the second. Tasks with no
  checks are judge-only.

Keeping both means the report can show a **judge calibration table**: how far the judge's opinion
sits from the measured result, per category. That number tells you how much to trust the judge on
the tasks where nothing can be measured.

The coding suites are the benchmark's tests, not the model's — a model cannot raise its own score
by writing weak tests.

**Code execution:** `code_exec` runs model-written code on your machine. It is sandboxed — fresh
temp dir, scrubbed environment, `python -I`, address-space/CPU/file-size/process rlimits, wall-clock
timeout enforced by killing the process group, and network dropped via an unprivileged namespace
where the kernel permits. That is real containment, not a guarantee. Set `CODE_EXEC=0` (or pass
`--no-code-exec`) to disable it entirely.

## Output structure

```
output/
└── 2026-04-06_17-02-25/
    ├── results.json                      # canonical data — everything, including raw per-repeat records
    ├── results.md                        # tables: performance, quality, calibration, per-task
    ├── report.html                       # self-contained visual report (no network needed)
    ├── state.json                        # resume checkpoint
    └── model-name/
        ├── aggregate_benchmark.md        # per-model summary + responses
        └── <task-id>.md                  # every repeat, with its seed, score, and metrics
```

`results.json` is the source of truth; the markdown and HTML are views of it. It keeps the raw
per-repeat records alongside the summary statistics, so any number in the report can be checked,
and old runs can be re-analysed without re-running anything.

### Comparing runs

```bash
uv run benchmark.py compare output/<before> output/<after>   # what changed
uv run benchmark.py compare --all                            # best score per model, all runs
```

A difference is only called a change when it exceeds the run's **measured noise floor**; everything
else is reported as unchanged. Runs predating `results.json` can be brought forward with `--resume`.

`results.md` contains a performance table and a quality table:

| Model | Params | Quant | Tokens/s | IQR | Prefill tok/s | TTFT p50 | TTFT p90 | Cold load (s) |
|-------|--------|-------|----------|-----|---------------|----------|----------|---------------|
| qwen3.5:27b | 27.8B | Q4_K_M | 38.9 | ±0.7 | 1161.4 | 0.25 | 0.41 | 6.2 |

| Model | Coding | … | Avg Score | Noise ± |
|-------|--------|---|-----------|---------|
| qwen3.5:27b | 9.7 ±0.6 | … | **9.20** | 0.45 |

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
| `JUDGE_MODELS` | *(interactive)* | Comma-separated judge panel; verdict is the median |
| `ALLOW_SELF_JUDGE` | `0` | Let a model score its own answers (off by default) |
| `PROMPT_TIMEOUT` | `2700` | Per-prompt request timeout (seconds) |
| `JUDGE_TIMEOUT` | `1800` | Per-judge request timeout (seconds) |
| `TEMPERATURE` | `0.0` | Sampling temperature |
| `TOP_P` / `TOP_K` | `1.0` / `40` | Sampling cutoffs |
| `SEED` | `0` | Base seed; repeat *i* uses `SEED + i` |
| `NUM_CTX` | `8192` | Context window — set explicitly, or Ollama silently truncates |
| `QUALITY_REPEATS` | `3` | Times each prompt is answered |
| `RETRIES` | `3` | Retries on connection errors (timeouts are never retried) |
| `PERF_PROBE` | `true` | Run the dedicated throughput probe |
| `PERF_REPEATS` | `5` | Probe samples per model |
| `PERF_PREDICT_TOKENS` | `256` | Tokens generated per probe sample |
| `PREFILL_SWEEP` | `512,4096,16384` | Input lengths for the prefill-scaling measurement |
| `PREFILL_REPEATS` | `2` | Samples per sweep point |
| `OBJECTIVE_WEIGHT` | `0.6` | Weight of deterministic checks in the blended score |
| `CODE_EXEC` | `1` | Set to `0` to never execute model-generated code |
| `TASKS_DIR` | `./tasks` | Task directory |

Prompt failures (timeout, connection error, HTTP error) are recorded as results rather than
aborting the run. Judge failures leave the category **unscored** (`—`) and it is excluded from
the model's average — a judge that fails to answer should not look like a model that scored badly.

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
uv run benchmark.py                    # full run
uv run benchmark.py --quick            # 1 repeat, no throughput probe — smoke test
uv run benchmark.py --repeats 5        # tighter noise estimate, 5x the cost
uv run benchmark.py --no-perf-probe    # skip throughput measurement only
uv run benchmark.py list-models        # what Ollama has available
uv run benchmark.py validate           # check the task set
uv run benchmark.py compare A B        # diff two runs
```

A full run costs roughly `models × prompts × QUALITY_REPEATS` generations plus
`models × PERF_REPEATS` short probes. `--quick` is about the cost of the old single-sample runs.

### Resuming a crashed or interrupted run

Each run writes a `state.json` checkpoint into its output folder after every
completed prompt and judge call. To resume, pass `--resume` with the run folder:

```bash
./start_benchmark.sh --resume ./output/2026-04-11_23-43-10
```

Already-completed prompts and judge scores are reused from the checkpoint; only
the remaining work runs. Total benchmark runtime is accumulated across sessions.

## Adding tasks

1. Add a `.toml` file to `tasks/<category>/` — the directory name becomes the category
   and the filename stem becomes the task id (both overridable with `category` / `id` keys).
2. Write the `prompt`, and `criteria` describing what a correct answer contains — the judge
   uses that as its primary scoring guide.
3. Add `[[checks]]` for anything verifiable, so the task does not rest entirely on the judge.
4. Run `uv run benchmark.py validate` to confirm the set still loads.

## Performance metrics

| Metric | Description |
|--------|-------------|
| **Tokens/s** | Generation throughput (output tokens per second) |
| **Prefill tok/s** | Prompt ingestion throughput (input tokens per second) |
| **TTFT** | Prompt evaluation time (seconds) — *excludes* model load |
| **Load** | Model load time (seconds), reported separately so a cold first prompt stays comparable |
| **Gen Time** | Total generation time (seconds) |
| **Output Tokens** | Number of tokens generated |
| **Score** | Judge score 1–10 per category; average shown in final column |

## Development

```bash
uv sync --all-groups
uv run pytest        # tests (no Ollama or GPU needed — the API is mocked)
uv run ruff check .
uv run mypy
```

The implementation lives in `llmbench/`; `benchmark.py` is a thin entry point.

| Module | Responsibility |
|--------|----------------|
| `config.py` | Environment-derived configuration |
| `models.py` | Ollama model discovery and sorting |
| `runner.py` | Running prompts, extracting timing metrics |
| `tasks.py` | Loading prompts and criteria |
| `state.py` | Resume checkpoints |
| `scoring/` | LLM judging and score aggregation |
| `telemetry.py` | GPU sampling and per-model memory accounting |
| `stats.py` | Median, percentile, and spread helpers |
| `report/` | JSON, markdown, HTML, and cross-run comparison |
| `sysinfo.py` | Host CPU/RAM/GPU detection |
| `cli.py` | Argument parsing and the run loop |
