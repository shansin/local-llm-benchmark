# Local LLM Benchmark

A tool for benchmarking locally-running Ollama models across multiple prompt categories, with automated LLM-as-judge scoring and detailed markdown output.

## How it works

1. Fetches available models from your local Ollama instance (embedding models are automatically excluded)
2. You select which models to benchmark and which model acts as judge
3. Each model is warmed up, then given a **preflight**: one trivial prompt under the real
   protocol, walking Ollama's thinking modes until one produces a scorable answer. That mode is
   used for the whole run. A model that answers under no mode is **excluded, loudly**, instead of
   being benchmarked into a row of zeros over several hours
4. Each model is measured with a short fixed-length **throughput probe**, then answers every
   prompt once at temperature 0
5. The judge model scores every response 1–10 against optional per-category evaluation criteria
6. Results are written to a timestamped directory under `output/`

### The protocol

Every model is measured the same way, and the protocol is fixed in code rather than configured:

- **Sampling is pinned** — temperature 0, fixed seed, explicit `num_ctx` — so a run can be
  repeated exactly; the parameters used are recorded in `results.md`.
- **Every prompt asks for the final answer inside `<answer></answer>`.** Models that emit
  reasoning as plain prose are otherwise unscoreable: every word-count and forbidden-pattern
  check measures the planning notes, and the model reads as far worse than it is. The tags make
  the split explicit; a model that ignores them is counted as **leaked reasoning** in the report.
- **The thinking mode is per-model, discovered by the preflight**, preferring the separate
  thinking channel (the deliberation is kept as evidence), then the model's own default, then no
  reasoning at all. Whatever was chosen is recorded per model in `results.json`.

Each prompt is sampled once: at temperature 0 decoding is greedy, so repeats come back
byte-identical and measure nothing. Treat small differences between models as ties.

Throughput is measured by a dedicated probe (256 tokens, repeated 5 times, reported as
median ± IQR) rather than inferred from the benchmark answers,
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

43 tasks across 9 categories, spread across easy/medium/hard so the set does not
saturate at the top:

| Category | Measures |
|----------|----------|
| `coding` | Implementing algorithms correctly, and repairing subtly broken ones — scored by running real tests against the generated code |
| `reasoning` | Multi-constraint logic puzzles with one correct answer |
| `knowledge` | Factual explanation under a length limit, with precise criteria for what counts as accurate |
| `instruction` | Following exact, sometimes adversarial, formatting and negative constraints |
| `writing` | Craft under constraint — dramatic irony, dialogue-only scenes, register control |
| `longcontext` | Aggregating and cross-referencing over documents of 15–36k characters |
| `statetrack` | Carrying a changing value correctly through 26–40 mechanical steps |
| `faithfulness` | Answering from the source given, abstaining where it is silent, and refusing a false premise |
| `transformation` | Converting 24–30 records exactly, without abbreviating or drifting |

The last four categories exist because the first five saturate. Canonical interview problems
and "explain TCP versus UDP" are answered well by every model in the 25–35B band, so they
measure little. These do not have memorised answers:

- **`longcontext`** documents are deliberately homogeneous — 400 identically-shaped ledger
  rows, 120 clauses in one house style. A needle written in a different voice can be found by
  style alone; when every line looks alike, the only way through is to read and aggregate.
- **`statetrack`** has no insight to have. Every step is trivial and the score measures only
  whether a model drifts.
- **`faithfulness`** includes sound premises alongside false ones, so blanket scepticism
  scores no better than blanket credulity, and questions whose answer is "not stated".
- **`transformation`** catches the failure no other check sees: fifteen correct rows followed
  by "... and the rest follow the same pattern".

Tasks whose answer key has to be computed — a ledger that must be summed, a machine that must
be run — are emitted by `tasks/generators/`, which writes the prompt and its `[[checks]]` from
the same code, so the key cannot drift from the question. Regenerate with
`uv run python tasks/generators/build.py`; the output is committed.

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
  one model's taste cannot decide the leaderboard. **A model never scores its own answers**; if
  that leaves no independent judge, the task is left unscored rather than scored by an
  interested party. Every individual vote is kept in `state.json`.
- **Blended** — 60% checks, 40% judge (fixed: many checks are necessary but not sufficient — a
  knowledge answer inside its word limit can still be wrong — so neither pure-objective nor
  pure-judge scoring would be correct, and a tunable blend makes runs incomparable). Tasks with
  no checks are judge-only.

Keeping both means the report can show a **judge calibration table**: how far the judge's opinion
sits from the measured result, per category. That number tells you how much to trust the judge on
the tasks where nothing can be measured.

The two halves are given the same text and a clean division of labour. Whatever the checks measured
is handed to the judge as settled fact — language models cannot count letters or words reliably,
and asking them to try produces confident nonsense, so the measurable half of a rubric is measured
and the judge scores only the half that cannot be.

### What gets scored

Not every response contains an answer, and the difference matters more than it looks:

- **Reasoning is separated from the answer** before anything is scored. `<think>` tags and Ollama's
  separate `thinking` channel are both recognised, and the reasoning is kept in the per-task report
  (folded away) rather than discarded.
- **A generation that ran out of context is not a wrong answer.** Truncated generations are
  detected, scored as missing, and counted in an **answer completeness** table with the `num_ctx`
  that caused them — not silently averaged in as near-zeros that look like a model being bad at
  puzzles.
- **Reasoning that reached neither channel is counted separately.** Asked for no thinking mode in
  particular, Ollama runs a hybrid-reasoning model in reasoning mode and streams none of it: the
  tokens are spent, the answer never starts, and both `response` and `thinking` come back empty.
  It looks identical to a context-wall truncation but `NUM_CTX` is not the remedy — the preflight
  exists to route each model around this, and the **…discarded** column names the cases where a
  model behaved differently at task length anyway.
- **An empty answer fails every check, including the negated ones.** "Must not contain the word
  *wolf*" is otherwise satisfied by silence, which used to hand a model that produced nothing a
  10/10 on the constraint half of a task.
- **Judges are never shown an empty response.** Asked to score nothing, they invent something to
  score; one reported that an empty response "contains many 'e' letters".
- **Undelimited reasoning is counted and named.** A model that answers with planning prose
  despite the answer-tag instruction gets its checks run over the planning prose — a 260-word
  scene counted at 4311 words. The report counts those responses as **leaked reasoning** instead
  of leaving a mysterious score gap.

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
    ├── results.json                      # canonical data — everything, including raw generation records
    ├── results.md                        # tables: performance, quality, calibration, per-task
    ├── report.html                       # self-contained visual report (no network needed)
    ├── state.json                        # resume checkpoint
    └── model-name/
        ├── aggregate_benchmark.md        # per-model summary + responses
        └── <task-id>.md                  # each response, with its seed, score, and metrics
```

`results.json` is the source of truth; the markdown and HTML are views of it. It keeps the raw
generation records alongside the summary statistics, so any number in the report can be checked,
and old runs can be re-analysed without re-running anything.

### Comparing runs

Two runs are only comparable if they were measured the same way. `compare` reports any difference
in the judge panel, sampling settings, or task set **before** it reports a single delta, and
`--all` warns when it is pooling runs scored by different judges — swapping the judge moves every
score in the table at once, in a direction no per-model reading can recover.


```bash
uv run benchmark.py compare output/<before> output/<after>   # what changed
uv run benchmark.py compare --all                            # best score per model, all runs
```

Each prompt is sampled once at temperature 0, so treat small deltas as sampling variation rather
than change. Runs predating `results.json` can be brought forward with `--resume`.

`results.md` contains a performance table and a quality table:

| Model | Params | Quant | Tokens/s | IQR | Prefill tok/s | TTFT p50 | TTFT p90 | Cold load (s) |
|-------|--------|-------|----------|-----|---------------|----------|----------|---------------|
| qwen3.5:27b | 27.8B | Q4_K_M | 38.9 | ±0.7 | 1161.4 | 0.25 | 0.41 | 6.2 |

| Model | Coding | … | Avg Score |
|-------|--------|---|-----------|
| qwen3.5:27b | 9.7 | … | **9.20** |

## Setup

**Prerequisites:** [Ollama](https://ollama.com) running locally, [uv](https://docs.astral.sh/uv/) installed.

```bash
# Install dependencies
uv sync
```

## Configuration

Copy `.env.example` to `.env` (or just set environment variables directly). The measurement
protocol — pinned sampling, answer tags, per-model thinking mode, probe and sweep settings —
is fixed in code; what remains configurable is what genuinely varies between machines and runs:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OUTPUT_DIR` | `./output` | Directory for benchmark results |
| `BENCHMARK_MODELS` | *(interactive)* | Comma-separated model names to benchmark, or `all` |
| `JUDGE_MODELS` | *(interactive)* | Comma-separated judge panel; verdict is the median. A model never scores its own answers |
| `PROMPT_TIMEOUT` | `2700` | Per-prompt request timeout (seconds) — size to your hardware |
| `NUM_CTX` | `32768` | Context window — a fact about your hardware, sized as large as the KV cache affords. Too small and reasoning models are cut off before they answer; the report's Truncated column says when to raise it |
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
uv run benchmark.py --quick            # skip throughput probe and prefill sweep — smoke test
uv run benchmark.py --no-code-exec     # coding tasks judged, never executed
uv run benchmark.py list-models        # what Ollama has available
uv run benchmark.py validate           # check the task set
uv run benchmark.py compare A B        # diff two runs
```

A full run costs one generation per model per task, plus a preflight and a handful of short
throughput probes per model.

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
3. Add `[[checks]]` for anything verifiable, so the task does not rest entirely on the judge:

   | Check | Verifies |
   |-------|----------|
   | `contains_all` / `contains_any` | Named substrings, case-insensitive; `contains_all` gives partial credit |
   | `regex` | A pattern is present, or with `negate` that it is absent |
   | `word_count` | `min` / `max` words |
   | `line_count` | `min` / `max` / `equals` non-blank lines, optionally only those matching `pattern` |
   | `match_count` | `min` / `max` / `equals` occurrences of `pattern` anywhere |
   | `json_valid` | The whole response is JSON — a code fence fails |
   | `json_path` | The value at `path` (e.g. `dept.ids[0]`) `equals` an expected one, within an optional `tolerance` |
   | `answer_equals` | The model's *stated* answer matches `expected` — reads the last `Answer:` line, or the closing lines; `numeric` compares as a number |
   | `code_exec` | A pytest `suite` run against the generated code, scored per test passed |

   Prefer several narrow checks to one broad one: each is weighted separately, so a
   six-of-seven answer scores as such rather than as a failure.
4. Run `uv run benchmark.py validate` to confirm the set still loads, and
   `uv run pytest tests/test_task_suite.py` to confirm a correct answer actually scores 10 —
   a task whose checks cannot be passed looks, in the report, exactly like a question every
   model failed.

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
| `config.py` | Environment-derived configuration and the fixed protocol values |
| `models.py` | Ollama model discovery and sorting |
| `preflight.py` | Per-model elicitation: choose a working thinking mode, or exclude the model |
| `runner.py` | Running prompts, extracting timing metrics |
| `tasks.py` | Loading prompts and criteria |
| `state.py` | Resume checkpoints |
| `scoring/` | LLM judging and score aggregation |
| `telemetry.py` | GPU sampling and per-model memory accounting |
| `stats.py` | Median, percentile, and spread helpers |
| `report/` | JSON, markdown, HTML, and cross-run comparison |
| `sysinfo.py` | Host CPU/RAM/GPU detection |
| `cli.py` | Argument parsing and the run loop |
