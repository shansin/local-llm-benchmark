"""Configuration, resolved from environment variables and an optional .env file.

Deliberately small. Every knob this file once exposed was added to work around
a failure observed in some run, and each one became a way to misconfigure the
next run. The measurement protocol — sampling pinned at temperature 0, answer
tags on every prompt, thinking mode discovered per model by the preflight — is
now fixed in code. What remains configurable is what genuinely varies between
machines and runs: where Ollama is, which models to measure, how much context
the hardware affords, and how long a slow machine is given per prompt.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_PROMPT_TIMEOUT = 2700
DEFAULT_JUDGE_TIMEOUT = 1800

# Weight of a task's deterministic checks in its blended score; the judge
# supplies the rest. Fixed rather than configurable: many checked tasks carry
# checks that are necessary but not sufficient (a knowledge answer inside its
# word limit can still be wrong), so neither pure-objective nor pure-judge
# scoring is correct for them, and a tunable blend makes runs incomparable.
OBJECTIVE_WEIGHT = 0.6

# The prompt used to measure throughput independently of how long a model
# happens to make its answers. Padded to a target length at runtime.
PERF_PROBE_PROMPT = (
    "Write a detailed technical explanation of how a CPU cache hierarchy works. "
    "Cover L1, L2, and L3 caches, cache lines, associativity, and eviction policies."
)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Error: {name} must be an int, got {raw!r}.")
        sys.exit(1)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class GenerationParams:
    """Sampling parameters sent to Ollama.

    Ollama's defaults sample at temperature 0.8 with no seed, which makes runs
    unreproducible and leaderboard gaps unreadable. Everything is pinned and
    recorded in the results so a run can be repeated exactly. Only `num_ctx`
    is configurable (NUM_CTX): it is a fact about the hardware, not a tuning
    choice, and left unset Ollama picks a per-model default and silently
    truncates anything longer.

    The `num_ctx` default is generous because the alternative is worse. At 8192
    a reasoning model given a hard puzzle spends the whole window thinking and
    the generation is cut off before it ever writes an answer — measured as a
    zero, indistinguishable in the report from a model that answered wrongly.
    Three of five models in one run scored 0.4 on two tasks for exactly this
    reason. A window big enough that models stop for their own reasons costs
    KV-cache memory and runtime; a window that decides the result costs the
    benchmark its meaning.

    `num_predict` caps a single generation. Uncapped (-1) is a known hazard: one
    run produced a 170k-token generation that held the GPU in a single kernel
    long enough for the driver watchdog to reset it, killing the server
    mid-stream. A capped value turns a looping model into a truncated answer —
    a scored finding — instead of a lost sample. Left at -1 anyway (current
    choice) because a truncation floor also zeroes out slow-but-genuine
    convergers on hard tasks; if the watchdog reset recurs, drop this back to a
    finite cap (12,288 previously cleared the longest legitimate answer
    measured, 8,264 tokens, while staying under the 28,625-token runaway that
    triggered the reset).
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    seed: int = 0
    num_ctx: int = 65536
    num_predict: int = -1  # -1 = until the model stops

    def to_options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }

    def for_repeat(self, index: int) -> GenerationParams:
        """Params for sample `index`, with a distinct but deterministic seed."""
        return replace(self, seed=self.seed + index)


# Judging is a measurement instrument; it must not itself be a source of noise.
# The window has to hold the task prompt, the criteria, and the whole answer
# being judged. Too small and the judge silently scores a truncated view of a
# response — an instrument reading the wrong end of the ruler.
JUDGE_PARAMS = GenerationParams(temperature=0.0, top_p=1.0, seed=0, num_ctx=65536)


@dataclass(frozen=True)
class Config:
    """Everything the benchmark reads from the environment.

    Fields below the environment block are fixed protocol values: they exist as
    fields so tests can exercise other settings, not as user-facing knobs.
    """

    base_url: str = DEFAULT_BASE_URL
    tasks_dir: Path = Path("./tasks")
    output_dir: Path = Path("./output")
    prompt_timeout: int = DEFAULT_PROMPT_TIMEOUT
    # Comma-separated model names; empty means "ask interactively".
    benchmark_models: str = ""
    # Comma-separated judge panel; empty means "ask interactively". A model
    # never scores its own answers: with a panel the others cover for it, and a
    # task whose only judge is the model under test goes unscored rather than
    # scored by an interested party.
    judge_model: str = ""
    gen: GenerationParams = GenerationParams()
    code_exec: bool = True

    # --- Fixed protocol, not read from the environment ---
    judge_timeout: int = DEFAULT_JUDGE_TIMEOUT
    # Throughput is measured by a separate short probe rather than inferred from
    # however long each answer happened to be.
    perf_probe: bool = True
    perf_repeats: int = 5
    perf_predict_tokens: int = 256
    # Input lengths for the prefill sweep. Lengths beyond num_ctx are skipped
    # rather than silently truncated.
    prefill_sweep: tuple[int, ...] = (512, 4096, 16384)
    prefill_repeats: int = 2
    retries: int = 3

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()
        return cls(
            base_url=_env_str("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            tasks_dir=Path(_env_str("TASKS_DIR", "./tasks")),
            output_dir=Path(_env_str("OUTPUT_DIR", "./output")),
            prompt_timeout=_env_int("PROMPT_TIMEOUT", DEFAULT_PROMPT_TIMEOUT),
            benchmark_models=_env_str("BENCHMARK_MODELS", ""),
            # JUDGE_MODELS (plural) is the panel form; JUDGE_MODEL still works.
            judge_model=_env_str("JUDGE_MODELS", "") or _env_str("JUDGE_MODEL", ""),
            gen=GenerationParams(num_ctx=_env_int("NUM_CTX", 65536)),
            code_exec=_env_bool("CODE_EXEC", True),
        )

    def apply_cli(self, args: Any) -> Config:
        """Overlay CLI flags, which win over the environment."""
        updates: dict[str, Any] = {}
        if getattr(args, "quick", False):
            updates["perf_probe"] = False
        if getattr(args, "no_code_exec", False):
            updates["code_exec"] = False
        return replace(self, **updates) if updates else self
