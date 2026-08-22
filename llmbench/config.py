"""Configuration, resolved from environment variables and an optional .env file."""

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

# The prompt used to measure throughput independently of how long a model
# happens to make its answers. Padded to a target length at runtime.
PERF_PROBE_PROMPT = (
    "Write a detailed technical explanation of how a CPU cache hierarchy works. "
    "Cover L1, L2, and L3 caches, cache lines, associativity, and eviction policies."
)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_number(name: str, default: Any, cast: Any) -> Any:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"Error: {name} must be a {cast.__name__}, got {raw!r}.")
        sys.exit(1)


def _env_int(name: str, default: int) -> int:
    value: int = _env_number(name, default, int)
    return value


def _env_float(name: str, default: float) -> float:
    value: float = _env_number(name, default, float)
    return value


def _env_int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError:
        print(f"Error: {name} must be a comma-separated list of integers, got {raw!r}.")
        sys.exit(1)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_tristate(name: str) -> bool | None:
    """A flag that can also be left alone. Empty or `auto` means "don't ask"."""
    raw = os.getenv(name, "").strip().lower()
    if not raw or raw == "auto":
        return None
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class GenerationParams:
    """Sampling parameters sent to Ollama.

    Ollama's defaults sample at temperature 0.8 with no seed, which makes runs
    unreproducible and leaderboard gaps unreadable. We pin everything and record
    the values in the results so a run can be repeated exactly.

    `num_ctx` in particular must be set explicitly: left unset, Ollama picks a
    per-model default and silently truncates anything longer.

    The default is generous because the alternative is worse. At 8192 a
    reasoning model given a hard puzzle spends the whole window thinking and
    the generation is cut off before it ever writes an answer — measured as a
    zero, indistinguishable in the report from a model that answered wrongly.
    Three of five models in one run scored 0.4 on two tasks for exactly this
    reason. A window big enough that models stop for their own reasons costs
    KV-cache memory and runtime; a window that decides the result costs the
    benchmark its meaning.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    seed: int = 0
    num_ctx: int = 32768
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
        """Params for repeat `index`, with a distinct but deterministic seed."""
        return replace(self, seed=self.seed + index)


# Judging is a measurement instrument; it must not itself be a source of noise.
# The window has to hold the task prompt, the criteria, and the whole answer
# being judged. Too small and the judge silently scores a truncated view of a
# response — an instrument reading the wrong end of the ruler.
JUDGE_PARAMS = GenerationParams(temperature=0.0, top_p=1.0, seed=0, num_ctx=32768)


@dataclass(frozen=True)
class Config:
    """Everything the benchmark reads from the environment."""

    base_url: str = DEFAULT_BASE_URL
    tasks_dir: Path = Path("./tasks")
    output_dir: Path = Path("./output")
    prompt_timeout: int = DEFAULT_PROMPT_TIMEOUT
    judge_timeout: int = DEFAULT_JUDGE_TIMEOUT
    # Empty means "ask interactively".
    # Comma-separated; empty means "ask interactively".
    benchmark_models: str = ""
    judge_model: str = ""
    # A model scoring its own answers is not an independent measurement.
    allow_self_judge: bool = False

    gen: GenerationParams = GenerationParams()
    # How many times each prompt is answered. The spread across repeats is the
    # noise floor that any leaderboard difference has to clear.
    quality_repeats: int = 3
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

    # Objective scoring. Where a task has verifiable checks they carry most of
    # the weight; the judge still contributes the part that isn't verifiable.
    code_exec: bool = True
    objective_weight: float = 0.6

    # Reasoning handling. `think` maps to Ollama's thinking channel: True puts
    # the chain of thought in its own field, False asks for none at all, None
    # leaves the model's default alone. `answer_tags` appends an explicit
    # "put the answer between <answer></answer>" instruction, which is the only
    # reliable way to score a model that emits undelimited reasoning — at the
    # cost of changing the prompt, so it is off unless asked for.
    think: bool | None = None
    answer_tags: bool = False

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()
        return cls(
            base_url=_env_str("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            tasks_dir=Path(_env_str("TASKS_DIR", "./tasks")),
            output_dir=Path(_env_str("OUTPUT_DIR", "./output")),
            prompt_timeout=_env_int("PROMPT_TIMEOUT", DEFAULT_PROMPT_TIMEOUT),
            judge_timeout=_env_int("JUDGE_TIMEOUT", DEFAULT_JUDGE_TIMEOUT),
            benchmark_models=_env_str("BENCHMARK_MODELS", ""),
            # JUDGE_MODELS (plural) is the panel form; JUDGE_MODEL still works.
            judge_model=_env_str("JUDGE_MODELS", "") or _env_str("JUDGE_MODEL", ""),
            allow_self_judge=_env_bool("ALLOW_SELF_JUDGE", False),
            gen=GenerationParams(
                temperature=_env_float("TEMPERATURE", 0.0),
                top_p=_env_float("TOP_P", 1.0),
                top_k=_env_int("TOP_K", 40),
                seed=_env_int("SEED", 0),
                num_ctx=_env_int("NUM_CTX", 32768),
            ),
            quality_repeats=max(1, _env_int("QUALITY_REPEATS", 3)),
            perf_probe=_env_bool("PERF_PROBE", True),
            perf_repeats=max(1, _env_int("PERF_REPEATS", 5)),
            perf_predict_tokens=_env_int("PERF_PREDICT_TOKENS", 256),
            prefill_sweep=_env_int_list("PREFILL_SWEEP", (512, 4096, 16384)),
            prefill_repeats=max(1, _env_int("PREFILL_REPEATS", 2)),
            retries=max(1, _env_int("RETRIES", 3)),
            code_exec=_env_bool("CODE_EXEC", True),
            objective_weight=min(max(_env_float("OBJECTIVE_WEIGHT", 0.6), 0.0), 1.0),
            think=_env_tristate("THINK"),
            answer_tags=_env_bool("ANSWER_TAGS", False),
        )

    def apply_cli(self, args: Any) -> Config:
        """Overlay CLI flags, which win over the environment."""
        updates: dict[str, Any] = {}
        if getattr(args, "quick", False):
            updates.update(quality_repeats=1, perf_probe=False)
        if getattr(args, "repeats", None) is not None:
            updates["quality_repeats"] = max(1, args.repeats)
        if getattr(args, "no_perf_probe", False):
            updates["perf_probe"] = False
        if getattr(args, "no_code_exec", False):
            updates["code_exec"] = False
        if getattr(args, "answer_tags", False):
            updates["answer_tags"] = True
        if getattr(args, "think", None) is not None:
            updates["think"] = args.think
        return replace(self, **updates) if updates else self
