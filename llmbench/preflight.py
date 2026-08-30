"""Per-model preflight: find settings this model can actually answer under.

The benchmark speaks one protocol — every prompt asks for the final answer
inside <answer></answer>, and reasoning goes on Ollama's thinking channel where
the model has one. But local models disagree about the thinking channel: some
have none, some must be asked explicitly, and a hybrid-reasoning model asked
for no mode in particular reasons anyway while Ollama streams none of it, so
the whole budget is spent and both channels come back empty.

Discovering that 43 tasks into a run wastes hours and produces leaderboard
rows that need footnotes. The preflight discovers it up front with one trivial
prompt: it walks the thinking modes in order of preference and keeps the first
one under which the model produces a scorable answer. A model that produces
nothing under any mode is excluded from the run — loudly, with the evidence —
rather than benchmarked into a row of zeros.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from llmbench.config import GenerationParams
from llmbench.runner import run_prompt
from llmbench.scoring.extract import extract_answer, with_answer_format

# Trivial on purpose: a model that cannot answer this cannot answer anything,
# and nothing else should be concluded from it.
SMOKE_PROMPT = "What is 2 + 3? Reply with just the number."

# The model is already warm when the preflight runs, and the prompt is trivial.
# Capped separately from the task timeout so a broken model costs minutes, not
# the 45 the tasks are allowed.
PREFLIGHT_TIMEOUT = 600

# Room for a reasoning model to think its way to "5" without inviting a runaway.
PREFLIGHT_PREDICT = 2048

# Modes in order of preference. `True` keeps the deliberation as evidence on
# its own channel; `None` leaves the model's default alone; `False` asks for no
# reasoning at all — the last resort, because it measures a reasoning model
# with its reasoning turned off.
THINK_LADDER: tuple[bool | None, ...] = (True, None, False)


@dataclass(frozen=True)
class ModelProfile:
    """How to talk to one model, as established by its preflight."""

    think: bool | None
    usable: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {"think": self.think, "usable": self.usable, "note": self.note}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelProfile:
        return cls(
            think=data.get("think"),
            usable=bool(data.get("usable")),
            note=str(data.get("note") or ""),
        )


def _mode_label(think: bool | None) -> str:
    return {True: "thinking on", False: "thinking off", None: "model default"}[think]


def preflight(
    base_url: str,
    model_name: str,
    params: GenerationParams,
    timeout: int = PREFLIGHT_TIMEOUT,
) -> ModelProfile:
    """Establish the thinking mode this model answers under, or that none works.

    Each attempt sends the smoke prompt under the benchmark's real protocol
    (answer tags on), so what passes here is exactly what the tasks will use.
    `run_prompt` already downgrades a `think` request the model rejects, so an
    attempt records the mode that was actually used; modes already tried that
    way are not tried again.
    """
    probe_params = replace(params, num_predict=PREFLIGHT_PREDICT)
    tried: set[bool | None] = set()
    evidence: list[str] = []

    for think in THINK_LADDER:
        if think in tried:
            continue
        result = run_prompt(
            base_url,
            model_name,
            with_answer_format(SMOKE_PROMPT),
            timeout,
            probe_params,
            retries=1,
            think=think,
        )
        # The mode the request ended up using, after any rejected-`think`
        # downgrade inside run_prompt.
        used = result.get("think_used", think) if result["error"] is None else think
        tried.add(think)
        tried.add(used)

        if result["error"] is not None:
            evidence.append(f"{_mode_label(think)}: {result['error']}")
            continue

        extraction = extract_answer(result["response"], result["thinking"])
        if not extraction.empty:
            return ModelProfile(think=used, usable=True, note=_mode_label(used))

        why = "reasoned without answering" if result["thinking"].strip() else "empty response"
        if result.get("discarded_reasoning"):
            why = "decoded tokens onto neither channel"
        evidence.append(f"{_mode_label(used)}: {why}")

    return ModelProfile(think=None, usable=False, note="; ".join(evidence))
