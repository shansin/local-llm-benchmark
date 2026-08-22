"""LLM-as-judge scoring.

Three things make a judge more than a vibe:

* **Structure.** The judge scores named dimensions rather than emitting one
  opaque number, so "9.6 vs 9.2" can be traced to something.
* **Determinism.** Judging runs at temperature 0 with a fixed seed. A noisy
  instrument cannot measure a small difference.
* **Plurality.** A panel of judges votes and the median wins, so no single
  model's taste decides the leaderboard. A model never counts its own vote.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

import requests

from llmbench.config import JUDGE_PARAMS, GenerationParams
from llmbench.stats import median

DIMENSIONS = ("accuracy", "completeness", "instruction_following", "clarity")

RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        **{d: {"type": "integer", "minimum": 1, "maximum": 10} for d in DIMENSIONS},
        "reason": {"type": "string"},
    },
    "required": [*DIMENSIONS, "reason"],
}


class JudgeResult(TypedDict):
    """A judge's verdict.

    `score` is None when the judge produced nothing parseable — a failure to
    read the verdict is not the same as a bad response, so it is excluded from
    averages rather than counted as a low score.
    """

    score: float | None
    reason: str
    dimensions: dict[str, int]
    votes: list[dict[str, Any]]
    response_chars: int


def format_verified(checks: list[dict[str, Any]]) -> list[str]:
    """Render check results as plain statements of fact for the judge."""
    return [f"{c['type']}: {c['detail']}" for c in checks if c.get("weight", 1.0) > 0]


def build_judge_prompt(
    category: str,
    prompt_text: str,
    response_text: str,
    expected_text: str = "",
    verified: list[str] | None = None,
) -> str:
    """Build the judging prompt. The model under test is never named.

    Anything already measured is handed to the judge as settled fact. Language
    models cannot count letters or words reliably, and asking them to try
    produces confident nonsense: on one task the checks recorded a response as
    421 words inside the limit while the judge failed it for length, and on
    another the judge's stated reasoning was six lines of it counting the same
    sentence twice. The measurable half of a rubric should be measured, and the
    judge left to score the half that cannot be.
    """
    expected_section = ""
    guidance = "Evaluate based on accuracy, completeness, instruction following, and clarity."
    if expected_text:
        expected_section = f"\nExpected answer / evaluation criteria:\n{expected_text}\n"
        guidance += (
            " Use the expected answer and evaluation criteria above as your primary scoring guide."
        )

    verified_section = ""
    if verified:
        facts = "\n".join(f"- {fact}" for fact in verified)
        verified_section = (
            "\nAutomated checks already measured the following. Treat these as "
            "ground truth: do not re-count words or letters yourself, and do not "
            "contradict them. Score the qualities they cannot capture.\n"
            f"{facts}\n"
        )

    dimensions = "\n".join(f'  "{d}": <integer 1-10>,' for d in DIMENSIONS)
    preamble = (
        "You are an expert evaluator. Score the AI response below on each "
        "dimension from 1 to 10 (10 = excellent)."
    )
    return f"""{preamble}

Category: {category}

Original prompt:
{prompt_text}
{expected_section}{verified_section}
AI response:
{response_text}

{guidance}

Respond with a single JSON object and nothing else:
{{
{dimensions}
  "reason": "<one line justification>"
}}"""


def _clamp(value: Any) -> int | None:
    try:
        return min(max(int(value), 1), 10)
    except (TypeError, ValueError):
        return None


def parse_judge_output(judge_text: str, response_chars: int = 0) -> JudgeResult:
    """Read a judge's verdict, preferring JSON and falling back to loose text."""
    empty: JudgeResult = {
        "score": None,
        "reason": "",
        "dimensions": {},
        "votes": [],
        "response_chars": response_chars,
    }

    payload = _extract_json(judge_text)
    if payload is not None:
        dimensions = {d: s for d in DIMENSIONS if (s := _clamp(payload.get(d))) is not None}
        if dimensions:
            reason = str(payload.get("reason", "")).strip()
            return {
                **empty,
                "score": sum(dimensions.values()) / len(dimensions),
                "reason": reason or "(no reason given)",
                "dimensions": dimensions,
            }

    # Older/simpler judges emit "Score: 8 / Reason: ...". Accept that too.
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", judge_text)
    if match:
        score = min(max(float(match.group(1)), 1.0), 10.0)
        reason_match = re.search(r"Reason:\s*(.+)", judge_text)
        reason = reason_match.group(1).strip() if reason_match else judge_text.strip()[:100]
        return {**empty, "score": score, "reason": reason}

    return {**empty, "reason": f"[UNPARSEABLE JUDGE OUTPUT] {judge_text.strip()[:100]}"}


def _extract_json(text: str) -> dict[str, Any] | None:
    """Find a JSON object in the judge's output, fenced or not."""
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    braced = re.search(r"\{.*\}", text, re.DOTALL)
    if braced:
        candidates.append(braced.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def judge_once(
    base_url: str,
    judge_model: str,
    category: str,
    prompt_text: str,
    response_text: str,
    expected_text: str = "",
    timeout: int = 1800,
    params: GenerationParams | None = None,
    verified: list[str] | None = None,
) -> JudgeResult:
    """Ask one judge model to score one response."""
    params = params or JUDGE_PARAMS
    judge_prompt = build_judge_prompt(category, prompt_text, response_text, expected_text, verified)

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": judge_model,
                "prompt": judge_prompt,
                "stream": False,
                "format": RUBRIC_SCHEMA,
                "options": params.to_options(),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return {
            "score": None,
            "reason": "[JUDGE TIMEOUT]",
            "dimensions": {},
            "votes": [],
            "response_chars": len(response_text),
        }
    except requests.exceptions.RequestException as exc:
        return {
            "score": None,
            "reason": f"[JUDGE ERROR: {type(exc).__name__}]",
            "dimensions": {},
            "votes": [],
            "response_chars": len(response_text),
        }

    return parse_judge_output(_verdict_text(resp.json()), len(response_text))


def _verdict_text(payload: dict[str, Any]) -> str:
    """Get the judge's verdict out of an Ollama reply.

    Reasoning models split their output: the visible answer lands in
    `response` and the chain of thought in `thinking`. When a schema is
    supplied, some of them emit the whole structured verdict as thinking and
    leave `response` empty — so an empty `response` is not an empty verdict.
    """
    text = str(payload.get("response") or "").strip()
    return text or str(payload.get("thinking") or "").strip()


def judge_response(
    base_url: str,
    judges: list[str],
    category: str,
    prompt_text: str,
    response_text: str,
    expected_text: str = "",
    timeout: int = 1800,
    params: GenerationParams | None = None,
    model_under_test: str | None = None,
    allow_self_judge: bool = False,
    verified: list[str] | None = None,
) -> JudgeResult:
    """Collect a panel's verdicts and return the median.

    A model is excluded from judging its own response unless explicitly
    allowed: its score for itself is not an independent measurement. If that
    leaves no eligible judge, the response goes unscored rather than being
    scored by an interested party.
    """
    eligible = [j for j in judges if allow_self_judge or j != model_under_test]
    if not eligible:
        return {
            "score": None,
            "reason": f"[NO INDEPENDENT JUDGE] every judge is {model_under_test}",
            "dimensions": {},
            "votes": [],
            "response_chars": len(response_text),
        }

    votes: list[dict[str, Any]] = []
    for judge_model in eligible:
        verdict = judge_once(
            base_url,
            judge_model,
            category,
            prompt_text,
            response_text,
            expected_text,
            timeout,
            params,
            verified,
        )
        votes.append({"judge": judge_model, **verdict})

    scored = [v for v in votes if v["score"] is not None]
    if not scored:
        return {**votes[0], "votes": votes}  # type: ignore[typeddict-item]

    combined = median([v["score"] for v in scored])
    # Report the reason from whichever judge landed closest to the panel's verdict.
    representative = min(scored, key=lambda v: abs(v["score"] - combined))

    dimensions: dict[str, int] = {}
    for dimension in DIMENSIONS:
        values = [v["dimensions"][dimension] for v in scored if dimension in v["dimensions"]]
        if values:
            dimensions[dimension] = round(median(values))

    return {
        "score": combined,
        "reason": representative["reason"],
        "dimensions": dimensions,
        "votes": [{"judge": v["judge"], "score": v["score"]} for v in votes],
        "response_chars": len(response_text),
    }
