"""Deterministic checks that do not require a judge's opinion.

Several of the benchmark's tasks have verifiable answers: the coding tasks
either pass their tests or do not, the puzzle has one correct answer, the
constrained-writing tasks either obey the constraint or do not. Scoring those
by asking a language model how it feels about the response throws away a signal
that can simply be measured.

Objective scores are kept alongside judge scores rather than replacing them —
where they disagree is itself a measurement of how far the judge can be trusted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypedDict

from llmbench.scoring.codeexec import run_code_check
from llmbench.tasks import Task


class CheckResult(TypedDict):
    """One check's verdict, as a fraction of the way to passing."""

    type: str
    passed: float  # 0.0 to 1.0
    weight: float
    detail: str


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text)


def _check_contains(response: str, patterns: list[str], require_all: bool) -> tuple[float, str]:
    found = [p for p in patterns if p.lower() in response.lower()]
    missing = [p for p in patterns if p not in found]
    if require_all:
        fraction = len(found) / len(patterns)
        detail = "all present" if not missing else f"missing: {', '.join(missing)}"
    else:
        fraction = 1.0 if found else 0.0
        detail = f"found: {', '.join(found)}" if found else f"none of: {', '.join(patterns)}"
    return fraction, detail


def _check_regex(response: str, pattern: str, negate: bool) -> tuple[float, str]:
    try:
        match = re.search(pattern, response)
    except re.error as exc:
        return 0.0, f"invalid regex in task definition: {exc}"
    if negate:
        if match is None:
            return 1.0, "forbidden pattern absent"
        return 0.0, f"forbidden pattern present: {match.group(0)!r}"
    return (1.0, "pattern found") if match else (0.0, "pattern not found")


def _check_word_count(response: str, low: int | None, high: int | None) -> tuple[float, str]:
    count = len(_words(response))
    if low is not None and count < low:
        return 0.0, f"{count} words, below the minimum of {low}"
    if high is not None and count > high:
        return 0.0, f"{count} words, above the maximum of {high}"
    return 1.0, f"{count} words, within range"


def _check_json_valid(response: str) -> tuple[float, str]:
    """The response must be JSON and nothing else — fences and prose fail."""
    text = response.strip()
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        if text.startswith("```"):
            return 0.0, "wrapped in a markdown code fence; the task asked for bare JSON"
        return 0.0, f"not valid JSON: {exc.msg}"
    return 1.0, "valid JSON with no surrounding prose"


def run_check(check: dict[str, Any], response: str, code_exec: bool = True) -> CheckResult:
    """Evaluate one check against one response."""
    kind = str(check["type"])
    weight = float(check.get("weight", 1.0))

    if kind == "contains_all":
        passed, detail = _check_contains(response, list(check["patterns"]), require_all=True)
    elif kind == "contains_any":
        passed, detail = _check_contains(response, list(check["patterns"]), require_all=False)
    elif kind == "regex":
        passed, detail = _check_regex(response, str(check["pattern"]), bool(check.get("negate")))
    elif kind == "word_count":
        passed, detail = _check_word_count(response, check.get("min"), check.get("max"))
    elif kind == "json_valid":
        passed, detail = _check_json_valid(response)
    elif kind == "code_exec":
        if not code_exec:
            return {"type": kind, "passed": 0.0, "weight": 0.0, "detail": "skipped (CODE_EXEC=0)"}
        outcome = run_code_check(response, Path(check["suite_path"]))
        passed, detail = outcome.fraction, outcome.detail
    else:  # pragma: no cover - task loading rejects unknown types
        return {"type": kind, "passed": 0.0, "weight": 0.0, "detail": f"unknown check type {kind}"}

    return {"type": kind, "passed": passed, "weight": weight, "detail": detail}


def _vacuous(check: dict[str, Any], reason: str) -> CheckResult:
    return {
        "type": str(check["type"]),
        "passed": 0.0,
        "weight": float(check.get("weight", 1.0)),
        "detail": reason,
    }


def run_checks(task: Task, response: str, code_exec: bool = True) -> list[CheckResult]:
    """Evaluate every check attached to a task.

    An empty answer fails every check, including the negated ones. Left to
    themselves, "this forbidden word must not appear" and "this pattern must be
    absent" are satisfied by silence, so a model that emitted nothing at all
    scored 10/10 on the constraint half of a task and 0 on the rest. That is
    not a partial success; it is a missing answer, and it is scored as one.
    """
    if not response.strip():
        return [_vacuous(check, "no answer to check") for check in task.checks]
    return [run_check(check, response, code_exec) for check in task.checks]


def objective_score(results: list[CheckResult]) -> float | None:
    """Combine check results into a 0-10 score, or None if nothing was checked.

    Weight-zero results (a skipped check) are excluded, so disabling code
    execution leaves the remaining checks scoring normally instead of dragging
    the task to zero.
    """
    active = [r for r in results if r["weight"] > 0]
    if not active:
        return None
    total_weight = sum(r["weight"] for r in active)
    return 10.0 * sum(r["passed"] * r["weight"] for r in active) / total_weight


def blended_score(
    objective: float | None, judge: float | None, objective_weight: float = 0.6
) -> float | None:
    """Combine the two scores, falling back to whichever one exists.

    Where a task has verifiable checks they carry most of the weight, because
    they measure whether the answer is right rather than whether it reads well.
    The judge still contributes: passing the tests says nothing about whether
    the code is clear, or whether the story is any good.
    """
    if objective is None:
        return judge
    if judge is None:
        return objective
    return objective_weight * objective + (1 - objective_weight) * judge
