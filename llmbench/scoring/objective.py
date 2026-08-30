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


def _loads_lenient(response: str) -> Any:
    """Parse the JSON in a response that may be wrapped in prose or a fence.

    `json_valid` is the strict check and stays strict — it is the one that
    measures whether a model can follow "bare JSON, nothing else". The content
    checks are a different question: they ask whether the *data* is right, and
    failing them because of a stray code fence would score the same mistake
    twice.
    """
    text = response.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost brace- or bracket-delimited span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in the response")


_PATH_STEP = re.compile(r"\[(\d+)\]|([^.\[\]]+)")


def _walk(document: Any, path: str) -> Any:
    """Resolve a dotted path like `meeting.attendees[2].name` against parsed JSON."""
    node = document
    for index, key in _PATH_STEP.findall(path):
        if index:
            if not isinstance(node, list):
                raise LookupError(f"{path}: expected a list at [{index}]")
            position = int(index)
            if position >= len(node):
                raise LookupError(
                    f"{path}: index {position} is past the end of a {len(node)}-item list"
                )
            node = node[position]
        else:
            if not isinstance(node, dict):
                raise LookupError(f"{path}: expected an object at {key!r}")
            if key not in node:
                raise LookupError(f"{path}: no key {key!r}")
            node = node[key]
    return node


def _equal(found: Any, expected: Any, tolerance: float | None) -> bool:
    """Compare a found value with the expected one, forgivingly but not loosely.

    Strings are compared case-insensitively with surrounding whitespace
    stripped: a model that writes "Redwood Room" where the source says
    "redwood room" has extracted the right value. Numbers compare within an
    optional tolerance, and a numeric string compares equal to the number it
    spells, because JSON-producing models are inconsistent about quoting.
    """
    if isinstance(expected, str) and isinstance(found, str):
        return found.strip().casefold() == expected.strip().casefold()
    if isinstance(expected, bool) or isinstance(found, bool):
        return found is expected
    if isinstance(expected, int | float):
        try:
            number = float(found)
        except (TypeError, ValueError):
            return False
        return abs(number - float(expected)) <= (tolerance or 0.0)
    return bool(found == expected)


def _check_json_path(
    response: str, path: str, expected: Any, tolerance: float | None
) -> tuple[float, str]:
    try:
        document = _loads_lenient(response)
    except ValueError as exc:
        return 0.0, str(exc)
    try:
        found = _walk(document, path)
    except LookupError as exc:
        return 0.0, str(exc)
    if _equal(found, expected, tolerance):
        return 1.0, f"{path} == {expected!r}"
    return 0.0, f"{path} is {found!r}, expected {expected!r}"


_ANSWER_MARKER = re.compile(
    r"(?:final\s+answer|answer)\s*[:\-—]\s*(.+?)(?:\n\s*\n|\Z)", re.IGNORECASE | re.DOTALL
)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _answer_span(response: str) -> str:
    """The part of a response that carries the final answer.

    Prefers an explicit `Answer:` line — the last one, since a model that
    restates the format tends to write the real answer after it. Falls back to
    the closing lines, which is where an answer lands when a model works
    through the problem and then states its conclusion.
    """
    markers = _ANSWER_MARKER.findall(response)
    if markers:
        return markers[-1].strip()
    lines = [line for line in response.strip().splitlines() if line.strip()]
    return "\n".join(lines[-3:])


def _check_answer_equals(
    response: str, expected: Any, numeric: bool, tolerance: float | None
) -> tuple[float, str]:
    """Does the model's stated answer match the known one?

    Scores the conclusion rather than the working: a correct derivation that
    ends on the wrong number is wrong, and a response that never commits to an
    answer is not a near miss.
    """
    span = _answer_span(response)
    if numeric:
        found = _NUMBER.findall(span)
        if not found:
            return 0.0, f"no number in the stated answer: {span[:60]!r}"
        # The last number in the span is the one the sentence concludes on.
        value = float(found[-1].replace(",", ""))
        if abs(value - float(expected)) <= (tolerance or 0.0):
            return 1.0, f"answered {value:g}"
        return 0.0, f"answered {value:g}, expected {float(expected):g}"

    accepted = expected if isinstance(expected, list) else [expected]
    haystack = span.casefold()
    for candidate in accepted:
        if str(candidate).strip().casefold() in haystack:
            return 1.0, f"answered {candidate!r}"
    return 0.0, f"stated answer {span[:60]!r} matches none of {accepted}"


def _within(count: int, check: dict[str, Any], noun: str) -> tuple[float, str]:
    exact = check.get("equals")
    if exact is not None:
        if count == int(exact):
            return 1.0, f"{count} {noun}, as required"
        return 0.0, f"{count} {noun}, expected exactly {exact}"
    low, high = check.get("min"), check.get("max")
    if low is not None and count < int(low):
        return 0.0, f"{count} {noun}, below the minimum of {low}"
    if high is not None and count > int(high):
        return 0.0, f"{count} {noun}, above the maximum of {high}"
    return 1.0, f"{count} {noun}, within range"


def _check_line_count(response: str, check: dict[str, Any]) -> tuple[float, str]:
    """Count non-blank lines, optionally only those matching a pattern.

    A model asked for thirty rows that writes twenty and an ellipsis has not
    partly succeeded at the task; counting the lines is what catches it.
    """
    lines = [line for line in response.splitlines() if line.strip()]
    pattern = check.get("pattern")
    if pattern:
        try:
            matcher = re.compile(str(pattern))
        except re.error as exc:
            return 0.0, f"invalid regex in task definition: {exc}"
        lines = [line for line in lines if matcher.search(line)]
        return _within(len(lines), check, "matching lines")
    return _within(len(lines), check, "non-blank lines")


def _check_match_count(response: str, check: dict[str, Any]) -> tuple[float, str]:
    """Count how many times a pattern occurs anywhere in the response."""
    try:
        matches = re.findall(str(check["pattern"]), response)
    except re.error as exc:
        return 0.0, f"invalid regex in task definition: {exc}"
    return _within(len(matches), check, "matches")


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
    elif kind == "json_path":
        passed, detail = _check_json_path(
            response, str(check["path"]), check.get("equals"), check.get("tolerance")
        )
    elif kind == "answer_equals":
        passed, detail = _check_answer_equals(
            response,
            check.get("expected"),
            bool(check.get("numeric")),
            check.get("tolerance"),
        )
    elif kind == "line_count":
        passed, detail = _check_line_count(response, check)
    elif kind == "match_count":
        passed, detail = _check_match_count(response, check)
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
