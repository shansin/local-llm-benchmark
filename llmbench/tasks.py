"""The benchmark dataset: tasks, grouped into categories.

A *task* is one prompt. A *category* is a group of tasks measuring the same
skill. The original layout conflated the two — one file per category meant one
prompt per category, so every category score rested on a single sample.

Tasks are TOML files under `tasks/<category>/<id>.toml`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECK_TYPES = {
    "code_exec",
    "contains_all",
    "contains_any",
    "regex",
    "word_count",
    "json_valid",
}

DIFFICULTIES = {"easy", "medium", "hard"}


class TaskError(Exception):
    """A task file is malformed."""


@dataclass(frozen=True)
class Task:
    """One prompt, with everything needed to score the answer."""

    id: str
    category: str
    prompt: str
    criteria: str = ""
    weight: float = 1.0
    difficulty: str = "medium"
    checks: list[dict[str, Any]] = field(default_factory=list)
    source: Path | None = None

    @property
    def key(self) -> str:
        """Stable identifier used as the checkpoint and report key."""
        return self.id


def categories_of(tasks: list[Task]) -> list[str]:
    """Category names in first-seen order."""
    seen: dict[str, None] = {}
    for task in tasks:
        seen.setdefault(task.category, None)
    return list(seen)


def group_by_category(tasks: list[Task]) -> dict[str, list[Task]]:
    grouped: dict[str, list[Task]] = {}
    for task in tasks:
        grouped.setdefault(task.category, []).append(task)
    return grouped


def _validate_check(check: dict[str, Any], where: str) -> None:
    kind = check.get("type")
    if not kind:
        raise TaskError(f"{where}: a check is missing its `type`")
    if kind not in CHECK_TYPES:
        raise TaskError(
            f"{where}: unknown check type {kind!r} (known: {', '.join(sorted(CHECK_TYPES))})"
        )
    if kind in ("contains_all", "contains_any") and not check.get("patterns"):
        raise TaskError(f"{where}: {kind} check needs a non-empty `patterns` list")
    if kind == "regex" and not check.get("pattern"):
        raise TaskError(f"{where}: regex check needs a `pattern`")
    if kind == "word_count" and "min" not in check and "max" not in check:
        raise TaskError(f"{where}: word_count check needs `min` and/or `max`")
    if kind == "code_exec" and not check.get("suite"):
        raise TaskError(f"{where}: code_exec check needs a `suite` path")


def load_task_file(path: Path) -> Task:
    """Parse one task TOML file."""
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise TaskError(f"{path}: invalid TOML — {exc}") from exc

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise TaskError(f"{path}: missing `prompt`")

    difficulty = str(data.get("difficulty", "medium"))
    if difficulty not in DIFFICULTIES:
        raise TaskError(
            f"{path}: difficulty {difficulty!r} must be one of {', '.join(sorted(DIFFICULTIES))}"
        )

    checks = list(data.get("checks", []))
    for check in checks:
        _validate_check(check, str(path))
        if check["type"] == "code_exec":
            suite = (path.parent / check["suite"]).resolve()
            if not suite.exists():
                raise TaskError(f"{path}: code_exec suite not found: {check['suite']}")
            check["suite_path"] = str(suite)

    return Task(
        id=str(data.get("id") or path.stem),
        category=str(data.get("category") or path.parent.name),
        prompt=prompt,
        criteria=str(data.get("criteria", "")).strip(),
        weight=float(data.get("weight", 1.0)),
        difficulty=difficulty,
        checks=checks,
        source=path,
    )


def load_tasks(tasks_dir: Path) -> list[Task]:
    """Load the task set from `tasks_dir`."""
    if not tasks_dir.exists():
        raise TaskError(f"task directory {tasks_dir} not found")

    files = sorted(tasks_dir.glob("*/*.toml")) + sorted(tasks_dir.glob("*.toml"))
    if not files:
        raise TaskError(f"no task files (*.toml) found under {tasks_dir}")

    tasks = [load_task_file(path) for path in files]

    seen: dict[str, Path] = {}
    for task in tasks:
        if task.id in seen:
            raise TaskError(
                f"duplicate task id {task.id!r} in {task.source} and {seen[task.id]}; "
                "ids must be unique across the whole task set"
            )
        seen[task.id] = task.source or Path(task.id)

    # Category order follows the directory listing; task order follows filename.
    return sorted(tasks, key=lambda t: (t.category, t.id))
