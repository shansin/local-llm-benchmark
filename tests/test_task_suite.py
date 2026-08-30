"""The task suite itself is data, and data can be wrong.

A task whose checks a correct answer cannot pass is worse than no task: it
scores every model at zero and reads, in the report, exactly like a question
they all failed. These tests answer each generated task perfectly and assert a
full objective score, so a broken JSON path, an off-by-one row count or a
regex that never matches is caught here rather than after a multi-hour run.

The answer *keys* are not verified here — they are computed by the same
simulation that writes the prompt (see `tasks/generators/`), so they cannot
disagree with it. What is verified is that the checks are answerable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from llmbench.scoring.objective import objective_score, run_checks
from llmbench.tasks import Task, load_tasks

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
sys.path.insert(0, str(TASKS_DIR / "generators"))

ALL_TASKS = load_tasks(TASKS_DIR)
BY_ID = {task.id: task for task in ALL_TASKS}


def _score(task: Task, response: str) -> float | None:
    return objective_score(run_checks(task, response, code_exec=False))


def _ideal_json(task: Task) -> str:
    """Build the JSON document that every `json_path` check on this task wants.

    Reading the answer back out of the checks proves the plumbing, not the
    arithmetic: that each path is syntactically valid, resolves against a
    document of the shape the prompt asks for, and compares equal to the value
    the generator computed.
    """
    root: dict | list = {}
    for check in task.checks:
        if check["type"] != "json_path":
            continue
        _place(root, str(check["path"]), check["equals"])
    return json.dumps(root)


def _steps(path: str) -> list[int | str]:
    out: list[int | str] = []
    for part in path.replace("[", ".[").split("."):
        if not part:
            continue
        out.append(int(part[1:-1]) if part.startswith("[") else part)
    return out


def _place(root: dict | list, path: str, value: object) -> None:
    steps = _steps(path)
    node: object = root
    for step, nxt in zip(steps, steps[1:], strict=False):
        child: dict | list = [] if isinstance(nxt, int) else {}
        if isinstance(step, int):
            assert isinstance(node, list)
            while len(node) <= step:
                node.append(None)
            if node[step] is None:
                node[step] = child
            node = node[step]
        else:
            assert isinstance(node, dict)
            node = node.setdefault(step, child)
    last = steps[-1]
    if isinstance(last, int):
        assert isinstance(node, list)
        while len(node) <= last:
            node.append(None)
        node[last] = value
    else:
        assert isinstance(node, dict)
        node[last] = value


JSON_ANSWER_TASKS = [
    "ledger-audit",
    "policy-lookup",
    "scattered-facts",
    "stack-machine",
    "robot-grid",
    "schema-migration",
]


@pytest.mark.parametrize("task_id", JSON_ANSWER_TASKS)
def test_every_json_path_check_is_satisfiable(task_id):
    """A path that never resolves would score every model zero, silently."""
    task = BY_ID[task_id]
    json_checks = [c for c in task.checks if c["type"] == "json_path"]
    assert json_checks, f"{task_id} has no json_path checks to verify"
    results = run_checks(task, _ideal_json(task), code_exec=False)
    failed = [
        (c["path"], r["detail"])
        for c, r in zip(task.checks, results, strict=True)
        if c["type"] == "json_path" and r["passed"] != 1.0
    ]
    assert not failed, f"{task_id}: unsatisfiable checks {failed}"


def test_a_wrong_value_actually_fails():
    """The satisfiability test above would pass on a check that passes anything."""
    task = BY_ID["robot-grid"]
    wrong = json.loads(_ideal_json(task))
    wrong["x"] = 99
    assert _score(task, json.dumps(wrong)) != 10.0


def test_partial_answers_score_partially():
    """Per-field checks exist so that six-of-seven is not the same as none."""
    task = BY_ID["stack-machine"]
    full = json.loads(_ideal_json(task))
    partial = {**full, "sum": full["sum"] + 1}
    score = _score(task, json.dumps(partial))
    assert score is not None and 0.0 < score < 10.0


# ---------- tasks whose ideal answer has to be written out in full ----------


def test_csv_to_json_scores_ten_on_a_correct_conversion():
    import transformation

    rows = transformation._staff(transformation.random.Random(606), 30)
    answer = [
        {
            "id": r["id"],
            "name": f"{r['first']} {r['last']}",
            "dept": r["dept"],
            "pay": round(r["hours"] * r["rate"], 2),
        }
        for r in rows
    ]
    assert _score(BY_ID["csv-to-json"], json.dumps(answer)) == 10.0


def test_csv_to_json_punishes_an_abbreviated_conversion():
    """The failure this task exists to catch: fifteen rows and an ellipsis."""
    import transformation

    rows = transformation._staff(transformation.random.Random(606), 30)
    short = json.dumps(
        [
            {
                "id": r["id"],
                "name": f"{r['first']} {r['last']}",
                "dept": r["dept"],
                "pay": round(r["hours"] * r["rate"], 2),
            }
            for r in rows[:15]
        ]
    )
    score = _score(BY_ID["csv-to-json"], short)
    assert score is not None and score < 6.0


def test_computed_table_scores_ten_on_a_correct_table():
    import transformation

    rows = transformation._staff(transformation.random.Random(112358), 28)
    lines = ["| ID | Hours | Rate | Pay | Band |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        pay = round(r["hours"] * r["rate"], 2)
        band = "low" if pay < 500 else ("mid" if pay < 1500 else "high")
        lines.append(f"| {r['id']} | {r['hours']} | {r['rate']} | {pay:.2f} | {band} |")
    assert _score(BY_ID["computed-table"], "\n".join(lines)) == 10.0


def test_computed_table_punishes_a_table_that_stops_early():
    import transformation

    rows = transformation._staff(transformation.random.Random(112358), 28)
    lines = ["| ID | Hours | Rate | Pay | Band |", "| --- | --- | --- | --- | --- |"]
    for r in rows[:12]:
        pay = round(r["hours"] * r["rate"], 2)
        band = "low" if pay < 500 else ("mid" if pay < 1500 else "high")
        lines.append(f"| {r['id']} | {r['hours']} | {r['rate']} | {pay:.2f} | {band} |")
    score = _score(BY_ID["computed-table"], "\n".join(lines))
    assert score is not None and score < 3.0


def test_ledger_balance_accepts_the_stated_answer():
    import statetrack

    task = BY_ID["ledger-balance"]
    expected = task.checks[0]["expected"]
    assert _score(task, f"Working through it...\n\nAnswer: {expected}") == 10.0
    assert _score(task, f"Answer: {expected + 25}") == 0.0
    assert statetrack  # the module is the source of the key above


# ---------- suite-wide invariants ----------


def test_every_task_has_a_criteria_block_for_the_judge():
    missing = [t.id for t in ALL_TASKS if not t.criteria.strip()]
    assert not missing, f"tasks with no judge criteria: {missing}"


def test_every_task_id_is_unique_and_matches_its_filename():
    for task in ALL_TASKS:
        assert task.source is not None
        assert task.id == task.source.stem


def test_the_long_context_tasks_are_actually_long():
    """A long-context task that fits in 2k tokens is testing something else."""
    for task_id in ("ledger-audit", "policy-lookup", "scattered-facts"):
        assert len(BY_ID[task_id].prompt) > 15_000, task_id


# ---------- repair tasks ----------

REPAIR_TASKS = ["fix-window-sum", "fix-run-length", "fix-date-overlap"]


@pytest.mark.parametrize("task_id", REPAIR_TASKS)
def test_the_code_in_a_repair_prompt_fails_its_own_suite(task_id):
    """A repair task is only a task if the code as given is actually broken.

    The snippet is pulled out of the prompt rather than restated here, so the
    two cannot drift: if someone fixes the bug while editing the prompt, this
    fails instead of the task quietly becoming free marks.
    """
    from llmbench.scoring.codeexec import run_code_check

    task = BY_ID[task_id]
    suite = Path(task.checks[0]["suite_path"])
    outcome = run_code_check(task.prompt, suite)
    assert outcome.fraction < 1.0, f"{task_id}: the broken code passes its own suite"


@pytest.mark.parametrize("task_id", REPAIR_TASKS)
def test_a_repair_task_still_gives_partial_credit(task_id):
    """All-or-nothing on a repair task would waste its main advantage.

    The broken code passes some of the suite, so a model that fixes one of two
    bugs lands between the two — which is the resolution this class of task is
    here to provide.
    """
    from llmbench.scoring.codeexec import run_code_check

    task = BY_ID[task_id]
    outcome = run_code_check(task.prompt, Path(task.checks[0]["suite_path"]))
    assert outcome.fraction > 0.0, f"{task_id}: the broken code fails every test"


# ---------- the new categories are wired in ----------


def test_the_new_categories_are_discovered():
    categories = {task.category for task in ALL_TASKS}
    assert {"longcontext", "statetrack", "transformation", "faithfulness"} <= categories


def test_faithfulness_tasks_do_not_reward_blanket_scepticism():
    """A model that calls every premise false must not score well.

    The task is only a measurement if refusing everything is punished, so the
    sound premises have to be checked with the same weight as the false ones.
    """
    task = BY_ID["false-premises"]
    all_false = json.dumps({f"q{i}": {"premise_ok": False, "note": "wrong"} for i in range(1, 7)})
    all_true = json.dumps({f"q{i}": {"premise_ok": True, "note": "sure"} for i in range(1, 7)})
    for response in (all_false, all_true):
        score = _score(task, response)
        assert score is not None and score < 5.0
