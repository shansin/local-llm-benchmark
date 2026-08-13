import pytest

from llmbench.tasks import (
    TaskError,
    categories_of,
    group_by_category,
    load_task_file,
    load_tasks,
)

MINIMAL = """
prompt = "Do the thing"
"""


def write(tmp_path, relpath, text):
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_id_and_category_default_to_the_path(tmp_path):
    path = write(tmp_path, "tasks/coding/my-task.toml", MINIMAL)
    task = load_task_file(path)
    assert task.id == "my-task"
    assert task.category == "coding"
    assert task.difficulty == "medium"
    assert task.weight == 1.0


def test_explicit_fields_win_over_the_path(tmp_path):
    path = write(
        tmp_path,
        "tasks/coding/x.toml",
        'id = "real-id"\ncategory = "reasoning"\nweight = 2.5\n'
        'difficulty = "hard"\nprompt = "p"\ncriteria = "c"\n',
    )
    task = load_task_file(path)
    assert (task.id, task.category, task.weight, task.difficulty) == (
        "real-id",
        "reasoning",
        2.5,
        "hard",
    )


def test_missing_prompt_is_an_error(tmp_path):
    path = write(tmp_path, "tasks/coding/x.toml", 'criteria = "c"\n')
    with pytest.raises(TaskError, match="missing `prompt`"):
        load_task_file(path)


def test_invalid_toml_is_reported_with_the_path(tmp_path):
    path = write(tmp_path, "tasks/coding/x.toml", "prompt = 'unterminated\n")
    with pytest.raises(TaskError, match="invalid TOML"):
        load_task_file(path)


def test_unknown_difficulty_is_rejected(tmp_path):
    path = write(tmp_path, "tasks/coding/x.toml", 'prompt = "p"\ndifficulty = "impossible"\n')
    with pytest.raises(TaskError, match="difficulty"):
        load_task_file(path)


def test_unknown_check_type_is_rejected(tmp_path):
    path = write(tmp_path, "tasks/coding/x.toml", 'prompt = "p"\n[[checks]]\ntype = "vibes"\n')
    with pytest.raises(TaskError, match="unknown check type"):
        load_task_file(path)


def test_checks_are_validated_for_required_fields(tmp_path):
    cases = [
        ('[[checks]]\ntype = "contains_all"\n', "patterns"),
        ('[[checks]]\ntype = "regex"\n', "pattern"),
        ('[[checks]]\ntype = "word_count"\n', "min"),
        ('[[checks]]\ntype = "code_exec"\n', "suite"),
    ]
    for i, (check, expected) in enumerate(cases):
        path = write(tmp_path, f"tasks/coding/x{i}.toml", f'prompt = "p"\n{check}')
        with pytest.raises(TaskError, match=expected):
            load_task_file(path)


def test_code_exec_suite_must_exist(tmp_path):
    path = write(
        tmp_path,
        "tasks/coding/x.toml",
        'prompt = "p"\n[[checks]]\ntype = "code_exec"\nsuite = "suites/nope.py"\n',
    )
    with pytest.raises(TaskError, match="suite not found"):
        load_task_file(path)


def test_code_exec_suite_path_is_resolved_relative_to_the_task(tmp_path):
    write(tmp_path, "tasks/coding/suites/s.py", "def test_x(): pass\n")
    path = write(
        tmp_path,
        "tasks/coding/x.toml",
        'prompt = "p"\n[[checks]]\ntype = "code_exec"\nsuite = "suites/s.py"\n',
    )
    task = load_task_file(path)
    assert task.checks[0]["suite_path"].endswith("tasks/coding/suites/s.py")


def test_duplicate_ids_across_categories_are_rejected(tmp_path):
    write(tmp_path, "tasks/coding/dup.toml", 'id = "same"\nprompt = "p"\n')
    write(tmp_path, "tasks/writing/other.toml", 'id = "same"\nprompt = "p"\n')
    with pytest.raises(TaskError, match="duplicate task id"):
        load_tasks(tmp_path / "tasks")


def test_loads_a_whole_task_tree(tmp_path):
    write(tmp_path, "tasks/coding/a.toml", 'prompt = "p"\n')
    write(tmp_path, "tasks/coding/b.toml", 'prompt = "p"\n')
    write(tmp_path, "tasks/writing/c.toml", 'prompt = "p"\n')
    tasks = load_tasks(tmp_path / "tasks")
    assert len(tasks) == 3
    assert categories_of(tasks) == ["coding", "writing"]
    assert [t.id for t in group_by_category(tasks)["coding"]] == ["a", "b"]


def test_a_missing_task_dir_is_an_error_not_an_empty_run(tmp_path):
    with pytest.raises(TaskError, match="not found"):
        load_tasks(tmp_path / "tasks")


def test_an_empty_task_dir_is_an_error_not_an_empty_run(tmp_path):
    """Running zero tasks and reporting success would be worse than failing."""
    (tmp_path / "tasks").mkdir()
    with pytest.raises(TaskError, match="no task files"):
        load_tasks(tmp_path / "tasks")


def test_the_shipped_task_set_is_valid():
    """Guards the real dataset against a malformed edit."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    tasks = load_tasks(root / "tasks")
    assert len(tasks) >= 25
    for task in tasks:
        assert task.prompt, f"{task.id} has an empty prompt"
        assert task.criteria, f"{task.id} has no criteria"
    for category, group in group_by_category(tasks).items():
        assert len(group) >= 3, f"{category} has only {len(group)} task(s)"
