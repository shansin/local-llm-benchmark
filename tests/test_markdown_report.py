from llmbench.report.markdown import write_model_benchmark, write_results
from llmbench.tasks import Task


def gen(response="a clean answer", **overrides):
    base = {
        "response": response,
        "thinking": "",
        "done_reason": "stop",
        "truncated": False,
        "tokens_per_sec": 100.0,
        "ttft": 0.2,
        "prefill_time": 0.1,
        "load_time": 0.0,
        "total_time": 3.0,
        "eval_count": 300,
        "prompt_eval_speed": 1500.0,
        "prompt_eval_count": 150,
        "seed": 0,
        "error": None,
        "gpu": None,
    }
    return {**base, **overrides}


TASKS = [Task(id="t1", category="writing", prompt="Write a scene.", criteria="c")]


def _write(tmp_path, results, gen_params=None):
    write_results(
        tmp_path,
        all_results=results,
        all_details={},
        judge_scores={},
        judge_model="j",
        tasks=TASKS,
        gen_params=gen_params or {"num_ctx": 8192},
    )
    return (tmp_path / "results.md").read_text()


def test_a_clean_run_gets_no_completeness_section(tmp_path):
    """The section is a warning; showing it when nothing is wrong dulls it."""
    md = _write(tmp_path, {"m": {"t1": [gen()]}})
    assert "Answer completeness" not in md


def test_truncated_generations_are_named_along_with_the_setting_that_caused_them(tmp_path):
    md = _write(
        tmp_path,
        {"m": {"t1": [gen("<think>never finished", truncated=True)]}},
        gen_params={"num_ctx": 8192},
    )
    assert "Answer completeness" in md
    assert "num_ctx=8192" in md
    assert "not as wrong" in md


def test_leaked_reasoning_is_explained_where_it_is_counted(tmp_path):
    md = _write(tmp_path, {"m": {"t1": [gen("Let me plan this out.\n\nThe scene.")]}})
    assert "Leaked reasoning" in md
    # Answer tags are always on now, so leakage means the model ignored them.
    assert "ignored" in md


def test_per_task_file_shows_the_text_that_was_actually_scored(tmp_path):
    write_model_benchmark(
        tmp_path,
        {"name": "m", "details": {}},
        {"t1": [gen("<think>deliberating</think>The scene.")]},
        TASKS,
    )
    body = (tmp_path / "t1.md").read_text()
    assert "**Answer (as scored):**" in body
    assert "The scene." in body
    # The reasoning is kept but folded away, not silently discarded.
    assert "deliberating" in body
    assert "<details>" in body


def test_per_task_file_flags_a_truncated_generation(tmp_path):
    write_model_benchmark(
        tmp_path,
        {"name": "m", "details": {}},
        {"t1": [gen("<think>cut", truncated=True)]},
        TASKS,
    )
    body = (tmp_path / "t1.md").read_text()
    assert "truncated" in body
    assert "_(none)_" in body


def test_discarded_reasoning_is_told_apart_from_the_context_window(tmp_path):
    """The two truncations have different remedies; the report must say which."""
    md = _write(
        tmp_path,
        {"m": {"t1": [gen("", truncated=True, discarded_reasoning=True)]}},
        gen_params={"num_ctx": 8192},
    )
    assert "discarded" in md
    assert "cannot fix" in md


def test_an_ordinary_truncation_does_not_mention_discarded_reasoning(tmp_path):
    md = _write(tmp_path, {"m": {"t1": [gen("<think>never finished", truncated=True)]}})
    assert "cannot fix" not in md


def test_excluded_models_are_listed_with_their_evidence(tmp_path):
    write_results(
        tmp_path,
        all_results={"m": {"t1": [gen()]}},
        all_details={},
        judge_scores={},
        judge_model="j",
        tasks=TASKS,
        gen_params={"num_ctx": 8192},
        excluded={"broken:7b": "thinking on: empty response; model default: timeout"},
    )
    md = (tmp_path / "results.md").read_text()
    assert "Excluded models" in md
    assert "broken:7b" in md
    assert "empty response" in md
