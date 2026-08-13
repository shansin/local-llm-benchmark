import pytest

from llmbench.scoring.objective import blended_score, objective_score, run_check, run_checks
from llmbench.tasks import Task


def check(kind, **kwargs):
    return {"type": kind, **kwargs}


# ---------- contains ----------


def test_contains_all_is_graded_not_pass_fail():
    """Partial credit matters: 4 of 5 right answers is not the same as none."""
    result = run_check(check("contains_all", patterns=["a", "b", "c", "d"]), "a b c")
    assert result["passed"] == 0.75
    assert "missing: d" in result["detail"]


def test_contains_all_passes():
    assert run_check(check("contains_all", patterns=["x", "y"]), "x and y")["passed"] == 1.0


def test_contains_is_case_insensitive():
    assert run_check(check("contains_all", patterns=["LXXVII"]), "lxxvii")["passed"] == 1.0


def test_contains_any_needs_only_one():
    result = run_check(check("contains_any", patterns=["F7", "0xF7"]), "the answer is F7")
    assert result["passed"] == 1.0


def test_contains_any_fails_when_none_present():
    assert run_check(check("contains_any", patterns=["a", "b"]), "zzz")["passed"] == 0.0


# ---------- regex ----------


def test_regex_found():
    assert run_check(check("regex", pattern=r"\d+"), "abc 123")["passed"] == 1.0


def test_regex_not_found():
    assert run_check(check("regex", pattern=r"\d+"), "abc")["passed"] == 0.0


def test_negated_regex_passes_when_absent():
    """The lipogram task: passing means the forbidden pattern never appears."""
    result = run_check(check("regex", pattern="[eE]", negate=True), "abc oxymoron")
    assert result["passed"] == 1.0


def test_negated_regex_fails_and_names_the_offender():
    result = run_check(check("regex", pattern="[eE]", negate=True), "the mug")
    assert result["passed"] == 0.0
    assert "'e'" in result["detail"]


def test_invalid_regex_in_a_task_fails_loudly_rather_than_silently_passing():
    result = run_check(check("regex", pattern="(unclosed"), "anything")
    assert result["passed"] == 0.0
    assert "invalid regex" in result["detail"]


# ---------- word count ----------


def test_word_count_within_range():
    assert run_check(check("word_count", min=2, max=5), "one two three")["passed"] == 1.0


def test_word_count_too_short():
    result = run_check(check("word_count", min=10), "too short")
    assert result["passed"] == 0.0
    assert "below the minimum" in result["detail"]


def test_word_count_too_long():
    result = run_check(check("word_count", max=2), "one two three four")
    assert result["passed"] == 0.0
    assert "above the maximum" in result["detail"]


def test_word_count_handles_punctuation_and_hyphens():
    result = run_check(check("word_count", min=4, max=4), "well-known, show-don't-tell is good")
    assert result["passed"] == 1.0


# ---------- json ----------


def test_json_valid_accepts_bare_json():
    assert run_check(check("json_valid"), '{"a": 1}')["passed"] == 1.0


def test_json_valid_rejects_a_code_fence():
    """The task asked for bare JSON; a fence is an instruction-following failure."""
    result = run_check(check("json_valid"), '```json\n{"a": 1}\n```')
    assert result["passed"] == 0.0
    assert "code fence" in result["detail"]


def test_json_valid_rejects_surrounding_prose():
    assert run_check(check("json_valid"), 'Here you go: {"a": 1}')["passed"] == 0.0


# ---------- aggregation ----------


def test_objective_score_is_a_weighted_mean_out_of_ten():
    results = [
        {"type": "a", "passed": 1.0, "weight": 1.0, "detail": ""},
        {"type": "b", "passed": 0.0, "weight": 1.0, "detail": ""},
    ]
    assert objective_score(results) == 5.0


def test_objective_score_respects_weights():
    results = [
        {"type": "a", "passed": 1.0, "weight": 3.0, "detail": ""},
        {"type": "b", "passed": 0.0, "weight": 1.0, "detail": ""},
    ]
    assert objective_score(results) == 7.5


def test_objective_score_is_none_without_checks():
    assert objective_score([]) is None


def test_skipped_checks_do_not_drag_the_score_down():
    """Disabling code execution must not make every coding task score zero."""
    results = [
        {"type": "code_exec", "passed": 0.0, "weight": 0.0, "detail": "skipped"},
        {"type": "contains_all", "passed": 1.0, "weight": 1.0, "detail": ""},
    ]
    assert objective_score(results) == 10.0


def test_all_checks_skipped_is_none_not_zero():
    results = [{"type": "code_exec", "passed": 0.0, "weight": 0.0, "detail": "skipped"}]
    assert objective_score(results) is None


def test_code_exec_is_skipped_when_disabled():
    result = run_check(check("code_exec", suite_path="/nonexistent"), "x", code_exec=False)
    assert result["weight"] == 0.0
    assert "CODE_EXEC=0" in result["detail"]


def test_run_checks_over_a_whole_task():
    task = Task(
        id="t",
        category="c",
        prompt="p",
        checks=[check("contains_all", patterns=["yes"]), check("word_count", max=3)],
    )
    results = run_checks(task, "yes indeed")
    assert [r["passed"] for r in results] == [1.0, 1.0]


def test_task_without_checks_produces_no_objective_score():
    task = Task(id="t", category="c", prompt="p")
    assert objective_score(run_checks(task, "anything")) is None


# ---------- blending ----------


@pytest.mark.parametrize(
    ("objective", "judge", "expected"),
    [
        (10.0, 5.0, 8.0),  # 0.6 * 10 + 0.4 * 5
        (0.0, 10.0, 4.0),
        (None, 7.0, 7.0),  # no checks: judge only
        (6.0, None, 6.0),  # judge failed: objective only
        (None, None, None),
    ],
)
def test_blended_score(objective, judge, expected):
    assert blended_score(objective, judge) == expected


def test_blend_weight_is_configurable():
    assert blended_score(10.0, 0.0, objective_weight=1.0) == 10.0
    assert blended_score(10.0, 0.0, objective_weight=0.0) == 0.0
