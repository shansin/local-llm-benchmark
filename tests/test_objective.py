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


# ---------- json_path ----------

RECORD = '{"meeting": {"room": "Redwood Room", "attendees": [{"name": "Ada"}, {"name": "Bo"}]}}'


def _path(response, path, expected, **kw):
    return run_check({"type": "json_path", "path": path, "equals": expected, **kw}, response)


def test_json_path_walks_objects_and_list_indices():
    assert _path(RECORD, "meeting.attendees[1].name", "Bo")["passed"] == 1.0


def test_json_path_compares_strings_case_insensitively():
    """A model that title-cases an extracted value still extracted the value."""
    assert _path(RECORD, "meeting.room", "redwood room")["passed"] == 1.0


def test_json_path_reports_what_it_found_instead():
    result = _path(RECORD, "meeting.room", "Cedar Room")
    assert result["passed"] == 0.0
    assert "Redwood Room" in result["detail"]


def test_json_path_names_the_missing_key():
    assert "no key 'floor'" in _path(RECORD, "meeting.floor", 3)["detail"]


def test_json_path_reads_through_a_code_fence():
    """Fence-wrapping is what `json_valid` is for; it must not fail twice."""
    fenced = f"Here you go:\n\n```json\n{RECORD}\n```\n"
    assert _path(fenced, "meeting.room", "Redwood Room")["passed"] == 1.0


def test_json_path_accepts_a_number_the_model_quoted():
    assert _path('{"total": "42"}', "total", 42)["passed"] == 1.0


def test_json_path_honours_a_numeric_tolerance():
    assert _path('{"pi": 3.14}', "pi", 3.14159, tolerance=0.01)["passed"] == 1.0
    assert _path('{"pi": 3.14}', "pi", 3.14159)["passed"] == 0.0


def test_json_path_fails_a_response_with_no_json_at_all():
    assert _path("I could not extract that.", "meeting.room", "x")["passed"] == 0.0


def test_json_path_does_not_coerce_booleans_to_numbers():
    assert _path('{"flag": true}', "flag", 1)["passed"] == 0.0


# ---------- answer_equals ----------


def _answer(response, expected, **kw):
    return run_check({"type": "answer_equals", "expected": expected, **kw}, response)


def test_answer_equals_reads_the_final_answer_line():
    response = "Step 1: ...\nStep 2: ...\n\nAnswer: 147"
    assert _answer(response, 147, numeric=True)["passed"] == 1.0


def test_answer_equals_scores_the_conclusion_not_the_working():
    """A right method that lands on the wrong number is a wrong answer."""
    response = "6 and 4 give 12 as the LCM.\n\nFinal answer: 5 hours"
    assert _answer(response, 3, numeric=True)["passed"] == 0.0


def test_answer_equals_takes_the_last_marker_when_a_model_restates_the_format():
    response = "I will end with `Answer: <n>`.\n\nAnswer: 12"
    assert _answer(response, 12, numeric=True)["passed"] == 1.0


def test_answer_equals_falls_back_to_the_closing_lines():
    assert (
        _answer("Working it through...\n\nThe tank fills in 3 hours.", 3, numeric=True)["passed"]
        == 1.0
    )


def test_answer_equals_ignores_numbers_from_the_working():
    """Only the stated answer counts, or every derivation matches by accident."""
    response = "Pipe A does 1/6, pipe B does 1/4, the drain removes 1/12.\n\nAnswer: 3 hours"
    assert _answer(response, 6, numeric=True)["passed"] == 0.0


def test_answer_equals_strips_thousands_separators():
    assert _answer("Answer: 1,048,576", 1048576, numeric=True)["passed"] == 1.0


def test_answer_equals_applies_a_numeric_tolerance():
    assert _answer("Answer: 3.33", 3.3333, numeric=True, tolerance=0.01)["passed"] == 1.0


def test_answer_equals_accepts_any_of_several_wordings():
    assert _answer("Answer: not stated in the passage", ["not stated", "unknown"])["passed"] == 1.0


def test_answer_equals_fails_a_response_that_never_commits():
    result = _answer("It depends on several factors.", 3, numeric=True)
    assert result["passed"] == 0.0
    assert "no number" in result["detail"]


# ---------- line_count and match_count ----------


def test_line_count_catches_a_model_that_stopped_early():
    """Twenty rows and an ellipsis is not a partial success."""
    response = "\n".join(f"row {i}" for i in range(20)) + "\n... and so on"
    assert (
        run_check({"type": "line_count", "pattern": r"^row ", "equals": 30}, response)["passed"]
        == 0.0
    )


def test_line_count_ignores_blank_lines():
    assert run_check({"type": "line_count", "equals": 3}, "a\n\n\nb\n\nc\n")["passed"] == 1.0


def test_line_count_accepts_a_range():
    check = {"type": "line_count", "min": 2, "max": 4}
    assert run_check(check, "a\nb\nc")["passed"] == 1.0
    assert run_check(check, "a")["passed"] == 0.0


def test_match_count_counts_every_occurrence_not_every_line():
    response = "not stated. not stated. answered."
    assert (
        run_check({"type": "match_count", "pattern": "not stated", "equals": 2}, response)["passed"]
        == 1.0
    )


def test_match_count_reports_the_number_it_found():
    result = run_check({"type": "match_count", "pattern": "x", "equals": 5}, "xxx")
    assert "3 matches" in result["detail"]


def test_a_counting_check_on_an_empty_answer_does_not_pass_by_being_zero():
    """Silence satisfies "at most 3 matches" unless empty answers fail outright."""
    task = Task(
        id="t",
        category="c",
        prompt="p",
        checks=[{"type": "match_count", "pattern": "wolf", "max": 0}],
    )
    assert run_checks(task, "")[0]["passed"] == 0.0
