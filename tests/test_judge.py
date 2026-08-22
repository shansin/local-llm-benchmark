import json

import requests

from llmbench.scoring import judge

RUBRIC = {
    "accuracy": 8,
    "completeness": 8,
    "instruction_following": 10,
    "clarity": 10,
    "reason": "Correct but terse.",
}


class _Resp:
    def __init__(self, text, field="response"):
        self._payload = {field: text}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _serving(*texts):
    """A fake Ollama that returns each text in turn."""
    queue = list(texts)

    def _post(*a, **k):
        return _Resp(queue.pop(0) if len(queue) > 1 else queue[0])

    return _post


# ---------- parsing ----------


def test_parses_the_rubric_and_averages_the_dimensions():
    result = judge.parse_judge_output(json.dumps(RUBRIC))
    assert result["score"] == 9.0
    assert result["reason"] == "Correct but terse."
    assert result["dimensions"]["accuracy"] == 8


def test_parses_rubric_inside_a_code_fence():
    text = f"```json\n{json.dumps(RUBRIC)}\n```"
    assert judge.parse_judge_output(text)["score"] == 9.0


def test_parses_rubric_with_surrounding_prose():
    text = f"Here is my evaluation:\n{json.dumps(RUBRIC)}\nHope that helps."
    assert judge.parse_judge_output(text)["score"] == 9.0


def test_clamps_out_of_range_dimensions():
    payload = dict(RUBRIC, accuracy=99, clarity=0)
    dimensions = judge.parse_judge_output(json.dumps(payload))["dimensions"]
    assert dimensions["accuracy"] == 10
    assert dimensions["clarity"] == 1


def test_partial_rubric_scores_on_the_dimensions_given():
    result = judge.parse_judge_output(json.dumps({"accuracy": 6, "reason": "eh"}))
    assert result["score"] == 6.0
    assert set(result["dimensions"]) == {"accuracy"}


def test_falls_back_to_the_plain_text_format():
    """Older judges emit 'Score: N / Reason: ...' — still accepted."""
    result = judge.parse_judge_output("Score: 7\nReason: Adequate.")
    assert result["score"] == 7.0
    assert result["reason"] == "Adequate."


def test_unparseable_output_is_none_not_a_low_score():
    result = judge.parse_judge_output("I'm not sure how to rate this.")
    assert result["score"] is None
    assert "UNPARSEABLE" in result["reason"]


def test_non_numeric_dimensions_are_ignored():
    payload = {"accuracy": "very good", "completeness": 8, "reason": "x"}
    assert judge.parse_judge_output(json.dumps(payload))["score"] == 8.0


def test_records_the_response_length_for_bias_analysis():
    assert judge.parse_judge_output(json.dumps(RUBRIC), 1234)["response_chars"] == 1234


# ---------- prompt ----------


def test_prompt_includes_criteria_when_given():
    with_criteria = judge.build_judge_prompt("coding", "p", "r", "expected thing")
    assert "expected thing" in with_criteria
    assert "primary scoring guide" in with_criteria
    assert "primary scoring guide" not in judge.build_judge_prompt("coding", "p", "r")


def test_prompt_never_names_the_model_under_test():
    prompt = judge.build_judge_prompt("coding", "the prompt", "the response", "criteria")
    assert "qwen" not in prompt.lower()
    assert "AI response" in prompt


def test_prompt_asks_for_every_dimension():
    prompt = judge.build_judge_prompt("coding", "p", "r")
    for dimension in judge.DIMENSIONS:
        assert dimension in prompt


# ---------- panel ----------


def test_single_judge_verdict(monkeypatch):
    monkeypatch.setattr(requests, "post", _serving(json.dumps(RUBRIC)))
    result = judge.judge_response("http://x", ["j"], "coding", "p", "r")
    assert result["score"] == 9.0
    assert [v["judge"] for v in result["votes"]] == ["j"]


def test_panel_takes_the_median_not_the_mean(monkeypatch):
    """One outlying judge must not move the verdict."""
    verdicts = [
        json.dumps(dict.fromkeys(judge.DIMENSIONS, n) | {"reason": f"score {n}"}) for n in (9, 8, 1)
    ]
    monkeypatch.setattr(requests, "post", _serving(*verdicts))
    result = judge.judge_response("http://x", ["a", "b", "c"], "coding", "p", "r")
    assert result["score"] == 8.0
    assert len(result["votes"]) == 3


def test_panel_records_every_vote(monkeypatch):
    verdicts = [json.dumps(dict.fromkeys(judge.DIMENSIONS, n) | {"reason": "r"}) for n in (10, 6)]
    monkeypatch.setattr(requests, "post", _serving(*verdicts))
    result = judge.judge_response("http://x", ["a", "b"], "coding", "p", "r")
    assert [v["score"] for v in result["votes"]] == [10.0, 6.0]


def test_a_model_does_not_vote_on_its_own_answer(monkeypatch):
    monkeypatch.setattr(requests, "post", _serving(json.dumps(RUBRIC)))
    result = judge.judge_response(
        "http://x", ["alpha", "beta"], "coding", "p", "r", model_under_test="alpha"
    )
    assert [v["judge"] for v in result["votes"]] == ["beta"]


def test_self_judging_can_be_explicitly_allowed(monkeypatch):
    monkeypatch.setattr(requests, "post", _serving(json.dumps(RUBRIC)))
    result = judge.judge_response(
        "http://x", ["alpha"], "coding", "p", "r", model_under_test="alpha", allow_self_judge=True
    )
    assert result["score"] == 9.0


def test_no_independent_judge_leaves_it_unscored(monkeypatch):
    """Better an honest gap than a score the model gave itself."""
    monkeypatch.setattr(requests, "post", _serving(json.dumps(RUBRIC)))
    result = judge.judge_response(
        "http://x", ["alpha"], "coding", "p", "r", model_under_test="alpha"
    )
    assert result["score"] is None
    assert "NO INDEPENDENT JUDGE" in result["reason"]


def test_panel_survives_one_judge_failing(monkeypatch):
    calls = []

    def _post(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ReadTimeout()
        return _Resp(json.dumps(RUBRIC))

    monkeypatch.setattr(requests, "post", _post)
    result = judge.judge_response("http://x", ["a", "b"], "coding", "p", "r")
    assert result["score"] == 9.0


def test_reasoning_models_that_answer_in_the_thinking_field_are_read(monkeypatch):
    """With a schema set, some reasoning models leave `response` empty and put
    the whole verdict in `thinking`. That is a verdict, not a failure."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(json.dumps(RUBRIC), "thinking"))
    result = judge.judge_response("http://x", ["j"], "coding", "p", "r")
    assert result["score"] == 9.0


def test_response_field_wins_when_both_are_present(monkeypatch):
    class _Both:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": json.dumps(RUBRIC), "thinking": "let me consider..."}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Both())
    assert judge.judge_response("http://x", ["j"], "coding", "p", "r")["score"] == 9.0


def test_genuinely_empty_reply_is_unscored(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(""))
    assert judge.judge_response("http://x", ["j"], "coding", "p", "r")["score"] is None


# ---------- transport failures ----------


def _raise(exc):
    def _post(*a, **k):
        raise exc

    return _post


def test_timeout_yields_unscored(monkeypatch):
    monkeypatch.setattr(requests, "post", _raise(requests.exceptions.ReadTimeout()))
    result = judge.judge_response("http://x", ["j"], "coding", "p", "r")
    assert result["score"] is None
    assert "TIMEOUT" in result["reason"]


def test_connection_error_yields_unscored(monkeypatch):
    monkeypatch.setattr(requests, "post", _raise(requests.exceptions.ConnectionError()))
    result = judge.judge_response("http://x", ["j"], "coding", "p", "r")
    assert result["score"] is None
    assert "ConnectionError" in result["reason"]


def test_judging_is_deterministic_by_construction(monkeypatch):
    """A noisy instrument cannot measure a small difference."""
    captured = {}

    rubric_text = json.dumps(RUBRIC)

    def _post(url, json=None, **k):
        captured.update(json)
        return _Resp(rubric_text)

    monkeypatch.setattr(requests, "post", _post)
    judge.judge_response("http://x", ["j"], "coding", "p", "r")
    assert captured["options"]["temperature"] == 0.0
    assert "format" in captured


# ---------- measured facts are given to the judge, not re-derived ----------


def test_verified_facts_are_stated_as_ground_truth_in_the_prompt():
    prompt = judge.build_judge_prompt(
        "writing",
        "Write 250 words.",
        "the response",
        verified=["word_count: 421 words, within range"],
    )
    assert "421 words, within range" in prompt
    assert "ground truth" in prompt


def test_no_verified_section_when_there_is_nothing_measured():
    assert "ground truth" not in judge.build_judge_prompt("writing", "p", "r")


def test_format_verified_skips_checks_that_did_not_run():
    from llmbench.scoring.judge import format_verified

    checks = [
        {"type": "word_count", "detail": "300 words, within range", "weight": 1.0},
        {"type": "code_exec", "detail": "skipped (CODE_EXEC=0)", "weight": 0.0},
    ]
    assert format_verified(checks) == ["word_count: 300 words, within range"]
