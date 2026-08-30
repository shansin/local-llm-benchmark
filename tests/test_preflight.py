from llmbench import preflight as pf
from llmbench.config import GenerationParams


def result(response="", thinking="", error=None, think_used=None, **overrides):
    base = {
        "response": response,
        "thinking": thinking,
        "done_reason": "stop",
        "truncated": False,
        "discarded_reasoning": False,
        "tokens_per_sec": 50.0,
        "ttft": 0.1,
        "prefill_time": 0.1,
        "load_time": 0.0,
        "total_time": 1.0,
        "eval_count": 10,
        "prompt_eval_speed": 500.0,
        "prompt_eval_count": 20,
        "seed": 0,
        "error": error,
        "gpu": None,
        "think_used": think_used,
    }
    return {**base, **overrides}


def _run(monkeypatch, responses):
    """Feed canned results to preflight, recording the think mode of each call."""
    calls = []

    def fake_run_prompt(base_url, model, prompt, timeout, params, retries=1, think=None):
        calls.append(think)
        return responses[len(calls) - 1]

    monkeypatch.setattr(pf, "run_prompt", fake_run_prompt)
    profile = pf.preflight("http://x", "m", GenerationParams())
    return profile, calls


def test_a_model_that_answers_with_thinking_on_keeps_thinking_on(monkeypatch):
    profile, calls = _run(
        monkeypatch, [result("<answer>5</answer>", thinking="2+3", think_used=True)]
    )
    assert profile.usable
    assert profile.think is True
    assert calls == [True]


def test_a_rejected_think_request_is_not_retried_as_a_separate_mode(monkeypatch):
    """run_prompt downgrades a rejected `think`; the ladder must notice that the
    model-default mode has then already been tried."""
    profile, calls = _run(monkeypatch, [result("<answer>5</answer>", think_used=None)])
    assert profile.usable
    assert profile.think is None
    assert calls == [True]


def test_the_ladder_falls_through_to_a_mode_that_produces_an_answer(monkeypatch):
    profile, calls = _run(
        monkeypatch,
        [
            # Thinking on: the model reasons and never answers.
            result("", thinking="endless deliberation", think_used=True),
            # Model default: decodes onto neither channel.
            result("", discarded_reasoning=True, think_used=None),
            # Thinking off: a plain answer.
            result("<answer>5</answer>", think_used=False),
        ],
    )
    assert profile.usable
    assert profile.think is False
    assert calls == [True, None, False]


def test_a_model_that_never_answers_is_excluded_with_the_evidence(monkeypatch):
    profile, _ = _run(
        monkeypatch,
        [
            result("", thinking="thought forever", think_used=True),
            result(error="timeout", response="[TIMEOUT]"),
            result("", think_used=False),
        ],
    )
    assert not profile.usable
    assert "reasoned without answering" in profile.note
    assert "timeout" in profile.note
    assert "empty response" in profile.note


def test_an_error_on_one_mode_does_not_condemn_the_model(monkeypatch):
    profile, calls = _run(
        monkeypatch,
        [
            result(error="HTTPError", response="[ERROR: HTTPError]"),
            result("<answer>5</answer>", think_used=None),
        ],
    )
    assert profile.usable
    assert profile.think is None
    assert calls == [True, None]


def test_profile_round_trips_through_state_storage():
    profile = pf.ModelProfile(think=False, usable=True, note="thinking off")
    assert pf.ModelProfile.from_dict(profile.as_dict()) == profile


def test_preflight_uses_the_real_protocol_prompt(monkeypatch):
    seen = {}

    def fake_run_prompt(base_url, model, prompt, timeout, params, retries=1, think=None):
        seen["prompt"] = prompt
        seen["num_predict"] = params.num_predict
        return result("<answer>5</answer>", think_used=think)

    monkeypatch.setattr(pf, "run_prompt", fake_run_prompt)
    pf.preflight("http://x", "m", GenerationParams())
    assert "<answer>" in seen["prompt"]
    assert seen["num_predict"] == pf.PREFLIGHT_PREDICT
