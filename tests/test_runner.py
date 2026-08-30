import json

import requests

from llmbench.config import GenerationParams
from llmbench.runner import metrics_from_response, run_prompt

FINAL = {
    "response": "",
    "done": True,
    "eval_count": 100,
    "eval_duration": 2_000_000_000,  # 2s -> 50 tok/s
    "load_duration": 3_000_000_000,
    "prompt_eval_duration": 500_000_000,
    "prompt_eval_count": 250,  # -> 500 tok/s prefill
    "total_duration": 6_000_000_000,
}


class _StreamResp:
    def __init__(self, lines, status_exc=None):
        self._lines = lines
        self._status_exc = status_exc

    def raise_for_status(self):
        if self._status_exc:
            raise self._status_exc

    def iter_lines(self):
        yield from (json.dumps(obj).encode() for obj in self._lines)

    def close(self):
        pass


def _ok_stream(*a, **k):
    return _StreamResp([{"response": "hel"}, {"response": "lo"}, FINAL])


def test_metrics_are_derived_from_ollama_durations():
    r = metrics_from_response(FINAL)
    assert r["tokens_per_sec"] == 50.0
    assert r["prompt_eval_speed"] == 500.0
    assert r["total_time"] == 6.0
    assert r["error"] is None


def test_prefill_and_load_are_reported_separately():
    r = metrics_from_response(FINAL)
    assert r["prefill_time"] == 0.5
    assert r["load_time"] == 3.0


def test_ttft_falls_back_to_load_plus_prefill_without_a_streamed_measurement():
    assert metrics_from_response(FINAL)["ttft"] == 3.5


def test_streamed_ttft_wins_when_measured():
    assert metrics_from_response(FINAL, client_ttft=0.25)["ttft"] == 0.25


def test_zero_durations_do_not_divide_by_zero():
    r = metrics_from_response({"response": "x"})
    assert r["tokens_per_sec"] == 0.0
    assert r["prompt_eval_speed"] == 0.0


def test_run_prompt_assembles_streamed_chunks(monkeypatch):
    monkeypatch.setattr(requests, "post", _ok_stream)
    r = run_prompt("http://x", "m", "p", 10)
    assert r["response"] == "hello"
    assert r["error"] is None
    assert r["ttft"] > 0


def test_generation_options_are_sent(monkeypatch):
    """Leaving sampling to Ollama's defaults is what made runs unreproducible."""
    captured = {}

    def _post(url, json=None, **k):
        captured.update(json)
        return _StreamResp([FINAL])

    monkeypatch.setattr(requests, "post", _post)
    run_prompt("http://x", "m", "p", 10, GenerationParams(temperature=0.0, seed=7, num_ctx=4096))
    assert captured["options"]["temperature"] == 0.0
    assert captured["options"]["seed"] == 7
    assert captured["options"]["num_ctx"] == 4096


def test_repeat_seeds_are_distinct_and_deterministic():
    base = GenerationParams(seed=100)
    assert [base.for_repeat(i).seed for i in range(3)] == [100, 101, 102]
    assert base.for_repeat(1).seed == base.for_repeat(1).seed


def test_seed_is_recorded_on_the_result(monkeypatch):
    monkeypatch.setattr(requests, "post", _ok_stream)
    r = run_prompt("http://x", "m", "p", 10, GenerationParams(seed=42))
    assert r["seed"] == 42


def test_timeouts_are_recorded_not_retried(monkeypatch):
    """A model that can't finish in time is a finding, not a flake."""
    calls = []

    def _post(*a, **k):
        calls.append(1)
        raise requests.exceptions.ReadTimeout()

    monkeypatch.setattr(requests, "post", _post)
    r = run_prompt("http://x", "m", "p", 10, retries=3)
    assert r["response"] == "[TIMEOUT]"
    assert len(calls) == 1


def test_connection_errors_are_retried_then_recorded(monkeypatch):
    calls = []

    def _post(*a, **k):
        calls.append(1)
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(requests, "post", _post)
    monkeypatch.setattr("llmbench.runner.time.sleep", lambda _: None)
    r = run_prompt("http://x", "m", "p", 10, retries=3)
    assert len(calls) == 3
    assert r["error"] == "ConnectionError"


def test_connection_error_that_recovers_succeeds(monkeypatch):
    calls = []

    def _post(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError()
        return _StreamResp([{"response": "hi"}, FINAL])

    monkeypatch.setattr(requests, "post", _post)
    monkeypatch.setattr("llmbench.runner.time.sleep", lambda _: None)
    r = run_prompt("http://x", "m", "p", 10, retries=3)
    assert r["response"] == "hi"
    assert r["error"] is None


def test_http_errors_are_not_retried(monkeypatch):
    calls = []

    def _post(*a, **k):
        calls.append(1)
        return _StreamResp([], status_exc=requests.exceptions.HTTPError("500"))

    monkeypatch.setattr(requests, "post", _post)
    r = run_prompt("http://x", "m", "p", 10, retries=3)
    assert r["error"] == "HTTPError"
    assert len(calls) == 1


def test_error_reported_inside_the_stream_is_caught(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: _StreamResp([{"error": "model not found"}])
    )
    r = run_prompt("http://x", "m", "p", 10)
    assert r["error"] == "HTTPError"


# ---------- reasoning channel and truncation ----------

THINKING_STREAM = [
    {"thinking": "let me work"},
    {"thinking": " this out"},
    {"response": "the answer"},
    {**FINAL, "done_reason": "stop"},
]


def test_thinking_is_captured_separately_from_the_answer(monkeypatch):
    """A model that puts everything on the thinking channel has not said nothing."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: _StreamResp(THINKING_STREAM))
    r = run_prompt("http://x", "m", "p", 10)
    assert r["response"] == "the answer"
    assert r["thinking"] == "let me work this out"


def test_ttft_ignores_thinking_only_chunks(monkeypatch):
    """Time to first token means the first token the caller can see."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: _StreamResp(THINKING_STREAM))
    assert run_prompt("http://x", "m", "p", 10)["ttft"] > 0


def test_done_reason_length_marks_a_generation_as_truncated():
    assert metrics_from_response({**FINAL, "done_reason": "length"})["truncated"]


def test_a_normal_stop_is_not_truncated():
    assert not metrics_from_response({**FINAL, "done_reason": "stop"}, num_ctx=8192)["truncated"]


def test_filling_the_context_window_counts_as_truncation_without_a_done_reason():
    """Some builds omit done_reason; a full window is the same finding."""
    payload = {**FINAL, "eval_count": 7998, "prompt_eval_count": 200}
    assert metrics_from_response(payload, num_ctx=8192)["truncated"]
    assert not metrics_from_response(payload, num_ctx=32768)["truncated"]


def test_thinking_option_is_sent_only_when_asked(monkeypatch):
    sent = {}

    def capture(url, json=None, **k):
        sent.update(json or {})
        return _StreamResp([{"response": "hi"}, FINAL])

    monkeypatch.setattr(requests, "post", capture)
    run_prompt("http://x", "m", "p", 10)
    assert "think" not in sent
    run_prompt("http://x", "m", "p", 10, think=True)
    assert sent["think"] is True


def test_a_model_without_a_thinking_channel_is_measured_anyway(monkeypatch):
    """Being refused the option is a fact about the model, not a run failure."""
    calls = []

    def flaky(url, json=None, **k):
        calls.append(json or {})
        if "think" in (json or {}):
            raise requests.exceptions.HTTPError('"thinking" is not supported')
        return _StreamResp([{"response": "hi"}, FINAL])

    monkeypatch.setattr(requests, "post", flaky)
    r = run_prompt("http://x", "m", "p", 10, think=True)
    assert r["error"] is None
    assert r["response"] == "hi"
    assert len(calls) == 2


def test_tokens_on_neither_channel_are_flagged_as_discarded_reasoning():
    """The gemma4 case: a full budget spent, both channels empty.

    Ollama runs a hybrid-reasoning model in reasoning mode when `think` is left
    unset and streams none of it. It reads as a truncation, but raising NUM_CTX
    is the wrong remedy, so it has to be countable on its own.
    """
    payload = {**FINAL, "done_reason": "length", "eval_count": 12288}
    r = metrics_from_response(payload, text="", thinking="")
    assert r["discarded_reasoning"]
    assert r["truncated"]


def test_truncation_with_a_partial_answer_is_not_discarded_reasoning():
    payload = {**FINAL, "done_reason": "length"}
    assert not metrics_from_response(payload, text="half an ans")["discarded_reasoning"]


def test_truncation_with_only_reasoning_is_not_discarded_reasoning():
    """Reasoning that reached the thinking channel was kept, not thrown away."""
    payload = {**FINAL, "done_reason": "length"}
    assert not metrics_from_response(payload, text="", thinking="let me think")[
        "discarded_reasoning"
    ]


def test_a_generation_that_decoded_nothing_is_not_discarded_reasoning():
    """No tokens spent means nothing was lost — that is some other failure."""
    assert not metrics_from_response({**FINAL, "eval_count": 0}, text="")["discarded_reasoning"]
