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
