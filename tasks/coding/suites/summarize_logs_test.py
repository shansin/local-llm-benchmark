"""Reference tests for the summarize_logs task."""

import pytest

from candidate import summarize_logs

LINES = [
    "2026-04-11T08:15:30Z INFO auth user logged in",
    "2026-04-11T08:15:31Z ERROR billing card declined",
    "2026-04-11T08:15:32Z WARN auth slow response",
    "2026-04-11T08:15:33Z ERROR auth token expired",
    "garbage line",
    "",
]


def test_counts_parsed_and_malformed_separately():
    summary = summarize_logs(LINES)
    assert summary["total"] == 4
    assert summary["malformed"] == 2


def test_counts_by_level():
    assert summarize_logs(LINES)["by_level"] == {"INFO": 1, "ERROR": 2, "WARN": 1}


def test_services_are_sorted_and_deduplicated():
    assert summarize_logs(LINES)["services"] == ["auth", "billing"]


def test_first_error_is_the_message_only():
    assert summarize_logs(LINES)["first_error"] == "card declined"


def test_first_error_is_none_without_errors():
    ok = ["2026-04-11T08:15:30Z INFO auth fine"]
    assert summarize_logs(ok)["first_error"] is None


def test_empty_input():
    summary = summarize_logs([])
    assert summary["total"] == 0
    assert summary["malformed"] == 0
    assert summary["services"] == []
    assert summary["first_error"] is None


def test_malformed_lines_do_not_raise():
    junk = ["", "   ", "not a log", "2026-04-11T08:15:30Z", "2026-04-11T08:15:30Z NOPE svc msg"]
    summary = summarize_logs(junk)
    assert summary["total"] + summary["malformed"] == len(junk)


def test_returns_exactly_the_specified_keys():
    assert set(summarize_logs(LINES)) == {
        "total",
        "malformed",
        "by_level",
        "services",
        "first_error",
    }


def test_multi_word_messages_are_kept_whole():
    line = ["2026-04-11T08:15:30Z ERROR api a b c d e"]
    assert summarize_logs(line)["first_error"] == "a b c d e"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
