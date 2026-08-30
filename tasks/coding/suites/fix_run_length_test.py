"""Reference tests for the run-length encode repair task.

The two bugs partly cancel on inputs of a single repeated character, so the
tests deliberately include the shapes where they do not: multiple runs, a run
of length one at the end, and a two-character string.
"""

import pytest

from candidate import encode


def test_the_docstring_example():
    assert encode("aaabbc") == [("a", 3), ("b", 2), ("c", 1)]


def test_every_character_distinct():
    assert encode("abc") == [("a", 1), ("b", 1), ("c", 1)]


def test_single_character():
    assert encode("a") == [("a", 1)]


def test_two_identical_characters():
    assert encode("aa") == [("a", 2)]


def test_two_different_characters():
    assert encode("ab") == [("a", 1), ("b", 1)]


def test_empty_string():
    assert encode("") == []


def test_final_run_is_not_dropped():
    assert encode("aab") == [("a", 2), ("b", 1)]


def test_repeated_characters_in_separate_runs():
    assert encode("aabaa") == [("a", 2), ("b", 1), ("a", 2)]


def test_longer_mixed_input():
    assert encode("wwwwaaadexxxxxxywww") == [
        ("w", 4),
        ("a", 3),
        ("d", 1),
        ("e", 1),
        ("x", 6),
        ("y", 1),
        ("w", 3),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
